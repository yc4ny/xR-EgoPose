# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, io
import argparse
import os
from model import resnet as pose_resnet
from model import encoder_decoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pprint
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml
from easydict import EasyDict as edict
from torchvision import transforms as transforms
import pdb

LOGGER = ConsoleLogger("Demo", 'test')

def parse_args():
    parser = argparse.ArgumentParser(description="demo script")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--load_model', type=str, default = "checkpoints/Finetune/model_best.tar")
    parser.add_argument('--data', default='test', type=str)  # "train", "val", "test"
    args = parser.parse_args()

    return args

def MPJPE(y_output, y_target):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between predicted and target 3D poses.

    Parameters:
    - y_output: Predicted 3D pose (numpy array of shape (n_frames, n_joints, 3))
    - y_target: Target 3D pose (numpy array of shape (n_frames, n_joints, 3))

    Returns:
    - mpjpe: Mean Per Joint Position Error
    """
    # Ensure the input arrays have the same shape
    assert y_output.shape == y_target.shape, "Input arrays must have the same shape"

    # Calculate the Euclidean distance for each joint in each frame
    joint_errors = np.linalg.norm(y_output - y_target, axis=1)

    # Calculate the mean error across all joints and frames
    mpjpe = np.mean(joint_errors)

    return mpjpe*1000


def show3Dpose(channels, ax, gt, mm=True):
    vals = channels.reshape((16, 3))
    if mm:
        channels *= 100
    if gt:
        color = "#3498db"
    else:
        color = "#e74c3c"
    I = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])  # start points
    J = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5])  # end points

    for i in range(16):
        ax.scatter(vals[i][0], vals[i][1], vals[i][2], c=color)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=color)

    RADIUS = 50  # space around the subject
    xroot, yroot, zroot = vals[8, 0], vals[8, 1], vals[8, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([RADIUS + zroot, -RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def main():
    """Main"""
    args = parse_args()
    print('Starting demo...')
    device = torch.device(f"cuda:{args.gpu}")
    LOGGER.info((args))

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    data = Mocap(
        config.dataset[args.data],
        SetType.TEST,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=16,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Evaluation -------------------

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- Model -------------------
    with open('model/model.yaml') as fin:
        model_cfg = edict(yaml.safe_load(fin))
    resnet = pose_resnet.get_pose_net(model_cfg, False)
    autoencoder = encoder_decoder.AutoEncoder()

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    else:
        raise ValueError("No checkpoint!")

    resnet.cuda(device)
    autoencoder.cuda(device)
    resnet.eval()
    autoencoder.eval()

    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    subplot_idx = 1

    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            p3d = p3d.to(device)

            heatmap2d_hat = resnet(img)
            p3d_hat, _ = autoencoder(heatmap2d_hat)

            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            for i in range(len(y_output)):
                print("MPJPE: " + str(MPJPE(y_output[i], y_target[i])) + " mm")
                ax = fig.add_subplot(111, projection='3d')
                show3Dpose(p3d[i].cpu().numpy(), ax, True)
                show3Dpose(p3d_hat[i].detach().cpu().numpy(), ax, False)
                plt.savefig(f"output_{i}.png")
                
    

if __name__ == "__main__":
    main()