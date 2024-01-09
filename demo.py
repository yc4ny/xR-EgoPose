# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config
from utils import evaluate, io
import argparse
import os
from model import resnet as pose_resnet
from model import encoder_decoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yaml
from easydict import EasyDict as edict
from torchvision import transforms as transforms
from io import BytesIO
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="demo script")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--load_model', type=str, default = "checkpoints/Finetune/model_best.tar")
    parser.add_argument('--data', default='test', type=str)
    parser.add_argument('--save_dir', default="demo_output", type=str)
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

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)  
    # Keep z pane

    # Get rid of the lines in 3d
    ax.xaxis.line.set_color(white) 
    ax.yaxis.line.set_color(white) 
    ax.zaxis.line.set_color(white)  


def main():
    """Main"""
    args = parse_args()
    print('Starting demo...')
    device = torch.device(f"cuda:{args.gpu}")

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

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir) 

    with torch.no_grad():
        index = 0
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            p3d = p3d.to(device)

            heatmap2d_hat = resnet(img)
            p3d_hat, _ = autoencoder(heatmap2d_hat)

            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            for i in range(len(y_output)):
                print(f"Output {index} -" +  " MPJPE: " + str(MPJPE(y_output[i], y_target[i])) + " mm")
                ax = fig.add_subplot(111, projection='3d')
                show3Dpose(p3d[i].cpu().numpy(), ax, True)
                show3Dpose(p3d_hat[i].detach().cpu().numpy(), ax, False)
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                plot_image = np.array(Image.open(buf))
                plot_image = plot_image[:, :, :3]
                tensor_image = ((np.transpose(img[i].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255)[:, :, ::-1].astype(np.uint8)
                h1, w1, _ = tensor_image.shape
                h2, w2, _ = plot_image.shape
                if h1 != h2:
                    plot_image = cv2.resize(plot_image, (int(w2 * h1 / h2), h1))
                concatenated_image = np.hstack((tensor_image, plot_image))
                cv2.imwrite(os.path.join(args.save_dir,f"output_{index}.jpg" ), concatenated_image)
                index += 1
                plt.clf()
                
if __name__ == "__main__":
    main()