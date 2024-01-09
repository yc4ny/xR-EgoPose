import os
import argparse
import numpy as np
import torch
import cv2
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easydict import EasyDict as edict
from model import resnet as pose_resnet
from model import encoder_decoder
from utils import config
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="demo script")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--load_model', type=str, default = "checkpoints/Finetune/model_best.tar")
    parser.add_argument('--input_dir', default = "imgs", type= str)
    parser.add_argument('--save_dir', default="inference_output", type=str)
    args = parser.parse_args()

    return args

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
    print('Starting Inference...')
    device = torch.device(f"cuda:{args.gpu}")

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

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Running Inference on input images:"):
        img_path = os.path.join(args.input_dir, filename)
        img = torch.from_numpy(np.transpose(cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (368, 368), interpolation=cv2.INTER_AREA), (2, 0, 1))).float().unsqueeze(0).div(255.0).sub(0.5).div(0.5).to(device)
        # Perform inference
        with torch.no_grad():
            heatmap2d_hat = resnet(img)
            p3d_hat, _ = autoencoder(heatmap2d_hat)

        # Visualize and save the output
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(p3d_hat.detach().cpu().numpy(), ax, False)
        output_filename = os.path.join(args.save_dir, f"{filename}")
        plt.savefig(output_filename)
        plt.close(fig) 

if __name__ == "__main__":
    main()