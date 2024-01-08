import os
import sys
import shutil
from tqdm import tqdm
import argparse
import yaml
import pprint

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from easydict import EasyDict as edict

from base import SetType
from utils import config, ConsoleLogger, evaluate 
from utils.loss import HeatmapLoss, LimbLoss, PoseLoss, HeatmapLossSquare

import dataset.transform as trsf
from dataset import Mocap

from model import resnet as pose_resnet
from model import encoder_decoder

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('-batch_size', '--batch_size', default=16, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--log_dir', type=str, default= "experiments/Train2d")
parser.add_argument('--training_type', type=str)
parser.add_argument('--load_model', help='the path of the checkpoint to load', type=str)
parser.add_argument('--load_2d_model', help='the path of the checkpoint to load 2D pose detector model', type=str)
parser.add_argument('--load_3d_model', help='the path of the checkpoint to load 3D pose detector model', type=str) 

# Data Loader for loading data in prep for training
def load_data():
    data_transform = transforms.Compose([trsf.ImageTrsf(), trsf.Joints3DTrsf(), trsf.ToTensor()])
    train_data = Mocap(config.dataset.train, SetType.TRAIN, transform = data_transform)
    test_data = Mocap(config.dataset.test, SetType.TEST, transform=data_transform)
    train_data_loader = DataLoader(train_data, batch_size = 16, shuffle = config.data_loader.shuffle, num_workers = 8)
    test_data_loader = DataLoader(test_data, batch_size = 2, shuffle = config.data_loader.shuffle, num_workers = 8)

    return data_transform, train_data, test_data, train_data_loader, test_data_loader

def validate_2d(LOGGER, data_loader, resnet, device, epoch):

    Loss2D = HeatmapLoss()
    val_losses = AverageMeter()
    Loss2D.cuda(device)

    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            heatmap = heatmap.to(device)
            heatmap2d_hat = resnet(img)
            loss2d = Loss2D(heatmap2d_hat, heatmap).mean()
            val_losses.update(loss2d.item(), img.size(0))
        LOGGER.info('Saving evaluation results...')
        msg = 'Test:\t' \
              'Loss {loss.avg:.5f}\t'.format(loss=val_losses)
        LOGGER.info(msg)

    return val_losses.avg

def validate_3d(LOGGER, data_loader, autoencoder, device, epoch):
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()
    with torch.no_grad(): 
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            p3d = p3d.to(device)
            heatmap = heatmap.to(device)

            p3d_hat, heatmap2d_recon = autoencoder(heatmap)
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)
        LOGGER.info('===========Evaluation on Val data==========')
        res = {'FullBody': eval_body.get_results(),
               'UpperBody': eval_upper.get_results(),
               'LowerBody': eval_lower.get_results()}
        LOGGER.info(pprint.pformat(res))

    return eval_body.get_results()['All']


def validate_finetune(LOGGER, data_loader, resnet, autoencoder, device, epoch):
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()
    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            p3d = p3d.to(device)
            heatmap2d_hat = resnet(img)
            p3d_hat, _ = autoencoder(heatmap2d_hat)
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()
            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)
        LOGGER.info('===========Evaluation on Val data==========')
        res = {'FullBody': eval_body.get_results(),
               'UpperBody': eval_upper.get_results(),
               'LowerBody': eval_lower.get_results()}
        LOGGER.info(pprint.pformat(res))

    return eval_body.get_results()['All']

def main():

    # Setting Training Type
    args = parser.parse_args()
    if args.training_type == "train2d":
        print("-------------Training 2D Heatmap Model-------------")
        LOGGER = ConsoleLogger('Train2d', 'train')
        print()
    elif args.training_type == "train3d":
        print("-------------Training 3D Lifting Model-------------")
        LOGGER = ConsoleLogger('Train3d', 'train')
    elif args.training_type == 'finetune':
        print("-------------Finetuning 2D, 3D Models-------------")
        LOGGER = ConsoleLogger('Finetune', 'train')
    else:
        print("You must choose training mode between train2d, train3d, and finetune!")
        sys.exit()
    
    # Loading training, testing data
    data_transform, train_data, test_data, train_data_loader, test_data_loader = load_data()
    print("-------------Loaded Data for Training-------------")

    if args.training_type == "train2d":
        # Load Resnet101 Model for Heatmap
        with open('model/model.yaml') as fin:
            model_cfg = edict(yaml.safe_load(fin))
        resnet = pose_resnet.get_pose_net(model_cfg, True)
        Loss2D = HeatmapLoss()

        # Send Models to GPU
        if torch.cuda.is_available():
            print("GPU is available!")
            device = torch.device(int(args.gpu))
            print("GPU : " + str(device).replace("cuda:", "") + " will be used for training")
            resnet = resnet.cuda(device)
            Loss2D = Loss2D.cuda(device)
        
        # Adam optimizer with scheduler for adjusting learning rate for optimal training 
        optimizer = optim.Adam(resnet.parameters(), lr= 0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

        if args.load_model:
            print("-------------Loading pretrained model to resume training-------------")
            checkpoint = torch.load(args.load_model)
            resnet.load_state_dict(checkpoint['resnet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("-------------Loaded pretrained model-------------")

    if args.training_type == "train3d":
        autoencoder = encoder_decoder.AutoEncoder(False,True)
        LossHeatmapRecon = HeatmapLoss()
        Loss3D = PoseLoss()
        LossLimb = LimbLoss()

        # Send Models to GPU
        if torch.cuda.is_available():
            print("GPU is available!")
            device = torch.device(int(args.gpu))
            print("GPU : " + str(device).replace("cuda:", "") + " will be used for training")
            autoencoder = autoencoder.cuda(device)
            LossHeatmapRecon.cuda(device)
            Loss3D.cuda(device)
            LossLimb.cuda(device)

        optimizer = optim.Adam(autoencoder.parameters(), lr = 0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma= 0.1)
    
        if args.load_model:
            print("-------------Loading pretrained model to resume training-------------")
            checkpoint = torch.load(args.load_model)
            autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("-------------Loaded pretrained model-------------")
    
    if args.training_type == "finetune":
        with open('model/model.yaml') as fin:
            model_cfg = edict(yaml.safe_load(fin))
        resnet = pose_resnet.get_pose_net(model_cfg, True)
        Loss2D = HeatmapLoss()
        autoencoder = encoder_decoder.AutoEncoder(False, True)
        LossHeatmapRecon = HeatmapLossSquare()
        Loss3D = PoseLoss()
        LossLimb = LimbLoss()

        # Send Models to GPU
        if torch.cuda.is_available():
            print("GPU is available!")
            device = torch.device(int(args.gpu))
            print("GPU : " + str(device).replace("cuda:", "") + " will be used for training")
            resnet = resnet.cuda(device)
            Loss2D = Loss2D.cuda(device)
            autoencoder = autoencoder.cuda(device)
            LossHeatmapRecon.cuda(device)
            Loss3D.cuda(device)
            LossLimb.cuda(device)
        
        optimizer = optim.Adam(list(resnet.parameters()) + list(autoencoder.parameters()), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
        print("-------------Loading 2D, 3D Pose Estimation Models for Finetuning-------------")
        checkpoint = torch.load(args.load_2d_model, map_location = device)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        checkpoint = torch.load(args.load_3d_model, map_location = device)
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    
    if args.training_type == "train2d":
        best_model = False 
        best_perf = float('inf')
        for epoch in range(3):
            print(f"Training epoch {epoch}")
            
            resnet.train()
            for it, (img, p2d, p3d, heatmap, action) in enumerate(tqdm(train_data_loader), start=0):
                img = img.to(device)
                p2d = p2d.to(device)
                p3d = p3d.to(device)
                heatmap = heatmap.to(device)
                heatmap2d_out = resnet(img)
                loss = Loss2D(heatmap2d_out, heatmap).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % 100 == 0:
                    print(f"Loss: {loss.item()}")
                    
            scheduler.step()

            checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            print("Saving checkpoint to " + checkpoint_dir)
            states = dict()
            states['resnet_state_dict'] = resnet.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            states['scheduler'] = scheduler.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))
            resnet.eval()
            val_loss = validate_2d(LOGGER, test_data_loader, resnet, device, epoch)
            if val_loss < best_perf:
                best_perf = val_loss
                best_model = True
            
            if best_model:
                shutil.copyfile(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'), os.path.join(checkpoint_dir, f'model_best.tar'))
                best_model = False
    
    if args.training_type == "train3d":
        best_model = False 
        best_perf = float('inf')
        for epoch in range(3):
            print(f"Training epoch {epoch}")
            eval_body = evaluate.EvalBody()
            eval_upper =  evaluate.EvalUpperBody()
            eval_lower = evaluate.EvalLowerBody()
            autoencoder.train()

            for it, (img, p2d, p3d, heatmap, action) in enumerate(tqdm(train_data_loader), start = 0 ):
                img = img.to(device)
                p3d = p3d.to(device)
                heatmap = heatmap.to(device)
                p3d_out, heatmap2d_out = autoencoder(heatmap)
                loss_recon = LossHeatmapRecon(heatmap2d_out, heatmap).mean()
                loss_3d = Loss3D(p3d_out, p3d).mean()
                loss_cos, loss_len = LossLimb(p3d_out, p3d)
                loss_cos = loss_cos.mean()
                loss_len = loss_len.mean()
                loss = 0.001 * loss_recon + 0.1 * loss_3d - 0.01 * loss_cos + 0.5 * loss_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % 1000 == 0:
                    print(f"Loss: {loss.item()}")
            scheduler.step()

            checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            print("Saving checkpoint to " + checkpoint_dir)
            states = dict()
            states['autoencoder_state_dict'] = autoencoder.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            states['scheduler'] = scheduler.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))
            autoencoder.eval()
            val_loss = validate_3d(LOGGER, test_data_loader, autoencoder, device, epoch)
            if val_loss < best_perf:
                best_perf = val_loss
                best_model = True
            
            if best_model:
                shutil.copyfile(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'), os.path.join(checkpoint_dir, f'model_best.tar'))
                best_model = False
        
    if args.training_type == "finetune":
        best_perf = float('inf')
        best_model = False
        for epoch in range(3):
            resnet.train()
            autoencoder.train()
            
            for it, (img, p2d, p3d, heatmap, action) in enumerate(tqdm(train_data_loader), 0):
                img = img.to(device)
                p3d = p3d.to(device)
                heatmap = heatmap.to(device)
                heatmap2d_out = resnet(img)
                p3d_out, heatmap2d_recon = autoencoder(heatmap2d_out)
                loss2d = Loss2D(heatmap2d_out, heatmap).mean()
                loss_recon = LossHeatmapRecon(heatmap2d_recon, heatmap2d_out).mean()
                loss_3d = Loss3D(p3d_out, p3d).mean()
                loss_cos, loss_len = LossLimb(p3d_out, p3d)
                loss_cos = loss_cos.mean()
                loss_len = loss_len.mean()
                loss = loss2d + 0.001 * loss_recon + 0.1 * loss_3d - 0.01 * loss_cos + 0.5 * loss_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % 1000 == 0:
                    print(f"Loss: {loss.item()}")
            scheduler.step()  

            checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            print("Saving checkpoint to " + checkpoint_dir)
            states = dict()
            states['resnet_state_dict'] = resnet.state_dict()
            states['autoencoder_state_dict'] = autoencoder.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            states['scheduler'] = scheduler.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))
            resnet.eval()
            autoencoder.eval()
            val_loss = validate_finetune(LOGGER, test_data_loader,resnet, autoencoder, device, epoch)
            if val_loss < best_perf:
                best_perf = val_loss
                best_model = True
            if best_model:
                shutil.copyfile(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'), os.path.join(checkpoint_dir, f'model_best.tar'))
                best_model = False

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, num=1):
        self.sum += val * num
        self.val = val
        self.count += num
        self.avg = self.sum / self.count if self.count!=0 else 0.0

if __name__ == "__main__":
    main()
