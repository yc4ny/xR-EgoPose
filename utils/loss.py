# Code adapted from @FloralZhao
import torch
import torch.nn as nn
import numpy as np


class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2).mean(dim=(-1, -2, -3))
        return loss

class HeatmapLossSquare(nn.Module):
    def __init__(self):
        super(HeatmapLossSquare, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt) ** 2)
        l = l.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        return l  ## l of dim bsize


class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
    def forward(self, pred, gt):
        l = (pred - gt) ** 2
        l = torch.sqrt(torch.sum(l, dim=-1)).mean(-1)
        return l  ## l of dim bsize


class LimbLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_point = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])
        self.end_point = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5]) 

    def forward(self, pred, gt):
        p, p_hat = self.getBones(pred, gt)
        CosineSimilarity = nn.CosineSimilarity(dim=2)  
        theta = CosineSimilarity(p, p_hat).sum(dim=-1) 

        R = torch.norm(p-p_hat, dim=-1)
        R = R.sum(dim=-1) 

        return theta, R

    def getBones(self, pred, gt):
        batch_size = pred.size(0)
        limb_num = self.start_point.shape[0]
        limb = torch.zeros(batch_size, limb_num, 3).to(gt.device)
        limb_hat = torch.zeros(batch_size, limb_num, 3).to(pred.device)
        for i in range(limb_num):
            limb[:, i, :] = gt[:, self.end_point[i], :] - gt[:, self.start_point[i], :]
            limb_hat[:, i, :] = pred[:, self.end_point[i], :] - pred[:, self.start_point[i], :]

        return limb, limb_hat


