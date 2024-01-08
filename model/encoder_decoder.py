import torch.nn as nn
import torch
from utils import config


class AutoEncoder(nn.Module):
    def __init__(self, bn=False, denis_activation=True):
        super().__init__()
        self.bn = bn
        self.denis_activation = denis_activation
        self.conv1 = nn.Conv2d(len(config.skel)-1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fc1 = nn.Linear(18432, 2048)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Linear(2048, 512)
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc3 = nn.Linear(512, 20)
        self.leaky_relu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # branch 1
        self.b1_fc1 = nn.Linear(20, 32)
        self.b1_leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b1_fc2 = nn.Linear(32, 32)
        self.b1_leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b1_fc3 = nn.Linear(32, 3*len(config.skel))

        # branch 2
        self.b2_fc1 = nn.Linear(20, 512)
        self.b2_leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b2_fc2 = nn.Linear(512, 2048)
        self.b2_leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b2_fc3 = nn.Linear(2048, 18432)
        self.b2_leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.b2_deconv1 = nn.ConvTranspose2d(512, 128, 4, 2, padding=1)
        self.b2_relu4 = nn.ReLU()
        self.b2_deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1)
        self.b2_relu5 = nn.ReLU()
        self.b2_deconv3 = nn.ConvTranspose2d(64, len(config.skel)-1, 4, 2, padding=1)
        self.b2_relu6 = nn.ReLU()

        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # encoder
        z = self.leaky_relu1(self.conv1(x))
        z = self.leaky_relu2(self.conv2(z))
        z = self.leaky_relu3(self.conv3(z))
        z = z.view(-1, 18432)
        z = self.leaky_relu4(self.fc1(z))
        z = self.leaky_relu5(self.fc2(z))
        z = self.fc3(z)

        # decoder--branch 1
        pose_3d = self.b1_fc1(z)
        pose_3d = self.b1_fc2(pose_3d)
        pose_3d = self.b1_fc3(pose_3d)
        pose_3d = pose_3d.view(-1, len(config.skel), 3)

        # decoder--branch 2
        heatmap = self.b2_leaky_relu1(self.b2_fc1(z))
        heatmap = self.b2_leaky_relu2(self.b2_fc2(heatmap))
        heatmap = self.b2_leaky_relu3(self.b2_fc3(heatmap))
        heatmap = heatmap.view(-1, 512, 6, 6)
        heatmap = self.b2_relu4(self.b2_deconv1(heatmap))
        heatmap = self.b2_relu5(self.b2_deconv2(heatmap))
        heatmap = self.b2_deconv3(heatmap)



        return pose_3d, heatmap













