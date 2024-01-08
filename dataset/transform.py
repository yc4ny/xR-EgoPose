# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Transformation to apply to the data

@author: Denis Tome'

"""
import torch
import numpy as np
from base import BaseTransform
from utils import config
from torchvision import transforms as transforms
from skimage.transform import resize


class ImageTrsf(BaseTransform):
    """Image Transform"""

    # def __init__(self, mean=None, std=None):
    #
    #     super().__init__()
    #     if mean is None:
    #         self.mean = [0.485, 0.456, 0.406]
    #     else:
    #         self.mean = mean
    #     if std is None:
    #         self.std = [0.229, 0.224, 0.225]
    #     else:
    #         self.std = std

    def __init__(self, mean=0.5, std=0.5):

        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        if 'image' not in list(data.keys()):
            return data

        # get image from all data
        img = data['image']
        img = resize(img, config.data.image_size, anti_aliasing=True)  # convert to scale [0, 1]

        # normalization
        img -= self.mean
        img /= self.std

        # channel last to channel first
        img = np.transpose(img, [2, 0, 1])

        data.update({'image': img})

        return data


class Joints3DTrsf(BaseTransform):
    """Joint Transform"""

    def __init__(self):

        super().__init__()
        joint_zeroed = config.transforms.norm   # Neck

        assert joint_zeroed in config.skel.keys()
        self.jid_zeroed = config.skel[joint_zeroed].jid

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        if 'joints3D' not in list(data.keys()):
            return data

        p3d = data['joints3D']
        joint_zeroed = p3d[self.jid_zeroed][np.newaxis]  # add one axis

        # centered at neck
        # update p3d
        p3d -= joint_zeroed
        data.update({'joints3D': p3d})

        return data


class ToTensor(BaseTransform):
    """Convert ndarrays to Tensors."""

    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """

        keys = list(data.keys())
        for k in keys:
            pytorch_data = torch.from_numpy(data[k]).float()
            data.update({k: pytorch_data})

        return data
