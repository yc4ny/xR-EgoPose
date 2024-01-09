# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.

@author: Denis Tome'

"""
import os
from skimage import io as sio
import numpy as np
from base import BaseDataset
from utils import config, io
from utils import config
import json
import cv2
import skimage.io as sio
from skimage.transform import resize as resize
import os
import pdb


class Mocap(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def index_db(self):

        return self._index_dir(self.path)  # data/Dataset/TrainSet

    def _index_dir(self, path):
        """Recursively add paths to the set of
        indexed files

        Arguments:
            path {str} -- folder path

        Returns:
            dict -- indexed files per root dir
        """

        indexed_paths = dict()
        sub_dirs, _ = io.get_subdirs(path)

        # if this is the last level of directory
        if set(self.ROOT_DIRS) <= set(sub_dirs):

            # get files from subdirs
            n_frames = -1

            # let's extract the rgba and json data per frame
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                _, paths = io.get_files(d_path)

                if n_frames < 0:
                    n_frames = len(paths)
                else:
                    if len(paths) != n_frames:
                        raise ValueError(
                            'Frames info in {} not matching other passes'.format(d_path))

                encoded = [p.encode('utf8') for p in paths]
                indexed_paths.update({sub_dir: encoded})

            return indexed_paths

        # initialize indexed_paths
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir: []})

        # check subdirs of path and merge info
        for sub_dir in sub_dirs:
            indexed = self._index_dir(os.path.join(path, sub_dir))

            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(indexed[r_dir])

        return indexed_paths

    def _process_points(self, data):
        """Filter joints to select only a sub-set for
        training/evaluation

        Arguments:
            data {dict} -- data dictionary with frame info

        Returns:
            np.ndarray -- 2D joint positƒions, format (J x 2)
            np.ndarray -- 3D joint positions, format (J x 3)
        """

        p2d_orig = np.array(data['pts2d_fisheye']).T
        p3d_orig = np.array(data['pts3d_fisheye']).T
        joint_names = {j['name'].replace('mixamorig:', ''): jid
                       for jid, j in enumerate(data['joints'])}

        # ------------------- Filter joints -------------------

        p2d = np.empty([len(config.skel), 2], dtype=p2d_orig.dtype)
        p3d = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)

        for jid, j in enumerate(config.skel.keys()):
            p2d[jid] = p2d_orig[joint_names[j]]
            p3d[jid] = p3d_orig[joint_names[j]]

        p3d /= self.CM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path)

        # (800, 1280, 3) -> (800, 800, 3)
        start = int((img.shape[1]-img.shape[0])/2)
        img = img[:, start:start+img.shape[0], :]

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        data = io.read_json(json_path)
        p2d, p3d = self._process_points(data)

        # p2d rescale according to img
        start = 240
        p2d[:, 0] -= start  # x-axis
        p2d[:, 0] /= (800 / config.data.image_size[1])
        p2d[:, 1] /= (800 / config.data.image_size[0])

        heatmap, _ = joint2heatmap(p2d)

        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']
            heatmap = self.transform({'heatmap': heatmap})['heatmap']
        # only use 15 heatmaps (w/o head)
        heatmap = np.delete(heatmap, 1, 0)

        return img, p2d, p3d, heatmap, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])

heatmap_size = (int(config.data.heatmap_size[0]), int(config.data.heatmap_size[1]))

# Code adapted from @FloralZhao
def joint2heatmap(p2d, sigma=2, heatmap_type='gaussian'):
    '''
    Args:
        joints: [num_joints, 3]
        sigma: std for gaussian
        heatmap_type

    Returns:
        visible(1: visible, 0: not visible)

    *********** NOTE: this function will change the value of p2d **********

    '''
    num_joints = len(config.skel)
    visible = np.ones((num_joints, 1), dtype=np.float32)

    assert heatmap_type == 'gaussian', 'Only support gaussian map now!'
    # p2d[:, 0] /= (368 / heatmap_size[1])
    # p2d[:, 1] /= (368 / heatmap_size[0])
    # p2d = p2d.astype(np.int)  # don't int here


    if heatmap_type == 'gaussian':
        heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        tmp_size = sigma*3


        for joint in range(num_joints):
            feat_stride = np.array(config.data.image_size) / np.array(config.data.heatmap_size)
            mu_x = int(p2d[joint][0] / feat_stride[1] + 0.5)
            mu_y = int(p2d[joint][1] / feat_stride[0] + 0.5)
            # mu_x = int(p2d[joint][0])
            # mu_y = int(p2d[joint][1])

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                visible[joint] = 0
                continue

            # generate Gaussian
            sz = 2 * tmp_size + 1
            x = np.arange(0, sz, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = sz // 2
            # The gaussian is not normalized, we want the cneter value to equal 1
            g = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

            heatmaps[joint][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps, visible