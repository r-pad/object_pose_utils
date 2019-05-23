# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:38:19 2018

@author: bokorn
"""
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import os
import pickle

from object_pose_utils.datasets.pose_dataset import PoseDataError

from object_pose_utils.datasets.uniform_dataset import UniformYCBDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset

class FeatureDataset(YCBDataset):
    def __init__(self, feature_path, *args, **kwargs):
        super(FeatureDataset, self).__init__(output_data = [], 
                                                    *args, **kwargs)

        self.feature_path = feature_path

    def __getitem__(self, index):
        obj = torch.LongTensor([self.object_label])
        data = np.load(self.feature_path + self.getPath(index))
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data['feat'].astype(np.float32))
        return obj, feat, quat

class UniformFeatureDataset(UniformYCBDataset):
    def __init__(self, feature_path, *args, **kwargs):
        super(UniformFeatureDataset, self).__init__(output_data = [], 
                                                    *args, **kwargs)

        self.feature_path = feature_path

    def __getitem__(self, index):
        meta_data = self.getMetaData(index)
        obj = torch.LongTensor([meta_data['object_label']])
        data = np.load(self.feature_path + self.getPath(index))
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data['feat'].astype(np.float32))
        return obj, feat, quat
