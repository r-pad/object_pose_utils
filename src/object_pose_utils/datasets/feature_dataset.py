# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:38:19 2018

@author: bokorn
"""
import numpy as np
import torch 
import os

from object_pose_utils.datasets.pose_dataset import PoseDataError

from object_pose_utils.datasets.uniform_ycb_dataset import UniformYCBDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset

class FeatureDataset(YCBDataset):
    def __init__(self, feature_root, *args, **kwargs):
        super(FeatureDataset, self).__init__(output_data = [], 
                                             *args, **kwargs)

        self.feature_root = feature_root

    def __getitem__(self, index):
        try:
            meta_data = self.getMetaData(index)
            obj = meta_data['object_label']
            data = np.load(os.path.join(self.feature_root, 'data', self.getPath(index)) + 
                    '_{}_feat.npz'.format(self.classes[obj]))
        except (IOError, PoseDataError)  as e:
            print('Exception on index {}: {}'.format(index, e))
            if(self.resample_on_error):
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                return [], [], []

        obj = torch.LongTensor([obj])
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data['feat'].astype(np.float32))
        return obj, feat, quat

class UniformFeatureDataset(UniformYCBDataset):
    def __init__(self, feature_root, *args, **kwargs):
        super(UniformFeatureDataset, self).__init__(output_data = [], 
                                                    *args, **kwargs)

        self.feature_root = feature_root

    def __getitem__(self, index):
        try:
            data = np.load(os.path.join(self.feature_root, self.getPath(index)) + 
                    '_{}_feat.npz'.format(self.classes[self.object_label]))
        except (IOError, PoseDataError)  as e:
            print('Exception on index {}: {}'.format(index, e))
            if(self.resample_on_error):
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                return [], [], []

        obj = torch.LongTensor([self.object_label])
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data['feat'].astype(np.float32))
        return obj, feat, quat
