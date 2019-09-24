# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:38:19 2018

@author: bokorn
"""
import numpy as np
import torch 
import os
import scipy.io as scio

from object_pose_utils.datasets.pose_dataset import PoseDataError

from object_pose_utils.datasets.uniform_ycb_dataset import UniformYCBDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset

from transforms3d.quaternions import quat2mat, mat2quat

def getPoseCNNQuat(data, obj):
    pose_idx = np.where(data['rois'][:,1].flatten()==obj.item())[0]
    if(len(pose_idx) == 0):
        return None, None
    else:
        pose_idx = pose_idx[0]
    pose = data['poses'][pose_idx]
    q = pose[:4][[1,2,3,0]]
    q /= np.linalg.norm(q)
    t = pose[4:7]
    return q, t

class FeatureDataset(YCBDataset):
    def __init__(self, feature_root, 
                 num_augs = 0, 
                 feature_key = 'feat', 
                 return_pred=False, 
                 *args, **kwargs):
        
        super(FeatureDataset, self).__init__(output_data = [], 
                                             *args, **kwargs)

        self.feature_root = feature_root
        self.num_augs = num_augs
        self.feature_key = feature_key
        self.return_pred = return_pred
    def __getitem__(self, index):
        try:
            meta_data = self.getMetaData(index)
            obj = meta_data['object_label']
            if(self.num_augs > 0):
                aug_idx = np.random.randint(self.num_augs)
                data = np.load(os.path.join(self.feature_root, 'data', self.getPath(index)) + 
                        '_{}_{}_feat.npz'.format(self.classes[obj], aug_idx))
            else:
                data = np.load(os.path.join(self.feature_root, 'data', self.getPath(index)) + 
                        '_{}_feat.npz'.format(self.classes[obj]))
        except (IOError, PoseDataError)  as e:
            print('Exception on index {}: {}'.format(index, e))
            if(self.resample_on_error):
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                if(self.return_pred):
                    return [], [], [], []
                else:
                    return [], [], []

        obj = torch.LongTensor([obj])
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data[self.feature_key].astype(np.float32))
        if(self.return_pred):
            if('pred_q' in data.keys()):
                pred_q = data['pred_q']
            elif('max_q' in data.keys()):
                pred_q = data['max_q']
            elif(self.feature_key == 'fc6'):
                posecnn_mat = scio.loadmat('{0}/data/{1}-posecnn.mat'.format(self.dataset_root, self.getPath(index)))
                pred_q, pred_t = getPoseCNNQuat(posecnn_mat, obj)
            else:
                raise NotImplemented()
            
            if(pred_q is None or np.abs(pred_q).sum() == 0):
                if(self.resample_on_error):
                    return self.__getitem__(np.random.randint(0, len(self)))
                else:
                    return [], [], [], []
            pred_q = torch.from_numpy(pred_q.astype(np.float32))
            return obj, feat, quat, pred_q

        return obj, feat, quat

class UniformFeatureDataset(UniformYCBDataset):
    def __init__(self, feature_root,
                 num_augs = 0,
                 feature_key = 'feat', 
                 return_pred=False,
                 *args, **kwargs):
        super(UniformFeatureDataset, self).__init__(output_data = [], 
                                                    *args, **kwargs)

        self.feature_root = feature_root
        self.num_augs = num_augs
        self.feature_key = feature_key
        self.return_pred = return_pred

    def __getitem__(self, index):
        try:
            if(self.num_augs > 0):
                aug_idx = np.random.randint(self.num_augs)
                data = np.load(os.path.join(self.feature_root, self.getPath(index)) + 
                        '_{}_{}_feat.npz'.format(self.classes[self.object_label], aug_idx))
            else:
                data = np.load(os.path.join(self.feature_root, self.getPath(index)) + 
                        '_{}_feat.npz'.format(self.classes[self.object_label]))
        except (IOError, PoseDataError)  as e:
            print('Exception on index {}: {}'.format(index, e))
            if(self.resample_on_error):
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                if(self.return_pred):
                    return [], [], [], []
                else:
                    return [], [], []

        obj = torch.LongTensor([self.object_label])
        quat = torch.from_numpy(data['quat'].astype(np.float32))
        feat = torch.from_numpy(data[self.feature_key].astype(np.float32))
        if(self.return_pred):
            if('pred_q' in data.keys()):
                pred_q = data['pred_q']
            elif('max_q' in data.keys()):
                pred_q = data['max_q']
            elif(self.feature_key == 'fc6'):
                posecnn_mat = scio.loadmat('{0}/{1}-posecnn.mat'.format(self.dataset_root, self.getPath(index)))
                pred_q, pred_t = getPoseCNNQuat(posecnn_mat, obj)
            else:
                raise NotImplemented()
            
            if(pred_q is None or np.abs(pred_q).sum() == 0):
                if(self.resample_on_error):
                    return self.__getitem__(np.random.randint(0, len(self)))
                else:
                    return [], [], [], []
                
            pred_q = torch.from_numpy(pred_q.astype(np.float32))
            return obj, feat, quat, pred_q
        
        return obj, feat, quat
