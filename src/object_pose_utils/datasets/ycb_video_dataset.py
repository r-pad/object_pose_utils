# A wrapper class for ycb_dataset.py that supports:
# Given a video dataset, n = number of frames to sample, k = interval between sampled frames
# Output information about the sampled frames (color images, depth images, metadata, etc) with a specified start frame d.  

import numpy as np
from .pose_dataset import PoseDataset
from object_pose_utils.datasets.ycb_dataset import YcbDataset
import os
import scipy.io as scio
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer
from object_pose_utils.datasets.pose_dataset import PoseDataError

from torch.utils.data import Dataset
from object_pose_utils.utils.multi_view_utils import computeCameraTransform

class YcbVideoDataset(Dataset):
    def __init__(self, ycb_dataset,
                 interval, video_len, 
                 *args, **kwargs):
        super(YcbVideoDataset, self).__init__()

        self.interval = interval
        self.video_len = video_len
        
        self.ycb_dataset = ycb_dataset
        self.ycb_dataset.resample_on_error = False
        self.dataset_root = self.ycb_dataset.dataset_root
        self.video_indices = {}
        for j in range(len(self.ycb_dataset)):
            file_id_line, object_id = ycb_dataset.image_list[j]
            if(object_id not in self.video_indices.keys()):
                self.video_indices[object_id] = {}
            video_id = file_id_line.split('/')[-2]
            if(video_id not in self.video_indices[object_id].keys()):
                self.video_indices[object_id][video_id] = []
            self.video_indices[object_id][video_id].append(j)

        self.setObjectId(self.ycb_dataset.object_list[0])
        self.setVideoId(self.getVideoIds()[0])

    def getObjectIds(self):
        return list(self.video_indices.keys())

    def setObjectId(self, object_id):
        assert object_id in self.video_indices.keys(), \
                '{} not in {}'.format(object_id, list(self.video_indices.keys()))
        self.object_id = object_id
   
    def getVideoIds(self):
        return list(self.video_indices[self.object_id].keys())

    def setVideoId(self, video_id):
        self.index_list = self.video_indices[self.object_id][video_id]

    def getItem(self, index):
        # Input: index wrt local index in the specifide video
        outputs_list = []
        current_index = index
        count = 0 
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            try:
                outputs = self.ycb_dataset.__getitem__(global_index)
            except PoseDataError as e:
                print('Exception on index {}: {}'.format(index, e))
                return []
            outputs_list.append(outputs)
            count += 1
            current_index += self.interval
        return outputs_list

    def getCameraTransforms(self, index):
        transform_init = None
        transform_list = []
        current_index = index
        count = 0
        while current_index < len(self.index_list) and count < self.video_len:
            global_index = self.index_list[current_index]
            sub_path = self.ycb_dataset.getPath(global_index)
            path = '{0}/data/{1}-meta.mat'.format(self.dataset_root, sub_path)
            meta = scio.loadmat(path)
            rotation_translation_matrix = meta["rotation_translation_matrix"]
            if(transform_init is None):
                transform_init = rotation_translation_matrix
                transform_list.append(np.eye(4)) 
            else:
                transform_list.append(computeCameraTransform(transform_init, 
                                                             rotation_translation_matrix))
            current_index += self.interval
            count += 1

        return transform_list

    def __getitem__(self, index):
        return self.getItem(index), self.getCameraTransforms(index)

    def __len__(self):
        return len(self.index_list)
