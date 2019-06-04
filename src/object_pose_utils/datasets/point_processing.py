# -*- coding: utf-8 -*-
"""
Created on a Thursday long long ago

@author: bokorn
"""

import torch
import numpy as np
import random
from object_pose_utils.datasets.pose_dataset import MODEL_POINT_OUTPUTS, DEPTH_POINT_OUTPUTS

class PointShifter(object):
    def __init__(self, noise_trans = 0.03):
        self.noise_trans = noise_trans

    def __call__(self, outputs, meta_data, output_types):
        res = []
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        add_t = torch.Tensor(add_t.astype(np.float32))
        for x, ot in zip(outputs, output_types):
            if(ot in MODEL_POINT_OUTPUTS or ot in DEPTH_POINT_OUTPUTS):
                x = x + add_t 
            res.append(x)
        return res

