# -*- coding: utf-8 -*-
"""
Created on a Thursday long long ago

@author: bokorn
"""

import os
import copy
import glob
import random
import numpy as np
import numpy.ma as ma
from PIL import Image

class YCBOcclusionAugmentor(object):
    
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.front_num = 2
        image_filenames = sorted(glob.glob(os.path.join(dataset_root, 'data_syn/')+'*-color.png'))
        self.syn = [os.path.relpath(fn, self.dataset_root).split('-')[0] for fn in image_filenames]
        assert len(self.syn) > 0, 'Occlususion image folder must contain atleast one image'

    def __call__(self, meta_data, img, depth, points): 
        meta_data_occ = copy.deepcopy(meta_data)
        label = meta_data_occ['mask']
        for k in range(5):
            subpath = random.choice(self.syn)
            front = np.array(Image.open('{0}/{1}-color.png'.format(self.dataset_root, subpath)).convert("RGB"))
            f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.dataset_root, subpath)))
            front_label = np.unique(f_label).tolist()[1:]
            if len(front_label) < self.front_num:
               continue
            front_label = random.sample(front_label, self.front_num)
            for f_i in front_label:
                mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                if f_i == front_label[0]:
                    mask_front = mk
                else:
                    mask_front = mask_front * mk
            t_label = label * mask_front
            if len(t_label.nonzero()[0]) > 1000:
                label = t_label
                break

        mask_front = np.expand_dims(mask_front, 2)
        img_occ = img * mask_front + front * ~mask_front
        meta_data_occ['mask'] = label
        return meta_data_occ, img_occ, depth, points
