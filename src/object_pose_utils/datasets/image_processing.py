# -*- coding: utf-8 -*-
"""
Created on a Thursday long long ago

@author: bokorn
"""

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from object_pose_utils.datasets.pose_dataset import IMAGE_OUTPUTS


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ImageNormalizer(object):
    def __call__(self, outputs, meta_data, output_types):
        res = []
        for x, ot in zip(outputs, output_types):
            if(ot in IMAGE_OUTPUTS):
                x = norm(x)
            res.append(x)
        return res

class ColorJitter(object):
    def __init__(self, brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05):
        self.trancolor = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, meta_data, img, depth, points): 
        img_jit = np.array(self.trancolor(Image.fromarray(img)))
        return meta_data, img_jit, depth, points

def get_bbox_label(label, image_size = None):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    if(image_size is None):
        img_width = label.shape[1]
        img_length = label.shape[0]
    else:
        img_width = image_size[1]
        img_length = image_size[0]
 
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return cmin, rmin, cmax-cmin, rmax-rmin


