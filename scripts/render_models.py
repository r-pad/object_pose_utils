# -*- coding: utf-8 -*-
"""
Created on a beautiful Sunday
@author: bokorn
"""
import bpy

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image
import pathlib

from model_renderer.pose_renderer import BpyRenderer
from generic_pose.datasets.ycb_dataset import ycbRenderTransform

from quat_math import euler_matrix, quaternion_matrix, random_quaternion
from dense_fusion.evaluate_likelihood import getYCBClassData
from object_pose_utils.bbTrans.discretized4dSphere import S3Grid
from tqdm import tqdm
import argparse

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def getYCBTransform(q, t=[0,0,.5]):
    trans_mat = quaternion_matrix(q)
    ycb_mat = euler_matrix(-np.pi/2,0,0)
    trans_mat = trans_mat.dot(ycb_mat)
    trans_mat[:3,3] = t
    return trans_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Data render output location')
    parser.add_argument('--model_list', type=str, default =  'datasets/ycb/YCB_Video_Dataset',
        help='Dataset root dir (''YCB_Video_Dataset'')')
    parser.add_argument('--models', type=str, nargs='+', help='List of model files or text file containing model file names')
    parser.add_argument('--quats', type=str, default = None, help='\'grid\' for 3885 grid or N for random quaternions')
    parser.add_argument('--name_start_idx', type=int, default = -2, help='Path depth for start of model name')
    parser.add_argument('--name_end_idx', type=int, default = -1, help='Path depth for end of model name')

    args = parser.parse_args()

    if(len(args.models) == 1 and args.models[0][-4:] == '.txt'):
        with open(args.models[0]) as f:
            args.models = f.readlines()
        args.models = [x.strip() for x in args.models]

    if(args.quats == 'grid'):
        grid = S3Grid(2)
        grid.Simplify()
        quats = grid.vertices
    elif(is_int(args.quats)):
        quats = [random_quaternion() for _ in range(int(args.quats))]
    elif(args.quats[-4:] == '.txt'):
        with open(args.quats) as f:
            quats = f.readlines()
        for j, q in enumerate(quats):
            quats[j] = np.array(q, dtype=float)
            quats[j] /= np.linalg.norm(quats[j])
    else:
        raise ValueError('Bad quaternion format. Valid formats are \'grid\', N, or [file].txt')

    digits = len(str(len(quats)))

    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109

    pbar_model = tqdm(args.models)
    for model_filename in pbar_model:
        model = '/'.join(model_filename.split('/')[args.name_start_idx:args.name_end_idx])
        pbar_model.set_description('Rendering {}'.format(model))
        renderer = BpyRenderer(transform_func = ycbRenderTransform)
        renderer.loadModel(model_filename, emit = 0.5)
        renderer.setCameraMatrix(fx, fy, px, py, 640, 480)
        renderer.setDepth()
        render_dir = '{}/{}/'.format(args.output_folder, model)

        pathlib.Path(render_dir).mkdir(parents=True, exist_ok=True)
        filename_template = render_dir + '{:0'+str(digits)+'d}-{}.{}'

        for j, q in tqdm(enumerate(quats), total=len(quats)):
            # change if there are different model ids
            obj_label = 1
            trans_mat = getYCBTransform(q, [0,0,1])
            img, depth = renderer.renderTrans(trans_mat)
            depth[depth > 1000] = 0
            depth = depth*10000
            cv2.imwrite(filename_template.format(j, 'color', 'png'), img)
            Image.fromarray(depth.astype(np.int32), "I").save(filename_template.format(j,'depth', 'png'))
            np.save(filename_template.format(j, 'trans', 'npy'), q)
            label = np.where(np.array(img[:,:,-1])==255, obj_label, 0)
            cv2.imwrite(filename_template.format(j,'label', 'png'), label)
            poses = np.zeros([3,4,1])
            poses[:3,:3,0] = quaternion_matrix(q)[:3,:3]
            poses[:3,3,0] = [0.,0.,1.]
            scio.savemat(filename_template.format(j,'meta', 'mat'),
                         {'cls_indexes':np.array([[obj_label]]),
                          'factor_depth':np.array([[10000]]),
                          'poses':poses})


if __name__=='__main__':
    main()
