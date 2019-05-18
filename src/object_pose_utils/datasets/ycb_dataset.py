# -*- coding: utf-8 -*-
"""
Created on Tue April 29 2019

@author: bokorn
"""

from object_pose_utils.datasets.pose_dataset import PoseDataset

def ycbRenderTransform(q):
    trans_quat = q.copy()
    trans_quat = tf_trans.quaternion_multiply(trans_quat, tf_trans.quaternion_about_axis(-np.pi/2, [1,0,0]))
    return viewpoint2Pose(trans_quat)

def setYCBCamera(renderer, width=640, height=480):
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    renderer.setCameraMatrix(fx, fy, px, py, width, height)

def getYCBSymmeties(obj):
    if(obj == 13):
        return [[0,0,1]], [[np.inf]]
    elif(obj == 16):
        return [[0.9789,-0.2045,0.], [0.,0.,1.]], [[0.,np.pi], [0.,np.pi/2,np.pi,3*np.pi/2]]
    elif(obj == 19):
        return [[-0.14142136,  0.98994949,0]], [[0.,np.pi]]
    elif(obj == 20):
        return [[0.9931506 , 0.11684125,0]], [[0.,np.pi]]
    elif(obj == 21):
        return [[0.,0.,1.]], [[0.,np.pi]]
    else:
        return [],[]

class YCBDataset(PoseDataset):
    def __init__(self, 
                 data_dir, 
                 image_set, 
                 obj = None, 
                 use_syn_data = False,
                 use_posecnn_masks = False,
                 *args, **kwargs):
        super(YCBDataset, self).__init__(*args, **kwargs)
        

