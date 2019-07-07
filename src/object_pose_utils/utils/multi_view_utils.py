import numpy as np
from quat_math import quaternion_from_matrix, quaternion_multiply
from torch.autograd import Variable
import torch
from object_pose_utils.utils.pose_processing import tensorMultiplyBatch

# Apply each camera transform to each pose
def applyTransform(poses, cameraTransforms):
    # Input: poses: non-first frame poses (quaternion)
    #        cameraTransforms: the camera transforms from the first frame to the ith frame 3x4
    assert len(poses) == len(cameraTransforms), "inputs don't have the same length"
    transformed = []
    for i in range(0, len(poses)):
        current_pose = poses[i]
        camera_transform = cameraTransforms[i][0]
        R1c = camera_transform[:3, :3].transpose(0,1)
        R1c_padded = np.identity(4)
        R1c_padded[:3, :3] = R1c
        R1c_quat = quaternion_from_matrix(R1c_padded)
        first_frame_pose = quaternion_multiply(R1c_quat, current_pose.cpu().detach().numpy())
        transformed.append(first_frame_pose)

    return transformed

# Apply each camera transform to each pose
def applyTransformBatch(poses, camera_transform):
    R1c = camera_transform[:3, :3].transpose()
    R1c_padded = np.identity(4)
    R1c_padded[:3, :3] = R1c
    trans_quat = torch.tensor(quaternion_from_matrix(R1c_padded)).unsqueeze(0).repeat(poses.shape[0],1).float()
    if poses.is_cuda:
        trans_quat = trans_quat.cuda()
    transformed = tensorMultiplyBatch(trans_quat, poses)
    return transformed


# Compute the camera transform matrix from the initial frame to the current frame
def computeCameraTransform(initial_camera_matrix, current_camera_matrix):
    # Initial camera matrix = from world to initial frame (Tiw) 3x4
    # current camera matrix = from world to current frame (Tcw) 3x4
    Rcw = current_camera_matrix[:3, :3]
    tcw = current_camera_matrix[:3, 3]
    Riw = initial_camera_matrix[:3, :3]
    tiw = initial_camera_matrix[:3, 3]

    Rci = np.matmul(Rcw, Riw.transpose())
    tci = tcw - np.matmul(Rci, tiw)

    Tci = np.eye(4)
    Tci[:3, :3] = Rci
    Tci[:3, 3] = tci

    return Tci

# Given an estimator and input data, output (normalized predicted pose, ground truth pose)
def get_prediction(estimator, data):
    points, choose, img, target, model_points, idx, quat = data
    idx = idx - 1
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
    pred_q /= pred_q.norm()
    
    return pred_q, quat[0]
