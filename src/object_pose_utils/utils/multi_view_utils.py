import numpy as np
from quat_math import quaternion_from_matrix, quaternion_multiply
from scipy.stats import multivariate_normal
import pybingham
from object_pose_utils.utils.interpolation import GaussianInterpolation, BinghamInterpolation
from torch.autograd import Variable
import torch

# Apply each camera transform to each pose
def applyTransform(poses, cameraTransforms):
    # Input: poses: non-first frame poses (quaternion)
    #        cameraTransforms: the camera transforms from the first frame to the ith frame 3x4
    assert len(poses) == len(cameraTransforms), "inputs don't have the same length"
    transformed = []
    for i in range(0, len(poses)):
        current_pose = poses[i]
        camera_transform = cameraTransforms[i]
        R1c = camera_transform[:3, :3].transpose()
        R1c_padded = np.identity(4)
        R1c_padded[:3, :3] = R1c
        R1c_quat = quaternion_from_matrix(R1c_padded)
        first_frame_pose = quaternion_multiply(R1c_quat, current_pose)
        transformed.append(first_frame_pose)

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

# The update function for combining two Gaussians
def update_function_gaussian(prediction, measurement):
    # Input: prediction and measurement are both quaternions
    # Output: updated state estimate (quaternion)

    # Construct Gaussian interpolations
    w = [1]
    sigma_gauss = np.pi/9
    
    interpolation_pred = GaussianInterpolation(vertices = [prediction], values = w, sigma = sigma_gauss)
    interpolation_mea = GaussianInterpolation(vertices = [measurement], values = w, sigma = sigma_gauss)

    est_mean_pred = interpolation_pred.vertices.cpu().detach().numpy().reshape((4,1))
    est_cov_pred = np.identity(4)*(interpolation_pred.sigma ** 2)

    est_mean_mea = interpolation_mea.vertices.cpu().detach().numpy().reshape((4,1))
    est_cov_mea = np.identity(4)*(interpolation_mea.sigma ** 2)
    
    step1 = np.linalg.inv(est_cov_pred+est_cov_mea)
    updated_cov = np.matmul(np.matmul(est_cov_pred, step1), est_cov_mea)
    step2 = np.matmul(np.matmul(est_cov_mea, step1), est_mean_pred)
    step3 = np.matmul(np.matmul(est_cov_pred, step1), est_mean_mea)
    updated_mean = step2 + step3

    return updated_mean


# The update function for combining two Binghams
def update_function_bingham(prediction, measurement):
    # Input: prediction and measurement are both quaternions
    # Output: updated state estimate (quaternion)

    # Construct Bingham interpolations
    w = [1]
    sigma_bingham = np.pi/9
    
    interpolation_pred = BinghamInterpolation(vertices = [prediction], values = torch.Tensor(w), sigma = sigma_bingham)
    interpolation_mea = BinghamInterpolation(vertices = [measurement], values = torch.Tensor(w), sigma = sigma_bingham)

    est_M_pred = interpolation_pred.M
    est_Z_pred = interpolation_pred.Z

    est_M_mea = interpolation_mea.M
    est_Z_mea = interpolation_mea.Z

    B1 = pybingham.Bingham(est_M_pred[:,1], est_M_pred[:,2], est_M_pred[:,3], est_Z_pred[1,1], est_Z_pred[2,2], est_Z_pred[3,3])
    B2 = pybingham.Bingham(est_M_mea[:,1], est_M_mea[:,2], est_M_mea[:,3], est_Z_mea[1,1], est_Z_mea[2,2], est_Z_mea[3,3])
    Bm = pybingham.Bingham()
    pybingham.bingham_mult(Bm, B1, B2)
    
    updated_mode = Bm.getMode()

    return updated_mode

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
