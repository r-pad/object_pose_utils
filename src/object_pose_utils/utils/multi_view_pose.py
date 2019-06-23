import torch
import os
import numpy as np
from object_pose_utils.utils.pose_processing import tensorAngularDiff
from generic_pose.utils import to_np, to_var
from torch.autograd import Variable
# Import video dataloader
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform
from quat_math.rotation_averaging import projectedAverageQuaternion
from quat_math.quat_math import quatAngularDiff
from quat_math import quaternion_from_matrix

# Import estimator
from dense_fusion.evaluate import DenseFusionEstimator
from dense_fusion.data_processing import getYCBGroundtruth
from dense_fusion.network import PoseNet

# Given a video and specified parameters, return the multi-view estimated pose for the initial frames of the subvideos, and their distance from the ground truth. The pose is estimated with averaged quaternion method
def estimate_average_quaternion(dataset_root, mode, model_checkpoint, video_num, sample_num, sample_interval, start_index, start_index_increment, object_label):
    
    num_points = 1000
    num_obj = 21

    estimator = PoseNet(num_points = num_points, num_obj = num_obj)
    estimator.load_state_dict(torch.load(model_checkpoint))
    estimator.cuda()

    # Create the ycb video dataloader to load samples
    dataset_video = YcbVideoDataset(dataset_root, mode, object_label, sample_interval, sample_num, video_num)

    print("ycb_video has {0} entries".format(dataset_video.getLen()))


    video_len = dataset_video.getLen()
    
    averaged_quaternion_dict = {}
    total_distance_dict = {}

    # Sample sample_num frames with sample_interval apart, starting at start_index
    while start_index < video_len:
        sample_camera_poses = dataset_video.getCameraTransforms(start_index)
        initial_camera_pose = sample_camera_poses[0]

        # Store the camera transform from the initial camera pose to the sampled camera poses
        camera_transform_list = []
        for current in range(1, len(sample_camera_poses)):
            current_camera_pose = sample_camera_poses[current]
            camera_transform = computeCameraTransform(initial_camera_pose, current_camera_pose)
            camera_transform_list.append(camera_transform)


        # Estimate the pose for the sampled frames with DenseFusion with refinement
        outputs_list = dataset_video.getItem(start_index)
    
        distance = []
        predicted_poses = []
        ground_truth_quaternions = []

    
        # For each sequence of samples, apply camera poses to get the final frame's predicted pose 
        for i in range(0, len(outputs_list)):
            points, choose, img, target, model_points, idx, quat = outputs_list[i]
            img = img.unsqueeze(0)
            points = points.unsqueeze(0)
            choose = choose.unsqueeze(0)
            idx = idx.unsqueeze(0)

            idx = idx - 1
            ground_truth_quaternions.append(quat)

            quat = quat.unsqueeze(0)
        
            points, choose, img, target, model_points, idx = Variable(points).cuda(), Variable(choose).cuda(), Variable(img).cuda(), Variable(target).cuda(), Variable(model_points).cuda(), Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
            pred_q /= pred_q.norm()
            distance.append(to_np(tensorAngularDiff(pred_q, quat.cuda()))*180/np.pi)
        
            predicted_poses.append(pred_q)
    
        # Transform every non-first sampled frame to the first sampled frame
    
        predicted_initial_pose = predicted_poses[0]
        predicted_initial_pose_data = predicted_poses[0].cpu().detach().numpy()
        transformed_pose_list = applyTransform(predicted_poses[1:], camera_transform_list)
        transformed_pose_list.insert(0, predicted_initial_pose_data)

        averaged_pose = projectedAverageQuaternion(transformed_pose_list)
        # Difference is in degrees
        difference = to_np(tensorAngularDiff(ground_truth_quaternions[0].unsqueeze(0), torch.from_numpy(averaged_pose).float()))*180/np.pi

        averaged_quaternion_dict[start_index] = averaged_pose
        total_distance_dict[start_index] = difference
        start_index += start_index_increment
    return averaged_quaternion_dict, total_distance_dict
