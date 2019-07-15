from object_pose_utils.utils.confusion_matrix_builder import ConfMatrixEstimator
from object_pose_utils.utils.temporal_filtering_framework_posecnn import TempFilterEstimator
import torch
import numpy as np
from dense_fusion.network import PoseNet
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_video_dataset import YcbVideoDataset as YCBVideoDataset
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.image_processing import ImageNormalizer
from object_pose_utils.utils.multi_view_utils import get_prediction
from object_pose_utils.utils.pose_processing import meanShift,tensorAngularDiff
from generic_pose.utils import to_np
import os.path
from torch.autograd import Variable
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform
from object_pose_utils.datasets.inplane_rotation_augmentation import InplaneRotator
import scipy.io as scio

# Return True if the subvideo is not complete or has invalid data
def invalid_sample(data, video_len):
    if len(data) < video_len:
        return True

    for i in range(0, len(data)):
        points, choose, img, target, model_points, idx, quat = data[i]
        if len(idx) == 0:
            return True
    return False

def getPoseCNNQuat(data, obj):
    pose_idx = np.where(data['rois'][:,1].flatten()==obj)[0]
    if(len(pose_idx) == 0):
        #import IPython; IPython.embed()
        return None
    else:
        pose_idx = pose_idx[0]
        pose = data['poses'][pose_idx]
        q = pose[:4][[1,2,3,0]]
        q /= np.linalg.norm(q)
        t = pose[4:7]
        return q
                                                    
def evaluate_estimator(estimator, obj_id, interval, video_len):
    dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb/YCB_Video_Dataset/'
    mode = 'test'
    object_list = [obj_id]
    model_checkpoint = '/home/mengyunx/DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
    output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                     otypes.IMAGE_CROPPED,
                     otypes.MODEL_POINTS_TRANSFORMED,
                     otypes.MODEL_POINTS,
                     otypes.OBJECT_LABEL,
                     otypes.QUATERNION]

    preprocessors = [InplaneRotator(theta=np.pi/2)]
    postprocessors = [ImageNormalizer()]
    
    num_objects = 21
    num_points = 1000

    pose_estimator = PoseNet(num_points = num_points,
                                            num_obj = num_objects)
    pose_estimator.load_state_dict(torch.load(model_checkpoint))
    pose_estimator.cuda()
    
    ycb_dataset = YCBDataset(dataset_root, mode=mode,
                             object_list = object_list,
                             output_data = output_format,
                             resample_on_error = True,
                             add_syn_background = False,
                             add_syn_noise = False,
                             preprocessors = preprocessors,
                             postprocessors = postprocessors,
                             image_size = [640, 480], num_points=1000)
    
    ycb_video_dataset = YCBVideoDataset(ycb_dataset, interval, video_len)
    
    ycb_video_dataset.setObjectId(obj_id)
    eval_result = {}
    
    for v_id in ycb_video_dataset.getVideoIds():
        ycb_video_dataset.setVideoId(v_id)
    
        dataloader = torch.utils.data.DataLoader(ycb_video_dataset, batch_size=1, shuffle=False, num_workers=20)
    
        angular_error_list = []
        log_likelihood_list = []
    
        for i, (data, trans) in enumerate(dataloader, 0):
            # In each video, for each subvideo, perform the evaluation
            points, choose, img, target, model_points, idx, quat = data[0]

            # If any frame is invalid or the sample has < 3 length, skip evaluating the whole subvideo
            if invalid_sample(data, video_len):
                print("Obj: {0}, Video: {1}, Subvideo: {2} gives nans.".format(str(obj_id), v_id, str(i)))
                angular_error_list.append(np.nan)
                log_likelihood_list.append(np.nan)
                continue


            count = 0
            return_nan = False
            for path in ycb_video_dataset.getPaths(i):
                pcnn_mat = scio.loadmat('{}/data/{}-posecnn.mat'.format(ycb_dataset.dataset_root, path))
                tmp = getPoseCNNQuat(pcnn_mat, idx.item())
                
                if tmp is None:
                    #import IPython; IPython.embed()
                    return_nan = True
                    break
                est_quat = torch.tensor(getPoseCNNQuat(pcnn_mat, idx.item())).float()
                
                data[count] = (data[count], est_quat)
                count += 1
                
            if return_nan:
                print("Obj: {0}, Video: {1}, Subvideo: {2} gives nans.".format(str(obj_id), v_id, str(i)))
                angular_error_list.append(np.nan)
                log_likelihood_list.append(np.nan)
                continue
            
            
            
            q_gt = quat[0]
                    
            estimator.fit(data, trans)

            lik = estimator.likelihood(q_gt.unsqueeze(0).cuda())
            q_est = estimator.mode()

            # Calculate angular error
            angular_error_degrees = to_np(tensorAngularDiff(torch.tensor(q_est).float().cuda(), quat.cuda()))*180/np.pi
            # Calculate log likelihood
            log_likelihood = np.log(lik)
                        
            angular_error_list.append(angular_error_degrees)
            log_likelihood_list.append(log_likelihood)
            if i % 100 == 0:
                print("Subvideo {0} has been evaluated".format(i))
        eval_result[v_id] = (angular_error_list, log_likelihood_list)
        print("Video {0} is evaluated".format(v_id))
    return eval_result

# ----- Evaluation code on Kalman Filter -----
obj_list = list(range(1,22))
interval = 4
video_len = 3

my_path = os.path.abspath(os.path.dirname(__file__))
precomputed_path_prefix = os.path.join(my_path, "..", "precomputed")

err = {}
lik = {}

#result_file = open("kalman_result.txt", "w")
for obj_id in obj_list:
  
    # ----- The below chunk is for the bingham method
    
    temp_filter_estimator_bingham = TempFilterEstimator()
    temp_filter_eval_result_bingham = evaluate_estimator(temp_filter_estimator_bingham, obj_id, interval, video_len)

    error_dict = {}
    lik_dict = {}
        
    print("----- Evaluation result of the temporal filtering with Bingham method -----")
    for video_id, result in temp_filter_eval_result_bingham.items():
        angular_error_list, log_likelihood_list = result
        #avg_error = np.average(angular_error_list)
        #avg_lik = np.average(log_likelihood_list)

        error_dict[video_id] = angular_error_list
        lik_dict[video_id] = log_likelihood_list
        
        result_text = "Obj: {0}, Video: {1}, Done.\n".format(str(obj_id), video_id)
        print(result_text)
        #result_file.write(result_text)

    err[obj_id] = error_dict
    lik[obj_id] = lik_dict
#result_file.close()
np.savez("kalman_posecnn_4_3_rot.npz", err=err, lik=lik)
    # ----- End of the bingham chunk
