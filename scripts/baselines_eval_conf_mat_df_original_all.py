from object_pose_utils.utils.confusion_matrix_builder import ConfMatrixEstimator
from object_pose_utils.utils.temporal_filtering_framework import TempFilterEstimator
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

# Return True if the subvideo is not complete or has invalid data
def invalid_sample(data, video_len):
    if len(data) < video_len:
        return True
    
    for i in range(0, len(data)):
        points, choose, img, target, model_points, idx, quat = data[i]
        if len(idx) == 0:
            return True

    return False
    
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
                             #preprocessors = [YCBOcclusionAugmentor(dataset_root), ColorJitter()],
                             #postprocessors = [ImageNormalizer(), PointShifter()],
                             postprocessors = [ImageNormalizer()],
                             image_size = [640, 480], num_points=1000)
    
    ycb_video_dataset = YCBVideoDataset(ycb_dataset, interval, video_len)
    
    ycb_video_dataset.setObjectId(obj_id)
    eval_result = {}
    
    for v_id in ycb_video_dataset.getVideoIds():
        ycb_video_dataset.setVideoId(v_id)
    
        dataloader = torch.utils.data.DataLoader(ycb_video_dataset, batch_size=1, shuffle=False, num_workers=0)
    
        angular_error_list = []
        log_likelihood_list = []
        
        for i, (data, trans) in enumerate(dataloader, 0):
            # In each video, for each subvideo, perform the evaluation
            points, choose, img, target, model_points, idx, quat = data[0]
            if invalid_sample(data, video_len):
                print("Obj: {0}, Video: {1}, Subvideo: {2} gives nans.".format(str(obj_id), v_id, str(i)))
                angular_error_list.append(np.nan)
                log_likelihood_list.append(np.nan)
                continue
            
            q_gt = quat[0]

            estimator.fit(data, trans)
           
            lik = estimator.likelihood(q_gt)
            
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

#----- Evaluation code on Confusion Matrix -----
obj_list = list(range(1,22))
interval = 0
video_len = 1

my_path = os.path.abspath(os.path.dirname(__file__))
precomputed_path_prefix = os.path.join(my_path, "..", "precomputed")

err = {}
lik = {}

#result_file = open("all_conf_result.txt", "w") 
for obj_id in obj_list:
    err_dict = {}
    lik_dict = {}
    
    conf_matrix = np.load(os.path.join(precomputed_path_prefix, "all_confusion_matrices", "{0}_confusion_matrix.npy".format(str(obj_id))))
    conf_matrix_estimator = ConfMatrixEstimator(conf_matrix)
    conf_matrix_eval_result = evaluate_estimator(conf_matrix_estimator, obj_id, interval, video_len)

    print("----- Evaluation result of the confusion matrix method -----")
    for video_id, result in conf_matrix_eval_result.items():
        angular_error_list, log_likelihood_list = result
        #avg_error = np.average(angular_error_list)
        #avg_lik = np.average(log_likelihood_list)

        err_dict[video_id] = angular_error_list
        lik_dict[video_id] = log_likelihood_list
        
        result_text = "Obj: {0}, Video: {1}, Done.\n".format(str(obj_id), video_id)
        print(result_text)
        
    err[obj_id] = err_dict
    lik[obj_id] = lik_dict
#result_file.close()
np.savez("all_conf_result_nan.npz", err=err, lik=lik)
