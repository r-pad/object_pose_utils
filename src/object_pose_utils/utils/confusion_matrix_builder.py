import torch
import numpy as np

from torch.autograd import Variable
from generic_pose.utils import to_np
from object_pose_utils.utils.pose_processing import quatAngularDiffBatch
from dense_fusion.network import PoseNet
from tqdm import tqdm_notebook as tqdm
from object_pose_utils.utils.multiscale_grid import MultiResGrid
import os.path
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform


class ConfMatrixEstimator(object):
    def __init__(self, confusion_matrix):
        # Input: pre-computed confusion matrix from validation set

        # Bins is a vector of vertex values
        self.bins = MultiResGrid(2).vertices
        self.bin_size = self.bins.shape[0]
        self.confusion_matrix = confusion_matrix

        num_points = 1000
        num_obj = 21
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_prefix = os.path.join(my_path, "..", "..", "..", "datasets")
        model_checkpoint = os.path.join(path_prefix, "DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth")
        
        # Initialize the estimator
        self.estimator = PoseNet(num_points = num_points, num_obj = num_obj)
        self.estimator.load_state_dict(torch.load(model_checkpoint))
        self.estimator.cuda()

        # Placeholder for the distribution estimated
        self.distribution = None

        
    # Fit a distribution on the data given and set self.distribution to it
    def fit(self, data, trans):
        # Input: data = a list of information bundle for all frames
        #        trans = camera transforms for transforming non-original frames into the original frame

        pre_transform_pred_q = []
        for i in range(0, len(data)):
            points, choose, img, target, model_points, idx, quat = data[i]
            idx = idx - 1
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
            pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
            pred_q /= pred_q.norm()

            pre_transform_pred_q.append(pred_q)

            #if i % 100 == 0:
            #    print("Image {0} / {1} has been processedand added to the confusion matrix".format(i, dataloader.__len__()))
            #if i == 500:
            #    break
        transformed_pose_list = applyTransform(pre_transform_pred_q, trans)
        
        # Set the predicted pose of the original frame
        self.q_est = pre_transform_pred_q[0]

        # Now fit a distribution on the original frame by combining all the transforms
        #import IPython; IPython.embed()
        self.distribution = self.confusion_matrix[self.get_index(pre_transform_pred_q[0].unsqueeze(0).cpu().detach().numpy())].copy()

        for i in range(1, len(transformed_pose_list)):
            transformed_pose = transformed_pose_list[i]
            index = self.get_index(transformed_pose)
            current_distribution = self.confusion_matrix[index, :]
            self.distribution += current_distribution
            #import IPython; IPython.embed()
            
    # Return P(q | I_origin)
    def likelihood(self, q):
        # Input: q: a unit quaternion pose

        # Smoothing constant is added to every entry of the matrix so no entry will be 0
        smoothing_constant = 1
        smoothed_distribution = self.distribution + np.ones(self.distribution.shape) * smoothing_constant
        index = self.get_index(q.unsqueeze(0).cpu().detach().numpy())
        lik = smoothed_distribution[index] / np.sum(smoothed_distribution)
        return lik

    
    # Return the mode
    def mode(self):
        max_idx = np.argmax(self.distribution)
        return self.bins[max_idx]

    
    # Combine with another distribution of the same kind
    def add(self, distribution):
        # Input: distribution: another ConfMatrixEstimator
        combined_distribution = self.distribution + distribution.distribution

        return combined_distribution
    

    # Map the pose to the closest bin
    def get_index(self, pose):
        dists = quatAngularDiffBatch(pose, self.bins)
        bin_id = np.argmin(dists)
        return bin_id
