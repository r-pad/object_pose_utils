import torch
import numpy as np

from torch.autograd import Variable
from generic_pose.utils import to_np
from object_pose_utils.utils.pose_processing import quatAngularDiffBatch
from dense_fusion.network import PoseNet


class PoseConfusionMatrix(object):
    def __init__(self, bins, estimator, dataloader = None):
        # Input: bins = [pose1, pose2, ..., poseN] that cover the pose space
        #        dataloader = any defined dataloader in the object_util repo
        self.dataloader = dataloader
        self.bins = bins
        self.bin_size = bins.shape[0]
        self.confusion_matrix = np.zeros((self.bin_size, self.bin_size))

        print("Start initializing the confusion matrix")
        
        if dataloader != None:
            #distance = []
            for i,data in enumerate(dataloader, 0):
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
                #distance.append(to_np(tensorAngularDiff(pred_q, quat.cuda())))

                self.add_entry(quat[0], pred_q)
                if i % 100 == 0:
                    print("Image {0} / {1} has been processed".format(i, dataloader.__len__()))
                
        print("confusion matrix initialized")
    # Add an entry into the confusion matrix
    def add_entry(self, true_pose, predicted_pose):
        col = self.get_index(true_pose)
        row = self.get_index(predicted_pose)
        self.confusion_matrix[row, col] += 1

        return
    
    # Map the pose to the closest bin
    def get_index(self, pose):
        dists = quatAngularDiffBatch(to_np(pose), self.bins)
        bin_id = np.argmin(dists)
        return bin_id

    # Return a distribution of the possible actual poses
    def get_conditional_probability(self, pose):
        row = self.get_index(pose)
        counts = self.confusion_matrix[row]
        if np.sum(counts) == 0:
            counts[row] = 1.0
        else:    
            counts = counts / np.sum(counts)
        return counts
