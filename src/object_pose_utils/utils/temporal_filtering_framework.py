from object_pose_utils.utils.bingham import bingham_likelihood
import numpy as np
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
import torch
from object_pose_utils.utils.multi_view_utils import applyTransform, computeCameraTransform
from object_pose_utils.utils.interpolation import BinghamInterpolation, GaussianInterpolation

class TempFilterEstimator(object):
    def __init__(self, M=None, Z=None):
        self.M = M
        self.Z = Z
        self.sigma_list = [8.383942118770577, 8.383942118770577, 4.007001106261114,
                           5.040256718720126, 27.958217393726432, 2.335158406235891,
                           9.417197731229585, 8.383942118770577, 5.040256718720126,
                           4.007001106261114, 23.581276381216973, 11.089040431254809,
                           0.6633157062106676, 5.040256718720126, 27.958217393726432,
                           0.6633157062106676, 6.71209941874535, 0.6633157062106676,
                           0.6633157062106676, 0.6633157062106676, 4.007001106261114]
    def fit(self, data, trans):
        pre_transform_pred_q = []
        trans_picked = []
        
        dataset_root = '/home/mengyunx/DenseFusion/datasets/ycb/YCB_Video_Dataset/'
        mode = 'test'
        object_list = [1]
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

        sigma_bingham = None
        for i in range(0, len(data)):

            points, choose, img, target, model_points, idx, quat = data[i]

            if len(idx) == 0:
                continue
            
            idx = idx - 1

            sigma_bingham = self.sigma_list[idx]

            points, choose, img, target, model_points, idx = Variable(points).cuda(), Variable(choose).cuda(), Variable(img).cuda(), Variable(target).cuda(), Variable(model_points).cuda(), Variable(idx).cuda()

            pred_r, pred_t, pred_c, emb = pose_estimator(img, points, choose, idx)
            
            pred_q = pred_r[0,torch.argmax(pred_c)][[1,2,3,0]]
            
            pred_q /= pred_q.norm()
            
            pre_transform_pred_q.append(pred_q)
            trans_picked.append(trans[i])
            
            #if i % 100 == 0:
            #    print("Image {0} / {1} has been processd".format(i, len(data)))
            
            #if i == 500:
            
            #    break
            
        transformed_pose_list = applyTransform(pre_transform_pred_q, trans_picked)
            
        # Set the predicted pose of the original frame
            
        self.q_origin = pre_transform_pred_q[0]
            
        w = [1]

        self.distribution = BinghamInterpolation(vertices = self.q_origin.unsqueeze(0).cpu().detach(), values = torch.Tensor(w), sigma = sigma_bingham)
        #import IPython; IPython.embed()
        M_np = self.distribution.M

        Z_np = self.distribution.Z

        #import IPython; IPython.embed()

        self.M = M_np[0]
        self.Z = Z_np
        #import IPython; IPython.embed()
        for i in range(1, len(transformed_pose_list)):
            transformed_pose = transformed_pose_list[i]
            current_distribution = BinghamInterpolation(vertices = [transformed_pose], values = torch.Tensor(w).cuda(), sigma = sigma_bingham)

            M = current_distribution.M[0]
            Z = current_distribution.Z
            new_M, new_Z = self.add(M, Z)
            self.M = new_M
            self.Z = new_Z
            #if i % 100 == 0:
            #    print("Transform {0} / {1} has been processd".format(i, len(transformed_pose_list)))
        return
    
        
    def mode(self):
        return self.M[:, 0]

    def likelihood(self, q):
        return bingham_likelihood(self.M.unsqueeze(0), self.Z.unsqueeze(0), q)

    def add(self, M, Z):
        first_part = torch.mm(torch.mm(self.M, self.Z), self.M.transpose(0,1))
        second_part = torch.mm(torch.mm(M, Z), M.transpose(0,1))
        w, v = torch.eig(first_part + second_part, eigenvectors=True)
        w = w[:, 0]
        order = np.argsort(-(w.numpy()))
        new_M = torch.zeros(self.M.shape)
        new_Z = torch.zeros(self.Z.shape)
        for i in range(0, order.size):
            new_M[:, i] = v[:, order[i]]
            new_Z[i, i] = w[order[i]]
        new_Z = new_Z - new_Z[0,0] * torch.eye(new_Z.shape[0])

        self.M = new_M
        self.Z = new_Z

        return new_M, new_Z

        # M1 = self.M
        # Z1 = self.Z
    
        # A = torch.mm(torch.mm(M1,Z1),M1.t()) + torch.mm(torch.mm(M2,Z2),M2.t())
        # e_vals, e_vecs = torch.eig(A, eigenvectors=True)
        # assert(torch.all(e_vals[:,1] == 0))
        # e_vals = e_vals[:,0]
        # idxs = np.argsort(-1*e_vals.numpy())
        # Z = e_vals[idxs] - torch.max(e_vals)
        # new_M = e_vecs[idxs].t()
        # new_Z = torch.zeros(4,4)
        # for i in range(0, 4):
        #     new_Z[i,i] = Z[i]
        # #import IPython; IPython.embed()
        # #self.M = MOA
        # #self.Z = Z

        # return new_M, new_Z
