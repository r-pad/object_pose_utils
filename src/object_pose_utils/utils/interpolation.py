# -*- coding: utf-8 -*-
"""
Created on Some night, Way to late

@author: bokorn
"""
import os
import torch
import numpy as np
from sklearn.neighbors import KDTree
import scipy
import scipy.io as sio
from functools import partial

from object_pose_utils.utils.pose_processing import getGaussianKernal, quatAngularDiffBatch
from object_pose_utils.utils.tetra_utils import SearchableTetraGrid
from object_pose_utils.utils.bingham import makeBinghamM, bingham_const

root_folder = os.path.dirname(os.path.abspath(__file__))

class TetraInterpolation(object):
    def __init__(self, grid_level, values = None):
        self.level = grid_level
        self.kd_grid = SearchableTetraGrid(self.level) 
        
        if(values is not None):
            self.setValues(values)

    def setValues(self, values):
        eta = np.abs(values).sum()*(np.pi**2)/values.shape[0]
        self.values = 1./eta * values

    def baryTetraTransform(self, t_id):
        return np.linalg.inv(np.vstack(self.kd_grid.grid.GetTetrahedron(t_id, self.level).vertices).T)

    def smooth(self, q, k=4):
        kd_dist, idxs = self.kd_grid.query(q, k=k)
        return (self.values[idxs]*kd_dist).sum(axis=1)/(kd_dist.sum(axis=1))

    def __call__(self, q):
        t_id = self.kd_grid.containingTetraFast(q)
        A = self.baryTetraTransform(t_id)
        v_ids = self.kd_grid.grid.GetTetras(self.level)[t_id]

        q_bar = np.matmul(A, q)
        q_bar /= q_bar.sum()
        val_verts = self.values[v_ids]
        return q_bar.dot(val_verts)

gaussian_normalization_data = sio.loadmat(os.path.join(root_folder, 
    'quat_angle_gaussian_normalization.mat'))
gaussian_normalization_eta = gaussian_normalization_data['eta'][0]
gaussian_normalization_sigma = gaussian_normalization_data['sigma'][0]
gaussianNormC = partial(np.interp, 
                        xp=gaussian_normalization_sigma, 
                        fp=gaussian_normalization_eta)

class GaussianInterpolation(object):
    def __init__(self, vertices, values, sigma=np.pi/9):
        self.vertices = torch.as_tensor(vertices).float()
        if(torch.cuda.is_available()):
            self.vertices = self.vertices.cuda()
        self.sigma = sigma
        self.eta = gaussianNormC(sigma) 
        self.setValues(values)

    def setValues(self, values, normalize = True):
        self.values = torch.as_tensor(values).float()
        if(normalize):
            self.values /= self.values.sum()
        if(len(self.values.shape) == 1):
            self.values = self.values.unsqueeze(0)

        if(torch.cuda.is_available()):
            self.values = self.values.cuda()

    def __call__(self, q):
        K = 1./self.eta * getGaussianKernal(q, self.vertices, sigma=self.sigma)
        return torch.mm(K, self.values.t()).flatten()

class BinghamInterpolation(object):
    def __init__(self, vertices, values = None, sigma=np.pi/9):
        if(type(vertices) is not torch.Tensor):
            vertices = torch.tensor(vertices).float()

        if(type(sigma) is not torch.Tensor):
            sigma = torch.tensor(sigma).float()

        self.vertices = vertices
        Ms = []
        for v in self.vertices:
            Ms.append(makeBinghamM(v))
        M = torch.stack(Ms)
        zeros = torch.zeros(1)
        if(vertices.is_cuda):
            zeros = zeros.cuda()
        Z = torch.cat([zeros,-sigma.repeat(3)])
        self.eta = bingham_const(Z[1:]).float()/2.0
        Z = torch.diag(Z)
        self.MZMt = torch.bmm(torch.bmm(M, Z.repeat([len(Ms),1,1])), torch.transpose(M,2,1))
        if(torch.cuda.is_available()):
            self.MZMt = self.MZMt.cuda()
            self.eta = self.eta.cuda()

        if(values is not None):
            self.setValues(values)

    def setValues(self, values):
        self.values = values/values.sum()
        if(torch.cuda.is_available()):
            self.values = self.values.cuda()

    def __call__(self, q, return_exponent = False):
#        bingham_p = 1./self.eta*torch.exp(torch.mul(q.transpose(1,0).unsqueeze(2), 
#            torch.matmul(q,self.MZMt.transpose(2,0))).sum([0]))
        bingham_p = torch.mul(q.transpose(1,0).unsqueeze(2), 
            torch.matmul(q,self.MZMt.transpose(2,0))).sum([0])
        if(return_exponent):
            return bingham_p
        else:
            bingham_p = 1./self.eta*torch.exp(bingham_p)
        return (self.values * bingham_p).sum(1)

