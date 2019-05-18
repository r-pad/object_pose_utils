# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
from quat_math import projectedAverageQuaternion
from object_pose_utils.utils.multiscale_grid import MultiResGrid
import scipy
from sklearn.neighbors import KDTree

eps = 1e-12

def vec_in_list(target, list_vecs):
    return next((True for elem in list_vecs if elem.size == target.size and np.array_equal(elem, target)), False)

def vec_close_in_list(target, list_vecs):
    return next((True for elem in list_vecs if elem.size == target.size and np.allclose(elem, target)), False)

def insideTetra(tetra, q):
    v0 = np.zeros(4)
    v1 = tetra.vertices[0]
    v2 = tetra.vertices[1]
    v3 = tetra.vertices[2]
    v4 = tetra.vertices[3]
    
    n1 = scipy.linalg.null_space(np.vstack([v2,v3,v4]))
    n2 = scipy.linalg.null_space(np.vstack([v1,v3,v4]))
    n3 = scipy.linalg.null_space(np.vstack([v1,v2,v4]))
    n4 = scipy.linalg.null_space(np.vstack([v1,v2,v3]))
    qs1 = q.dot(n1)
    vs1 = v1.T.dot(n1)
    qs2 = q.dot(n2) 
    vs2 = v2.T.dot(n2) 
    qs3 = q.dot(n3) 
    vs3 = v3.T.dot(n3)
    qs4 = q.dot(n4)
    vs4 = v4.T.dot(n4)
    
    if (vs1*qs1 >= -eps and vs2*qs2 >= -eps and vs3*qs3 >= -eps and vs4*qs4 >= -eps) or \
       (vs1*qs1 <=  eps and vs2*qs2 <=  eps and vs3*qs3 <=  eps and vs4*qs4 <=  eps):
        return True

    return False

def metricGreaterThan(sorted_vals, max_metrics, metric_func):
    for k in range(len(sorted_vals)):
        v = metric_func(sorted_vals[k:])
        if(v > max_metrics[k]):
            return True
        elif(v < max_metrics[k]):
            return False
    return False

def quatL2Dist(q1, q2):
    return min(np.linalg.norm(q1-q2),np.linalg.norm(q1+q2))

class SearchableTetraGrid(object):
    def __init__(self, grid_level):
        #self.vertices = vertices
        self.level = grid_level
        self.grid = MultiResGrid(self.level) 
        self.num_verts = self.grid.vertices.shape[0]
        self.num_tetra = len(self.grid.GetTetras(1))
        self.kd_tree = KDTree(np.vstack([self.grid.vertices, -self.grid.vertices]))
        
        self.max_edge = -np.inf
        for j, tet in enumerate(self.grid.GetTetrahedra(self.level)):
            v1 = tet.vertices[0]
            v2 = tet.vertices[1]
            v3 = tet.vertices[2]
            v4 = tet.vertices[3]
            d12 = quatL2Dist(v1,v2)
            d13 = quatL2Dist(v1,v3)
            d14 = quatL2Dist(v1,v4)
            d23 = quatL2Dist(v2,v3)
            d24 = quatL2Dist(v2,v4)
            d34 = quatL2Dist(v3,v4)
            self.max_edge = max(self.max_edge, d12, d13, d14, d23, d24, d34)

    def containingTetra(self, q, return_all=False):
        vert_ids = self.kd_tree.query_radius([q], r = self.max_edge)[0]
        vert_ids = np.where(vert_ids < self.num_verts, vert_ids, vert_ids - self.num_verts)
        tetra_ids = set()

        for v_id in vert_ids:
            _, t_ids = self.grid.GetNeighborhood(v_id, level=self.level)
            tetra_ids.update(t_ids)

        if(return_all):
            ids = []
        for t_id in tetra_ids:
            if(insideTetra(self.grid.GetTetrahedron(t_id, level=self.level), q)):
                if(return_all):
                    ids.append(t_id)
                else:
                    return t_id
        if(not return_all):
            raise ValueError('Did not find containing tetrahedra for {}'.format(q)) 
        return ids

def refineTetrahedron(q, tetrahedron, dist_func, metric_func, levels=2):
    if(levels == 0):
        dists = dist_func(tetrahedron.vertices)
        idx = np.argmax(dists)
        return tetrahedron.vertices[idx]
        return projectedAverageQuaternion(tetrahedron.vertices)#, weights = 1/np.array(dists))

    tetras = tetrahedron.Subdivide()

    max_idx = -1
    max_val = -float('inf')

    for j, tra in enumerate(tetras):
        dists = dist_func(tra.vertices)
        val = metric_func(dists)
        if(val > max_val):
            max_val = val
            max_idx = j
    return refineTetrahedron(q, tetras[max_idx], dist_func, metric_func, levels=levels-1)


