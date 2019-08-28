import matplotlib.pyplot as plt
import cv2
import numpy as np
from object_pose_utils.utils import to_np
from quat_math import quat2AxisAngle
from mpl_toolkits.mplot3d import Axes3D


def imshowCV(img, axis = False, show = True):
    if not axis:
        plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if(show):
        plt.show()
    
def imshow(img, axis = False, colorbar = False, show = True):
    if not axis:
        plt.axis('off')
    plt.imshow(img)
    if(colorbar):
        plt.colorbar()
    if(show):
        plt.show()
    
def torch2Img(img, normalized = False):
    disp_img = to_np(img)
    if len(disp_img.shape) == 4:
        disp_img = disp_img[0]
    disp_img = disp_img.transpose((1,2,0))
    if(normalized):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        disp_img = disp_img * std + mean
    return disp_img
    
def imshowTorch(img, normalized = False, axis = False, show = True):
    if not axis:
        plt.axis('off')
    disp_img = torch2Img(img, normalized=normalized).astype(np.uint8)
    plt.imshow(disp_img)
    if(show):
        plt.show()

def plotImageScatter(img, choose, show = True, sz = 50):
    coords = np.unravel_index(choose, img.shape[:2])    
    plt.axis('off')
    plt.imshow(img.astype(np.uint8))    
    plt.scatter(coords[1], coords[0], sz)
    #plt.colorbar()
    if(show):
        plt.show()
        
def imageBlend(im1, im2):
    return (0.5*im1 + 0.5*im2).astype(np.uint8)

def quats2Point(quats):
    pts = []
    for q in quats:
        xi, theta = quat2AxisAngle(q)
        pts.append(xi*theta)
    return np.array(pts)

def scatterSO3(vertices, vals = None, q_gt = None, 
               alpha_min = 0.0, alpha_max = 1.0, s=None, 
               ax = None, cmap = 'jet'):
    
    if(type(cmap) == str):
        cmap = plt.get_cmap(cmap)
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')

    #c = cmap(vals)
    if(vals is not None):
        a = np.maximum(0,np.minimum(1, (vals - min(vals))/(max(vals) - min(vals))))
        c = cmap(a)

        c[:,3] = (alpha_max-alpha_min)*a + alpha_min
    else:
        a = 1
        c = None
        
    if(s is None):
        s = a*10
    pts = quats2Point(vertices)
    h = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, c=c)

    if(q_gt is not None):
        pt_gt = quats2Point(q_gt)
        ax.scatter(pt_gt[:,0], pt_gt[:,1], pt_gt[:,2], c='r', marker='x')
    
    return h