import numpy as np
import cv2
from scipy.io import loadmat

def load_grid_data(dataset_root, grid_indices, obj, return_meta = False):
    with open('{0}/image_sets/classes.txt'.format(dataset_root)) as f:
        classes = f.read().split()
    classes.insert(0, '__background__')

    filename_format = '{}/depth_renders/{}/'.format(dataset_root, classes[obj]) + '{:04d}-{}.{}'

    images = []
    depths = []
    masks = []
    quats = []
    metas = []

    for j in grid_indices:
        img = cv2.cvtColor(cv2.imread(filename_format.format(j, 'color', 'png')), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(filename_format.format(j, 'depth', 'png'), cv2.IMREAD_UNCHANGED)
        mask = np.bitwise_and((cv2.imread(filename_format.format(j, 'label', 'png'))==obj)[:,:,0],
                              depth > 0).astype(np.uint8)
        q = np.load(filename_format.format(j, 'trans', 'npy'))
        meta = loadmat(filename_format.format(j, "meta", 'mat'))

        images.append(img)
        depths.append(depth)
        masks.append(mask)
        quats.append(q)
        metas.append(meta)

    if return_meta:
        return images, depths, masks, quats, metas
    else:
        return images, depths, masks, quats
