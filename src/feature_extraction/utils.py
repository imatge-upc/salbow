import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import cv2
from IPython import embed


def preprocess_image( path_image, dim=None, max_dim=None ):
    """
    Addapts image to size 'dim' outputs tensor for net
    args: path_image
    retu: X tensor
    """
    if 'str' in str(type(path_image)) or  'unicode' in str(type(path_image)):
        ima = cv2.imread( path_image )
    else:
        ima = path_image

    # check if it is the mask
    if len(ima.shape) == 2:
        if dim is None:
            dim = find_dimensions(ima, max_dim)
        ima = cv2.resize( ima, (dim[1], dim[0]) )
        ima[ima>0] = 1
        return ima
    else:
        if dim is None and max_dim is None:
            ima = ima[:,:,::-1].astype( dtype=np.float32 )
            ima = np.transpose( ima, (2,0,1) )
            ima = np.expand_dims( ima, axis=0 )
            return preprocess_input(ima)

        if dim is None:
            dim = find_dimensions(ima, max_dim)

        ima = cv2.resize( ima, (dim[1], dim[0]) )
        ima = ima[:,:,::-1].astype( dtype=np.float32 )
        ima = np.transpose( ima, (2,0,1) )
        ima = np.expand_dims( ima, axis=0 )
        return preprocess_input(ima)


def find_dimensions( ima, max_dim=340, r_ratio=False):
    """
    Find the dimensions for keeping the aspect ratio.
    It sets the larger dimension to 'max_dim'

    return tuple with the new dimensions
    """
    if len(ima.shape)==3:
        dim =np.array(map(float, ima.shape[:2]))
    else:
        dim =np.array(map(float, ima.shape))
    ratio = min( max_dim/dim )
    new_dim = tuple([int(ratio*d) for d in dim])
    if r_ratio:
        return new_dim, ratio
    else:
        return new_dim
