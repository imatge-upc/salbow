import numpy as np
import cv2

def get_distanceTransform(mask, k):
    img = (255*mask).astype(np.uint8)
    dist = cv2.distanceTransform(255-img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
    #mask_weights = np.exp(-0.05*dist)
    if np.max(dist)==0:
        return None
    dist = 1 - dist/np.max(dist)
    return dist


def l2_norm_maps( feat, dim_r=None ):
    norm_m = np.sqrt(np.sum( feat**2, axis=0 ))
    norm_m /= norm_m.max()
    if dim_r is not None:
        norm_m = cv2.resize( norm_m, (dim_r[1], dim_r[0] ))
    return norm_m

def get_spatial_weights( feat ):
    S = np.sum( feat, axis=0 )
    S_norm = np.sqrt(np.sum( S**2 ))
    return np.sqrt(S/S_norm)

def gaussian_weights(shape, center = None, sigma=None ):
    r1 = shape[0] /2
    r2 = shape[1] /2

    ys = np.linspace(-r1, r1, shape[0])
    xs = np.linspace(-r2, r2, shape[1])
    YS, XS = np.meshgrid(xs, ys)

    if center is not None:
        YS -= ( center[1]-r1 )
        XS -= (center[0]-r2 )

    if sigma is None:
        sigma = min(shape[0], shape[1]) / 3.0
    g = np.exp(-0.5 * (XS**2 + YS**2) / (sigma**2))

    # normalize
    g -= np.min(g)
    g /= np.max(g)
    return g

def weighted_distances( dx=10, dy=10, c=None):
    '''
    Map with weighted distances to a point
	args: Dimension maps and point
    '''
    if c is None:
        c = (dx/2, dy/2)

    a = np.zeros((dx,dy))
    a[c]=1

    indr = np.indices(a.shape)[0,:]
    indc = np.indices(a.shape)[1,:]

    difr = indr-c[0]
    difc = indc-c[1]

    map_diff = np.sqrt((difr**2)+(difc**2))

    map_diff = 1.0 - (map_diff/ map_diff.flatten().max())

    # Return inverse distance map
    return map_diff
