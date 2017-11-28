from codebook import GPUCodebook
from scipy.ndimage import zoom
import os
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize


class BoW_aggregator():
    '''
    Class to encode each local feature from a conv layer to a visual word

    args:
        ds        - Dataset object (src.datasets.datasets.py)
        pca_model - PCA-whitening sklearn object
        centroids - numpy array with centroids - (see train_visual_voc.py)
    '''
    def __init__(self, ds, pca_model, centroids):
        self.ds = ds
        self.pca_model = pca_model
        self.centroids = centroids
        self.codebook = GPUCodebook(self.centroids)

    def postprocess(self, feat, get_dim=False, interpolate=1):
        '''
        Post process prior to assignment computation (l2norm - PCAw - l2norm)

        args:
            feat          - 3D volume from a conv layer
            get_dim       - flag to return demensions with pos-processed features
            interpolation - optional bilinear interpolation on spatial dimension
                            of 'feat'
        '''
        if interpolate > 1:
            z= interpolate
            feat = zoom(feat, (1,z,z), order=1)
        dim = feat.shape
        feat = np.reshape( feat, (feat.shape[0], -1) )
        feat = np.transpose( feat, (1,0) )
        feat = normalize(feat)
        feat = self.pca_model.transform(feat)
        feat = normalize(feat)
        if get_dim:
            return feat.astype(np.float32), dim
        else:
            return feat.astype(np.float32)

    def compute_assignments(self, path_features=None, path_assignments=None, interpolate=1 ):
        """
        Compute assignment maps and store to disk.

        args:
            path_features     - abs path to raw features extracted from VGG16
            path_assignments  - path to store assignment maps
            interpolation     - resize raw features to obtain high ressolution on
                                assignment maps
        """

        if not os.path.exists( path_assignments ):
            os.makedirs( path_assignments )

        # get total targets
        N = len( os.listdir(path_features) )

        desc_text = "Computing assignments --> {}".format( N )
        computed = np.array( [int(k.split('.')[0]) for k in  os.listdir( path_assignments )] )

        if computed.shape[0] == N:
            return path_assignments
        else:
            last = computed.shape[0]

        for i in tqdm(range(last,N), desc=desc_text):
            path = os.path.join( path_features, str(i)+'.npy' )
            # load feat
            feat = np.load( path )

            # postprocess feat
            feat, feat_dim = self.postprocess(feat, get_dim=True, interpolate=interpolate)
            ch, x, y = feat_dim

            # compute assignments
            assignments = self.codebook.get_assignments(feat)
            np.save( os.path.join(path_assignments, "{}.npy".format(i)), assignments.reshape(x,y) )

        return path_assignments
