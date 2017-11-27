import glob, os, sys
import numpy as np
#vlfeat module
from vlfeat import kmeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import time
from IPython import embed
from tqdm import tqdm

def get_local_features( path_features, N_max=1568112):
    # local feartures # exp set with 10.000)
    '''
    Get random local features for training
    args: path_features
    '''
    # all features
    list_images = glob.glob( os.path.join( path_features, "*.npy" ) )

    #get dimension features
    feat = np.load( list_images[0] )
    c, x, y = feat.shape

    # set max num of images to process
    N = len(list_images)

    # random order
    np.random.shuffle( list_images )
    i = 0
    data = []

    count =0
    for path in tqdm(list_images[:int(N)]):
        feat = np.load( path )
        feat = np.transpose( feat, (1,2,0) )
        r, c, ch = feat.shape
        feat = np.reshape( feat, (r*c, -1) )
        data.extend(feat)
        count +=r*c
#        if count > N_max:
#            break
    data = np.array(data)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    print data.shape
    return data[idx,...]


def get_pca_model( path_pca_file, path_features, n_components=512):
    """
    Train/load PCA for a particular features
    args: pca id models/dataset/max_dim/layer/pca.npy
          path_features to the conv features
    """
    if os.path.exists( path_pca_file+'.npy' ):
        pca_model = np.load(  path_pca_file+'.npy'  ).tolist()
    else:
        print "computing PCA-w..."
        # load taining features
        training_feats = get_local_features(path_features)
        #l2norm
        training_feats = normalize(training_feats)
        t0 = time.time()
        pca_model = PCA(n_components, whiten=True)
        pca_model.fit(training_feats)
        t1 = time.time()
        print "DONE! %.2fs" % (t1-t0)

        np.save( path_pca_file, pca_model )
    return pca_model

def get_clustering( path_file, path_features, pca_model, n_centroids=25000 ):
    """
    Train/load PCA for a particular features
    args: pca id models/dataset/max_dim/layer/pca.npy
          path_features to the conv features
    """
    if pca_model is not None:
        filename = path_file+'.npy'
    else:
        filename = path_file+'_l2.npy'

    if os.path.exists( filename ):
        centers = np.load(  filename  )
    else:
        # load taining features
        training_feats = get_local_features(path_features)
        print "loaded target features for training. Shape={}".format(training_feats.shape)
        #postprocess conv features: l2-pca-l2
        if pca_model!=None:
            training_feats = normalize(pca_model.transform((normalize(training_feats)))).astype(np.float32)
        else:
            training_feats = normalize(training_feats).astype(np.float32)

        print "target features post-processed. Shape={}".format(training_feats.shape)

        print "Fitting vocabulary"
        t0 = time.time()
        clustering_model = kmeans.KMeans( num_centers = n_centroids, algorithm='ann', initialization='random' )
        clustering_model.fit(training_feats)
        centers = clustering_model.centers.copy()
        t1 = time.time()
        print "DONE! %.2fs" % (t1-t0)
        np.save( filename.split('.')[0], centers )

    return centers


def get_codebook( path_features, path_out, n_clusters=25000, n_components=512 ):
    """
    Compute PCA and codebook models

    arg: path_features -- original raw conv_features
         path_out -- path to store the models PCA and centroids
         n_clusters --  size of vocabulary
         n_components -- dim PCA model / None if not computing PCA
    """
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    path_pca_file = os.path.join( path_out, "pca" )

    if n_components is not None:
        path_centroids =  os.path.join( path_out, "centroids" )
        pca_model = get_pca_model( path_pca_file, path_features, n_components)

    else:
        path_centroids =  os.path.join( path_out,"centroids_l2" )
        pca_model=False

    centroids = get_clustering( path_centroids, path_features, pca_model, n_clusters )

    return pca_model, centroids
