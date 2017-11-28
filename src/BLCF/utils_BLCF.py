import numpy as np
import os, sys
from tqdm import tqdm

from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse import hstack, vstack
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

from spatial_weighting_schemes import l2_norm_maps, gaussian_weights


def save_sparse_csr(filename, array):
    '''
    Store sparce numpy matrix to disk

    args:
        filename   - abs path to store the matrix
        array      - numpy sparse matrix
    '''
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    '''
    Load sparce numpy matrix

    args:
        filename   - abs path to store the matrix
        array      - numpy sparse matrix
    '''
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def get_bow( assignments, weights=None, n=25000 ):
    '''
    Funtion to build BoW representation given an assignment map.

    args:
        assignments - 2D map with assignments associated to each local feature
        weights     - 2D maps with normalized (0-1) spatial weighting scheme
        n           - size of the visual vocabulary (default 25,000)
    '''
    # sparse encoding !
    rows = np.array([], dtype=np.int)
    cols = np.array([], dtype=np.int)
    vals = np.array([], dtype=np.float)
    n_docs = 0

    # get counts
    cnt = Counter(assignments.flatten())
    ids = np.array(cnt.keys())
    if weights is None:
        weights = np.array(cnt.values())
    else:
        weights = weights.flatten()
        weights = np.array([weights[np.where(assignments.flatten()==i)[0]].sum() for i in ids])

    #save index
    cols = np.append( cols, np.array(ids).astype(int) )
    rows = np.append( rows, np.ones( len(cnt.keys()), dtype=int )*n_docs )
    vals = np.append( vals, weights.astype(float) )
    n_docs +=1

    bow = coo_matrix( ( vals, (rows, cols) ), shape=(n_docs,n) )
    bow = bow.tocsr()
#    bow = normalize(bow)
    return bow


def load_targets( ds, path_assignments, mask=None ):
    """
    Function to encode into sparse matrix assignment
    with a particular spatial weighting scheme

    args:
        path_assignments   - abs path to the assignment maps
        mask               - string indicating the spatial weighting scheme
    """

    N = len(os.listdir(path_assignments))
    desc = "Generating sparse matrix -- Spatial weight {}, total assignment maps {}".format( mask, N )
    for i in tqdm(range(N), desc=desc):
        path = os.path.join( path_assignments, "{}.npy".format(i) )
        assignments = np.load(path)
        weights = None

        if mask is None:
            weights = None
        elif mask == 'gaussian':
            weights = gaussian_weights(assignments.shape)
        elif mask == 'l2norm':
            # load raw features
            path = path_assignments.replace( 'assignments', 'conv_features' )
            feat = np.load( os.path.join( path, "{}.npy".format(i) ) )
            weights = l2_norm_maps( feat, dim_r=assignments.shape )
        elif mask in ds.saliency_masks:
            weights = ds.get_mask_saliency( i, size=assignments.shape, mode='keyframes')
        else:
            print "--> Mask '{}' is not valid!".format( mask )
            print
            sys.exit()

        bow = get_bow( assignments, weights=weights)

        if i == 0:
            total_bow = bow
        else:
            total_bow = vstack( [total_bow, bow] )
    return normalize(total_bow)


def load_queries( ds, path_assignments, mode='global'):
    """
    Function to encode into sparse matrix assignment maps of query images.

    args:
        path_assignments   - abs path to the assignment maps
        mode               - global - Encode all visual words
                             crop   - Encode only visual words within bbx
                             [TODO] - weight bg on query
    """

    N = len(os.listdir(path_assignments))

    for i in tqdm(range(N)):
        path = os.path.join( path_assignments, "{}.npy".format(i) )
        assignments = np.load(path)
        if  mode=='crop': # only trec has computed ass!
            weights = ds.get_mask_crop_query( i, ass_dim=assignments.shape )
        elif mode=='gaussian_crop':
            mask = ds.get_mask_frame( i, dimension=assignments.shape )
            weights = gaussian_weights(s_a, center = None, sigma=None )
            weights[mask>0]=1
        elif mode=='d_weighting':
            weights = ds.get_mask_crop_query( i, dimension=assignments.shape )
            weights=get_distanceTransform(weights, i)
        else:
            weights=None
        bow = get_bow( assignments, weights=weights)
        if i == 0:
            total_bow = bow
        else:
            total_bow = vstack( [total_bow, bow] )

    return normalize(total_bow)

from IPython import embed

def query_expansion( targets, queries, ranks, N=10 ):
    """
    Average Query Expansion function. It created a new query based on the top N
    retrieved results.

    args:
         targets  -  Sparse Matrix with target descriptors
         queries  -  Sparse Matrix with query descriptors
         N        -  Top results to generate new query (N=10)
    """

    # get scores
    NQ = queries.shape[0]

    new_queries = None
    for i in range(NQ):
        # get rankings
        idx = ranks[:,i]

        # get 10 features
        new = None
        for k in range(N+1):
            if new is None:
                new = queries[i,:]
            else:
                new = vstack( (new, targets[idx[k-1],:]) )
        new = csr_matrix(new.sum(axis=0))
        if new_queries is None:
            new_queries = new
            new_queries = vstack( (new_queries, new) )

    new_queries = normalize( new_queries )
    return new_queries
