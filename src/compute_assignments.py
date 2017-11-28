from BLCF import get_codebook
from BLCF import BoW_aggregator
import os

def compute_assignments( ds, path_data, layer, max_dim, mode='keyframes', interpolate=1  ):
    """
    Function to compute assignment maps. - separate keyframes / query folders

    args: Dataset object
          path_data       -- where to store features PATH_OUT
          layer           -- layer to store (conv5_1)
          max_dim         -- max dimension to resize image keeping aspect ratio (340)
          mode            -- whether to extract keyframes or queries
          interpolate     -- Whether to interpolate or not raw features to obtain higher ressolution
                             assignment map. (1= No interpolation/ 2= double each spatial dimension)
    """

    if mode == 'keyframes':
        keyframes = ds.keyframes
    else:
        keyframes = ds.q_keyframes

    # path to assignments
    path_assignments = os.path.join( path_data, ds.dataset, 'assignments', mode, layer, str(max_dim) )
    if not os.path.exists( path_assignments ):
        os.makedirs( path_assignments )
    else:
        # check if assignments have been computed
        if keyframes.shape[0] == len( os.listdir(path_assignments) ):
            return path_assignments

    # make sure features exist
    path_raw_features = extract_raw_features( ds, path_data, layer, max_dim, mode='keyframes' )
    # compute/read model
    path_models = os.path.join(path_data, ds.dataset, "models", layer, str(max_dim))
    pca_model, centroids = get_codebook( path_raw_features, path_models )
    # check mode
    path_raw_features = extract_raw_features( ds, path_data, layer, max_dim, mode=mode )
    # get assignments
    path_assignments = os.path.join( path_data, ds.dataset, 'assignments', mode, layer, str(max_dim) )
    aggregator = BoW_aggregator( ds, pca_model, centroids)
    path_assignments = aggregator.compute_assignments( path_raw_features, path_assignments, interpolate )

    return path_assignments
