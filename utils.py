from src.feature_extraction.compute_features import compute_features
from src.BLCF.train_visual_voc import get_codebook
from src.BLCF.encode_assignments import BoW_aggregator
import os

def extract_raw_features( ds, path_data, layer, max_dim, mode='keyframes' ):
    """
    Funtion to extract raw features from VGG16.

    args:
        Dabaset object
        path_data       -- where to store features PATH_DATA/dataset/conv_features/MODE/LAYER/MAX_DIM
        layer           -- layer to store (conv5_1)
        max_dim         -- max dimension to resize image keeping aspect ratio (340)
        mode            -- whether to extract keyframes or queries
    """
    # path to store raw features for targets
    path_raw_features = os.path.join( path_data, ds.dataset, 'conv_features', mode, layer, str(max_dim) )
    if mode == 'keyframes':
        list_images = ds.keyframes
    else:
        list_images = ds.q_keyframes
    path_raw_features = compute_features( list_images, path_raw_features, layer, max_dim )
    return path_raw_features


def compute_assignments( ds, path_data, layer, max_dim, mode='keyframes', interpolate=1  ):
    """
    Function to compute assignment maps.

    args: Dataset object
          path_data       -- where to store features PATH_DATA/dataset/conv_features/MODE/LAYER/MAX_DIM
          layer           -- layer to store (conv5_1)
          max_dim         -- max dimension to resize image keeping aspect ratio (340)
          mode            -- whether to extract keyframes or queries
          interpolate     -- Whether to interpolate or not raw features to obtain higher ressolution
                             assignment map. (1= No interpolation/ 2= double each spatial dimension)
    """
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
