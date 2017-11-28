from feature_extraction import compute_features


def extract_raw_features( ds, path_data, layer, max_dim, mode='keyframes' ):
    """
    Funtion to extract raw features from VGG16 - separate keyframes / query folders

    args:
        Dabaset object
        path_data       -- where to store features PATH_OUT
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
