from model import init_model
from utils import preprocess_image
import os
from tqdm import tqdm
import numpy as np
from IPython import embed

def compute_features( keyframes, path_out, layer='conv5_1', max_dim=340 ):
    """
    store all local features from conv5_1 of vgg16 at 340 mac res and
    store then in path_data/[mode]/layer/max_dim
    """

    # create folder if it does not exist
    if not os.path.exists( path_out ):
        os.makedirs( path_out )

        # init model
        model = init_model(  layer )
        # message
        desc_text = "Feature extraction --> Layer: {}, Max_dim: {}, total_images={}".format( layer, max_dim, keyframes.shape[0] )
        # process keyframes
        for k, keyframe in tqdm(enumerate(keyframes), ascii=True, desc=desc_text):
            feats = model.predict( preprocess_image( keyframe, max_dim=max_dim ) ).squeeze(axis=0)
            np.save( os.path.join( path_out, "{}".format(k) ), feats )

    # resume computation
    else:
        computed = np.array( [int(k.split('.')[0]) for k in  os.listdir( path_out )] )
        # if features has been computed...
        if computed.shape[0] == keyframes.shape[0]:
            return path_out
        # start from the last computed...
        elif computed.shape[0]==0:
            last = 0
        else:
            last = np.sort(computed)[::-1][0]

        # init model
        model = init_model(  layer )
        desc_text = "Feature extraction --> Layer: {}, Max_dim: {}, total_images={}".format( layer, max_dim, keyframes.shape[0]-last )
        for k, keyframe in tqdm(enumerate(keyframes[last:]), ascii=True, desc=desc_text):
            feats = model.predict( preprocess_image( keyframe, max_dim=max_dim ) ).squeeze(axis=0)
            k+=last
            np.save( os.path.join( path_out, "{}".format(k)), feats )

    return path_out
