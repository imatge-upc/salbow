from config import PATH_DATASET, PATH_SALIENCY, SALIENCY_MASKS, SALIENCY_URL
import os
import numpy as np
import cv2
from skimage.measure import block_reduce
from IPython import embed

def saliency_mask_blocks( path_saliency, name, size, ext='.png' ):
    """
    Get saliency for a particular image.

    size -- assignment map size
    """
    mask = cv2.imread( os.path.join(path_saliency, name+ext) )
    # reduce to size of image processing for feats
    if size is not None:
        mask = cv2.resize( mask, (size[1]*16, size[0]*16) )
        mask = block_reduce( mask[:,:,0], (16,16), np.max )
    mask = mask.astype(np.float32)
    if not np.any(mask):
        mask[...]=1
    return mask / mask.max()

def compute_ranks( targets, queries ):
    distances = targets.dot( queries.T )
    if 'sparse' in str(type(distances)):
        distances = np.array(distances.toarray())
    distances = distances.squeeze()
    ranks = []
    for i in range(queries.shape[0]):
        rank = np.argsort( distances[:,i] )[::-1]
        ranks.append(rank)
    return np.array(ranks).T, distances

class Dataset( ):
    def __init__(self, dataset ='oxford', mask=None):
        self.dataset = dataset

        if dataset == 'oxford':
            import oxford_utils as data_utils
        elif dataset == 'paris':
            import paris_utils as data_utils
        else:
            import instre_utils as data_utils

        # make sure data has been downloaded and it exists
        data_utils.get_data()
        data_utils.make_target_list()
        data_utils.make_query_list()
        self.mask = mask

        # setup data with full paths
        self.set_data( dataset )

    def set_data( self, dataset ):
        self.path_dataset = os.path.join( PATH_DATASET, dataset )
        # set path query images
        self.keyframes = np.loadtxt( os.path.join( self.path_dataset, 'imlist.txt' ), dtype='str' )
        self.base_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'imlist.txt' ), dtype='str' )

        #self.m_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'eva_imlist.txt' ), dtype='str' )

        self.q_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )
        self.base_q_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'qimlist.txt' ), dtype='str' )

        #self.mq_keyframes = np.loadtxt( os.path.join( self.path_dataset, 'eva_imlist.txt' ), dtype='str' )

        self.q_topics = np.loadtxt( os.path.join( self.path_dataset, 'qtopic.txt' ), dtype='str' )
        #self.mq_topics = np.loadtxt( os.path.join( self.path_dataset, 'eva_topic.txt' ), dtype='str' )

        self.bbx = np.loadtxt( os.path.join( self.path_dataset, 'qbbx.txt' ), dtype='str' ).astype(float)

        # format oxford is x, y, x+w, y+h
        self.bbx[:,2] = self.bbx[:,2]-self.bbx[:,0]
        self.bbx[:,3] = self.bbx[:,3]-self.bbx[:,1]

        # make full path
        self.keyframes = np.array( [os.path.join( self.path_dataset, 'images',k ) for k in self.keyframes] )
        self.q_keyframes = np.array( [os.path.join( self.path_dataset, 'images',k ) for k in self.q_keyframes] )
        self.saliency_masks = SALIENCY_MASKS

        if self.mask in SALIENCY_MASKS:
            self.path_saliency = os.path.join(PATH_SALIENCY, self.dataset, self.mask)
            # check is it does not exists
            if not os.path.exists(os.path.join(PATH_SALIENCY, self.dataset, self.mask)):
                p_dest = os.path.join(PATH_SALIENCY, self.dataset)
                if not os.path.exists( p_dest ):
                    os.makedirs( p_dest )
                # download saliency
                cmd = "wget -O {}/tmp.zip '{}'".format( p_dest, SALIENCY_URL["{}_{}".format(dataset, self.mask)] )
                os.system(cmd)
                cmd = "unzip '{}/tmp.zip' -d '{}'".format( p_dest, p_dest )
                os.system(cmd)
                os.system( "rm '{}/tmp.zip'".format(p_dest) )


    def get_mask_crop_query( self, q, ass_dim=None ):
        """ Get mask for a given query and resize to the feature map dimension
        desired """
        # frame ID without path and extension
        q_keyframes_dim = cv2.imread( self.q_keyframes[q] ).shape[:2]
        dim_ima = map(int, q_keyframes_dim)
        if ass_dim == None:
            ass_dim = dim_ima
        mask = np.zeros( (dim_ima) )

        dim_ima = map(float, q_keyframes_dim)

        # get bbx coordinates
        x,y,w,h = self.bbx[q,:].astype(int)
        if w == 0:
            w=1
        if h == 0:
            h=1

        mask[ y:y+h, x:x+w ]=1
        # init output mask

        #first resize
        mask = cv2.resize( mask, (ass_dim[1],ass_dim[0]) )
        mask[mask>=0.5]=1
        mask[mask<0.5]=0
        name = os.path.basename(self.q_keyframes[q]).split('.')[0]
        cv2.imwrite( 'test/{}_{}.png'.format(q, name), mask*255 )
        return mask

    def get_mask_saliency( self, id_, size=None, mode='keyframes'):
        """ Get mask for a given query and resize to the feature map dimension
        desired """
        # frame ID without path and extension
        path_saliency = os.path.join( self.path_saliency, mode  )
        if mode == 'keyframes':
            name = self.base_keyframes[id_].split('.')[0]
        else:
            name = self.base_q_keyframes[id_].split('.')[0]
        mask = saliency_mask_blocks( path_saliency, name, size, ext='.png' )
        return mask

    def evaluate_ranks(self,ranks):
        """
        given a list of ranks (n_queries, n_dataset_images),
        compute mAP for a given dataset

        return:
            APs and ranks
        """
        if self.dataset == 'instre':
            from instre_utils import compute_map
            mAP = compute_map(ranks)
            print np.mean(mAP)

        elif self.dataset == 'oxford':
            from oxford_utils import compute_map
            mAP = compute_map(ranks, self.path_dataset).values()

        elif self.dataset == 'paris':
            from paris_utils import compute_map
            mAP = compute_map(ranks, self.path_dataset).values()
        else:
            print "Datset {} not known!".format(self.dataset)
        return mAP, ranks

    def evaluate( self, targets, queries ):
        mAP, ranks = self.evaluate_ranks( compute_ranks( targets, queries )[0] )
        return mAP, ranks

    def get_keyframes(self):
        return self.keyframes

    def get_qkeyframes(self):
        return self.q_keyframes
