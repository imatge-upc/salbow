import numpy as np
import sys, os, glob
from scipy.io import loadmat
from config import PATH_DATASET
from IPython import embed
from scipy.io import loadmat


# make sure we have the data
def get_data():
    """
    get data for Instre

    returns: groundtruth labels and target/query order
    """

    path_instre = os.path.join( PATH_DATASET, 'instre')
    if not os.path.exists( path_instre ):
        print "Creating Instre dataset folder"
        path_instre_images =os.path.join(path_instre, "images")

        # create image folder
        if not os.path.exists( path_instre ):
            os.makedirs(path_instre_images)

        # download images
        cmd="wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/instre.tar.gz -O {}/tmp.tar.gz".format(path_instre)
        os.system( cmd )
        # uncompress images
        cmd = "tar -C {} -zxvf {}/tmp.tar.gz".format( path_instre_images, path_instre )
        os.system( cmd )
        #get groundtruth
        cmd="wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/gnd_instre.mat -O {}/gnd_instre.mat".format(path_instre)
        os.system( cmd )

        #rm temporal file
        os.remove( "{}/tmp.tar.gz".format(path_instre) )

    return loadmat( os.path.join( path_instre,'gnd_instre.mat' ) )


def make_target_list():
    """
    create the list with all target images in Instre
    """
    if not os._exists( os.path.join(PATH_DATASET, 'instre', 'imlist.txt') ):

        # check broken image
        data_gnd = loadmat( os.path.join( PATH_DATASET, 'instre','gnd_instre.mat' ) )
        data = data_gnd['imlist'].squeeze()
        fid = open( os.path.join(PATH_DATASET, 'instre', 'imlist.txt'), 'w' )

        for name in data:
            fid.write( "{}\n".format(name[0]) )
        fid.close()

def make_query_list():
    """
    create the list with all queries images in Instre
    """
    if not os._exists( os.path.join(PATH_DATASET, 'instre', 'qimlist.txt') ):
        # check broken image
        data_gnd = loadmat( os.path.join( PATH_DATASET, 'instre','gnd_instre.mat' ) )
        data = data_gnd['qimlist'].squeeze()
        fid = open( os.path.join(PATH_DATASET, 'instre', 'qimlist.txt'), 'w' )
        fid_topic = open( os.path.join(PATH_DATASET, 'instre', 'qtopic.txt'), 'w' )
        fid_bbx = open( os.path.join(PATH_DATASET, 'instre', 'qbbx.txt'), 'w' )

        for i, name in enumerate(data):
            bbx = data_gnd['gnd'][0][i][1].squeeze()
            fid.write( "{}\n".format(name[0]) )
            fid_topic.write( "{}\n".format( name[0].split('/')[1] ) )
            fid_bbx.write( "{}\t{}\t{}\t{}\n".format(bbx[0],bbx[1],bbx[2],bbx[3]) )
        fid.close()
        fid_topic.close()
        fid_bbx.close()


def get_gnd():
    data_gnd = loadmat( os.path.join( PATH_DATASET, 'instre','gnd_instre.mat' ) )
    class Query():
        def __init__(self, ok, bbx):
            self.ok = ok[0,:].squeeze()
            self.bbx = bbx[0,:].squeeze()

    def convert_gnt( data_gnd ):
        # for each quey I have the ok list and
        gt_queries = []
        for i in range( data_gnd['gnd'].shape[1] ):
            gt_queries.append( Query( data_gnd['gnd'][0][i][0], data_gnd['gnd'][0][i][1] ) )
        return gt_queries
    return convert_gnt( data_gnd )


def compute_map( ranks, verbose=False, per_query=True):
    gnd = get_gnd()
    mAP = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    precision = []
    # for each query
    for i in range(nq):
        gt_list = gnd[i].ok
        intersection = np.in1d(ranks[:,i], gt_list)
        pos = np.arange(ranks[:,i].shape[0])[intersection]
        ap = score_ap_from_ranks( pos, gt_list.shape[0] )
        if verbose:
            print "query{} -- {}".format( i, ap )
        mAP += ap
        precision.append( ap )
    if per_query:
        return precision
    else:
        return mAP/nq

def score_ap_from_ranks( _ranks, nres ):
    _ranks = _ranks.astype(float)
    nimgranks = _ranks.shape[0]
    ap = 0
    recall_step = 1.0/nres
    for j in range(1, nimgranks+1):
        rankval = _ranks[j-1]

        if rankval == 0:
            p0 = 1.0
        else:
            p0 = (j-1.0)/rankval

        p1 = j/(rankval+1)
        ap = ap+(p0+p1)*recall_step/2

    return ap
