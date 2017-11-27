import numpy as np
import os, sys, glob
from config import PATH_DATASET
import shutil
from IPython import embed


# make sure we have the data
def get_data():
    """
    get data for Oxford/Paris
    """
    path_dataset = os.path.join( PATH_DATASET, 'oxford')
    if  not os.path.exists( path_dataset ):
        print "Creating {} dataset folder".format( 'oxford' )
        path_dataset_images =os.path.join(path_dataset, "images")

        # create image folder
        if not os.path.exists( path_dataset ):
            os.makedirs(path_dataset_images)

        # download images
        cmd="wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz -O {}/tmp.tar.gz".format(path_dataset)
        os.system( cmd )
        # uncompress images
        cmd = "tar -C {} -zxvf {}/tmp.tar.gz".format( path_dataset_images, path_dataset )
        os.system( cmd)

        #remove temporal file
        os.remove( "{}/tmp.tar.gz".format(path_dataset) )

        #get groundtruth
        cmd="wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz -O {}/tmp.tgz".format(path_dataset)
        os.system(cmd)

        path_gt_files =os.path.join(path_dataset, "gt_files")
        if not os.path.exists( path_gt_files ):
            os.makedirs(path_gt_files)

        # uncompress data
        cmd = "tar -C {} -zxvf {}/tmp.tgz".format( path_gt_files, path_dataset )
        os.system( cmd)
        #rm temporal files
        os.remove( "{}/tmp.tgz".format(path_dataset) )

        # get evaluation protocol
    if not os.path.exists("compute_ap"):
        cmd="wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp"
        os.system(cmd)
        os.system( "chmod 777 compute_ap.cpp")

        # add missing line and compile
        os.system( "echo '#include<stdlib.h>' > new_compute_ap.cpp" )
        os.system( "cat compute_ap.cpp >> new_compute_ap.cpp" )
        os.system( "g++ -O new_compute_ap.cpp -o compute_ap" )
        os.remove( "new_compute_ap.cpp" )
        os.remove( "compute_ap.cpp" )

def make_target_list():
    if not os._exists( os.path.join(PATH_DATASET, 'oxford', 'imlist.txt') ):
        path_list = os.path.join( PATH_DATASET, 'oxford', 'images' )
        print os.path.exists( path_list )
        fid = open( os.path.join(PATH_DATASET, 'oxford', 'imlist.txt'), 'w' )
        lst = os.listdir( path_list )
        lst.sort()
        for i in lst:
            fid.write( "{}\n".format(i) )
        fid.close()

def make_query_list():
    if not os._exists( os.path.join(PATH_DATASET, 'oxford', 'qimlist.txt') ):

        path_list = os.path.join( PATH_DATASET, 'oxford', 'gt_files' )
        queries = glob.glob( path_list+'/*_query*' )
        queries.sort()

        fid = open( os.path.join(PATH_DATASET, 'oxford','qimlist.txt'), 'w' )
        fid_t = open( os.path.join(PATH_DATASET, 'oxford','qtopic.txt'), 'w' )
        fid_bbx = open( os.path.join(PATH_DATASET, 'oxford','qbbx.txt'), 'w' )

        for file_query in queries:
            data = np.loadtxt( file_query, dtype='str' )
            name =  data[0].split('oxc1_')[1]+'.jpg'

            id_query = os.path.basename(file_query).split('_query.txt')[0]
            fid.write( "{}\n".format( name ))
            fid_t.write( "{}\n".format( id_query ))
            fid_bbx.write( "{}\t{}\t{}\t{}\n".format(data[1], data[2], data[3], data[4]))

        fid.close()
        fid_t.close()
        fid_bbx.close()

def compute_map(ranks, details=True):
    target_names = os.path.join( PATH_DATASET, 'oxford', 'imlist.txt' )
    query_names = os.path.join( PATH_DATASET, 'oxford', 'qtopic.txt' )

    target_names = np.loadtxt( target_names, dtype='str' )
    query_names = np.loadtxt( query_names, dtype='str' )

    path_out = os.path.join( os.getcwd(), 'ranks_oxford')

    if os.path.exists(path_out):
        shutil.rmtree(path_out)
        os.makedirs(path_out)
    else:
        os.makedirs(path_out)

    for i, name in enumerate(query_names):
        fid = open(  os.path.join(path_out, name+'.txt'), 'w' )
        for k in range(ranks.shape[0]):
            name = target_names[ranks[k,i]]
            fid.write( "{}\n".format(name.split('.')[0]) )
        fid.close()
    results = compute_ap_oxf( path_out, os.path.join(PATH_DATASET, 'oxford', 'gt_files'))
    if details:
        return results
    else:
        return np.average( results.values() )

def compute_ap_oxf( path_ranks, gt_path ):
    """
    Generate AP per rankfile
    params:
        path_ranks: where .txt files are allocated
        gt_path: where OX/PAR queries are allocated
    """
    results = {}
    for rank in os.listdir(path_ranks):
        q_id = os.path.basename( rank ).split('.')[0]
        cmd = "./compute_ap '{}/{}' '{}/{}'".format( gt_path, q_id, path_ranks, rank )
        ap = float(os.popen(cmd).read())
        results[ os.path.basename( rank ).split('.')[0] ] = ap
    return results
