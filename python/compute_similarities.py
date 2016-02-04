from scipy.io import savemat, loadmat
from configs import *
import numpy as np
import caffe, os, glob, sys

def getVSB100Representation(conf, solver):
    db_settings = conf.db_settings
    action = db_settings['action_name']
    video_info_path = db_settings['video_info_path'].format(action_name=action) #'/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_vidinfo.mat'
    video_info = loadmat(video_info_path) #video_info = [mapped, labelledlevelvideo, numberofsuperpixelsperframe]
    framebelong = video_info['framebelong']
    superpixels_num = len(framebelong)
    #net = solver.test_nets[0]
    return getRepresentations(conf, solver, superpixels_num)

def getJHMDBRepresentations(conf, solver):
    db_settings = conf.db_settings
    action = db_settings['action_name']
    video_name = db_settings['video_name']
    level = db_settings['level']
    # mat = loadmat('/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/pickle/pour1.mat');
    path = db_settings['labelledlevelvideo_path'].format(action_name=action[0], video_name=video_name[action[0]][0], level=level)
    print path
    mat = loadmat(path)
    # pickle_path =
    superpixels_num = mat['total_number_of_supervoxels']
    return getRepresentations(conf, solver, superpixels_num)

def getRepresentations(conf, net, superpixels_num):
    data = net.blobs['inner_product_target'].data
    assert data.shape[0] == 1, 'batch size != ? ... this assert is not important'
    feature_len = data.shape[1]
    reps = np.zeros((superpixels_num, feature_len))
    for i in xrange(superpixels_num):
        if i%1000==1:
            print i
        net.forward()
        reps[i][...] = net.blobs['inner_product_target'].data[...]
        # print net.blobs['inner_product_target'].data[1:10]
    return reps


def computeDistanceMatrix(representations):
    return (representations.dot(representations.T))
    #TODO: check if it should be negative or positive

def getLastAddedFile(path):
    search_dir = path
    # remove anything from the list that is not a file (directories, symlinks)
    # thanks to J.F. Sebastion for pointing out that the requirement was a list 
    # of files (presumably not including directories)  
    files = filter(os.path.isfile, glob.glob(search_dir + "*"))
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-2]

def computeSimilarities(config_id):
    print 'Experiment number:', config_id
    conf = getConfigs(config_id)
    snapshot_path = conf.solver['snapshot_prefix']
    caffemodel_path = getLastAddedFile(snapshot_path + '/')
    print "last snapshot is:", caffemodel_path
    caffe.set_mode_gpu()
    db_settings = conf.db_settings
    test_model_path = conf.model['test_prototxt_path']
    test_model =  caffe.Net(test_model_path, caffemodel_path, caffe.TEST)
    print 'creating model'
    if conf.db == 'vsb100':
        print 'vsb100'
        representations = getVSB100Representation(conf, test_model)
    elif conf.db == 'jhmdb':
        print 'jhmdb'
        representations = getJHMDBRepresentations(conf, test_model)
    else:
        raise
    similarities = computeDistanceMatrix(representations)
    print representations[:10, :] #'creating model'
    savemat(conf.experiments_path+'/similarities.mat', {'similarities':similarities})

if __name__ == '__main__':
    if len(sys.argv) == 1:
        computeSimilarities(-1)
    else:
        computeSimilarities(int(sys.argv[1]))
