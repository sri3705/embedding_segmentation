from scipy.io import savemat, loadmat
from configs import *
import numpy as np
import caffe, os, glob, sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-e', '--expid', dest='exp_id', default='-1')
parser.add_option('-l', '--layer', dest='layer', default='inner_product_target')
(options, args) = parser.parse_args()

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
    path = db_settings['voxellabelledlevelvideo_path'].format(action_name=action[0], video_name=video_name[action[0]][0], level=level)
    print path
    mat = loadmat(path)
    # pickle_path =
    superpixels_num = mat['total_number_of_supervoxels']
    return getRepresentations(conf, solver, superpixels_num)

def getweights(conf, solver):
    db_settings = conf.db_settings
    action = db_settings['action_name']
    video_name = db_settings['video_name']
    level = db_settings['level']
    # mat = loadmat('/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/pickle/pour1.mat');
    path = db_settings['voxellabelledlevelvideo_path'].format(action_name=action[0], video_name=video_name[action[0]][0], level=level)
    print path
    mat = loadmat(path)
    # pickle_path =
    superpixels_num = mat['total_number_of_supervoxels']
    return getips(conf, solver, superpixels_num)

def getips(conf, net, superpixels_num, layer='inner_product_target'):
    layer = options.layer
    data = net.blobs[layer].data
    #data = net.blobs['InnerProduct1'].data
    feature_len = data.shape[1]
    try:
        negative_numbers = conf.model['number_of_negatives']
    except:
        negative_numbers = 1
    reps = np.zeros((superpixels_num*negative_numbers, feature_len))
    for i in xrange(superpixels_num*negative_numbers):
        if i%1000==1:
            print i
        net.forward()
        reps[i] = np.sum(net.blobs[layer].data, axis=1)
    reps_slice = reps[..., 0]
    from sklearn.preprocessing import MinMaxScaler
    clf = MinMaxScaler()
    reps_slice = clf.fit_transform(reps_slice)
    reps_slice = np.square(reps_slice)
    #reps_slice[reps_slice<np.mean(reps_slice)] = 0
    for i in xrange(reps_slice.shape[0]):
        reps[i] = reps_slice[i]
        # print net.blobs['inner_product_target'].data[1:10]
    return reps

def getRepresentations(conf, net, superpixels_num, layer='inner_product_target'):
    data = net.blobs[layer].data
#this one works on databases that records of each target are together in the database table!
# target0 [...neighbors...] negative0
# target1 [...neighbors...] negative0
# ...
# target0 [...neighbors...] negative1
# target1 [...neighbors...] negative1
# ...
    #data = net.blobs['InnerProduct1'].data
    assert data.shape[0] == 1, 'batch size != ? ... this assert is not important'
    feature_len = data.shape[1]
    reps = np.zeros((superpixels_num, feature_len))
    try:
        negative_numbers = conf.model['number_of_negatives']
    except:
        negative_numbers = 1
    #for i in xrange(superpixels_num*negative_numbers):
    for i in xrange(superpixels_num):
=======
    for i in xrange(superpixels_num):
        if i%1000==1:
            print i,'/',superpixels_num
        net.forward()
        reps[i][...] = net.blobs['inner_product_target'].data
        # print net.blobs['inner_product_target'].data[1:10]
    return reps

#this one works on databases that records of each target are together in the database table!
# target0 [...neighbors...] negative0
# target0 [...neighbors...] negative1
# ...
# target1 [...neighbors...] negative0
# target1 [...neighbors...] negative1
# ...
def getRepresentations_old(conf, net, superpixels_num):
    data = net.blobs['inner_product_target'].data
    #data = net.blobs['InnerProduct1'].data
    assert data.shape[0] == 1, 'batch size != ? ... this assert is not important'
    feature_len = data.shape[1]
    reps = np.zeros((superpixels_num, feature_len))
    try:
        negative_numbers = conf.model['number_of_negatives']
    except:
        negative_numbers = 1
    for i in xrange(superpixels_num*negative_numbers):
>>>>>>> upstream/master
        if i%1000==1:
            print i
        net.forward()
        #if i%negative_numbers == 0:
        #    reps[i/negative_numbers][...] = net.blobs[layer].data
        reps[i] = net.blobs[layer].data
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
    print path
    files = filter(os.path.isfile, glob.glob(search_dir + "*"))
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-2]

def computeSimilarities(config_id):
    conf = getConfigs(config_id)
    snapshot_path = conf.solver['snapshot_prefix']
    caffemodel_path = getLastAddedFile(snapshot_path)
    caffe.set_mode_gpu()
    db_settings = conf.db_settings
    test_model_path = conf.model['test_prototxt_path']
    db_list = [x.split('\n')[0] for x in open(conf.model['database_list_path'])]
    test_model =  caffe.Net(test_model_path, caffemodel_path, caffe.TEST)
    print "last snapshot is:", caffemodel_path
    print 'Experiment number:', conf.experiment_number
    print 'creating model'
    if conf.db == 'vsb100':
        print 'vsb100'
        representations = getVSB100Representation(conf, test_model)
    elif conf.db == 'jhmdb':
        print 'jhmdb'
        if options.layer == 'weights':
            representations = getweights(conf, test_model)
        else:
            representations = getJHMDBRepresentations(conf, test_model)
    else:
        raise
    similarities = computeDistanceMatrix(representations)
    print representations[:10, :] #'creating model'
    if options.layer == 'weights':
        configs = getConfigs(config_id)
        count = 0
        for x in db_list:
            h5_f = h5py.File(x, 'r+')
            n_data = h5_f['data_weights'].shape[0]
            h5_f['data_weights'].write_direct(representations[count:n_data])
            count += n_data
            h5_f.close()
    else:
        savemat(conf.experiments_path+'/similarities.mat', {'similarities':similarities})

if __name__ == '__main__':
    computeSimilarities(int(options.exp_id))
