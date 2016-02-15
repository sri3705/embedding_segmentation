from scipy.io import savemat, loadmat
from configs import *
import numpy as np
import caffe, os, glob, sys
import h5py

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
    negative_numbers = conf.model['number_of_negatives'] 
    for i in xrange(superpixels_num*negative_numbers):
        if i%1000==1:
            print i
        net.forward()
        if i%negative_numbers == 0:
                reps[i/negative_numbers][...] = net.blobs['inner_product_target'].data[...]
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
    conf = getConfigs(config_id)
    snapshot_path = conf.solver['snapshot_prefix']
    caffemodel_path = getLastAddedFile(snapshot_path + '/')
    caffe.set_mode_gpu()
    db_settings = conf.db_settings
    test_model_path = conf.model['test_prototxt_path']
    test_model =  caffe.Net(test_model_path, caffemodel_path, caffe.TEST)
    print "last snapshot is:", caffemodel_path
    print 'Experiment number:', conf.experiment_number 
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

def buildVoxel2PixelSimilarities(config_id):
    conf = getConfigs(config_id)
    #Load similarities
    mat = loadmat(conf.experiments_path+'/similarities.mat')
    voxelSimilarities = mat['similarities']
    #Load superpixel information (1-based)
    mat = loadmat('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat')
    mapped = mat['mapped']
    numberofsuperpixelsperframe = mat['numberofsuperpixelsperframe']
    total_superpixels_num = np.sum(numberofsuperpixelsperframe)
    pixellabelledlevelvideo=mat['labelledlevelvideo']
    #Load supervoxel information (1-based)
    db_settings = conf.db_settings
    action = db_settings['action_name']
    video_name = db_settings['video_name']
    level = db_settings['level']
    path = db_settings['labelledlevelvideo_path'].format(action_name=action[0], video_name=video_name[action[0]][0], level=level)
    mat = loadmat(path)
    voxellabelledlevelvideo=mat['labelledlevelvideo']
    #A simple check
    # assert voxellabelledlevelvideo.shape == pixellabelledlevelvideo.shape, "voxel and pixel information doesn't matchi: %s != %s" % (voxellabelledlevelvideo.shape, pixellabelledlevelvideo.shape)
    voxel_shape = voxellabelledlevelvideo.shape
    pixel_shape = pixellabelledlevelvideo.shape
    assert voxel_shape[0] == 2 * pixel_shape[0] 
    assert voxel_shape[1] == 2 * pixel_shape[1] 
    height, width, frames = pixellabelledlevelvideo.shape
    print 'frames, width, height', frames, width, height
    #Do the mapping
    print 'total_superpixels_num', total_superpixels_num
    pixel_to_voxel = [-1 for i in xrange(total_superpixels_num)]
    for f in xrange(frames):
        print '[compute_similarities_vox2pix] frame:', f
        for w in xrange(width):
            for h in xrange(height):
                pixel_label = pixellabelledlevelvideo[h][w][f]-1 #TO make it 0-based
                voxel_label = voxellabelledlevelvideo[h*2][w*2][f]-1 #To make it 0-based
                superpixel_idx = mapped[f][pixel_label]-1
                if pixel_to_voxel[superpixel_idx] == -1:
                    pixel_to_voxel[superpixel_idx] = int(voxel_label)
                else:
                    assert pixel_to_voxel[superpixel_idx] == voxel_label, "Supervoxel should compeletely overlap with supervoxel"
                # except Exception as e:
                    # print e
                    # print superpixel_idx
                    # print voxel_label
                    # print pixel_label
                    # print f,w,h
                    # exit()
    # similarities = np.zeros((total_superpixels_num, total_superpixels_num))
    print 'changing voxel similarities to list'
    voxelSimilarities = voxelSimilarities.tolist()
    print 'initializing similarities'
    similarities = [[0 for _ in xrange(total_superpixels_num)] for ii in xrange(total_superpixels_num)]
    print 'initialize mapping'
    for i in xrange(total_superpixels_num):
        for j in xrange(total_superpixels_num):
            similarities[i][j] = voxelSimilarities[pixel_to_voxel[i]][pixel_to_voxel[j]]
        if i % 100 == 0:
            print i
    # savemat(conf.experiments_path+'/similarities_superpixels.mat', {'similarities':similarities})
    with h5py.File(conf.experiments_path+'/similarities_superpixels.h','w') as hf:
        similarities = np.array(similarities)
        print 'similarities converted to np.array'
        hf.create_dataset('similarities', data=similarities) 

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # computeSimilarities(-1)
        buildVoxel2PixelSimilarities(-1)
    else:
        print '[compute_similarities_vox2pix.py::__main__] compute voxel similarities'
        config_id = int(sys.argv[1])
        # computeSimilarities(config_id)
        print '[compute_similarities_vox2pix.py::__main__] compute voxel2pixel similarities' 
        buildVoxel2PixelSimilarities(config_id)
