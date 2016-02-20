import numpy as np
from scipy.io import loadmat, savemat
import caffe, sys
from configs import *
import cPickle as pickle
from scipy.spatial import cKDTree

def normalizeSimilarities(similarities):
    min_sim = similarities.min()
    max_sim = similarities.max()
    print 'min:', min_sim, ' max:', max_sim
    # similarities = similarities.astype('float16')
    normalized_similarities = (similarities-min_sim) / (max_sim-min_sim)
    print 'min:', normalized_similarities.min(), ' max:', normalized_similarities.max()
    return normalized_similarities

def refineSimilarities(config_id):
    conf = getConfigs(config_id)
    print 'refineSimilarities', conf.comment
    mat = loadmat(conf.experiments_path+'/similarities_old.mat')
    similarities = mat['similarities']
    similarities = normalizeSimilarities(similarities)
    fade_multiplier = 0.7
    refined_similarities = np.ones(similarities.shape)*fade_multiplier
    # refined_similarities = np.ones(similarities.shape)*similarities.min()
    print 'normalization done'
    segment = pickle.load(open(conf.getPath('pickle_path')))
    centers = np.array([sv.center() for sv in segment.supervoxels_list])
    kdtree = cKDTree(centers)
    sup_num = similarities.shape[0]
    for i in xrange(sup_num):
        if i % 1000 == 0:
            print i, '/', sup_num
        neighbors = kdtree.query(centers[i], 50)[1]
        for nei in neighbors:
            refined_similarities[i][nei] = 1.0 
            # refined_similarities[i][nei] = similarities[i][nei]
    # savemat(conf.experiments_path+'/refined_similarities.mat', {'similarities':refined_similarities})    
    refined_similarities = np.multiply(similarities, refined_similarities)
    savemat(conf.experiments_path+'/similarities.mat', {'similarities':refined_similarities})    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        refineSimilarities(-1)
    else:
        refineSimilarities(int(sys.argv[1]))
