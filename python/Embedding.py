#In the name of God
from Supervoxel import *
import caffe
import numpy as np
import cPickle as pickle
from scipy.spatial.distance import euclidean
class EmbeddingTester:
	def __init__(self, segments=None, features=None):
		self.segments = segments
		self.features = features

	def loadModel(self, snapshot_path, net_path):
		self.net_path = net_path
		self.snapshot_path = snapshot_path
		self.model = caffe.Net(net_path, snapshot_path, caffe.TEST)
	
def extractFeatures(seg):
	seg.__class__ = MySegmentation
	random_i = 10
	from_path = seg.original_path
	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/01/{0:05d}.ppm'
	seg.visualizeSegments([seg.supervoxels_list[random_i]], from_path, to_path)
	print 'extract features ...'
	features = MySegmentation.getFeatures(seg, 6)
	print 'done extracting ...'
	return features

def prepare(test_model, snapshot_model, features):
	sh = features['target'].shape
	for key in features.keys():
		test_model.blobs[key].reshape(sh[0], sh[1])
	test_model.params['inner_product_target'][0].data[...] = snapshot_model.params['inner_product_target'][0].data[...]
	test_model.params['inner_product_target'][1].data[...] = snapshot_model.params['inner_product_target'][1].data[...]
	test_model.blobs['target'].data[...] = features['target'][...]
	
def getRepresentations(test_model):
	test_model.forward()
	return test_model.blobs['inner_product_target'].data[...]
def getEuDistances(reps, target):
	dists = np.zeros((reps.shape[0], 2))
	for i in range(reps.shape[0]):
		dists[i][0] = euclidean(reps[i][...], reps[target][...])
		dists[i][1] = i
	dists= sorted(dists, key=lambda x: x[0])
	return dists

#TODO implement cosine distance
def getCosineDistances(reps, target):
	dists = np.zeros((reps.shape[0], 2))
	for i in range(reps.shape[0]):
		dists[i][0] = -1*np.dot(reps[i][...], reps[target][...])
		dists[i][1] = i
	dists= sorted(dists, key=lambda x: x[0])
	return dists

def getCosineDistances_old(reps, target):
	target_s = np.tile(reps[target][...], (reps.shape[0], 1))
	cos_dot = target_s * reps
	dists = np.sum(cos_dot, axis=1)
	dists = np.column_stack((dists, np.arange(reps.shape[0])))
	dists = sorted(dists, key=lambda x: x[0])
	return dists



def visualizeTopN(dists, seg, n):
	dd = sorted(dists, key=lambda x: x[0])
	indices = [int(x[1]) for x in dd[1:n]]
	retrieval = [seg.supervoxels_list[i] for i in indices]
	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/02/{0:05d}.ppm'
	seg.visualizeSegments(retrieval, seg.original_path, to_path)

def visualize(target):
	seg.__class__ = MySegmentation
	from_path = seg.original_path
	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/01/{0:05d}.ppm'
	seg.visualizeSegments([seg.supervoxels_list[target]], from_path, to_path)
	

def main():
	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
	snapshot_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/snapshot/vml_gpu/many/{name}'	
	net_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/{name}'	
	snapshot_name = '_iter_870000.caffemodel'
	net_name = 'model.prototxt'
	embeddingTester = EmbeddingTester()
	embeddingTester.loadModel(snapshot_path.format(name=snapshot_name), net_path.format(name=net_name))
	embeddingTester.test_model = caffe.Net(net_path.format(name='model_test.prototxt'), snapshot_path.format(name=snapshot_name), caffe.TEST)

	print 'pickling segments/features ...'
	segments = pickle.load(open(dataset_path.format(name='segmentors_lvl1.p'), 'r'))
	features = pickle.load(open(dataset_path.format(name='features_lvl1.p'), 'r'))
	print 'done pickling ...'

	embeddingTester.segments = segments
	embeddingTester.features = features

	return embeddingTester
	#for key in features.keys():
	#	embeddingTester.model.blobs[key].reshape(shape[0], shape[1])
	#embeddingTester.model.blobs['target'].data[...] = features['target'][...]
	#embeddingTester.model.forward()
	
	#target_embedding = embedding.model.blobs['inner_product_target']
	
	#return embeddingTetser

#if __name__ == '__main__':
#	main()
	
