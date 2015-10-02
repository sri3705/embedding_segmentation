from PIL import Image
import math

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass

class MyImage:
	def __init__(self, path):
		self.img = Image.open(path)
		self.size = self.img.size

	def getcolors(self):
		#return self.img.convert('RGB').getcolors()
		colors = {}
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				colors[self.getpixel(x,y)] = True
		return colors.keys()

	def getpixel(self, i, j):
		return self.img.getpixel((i, j))

	def putpixel(self, i, j, color):
		self.img.putpixel((i,j), color)

	
	def save(self, path):
		self.img.save(path)



class Supervoxel:

	def __init__(self, ID, bin_num=11):
		'''
		:param arg1: the id of the supervoxel (we use color-tuple -> (r,g,b))
		:param arg2: number of histogram bins per channel
		:type arg1: any hashable object (we use tuple)
		:type arg2: int
		'''
		try:
			hash(ID)
		except TypeError:
			raise Exception('ID must be immutable (hashable)')
		self.ID = ID
		self.pixels = {} # frame -> set of (x,y)
		self.colors_dict = {} # (x,y,f) -> (R, G, B) actual color in the frame		
		self.__initializeCenter()
		self.__initializeHistogram()

	def __initializeCenter(self):
		self.sum_x = 0
		self.sum_y = 0
		self.sum_t = 0
		self.number_of_pixels = 0
			
	def __initializeHistogram(self):
		self.R_hist = [0 for i in xrange(256)]
		self.G_hist = [0 for i in xrange(256)]
		self.B_hist = [0 for i in xrange(256)]

	def addVoxel(self, x,y,t, color):
		if t not in self.pixels.keys():
			self.pixels[t] = set()
		self.pixels[t].add((x,y))
		self.colors_dict[ (x, y, t) ] = color
		self.sum_x += x
		self.sum_y += y
		self.sum_t += t
		self.number_of_pixels += 1
		self.__updateHistogram(color)

	def __updateHistogram(self, color):
		self.R_hist[color[0]] += 1				
		self.G_hist[color[1]] += 1		
		self.B_hist[color[2]] += 1		
		
	#def __recomputeFeatures(self, bin_num):
	#	self.n_bins = bin_num
	#	self.__initializeHistogram()
	#	for color in colors_dict.values():
	#		self.__updateHitsogram(color)
	#	return self.getFeature()

	def getFeature(self, number_of_bins=16):
		bin_width = 256/number_of_bins
		bin_num = -1
		r_hist = [0 for i in xrange(number_of_bins)]
		g_hist = r_hist[:]
		b_hist = r_hist[:]

		for i in xrange(256):
			if i%bin_width == 0:
				bin_num+=1
			r_hist[bin_num]+=self.R_hist[i]
			g_hist[bin_num]+=self.G_hist[i]
			b_hist[bin_num]+=self.B_hist[i]
		return [i*1.0/self.number_of_pixels for i in r_hist+g_hist+b_hist]


	def getPixelsAtFrame(self, f):
		return self.pixels[f]

	def availableFrames(self):
		return self.pixels.keys()

	def center(self):
		n = self.number_of_pixels
		return (self.sum_x/n, self.sum_y/n, self.sum_t/n)

	def __str__(self):
		return "Supervoxel [ID:"+str(self.ID)+ ", Center:"+str(self.center()) + "]"

	def __eq__(self, other):
		return self.ID == other.ID

	def __hash__(self):
		return hash(self.ID)


from scipy.spatial import cKDTree
import numpy as np
import os.path
from os import makedirs 

def mkdirs(path):
	dir_path = os.path.dirname(path)
	try:
		makedirs(dir_path)
	except Exception as e:
		print e

class Segmentation:
	

	def __init__(self, original_path='./orig/{0:05d}.ppm', segmented_path='./seg/{0:05d}.ppm', segment=None):
		if segment is not None:
			attrs = [a for a in dir(segment) if not a.startswith('__') and not callable(getattr(segment,a))]
			for attr in attrs:
				print attr
				setattr(self, attr, getattr(segment, attr))
			return
		self.supervoxels = {} # ID -> Supervoxel
		self.frame_to_voxels = {} # frame (int) -> Supervoxel
		self.current_frame = 1
		self.original_path = original_path
		self.segmented_path = segmented_path
		self.__cKDTRee__ = None #cKDTree() for finding the neighbors. This attribute is set in donePrecessing method
		self.supervoxels_list = None # list of all supervoxels. This attribute is set in donePrecessing method
	
	#Not in use
#	def __extract_supervoxels__(self, original_img_path, segmented_img_path, frame_number):
#		orig_img = MyImage(original_img_path)
#		img = MyImage(segmented_img_path)
#		voxel_colors = img.getcolors()
#		supervoxels = {}
#		for color in voxel_colors:
#			supervoxels[color] = Supervoxel(color)		
#		for x in range(img.size[0]):
#			for y in range(img.size[1]):
#				color = img.getpixel(x, y)
#				supervoxels[color].addVoxel(x, y, frame_number, orig_img.getpixel(x, y))
#		return supervoxels
	
	def addSupervoxels(self, original_img_path, segmented_img_path, frame_number):
		orig_img = MyImage(original_img_path)
		img = MyImage(segmented_img_path)
		voxel_colors = img.getcolors()
		#print "Colors"
		#for c in voxel_colors:
		#	print c

		self.frame_to_voxels[frame_number] = set()
		for color in voxel_colors:
			if color not in self.supervoxels:
				self.supervoxels[color] = Supervoxel(color)
			self.frame_to_voxels[frame_number].add(self.supervoxels[color])


		for x in range(img.size[0]):
			for y in range(img.size[1]):
				color = img.getpixel(x, y)
				self.supervoxels[color].addVoxel(x, y, frame_number, orig_img.getpixel(x, y)) 	

	def processNewFrame(self):
		orig_path = self.original_path.format(self.current_frame)
		seg_path = self.segmented_path.format(self.current_frame)
		self.addSupervoxels(orig_path, seg_path, self.current_frame)
		self.current_frame += 1

	#TODO: Re-implement this one!
	#def saveSegments(self, supervoxels_set, from_path='./orig/{0:05d}.ppm', to_path='./save/{0:05d}.ppm'):
	#	mkdirs(to_path)
	#	all_voxels = []
	#	for sv in supervoxels_set:
	#		for p in sv.pixels:
	#			all_voxels.append((p, sv.ID)) #pair of (pixel, ID). ID is usually color in segmented image
	#
	#	all_voxels.sort(key=lambda x: x[0][2]) #sort all pixels of all supervoxels based on frame number
	#
	#	open_frame = all_voxels[0][0][1]
	#	img = MyImage(from_path.format(open_frame))
	#	for pixel, color in all_voxels:
	#		x,y,f = pixel
	#		if open_frame != f:
	#			img.save(to_path.format(open_frame))
	#			open_frame = f
	#			img = MyImage(from_path.format(open_frame))
	#		img.putpixel(x,y, color)		

	def visualizeSegments(self, supervoxels_set, from_path='./orig/{0:05d}.ppm', to_path='./save/{0:05d}.ppm', colors={}):
		mkdirs(to_path)
		all_frames = set()
		for sv in supervoxels_set:
			all_frames.update(sv.pixels.keys())

		for f in all_frames:
			img = MyImage(from_path.format(f))
			for sv in supervoxels_set:
				#clr = colors[sv.ID]
				if f in sv.pixels:
					for x,y in sv.pixels[f]:
						#img.putpixel(x,y, clr)
						img.putpixel(x,y, sv.ID)
			img.save(to_path.format(f))		

	def doneProcessing(self):
		self.supervoxels_list = self.supervoxels.values()
		
		

	def getSupervoxelAt(self, x, y, t):
		pixel = (x,y)
		for sv in self.frame_to_voxels[t]:
			if pixel in sv.pixels[t]:
				return sv
	
	
class MySegmentation(Segmentation):
	def __init__(self, segment=None):
		if  segment is None:
			super(Segmentation, self).__init__(segment.original_path, segmented_path)
		else:
			super(Segmentation, self).__init__(segment.original_path, segmented_path, segment)
			
	def getNearestSupervoxelsOf(self, supervoxel, threshold=30):
		pass

	def getKNearestSupervoxelsOf(self, supervoxel, k=6):
		'''
		:param arg1: supervoxel
		:param arg2: number of neighbors
		:type arg1: Supervoxel()
		:type arg2: int
		:return: set of neighbors of supervoxel
		:rtype: set()
		
		'''
		if not hasattr(self, 'cKDTree'):
			self.__cKDTree__ = cKDTree(np.array([sv.center() for sv in self.supervoxels_list]))
		nearestNeighbors = self.__cKDTree__.query(np.array(supervoxel.center()), k+1)[1] # Added one to the neighbors because the target itself is included
	
		return set([self.supervoxels_list[i] for i in nearestNeighbors[1:]])
	
	def prepareData(self, k, number_of_data, feature_vec_size):
		feature_size = feature_vec_size * (1 + k + 1) #One for the target, k for neighbors, one for negative
		data = np.arange(number_of_data*feature_size)
		#data = data.reshape(number_of_data, 1, 1, feature_size)
		data = data.reshape(number_of_data, feature_size)
		data = data.astype('float32')

		print "data.shape: ", data.shape
		return data

	def dummyData(self, number_of_data, feature_vec_size):
		'''
		:param arg1: number of data (n)
		:param arg2: length of the feature vector of each supervoxel (k)
		:type arg1: int
		:type arg2: int
		:return: an array of size n by k
		:rtype: numpy.array
		
		'''
		data = np.arange(number_of_data*feature_vec_size)
		data = data.reshape(number_of_data, feature_vec_size)
		data = data.astype('float32')

		return data

	def getFeatures(self, k):
		'''
		:param arg1: number of nieghbors (k)
		:type arg1: int
		:return: a dictionary that has the following keys: target, negative, neighbor0, neighbor1, ..., neighbork
			the value of each key is a numpy.array of size n by f, where n is the number of supervoxels in the
			video and f is the size of the feature vector of each supervoxel
		:rtype: dict
		
		'''
		supervoxels = set(self.supervoxels_list)
		feature_len = len(self.supervoxels_list[0].getFeature())
		n = len(supervoxels)
		data = {'target':self.dummyData(n, feature_len), 'negative':self.dummyData(n, feature_len)}
		for i in range(k):
			data['neighbor{0}'.format(i)] = self.dummyData(n, feature_len)
		for i, sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k) 
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negative = random.sample(supervoxels, 1)[0] #Sample one supervoxel as negatie
			#neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)

			data['target'][i][...] = sv.getFeature()
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][i][...] = nei.getFeature()
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][i][...] = negative.getFeature()
		#print data.keys()
		return data

	def getFeaturesInOne(self):
		supervoxels = set(self.supervoxels_list)
		k = 1
		feature_len = len(self.supervoxels_list[0].getFeature())
		data = self.prepareData(k, len(supervoxels), feature_len)

		for i,sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negative = random.sample(supervoxels, 1)[0] #Sample one supervoxel as negatie
			neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)

			data[i][0:feature_len] = sv.getFeature()
			for j, nei in enumerate(neighbors):
				data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data[i][(k+1)*feature_len:(k+2)*feature_len] = negative.getFeature()
		return data
	

import random
import h5py

class DB:
	
	def __init__(self, path):
		self.path = path
		self.h5pyDB = h5py.File(path, 'w')

	def __db__(self):
		return self.h5pyDB

	def save(self, data, name='data'):
		if isinstance(data, dict):
			for name, dataset in data.iteritems():
				self.h5pyDB.create_dataset(name, data=dataset, compression='gzip', compression_opts=1)
	
		else:
			data = np.array(data)
			data = data.astype('float32')
			self.h5pyDB.create_dataset(name, data=data, compression='gzip', compression_opts=1)
	
	
	def close(self):
		self.h5pyDB.close()

def doSegmentation(**kargs):
	pass
def doDataCollection(**kargs):
	pass


import cPickle as pickle

def main():

	frame_format = '{0:05d}.ppm'
	seg_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/seg/{1:02d}/' #+ frame_format
	orig_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/' #+ frame_format
	first_output = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/mymethod/{1:02d}/first/'#.format(level)	
	output_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/mymethod/{1:02d}/output/'#.format(level)
	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
	
	# Preparing data for 
	#segmentor = Segmentation(orig_path, seg_path+frame_format)
	level = 1
	segmentors = []
	vid_num = 4
	frames_per_video = 31
	for d in range(1,vid_num):
		print 'b{0}'.format(d)
		segmentor = MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format)
		for i in range(1, frames_per_video):
			print "processing frame {i}".format(i=i)
			segmentor.processNewFrame()
		segmentor.doneProcessing()
		segmentors.append(segmentor)
		print "Total number of supervoxels: {0}".format(len(segmentor.supervoxels))
		print
		
	#sv = segmentor.getSupervoxelAt(27, 127, 20)
	#print sv
	#supervoxels = segmentor.getKNearestSupervoxelsOf(sv, 6)
	#supervoxels.remove(sv)
	#for s in supervoxels:
	#	print s


	#TODO check if features are correct
	##for sv in segmentor.supervoxels_list:
		##print sv.getFeature()
		##print "ID: {0}".format(sv.ID)		

		#R_hist = [0 for i in xrange(13)]
		#G_hist = [0 for i in xrange(13)]
		#B_hist = [0 for i in xrange(13)]
		#R_hist[int(sv.ID[0]/20)] += 1
		#G_hist[int(sv.ID[1]/20)] += 1
		#B_hist[int(sv.ID[2]/20)] += 1
		#print R_hist+G_hist+B_hist
		#print sum(sv.getFeature())/3		
		#print "Num pixels: {0}".format(sv.number_of_pixels)

	

	#TODO create database
	mkdirs(dataset_path)
	database = DB(dataset_path.format(name='b1b2_train_16bins_lvl{0}.h5'.format(level)))

	print 'Collecting features ...'
	neighbor_num = 6
	keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]	
	features = segmentors[0].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 1 done!'
	for i in range(1, len(segmentors)-1):
		tmp = segmentors[i].getFeatures(neighbor_num)
		feats.append(tmp)
		for key in keys:
			features[key] = np.append(features[key], tmp[key], axis=0)	
		print 'video {0} done!'.format(i+1)
	#print data
	#database_path = '
	print 'saving to database ...'
	for name, data in features.iteritems():
		database.save(data, name)
	#database.save(dataset)	
	database.close()


	database = DB(dataset_path.format(name='b3_test_16bins_lvl{0}.h5'.format(level)))

	print 'Collecting features ...'
	neighbor_num = 6
	features = segmentors[-1].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 3 done!'
	#print data
	#database_path = '
	print 'saving to database ...'
	for name, data in features.iteritems():
		database.save(data, name)
	#database.save(dataset)	
	database.close()



	print 'done!'

	for i in range(len(segmentors)):
		print i
		segmentors[i] = Segmentation(segment=segmentors[i])

	print 'pickle segments ...'
	pickle.dump( segmentors, open(dataset_path.format(name='segmentors_lvl1.p'), 'w'))
	print 'pickle features ...'	
	pickle.dump( feats, open(dataset_path.format(name='features_lvl1.p'), 'w'))
	


if __name__ == "__main__":
	main()





