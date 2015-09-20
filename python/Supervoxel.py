from PIL import Image
import math

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

	def __init__(self, ID):
		try:
			hash(ID)
		except TypeError:
			raise Exception('ID must be immutable (hashable)')
		self.__ID__ = ID
		self.pixels = {} # frame -> set of (x,y)
#		self.pixels = set() # set of tupels (x, y, frame_number)
		self.colors_dict = {} # (x,y,f) -> (R, G, B) actual color in the frame
		self.colors_hist = {} # histogram of actual colors
		self.sum_x = 0
		self.sum_y = 0
		self.sum_t = 0
		self.number_of_pixels = 0

		self.bin_width = 130
		self.n_bins = int(math.ceil(255.0/self.bin_width))
		
		self.R_hist = [0 for i in xrange(self.n_bins)]
		self.G_hist = [0 for i in xrange(self.n_bins)]
		self.B_hist = [0 for i in xrange(self.n_bins)]
	
	def addVoxel(self, x,y,t, color):
		if t not in self.pixels.keys():
			self.pixels[t] = set()
		self.pixels[t].add((x,y))
		self.colors_dict[ (x, y, t) ] = color
		self.sum_x += x
		self.sum_y += y
		self.sum_t += t
		self.number_of_pixels += 1
		self.__update_hist__(color)

	def __update_hist__(self, color):
		self.R_hist[int(color[0]/self.bin_width)] += 1				
		self.G_hist[int(color[1]/self.bin_width)] += 1		
		self.B_hist[int(color[2]/self.bin_width)] += 1		
		
	def getFeature(self):
		return [i*1.0/self.number_of_pixels for i in self.R_hist+self.G_hist+self.B_hist]

	def getPixelsAtFrame(self, f):
		return self.pixels[f]

	def availableFrames(self):
		return self.pixels.keys()

	def center(self):
		n = self.number_of_pixels
		return (self.sum_x/n, self.sum_y/n, self.sum_t/n)

	def __str__(self):
		return "Supervoxel [ID:"+str(self.__ID__)+ ", Center:"+str(self.center()) + "]"

	def __eq__(self, other):
		return self.__ID__ == other.__ID__

	def __hash__(self):
		return hash(self.__ID__)


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
	
	def __init__(self, original_path='./orig/{0:05d}.ppm', segmented_path='./seg/{0:05d}.ppm'):
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
	#			all_voxels.append((p, sv.__ID__)) #pair of (pixel, ID). ID is usually color in segmented image
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

	def visualizeSegments(self, supervoxels_set, from_path='./orig/{0:05d}.ppm', to_path='./save/{0:05d}.ppm'):
		mkdirs(to_path)
		all_frames = set()
		for sv in supervoxels_set:
			all_frames.update(sv.pixels.keys())

		for f in all_frames:
			img = MyImage(from_path.format(f))
			for sv in supervoxels_set:
				if f in sv.pixels:
					for x,y in sv.pixels[f]:
						img.putpixel(x,y, sv.__ID__)
			img.save(to_path.format(f))		

	def doneProcessing(self):
		self.supervoxels_list = self.supervoxels.values()
		self.cKDTree = cKDTree(np.array([sv.center() for sv in self.supervoxels_list]))
		

	def getSupervoxelAt(self, x, y, t):
		pixel = (x,y)
		for sv in self.frame_to_voxels[t]:
			if pixel in sv.pixels[t]:
				return sv
	
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
		nearestNeighbors = self.cKDTree.query(np.array(supervoxel.center()), k+1)[1] # Added one to the neighbors because the target itself is included
	
		return set([self.supervoxels_list[i] for i in nearestNeighbors])
	
	def prepareData(self, k, number_of_data, feature_vec_size):
		feature_size = feature_vec_size * (1 + k + 1) #One for the target, k for neighbors, one for negative
		data = np.arange(number_of_data*feature_size)
		#data = data.reshape(number_of_data, 1, 1, feature_size)
		data = data.reshape(number_of_data, feature_size)
		data = data.astype('float32')

		print "data.shape: ", data.shape
		return data

	def dummyData(self, number_of_data, feature_vec_size):
		data = np.arange(number_of_data*feature_vec_size)
		data = data.reshape(number_of_data, feature_vec_size)
		data = data.astype('float32')

		return data

	def getFeatures(self, k):
		supervoxels = set(self.supervoxels_list)
		k = 1
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
			neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)

			data['target'][i][...] = sv.getFeature()
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][i][...] = nei.getFeature()
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][i][...] = negative.getFeature()
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

def main():
	level = 20
	frame_format = '{0:05d}.ppm'
	seg_path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/output_gbh/{0:02d}/'.format(level)
	orig_path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/frames_ppm/' + frame_format
	first_output = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/mehran_gbh/{0:02d}/'.format(level)	
	output_path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/mehran_gbh/{0:02d}/0/'.format(level)

	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/dataset1.h5'
	
	# Preparing data for 
	#segmentor = Segmentation(orig_path, seg_path+frame_format)
	segmentor = Segmentation(seg_path+frame_format, seg_path+frame_format)
	for i in range(1, 3):
		print i
		segmentor.processNewFrame()
	segmentor.doneProcessing()
	print "Total number of supervoxels: {0}".format(len(segmentor.supervoxels))

	
	#sv = segmentor.getSupervoxelAt(27, 127, 20)
	#print sv
	#supervoxels = segmentor.getKNearestSupervoxelsOf(sv, 6)
	#supervoxels.remove(sv)
	#for s in supervoxels:
	#	print s


	#TODO check if features are correct
	for sv in segmentor.supervoxels_list:
		#R_hist = [0 for i in xrange(13)]
		#G_hist = [0 for i in xrange(13)]
		#B_hist = [0 for i in xrange(13)]
		#R_hist[int(sv.__ID__[0]/20)] += 1
		#G_hist[int(sv.__ID__[1]/20)] += 1
		#B_hist[int(sv.__ID__[2]/20)] += 1
		print sv.getFeature()
		#print R_hist+G_hist+B_hist
		print "ID: {0}".format(sv.__ID__)		
		#print sum(sv.getFeature())/3		
		#print "Num pixels: {0}".format(sv.number_of_pixels)

		

	#TODO create database
	datasets = segmentor.getFeatures(1)
	#print data
	#database_path = '
	mkdirs(dataset_path)
	database = DB(dataset_path)
	for name, data in datasets.iteritems():
		database.save(data, name)
	#database.save(dataset)	
	database.close()

	#f = h5py.File(dataset_path, 'r')
	#print f['data'].value
	

	#segmentor.visualizeSegments([sv], orig_path, first_output+frame_format)
	#segmentor.visualizeSegments(supervoxels, first_output+frame_format, output_path+frame_format)


if __name__ == "__main__":
	main()





