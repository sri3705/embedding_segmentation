#In the name of GOD


import cPickle as pickle
from Segmentation import *
from Annotation import JHMDBAnnotator as JA
import time
from configs import getConfigs

def createDatabase(db_name, db_settings, logger):
	if db_name == 'jhmdb':
		createJHMDB(db_settings, logger)
	elif db_name == 'ucf_sports':
		createUCFSports(db_settings, logger)

def createJHMDB(db_settings, logger):
	frame_format = db_settings['frame_format']
	action_name = db_settings['action_name']
	video_name = db_settings['video_name'] 
	annotation_path = db_settings['annotation_path']	
	segmented_path = db_settings['segmented_path']	
	orig_path = db_settings['orig_path']
	level = db_settings['level']
	frame = db_settings['frame']
	pickle_path = db_settings['pickle_path']
	neighbor_num = db_settings['number_of_neighbors'] #TODO add this to db_settings in experimentSetup	
	database_path = db_settings['database_path']
	database_list_path = db_settings['database_list_path']
	#TODO: maybe we should save them segarately
	#TODO: write a merge segment function?
	segmentors = {}
	logger.log('*** Segment parsing ***')
	for action in action_name:
		segmentors[action] = {}
		for video in video_name[action]:
			logger.log('Processing action:`{action}`, video:`{video}`:'.format(action=action, video=video))
			annotator = JA(annotation_path.format(action_name=action, video_name=video))
			segmentor = MySegmentation(orig_path.format(action_name=action, video_name=video, level=level)+frame_format,
							segmented_path.format(action_name=action, video_name=video, level=level)+frame_format,
							annotator)
			for i in xrange(frame):
				logger.log('frame {0}'.format(i+1))
				segmentor.processNewFrame()
			segmentor.doneProcessing()
			logger.log("Total number of supervoxels: {0}".format(len(segmentor.supervoxels)))
			segmentors[action][video]= segmentor

	logger.log('*** Pickling ***')
	s = time.time()
	for action in action_name:
		for video in video_name[action]:
			logger.log('Piclking action:`{action}`, video:`{video}` ...'.format(action=action, video=video))
			pickle.dump(segmentors[action][video], open(pickle_path.format(action_name=action, video_name=video, level=level), 'w'))
			logger.log('Elapsed time: {0}'.format(time.time()-s))
			s = time.time()
		
	logger.log('*** Collecting features / Creating databases ***')
	keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]	
	feats = []	
#	feats = [features]
	#logger.log('video 1 done!')
	with open(database_list_path, 'w') as db_list:
		for action in action_name:
			for video in video_name[action]:
				db_path = database_path.format(action_name=action, video_name=video, level=level)
				database = DB(db_path)
				features = segmentors[action][video].getFeatures(neighbor_num)
				for name, data in features.iteritems():
					database.save(data, name)
				database.close()
				db_path.write(db_path);
	logger.log('done!')

def createUCFSports(db_settings, log_path):
	pass

	










	
def create_dbs():
	configs = getConfigs()
	
	frame_format = configs.frame_format
	seg_path = configs.seg_path
	orig_path = configs.orig_path
	first_output = configs.first_output
	output_path = configs.output_path
	dataset_path = configs.dataset_path
	annotation_path = configs.annotation_path
	action
	feature_name = '256bin'
	level = 2
	segmentors = []
	vid_num = 2
	frames_per_video = 31
	if 1 == 1:
		for dd in range(vid_num):
			d = dd+1
			print 'b{0}'.format(d)
			annotator = JA(annotation_path.format(name='b'+str(d)))
			segmentor = MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format, annotator)
			for i in range(1, frames_per_video):
				print "processing frame {i}".format(i=i)
				segmentor.processNewFrame()
			segmentor.doneProcessing()
			segmentors.append(segmentor)
			print "Total number of supervoxels: {0}".format(len(segmentor.supervoxels))
			print
	
		try:
			mkdirs(dataset_path)
		except:
			pass
		print 'Piclking ...'
		t = time.time()
		for i in range(vid_num):
			pickle.dump(segmentors[i], open(dataset_path.format(name='segment_{0}.p'.format(i+1)), 'w'))
			print '{0}-th done. time elapsed: {1}'.format(i+1, time.time()-t)
			t = time.time()

		#TODO create database
	else:
		for i in range(vid_num):
			segmentors.append(pickle.load(open(dataset_path.format(name='segment_{0}.p'.format(i+1)), 'r')))
		

	database = DB(dataset_path.format(name='videos{v}_feature{f}_lvl{l}.h5'.format(\
							v='_'.join(map(str,range(1,vid_num))),
							f=feature_name,
							l=level)))
		
	print 'Collecting features ...'
	neighbor_num = 6
	keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]	
	features = segmentors[0].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 1 done!'
	for i in range(1, len(segmentors)):
		tmp = segmentors[i].getFeatures(neighbor_num)
		#feats.append(tmp)
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


	print 'done!'

	#for i in range(len(segmentors)):
	#	print i
	#	segmentors[i] = Segmentation(segment=segmentors[i])

	#print 'pickle segments ...'
	#pickle.dump( segmentors, open(dataset_path.format(name='segmentors_lvl1.p'), 'w'))
	#print 'pickle features ...'	
	#pickle.dump( feats, open(dataset_path.format(name='features_lvl1.p'), 'w'))

if __name__ == '__main__':
	create_dbs()
