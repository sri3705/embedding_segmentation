#In the name of GOD
from utils import mkdirs, getDirs
from Logger import LogType
import cPickle as pickle
class Config:
	def __init__(self, experiment_number=None):
		
		self.experiments_root = '/cs/vml2/mkhodaba/cvpr16/expriments/'

		if not experiment_number:
			self.__create_config__()

	def __create_config__(self):
		self.frame_format = '{0:05d}.png'
		
		self.experiment_number = max([0]+map(int, getDirs(self.experiments_root)))+1
		self.experiments_path = self.experiments_root+'/{0}/'.format(self.experiment_number)
		mkdirs(self.experiments_path)

		self.log_path = self.experiments_path+'/log.txt'
		self.log_type = LogType.PRINT			

		self.db = 'jhmdb'

	
		self.model = {
			'batch_size':		20,
			'number_of_neighbors':	6,
			'inner_product_output':	3*256,
			'weight_lr_mult':	1,
			'weight_decay_mult':	1,
			'b_lr_mult':		2,
			'b_decay_mult':		0,
			'model_prototxt_path':	self.experiments_path+'/model.prototxt',
			'dataset_list_path':	self.experiments_path+'/dataset_list.txt',
			
			
		}

		self.solver = {
			'weight_decay':		0.0005,
			'base_lr':		0.09,
			'momentum': 		0.1,
			'gamma':		0.0001,
			'power':		0.75,
			'display':		100000,
			'test_interval':	500,
			'test_iter':		100,
			'snapshot':		4,
			'snapshot_prefix':	self.experiments_path+'/snapshot/',
			'solver_prototxt_path':	self.experiments_path+'/solver.prototxt',						
			'model_prototxt_path':	self.model['model_prototxt_path'],
			'train_net':		self.model['model_prototxt_path'],
			'test_net':		self.model['model_prototxt_path'],
		}


		self.results_path = self.experiments_path+'/results.txt'

		


		self.db_settings = {
			'jhmdb': self.__jhmdb__(),
		}

	def __jhmdb__(self):
		jhmdb = {
			'action_name':		['pour'],
			'level':		0,
			'video_name':		{},
			'frame':		29,
			'frame_format':		self.frame_format,
			'number_of_neighbors':	self.model['number_of_neighbors'],
			'root_path':		'/cs/vml2/mkhodaba/datasets/JHMDB/frames/{action_name}/',
			'orig_path':		'/cs/vml2/mkhodaba/datasets/JHMDB/frames/{action_name}/{video_name}/', #+frame_format
			'annotation_path':	'/cs/vml2/mkhodaba/datasets/JHMDB/puppet_mask/{action_name}/{video_name}/puppet_mask.mat',
			'segmented_path':	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/features/{action_name}/{video_name}/data/results/images/motionsegmentation/{level:02d}/',  #+frame_format,
			'features_path': 	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/features/{action_name}/{video_name}/features.txt',
			'output_path':		'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/output/{action_name}/{video_name}/{level:02d}/{experiment_number}/', #+frame_format
			'database_path':	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/databases/{action_name}/{video_name}/{level:02d}.h5',
			'pickle_path':		'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/pickle/{action_name}/{video_name}/{level:02d}.p',
			'database_list_path':	self.model['database_list_path']
		}		
		num_videos = 1 #set to None for all
		
		for action in jhmdb['action_name']:
			#TODO this line!
			jhmdb['video_name'][action] = getDirs(jhmdb['root_path'].format(action_name=action))[2:num_videos+2] #TODO #TODO This should be changed!!!!!!!!!!!!!


		return jhmdb

	def save(self):
		'''
			dic = {action_name:--, level:--, video_name:--}
		'''
		
		#with open(self.experiments_path+'exp.txt', 'w') as f:
		#	f.write(str(self))
		pickle.dump(self, open(self.experiments_path+'configs.txt', 'w'))

	#def __str__(self):
	#	from yaml import dump
	#	s = ''+\
	#	'experiment numebr: {0}\n'.format(self.experiment_number)+\
	#	'experiment path: {0}\n\n'.format(self.experiments_path)+\
	#	'model:\n{0}\n'.format(dump(self.model, default_flow_style=False, indent=4))+\
	#	'solver:\n{0}\n'.format(dump(self.solver, default_flow_style=False, indent=4))+\
	#	'db -> {0}\n'.format(self.db)
	#	db = self.db_settings[self.db]
	#	db['experiment_number'] = self.experiment_number
		#for key in db.keys():
		#	if key.endswith('_path'):
		#		
		#		db[key] = db[key].format(**db)
	#	s+='db settings:\n{0}\n'.format(dump(db, default_flow_style=False, indent=4))
		#print s
	#	return s

def getConfigs(experiment_num=None):
	
	conf = Config(experiment_num)
	if experiment_num is not None:
		conf = pickle.load(open(conf.experiments_root+str(experiment_num)+'/configs.txt', 'r'))
	return conf


