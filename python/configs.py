#In the name of GOD
from utils import *
from Logger import *
import cPickle as pickle
from Segmentation import *

class Config:
    def __init__(self, experiment_number=None, comment=None, action_name=None):
        # self.experiments_root = '/cs/vml2/mkhodaba/cvpr16/expriments/'
        self.experiments_root = '/local-scratch/expriments/'
        # self.visualization_path = '/cs/vml2/smuralid/projects/embedding_segmentation/python/Visualization/'
        self.comment = comment
        if not experiment_number:
            self.__create_config__(action_name)

    def __create_config__(self, action_name=None):
        self.frame_format = '{0:05d}.ppm'
        vals = [0]
        for x in getDirs(self.experiments_root):
            try:
                vals.append(int(x))
            except ValueError:
                vals.append(int(x.split('-')[0]))

        self.experiment_number = max(vals)+1
        print 'Experiment number is:', self.experiment_number
        if self.comment:
            self.experiments_path = self.experiments_root+'/{0}-{1}/'.format(self.experiment_number,self.comment)
        else:
            self.experiments_path = self.experiments_root+'/{0}/'.format(self.experiment_number)

        mkdirs(self.experiments_path)
        self.log_path = self.experiments_path+'/log.txt'
        self.log_type = LogType.FILE
        # self.db = 'vsb100' # self.db = 'jhmdb' or 'vsb100'
        self.db = 'jhmdb'
        self.model = {
            'batch_size':    	64,
            'number_of_neighbors':    8, #number of neighbors around the target superpixel
            'number_of_negatives':  4,
            'inner_product_output':    128, #2*(3*256+192),
            'weight_lr_mult':    1,
            'weight_decay_mult':    1,
            'b_lr_mult':    	2,
            'b_decay_mult':    	0,
            'model_prototxt_path':    self.experiments_path+'/model.prototxt',
            'test_prototxt_path':    self.experiments_path+'/test.prototxt',
            'database_list_path':    self.experiments_path+'/database_list.txt',
            'feature_type':    	[FeatureType.FCN, FeatureType.HOF, FeatureType.CLR]#FeatureType.COLOR_HISTOGRAM#
        }

        self.solver = {
            'weight_decay':    	0.00001,
            'base_lr':    	    0.01,
            'momentum':     	0.9,
            'gamma':    	    0.8,
            'power':    	    0.75,
            'display':    	    500,
            'test_interval':    100000,
            'test_iter':        1,
            'snapshot':        2000,
            'lr_policy':         "step",
            'stepsize':        1500,
            'snapshot_prefix':    self.experiments_path+'/snapshot/',
            'net':    		self.model['test_prototxt_path'],
            '_train_net':    	self.model['model_prototxt_path'],
            '_test_nets':    	self.model['test_prototxt_path'],
            'max_iter':    	10000,
            '_train_interval':    500,
            '_termination_threshold':0.0004,
            '_solver_prototxt_path':    self.experiments_path+'/solver.prototxt',
            '_model_prototxt_path':    self.model['model_prototxt_path'],
            '_solver_log_path':    self.experiments_path+'/solver.log',
        }
        mkdirs(self.solver['snapshot_prefix'])
        self.results_path = self.experiments_path+'/results.txt'
        db_settings = getattr(self, '__' + self.db + '__')
        self.db_settings = db_settings(action_name)
        # old version: (but why?)
        #self.db_settings = {
    	#	'jhmdb': self.__jhmdb__(),
    	#	'vsb100': self.__vsb100__(),
    	#}


    def __jhmdb__(self, action_name=None):
        jhmdb = {
            'db':    			'jhmdb',
            'action_name':    	['arctic_kayak'] if action_name is None else [action_name],# ['vw_commercial'], #['pour'],
            'level':    		7,
            'video_name':    {},	
            # 'starting_frame':   0,
            'frame':    		None, #if you set frame to None it will run on all the frames in orig_path folder
            'frame_format':    		self.frame_format,
            'number_of_negatives':  self.model['number_of_negatives'],
            'number_of_neighbors':    	self.model['number_of_neighbors'],
            'root_path':    		'/cs/vml3/mkhodaba/{action_name}/',
            'orig_path':    		'/cs/vml2/mkhodaba/datasets/VSB100/Test/{action_name}/ppm/',
            # 'annotation_path':    	'/cs/vml3/mkhodaba/cvpr16/dataset/{action_name}/{video_name}/puppet_mask.mat',
            'annotation_path':    	'/cs/vml2/mkhodaba/datasets/VSB100/files/{action_name}/puppet_mask.mat',
            # 'segmented_path':    	'/cs/vml3/mkhodaba/cvpr16/dataset/{action_name}/{video_name}/seg/{level:02d}/',  #+frame_format,
            'segmented_path':    	'/cs/vml2/mkhodaba/datasets/VSB100/segmented_frames/{action_name}/{level:02d}/',  #+frame_format,
            'optical_flow_path':    	'/cs/vml2/mkhodaba/datasets/VSB100/Test_flow/{action_name}/',
            'fcn_path':                 '/cs/vml2/smuralid/projects/eccv16/python/preprocessing/fcn/Test/{action_name}/',
            'features_path':     	'/cs/vml2/mkhodaba/datasets/VSB100/files/{action_name}/{feature_name}_{level}.npz',
            'output_path':    		self.experiments_path + 'indices.mat',#+frame_format
            'pickle_path':            '/cs/vml2/mkhodaba/datasets/VSB100/files/{action_name}/{level:02d}.p',
            'database_path':        self.experiments_path+'/{level:02d}.h5',
            'pixellabelledlevelvideo_path':   '/cs/vml2/mkhodaba/datasets/VSB100/files/{action_name}/pixellabelledlevelvideo_{level:02d}.mat',
            'voxellabelledlevelvideo_path':   '/cs/vml2/mkhodaba/datasets/VSB100/files/{action_name}/voxellabelledlevelvideo_{level:02d}.mat',
            'test_database_list_path':    self.experiments_path+'/database_list_{name}.txt',
            'database_list_path':    	self.model['database_list_path'],
            'feature_type':    		self.model['feature_type'],
        }
        if jhmdb['frame'] is None:
            jhmdb['frame'] = getNumberOfFiles(jhmdb['orig_path'].format(action_name=jhmdb['action_name'][0]))
            print 'This video has {} frames', jhmdb['frame']
        print 'action_name', jhmdb['action_name']
        for action in jhmdb['action_name']:
            jhmdb['video_name'][action] = ['b1'] 
        return jhmdb


    def __vsb100__(self,action_name=None):
            vsb100 = {
                'db':    			'vsb100',
                'action_name':    		'vw_commercial', #['pour'],
                'frame_format':    		self.frame_format,
                'root_path':    		'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/{action_name}/',
                'features_path':     	'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/features/{action_name}_features.mat',
                'video_info_path':    	'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/features/{action_name}_vidinfo.mat',
                'output_path':    		'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/output/{action_name}/', #+frame_format
                'number_of_neighbors':    	self.model['number_of_neighbors'], #number of neighbors per frame
                'neighbor_frames_num':    	2,
                'test_database_list_path':    self.experiments_path+'/database_list_test.txt',
                'database_list_path':    	self.model['database_list_path'],
            }
            vsb100['database_path'] = '/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/databases/{action_name}' + '_' + \
                    str(self.model['number_of_neighbors']) + '_' + \
                    str(vsb100['neighbor_frames_num']) +'.h5'
                    # str(self.model['inner_product_output']) + '_' + \
            self.model['number_of_neighbors'] = vsb100['number_of_neighbors'] * (2 * vsb100['neighbor_frames_num']+1) #total number of neighbors
            return vsb100

    def getPath(self, key):
        level = self.db_settings['level']
        action_name = self.db_settings['action_name'][0]
        video_name = self.db_settings['video_name'][action_name][0]
        path = self.db_settings[key]
        return path.format(level=level, action_name=action_name, video_name=video_name)

    def save(self):
        '''
        dic = {action_name:--, level:--, video_name:--}
        '''
        pickle.dump(self, open(self.experiments_path+'configs.txt', 'w'))
        with open(self.experiments_path+'configs.readable', 'w') as config_txt:
            import pprint
            pp = pprint.PrettyPrinter(indent=4,stream=config_txt)
            dic = {'db':self.db, 'model': self.db_settings, 'solver':self.solver}
            pp.pprint(dic)
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

def getConfigs(experiment_num=None, comment=None, action_name=None):
    conf = Config(experiment_num, comment, action_name)
    vals = [0]
    for x in getDirs(conf.experiments_root):
        try:
            vals.append(int(x))
        except ValueError:
            vals.append(int(x.split('-')[0]))

    if experiment_num is not None:
    	if experiment_num == -1:
            experiment_num = max(vals)
        experiment_folder_name = ''
        for dir_name in getDirs(conf.experiments_root):
            if dir_name.startswith(str(experiment_num)):
                experiment_folder_name = dir_name
                break
    	# conf = pickle.load(open(conf.experiments_root+str(experiment_num)+'/configs.txt', 'r'))
    	conf = pickle.load(open(conf.experiments_root+experiment_folder_name+'/configs.txt', 'r'))
    return conf


