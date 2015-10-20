from configs import getConfigs
from NetworkFactory import createNetwork
from DatabaseGenerator import createDatabase
from Logger import Logger

def setup_experiment(extract_features=False, visualization=False):
	# need to extract features?
	config = getConfigs()
	logger = Logger(config.log_type, config.log_path)
	logger.log('Configs created:')
	logger.log(str(config))
	logger.log('Creating Network ...')
	createNetwork(config.model)
	logger.log('Creating Solver prototxt ...')
	with open(config.solver['solver_prototxt_path'], 'w') as f:
		for key, val in config.solver.iteritems():
			f.write('{key}:{val}\n'.format(key=key, val=val))
	if extract_features:
		#TODO: ^^^^ add neighbor_num to db_settigs shit!
		print 'extract features'
		createDatabase(config.db, config.db_settings[config.db], logger)
		#TODO create the database list
		#TODO: probably in configs need to set how to merge them: for now separately
	logger.close()

	#TODO save configs
	config.save()

if __name__=='__main__':

	setup_experiment(extract_features=True, visualization=False)
