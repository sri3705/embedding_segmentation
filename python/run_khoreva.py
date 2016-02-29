from configs import *
import sys, os

if len(sys.argv) == 1: 
    conf = getConfigs(-1) 
else: 
    conf = getConfigs(int(sys.argv[1])) 
try:
    print conf.experiment_folder_name
except:
    if conf.comment:
            conf.experiment_folder_name = '{0}-{1}'.format(conf.experiment_number,conf.comment)
    else:
        conf.experiment_folder_name = '{0}'.format(conf.experiment_number)
    print conf.experiment_folder_name

# os.system('cd $graph; matlab_sh -r "run_all('{}','{}','{}');exit;"',conf.experiment_folder_name, conf.level, conf.db_settings['action_name'][0])
os.system('matlab -nodisplay -nosplash -nodesktop -r "run_all(\'{}\',\'{}\',\'{}\');exit;"'.format(conf.experiment_folder_name, conf.db_settings['level'], conf.db_settings['action_name'][0]))
