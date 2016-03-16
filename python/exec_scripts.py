import os
from configs import getConfigs
import sys
import tempfile, subprocess, pprint
from optparse import OptionParser
from scipy.io import loadmat, savemat
os.environ['GLOG_minloglevel'] = '1'

parser = OptionParser()
parser.add_option('-s', '--recompute_features', dest='s', action="store_true", default=False)
parser.add_option('-f', '--compute-db', dest='f', action="store_true", default=False)
parser.add_option('-v', '--video', dest='v', default=None)
parser.add_option('-m', '--model', dest='m', default='0')
parser.add_option('-n', '--net', dest='n', default=None)
parser.add_option('-c', '--comment', dest='c', default=None)
parser.add_option('-b', '--batch-size', dest='b', default=None)
parser.add_option('-a', '--neighbors', dest='a', default=None)
parser.add_option('-A', '--negatives', dest='A', default=None)
parser.add_option('-S', '--stepsize', dest='S', default=None)
parser.add_option('-o', '--innerprod', dest='o', default=None)
parser.add_option('-O', '--innerprod_l1', dest='O', default=None)
parser.add_option('-F', '--features', dest='F', default=None, help='feature type')
parser.add_option('-l', '--level', dest='l', default=None)
parser.add_option('-L', '--baselr', dest='L', default=None)
parser.add_option('-N', '--negative_selector_param', dest='N', default=None)
parser.add_option('-B', '--bag_size', dest='B', default=None)

###Features need to be like this: FCN_HOF...

(options, args) = parser.parse_args()
if not options.c:
    parser.error('Comment not provided...exiting')
    sys.exit()

if options.F is not None:
    feats = options.F.split('_')
    if '' in feats:
        feats = feats[feats!='']
    try:
        [x in ['HOF', 'FCN', 'CLR'] for x in feats]
    except:
        print 'wrong feature type!\nexiting....'
        sys.exit()

#if options.video.endswith('txt'):
if options.v is None:
    options.v = 'rock_climbing'
#model = {'network':}
#solver = {'stepsize':}
#args = {'model':model, 'solver':solver}


in_args = ["python", "ExperimentSetup.py" ]
def select(x):
    if type(x[1]) is not bool:
        print 'not bool:', ['-' + x[0], x[1]]
        return ['-' + x[0], x[1]]
    else:
        if x[1]:
            print 'x[1]:', ['-' + x[0]]
            return ['-'+x[0]]
        else:
            print 'x[0]:', ['']
            return ['']
[in_args.extend(select(x)) for x in options.__dict__.items() if x[1] is not None]

result = subprocess.Popen(in_args, stdout=subprocess.PIPE)
result = result.stdout.read().split('\n')
pprint.pprint(result)

exp_id = [x for x in result if x.startswith('Experiment number: ')][0]
args = exp_id.split(': ')[-1]
os.system('python new_solver.py ' + args)
conf = getConfigs(int(args))
snapshot_path = conf.solver['snapshot_prefix']
exp_root = os.path.basename(os.path.dirname(snapshot_path[:-1]))
exp_root = conf.experiment_folder_name
os.system('python compute_similarities.py -e ' + args)
#os.system('python compute_similarities_vox2pix.py ' + args)
#os.system('mv -f ' + conf.experiments_path + '/similarities.mat '+ conf.experiments_path + '/similarities_1.mat')
#result = subprocess.Popen(in_args.append['-F', 'FCN', '-E', exp_root], stdout=subprocess.PIPE)
#result = result.stdout.read().split('\n')
#os.system('python new_solver.py ' + args)
#os.system('python compute_similarities.py -e ' + args)
#x = np.add(loadmat(conf.experiments_path + 'similarities_1.mat')['similarities'], loadmat(conf.experiments_path + 'similarities.mat')['similarities'])
#np.savemat(conf.experiments_path + 'similarities.mat', similarities=x)
from scipy.io import loadmat
try:
    loadmat('/local-scratch/experiments/' + exp_root + '/similarities.mat')
except:
    sys.exit()

gt_list = [(x.split('\n')[0]).split(' ') for x in open('gt_list.txt')]
gt_idx = [x[1] for x in gt_list if x[0] == options.v]
if len(gt_idx) == 1:
    #cmd = 'matlab -nosplash -nodisplay -r ' + '"' + 'run_all(' + "'" + exp_root + "'" + ',' + "'{0}'".format(conf.db_settings['level']) + ', ' + "'{0}'".format(options.v) + ');exit;' + '"'
    cmd = 'matlab -nosplash -nodisplay -r ' + '"' + 'run_all(' + "'" + exp_root + "'" + ',' + "'{0}'".format(conf.db_settings['level']) + ', ' + "'{0}'".format(options.v) + ',' + str(gt_idx) + ');exit;' + '"'
else:
    cmd = 'matlab -nosplash -nodisplay -r ' + '"' + 'run_all(' + "'" + exp_root + "'" + ',' + "'{0}'".format(conf.db_settings['level']) + ', ' + "'{0}'".format(options.v) + ');exit;' + '"'

print cmd
os.system(cmd)
