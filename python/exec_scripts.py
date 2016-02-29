import os
from configs import getConfigs
import sys
import tempfile, subprocess
from optparse import OptionParser
os.environ['GLOG_minloglevel'] = '1'

parser = OptionParser()
parser.add_option('-s', '--recompute_features', dest='s', action="store_true", default=False)
parser.add_option('-f', '--compute-db', dest='f', action="store_true", default=False)
parser.add_option('-v', '--video', dest='v', default=None)
parser.add_option('-m', '--model', dest='m', action="store_true", default=False)
parser.add_option('-n', '--net', dest='n', default=None)
parser.add_option('-c', '--comment', dest='c', default=None)
parser.add_option('-b', '--batch-size', dest='b', default=None)
parser.add_option('-a', '--neighbors', dest='a', default=None)
parser.add_option('-A', '--negatives', dest='A', default=None)
parser.add_option('-S', '--stepsize', dest='S', default=None)
parser.add_option('-o', '--innerprod', dest='o', default=None)
parser.add_option('-F', '--features', dest='F', default=None, help='feature type')
parser.add_option('-l', '--level', dest='l', default=None)
parser.add_option('-L', '--baselr', dest='L', default=None)
###Features need to be like this: FCN_HOF...

(options, args) = parser.parse_args()
if not options.c:
    parser.error('Comment not provided...exiting')
    sys.exit()

if options.F is not None:
    feats = options.F.split('_')
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

print options.__dict__.items()

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
print in_args


result = subprocess.Popen(in_args, stdout=subprocess.PIPE)
result = result.stdout.read().split('\n')
exp_id = [x for x in result if x.startswith('Experiment number: ')][0]
args = exp_id.split(': ')[-1]
os.system('python new_solver.py ' + args)
conf = getConfigs(int(args))
snapshot_path = conf.solver['snapshot_prefix']
exp_root = os.path.basename(os.path.dirname(snapshot_path[:-1]))
if options.m == True:
    os.system('python compute_similarities.py -e ' + args + ' -l weights')
    print 'Finetuning data!'
    from compute_similarities import getLastAddedFile
    #conf = getConfigs(-1)
    caffemodel_path = getLastAddedFile(snapshot_path).replace('caffemodel', 'solverstate')
    solver_model_path = conf.solver['_solver_prototxt_path']

    if conf.solver['stepsize'] < 800:
        solver = [x.split('\n')[0] for x in open(solver_model_path)]
        f = open(solver_model_path, 'w')
        #max_iter = 0.5*conf.solver['max_iter']
        #base_lr = conf.solver['base_lr']
        #new_lr = (0.001/base_lr)
        for xid, x in enumerate(solver):
            if x.startswith('base_lr'):
                solver[xid] = 'base_lr: 0.09'

        for x in solver:
            f.write(x + '\n')

        f.close()
    os.system('caffe train -snapshot ' + caffemodel_path + ' -solver ' + solver_model_path)

os.system('python compute_similarities.py -e ' + args)
cmd = 'matlab -nosplash -nodisplay -r ' + '"' + 'run_all(' + "'" + exp_root + "'" + ',' + "'{0}'".format(conf.db_settings['level']) + ', ' + "'{0}'".format(options.v) + ');exit;' + '"'
print cmd
# os.system(cmd)
