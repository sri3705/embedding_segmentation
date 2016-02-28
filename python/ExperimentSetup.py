print "experiment setups initiating ... "
from configs import *
print "configs imported"
from NetworkFactory import *
print "networkfactory imported"
from DatabaseGenerator import *
print "databasegenerator imported"
from Logger import *
print "logger imported"
import sys
import os
from optparse import OptionParser
print "os, sys  imported"

parser = OptionParser()
parser.add_option('-s', '--recompute_features', dest='s', action="store_true", default=False)
parser.add_option('-f', '--compute-db', dest='f', action="store_true", default=False)
parser.add_option('-v', '--video', dest='v', default=None)
parser.add_option('-n', '--net', dest='n', default=None, type="int")
parser.add_option('-c', '--comment', dest='c')
parser.add_option('-b', '--batch-size', dest='b', default=None, type="int")
parser.add_option('-a', '--neighbors', dest='a', default=None, type="int")
parser.add_option('-A', '--negatives', dest='A', default=None, type="int")
parser.add_option('-S', '--stepsize', dest='S', default=None, type="int")
parser.add_option('-o', '--innerprod', dest='o', default=None, type="int")
parser.add_option('-F', '--features', dest='F', default=None, help='feature type')
parser.add_option('-m', '--model', dest='m', action="store_true", default=False)
parser.add_option('-l', '--level', dest='l', default=None)
parser.add_option('-L', '--baselr', dest='L', default=None)
(options, args) = parser.parse_args()




def labelledlevelvideo_generator(conf):
    # This function gives you the label of SuperPIXELS not supervoxels
    #TODO put the labelledllevelvideo.py here
    # from PIL import Image
    # from scipy.io import savemat
    # import numpy as np

    # action_name = conf.db_settings['action_name'][0]
    # video_name = conf.db_settings['video_name'][action_name][0]
    # level = 10
    segmented_path = conf.getPath('segmented_path')
    print segmented_path
    # out_path1 = conf.db_settings['pixellabelledlevelvideo_path'].format(video_name=video_name, action_name=action_name, level=level)
    frame_number = conf.db_settings['frame']
    print '[ExperimentSetup::labelledlevelvideo_generator] frame:', frame_number

    #frame_number = 21
    out_path = conf.getPath('pixellabelledlevelvideo_path')
    # assert out_path1 == out_path2
    import glob
    print 'segmented_path=', segmented_path
    img = Image.open(glob.glob(segmented_path+"*.ppm")[0])
    size = img.size
    sups_nums = np.zeros((1,frame_number))
    mat = np.zeros((size[1]/2, size[0]/2, frame_number), dtype=np.int32)
    print '[ExperimentSetup::labelledlevelvideo_generator] creating labelledlevelvideo.mat'
    for i,img_path in enumerate(glob.glob(segmented_path+"*.ppm")):
        if i == frame_number:
            break
        print 'image', i
        img = Image.open(img_path)
        width, height = img.size
        colors = {}
        counter = 1
        for w in xrange(0,width,2):
            for h in xrange(0,height,2):
                pix = img.getpixel((w,h))
                if pix not in colors:
                    colors[pix] = counter
                    counter += 1
                mat[h/2][w/2][i] = colors[pix]
        sups_nums[0,i] = counter-1
    from scipy.io import savemat
    savemat(out_path, {'labelledlevelvideo':mat, 'numberofsuperpixelsperframe':sups_nums})


def setup_experiment(extract_features=False, visualization=False, comment=None, compute_segment=False, action_name=None, args=None):
    # need to extract features?
    config = getConfigs(comment=comment, action_name=action_name, args=args)
    experiment_path = config.experiments_root
    print "Experiment number:", config.experiment_number
    logger = Logger(config.log_type, config.log_path)
    logger.log('Configs created:')
    logger.log(str(config))
    logger.log('Creating Network ...')
    createNetwork(config.model)
    logger.log('Adding test layers ...')
    addTestLayers(config)
    logger.log('Creating Solver prototxt ...')
    with open(config.solver['_solver_prototxt_path'], 'w') as f:
       for key, val in config.solver.iteritems():
           if not key.startswith('_'):
               f.write('{key}:{val}\n'.format(key=key, val='"{0}"'.format(val) if isinstance(val, str) else val))

    out_path = config.getPath('pixellabelledlevelvideo_path')
    if extract_features:
        print 'extract features'
        if compute_segment:
            config.db_settings['compute_segment'] = True
        else:
            config.db_settings['compute_segment'] = False
        if not os.path.exists(out_path):
            print 'labels are not there. Computing labelledlevelvideo_pixels'
            labelledlevelvideo_generator(config)
        createDatabase(config.db, config.db_settings, logger)
        #TODO create the database list
        #TODO: probably in configs need to set how to merge them: for now separately
    else:
        import glob
        folders = sorted(glob.glob(experiment_path + '*/'))[-2:]
        os.system('cp -f ' + folders[0] + '/*h5 ' + folders[1] + '/')
        write_db_list(config.db_settings, logger)
    logger.close()
    if not os.path.exists(out_path):
        print 'labels are not there'
        labelledlevelvideo_generator(config)
   #TODO save configs
    config.save()

if __name__=='__main__':

    model = {'net': options.n, 'batch_size': options.b, 'number_of_neighbors': options.a, \
        'number_of_negatives': options.A, 'inner_product_output': options.o, \
        'feature_type': options.F}

    db_args = {'level': options.l}

    solver = {'stepsize': options.S, 'base_lr':options.L}
    args = {'model': model, 'solver': solver, 'db_args': db_args}
    for arg_i in args.keys():
        for arg_key in args[arg_i].keys():
            if args[arg_i][arg_key] is None:
                del args[arg_i][arg_key]

    extract_features = options.f
    comment = options.c
    action_name = options.v
    compute_segment = options.s
    setup_experiment(extract_features=extract_features, visualization=False, comment=comment, compute_segment=compute_segment, action_name=action_name, args=args)



#if __name__=='__main__':
#
#    if "-f" in sys.argv:
#        extract_features = True
#    else:
#        extract_features = False
#    if "-s" in sys.argv:
#        compute_segment = True
#    else:
#        compute_segment = False
#    if "-v" in sys.argv:
#        j = sys.argv.index('-v')
#        action_name = sys.argv[j+1]
#    else:
#        action_name = None
#    comment = ''
#    i = sys.argv.index('-m')
#    comment = sys.argv[i+1]
#    print "extract_features = ", extract_features
#    setup_experiment(extract_features=extract_features, visualization=False, comment=comment, compute_segment=compute_segment, action_name=action_name)
