#In the name of GOD


import cPickle as pickle
from Segmentation import *
from Annotation import JHMDBAnnotator as JA
import time
from configs import getConfigs


def createDatabase(db_name, db_settings, logger):
    if db_name == 'jhmdb':
        # createJHMDB(db_settings, logger)
        createJHMDBParallel(db_settings, logger)
    elif db_name == 'ucf_sports':
        createUCFSports(db_settings, logger)
    elif db_name == 'vsb100':
        createVSB100(db_settings, logger)

def parallelProcess(inp):
    i, segg = inp
    print 'parallelProcess on frame', i
    segg.processNewFramePar(i+1)
    return segg

def createJHMDBParallel(db_settings, logger):
    frame_format = db_settings['frame_format']
    action_name = db_settings['action_name']
    video_name = db_settings['video_name']
    annotation_path = db_settings['annotation_path']
    segmented_path = db_settings['segmented_path']
    orig_path = db_settings['orig_path']
    level = db_settings['level']
    frame = db_settings['frame']
    n_neg = db_settings['number_of_negatives']
    pickle_path = db_settings['pickle_path']
    neighbor_num = db_settings['number_of_neighbors'] #TODO add this to db_settings in experimentSetup
    database_path = db_settings['database_path']
    database_list_path = db_settings['database_list_path']
    features_path = db_settings['features_path']
    feature_type = db_settings['feature_type']
    labelledlevelvideo_path = db_settings['voxellabelledlevelvideo_path']
    optical_flow_path = db_settings['optical_flow_path']
    compute_segment = db_settings['compute_segment']
    #TODO: maybe we should save them segarately
    #TODO: write a merge segment function?
    logger.log('*** Segment parsing ***')
    keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]
    fcn_path = db_settings['fcn_path']
    for action in action_name:
        for video in video_name[action]:
            logger.log('Processing action:`{action}`, video:`{video}`:'.format(action=action, video=video))
            try:
                annotator = JA(annotation_path.format(action_name=action, video_name=video))
            except:
                annotator = None
            segmentor_list = []
            if compute_segment:
                for i in xrange(frame):
                    print 'segment of frame:', i
                    segmentor = MySegmentation(orig_path.format(action_name=action, video_name=video, level=level)+frame_format,
                                    segmented_path.format(action_name=action, video_name=video, level=level)+frame_format,
                                    features_path.format(action_name=action, video_name=video, level=level),
                                    annotator,
                                    None,
                                    labelledlevelvideo_path.format(action_name=action, video_name=video, level=level),
                                    optical_flow_path.format(action_name=action, video_name=video, level=level)+frame_format,
                                    negative_neighbors=n_neg,
                                    fcn_path=fcn_path.format(action_name=action, video_name=video, level=level)+frame_format)
                    segmentor.setFeatureType(feature_type)
                    segmentor_list.append((i, segmentor))
                    # segmentor_list.append((i, MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format, annotator)))
                # parallelProcess = lambda pair: pair[1].processNewFrame(pair[0]) #pair = (frame_number, segment)
                from multiprocessing import Pool
                print 'create pool'
                pool = Pool()
                print 'defining function'
                        # pair[1].processNewFrame(pair[0]) #pair = (frame_number, segment)
                s = time.time()
                print 'parallelizing begins', 'Elapsed time:', time.time()-s
                s = time.time()
                parallelized_segmentor_list = pool.map(parallelProcess, segmentor_list)
                segmentor_list = None
                # for segg in parallelized_segmentor_list:
                    # segg.processNewFramePar(i+1)
                print 'Parallelizing DONE', 'Elapsed time: ', time.time()-s

                print 'Merging begins'
                s = time.time()
                segmentor = parallelized_segmentor_list[0]
                for i in xrange(1, len(parallelized_segmentor_list)):
                    segmentor.merge(parallelized_segmentor_list[i])
                print 'Elapsed time: ', time.time()-s
                # for i in xrange(frame):
                    # logger.log('frame {0}'.format(i+1))
                    # segmentor.processNewFrame()
                segmentor.doneProcessing()
                logger.log("Total number of supervoxels: {0}".format(len(segmentor.supervoxels)))
                logger.log('*** Pickling ***')
                s = time.time()
                logger.log('Elapsed time: {0}'.format(time.time()-s))
                pickle.dump(segmentor, open(pickle_path.format(action_name=action, video_name=video, level=level), 'w'))
                s = time.time()
                logger.log('Piclking action:`{action}`, video:`{video}` ...'.format(action=action, video=video))
                logger.log('*** Collecting features / Creating databases ***')
            else: # Don't compute segments
                logger.log('No need to compute segments')
                logger.log('loading segments')
                segmentor = pickle.load(open(pickle_path.format(action_name=action, video_name=video, level=level), 'r'))
                segmentor.__class__ = MySegmentation

            db_path = database_path.format(action_name=action, video_name=video, level=level)
            database = DB(db_path)
            features = segmentor.getFeatures(neighbor_num,feature_type=feature_type)
            for name, data in features.iteritems():
                database.save(data, name)
            database.close()
            logger.log("Segment {0} Done!\n".format(action))
    write_db_list(db_settings, logger)
    logger.log('done!')


def createJHMDB(db_settings, logger):
    frame_format = db_settings['frame_format']
    action_name = db_settings['action_name']
    video_name = db_settings['video_name']
    annotation_path = db_settings['annotation_path']
    segmented_path = db_settings['segmented_path']
    orig_path = db_settings['orig_path']
    level = db_settings['level']
    frame = db_settings['frame']
    n_neg = db_settings['number_of_negatives']
    pickle_path = db_settings['pickle_path']
    neighbor_num = db_settings['number_of_neighbors'] #TODO add this to db_settings in experimentSetup
    database_path = db_settings['database_path']
    database_list_path = db_settings['database_list_path']
    features_path = db_settings['features_path']
    feature_type = db_settings['feature_type']
    labelledlevelvideo_path = db_settings['voxellabelledlevelvideo_path']
    optical_flow_path = db_settings['optical_flow_path']
    fcn_path = db_settings['fcn_path']
    output_path = db_settings['output_path']
    #TODO: maybe we should save them segarately
    #TODO: write a merge segment function?
    logger.log('*** Segment parsing ***')
    keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]
    # from multiprocessing import Pool
    # pool = Pool()
    # parallelProcess = lambda seg: seg[1].processNewFrame(seg[0])
    fcn_path = db_settings['fcn_path']
    for action in action_name:
        for video in video_name[action]:
            logger.log('Processing action:`{action}`, video:`{video}`:'.format(action=action, video=video))
            try:
                annotator = JA(annotation_path.format(action_name=action, video_name=video))
            except:
                annotator = None
            segmentor = MySegmentation(orig_path.format(action_name=action, video_name=video, level=level)+frame_format,
                            segmented_path.format(action_name=action, video_name=video, level=level)+frame_format,
                            features_path.format(action_name=action, video_name=video, level=level),
                            annotator,
                            None,
                            labelledlevelvideo_path.format(action_name=action, video_name=video, level=level),
                            optical_flow_path.format(action_name=action, video_name=video, level=level)+frame_format,
                            negative_neighbors=n_neg,
                            fcn_path=fcn_path.format(action_name=action, video_name=video, level=level)+frame_format,
                            output_path=output_path)
            segmentor.setFeatureType(feature_type)
            # segmentor_list = []
            # for i in xrange(frames_per_vidoe):
                # segmentor_list.append((i, MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format, annotator)))
            # parallelized_segmentor_list = pool.map(parallelProcess, segmentor_list)

            for i in xrange(frame):
                logger.log('frame {0}'.format(i+1))
                segmentor.processNewFrame()
            segmentor.doneProcessing()
            logger.log("Total number of supervoxels: {0}".format(len(segmentor.supervoxels)))
            logger.log('*** Pickling ***')
            s = time.time()
            logger.log('Elapsed time: {0}'.format(time.time()-s))
            pickle.dump(segmentor, open(pickle_path.format(action_name=action, video_name=video, level=level), 'w'))
            s = time.time()
            logger.log('Piclking action:`{action}`, video:`{video}` ...'.format(action=action, video=video))
            logger.log('*** Collecting features / Creating databases ***')
            db_path = database_path.format(action_name=action, video_name=video, level=level)
            database = DB(db_path)
            features = segmentor.getFeatures(neighbor_num,feature_type=feature_type)
            #if type(feature_type) is list:
            #    feat_size = features[-1]
            #    features = features[0]
            #    for _id, feature_type_i in enumerate(feature_type):
            #        idx1 = sum(feat_size[:_id])
            #        idx2 = sum(feat_size[:(_id + 1)])
            #        for name, data in features.iteritems():
            #            database.save(data[..., idx1:idx2], feature_type_i.name + '_' + name)
            #else:
            #    for name, data in features.iteritems():
            #        database.save(data, name)
            for name, data in features.iteritems():
                database.save(data, name)
            database.close()
            logger.log("Segment {0} Done!\n".format(action))
    write_db_list(db_settings, logger)
    logger.log('done!')

def createJHMDB2(db_settings, logger):
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
    features_path = db_settings['features_path']
    feature_type = db_settings['feature_type']
    #TODO: maybe we should save them separately
    #TODO: write a merge segment function?
    segmentors = {}
    logger.log('*** Segment parsing ***')
    for action in action_name:
        segmentors[action] = {}
        for video in video_name[action]:
            logger.log('Processing action:`{action}`, video:`{video}`:'.format(action=action, video=video))
            try:
                annotator = JA(annotation_path.format(action_name=action, video_name=video))
                segmentor = MySegmentation(orig_path.format(action_name=action, video_name=video, level=level)+frame_format,
                                segmented_path.format(action_name=action, video_name=video, level=level)+frame_format,
                                features_path.format(action_name=action, video_name=video, level=level),
                                annotator)
                segmentor.setFeatureType(feature_type)
                for i in xrange(frame):
                    logger.log('frame {0}'.format(i+1))
                    segmentor.processNewFrame()
                segmentor.doneProcessing()
                logger.log("Total number of supervoxels: {0}".format(len(segmentor.supervoxels)))
                segmentors[action][video]= segmentor
            except Exception as e:
                logger.log('============================\n ERROR: video: "{0}" has problems...: {1}\n==========================='.format(video, str(e)))
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
#    feats = [features]
    #logger.log('video 1 done!')
    #with open(database_list_path, 'w') as db_list:
    for action in action_name:
        for video in video_name[action]:
            db_path = database_path.format(action_name=action, video_name=video, level=level)
            database = DB(db_path)
            features = segmentors[action][video].getFeatures(neighbor_num,feature_type=feature_type)
            for name, data in features.iteritems():
                database.save(data, name)
            database.close()
    #        db_list.write(db_path);
    write_db_list(db_settings, logger)
    logger.log('done!')



def write_db_list(db_settings, logger):
    if db_settings['db'] == 'jhmdb':
        action_name = db_settings['action_name']
        video_name = db_settings['video_name']
        database_path = db_settings['database_path']
        database_list_path = db_settings['database_list_path']
        test_database_list_path = db_settings['test_database_list_path']
        level = db_settings['level']
        with open(database_list_path, 'w') as db_list:
            for action in action_name:
                for i,video in enumerate(video_name[action]):
                    db_path = database_path.format(action_name=action, video_name=video, level=level)
                    db_list.write(db_path+'\n');
                    with open(test_database_list_path.format(name=i), 'w') as f:
                        f.write(db_path)
    elif db_settings['db'] == 'vsb100':
        action_name = db_settings['action_name']
        database_path = db_settings['database_path']
        database_list_path = db_settings['database_list_path']
        test_database_list_path = db_settings['test_database_list_path']
        with open(database_list_path, 'w') as db_list:
            db_path = database_path.format(action_name=action_name)
            db_list.write(db_path+'\n');
        with open(test_database_list_path, 'w') as db_test_list:
            db_path = database_path.format(action_name=action_name+'_test')
            db_test_list.write(db_path+'\n');

def createUCFSports(db_settings, log_path):
    pass


def createVSB100(db_settings, logger):
    '''
    This method creates the database needed for caffe.
    '''
    action = 'vw_commercial'
    database_path = db_settings['database_path']
    features_path = db_settings['features_path'] #'/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_features.mat'
    video_info_path = db_settings['video_info_path'] #'/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_vidinfo.mat'
    #database_path = '/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/databases/{action_name}.h5'
    #features_path = '/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_features.mat'
    #video_info_path = '/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_vidinfo.mat'
    features_path = features_path.format(action_name=action)
    video_info_path = video_info_path.format(action_name=action)
    database_path = database_path.format(action_name=action)
    neighbors_num = db_settings['number_of_neighbors']
    neighbor_frames_num = db_settings['neighbor_frames_num']

    from scipy.io import loadmat
    import numpy as np
    from scipy.spatial import cKDTree
    from random import randint
    from sklearn.preprocessing import StandardScaler
    try:
        features = loadmat(features_path)['features'] #number_of_frames x number_of_supervoxels_per_frame x feature_length
    except:
        import h5py
        features = h5py.File(features_path)
        print features.keys()
    video_info = loadmat(video_info_path) #video_info = [mapped, labelledlevelvideo, numberofsuperpixelsperframe]
                        #mapped -> #number_of_frames x number_of_supervoxels_per_frame
                        #labelledlevelvideo -> height x width x number_of_frames
                        #framebelong -> total_number_of_super_pixels x 1
                        #labelsatframe -> total_number_of_super_pixels x 1
    kdtrees = []
    labelledlevelvideo = video_info['labelledlevelvideo']
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe']
    numberofsuperpixelsperframe = numberofsuperpixelsperframe[0]
    print features.shape
    frames_num = len(features)
    superpixels_num = len(features[0]) #per frame
    feature_len = len(features[0][0])
    print features[0][0][1:50]
    normalize_data = False
    if normalize_data:
        features_normalized = np.zeros((np.sum(numberofsuperpixelsperframe), feature_len))
        print features_normalized.shape
        idx = 0
        for f in xrange(frames_num):
            for s in xrange(numberofsuperpixelsperframe[f]):
                features_normalized[idx][...] = features[f][s][...]
                idx += 1
        clf = StandardScaler()
        features_normalized_2 = clf.fit_transform(features_normalized)
        idx = 0
        for f in xrange(frames_num):
            for s in xrange(numberofsuperpixelsperframe[f]):
                features[f][s][...] = features_normalized_2[idx][...]
                idx +=1

    print features[0][0][1:50]
    print features.shape
    print frames_num, superpixels_num, feature_len
    print numberofsuperpixelsperframe
    #centers[f][i] -> h,w of center
    centers = np.zeros((frames_num, superpixels_num, 2)) #[[[0.0,0.0] for i in xrange(superpixels_num)] for j in xrange(frames_num)] #frames_num x superpixels_num x 2
    pixels_count = [[0 for i in xrange(superpixels_num)] for j in xrange(frames_num)] #frames_num x superpixels_num
    height = len(labelledlevelvideo)
    width = len(labelledlevelvideo[0])
    logger.log('Computing centers of superpixels ...')
    for f in xrange(frames_num):
        logger.log('Frame %d' % f)
        for h in xrange(height):
            for w in xrange(width):
                try:
                    idx = labelledlevelvideo[h][w][f]-1
                except:
                    print h, w, f
                    raise
                centers[f][idx][0] += h
                centers[f][idx][1] += w
                pixels_count[f][idx] += 1
        for i in xrange(numberofsuperpixelsperframe[f]):
            centers[f][i][0] /= pixels_count[f][i]
            centers[f][i][1] /= pixels_count[f][i]
        logger.log('Building kdtree')
        kdtree = cKDTree(np.array(centers[f][:numberofsuperpixelsperframe[f]]))
        kdtrees.append(kdtree)
    framebelong = video_info['framebelong']
    print framebelong.shape
    labelsatframe = video_info['labelsatframe']
    target_superpixel_num = 0
    for f in xrange(neighbor_frames_num, frames_num-neighbor_frames_num):
        target_superpixel_num += numberofsuperpixelsperframe[f]
    n = target_superpixel_num
    #len(framebelong)
    superpixel_skip_num = 0
    n_neg = 10
    for f in xrange(neighbor_frames_num):
        superpixel_skip_num += numberofsuperpixelsperframe[f]
    data = {'target':np.zeros((n*n_neg, feature_len)), 'negative':np.zeros((n*n_neg, feature_len))}
    #data = {'target':np.zeros((n, feature_len)), 'negative':np.zeros((n, feature_len))}
    #Tracer()()
    total_number_of_neighbors = neighbors_num  * (2*neighbor_frames_num+1)
    total_number_of_neighbors = neighbors_num  * (2*neighbor_frames_num+1)
    for i in range(total_number_of_neighbors):
        data['neighbor{0}'.format(i)] = np.zeros((n*n_neg, feature_len))
        #data['neighbor{0}'.format(i)] = np.zeros((n, feature_len))
    superpixel_idx = -1
    logger.log('Creating the database of superpixels:features')
    for f in xrange(neighbor_frames_num, frames_num-neighbor_frames_num): #TODO: start from a frame that has at least neighbor_frames_num number of frames before it
        logger.log('Frame %d' % f)
        logger.log('There are %d superpixels in in this frame' % numberofsuperpixelsperframe[f])
        for i in xrange(numberofsuperpixelsperframe[f]):
            superpixel_idx += 1
            assert f == framebelong[superpixel_idx+superpixel_skip_num]-1, 'Something went wrong in mapping superpixel index to frames/label at frame (1)'
            assert i == labelsatframe[superpixel_idx+superpixel_skip_num]-1, 'Something went wrong in mapping superpixel index to frames/label at frame (2)'
            data['target'][superpixel_idx*n_neg:(superpixel_idx + 1)*n_neg][...] = features[f][i][...]
            #data['target'][superpixel_idx][...] = features[f][i][...]

            center = centers[f][i]
            frame_start = max(0, f-neighbor_frames_num)
            frame_end = min(frames_num, f+neighbor_frames_num)
            neighbor_idx = 0
            #print frame_start, frame_end
            for target_frame in xrange(frame_start, frame_end+1):
                if f == target_frame:
                    nearest_neighbors = kdtrees[target_frame].query(center, neighbors_num+1)[1] # Added one to the neighbors because the target itself is included
                    nearest_neighbors = nearest_neighbors[1:]
                else:
                    nearest_neighbors = kdtrees[target_frame].query(center, neighbors_num)[1]
                for idx in nearest_neighbors:
                    #data['neighbor{0}'.format(neighbor_idx)][superpixel_idx*n_neg:(superpixel_idx + 1)*n_neg][...] = features[target_frame][idx][...]
                    data['neighbor{0}'.format(neighbor_idx)][superpixel_idx][...] = features[target_frame][idx][...]
                    neighbor_idx += 1
            assert neighbor_idx == total_number_of_neighbors, "Number of neighbors doesn't match ( %d != %d )" % (neighbor_idx, total_number_of_neighbors)
            #TODO: print "Random frame ... (Warning: if it's taknig too long stop it! \n Apparantly, the number of neighboring frames are relatively large \n with respect to the number of video frames)"
            # frame_random = randint(0, frames_num-1)
            # while frame_end-frame_start < 0.5*frames_num and frame_start <= frame_random <= frame_end:
                # frame_random = randint(0, frames_num-1)
            # idx_random = randint(0, numberofsuperpixelsperframe[ frame_random]-1)
            # data['negative'][superpixel_idx][...] = features[frame_random][idx_random][...]
            nearest_neighbors = kdtrees[f].query(center, 5*neighbors_num+n_neg)[1]
            #nearest_neighbors = kdtrees[f].query(center, 5*neighbors_num)[1]
            #It's the nearest of farthest superpixels to this one
            idx_random = nearest_neighbors[-1]
            if i == 10:
                print 'f, i, superpixel_idx, idx_random', f, i, superpixel_idx, idx_random
            #data['negative'][superpixel_idx][...] = features[f][idx_random][...]
            for j in xrange(n_neg):
                idx_random = nearest_neighbors[-j]
                data['negative'][superpixel_idx*n_neg + j][...] = features[f][idx_random][...]

    assert superpixel_idx+1 == target_superpixel_num, "Total number of superpixels doesn't match (%d != %d)" % (superpixel_idx, target_superpixel_num)
    db_path = database_path.format(action_name=action)
    print db_path
    database = DB(db_path)
    for name, datum in data.iteritems():
        database.save(datum, name)
    database.close()
    #Creating the database for extracting the final representations. It just needs to have the targets nothing else.

#    n = len(framebelong)
#    print 'n', n
#    data = {'target':np.zeros((n*n_neg, feature_len)), 'negative':np.zeros((n*n_neg, feature_len))}
#    total_number_of_neighbors = neighbors_num  * (2*neighbor_frames_num+1)
#    for i in range(total_number_of_neighbors):
#        data['neighbor{0}'.format(i)] = np.zeros((n*n_neg, feature_len))
#    superpixel_idx = 0
#    for f in xrange(1,frames_num-1):
#        for i in xrange(numberofsuperpixelsperframe[f]):
#            try:
#                data['target'][superpixel_idx*n_neg:(superpixel_idx + 1)*n_neg][...] = features[f][i][...]
#            except:
#                print superpixel_idx, f, i
#                raise
#            superpixel_idx +=1
#    database_path = db_settings['database_path']
#    db_path = database_path.format(action_name=(action+'_test'))
#    print 'test db path', db_path
#    database = DB(db_path)
#    for name, datum in data.iteritems():
#        database.save(datum, name)
#    database.close()
#
    write_db_list(db_settings, logger)

    logger.log('Creating database Done!')
    #TODO:
    # 1-Read the features, vid-labels mat files
    # 2-map each superpixel to an ID
    # 3-create a kdtree for superpixels of each frame
    # 4-loop over all superpixels of all frames.
    #     4.1-for each superpixel loop over current, previous, next frames and find neighbors
    #    4.2- concatenate features then push it in the database
    # done

    #This method is deprecated
def create_dbs():
    configs = getConfigs()

    frame_format = configs.frame_format
    seg_path = configs.seg_path
    orig_path = configs.orig_path
    first_output = configs.first_output
    output_path = configs.output_path
    dataset_path = configs.dataset_path
    annotation_path = configs.annotation_path
    n_neg = configs.number_of_negatives
    print 'n_neg \t= \t' + str(n_neg)
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
            segmentor = MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format, annotator, negative_neighbors=n_neg)
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
    #    print i
    #    segmentors[i] = Segmentation(segment=segmentors[i])

    #print 'pickle segments ...'
    #pickle.dump( segmentors, open(dataset_path.format(name='segmentors_lvl1.p'), 'w'))
    #print 'pickle features ...'
    #pickle.dump( feats, open(dataset_path.format(name='features_lvl1.p'), 'w'))

if __name__ == '__main__':
    create_dbs()
