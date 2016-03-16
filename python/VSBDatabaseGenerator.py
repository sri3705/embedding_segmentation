import h5py, numpy as np
from scipy.spatial import cKDTree
from Segmentation import FeatureType
from scipy.io import loadmat, savemat
import sys
def getNegatives(method, param, neighbors_all, negative_numbers, number_of_neighbors):
    neighbors_negatives = neighbors_all#[4*number_of_neighbors:]
    start = param
    # start = param*number_of_neighbors
    if method == 'close':
        negatives = neighbors_negatives[start:start+negative_numbers]
    elif method == 'far':
        negatives = neighbors_negatives[-(start+negative_numbers):-start]
    elif method == 'random':
        import random
        negatives = random.sample(neighbors_negatives[start+negative_numbers:])
    else:
        raise
    return negatives


def createVSB100Database(data, db_settings, logger):
    negative_numbers = db_settings['number_of_negatives']
    k = db_settings['number_of_neighbors']
    colors = data['colors']
    centers = data['centers']
    method = db_settings['method']
    level = int(db_settings['level'])
    output_path = db_settings['output_path']
    database_path = db_settings['database_path'].format(level=level)
    negative_selector_method = db_settings['negative_selector_method']
    negative_selector_param = db_settings['negative_selector_param']
    bag_size = db_settings['bag_size']
    feature_type = db_settings['feature_type']
    action = db_settings['action_name']

    similarities_path = db_settings['similarities_path']
    if method == 0:
        try:
            assert type(feature_type) is str
        except:
            print 'Feature type 0 takes only one feature as input\nExiting...'
            sys.exit()
    feature_set = ['HOF', 'FCN']
    #feature_type = db_settings['feature_type']
    if type(feature_type) is str:
        features = data[getattr(FeatureType, feature_type).name]
        if method == 0:
            other_feature = feature_set[1 - feature_set.index(feature_type)]
            other_similarity = loadmat(similarities_path.format(action_name=action[0], feature_name=other_feature))['similarities']
    if (type(feature_type) is list) and type(feature_type[0]) is str:
        feature_type = map(lambda x: getattr(FeatureType, feature_type))
    if type(feature_type) is list:
    #feature_names = [x.name for x in db_settings['feature_type']]
        features = [data[x.name] for x in db_settings['feature_type']]
    #assert colors.shape[0] == features[0].shape[0], 'Feature dimensions mismatch %s != %s' % (colors.shape[0],  features[0].shape[0])
        features = np.concatenate(features, axis=1)
    #TODO: if voxel_labels is not there produce it
    #TODO: if pixel_labels is not there produce it
    #basic_similarities = data['FCN'].dot(data['FCN'].T)
    if method == 0:
        basic_similarities = other_similarity.astype('int')
    number_of_voxels = colors.shape[0]
    n = number_of_voxels * negative_numbers
    database_negative_indices = np.zeros((number_of_voxels, negative_numbers), dtype=np.int32)
    database_neighbor_indices = np.zeros((number_of_voxels, k), dtype=np.int32)
    kdtree = cKDTree(centers.tolist())
    if method == 0:
        print 'Using Other Feature similarities for computations'
    for i in xrange(number_of_voxels):
        neighbors_all = kdtree.query(centers[i], bag_size)[1][1:]
        weights = basic_similarities[i, neighbors_all]
        arg_weights = np.argsort(np.array(weights)).tolist()
        neighbors_all = weights[arg_weights][::-1][:bag_size]
        neighbors = neighbors_all[:k]
        negatives = neighbors_all[-negative_numbers:]
        #negatives = getNegatives(negative_selector_method, negative_selector_param, neighbors_all, negative_numbers, k)
        database_negative_indices[i][...] = negatives
        database_neighbor_indices[i][...] = neighbors
    savemat(output_path, {'database_negative_indices':database_negative_indices, 'database_neighbor_indices':database_neighbor_indices})

    database = h5py.File(database_path, 'w')
    database['target'] = np.tile(features, (negative_numbers, 1))

    # database.close()
    print 'target done'
    for nei in xrange(k):
        # database = h5py.File(database_path, 'r+')
        neighbor_features = features[database_neighbor_indices[:,nei]][...]
        database['neighbor{}'.format(nei)] = np.tile(neighbor_features, (negative_numbers, 1))
        print 'neighbor{} done'.format(nei)
        # database.close()

    # database = h5py.File(database_path, 'r+')
    neighbor_features = np.zeros((n, features.shape[1]))
    for neg in xrange(negative_numbers):
        s_idx = features.shape[0]*neg
        e_idx = features.shape[0]*(neg+1)
        neighbor_features[s_idx:e_idx] = features[database_negative_indices[:,neg]][...]

    database['negative'] = neighbor_features
    database['data_weights'] = np.ones((neighbor_features.shape[0], db_settings['inner_product_output']))
    print 'negative done'
    database.close()


def createVoxelLabelledlevelvideoData(db_settings, colors):
    #TODO:
    #
    # self.current_frame = 22
    print "[Segmentation::VoxelLabelledlevelvideoData]  self.current_frame = {}".format(self.current_frame)
    frame_num = db_settings['frame']
    action_name = db_settings['action_name'][0]
    level = db_settings['level']
    segmented_path = db_settings['segmented_path'].format(action_name=action_name, level=level)+'{0:05d}.ppm'
    labelledlevelvideo_path = db_settings['voxellabelledlevelvideo_path'].format(level=level)
    orig_img = Image.open(segmented_path.format(frame_num))
    width, height = orig_img.size
    mapped = np.zeros(( height, width, frame_num-1))
    colors_to_id = {}
        # 1-based
    for i, color in enumerate(colors.tolist()):
        colors_to_id[tuple(color)] = i+1
    print "%%%%%%%% frames:", frame_num
    for f in xrange(frame_num-1):
        img = Image.open(segmented_path.format(f+1))
        for h in xrange(height):
            for w in xrange(width):
                mapped[h][w][f]= colors_to_id[img.getpixel((w,h))]
    from scipy.io import savemat
    labelledlevelvideo = mapped
    savemat(labelledlevelvideo_path, {'labelledlevelvideo':labelledlevelvideo, 'total_number_of_supervoxels':len(colors_to_id)})


# def labelledlevelvideo_generator(conf):
    # segmented_path = conf.getPath('segmented_path')
    # frame_number = conf.db_settings['frame']
    # print '[VSBDatabaseGenerator::labelledlevelvideo_generator] frame:', frame_number
    # out_path = conf.getPath('pixellabelledlevelvideo_path')
    # # assert out_path1 == out_path2
    # import glob
    # img = Image.open(glob.glob(segmented_path+"*.ppm")[0])
    # size = img.size
    # sups_nums = np.zeros((1,frame_number))
    # mat = np.zeros((size[1]/2, size[0]/2, frame_number))
    # print '[ExperimentSetup::labelledlevelvideo_generator] creating labelledlevelvideo.mat'
    # for i,img_path in enumerate(glob.glob(segmented_path+"*.ppm")):
        # if i == frame_number:
            # break
        # print 'image', i
        # img = Image.open(img_path)
        # width, height = img.size
        # colors = {}
        # counter = 1
        # for w in xrange(0,width,2):
            # for h in xrange(0,height,2):
                # pix = img.getpixel((w,h))
                # if pix not in colors:
                    # colors[pix] = counter
                    # counter += 1
                # mat[h/2][w/2][i] = colors[pix]
        # sups_nums[0,i] = counter-1
    # from scipy.io import savemat
    # savemat(out_path, {'labelledlevelvideo':mat, 'numberofsuperpixelsperframe':sups_nums})


