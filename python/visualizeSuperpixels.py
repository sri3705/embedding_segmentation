#INTNOG
from PIL import Image
from scipy.io import loadmat,savemat
import numpy as np
from scipy.spatial import cKDTree
from sys import argv

def revised_similarities(experiment_num):
    video_info_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/vw_commercial_vidinfo.mat'
    video_info = loadmat(video_info_path)
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe'][0]
    labelsatframe = video_info['labelsatframe']
    framebelong = video_info['framebelong']
    labelledlevelvideo = video_info['labelledlevelvideo']
    mapped = video_info['mapped']
    mat = loadmat('/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/centers.mat')
    centers = mat['centers']
    kdtrees = []
    total_num_frames = 19
    for f in xrange(total_num_frames):
        kdtree = cKDTree(np.array(centers[f][:numberofsuperpixelsperframe[f]])) 
        kdtrees.append(kdtree)
    neighbors = 16
    frames_limit = 6
    similarities = loadmat('/cs/vml2/mkhodaba/cvpr16/expriments/'+str(experiment_num)+'/similarities.mat')
    similarities = similarities['similarities']
    # similarities = similarities * -1;
    n = similarities.shape[0] 
    print n
    print centers.shape
    print framebelong.shape
    print labelsatframe.shape
    new_similarities = np.zeros((n,n)) 
    for i in xrange(n):
        frame = framebelong[0, i]-1
        lbl = labelsatframe[i, 0]-1
        centr = centers[frame][lbl][...]
        if i == 0:
            print 'frame', frame
            print 'label', lbl
            print 'centr', centr
        for f in xrange(max(0, frame-frames_limit), min(frame+frames_limit, total_num_frames)):
            _, nearest_neighbors = kdtrees[f].query(centr, neighbors)
            if i == 0:
                print f, nearest_neighbors
            for nei in nearest_neighbors:
                idx = mapped[f][nei]
                new_similarities[i][idx] = similarities[i][idx]
    savemat('/cs/vml2/mkhodaba/cvpr16/expriments/'+str(experiment_num)+'/new_similarities.mat', {'similarities':new_similarities})

def visualAdjacents(frame,lbl):
    video_info_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/vw_commercial_vidinfo.mat'
    video_info = loadmat(video_info_path)
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe'][0]
    labelsatframe = video_info['labelsatframe']
    framebelong = video_info['framebelong']
    labelledlevelvideo = video_info['labelledlevelvideo']
    mapped = video_info['mapped']
    centers = mat['centers']
    kdtrees = []
    total_num_frames = 19
    for f in xrange(total_num_frames):
        kdtree = cKDTree(np.array(centers[f][:numberofsuperpixelsperframe[f]])) 
        kdtrees.append(kdtree)
    neighbors = 6 
    frames_limit = 20 
    video_size = labelledlevelvideo.shape#320x640xtotal_num_frames
    # video = np.zeros(video_size) 
    # frame = framebelong[0, idx]-1
    # lbl = labelsatframe[idx, 0]-1
    centr = centers[frame][lbl]
    for f in xrange(total_num_frames):
        img = Image.new('RGB', (video_size[1], video_size[0]))
        nearest_neighbors = set([])
        if abs(f-frame)<=frames_limit:
            _, nearest_neighbors = kdtrees[f].query(centr, neighbors)
            print nearest_neighbors
            nearest_neighbors = set(nearest_neighbors)
        for h in xrange(video_size[0]):
            for w in xrange(video_size[1]):
                if labelledlevelvideo[h][w][f] in nearest_neighbors:
                    img.putpixel((w,h), (255,20,20))
                if f == frame and lbl==labelledlevelvideo[h][w][f]-1:
                    img.putpixel((w,h), (20,255,20))
        img.save('/cs/vml2/mkhodaba/cvpr16/test/{0}.jpg'.format(f))

def visualizeJustOne(frame, lbl):
    video_info_path = '/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat'
    video_info = loadmat(video_info_path)
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe'][0]
    labelledlevelvideo = video_info['labelledlevelvideo']
    mat = loadmat('/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/centers.mat')
    centers = mat['centers']
    frames_limit = 20 
    video_size = labelledlevelvideo.shape#320x640xtotal_num_frames
    # video = np.zeros(video_size) 
    # frame = framebelong[0, idx]-1
    # lbl = labelsatframe[idx, 0]-1
    centr = centers[frame][lbl]
    print 'Center is:', centr
    # ranged = set(range(100))
    ranged = [lbl]
    img = Image.new('RGB', (video_size[1], video_size[0]))
    for h in xrange(video_size[0]):
        for w in xrange(video_size[1]):
            # if lbl==labelledlevelvideo[h][w][frame]-1:
            if labelledlevelvideo[h][w][frame]-1 in ranged:
                img.putpixel((w,h), (20,255,20))
    img.save('/cs/vml2/mkhodaba/cvpr16/test/justone-{0}-{1}.jpg'.format(frame, lbl))



def visualizeNeighbors(frame,lbl):
    # video_info_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/vw_commercial_vidinfo.mat'
    video_info_path = '/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat'
    video_info = loadmat(video_info_path)
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe'][0]
    labelsatframe = video_info['labelsatframe']
    framebelong = video_info['framebelong']
    labelledlevelvideo = video_info['labelledlevelvideo']
    mapped = video_info['mapped']
    video_size = labelledlevelvideo.shape#320x640xtotal_num_frames
    mat = loadmat('/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/centers.mat')
    centers = mat['centers']
    img = Image.new('RGB', (video_size[1], video_size[0]))
    kdtrees = []
    total_num_frames = 19
    for f in xrange(total_num_frames):
        kdtree = cKDTree(np.array(centers[f][:numberofsuperpixelsperframe[f]])) 
        kdtrees.append(kdtree)
    neighbors = 6 
    frames_limit = 20 
    # video = np.zeros(video_size) 
    # frame = framebelong[0, idx]-1
    # lbl = labelsatframe[idx, 0]-1
    centr = centers[frame][lbl]
    for f in xrange(total_num_frames):
        img = Image.new('RGB', (video_size[1], video_size[0]))
        nearest_neighbors = set([])
        if abs(f-frame)<=frames_limit:
            _, nearest_neighbors = kdtrees[f].query(centr, neighbors)
            print nearest_neighbors
            nearest_neighbors = set(nearest_neighbors)
        for h in xrange(video_size[0]):
            for w in xrange(video_size[1]):
                if labelledlevelvideo[h][w][f]-1 in nearest_neighbors:
                    img.putpixel((w,h), (255,20,20))
                if f == frame and lbl==labelledlevelvideo[h][w][f]-1:
                    img.putpixel((w,h), (20,255,20))
        for cent in centers[f]:
            h = int(cent[0])
            w = int(cent[1])
            img.putpixel((w,h), (20,20,255))
            img.putpixel((w+1,h), (20,20,255))
            img.putpixel((w,h+1), (20,20,255))
            img.putpixel((w+1,h+1), (20,20,255))
    # img.save('/cs/vml2/mkhodaba/cvpr16/test/centers-{0}.jpg'.format(0))
        img.save('/cs/vml2/mkhodaba/cvpr16/test/{0}.jpg'.format(f))
    
def buildTrees():
    # video_info_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/vw_commercial_vidinfo.mat'
    video_info_path = '/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat'
    video_info = loadmat(video_info_path)
    labels = video_info['labelledlevelvideo']
    frames_num = 19
    numberofsuperpixelsperframe = video_info['numberofsuperpixelsperframe'][0]
    superpixels_num = max(numberofsuperpixelsperframe)
    print superpixels_num
    # last two are H and W. Center = (h,w)
    centers = np.zeros((frames_num, superpixels_num, 2)) #[[[0.0,0.0] for i in xrange(superpixels_num)] for j in xrange(frames_num)] #frames_num x superpixels_num x 2
    pixels_count = [[0 for i in xrange(superpixels_num)] for j in xrange(frames_num)] #frames_num x superpixels_num
    height = len(labels)
    width = len(labels[0])
    for f in xrange(frames_num):
        print 'frame', f
        for h in xrange(height):
            for w in xrange(width):
				idx = labels[h][w][f]-1
				centers[f][idx][0] += h		
				centers[f][idx][1] += w
				pixels_count[f][idx] += 1
        for i in xrange(numberofsuperpixelsperframe[f]):
            centers[f][i][0] /= pixels_count[f][i]
            centers[f][i][1] /= pixels_count[f][i]
            # kdtree = cKDTree(np.array(centers[f][:numberofsuperpixelsperframe[f]]))
            # kdtrees.append(kdtree)	
    savemat('/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/visualization/centers.mat', {'centers': centers}) 
    
if __name__=='__main__':
    if len(argv) == 4:
        if argv[1] == '-v':
            f = int(argv[2])
            i = int(argv[3])
            visualizeNeighbors(f, i)
        if argv[1] == '-j':
            f = int(argv[2])
            i = int(argv[3])
            visualizeJustOne(f, i) 
    if len(argv) == 3:
       if argv[1] == '-s':
            exp_num = int(argv[2])
            revised_similarities(exp_num)
    if len(argv) == 2 and argv[1] == '-c':
        buildTrees()
        








