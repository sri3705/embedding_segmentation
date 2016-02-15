from PIL import Image
from scipy.io import savemat
import numpy as np
import glob

level = '06'
orig_path = '/cs/vml3/mkhodaba/cvpr16/dataset/vw_commercial/b1/seg/{}/'.format(level)
out_path = '/cs/vml2/mkhodaba/cvpr16/VSB100/VideoProcessingTemp/vw_commercial/labelledlevelvideo.mat'

img = Image.open(glob.glob(orig_path+"*.ppm")[0])
size = img.size
sups_nums = np.zeros((1,21))
mat = np.zeros((size[1]/2, size[0]/2, 21))
for i,img_path in enumerate(glob.glob(orig_path+"*.ppm")):
    if i == 21:
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

savemat(out_path, {'labelledlevelvideo':mat, 'numberofsuperpixelsperframe':sups_nums}) 

