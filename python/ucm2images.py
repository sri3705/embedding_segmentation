from PIL import Image
import glob
import sys
name = sys.argv[1]
level = 8 
# orig_path = '/cs/vml3/mkhodaba/cvpr16/dataset/{}/b1/seg/{}/'.format(name,level)
orig_path = '/cs/vml2/mkhodaba/datasets/VSB100/segmented_frames/{0}/{1:02d}/'.format(name,level)
out_path = '/cs/vml2/mkhodaba/cvpr16/VSB100/VideoProcessingTemp/{0}/ucm2images/ucm2imagenew{1:03d}_ucm2.bmp'
# orig_path = '/cs/vml2/mkhodaba/cvpr16/VSB100/VideoProcessingTemp/vw_commercial/origimages/image{0:03d}.png'

moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
for i,img_path in enumerate(glob.glob(orig_path+"*.ppm")):
    print 'image', i
    img = Image.open(img_path)
    out_img = Image.new('L', img.size, 0)
    width, height = img.size
    for w in xrange(width):
        for h in xrange(height):
            pix = img.getpixel((w,h))
            # out_img.putpixel(
            # for dw,dh in moves:
            if (0<=1+w<width and pix != img.getpixel((1+w, h))) or (0<=1+h<height and pix != img.getpixel((w, 1+h))):
                out_img.putpixel((w,h), 170)
                    # break
    out_img.save(out_path.format(name, i))
                    

