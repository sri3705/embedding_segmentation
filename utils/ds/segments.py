#import cv2
#start = time.clock()
#img = cv2.imread('00001.ppm')
#colors = {}
#for i in range(img.shape[0]):
#	for j in range(img.shape[1]):
#		pixel = (img[0][1][0], img[0][1][1], img[0][1][2])
#		if pixel not in colors:
#			colors[pixel] = 1
#		else:
#			colors[pixel] += 1
#print time.clock()-start

from PIL import Image
import time


path = '/cs/vml3/mkhodaba/cvpr16/libsvx.v3.0/example/output_gbh/11/'
#path = './'
start = time.clock()
img = Image.open(path+'00001.ppm')
print path+'00001.ppm'
cs = img.convert('RGB').getcolors()
supervoxels = {}	
opens = {}
sums = {}
counts = {}
closed_colors = {}
for i, c in cs:
	opens[c] = set()
	sums[c] = [0,0,0]
	counts[c] = 0
for f in range(1,31):
	img = Image.open(path+'{0:05d}.ppm'.format(f))
	print '{0:05d}.ppm'.format(f)
	colors = img.convert('RGB').getcolors()
	for i,c in colors:
		if c not in opens.keys():
			opens[c] = set()
			sums[c] = [0,0,0]
			counts[c] = 0
		counts[c] += i	
	existing = {}
	for c in opens.keys():
		existing[c] = False
	print "colors: ", len(colors)
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			color = img.getpixel((i,j))
			existing[color] = True
			opens[color].add((i,j,f))
			sums[color] = [sums[color][0]+i, sums[color][1]+j, sums[color][2]+f]
	for c in existing.keys():
		if existing[c] == False:			
			center = sums.pop(c)
			num = counts.pop(c)
			center = [center[0]/num, center[1]/num, center[2]/num]
			supervoxels[(c,f)] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to the final set
			closed_colors[c] = True
	print "opens: ", len(opens)
	print "supervoxels: ", len(supervoxels)
for c in opens.keys():
	center = sums.pop(c)
	num = counts.pop(c)
	center = (center[0]/num, center[1]/num, center[2]/num)
	supervoxels[(c,f)] = [opens.pop(c), center] #remove the supervoxel from open ones and add it to 
print "opens: ", len(opens)
print "supervoxels: ", len(supervoxels)

print supervoxels[(c,30)]
