import numpy as np
import matplotlib.pyplot as plt

import caffe

root = '/cs/vml3/mkhodaba/cvpr16/code/'

caffe.set_mode_cpu()
net = caffe.Net(root + 'model.prototxt', caffe.TRAIN)

# TODO: Feed dummy data based on the html page. So something like:
#net.blobs['data'].reshape(1,6,1,2)
net.blobs['data'].data[...] = np.array([[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[5, 5]]],
					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[15, -5]]],
					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[7, 16]]],
					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[-5, 8]]],
					])

#net.blobs['data'].data[...] = np.array([[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[5, 5]]],])

items = [(k, v.data.shape) for k, v in net.blobs.items()]
print "Layers: "
for i in items:
	print i 

print 
net.params['embedding_function_context'][0].data[...] = np.array([[1,1],[2,4]], dtype=np.float32)
out = net.forward()
out1 = net.backward()
print net.blobs['target'].data
print net.blobs['context_sum'].data
print net.blobs['embedding_function_context'].data
#print net.params['embedding_function_context'].data
#print net.params['embedding_function_target'].data
#print net.params['embedding_function'].data
print out
