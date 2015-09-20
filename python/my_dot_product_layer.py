#In the name of God
import caffe
import numpy as np


class MyHingLossDotProductLayer(caffe.Layer):
    """
    Compute the hing loss of the dot product of the 2 input layers
    """

    def setup(self, bottom, top):
        # check input pair
#	top[0].reshape(4, 1)
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute dot product.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        
	# loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #self.diff[...] = bottom[0].data - bottom[1].data
	#print "data0:", bottom[0].data
	#print "data1:", bottom[1].data
#	print dir(bottom[0].size())
	#print "dot:", bottom[0].data * bottom[1].data
	
	hinge = np.vectorize(lambda x: max(0, x), otypes=[bottom[0].data.dtype])
	self.res = hinge(np.ones(bottom[0].data.shape[0]) - np.sum(bottom[0].data * bottom[1].data, axis=1))
        top[0].data[...] = np.sum(self.res)

    def backward(self, top, propagate_down, bottom):
	print np.sign(self.res).shape
	print np.sum(bottom[0].data, axis=1).shape
        for i in range(2):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = -1 * np.transpose(np.tile(np.sign(self.res), (2,1))) * bottom[1-i].data

	print "diff:", bottom[0].diff
	print "diff:", bottom[1].diff
