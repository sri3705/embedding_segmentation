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
	print "data0:", bottom[0].data
	print "data1:", bottom[1].data
#	print dir(bottom[0].size())
        top[0].data[...] = 1 - np.sum(np.dot(bottom[0].data, np.transpose(bottom[1].data)))

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = -1 * np.sum(bottom[1-i].data, axis=0)
	print "diff:", bottom[0].diff
	print "diff:", bottom[1].diff
