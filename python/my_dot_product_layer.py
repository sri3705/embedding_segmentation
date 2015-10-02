#In the name of God
import caffe
import numpy as np


class MyHingLossDotProductLayer(caffe.Layer):
    """
    Compute the hing loss of the dot product of the 2 input layers
    """
	# max(0, 1 - a.b)

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
	#TODO normalization
	#TODO 2 batch size check beshe.
	#print "shape bottom", bottom[0].data.shape
	#print "shape bottom diff", bottom[0].diff.shape
	#print "bottom diff [0]", bottom[0].diff[0]
	#print "bottom data [0]", bottom[0].data
	#print "bottom diff [1]", bottom[1].diff[0]
	#print "bottom data [1]", bottom[1].data
	hinge = np.vectorize(lambda x: max(0, x), otypes=[bottom[0].data.dtype])
	self.res = hinge(np.ones(bottom[0].data.shape[0]) - np.sum(bottom[0].data * bottom[1].data, axis=1))
	#print "self.res", self.res
        top[0].data[...] = np.sum(self.res) / bottom[0].num
	#print "top[0].data", top[0].data

    def backward(self, top, propagate_down, bottom):
	#print "in backward"
	hing_res = np.sign(self.res)
	#print "hing_res", hing_res
	#print np.sum(bottom[0].data, axis=1).shape
	#print "bottom[0].diff.shape: ", bottom[0].diff.shape
	#print "res: ", np.transpose(np.tile(np.sign(self.res), (2,1))).shape
        for i in range(2):
            	if not propagate_down[i]:
	#		print "not propagate_down[{0}]".format(i)
                	continue
            #bottom[i].diff[...] = -1 * np.transpose(np.tile(np.sign(self.res), (2,1))) * bottom[1-i].data
		#print bottom
		for d in range(bottom[0].data.shape[0]):
			bottom[i].diff[d][...] = -1 * hing_res[d] * bottom[1-i].data[d][...]

		bottom[i].diff[...] /= bottom[i].num
	
	#print "bottom diff[0] after backward", bottom[0].diff[0]
	#print "bottom diff[1] after backward", bottom[1].diff[0]
	#print "bottom data[0] after backward", bottom[0].data
	#print "bottom data[1] after backward", bottom[1].data


	#print "bottom.data", bottom[0].data
	#print "bottom.shape", bottom[0].data.shape
	#print "diff.shape:", bottom[0].diff.shape
	#print "diff:", bottom[0].diff
