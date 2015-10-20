import caffe
from numpy import zeros
import numpy as np
root = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/'

#caffe.set_decive(2)
caffe.set_mode_gpu()
#caffe.set_device(2)


model_prototxt_name = 'model.prototxt'
solver_prototxt_name = 'solver_vml_gpu2.prototxt'
niter = 500000
train_interval = 1000
test_interval = 10000
net = caffe.Net(root + model_prototxt_name, caffe.TRAIN)

#solver = caffe.SGDSolver(root+'solver.prototxt')
solver = caffe.SGDSolver(root+solver_prototxt_name)
# losses will also be stored in the log
#train_loss = zeros(niter)
train_loss = np.array([])
test_acc = zeros(int(np.ceil(niter / test_interval)))
#output = zeros((niter, 8, 10))
test_loss = 100
# the main solver loop
for it in range(niter):
	#print 'iter', it
	solver.step(1)  # SGD by Caffe
    	#print 'step done'
    # store the train loss
	#target_data = solver.net.blobs['target'].data
	
	#print 'target_data', target_data
	#print target_data.shape
	train_loss= np.append(train_loss, solver.net.blobs['loss'].data)
#	if it > 0 and train_loss[it-1] == 0:
#		train_loss[it-1] = train_loss[it]
	if it % train_interval == 0:
		print 'Iteration', it, '...'
		#print 'Loss:', train_loss[it]
		print 'Average Train Loss:', np.mean(train_loss), '-- Train Loss Std', np.std(train_loss)
		print 'Minimum Train Loss:', np.amin(train_loss)	
		print 'Test Loss so far:', test_loss
	#solver.test_nets[0].forward(start='conv1')
	if it % test_interval == 0:
		for test_it in range(100):
			solver.test_nets[0].forward()
		print 'Test Loss:', solver.test_nets[0].blobs['loss'].data
		test_loss = solver.test_nets[0].blobs['loss'].data

np.savetxt(root+'embedding.txt', solver.net.params['inner_product_target'][0].data)
#with open(root+'embedding.txt', 'w') as f:
#	f.write(str(solver.net.params['inner_product_target'][0].data))
#print solver.net.params['inner_product_negative'][0].data

#print solver.net.params['inner_product_target'][0].data


    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    #solver.test_nets[0].forward(start='conv1')
    #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    #if it % test_interval == 0:
    #   print 'Iteration', it, 'testing...'
    #    correct = 0
    #    for test_it in range(100):
    #        solver.test_nets[0].forward()
    #        correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
    #                       == solver.test_nets[0].blobs['label'].data)
    #    test_acc[it // test_interval] = correct / 1e4
