#In the name of God

from caffe import layers as L
from caffe import params as P
from caffe import NetSpec

class Network:
	def __init__(self, number_of_neighbors):
		self.number_of_neighbors = number_of_neighbors
		#self.netP = 
		self.net = NetSpec()
		self.shared_weight_counter = 0

	def getInnerProduct(self, input_name, output_name, ID):
		return L.InnerProduct(getattr(self.net, input_name),
						name=output_name,
						num_output=300, #TODO What should be the output of this layer?
						weight_filler=dict(type='xavier'),
						bias_filler=dict(type='xavier', value=0.2),
						#in_place=True,
						param=list([dict(name="embed_w{0}".format(ID), lr_mult=1, decay_mult=1), 
							    dict(name="embed_b{0}".format(ID), lr_mult=2, decay_mult=0)])
						)

	def createEmbeddingNetwork(self, dataset_path, batch_size):
		dataLayer = L.HDF5Data(name='dataLayer', 
						source=dataset_path, 
						batch_size=batch_size, 
						ntop=2+self.number_of_neighbors)# tops-> target, [neighbors], negative
		#data -> [target, neighbor1, neighbor2, ..., neighbork, negative]
		self.net.target = dataLayer[0]
		self.net.negative = dataLayer[-1]
		for l in range(1, self.number_of_neighbors+1):
			setattr(self.net, 'neighbor{0}'.format(l-1), dataLayer[l])		

		
		#First layer of inner product 
		self.net.inner_product_target = self.getInnerProduct('target', 'inner_product_target', 1)
		self.net.inner_product_negative = self.getInnerProduct('negative', 'inner_product_negative', 1)
		for i in range(0, self.number_of_neighbors):
			layer = self.getInnerProduct('neighbor{0}'.format(i), 'inner_product_neighbor{0}'.format(i), 1)
			setattr(self.net, 'inner_product_neighbor{0}'.format(i), layer)
		
		#ReLU
		self.net.relu_target = L.ReLU(self.net.inner_product_target, name='relu_target', in_place=True)
		self.net.relu_negative = L.ReLU(self.net.inner_product_negative, name='relu_negative', in_place=True)
		for i in range(0, self.number_of_neighbors):
			layer = L.ReLU(getattr(self.net, 'inner_product_neighbor{0}'.format(i)), 
					name='relu_neighbor{0}'.format(i),
					in_place=True)
			setattr(self.net, 'relu_neighbor{0}'.format(i), layer)
		
		#Second layer of inner product
		self.net.inner_product2_target = self.getInnerProduct('inner_product_target', 'inner_product2_target', 2)
		self.net.inner_product2_negative = self.getInnerProduct('inner_product_negative', 'inner_product2_negative', 2)
		for i in range(0, self.number_of_neighbors):
			layer = self.getInnerProduct('inner_product_neighbor{0}'.format(i), 
							'inner_product2_neighbor{0}'.format(i), 2)
			setattr(self.net, 'inner_product2_neighbor{0}'.format(i), layer)
			
		#Context
		context_sum_bottom = []
		for i in range(0, self.number_of_neighbors):
			context_sum_bottom.append(getattr(self.net, 'inner_product2_neighbor{0}'.format(i)))
		coeff = 1.0/self.number_of_neighbors		
		self.net.context_sum = L.Eltwise(*context_sum_bottom,
						name='context_sum',
						operation=P.Eltwise.SUM, # 1 -> SUM
						coeff=list([coeff for i in range(self.number_of_neighbors)]))
		
		#Target - Negative
		self.net.target_negative_diff = L.Eltwise(self.net.inner_product2_target, self.net.inner_product2_negative,
								name='target_negative_diff',
								operation=P.Eltwise.SUM, # SUM
								coeff=list([1,-1])) # target - negative
		
		#Loss layer
		self.net.loss = L.Python(self.net.context_sum, self.net.target_negative_diff,
						name='loss',
						module='my_dot_product_layer',
						layer='MyHingLossDotProductLayer')

	def saveNetwork(self, path):
		with open(path, 'w') as f:
			f.write("force_backward:true\n"+str(self.net.to_proto()))

def main():
	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/datalist.txt' #dataset1.h5'
	model_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/model.prototxt'
	network = Network(2)
	network.createEmbeddingNetwork(dataset_path, 2)
	network.saveNetwork(model_path)


if __name__ == '__main__':
	main()
