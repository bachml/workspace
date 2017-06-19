import sys
sys.path.insert(0, '/home/zeng/caffe_wyd/python')
import caffe
import numpy as np
#import yaml



'''
layer{
  name: "fe_layer"
  type: "Python"
  bottom: "connect1"
  top: "Leaky25"
  python_param{
    module: "feature_extract_layer"
    layer: "FeatExLayer"#层的名称为LeakyLayer
  }
}
'''

def task_face_feature_extraction(filelist, taskPath, net, transformer, dim, imageSize,  blob_name, isColor):
    file = open(filelist)
    if isColor == 1:
	channels = 3
    else:
	channels = 1
    num = util_countLine(filelist)
    feature_batch = np.zeros((num, dim))
    i = 0
    while True:
	i = i + 1
	line = file.readline()
	if not line:
	    break
	pass
	
	if (i%100==0):
	    print('iteration ' + str(i))
	imagePath = line.strip('\n')
	net.blobs['data'].reshape(1, channels, imageSize, imageSize)
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(taskPath+imagePath, color=isColor),(imageSize, imageSize)))
	net.forward()
	feature = net.blobs[blob_name].data[0]
	feature_batch[i-1,:] = feature

    return feature_batch


def net_config(deployNet_path, caffemodel_path, isColor):
    caffe.set_mode_gpu()
    net = caffe.Net(deployNet_path,caffemodel_path,caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    if isColor == 1:
        transformer.set_mean('data', np.ones((3))*127.5)
    if isColor == 0:
        transformer.set_mean('data', np.ones((1))*127.5)

    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_input_scale('data', 0.0078125)
    if isColor == 1:
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)

class FeatExLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1 :
            raise Exception("Need one inputs to compute distance.")
	deployNet_path = 'dasdf'
	caffemodel_path = 'qweqe'
    	net = caffe.Net(deployNet_path,caffemodel_path,caffe.TEST)
		

    def reshape(self, bottom, top):
	batch_size = bottom[0].data.shape[0] # B x C x H x W
	feature_dimension = 512
        top[0].reshape(batch_size, feature_dimension)

    def forward(self, bottom, top):
        data_top = top[0].data
	bottom_shape = bottom[0].data.shape
	net_blobs['data'].reshape(bottom_shape[0], bottom_shape[1], bottom_shape[2], bottom_shape[3])
	net.blobs['data'].data = bottom.data
	net.forward()
	data_top = net.blobs['fc5'].data
	'''	
        count = bottom[0].count
        data_bot = bottom[0].data
        data_top = top[0].data
        shape = data_bot.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                if data_bot[i][j] < 0:
                    data_top[i][j] = data_bot[i][j]*0.1
                else:
                    data_top[i][j] = data_bot[i][j]
	'''

    def backward(self, top, propagate_down, bottom):
        pass

'''
class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

'''
