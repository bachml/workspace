import sys
sys.path.insert(0, '/home/zeng/dp-caffe/python')
import numpy as np
import caffe
import argparse



def parse_args():


    parser = argparse.ArgumentParser(description='Extract Feature')

    parser.add_argument('--c', dest='isColor', default=1, type=int,
                        help='color img = 1, gray image = 0')

    parser.add_argument('--d', dest='dim', default=256, type=int,
			help='feature dimensions')

    parser.add_argument('--h', dest='imageSize_h', default=256, type=int,
                        help='image size ')

    parser.add_argument('--w', dest='imageSize_w', default=256, type=int,
                        help='image size ')

    parser.add_argument('--n', dest='blobname', default='prob', type=str,
			help='name of output blob')

    parser.add_argument('--t', dest='task_path', default='../model/Wen_ECCV/', type=str,
			help='path of task folder')

    #parser.add_argument('--m', dest='model_path', default='../buffer_/Wen_ECCV.caffemodel', type=str,
    #			help='path of trained model')

    parser.add_argument('--td', dest='discript', default='None', type=str,
			help='task discription')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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


def extract_feature(net, transformer, filelist, dimension, imageSize, test_dataPath):
    file = open(filelist)

    nan = np.empty(shape=[0, dimension])
    _label = np.empty(shape=[0, 1])
    looptimes = 0
    i=0
    while 1:
        i=i+1
        line = file.readline()
        if not line:
            break
        pass

        if (i%100==0):
            print(str(i))
        #spaceIndex = line.find(" ")
        #imagePath = line[0:spaceIndex]
        imagePath = line.strip('\n')
        #thisLabel = int(line[spaceIndex + 1:len(line)])
        net.blobs['data'].reshape(1, 3, 256, 256)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(test_dataPath + imagePath, color=True),(256,256)))
        out = net.forward()
        feat = net.blobs['eltwise_fc1'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model

        nan = np.vstack((nan, feat))
        #_label = np.vstack((_label, thisLabel))
    return (nan, _label)


def task_face_feature_extraction(filelist, taskPath, net, transformer, dim, imageSize_h, imageSize_w,  blob_name, isColor):
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
	net.blobs['data'].reshape(1, channels, imageSize_h, imageSize_w)
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(taskPath+imagePath, color=isColor),(imageSize_h, imageSize_w)))
	net.forward()
	feature = net.blobs[blob_name].data[0]
	feature_batch[i-1,:] = feature

    return feature_batch


def util_countLine(path):
    numLine = 0
    reader = open(path)
    while 1:
        line = reader.readline()
        if not line:
            break
        numLine = numLine + 1
    return numLine


if __name__ == "__main__":

    args = parse_args()
    test_dataPath = "/home/zeng/data4my_dirtywork/id_rgb_224x192/"

    deployNet_path =  args.task_path + '/deploy.prototxt'
    caffemodel_path = args.task_path + '/deploy.caffemodel'

    (net, transformer) = net_config(deployNet_path, caffemodel_path, args.isColor)

    intra_feature = task_face_feature_extraction(test_dataPath+'intra.txt', test_dataPath, net, transformer, args.dim, args.imageSize_h, args.imageSize_w, args.blobname, args.isColor)
    np.save('../metric_results_/'+args.discript+'_intra', intra_feature)
    extra_feature = task_face_feature_extraction(test_dataPath+'extra.txt', test_dataPath, net, transformer, args.dim, args.imageSize_h, args.imageSize_w, args.blobname, args.isColor)
    np.save('../metric_results_/'+args.discript+'_extra', extra_feature)
