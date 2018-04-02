import sys
sys.path.insert(0, '/home/zeng/lconv-sfm-caffe/python')
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








def net_config(data_name, deployNet_path, caffemodel_path, isColor):
    caffe.set_mode_gpu()
    net = caffe.Net(deployNet_path,caffemodel_path,caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({data_name: net.blobs[data_name].data.shape})
    transformer.set_transpose(data_name, (2, 0, 1))
    #transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    if isColor == 1:
        transformer.set_mean(data_name, np.ones((3))*127.5)
    if isColor == 0:
        transformer.set_mean(data_name, np.ones((1))*127.5)

    transformer.set_raw_scale(data_name, 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_input_scale(data_name, 0.0078125)
    if isColor == 1:
        transformer.set_channel_swap(data_name, (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)




def util_countLine(path):
    numLine = 0
    reader = open(path)
    while 1:
        line = reader.readline()
        if not line:
            break
        numLine = numLine + 1
    return numLine




def task_face_feature_extraction(filelist, taskPath, net, transformer, dim, imageSize_h, imageSize_w,  blob_name, isColor, transformer_le_data, transformer_n_data, transformer_lm_data):
    file = open(filelist)
    if isColor == 1:
	channels = 3
    else:
	channels = 1
    num = util_countLine(filelist)
    feature_batch = np.zeros((num, dim))
    i = 0

    path_task = '/home/zeng/data4my_dirtywork/id_patch_'

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
	net.blobs['le_data'].reshape(1, channels, imageSize_h/2, imageSize_w/2)
	net.blobs['n_data'].reshape(1, channels, imageSize_h/2, imageSize_w/2)
	net.blobs['lm_data'].reshape(1, channels, imageSize_h/2, imageSize_w/2)
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(taskPath+imagePath, color=isColor),(imageSize_h, imageSize_w)))
	net.blobs['le_data'].data[...] = transformer_le_data.preprocess('le_data', caffe.io.resize_image(caffe.io.load_image(path_task+'le/'+imagePath, color=isColor),(imageSize_h/2, imageSize_w/2)))
	net.blobs['n_data'].data[...] = transformer_n_data.preprocess('n_data', caffe.io.resize_image(caffe.io.load_image(path_task+'n/'+imagePath, color=isColor),(imageSize_h/2, imageSize_w/2)))
	net.blobs['lm_data'].data[...] = transformer_lm_data.preprocess('lm_data', caffe.io.resize_image(caffe.io.load_image(path_task+'lm/'+imagePath, color=isColor),(imageSize_h/2, imageSize_w/2)))
	net.forward()
	feature = net.blobs[blob_name].data[0]
	feature_batch[i-1,:] = feature

    return feature_batch


if __name__ == "__main__":

    args = parse_args()
    test_dataPath = "/home/zeng/data4my_dirtywork/id_rgb_256/"
    #test_dataPath = "/home/zeng/data4my_dirtywork/id_rgb_256/"

    deployNet_path =  args.task_path + '/deploy.prototxt'
    caffemodel_path = args.task_path + '/deploy.caffemodel'

    (net, transformer) = net_config('data', deployNet_path, caffemodel_path, args.isColor)
    (net, transformer_le_data) = net_config('le_data', deployNet_path, caffemodel_path, args.isColor)
    (net, transformer_n_data) = net_config('n_data', deployNet_path, caffemodel_path, args.isColor)
    (net, transformer_lm_data) = net_config('lm_data', deployNet_path, caffemodel_path, args.isColor)

    intra_feature = task_face_feature_extraction(test_dataPath+'intra.txt', test_dataPath, net, transformer, args.dim, args.imageSize_h, args.imageSize_w, args.blobname, args.isColor, transformer_le_data, transformer_n_data, transformer_lm_data)
    np.save('../metric_results_/'+args.discript+'_intra', intra_feature)
    extra_feature = task_face_feature_extraction(test_dataPath+'extra.txt', test_dataPath, net, transformer, args.dim, args.imageSize_h, args.imageSize_w, args.blobname, args.isColor, transformer_le_data, transformer_n_data, transformer_lm_data)
    np.save('../metric_results_/'+args.discript+'_extra', extra_feature)
