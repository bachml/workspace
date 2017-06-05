import sys
sys.path.insert(0, '/home/zeng/caffe_wyd/python')
import numpy as np
import caffe


def readDeepNet(trainNet_path, caffemodel_path, proj_root):
    caffe.set_mode_gpu()
    net = caffe.Net(trainNet_path,
                    caffemodel_path,
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    #transformer.set_mean('data', np.ones((1))*127.5)
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_input_scale('data', 0.0078125)
    transformer.set_input_scale('data', 0.003921)
    #transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)


def extract_feature(net, transformer, filelist, dimension, imageSize, data_folder):
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
        net.blobs['data'].reshape(1, 1, 128, 128)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(imagePath, color=False),(128, 128)))
        out = net.forward()
        feat = net.blobs['eltwise_fc1'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model

        nan = np.vstack((nan, feat))
        #_label = np.vstack((_label, thisLabel))
    return (nan, _label)


if __name__ == "__main__":



    data_folder = "../../../data4my_dirtywork/id_gray_128/"
    save_folder = "./"
    meanFile_path = "model/image_mean/rgb_112x96_mean.npy"

    model_name = sys.argv[1]
    imagePath = sys.argv[2]

    trainNet_path = "../../model/" + model_name + "/deploy.prototxt"
    caffemodel_path = "../../model/" + model_name + "/" + model_name + ".caffemodel"



    (net, transformer) = readDeepNet(trainNet_path, caffemodel_path, meanFile_path)
    
    net.blobs['data'].reshape(1, 1, 128, 128)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(imagePath, color=False),(128, 128)))
    out = net.forward()
    feat = net.blobs['eltwise_fc1'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model
    np.save('id_photo.npy', feat)

