import sys
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
    #transformer.set_transpose('data', (0,1,2))
    transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_input_scale('data', 0.0078125)
    #transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)


def extract_feature(net, transformer, filelist, dimension, imageSize):
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

        if i%10==0:
            print(str(i))
        spaceIndex = line.find(" ")
        imagePath = line[0:spaceIndex]
        thisLabel = int(line[spaceIndex + 1:len(line)])
        net.blobs['data'].reshape(1, 3, 112, 96)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('lfw/' + imagePath, color=True))
        out = net.forward()
        feat = net.blobs['fc5'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model

        nan = np.vstack((nan, feat))
        _label = np.vstack((_label, thisLabel))
    return (nan, _label)


if __name__ == "__main__":


    test_arg = "lfw"
    meanFile_path = "model/image_mean/rgb_112x96_mean.npy"

    testname = sys.argv[1]

    trainNet_path = "../" + testname + "/deploy.prototxt"
    caffemodel_path = "../" + testname + "/" + testname + ".caffemodel"



    (net, transformer) = readDeepNet(trainNet_path, caffemodel_path, meanFile_path)


    filelist = test_arg + "/intra.txt"
    (feature, label) = extract_feature(net, transformer, filelist, 512, 50)
    outputFileName = test_arg + "/feature/" + testname + "_intra.npy"
    print(feature.shape)
    np.save(outputFileName, feature)


    filelist = test_arg + "/extra.txt"
    (feature, label) = extract_feature(net, transformer, filelist, 512, 50)
    outputFileName = test_arg + "/feature/" + testname + "_extra.npy"
    print(feature.shape)
    np.save(outputFileName, feature)

    '''

    filelist = test_arg + "_test/cacd_withNoLFWebface.txt"
    (feature, label) = extract_feature(net, transformer, filelist, 160, 50)
    outputFileName = test_arg + "_test/cacd112x96_feature.npy"
    print(feature.shape)
    np.save(outputFileName, feature)

    np.save(test_arg + "_test/cacd112x96_label.npy", label)
    '''
