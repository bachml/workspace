import sys
sys.path.insert(0, '/home/zeng/caffe_wyd/python')
import numpy as np
import caffe


def readDeepNet(trainNet_path, caffemodel_path):
    caffe.set_mode_cpu()
    net = caffe.Net(trainNet_path,
                    caffemodel_path,
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    transformer.set_mean('data', np.ones((3))*127.5)
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_input_scale('data', 0.0078125)
    #transformer.set_input_scale('data', 0.003921)
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)




if __name__ == "__main__":



    task_name = sys.argv[1]
    imagePath = sys.argv[2]
    prob_name = sys.argv[3]

    workspace_path = '/home/zeng/workspace/'
    
    trainNet_path =   task_name + "/deploy.prototxt"
    caffemodel_path =  task_name + '/' + task_name + '.caffemodel'


    (net, transformer) = readDeepNet(trainNet_path, caffemodel_path)
    
    net.blobs['data'].reshape(1, 3, 128, 128)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(imagePath, color=True),(128, 128)))
    out = net.forward()
    feat = net.blobs[prob_name].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model
    print(feat)
