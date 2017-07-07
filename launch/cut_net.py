import sys
sys.path.insert(0, '/home/zeng/dp-caffe/python')
import caffe
proto = sys.argv[1]
model = sys.argv[2]
net_new = caffe.Net(proto,model,caffe.TEST)
net_new.save('cutted_' + model)
