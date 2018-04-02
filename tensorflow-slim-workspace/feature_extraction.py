import tensorflow as tf
import numpy as np
from nets import inception_resnet_v1
from nets.inception_resnet_v1 import *
from nets.mobilenet_v2 import *
from nets.wynet import *
from preprocessing import face_preprocessing
from tensorflow.contrib import slim
import urllib2
import math
from scipy import misc
from preprocessing import inception_preprocessing
from nets.nets_factory import *

import argparse
import sys

#IMAGE_SIZE = 256
#CHANNELS = 3
#FEATURE_DIM = 512   #1792
batch_size = 60

#filelist = '/home/zeng/workspace/tensorflow-workspace/extra_id256.txt'

#image_paths = [l.strip('\n') for l in
#    tf.gfile.FastGFile(filelist, 'r').readlines()]

tf.reset_default_graph()



def parse_args():


    parser = argparse.ArgumentParser(description='Extract Feature')


    parser.add_argument('--f', dest='filelist', default=None, type=str) #filelist = '/home/zeng/workspace/tensorflow-workspace/extra_id256.txt'
    parser.add_argument('--n', dest='network_name', default=None, type=str)
    parser.add_argument('--sn', dest='network_scope_name', default=None, type=str)
    parser.add_argument('--s', dest='image_size', default=256, type=int)
    parser.add_argument('--c', dest='channel', default=3, type=int)
    parser.add_argument('--d', dest='feature_dim', default=None, type=int)
    parser.add_argument('--nc', dest='num_classes', default=None, type=int)

    parser.add_argument('--path', dest='checkpoint_path', default=None, type=str)
    parser.add_argument('--e', dest='embedding_name', default=None, type=str)
    parser.add_argument('--o', dest='output_name', default=None, type=str)  ### example: wynet_intra

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def go_tf(image_paths, image_size, channel):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, channel))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        images[i,:,:,:] = img
    return images


def preproc_image_batch(batch_size, image_size, num_threads=1):



    filename_queue = tf.FIFOQueue(100000, [tf.string], shapes=[[]], name="filename_queue")
    reader = tf.WholeFileReader()
    image_filename, image_raw = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    # Image preprocessing
    #preproc_func_name = self._network_name if self._preproc_func_name is None else self._preproc_func_name
    #image_preproc_fn = preprocessing_factory.get_preprocessing(preproc_func_name, is_training=False)
    #image_preproc = image_preproc_fn(image, self.image_size, self.image_size)
    image_preproc = face_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    #image_preproc  = tf.expand_dims(image_preproc, 0)
    # Read a batch of preprocessing images from queue
    image_batch = tf.train.batch(
        [image_preproc, image_filename], batch_size, num_threads=num_threads,
        allow_smaller_final_batch=True)
    return image_batch

if __name__ == '__main__':
    args = parse_args()

    image_paths = [l.strip('\n') for l in
                tf.gfile.FastGFile(args.filelist, 'r').readlines()]

    with tf.Graph().as_default():
        batch_from_queue, _batch_filenames = preproc_image_batch(batch_size, args.image_size, num_threads=1)
        image_batch = tf.placeholder_with_default(
            batch_from_queue, shape=[None, args.image_size, args.image_size, args.channel])

        arg_scope = arg_scopes_map[args.network_name]()
        func = networks_map[args.network_name]
        #@functools.wraps(func)
        with slim.arg_scope(arg_scope):
           logits, end_points = func(image_batch, num_classes=args.num_classes)
        #with slim.arg_scope(mobilenet_v2_arg_scope()):
        #    logits, end_points = mobilenet_v2_075(image_batch, num_classes=args.num_classes)

        fetches = {}
        fetches[args.embedding_name] = end_points[args.embedding_name]



        with tf.Session() as sess:
            init_fn = slim.assign_from_checkpoint_fn(
                args.checkpoint_path,slim.get_model_variables(args.network_scope_name))
            init_fn(sess)

            #image_paths = [l.strip('\n') for l in
                #tf.gfile.FastGFile(filelist, 'r').readlines()]


            nrof_images = len(image_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            feature_batch = np.zeros((nrof_images, args.feature_dim))
            for i in range(nrof_batches):
                print(str(i) + '/' + str(nrof_batches) + '\n')
                start_index = i * batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = image_paths[start_index:end_index]
                input_batch = go_tf(paths_batch, args.image_size, args.channel)
                feed_dict = {image_batch: input_batch}
                outputs = sess.run(fetches, feed_dict=feed_dict)
                feature_batch[start_index:end_index,:] = np.squeeze( outputs[args.embedding_name] )

            np.save('/home/zeng/workspace/metric_results_/'+args.output_name, feature_batch)
            print('done')



###
