import numpy,scipy.misc, os, array
from PIL import Image
#from datasets import dataset_utils
import tensorflow as tf
import os
import os.path
import pandas as pd
import cv2
#train_file = 'trainLabels.csv'  #类别文件
#name='train'      #图片储存文件夹
#output_directory='./tfrecords'  #输出文件夹


resize_height=112
resize_width=96


shape = (resize_height, resize_width, 3)

def get_feature(output_filename, input_filelist):

    filename = output_filename '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_jpeg = tf.image.encode_jpeg(image)
        fin = file(input_filelist)
        with tf.Session('') as sess:
            count = 0
            while True:
                count = count + 1
                if count % 1000 == 0:
                    print(count)
                line = fin.readline()
                if not line:
                    break
                line = line.strip('\n')
                img_path = line.split(' ')[0]
                label = line.split(' ')[1]
                img = read_image(img_path)
                jpeg_string = sess.run(encoded_jpeg, feed_dict={image: img})
                example = dataset_utils.image_to_tfexample(
                    jpeg_string, 'jpeg', resize_height, resize_width, int(label))
                writer.write(example.SerializeToString())
    writer.close()


def read_image(filename):
    image = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)
    image = cv2.resize(image, (resize_height, resize_width))
    return image

if __name__ == '__main__':
    get_feature(sys.argv[1], argv[2])
    print 'DONE asm image features!'
