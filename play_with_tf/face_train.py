# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from net import *

FLAGS = None


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(
        serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
            'img_raw' : tf.FixedLenFeature([], tf.string),
        }
    )
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [112, 96, 3])
    #img = tf.cast(img, tf.float32) * (1./ 255) - 0.5
    img = (tf.cast(img, tf.float32) - 127.5 ) - (1./128)
    label = tf.cast(features['label'], tf.int32)
    return img, label

'''
img, label = read_and_decode("test_train.tfrecords")
 
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=100, capacity=2000,
                                                min_after_dequeue=1000)
x = tf.placeholder("float",[None,784])
 
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
 
y = tf.nn.softmax(tf.matmul(x,w)+b)
 
y_ = tf.placeholder("float",[None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
 
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
 
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                   batch_size=100, capacity=2000,
                                                min_after_dequeue=1000)
threads = tf.train.start_queue_runners(sess=sess)
for i in range(1000):
    img_xs,label_xs = sess.run([img_batch,label_batch])
    sess.run(train_step, feed_dict={x: img_xs,y_:label_xs})
'''

def main(train_data, val_data):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  img, label = read_and_decode(train_data)
  img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=100, capacity=2000,
                                                min_after_dequeue=1000)

  
  val_img, val_label = read_and_decode(val_data)


  # Create the model
  x = tf.placeholder(tf.float32, [None, 112*96*3])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10572])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
