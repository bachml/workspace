from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def module_2convBlock():
   

def deepid(inputs,
          num_classes=10572,
          is_training=True,
          width_multiplier=1,
          scope='deepid'):

  end_points = {}

  with tf.variable_scope(scope, 'deepid', [inputs, num_classes]):
    net = slim.conv2d(inputs, 32, [3, 3], scope='conv1a')
    net = slim.conv2d(inputs, 64, [3, 3], scope='conv1b')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1b')


    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 64, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    #net = tf.nn.relu(net + scope='pool1')
    net = slim.conv2d(net, 96, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
    shortcut = slim.conv2d(net, 96, [3,3], stride=1, scope='conv5')
    shortcut = slim.conv2d(shortcut, 96, [3,3], stride=1, scope='conv6')
    net = net + shortcut
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool5')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 160, scope='fc3')
    net = slim.dropout(net, 0.5, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

  end_points['Logits'] = logits
  end_points['Predictions'] = slim.softmax(logits, scope='Predictions')

  return logits, end_points



'''
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=tf.nn.relu
			):
                        #outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu):
        net = slim.convolution2d(inputs, 32 , [3, 3], stride=2, padding='SAME', scope='conv_1')
    	net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.convolution2d(net, 64 , [3, 3], stride=2, padding='SAME', scope='conv_2')
    	net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.convolution2d(net, 64 , [3, 3], stride=2, padding='SAME', scope='conv_3')
    	net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
        net = slim.convolution2d(net, 96 , [3, 3], stride=2, padding='SAME', scope='conv_4')
    	net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
	net = slim.flatten(net)
	
	net = slim.fully_connected(net, 160, scope='fc')



    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    #net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    end_points['Flatten'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_class')
    predictions = slim.softmax(logits, scope='Predictions')

    end_points['Logits'] = logits
    end_points['Predictions'] = predictions

  return logits, end_points
'''
deepid.default_image_size = 224


'''
def deepid_arg_scope(weight_decay=0.0):
  """Defines the default deepid argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc
'''

def deepid_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.prelu) as sc:
    return sc
