from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def deepid(inputs,
          num_classes=10572,
          is_training=True,
          width_multiplier=1,
          scope='deepid'):

  end_points = {}

  with tf.variable_scope(scope, 'deepid', [inputs, num_classes]):
    net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 64, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = slim.conv2d(net, 96, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
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



deepid.default_image_size = 224




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
      activation_fn=tf.nn.relu) as sc:
    return sc
