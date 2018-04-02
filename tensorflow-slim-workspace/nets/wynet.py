from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim



def bottleneck_group(inputs, num_output, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'resnet_block', [inputs]) as sc:
        shortcut = slim.conv2d(inputs, num_output, [3, 3], stride=1, scope='conv1')
        #shortcut = slim.batch_norm(shortcut, scope='conv1_bn')
        shortcut = tf.nn.relu(shortcut)
        shortcut = slim.conv2d(shortcut, num_output, [3, 3], stride=1, scope='conv2')
        #shortcut = slim.batch_norm(shortcut, scope='conv2_bn')
        shortcut = tf.nn.relu(shortcut)
        output = inputs + shortcut
        return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)

def bottleneck_groupdd(inputs, num_output, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'resnet_block', [inputs]) as sc:

        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=1, depth_multiplier=1, kernel_size=[3,3], scope='conv_dw')
        depthwise_conv = prelu(depthwise_conv)
        #bn = slim.batch_norm(depthwise_conv, scope='conv_dw_bn')
        pointwise_conv = slim.convolution2d(depthwise_conv, num_output, [1, 1], scope='conv_pw')
        pointwise_conv = prelu(pointwise_conv)
        #bn = slim.batch_norm(pointwise_conv, scope='conv_pw_bn')

        output = inputs + pointwise_conv
        return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)


def prelu(_x, scope=None):
    with tf.variable_scope(scope, "prelu", [_x]):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

def wynet(inputs,
          num_classes=10572,
          is_training=True,
          width_multiplier=1,
          scope='wynet'):



  end_points = {}




  with tf.variable_scope(scope, 'wynet', [inputs, num_classes]):
    net = slim.conv2d(inputs, 64, [6, 6], stride=4,  scope='conv1a',weights_initializer=slim.initializers.xavier_initializer())
    net = tf.nn.relu(net)

    net = bottleneck_group(net, 64)


    net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv2', weights_initializer=slim.initializers.xavier_initializer())
    net = tf.nn.relu(net)
   

    net = bottleneck_group(net, 128)
    net = bottleneck_group(net, 128)

    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3', weights_initializer=slim.initializers.xavier_initializer())
    net = tf.nn.relu(net)
 

    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
   

    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv4', weights_initializer=slim.initializers.xavier_initializer())
    net = tf.nn.relu(net)

    net = bottleneck_group(net, 512)
 

    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 512, scope='fc5')
    end_points['fc5'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc6')

  end_points['Logits'] = logits
  end_points['Predictions'] = slim.softmax(logits, scope='Predictions')

  return logits, end_points




wynet.default_image_size = 256


'''
def wynet_arg_scope(weight_decay=0.0):
  """Defines the default wynet argument scope.

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

def wynet_arg_scope(weight_decay=0.0005):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)) as sc:
    return sc

      #activation_fn=tf.contrib.keras.layers.PReLU,
      #activation_fn=tf.nn.relu
