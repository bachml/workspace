from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim



def bottleneck_group(inputs, num_output, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'resnet_block', [inputs]) as sc:
        shortcut = slim.conv2d(inputs, num_output, [3, 3], stride=1, scope='conv1')
        shortcut = slim.conv2d(shortcut, num_output, [3, 3], stride=1, scope='conv2')
        output = inputs + shortcut
        return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)

def bottleneck_grouhp(inputs, num_output, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'resnet_block', [inputs]) as sc:

        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=num_output, stride=1, depth_multiplier=1, kernel_size=[3,3], scope='conv_dw')
        #bn = slim.batch_norm(depthwise_conv, scope='conv_dw_bn')
        pointwise_conv = slim.convolution2d(depthwise_conv, num_output, [1, 1], scope='conv_pw')
        #bn = slim.batch_norm(pointwise_conv, scope='conv_pw_bn')

        output = inputs + pointwise_conv
        return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)


def wenet(inputs,
          num_classes=10572,
          is_training=True,
          width_multiplier=1,
          scope='wenet'):



  end_points = {}




  with tf.variable_scope(scope, 'wenet', [inputs, num_classes]):
    net = slim.conv2d(inputs, 32, [3, 3], stride=1,  scope='conv1a',weights_initializer=slim.initializers.xavier_initializer())
    net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv1b', weights_initializer=slim.initializers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1b')


    net = bottleneck_group(net, 64)


    net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv2', weights_initializer=slim.initializers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

    net = bottleneck_group(net, 128)
    net = bottleneck_group(net, 128)

    net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3', weights_initializer=slim.initializers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')

    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)
    net = bottleneck_group(net, 256)

    net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv4', weights_initializer=slim.initializers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')

    net = bottleneck_group(net, 512)
    net = bottleneck_group(net, 512)
    net = bottleneck_group(net, 512)

    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 512, scope='fc5')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc6')

  end_points['Logits'] = logits
  end_points['Predictions'] = slim.softmax(logits, scope='Predictions')

  return logits, end_points




wenet.default_image_size = 224


'''
def wenet_arg_scope(weight_decay=0.0):
  """Defines the default wenet argument scope.

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

def wenet_arg_scope(weight_decay=0.0005):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      activation_fn=tf.nn.relu) as sc:
    return sc
