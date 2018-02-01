from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops



def preprocess_for_train(image,
                         height,
                         width,
                         resize_height,
                         resize_width,
                         crop_height,
                         crop_width,
                         scope=None
                        ):
  '''
  if not is_resize:
    pass
  else:
    image = tf.image.resize_image_with_crop_or_pad(image, new_height, new_width)
    #image = tf.image.resize_images(image, [new_height, new_width])
  
  if not is_random_crop:
    pass
  else:
    image = tf.random_crop(image, [224, 224, 3]) 
  



  image = tf.image.random_flip_left_right(image)
  image = tf.to_float(image)
  MEAN = [127.5, 127.5, 127.5]    

  means = tf.reshape(tf.constant(MEAN), [1, 1, 3])
   
  image = image - means 
  image = image / 128
  
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
  image = tf.squeeze(image, [0])
  
  image.set_shape([height, width, 3])
  '''
  with tf.name_scope(scope, 'train_image', [image, height, width]):
    means = tf.reshape(tf.constant([127.5,127.5,127.5]), [1, 1, 3])
    #image = image - means 
    #image = image / 128
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    
    if resize_height and resize_width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [resize_height, resize_width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    if crop_height and crop_width:
      image = tf.random_crop(image, [crop_height, crop_width, 3]) 
        
    #image = tf.subtract(image, 0.5)
    #image = tf.multiply(image, 2.0)
    return image

  
  return image



def preprocess_image(image, 
                     is_training=False,
                     height=None,
                     width=None,
                     resize_height=None,
                     resize_width=None,
                     crop_height=None,
                     crop_width=None):

  if is_training:
    return preprocess_for_train(image, height, width, resize_height, resize_width, crop_height, crop_width)
  else:
    return preprocess_for_train(image, height, width, resize_height, resize_width, crop_height, crop_width) #TODO
