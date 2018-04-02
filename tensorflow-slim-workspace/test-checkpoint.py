import os
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.contrib import slim
from preprocessing import inception_preprocessing
from preprocessing import face_preprocessing
#from preprocessing.face_preprocessing import *
from nets.inception_resnet_v2 import *
import numpy as np
from tensorflow.python.training import training_util
from tensorflow.python.training import learning_rate_decay
import os


# In[2]:

SPLITS_TO_SIZES = {
  'train': 481008,
  'validation': 10572,
}
_FILE_PATTERN = 'webface_%s-*.tfrecord'
_NUM_CLASSES = 10572
train_dir = '/data/zeng/tf_models'

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    
    

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  if reader is None:
    reader = tf.TFRecordReader
    
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }


  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  #labels_to_names = None
  #if dataset_utils.has_labels(dataset_dir):
  #  labels_to_names = dataset_utils.read_label_file(dataset_dir)
    
  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=None,
      num_classes=_NUM_CLASSES,
      labels_to_names=None)


def load_batch(dataset, batch_size=128, height=256, width=256, is_training=False):

    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = face_preprocessing.preprocess_image(image_raw, is_training=True, height=256,
                                                width=256,resize_height=256, resize_width=256)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=4 * batch_size)
    
    return images, images_raw, labels

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Create a single Session to run all image coding calls.
    sess = tf.Session(config=config)

    
    dataset = get_split('train', '/data/zeng/tf_webface_256x256')
    images, _, labels = load_batch(dataset)

    
    # Create the model:
    logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)
    #print(logits.get_shape())
    # Specify the loss function:
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    
    
  
    #with tf.device(deploy_config.variables_device()):
    #global_step = slim.create_global_step()
    global_step = training_util.get_or_create_global_step()
    global_step = tf.cast(global_step, tf.int32)
    #global_step = tf.Variable(0, trainable=False)
    boundaries = [16000,24000,28000]
    #boundaries = tf.convert_to_tensor(np.array([100,200]), dtype=tf.int64)
    values = [0.1, 0.01, 0.001,0.0001]
    #learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    learning_rate = learning_rate_decay.piecewise_constant(global_step, boundaries, values)
    #print(learning_rate)

    
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=0.9,
        name='Momentum')
    
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    
    
    
    # Create some summaries to visualize the training process:
    #summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
    #                                  first_clone_scope))\
    #summaries = []
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.scalar('losses/Total_Loss', total_loss))
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Run the training:
    final_loss = slim.learning.train(
      train_op,
      logdir=train_dir,
      summary_op=summary_op,
      number_of_steps=28000, # For speed, we just do 1 epoch
      save_summaries_secs=5)
  
    print('Finished training. Final batch loss %d' % final_loss)
    

    
  


# In[ ]:

a=1000000000
type(a)


# In[ ]:



