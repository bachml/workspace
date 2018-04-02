
import tensorflow as tf
from tensorflow.contrib import slim
#from preprocessing import inception_preprocessing
#from preprocessing import face_preprocessing
#from preprocessing.face_preprocessing import *
#from nets.inception_resnet_v2 import *
import numpy as np
#from tensorflow.python.training import training_util
#from tensorflow.python.training import learning_rate_decay
import os

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

