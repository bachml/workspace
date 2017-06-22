#from datasets import dataset_utils
import tensorflow as tf
slim = tf.contrib.slim
_NUM_CLASSES = 9
_SPLITS_TO_SIZES = {'train': 10868, 'test': 4000}
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [299 x 299 x 3]  image.',
    'label': 'A single integer between 0 and 8',
}



def get_dataset():
	reader = tf.TFRecordReader
	keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
        items_to_handlers = {
         'img_raw': slim.tfexample_decoder.Image(shape=[112, 96, 3]),
         'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }
        decoder = slim.tfexample_decoder.TFExampleDecoder(
         keys_to_features, items_to_handlers)
        labels_to_names = None
        dataset_dir = '/data/zeng/webface_112x96/webface_112x96_train.tfrecords'
        return slim.dataset.Dataset(
             data_sources=dataset_dir,
             reader=reader,
             decoder=decoder,
             num_samples=481008,
             num_classes=10572,
             items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
             labels_to_names=labels_to_names)

if __name__ == '__main__':
    x = get_dataset()
    print(x)