import os
import tensorflow as tf
from PIL import Image


def write_tfrecord(filelist, output_name, img_h, img_w):
    writer = tf.python_io.TFRecordWriter(output_name)
    fin = file(filelist)
    count = 0
    while True:
        count = count + 1
        if count%1000 == 0 :
            print(count)
        line = fin.readline()
        if not line:
            break
        line = line.strip('\n')
        img_path = line.split(' ')[0]
        label = line.split(' ')[1]

        img = Image.open(img_path)
        img = img.resize((img_h, img_w))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    output_name = "webface.tfrecords"
    write_tfrecord('train.txt', output_name, 256,256)
