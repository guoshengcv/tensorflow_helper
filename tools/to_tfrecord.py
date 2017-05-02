from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('record_file', '', 'file path to store tf records')
tf.app.flags.DEFINE_string('source', '', '(image_name label) list file')
tf.app.flags.DEFINE_string('root', '', 'prefix to add to image_name')
tf.app.flags.DEFINE_bool('encode', True, 'store encoded image if True')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecords(record_file, source, root='', encode=True):
  with open(source) as fid:
    filenames, labels = [], []
    for line in fid:
      sp = line.split(' ')
      filenames.append(root+sp[0])
      labels.append(int(sp[1]))

  writer = tf.python_io.TFRecordWriter(record_file)
  for idx, filename in enumerate(filenames):
    label = labels[idx]
    if encode:
      with open(filename, 'rb') as fid:
        image_encode = fid.read()
        example = tf.train.Example(
          features = tf.train.Features(
            feature = {
              'label': _int64_feature(label),
              'image_encode': _bytes_feature(image_encode)
            }
          )
        )
    else:
      image = cv2.imread(filename)
      height, width, depth = img.shape
      example = tf.train.Example(
        features = tf.train.Features(
          feature = {
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image': _bytes_feature(image.to_string())
          }
        )
      )
    writer.write(example.SerializeToString())
    if idx % 1000 == 0:
      print(idx, 'examples have been processed')
  writer.close()


if __name__ == '__main__':
  convert_to_tfrecords(
      record_file = FLAGS.record_file,
      source = FLAGS.source,
      root = FLAGS.root,
      encode = FLAGS.encode
  )
