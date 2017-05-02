from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2


__all__ = [
  'tfrecord_data_layer'
]


def tfrecord_data_layer(record_files, encode=True, shuffle=True, num_epochs=None,
                        batch_size=1, num_thread=1, transforms=[], image_shape=(None,None,3),
                        summary=True, image_data_format='channels_first'):
  with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(
        [record_files],
        num_epochs = num_epochs
    )
    # Read and decode
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if encode:
      example = tf.parse_single_example(
        serialized_example,
        features = {
          'label': tf.FixedLenFeature([], tf.int64),
          'image_encode': tf.FixedLenFeature([], tf.string)
        }
      )
      image = tf.image.decode_image(
          contents = example['image_encode'],
          channels = None
      )
      label = tf.cast(
          example['label'],
          dtype = tf.int64
      )
    else:
      example = tf.parse_single_example(
        serialized_example,
        features = {
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'image': tf.FixedLenFeature([], tf.string)
        }
      )
      image = tf.decode_raw(
        example['image'],
        out_type = tf.uint8
      )
      label = tf.cast(
        example['label'],
        dtype = tf.int64
      )
    # Convert dtype of image to float
    image = tf.image.convert_image_dtype(
      image = image,
      dtype = tf.float32
    )

    # Preprocess
    for transform in transforms:
      image, label = transform(image, label)
    # Set image shape (H,W,C)
    image.set_shape(
      tf.TensorShape((None,None,None)).merge_with(image_shape))

    # Create a Queue for input example
    if shuffle:
      images, labels = tf.train.shuffle_batch(
        tensors = [image, label],
        batch_size = batch_size,
        num_threads = num_thread,
        capacity = 1000 + 4*batch_size,
        min_after_dequeue = 1000, # Ensures a minimum amount of shuffling of examples.
        shapes = None
      )
    else:
      images, labels = tf.train.batch(
        tensors = [image, label],
        batch_size = batch_size,
        num_threads = num_thread,
        capacity = 4*batch_size,
        shapes = None,
        dynamic_pad = False
      )

    # Display the training images in the visualizer.
    if summary:
      tf.summary.image(
        name = 'images',
        tensor = images
      )
    
    # Change (NHWC) to (NCHW) if necessary
    if 'channels_first' == image_data_format:
      images = tf.transpose(
        images,
        perm = [0, 3, 1, 2]
      )

    # Return result
    return images, labels
