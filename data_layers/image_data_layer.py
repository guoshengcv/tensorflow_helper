from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import six


__all__ = [
  'image_data_layer',
  'image_data_layer_v2'
]


def image_data_layer(source, root='', shuffle=True, num_epochs=None,
                     batch_size=1, num_thread=1, transforms=[], image_shape=(None,None,3),
                     summary=True, image_data_format='channels_first'):
  # Read input filenames and corresponding labels
  with open(source) as fid:
    filenames, labels = [], []
    for line in fid:
      sp = line.split(' ')
      filenames.append(root+sp[0])
      labels.append(int(sp[1]))

  with tf.device('/cpu:0'):
    # Create a Queue for filenames and lables
    inputQueue = tf.train.slice_input_producer(
        tensor_list = [filenames, labels],
        num_epochs = num_epochs,
        shuffle = shuffle,
        capacity = 4*batch_size
    )
    # Read and decode
    image = tf.image.decode_image(
      contents = tf.read_file(inputQueue[0]),
      channels = None
    )
    label = tf.cast(
      inputQueue[1],
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
      tf.TensorShape((None,None,None)).merge_with(image_shape)
    )

    # Create a Queue for input example
    images, labels = tf.train.batch_join(
        tensors_list = [(image, label) for _ in six.moves.range(num_thread)],
        batch_size = batch_size,
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


def image_data_layer_v2(source, root='', shuffle=True, num_epochs=None,
                        batch_size=1, num_thread=1, preprocess=None, image_shape=(None,None,3),
                        summary=True, image_data_format='channels_first'):
  with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(
        [source], num_epochs=num_epochs)
    # Read and decode
    reader = tf.TextLineReader()
    key, text_line = reader.read(filename_queue)
    
    img_name,  label = tf.decode_csv(text_line,
                                     record_defaults=[[''],[int(-1)]],
                                     field_delim=' ')
    image = tf.image.decode_image(
              tf.read_file(tf.string_join([root, img_name])),
              channels=None)
  
    # Preprocess
    if preprocess:
      image, label = preprocess(image, label)
    # Set image shape (H,W,C)
    image.set_shape(
      tf.TensorShape((None,None,None)).merge_with(image_shape))
  
    # Create a Queue for input example
    if shuffle:
      images, labels = tf.train.shuffle_batch(
          tensors=[image, label],
          batch_size=batch_size,
          num_threads=num_thread,
          capacity=1000 + 4*batch_size,
          min_after_dequeue=1000, # Ensures a minimum amount of shuffling of examples.
          shapes=None)
    else:
      images, labels = tf.train.batch(
          tensors=[image, label],
          batch_size=batch_size,
          num_threads=num_thread,
          capacity=4*batch_size,
          shapes=None,
          dynamic_pad=False)

    # Display the training images in the visualizer.
    if summary:
      tf.summary.image('images', images)

    # Change (NHWC) to (NCHW) if necessary
    if 'channels_first' == image_data_format:
      images = tf.transpose(images, [0, 3, 1, 2])

    # Return result
    return images, labels
