from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
import numpy as np
import six

import tensorflow_helper as tfh



class AlexNet(object):
  def __init__(self, num_cls, image_data_format = 'channels_first'):
    self._num_cls = num_cls
    self._image_data_format = image_data_format

  def model(self, inputs):
    phase = tfh.get_global_phase()
    images = inputs['images']
    # conv1
    conv1 = tf.layers.conv2d(
      inputs = images,
      filters = 64,
      kernel_size = (5,5),
      strides = (1,1),
      padding = 'same',
      data_format = self._image_data_format,
      activation = tf.nn.relu,
      use_bias = True,
      kernel_initializer = tf.truncated_normal_initializer(0, 5e-2),
      bias_initializer = tf.constant_initializer(0),
      name = 'conv1'
    )
    pool1 = tf.layers.max_pooling2d(
      inputs = conv1,
      pool_size = (3,3),
      strides = (2,2),
      padding = 'same',
      data_format = self._image_data_format,
      name='pool1'
    )
    norm1 = tf.nn.lrn(
      input = pool1,
      depth_radius = 4,
      bias = 1.0,
      alpha = 0.001 / 9.0,
      beta = 0.75,
      name='norm1'
    )
  
    # conv2
    conv2 = tf.layers.conv2d(
      inputs = norm1,
      filters = 64,
      kernel_size = (5,5),
      strides = (1,1),
      padding = 'same',
      data_format = self._image_data_format,
      activation = tf.nn.relu,
      use_bias = True,
      kernel_initializer = tf.truncated_normal_initializer(0, 5e-2),
      bias_initializer = tf.constant_initializer(0),
      name = 'conv2'
    )
    norm2 = tf.nn.lrn(
      input = conv2,
      depth_radius = 4,
      bias = 1.0,
      alpha = 0.001 / 9.0,
      beta = 0.75,
      name = 'norm2'
    )
    pool2 = tf.layers.max_pooling2d(
      inputs = norm2,
      pool_size = (3,3),
      strides = (2,2),
      padding = 'same',
      data_format = self._image_data_format,
      name = 'pool2'
    )
  
    # flatten
    flatten = tf.reshape(
      tensor = pool2,
      shape = [pool2.shape[0].value, -1]
    )
  
    # fc3
    fc3 = tf.layers.dense(
      inputs = flatten,
      units = 384,
      activation = tf.nn.relu,
      use_bias = True,
      kernel_initializer = tf.truncated_normal_initializer(0, 4e-2),
      bias_initializer = tf.constant_initializer(0.1),
      kernel_regularizer = lambda _: 4e-3 * tf.nn.l2_loss(_),
      name = 'fc3'
    )
  
    # fc4
    fc4 = tf.layers.dense(
      inputs = fc3,
      units = 192,
      activation = tf.nn.relu,
      use_bias = True,
      kernel_initializer = tf.truncated_normal_initializer(0, 4e-2),
      bias_initializer = tf.constant_initializer(0.1),
      kernel_regularizer = lambda _: 4e-3 * tf.nn.l2_loss(_),
      name = 'fc4'
    )
  
    # pred
    pred = tf.layers.dense(
      inputs = fc4,
      units = self._num_cls,
      use_bias = True,
      kernel_initializer = tf.truncated_normal_initializer(0, 1/192.0),
      bias_initializer = tf.constant_initializer(0),
      name = 'pred'
    )
  
    return {'logits': pred}


  def loss_fn(self, infers, targets, scope):
    """
    Construct total loss by summing regularization losses and target losses.
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    logits = infers['logits']
    labels = tf.cast(
      targets['labels'],
      dtype = tf.int64
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels = labels,
      logits = logits,
      name = 'cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(
      input_tensor = cross_entropy,
      name = 'cross_entropy'
    )
    tf.add_to_collection(
      name = tf.GraphKeys.LOSSES,
      value = cross_entropy_mean
    )
  
    # The total loss is defined as the cross entropy loss.
    return {'tower_loss': tf.add_n(
      inputs = tf.get_collection(
        key = tf.GraphKeys.LOSSES,
        scope = scope
      )
    )
    }


  def eval_metric(self, infers, targets):
    with tf.device('/cpu:0'):
      return {'cls_acc': tf.reduce_mean(
        input_tensor = tf.cast(
          x = tf.nn.in_top_k(
            predictions = infers['logits'],
            targets = targets['labels'],
            k = 1
          ),
          dtype = tf.float32
        ),
        axis = 0
      )
      }
