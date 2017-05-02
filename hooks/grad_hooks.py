from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib
import six
from six.moves import range as range

from tensorflow_helper.utils.string_utils import in_res

import re


__all__ = [
  'grad_clip_by_value_hook',
  'grad_print_hook',
  'grad_scale_hook',
  'grad_double_bias_hook'
]


def grad_clip_by_value_hook(clip_vale_min, clip_value_max):
  def fun(grads):
    processed_grads = []
    for grad, var in grads:
      if grad == None:
        cliped_grad = None
      else:
        cliped_grad = tf.clip_by_value(
          grad,
          clip_value_min = clip_value_min,
          clip_value_max = clip_value_max
        )
      processed_grads.append(
        (cliped_grad, var)
      )
  
    return processed_grads
  return fun

def grad_print_hook():
  def fun(grads):
    for grad, var in grads:
      print('grad: {} var: {}'.format(
          grad, var
      )
      )
    return grads
  return fun

def grad_scale_hook(alpha, regexs):
  _regexs = [re.compile(_) for _ in regexs]
  def fun(grads):
    processed_grads = []
    for grad, var in grads:
      if grad != None \
         and in_res(var.name, _regexs):
        grad *= alpha
      processed_grads.append(
        (grad, var)
      )
    return processed_grads
  return fun

def grad_double_bias_hook():
  return grad_scale_hook(
    alpha = 2.0,
    regexs = [
      '.*/bias:0'
    ]
  )
