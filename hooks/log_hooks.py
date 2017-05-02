from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
from tensorflow import logging as logging
import six

import numpy as np
from datetime import datetime


__all__ = [
  'LoggerHook'
]


class LoggerHook(tf.train.SessionRunHook):
  """Logs loss, evaluation and runtime."""

  def __init__(self, losses, learning_rate, global_step = None,
               log_interval = 20):
    self._losses = losses
    self._global_step = global_step or \
                        tf.contrib.framework.get_global_step()
    self._log_interval = log_interval
    self._learning_rate = learning_rate

  def begin(self):
    self._step = -1

  def before_run(self, run_context):
    self._step += 1
    return tf.train.SessionRunArgs([
      self._losses,
      self._global_step,
      self._learning_rate
    ])

  def after_run(self, run_context, run_values):
    loss_values, step_value, lr_value = run_values.results
    if self._step % self._log_interval == 0:
      # Main log message include total_loss
      log_str = ' {}] Train: step = {:d}, lr = {:g} '.format(
        datetime.now(), step_value, lr_value
      )
      # total_loss message
      total_loss_str = 'total_loss = {:g}'.format(
        loss_values['total_loss']
      )
      # losses message except total_loss
      loss_str = ''
      loss_format_str = ('\n'
                         + ' '*len(log_str + 'INFO:tensorflow:')
                         + '{} = {:g}'
      )
      for k in loss_values:
        if k != 'total_loss':
          loss_str += loss_format_str.format(
            k, loss_values[k]
          )
      logging.info(
        log_str + total_loss_str + loss_str
      )
