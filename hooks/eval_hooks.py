from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib
from tensorflow import logging as logging
from six.moves import range as range

from tensorflow_helper.framework.global_var import get_global_phase

import numpy as np
from datetime import datetime


__all__ = [
  'EvalHook'
]


class EvalHook(tf.train.SessionRunHook):
  """Run evaluation."""

  def __init__(self, eval_results, global_step = None,
               test_interval = 5000, test_iters = 1,
               checkpoint_dir = None):
    self._global_step = global_step or \
                        tf.contrib.framework.get_global_step()
    self._global_phase = get_global_phase()
    self._eval_results = eval_results
    self._test_interval = test_interval
    self._test_iters = test_iters
    self._summary_writer = tf.summary.FileWriterCache.get(checkpoint_dir) \
                           if checkpoint_dir else None

  def begin(self):
    self._step = -1

  def before_run(self, run_context):
    self._step += 1
    return tf.train.SessionRunArgs(self._global_step)

  def after_run(self, run_context, run_values):
    step_value = run_values.results
    # Evaluation
    if int(step_value) % self._test_interval == 0:
      result_values = []
      for idx in range(self._test_iters):
        result_values.append(
          run_context.session.run(
            self._eval_results,
            feed_dict = {self._global_phase: False}
          )
        )
      # Aggregate evaluation results
      all_results = {}
      for k in result_values[0].keys():
        all_results[k] = np.average(
          np.vstack(
            [results[k] for results in result_values]
          ),
          axis = 0
        )[0]
      # Log to std
      log_str = ' {}] Evaluation: step = {:d} '.format(
        datetime.now(), step_value
      )
      result_str = ''
      result_format_str = ('\n'
                           + ' '*len(log_str + 'INFO:tensorflow:')
                           + '{} = {:g}'
      )
      for idx,k in enumerate(all_results.keys()):
        if idx == 0:
          result_str += ('{} = {:g}').format(
            k, all_results[k]
          )
        else:
          result_str += result_format_str.format(
            k, all_results[k]
          )
      logging.info(
        log_str + result_str
      )
      # Summary to tensorboard
      if self._summary_writer:
        for k in all_results:
          summary_proto = tf.Summary(
            value = [
              tf.Summary.Value(
                tag = 'evaluation/' + k,
                simple_value = all_results[k]
              )
            ]
          )
          self._summary_writer.add_summary(
            summary = summary_proto,
            global_step = step_value
          )

  def end(self, session):
    # Final Evaluation
    step_value = session.run(self._global_step)
    if int(step_value) % self._test_interval != 0:
      result_values = []
      for idx in range(self._test_iters):
        result_values.append(
          session.run(
            self._eval_results,
            feed_dict = {self._global_phase: False}
          )
        )
      # Aggregate evaluation results
      all_results = {}
      for k in result_values[0].keys():
        all_results[k] = np.average(
          np.vstack(
            [results[k] for results in result_values]
          ),
          axis = 0
        )[0]
      # Log to std
      log_str = ' {}] Evaluation: step = {:d} '.format(
        datetime.now(), step_value
      )
      result_str = ''
      result_format_str = ('\n'
                           + ' '*len(log_str + 'INFO:tensorflow:')
                           + '{} = {:g}'
      )
      for idx,k in enumerate(all_results.keys()):
        if idx == 0:
          result_str += '{} = {:g}'.format(
            k, all_results[k]
          )
        else:
          result_str += result_format_str.format(
            k, all_results[k]
          )
      logging.info(
        log_str + result_str
      )
      # Summary to tensorboard
      if self._summary_writer:
        for k in all_results:
          summary_proto = tf.Summary(
            value = [
              tf.Summary.Value(
                tag = 'evaluation/' + k,
                simple_value = all_results[k]
              )
            ]
          )
          self._summary_writer.add_summary(
            summary = summary_proto,
            global_step = step_value
          )
