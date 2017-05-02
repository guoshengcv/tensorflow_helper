from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
import numpy as np
import six
from six.moves import range as range


__all__ = [
  'eval_model'
]


def eval_model(model, metric_fn, inputs, targets, num_gpu):
  """
  Create evaluation op. Distribute evaluation on multi gpus.

  Args:
    model: A callable function. Has a named argument `inputs` which is a python dict,
           e.g. {'input1': input_tensor1, 'input2': input_tensor2}. Return a python
           dict, e.g. {'infer1': infer_tensor1, 'infer2': infer_tensor_2}.
    metric_fn: A callable function. Has two name arguments: 1. `infers`: a python dict
             returned by `model`; 2. `targets`: a python dict, e.g. {'target1': 
             target_tensor1, 'target2': target_tensor2};
             return a python dict, e.g. {'acc1': acc_tensor1, 'acc2': acc_tensor2}.
    inputs: A python dict fed to `model`.
    targets: A python dict fed `metric_fn`.
    num_gpu: number of gpu used to train.

  Returns:
    all_results: A python dict. e.g. {'acc1': acc_tensor1, 'acc2': acc_tensor2}.
  """
  # Get batch size from any input item
  batch_size = six.next(six.itervalues(inputs)).shape[0].value
  split_size = [int(batch_size/num_gpu) for _ in range(num_gpu)]
  if batch_size % num_gpu != 0:
    split_size[-1] += (batch_size % num_gpu)
  # Split inputs
  split_inputs = [{} for _ in range(num_gpu)]
  for k in inputs:
    splits = tf.split(inputs[k], split_size, axis=0)
    for i in range(num_gpu):
      split_inputs[i][k] = splits[i]
  # Split targets
  split_targets = [{} for _ in range(num_gpu)]
  for k in targets:
    splits = tf.split(targets[k], split_size, axis=0)
    for i in range(num_gpu):
      split_targets[i][k] = splits[i]
  # Place evalution compution on gpus
  tower_results = []
  for idx in range(num_gpu):
    with tf.variable_scope(tf.get_variable_scope(),
                           caching_device='/cpu:0', reuse=True),\
        tf.device('/gpu:{}'.format(idx)), \
        tf.name_scope('eval_tower_{}'.format(idx)):
      infers = model(
        inputs = split_inputs[idx]
      )
      results = metric_fn(
        infers = infers,
        targets = split_targets[idx]
        )
      tower_results.append(results)
  # Aggregate losses from towers
  all_results = {}
  for k in tower_results[0].keys():
    all_results[k] = tf.reduce_mean(
      input_tensor = [results[k] for results in tower_results]
    )

  return all_results
