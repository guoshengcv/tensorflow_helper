from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib
import six
from six.moves import range as range

import re


__all__ = [
  'activation_summary',
  'train'
]


def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  with tf.device('/cpu:0'):
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _validate_loss_names(losses):
  reserved_names = set(
    ['total_loss', 'regularization_loss']
  )
  for k in losses:
    if k in reserved_names:
      raise(
        KeyError(
          'Loss name ' + k + 'is reserved'
        )
      )


def train(model, loss_fn, inputs, targets, opt, global_step = None,
          regularize = True, num_gpu = 1, grad_hooks = [],
          param_decay = 0.9999, loss_decay = 0.9):
  """
  Create train op. Distribute training on multi gpus.

  Args:
    model: A callable function. Has a named argument `inputs` which is a python dict,
           e.g. {'input1': input_tensor1, 'input2': input_tensor2}. Return a python
           dict, e.g. {'infer1': infer_tensor1, 'infer2': infer_tensor_2}.
    loss_fn: A callable function. Has three name arguments: 1. `infers`: a python dict
             returned by `model`; 2. `targets`: a python dict, e.g. {'target1': 
             target_tensor1, 'target2': target_tensor2}; 3. `scope`: tf.name_scope.
             return a python dict, e.g. {'loss1': loss_tensor1, 'loss2': loss_tensor2}.
             The returned dict can not use 'total_loss' and 'regularization_loss' as key.
    inputs: A python dict fed to `model`.
    targets: A python dict fed `loss_fn`.
    opt: A subclass of tf.train.Optimizer.
    global_step: if None, use tf.contrib.framework.get_global_step().
    regularize: python bool, whether to add regularization loss.
                Add loss in collection tf.GraphKeys.REGULARIZATION_LOSSES.
    num_gpu: number of gpu used to train.
    grad_hooks: A python list of callable function. These function take a named argument
                `grads` and return a processed `grads`. `grads` will be fed to 
                opt.apply_gradients, e.g. [(grad1, var1), (None, var2), (grad3, var2)].

  Returns:
    train_op: A tensorflow op to train the model one step.
    all_losses: A python dict. e.g. {'total_loss': total_loss_tensor, 'regularization_loss':
                regularization_loss_tensor, 'cls_loss1': cls_loss_tensor1, 'cls_loss2':
                cls_loss_tensor2}.
  """
  # Get global step
  global_step = global_step or \
                tf.contrib.framework.get_global_step()
  # Get batch size from any input item
  batch_size = six.next(six.itervalues(inputs)).shape[0].value
  # Deal with the case that batch_size not evenly divede by num_gpu
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
  # Forward network and compute gradients on gpus
  tower_grads = []
  tower_losses = []
  for idx in range(num_gpu):
    with tf.variable_scope(
        tf.get_variable_scope(), caching_device='/cpu:0', reuse=(idx>0)), \
        tf.device('/gpu:{}'.format(idx)), \
        tf.name_scope('tower_{}'.format(idx)) as scope:
      infers = model(
        inputs = split_inputs[idx]
      )
      losses = loss_fn(
        infers = infers,
        targets = split_targets[idx],
        scope = scope
      )
      # Protect reserved loss name
      _validate_loss_names(losses)
      grad = opt.compute_gradients(
        loss = tf.add_n(
          inputs = [losses[k] for k in losses]
        )
      )
      tower_losses.append(losses)
      tower_grads.append(grad)
  # Place regularization loss on gpu 0
  if regularize:
    with tf.device('/gpu:0'):
      regularization_loss = tf.add_n(
        inputs = tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES
        ),
        name='regularization_loss'
      )
      regularization_grad = opt.compute_gradients(
        loss = regularization_loss
      )

  # Aggregate gradients and regularization gradients for updating
  grads = []
  for idx, grad_vars in enumerate(zip(*tower_grads)):
    grad = tf.reduce_mean(
             input_tensor = tf.concat(
               values = [tf.expand_dims(_[0], 0) for _ in grad_vars],
               axis = 0
             ),
             axis = 0
    )
    if regularize \
       and regularization_grad[idx][0] != None \
       and grad != None:
      grad += regularization_grad[idx][0]
    grads.append((grad, grad_vars[0][1]))

  # Apply gradient hooks
  for grad_hook in grad_hooks:
    grads = grad_hook(
      grads = grads
    )
  # Apply gradients
  apply_gradient_op = opt.apply_gradients(
    grads_and_vars = grads,
    global_step = global_step
  )

  # Aggregate losses from towers
  all_losses = {}
  for k in tower_losses[0].keys():
    all_losses[k] = tf.reduce_mean(
      input_tensor = [losses[k] for losses in tower_losses]
    )
  # Sum all losses except regularization loss
  all_losses['total_loss'] = tf.reduce_mean(
    input_tensor = [all_losses[k] for k in all_losses]
  )
  if regularize:
    all_losses['regularization_loss'] = regularization_loss
    all_losses['total_loss'] = tf.add(
      x = all_losses['total_loss'],
      y = all_losses['regularization_loss']
    )
  # Track moving averages of tower losses, total_loss and regularization loss
  # and summary them to tensorboard
  loss_averager = tf.train.ExponentialMovingAverage(
    decay = loss_decay,
    name = 'loss_average'
  )
  loss_average_op = loss_averager.apply(
    var_list = [all_losses[k] for k in all_losses]
  )
  for k in all_losses:
    tf.summary.scalar(
      name = 'losses/' + k + '/raw',
      tensor = all_losses[k]
    )
    tf.summary.scalar(
      name = 'losses/' + k + '/avg',
      tensor = loss_averager.average(
        var = all_losses[k]
      )
    )

  # Track the moving averages of all trainable variables.
  variable_averager = tf.train.ExponentialMovingAverage(
    decay = param_decay,
    num_updates = global_step,
    name = 'weights_average'
  )
  variable_average_op = variable_averager.apply(
    var_list = tf.trainable_variables()
  )

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(
        name = var.op.name + '/gradients',
        values = grad
      )

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(
      name = var.op.name,
      values = var)

  with tf.control_dependencies([
      apply_gradient_op, variable_average_op, loss_average_op]):
    train_op = tf.no_op(name='train')

  return train_op, all_losses
