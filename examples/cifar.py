from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
import numpy as np
import six

import os

import tensorflow_helper as tfh
from tensorflow_helper.models.alexNet import AlexNet



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', '0,1', 'GPU to be used')
tf.app.flags.DEFINE_integer('max_steps', 400000,
                            'the number of iterations to train')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch_size')
tf.app.flags.DEFINE_integer('test_iters', 40,
                            'the number of iterations to test')
tf.app.flags.DEFINE_string('image_data_format', 'channels_first',
                           'channels_first for (NCHW), channels_last for (NHWC)')
tf.app.flags.DEFINE_float('base_lr', 0.1, 'base learning rate')
tf.app.flags.DEFINE_integer('lr_decay_step', 100000,
                            'decay learning rate every lr_decay_step')
tf.app.flags.DEFINE_float('lr_decay_rate', 0.1, 'learning decay rate')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/tf-cifar/',
                           'directory to save checkpoint')
tf.app.flags.DEFINE_integer('snapshot_interval', 20000,
                            'save checkpoint every snapshot_interval steps')
tf.app.flags.DEFINE_string('snapshot_prefix', 'cifar-model',
                           'prefix added to checkpoint files')




os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
FLAGS.num_gpu = len(FLAGS.gpu.split(','))
tf.logging.set_verbosity(tf.logging.INFO)


alexNet = AlexNet(
  num_cls = 10,
  image_data_format = FLAGS.image_data_format
)
model = tf.make_template(
  'alexNet',
  alexNet.model
)

with tf.Graph().as_default():
  # Get global_phse and global_step
  global_phase = tfh.get_or_create_global_phase()
  global_step = tf.contrib.framework.get_or_create_global_step()
  # Create optimizer
  lr = tf.train.exponential_decay(
    learning_rate = FLAGS.base_lr,
    global_step = global_step,
    decay_steps = FLAGS.lr_decay_step,
    decay_rate = FLAGS.lr_decay_rate,
    staircase = True
  )
  opt = tf.train.GradientDescentOptimizer(
    learning_rate = lr
  )
  # Summary learning rate
  tf.summary.scalar(
    name = 'learning_rate',
    tensor = lr
  )


  with tf.name_scope('train'):
    # images, labels = tfh.image_data_layer(
    #   source='data/cifar10/train/jpeg/train.txt',
    #   root='data/cifar10/train/jpeg/imgs/',
    #   shuffle=True,
    #   num_epochs=None,
    #   batch_size=FLAGS.batch_size,
    #   num_thread=80,
    #   transforms = [
    #     tfh.image_transforms.random_crop(
    #       size = (24,24,3)
    #     ),
    #     tfh.image_transforms.random_flip_left_right(),
    #     tfh.image_transforms.random_brightness(
    #       max_delta = 0.2
    #     ),
    #     tfh.image_transforms.random_contrast(
    #       lower = 0.2,
    #       upper = 1.8
    #     ),
    #     tf.image.per_image_standardization
    #   ],
    #   image_shape = (None,None,3),
    #   summary = True,
    #   image_data_format = FLAGS.image_data_format
    # )

    images, labels = tfh.tfrecord_data_layer(
      record_files = 'data/cifar10/train/train.tf',
      shuffle = True,
      num_epochs = None,
      batch_size = FLAGS.batch_size,
      num_thread = 80,
      transforms = [
        tfh.image_transforms.sample_distorted_bounding_box(),
        tfh.image_transforms.resize_image(
          size = (24,24)
        ),
        # tfh.image_transforms.random_crop(
        #   size = (24,24,3)
        # ),
        tfh.image_transforms.random_flip_left_right(),
        tfh.image_transforms.random_brightness(
          max_delta = 0.2
        ),
        tfh.image_transforms.random_contrast(
          lower = 0.2,
          upper = 1.8
        ),
        tfh.image_transforms.per_image_standardization()
      ],
      image_shape = (None,None,3),
      summary = True,
      image_data_format = FLAGS.image_data_format
    )

    train_op, all_losses = tfh.train(
      model = model,
      loss_fn = alexNet.loss_fn,
      inputs = {'images': images},
      targets = {'labels': labels},
      opt = opt,
      num_gpu = FLAGS.num_gpu,
      grad_hooks = [
        tfh.hooks.grad_double_bias_hook(),
        tfh.hooks.grad_print_hook()
      ]
    )


  with tf.name_scope('eval'):
    eval_images, eval_labels = tfh.image_data_layer(
      source = 'data/cifar10/test/jpeg/test.txt',
      root = 'data/cifar10/test/jpeg/imgs/',
      shuffle = False,
      num_epochs = None,
      batch_size = FLAGS.batch_size,
      num_thread = 10,
      transforms = [
        tfh.image_transforms.resize_image_with_crop_or_pad(
          target_height = 24,
          target_width = 24
        ),
        tfh.image_transforms.per_image_standardization()
      ],
      image_shape = (None,None,3),
      summary = True,
      image_data_format = FLAGS.image_data_format
    )
  
    eval_results = tfh.eval_model(
      model = model,
      metric_fn = alexNet.eval_metric,
      inputs = {'images': eval_images},
      targets = {'labels': eval_labels},
      num_gpu = 2
    )


  scaffold = tf.train.Scaffold(
    saver = tf.train.Saver(
      allow_empty = False,
      max_to_keep = 0
    ),
#     init_fn = tfh.model_load_from_pickle(
#       data_path = '/tmp/model_dump_load/4W_alexNet_model.pkl',
#       skip_regexs = [
#         'alexNet/fc3/bias'
#       ],
#       reshape=False
#     )
  )
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.2
  config.log_device_placement = False
  config.allow_soft_placement = True


  with tf.train.MonitoredTrainingSession(
      config = config,
      scaffold = scaffold,
      checkpoint_dir = FLAGS.checkpoint_dir,
      save_checkpoint_secs = None,
      hooks=[
        tf.train.StopAtStepHook(
          last_step = FLAGS.max_steps
        ),
        tf.train.CheckpointSaverHook(
          scaffold = scaffold,
          checkpoint_dir = FLAGS.checkpoint_dir,
          save_steps = FLAGS.snapshot_interval,
          checkpoint_basename = FLAGS.snapshot_prefix
        ),
        tf.train.NanTensorHook(
          loss_tensor = all_losses['total_loss']
        ),
        tfh.LoggerHook(
          losses = all_losses,
          learning_rate = lr,
          log_interval = 200,
        ),
        tfh.EvalHook(
          eval_results = eval_results,
          test_iters = FLAGS.test_iters,
          checkpoint_dir = FLAGS.checkpoint_dir
        )
      ]
  ) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(
        train_op,
        feed_dict = {
          global_phase: True
        }
      )
