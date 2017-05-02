from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow_helper.utils.string_utils import in_res

import re
try:
  import cPickle as pickle
except ImportError:
  import pickle


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('act', 'print', 'action to peform: '
                           '(print: print all tensors in checkpoint_file)'
                           '(extract: extract tensor in net_name to pickle)')
tf.app.flags.DEFINE_string('checkpoint_file', '', 'file stores the checkpoint')
tf.app.flags.DEFINE_string('net_name', '', 'tensors in this net to be extracted')
tf.app.flags.DEFINE_string('save_path', '', 'prefix to model name')


def print_all_tensors(checkpoint_file):
  """
  Args:
    Prints tensors in a checkpoint file and their shape.
  """
  try:
    reader = tf.train.NewCheckpointReader(
      filepattern = checkpoint_file
    )
    tensor_shape_dict = reader.get_variable_to_shape_map()
    for tensor_name in tensor_shape_dict:
      print("Tensor_name: {} -> Shape: {}".format(
        tensor_name,
        tensor_shape_dict[tensor_name]
      )
      )
  except Exception as e:
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")


def extract_models(checkpoint_file, models_regexs):
  """
  Args:
    models_regexs: {
      'model1': [re11, re12, re13],
      'model2': [re21, re22]
    }
  Returns:
    models: {
      'model1': {'tensor1': np.ndarray, 'tensor2': np.ndarray},
      'model2': {'tensor3': np.ndarray, 'tensor4': np.ndarray}
    }
  """
  # Get all tensors in checkpoint
  try:
    reader = tf.train.NewCheckpointReader(
      filepattern = checkpoint_file
    )
    tensor_shape_dict = reader.get_variable_to_shape_map()
    all_tensors = {}
    for tensor_name in tensor_shape_dict:
      all_tensors[tensor_name] = reader.get_tensor(tensor_name)
  except Exception as e:
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
  # Save models
  models = {}
  for model_name in models_regexs:
    regexs = [re.compile(_) for _ in models_regexs[model_name]]
    model = {}
    for tensor_name in all_tensors:
      if in_res(tensor_name, regexs):
        model[tensor_name] = all_tensors[tensor_name]
    models[model_name] = model

  return models


def extract_models_to_pickle(checkpoint_file, models_regexs, save_path):
  """
  Store models returned by extract models to pickle file.
  """
  models = extract_models(
    checkpoint_file = checkpoint_file,
    models_regexs = models_regexs
  )
  for model_name in models:
    model = models[model_name]
    if len(model.keys()) == 0:
      print('Warning: model `{}` is empty'.format(model_name))
    else:
      with open(save_path + model_name +'.pkl', 'wb') as fid:
        pickle.dump(model, fid, pickle.HIGHEST_PROTOCOL)
      print('Save model `{}` to {}'.format(
        model_name, save_path + model_name + '.pkl')
      )


def extract_net(checkpoint_file, net_name, save_path):
  models_regexs = {
    net_name + '_model': [
      '^{}/.*(?<!_average)$'.format(net_name)
    ],
    net_name + '_smooth_model': [
      '^alexNet/.*(?<=_average)$'.format(net_name)
    ]
  }
  extract_models_to_pickle(
    checkpoint_file = checkpoint_file,
    models_regexs = models_regexs,
    save_path = save_path
  )


if __name__ == '__main__':
  if FLAGS.act == 'print':
    print_all_tensors(
      checkpoint_file = FLAGS.checkpoint_file
    )
  elif FLAGS.act == 'extract':
    extract_net(
      checkpoint_file = FLAGS.checkpoint_file,
      net_name = FLAGS.net_name,
      save_path = FLAGS.save_path
    )
  else:
    print('Unknow action')
