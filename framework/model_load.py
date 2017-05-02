from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging as logging

from tensorflow_helper.utils.string_utils import in_res

import re
try:
  import cPickle as pickle
except ImportError:
  import pickle


__all__ = [
  'model_load',
  'model_load_from_pickle'
]


def model_load(load_dict, data_dict, reshape=False):
  """
  Args:
    load_dict: {'cur_var_1': 'pre_var_1', 'cur_var_2': 'pre_var_2'}
    data_dict: {'pre_var_1': np_ndarray_1, 'pre_var_2': np_ndarray_2}
  Returns:
    init_fn: a callable passed to scaffold.init_fn to load data to tensor
  """
  var_load_ops = []
  vars_loaded = []
  load_dict_keys = set(load_dict.keys())
  for var in tf.global_variables():
    if var.name in load_dict_keys:
      data = tf.convert_to_tensor(
        value = data_dict[load_dict[var.name]],
        dtype = var.value().dtype
      )
      if reshape:
        data = tf.reshape(
          tensor = data,
          shape = var.shape
        )
      var_load_ops.append(
        var.assign(
          value = data
        )
      )
      vars_loaded.append(
        var.name
      )
  load_op = tf.group(*var_load_ops)

  def init_fn(scaffold, sess):
    # Log variables to be loaded from pretrained model
    log_format = 'Load variable <{}> from pretrained model.'
    for var_name in vars_loaded:
      logging.info(
        log_format.format(var_name)
      )
    # Run session to load
    sess.run(load_op)
  return init_fn


def model_load_from_pickle(data_path, skip_regexs=[], reshape=False):
  """
  Args:
    data_path: str, the path of pickle file which stores pretrained model.
    skip_regexs: a list of str which will be compiled as regular expression 
              pattern. (key, np.ndarray) item in `pickle file` will be skiped.
  Returns:
    init_fn: a callable passed to scaffold.init_fn to load data to tensor
  """
  with open(data_path, 'rb') as fid:
    data_dict = pickle.load(fid)
  regexs = [re.compile(_) for _ in skip_regexs]
  load_dict = {}
  for k in data_dict.keys():
    if not in_res(k, regexs):
      load_dict[k + ':0'] = k
  return model_load(load_dict, data_dict, reshape)
