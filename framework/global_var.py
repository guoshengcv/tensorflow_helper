from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

import tensorflow as tf


__all__ = [
  'get_global_phase',
  'create_global_phase',
  'get_or_create_global_phase'
]


ops.GraphKeys.GLOBAL_PHASE = 'global_phase'


def get_global_phase(graph=None):
  """Get the global phase tensor.

  The global phase tensor must be an bool variable. We first try to find it
  in the collection `GLOBAL_PHASE`, or by name `global_phase:0`.

  Args:
    graph: The graph to find the global phase in. If missing, use default graph.

  Returns:
    The global phase variable, or `None` if none was found.

  Raises:
    TypeError: If the global phase tensor has a bool type, or if it is not
      a `Variable`.
  """
  graph = graph or ops.get_default_graph()
  global_phase_tensor = None
  global_phase_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_PHASE)
  if len(global_phase_tensors) == 1:
    global_phase_tensor = global_phase_tensors[0]
  elif not global_phase_tensors:
    try:
      global_phase_tensor = graph.get_tensor_by_name('global_phase:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_phase collection.')
    return None

  assert_global_phase(global_phase_tensor)
  return global_phase_tensor


def create_global_phase(graph=None):
  """Create global phase tensor in graph.

  Args:
    graph: The graph in which to create the global phase tensor. If missing,
      use default graph.

  Returns:
    Global phase tensor.

  Raises:
    ValueError: if global phase tensor is already defined.
  """
  graph = graph or ops.get_default_graph()
  if get_global_phase(graph) is not None:
    raise ValueError('"global_phase" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    global_phase = tf.placeholder_with_default(
      True,
      shape=[],
      name=ops.GraphKeys.GLOBAL_PHASE)
    graph.add_to_collection(
      name=ops.GraphKeys.GLOBAL_PHASE,
      value=global_phase)
    return global_phase


def get_or_create_global_phase(graph=None):
  """Returns and create (if necessary) the global phase tensor.

  Args:
    graph: The graph in which to create the global phase tensor. If missing, use
      default graph.

  Returns:
    The global phase tensor.
  """
  graph = graph or ops.get_default_graph()
  global_phase_tensor = get_global_phase(graph)
  if global_phase_tensor is None:
    global_phase_tensor = create_global_phase(graph)
  return global_phase_tensor


def assert_global_phase(global_phase_tensor):
  """Asserts `global_phase_tensor` is a scalar bool `Variable` or `Tensor`.

  Args:
    global_phase_tensor: `Tensor` to test.
  """
  if not (isinstance(global_phase_tensor, variables.Variable) or
          isinstance(global_phase_tensor, ops.Tensor) or
          isinstance(global_phase_tensor,
                     resource_variable_ops.ResourceVariable)):
    raise TypeError(
        'Existing "global_phase" must be a Variable or Tensor: {}.'.format(
        global_phase_tensor))

  if not global_phase_tensor.dtype.is_bool:
    raise TypeError('Existing "global_phase" does not have bool type: {}'.format(
                    global_phase_tensor.dtype))

  if global_phase_tensor.get_shape().ndims != 0:
    raise TypeError('Existing "global_phase" is not scalar: {}'.format(
                    global_phase_tensor.get_shape()))
