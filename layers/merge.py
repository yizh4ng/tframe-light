import tensorflow as tf
from tframev2.layers.layer import Layer



class Merge(Layer):

  PROD = 'prod'
  SUM = 'sum'
  CONCAT = 'concat'
  CROSS_CONCAT = 'cross-concat'
  CONCAT_SUM = 'concat-sum'
  HIGHWAY = 'highway'

  def __init__(self, merge_method, **kwargs):
    """This layer class provides some build-in merge method, including
    those listed in the class variables with capitalized names. When using
    `CONCAT_SUM` method, one needs to specify `sum_indices`. Other tensors
    will be concatenated first and added to those within `sum_indices`.
    Currently a more general design is not needed."""
    self.merge_method = merge_method
    self._axis = kwargs.get('axis', -1)
    self._sum_indices = kwargs.get('sum_indices', (0,))
    self.max_trim = kwargs.get('max_trim', 0)
    # Store other keyword arguments
    self.kwargs = kwargs
    self._trainable_parameters = []

  def _link(self, *input_list, **kwargs):
    # Check input_list
    assert len(input_list) > 0
    if len(input_list) == 1: input_list = input_list[0]
    if not (isinstance(input_list, (list, tuple)) and len(input_list) > 1):
      raise ValueError('!! Illegal input tensors flow into merge layer.')

    # Slice if necessary
    # input_list = self._check_input_list(input_list)

    # Merge according to specification
    if self.merge_method == self.SUM:
      return tf.add_n(input_list)
    elif self.merge_method == self.CONCAT:
      return tf.concat(input_list, axis=self._axis)
    elif self.merge_method == self.CROSS_CONCAT:
      assert len(input_list) == 2
      x: tf.Tensor = input_list[0]
      y: tf.Tensor = input_list[1]
      assert x.shape.as_list() == y.shape.as_list()
      xy = tf.multiply(x, y, name='cross')
      return tf.concat([x, y, xy], axis=self._axis)
    elif self.merge_method == self.PROD:
      output = input_list.pop()
      for tensor in input_list: output *= tensor
      return output
    elif self.merge_method == self.CONCAT_SUM:
      assert len(input_list) > 2
      assert 0 < len(self._sum_indices) <= len(input_list) - 2
      y = tf.concat([x for i, x in enumerate(input_list)
                     if i not in self._sum_indices], axis=self._axis)
      inputs = [x for i, x in enumerate(input_list) if i in self._sum_indices]
      inputs.append(y)
      return tf.add_n(inputs)
    elif self.merge_method == self.HIGHWAY:
      assert len(input_list) == 3
      x, x_bar, gate = input_list
      y = tf.multiply(gate, x) + tf.multiply(1. - gate, x_bar)
      return y
    else: raise KeyError('!! Unknown merge method {}'.format(self.merge_method))

  def _check_input_list(self, input_list):
    # Make sure each input has the same shape length
    shapes = [x.shape.as_list() for x in input_list]
    if not all([len(shape) == len(shapes[0]) for shape in shapes]):
      raise AssertionError('!! tensors to be merged must have a same rank')

    # TODO: some more checks should be done
    if self.merge_method not in (self.SUM, self.PROD): return input_list
    dims = [shape[self._axis] for shape in shapes]
    min_dim = min(dims)
    deltas = [d - min_dim for d in dims]
    if not any(deltas): return input_list

    # Try to automatically truncate overlong tensors
    if max(deltas) > self.max_trim: raise ValueError(
      '!! Failed to merge tensors because of unequal dimension of the'
      ' corresponding axis which can not be truncated automatically.')

    # Truncate overlong tensors
    begin, size = [[i] * len(shapes[0]) for i in (0, 1)]
    size[self._axis] = min_dim
    for i, delta in enumerate(deltas):
      if delta == 0: continue
      input_list[i] =tf.slice(input_list[i], begin, size)

    self.full_name += '[t]'
    return input_list

  @classmethod
  def Sum(cls, **kwargs):
    return Merge(cls.SUM, **kwargs)

  @classmethod
  def Prod(cls, **kwargs):
    return Merge(cls.PROD, **kwargs)

  @classmethod
  def Concat(cls, axis=-1, **kwargs):
    return Merge(cls.CONCAT, axis=axis, **kwargs)

  @classmethod
  def CrossConcat(cls, axis=-1, **kwargs):
    return Merge(cls.CROSS_CONCAT, axis=axis, **kwargs)

  @classmethod
  def ConcatSum(cls, sum_indices=(0,), **kwargs):
    return Merge(cls.CONCAT_SUM, sum_indices=sum_indices, **kwargs)

  @classmethod
  def Highway(cls, **kwargs):
    return Merge(cls.HIGHWAY, **kwargs)

  @property
  def trainable_parameters(self):
    return  self._trainable_parameters