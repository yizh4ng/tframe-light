import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
from collections import defaultdict



class Quantity():
  def __init__(self, name, smaller_is_better):
    self.name = name
    self._smaller_is_better = smaller_is_better
    self._record_appears = defaultdict(lambda:False, {})
    if self.smaller_is_better:
      self._record = defaultdict(lambda: np.inf, {})
    else:
      self._record = defaultdict(lambda: -np.inf, {})

  @property
  def smaller_is_better(self):
    return self._smaller_is_better

  @property
  def larger_is_better(self):
    return not self._smaller_is_better

  def try_set_record(self, value, dataset):
    self._record_appears[dataset] = False
    if self.smaller_is_better:
      if value < self._record[dataset]:
        self._record[dataset] = value
        self._record_appears[dataset] = True
    else:
      if value > self._record[dataset]:
        self._record[dataset] = value
        self._record_appears[dataset] = True

  def __call__(self, predictions, targets):
    assert predictions.shape == targets.shape
    result = self.function(predictions, targets)
    return result

  def __add__(self, other_quantity):
    assert isinstance(other_quantity, Quantity)
    assert self.smaller_is_better == other_quantity._smaller_is_better
    new_quantity = Quantity(name=', '.join([self.name + other_quantity.name]),
                            smaller_is_better=self.smaller_is_better)
    def new_function(predictions, targets):
      return self.function(predictions, targets) \
             + other_quantity(predictions, targets)
    new_quantity.function = new_function
    return new_quantity

  def __sub__(self, other_quantity):
    assert isinstance(other_quantity, Quantity)
    assert self.smaller_is_better != other_quantity._smaller_is_better
    new_quantity = Quantity(name=', '.join([self.name + other_quantity.name]),
                            smaller_is_better=self.smaller_is_better)
    def new_function(predictions, targets):
      return self.function(predictions, targets) \
             - other_quantity(predictions, targets)
    new_quantity.function = new_function
    return new_quantity

  def __mul__(self, weight):
    assert isinstance(weight, float)
    new_quantity = Quantity(name='{} * {}'.format(self.name, weight),
                            smaller_is_better=self.smaller_is_better)
    def new_function(predictions, targets):
      return self.function(predictions, targets) * weight
    new_quantity.function = new_function
    return new_quantity

  def function(self, predictions, targets):
    raise NotImplementedError

class MSE(Quantity):
  def __init__(self):
    super(MSE, self).__init__('MSE',True)

  def function(self, predictions, targets):
    return tf.reduce_mean(tf.square(predictions - targets))

class MAE(Quantity):
  def __init__(self):
    super(MAE, self).__init__('MAE', True)

  def function(self, predictions, targets):
    return tf.reduce_mean(tf.abs(predictions - targets))

class CrossEntropy(Quantity):
  def __init__(self):
    super(CrossEntropy, self).__init__('CrossEntropy', True)

  def function(self, predictions, targets):
    return tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=targets,
                                                                logits=predictions)
    # return tf.losses.CategoricalCrossentropy()(predictions, targets)

class Accraucy(Quantity):
  def __init__(self):
    super(Accraucy, self).__init__('Accuracy', False)

  def function(self, predictions, targets):
    return metrics.Accuracy()(tf.argmax(predictions, 1),
                                tf.argmax(targets, 1))

class GlobalBalancedError(Quantity):
  def __init__(self):
    super(GlobalBalancedError, self).__init__('GBE', True)

  def function(self, y_predict, y_true):
    reduce_axis = list(range(1, len(y_true.shape)))

    # Define similarity ratio
    _min = tf.minimum(y_true, y_predict)
    _max = tf.maximum(y_true, y_predict)
    sr = _min / tf.maximum(_max, 1e-6)
    tp = tf.reduce_sum(sr * y_true, axis=reduce_axis)

    # Calculate F1 score
    return 1 - 2 * tp / tf.maximum(
      tf.reduce_sum(y_true + y_predict, axis=reduce_axis), 1e-6)

class WMAE(Quantity):
  def __init__(self, min_weight=0.0001):
    super(WMAE, self).__init__('WMAE', True)
    self.min_weight = min_weight

  def function(self, y_predict, y_true):
    reduce_axis = list(range(1, len(y_true.shape)))
    weights = tf.maximum(y_true, self.min_weight)
    weights = tf.cast(weights, dtype=tf.float32)
    nume = tf.reduce_sum(tf.abs(y_true - y_predict) * weights,
                         axis=reduce_axis)
    deno = tf.reduce_sum(weights, axis=reduce_axis)
    return tf.divide(nume, deno)

class SSIM(Quantity):
  def __init__(self, max_val=1.0):
    super(SSIM, self).__init__('SSIM', False)
    self.max_val = max_val

  def function(self, y_predict, y_true):
    y_true = tf.cast(y_true,tf.float32)
    y_predict = tf.cast(y_predict,tf.float32)
    return tf.image.ssim(y_true, y_predict, max_val=self.max_val)

class PSNR(Quantity):
  def __init__(self, max_val=1.0):
    super(PSNR, self).__init__('PSNR', False)
    self.max_val = max_val

  def function(self, y_predict, y_true):
    y_true = tf.cast(y_true,tf.float32)
    y_predict = tf.cast(y_predict,tf.float32)
    return tf.image.psnr(y_true, y_predict, max_val=self.max_val)
