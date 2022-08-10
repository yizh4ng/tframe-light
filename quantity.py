import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics



class Quantity():
  def __init__(self, name, smaller_is_better):
    self.name = name
    self._smaller_is_better = smaller_is_better
    if self._smaller_is_better:
      self.record = np.inf
    else:
      self.record = -np.inf
    self._record_appears = False

  @property
  def smaller_is_better(self):
    return self._smaller_is_better

  @property
  def larger_is_better(self):
    return not self._smaller_is_better

  @property
  def record_appears(self):
    return self._record_appears

  def try_set_record(self, value):
    self._record_appears = False
    if self.smaller_is_better:
      if value < self.record:
        self.record = value
        self._record_appears = True
    else:
      if value > self.record:
        self.record = value
        self._record_appears = True

  def __call__(self, predictions, targets):
    assert predictions.shape == targets.shape
    result = self.function(predictions, targets)
    return result


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
    return tf.losses.CategoricalCrossentropy()(predictions, targets)


class Accraucy(Quantity):
  def __init__(self):
    super(Accraucy, self).__init__('Accraucy', False)

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
