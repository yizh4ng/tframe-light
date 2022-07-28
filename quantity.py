import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics



class Quantity():
  def __init__(self, name, smaller_is_better):
    self.name = name
    self._smaller_is_better = smaller_is_better

  @property
  def smaller_is_better(self):
    return self.smaller_is_better

  @property
  def larger_is_better(self):
    return not self.smaller_is_better

  def __call__(self, predictions, targets):
    assert predictions.shape == targets.shape
    return self.function(predictions, targets)

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
