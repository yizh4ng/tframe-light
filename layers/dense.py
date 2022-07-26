import tensorflow as tf
from tframev2.layers.layer import Layer
from tframev2.layers.common import single_input

class Dense(Layer):

  def __init__(self, *args, **kwargs):
    self.function = tf.keras.layers.Dense(*args, **kwargs)

  @single_input
  def _link(self, input, **kwargs):
    return  self.function(input)

  @property
  def trainable_variables(self):
    return self.function.trainable_variables

