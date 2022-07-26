import tensorflow as tf
from tframe.layers.layer import Layer


def single_input(_link):

  def wrapper(*args, **kwargs):
    # Currently not sure if the decorator is for class method only
    input_ = args[1] if isinstance(args[0], (Layer)) else args[0]
    if isinstance(input_, (tuple, list)):
      if len(input_) != 1:
        raise ValueError('!! This layer only accept single input')
      input_ = input_[0]


    args = (args[0], input_) if isinstance(args[0], Layer) else (input_,)

    return _link(args[0], input_, *kwargs)

  return wrapper


class Flatten(Layer):

  def __init__(self, *args, **kwargs):
    self.function = tf.keras.layers.Flatten(*args, **kwargs)
  @single_input
  def _link(self, input, **kwargs):
    return  self.function(input)

  @property
  def trainable_variables(self):
    return self.function.trainable_variables

