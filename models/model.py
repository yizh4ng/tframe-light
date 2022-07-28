from tensorflow import keras
import tensorflow as tf
from tframe import console

class Model(object):
  def __init__(self,loss, metrics, net):
    assert callable(loss)
    assert isinstance(metrics, (list, tuple))
    self.loss = loss
    self.metrics = metrics
    self.net = net
    self._mark = None
    self.keras_model = None

  @property
  def mark(self):
    if self._mark is None:
      return 'default_mark'
    else:
      return self._mark

  @mark.setter
  def mark(self, mark):
    self._mark = mark

  def build(self, input_shape):
    input = tf.keras.layers.Input(input_shape)
    output = self.net(input)
    self.keras_model = keras.Model(inputs=input, outputs=output,
                                   name=self.net.name)




