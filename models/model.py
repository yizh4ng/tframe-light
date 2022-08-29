import numpy as np
from tensorflow import keras
import tensorflow as tf
from tframe import console, DataSet
from lambo.gui.vinci.vinci import DaVinci


class Model(object):
  def __init__(self,loss, metrics, net, name='DefaultModel'):
    assert callable(loss)
    assert isinstance(metrics, (list, tuple))
    self.loss = loss
    self.metrics = metrics
    self.net = net
    self._mark = None
    self.keras_model = None
    self.name = name

  @property
  def mark(self):
    if self._mark is None:
      return 'default_mark'
    else:
      return self._mark

  @mark.setter
  def mark(self, mark):
    self._mark = mark

  @property
  def num_of_parameters(self):
    trainableParams = int(np.sum(
      [np.prod(v.get_shape()) for v in self.keras_model.trainable_weights]))
    nonTrainableParams = int(np.sum(
      [np.prod(v.get_shape()) for v in self.keras_model.non_trainable_weights]))

    total_paras = trainableParams + nonTrainableParams

    return total_paras

  @tf.function
  def link(self, input):
    return self.net(input)

  def build(self, input_shape):
    input = tf.keras.layers.Input(input_shape)
    output = self.net(input)
    self.keras_model = keras.Model(inputs=input, outputs=output,
                                   name=self.net.name)

  def show_feauture_maps(self, dataset:DataSet):
    assert isinstance(self.keras_model, tf.keras.Model)
    feature_maps = []
    titles = []
    for layer in self.keras_model.layers:
      # dealing with images
      if len(layer.output.shape) == 4:
        for i in range(layer.output.shape[-1]):
          feature_maps.append(
            tf.reduce_sum(
              tf.slice(
                layer.output, [0,0,0,i], [-1,-1,-1,1]),
              axis=(0, -1))
          )
          titles.append('{} {}'.format(layer.name, i))
      # dealing with dense layers
      elif len(layer.output.shape) == 2:
        feature_maps.append(layer.output)
        titles.append('{}'.format(layer.name))
      else:
        console.show_status(
          'Unknow layer types {} or shapes {}'.format(type(layer),
                                                      layer.output.shape))
    da = DaVinci()
    da.objects = dataset.features

    model_temp = tf.keras.Model(inputs=self.keras_model.input,
                                outputs=feature_maps)

    def _show_feature_map(x, feature_index):
      da.imshow(model_temp(np.expand_dims(x, 0))[feature_index], title=titles[feature_index])

    da.add_plotter(da.imshow)
    for i in range(len(feature_maps)):
      da.add_plotter(lambda x, _i = i: _show_feature_map(x, _i))
    da.show()