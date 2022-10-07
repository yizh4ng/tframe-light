import numpy as np
import tensorflow.keras as keras
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


  def reset_weights(self, layer_index):
    assert isinstance(self.keras_model, keras.Model)
    model = self.keras_model
    ix = layer_index
    new_weights = []
    if hasattr(model.layers[ix], 'kernel_initializer'):
      weight_initializer = model.layers[ix].kernel_initializer
      new_weights.append(weight_initializer(
        model.layers[ix].get_weights()[0].shape))
    if hasattr(model.layers[ix], 'bias_initializer'):
      bias_initializer = model.layers[ix].bias_initializer
      new_weights.append(bias_initializer(
        model.layers[ix].get_weights()[1].shape))

      model.layers[ix].set_weights(new_weights)

  def show_feauture_maps(self, dataset:DataSet, class_key=None):
    assert isinstance(self.keras_model, tf.keras.Model)
    da = DaVinci()

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
              axis=(-1))
          )
          titles.append('{} {}'.format(layer.name, i))
        da.register_bookmarks(len(feature_maps))
      # dealing with dense layers
      elif len(layer.output.shape) == 2:
        feature_maps.append(tf.expand_dims(layer.output, -1))
        titles.append('{}'.format(layer.name))
        da.register_bookmarks(len(feature_maps))
      else:
        console.show_status(
          'Unknow layer types {} or shapes {}'.format(type(layer),
                                                      layer.output.shape))

    da.objects = dataset.features

    model_temp = tf.keras.Model(inputs=self.keras_model.input,
                                outputs=feature_maps)

    outputs = model_temp(dataset.features)
    da.objects = outputs
    #object_cursor for deep learning model layers, layer_curosr for data index

    def _show_feature_map(x, data_index):
      title = titles[da.object_cursor]
      if class_key is not None:
        title += ' {}'.format(dataset.properties[class_key][data_index])

      da.imshow_pro(x[data_index], title=title)

    # da.add_plotter(show_raw)
    for i in range(len(dataset)):
      da.add_plotter(lambda x, _i = i: _show_feature_map(x, _i))
    da.show()