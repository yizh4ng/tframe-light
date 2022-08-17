from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.nets.net import Net


class UNet(Net):

  def __init__(self, filters=16, kernel_size=3, activation='relu', thickness=2,
               use_batchnorm=False, use_bottleneck=True, use_maxpool=False,
               height=4, link_indices=(0,1,2,3), name='Unet', contraction_kernel_size = None,
               expansion_kernel_size = None, **kwargs):
    super(UNet, self).__init__(name, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.thickness = thickness
    self.use_batchnorm = use_batchnorm
    self.use_bottleneck = use_bottleneck
    self.height = height
    self.link_indices = link_indices
    if isinstance(link_indices, str):
      self.link_indices = []
      for i in link_indices.split(','):
        if i == '':continue
        self.link_indices.append(int(i))
    self.use_maxpool = use_maxpool


    self.contraction_kernel_size = (
      self.kernel_size if contraction_kernel_size is None
      else contraction_kernel_size)
    self.expansion_kernel_size = (
      self.kernel_size if expansion_kernel_size is None
      else expansion_kernel_size)

  def Conv2D(self, filters, kernel_size):
    def _conv2d(input):
      output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                        strides=1, padding='same', use_bias=False)(input)
      if self.use_batchnorm:
        output = tf.keras.layers.BatchNormalization()(output)

      output = tf.keras.layers.Activation(self.activation)(output)
      return output
    return _conv2d

  def InputBlock(self, input, filters):
    for _ in range(self.thickness):
      input = self.Conv2D(filters=filters, kernel_size=self.contraction_kernel_size,
                                      )(input)
    return input

  def ContractingPathBlock(self, input, filters):
    if self.use_maxpool:
      down_sampling = tf.keras.layers.MaxPool2D((2, 2), strides=2,
                                                padding='same')(input)
    else:
      down_sampling = tf.keras.layers.Conv2D(filters=filters/2,
                                             kernel_size=self.contraction_kernel_size,
                                             strides=2, padding='same', use_bias=False)(input)
    for _ in range(self.thickness):
      down_sampling = self.Conv2D(filters=filters,
                                  kernel_size=self.contraction_kernel_size
                                  )(down_sampling)
    return down_sampling

  def ExpansivePathBlock(self, input, con_feature, filters):
    upsampling = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=self.expansion_kernel_size,
                                                 strides=2,
                                                 padding='same', use_bias=False)(
      input)
    if con_feature is not None:
      concat_feature = tf.concat([con_feature, upsampling],
                                 axis=3)
      if self.use_bottleneck:
        concat_feature = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                        strides=1, padding='same', use_bias=False)(concat_feature)
    else:
      concat_feature = upsampling
    # concat_feature = upsampling

    if self.use_batchnorm:
      concat_feature = tf.keras.layers.BatchNormalization()(concat_feature)

    for _ in range(self.thickness):
      concat_feature = self.Conv2D(filters=filters,
                                  kernel_size=self.expansion_kernel_size
                                  )(concat_feature)
    return concat_feature

  def _link(self, *input):
    if len(input) == 1:
      input = input[0]
    else:
      raise SyntaxError('!! Too much inputs')

    filters = self.filters
    output = self.InputBlock(input, filters=filters)
    filters *=2

    features = [output]

    for i in range(self.height):
      output = self.ContractingPathBlock(output, filters)
      filters *= 2
      features.append(output)

    features.pop(-1)

    features.reverse()
    filters /= 2
    for i in range(self.height):
      filters = int(filters/2)
      if i in self.link_indices:
        output = self.ExpansivePathBlock(output, features[i], filters)
        # output = self.ExpansivePathBlock(output, None, filters)
      else:
        output = self.ExpansivePathBlock(output, None, filters)

    return output

