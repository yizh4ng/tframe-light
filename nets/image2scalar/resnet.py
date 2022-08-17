import tensorflow as tf

from tframe.nets.net import Net


class ResNet(Net):

  def __init__(self, filters=16, kernel_size=3, block_repetitions=(1,2,1),
               dense_layers = (), activation='relu', name='resnet', **kwargs):
    super(ResNet, self).__init__(name, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    if isinstance(block_repetitions, str):
      block_repetitions = [int(s) for s in block_repetitions.split('-')]
    self.block_repetitions = block_repetitions
    self.activation = activation
    self.dense_layers = dense_layers

  def Conv(self, filters, stride=1):
    return lambda input : tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=self.kernel_size,
                                                 strides=stride,
                                                 padding='same',
                                                 use_bias=True)(input)

  def residual_block(self, filters, stride=1):
    def residual_block(input):
      output = self.Conv(filters, stride)(input)
      output = tf.keras.layers.BatchNormalization()(output)
      output = tf.keras.layers.Activation(self.activation)(output)
      output = self.Conv(filters)(output)
      output = tf.keras.layers.BatchNormalization()(output)
      output = tf.keras.layers.Activation(self.activation)(output)
      if stride == 1:
        output = output + input
      else:
        output = output + tf.keras.layers.Conv2D(filters, kernel_size=1,
                                                 strides=2, padding='same')(input)
      return output
    return lambda input: residual_block(input)

  def _link(self, *input):
    if len(input) == 1:
      input = input[0]
    else:
      raise SyntaxError('!! Too much inputs')

    filters = self.filters

    # Construct The first Contracting Block
    output = self.Conv(filters, stride=2)(input)

    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Activation(self.activation)(output)

    output = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(output)

    # Adding Residual Blocks
    for i, block_repetition in enumerate(self.block_repetitions):
      for r in range(int(block_repetition)):
        # The first residual block should not contract
        if i == 0 and r == 0:
          output = self.residual_block(filters, stride=1)(output)
        elif r == 0:
          output = self.residual_block(filters, stride=2)(output)
        else:
          output = self.residual_block(filters, stride=1)(output)
      filters *= 2

    output = tf.keras.layers.GlobalAvgPool2D()(output)
    output = tf.keras.layers.Flatten()(output)

    for i in self.dense_layers:
      output = tf.keras.layers.Dense(units=i, activation=self.activation,
                                     use_bias=False)(output)

    return output

