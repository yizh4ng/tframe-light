from tframev2.core.function import Function



class Layer(Function):

  @property
  def trainable_parameters(self):
    raise NotImplementedError