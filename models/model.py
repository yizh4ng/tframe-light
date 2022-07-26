

class Model(object):
  def __init__(self,loss_function, metrics, net):

    self.loss_function = loss_function
    self.metrics = metrics
    self.net = net




