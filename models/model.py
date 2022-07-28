

class Model(object):
  def __init__(self,loss, metrics, net):
    assert callable(loss)
    assert isinstance(metrics, (list, tuple))
    self.loss = loss
    self.metrics = metrics
    self.net = net





