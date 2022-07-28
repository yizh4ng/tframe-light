

class Model(object):
  def __init__(self,loss, metrics, net):
    assert callable(loss)
    assert isinstance(metrics, (list, tuple))
    self.loss = loss
    self.metrics = metrics
    self.net = net
    self._mark = None

  @property
  def mark(self):
    if self._mark is None:
      return 'default_mark'
    else:
      return self._mark

  @mark.setter
  def mark(self, mark):
    self._mark = mark


