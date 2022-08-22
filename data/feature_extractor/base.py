from tframe.data.dataset import DataSet
import numpy



class FeatureExtractor(object):
  def __init__(self, name=None):
    self.name = name

  def extract(self, features):
    raise NotImplementedError

  def __call__(self, data_set:DataSet):
    data_set.data_dict[str(self)] = self.extract(data_set.features)


  def view(self, data_set:DataSet, key):
    raise NotImplementedError


  def __str__(self):
    if self.name is not None:
      return self.name
    else:
      raise NotImplementedError
