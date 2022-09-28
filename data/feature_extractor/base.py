import os
import numpy as np
from matplotlib import  pyplot as plt



from tframe.data.dataset import DataSet



class FeatureExtractor(object):
  def __init__(self, name=None):
    self._name = name
    self._default_name = None

  @property
  def default_name(self):
    if self._default_name is None:
      raise NotImplementedError
    else:
      return self._default_name

  @default_name.setter
  def default_name(self, default_name):
    self._default_name = default_name

  @property
  def name(self):
    if self._name is None:
      return self.default_name
    else:
      return self._name

  def extract(self, features):
    raise NotImplementedError

  def __call__(self, data_set:DataSet, key='features'):
    features = self.extract(data_set[key])
    if key != 'features':
      self._name = '{} of {}'.format(self.name, key)
    data_set.data_dict[self.name] = features
    return features


  def view(self, data_set:DataSet, key):
    raise NotImplementedError


  def view_hist(self, data_set: DataSet, key=None, bins_num=20, save_path=None):
    if key is not None:
      groups = data_set.sub_groups(key)
      features = data_set.data_dict[self.name]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      for group in groups:
        assert self.name in group.data_dict.keys()
        label = group.properties[key][0]
        plt.hist(group.data_dict[self.name], self.bins, alpha=0.5, label=label)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, self.name + ' '.join(
          set(data_set.properties[key]))))
      plt.show()
    else:
      features = data_set.data_dict[self.name]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      plt.hist(features, self.bins, alpha=0.5)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, self.name))
      plt.show()
