import os

from matplotlib import pyplot as plt
import numpy as np
from .base import FeatureExtractor
from tframe.data.dataset import DataSet


class RegionSize(FeatureExtractor):
  def __init__(self, threshold, name=None, bins=None):
    super(RegionSize, self).__init__(name=name)
    self.threshold = threshold
    self.bins = bins
  def __str__(self):
    return 'region size above {}'.format(self.threshold)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.threshold] = 1
    return np.sum(mask, axis=(1, 2)) / np.prod(img_size[1:])

  def view(self, data_set:DataSet, key=None, bins_num=20, save_path=None):
    if key is not None:
      groups = data_set.sub_groups(key)
      features = data_set.data_dict[str(self)]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      for group in groups:
        assert str(self) in group.data_dict.keys()
        label = group.properties[key][0]
        plt.hist(group.data_dict[str(self)], self.bins, alpha=0.5, label=label)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, str(self) + ' ' + ' '.join(
          set(data_set.properties[key]))))
      plt.show()
    else:
      features = data_set.data_dict[str(self)]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      plt.hist(features, self.bins, alpha=0.5)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, str(self)))
      plt.show()

class RegionMask(FeatureExtractor):
  def __init__(self, threshold, name=None,):
    super(RegionMask, self).__init__(name=name)
    self.threshold = threshold

  def __str__(self):
    return 'region mask above {}'.format(self.threshold)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.threshold] = 1
    return mask

class RegionIntegrate(FeatureExtractor):
  def __init__(self, threshold, name=None, bins=None):
    super(RegionIntegrate, self).__init__(name=name)
    self.threshold = threshold
    self.bins = bins

  def __str__(self):
    return 'region integrate above {}'.format(self.threshold)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.threshold] = imgs[imgs > self.threshold]
    return np.sum(mask, axis=(1, 2))

  def view(self, data_set:DataSet, key=None, bins_num=20, save_path=None):
    if key is not None:
      groups = data_set.sub_groups(key)
      features = data_set.data_dict[str(self)]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      for group in groups:
        assert str(self) in group.data_dict.keys()
        label = group.properties[key][0]
        plt.hist(group.data_dict[str(self)], self.bins, alpha=0.5, label=label)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, str(self) + ' '.join(set(data_set.properties[key]))))
      plt.show()
    else:
      features = data_set.data_dict[str(self)]
      if self.bins is None:
        self.bins = np.linspace(np.min(features), np.max(features), bins_num)
      plt.hist(features, self.bins, alpha=0.5)
      plt.xlabel(self.name)
      plt.ylabel('count')
      plt.legend(loc='upper right')
      if save_path is not None:
        plt.savefig(os.path.join(save_path, str(self)))
      plt.show()
