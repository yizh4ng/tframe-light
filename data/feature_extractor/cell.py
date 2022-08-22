import os

from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import numpy as np
from .base import FeatureExtractor
from roma import console
from tframe.data.dataset import DataSet

# TODO: Reorganize the histogram visualization
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

class SphereError(FeatureExtractor):
  def __init__(self, threshold, name=None, bins=None):
    super(SphereError, self).__init__(name=name)
    self.threshold = threshold
    self.bins = bins
    # self.bins = np.linspace(0, 1, 20)
    # if bins is not None:
    #   self.bins = bins

  def __str__(self):
    return 'sphere error'.format(self.threshold)

  def sphereFit(self, x, y, z):
    # A, a, b, C
    def func(v):
      return (x - v[1]) ** 2 / v[0] ** 2 + (y - v[2]) ** 2 / v[0] ** 2 + z ** 2 / v[3] ** 2 - 1

    result = leastsq(func, np.array([75, 50, 50, 7]))[0]
    result = np.mean(np.abs(func(result)))

    # residual = func(result[0])

    return result

  def extract(self, imgs):
    #TODO: Not using for loop
    result = []
    console.show_status('Calculating sphere error...')
    for i, img in enumerate(imgs):
      indices = np.meshgrid(*[range(s) for s in img.shape], indexing='ij')
      indices = tuple([v.ravel() for v in indices])
      indices_with_height = np.transpose(np.array(indices + (img.ravel(), )))
      indices_with_height_filtered = np.transpose(indices_with_height[indices_with_height[:, -1] > self.threshold])
      residual = self.sphereFit(*indices_with_height_filtered)
      result.append(residual)
      console.print_progress(i, len(imgs))
    return np.array(result)