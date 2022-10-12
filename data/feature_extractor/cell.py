import os

from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import numpy as np
from .base import FeatureExtractor
from roma import console
from tframe.data.dataset import DataSet



class RegionSize(FeatureExtractor):
  def __init__(self, low=np.inf, high=-np.inf, name=None, bins=None):
    super(RegionSize, self).__init__(name=name)
    self.low = low
    self.high = high
    self.bins = bins
    self.default_name = 'region size above {} below {}.'.format(self.low,
                                                                self.high)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.low] = 1
    mask[imgs < self.high] = 1
    return np.sum(mask, axis=(1, 2)) / np.prod(img_size[1:])

class Variance(FeatureExtractor):
  def __init__(self, name=None, bins=None):
    super(Variance, self).__init__(name=name)
    self.bins = bins
    self.default_name = 'Variance'

  def extract(self, imgs):
    return np.std(imgs, axis=(1, 2))

class Maximum(FeatureExtractor):
  def __init__(self, name=None, bins=None):
    super(Maximum, self).__init__(name=name)
    self.bins = bins
    self.default_name = 'Maximum'

  def extract(self, imgs):
    return np.max(imgs, axis=(1, 2))

class MaximumGap(FeatureExtractor):
  def __init__(self, step, name=None, bins=None):
    super(MaximumGap, self).__init__(name=name)
    self.bins = bins
    self.default_name = 'MaximumGap with step {}.'.format(step)
    self.step = step

  def extract(self, imgs):
    W, H = imgs.shape[1], imgs.shape[2]
    step = self.step
    x_difference = np.abs(imgs[:, step:,:H-step,:] - imgs[:, :W-step,:H-step,:])
    y_difference = np.abs(imgs[:,:W-step, step:,:] - imgs[:, :W-step, :H-step, :])
    max_difference = np.maximum(x_difference, y_difference)
    return np.max(max_difference, axis=(1, 2))

class RegionMask(FeatureExtractor):
  def __init__(self, threshold, name=None,):
    super(RegionMask, self).__init__(name=name)
    self.threshold = threshold
    self.default_name = 'region mask above {}'.format(self.threshold)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.threshold] = 1
    return mask

class Crop(FeatureExtractor):
  def __init__(self, x_range=None, y_range=None, name=None, bins=None):
    super(Crop, self).__init__(name=name)
    self.x_range = x_range
    self.y_range = y_range
    self.bins = bins
    self.default_name = 'Crop region'

  def extract(self, imgs):
    if self.x_range is not None:
      x_range = self.x_range
      assert len(x_range) == 2
      imgs = imgs[:, x_range[0]: x_range[1],:, :]
    if self.y_range is not None:
      y_range = self.y_range
      assert len(y_range) == 2
      imgs = imgs[:, :, y_range[0]: y_range[1], :]
    return imgs


class RegionVariance(FeatureExtractor):
  def __init__(self, low=np.inf, high=-np.inf, name=None, bins=None):
    super(RegionVariance, self).__init__(name=name)
    self.low = low
    self.high = high
    self.bins = bins
    self.default_name = 'region variance above {} and below'.format(self.low,
                                                                    self.high)

  def extract(self, imgs):
    variance = []
    for img in imgs:
      _img = np.zeros(img.shape)
      _img[img > self.low] = img[img > self.low]
      _img[img < self.high] = img[img < self.high]
      variance.append([np.std(_img)])
    return np.array(variance)

class RegionIntegrate(FeatureExtractor):
  def __init__(self, low=np.inf, high=-np.inf, name=None, bins=None):
    super(RegionIntegrate, self).__init__(name=name)
    self.low = low
    self.high = high
    self.bins = bins
    self.default_name = 'region integrate from {} to {}'\
      .format(self.low, self.high)

  def extract(self, imgs):
    img_size = imgs.shape
    mask = np.zeros(img_size)
    mask[imgs > self.low] = imgs[imgs > self.low]
    mask[imgs < self.high] = imgs[imgs < self.high]
    return np.sum(mask, axis=(1, 2))

class SphereError(FeatureExtractor):
  def __init__(self, threshold, name=None, bins=None):
    super(SphereError, self).__init__(name=name)
    self.threshold = threshold
    self.bins = bins
    self.default_name = 'sphere error'.format(self.threshold)
    # self.bins = np.linspace(0, 1, 20)
    # if bins is not None:
    #   self.bins = bins

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
    if len(imgs.shape) == 4:
      imgs = np.sum(imgs, axis=-1)
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