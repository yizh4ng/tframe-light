from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.ndimage import map_coordinates

import tframe as tfr
from tframe.utils.arg_parser import Parser
from tframe.data.dataset import DataSet

from typing import Optional


def image_augmentation_processor(aug_config,
    data_batch: DataSet, is_training: bool, proceed_target: bool = False):
  # Get hub
  if not is_training or aug_config is None: return data_batch
  # Parse augmentation setting
  assert isinstance(aug_config, str)
  if aug_config in ('-', 'x'): return data_batch
  configs = [Parser.parse(s) for s in aug_config.split('|')]
  if len(configs) == 0: return data_batch

  # Apply each method according to configs
  for cfg in configs:
    # Find method
    if cfg.name == 'rotate': method = _rotate
    elif cfg.name == 'flip': method = _flip
    else: raise KeyError('!! Unknown augmentation option {}'.format(cfg.name))
    # Do augmentation
    if proceed_target:
      data_batch.features, data_batch.targets = method(
        data_batch.features, data_batch.targets, *cfg.arg_list, **cfg.arg_dict)
    else: data_batch.features = method(
      data_batch.features, *cfg.arg_list, **cfg.arg_dict)

  return data_batch


"""Currently this method works only for channel-last format.
   That is, the H and W dim of x correspond to x.shape[1] and x.shape[2].
   This applies to all the methods below.
"""

def _rotate(x: np.ndarray, y: Optional[np.ndarray] = None, bg=0):
  # Check x shape
  assert x.shape[1] == x.shape[2]
  # Decide k
  k = np.random.choice(4, 1)[0]
  # Rotate x
  x = np.rot90(x, k, axes=[1, 2])

  if y is not None:
    assert y.shape[1] == y.shape[2]
    y = np.rot90(y, k, axes=[1, 2])
    return x, y

  return x


def _flip(x: np.ndarray, horizontal=True, vertical=True, p=0.5,
          y: Optional[np.ndarray] = None):
  """Randomly flip image batch.

  :param x: images with shape (batch_size, H, W[, C])
  :param horizontal: whether to do flip horizontally
  :param vertical: whether to do flip vertically
  :param p: probability to do flip
  :return: flipped image batch
  """
  assert 0 < p < 1 and any([horizontal, vertical])

  def _rand_flip(axis):
    mask = np.random.choice([True, False], size=x.shape[0], p=[p, 1 - p])
    x[mask] = np.flip(x[mask], axis=axis)
    if y is not None: y[mask] = np.flip(y[mask], axis=axis)

  if horizontal: _rand_flip(2)
  if vertical: _rand_flip(1)

  if y is not None: return x, y
  return x

def alter(data_batch:DataSet, is_training:bool, mode:str):
  if is_training is False:
    return data_batch
  assert mode in ('lr', 'ud')

  features = data_batch.features
  assert isinstance(features, np.ndarray)

  new_features = None
  if mode == 'ud':
    random_index = np.random.randint(0, features.shape[1])
    new_features = np.concatenate(
      (features[:,random_index:], features[:,:random_index]), axis=1)

  elif mode == 'lr':
    random_index = np.random.randint(0, features.shape[2])
    new_features = np.concatenate(
      (features[:,:,random_index:], features[:,:,:random_index]), axis=2)

  assert new_features is not None

  data_batch.features = new_features
  return data_batch

def rotagram(data_batch, is_training):
  features = data_batch.features
  assert len(features.shape) == 4
  sampling_num = min(features.shape[1:2])

  new_features = []
  def map_images_values(imgs, x:np.ndarray, y:np.ndarray, sampling_num, mode='scipy'):
    assert mode in ('numpy', 'scipy')
    x0, x1 = x
    y0, y1 = y
    if mode == 'numpy':
      x, y = np.linspace(x0, y0, sampling_num).astype(int), \
             np.linspace(x1, y1, sampling_num).astype(int)
      x = np.clip(x, 0, imgs.shape[1] - 1)
      y = np.clip(y, 0, imgs.shape[2] - 1)
      zi = imgs[:,x, y,:]
      return zi
    elif mode == 'scipy':
      x, y = np.linspace(x0, y0, sampling_num), \
             np.linspace(x1, y1, sampling_num)
      x = np.clip(x, 0, imgs.shape[1] - 1)
      y = np.clip(y, 0, imgs.shape[2] - 1)
      new_imgs = []
      for img in imgs:
        new_img = map_coordinates(img[:,:,0], np.array([x, y]), order=3)
        new_imgs.append(new_img)
      zi = np.expand_dims(new_imgs, -1)
      return zi

  def index2coor(index, sample_num):
    angle = 2 * np.pi * index / sample_num
    return np.array([np.sin(angle) * int(min(features.shape[1:2])/2),
                     np.cos(angle) * int(min(features.shape[1:2])/2)])


  origin = np.array([(features.shape[1]) / 2, (features.shape[2]) / 2])
  for i in range(sampling_num):
    coordinate = (index2coor(i, sampling_num) + origin)
    # print((origin[0] - coordinate[0]) ** 2 + (origin[1] - coordinate[1]) ** 2)
    new_features.append(map_images_values(features,
                                          origin, coordinate, sampling_num))
  # return np.expand_dims(np.concatenate(new_features, axis=2), -1)

  data_batch.features = np.expand_dims(np.concatenate(new_features, axis=2), -1)
  return data_batch

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  x, y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
  z = np.sqrt(x ** 2 + y ** 2) + np.sin(x ** 2 + y ** 2)
  features = np.expand_dims(z,(0, -1))

  data_batch = DataSet(features=features)
  plt.imshow(data_batch.features[0])
  plt.show()

  data_batch = alter(data_batch, is_training=True, mode='lr')
  plt.imshow(data_batch.features[0])
  plt.show()
