import numpy as np
from scipy.ndimage import map_coordinates
from roma import console


def centralize(features):
  x = features
  assert len(x.shape) == 4
  shape = [len(x), 1, 1, x.shape[3]]
  mu = np.mean(x, axis=(1, 2)).reshape(shape)
  return features - mu

def normalize(features):
  x = centralize(features)
  assert len(x.shape) == 4
  shape = [len(x), 1, 1, x.shape[3]]
  sigma = np.std(x, axis=(1, 2)).reshape(shape)
  return  x / sigma

def rotagram(features, verbose=False):
  assert len(features.shape) == 4
  sampling_num = min(features.shape[1:2])

  new_features = []
  def map_images_values(imgs, x:np.ndarray, y:np.ndarray, sampling_num, mode='scipy'):
    assert mode in ('numpy', 'scipy')
    x0, x1 = x
    y0, y1 = y
    if mode == 'numpy':
      x, y = np.linspace(x0, y0, sampling_num).astype(int),\
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
  if verbose: console.show_status('Generating rotagram...')
  for i in range(sampling_num):
    coordinate = (index2coor(i, sampling_num) + origin)
    # print((origin[0] - coordinate[0]) ** 2 + (origin[1] - coordinate[1]) ** 2)
    new_features.append(map_images_values(features,
                                          origin, coordinate, sampling_num))
    if verbose: console.print_progress(i, len(range(sampling_num)))
  return np.expand_dims(np.concatenate(new_features, axis=2), -1)



if __name__ == '__main__':
  import matplotlib.pyplot as plt
  x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
  z = np.sqrt(x ** 2 + y ** 2 + 10) + np.sin(x ** 2 + y ** 2 + 10)
  features = np.expand_dims(z,(0, -1))
  plt.imshow(features[0])
  plt.show()
  features = rotagram(features)
  plt.imshow(features[0])
  plt.show()
