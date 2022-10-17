from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker
from tframe import pedia

from tframe.utils import misc

from tframe.data.base_classes import TFRData
from tframe.utils.misc import convert_to_one_hot


class DataSet(TFRData):

  EXTENSION = 'tfd'

  FEATURES = pedia.features
  TARGETS = pedia.targets

  def __init__(self, features=None, targets=None, data_dict=None,
               name='dataset', is_rnn_input=False, **kwargs):
    """
    A DataSet is the only data structure which can be fed into tframe model
    directly. Data stored in data_dict must be a regular numpy array with the
    same length.
    """
    # Call parent's constructor
    super().__init__(name)

    # Attributes
    self.data_dict = {} if data_dict is None else data_dict
    self.features = features
    self.targets = targets
    self.properties.update(kwargs)


  # region : Properties

  @property
  def features(self): return self.data_dict.get(self.FEATURES, None)

  @property
  def dense_labels(self):
    return self.properties['dense_labels']

  @features.setter
  def features(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.FEATURES] = val
      else: raise TypeError('!! Unsupported feature type {}'.format(type(val)))

  @property
  def targets(self): return self.data_dict.get(self.TARGETS, None)

  @targets.setter
  def targets(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.TARGETS] = val
      else: raise TypeError('!! Unsupported target type {}'.format(type(val)))

  @property
  def representative(self):
    array = list(self.data_dict.values())[0]
    assert isinstance(array, np.ndarray)
    return array

  @property
  def size(self): return len(self.representative)



  # region : Overriden Methods

  def __len__(self): return self.size

  def __getitem__(self, item):
    if isinstance(item, str):
      if item in self.data_dict.keys(): return self.data_dict[item]
      elif item in self.properties.keys(): return self.properties[item]
      else: raise KeyError('!! Can not resolve "{}"'.format(item))

    # If item is index array
    f = lambda x: self._get_subset(x, item)

    data_set = type(self)(data_dict=self._apply(f), name=self.name + '(slice)')
    return self._finalize(data_set, item)

  # endregion : Overriden Methods

  # region : Basic APIs

  def get_round_length(self, batch_size, num_steps=None, training=False,
                       updates_per_round=None, shuffle=True):
    # Calculate round_len according to updates_per_round
    if training and updates_per_round and updates_per_round > 0:
      assert shuffle is True
      self._set_dynamic_round_len(updates_per_round)
      return updates_per_round

    # Calculate round_len according to batch_size and num_steps
    assert isinstance(batch_size, int) and isinstance(training, bool)
    if batch_size < 0:
      round_len = 1
      if training: self._set_dynamic_round_len(round_len)
      return round_len

    elif num_steps is None:
      round_len = np.ceil(self.size / batch_size)

      round_len = int(round_len)
      if training: self._set_dynamic_round_len(round_len)
      return round_len
    else:
      raise NotImplementedError('Unknown parameters cases')

  def gen_batches(self, batch_size, updates_per_round=None,
                  shuffle=False, is_training=False):
    """Yield batches of data with the specific size"""
    round_len = self.get_round_length(batch_size, training=is_training,
                                      updates_per_round=updates_per_round)
    if batch_size == -1: batch_size = self.size

    # Generate batches
    training_indices = np.array(list(range(self.size)))
    if shuffle:
      np.random.shuffle(training_indices)
    for i in range(round_len):
      if is_training:
        if updates_per_round is None:
          indices = self._select(i, batch_size, training_indices)
        else:
          indices = list(np.random.randint(0, self.size, batch_size))
      else:
        indices = self._select(i, batch_size, np.array(list(range(self.size))))
      # Get subset
      data_batch = self[indices]
      # Preprocess if necessary
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch, is_training)
      # Yield data batch
      yield data_batch

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()


  # endregion : Basic APIs

  # region : Public Methods


  def shuffle(self):
    indices = np.arange(self.size)
    np.random.shuffle(indices)
    return self[indices]

  def split(self, sizes, names, random=False, over_key=None):
    for i, size in enumerate(sizes):
      if 0 < size and size < 1:
        sizes[i] = size * self.size

    assert np.sum(sizes) < self.size
    assert len(names) == len(sizes) + 1

    thisdataset = self
    if random:
      thisdataset = self.shuffle()

    indices_group = []
    if over_key is None:
      start_index = 0
      for size in sizes:
        indices_group.append(np.arange(start_index, start_index + size))
        start_index += size

      indices_group.append(np.arange(start_index, thisdataset.size))
    else:
      #TODO: To splitting equally an imbalanced dataset needs further reformatting
      indices_groups = thisdataset.indices_groups(over_key)
      num_groups = len(indices_groups)
      groups_weights = [len(indices_group)/thisdataset.size
                        for indices_group in indices_groups]
      start_index = [0] * num_groups

      indices = [[] for _ in range(len(sizes) + 1)]

      for i, indices_group in enumerate(indices_groups):
        for j, indice in enumerate(indices):
          if j < len(sizes):
            indice.extend((np.array(indices_group)[np.arange(start_index[i], start_index[i] + np.ceil(sizes[j] * groups_weights[i])).astype(int)]).astype(int))
            start_index[i] = start_index[i] + np.ceil(sizes[j] * groups_weights[i])
          else:
            indice.extend((np.array(indices_group)[np.arange(start_index[i], len(indices_group)).astype(int)]).astype(int))

      indices_group = indices

    datasets = []

    for i, indices in enumerate(indices_group):
      dataset = thisdataset[indices]
      dataset.name = names[i]
      datasets.append(dataset)

    return datasets

  def indices_group(self, key, value):
    assert key in self.properties.keys()
    data = self.properties[key]

    if isinstance(value, (list, tuple)):
      indices = [i for i,x in enumerate(data)
                 if x in value]
    else:
      indices = [i for i,x in enumerate(data)
                 if x == value]
    return indices

  def indices_groups(self, key):
    assert key in self.properties.keys()
    targets = set(self.properties[key])
    data = self.properties[key]
    indices_groups = []

    for target in targets:
      indices_groups.append([i for i, x in enumerate(data)
                             if x == target])
    return indices_groups

  def sub_groups(self, key):
    assert key in self.properties.keys()
    indices_groups = self.indices_groups(key)
    return [self[indices] for indices in indices_groups]

  def sub_group(self, key, value):
    assert key in self.properties.keys()
    indices = self.indices_group(key, value)
    return self[indices]

  def sample_balanced_dataset(self, key):
    indice_groups = self.indices_groups(key)
    num_groups = len(indice_groups)
    selected_indices = []
    for i in range(self.size):
      selected_indices.append(np.random.choice(indice_groups[i % num_groups]))
    return self[selected_indices]

  def report(self):
    print(f'Report Dataset Details on {self.name}:')
    print(f':: Size: {self.size}')
    for key in self.properties.keys():
      if isinstance(self.properties[key], (str, int, float)):
        print(f':: {key}: {self.properties[key]}')
      elif isinstance(self.properties[key], (list, tuple, np.ndarray)):
        if not isinstance(self.properties[key][0], (int, float, str, np.uint8,
                                                    np.int32)):
          print(':: Unknown data type {} for key {}.'.format(
            type(self.properties[key][0]).__name__, key))
          continue
        targets = set(self.properties[key])
        if len(targets) > 20:
          print(':: Too many data types for key {}.'.format(key))
          continue
        print(f':: In the dimenstion of {key}:')
        group = self.sub_groups(key)
        for i, target in enumerate(targets):
          print(f':::: {target} has {group[i].size}')
      else:
        print(f'unknown type {type(self.properties[key])} for {key}.')


  def set_classification_target(self, key, targets_set=None, func=None):
    assert key in self.properties.keys()
    data = self.properties[key]
    if func is not None:
      assert callable(func)
      data = [func(v) for v in data]
      self.properties[key] = data
    if targets_set is None:
      targets_set = list(set(data))
      targets_set.sort()
    self.properties['CLASSES'] = [key + ':' + str(target) for target in targets_set]
    self.properties['NUM_CLASSES'] = len(targets_set)

    labels = []
    for i in data:
      label = targets_set.index(i)
      labels.append(label)
    # labels = np.array(labels)
    self.properties['dense_labels'] = labels
    self.data_dict['targets'] = \
      convert_to_one_hot(labels, self.properties['NUM_CLASSES'])


  def split_k_fold(self, K: int, i: int):
    # Sanity check
    assert 0 < i <= K
    # Calculate fold size
    N = self.size // K
    # Find indices
    i1, i2 = (i - 1) * N, (i * N if i < K else self.size)
    val_indices = set(range(i1, i2))
    train_indices = set(range(self.size)) - val_indices
    train_set, val_set = self[list(train_indices)], self[list(val_indices)]
    train_set.name, val_set.name = 'Train Set', 'Val Set'
    return train_set, val_set

  # endregion : Public Methods

  # region : Private Methods

  def _finalize(self, data_set, indices=None):
    assert isinstance(data_set, DataSet)
    data_set.__class__ = self.__class__
    data_set.properties = self.properties.copy()

    if indices is not None:
      for k, v in self.properties.items():
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) == self.size:
          data_set.properties[k] = self._get_subset(v, indices)

    return data_set

  def _select(self, batch_index, batch_size, indices):
    """The result indices may have a length less than batch_size specified.
       * shuffle option is handled in  gen_[rnn_]batches method
    """
    upper_bound = self.size
    assert isinstance(batch_index, int) and batch_index >= 0

    # Green pass
    # if training and shuffle:
    #   return list(np.random.randint(0, self.size, batch_size))

    from_index = batch_index * batch_size
    to_index = min((batch_index + 1) * batch_size, upper_bound)
    selected_indices = list(range(from_index, to_index))

    # return indices
    return  indices[selected_indices]


  def _apply(self, f, data_dict=None):
    """Apply callable method f to all data in data_dict. If data_dict is not
       provided, self.data_dict will be used as default"""
    assert callable(f)
    if data_dict is None: data_dict = self.data_dict
    result_dict = {}
    for k, v in data_dict.items(): result_dict[k] = f(v)
    return result_dict


  @staticmethod
  def _get_subset(data, indices):
    """Get subset of data. For DataSet, data is np.ndarray.
       For SequenceSet, data is a list.
    """
    if np.isscalar(indices):
      if isinstance(data, (list, tuple)): return [data[indices]]
      elif isinstance(data, np.ndarray):
        subset = data[indices]
        return np.reshape(subset, (1, *subset.shape))
    elif isinstance(indices, (list, tuple, np.ndarray)):
      if isinstance(data, (list, tuple)): return [data[i] for i in indices]
      elif isinstance(data, np.ndarray): return data[np.array(indices)]
    elif isinstance(indices, slice): return data[indices]
    else: raise TypeError('Unknown indices format: {}'.format(type(indices)))

    raise TypeError('Unknown data format: {}'.format(type(data)))



  # def _shuffle_training_indices(self):
  #   # indices = list(range(len(self.features)))
  #   indices = list(range(self.size))
  #   np.random.shuffle(indices)
  #   self._indices_for_training = np.array(indices)

  def _set_dynamic_round_len(self, val):
    # To be compatible with old version
    assert getattr(self, '_dynamic_round_len', None) is None
    self._dynamic_round_len = checker.check_positive_integer(val)

  def _clear_dynamic_round_len(self):
    self._dynamic_round_len = None


  @staticmethod
  def _get_dynamic_round_len(act_lens, num_steps, training):
    """
                                 not train
    x x x x x|x x x x x|x x x/x x:x x x x x x x
    x x x x x|x x x x x|x x x/   :
    x x x x x|x x x x x|x x x/x  :
    x x x x x|x x x x x|x x x/x x:x x x x x
                           train
    """
    assert isinstance(act_lens, (np.ndarray, list)) and len(act_lens) > 0
    checker.check_positive_integer(num_steps)
    counter = 0
    while len(act_lens) > 0:
      # Find the shortest sequence
      sl = min(act_lens)
      assert sl > 0
      # Calculate iterations (IMPORTANT). Note that during training act_len
      # .. does not help to avoid inappropriate gradient flow thus sequences
      # .. have to be truncated
      n = int(np.ceil(sl / num_steps))
      counter += n
      # Update act_lens list
      L = sl if training else n * num_steps
      act_lens = [al for al in [al - L for al in act_lens] if al > 0]
    return counter

  # endregion : Private Methods


if __name__ == '__main__':
  features = np.arange(12)
  data_set = DataSet(features)


