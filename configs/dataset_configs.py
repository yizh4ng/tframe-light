from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class DataConfigs(object):
  data_config = Flag.string(None, 'Data set config string', is_key=None)

  train_size = Flag.integer(0, 'Size of training set')
  val_size = Flag.integer(0.1, 'Size of validation set')
  test_size = Flag.integer(0.1, 'Size of test set')

  train_config = Flag.string(None, 'Config string for train_set', is_key=None)
  val_config = Flag.string(None, 'Config string for val_set', is_key=None)
  test_config = Flag.string(None, 'Config string for test_set', is_key=None)

  sequence_length = Flag.integer(0, 'Sequence length', is_key=None)
  prediction_threshold = Flag.float(
    None, 'The prediction threshold used as an parameter for metric function',
    is_key=None)

  # BETA configs
  use_wheel = Flag.boolean(
    True, 'Whether to used wheel to select sequences', is_key=None)
  sub_seq_len = Flag.integer(
    None, 'Length of sub-sequence used in seq_set.get_round_len or '
          'gen_rnn_batches', is_key=None)

  # Data augmentation options
  augmentation = Flag.boolean(False, 'Whether to augment data', is_key=None)
  aug_config = Flag.string(
    None, 'Configuration for data augmentation', is_key=None)
  pad_mode = Flag.string(None, 'Padding option for image padding', is_key=None)


  @property
  def sample_among_sequences(self):
    if self.sub_seq_len in [None, 0]: return False
    assert isinstance(self.sub_seq_len, int) and self.sub_seq_len > 0
    return True



