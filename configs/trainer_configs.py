from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf
from .flag import Flag
from tframe.utils.arg_parser import Parser


class TrainerConfigs(object):
  """TODO: Somehow merge this class to TrainerHub
  """

  train = Flag.boolean(True, 'Whether this is a training task')
  smart_train = Flag.boolean(False, 'Whether to use smart trainer', is_key=None)
  save_model = Flag.boolean(True, 'Whether to save model during training')
  load_model = Flag.boolean(False, 'Whether to load model during training')
  overwrite = Flag.boolean(True, 'Whether to overwrite records')
  summary = Flag.boolean(False, 'Whether to write summary')
  epoch_as_step = Flag.boolean(True, '...')
  snapshot = Flag.boolean(False, 'Whether to take snapshot during training')

  evaluate_train_set = Flag.boolean(
    False, 'Whether to evaluate train set after training')
  evaluate_val_set = Flag.boolean(
    False, 'Whether to evaluate validation set after training')
  evaluate_test_set = Flag.boolean(
    False, 'Whether to evaluate test set after training')

  val_batch_size = Flag.integer(None, 'Batch size in batch validation')
  eval_batch_size = Flag.integer(None, 'Batch size in batch evaluation')
  eval_num_steps = Flag.integer(None, 'Step number in batch evaluation')

  updates_per_round = Flag.integer(None, 'Number of updates per round')

  epoch = Flag.integer(1, 'Epoch number to train', is_key=None)
  print_cycle = Flag.integer(0, 'Print cycle')
  batch_size = Flag.integer(1, 'Batch size', is_key=None, hp_scale='log')
  shuffle = Flag.boolean(True, 'Whether to shuffle', is_key=None)
  probe = Flag.boolean(False, 'Whether to probe')
  probe_cycle = Flag.integer(0, 'Probe cycle')
  round_name = Flag.string('Epoch', 'Name of outer loop during training')
  patience = Flag.integer(
    20, 'Tolerance of idle rounds(or iterations) when early stop is on',
    is_key=None)
  validate_train_set = Flag.boolean(
    False, 'Whether to validate train set in trainer._validate_model')
  validate_test_set = Flag.boolean(
    False, 'Whether to test train set in trainer._validate_model')


