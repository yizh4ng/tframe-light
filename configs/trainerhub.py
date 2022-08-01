from tframe.configs.config_base import Config, Flag
from tframe.enums import InputTypes, SaveMode


class TrainerHub(Config):
  """Trainer Hub manages configurations for Trainer and stores status during
     training"""

  # region : Class Attributes

  epoch = Flag.integer(1, 'Epoch number to train', is_key=None)
  max_iterations = Flag.integer(None, 'Max inner iterations')
  batch_size = Flag.integer(1, 'Batch size', is_key=None, hp_scale='log')
  num_steps = Flag.integer(None, 'Number of time steps', is_key=None)
  shuffle = Flag.boolean(True, 'Whether to shuffle', is_key=None)

  print_cycle = Flag.integer(0, 'Print cycle')
  validate_cycle = Flag.integer(0, 'Validate cycle')
  validate_at_the_beginning = Flag.boolean(
    False, 'Whether to validate before outer_loop')
  validation_per_round = Flag.integer(0, 'Validation per round',
                                      name='val_per_rnd')
  snapshot_cycle = Flag.integer(0, 'Snapshot cycle')
  probe_cycle = Flag.integer(0, 'Probe cycle')
  probe_per_round = Flag.integer(0, 'Probe per round')
  match_cycle = Flag.integer(0, 'Match cycle for RL')

  etch_per_round = Flag.integer(0, 'Etch per round')
  etch_cycle = Flag.integer(0, 'Etch cycle', is_key=None)

  early_stop = Flag.boolean(False, 'Early stop option', is_key=None)
  record_gap = Flag.float(0.0, 'Minimum improvement')
  patience = Flag.integer(
    20, 'Tolerance of idle rounds(or iterations) when early stop is on',
    is_key=None)
  save_mode = Flag.enum(SaveMode.ON_RECORD, SaveMode,
                        "Save mode, \in  ['naive', 'on_record']")
  warm_up_thres = Flag.integer(1, 'Warm up threshold', is_key=None)
  warm_up = Flag.boolean(False, 'Whether to warm up')
  at_most_save_once_per_round = Flag.integer(False, '...')

  round_name = Flag.string('Epoch', 'Name of outer loop during training')
  round = Flag.integer(1, 'General concept of total outer loops, used'
                          ' when outer loop is not called epochs', is_key=None)
  hist_buffer_len = Flag.integer(
    20, 'Max length of historical statistics buffer length')
  validate_train_set = Flag.boolean(
    False, 'Whether to validate train set in trainer._validate_model')
  validate_test_set = Flag.boolean(
    False, 'Whether to test train set in trainer._validate_model')
  terminal_threshold = Flag.float(0., 'Terminal threshold')

  # endregion : Class Attributes

  def __init__(self, trainer=None, as_global=False):
    # Call parent's constructor
    Config.__init__(self, as_global)

    self.trainer = trainer
    self.record_rnd = 0
    # metric log is a list of list
    self.metric_log = []

    self._time_stamp = {'__start_time': None}
    self._stop = False

    self._round_length = None
    self.cursor = None

    self.force_terminate = False
    # Sometimes probe method should know the accuracy history
    self.logs = {}

  # region : Properties

  # @property
  # def round_length(self):
  #   assert isinstance(self.trainer.training_set, TFRData)
  #   # For being compatible with old versions
  #   if hasattr(self.trainer.training_set, 'dynamic_round_len'):
  #     return self.trainer.training_set.dynamic_round_len
  #   else: return getattr(self.trainer.training_set, '_dynamic_round_len', None)

  @property
  def total_outer_loops(self):
    """In most supervised learning tasks, each outer training loop is called
       an epoch. If epoch is specified in config, it will be returned as
       total outer loops. In other tasks such as reinforcement learning,
       an outer loop may be called an episode. In this case, set 'total_rounds'
        in config instead of epoch."""
    assert 1 in (self.epoch, self.round)
    return max(self.epoch, self.round)

  # @property
  # def validation_on(self):
  #   mm = self.trainer.metrics_manager
  #   assert isinstance(mm, MetricsManager)
  #   if not mm.has_metric: return False
  #   val_data = self.trainer.validation_set
  #
  #   # return val_data is not None and self.validate_modulus > 0
  #   return all([val_data is not None,
  #               self.validate_cycle > 0 or self.validation_per_round > 0])

  @property
  def start_time(self):
    return self._time_stamp['__start_time']

  # @property
  # def stop(self):
  #   value = self._stop and self.early_stop
  #   self._stop = False
  #   return value

  # @property
  # def round_progress(self):
  #   if self.round_length is None or self.cursor is None: return None
  #   return 1.0 * self.cursor / self.round_length

  # region : Modulus

  # @property
  # def round_len_is_active(self):
  #   assert isinstance(self.trainer.training_set, TFRData)
  #   return not isinstance(self.trainer.training_set, PerpetualMachine)

  # def _get_modulus(self, verb, act_per_round_key=None, act_cycle_key=None):
  #   assert isinstance(verb, str)
  #   if act_per_round_key is None:
  #     act_per_round_key = '{}_per_round'.format(verb)
  #   if act_cycle_key is None: act_cycle_key = '{}_cycle'.format(verb)
  #   # Get value
  #   act_per_round = getattr(self, act_per_round_key)
  #   act_cycle = getattr(self, act_cycle_key)
  #   # act_cycle has the highest priority
  #   if any([act_cycle > 0, not self.round_len_is_active, act_per_round <= 0]):
  #     return act_cycle
  #   # [Compromise] avoid error in Trainer._show_configuration method
  #   if self.round_length is None: return None
  #   return self.round_length // act_per_round
  #
  # @property
  # def validate_modulus(self):
  #   return self._get_modulus(
  #     'validate', act_per_round_key='validation_per_round')
  #
  # @property
  # def probe_modulus(self): return self._get_modulus('probe')
  #
  # @property
  # def etch_modulus(self): return self._get_modulus('etch')
  #
  # @property
  # def note_modulus(self):
  #   if self.note_cycle <= 0 and self.export_tensors_upon_validation:
  #     return self.validate_modulus
  #   return self._get_modulus('note')

  # endregion : Modulus

  # endregion : Properties

  # region : Public Methods

  def set_up(self, **kwargs):
    for key, arg in kwargs.items():
      if hasattr(self, key): self.__setattr__(key, arg)
      else: raise ValueError('!! can not resolve key {}'.format(key))

  # def sanity_check(self):
  #   assert isinstance(self.trainer, Trainer)

  def tic(self, key='__start_time'):
    self._time_stamp[key] = time.time()

  def toc(self, key='__start_time'):
    assert self._time_stamp[key] is not None
    return time.time() - self._time_stamp[key]

  def raise_stop_flag(self):
    self._stop = True

  # endregion : Public Methods

TrainerHub.register()
