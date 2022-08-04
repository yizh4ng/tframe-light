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

  def __init__(self):
    # Call parent's constructor
    Config.__init__(self)

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

  @property
  def start_time(self):
    return self._time_stamp['__start_time']


  # region : Modulus


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

  # endregion : Public Methods

TrainerHub.register()
