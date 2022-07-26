import tensorflow as tf
from tframe import console
from tframe.data.dataset import TFRData
from tensorflow.keras.optimizers import Adam
from tframe.configs.config_base import Config, Flag
from tframe.enums import InputTypes, SaveMode
import numpy as np


class Trainer():
  """Base class of trainer for training tframe models.

     Model save mechanism when save_mode is
       (1) SaveMode.NAIVE:
           Model will be saved only at the end of each round naively
       (2) SaveMode.ON_RECORD:
           Model will be saved only when a new metric record appears
           after model finishes its warm-up rounds
   """
  HubClass = None
  def __init__(
      self,
      model,
      training_set=None,
      validation_set=None,
      test_set=None,
      snapshot=None,
      probe=None,
      evaluate=None,
      terminator=None
  ):
    # Set model for trainer
    self.model = model

    # Date set attributes
    self._training_set = None
    self._validation_set = None
    self._test_set = None
    self.set_data(training_set, validation_set, test_set)
    self.counter = 0
    self.round = 0
    self.optimizer = Adam(learning_rate=0.0001)
    self.th = TrainerHub(self)

    # Set callable attributes


  # region : Properties

  @property
  def key_metric(self):
    return self.metrics_manager.early_stop_slot

  @property
  def training_set(self):
    if self._training_set is not None:
      assert isinstance(self._training_set, TFRData)
    return self._training_set

  @property
  def validation_set(self):
    if self._validation_set is not None:
      assert isinstance(self._validation_set, TFRData)
    return self._validation_set

  @property
  def test_set(self):
    if self._test_set is not None:
      assert isinstance(self._test_set, TFRData)
    return self._test_set


  # @property
  # def total_rounds(self):  # TODO: CC
  #   # TODO: Batch size must be kept the same among different trials
  #   if self.th.round_length is None: return None
  #   return self.counter / self.th.round_length




  # @property
  # def effective_batch_size(self):
  #   if self.th.bs_mar in (None, 1.0): return self.th.batch_size
    # assert self.th.bs_mar > 0
    # return self.get_from_pocket(
    #   'EFFECTIVE_BATCH_SIZE', initializer=lambda: self.th.batch_size)

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None, test_set=None):
    if training_set is not None:
      self._training_set = training_set
    if validation_set is not None:
      self._validation_set = validation_set
    if test_set is not None:
      self._test_set = test_set

  def recover_progress(self, start_time=None):
    # Print progress bar
    if self.th.progress_bar and self.th.round_length is not None:
      assert isinstance(self._training_set, TFRData)
      progress = self.th.round_progress
      assert progress is not None
      console.print_progress(progress=progress, start_time=start_time)

  # endregion : Public Methods

  # region : Train

  def train(self):

    rounds = self._outer_loop()

    # :: After training
    # self._end_training(rounds)




  # region : During training

  def _outer_loop(self):
    rnd = 0
    for _ in range(self.th.total_outer_loops): #TODO: epcoh num
      rnd += 1
      console.section('round {}:'.format(rnd))

      self._inner_loop(rnd)
      self.round += 1
    return rnd

  def _inner_loop(self, rnd):
    self._record_count = 0
    for i, batch in enumerate(self.training_set.gen_batches(
        self.th.batch_size, updates_per_round =self.th.updates_per_round,
        shuffle=self.th.shuffle, is_training=True)):
      self.counter += 1
      # Update model
      print(self._update_model(batch))

      # Validation
      # if self._validate_model(rnd) and self._save_model_when_record_appears:
      #   if not self.is_online: assert np.isscalar(self.th.round_progress)
      #   self._save_model(inter_cut=True, progress=self.th.round_progress)
      # Etch (i begins from 0, while rnd begins from 1)
      # Probe
      # self._run_probe()
      # Take notes
      # self._take_notes_for_export()



  # endregion : During training

  # region : After training

  def _end_training(self, rounds):
    if self.th.progress_bar: console.clear_line()
    # If this is a hp-tuning task, write record summary
    if self.th.hp_tuning:
      assert not self.th.summary
      self.key_metric.write_record_summary()
    # Flush summary
    if self.th.summary or self.th.hp_tuning:
      self.model.agent.summary_writer.flush()
    # Take notes
    if self.is_online:
      self.model.agent.take_notes(
        'End  after {} iterations'.format(self.counter))
    else:
      total_round = ('' if self.total_rounds is None
                     else ' ({:.1f} total)'.format(self.total_rounds))
      self.model.agent.take_notes(
        'End training after {} rounds{}'.format(rounds, total_round))
    # Evaluate
    if self._evaluate is not None:
      # Load the best model if necessary
      if self.th.save_model:
        flag, _, _ = self.model.agent.load()
        assert flag
      # Evaluate model
      self._evaluate(self)
    # Show RAS if necessary
    if self.th.lives > 0:
      ras_info = self.metrics_manager.RAR_string
      console.show_status(ras_info)
      self.model.agent.take_notes(ras_info)



  # endregion : After training

  # endregion : Train

  # region : Private Methods

  def _update_model(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features

    with tf.GradientTape() as tape:
      loss = self.model.loss_function(self.model.net(feature), target)
    grads = tape.gradient(loss, self.model.net.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.net.trainable_variables))
    return np.mean(loss)




  @staticmethod
  def _dict_to_string(dict_):
    assert isinstance(dict_, dict)
    string_array = ['{} = {:.3f}'.format(k, v) for k, v in dict_.items()]
    return ', '.join(string_array)



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
  trainer_class = Trainer

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

  @property
  def round_length(self):
    assert isinstance(self.trainer.training_set, TFRData)
    # For being compatible with old versions
    if hasattr(self.trainer.training_set, 'dynamic_round_len'):
      return self.trainer.training_set.dynamic_round_len
    else: return getattr(self.trainer.training_set, '_dynamic_round_len', None)

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
  def validation_on(self):
    mm = self.trainer.metrics_manager
    assert isinstance(mm, MetricsManager)
    if not mm.has_metric: return False
    val_data = self.trainer.validation_set

    # return val_data is not None and self.validate_modulus > 0
    return all([val_data is not None,
                self.validate_cycle > 0 or self.validation_per_round > 0])

  @property
  def start_time(self):
    return self._time_stamp['__start_time']

  @property
  def stop(self):
    value = self._stop and self.early_stop
    self._stop = False
    return value

  @property
  def round_progress(self):
    if self.round_length is None or self.cursor is None: return None
    return 1.0 * self.cursor / self.round_length

  # region : Modulus

  @property
  def round_len_is_active(self):
    assert isinstance(self.trainer.training_set, TFRData)
    return not isinstance(self.trainer.training_set, PerpetualMachine)

  def _get_modulus(self, verb, act_per_round_key=None, act_cycle_key=None):
    assert isinstance(verb, str)
    if act_per_round_key is None:
      act_per_round_key = '{}_per_round'.format(verb)
    if act_cycle_key is None: act_cycle_key = '{}_cycle'.format(verb)
    # Get value
    act_per_round = getattr(self, act_per_round_key)
    act_cycle = getattr(self, act_cycle_key)
    # act_cycle has the highest priority
    if any([act_cycle > 0, not self.round_len_is_active, act_per_round <= 0]):
      return act_cycle
    # [Compromise] avoid error in Trainer._show_configuration method
    if self.round_length is None: return None
    return self.round_length // act_per_round

  @property
  def validate_modulus(self):
    return self._get_modulus(
      'validate', act_per_round_key='validation_per_round')

  @property
  def probe_modulus(self): return self._get_modulus('probe')

  @property
  def etch_modulus(self): return self._get_modulus('etch')

  @property
  def note_modulus(self):
    if self.note_cycle <= 0 and self.export_tensors_upon_validation:
      return self.validate_modulus
    return self._get_modulus('note')

  # endregion : Modulus

  # endregion : Properties

  # region : Public Methods

  def set_up(self, **kwargs):
    for key, arg in kwargs.items():
      if hasattr(self, key): self.__setattr__(key, arg)
      else: raise ValueError('!! can not resolve key {}'.format(key))

  def sanity_check(self):
    assert isinstance(self.trainer, Trainer)

  def tic(self, key='__start_time'):
    self._time_stamp[key] = time.time()

  def toc(self, key='__start_time'):
    assert self._time_stamp[key] is not None
    return time.time() - self._time_stamp[key]

  def raise_stop_flag(self):
    self._stop = True

  # endregion : Public Methods

TrainerHub.register()
