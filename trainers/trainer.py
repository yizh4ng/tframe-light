import tensorflow as tf
from tframe import console, hub
from tframe.data.dataset import TFRData
from tensorflow.keras.optimizers import Adam
from tframe.core.agent import Agent
import numpy as np

# Only trainer knows the trainer hub right?

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
      agent,
      training_set=None,
      validation_set=None,
      test_set=None,
  ):
    # Set model for trainer
    self.model = model

    # Date set attributes
    self._training_set = None
    self._validation_set = None
    self._test_set = None
    self.set_data(training_set, validation_set, test_set)
    self.round = 0
    self.th = hub
    self.counter = 0
    self.cursor = None
    self.agent = agent

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

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None, test_set=None):
    if training_set is not None:
      self._training_set = training_set
    if validation_set is not None:
      self._validation_set = validation_set
    if test_set is not None:
      self._test_set = test_set

  def _inter_cut(self, i, total, content, prompt='>>', start_time=None):
    # Show content
    console.show_status(content, symbol=prompt)
    # Print progress bar
    console.print_progress(i, total, start_time=start_time)
    # self.recover_prototal_roundsgress(start_time)

  def recover_progress(self, start_time=None):
    # Print progress bar
    if self.th.progress_bar and self.th.round_length is not None:
      assert isinstance(self._training_set, TFRData)
      progress = self.th.round_progress
      assert progress is not None
      console.print_progress(progress=progress, start_time=start_time)

  @staticmethod
  def _dict_to_string(dict_, show_records=False):
    assert isinstance(dict_, dict)
    string_array = []
    for k, v in dict_.items():
      string = '{} = {:.3f}'.format(k.name, v)
      if k.record_appears and show_records:
        string +=' [New Record]'
      string_array.append(string)
    return ', '.join(string_array)

  def _print_progress(self, i, total, rnd, loss_dict):
    content = '{} {} '.format(
      self.th.round_name, rnd, loss_dict)
    content += self._dict_to_string(loss_dict)

    # Show time elapsed for a single update if required
    if self.th.tic_toc:
      # Here the key for toc should be taken care of
      # TODO: note that print progress may take a lot of time
      content += ' ({:.1f}ms)'.format(self.th.toc('__update') * 1000)

    self._inter_cut(i, total, content, prompt='[Train]', start_time=self.th.start_time)
  # endregion : Public Methods

  # region : Train

  def train(self):
    if self.th.load_model:
      # self.model.build(self.th.input_shape)
      self.model.keras_model, self.counter = self.agent.load_model(self.model.mark)
      console.show_status('Model loaded from counter {}'.format(self.counter))
    else:
      self.model.build(self.th.input_shape)
      console.show_status('Model built.')
    self.model.keras_model.summary()

    if self.th.overwrite:
      self.agent.clear_dirs()

    self.agent.create_bash()

    rounds = self._outer_loop()

    # :: After training
    # self._end_training(rounds)




  # region : During training

  def _outer_loop(self):
    rnd = 0
    self.patenice = self.th.patience
    for _ in range(self.th.total_outer_loops): #TODO: epcoh num
      rnd += 1
      console.section('round {}:'.format(rnd))

      self._inner_loop(rnd)
      self.round += 1
      if self.th._stop:
        break

    console.show_status('Training ends at round {}'.format(rnd), symbol='[Patience]')
    return rnd

  def _inner_loop(self, rnd):
    self.cursor = 0
    self._record_count = 0
    for i, batch in enumerate(self.training_set.gen_batches(
        self.th.batch_size, updates_per_round =self.th.updates_per_round,
        shuffle=self.th.shuffle, is_training=True)):
      self.cursor += 1
      self.counter += 1
      # Update model
      loss_dict = self._update_model(batch)

      if np.mod(self.counter - 1, self.th.print_cycle) == 0:
        self._print_progress(i, self.training_set._dynamic_round_len, rnd, loss_dict)

    if self.th.validate_train_set:
     loss_dict = self.validate_model(self.training_set,
                                     batch_size=self.th.val_batch_size)
     self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='train')

     console.show_status('Train set: ' +self._dict_to_string(loss_dict, show_records=True), symbol='[Validation]')

    loss_dict = self.validate_model(self.validation_set,
                                    batch_size=self.th.val_batch_size,
                                    update_record=True)
    self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='validation')

    console.show_status('Validation set: ' + self._dict_to_string(loss_dict,
                                                                  show_records=True),
                        symbol='[Validation]')

    if self.th.validate_test_set:
      loss_dict = self.validate_model(self.test_set,
                                      batch_size=self.th.val_batch_size)
      console.show_status('Test set: ' + self._dict_to_string(loss_dict,
                                                              show_records=True),
                          symbol='[Validation]')
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='test')

    if self.model.metrics[0].record_appears:
      self.patenice = self.th.patience
      console.show_status('Record appears', symbol='[Patience]')
      if self.th.save_model:
        console.show_status('Saving the model to {}'.format(
          self.agent.ckpt_dir ), symbol='[Saving]')
        self.agent.save_model(self.model.keras_model,
                                      self.counter, self.model.mark)
    else:
      self.patenice -= 1
      if self.patenice < 0:
        self.th._stop = True
      else:
        console.show_status('Record does not appear [{}/{}]'.format(
          self.patenice + 1, self.th.patience), symbol='[Patience]')


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
    loss_dict = {}
    with tf.GradientTape() as tape:
      prediction = self.model.keras_model(feature)
      loss = self.model.loss(prediction, target)
      loss_dict[self.model.loss] = loss
      for metric in self.model.metrics:
        loss_dict[metric] = metric(prediction, target)
    grads = tape.gradient(loss, self.model.keras_model.trainable_variables)
    Adam(learning_rate=self.th.learning_rate).apply_gradients(zip(grads, self.model.keras_model.trainable_variables))
    return loss_dict

  def validate_model(self, data_set:TFRData, batch_size=None, update_record=False):
    _loss_dict = {}
    _loss_dict[self.model.loss] = 0
    for metric in self.model.metrics:
      metric_key = metric
      _loss_dict[metric_key] = 0
    for i, data_batch in enumerate(data_set.gen_batches(batch_size,
                                                        is_training=False)):
      target = data_batch.targets
      feature = data_batch.features
      prediction = self.model.keras_model(feature)
      loss = self.model.loss(prediction, target)

      _loss_dict[self.model.loss] += loss * data_batch.size / data_set.size
      for metric in self.model.metrics:
        _loss_dict[metric] += metric(prediction, target)\
                                 * data_batch.size / data_set.size
    if update_record:
      self.model.loss.try_set_record(_loss_dict[self.model.loss])
      for metric in self.model.metrics:
        metric.try_set_record(_loss_dict[metric])
    return _loss_dict




