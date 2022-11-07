from collections import OrderedDict
import numpy as np

import tensorflow as tf


from tframe import console, DataSet
from tframe.quantity import Quantity
from tframe.trainers.trainer import Trainer



class GANTrainer(Trainer):
  def __init__(
      self,
      generator,
      discriminator,
      agent,
      config,
      training_set=None,
      valiation_set=None,
      test_set=None,
      probe=None
  ):
    super(GANTrainer, self).__init__(None, agent, config, training_set,
                                     valiation_set, test_set, probe)
    self.generator = generator
    self.discriminator = discriminator
    self.model = self.generator

  def train(self):
    # :: Before training
    if self.th.overwrite:
      self.agent.clear_dirs()

    if self.th.load_model:
      self.generator.keras_model, self.counter = self.agent.load_model(
        'gen')
      self.discriminator.keras_model, self.counter = self.agent.load_model(
        'load')
      console.show_status(
        'Model loaded from counter {}'.format(self.counter))
    else:
      self.generator.build(self.th.input_shape)
      self.discriminator.build(self.th.input_shape)
      tf.summary.trace_on(graph=True, profiler=True)

      # self.model.link(tf.random.uniform((self.th.batch_size, *self.th.input_shape)))
      # _ = self.model.keras_model(tf.keras.layers.Input(self.th.input_shape))
      @tf.function
      def predict(x):
        self.generator.keras_model(x)
        self.discriminator.keras_model(x)
        return None

      predict(tf.random.uniform(
        (self.th.batch_size, *self.th.non_train_input_shape)))
      self.agent.write_model_summary()
      tf.summary.trace_off()
      console.show_status('Model built.')
    self.generator.keras_model.summary()
    self.discriminator.keras_model.summary()
    self.agent.create_bash()

    # :: During training
    if self.th.rehearse:
      return

    rounds = self._outer_loop()

    # :: After training
    # self._end_training(rounds)
    # Put down key configurations to note
    self.agent.note.put_down_configs(self.th.key_options)
    # Export notes if necessary
    # Gather notes if necessary
    if self.th.gather_note:
      self.agent.gather_to_summary()

    if self.th.save_last_model:
      self.agent.save_model(self.discriminator.keras_model,
                            self.counter, 'dis')
      self.agent.save_model(self.generator.keras_model,
                            self.counter, 'gen')

    self.generator.keras_model, _ = self.agent.load_model('gen')
    self.discriminator.keras_model, _ = self.agent.load_model('dis')
    if self.th.probe:
      self.probe()


  # region : During training

  def _outer_loop(self):
    rnd = 0
    self.patenice = self.th.patience
    for _ in range(self.th.epoch):
      rnd += 1
      console.section('round {}:'.format(rnd))

      self._inner_loop(rnd)
      self.round += 1
      if self.th.probe and rnd % self.th.probe_cycle == 0:
        assert callable(self.probe)
        self.probe()

      if self.th._stop:
        break

    console.show_status('Training ends at round {}'.format(rnd),
                        symbol='[Patience]')

    if self.th.gather_note:
      self.agent.note.put_down_criterion('Total Parameters of Generator',
                                         self.generator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Parameters of Discriminator',
                                         self.discriminator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Iterations', self.counter)
      self.agent.note.put_down_criterion('Total Rounds', rnd)
      # Evaluate the best model if necessary
      ds_dict = OrderedDict()
      ds_dict['Train'] = self.training_set
      ds_dict['Val'] = self.validation_set
      ds_dict['Test'] = self.test_set
      if len(ds_dict) > 0:
        # Load the best model
        if self.th.save_model:
          self.generator.keras_model, self.counter = self.agent.load_model(
            'gen')
          self.discriminator.keras_model, self.counter = self.agent.load_model(
            'dis')

        # Evaluate the specified data sets
        for name, data_set in ds_dict.items():
          loss_dict = self.validate_model(data_set,
                                          batch_size=self.th.val_batch_size)
          for key in loss_dict:
            title = '{} {}'.format(name, key.name)
            # print(title, loss_dict[key])
            self.agent.note.put_down_criterion(title, loss_dict[key].numpy())

    return rnd

  # region : Private Methods

  def _update_model_by_batch(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape(persistent=True) as tape:
      generator_prediction = self.generator.keras_model(feature)
      generator_loss = self.generator.loss(generator_prediction, target)
      loss_dict[self.generator.loss] = generator_loss

      fake_index = np.random.randint(2, size=len(data_batch))
      _feature = []
      _target = []
      for i, index in enumerate(fake_index):
        if index == 0:
          _feature.append(feature[i])
          _target.append([1, 0])
        else:
          _feature.append(generator_prediction[i])
          _target.append([0, 1])
      _feature = np.array(_feature)
      _target = np.array(_target)

      discriminator_prediction = self.discriminator.keras_model(_feature)
      _target = tf.constant(_target, dtype=tf.float32)

      discriminator_loss = -0.003 * tf.math.log(
        tf.clip_by_value(
          tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(_target,
            discriminator_prediction), 1e-10,1.0))

      # print(_target, discriminator_prediction)
      # print(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(_target,
      #       discriminator_prediction))
      # print(discriminator_loss)
      # quit()
      generator_loss = generator_loss + discriminator_loss

      dis_loss = Quantity('Disloss', smaller_is_better=True)

      # print(discriminator_loss)
      # quit()
      loss_dict[dis_loss] = discriminator_loss
      loss_dict[self.discriminator.loss] = self.discriminator.loss(
        discriminator_prediction, _target)

      for metric in self.generator.metrics:
        loss_dict[metric] = metric(generator_prediction, target)
      for metric in self.discriminator.metrics:
        loss_dict[metric] = metric(discriminator_prediction, _target)
    # print(generator_loss, discriminator_loss)
    # quit()

    gen_grads = tape.gradient(generator_loss, self.generator.keras_model.trainable_variables)
    dis_grads = tape.gradient(loss_dict[self.discriminator.loss], self.discriminator.keras_model.trainable_variables)

    self.optimizer.apply_gradients(zip(gen_grads, self.generator.keras_model.trainable_variables))
    self.optimizer.apply_gradients(zip(dis_grads, self.discriminator.keras_model.trainable_variables))

    return loss_dict


  def validate_model(self, data_set:DataSet, batch_size=None, update_record=True):
    _loss_dict = {}
    _loss_dict[self.generator.loss] = 0
    _loss_dict[self.discriminator.loss] = 0
    for metric in self.generator.metrics:
      _loss_dict[metric] = 0
    for metric in self.discriminator.metrics:
      _loss_dict[metric] = 0

    batch_size_sum = 0
    for i, data_batch in enumerate(data_set.gen_batches(batch_size,
                                                        is_training=False)):
      target = data_batch.targets
      feature = data_batch.features
      generator_prediction = self.generator.keras_model(feature)
      loss = self.generator.loss(generator_prediction, target)

      _loss_dict[self.generator.loss] += tf.reduce_mean(loss) * data_batch.size
      for metric in self.generator.metrics:
        _loss_dict[metric] += tf.reduce_mean(metric(generator_prediction, target)) \
                              * data_batch.size
      batch_size_sum += data_batch.size

    _loss_dict[self.generator.loss] /= batch_size_sum
    for metric in self.generator.metrics:
      _loss_dict[metric] /= batch_size_sum

    if update_record:
      self.generator.loss.try_set_record(_loss_dict[self.generator.loss], data_set)
      for metric in self.generator.metrics:
        metric.try_set_record(_loss_dict[metric], data_set)
    return _loss_dict



  def _inner_loop(self, rnd):
    self.cursor = 0
    self._record_count = 0
    self._update_model_by_dataset(self.training_set, rnd)

    if self.th.validate_train_set:
      loss_dict = self.validate_model(self.training_set,
                                      batch_size=self.th.val_batch_size, update_record=False)
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='train')

      console.show_status('Train set: ' +self._dict_to_string(loss_dict, self.training_set), symbol='[Validation]')

    if self.th.validate_val_set:
      loss_dict = self.validate_model(self.validation_set,
                                      batch_size=self.th.val_batch_size)
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='validation')

      console.show_status('Validation set: ' + self._dict_to_string(loss_dict,
                                                                    self.validation_set),
                          symbol='[Validation]')

    if self.th.validate_test_set:
      loss_dict = self.validate_model(self.test_set,
                                      batch_size=self.th.val_batch_size)
      console.show_status('Test set: ' + self._dict_to_string(loss_dict,
                                                              self.test_set),
                          symbol='[Validation]')
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='test')

    # self.th._stop = True #Test

    if self.generator.metrics[0]._record_appears[self.validation_set]:
      self.patenice = self.th.patience
      console.show_status('Record appears', symbol='[Patience]')
      if self.th.save_model:
        console.show_status('Saving the model to {}'.format(
          self.agent.ckpt_dir ), symbol='[Saving]')
        self.agent.save_model(self.generator.keras_model,
                              self.counter, 'gen')
        self.agent.save_model(self.discriminator.keras_model,
                              self.counter, 'dis')
    else:
      self.patenice -= 1
      if self.patenice < 0:
        self.th._stop = True
      else:
        console.show_status('Record does not appear [{}/{}]'.format(
          self.patenice + 1, self.th.patience), symbol='[Patience]')

