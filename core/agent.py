import tensorflow as tf
import os, sys
import re
from tframe import console
from tframe.utils.local import check_path, clear_paths
from tensorflow import keras
import shutil



class Agent(object):
  def __init__(self, trainer):
    self.saved_model_paths = []
    self.trainer = trainer
    self._model = self.trainer.model
    self.config_dir()

  @property
  def root_path(self):
   return self.job_dir


  @property
  def log_dir(self):
    return check_path(self.root_path, 'log',
                      self._model.mark)


  @property
  def ckpt_dir(self):
    return check_path(self.root_path, 'checkpoints',
                      self._model.mark)


  @property
  def snapshot_dir(self):
    return check_path(self.root_path, 'snapshot',
                      self._model.mark)


  def config_dir(self, dir_depth=1):
    """This method should be called only in XX_core.py module for setting
       default job_dir and data_dir.
    """
    self.job_dir = os.path.join(sys.path[dir_depth - 1])
    self.data_dir = os.path.join(self.job_dir, 'data')
    console.show_status('Job directory set to `{}`'.format(self.job_dir))


  def save_model(self, model, counter, mark,
                      maxmium_number_to_save=2, suffix='.sav'):
    if len(self.saved_model_paths) >= maxmium_number_to_save:
      saved_model_to_delete = self.saved_model_paths.pop(0)
      shutil.rmtree(saved_model_to_delete)
    file_name = 'model{}-c{}{}'.format(mark, counter, suffix)

    path =check_path(self.ckpt_dir, file_name)

    model.save(path)
    self.saved_model_paths.append(path)

  def load_model(self, mark, suffix='.sav'):
    counter = 0
    for root, dirs, files in os.walk(self.ckpt_dir):
      for dir in dirs:
        _counter = self.get_model_counter_from_name(dir)
        if _counter is not None:
          if _counter > counter:
            counter = _counter
    file_name = 'model{}-c{}{}'.format(mark, counter, suffix)
    path = check_path(self.ckpt_dir, file_name)
    return keras.models.load_model(path), counter


  def get_model_counter_from_name(self, path):
    matched = re.search('-c(\d+)', path)
    if matched:
      counter = matched.group(1)
      return int(counter)
    else:
      return None


