import tensorflow as tf
import os, sys
import re
from tframe import console
from tframe.utils.local import check_path


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


  def config_dir(self, dir_depth=2):
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
      os.remove(saved_model_to_delete)

    file_name = 'model{}-c{}{}'.format(mark, counter, suffix)

    path =check_path(self.ckpt_dir, file_name)

    model.save(path)



  def get_model_counter_from_name(self, path):
    matched = re.search(r'-c(\d+)', path)
    assert matched
    counter = matched.group()
    return counter


