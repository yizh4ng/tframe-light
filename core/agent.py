import tensorflow as tf
import os
import re


class Agent(object):
  def __init__(self, trainer):
    self.saved_model_paths = []
    self.trainer = trainer
    self.hub = self.trainer.th
    self._model = self.trainer.model

  @property
  def root_path(self):
    if self.hub.job_dir == './':
      return self.hub.record_dir
    else:
      return self.hub.job_dir


  @property
  def log_dir(self):
    return self.check_path(self.root_path, self.hub.log_folder_name,
                      self._model.mark)


  @property
  def ckpt_dir(self):
    if self.hub.specified_ckpt_path is not None: return self.hub.specified_ckpt_path
    return self.check_path(self.root_path, self.hub.ckpt_folder_name,
                      self._model.mark)


  @property
  def snapshot_dir(self):
    return self.check_path(self.root_path, self.hub.snapshot_folder_name,
                      self._model.mark)


  def check_path(*paths):
      assert len(paths) > 0
      if len(paths) == 1:
        paths = re.split(r'/|\\', paths[0])
        if paths[0] in ['.', '']:
          paths.pop(0)
        if len(paths) > 0 and paths[-1] == '':
          paths.pop(-1)
      path = ""
      for i, p in enumerate(paths):
        # The first p should be treated differently
        if i == 0:
          assert path == "" and p != ""
          # if p[-1] != ':':
          if ':' not in p:
            # Put `/` back to front for Unix-like systems
            path = '/' + p
          else:
            # This will only happen in Windows system family
            path = p + '/'
            # continue
        else:
          path = os.path.join(path, p)

        # Make directory if necessary
        if not os.path.exists(path):
          # TODO: flag in context.hub should not be used here
          os.mkdir(path)
      return path

  def save_parameters(self, paras, counter, dir_path=ckpt_dir,
                      maxmium_number_to_save=2, suffix='.sav'):
    if len(self.saved_model_paths) >= maxmium_number_to_save:
      saved_model_to_delete = self.saved_model_paths.pop(0)
      os.remove(saved_model_to_delete)

    file_name = 'model{}-c{}{}'.format(self._model.mark,
                                        counter, suffix)

    path = self.check_path(dir_path, file_name)

    tf.saved_model.save(paras, path)


  def get_model_counter_from_name(self, path):
    matched = re.search(r'-c(\d+)', path)
    assert matched
    counter = matched.group()
    return counter


