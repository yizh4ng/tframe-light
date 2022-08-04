from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
from tframe import tf
from collections import OrderedDict
import time

import tframe as tfr
from tframe import pedia

from .flag import Flag
from .dataset_configs import DataConfigs
from .model_configs import ModelConfigs
from .note_configs import NoteConfigs
from .trainer_configs import TrainerConfigs


class Config(
  DataConfigs,
  ModelConfigs,
  NoteConfigs,
  TrainerConfigs,
):
  registered = False


  visualize_tensors = Flag.boolean(
    False, 'Whether to visualize tensors in core')
  visualize_kernels = Flag.boolean(
    False, 'Whether to visualize CNN kernels in core')

  tensor_dict = Flag.whatever(None, 'Stores tensors for visualization')

  tic_toc = Flag.boolean(True, 'Whether to track time')

  # A dictionary for highest priority setting
  _backdoor = {}

  def __init__(self):
    # Try to register flags into tensorflow
    if not self.__class__.registered:
      self.__class__.register()
    self._time_stamp = {'__start_time': None}

  # region : Properties
  @property
  def start_time(self):
    return self._time_stamp['__start_time']

  @property
  def key_options(self):
    ko = OrderedDict()
    for name in self.__dir__():
      if name in ('key_options', 'config_strings'): continue
      attr = self.get_attr(name)
      if not isinstance(attr, Flag): continue
      if attr.is_key:
        ko[name] = attr.value
    return ko


  # endregion : Properties

  # region : Override

  def __getattribute__(self, name):
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag): return attr
    else:
      if name in self._backdoor: return self._backdoor[name]
      return attr.value

  def __setattr__(self, name, value):
    # Set value to backdoor if required, e.g., th.batch_size = {[256]}
    if isinstance(value, list) and len(value) == 1:
      if isinstance(value[0], set) and len(value[0]) == 1:
        self._backdoor[name] = list(value[0])[0]

    # If attribute is not found (say during instance initialization),
    # .. use default __setattr__
    if not hasattr(self, name):
      object.__setattr__(self, name, value)
      return

    # If attribute is not a Flag, use default __setattr__
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag):
      object.__setattr__(self, name, value)
      return

    # Now attr is definitely a Flag
    # if name == 'visible_gpu_id':
    #   import os
    #   assert isinstance(value, str)
    #   os.environ['CUDA_VISIBLE_DEVICES'] = value

    if attr.frozen and value != attr._value:
      raise AssertionError(
        '!! config {} has been frozen to {}'.format(name, attr._value))
    # If attr is a enum Flag, make sure value is legal
    if attr.is_enum:
      if value not in list(attr.enum_class):
        raise TypeError(
          '!! Can not set {} for enum flag {}'.format(value, name))

    attr._value = value
    if attr.ready_to_be_key: attr._is_key = True

    # Replace the attr with a new Flag TODO: tasks with multi hubs?
    # object.__setattr__(self, name, attr.new_value(value))


  # endregion : Override


  @classmethod
  def register(cls):
    queue = {key: getattr(cls, key) for key in dir(cls)
             if isinstance(getattr(cls, key), Flag)}
    for name, flag in queue.items():
      if flag.should_register: flag.register(name)
      elif flag.name is None: flag.name = name
    cls.registered = True



  def get_attr(self, name):
    return object.__getattribute__(self, name)

  def get_flag(self, name):
    flag = super().__getattribute__(name)
    if not isinstance(flag, Flag):
      raise TypeError('!! flag {} not found'.format(name))
    return flag


  # endregion : Public Methods


  def tic(self, key='__start_time'):
    self._time_stamp[key] = time.time()


  def toc(self, key='__start_time'):
    assert self._time_stamp[key] is not None
    return time.time() - self._time_stamp[key]


Config.register()
