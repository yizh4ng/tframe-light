from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Ignore FutureWarnings before import tensorflow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys, os
import numpy as np
import tensorflow as tf

# [VP] the code block below is used for making tframe compatible with
#   tensorflow 2.*. All modules related to tframe using tensorflow are
#   recommended to import tf in the way below:
#      from tframe import tf

from . import pedia
from .enums import *

from .import core

from .utils import checker
from .utils import console
from .utils import local
from tframe.data.dataset import DataSet

# from .trainers.smartrainer import SmartTrainerHub as DefaultHub
from .configs.trainerhub import TrainerHub as DefaultHub
hub = DefaultHub()
