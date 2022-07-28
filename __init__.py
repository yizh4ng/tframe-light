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
# from .utils import linker
from tframe.data.dataset import DataSet

# from .trainers.smartrainer import SmartTrainerHub as DefaultHub
from .trainers.trainer import Trainer as DefaultHub
# from .import models


# from .utils.organizer import mu





# Try to register folders
# try:
#   from roster import folders
#   root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#   # Register these folders behind potential first few directories inside root.
#   # This is to avoid `th.config_dir()` inside xx_core.py scripts perform
#   # incorrectly. TODO: this code block should be refactored
#   for i, p in enumerate(sys.path):
#     if root in p: continue
#     break
#   for fd in folders: sys.path.insert(i, os.path.join(root, fd))
#   console.show_status('Registered {} (root: `{}`) to sys.path.'.format(
#     ', '.join(['`{}`'.format(fd) for fd in folders]), root))
# except:
#   pass
#
