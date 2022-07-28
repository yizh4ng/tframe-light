from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.core import Function
import tensorflow.keras.backend as K
import numpy as np
from tframe.layers.common import single_input


class Net(Function):
  """Function which can packet sub-functions automatically when calling add
     method"""

  # CASCADE = pedia.cascade
  # PROD = pedia.prod
  # SUM = pedia.sum
  # FORK = pedia.fork
  # CONCAT = pedia.concat
  # RECURRENT = 'RECURRENT'

  def __init__(self, name,  **kwargs):
    """Instantiate Net, a name must be given
       TODO: deprecate inter_type
       :param level: level 0 indicates the trunk
       :param inter_type: \in {cascade, fork, sum, prod, concat}
    """
    self.name = name

    self.children = []

  # region : Overrode Method

  def _link(self, *input, **kwargs):
    # region : Check inputs
    if len(input) == 1:
      input = input[0]
    else:
      raise SyntaxError('!! Too much inputs')

    # Check children
    assert isinstance(self.children, list)
    # if len(self.children) == 0: raise ValueError('!! Net is empty')

    input_ = input

    pioneer = input_

    output = None
    # Link all functions in children
    for f in self.children:
      output = f(pioneer)
      pioneer = output


    # This will only happens when Net is empty
    if output is None: output = input_

    # Return
    return output

  # endregion : Overrode Methods

  def add(self, f):
    self.children.append(f)