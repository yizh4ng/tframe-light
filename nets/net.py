from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.core import Function
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

  def __init__(self, name, level=0,  **kwargs):
    """Instantiate Net, a name must be given
       TODO: deprecate inter_type
       :param level: level 0 indicates the trunk
       :param inter_type: \in {cascade, fork, sum, prod, concat}
    """
    self.name = name
    self._level = level

    self.input_ = None
    self._output_scale = None

    self.children = []
    self.branch_outputs = []
    self.kwargs = kwargs

    # Losses
    self._extra_loss = None
    # self._reg_loss = None

    # Tensor extractor
    self._tensor_extractors = []

    self._output_slots = []

  # region : Properties

  # region : Overrode Method

  # TODO: modify with_logits mechanism
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

  @property
  def trainable_variables(self):
    trainalbe_variables = []
    for f in self.children:
      # print(f)
      assert hasattr(f, 'trainable_variables')
      trainalbe_variables.extend(f.trainable_variables)
    return trainalbe_variables