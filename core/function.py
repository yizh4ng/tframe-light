from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tframe import tf


class Function(object):
  """A core concept in tframe"""
  master = None
  parameters = None

  def group_name(self):
    raise NotImplementedError('Property "group_name" has not implemented yet')

  def __call__(self, *inputs, **kwargs):
    """When a Function is called, it will be linked into a model and the
       corresponding parameters are registered

    :return: the output tf tensor
    """
    # Get the link method
    link = lambda: self._link(*inputs, **kwargs)
    # Handle the situation when self can be both feed-forward and recurrent
    if self.master is not None:
      assert issubclass(self.master, Function)
      link = lambda: self.master._link(self, *inputs, **kwargs)

    # Call _link to get the output tensor and register parameters
    def get_output_and_register():
      output = link()
      return output

    output = get_output_and_register()

    self.linked = True

    return output


  def _link(self, *inputs, **kwargs):
    raise NotImplementedError('_link method not implemented')

  @property
  def trainable_parameters(self):
    raise NotImplementedError
