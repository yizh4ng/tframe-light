from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.core.function import Function
from tframe.layers.layer import Layer
from tframe.layers.merge import Merge
from tframe.nets.net import Net


class ForkMergeDAGMIMO(Net):
  """A more general fork-merge neural module that allows the structure between
  input and output operation to be any DAG, similar to what has been used
  in NAS-X01 serial papers. """

  def __init__(self, vertices, edges, input_projections=None,
               name='DAG', max_trim=None, auto_merge=False, **kwargs):
    """A neural network module with one input and one output. The internal
    structure is represented as a DAG. For vertices accepting multiple inputs,
    a merge layer must be provided. Otherwise, concatenation layers will be
    used as default. Using this class, one needs to consider very carefully
    how the tensor shape changes and make sure the subgraph can be built
    properly.

    SYNTAX:
    -------
      # Initiate a classifier
      model = Classifier('GooNet')
      model.add(Input(shape=...))

      # Add FMDAG
      fm_dag = ForkMergeDAG(
        [m.Conv2D(filters, 3, activation='relu'),    # 1
         m.MaxPool2D(3, 1),                          # 2
         m.Conv2D(filters ,3, activation='relu'),    # 3
         m.Conv2D(filters, 1, activation='relu'),    # 4
         [m.Merge.Concat(), m.Conv2D(filters, 3)],   # 5
         m.Merge.Concat()],                          # 6 (output vertex)
        edges='1;01;001;1000;10011;100001', name='DAG Example')
      model.add(fm_dag)

      # Add some more layers
      ...

      # Check model by rehearsal
      model.rehearse(export_graph=True)

    :param vertices: list or tuple, each entry can be a Function or
                     list/tuple of Functions
    :param edges: a string or matrix (upper triangular) representing graph edge
    :param input_projections: projection function applied to DAG input tensor
                              for each vertex
    :param name: network name
    :param max_trim: argument passed to Merge layers for truncate unaligned
                     tensors
    :param auto_merge: option to merge multiple inputs automatically
    :param kwargs: other keyword arguments
    """
    # Call parent's initializer
    super(ForkMergeDAGMIMO, self).__init__(name, **kwargs)
    # Attributes
    self.vertices = vertices
    self.edges = edges
    self.adj_mat = None
    self.input_projections = input_projections
    self._init_graph()

    self.max_trim = max_trim
    self.auto_merge = auto_merge

    # Buffers
    self._predecessor_dict = {}
    self._front_vertices = []


  def _init_graph(self):
    """Formulate the adjacent matrix"""
    # Check vertices first
    assert isinstance(self.vertices, (list, tuple))

    mat_size = len(self.vertices)
    mat_shape = (mat_size, mat_size)
    # Check edges and formalize adjacent matrix
    if isinstance(self.edges, str):
      self.adj_mat = self.parse_edge_str(self.edges, mat_size)
    else:
      assert isinstance(self.edges, np.ndarray)
      self.adj_mat = self.edges.astype(dtype=np.bool)

  def build(self):
    input_tensors = []
    for vertex in self.vertices:
      if callable(vertex):
        input_tensors.append(None)
      elif isinstance(vertex, tf.Tensor):
        input_tensors.append(vertex)
      else:
        raise TypeError
    for j in range(len(self.vertices)):
      input_index = np.sum(np.argwhere(self.adj_mat[:, j]), -1)
      if len(input_index) == 1:
        input_tensors[j] = self.vertices[j](input_tensors[input_index[0]])
      elif len(input_index) > 1:
        input = [input_tensors[k] for k in input_index]
        input_tensors[j] = self.vertices[j](input)
    return input_tensors

  @staticmethod
  def parse_edge_str(edge_spec, mat_size):
    """Parse a string representing an upper triangular matrix. The string
    should list each column of the upper triangular part of the matrix in order.
    E.g., 1;01;001;1000;10011;100001
    """
    assert isinstance(edge_spec, str)
    columns = edge_spec.split(';')
    # (1) check column number
    assert len(columns) == mat_size
    adj_mat = np.zeros(shape=[mat_size, mat_size], dtype=np.bool)
    for j, col in enumerate(columns):
      # (2) each column should represent an upper-triangular matrix column
      # assert isinstance(col, str) and len(col) == j
      for i, r in enumerate(col):
        # assert r in '01'
        adj_mat[i, j] = r == '1'

    return adj_mat
