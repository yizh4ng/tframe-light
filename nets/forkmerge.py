from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.core.function import Function
from tframe.layers.layer import Layer
from tframe.layers.merge import Merge
from tframe.nets.net import Net


class ForkMergeDAG(Net):
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
    super(ForkMergeDAG, self).__init__(name, **kwargs)
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
    vertices = []
    for vertex in self.vertices:
      if not isinstance(vertex, (list, tuple)): vertex = [vertex]
      # Each vertex is organized as a list/tuple of Functions
      vertices.append(vertex)
    # Set vertices back
    self.vertices = vertices

    mat_size = len(self.vertices) + 1
    mat_shape = (mat_size, mat_size)
    # Check edges and formalize adjacent matrix
    if isinstance(self.edges, str):
      self.adj_mat = self.parse_edge_str(self.edges, mat_size)
    else:
      assert isinstance(self.edges, np.ndarray)
      self.adj_mat = self.edges.astype(dtype=np.bool)

    # # Check adjacent matrix
    # assert self.adj_mat.shape == mat_shape
    # assert np.allclose(self.adj_mat, np.triu(self.adj_mat))
    # if not all([
    #   all([np.sum(self.adj_mat[:, j]) > 0 for j in range(1, mat_size)]),  # in
    #   all([np.sum(self.adj_mat[i, :]) > 0 for i in range(mat_size - 1)]),
    #   # out
    # ]): raise AssertionError('!! Adjacent matrix {} is illegal'.format(
    #   self.adj_mat))

    # Check input projections, make sure self.input_projections is a list of
    #   layer list (may be empty), e.g., [[], [Conv], [], [], [Conv, Conv]]
    # Input projections only apply to in-vertex
    if self.input_projections is None:
      self.input_projections = [[]] * len(self.vertices)
      return
    assert isinstance(self.input_projections, (tuple, list))
    assert len(self.input_projections) == len(self.vertices)
    projections = []
    for j, p in enumerate(self.input_projections):
      # self.adj_mat[i, j] means `whether vertex i goes to vertex j`
      if not self.adj_mat[0, j + 1] or p is None:
        p = []
      elif isinstance(p, (tuple, list)):
        if len(p) > 0: assert all([isinstance(f, Layer) for f in p])
        p = list(p)
      elif isinstance(p, Layer):
        p = [p]
      else:
        raise TypeError('!! Illegal projection `{}`'.format(p))
      projections.append(p)
    self.input_projections = projections

  def _link(self, input_: tf.Tensor, **kwargs):
    # Output tensors of each vertex
    outputs = [input_]
    for j, funcs in enumerate(self.vertices):
      assert isinstance(funcs, list)

      # Get input tensors according to adjacent matrix
      #    in|vertices |out
      # j = 0|1 2 3 4 5|6          <= `outputs` indices
      #   [[+|1 0 0 1 1|1] 0       <= len(vertices) == 5
      #    [ |+ 1 0 0 0|0] 1
      #    [ |  + 1 0 0|0] 2       <= adjacent matrix
      #    [ |    + 0 1|0] 3
      #    [ |      + 1|0] 4
      #    [ |        +|1] 5
      #    [ |         |+]]
      input_tensors = [
        t for i, t in enumerate(outputs) if self.adj_mat[i, j + 1]]

      # Apply input projection if necessary
      # Note that input projection only applies to DAG input tensor
      proj_flag = input_ in input_tensors and self.input_projections[j] != []
      if proj_flag:
        layers = self.input_projections[j]
        x = input_
        for layer in layers:
          # Add before link, otherwise variable may be shared
          self.add(layer)
          x = layer(x)
        # Set tensors back
        input_tensors[0] = x
        # Set layers[0] as front vertices
        self._front_vertices.append(layers[0])

      # Take down predecessors for structure details
      predecessors = [fs[-1] for i, fs in enumerate(self.vertices)
                      if self.adj_mat[i + 1, j + 1]]
      # Projection is only applied to the first input tensor, thus
      #  it should be put at the first place of predecessors
      if proj_flag: predecessors.insert(0, self.input_projections[j][-1])
      if predecessors: self._predecessor_dict[funcs[0]] = predecessors
      # Add input
      if self.adj_mat[0, j + 1] and not proj_flag:
        self._front_vertices.append(funcs[0])

      # Feed input_tensor(s) to funcs
      x = input_tensors
      for f in funcs:
        # Add funcs to children list for structure details
        if isinstance(x, (list, tuple)):
          if len(x) == 1:
            x=x[0]

        self.add(f)
        # Call function
        x = f(x)

      outputs.append(x)

    # Clear predecessor_dict for conciser structure detail
    for i, f in enumerate(self.children[:-1]):
      if f not in self._predecessor_dict: continue
      preds = self._predecessor_dict[f]
      if len(preds) == 1 and preds[0] is self.children[i - 1]:
        self._predecessor_dict.pop(f)

    # Return the result from the output vertex
    return outputs[-1]

  @staticmethod
  def parse_edge_str(edge_spec, mat_size):
    """Parse a string representing an upper triangular matrix. The string
    should list each column of the upper triangular part of the matrix in order.
    E.g., 1;01;001;1000;10011;100001
    """
    assert isinstance(edge_spec, str)
    columns = edge_spec.split(';')
    # (1) check column number
    assert len(columns) == mat_size - 1
    adj_mat = np.zeros(shape=[mat_size, mat_size], dtype=np.bool)
    for j, col in enumerate(columns):
      # (2) each column should represent an upper-triangular matrix column
      assert isinstance(col, str) and len(col) == j + 1
      for i, r in enumerate(col):
        assert r in '01'
        adj_mat[i, j + 1] = r == '1'

    return adj_mat