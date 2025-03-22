import numpy as np
from core import Node

class Grid:
    """A multi-dimensional container of Nodes, built on top of numpy arrays."""

    def __init__(self, data, requires_grad=True):
        data = [Node._to_node(x, requires_grad=requires_grad) for x in data]
        # Using dtype=Node behaves like a fixed-type container;
        # may need to switch to dtype=object if problems arise
        self.data = np.array(data, dtype=Node)
        self.requires_grad = requires_grad

    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

    def __getattr__(self, name):
        # Delegate attribute access to the internal data array
        return getattr(self.data, name)

    def __add__(self, other):
        other = Grid._to_grid(other)
        return Grid(self.data + other.data)

    def __sub__(self, other):
        other = Grid._to_grid(other)
        return Grid(self.data - other.data)

    def __mul__(self, other):
        other = Grid._to_grid(other)
        return Grid(self.data * other.data)
    
    def backward(self):
        for idx in np.ndindex(self.data.shape):
            self.data[idx].backward()

    @staticmethod
    def sum(g: "Grid") -> "Grid":
        result = np.sum(g.data)
        return Grid([result])
    
    @staticmethod
    def _to_grid(g, requires_grad=False) -> "Grid":
        return g if isinstance(g, Grid) else Grid(g, requires_grad=requires_grad)
    
    def __repr__(self: "Grid"):
        value_array = np.array([node.value for node in self.data])
        return f"\nGrid({np.array2string(value_array, separator=', ', prefix='Grid(')}, 'requires_grad={self.requires_grad}')\n"
