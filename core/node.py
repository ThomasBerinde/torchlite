from numbers import Real
from typing import Callable, Optional, Tuple, Union

GradFn = Callable[["Node", "Node", "Node"], None]
Inputs = Tuple["Node", "Node"]

class Node:
    """Basic unit in the Torchlite autograd system, tracking values and gradients 
    as part of the computation graph."""

    def __init__(self: "Node", 
                 value: Real, 
                 requires_grad: bool = True, 
                 inputs: Optional[Inputs] = None, 
                 grad_fn: Optional[GradFn] = None) -> None:
        self.value = value
        self.grad = None
        self.requires_grad = requires_grad
        self.inputs = inputs if inputs is not None else []
        self.grad_fn = grad_fn
    
    def __add__(self: "Node", other: Union["Node", Real]) -> "Node":
        other = Node._to_node(other)
        return Node(
            value = self.value + other.value,
            inputs = [self, other],
            grad_fn = Node._backward_add)
    
    def __sub__(self: "Node", other: Union["Node", Real]) -> "Node":
        other = Node._to_node(other)
        return Node(
            value = self.value - other.value,
            inputs = [self, other],
            grad_fn = Node._backward_sub)
    
    def __mul__(self: "Node", other: Union["Node", Real]) -> "Node":
        other = Node._to_node(other)
        return Node(
            value = self.value * other.value,
            inputs = [self, other],
            grad_fn = Node._backward_mul)
    
    def backward(self: "Node") -> None:
        if self.grad is None:
            self.grad = 1.
        if self.grad_fn and self.inputs:
            self.grad_fn(self, *self.inputs)
        for input in self.inputs:
            input.backward()
    
    @staticmethod
    def _backward_add(self: "Node", n1: "Node", n2: "Node") -> None:
        if n1.requires_grad is True:
            n1._ensure_grad()
            n1.grad += self.grad
        if n2.requires_grad is True:
            n2._ensure_grad()
            n2.grad += self.grad

    @staticmethod
    def _backward_sub(self: "Node", n1: "Node", n2: "Node") -> None:
        if n1.requires_grad is True:
            n1._ensure_grad()
            n1.grad += self.grad
        if n2.requires_grad is True:
            n2._ensure_grad()
            n2.grad -= self.grad
    
    @staticmethod
    def _backward_mul(self: "Node", n1: "Node", n2: "Node") -> None:
        if n1.requires_grad is True:
            n1._ensure_grad()
            n1.grad += self.grad * n2.value
        if n2.requires_grad is True:
            n2._ensure_grad()
            n2.grad += self.grad * n1.value

    @staticmethod
    def _to_node(x: Union["Node", Real]) -> "Node":
        return x if isinstance(x, Node) else Node(x, requires_grad=False)
    
    def _ensure_grad(self: "Node") -> None:
        if self.grad is None:
            self.grad = 0
    
    def __repr__(self: "Node") -> str:
        return f"Node(value={self.value}, grad={self.grad})"