"""
<img src="/assets/images/posts/autograd/pytorch-autograd.png" width="350">

A simple implementation of PyTorch's autograd. Heavily inspired by Andrej Karpathy's micrograd lecture.

[`torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) is 
PyTorch's automatic differentiation engine that powers Neural Network training. 
Autograd stands for automatic gradients, which is fundamental for efficient gradient computation and backpropagation.

All Neural Networks utilize backpropagation to adjust weights and biases. Backpropagation 
adjusts the values of the weights in the network by calculating gradients and using a learning rate. 
To calculate the gradients, autograd needs to track operations performed througout the network and perform 
chain rule of calculus.

The tensors keep track of operations and their differentiation using individual `_backward()` functions. 
The actual gradient is calculated with the tensor's `backward()` function, which utilizes topological sort.

The end of the file includes a small Multi-Layer Perceptron using this version of autograd.
"""

import math
import numpy as np

# === Gradient Descent Explained ===

class GradientDescent:
    def __init__(self):
        pass

# === Autograd and Tensor Implementation ===

class Tensor:
    """ stores a single scalar value and its gradient """

    def __init__(self, value, _children=(), _op=''):
        self.value = value
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.value**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(0 if self.value < 0 else self.value, (self,), 'ReLU')

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

# === Implementation of nn.Module ===

# === A Single Neuron ===

# === A Layer of Neuron ===

# === Multi-layer Perceptron ===
