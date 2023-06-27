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

The end of the file includes implementations of PyTorch's `nn.Module` and `nn.Linear` using this version of autograd.
"""

import math
import numpy as np

# === Gradient Descent Explained ===

"""
Gradient Descent is an iterative optimization algorithm that adjusts the parameters 
of the model in the direction of the steepest descent, calculated using 
the gradient of the loss function. Since we are interested in minimizing 
the loss of the model, we want to find the minimum of this function.

To find the direction of the steepest descent of the function, partial differentiation 
is needed.
"""

# === Autograd and Tensor Implementation ===

class Tensor:
    """
    A tensor stores a single scalar value and its gradient, and contains 
    information on the tensor's 'children' tensors, various functions, and the 
    respective function (partial) derivatives for backpropagation.
    
    #### Gradients
    Let's say we have two equations:
    $$z = x + y$$
    $$L = z \times w$$
    The 'local gradient' of \\(x\\) is \\(\frac{\partial z}{\partial x}\\), 
    and the global gradient of \\(x\\) is \\(\frac{\partial L}{\partial x}\\)
    
    The `self.grad` stored for each tensor is the global gradient.
    """

    def __init__(self, value, _children=(), _op=''):
        """
        #### Parameters
        `value`: number value to be stored in the tensor  
        `_children`: children tensors connected to this tensor  
        `_op`: string indicator of the operation used to calculate this tensor
        """
        self.value = value
        # Initialize the gradient
        self.grad = 0
        # Lambda funtion storing backward propagation
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ### Addition operator
    # Example:  
    # $$z = x + y$$
    # $$\frac{\partial z}{\partial x} = 1 + 0 = 1$$
    # If \\(z\\) were part of \\(L\\), then
    # $$\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \times \frac{\partial L}{\partial z}$$
    # $$\frac{\partial L}{\partial x} = 1*\frac{\partial L}{\partial z},$$
    # where \\(\frac{\partial L}{\partial z}\\) here is `out.grad`
    def __add__(self, other):
        # Handle the scenario where the other is an integer/float, not a tensor
        other = other if isinstance(other, Tensor) else Tensor(other)
        # `self.value`, `other.value`, and `out` represent `x`, `y`, and `z` respectively
        out = Tensor(self.value + other.value, (self, other), '+')

        # Apply local backward propagation - i.e. update the gradients for the current and relevant tensors
        def _backward():
            # $$\frac{\partial L}{\partial x} = 1*\frac{\partial L}{\partial z}$$
            # The gradients are added, in case the same tensor is visited multiple times in the equation graph.
            self.grad += 1.0*out.grad
            # $$\frac{\partial L}{\partial y} = 1*\frac{\partial L}{\partial z}$$
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out

    # ### Multiplication operator
    # Example:  
    # $$z = x \times y$$
    # $$\frac{\partial z}{\partial x} = 1 \times y = y$$
    def __mul__(self, other):
        # Handle the scenario where the other is an integer/float, not a tensor
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, (self, other), '*')

        # Apply local backward propagation - i.e. update the gradients for the current and relevant tensors
        def _backward():
            # $$\frac{\partial L}{\partial x} = y \times \frac{\partial L}{\partial z}$$
            self.grad += other.value * out.grad
            # $$\frac{\partial L}{\partial y} = x \times \frac{\partial L}{\partial z}$$
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    # ### Power operator
    # Example:  
    # $$z = x^y$$
    # $$\frac{\partial z}{\partial x} = y \times x^{y-1} $$
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.value**other, (self,), f'**{other}')

        def _backward():
            # $$\frac{\partial L}{\partial x} = y \times x^{y-1} \times \frac{\partial L}{\partial z} $$
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward

        return out

    # ReLU operator
    # Example:  
    # \\(z = ReLU(x)\\), \\(ReLU(x) = 0\\) if \\(x < 0\\) and \\(ReLU(x) = x\\) if \\(x >= 0\\)
    # \\(\frac{\partial z}{\partial x} = 0 \\) if z <= 0.  
    # \\(\frac{\partial z}{\partial x} = 1 \\) if z <= 0.
    def relu(self):
        out = Tensor(0 if self.value < 0 else self.value, (self,), 'ReLU')

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out

    # ### Tanh operator
    # Example:  
    # $$z = \tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^{2x}-1}{e^{2x}+1}$$
    # $$\frac{\partial z}{\partial x} = 1 - \tanh(x)^2 $$
    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            # $$\frac{\partial L}{\partial x} = (1 - \tanh(x)^2) \times \frac{\partial L}{\partial z}$$
            self.grad += (1 - t**2)*out.grad
        out.backward = _backward
        return out

    # ### Overall Backward propagation
    # If this function is called, we are assuming that the current (`self`) tensor is the finale tensor in the equation graph.
    def backward(self):
        # Build a Topological graph all of nodes in the propagation
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # \\(\frac{\partial L}{\partial L} = 1,\\)
        # where \\(L\\) is final tensor node the equation graph.
        self.grad = 1
        # Go one tensor at a time and apply the chain rule to get its gradient.
        # In the end, all tensors in the graph will have their gradients updated accordingly.
        for v in reversed(topo):
            v._backward()

    # Calculate a tensor with the negative value
    def __neg__(self): # -self
        return self * -1

    # Calculate reverse addition (`other` + `self`)
    def __radd__(self, other):
        return self + other

    # Calculate substraction (`self` - `other`)
    def __sub__(self, other):
        return self + (-other)

    # Calculate reverse substraction (`other` - `self`)
    def __rsub__(self, other): # other - self
        return other + (-self)

    # Calculate reverse multiplication (`other`*`self`)
    def __rmul__(self, other):
        return self * other

    # Calculate division as multiplication (`self`/`other`)
    def __truediv__(self, other): # self / other
        return self * other**-1

    # Calculate reverse division (`other`/`self`)
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    # ### Wrapper function
    # Print in the following format
    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

# === Implementation of nn.Module ===
class Module:
    """
    Small scale implementation of PyTorch `nn.Module` using `Tensor` above
    """
    def __init__(self, num_inputs, non_linearity):
        """
        Initialize the model.

        #### Parameters
        `num_inputs`: number of inputs  
        `non_linearity`: What non-linearity to use 
        """
        pass
    
    def forward(self, x):
        """
        Forward propagation through the network/neuron.

        #### Parameters
        `x`: input feature map  
        
        #### Return
        `x`: output feature map
        """
        return x
    
    def zero_grad(self):
        """
        Reset all gradients to be 0 for the next backward propagation
        """
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        """
        Return all parameters as a list
        """
        return []

# === A Single Neuron ===
class Neuron(Module):
    """
    An implementation of a single neuron
    """
    def __init__(self, num_inputs, non_linearity=None):
        # Create a \\(num inputs \times 1\\) dimension of weights
        self.weights = [Tensor(random.uniform(-1,1)) for _in range(num_inputs)]
        # Create a bias
        self.bias = Tensor(0)
        self.non_linearity = non_linearity
    
    # Forward propagation
    def forward(self, x):
        z = sum(wi*xi for wi,xi in zip(self.weights, x), self.bias)
        return z.relu() if self.non_linearity else z
    
    # Show parameters
    def parameters(self):
        return self.weights + [self.bias]


# === Implementation of nn.Linear (a fully connected layer) ===
class Linear(Module):
    def __init__(self, num_inputs, num_outputs, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]
    
    def forward(self, x):
        z = [n(x) for n in self.neurons]
        return z[0] if len(z) == 1 else z
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    