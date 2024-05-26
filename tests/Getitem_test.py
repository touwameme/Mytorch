import sys
sys.path.append('..')
sys.path.append('.')
import torch
import numpy as np
import Mytorch
from Mytorch import Tensor, log,exp,matmul
# from torch import tensor as Tensor
from collections import OrderedDict

from Mytorch.nn.layers import Softmax as MySoftmax
from Mytorch import Tensor 

import torch

# GETITEM
x = torch.Tensor(np.random.random([4,5])).requires_grad_(True)
x.retain_grad()
y = x[1,:]
# y[1,:]=1
y.backward(torch.ones(y.shape))
print(x.grad)


x = Tensor(np.random.random([4,5]))
x.retain_grad()
y = x[1,:]
# y = x
# y[1,:]=1
y.backward(Tensor(np.ones(y.shape)))
print(x.grad)
