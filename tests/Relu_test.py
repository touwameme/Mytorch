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
from torch.nn import ReLU




### Relu
import torch
from torch.nn import ReLU
from Mytorch import Tensor 
func= ReLU()
x = torch.Tensor([-1,0,1,2.]).requires_grad_(True)
x.retain_grad()

y = func(x)
print("pytorch out_tensor",y)
y.backward(torch.ones(y.shape))

print("pytorch grad",x.grad)


func = Mytorch.nn.ReLU()
x = Tensor([-1,0,1,2.]).requires_grad_(True)
x.retain_grad()

y = func(x)
print("Mytorch out_tensor",y)
y.backward(Tensor(np.ones(y.shape)))

print("Mytorch grad",x.grad)