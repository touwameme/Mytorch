

import numpy as np
from typing import Union, Tuple, List
from abc import ABC, abstractmethod
from Mytorch import Value,Tensor
import Mytorch
import warnings
warnings.filterwarnings("ignore")
def autogradTest():
    x=Tensor([1.,2.,3.,4,5,6]).requires_grad_(True)
    # y=x.reshape([2])
    y =Mytorch.exp(Tensor([-2]).requires_grad_(True))
    # z=Tensor([6.,4,3,8,3,1]).requires_grad_(True).reshape([3,2])                                                                                                              
    # z=Tensor.make_const([4])
    
    # out = Mytorch.log(Mytorch.matmul(y,z))
    z = x**y 
    out=Mytorch.exp(z)
    print("TensorOut:",out)
    # print(z.backward())
    out.backward(Tensor(np.ones(out.shape)))
    
    print("grad_out",out.grad)
    print("grad_z",z.grad)
    print("grad_x",x.grad)
    print("grad_y",y.grad)

print("-------My Autograd----------")
autogradTest()

print("-----Pytorch Version-------")
from torch import Tensor
import torch as Mytorch
autogradTest()


# x = Tensor(np.arange(24)).reshape([2,3,4])
# x = x.transpose([0,2,1])
# print(x.shape)