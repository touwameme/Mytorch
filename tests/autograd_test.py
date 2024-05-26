

import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
from typing import Union, Tuple, List
from abc import ABC, abstractmethod
from Mytorch import Value,Tensor,exp,log,matmul



import warnings
warnings.filterwarnings("ignore")


def autogradTest1():
    x=Tensor([1.,2.,3.,4,5,6]).requires_grad_(True).reshape([2,3])
    y=exp(Tensor([-2.]).requires_grad_(True))
    z = 2+x*y+x**2
    t=Tensor([6.,4,3,8,3,1]).requires_grad_(True).reshape([3,2])
    out =log(matmul(z,t))
    
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()
    out.retain_grad()
    out.backward(Tensor(np.ones(out.shape)))

    print("[Tensor]")
    print("out",out)        
    print("x",x)
    print("y",y)
    print("z",z)
    
    print("\n[Grad]")
    print("grad_out",out.grad)
    print("grad_x",x.grad)
    print("grad_y",y.grad)
    print("grad_z",z.grad)
    
def TensorOpTest():
    
    x=Tensor([1.,2.,3.,4,5,6]).requires_grad_(True).reshape([2,1,3])
    y=Tensor([1.,3.,1.,7,4,3,6,4,2,4,3,2]).requires_grad_(True).reshape([4,3])
    z = 1+x*y
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()
    z.backward(Tensor(np.ones(z.shape)))

    print("[Tensor]")
    print("x",x)
    print("y",y)
    print("z",z)
    
    print("\n[Grad]")
    print("grad_x",x.grad)
    print("grad_y",y.grad)
    print("grad_z",z.grad)


testcases=[autogradTest1]


print("\n-------My Autograd----------")
for case in testcases:
    case()

print("\n-----Pytorch Version-------")
from torch import Tensor,exp,log,matmul
for case in testcases:
    case()

