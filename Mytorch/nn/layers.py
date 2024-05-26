import sys 
sys.path.append(".")
sys.path.append("..")
from  .Modules import Module,Parameter
from ..basic import Tensor,exp,log
from .init import *
import numpy as np

import random
import numpy as np
# import torch




def _get_softmax_dim( ndim: int) -> int:
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


class Sequential(Module):
    def __init__(self,*modules):
        super().__init__()
        self.modulelist = list(modules)
    def modules(self):
        modules=[]
        for m in self.modulelist:
            modules.append(m)
            modules.extend(m.modules())
        return modules
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self.modulelist:
            x = module(x)
        return x

class Flatten(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.reshape(-1)
    def __repr__(self):
        return "(Flatten)"
    def __str__(self):
        return "(Flatten)"
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init_He (in_features, out_features, dtype) #请自行实现初始化算法
        if bias:
            self.bias = Tensor(np.zeros(self.out_features))
        else:
            self.bias = None
    def __str__(self):
        return "(Linear:[{},{}])".format(self.in_features,self.out_features)
    def __repr__(self):
        return "(Linear:[{},{}])".format(self.in_features,self.out_features)
    
    def forward(self, X: Tensor) -> Tensor:
        X_out = X @ self.weight
        if self.bias:
            return X_out + self.bias
        return X_out
    
    
    
# Activation Layers

class Softmax(Module):
    def __init__(self,dim=None):
        super().__init__()
        self.dim=dim
    def forward(self,x):
        exp_x = exp(x)
        # if(len(x.shape)==1):
        #     div_term = exp_x.sum()
        # else:
        if self.dim is None:
            self.dim = _get_softmax_dim(len(x.shape))
        div_term = exp_x.sum(dim=[self.dim],keepdim=True)
        logits = exp_x/div_term
        return logits
    def __repr__(self):
        return "(Softmax)"
    def __str__(self):
        return "(Softmax)"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        output =1/(1+exp(-x))
        return output
    def __repr__(self):
        return "(Sigmoid)"
    def __str__(self):
        return "(Sigmoid)"
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        neg_index = (x<=0)
        x[neg_index]=0
        return x
    def __repr__(self):
        return "(ReLU)"
    def __str__(self):
        return "(ReLU)"


class Residual(Module):
    def __init__(self,module):
        super().__init__()
        self.module=module
    def forward(self,x):
        y = self.module(x)
        output = x+y 
        return output
    def __repr__(self):
        return "(Residule:{})".format(self.module)
    def __str__(self):
        return "(Residule:{})".format(self.module)

class Dropout(Module):
    def __init__(self,p=0.2):
        super().__init__()
        self.p=0.2
    def forward(self,x):
        rand_prob = np.random.rand(*x.shape)
        drop_index = rand_prob<self.p
        x[drop_index]=0
        return x
    def __repr__(self):
        return "(Dropout:{})".format(self.p)
    def __str__(self):
                return "(Dropout:{})".format(self.p)


if __name__=='__main__':
    model = Linear(3,5)
    model = Sequential(Linear(3,5),Linear(5,4))
    X = Tensor(np.random.random([6,3]))
    y = model(X)
    print(y.shape)
    # y.backward()
    # print(model.weight.grad)
    # print(model.bias.grad)    


