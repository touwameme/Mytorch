
from abc import ABC, abstractmethod
from ..basic import Tensor
import numpy as np
class Optimizer(ABC):
    def __init__(self, params):
        self.params = params
    @abstractmethod
    def step(self):
        pass
    def reset_grad(self):
        for p in self.params:
            p.grad = None
            
class SGD(Optimizer):
    def __init__(self, params, lr=0.001,momentum=0,weight_decay=0):
        super().__init__(params)
        self.lr = lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.veloc=[np.zeros(x.data.shape) for x in params]
    def step(self):
        for i, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            delta_grad = -grad*self.lr-2*self.weight_decay*self.lr*param.data
            self.veloc[i]=self.veloc[i]*self.momentum+(1-self.momentum)*delta_grad
            param.data=param.data+self.veloc[i]


class Adam(Optimizer):
    def __init__(self, params, lr=0.001,beta1=0.9,beta2=0.999,weight_decay=0):
        super().__init__(params)
        self.lr = lr
        self.eps=1e-10
        self.beta1=beta1
        self.beta2=beta2
        self.weight_decay=weight_decay
        self.t=0
        self.m=[np.zeros(x.data.shape) for x in params]
        self.v=[np.zeros(x.data.shape) for x in params]
    def step(self):
        self.t+=1
        for i, param in enumerate(self.params):
            assert(param.grad is not None),"Params {} with grad None ".format(param.shape)
            delta_grad = Tensor(param.grad+2*self.weight_decay*self.lr*param.data, dtype='float32').data
            self.m[i] = self.beta1*self.m[i]+(1-self.beta1)*delta_grad
            self.v[i]=self.v[i]*self.beta2+(1-self.beta2)*(delta_grad**2)
            m_hat = self.m[i]/(1-self.beta1**self.t)
            v_hat = self.v[i]/(1-self.beta2**self.t)
            param.data=param.data-self.lr*m_hat/(np.sqrt(v_hat)+self.eps)
          
