
from abc import ABC, abstractmethod
from ..basic import Tensor
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
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr
    def step(self):
        for i, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            param.data= param.data - grad * self.lr

