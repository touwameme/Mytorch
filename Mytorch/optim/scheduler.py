from abc import ABC, abstractmethod
from ..basic import Tensor
import numpy as np

class Scheduler(ABC):
    def __init__(self, optimizer):
        self.params = optimizer
    @abstractmethod
    def step(self):
        pass

class StepLR(Scheduler):
    def __init__(self,optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer 
        self.step_size=10
        self.gamma=0.1
        self.t=0
    def step(self):
        self.t+=1
        if(self.t%10==0):
            self.optimizer.lr*=self.gamma
            

class LinearWarmup(Scheduler):
    def __init__(self,optimizer, max_lr=0.001, warmup_steps=128):
        self.optimizer = optimizer 
        self.max_lr=max_lr
        self.warmup_steps=warmup_steps
        self.t=0
        self.init_lr=1e-6
    def step(self):
        self.optimizer.lr = self.init_lr+(self.t/self.warmup_steps)*self.max_lr
        self.t+=1
        
        
        
def cosine_annealing(current_step, total_steps, max_lr, min_lr):
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(current_step / total_steps * np.pi))

class CosineDecayWithWarmRestarts(Scheduler):
    def __init__(self,optimizer, max_lr=0.001,min_lr=1e-6,T_max=100,warmup_steps=20):
        self.optimizer = optimizer 
        self.max_lr=max_lr
        self.min_lr = min_lr
        self.total_steps=T_max
        self.warmup_steps=warmup_steps
        self.t=0
        self.init_lr=1e-6
        self.state=1 #1: warmup 2:cosine annealing
    def step(self):
        if(self.state==1):
            new_lr = self.min_lr+self.t/self.warmup_steps*(self.max_lr-self.min_lr)
            self.t+=1 
            if(self.t==self.warmup_steps):
                self.t=0
                self.state=2
        else:
            new_lr = cosine_annealing(self.t,self.total_steps,self.max_lr,self.min_lr)
            self.t+=1
            if(self.t==self.total_steps):
                self.t=0
                self.state=1
        self.optimizer.lr = new_lr
    