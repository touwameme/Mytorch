import numpy as np
import sys 
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
from ..basic import Tensor

from collections import OrderedDict


class Module(ABC):
    def __init__(self):
        self.training = True
        self._parameters=OrderedDict()
        self._modules=OrderedDict()
        self.prefix="_"
    
        
    
    def register_module(self,name:str, module):
        self._modules[name]=module
    
    def register_parameter(self,name,parameter):
        self._parameters[name]=parameter
    
    def __str__(self,):
        return self.prefix
    
    def __repr__(self,prefix="",recurse=True):
        return self.prefix


    def __getattr__(self,name):
        if '_parameters' in self.__dict__:
            _parameters=self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            _modules=self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        return self.__dict__[name]

    def __setattr__(self,name,value):
        if isinstance(value,Parameter):
            self.register_parameter(name,value)
            return 
        if isinstance(value,Module):
            self.register_module(name,value)
            return 
        self.__dict__[name]=value
        

    def parameters(self) -> List[Tensor]:
        parameters=[v for k,v in self._parameters.items()]
        for m in self.modules():
            parameters.extend([v for k,v in m._parameters.items()])
        return parameters
        
    def set_recursive_prefix(self,prefix="_"):
        for k,v in self._modules.items():
            if len(v._modules)==0:
                v.set_prefix(prefix+"."+k)
            else:
                v.set_prefix(prefix+"."+k)
                v.set_recursive_prefix(prefix+"."+k)
        return 
    
    def modules(self,firstcall=True):
        if(firstcall):
            self.set_recursive_prefix("_")
        modules=[]
        for k,v in self._modules.items():
            if len(v._modules)==0:
                # v.set_prefix(prefix+"."+k)
                modules.append(v)
            else:
                # v.set_prefix(prefix+"."+k)
                modules.append(v)
                modules.extend(v.modules(False))
        return modules
        
    def set_prefix(self,prefix,firstcall=True):
        self.prefix=prefix
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass
    
    def eval(self):
        self.training=False
        for m in self.modules():
            m.training=False
            
    def train(self):
        self.training=True
        for m in self.modules():
            m.training=True


class Parameter(Tensor): # 声明一个类专门表示网络参数
    
    def __init__(self,data):
        self.is_param=True
        super(Parameter,self).__init__(data,requires_grad=True)
        return 
            
    def __str__(self):
        return "Parameter containing:\n("+super(Parameter,self).__repr__()+" grad {}".format(self.grad)+")\n"
    def __repr__(self):
        return "Parameter containing:\n("+super(Parameter,self).__repr__()+" grad {}".format(self.grad)+")\n"


if __name__=='__main__':
    class Dummy(Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            pass


    class M1(Module):
        def __init__(self):
            super().__init__()
            self.param1=Parameter(Tensor([1,2,3]))
            self.m0=Dummy()
            self.m1=Dummy()
        def forward(self):
        
            pass

    class M2(Module):
        def __init__(self):
            super().__init__()
            self.param2=Parameter(Tensor([4,5]))
            self.m2=Dummy()
            self.m3=M1()
        def forward(self):
        
            pass

    m=M2()        
    print(m.modules())
    print(m.parameters())
