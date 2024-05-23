import numpy as np

from ..basic import Tensor
from .Modules import Module,Parameter
from ..basic import log
from .layers import Softmax

import torch
class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
        self.eps =1e-10
    def forward(self,x,y):
        assert(len(y.shape)==1),"Input true label should be 1 dim"
        # y [bs,1] x [bs,class_num]
        if(isinstance(y,Tensor)):
            y = y.data # ndarray
        assert(y.dtype=='int64'),"Expect dtype Long, find dtype {} ".format(y.dtype)
        class_num = x.shape[1]
        soft_x = Softmax()(x)
        y_true_one_hot = Tensor(np.eye(class_num)[y])
        output=-(y_true_one_hot*log(soft_x+self.eps)).sum()/len(y)
        return output
    
class MSELoss(Module):
    def  __init__(self):
        super(MSELoss,self).__init__()
    
    def forward(self,x,y):
        # x [bs,dim] y [bs,dim]
        output= ((x-y)**2).sum()/(x.size)
        return output
    
class BCELoss(Module):
    def __init__(self):
        super(BCELoss,self).__init__()
        self.eps =1e-10
        
    def forward(self,x,y):
        assert(isinstance(y,Tensor)),"y should be Tensor"
        assert(x.shape[-1]==2),"input shoult be binary predictions"
        positive_loss=-(y*log(x+self.eps)).sum()/(y.shape[0])
        negtive_loss=-((1-y)*log(1-x+self.eps)).sum()/(y.shape[0])
        output=(positive_loss+negtive_loss)/2
        return output