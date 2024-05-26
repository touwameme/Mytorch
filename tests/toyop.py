import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import Mytorch
from Mytorch import Tensor, log,exp,matmul
# from torch import tensor as Tensor

from collections import OrderedDict

from Mytorch.optim import *
from Mytorch.nn import MSELoss,CrossEntropyLoss
from Mytorch.nn import Linear,Module,Sequential,ReLU
from utils import generate_linearly_separable_data


X,y=generate_linearly_separable_data()

X = Tensor(X)
y=Tensor(y).long()

model = Linear(in_features=3,out_features=2)


class LinearReluBlock(Module):
    def __init__(self,in_f,out_f):
        super().__init__()
        self.in_f = in_f
        self.out_f=out_f
        self.linear= Linear(in_f,out_f)
        self.relu = ReLU()
        
        
    def forward(self,x):
        x =self.linear(x)
        x = self.relu(x)
        return x    

class Mymodel(Module):
    def __init__(self,in_f,out_f):
        super().__init__()
        self.in_f = in_f
        self.out_f=out_f
        self.linear1= Linear(in_f,16)
        self.linear2=Linear(16,out_f)
    def forward(self,x):
        x = self.linear1(x)
        x =self.linear2(x)
        return x
        
    
model = Sequential(LinearReluBlock(3,16),Linear(16,2))
# model = Mymodel(3,2)
# model = Linear(3,2)


pred_y = model(X)
hard_y = np.argmax(pred_y.data,axis=1)
print("label ",y)
print("init pred",hard_y)
print(hard_y==y.data)
criterion = CrossEntropyLoss()
print("init CE loss:",criterion(X,y))

# optimizer = SGD(model.parameters(),lr=1,momentum=0.8,weight_decay=0.05)
optimizer = Adam(model.parameters(),lr=0.1,weight_decay=0.001)

for i in range(100):
    pred_y = model(X)
    loss = criterion(pred_y,y)
    loss.backward()
    if((i+1) %10 ==0):
        print("Epoch {} loss {}".format(i,loss))
    optimizer.step()

pred_y = model(X)
hard_y = np.argmax(pred_y.data,axis=1)
print("After trainning:",hard_y==y.data)