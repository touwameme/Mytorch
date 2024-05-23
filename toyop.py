import torch
import numpy as np
import Mytorch
from Mytorch import Tensor, log,exp,matmul
# from torch import tensor as Tensor

from collections import OrderedDict

from Mytorch.optim import SGD 
from Mytorch.nn import MSELoss,CrossEntropyLoss
from Mytorch.nn import Linear,Module
from utils import generate_linearly_separable_data


X,y=generate_linearly_separable_data()

X = Tensor(X)
y=Tensor(y).long()

model = Linear(in_features=3,out_features=2)
pred_y = model(X)
hard_y = np.argmax(pred_y.data,axis=1)
print("label ",y)
print("init pred",hard_y)
print(hard_y==y.data)
criterion = CrossEntropyLoss()
print("init CE loss:",criterion(X,y))

optimizer = SGD(model.parameters(),lr=1)
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