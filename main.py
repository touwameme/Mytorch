import sys 
sys.path.append(".")
sys.path.append("..")
import numpy as np
from Mytorch.utils.data import CsvDataset,DataLoader
from Mytorch.nn import Sequential,Linear,ReLU,Residual
from Mytorch.nn import Parameter,Module,CrossEntropyLoss
from Mytorch.optim import Adam, CosineDecayWithWarmRestarts
from Mytorch import Tensor
import Mytorch

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

  

model = Sequential(Linear(9,16),Residual(LinearReluBlock(16,16)),Residual(LinearReluBlock(16,16)) ,Linear(16,3))

  
D=CsvDataset("./data/iris.csv")
DL = DataLoader(D,batch_size=16,shuffle=True)


optimizer = Adam(model.parameters(),lr=0.01,weight_decay=0.1)
scheduler = CosineDecayWithWarmRestarts(optimizer,max_lr=optimizer.lr,warmup_steps=20,T_max=200)
criterion = CrossEntropyLoss()

model.train()
for i in range(100):
    loss_list=[]
    for x,y in DL:
        pred_y = model(x)
        loss = criterion(pred_y,y)
        loss_list.append(loss.data)
        loss.backward(Tensor(np.ones(loss.shape)))
        scheduler.step()
        optimizer.step()
    if(((i) %10)==0):
        print("Epoch {} Avg loss {}".format(i,np.sum(loss_list)/len(loss_list)))  
    
def test(model,dataloader):
    corrects=0
    total=0
    for x,y in dataloader:
        pred_y = model(x)
        hard_y = np.argmax(pred_y.data,axis=1)
        corrects+=np.sum(hard_y==y.data)
        total+=len(y.data)
    
    print("Accuracy:",corrects/total)      


Mytorch.no_grad()
model.eval()
test(model,DL)