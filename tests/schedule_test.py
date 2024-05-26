import sys
sys.path.append('..')
sys.path.append('.')
from Mytorch.optim import SGD,CosineDecayWithWarmRestarts
from Mytorch.nn import Parameter
from Mytorch import Tensor
from matplotlib import pyplot as plt


optimizer = SGD(params=Parameter(Tensor([1,2,3]).requires_grad_(True)))
scheduler = CosineDecayWithWarmRestarts(optimizer)
L=[]
for i in range(500):
    scheduler.step()
    L.append(optimizer.lr)
    
plt.plot(list(range(500)),L)
plt.savefig("CosineDecayWithWarmRestart.png")