
import sys
sys.path.append('..')
sys.path.append('.')
import torch
import numpy as np
import Mytorch
from Mytorch import Tensor, log,exp,matmul
from Mytorch.nn import CrossEntropyLoss, MSELoss
if __name__=='__main__':
    ###MSE
    print("\n MSELoss")
    X = Tensor(np.random.random([2,5,4]))
    Y = Tensor(np.random.random([2,5,4]))
    f = MSELoss()
    print("Mytorch:",f(X,Y))
    from torch.nn import MSELoss,CrossEntropyLoss
    from torch import Tensor
    X = Tensor(X.data)
    Y = Tensor(Y.data)
    f = MSELoss()
    print("pytorch:",f(X,Y))
    
    ##CE
    print("\n CrossEntropyLoss ")
    X = Tensor(np.random.random([8,5]))
    Y =Tensor(np.random.randint(0,5,size=[8])).long()
        
    f = CrossEntropyLoss()
    print("Mytorch:",f(X,Y))
    from torch.nn import MSELoss,CrossEntropyLoss,NLLLoss
    from torch import Tensor
    X = Tensor(X.data)
    Y = Tensor(Y.data).long()
    f = CrossEntropyLoss()
    print("pytorch:",f(X,Y))
    
    
    
    #BCE
    print("\n BCELoss ")
    from Mytorch.nn import Tensor,BCELoss
    X = Tensor(np.random.random([8,2]))
    Y =Tensor(np.random.random([8,2]))
        
    f = BCELoss()
    print("Mytorch:",f(X,Y))
    from torch.nn import BCELoss
    from torch import Tensor
    X = Tensor(X.data)
    Y = Tensor(Y.data)
    f = BCELoss()
    print("pytorch:",f(X,Y))
    
    
