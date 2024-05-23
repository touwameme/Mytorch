##Softmax:
    # data=np.random.random([2,3,2])
    # data=[[[2,3],[5,5]],[[1,5],[7,2]]]
    # d=None
    # X = Tensor(data).requires_grad_(True)
    # y=Softmax(dim=d)(X)
    # y.backward()
    # print(y.shape)
    # print(y)
    # # print(y.grad)
    # # print(X.grad)
    
    # from torch import Tensor
    # from torch.nn import Softmax 
    # X = Tensor(data).requires_grad_(True)
    # X.retain_grad()

    # y=Softmax(dim=d)(X)
    # y.retain_grad()
    # y.backward(Tensor(np.ones(y.shape)))
    # print("---------")
    # print(y.shape)
    # print(y)
    # print(y.grad)
    # print(X.grad)
    