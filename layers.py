class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
super().__init__()
self.in_features = in_features
self.out_features = out_features
self.weight = init_He (in_features, out_features, dtype) #请自行实现初始化算法
if bias:
self.bias = Tensor(numpy.zeros(self.out_features))
else:
self.bias = None
def forward(self, X: Tensor) -> Tensor:
X_out = X @ self.weight
if self.bias:
return X_out + self.bias.broadcast_to(X_out.shape)
return X_ou