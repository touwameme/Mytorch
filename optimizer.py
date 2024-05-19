class Optimizer(ABC):
    def __init__(self, params):
self.params = params
@abstractmethod
def step(self):
pass
def reset_grad(self):
for p in self.params:
p.grad = None