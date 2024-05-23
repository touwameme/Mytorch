# 神经网络大作业 梁天润 502023370027

### 一、Value,Op,Tensor

* Value 实现如下：

  ```python
  class Value:
      # op: Optional[Op] # 节点对应的计算操作， Op是自定义的计算操作类
      # inputs: List["Value"]
      # cached_data: NDArray
      # requires_grad: bool
      def __init__(self,op,inputs,required_grad=True):
          self.op = op
          self.inputs = inputs
          self.requires_grad=required_grad 
          self.cached_data=None
      def realize_cached_data(self): # 进行计算得到节点对应的变量，存储在cached_data里
          if self.is_leaf() or self.cached_data is not None:
              return self.cached_data 
          else:
              self.cached_data=self.op.compute([x.realize_cached_data() for x in self.inputs])
              return self.cached_data
      def is_leaf(self):
          return self.op is None
      def __del__(self):
          pass
      @classmethod
      def make_const(cls, data,requires_grad=False): # 建立一个用data生成的独立节点
          value =Value(None,inputs=[],required_grad=requires_grad)
          if not isinstance(input,np.ndarray):
              data = np.array(data)
          value.cached_data=data
          return value
      @classmethod
      def make_from_op(cls,op,inputs):
          value = Value(op,inputs=inputs,required_grad=True)
          value.cached_data=value.realize_cached_data()
          return value
  ```

* Tensor实现如下

  ```python
  class Tensor (Value):
      cnt=0
      def __init__(self, op=None, inputs=[],requires_grad=True,  dtype=None, **kwargs):
          if (op is not None and not isinstance(op,TensorOp)): 
              # use Tensor(data) to init
              data=op
              self.op=None 
              self.inputs=[]
              self.requires_grad=False
              if isinstance(data,Tensor):
                  tensor_data =data.data
              else:
                  if isinstance(data,Value):
                      tensor_data=data.realize_cached_data() 
                  else:#NDarray
                      if not isinstance(data,np.ndarray):
                          if not isinstance(data,list):
                              data =[data]
                          data = np.array(data)
                      tensor_data=data
              self.cached_data=tensor_data
          else:
              super(Tensor,self).__init__(op,inputs,requires_grad)
          self.grad=None
          self.id=Tensor.cnt
          Tensor.cnt+=1
      
      
      @staticmethod
      def from_numpy(numpy_array, dtype="float64"):
          return self.make_const(numpy_array,requires_grad=False)
          
      @staticmethod
      def make_const(data, requires_grad=False):
          tensor=Tensor.__new__(Tensor)
          if isinstance(data,Tensor):
              tensor_data =data.data
          else:
              if isinstance(data,Value):
                  tensor_data=data.realize_cached_data() 
              else:#NDarray
                  if not isinstance(data,np.ndarray):
                      if not isinstance(data,list):
                          data =[data]
                      data = np.array(data)
                  tensor_data=data
          tensor.__init__(None,[],requires_grad,dtype=tensor_data.dtype)
          tensor.cached_data=tensor_data
          
          return tensor
      @staticmethod
      def make_from_op(op:Op,inputs):
          tensor=Tensor.__new__(Tensor)
          tensor.__init__(op,inputs)
          tensor.cached_data=tensor.realize_cached_data()
          return tensor
  
      @property
      def data(self):
          if self.cached_data is None:
              return self.realize_cached_data()
          return self.cached_data
      @property
      def shape(self):
          return self.data.shape
      
      @data.setter
      def data(self,value):
          self.data=value
          
      @property
      def dtype(self):
          return self.data.dtype
  ```

  * 设置了Tensor.cnt用来记录创建的节点的序号
  * 支持和pytorch同样的初始化方法 ，即 tensor_x = Tensor(data)
  * \_\_init\_\_部分会判断输入的数据类型 ,支持 Tensor(int), Tensor(list), Tensor(ndrray), Tensor(Tensor)

### 二、TensorOps,以及Topo排序和自动求导

#### **代码在Mytorch/basic.py中**

##### 1.张量计算

实现了以下功能

```python
class EWiseAdd(TensorOp): # 对应元素相加
c
class EWiseMul(TensorOp): # 对应元素乘
class MulScalar(TensorOp): # 乘常数
class PowerScalar(TensorOp): # 常数幂
class EWiseDiv(TensorOp): # 对应元素除
class DivScalar(TensorOp): # 除以常数
class Transpose(TensorOp): # 矩阵转置
class Reshape(TensorOp): # 变形
class Summation(TensorOp): # 按维度求
class MatMul(TensorOp): # 矩阵相乘
class Negate(TensorOp): # 求相反数
class Log(TensorOp): # 求对数
class Exp(TensorOp): # 求指
    
class AddTensor(TensorOp): # 广播相加
class SubTensor(TensorOp): # 广播相减
class MulTensor(TensorOp): # 广播相乘
class DivTensor(TensorOp): # 广播相除
```

其中OpScalar函数,输入input格式为 [Tensor,Scalar]

实现了scalar项的求导，以MulScalar为例：

```python
class MulScalar(TensorOp):
    def __call__(self,inputs):
        #inputs [Tensor,Scalar]
        return super().__call__(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]*inputs[1]
    def gradient(self, out_grad, node):
        return out_grad*node.inputs[1], Tensor.make_const(np.sum(out_grad.data*node.inputs[0].data))
```

###### 实现了OpTensor类，支持广播操作的运算，以MulTensor为例：

```python
class MulTensor(TensorOp):
    def __call__(self,inputs):
        
        self.Ashape=inputs[0].shape
        self.Bshape=inputs[1].shape
        
        self.Ashape_aln,self.Bshape_aln=align_shape(self.Ashape,self.Bshape)
        
        return super().__call__(op=self,inputs=inputs)
    def compute(self,inputs):
        return inputs[0]*inputs[1]
    
    def gradient(self, out_grad, node):
        self.sumIndexA=np.where(self.Ashape_aln==1)[0]
        self.sumIndexB=np.where(self.Bshape_aln==1)[0]

        out_grad_0,out_grad_1=(out_grad*node.inputs[1],out_grad*node.inputs[0])

        out_grad_A =out_grad_0.sum(dim=self.sumIndexA)
        if(len(self.Ashape)!=len(out_grad_A.shape)):
            out_grad_A=out_grad_A.reshape(self.Ashape)
        
        out_grad_B =out_grad_1.sum(dim=self.sumIndexB)
        if(len(self.Bshape)!=len(out_grad_B.shape)):
            out_grad_B=out_grad_B.reshape(self.Bshape)
        return out_grad_A, out_grad_B
```

Summation,Reshape,Matmul,Transpose的运算均借助numpy完成。以Matmul为例

```python
class MatMul(TensorOp):
    def __call__(self,inputs):
        return super().__call__(op=self,inputs=inputs)    
    def compute(self,inputs):
        return np.matmul(inputs[0],inputs[1])
    def gradient(self,out_grad,node):
        # [a,b]x[b,c]->[a,c]  
        inputs=node.inputs
        index0=np.arange(len(inputs[0].shape))
        index0[-1]=-2
        index0[-2]=-1
        index1=np.arange(len(inputs[1].shape))
        index1[-1]=-2
        index1[-2]=-1
        BT=inputs[1].transpose(index1)
        AT=inputs[0].transpose(index0)
        g1=MatMul()([out_grad,BT])
        g2=MatMul()([AT,out_grad])
        return (g1,g2)
```

###### 其余运算的求导详见Mytorch/basic.py



##### 2.运算符重载 以及常用函数

##### 重载了[+,-(sub),-(negation),*,/,**,@] 运算符

对于需要判断EWiseOp还是OpScalar,OpTensor的运算符定义了如下模板：

```python
def tensor_scalar_op(self,other,ops):
        # ops[EWiseOp,OpScalar,OpTensor]
        EWiseOp=ops[0]
        OpScalar=ops[1]
        OpTensor=ops[2]
        if isinstance(other,Tensor):
            #print("EWiseAdd")
            if(self.shape==other.shape):
                return EWiseOp()([self,other])
            else:
                return OpTensor()([self,other])
        else:
            #print("ScalarAdd")
            other = Tensor.make_const(other,requires_grad=False)
            return OpScalar()([self,other])
```

其中[+,-,*,/]需要重载左绑定和右绑定的运算符，以sub运算符为例，需要重载\_\_sub\_\_,和\_\_rsub\_\_

```python
def __sub__(self,other):
        return self.tensor_scalar_op(other,[EWiseSub,SubScalar])
def __rsub__(self,other):
    if isinstance(other,Tensor):
        if(self.shape==other.shape):
            return EWiseSub()([other,self])
        else:
            return AddScalar()([-self,other])
    else:
        other = Tensor.make_const(other,requires_grad=False)
        return AddScalar()([-self,other])
```

##### 常用的函数封装成了exp和log函数，求导详见代码



#### 2.拓扑排序和自动求导计算图实现

* 递归求解计算图和梯度的代码使用框架源码

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = (find_topo_sort_reverse(output_tensor)) # 请自行实现拓扑排序函数    
    for node in reverse_topo_order:   
        node.grad = sum(node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        node_grads=node.op.gradient(node.grad, node)
        for i, grad in enumerate(node_grads): # 计算node.inputs的partial adjoint
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict
```

* Topo排序  使用dfs实现

  ```python
  def __dfs(topo_dict,node,depth=0):
      if(node==None or not isinstance(node,Tensor)):
          return 
      for n in node.inputs:
          __dfs(topo_dict,n,depth+1)
      if(node in topo_dict):
          topo_dict[node]=max(topo_dict[node],depth)
      else:
          topo_dict[node]=depth
  
  def find_topo_sort_reverse(output_tensor):
      # topo sort the compute graph
      Dic={}
      __dfs(Dic,output_tensor)
      sorted_dict = sorted(Dic.items(), key=lambda item: item[1])
      return [k for k,v in sorted_dict]
  ```

  



#### 测试autograd

##### 运行python autograd_test.py，会分别调用Mytoch和pytorch计算grad，并打印结果



测试函数

```python
def autogradTest1():
    x=Tensor([1.,2.,3.,4,5,6]).requires_grad_(True).reshape([2,3])
    y=exp(Tensor([-2.]).requires_grad_(True))
    z = 2+x*y+x
    t=Tensor([6.,4,3,8,3,1]).requires_grad_(True).reshape([3,2])                    
    out =log(matmul(z,t))
    
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()
    out.retain_grad()
    out.backward(Tensor(np.ones(out.shape)))
```



测试结果

![image-20240521204035900](C:\Users\ltr\AppData\Roaming\Typora\typora-user-images\image-20240521204035900.png)

### 三、Module、Optimizer、Scheduler子类，以及He和Xavier初始化

#### **代码在Mytorch/nn中**

#### （1）Module 和 Parameter 以及初始化算法

* ##### Module

  主要通过重载\_\_getattr__, \_\_setatter\_\_ 实现把类型为Module和Parameter的属性分别存放到\_modules和\_parameters中

  并且重载了\_\_repr\_\_, \_\_str\_\_

  ```python
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
  ```

  

* ##### Parameter

  重载了\_\_repr\_\_, \_\_str\_\_

```python
class Parameter(Tensor): # 声明一个类专门表示网络参数
    
    def __init__(self,data):
        self.is_param=True
        super(Parameter,self).__init__(data,requires_grad=True)
        return 
            
    def __str__(self):
        return "Parameter containing:\n("+super(Parameter,self).__repr__()+")\n"
    def __repr__(self):
        return "Parameter containing:\n("+super(Parameter,self).__repr__()+")\n"
```

使用以下测试代码：python Module.py

```python
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

```

测试结果

![image-20240522212949832](C:\Users\ltr\AppData\Roaming\Typora\typora-user-images\image-20240522212949832.png)



* He初始化

  ```python
  def init_He (in_features, out_features, dtype="float64"):
      he_stddev=np.sqrt(2/in_features)
      array=(np.random.randn(in_features,out_features)*he_stddev).astype(dtype)
      params=Parameter(Tensor(array))
      return params
  ```

  

* Xavier初始化

```python
def init_Xavier(in_features, out_features, dtype="float32"):
    xavier_stddev = np.sqrt(2 / (in_features + out_features))
    initial_weights = np.random.randn(in_features,out_features) * xavier_stddev
    return initial_weights
```

#### （2）常用的Loss functions以及 Layers 

#### 实现了

```
class Linear(Module) # 线性层
class Flatten(Module) # 平铺层
class ReLU(Module) # ReLU激活函数
class Sigmoid(Module) # Sigmoid激活函数
class Softmax(Module) # Softmax层
class CrossEntrophyLoss(Module) # 交叉熵损失
class BinaryCrossEntrophyLoss(Module) # 二元交叉熵损失
class MSELoss(Module) # 均方损失
class BatchNorm1d(Module) # 一维批归一化 （选做）
class LayerNorm1d(Module) # 一维层归一化 （选做）
class DropOut(Module) # Dropout层 （选做）
class Sequential(Module) # 多层模型
class Residual(Module) # 残差连接
```

#### Loss functions

##### Loss function放在Mytorch/nn/loss.py中

* MSELoss

```python
class MSELoss(Module):
    def  __init__(self):
        super(MSELoss,self).__init__()
    
    def forward(self,x,y):
        # x [bs,dim] y [bs,dim]
        output= ((x-y)**2).sum()/(x.size)
        return output
```



* CrossEntropyLoss:

  ```python
  class CrossEntropyLoss(Module):
      def __init__(self):
          super(CrossEntropyLoss,self).__init__()
          self.eps =1e-10
      def forward(self,x,y):
          assert(len(y.shape)==1),"Input true label should be 1 dim"
          # y [bs,1] x [bs,class_num]
          if(isinstance(y,Tensor)):
              y = y.data # ndarray
          print("Y datatype,",y.dtype)
          assert(y.dtype=='int64'),"Expect dtype Long, find dtype {} ".format(y.dtype)
          class_num = x.shape[1]
          soft_x = Softmax()(x)
          y_true_one_hot = Tensor(np.eye(class_num)[y])
          print("Y one hot,",y_true_one_hot)
          output=-(y_true_one_hot*log(soft_x+self.eps)).sum()/len(y)
          return output
  ```

* BCELoss

  ```python
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
  ```

  



#### 验证Loss实现 python loss_test.py

##### 实验结果

![image-20240523212053883](C:\Users\ltr\AppData\Roaming\Typora\typora-user-images\image-20240523212053883.png)

#### Layers 放在Mytorch/nn/layers.py中

* Linear

  ```python
  class Linear(Module):
      def __init__(self, in_features, out_features, bias=True, dtype="float32"):
          super().__init__()
          self.in_features = in_features
          self.out_features = out_features
          self.weight = init_He (in_features, out_features, dtype) 
          if bias:
              self.bias = Tensor(np.zeros(self.out_features))
          else:
              self.bias = None
      def forward(self, X: Tensor) -> Tensor:
          X_out = X @ self.weight
          if self.bias:
              return X_out + self.bias
          return X_out
  ```

  

* Sequential

  ```python
  class Sequential(Module):
      def __init__(self,*modules):
          super().__init__()
          self.modules = modules
      def forward(self, x: Tensor) -> Tensor:
          for module in self.modules:
              x = module(x)
          return x
  ```

* Sigmoid

  ```python
  
  class Sigmoid(Module):
      def __init__(self):
          super().__init__()
      def forward(self,x):
          output =1/(1+exp(-x))
          return output
  ```

  

* Softmax

  ```python
  class Softmax(Module):
      def __init__(self,dim=None):
          super().__init__()
          self.dim=dim
      def forward(self,x):
          exp_x = exp(x)
          if self.dim is None:
              self.dim = _get_softmax_dim(len(x.shape))
          div_term = exp_x.sum(dim=[self.dim],keepdim=True)
          logits = exp_x/div_term
          return logits
  ```

  

#### （3）Optimizer

```python
class Optimizer(ABC):
    def __init__(self, params):
        self.params = params
    @abstractmethod
    def step(self):
        pass
    def reset_grad(self):
        for p in self.params:
            p.grad = None
```



* SGD

  ```python
  class SGD(Optimizer):
      def __init__(self, params, lr=0.001):
          super().__init__(params)
          self.lr = lr
      def step(self):
          for i, param in enumerate(self.params):
              grad = Tensor(param.grad, dtype='float32').data
              param.data= param.data - grad * self.lr
  ```

  

##### 验证优化框架：python toyop.py

##### 生成了一组线性可分的数据，使用Linear 层加CrossEntropyLoss训练100个epochs

实验结果如下

![image-20240523214533989](C:\Users\ltr\AppData\Roaming\Typora\typora-user-images\image-20240523214533989.png)
