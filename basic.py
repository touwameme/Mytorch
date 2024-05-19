import numpy as np

from abc import ABC, abstractmethod



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
            self.cached_data=self.op.compute(np.array([x.realize_cached_data() for x in self.inputs]))
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
 


class Op(ABC):
    @abstractmethod
    #  compute(self, *args: Tuple["NDArray"]) -> NDArray:
    def compute(self,inputs):
        pass
    # def gradient(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
    # 后向求导. 计算每个输入变量对应的局部伴随值(partial adjoint)
    # 参数out_grad是输出变量对应的伴随值，node是计算操作所在的计算图节点
    # 为方便编程，输出总是一个序列Tuple
    @abstractmethod
    def gradient(self,out_grad,inputs=None):
        pass


class AddOp(Op):
    def __init__(self):
        self.name="ADD"    
    def compute(self,inputs):
        return np.sum(inputs,axis=0,keepdims=True)
    def gradient(self,out_grad,inputs):
        self.input_shape=inputs.shape
        assert(out_grad.shape[-1]==self.inputs_shape[-1])
        replicates=np.ones(self.input_shape[0],1)
        return replicates*out_grad;
    

class TensorOp(Op):
    # 继承计算操作类，实现张量特有的计算
    def __call__(self, *args):
        print("TensrpOP")
        return Tensor.make_from_op(self, args)
  
        


    
class AddScalar(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]+inputs[1]
    def gradient(self,out_grad,node):
        retunr (out_grad)

class EWiseAdd(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=EWiseAdd(),inputs=inputs)
    def compute(self,inputs):
        return np.sum(inputs,axis=0)
    
    def gradient(self, out_grad, node):
        return out_grad, out_grad  
    
  
    
class Tensor (Value):
    # grad: "Tensor" 
    def __init__(self, op, inputs,requires_grad=True,  dtype=None, **kwargs):
        # self.Vdata=self.super(Tensor,se)
        super(Tensor,self).__init__(op,inputs,requires_grad)
        self.dtype=dtype

    
    def __add__(self,other):
        if isinstance(other,Tensor):
            return EWiseAdd()([self,other])
        else:
            return AddScalar(other)(self)
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
                if not isinstance(input,np.ndarray):
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
        return self.realize_cached_data()