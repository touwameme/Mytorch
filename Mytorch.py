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
            # print("REALIZE_CACHED_DATA:",self.op)
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
    
### Tenso Operators
class TensorOp(Op):
    # 继承计算操作类，实现张量特有的计算
    def __call__(self, *args):
        #print("TensrpOP")
        return Tensor.make_from_op(self, args)
  
        
class MulScalar(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]+inputs[1]
    def gradient(self,out_grad,node):
        return (out_grad)

    
class AddScalar(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]+inputs[1]
    def gradient(self,out_grad,node):
    # gradient for scalar is sum(grad_out)
    
        return (out_grad,Tensor(np.sum(out_grad.data)))
    

class EWiseAdd(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        return np.sum(inputs,axis=0)
    
    def gradient(self, out_grad, node):
        #print("GRAD")
        return out_grad, out_grad  
    
class SubScalar(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]-inputs[1]
    def gradient(self,out_grad,node):
        return (out_grad,Tensor(-np.sum(out_grad.data)))

class EWiseSub(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        
        assert(len(inputs)==2)
        return inputs[0]-inputs[1]
    
    def gradient(self, out_grad, node):
        return out_grad, -out_grad  


class DivScalar(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]/inputs[1]
    def gradient(self,out_grad,node):
        return (out_grad/node.inputs[1],Tensor(-np.sum(out_grad.data*node.inputs[0].data)/(node.inputs[1].data**2)))

class EWiseDiv(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        
        assert(len(inputs)==2)
        return inputs[0]/inputs[1]
    
    def gradient(self, out_grad, node):
        return out_grad/node.inputs[1], -out_grad*node.inputs[0]/(node.inputs[1]*node.inputs[1])  

class MulScalar(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        
        assert(len(inputs)==2)
        return inputs[0]*inputs[1]
    
    def gradient(self, out_grad, node):
        return out_grad*node.inputs[1], out_grad*(np.sum(node.inputs[0].data))

class EWiseMul(TensorOp):
    def __call__(self,inputs):
        # inputs[Tensor,Scalar]
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        assert(len(inputs)==2)
        return inputs[0]*inputs[1]
    def gradient(self,out_grad,node):
        return (out_grad*node.inputs[1],out_grad*node.inputs[0])


class Reshape(TensorOp):
    def __call__(self,inputs,rshape):
        self.rshape=rshape
        assert(len(inputs)==1)
        self.oshape=inputs[0].shape
        return Tensor.make_from_op(op=self,inputs=inputs)
    
    def compute(self,inputs):
        #compute inputs ndarray
        assert(len(inputs)==1)
        return inputs[0].reshape(self.rshape)
    
    def gradient(self,out_grad,node):
        return (out_grad.reshape(self.oshape),)
    

class Transpose(TensorOp):
    def __call__(self,inputs,index):
        # example x.transpose([0,2,1])
        self.len = len(index)
        assert(self.len==len(inputs[0].shape))
        self.index=np.arange(self.len)[index]
        self.inverse_index=np.argsort(self.index)
        return Tensor.make_from_op(op=self,inputs=inputs)
    
    def compute(self,inputs):
        #compute inputs ndarray
        assert(len(inputs)==1)
        return inputs[0].transpose(self.index)
    
    def gradient(self,out_grad,node):
        return (out_grad.transpose(self.inverse_index),)

class Exp(TensorOp):
    def __call__(self,inputs):
        assert(len(inputs)==1)
        return Tensor.make_from_op(op=self,inputs=inputs)

    def compute(self,inputs):
        assert(len(inputs)==1)
        return np.exp(inputs[0])
    def gradient(self,out_grad,node):
        return (out_grad*exp(node.inputs[0]),)

def exp(x):
    return Exp()([x])


class Ln(TensorOp):
    def __call__(self,inputs):
        assert(len(inputs)==1)
        return Tensor.make_from_op(op=self,inputs=inputs)

    def compute(self,inputs):
        assert(len(inputs)==1)
        return np.log(inputs[0])
    def gradient(self,out_grad,node):
        return (out_grad/node.inputs[0],)

def log(x):
    return Ln()([x])


class PowerScalar(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        #[tensor,scalar] 
        return inputs[0]**inputs[1]
    def gradient(self,out_grad,node):
        a = node.inputs[1]
        x = node.inputs[0]
        gg=out_grad*(x**a).data*np.log(x.data)
        return (out_grad*a*(x**(a-1)),Tensor.make_const(np.sum(gg.data)))


class MatMul(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    
    def compute(self,inputs):
        # print("MATMUL compute on:",inputs[0].shape,inputs[1].shape)
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
    
def matmul(A,B):
    return MatMul()([A,B])    

class Negate(TensorOp):
    def __call__(self,inputs):
        return Tensor.make_from_op(op=self,inputs=inputs)
    def compute(self,inputs):
        
        assert(len(inputs)==1)
        return -inputs[0]
    
    def gradient(self, out_grad, node):
        # print("NEGATE G")
        return (-out_grad,)  

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
    

def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = (find_topo_sort_reverse(output_tensor)) # 请自行实现拓扑排序函数
    # print([x.id for x in reverse_topo_order])
    
    for node in reverse_topo_order:
        # print("Node {} Grads".format(node.id),node_to_output_grads_list[node])        
        node.grad = sum(node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        node_grads=node.op.gradient(node.grad, node)
        # print("Node gradients")
        # print(node_grads)
        for i, grad in enumerate(node_grads): # 计算node.inputs的partial adjoint
            # print("Op ",node.op)
            # print("cal grad ",i," grad is ",grad)
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict



    
class Tensor (Value):
    # grad: "Tensor" 
    cnt=0
    
    def __init__(self, op=None, inputs=[],requires_grad=True,  dtype=None, **kwargs):
        if (op is not None and not isinstance(op,TensorOp)): 
            # use Tensor(data) to init
            data=op
            self.op=None 
            self.inputs=[]
            self.requires_grad=False
            self.grad=None
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
    
    @property
    def T(self):
        assert(len(self.shape)==2)
        index = [-1,-2]
        return Transpose()([self],index)
    
    def transpose(self,index):
        return Transpose()([self],index)
    def detach(self):
        return Tensor.make_const(self.data)
    
    def backward(self,out_grad=None):
        if out_grad:
            out_grad=out_grad
        else:
            out_grad=Tensor.make_const(np.ones(self.shape))
        compute_gradient_of_variables(self,out_grad)
    def requires_grad_(self,req_grad):
        self.requires_grad=req_grad
        return self
    
    def __str__(self):
        return "Tensor( {})".format(self.data)
    def __repr__(self):
        return "Tensor( {})".format(self.data)
    def __neg__(self):
        return Negate()([self])
    
    def tensor_scalar_op(self,other,ops):
        # ops[TensorOp,ScalarOp,..]
        EWiseOp=ops[0]
        OpScalar=ops[1]
        
        if isinstance(other,Tensor):
            #print("EWiseAdd")
            if(self.shape==other.shape):
                return EWiseOp()([self,other])
            else:
                return OpScalar()([self,other])
        else:
            #print("ScalarAdd")
            other = Tensor.make_const(other,requires_grad=False)
            return OpScalar()([self,other])
    
    def __add__(self,other):
        return self.tensor_scalar_op(other,[EWiseAdd,AddScalar])
        
    def __radd__(self,other):
        return self.tensor_scalar_op(other,[EWiseAdd,AddScalar])
        
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
        
    def __mul__(self,other):
        return self.tensor_scalar_op(other,[EWiseMul,MulScalar])
    
    def __rmul__(self,other):
        return self.tensor_scalar_op(other,[EWiseMul,MulScalar])
    
    def __truediv__(self,other):
        return self.tensor_scalar_op(other,[EWiseDiv,DivScalar])
    
    def __rtruediv__(self,other):
        if isinstance(other,Tensor):
            if(self.shape==other.shape):
                return EWiseDiv()([other,self])
            else:
                inverse = EWiseDiv()([Tensor(np.ones(self.shape)),self])
                return MulScalar()([inverse,other])
        else:
            other = Tensor.make_const(other,requires_grad=False)
            inverse = EWiseDiv()([Tensor(np.ones(self.shape)),self])
            return MulScalar()([inverse,other])
     
    def __pow__(self,other):
        return PowerScalar()([self,other]) 
        
    def reshape(self,rshape):
        return Reshape()([self],rshape)