

import numpy as np
from typing import Union, Tuple, List
from abc import ABC, abstractmethod
from basic import Value,Tensor

# x=Value.make_const([1])
# y=Value.make_const([2])
# z=Value.make_from_op(AddOp(),[x,y])
# print(z.realize_cached_data())

x=Tensor.make_const([1,2])
y=Tensor.make_const([2,3])
# z=Tensor.make_const([4,6])
y=Tensor.make_const([2])
r=x+y
# r = x+y+z
print(r.data)