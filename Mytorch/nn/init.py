import numpy as np
import sys 
sys.path.append(".")
sys.path.append("..")
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
from ..basic import Tensor
from .Modules import Parameter,Module
from .layers import *

def init_He (in_features, out_features, dtype="float32"):
    he_stddev=np.sqrt(2/in_features)
    array=(np.random.randn(in_features,out_features)*he_stddev).astype(dtype)
    params=Parameter(Tensor(array))
    return params

def init_Xavier(in_features, out_features, dtype="float32"):
    xavier_stddev = np.sqrt(2 / (in_features + out_features))
    initial_weights = np.random.randn(in_features,out_features) * xavier_stddev
    return initial_weights