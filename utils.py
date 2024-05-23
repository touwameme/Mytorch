import numpy as np 

import torch              
def generate_linearly_separable_data(num_samples=16):
    # 生成随机分布的特征向量
    X = np.random.randn(num_samples, 3)
    
    # 生成随机分布的标签，使得数据是线性可分的
    y = np.zeros(num_samples)
    for i in range(num_samples):
        if X[i][0] + X[i][1] - X[i][2] > 0:
            y[i] = 1
        else:
            y[i] = 0
    
    return X, y


def set_seeds(seed):
    # 设置Python内置random模块的种子
    random.seed(seed)
    
    # 设置NumPy的种子
    np.random.seed(seed)
    
    # 设置Torch的种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 生成16个点的线性可分数据

