from ..basic import Tensor 


from abc import ABC, abstractmethod

import numpy as np

class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self):
        pass
    

import pandas as pd
class CsvDataset(Dataset):
    def __init__(self,csv_file,label_dim=-1):
        data_frame = pd.read_csv(csv_file)
        raw_data = data_frame.to_numpy()
        self.labels=raw_data[:,label_dim]
        self.data = np.hstack((raw_data[:, :label_dim], raw_data[:, label_dim+1:]))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        feature = Tensor(self.data[idx,:]).float()
        label = Tensor(self.labels[idx]).long()
        return feature,label
    
import os
import numpy as np
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        labels = []
        for cls_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(self.root, cls)
            for file in os.listdir(class_dir):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(class_dir, file)
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        img = np.asarray(img)
                    images.append(img)
                    labels.append(cls_idx)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label

    
    
    
class DataLoader(ABC):
    def __init__(self,dataset,batch_size,shuffle=False):
        self.batch_size=batch_size
        self.dataset=dataset
        self.shuffle=shuffle

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        # 按照batch_size拆分数据集索引
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if(i+self.batch_size>len(self.dataset)):
                batch_indices = indices[i:]
            # 根据索引获取对应的样本并返回批次
            batch = self.dataset[batch_indices]
            yield batch