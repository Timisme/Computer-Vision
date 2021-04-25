import os, torch, torchvision, random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torch.optim import lr_scheduler
from torchsummary import summary
import time

class Dataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        
        self.filenames = filenames # 資料集的所有檔名
        self.labels = labels # 影像的標籤
        self.transform = transform # 影像的轉換方式
 
    def __len__(self):
        
        return len(self.filenames) # return DataSet 長度
 
    def __getitem__(self, idx):
        
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        label = np.array(self.labels[idx])
                
        return image, label # return 模型訓練所需的資訊


normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 1.2. 填入 ??? 的部份

def split_Train_Val_Data(data_dir, batch_size= 32):
    
    dataset = ImageFolder(data_dir) 
    
    # 建立 20 類的 list
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        
        np.random.seed(42)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------
        
        num_sample_train = 0.8
        num_sample_test = 0.2
        
        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))
        
        for x in data[:int(len(data)*num_sample_train)] : # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[int(len(data)*num_sample_train):] : # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(Dataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True, num_workers= 4)
    test_dataloader = DataLoader(Dataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False, num_workers= 4)
 
    return train_dataloader, test_dataloader

if __name__ == '__main__':
     data_dir = 'dataset/PetImages'

     dataset =ImageFolder(data_dir)

     # print(dataset.classes)

     train_dataloader, test_dataloader = split_Train_Val_Data(data_dir= data_dir, batch_size= 32)

     print(next(iter(train_dataloader)))