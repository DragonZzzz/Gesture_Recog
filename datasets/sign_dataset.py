import numpy as np
import torch 
from torch.utils import data
from torchvision import transforms
from PIL import Image
import sys 
import os
import copy
from sklearn.model_selection import train_test_split
sys.path.append("..") 
import config
import skimage.io as io 


def make_train_test_split(data_path, test_size=0.1):
    '''
    按test_size的比例随机划分训练集和测试集
    '''
    image_list = []
    label_list = []
    gesture_list = os.listdir(data_path)
    label_idx = 0
    for gesture in gesture_list:
        # print(gesture)
        file_path = os.path.join(data_path, gesture)
        file_list = os.listdir(file_path)
        for file_name in file_list:
            #只处理其中的jpg图片，忽略json文件
            if '.jpg' in file_name:
                file_dir = os.path.join(file_path, file_name)
                # print(file_dir, label_id)
                image_list.append(file_dir)
                label_list.append(label_idx)
        label_idx += 1
    x_train,x_test,y_train,y_test = train_test_split(image_list,label_list,test_size = 0.1)
    return x_train,x_test,y_train,y_test



class TrainDataset(data.Dataset):
    def __init__(self, image_list, label_list):
        self.img_w = config.img_w
        self.img_h = config.img_h
        self.num_classes = config.num_classes
        self.data_list = []
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.image_list = copy.deepcopy(image_list)
        self.label_list = copy.deepcopy(label_list)

        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        file_dir = self.image_list[index]
        label  = self.label_list[index]
        image = self.transform(Image.open(file_dir).convert('RGB'))
        return image, label

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop((config.img_w, config.img_h)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

class TestDataset(data.Dataset):
    def __init__(self, image_list, label_list):
        self.img_w = config.img_w
        self.img_h = config.img_h
        self.num_classes = config.num_classes
        self.data_list = []
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.image_list = copy.deepcopy(image_list)
        self.label_list = copy.deepcopy(label_list)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        file_dir = self.image_list[index]
        label  = self.label_list[index]
        image = self.transform(Image.open(file_dir).convert('RGB'))
        return image, label


    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop((config.img_w, config.img_h)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

