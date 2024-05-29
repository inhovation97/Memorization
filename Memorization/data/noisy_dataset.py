from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import cv2
from typing import List, Tuple
import glob
import os
from torchvision import transforms
import torch
import numpy as np
import random
from collections import defaultdict
import json

class Noisy_dataset(Dataset):
    
    def __init__(self, json_path, transform):
        with open( json_path, 'r' ) as f:
            data_dict = json.load(f)
        
        self.true_label_list = data_dict['true_label']
        self.noisy_label_list = data_dict['noisy_label']
        self.image_path_list = data_dict['image_path']
        self.transform = transform
               
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()

        # image data
        data_path = self.image_path_list[index]
        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"] if self.transform else Image

        # label and path
        true_label = self.true_label_list[index]
        noisy_label = self.noisy_label_list[index]
        image_path = self.image_path_list[index]
        
        # output dict
        output_dict={}
        output_dict['batch_image'] = image
        output_dict['true_label'] = int(true_label)
        output_dict['noisy_label'] = int(noisy_label)
        output_dict['image_path'] = image_path
    
        return output_dict
       
    def __len__(self):
        length = len(self.true_label_list)
        return length
        