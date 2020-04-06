import torch
from torchvision.datasets import VOCSegmentation
import torchvision
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor,to_pil_image


class DataSet(VOCSegmentation):
     def __init__(self,mode,path,transform=None):
       self.transformation = transform  
       self.path=path 
       self.mode=mode
       self.img_paths=os.listdir(self.path+'/data/pascal_'+self.mode+'/images')      
       
       
     def __getitem__(self,idx):
        img = Image.open(self.path+'/data/pascal_'+self.mode+'/images' + "/" + self.img_paths[idx]).convert('RGB')   
        target = Image.open(self.path+'/data/pascal_'+self.mode+'/seg_class' + "/" + self.img_paths[idx].split('.')[0]+'.png' )
        if self.transformation is not None:
           augmented = self.transformation(image=np.array(img),mask=np.array(target))
           img=augmented['image']
           target=augmented['mask']
           target[target>20]=0 
        img=to_tensor(img)
        target=torch.from_numpy(target).type(torch.long)
        return img,target 

     def __len__(self):
        return len(self.img_paths)
