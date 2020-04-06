import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from   PIL import Image
import matplotlib.pyplot as plt
from   torch.optim import lr_scheduler
import os,argparse,time
import scipy.ndimage.morphology as morph
import numpy as np
from torch.utils.data import  DataLoader
from torchsummary import summary
from torchvision.transforms.functional import to_tensor,to_pil_image
from albumentations import (HorizontalFlip, VerticalFlip, Compose, Resize,Normalize)
import copy
from  Unet import Unet
from  dataset import DataSet
from  ce_loss import *

class TrainUnet(object):
  def __init__(self,args,saved_weights):
    self.args = args
    self.transforms_train=Compose([ Resize(256,256),
                           HorizontalFlip(0.5),  
                           VerticalFlip (0.5),   
                           Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                          ]) 
    self.transforms_validation =Compose([ Resize(256,256),
                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                      
                         ])
    self.train_dataset=DataSet('train',self.args.path,self.transforms_train)
    self.validation_dataset=DataSet('validation',self.args.path,self.transforms_validation) 
    self.best_loss=0
    self.model=Unet(3,21)    
    self.best_model_wts=copy.deepcopy(self.model.state_dict())
    
    ####
    if saved_weights:
       self.model.load_state_dict(torch.load(self.args.path+'/Unet_results'+'/training_results.pt'))
    ####   
    self.model.cuda()
    self.train_hist= {'tot_loss_validation': [],                      
                      'tot_loss_train': [],
                      'bce_loss_validation': [],                      
                      'bce_loss_train': [],
                      'dice_loss_validation': [],                      
                      'dice_loss_train': [],
                      'per_epoch_time':[],
                      'total_time':[]                     
                      }

  def train_model(self,optimizer,loss,t_loader):  #scheduler,    
      self.model.train()      
      train_loss,train_bce_loss,train_dice_loss,num_samples=0,0,0,0

      for iter,cur_batch in enumerate(t_loader):           
          inputs,labels=cur_batch
          inputs,labels=inputs.cuda(),labels.cuda()  
          optimizer.zero_grad()
          predicted=self.model(inputs)
          bce_loss,d_loss,tot_loss=loss(predicted,labels)
          tot_loss.backward()
          optimizer.step()
          tot_loss.detach()
          train_loss+=tot_loss*len(inputs)
          train_bce_loss+=bce_loss*len(inputs)
          train_dice_loss+=d_loss*len(inputs)
          num_samples+=len(inputs)
      return train_loss/num_samples,train_bce_loss/num_samples,train_dice_loss/num_samples

  def evaluate_model(self,loss,t_loader):
      self.model.eval()
      val_loss,val_bce_loss,val_dice_loss,num_samples=0,0,0,0
      with torch.no_grad():
        for iter,cur_batch in enumerate(t_loader):
            inputs,labels=cur_batch
            inputs,labels=inputs.cuda(),labels.cuda()
            predicted=self.model(inputs)
            bce_loss,d_loss,tot_loss=loss(predicted,labels)
            val_loss+=tot_loss*len(inputs)
            val_bce_loss+=bce_loss*len(inputs)
            val_dice_loss+=d_loss*len(inputs)
            num_samples+=len(inputs)
            if  val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
      return val_loss/num_samples,val_bce_loss/num_samples,val_dice_loss/num_samples

  def plot_results(self):
      plt.figure()
      train_loss=plt.plot(self.train_hist['tot_loss_train'])
      val_loss=plt.plot(self.train_hist['tot_loss_validation'])      
      plt.legend([train_loss, val_loss], ['train', 'val'], loc='upper right')
      plt.title('Loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.savefig(self.args.path+'/Unet_results'+'/loss.png', bbox_inches='tight' )

  def update_history(self,train_l,train_bce_l,train_dice_l,val_l,val_bce_l,val_dice_l,epoch_run_time):  
      self.train_hist['tot_loss_train'].append(train_l)
      self.train_hist['tot_loss_validation'].append(val_l)   

      self.train_hist['bce_loss_train'].append(train_bce_l)
      self.train_hist['bce_loss_validation'].append(val_bce_l) 

      self.train_hist['dice_loss_train'].append(train_dice_l)
      self.train_hist['dice_loss_validation'].append(val_dice_l)  

      self.train_hist['per_epoch_time'].append(epoch_run_time) 

  def run(self):
      optimizer=optim.Adam(self.model.parameters(),lr=self.args.lrG,betas=(self.args.beta1,self.args.beta2) )
      loss=combined_loss(0.84)
      start_time=time.time()
      print('---Start training---')
      
      train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, num_workers=4,shuffle=True)    
      validation_loader = DataLoader(dataset=self.validation_dataset, batch_size=self.args.batch_size, num_workers=4,shuffle=True) 
      for epoch in range(self.args.num_epochs):
          print("epoch num",epoch)
          epoch_start_time=time.time()
          t_loss,t_bce_loss,t_dice_loss=self.train_model(optimizer,loss,train_loader)      ##scheduler,    
          v_loss,v_bce_loss,v_dice_loss=self.evaluate_model(loss,validation_loader)
          print("%0.2f train loss"% (v_loss))
          self.update_history(t_loss,t_bce_loss,t_dice_loss,v_loss,v_bce_loss,v_dice_loss,(time.time()-epoch_start_time))
      print('---End training---')    
      self.train_hist['total_time'].append((time.time()-start_time))   
      self.plot_results()
      torch.save(self.best_model_wts, self.args.path+'/Unet_results'+'/training_results.pt')
      self.model.load_state_dict(self.best_model_wts) 
      return self.model
      
           
def parse_arguments(): 
     parser=argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--num_epochs', type=int, default=35, help='The number of epochs to run') 
     parser.add_argument('--lrG', type=float, default=0.0001)
     parser.add_argument('--beta1', type=float, default=0.9)
     parser.add_argument('--beta2', type=float, default=0.999)
     parser.add_argument('--batch_size', type=int, default=16) 
     parser.add_argument('--path', type=str, default='/content/gdrive/My Drive/gdrive') 
     return parser.parse_args()


def main(saved_w):
    args=parse_arguments()
    u_net=TrainUnet(args,saved_w)  
    my_model=u_net.run() 
    return my_model 

if __name__=='__main__':
    my_model=main(False)



