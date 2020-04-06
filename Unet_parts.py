import torch
import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class conv_block(nn.Module):
    def __init__(self,channels_in, channels_out):
        super(conv_block,self).__init__()
        self.conv_b=nn.Sequential(
                      nn.Conv2d(channels_in, channels_out,3,padding=1),
                      nn.BatchNorm2d( channels_out),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(channels_out,channels_out,3,padding=1),
                      nn.BatchNorm2d(channels_out),
                      nn.ReLU(inplace=True) ) 

    def forward (self,x):
          output= self.conv_b(x)
          return output  

class contraction_block(nn.Module): 
      def __init__(self,channels_in, channels_out):
        super(contraction_block,self).__init__() 
        self.block=nn.Sequential(nn.MaxPool2d(2),
                                       conv_block(channels_in, channels_out))
      def forward(self,x):
          output=self.block(x)
          return output

class expantion_block(nn.Module):
      def __init__(self, channels_in, channels_out):
          super(expantion_block,self).__init__()
          self.conv_t=nn.ConvTranspose2d(channels_in//2,channels_in//2,2,stride=2) 
          self.conv=conv_block(channels_in, channels_out)  

      def stack(self, x_1,x_2):
          dy  = x_2.size()[2] - x_1.size()[2]
          dx  =  x_2.size()[3] - x_1.size()[3]
          x_1 = F.pad(x_1, (dx // 2, dx - dx//2, dy // 2, dy - dy//2))
          output = torch.cat([x_2, x_1], dim=1)
          return output

      def forward(self,x_1,x_2):
          output=self.conv_t(x_1)
          output=self.stack(output,x_2)
          output=self.conv(output)  
          return output         
