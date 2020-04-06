import os,argparse,time
from  dataset import DataSet
from torchvision.transforms.functional import to_tensor,to_pil_image
from albumentations import (HorizontalFlip, VerticalFlip, Compose, Resize,Normalize)
from torch.utils.data import  DataLoader
from Unet import Unet
import torch
import matplotlib.pyplot as plt


mean=[0.485 , 0.456, 0.406]
std=[0.229,0.224,0.225]

def re_normalize(x, mean=mean,std=std):
    print(type(x))
    input=x.clone()
    for channel ,(mean_c,std_c) in enumerate(zip(mean,std)):        
        input[channel]*=std_c
        input[channel]+=mean_c
    return input   


def test_model(model,test_loader):
    inputs_1, labels_1 = next(iter(test_loader))
    inputs_1 = inputs_1.cuda() 
    labels_1 = labels_1.cuda() 

    # Predict
    pred_1 = model(inputs_1)
    pred_1 = torch.argmax(pred_1,axis=1)
    pred_1=pred_1.type(torch.LongTensor)
    return inputs_1,labels_1,pred_1

def parse_arguments(): 
     parser=argparse.ArgumentParser()
     parser.add_argument('--path', type=str, default='your_path_') 
     return parser.parse_args() 

if __name__ == '__main__':
    transforms_test =Compose([ Resize(256,256),
                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                      
                         ])
    args=parse_arguments()  
    batch_s=3                   
    test_dataset= DataSet('test',args.path,transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False, num_workers=0)
    model = Unet(3,21)
    model=model.cuda()
    
    model.load_state_dict(torch.load(args.path+'/Unet_results'+'/training_results.pt'))
    inputs,labels,predctions=test_model(model, test_loader)    
    inputs=inputs.cpu()
    labels=labels.cpu()
    predctions = predctions.cpu()
    fig=plt.figure(figsize=(10,10))
    plt.clf() 
    columns = 3
    rows = batch_s
    for i in range(0,columns*rows):  
      if i%3==0: 
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(to_pil_image(re_normalize((inputs[i//3]))))
      if i%3==1:  
        fig.add_subplot(rows, columns, i+1)
        plt.imshow((((labels[i//3]))))
      if i%3==2:
        fig.add_subplot(rows, columns, i+1)
        plt.imshow((((predctions[i//3]))))
    plt.show()  
    plt.savefig(args.path+'/Unet_results'+'/test_pics.png', bbox_inches='tight' )
  
      

    
