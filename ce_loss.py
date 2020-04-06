import torch
from torch import nn


class CE_Loss(nn.Module):    
    def __init__(self):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, predicted, target):
        predict = predicted
        target = target
        return self.ce_loss(predict, target)

def dice_loss(predicted,target,smooth=0.01):    
    batch_size=predicted.size()[0] 
    predicted=torch.argmax(predicted,dim=1)
    target_mask=(target>0).type(torch.IntTensor)
    predicted_mask=(predicted>0).type(torch.IntTensor)    

    predicted_mask=predicted_mask.view(batch_size,-1)
    target_mask=target_mask.view(batch_size,-1)
 
    tot=(predicted_mask*target_mask).sum(-1)

    loss=1-(2*tot+smooth)/(predicted_mask.sum(-1)+target_mask.sum(-1)+smooth)    
    return loss.mean().item()

class combined_loss(nn.Module):
    def __init__(self,ce_weight):
        super(combined_loss, self).__init__()
        self.ce_loss = CE_Loss() 
        self.weight=ce_weight

    def forward(self,predicted, target):        
        ce_loss=self.ce_loss(predicted,target)
        dc_loss=dice_loss(predicted,target,self.weight)
        loss = ce_loss* self.weight + dc_loss * (1 - self.weight)
        return ce_loss,dc_loss,loss 
