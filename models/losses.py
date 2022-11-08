import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self,
                weight=None,
                reduction='mean',
                ignore_idx=255):
        super().__init__()

        if weight: 
            weight = torch.tensor(weight)
        
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_idx, reduction='none')

        self.reduction = reduction
        self.ignore_idx = ignore_idx
        
    def forward(self, outputs, targets):
        # print(outputs.shape)
        # print(outputs[0, 5, 500:508, 800:808])
        softmax = F.softmax(outputs, dim=1)
        # print(softmax[0, :, 500, 800])
        loss = self.loss(outputs, targets)

        if self.reduction == 'mean':
            num_element = targets.numel() - (targets == self.ignore_idx).sum().item()
            loss = loss.sum() / num_element

        return loss