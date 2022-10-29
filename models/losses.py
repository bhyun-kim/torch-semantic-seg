import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                cls_score,
                label,
                weight=None,
                reduction='mean',
                ignore_idx=255):

        loss_cls = F.cross_entropy(cls_score, 
                                   label,
                                   weight=weight,
                                   reduction=reduction,
                                   ignore_index=ignore_idx)

        return loss_cls