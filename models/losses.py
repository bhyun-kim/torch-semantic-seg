import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self,
                weight=None,
                reduction='mean',
                ignore_idx=255):
        super().__init__()

        self.args = dict(
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_idx
        )

    def forward(self, 
                cls_score,
                label):

        loss_cls = F.cross_entropy(cls_score, 
                                   label,
                                   **self.args)

        return loss_cls