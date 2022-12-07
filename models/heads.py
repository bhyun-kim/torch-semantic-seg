import torch.nn as nn 
import torch.nn.functional as F 

class Interpolate(nn.Module):
    def __init__(self, 
                 loss,
                 size=None,
                 scale_factor=None, 
                 mode='nearest', 
                 align_corners=None,
                 recompute_scale_factor=None,
                 antialias=False
    ):
        """
        Args:
            
        """
        super().__init__()

        self.args_interpolate = dict(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )

        self.criterion = loss

    def forward(self, input, labels=None):
        output = F.interpolate(input=input, **self.args_interpolate)

        if labels == None: 
            return output 
        else: 
            return self.criterion(output, labels)
            
        