from . import HeadRegistry

import torch.nn as nn 
import torch.nn.functional as F 

@HeadRegistry.register('Interpolate')
class Interpolate(nn.Module):
    def __init__(self, 
                 criterion,
                 size=None,
                 scale_factor=None, 
                 mode='nearest', 
                 align_corners=None,
                 recompute_scale_factor=None,
                 antialias=False
    ):
        """
        Args:
            size 
            scale_factor 
            mode (str) 
            align_corners (bool) 
            recompute_scale_factor () 
            antialias (bool) 
            
        """
        super().__init__()

        self.args = dict(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )
        self.criterion = criterion

    def forward(self, input, labels):
        output = self.predict(input)
        return self.criterion(output, labels)

    def predict(self, input):
        return F.interpolate(input=input, **self.args)