from . import HeadRegistry

import torch.nn as nn 
import torch.nn.functional as F 

@HeadRegistry.register('Interpolate')
class Interpolate(nn.Module):
    def __init__(self, 
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

        self.args = dict(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )

    def forward(self, input):
        output = F.interpolate(input=input, **self.args)
        return output