import torch.nn as nn 
import torch.nn.functional as F 

class Interpolate(nn.Module):
    def __init__(self, 
                 size=None,
                 scale_factor=None, 
                 mode='nearest', 
                 align_corners=None,
                 recompute_scale_factor=None,
                 antialias=True
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
        print(self.args)

    def forward(self, input):
        output = F.interpolate(input=input, **self.args)
        return F.log_softmax(output, 1)