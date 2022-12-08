from . import HeadRegistry

import torch.nn as nn 
import torch.nn.functional as F 

@HeadRegistry.register('Interpolate')
class Interpolate(nn.Module):
    def __init__(self, 
<<<<<<< HEAD:models/heads/heads.py
                 criterion,
=======
                 loss,
>>>>>>> upstream/develop:models/heads.py
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

        self.args_interpolate = dict(
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

<<<<<<< HEAD:models/heads/heads.py
    def predict(self, input):
        return F.interpolate(input=input, **self.args)
=======
        self.criterion = loss

    def forward(self, input, labels=None):
        output = F.interpolate(input=input, **self.args_interpolate)

        if labels == None: 
            return output 
        else: 
            return self.criterion(output, labels)
            
        
>>>>>>> upstream/develop:models/heads.py
