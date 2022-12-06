import torch.nn as nn 

from . import FrameRegistry 

@FrameRegistry.register('ModelFramer')
class ModelFramer(nn.Module):

    def __init__(self,
                 encoder,
                 head,
                 decoder=None,
                 ):
        super().__init__()

        self.encoder = encoder 
        self.decoder = decoder 
        self.head = head 

    def forward(self, input):
        
        feat = self.encoder(input)

        if self.decoder: 
            feat = self.decoder(feat)
            
        feat = self.head(feat) 
        
        return feat