import torch.nn as nn 

class ModelWrapper(nn.Module):

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

        if self.decoder : 
            feat = self.decoder(feat)
        if self.head :
            feat = self.head(feat) 
        return feat