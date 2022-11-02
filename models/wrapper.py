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
        
        enc_feat = self.encoder(input)

        if self.decoder : 
            dec_feat = self.decoder(enc_feat)
            feat_out = self.head(dec_feat) 
        else : 
            feat_out = self.head(enc_feat)

        return feat_out