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

    def forward(self, input, labels=None):
        
        enc_output = self.encoder(input)

        if self.decoder : 
            dec_output = self.decoder(enc_output)
        else: 
            dec_output = enc_output

        if self.head :
            output = self.head(dec_output, labels) 
        else: 
            output = dec_output
            
        return output
