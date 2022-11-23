from models import *
from models.encoders import *
from models.decoders import * 
from models.heads import *

from .utils import build_from_cfg

def build_model(cfg):
    """Build model from config 
    Args: 
        cfg (dict): model configuration
    
    Returns:
        model (torch.nn.Module)
    """
    
    encoder = build_from_cfg(cfg['encoder'], globals())

    if cfg['decoder']:
        decoder = build_from_cfg(cfg['decoder'], globals())
    else: 
        decoder = None 

    if cfg['head']:
        head = build_head(cfg['head'])
    else:
        head = None

    return ModelWrapper(encoder=encoder, decoder=decoder, head=head)



def build_head(cfg):
    """Build segmentation head 
    Args: 
        cfg (dict): Segmentation head configuration 

    Returns: 
        model (torch.nn.Module)
    """

    # replace loss with loss object
    loss = build_from_cfg(cfg['loss'], globals())
    cfg['loss'] = loss

    return build_from_cfg(cfg, globals())