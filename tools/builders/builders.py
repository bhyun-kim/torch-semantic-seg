import sys 
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
print(sys.path)

import torch
import torchvision as tv 

from pipelines import *
from datasets import *
from models import *
from models.encoders import *
from models.decoders import * 
from models.heads import *

from torch.optim import *
from torch.optim.lr_scheduler import *

from runners.supervised_learning import *


def build_pipelines(cfg_pipelines):
    """Build pipelines from config

    Args: 
        cfg_pipelines (list): sequence of pipeline configurations

    Returns:
        transforms (torchvision.transforms.Compose)
    
    """

    pipelines = [] 

    for pipeline in cfg_pipelines: 
        name = pipeline.pop('type')
        pipelines.append(globals()[name](**pipeline))

    return tv.transforms.Compose(pipelines)

def build_dataset(cfg, transforms=None):
    """Build dataset from config 
    Args: 
        cfg (dict): dataset configuration 
        transforms (torchvision.transforms)

    Returns: 
        dataset (torch.utils.data.Dataset)
    """
    if cfg['type'] == 'ConcatDataset': 
        dataset = _concat_dataset(cfg, transforms=transforms)
    else:  
        dataset = _build_dataset(cfg, transforms=transforms)
    return dataset

def _concat_dataset(cfg, transforms=None):
    """Concat datasets in cfg 
    Args: 
        cfg (dict): dataset configuration 
        transforms (torchvision.transforms)

    Returns: 
        dataset (torch.utils.data.ConcatDataset)

    """
    datasets = []

    for dataset in cfg['datasets']:
        datasets.append(_build_dataset(dataset, transforms=transforms))
        

    return torch.utils.data.ConcatDataset(datasets)

def _build_dataset(cfg, transforms=None):
    """Build dataset from config 
    Args: 
        cfg (dict): dataset configuration 
        transforms (torchvision.transforms)

    Returns: 
        dataset (torch.utils.data.Dataset)
    """
    dataset_type = cfg.pop('type')
    return globals()[dataset_type](**cfg, transform=transforms)

def build_data_loader(cfg, dataset):
    """Build data loader from config
    Args:
        cfg (dict): data loader configuration 
        dataset (torch.utils.data.Dataset)

    Returns: 
        data_loader (torch.utils.data.DataLoader)
    """
    
    return torch.utils.data.DataLoader(dataset, **cfg)


def build_loss(cfg):
    """Build loss from config 
    Args:
        cfg (dict): loss configuration 
    
    Returns: 
        loss (torch.nn.Module)
    """
    loss_type = cfg.pop('type')

    return globals()[loss_type](**cfg)

def build_model(cfg):
    """Build model from config 
    Args: 
        cfg (dict): model configuration
    
    Returns:
        model (torch.nn.Module)
    """
    
    encoder_name = cfg['encoder'].pop('type')
    # if cfg['encoder']:
    encoder = globals()[encoder_name](**cfg['encoder'])
    # else: 
        # encoder = globals()[encoder_name]()

    if cfg['decoder']:
        decoder_name = cfg['decoder'].pop('type')
        decoder = globals()[decoder_name]( **cfg['decoder'])
    else: 
        decoder = None 

    if cfg['head']:

        head_type = cfg['head'].pop('type')
        head = globals()[head_type](**cfg['head'])
    else:
        head = None

    return ModelWrapper(encoder=encoder, decoder=decoder, head=head)



def build_optimizer(cfg, model):
    """Build optimizer from config 
    Args: 
        cfg (dict): optimizer configuration
        model (torch.nn.Module)
    
    Returns:
        optimizer (torch.optim)
    """

    optim_type = cfg.pop('type')

    return globals()[optim_type](model.parameters(), **cfg)


def build_loaders(cfg):
    """Build data loaders from config 
    Args: 
        cfg (dict): data loaders configuration
        model (torch.nn.Module)
    
    Returns:
        optimizer (torch.optim)
    """
    loaders = dict()
    splits = [k for k, v in cfg.items()]
    for split in splits:
        _cfg = cfg[split]
        
        # build pipelines 
        transforms = build_pipelines(_cfg['pipelines'])

        # build dataset 
        dataset = build_dataset(_cfg['dataset'], transforms=transforms)
        
        # build data loader 
        loaders[split] = build_data_loader(_cfg['loader'], dataset) 

    return loaders 


def build_runner(cfg):
    """Build runner from config 
    Args: 
        cfg (dict): runner configuration
    
    Returns:
        runner (obj)
    """

    runner_type = cfg.pop('type')

    return globals()[runner_type](**cfg)

def build_lr_config(cfg, optim):
    """Build lr_config
    Args: 
        cfg (dict): learning rate config
        optim (torch.optim)
    
    Returns:
        torch.optim.lr_scheduler 
    """

    lr_scheduler_type = cfg.pop('type')

    return globals()[lr_scheduler_type](optimizer=optim, **cfg)
