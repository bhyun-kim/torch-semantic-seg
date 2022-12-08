import torch
import torchvision as tv 

from pipelines import *
from datasets import *

from .utils import build_from_cfg

from torch.utils.data import *


def build_pipelines(cfg_pipelines):
    """Build pipelines from config

    Args: 
        cfg_pipelines (list): sequence of pipeline configurations

    Returns:
        transforms (torchvision.transforms.Compose)
    
    """

    pipelines = [] 

    for pipeline in cfg_pipelines: 
        pipelines.append(build_from_cfg(pipeline, globals()))

    return tv.transforms.Compose(pipelines)



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
        dataset['transforms'] = transforms
        datasets.append(build_from_cfg(dataset, globals()))
        
    return ConcatDataset(datasets)


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
        cfg['transforms'] = transforms
        dataset = build_from_cfg(cfg, globals())

    return dataset

def build_data_loader(cfg, dataset):
    """Build data loader from config
    Args:
        cfg (dict): data loader configuration 
        dataset (torch.utils.data.Dataset)

    Returns: 
        data_loader (torch.utils.data.DataLoader)
    """
    return DataLoader(dataset, **cfg)


def build_loaders(cfg, rank, num_replicas=0):
    """Build data loaders from config 
    Args: 
        cfg (dict): data loaders configuration
        rank (int): Processor identifier
            
    Returns:
        DataLoader (torch.DataLoader)
    """
    loaders = dict()
    splits = [k for k, v in cfg.items()]
    for split in splits:
        _cfg = cfg[split]
        
        # build pipelines 
        transforms = build_pipelines(_cfg['pipelines'])

        # build dataset 
        dataset = build_dataset(_cfg['dataset'], transforms=transforms)
        
        # build samler
        if num_replicas > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
            _cfg['loader']['sampler'] = sampler
            _cfg['loader']['shuffle'] = False

        # build data loader 
        loaders[split] = build_data_loader(_cfg['loader'], dataset) 

    return loaders 