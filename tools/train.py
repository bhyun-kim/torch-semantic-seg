import sys 
import os
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(sys.path)

import argparse
import logging 
import json

import torch 
import torchvision as tv 

from importlib import import_module

from pipelines import *
from datasets import *
from models import *
from models.encoders import *
from models.decoders import * 
from models.heads import *
from torch.optim import *


from datetime import datetime

def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    args = parser.parse_args()

    return args


def cvt_pathToModule(file_path):
    """Convert path (string) to module form.
    
    Args :
        file_path (str) : file path written in nomal path form
    
    Returns :
        module_form (str) : file path in module form (i.e. matplotlib.pyplot)
    
    """
    
    file_path = file_path.replace('/', '.')
    module_form  = file_path.replace('.py', '')
    
    return module_form

def cvt_moduleToDict(mod) :
    """
    Args : 
        mod (module)  
    
    Returns :
        cfg (dict)
    
    """
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        }
    
    return cfg

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
    return globals()[cfg['type']](cfg['root_dir'], transform=transforms)

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

    if cfg:
        loss = globals()[loss_type](cfg)
    else:
        loss = globals()[loss_type]()
    
    return loss

def build_model(cfg):
    """Build model from config 
    Args: 
        cfg (dict): model configuration
    
    Returns:
        model (torch.nn.Module)
    """
    
    encoder_name = cfg['encoder'].pop('type')
    print(encoder_name)
    print(cfg['encoder'])
    # if cfg['encoder']:
    encoder = globals()[encoder_name](**cfg['encoder'])
    # else: 
        # encoder = globals()[encoder_name]()

    if cfg['decoder']:
        decoder_name = cfg['decoder'].pop('type')
        decoder = globals()[decoder_name]( **cfg['decoder'])
    else: 
        decoder = None 

    head_type = cfg['head'].pop('type')
    head = globals()[head_type](**cfg['head'])

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


def evaluate(model, dataloader, device, num_classes):
    # again no gradients needed
    model.eval()

    import numpy as np
    with torch.no_grad():
        category_vectors = []
        for idx, data in enumerate(dataloader):
            inputs, labels = data['image'], data['segmap']

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)


            category_vectors.append(predictions * num_classes + labels)

        
        category_vectors = torch.cat(category_vectors)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        category_vectors = category_vectors.detach().cpu().numpy()
        
        for i in range(num_classes ** 2):
            row, col = i // num_classes, i % num_classes
            _cat = category_vectors == i
            confusion_matrix[row, col] = np.count_nonzero(_cat) 

        IoUs = []
        for i in range(num_classes): 
            intersection = confusion_matrix[i, i]
            union = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
            IoUs.append(intersection / union)
        
        mIoU = np.mean(IoUs)
            
    model.train()
    return mIoU, IoUs


def main():
    args = parse_args()

    # build config 
    _cfg = args.config

    abs_path = osp.abspath(_cfg)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    cfg = cvt_moduleToDict(_mod)

    # build logger  
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    os.makedirs(cfg['WORK_DIR'], exist_ok=True)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = cfg['WORK_DIR'] + f'/{current_time}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    
    logger.info(json.dumps(cfg, indent=4))

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### build data loaders 
    data_loaders = build_loaders(cfg['DATA_LOADERS']) 

    # build loss 
    criterion = build_loss(cfg['LOSS'])

    # build model     
    model = build_model(cfg['MODEL'])    
    model.to(device)

    # build optimizer 
    optimizer = build_optimizer(cfg['OPTIMIZER'], model)

    # build runner -----------------------------------------------------
    for epoch in range(cfg['EPOCHS']): # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(data_loaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['segmap']

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

        evaluate(model, data_loaders['val'], device, 19) 
        if epoch % 5 == 1: 
            save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

        

    logger.info('Finished Training')
        
if __name__ == '__main__':
    main()