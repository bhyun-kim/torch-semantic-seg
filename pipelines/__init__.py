from .transforms import * 
from registry import Registry

PipelineRegistry = Registry()

__all__ = [
    'Rescale', 'RandomRescale', 'RandomCrop', 'Normalization', 
    'ToTensor', 'RandomFlipLR'
]