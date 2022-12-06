from .cityscapes import CityscapesDataset
from registry import Registry

DatasetRegistry = Registry()

__all__ = [
    'CityscapesDataset'
]