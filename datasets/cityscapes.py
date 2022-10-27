import cv2

import os.path as osp 

from glob import glob 
from torch.utils.data import Dataset, DataLoader

"""
References: 

[1] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

class CityscapesDataset(Dataset):
    """Cityscapes dataset."""

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args: 
            root_dir (str): Directory with all the images.
            split (str): The dataset split, supports 'train', 'val', or 'test'

        Folder structure:
            root_dir 
                └leftImg8bit
                    └ train/val/test
                        └ cities 
                            └ ***_leftImg8bit.png
                └gtFine
                    └ train/val/test
                        └ cities
                            └ ***_gtFine_labelTrainIds.png
                
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self.img_list = glob(osp.join(self.root_dir, 'leftImg8bit', self.split, '*', '*_leftImg8bit.png'))

        # class name 
        # class palette 
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Args: 
            idx (int): data index

        Returns:
            img (np.arr) 
            gt (np.arr) 
        
        """

        img_path = self.img_list[idx]
        gt_path = img_path.replace('leftImg8bit', 'gtFine')
        gt_path = gt_path.replace('.png', '_labelTrainIds.png')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        return img, gt 



            

            