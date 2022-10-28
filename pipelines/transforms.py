
# To dos 
# [v] rescale 
# [v] random rescale 
# [v] random crop 
# [v] random flipLR
# [] random flipUD
# [] negative sample 

# [v] to tensor 
# [v] normalization 

import random

import cv2 
import torch

import numpy as np 


"""
References: 

[1] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
[2] https://mmcv-jm.readthedocs.io/en/stable/_modules/mmcv/image/normalize.html
"""



class Rescale(object): 
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size(H x W). If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

        assert isinstance(output_size, (int, tuple))
        # assert tuple must have two elements, height and width 
        self.output_size = output_size


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        segmap = cv2.resize(segmap, (new_w, new_h),  interpolation = cv2.INTER_NEAREST)

        return {'image': img, 'segmap': segmap}



class RandomRescale(object): 
    """Rescale the image to a randomly selected output size within the given range.
    """

    def __init__(self, output_range):
        """
        Args:
            output_range (tuple): Desired output range. 
        """

        assert isinstance(output_range, (tuple))
        self.output_range = output_range


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        output_size = random.randint(self.output_range[0], self.output_range[1])
        rescale = Rescale(output_size)
        return rescale(sample)



class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size(H x W). If int, square crop is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        segmap = segmap[top: top + new_h,
                        left: left + new_w]

        return {'image': image, 'segmap': segmap}



class RandomFlipLR(object):
    """
    Horizontally flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        if np.random.rand() < self.prob:
            # flip image
            image = cv2.flip(image, 1)
            # flip segmap
            segmap = cv2.flip(segmap, 1)

        return {'image': image, 'segmap': segmap}



class RandomFlipUD(object):
    """
    Vertically flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        if np.random.rand() < self.prob:
            # flip image
            image = cv2.flip(image, 0)
            # flip segmap
            segmap = cv2.flip(segmap, 0)

        return {'image': image, 'segmap': segmap}


class Normalization(object):
    """Normalize image 
    """
    def __init__(self, mean, std):
        """
        Args:
            mean (tuple, list): (R, G, B)
            std (tuple, list): (R, G, B)
        """
        assert isinstance(mean, (tuple, list))
        assert isinstance(std, (tuple, list))

        mean, std = np.array(mean), np.array(std)
        self.mean = np.float64(mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(std.reshape(1, -1))
        print(f'mean: {self.mean}, stdinv: {self.stdinv}')
        

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, float32), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']
        image = np.float32(image) if image.dtype != np.float32 else image.copy()

        cv2.subtract(image, self.mean, image)
        cv2.multiply(image, self.stdinv, image)

        return {'image': image, 'segmap': segmap}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C), segmap: np.arr (H x W)})
        
        Returns:
            sample (dict, {image: torch.tensor (C x H x W), segmap: torch.tensor (H x W)})
        """
        image, segmap = sample['image'], sample['segmap']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'segmap': torch.from_numpy(segmap)}



