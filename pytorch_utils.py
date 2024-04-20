from torch.utils.data import Dataset
import torch
import numpy as np


class FramesDataset(Dataset):

    def __init__(self, im_dir, moments_dir, locs_dir, shape=(40, 40), transform=None):
        
        self.moments = np.load(moments_dir).astype(np.float32)
        self.locs = np.tile(np.load(locs_dir).astype(np.float32), self.moments.shape[1]).reshape(-1,2)/shape[0]
        self.moments = self.moments.reshape(-1,6)
        self.target = np.hstack([self.moments, self.locs])
        self.image = np.load(im_dir).astype(np.float32).reshape(-1, 1, *shape)
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.target[idx]
        image = self.image[idx]
        
        sample = {'image': image, 
                  'target': target,
                 }

        if self.transform:
            sample = self.transform(sample)
        return sample

    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target, minmax = sample['image'], sample['target'], sample['minmax']
        
        if len(image.shape) == 2:    
            image = image.reshape(1, image.shape[0], image.shape[1])
        
        return {'image': torch.from_numpy(image.astype(np.float32)),
                'target': torch.from_numpy(target.astype(np.float32)),
                'minmax': minmax
               
               }
    
    
class MinMaxNorm(object):
    """Normalizes images."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        
        image_norm = (image-image.min())/(image.max()-image.min())
        moments, locs = np.hsplit(target, [6]) # locations have been already normalized
        a = moments.min()
        b = moments.max()
        moments_norm = (moments - a)/(b - a)
        target = np.hstack([moments_norm, locs])
        
        return {'image': image_norm,
                'target': target,
                'minmax': (a,b)}
    
    
 