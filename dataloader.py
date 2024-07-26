import nibabel as nib
from torch.utils.data import Dataset
import os
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
import numpy as np

class TumorDataset(Dataset):
    def __init__(self, config_path, mutation=True, transform=ToTensor()):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        self.image_dir = cfg.paths.mutation_dir.images if mutation else cfg.paths.no_mutation_dir.images
        self.mask_dir = cfg.paths.mutation_dir.masks if mutation else cfg.paths.no_mutation_dir.masks
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.nii')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.nii')])
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = np.array(nib.load(image_path).get_fdata())
        mask = np.array(nib.load(mask_path).get_fdata())
        
        # rotate, augment, anything needed
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask
    
# Usage:

# a = TumorDataset(config_path='config/config.yaml')
# for b, c in a:
#     plt.imshow(b[12])
#     plt.show()