import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
import numpy as np

class TumorDataset(Dataset):
    def __init__(self, config_path, transform=ToTensor(), mutation=True):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        self.image_dir = cfg.paths.mutation_dir.images if mutation else cfg.paths.no_mutation_dir.images
        self.mask_dir = cfg.paths.mutation_dir.masks if mutation else cfg.paths.no_mutation_dir.masks
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.nii')]
        self.mask_files = [image_file.split('.')[0] + '_label.nii' for image_file in self.image_files]
        print(self.image_files, self.mask_files)
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

def make_loader(config_path, transform=ToTensor(), batch_size=4, mutation=True, shuffle=True, num_workers=4):
    dataset = TumorDataset(config_path, mutation=mutation, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader

# Usage:
# No batching:
# a = TumorDataset(config_path='config.yaml')
# for b, c in a:
#     plt.imshow(b[12])
#     plt.show()
# For batching use make_loader but will throw errors maybe because data has different shapes and needs interpolation.