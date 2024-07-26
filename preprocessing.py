import os
import pandas as pd
import radiomics
import numpy as np
from radiomics import featureextractor
from omegaconf import OmegaConf
import logging
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm

loggerrdm = logging.getLogger("radiomics")
loggerrdm.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel('INFO') # change

class RadiomicsPreprocessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.extractor = self.get_extractor()
        self._df_processed = None

    @staticmethod
    def _load_config(config_path: str):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        return cfg

    def get_extractor(self):
        return featureextractor.RadiomicsFeatureExtractor(**self.config.extractor_params)

    def rename_png_to_nii(self, path):
        for file in os.listdir(path):
            file_prev = os.path.join(path, file)
            file_new = os.path.join(path, f"{os.path.splitext(file)[0]}.nii")
            os.rename(file_prev, file_new)

    def extract_single(self, path_img, path_mask) -> dict:
        exclude_features = self.config.get('exclude_features', [])
        print(path_img, path_mask)
        results = self.extractor.execute(path_img, path_mask)
        for feature in exclude_features:
            if feature in results:
                del results[feature]

        results = {k: [v] if np.isscalar(v) else [v.item()] for k, v in results.items()}
        if self._df_processed is None:
            self._df_processed = results
        else:
            for feature, value in results.items():
                self._df_processed[feature].extend(value)
        return results

    def extract_batch(self, mutation=True, save=False):
        index = []
        if mutation:
            image_dir = self.config.paths.mutation_dir.images
            mask_dir = self.config.paths.mutation_dir.masks
        else:
            image_dir = self.config.paths.no_mutation_dir.images
            mask_dir = self.config.paths.no_mutation_dir.masks

        # check if files format correct first
        for img_file in os.listdir(image_dir):
            if not img_file.endswith('.nii'):
                logger.warn('PNG not converted to NII')
                self.rename_png_to_nii(image_dir)
                self.rename_png_to_nii(mask_dir)
                return self.extract_batch(mutation)

        for img_file in tqdm(list(os.listdir(image_dir)), desc='Feature extraction'):
            index.append(str(img_file).split('.')[0])
            mask_file = img_file
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            self.extract_single(img_path, mask_path)
        df = pd.DataFrame(self._df_processed, index=index)
        if save:
            df.to_csv('mutation.csv' if mutation else 'no_mutation.csv', index=True, index_label='filename')
            # pd.read_csv('mutation.csv', index_col='filename') access to csv
        return pd.DataFrame(self._df_processed, index=index)

# Usage
# p = RadiomicsPreprocessor('config/config.yaml')
# mutation_df = p.extract_batch(mutation=True, save=True) # Works
# no_mutation_df = p.extract_batch(mutation=False) # Person10 mask and image shape mismatch