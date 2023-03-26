import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models.unet_blocks import *


class MedicalDataset(Dataset):
    """
    A PyTorch dataset class for loading medical imaging data.

    Args:
        root_dir (str): The path to the root directory containing the data.
        test (bool): Whether the dataset is for testing purposes or not. Defaults to False.
        transforms (callable): A function/transform that takes in a sample and returns a transformed version.

    Attributes:
        transforms (callable): The input data transformation function.
        root_dir (str): The path to the root directory containing the data.
        sample_list (list): A list of filenames in the root directory.
        test (bool): Whether the dataset is for testing purposes or not.

    Methods:
        __len__(): Returns the length of the dataset.
        get_sample(idx): Returns a dictionary containing the sample data corresponding to the given index.
        __getitem__(idx): Gets the dataset sample given an index.

    """
    def __init__(self, root_dir, test=False,
                 transforms=None):      
        self.transforms = transforms
        self.root_dir = root_dir
        self.sample_list = os.listdir(self.root_dir)
        self.test= test

    def __len__(self):
        """Dataset length."""
        return len(self.sample_list)

    def get_sample(self, idx):
        """
        Returns a dictionary containing the sample data corresponding to the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the following fields:
                - idx: The index of the sample.
                - sample_name: The name of the sample file.
                - ct: The CT scan data as a numpy array.
                - possible_dose_mask: The possible dose mask as a numpy array.
                - structure_masks: The structure masks as a numpy array.
                - dose: The dose data as a numpy array (only if `test=False`).
                - organ_dose: The organ dose data as a numpy array (only if `test=False`).

        """
        # Add image information
        sample_name = self.sample_list[idx]
        sample_path = os.path.join(self.root_dir, sample_name)
        ct_path = os.path.join(sample_path, "ct.npy")
        possible_dose_mask_path = os.path.join(sample_path, "possible_dose_mask.npy")
        structure_masks_path = os.path.join(sample_path, "structure_masks.npy")
        dose_path = os.path.join(sample_path, "dose.npy")

        sample = {
            'idx': idx,
            'sample_name': sample_name,
            'ct': np.load(ct_path),
            'possible_dose_mask': np.load(possible_dose_mask_path),
            'structure_masks': np.load(structure_masks_path),
        }
        if not self.test:
            sample['dose'] = np.load(dose_path)
            organ_dose_list = [np.multiply(organ_mask, sample["dose"])[None, ...] for organ_mask in sample["structure_masks"]]
            sample["organ_dose"] = np.concatenate(organ_dose_list)

        return sample
        
    
    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        raise NotImplementedError
