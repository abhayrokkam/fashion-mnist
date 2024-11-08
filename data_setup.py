import os

import torch
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path

from typing import Tuple, List

def create_data_path() -> str:
    """
    Creating a directory to set up the data. 
    The functions returns a string object which contains the created data path.
    """
    data_path = Path('./data/')
    if not data_path.is_dir():
        os.mkdir(data_path)
        
    return str(data_path)

def get_datasets(root: str | Path) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Function to load and return the FashionMNIST train and test datasets.
    
    Args:
        root: The path to store the data files.

    Returns:
        train_data (torch.utils.data.Dataset): The training dataset.
        test_data (torch.utils.data.Dataset): The testing dataset.
    """
    train_data = torchvision.datasets.FashionMNIST(root=root,
                                                   train=True,
                                                   transform=torchvision.transforms.ToTensor(),
                                                   download=True)
    
    test_data = torchvision.datasets.FashionMNIST(root=root,
                                                  train=False,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    
    return train_data, test_data

def get_class_names(dataset: torch.utils.data.Dataset) -> List[str]:
    """
    A funciton to return the classes of a dataset.
    """
    return dataset.classes

def get_dataloader(train_dataset: torch.utils.data.Dataset, 
                   test_dataset: torch.utils.data.Dataset,
                   batch_size: int,
                   num_workers: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates and returns PyTorch DataLoader instances for training and testing datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset to be used for training.
        test_dataset (torch.utils.data.Dataset): The dataset to be used for testing.
        batch_size (int): The number of samples per batch in the DataLoader.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
        A tuple containing two DataLoader objects:
            - The first is the DataLoader for the training dataset with the specified batch size and shuffling enabled.
            - The second is the DataLoader for the test dataset with double the batch size and shuffling enabled.
    """
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size * 2,
                                 shuffle=True,
                                 num_workers=num_workers)
    
    return train_dataloader, test_dataloader