import random
import torch
from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


def get_device() -> torch.device:
    
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    
    return './data/'


def set_random_seed(seed: int) -> None:
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ContinualDataset:
    
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
       
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
       
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        
        pass
