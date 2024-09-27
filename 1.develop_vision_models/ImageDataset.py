import pathlib
from typing import Optional

import numpy as np
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Iterable Image Dataset for Stained Images, which supports applying transformations to the inputs and targets"""

    def __init__(
        self,
        _input_dir: pathlib.Path,
        _target_dir: pathlib.Path,
        _input_transform: Optional[ImageOnlyTransform] = None,
        _target_transform: Optional[ImageOnlyTransform] = None
    ):
        self.__input_dir = _input_dir
        self.__target_dir = _target_dir

        # Retrieve all data from the specified directory
        self.__image_path = list(self.__input_dir.glob('*'))

        self.__input_transform = _input_transform
        self.__target_transform = _target_transform

    def __len__(self):
        return len(self.__image_path)

    @property
    def input_transform(self):
        return self.__input_transform

    @property
    def target_transform(self):
        return self.__target_transform

    @property
    def input_name(self):
        if not self.__input_name:
            raise ValueError("The input is not yet defined, so __input_name is not defined.")
        return self.__input_name

    @property
    def target_name(self):
        if not self.__target_name:
            raise ValueError("The target is not yet defined, so __target_name is not defined.")
        return self.__target_name

    def __getitem__(self, _idx):
        """Retrieve input and target image stain"""

        self.__input_name = self.__image_path[_idx].name
        self.__target_name = str(self.__input_name).replace("CH0", "CH2").replace("dapi", "gold")
        input_image = np.array(Image.open(self.__input_dir / self.__input_name).convert('I;16'))
        target_image = np.array(Image.open(self.__target_dir / self.__target_name).convert('I;16'))

        if self.__input_transform:
            input_image = self.__input_transform(image=input_image)["image"]
            input_image = torch.from_numpy(input_image).unsqueeze(0).float()

        if self.__target_transform:
            target_image = self.__target_transform(image=target_image)["image"]
            target_image = torch.from_numpy(target_image).unsqueeze(0).float()

        return input_image, target_image
