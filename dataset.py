import os
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped, Lambdad
)

def get_brats_transforms():
    """
    Real 3D Medical MRI Preprocessing Transformations using MONAI
    - Remaps BraTS label 4 to 3 using Lambdad
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # BraTS-ல் உள்ள லேபிள் 4-ஐ 3 ஆக மாற்றுதல் (0: Background, 1: NCR/NET, 2: ED, 3: ET)
        Lambdad(keys=["label"], func=lambda x: torch.where(x == 4, torch.tensor(3, dtype=x.dtype), x)),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(64, 64, 64), mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ])

class RealBraTSDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms or get_brats_transforms()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = self.transforms(item)
        return data["image"], data["label"]