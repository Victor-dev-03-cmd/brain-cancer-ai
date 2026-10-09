import os
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped
)

def get_brats_transforms():
    """
    Real 3D Medical MRI Preprocessing Transformations using MONAI
    - Intensity Normalization
    - RAS Orientation Standard
    - Resizing to 64x64x64 for GPU Training Efficiency
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(64, 64, 64)),
        EnsureTyped(keys=["image", "label"]),
    ])

class RealBraTSDataset(Dataset):
    """
    Real BraTS Dataset Wrapper
    Expected data structure: List of dicts with 'image' and 'label' file paths (.nii.gz)
    """
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms or get_brats_transforms()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = self.transforms(item)
        return data["image"], data["label"]
