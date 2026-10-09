import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, CropForegroundd, Resized, EnsureTyped, Lambdad
)


def _remap_brats_label(label):
    """Remap BraTS label 4 → 3 so classes are strictly [0, 1, 2, 3]."""
    out = label.copy()
    out[label == 4] = 3
    return out


def get_brats_transforms():
    """
    3D Medical MRI Preprocessing Transformations using MONAI.
    Uses per-channel non-zero intensity normalization to handle wide
    and modality-varying intensity ranges across BraTS MRI volumes.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Lambdad(keys=["label"], func=_remap_brats_label),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
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
