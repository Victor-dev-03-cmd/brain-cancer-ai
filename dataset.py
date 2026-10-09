import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose, EnsureChannelFirstd, Resized, ScaleIntensityRanged
)

class BrainTumorDataset(Dataset):
    """
    3D MRI Medical Image Pipeline for Glioblastoma Research
    Input: 4 Channels (T1, T1ce, T2, FLAIR)
    Output: 3 Classes (Background, Edema, Enhancing Tumor Core)
    """
    def __init__(self, num_samples=20, spatial_size=(64, 64, 64)):
        self.num_samples = num_samples
        self.spatial_size = spatial_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 4 MRI Modalities
        images = torch.randn(4, *self.spatial_size)
        # 3 Tumor Classes Mask
        masks = torch.randint(0, 3, self.spatial_size, dtype=torch.long)
        return images, masks
