import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """(3D Convolution -> Batch Normalization -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D UNet Architecture for Brain Tumor Segmentation
    Inputs: 3D MRI Volume (e.g., [Batch, 4_Channels, Depth, Height, Width])
    Outputs: Tumor Mask (e.g., [Batch, Num_Classes, Depth, Height, Width])
    """
    def __init__(self, in_channels=4, num_classes=3):
        super(UNet3D, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck (Bridge)
        self.bottleneck = DoubleConv3D(64, 128)
        
        # Decoder (Expanding Path)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(128, 64) # 64 (up) + 64 (skip) = 128
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)  # 32 (up) + 32 (skip) = 64
        
        # Final Output Layer
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool2)
        
        # Decoder with Skip Connections
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1) # Skip connection 2
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1) # Skip connection 1
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)


# Local Testing
if __name__ == "__main__":
    # Test Input: [Batch_Size=1, Modalities=4 (T1, T1ce, T2, FLAIR), Depth=64, Height=64, Width=64]
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    model = UNet3D(in_channels=4, num_classes=3)
    output = model(dummy_input)
    
    print("=" * 55)
    print("  3D UNET MODEL ARCHITECTURE CREATED SUCCESSFULLY!  ")
    print("=" * 55)
    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")
    print("Status       : Ready for Training Pipeline on Cloud GPU!")
    print("=" * 55)
