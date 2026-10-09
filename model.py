import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate3D(nn.Module):
    """Soft attention gate to focus on small tumor regions and suppress background."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int),
        )
        self.W_l = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.psi(self.relu(self.W_g(g) + self.W_l(x)))
        return x * psi


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        # Encoder: 32 -> 64 -> 128 -> 256, Bottleneck: 512
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = DoubleConv3D(64, 128)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.enc4 = DoubleConv3D(128, 256)
        self.pool4 = nn.MaxPool3d(2, 2)

        self.bottleneck = DoubleConv3D(256, 512)

        # Decoder level 4
        self.up4  = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.ag4  = AttentionGate3D(F_g=256, F_l=256, F_int=128)
        self.dec4 = DoubleConv3D(512, 256)

        # Decoder level 3
        self.up3  = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.ag3  = AttentionGate3D(F_g=128, F_l=128, F_int=64)
        self.dec3 = DoubleConv3D(256, 128)

        # Decoder level 2
        self.up2  = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.ag2  = AttentionGate3D(F_g=64, F_l=64, F_int=32)
        self.dec2 = DoubleConv3D(128, 64)

        # Decoder level 1
        self.up1  = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.ag1  = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.dec1 = DoubleConv3D(64, 32)

        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck(self.pool4(e4))

        d4 = self.up4(bn)
        d4 = self.dec4(torch.cat([d4, self.ag4(g=d4, x=e4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.ag3(g=d3, x=e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.ag2(g=d2, x=e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.ag1(g=d1, x=e1)], dim=1))

        return self.final_conv(d1)
