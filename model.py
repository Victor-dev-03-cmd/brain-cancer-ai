import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
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

class AttentionGate3D(nn.Module):
    """Attention Gate to focus on small tumor regions and suppress background noise"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        net = self.relu(g1 + x1)
        psi = self.psi(net)
        return x * psi

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super(UNet3D, self).__init__()
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv3D(64, 128)

        # Attention Gates
        self.ag2 = AttentionGate3D(F_g=64, F_l=64, F_int=32)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(128, 64)

        self.ag1 = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)

        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        bottleneck = self.bottleneck(pool2)

        dec2 = self.up2(bottleneck)
        ag2 = self.ag2(g=dec2, x=enc2)
        dec2 = torch.cat((dec2, ag2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        ag1 = self.ag1(g=dec1, x=enc1)
        dec1 = torch.cat((dec1, ag1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)
