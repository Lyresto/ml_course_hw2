import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential

from config import UNetModelConfig


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(GCN, self).__init__()

        padding = kernel_size // 2
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0))

        self.conv_r1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding))

    def forward(self, x):
        left = self.conv_l1(x)
        left = self.conv_l2(left)

        right = self.conv_r1(x)
        right = self.conv_r2(right)

        return left + right


class UNet(nn.Module):
    def __init__(self, config: UNetModelConfig, in_channels: int=1, out_channels: int=1):
        super(UNet, self).__init__()

        self.config = config

        self.encoder1 = self.double_conv(in_channels, 64, first_layer=True)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        if config.use_GCN:
            self.encoder3 = GCN(128, 256)
        else:
            self.encoder3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        if config.depth == 4:
            self.bottleneck = self.double_conv(512, 1024)
            self.encoder5 = self.pool5 = self.upconv5 = self.decoder5 = None
        elif config.depth == 5:
            self.encoder5 = self.double_conv(512, 1024)
            self.pool5 = nn.MaxPool2d(2)
            self.bottleneck = self.double_conv(1024, 2048)
            self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
            self.decoder5 = self.double_conv(2048, 1024)
        else:
            raise ValueError("Depth must be 4 or 5")

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(128, 64)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels: int, out_channels: int, first_layer: bool=False) -> Sequential:
        encoder_kernel_size = self.config.encoder_kernel_size
        if not first_layer:
            encoder_kernel_size -= self.config.encoder_kernel_size_reduction
        padding = (encoder_kernel_size - 1) // 2
        return Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=encoder_kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=encoder_kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        if self.config.depth == 4:
            bottleneck = self.bottleneck(self.pool4(enc4))
            dec4 = self.upconv4(bottleneck)
        elif self.config.depth == 5:
            enc5 = self.encoder5(self.pool4(enc4))
            bottleneck = self.bottleneck(self.pool5(enc5))
            dec5 = self.upconv5(bottleneck)
            dec5 = torch.cat((dec5, enc5), dim=1)
            dec5 = self.decoder5(dec5)
            dec4 = self.upconv4(dec5)
        else:
            raise ValueError("Depth must be 4 or 5")

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = torch.sigmoid(self.output_conv(dec1))

        return output
