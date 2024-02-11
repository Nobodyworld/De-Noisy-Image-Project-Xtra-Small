# /models/unet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc_conv4 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc_conv5 = nn.Sequential(nn.Conv2d(64, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(2, 2)
        self.enc_conv6 = nn.Sequential(nn.Conv2d(80, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.pool6 = nn.MaxPool2d(2, 2)
        self.enc_conv7 = nn.Sequential(nn.Conv2d(96, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.pool7 = nn.MaxPool2d(2, 2)
        self.enc_conv8 = nn.Sequential(nn.Conv2d(112, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))


        # Decoder
        self.dec_conv8 = nn.Sequential(nn.Conv2d(128, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv7 = nn.Sequential(nn.Conv2d(112, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv6 = nn.Sequential(nn.Conv2d(96, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = nn.Sequential(nn.Conv2d(80, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Sequential(nn.Conv2d(64, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(48, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Modify the last decoder layer to have 32 output channels
        self.dec_conv1 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        # Keep the output layer's input channels at 32
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()  # or nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool1(x1)
        x3 = self.enc_conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc_conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.enc_conv4(x6)
        x8 = self.pool4(x7)
        x9 = self.enc_conv5(x8)
        x10 = self.pool5(x9)
        x11 = self.enc_conv6(x10)
        x12 = self.pool6(x11)
        x13 = self.enc_conv7(x12)
        x14 = self.pool7(x13)
        x15 = self.enc_conv8(x14)

        # Decoder
        x16 = self.dec_conv8(x15)
        x17 = self.up7(x16)
        x18 = self.dec_conv7(x17 + x13)
        x19 = self.up6(x18)
        x20 = self.dec_conv6(x19 + x11)
        x21 = self.up5(x20)
        x22 = self.dec_conv5(x21 + x9)
        x23 = self.up4(x22)
        x24 = self.dec_conv4(x23 + x7)
        x25 = self.up3(x24)
        x26 = self.dec_conv3(x25 + x5)
        x27 = self.up2(x26)
        x28 = self.dec_conv2(x27 + x3)
        x29 = self.up1(x28)
        x30 = self.dec_conv1(x29 + x1)

        # Output
        x31 = self.out_conv(x30)
        return x31