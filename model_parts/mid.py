# ./model_parts/mid.py
import torch.nn as nn

class Mid(nn.Module):
    def __init__(self):
        super(Mid, self).__init__()
        # Mid
        self.mid_conv1 = nn.Sequential(nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.mid_conv2 = nn.Sequential(nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.mid_conv3 = nn.Sequential(nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.mid_conv4 = nn.Sequential(nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
 
    def forward(self, x13):
        x13 = self.mid_conv1(x13)
        x13 = self.mid_conv2(x13)
        x13 = self.mid_conv3(x13)
        x13 = self.mid_conv4(x13)

        return x13