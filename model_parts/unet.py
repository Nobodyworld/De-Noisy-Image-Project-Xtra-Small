import torch.nn as nn
from model_parts.encoder import Encoder
from model_parts.mid import Mid
from model_parts.decoder import Decoder

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.mid = Mid()
        self.decoder = Decoder()

    def forward(self, x):
        # Use the encoder
        x1, x3, x5, x7, x9, x11, x13 = self.encoder(x)

        # Use the mid
        x13 = self.mid(x13)

        # Use the decoder
        x27 = self.decoder(x13, x11, x9, x7, x5, x3, x1)

        return x27