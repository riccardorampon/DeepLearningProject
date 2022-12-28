from torch import nn
from utils import conv_block


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = conv_block.ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

