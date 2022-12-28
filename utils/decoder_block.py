from torch import nn
from utils import conv_block


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = conv_block.ConvBlock(out_channels+out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
