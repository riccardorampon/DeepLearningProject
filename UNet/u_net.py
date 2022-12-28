from torch import nn
from utils import conv_block
from utils import encoder_block
from utils import decoder_block


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block.EncoderBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.e2 = encoder_block.EncoderBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.e3 = encoder_block.EncoderBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.e4 = encoder_block.EncoderBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        """ Bottleneck """
        self.b = conv_block.ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        """ Decoder """
        self.d1 = decoder_block.DecoderBlock(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.d2 = decoder_block.DecoderBlock(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.d3 = decoder_block.DecoderBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.d4 = decoder_block.DecoderBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """ Encoder """
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


