import torch
import torch.nn as nn
import time


class Conv2d(nn.Module):
    def __init__(self, inc, ouc, k, s, p):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, ouc, k, s, p),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvSet(nn.Module):  # inc->ouc
    def __init__(self, inc, ouc):
        super(ConvSet, self).__init__()
        self.convset = nn.Sequential(
            Conv2d(inc, ouc, 1, 1, 0),
            Conv2d(ouc, ouc, 3, 1, 1),
            Conv2d(ouc, ouc * 2, 1, 1, 0),
            Conv2d(ouc * 2, ouc * 2, 3, 1, 1),
            Conv2d(ouc * 2, ouc, 1, 1, 0)
        )

    def forward(self, x):
        return self.convset(x)


class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class Downsampling(nn.Module):
    def __init__(self, inc, ouc):
        super(Downsampling, self).__init__()
        self.d = nn.Sequential(
            Conv2d(inc, ouc, 3, 2, 1)
        )

    def forward(self, x):
        return self.d(x)


class Residual(nn.Module):  # inc->inc
    def __init__(self, inc):
        super(Residual, self).__init__()
        self.r = nn.Sequential(
            Conv2d(inc, inc // 2, 1, 1, 0),
            Conv2d(inc // 2, inc, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.r(x)


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()

        self.d52 = nn.Sequential(
            Conv2d(3, 32, 3, 1, 1),  # 416
            Conv2d(32, 64, 3, 2, 1),  # 208

            # 1x
            Conv2d(64, 32, 1, 1, 0),
            Conv2d(32, 64, 3, 1, 1),
            Residual(64),

            Downsampling(64, 128),  # 104

            # 2x
            Conv2d(128, 64, 1, 1, 0),
            Conv2d(64, 128, 3, 1, 1),
            Residual(128),

            Conv2d(128, 64, 1, 1, 0),
            Conv2d(64, 128, 3, 1, 1),
            Residual(128),

            Downsampling(128, 256),  # 52

            # 8x
            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256)
        )

        self.d26 = nn.Sequential(
            Downsampling(256, 512),  # 26

            # 8x
            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512)
        )

        self.d13 = nn.Sequential(
            Downsampling(512, 1024),  # 13

            # 4x
            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024)
        )
        '---------------------------------------------------------'

        self.convset_13 = nn.Sequential(
            ConvSet(1024, 512)
        )

        self.detection_13 = nn.Sequential(
            Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 266, 1, 1, 0)  # ?????????????????18
        )

        self.conv_13 = nn.Sequential(
            Conv2d(512, 256, 1, 1, 0)
        )

        self.up_to_26 = nn.Sequential(
            Upsampling()
        )
        '---------------------------------------------------------'

        self.convset_26 = nn.Sequential(
            ConvSet(768, 512)  # 经concat，通道相加512+256=768
        )

        self.detection_26 = nn.Sequential(
            Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 255, 1, 1, 0)
        )

        self.conv_26 = nn.Sequential(
            Conv2d(512, 256, 1, 1, 0)
        )

        self.up_to_52 = nn.Sequential(
            Upsampling()
        )
        '---------------------------------------------------------'

        self.convset_52 = nn.Sequential(
            ConvSet(512, 512)  # 经concat，通道相加256+256=512
        )

        self.detection_52 = nn.Sequential(
            Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 255, 1, 1, 0)
        )

    def forward(self, x):
        x_52 = self.d52(x)
        x_26 = self.d26(x_52)
        x_13 = self.d13(x_26)

        x_13_ = self.convset_13(x_13)
        out_13 = self.detection_13(x_13_)  # 13*13输出

        y_13_ = self.conv_13(x_13_)
        y_26 = self.up_to_26(y_13_)
        '----------------------------------------------------------'

        y_26_cat = torch.cat((y_26, x_26), dim=1)  # 26*26连接
        x_26_ = self.convset_26(y_26_cat)
        out_26 = self.detection_26(x_26_)

        y_26_ = self.conv_26(x_26_)
        y_52 = self.up_to_52(y_26_)
        '----------------------------------------------------------'

        y_52_cat = torch.cat((y_52, x_52), dim=1)
        x_52_ = self.convset_52(y_52_cat)
        out_52 = self.detection_52(x_52_)

        return out_52


if __name__ == '__main__':
    trunk = MainNet()
    x = torch.rand((1, 3, 416, 416))
    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)


