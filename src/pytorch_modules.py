import torch
from torch import nn


class DepthwiseSeparableConv2d(nn.Module):
    # Implementation of the depthwise separable convolution layer. Code borrowed from:
    # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseSeparableConv1d(nn.Module):
    # Implementation of the depthwise separable convolution layer. Code based on:
    # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class TransferenceFunctionModule(nn.Module):
    def __init__(self):
        super(TransferenceFunctionModule, self).__init__()

    def forward(self, x):
        return x


class XceptionModule1d(nn.Module):
    # Implementation of the xception basic conv block (1-dimensional). Code based on:
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modules,
        kernel_size=3,
        pooling_stride=1,
        bn_momentum=0.995,
    ):
        super(XceptionModule1d, self).__init__()
        padding = int((kernel_size - 1) / 2)
        padding_pool = int((pooling_stride - 1) / 2)
        if pooling_stride > 1 or in_channels != out_channels:
            self.skip_conn_module = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=pooling_stride,
                padding=0,
            )
        else:
            self.skip_conn_module = TransferenceFunctionModule()

        modules = []
        for i in range(n_modules):
            modules.append(nn.ReLU(True))
            modules.append(nn.InstanceNorm1d(in_channels))
            modules.append(
                DepthwiseSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            in_channels = out_channels
        if pooling_stride > 1:
            modules.append(
                nn.AvgPool1d(
                    kernel_size=pooling_stride,
                    stride=pooling_stride,
                    ceil_mode=True,
                    padding=0,
                )
            )
        self.core = nn.Sequential(*modules)

    def forward(self, x):
        main = self.core.forward(x)
        skip = self.skip_conn_module.forward(x)
        return main + skip


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
