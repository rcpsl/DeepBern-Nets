import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def get_output_shape(self, input_shape):
        Ci, Hi, Wi = input_shape
        Co = self.out_channels
        padding = self.padding
        dilation = self.dilation
        kernel_size = self.kernel_size
        stride = self.stride
        Ho = (Hi + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
        Wo = (Wi + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
        Ho = int(Ho + 1)
        Wo = int(Wo + 1)

        return torch.tensor([Co, Ho, Wo])

    def _conv_args(self):
        params = {}
        params["padding"] = self.padding
        params["dilation"] = self.dilation
        params["stride"] = self.stride
        params["groups"] = self.groups
        return params

    def forward_interval(self, interval):
        params = self._conv_args()
        W = self.weight
        b = self.bias
        zeros_W = torch.zeros_like(W)
        W_pos = torch.maximum(zeros_W, W)
        W_neg = torch.minimum(zeros_W, W)
        post_l = F.conv2d(interval[..., 0], W_pos, **params) + F.conv2d(
            interval[..., 1], W_neg, bias=b, **params
        )
        post_u = F.conv2d(interval[..., 0], W_neg, **params) + F.conv2d(
            interval[..., 1], W_pos, bias=b, **params
        )
        out_interval = torch.stack((post_l, post_u), dim=-1)
        return out_interval
