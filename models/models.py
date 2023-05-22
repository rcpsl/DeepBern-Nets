import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BernsteinLayer import BernsteinLayer
from models.ConvLayer import ConvLayer


class FCModel(nn.Module):
    def __init__(
        self, layer_sizes, degree, act="bern", input_bounds=None, last_bern=False
    ):
        super().__init__()

        layers = []
        for i, l_size in enumerate(layer_sizes[:-1]):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < (len(layer_sizes) - 2):
                if act == "bern":
                    BATCH_NORM = False
                    SIGMOID = False
                    if BATCH_NORM:
                        layers.append(nn.BatchNorm1d(layer_sizes[i + 1], affine=False))
                    if SIGMOID:
                        layers.append(nn.Sigmoid())
                    layers.append(BernsteinLayer([layer_sizes[i + 1]], degree))
                else:
                    layers.append(nn.ReLU())

        if last_bern:
            layers.append(BernsteinLayer([layer_sizes[-1]], 1))
        self.input_bounds = input_bounds
        self.net = nn.Sequential(*layers)
        self.layers = layers
    def forward_with_bounds(self, x):
        y = x
        prev_bounds = self.input_bounds
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    W = layer.weight
                    b = layer.bias
                    W_pos = F.relu(W)
                    W_neg = -F.relu(-W)
                    lb = W_pos @ prev_bounds[..., 0].T + W_neg @ prev_bounds[..., 1].T + b
                    ub = W_pos @ prev_bounds[..., 1].T + W_neg @ prev_bounds[..., 0].T + b
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                y = layer(y)

            elif isinstance(layer, nn.BatchNorm1d):
                y = layer(y)
                lb = F.batch_norm(
                    prev_bounds[..., 0].unsqueeze(0),
                    layer.running_mean,
                    layer.running_var,
                )
                ub = F.batch_norm(
                    prev_bounds[..., 1].unsqueeze(0),
                    layer.running_mean,
                    layer.running_var,
                )
                prev_bounds = torch.concat((lb.T, ub.T), dim=-1)

            elif isinstance(layer, nn.Sigmoid):
                y = layer(y)
                lb = torch.sigmoid(prev_bounds[..., 0].unsqueeze(0))
                ub = torch.sigmoid(prev_bounds[..., 1].unsqueeze(0))
                prev_bounds = torch.concat((lb.T, ub.T), dim=-1)

            elif isinstance(layer, BernsteinLayer):
                if not isinstance(self.layers[i - 1], nn.BatchNorm1d):
                    layer.input_bounds = prev_bounds
                elif (
                    isinstance(self.layers[i - 1], nn.BatchNorm1d)
                    and not self.layers[i - 1].training
                ):
                    layer.input_bounds = prev_bounds
                with torch.no_grad():
                    prev_bounds = layer.bern_bounds
                y = layer(y)

        return y

    def forward_subinterval(self, x, C=None):
        prev_bounds = x
        for i, layer in enumerate(self.layers):
            last_layer = i == len(self.layers) - 1
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                if last_layer and C is not None:
                    W = C.matmul(W)
                    b = C @ b
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = b
                    ub = b
                    lb = (
                        lb
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 0].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 1].unsqueeze(-1))
                        ).squeeze()
                    )
                    ub = (
                        ub
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 1].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 0].unsqueeze(-1))
                        ).squeeze()
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                else:
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = (
                        prev_bounds[..., 0] @ W_pos.T
                        + prev_bounds[..., 1] @ W_neg.T
                        + b
                    )
                    ub = (
                        prev_bounds[..., 1] @ W_pos.T
                        + prev_bounds[..., 0] @ W_neg.T
                        + b
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, nn.BatchNorm1d):
                lb = F.batch_norm(
                    prev_bounds[..., 0], layer.running_mean, layer.running_var
                )
                ub = F.batch_norm(
                    prev_bounds[..., 1], layer.running_mean, layer.running_var
                )
                prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, nn.Sigmoid):
                lb = torch.sigmoid(prev_bounds[..., 0])
                ub = torch.sigmoid(prev_bounds[..., 1])
                prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, BernsteinLayer):
                prev_bounds = layer.subinterval_bounds(prev_bounds)
        return prev_bounds

    def forward(self, x, with_bounds=False):
        if self.training:
            y = self.forward_with_bounds(x)
            # y = self.net(x)
        else:
            if with_bounds:
                y = self.forward_with_bounds(x)
            else:
                y = self.net(x)
        return y

class CNNa(nn.Module):
    def __init__(self, degree, act="bern", input_bounds=None, num_outs = 10):
        super().__init__()
        self.input_bounds = input_bounds
        in_shape = torch.tensor(input_bounds.shape)[:-1]
        layers = []
        conv1 = ConvLayer(in_shape[0], 16, 4, stride=2, padding=1)
        conv1_out_shape = conv1.get_output_shape(in_shape)
        layers.append(conv1)
        if act == "bern":
            layers.append(BernsteinLayer(conv1_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv2 = ConvLayer(16, 32, 4, stride=2, padding=1)
        conv2_out_shape = conv2.get_output_shape(conv1_out_shape)
        layers.append(conv2)
        if act == "bern":
            layers.append(BernsteinLayer(conv2_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        flattened_shape = conv2_out_shape.prod().item()
        layers.append(nn.Linear(flattened_shape, 100))
        if act == "bern":
            layers.append(BernsteinLayer([100], degree))
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(100, num_outs))

        self.net = nn.Sequential(*layers)

    def forward_with_bounds(self, x):
        y = x
        prev_bounds = self.input_bounds
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                W_pos = F.relu(W)
                W_neg = -F.relu(-W)
                lb = W_pos @ prev_bounds[..., 0].T + W_neg @ prev_bounds[..., 1].T + b
                ub = W_pos @ prev_bounds[..., 1].T + W_neg @ prev_bounds[..., 0].T + b
                prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds.unsqueeze(0))
                prev_bounds = prev_bounds.squeeze()

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(-1, 2)

            elif isinstance(layer, BernsteinLayer):
                layer.input_bounds = prev_bounds
                prev_bounds = layer.bern_bounds

            y = layer(y)

        return y

    def forward_subinterval(self, x, C=None):
        B = x.shape[0]
        prev_bounds = x
        for i, layer in enumerate(self.net):
            last_layer = i == len(self.net) - 1
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                if last_layer and C is not None:
                    W = C.matmul(W)
                    b = C @ b
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = b
                    ub = b
                    lb = (
                        lb
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 0].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 1].unsqueeze(-1))
                        ).squeeze()
                    )
                    ub = (
                        ub
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 1].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 0].unsqueeze(-1))
                        ).squeeze()
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                else:
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = (
                        prev_bounds[..., 0] @ W_pos.T
                        + prev_bounds[..., 1] @ W_neg.T
                        + b
                    )
                    ub = (
                        prev_bounds[..., 1] @ W_pos.T
                        + prev_bounds[..., 0] @ W_neg.T
                        + b
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds)

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(B, -1, 2)

            elif isinstance(layer, BernsteinLayer):
                prev_bounds = layer.subinterval_bounds(prev_bounds)

        return prev_bounds

    def forward(self, x):
        if self.training:
            y = self.forward_with_bounds(x)
        else:
            y = self.net(x)
        return y

class CNN7(nn.Module):
    def __init__(self, degree, act="bern", input_bounds=None, num_outs = 10):
        super().__init__()
        BATCH_NORM = False
        self.input_bounds = input_bounds
        in_shape = torch.tensor(input_bounds.shape)[:-1]
        layers = []
        conv1 = ConvLayer(in_shape[0], 64, 3, stride=1, padding=1)
        conv1_out_shape = conv1.get_output_shape(in_shape)
        layers.append(conv1)
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(64, affine=False))
        if act == "bern":
            layers.append(BernsteinLayer(conv1_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv2 = ConvLayer(64, 64, 3, stride=1, padding=1)
        conv2_out_shape = conv2.get_output_shape(conv1_out_shape)
        layers.append(conv2)
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(64, affine=False))
        if act == "bern":
            layers.append(BernsteinLayer(conv2_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv3 = ConvLayer(64, 128, 3, stride=2, padding=1)
        conv3_out_shape = conv3.get_output_shape(conv2_out_shape)
        layers.append(conv3)
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(128, affine=False))
        if act == "bern":
            layers.append(BernsteinLayer(conv3_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv4 = ConvLayer(128, 128, 3, stride=1, padding=1)
        conv4_out_shape = conv4.get_output_shape(conv3_out_shape)
        layers.append(conv4)
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(128, affine=False))
        if act == "bern":
            layers.append(BernsteinLayer(conv4_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv5 = ConvLayer(128, 128, 3, stride=1, padding=1)
        conv5_out_shape = conv5.get_output_shape(conv4_out_shape)
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(128, affine=False))
        layers.append(nn.Flatten())
        flattened_shape = conv5_out_shape.prod().item()
        if act == "bern":
            layers += [
                nn.Linear(flattened_shape, 512),
                BernsteinLayer([512], degree),
                nn.Linear(512, 512),
                BernsteinLayer([512], degree),
                nn.Linear(512, num_outs),
            ]
        else:
            layers += [
                nn.Linear(flattened_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_outs),
            ]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward_with_bounds(self, x):
        y = x
        with torch.no_grad():
            prev_bounds = self.input_bounds
        for i, layer in enumerate(self.net):
           
            if isinstance(layer, nn.Linear):
                y = layer(y)
                with torch.no_grad():
                    W = layer.weight
                    b = layer.bias
                    W_pos = F.relu(W)
                    W_neg = -F.relu(-W)
                    lb = W_pos @ prev_bounds[..., 0].T + W_neg @ prev_bounds[..., 1].T + b
                    ub = W_pos @ prev_bounds[..., 1].T + W_neg @ prev_bounds[..., 0].T + b
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                

            elif isinstance(layer, ConvLayer):
                y = layer(y)
                with torch.no_grad():
                    prev_bounds = layer.forward_interval(prev_bounds.unsqueeze(0))
                    prev_bounds = prev_bounds.squeeze()

            elif isinstance(layer, nn.BatchNorm2d):
                with torch.no_grad():
                    layer(y)
                    mean = layer.running_mean
                    var = layer.running_var
                    lb = (prev_bounds[..., 0] - mean.reshape(1, -1, 1, 1)) / torch.sqrt(var + 1e-5).reshape(1, -1, 1, 1)
                    ub = (prev_bounds[..., 1] - mean.unsqueeze(-1).unsqueeze(-1)) / torch.sqrt(var + 1e-5).reshape(1, -1, 1, 1)
                    prev_bounds = torch.stack((lb, ub), dim=-1).squeeze()
                y = (y - mean.reshape(1, -1, 1, 1)) / torch.sqrt(var + 1e-5).reshape(1, -1, 1, 1)
            elif isinstance(layer, nn.Flatten):
                y = layer(y)
                with torch.no_grad():
                    prev_bounds = prev_bounds.reshape(-1, 2)

            elif isinstance(layer, BernsteinLayer):
                with torch.no_grad():
                    layer.input_bounds = prev_bounds
                    prev_bounds = layer.bern_bounds
                y = layer(y)
        return y

    def forward_subinterval(self, x, C=None):
        B = x.shape[0]
        prev_bounds = x
        for i, layer in enumerate(self.net):
            last_layer = i == len(self.net) - 1
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                if last_layer and C is not None:
                    W = C.matmul(W)
                    b = C @ b
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = b
                    ub = b
                    lb = (
                        lb
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 0].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 1].unsqueeze(-1))
                        ).squeeze()
                    )
                    ub = (
                        ub
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 1].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 0].unsqueeze(-1))
                        ).squeeze()
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                else:
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = (
                        prev_bounds[..., 0] @ W_pos.T
                        + prev_bounds[..., 1] @ W_neg.T
                        + b
                    )
                    ub = (
                        prev_bounds[..., 1] @ W_pos.T
                        + prev_bounds[..., 0] @ W_neg.T
                        + b
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds)

            elif isinstance(layer, nn.BatchNorm2d):
                    lb = F.batch_norm(
                        prev_bounds[..., 0].unsqueeze(0),
                        layer.running_mean,
                        layer.running_var,
                    )
                    ub = F.batch_norm(
                        prev_bounds[..., 1].unsqueeze(0),
                        layer.running_mean,
                        layer.running_var,
                    )
                    prev_bounds = torch.concat((lb.T, ub.T), dim=-1)

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(B, -1, 2)

            elif isinstance(layer, BernsteinLayer):
                prev_bounds = layer.subinterval_bounds(prev_bounds)

        return prev_bounds

    def forward(self, x):
        if self.training:
            y = self.forward_with_bounds(x)
        else:
            y = self.net(x)
        return y    

class CNNb(nn.Module):
    def __init__(self, degree, act="bern", input_bounds=None, num_outs = 10):
        super().__init__()
        self.input_bounds = input_bounds
        in_shape = torch.tensor(input_bounds.shape)[:-1]
        layers = []
        conv1 = ConvLayer(in_shape[0], 16, 3, stride=1, padding=1)
        conv1_out_shape = conv1.get_output_shape(in_shape)
        layers.append(conv1)
        if act == "bern":
            layers.append(BernsteinLayer(conv1_out_shape, degree=degree))
        elif act == "relu":
            layers.append(nn.ReLU())

        conv2 = ConvLayer(16, 16, 4, stride=2, padding=1)
        conv2_out_shape = conv2.get_output_shape(conv1_out_shape)
        layers.append(conv2)
        if act == "bern":
            layers.append(BernsteinLayer(conv2_out_shape, degree=degree))
        elif act == "relu":
            layers.append(nn.ReLU())
        conv3 = ConvLayer(16, 32, 3, stride=1, padding=1)
        conv3_out_shape = conv3.get_output_shape(conv2_out_shape)
        layers.append(conv3)
        if act == "bern":
            layers.append(BernsteinLayer(conv3_out_shape, degree=degree))
        elif act == "relu":
            layers.append(nn.ReLU())

        conv4 = ConvLayer(32, 32, 4, stride=2, padding=1)
        conv4_out_shape = conv4.get_output_shape(conv3_out_shape)
        layers.append(conv4)
        if act == "bern":
            layers.append(BernsteinLayer(conv4_out_shape, degree=degree))
        elif act == "relu":
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())
        flattened_shape = conv4_out_shape.prod().item()
        layers += [
            nn.Linear(flattened_shape, 512),
            BernsteinLayer([512], degree) if act == "bern" else nn.ReLU(),
            nn.Linear(512, num_outs),
        ]

        self.net = nn.Sequential(*layers)

    def forward_with_bounds(self, x):
        y = x
        prev_bounds = self.input_bounds
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                # z_like_W = torch.zeros_like(W)
                # W_pos = torch.maximum(z_like_W, W)
                # W_neg = torch.minimum(z_like_W, W)
                W_pos = F.relu(W)
                W_neg = -F.relu(-W)
                # W_pos = torch.maximum(z_like_W, W)
                # W_neg = torch.minimum(z_like_W, W)
                lb = W_pos @ prev_bounds[..., 0].T + W_neg @ prev_bounds[..., 1].T + b
                ub = W_pos @ prev_bounds[..., 1].T + W_neg @ prev_bounds[..., 0].T + b
                prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds.unsqueeze(0))
                prev_bounds = prev_bounds.squeeze()

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(-1, 2)

            elif isinstance(layer, BernsteinLayer):
                layer.input_bounds = prev_bounds
                prev_bounds = layer.bern_bounds

            y = layer(y)

        return y

    def forward_subinterval(self, x, C=None):
        B = x.shape[0]
        prev_bounds = x
        for i, layer in enumerate(self.net):
            last_layer = i == len(self.net) - 1
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                if last_layer and C is not None:
                    W = C.matmul(W)
                    b = C @ b
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = b
                    ub = b
                    lb = (
                        lb
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 0].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 1].unsqueeze(-1))
                        ).squeeze()
                    )
                    ub = (
                        ub
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 1].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 0].unsqueeze(-1))
                        ).squeeze()
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                else:
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = (
                        prev_bounds[..., 0] @ W_pos.T
                        + prev_bounds[..., 1] @ W_neg.T
                        + b
                    )
                    ub = (
                        prev_bounds[..., 1] @ W_pos.T
                        + prev_bounds[..., 0] @ W_neg.T
                        + b
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds)

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(B, -1, 2)

            elif isinstance(layer, BernsteinLayer):
                prev_bounds = layer.subinterval_bounds(prev_bounds)

        return prev_bounds

    def forward(self, x):
        if self.training:
            y = self.forward_with_bounds(x)
        else:
            y = self.net(x)
        return y


class CNNc(nn.Module):
    def __init__(self, degree, act="bern", input_bounds=None, num_outs = 10):
        super().__init__()
        self.input_bounds = input_bounds
        in_shape = torch.tensor(input_bounds.shape)[:-1]
        layers = []
        conv1 = ConvLayer(in_shape[0], 32, 3, stride=1, padding=1)
        conv1_out_shape = conv1.get_output_shape(in_shape)
        layers.append(conv1)
        if act == "bern":
            layers.append(BernsteinLayer(conv1_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv2 = ConvLayer(32, 32, 4, stride=2, padding=1)
        conv2_out_shape = conv2.get_output_shape(conv1_out_shape)
        layers.append(conv2)
        if act == "bern":
            layers.append(BernsteinLayer(conv2_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv3 = ConvLayer(32, 64, 3, stride=1, padding=1)
        conv3_out_shape = conv3.get_output_shape(conv2_out_shape)
        layers.append(conv3)
        if act == "bern":
            layers.append(BernsteinLayer(conv3_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        conv4 = ConvLayer(64, 64, 4, stride=2, padding=1)
        conv4_out_shape = conv4.get_output_shape(conv3_out_shape)
        layers.append(conv4)
        if act == "bern":
            layers.append(BernsteinLayer(conv4_out_shape, degree=degree))
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        flattened_shape = conv4_out_shape.prod().item()
        if act == "bern":
            layers += [
                nn.Linear(flattened_shape, 512),
                BernsteinLayer([512], degree),
                nn.Linear(512, 512),
                BernsteinLayer([512], degree),
                nn.Linear(512, num_outs),
            ]
        else:
            layers += [
                nn.Linear(flattened_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_outs),
            ]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward_with_bounds(self, x):
        y = x
        prev_bounds = self.input_bounds
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                W_pos = F.relu(W)
                W_neg = -F.relu(-W)
                lb = W_pos @ prev_bounds[..., 0].T + W_neg @ prev_bounds[..., 1].T + b
                ub = W_pos @ prev_bounds[..., 1].T + W_neg @ prev_bounds[..., 0].T + b
                prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds.unsqueeze(0))
                prev_bounds = prev_bounds.squeeze()

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(-1, 2)

            elif isinstance(layer, BernsteinLayer):
                layer.input_bounds = prev_bounds
                prev_bounds = layer.bern_bounds

            y = layer(y)

        return y

    def forward_subinterval(self, x, C=None):
        B = x.shape[0]
        prev_bounds = x
        for i, layer in enumerate(self.net):
            last_layer = i == len(self.net) - 1
            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias
                if last_layer and C is not None:
                    W = C.matmul(W)
                    b = C @ b
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = b
                    ub = b
                    lb = (
                        lb
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 0].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 1].unsqueeze(-1))
                        ).squeeze()
                    )
                    ub = (
                        ub
                        + (
                            torch.matmul(W_pos, prev_bounds[..., 1].unsqueeze(-1))
                            + torch.matmul(W_neg, prev_bounds[..., 0].unsqueeze(-1))
                        ).squeeze()
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)
                else:
                    z_like_W = torch.zeros_like(W)
                    W_pos = torch.maximum(z_like_W, W)
                    W_neg = torch.minimum(z_like_W, W)
                    lb = (
                        prev_bounds[..., 0] @ W_pos.T
                        + prev_bounds[..., 1] @ W_neg.T
                        + b
                    )
                    ub = (
                        prev_bounds[..., 1] @ W_pos.T
                        + prev_bounds[..., 0] @ W_neg.T
                        + b
                    )
                    prev_bounds = torch.stack((lb, ub), dim=-1)

            elif isinstance(layer, ConvLayer):
                prev_bounds = layer.forward_interval(prev_bounds)

            elif isinstance(layer, nn.Flatten):
                prev_bounds = prev_bounds.reshape(B, -1, 2)

            elif isinstance(layer, BernsteinLayer):
                prev_bounds = layer.subinterval_bounds(prev_bounds)

        return prev_bounds

    def forward(self, x):
        if self.training:
            y = self.forward_with_bounds(x)
        else:
            y = self.net(x)
        return y
