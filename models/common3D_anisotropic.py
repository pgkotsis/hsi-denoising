import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[3] for x in inputs]
        inputs_shapes3 = [x.shape[4] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(3) - target_shape2) // 2 
                diff3 = (inp.size(4) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm3d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """
    3D convolution with optional anisotropic kernel/stride.
    Fully backward compatible with original conv().
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    downsampler = None
    if stride != (1, 1, 1) and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool3d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool3d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = (1, 1, 1)

    padder = None
    if pad == 'reflection':
        padding = tuple(k // 2 for k in kernel_size)
        padder = nn.ReplicationPad3d(
            (padding[2], padding[2],
             padding[1], padding[1],
             padding[0], padding[0])
        )
        padding = (0, 0, 0)
    else:
        padding = tuple(k // 2 for k in kernel_size)

    convolver = nn.Conv3d(in_f, out_f, kernel_size, stride, padding=padding, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

class SpectralSpatialConv3d(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kD=3,
                 kH=3,
                 kW=3,
                 stride=(1, 1, 1),
                 bias=True,
                 pad='reflection',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 mode='series'):
        super().__init__()

        assert mode in ['series', 'parallel']
        self.mode = mode

        self.spec = conv(
            in_ch, out_ch,
            kernel_size=(kD, 1, 1),
            stride=stride if mode == 'parallel' else (1, 1, 1),
            pad=pad,
            bias=bias,
            downsample_mode=downsample_mode
        )

        self.spat = conv(
            in_ch if mode == 'parallel' else out_ch,
            out_ch,
            kernel_size=(1, kH, kW),
            stride=stride,
            pad=pad,
            bias=bias,
            downsample_mode=downsample_mode
        )

        if mode == 'parallel':
            self.merge = conv(
                out_ch * 2, out_ch,
                kernel_size=1,
                pad=pad,
                bias=bias,
                downsample_mode=downsample_mode
            )

        self.bn = bn(out_ch)
        self.act = act(act_fun)

    def forward(self, x):
        if self.mode == 'series':
            x = self.spec(x)
            x = self.spat(x)
        else:
            xs = self.spec(x)
            xp = self.spat(x)
            x = torch.cat([xs, xp], dim=1)
            x = self.merge(x)

        x = self.bn(x)
        x = self.act(x)
        return x

class SepConv2p1D(nn.Module):
    """
    Feature-complete replacement for Conv3d using 2+1D factorization.
    Preserves (N, C, D, H, W).
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 kD=3,
                 kH=3,
                 kW=3,
                 stride=(1, 1, 1),
                 bias=True,
                 pad='zero',
                 downsample_mode='stride',
                 act_fun='LeakyReLU'):
        super().__init__()

        assert stride[0] == 1, "Spectral downsampling is not supported"

        self.stride_hw = stride[1:]

        # ---- spectral conv (Conv1d) ----
        self.spec = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kD,
            padding=kD // 2,
            bias=bias
        )

        # ---- spatial conv (Conv2d) ----
        self.spat = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(kH, kW),
            padding=(kH // 2, kW // 2),
            bias=bias
        )

        # ---- optional spatial downsampling ----
        self.downsampler = None
        if self.stride_hw != (1, 1) and downsample_mode != 'stride':
            if downsample_mode == 'avg':
                self.downsampler = nn.AvgPool2d(self.stride_hw)
            elif downsample_mode == 'max':
                self.downsampler = nn.MaxPool2d(self.stride_hw)
            else:
                raise NotImplementedError
        self.use_stride = (self.stride_hw != (1, 1)) and downsample_mode == 'stride'

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(act_fun)

    def forward(self, x):
        # x: (N, C, D, H, W)
        N, C, D, H, W = x.shape

        # ---- spectral conv ----
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(N * H * W, C, D)
        x = self.spec(x)
        C2 = x.shape[1]
        x = x.view(N, H, W, C2, D).permute(0, 3, 4, 1, 2).contiguous()

        # ---- spatial conv ----
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(N * D, C2, H, W)

        if self.use_stride:
            x = self.spat(x)
            x = x[:, :, ::self.stride_hw[0], ::self.stride_hw[1]]
        else:
            x = self.spat(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        x = self.bn(x)
        x = self.act(x)

        # back to (N, C, D, H, W)
        H2, W2 = x.shape[-2:]
        x = x.view(N, D, C2, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        return x