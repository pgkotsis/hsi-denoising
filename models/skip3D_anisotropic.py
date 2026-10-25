import torch
import torch.nn as nn
from .common3D_anisotropic import *

def skip(
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        spectral_kernel_size=3,
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)
    last_scale = n_scales - 1

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up]*n_scales

    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(n_scales):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(
                input_depth,
                num_channels_skip[i],
                kernel_size=(filter_skip_size, filter_skip_size, filter_skip_size),
                bias=need_bias,
                pad=pad
            ))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(
            input_depth,
            num_channels_down[i],
            kernel_size=(spectral_kernel_size, filter_size_down[i], filter_size_down[i]),
            stride=(2, 2, 2),
            bias=need_bias,
            pad=pad,
            downsample_mode=downsample_mode[i]
        ))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(
            num_channels_down[i],
            num_channels_down[i],
            kernel_size=(spectral_kernel_size, filter_size_down[i], filter_size_down[i]),
            bias=need_bias,
            pad=pad
        ))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_mode[i], align_corners=False))

        model_tmp.add(conv(
            num_channels_skip[i] + k,
            num_channels_up[i],
            kernel_size=(spectral_kernel_size, filter_size_up[i], filter_size_up[i]),
            bias=need_bias,
            pad=pad
        ))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], kernel_size=1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, kernel_size=1, bias=need_bias, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

def skip2(
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        spectral_kernel_size=3,
        conv_mode='series',
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections. Uses separable 3D convolutions,
    but is still built on Conv3d layers.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    last_scale = n_scales - 1

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up]*n_scales

    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(n_scales):

        deeper = nn.Sequential()
        skip_branch = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_branch, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(num_channels_skip[i] +
               (num_channels_up[i + 1] if i < last_scale else num_channels_down[i]))
        )

        if num_channels_skip[i] != 0:
            skip_branch.add(SpectralSpatialConv3d(
                input_depth,
                num_channels_skip[i],
                kD=filter_skip_size,
                kH=filter_skip_size,
                kW=filter_skip_size,
                stride=(1, 1, 1),
                pad=pad,
                bias=need_bias,
                act_fun=act_fun,
                mode=conv_mode
            ))

        deeper.add(SpectralSpatialConv3d(
            input_depth,
            num_channels_down[i],
            kD=spectral_kernel_size,
            kH=filter_size_down[i],
            kW=filter_size_down[i],
            stride=(2, 2, 2),
            pad=pad,
            bias=need_bias,
            act_fun=act_fun,
            mode=conv_mode,
            downsample_mode=downsample_mode[i]
        ))

        deeper.add(SpectralSpatialConv3d(
            num_channels_down[i],
            num_channels_down[i],
            kD=spectral_kernel_size,
            kH=filter_size_down[i],
            kW=filter_size_down[i],
            stride=(1, 1, 1),
            pad=pad,
            bias=need_bias,
            act_fun=act_fun,
            mode=conv_mode
        ))

        deeper_main = nn.Sequential()

        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(
            scale_factor=(2, 2, 2),
            mode=upsample_mode[i],
            align_corners=False
        ))

        model_tmp.add(SpectralSpatialConv3d(
            num_channels_skip[i] + k,
            num_channels_up[i],
            kD=spectral_kernel_size,
            kH=filter_size_up[i],
            kW=filter_size_up[i],
            stride=(1, 1, 1),
            pad=pad,
            bias=need_bias,
            act_fun=act_fun,
            mode=conv_mode
        ))

        if need1x1_up:
            model_tmp.add(conv(
                num_channels_up[i],
                num_channels_up[i],
                kernel_size=1,
                bias=need_bias,
                pad=pad
            ))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(
        num_channels_up[0],
        num_output_channels,
        kernel_size=1,
        bias=need_bias,
        pad=pad
    ))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

def skip3(
        num_input_channels=1,
        num_output_channels=1,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        spectral_kernel_size=3,
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections. Uses separable 3D convolutions.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    last_scale = n_scales - 1

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up]*n_scales

    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(n_scales):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(num_channels_skip[i] +
               (num_channels_up[i + 1] if i < last_scale else num_channels_down[i]))
        )

        # ---- skip branch ----
        if num_channels_skip[i] != 0:
            skip.add(SepConv2p1D(
                input_depth,
                num_channels_skip[i],
                kD=1,
                kH=filter_skip_size,
                kW=filter_skip_size,
                bias=need_bias,
                act_fun=act_fun
            ))

        # ---- encoder ----
        deeper.add(SepConv2p1D(
            input_depth,
            num_channels_down[i],
            kD=spectral_kernel_size,
            kH=filter_size_down[i],
            kW=filter_size_down[i],
            stride=(1, 2, 2),
            bias=need_bias,
            downsample_mode=downsample_mode[i],
            act_fun=act_fun
        ))

        deeper.add(SepConv2p1D(
            num_channels_down[i],
            num_channels_down[i],
            kD=spectral_kernel_size,
            kH=filter_size_down[i],
            kW=filter_size_down[i],
            bias=need_bias,
            act_fun=act_fun
        ))

        deeper_main = nn.Sequential()
        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        # ---- upsampling (spatial only) ----
        deeper.add(nn.Upsample(
            scale_factor=(1, 2, 2),
            mode=upsample_mode[i],
            align_corners=False if 'linear' in upsample_mode[i] else None
        ))

        # ---- decoder ----
        model_tmp.add(SepConv2p1D(
            num_channels_skip[i] + k,
            num_channels_up[i],
            kD=spectral_kernel_size,
            kH=filter_size_up[i],
            kW=filter_size_up[i],
            bias=need_bias,
            act_fun=act_fun
        ))

        if need1x1_up:
            model_tmp.add(SepConv2p1D(
                num_channels_up[i],
                num_channels_up[i],
                kD=1,
                kH=1,
                kW=1,
                bias=need_bias,
                act_fun=act_fun
            ))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(SepConv2p1D(
        num_channels_up[0],
        num_output_channels,
        kD=1,
        kH=1,
        kW=1,
        bias=need_bias,
        act_fun='none'
    ))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model