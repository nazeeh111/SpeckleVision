# Chris Metzler
# 2/13/20
"""Differentiable correlations and Fourier power, with the retained lag convention.

Cross-correlation is full linear correlation (negative through positive lags).
Autocorrelation is unshifted: zero lag is index [0,0], as in the original code.
"""
import torch


def flip(x, dim):
    return torch.flip(x, (dim,))


def _image(a):
    if a.ndim != 4 or a.shape[1] != 1 or min(a.shape[0], *a.shape[2:]) < 1:
        raise ValueError('Expected nonempty (batch,1,height,width) grayscale tensor')
    if not a.is_floating_point():
        raise ValueError('Expected a real floating-point tensor')
    return a[:, 0]


def xcorr2_torch(a, b=None):
    """Full real cross-correlation, or unshifted padded autocorrelation.

    An empty b retains the original autocorrelation sentinel. A single b image
    can broadcast across a batch. Dtype/device and autograd are preserved.
    """
    av = _image(a)
    if b is None or b.numel() == 0:
        shape = (2 * a.shape[2] - 1, 2 * a.shape[3] - 1)
        spectrum = torch.fft.fft2(av, s=shape)
        return torch.fft.ifft2(spectrum * spectrum.conj(), s=shape).real[:, None]
    bv = _image(b)
    if b.shape[0] not in (1, a.shape[0]) or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('b must have one or matching batches and the same dtype/device as a')
    shape = (a.shape[2] + b.shape[2] - 1, a.shape[3] + b.shape[3] - 1)
    af = torch.fft.fft2(av, s=shape)
    bf = torch.fft.fft2(torch.flip(bv, (-2, -1)), s=shape)
    return torch.fft.ifft2(af * bf, s=shape).real[:, None]


def xcorr2_torch_CPU(a, b=None):
    # Historical alias already followed a.device; retain that behavior.
    return xcorr2_torch(a, b)


def _fourier(a, *, padded):
    value = _image(a)
    shape = (2 * a.shape[2] - 1, 2 * a.shape[3] - 1) if padded else a.shape[2:]
    spectrum = torch.fft.fft2(value, s=shape)[:, None]
    return spectrum.abs().square(), spectrum.real, spectrum.imag


def FourierMod2_nopad(a):
    return _fourier(a, padded=False)


def FourierMod2_nopad_complex(a):
    if a.ndim != 5 or a.shape[1] != 1 or a.shape[-1] != 2 or not a.is_floating_point():
        raise ValueError('Expected real/imag pairs shaped (batch,1,height,width,2)')
    spectrum = torch.fft.fft2(torch.view_as_complex(a.contiguous()), dim=(-2, -1))
    return spectrum.abs().square(), spectrum.real, spectrum.imag


def FourierMod2(a):
    return _fourier(a, padded=True)


def FourierMod2_CPU(a):
    # This historical helper explicitly produced CPU power only.
    return FourierMod2(a.cpu())[0]
