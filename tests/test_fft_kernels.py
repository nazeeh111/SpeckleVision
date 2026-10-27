"""Spectral kernels checked against independent SciPy spatial correlation."""
import numpy as np
import pytest
from scipy.signal import correlate2d
import torch
from utils import xcorr2


@pytest.mark.parametrize('function', [xcorr2.xcorr2_torch, xcorr2.xcorr2_torch_CPU])
def test_cross_correlation_matches_spatial_reference(function):
    a = torch.arange(24, dtype=torch.float64).reshape(2, 1, 3, 4).requires_grad_()
    b = torch.arange(12, dtype=torch.float64).reshape(2, 1, 2, 3) / 7
    actual = function(a, b)
    expected = np.stack([correlate2d(x[0], y[0], mode='full')
                         for x, y in zip(a.detach().numpy(), b.numpy())])[:, None]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    actual.sum().backward()
    expected_gradient = b.sum(dim=(1, 2, 3)).reshape(2, 1, 1, 1).expand_as(a)
    torch.testing.assert_close(a.grad, expected_gradient)


@pytest.mark.parametrize('function', [xcorr2.xcorr2_torch, xcorr2.xcorr2_torch_CPU])
def test_autocorrelation_preserves_unshifted_output(function):
    a = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    actual = function(a).numpy()[0, 0]
    expected = correlate2d(a.numpy()[0, 0], a.numpy()[0, 0], mode='full')
    np.testing.assert_allclose(np.fft.fftshift(actual), expected, atol=1e-12)


@pytest.mark.parametrize('function', [xcorr2.FourierMod2_nopad, xcorr2.FourierMod2, xcorr2.FourierMod2_CPU])
def test_power_spectrum_matches_numpy(function):
    a = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    result = function(a)
    power = result[0] if isinstance(result, tuple) else result
    shape = (3, 4) if function is xcorr2.FourierMod2_nopad else (5, 7)
    spectrum = np.fft.fft2(a.numpy(), s=shape)
    np.testing.assert_allclose(power.numpy(), abs(spectrum)**2, atol=1e-10)
    if isinstance(result, tuple):
        np.testing.assert_allclose(result[1].numpy(), spectrum.real, atol=1e-12)
        np.testing.assert_allclose(result[2].numpy(), spectrum.imag, atol=1e-12)


def test_complex_spectrum_preserves_real_imaginary_contract():
    a = torch.arange(24, dtype=torch.float64).reshape(1, 1, 3, 4, 2)
    power, real, imag = xcorr2.FourierMod2_nopad_complex(a)
    expected = np.fft.fft2(a.numpy()[..., 0] + 1j*a.numpy()[..., 1])
    np.testing.assert_allclose(power.numpy(), abs(expected)**2, atol=1e-10)
    np.testing.assert_allclose(real.numpy(), expected.real, atol=1e-12)
    np.testing.assert_allclose(imag.numpy(), expected.imag, atol=1e-12)
