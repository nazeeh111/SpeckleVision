"""Independent spatial/Fourier references and differentiability, with no weights."""
import unittest
import numpy as np
import torch
from utils import xcorr2 as corr


def spatial(a, b):
    out = np.zeros((a.shape[0]+b.shape[0]-1, a.shape[1]+b.shape[1]-1))
    # Direct convolution with spatially reversed b, independent of FFT code.
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            out[y:y+b.shape[0],x:x+b.shape[1]] += a[y,x]*b[::-1,::-1]
    return out


class NumericalTests(unittest.TestCase):
    def test_cross_and_autocorrelation_lags_batches_and_changed_input(self):
        for seed in (4, 19):
            rng = np.random.default_rng(seed)
            av, bv = rng.normal(size=(2, 3, 5)), rng.normal(size=(1, 2, 4))
            a, b = torch.tensor(av[:,None]), torch.tensor(bv[:,None])
            expected = np.stack([spatial(v,bv[0]) for v in av])[:,None]
            np.testing.assert_allclose(corr.xcorr2_torch(a,b), expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(corr.xcorr2_torch_CPU(a,b), corr.xcorr2_torch(a,b))
            autocorr = np.stack([spatial(v,v) for v in av])[:,None]
            actual = corr.xcorr2_torch(a).numpy()
            np.testing.assert_allclose(np.fft.fftshift(actual,axes=(-2,-1)), autocorr, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(corr.xcorr2_torch(a,torch.tensor([])), actual)
            np.testing.assert_allclose(actual[:,0,0,0], (av*av).sum(axis=(1,2)), atol=1e-12)

    def test_all_fourier_helpers_match_independent_numpy(self):
        a = torch.arange(30,dtype=torch.float64).view(2,1,3,5)/11
        for padded, fn in [(False,corr.FourierMod2_nopad),(True,corr.FourierMod2)]:
            shape=(5,9) if padded else (3,5)
            expected=np.fft.fft2(a.numpy(),s=shape,axes=(-2,-1))
            power, real, imag=fn(a)
            for actual, ref in [(power,abs(expected)**2),(real,expected.real),(imag,expected.imag)]:
                np.testing.assert_allclose(actual,ref,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(corr.FourierMod2_CPU(a),corr.FourierMod2(a)[0])
        paired=torch.stack([a,a/7],dim=-1)
        expected=np.fft.fft2(a.numpy()+1j*a.numpy()/7,axes=(-2,-1))
        for actual, ref in zip(corr.FourierMod2_nopad_complex(paired),[abs(expected)**2,expected.real,expected.imag]):
            np.testing.assert_allclose(actual,ref,rtol=1e-12,atol=1e-12)

    def test_training_autocorrelation_loss_has_correct_gradients(self):
        x=torch.randn(1,1,2,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda value:corr.xcorr2_torch(value),(x,),atol=1e-5))

    def test_bad_shapes_dtype_and_batch_fail_clearly(self):
        for a in (torch.ones(2,3),torch.ones(1,2,3,3),torch.ones(1,1,3,3,dtype=torch.int64)):
            with self.assertRaises(ValueError):corr.xcorr2_torch(a)
        with self.assertRaises(ValueError):corr.xcorr2_torch(torch.ones(2,1,3,3),torch.ones(3,1,2,2))


if __name__=='__main__':unittest.main()
