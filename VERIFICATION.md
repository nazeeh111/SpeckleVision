# Local verification

Checked 2026-09-22. These are bounded local checks, not research replication.

- 106 numerical source and input-image files now match the retained upstream snapshot byte-for-byte. The two original author comments were restored on 2026-10-02; numerical statements and input data are unchanged.
- The default U-net architecture runs on CPU with deterministic random weights and a 1 × 1 × 128 × 128 fixture. Output is finite, preserves shape, and is exactly equal to the retained baseline initialized with the same seed.
- All 96 included input PNGs pass Pillow image verification.

Not run: pretrained inference, training, or CUDA. Trained weights are absent. The default demo explicitly loads to CUDA, and historical correlation helpers call `torch.rfft`/`torch.irfft`, which are absent from the installed PyTorch. Those APIs and numerical behavior were preserved rather than silently ported. A random-weight smoke check is not learned reconstruction or evidence of accuracy.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.

## Correlation compatibility update, 26 October 2026

The retained `utils/xcorr2.py` is now intentionally changed to replace removed
`torch.rfft`/`torch.irfft` calls with `torch.fft` complex transforms. Function
arguments, output shapes, normalization and unshifted autocorrelation order are
preserved. Eight CPU regression cases compare unequal-sized batched correlation
and gradients, autocorrelation order, padded/unpadded power and complex input
against independent SciPy/NumPy calculations. Run `python -m pytest -q tests`.
The earlier 106-file identity statement and machine-readable verification record
are historical; this utility no longer matches that snapshot byte for byte.
No trained weights or data were changed. Full pretrained inference, training
and CUDA remain unverified because the trained checkpoint is absent.
