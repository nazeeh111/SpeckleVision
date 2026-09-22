# Local verification

Checked 2026-09-22. These are bounded local checks, not research replication.

- 106 numerical source and input-image files preserve the numerical statements and data from the retained pre-publication checkout. 104 files are byte-identical; the remaining files differ only in author/branding comments.
- The default U-net architecture runs on CPU with deterministic random weights and a 1 × 1 × 128 × 128 fixture. Output is finite, preserves shape, and is exactly equal to the retained baseline initialized with the same seed.
- All 96 included input PNGs pass Pillow image verification.

Not run: pretrained inference, training, or CUDA. Trained weights are absent. The default demo explicitly loads to CUDA, and historical correlation helpers call `torch.rfft`/`torch.irfft`, which are absent from the installed PyTorch. Those APIs and numerical behavior were preserved rather than silently ported. A random-weight smoke check is not learned reconstruction or evidence of accuracy.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.
