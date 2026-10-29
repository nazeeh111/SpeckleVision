# Local verification

Checked 2026-09-22. These are bounded local checks, not research replication.

- 106 numerical source and input-image files now match the retained upstream snapshot byte-for-byte. The two original author comments were restored on 2026-10-02; numerical statements and input data are unchanged.
- The default U-net architecture runs on CPU with deterministic random weights and a 1 × 1 × 128 × 128 fixture. Output is finite, preserves shape, and is exactly equal to the retained baseline initialized with the same seed.
- All 96 included input PNGs pass Pillow image verification.

Not run: pretrained inference, training, or CUDA. Trained weights are absent. The default demo explicitly loads to CUDA, and historical correlation helpers call `torch.rfft`/`torch.irfft`, which are absent from the installed PyTorch. Those APIs and numerical behavior were preserved rather than silently ported. A random-weight smoke check is not learned reconstruction or evidence of accuracy.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.

## Current numerical/input contracts, 2026-10-29

The earlier source-identity receipt above describes the historical snapshot.
The correlation helpers now use modern `torch.fft`; direct spatial correlation,
NumPy Fourier references, zero-lag placement, changed inputs and finite-difference
gradient checks establish their retained numerical contracts. The current demo
loads to CPU by default, explicitly selects evaluation/inference mode, preserves
RGB/bicubic/channel/left-half/central-crop semantics and writes fresh results
with hashes. Safe state dictionaries work by default; legacy whole-model pickle
requires explicit trust. The U-net architecture and bundled input bytes are unchanged.

Run `python -m unittest discover -s tests -v` after installing
`requirements-inference.txt`. These checks include an actual U-net on two bundled
correlation inputs with temporary seeded random weights, not trained reconstruction
or task quality. The Stanford Box checkpoint link returned HTTP404 during the
check; no checkpoint was downloaded or redistributed. Compatible trained
weights with verified provenance/license and independent reconstruction quality
remain unresolved. Full MATLAB/BSD500 training and CUDA were not executed.
