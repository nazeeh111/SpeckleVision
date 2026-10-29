# SpeckleVision

**Hidden-scene structure from noisy speckle correlations.**

A U-net maps a correlation image into an estimated hidden scene. Inspect the bundled correlation inputs, follow the inference path, or set up the compatible research environment below.

<img src="datasets/SubsampledData/test/5_n2_corr_x100_nzero1_npatch1.png" width="508" alt="Retained grayscale sample correlation PNG, shown as stored with two side-by-side panels of noisy correlation structure.">

*Bundled sample input. Neither panel is a reconstructed scene. The demo consumes only the left half after resizing; no trained output is bundled.*

[Browse the inputs](#inspect-the-bundled-inputs) · [Follow the inference path](#what-the-demo-does) · [Run with trained weights](#trained-inference) · [Verification limits](VERIFICATION.md)

## Inspect the bundled inputs

You can inspect the data without a checkpoint or GPU. The three folders contain **96 grayscale PNGs**, each **254 × 127 pixels**:

| Retained input collection | Files | Browse |
| --- | --- | --- |
| Subsampled correlations; the default demo root | 28 | [`SubsampledData/test`](datasets/SubsampledData/test) |
| Short-exposure correlations | 28 | [`ShortExposureData/test`](datasets/ShortExposureData/test) |
| Simulated examples | 40 | [`SimData/test`](datasets/SimData/test) |

For a small executable input check, install Pillow in your Python environment and run this from the repository root:

```bash
python -m pip install Pillow
python - <<'PY'
from pathlib import Path
from PIL import Image

paths = sorted(Path('datasets').rglob('*.png'))
assert len(paths) == 96, f'Expected 96 bundled PNGs, found {len(paths)}'
for path in paths:
    with Image.open(path) as image:
        assert image.size == (254, 127) and image.mode == 'L', str(path)
        image.verify()
print('PASS: 96 grayscale input PNGs, each 254 x 127; no inference run')
PY
```

This checks the bundled files, not the model. A failed count or image check means the checkout differs from this input inventory; check the reported file or restore the missing sample before proceeding.

## What the demo does

[`demo.py`](demo.py) defines the exact path from a stored file to an output:

1. Load image files in sorted path order, including the collection's `test/` subfolder.
2. Resize each PNG to **256 wide × 128 high** and normalize the pixels.
3. Take channel 0 and the **left 128 columns** as a `1 × 1 × 128 × 128` network input. The stored right half is not a network prediction.
4. Run the loaded U-net in evaluation mode, extract its **central 64 × 64 crop**, and save a float32 `.npy` array and grayscale PNG, with input/checkpoint hashes in `run.json`.

The central crop is a display/output convention in this source, not the full network output. The [`network architecture`](utils/network.py), [`correlation utilities`](utils/xcorr2.py) and [`training-data generator`](create_training_data.m) are available for inspection.

<a id="inference"></a>

## Trained inference

The demo supports CPU by default and requires a compatible checkpoint. It never downloads weights or uses a camera. Install its inference dependencies, then supply legally obtained weights:

```bash
python -m pip install -r requirements-inference.txt
python demo.py --checkpoint /path/to/checkpoint.pth --device cpu \
  --input datasets/SubsampledData --output Reconstructions/my-run --limit 2
```

The output directory must be new; earlier results are never overwritten. Safe `state_dict` loading is the default. The original training code saved a complete Python model; **only for a trusted legacy checkpoint**, add `--trust-legacy-pickle`. This explicitly enables Python pickle loading, which can execute code. The unchanged U-net architecture is required for raw state dictionaries. The checkpoint hash identifies bytes, not training history or accuracy.

The original [Stanford Box checkpoint link](https://stanford.box.com/s/zyzdhzyk68xw6z8whpcup5cman55esl7) returned HTTP404 during the 2026-10-29 check. No trained checkpoint or new trained reconstruction is bundled. External checkpoint/data redistribution terms still require verification. No substitute or random-weight reconstruction is presented as a learned result.

Without weights, the independently working correlation/Fourier subset and software contracts can be checked:

```bash
python -m unittest discover -s tests -v
```

The tests compare the modern FFT implementation against direct spatial correlation and NumPy Fourier references, including lag order and differentiable training loss. A separate actual U-net run uses **temporary seeded random weights** with two bundled correlation images; it checks input/output contracts and repeatability, not reconstruction quality. Inference uses evaluation mode, so dropout and batch-normalization updates do not make repeated results stochastic.

## Training

Place the BSD-500 JPEG images directly in [`datasets/BSD500/`](datasets/BSD500), as expected by [`create_training_data.m`](create_training_data.m). The [BSD-500 source](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) is external. Run the generator in MATLAB with its image-processing functions available, then `python train_network.py` in a compatible PyTorch environment. The generator writes `datasets/Edges_b80_gammap0.015_res64/`, the training script's default input root.

Model architecture, normalization, training defaults and sample data remain unchanged. The correlation/Fourier helpers now use modern `torch.fft` with their original lag and scaling conventions; spatial, NumPy and gradient tests verify them. The full MATLAB/BSD500 training route has not been executed. [Raw speckle measurements](https://stanford.box.com/s/lta3jxsgrj49uio9v5qgko1betj8z77w) are referenced separately by the original project.

<a id="verification"></a>
<a id="source-and-license"></a>

## Verification and license

The [verification record](VERIFICATION.md) describes a prior randomly initialized CPU network check. That check is not trained reconstruction or evidence of accuracy. Current CPU numerical and inference-contract tests extend this evidence; no new trained inference, training or CUDA result is claimed here.

[Source provenance](NOTICE.md) · [Original license](LICENSE) · [New-addition terms](LICENSE-branding) · [U-net component terms](THIRD_PARTY_NOTICES.md).
