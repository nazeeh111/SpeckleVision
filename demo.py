"""Checkpoint inference on stored correlation images. No camera or auto-downloads."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from utils.network import Unet

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_model(checkpoint, device, *, trust_legacy_pickle=False):
    """Safe state_dict loading by default; old whole-model pickle requires opt-in."""
    try:
        saved = torch.load(checkpoint, map_location=device, weights_only=not trust_legacy_pickle)
    except Exception as error:
        if not trust_legacy_pickle and isinstance(error, pickle.UnpicklingError):
            raise ValueError('Checkpoint is not a safe state_dict. Only for a trusted legacy whole-model file, use --trust-legacy-pickle; pickle can execute code.') from error
        raise
    if isinstance(saved, Unet):
        model = saved
    elif isinstance(saved, dict):
        state = saved.get('state_dict', saved)
        if not isinstance(state, dict):
            raise ValueError('state_dict must be a parameter dictionary')
        model = Unet()
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError('Checkpoint must contain the retained Unet or its state_dict')
    return model.to(device).eval()


def prepare_image(path):
    """Preserve ImageFolder RGB, bicubic 128x256, channel0/left half, [-1,1]."""
    with Image.open(path) as image:
        resized = image.convert('RGB').resize((256, 128), Image.Resampling.BICUBIC)
        pixels = np.array(resized, dtype=np.uint8, copy=True)[:, :128, 0]
    values = (pixels.astype(np.float32) / 255.0 - .5) / .5
    return torch.from_numpy(values[None, None])


def run(checkpoint, inputs, output, *, device='cpu', limit=None, trust_legacy_pickle=False):
    checkpoint, inputs, output = Path(checkpoint), Path(inputs), Path(output)
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new directory: {output}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {checkpoint}; no trained weights are bundled or downloaded')
    if not inputs.is_dir():
        raise FileNotFoundError(f'Input collection missing: {inputs}')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    paths = sorted(p for p in inputs.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f'No supported images in {inputs}')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu')
    model = load_model(checkpoint, device, trust_legacy_pickle=trust_legacy_pickle)
    output.mkdir(parents=True)
    records = []
    with torch.inference_mode():
        for index, path in enumerate(paths):
            raw = model(prepare_image(path).to(device))
            if tuple(raw.shape) != (1, 1, 128, 128) or not bool(torch.isfinite(raw).all()):
                raise ValueError(f'Model output must be finite (1,1,128,128): {path}')
            full = raw[0, 0].cpu().numpy().copy()
            crop = full[32:96, 32:96]
            name = f'{index}_recon'
            np.save(output / (name + '.npy'), crop)
            display = np.rint(np.clip((crop + 1) * 127.5, 0, 255)).astype(np.uint8)
            Image.fromarray(display).save(output / (name + '.png'))
            records.append({'input': str(path), 'input_sha256': digest(path),
                            'array': name + '.npy', 'preview': name + '.png',
                            'shape': list(crop.shape), 'array_sha256': digest(output / (name + '.npy'))})
    report = {'checkpoint_sha256': digest(checkpoint), 'device': str(device),
              'torch': torch.__version__, 'mode': 'eval, inference_mode',
              'preprocessing': 'RGB, bicubic128x256, channel0, left128, pixel/255 then (x-.5)/.5',
              'output': 'central64x64 float32 array; grayscale PNG maps [-1,1] to [0,255]',
              'training_provenance': 'Not determined by loading weights; supply and verify provenance separately.',
              'inputs': records}
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/TrainedNetwork.pth')
    parser.add_argument('--input', type=Path, default=ROOT / 'datasets/SubsampledData')
    parser.add_argument('--output', type=Path, default=ROOT / 'Reconstructions/run')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--trust-legacy-pickle', action='store_true', help='Only trusted legacy whole-model checkpoints; pickle can execute code')
    args = parser.parse_args(argv)
    try:
        report = run(args.checkpoint, args.input, args.output, device=args.device,
                     limit=args.limit, trust_legacy_pickle=args.trust_legacy_pickle)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f'SpeckleVision: {error}\n')
    print(json.dumps({'images': len(report['inputs']), 'output': str(args.output), 'device': report['device']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
