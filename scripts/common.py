"""Shared preprocessing, affine reconstruction, configuration, and result I/O."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PATCH = 16
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--images', '--image', type=Path, nargs='+', help='Override configured images; paths relative to current directory.')
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'reproduced')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--local-files-only', action='store_true', help='Use the Hugging Face cache without network access.')
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.settings = json.loads(args.config.read_text())
    args.seed = args.settings['seed']
    args.images = [p.resolve() for p in args.images] if args.images else [(args.config.parent / p).resolve() for p in args.settings['images']]
    if not args.images:
        parser.error('Provide at least one image.')
    if len({p.stem for p in args.images}) != len(args.images):
        parser.error('Image stems must be unique to prevent output collisions.')
    for path in args.images:
        if not path.is_file():
            parser.error(f'Image not found: {path}')
    args.out = args.out.resolve()
    if args.out == ROOT / 'results' / 'legacy' or ROOT / 'results' / 'legacy' in args.out.parents:
        parser.error('Choose an output directory outside the preserved legacy results.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def load_model(args):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.settings['model_id'], revision=args.settings['revision'],
                                      local_files_only=args.local_files_only).to(args.device).eval()
    projection = model.embeddings.patch_embeddings
    if tuple(projection.weight.shape[1:]) != (3, PATCH, PATCH) or tuple(projection.stride) != (PATCH, PATCH):
        raise ValueError('These experiments require non-overlapping RGB 16x16 patches.')
    return model


def load_image_tensor(image_path, device='cpu'):
    # Match the original PIL RGB -> square resize; make PIL's RGB default explicit.
    with Image.open(image_path) as image:
        image = image.convert('RGB').resize((SIZE, SIZE), resample=Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.


def normalize(rgb):
    return (rgb - MEAN.to(rgb.device)) / STD.to(rgb.device)


def denormalize(x):
    return x * STD.to(x.device) + MEAN.to(x.device)


def extract_patches(x):
    return F.unfold(x, kernel_size=PATCH, stride=PATCH).transpose(1, 2)


def projection_parameters(model):
    projection = model.embeddings.patch_embeddings
    weight = projection.weight.detach().float().flatten(1)
    bias = projection.bias.detach().float() if projection.bias is not None else None
    return weight, bias


def projections(model, seed):
    weight, bias = projection_parameters(model)
    # Generate on CPU as in original 02/03, independent of image order and CUDA RNG.
    torch.manual_seed(seed)
    random_weight = (torch.randn_like(weight.cpu()) / (weight.shape[1] ** 0.5)).to(weight.device)
    random_bias = torch.zeros(random_weight.shape[0], dtype=weight.dtype, device=weight.device)
    return {name: (w, b, torch.linalg.pinv(w)) for name, w, b in
            [('dino', weight, bias), ('random', random_weight, random_bias)]}


def reconstruct_from_projection(patches, weight, bias, inverse=None):
    z = F.linear(patches, weight, bias)
    inverse = torch.linalg.pinv(weight) if inverse is None else inverse
    recovered = F.linear(z - (bias if bias is not None else 0), inverse)
    return F.fold(recovered.transpose(1, 2), output_size=(SIZE, SIZE), kernel_size=PATCH, stride=PATCH)


def reconstruct_pair(rgb, matrices):
    patches = extract_patches(normalize(rgb))
    return {name: denormalize(reconstruct_from_projection(patches, *parameters))
            for name, parameters in matrices.items()}


def patch_embeddings(model, normalized):
    return model.embeddings.patch_embeddings(normalized).flatten(2).transpose(1, 2)


def encoder_patch_tokens(model, normalized):
    out = model(pixel_values=normalized).last_hidden_state
    return out[:, 1 + model.config.num_register_tokens:, :]


def compare(a, b):
    a, b = a.float(), b.float()
    rel_l2 = torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12)
    cosine = F.cosine_similarity(a, b, dim=-1).mean()
    return float(rel_l2), float(cosine)


def to_img(rgb):
    """Return raw HWC values. Clip only at the RGB display call site."""
    return rgb.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()


def image_output(args, image_path):
    path = args.out / 'figures' / image_path.stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path}')


def write_manifest(args, model, experiment):
    packages = ['torch', 'transformers', 'huggingface_hub', 'safetensors', 'numpy', 'Pillow', 'matplotlib']
    data = dict(experiment=experiment, model_id=args.settings['model_id'],
                requested_revision=args.settings['revision'], resolved_revision=getattr(model.config, '_commit_hash', None),
                seed=args.seed, device=args.device, dtype='float32', python=platform.python_version(),
                packages={name: version(name) for name in packages},
                preprocessing=dict(size=[SIZE, SIZE], color='RGB', resize='PIL bicubic, no crop',
                                   mean=MEAN.flatten().tolist(), std=STD.flatten().tolist()),
                frequency=dict(grayscale=[0.2989, 0.5870, 0.1140], radius='Euclidean, centered FFT, divided by corner radius',
                               low_max=0.15, mid_max=0.45, power_fft_norm='ortho', magnitude_fft_norm='backward',
                               radial_bins='floor radius in FFT bins'),
                numerical_reconstruction='unclipped',
                images=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.images],
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').glob('*.py'))})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'manifest_{experiment}.json').write_text(json.dumps(data, indent=2) + '\n')
