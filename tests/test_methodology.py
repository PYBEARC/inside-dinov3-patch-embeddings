"""Model-free checks for the numerical invariants used by experiments 01–04."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import (SIZE, normalize, denormalize, extract_patches,
                    reconstruct_from_projection, load_image_tensor, ROOT)
from frequency import band_masks, frequency_band_mse, rgb_to_gray


class MethodologyTests(unittest.TestCase):
    def test_original_preprocessing(self):
        for path in (ROOT / 'data').iterdir():
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            with Image.open(path) as image:
                expected = torch.tensor(np.array(image.convert('RGB').resize((224, 224))), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.
            torch.testing.assert_close(load_image_tensor(path), expected, rtol=0, atol=0)

    def test_affine_projection_and_nullspace(self):
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(32, 768, generator=generator)
        bias = torch.randn(32, generator=generator)
        rgb = torch.rand(1, 3, SIZE, SIZE, generator=generator)
        normalized = normalize(rgb)
        patches = extract_patches(normalized)
        recovered = reconstruct_from_projection(patches, weight, bias)
        recovered_patches = extract_patches(recovered)
        torch.testing.assert_close(F.linear(patches, weight, bias),
                                   F.linear(recovered_patches, weight, bias), rtol=2e-4, atol=2e-4)
        residual_projection = F.linear(patches-recovered_patches, weight)
        self.assertLess(float(residual_projection.abs().max()), 2e-4)
        torch.testing.assert_close(denormalize(normalized), rgb)
        self.assertGreater(float(F.mse_loss(recovered, normalized)), 0)

    def test_unclipped_fft_parseval_and_band_boundaries(self):
        rng = np.random.default_rng(42)
        original = rng.random((224, 224, 3))
        recovered = rng.normal(0.5, 1.0, size=original.shape)
        gray, gray_recovered = rgb_to_gray(original), rgb_to_gray(recovered)
        bands = frequency_band_mse(gray, gray_recovered)
        self.assertAlmostEqual(sum(bands.values()), float(np.mean((gray-gray_recovered)**2)), places=12)
        clipped = frequency_band_mse(gray, rgb_to_gray(np.clip(recovered, 0, 1)))
        self.assertGreater(abs(sum(bands.values())-sum(clipped.values())), 0.1)
        masks = band_masks(gray.shape)
        np.testing.assert_array_equal(sum(masks.values()), np.ones(gray.shape))
        y, x = np.indices(gray.shape)
        radius = np.sqrt((x-112)**2+(y-112)**2)
        np.testing.assert_array_equal(masks['low'], radius <= .15*radius.max())
        np.testing.assert_array_equal(masks['mid'], (radius > .15*radius.max()) & (radius <= .45*radius.max()))
        np.testing.assert_array_equal(masks['high'], radius > .45*radius.max())


if __name__ == '__main__':
    unittest.main()
