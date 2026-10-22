"""Original grayscale, radial-bin, and frequency-band definitions (unclipped inputs)."""
import numpy as np


def rgb_to_gray(img):
    # img: H x W x 3; reconstructed values may lie outside [0, 1]
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return gray


def fft_magnitude(gray_img):
    fft2 = np.fft.fft2(gray_img)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude, magnitude


def radial_profile(magnitude):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(np.int32)

    radial_mean = np.bincount(r.ravel(), magnitude.ravel()) / np.bincount(r.ravel())
    return radial_mean


def band_masks(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = r.max()

    low = r <= 0.15 * r_max
    mid = (r > 0.15 * r_max) & (r <= 0.45 * r_max)
    high = r > 0.45 * r_max

    return {"low": low, "mid": mid, "high": high}



def frequency_band_mse(original, reconstructed):
    """
    Input: grayscale images, shape [H, W].
    Returns frequency-band contribution to image MSE.
    """
    error = original - reconstructed

    # Orthogonal FFT normalization
    spectrum = np.fft.fftshift(
        np.fft.fft2(error, norm="ortho")
    )

    power = np.abs(spectrum) ** 2
    masks = band_masks(error.shape)

    H, W = error.shape

    results = {}
    for name, mask in masks.items():
        results[name] = power[mask].sum() / (H * W)

    return results

def radial_error_power(original, reconstructed):
    error = original - reconstructed

    spectrum = np.fft.fftshift(
        np.fft.fft2(error, norm="ortho")
    )

    power = np.abs(spectrum) ** 2
    return radial_profile(power)

def relative_frequency_error(original, reconstructed):
    # Original image spectrum
    F_original = np.fft.fftshift(
        np.fft.fft2(original, norm="ortho")
    )

    # Reconstruction error spectrum
    error = original - reconstructed
    F_error = np.fft.fftshift(
        np.fft.fft2(error, norm="ortho")
    )

    masks = band_masks(original.shape)

    results = {}

    for name, mask in masks.items():
        original_energy = np.sum(np.abs(F_original[mask]) ** 2)
        error_energy = np.sum(np.abs(F_error[mask]) ** 2)

        results[name] = error_energy / (original_energy + 1e-12)

    return results
