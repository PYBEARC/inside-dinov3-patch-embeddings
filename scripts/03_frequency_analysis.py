"""03: characterize the unclipped reconstruction residual in frequency space."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import (arguments, load_model, projections, reconstruct_pair, load_image_tensor,
                    to_img, write_csv, write_manifest, image_output)
from frequency import (rgb_to_gray, fft_magnitude, radial_profile, frequency_band_mse,
                       relative_frequency_error, radial_error_power)


def plot_frequency(original, dino, random, output_dir):
    # Clipping occurs only in the two RGB imshow calls below.
    error_dino = np.abs(original - dino).mean(axis=2)
    error_random = np.abs(original - random).mean(axis=2)
    mse_dino = np.mean((original - dino) ** 2)
    mse_random = np.mean((original - random) ** 2)
    gray_original, gray_dino, gray_random = map(rgb_to_gray, (original, dino, random))
    log_spec_original, mag_original = fft_magnitude(gray_original)
    log_spec_dino, mag_dino = fft_magnitude(gray_dino)
    log_spec_random, mag_random = fft_magnitude(gray_random)
    log_spec_err_dino, _ = fft_magnitude(gray_original - gray_dino)
    log_spec_err_random, _ = fft_magnitude(gray_original - gray_random)
    radial_original = radial_profile(mag_original)
    radial_dino = radial_profile(mag_dino)
    radial_random = radial_profile(mag_random)
    # ----------------------------
    # Figure 1: spectra
    # ----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(np.clip(dino, 0, 1))
    axes[0, 1].set_title(f"DINO Reconstruction\nMSE={mse_dino:.6f}")

    axes[0, 2].imshow(np.clip(random, 0, 1))
    axes[0, 2].set_title(f"Random Reconstruction\nMSE={mse_random:.6f}")

    axes[1, 0].imshow(log_spec_original, cmap="gray")
    axes[1, 0].set_title("Original Spectrum")

    axes[1, 1].imshow(log_spec_dino, cmap="gray")
    axes[1, 1].set_title("DINO Spectrum")

    axes[1, 2].imshow(log_spec_random, cmap="gray")
    axes[1, 2].set_title("Random Spectrum")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "frequency_spectra.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Figure 2: error spectra
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(error_dino, cmap="inferno")
    axes[0].set_title("DINO Pixel Error")

    axes[1].imshow(error_random, cmap="inferno")
    axes[1].set_title("Random Pixel Error")

    axes[2].imshow(np.abs(dino - random).mean(axis=2), cmap="viridis")
    axes[2].set_title("DINO vs Random Difference")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "pixel_error_maps.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(log_spec_err_dino, cmap="gray")
    axes[0].set_title("DINO Error Spectrum")
    axes[0].axis("off")

    axes[1].imshow(log_spec_err_random, cmap="gray")
    axes[1].set_title("Random Error Spectrum")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "error_spectra.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Figure 3: radial profile
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(radial_original, label="Original")
    ax.semilogy(radial_dino, label="DINOv3")
    ax.semilogy(radial_random, label="Random")

    ax.set_xlim(1, 155)  # Ignore DC component
    ax.set_xlabel("Radial Spatial Frequency (FFT bins)")
    ax.set_ylabel("Mean FFT Magnitude (log scale)")
    ax.set_title("Radial Frequency Profile")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "radial_frequency_profile_log.png",
        dpi=180
    )
    plt.close()

    dino_error_profile = radial_error_power(
    gray_original, gray_dino
    )

    random_error_profile = radial_error_power(
        gray_original, gray_random
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(dino_error_profile, label="DINOv3")
    ax.semilogy(random_error_profile, label="Random")

    ax.set_xlim(1, 155)
    ax.set_xlabel("Spatial Frequency Radius (FFT bins)")
    ax.set_ylabel("Mean Squared FFT Error")
    ax.set_title("Radial Reconstruction Error Power")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "radial_error_power.png",
        dpi=180
    )

    plt.close()


def run(model, args):
    rows, band_rows, radial_rows = [], [], []
    matrices = projections(model, args.seed)
    for image_path in args.images:
        rgb = load_image_tensor(image_path, args.device)
        original = to_img(rgb)
        recovered = reconstruct_pair(rgb, matrices)
        arrays = {name: to_img(value) for name, value in recovered.items()}
        output_dir = image_output(args, image_path)
        plot_frequency(original, arrays["dino"], arrays["random"], output_dir)
        gray = rgb_to_gray(original)
        for method, array in arrays.items():
            gray_recon = rgb_to_gray(array)
            rows.append(dict(image=image_path.name, method=method,
                             rgb_mse=float(np.mean((original-array)**2)),
                             grayscale_mse=float(np.mean((gray-gray_recon)**2)),
                             reconstruction_min=float(array.min()), reconstruction_max=float(array.max())))
            bands = frequency_band_mse(gray, gray_recon)
            relative = relative_frequency_error(gray, gray_recon)
            for band in bands:
                band_rows.append(dict(image=image_path.name, method=method, band=band,
                                      grayscale_mse_contribution=float(bands[band]),
                                      relative_error_energy=float(relative[band])))
            _, magnitude = fft_magnitude(gray_recon)
            for radius, (mag, power) in enumerate(zip(radial_profile(magnitude), radial_error_power(gray, gray_recon))):
                radial_rows.append(dict(image=image_path.name, method=method, radius_fft_bins=radius,
                                        mean_fft_magnitude=float(mag), mean_error_power=float(power)))
        _, magnitude = fft_magnitude(gray)
        for radius, mag in enumerate(radial_profile(magnitude)):
            radial_rows.append(dict(image=image_path.name, method="original", radius_fft_bins=radius,
                                    mean_fft_magnitude=float(mag), mean_error_power=0.0))
    write_csv(args.out / "03_frequency_summary.csv", rows)
    write_csv(args.out / "03_frequency_bands.csv", band_rows)
    write_csv(args.out / "03_radial_profiles.csv", radial_rows)
    write_manifest(args, model, "03")


if __name__ == "__main__":
    args = arguments(__doc__)
    run(load_model(args), args)
