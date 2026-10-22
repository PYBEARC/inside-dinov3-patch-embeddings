"""02: compare pretrained and seeded random pseudoinverse reconstructions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import (arguments, load_model, projections, reconstruct_pair, load_image_tensor,
                    to_img, write_csv, write_manifest, image_output)


def plot_comparison(
    original,
    dino_recovered,
    random_recovered,
    dino_error,
    random_error,
    diff_recon,
    save_path,
    mse_dino,
    mse_random
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(np.clip(dino_recovered, 0, 1))
    axes[0, 1].set_title(f"DINOv3 Reconstruction\nMSE={mse_dino:.6f}")

    axes[0, 2].imshow(np.clip(random_recovered, 0, 1))
    axes[0, 2].set_title(f"Random Reconstruction\nMSE={mse_random:.6f}")

    vmax_error = max(dino_error.max(), random_error.max())

    im1 = axes[1, 0].imshow(dino_error, cmap="inferno", vmin=0, vmax=vmax_error)
    axes[1, 0].set_title("DINOv3 Absolute Error (unclipped)")

    im2 = axes[1, 1].imshow(random_error, cmap="inferno", vmin=0, vmax=vmax_error)
    axes[1, 1].set_title("Random Absolute Error (unclipped)")

    im3 = axes[1, 2].imshow(diff_recon, cmap="viridis")
    axes[1, 2].set_title("DINO vs Random Difference")

    for ax in axes.ravel():
        ax.axis("off")

    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")


def run(model, args):
    rows = []
    matrices = projections(model, args.seed)
    for image_path in args.images:
        rgb = load_image_tensor(image_path, args.device)
        original = to_img(rgb)
        arrays = {name: to_img(value) for name, value in reconstruct_pair(rgb, matrices).items()}
        mse = {name: float(np.mean((original-value)**2)) for name, value in arrays.items()}
        plot_comparison(original, arrays["dino"], arrays["random"],
                        np.abs(original-arrays["dino"]).mean(axis=2),
                        np.abs(original-arrays["random"]).mean(axis=2),
                        np.abs(arrays["dino"]-arrays["random"]).mean(axis=2),
                        image_output(args, image_path) / "reconstruction.png",
                        mse["dino"], mse["random"])
        for name, value in arrays.items():
            rows.append(dict(image=image_path.name, method=name, rgb_mse=mse[name],
                             reconstruction_min=float(value.min()), reconstruction_max=float(value.max())))
    write_csv(args.out / "02_reconstruction.csv", rows)
    write_manifest(args, model, "02")


if __name__ == "__main__":
    args = arguments(__doc__)
    run(load_model(args), args)
