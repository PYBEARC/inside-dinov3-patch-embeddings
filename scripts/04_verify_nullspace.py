"""04: verify pseudoinverse reconstruction preserves patch and encoder tokens."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from common import (arguments, load_model, load_image_tensor, normalize, denormalize,
                    projection_parameters, extract_patches, reconstruct_from_projection,
                    patch_embeddings, encoder_patch_tokens, compare, to_img,
                    image_output, write_csv, write_manifest)


def run(model, args):
    rows = []
    weight, bias = projection_parameters(model)
    inverse = torch.linalg.pinv(weight)
    for image_path in args.images:
        rgb = load_image_tensor(image_path, args.device)
        original = normalize(rgb)
        recovered_norm = reconstruct_from_projection(extract_patches(original), weight, bias, inverse)
        recovered_rgb = denormalize(recovered_norm)
        # Neither model inputs nor metrics are clipped or re-normalized after reconstruction.
        with torch.inference_mode():
            patch_orig = patch_embeddings(model, original)
            patch_recon = patch_embeddings(model, recovered_norm)
            encoder_orig = encoder_patch_tokens(model, original)
            encoder_recon = encoder_patch_tokens(model, recovered_norm)
        patch_rel, patch_cos = compare(patch_orig, patch_recon)
        encoder_rel, encoder_cos = compare(encoder_orig, encoder_recon)
        row = dict(image=image_path.name, rgb_mse=F.mse_loss(rgb, recovered_rgb).item(),
                   patch_rel_l2=patch_rel, patch_cosine=patch_cos,
                   encoder_rel_l2=encoder_rel, encoder_cosine=encoder_cos,
                   projection_max_abs=(patch_orig-patch_recon).abs().max().item())
        rows.append(row)
        print(row)
        a, b = to_img(rgb), to_img(recovered_rgb)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(a)
        ax[0].set_title('Original')
        ax[1].imshow(b.clip(0, 1))
        ax[1].set_title('Pseudoinverse reconstruction (display clipped)')
        im = ax[2].imshow(abs(a-b).mean(2), cmap='inferno')
        ax[2].set_title('Pixel absolute error (unclipped)')
        fig.colorbar(im, ax=ax[2], fraction=.045)
        for axis in ax:
            axis.axis('off')
        fig.tight_layout()
        fig.savefig(image_output(args, image_path) / 'nullspace.png', dpi=160)
        plt.close(fig)
    write_csv(args.out / '04_nullspace.csv', rows)
    write_manifest(args, model, '04')


if __name__ == '__main__':
    args = arguments(__doc__)
    run(load_model(args), args)
