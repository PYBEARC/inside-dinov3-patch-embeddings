"""01: inspect model dimensions and patch-projection rank."""
import torch
from common import arguments, load_model, projection_parameters, write_csv, write_manifest, SIZE


def run(model, args):
    weight, _ = projection_parameters(model)
    rank = torch.linalg.matrix_rank(weight).item()
    torch.manual_seed(args.seed)
    dummy = torch.randn(1, 3, SIZE, SIZE, device=args.device)
    with torch.inference_mode():
        output = model(pixel_values=dummy).last_hidden_state
    row = dict(model_type=model.config.model_type, hidden_size=model.config.hidden_size,
               patch_size=model.config.patch_size, projection_rows=weight.shape[0],
               projection_columns=weight.shape[1], rank=rank, nullity=weight.shape[1]-rank,
               output_shape=str(tuple(output.shape)))
    for key, value in row.items():
        print(f'{key}: {value}')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f'{name}: weight={tuple(module.weight.shape)}, kernel={module.kernel_size}, stride={module.stride}')
    write_csv(args.out / '01_model.csv', [row])
    write_manifest(args, model, '01')


if __name__ == '__main__':
    args = arguments(__doc__)
    run(load_model(args), args)
