"""Run the existing experiments 01–04 with one model and one image configuration."""
import importlib.util
from pathlib import Path
from common import arguments, load_model


def main():
    args = arguments(__doc__)
    model = load_model(args)
    for filename in ['01_inspect_model.py', '02_patch_reconstruction.py',
                     '03_frequency_analysis.py', '04_verify_nullspace.py']:
        path = Path(__file__).parent / filename
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f'Running {path.name}', flush=True)
        module.run(model, args)


if __name__ == '__main__':
    main()
