"""Optionally regenerate the two scikit-image examples, without overwriting inputs."""
from pathlib import Path
from PIL import Image
from skimage import data

ROOT = Path(__file__).resolve().parents[1]


def main():
    (ROOT / 'data').mkdir(exist_ok=True)
    for name, loader in [('sample.jpg', data.astronaut), ('coffee.jpg', data.coffee)]:
        path = ROOT / 'data' / name
        if path.exists():
            print(f'Keeping existing {path}')
        else:
            Image.fromarray(loader()).save(path)
            print(f'Created {path}')


if __name__ == '__main__':
    main()
