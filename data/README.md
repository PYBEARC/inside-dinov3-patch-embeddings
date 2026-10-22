# Image sources / 圖片來源

The committed image bytes are the reference inputs. Checksums and download provenance are recorded in [sources.json](sources.json), and every experiment manifest records the input SHA-256 hashes.

| File | Creator / source | Recorded license | Preparation |
| --- | --- | --- | --- |
| `sample.jpg` | NASA, via [`skimage.data.astronaut()`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) | Public domain, as documented by scikit-image | Existing PIL JPEG export retained. |
| `coffee.jpg` | Rachel Michetti; courtesy of Pikolo Espresso Bar, via [`skimage.data.coffee()`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee) | CC0 | Existing PIL JPEG export retained. |
| `street.jpg` | Ali Sabbagh, [Paris Street scene (42791982090).jpg](https://commons.wikimedia.org/wiki/File:Paris_Street_scene_(42791982090).jpg), originally [on Flickr](https://www.flickr.com/photos/32174721@N04/42791982090) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) as recorded on Commons; Flickr license reviewed there on 16 November 2019 | Original 4,311 × 2,597 JPEG, unchanged bytes. |

`street.jpg` is a street-scene photograph, not a driving benchmark dataset. It replaces the former NVIDIA-sourced `driving.png` for public distribution. The former image and historical figures remain in the ignored local `.local-archive/` directory. Public CSVs were recomputed for the replacement input with the same methodology.

All inputs undergo the same RGB conversion, 224 × 224 bicubic resize, and normalization during each experiment. These transformations are reflected in the generated figures; clipping is used only for RGB display.

Regenerating sample JPEGs with other library versions can change their bytes. Prefer the committed files for reproduction. `scripts/download_sample_images.py` optionally creates missing scikit-image examples and never overwrites existing images (requires `pip install scikit-image`). The street JPEG can be downloaded from the URL in `sources.json` and verified against its SHA-256. Image license information applies to the source images and does not assign a license to the project code or model weights.

## 繁體中文

三張圖片的來源、授權與 SHA-256 見上表及 [sources.json](sources.json)。`street.jpg` 是 Ali Sabbagh 的巴黎街景照片，來源頁標示 CC0 1.0；專案保留原始 JPEG 位元組，實驗時才套用既有前處理。Coffee 圖片由 Rachel Michetti 以 CC0 提供；astronaut 圖片依 scikit-image 文件記載屬公有領域。

原 NVIDIA 圖片與含有它的歷史圖表僅保留於本機 `.local-archive/`，不加入 Git。公開版結果已改用街景照片重新計算，不能與舊 driving 圖片的數值混用。重現時建議直接使用 repository 內的圖片，以避免不同 JPEG 編碼版本造成數值差異。
