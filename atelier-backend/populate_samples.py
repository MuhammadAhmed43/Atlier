"""
Populate atelier-backend/samples/ with fashion reference images.

Three modes, in priority order:
  1. --h5 PATH       Extract evenly-spaced samples from a local FashionGen H5 file
  2. --hf            Stream from a public HuggingFace fashion dataset (default)
  3. --stitch        Just copy the 6 Unsplash URLs Stitch baked into our mockups

Examples:
    python populate_samples.py --hf -n 60
    python populate_samples.py --h5 "C:/path/to/fashiongen_256_256_train.h5" -n 100
    python populate_samples.py --stitch
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 512  # display-friendly; backend will further center-crop to 256 at inference


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def save_image(img: Image.Image, name: str) -> Path:
    img = center_crop_square(img.convert("RGB")).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    path = SAMPLES_DIR / f"{name}.png"
    img.save(path, "PNG", optimize=True)
    return path


# ─────────────────────────── H5 extractor ───────────────────────────

def extract_from_h5(h5_path: Path, n: int) -> int:
    import h5py

    if not h5_path.is_file():
        print(f"[h5] not found: {h5_path}", file=sys.stderr)
        return 0

    saved = 0
    with h5py.File(h5_path, "r") as f:
        print(f"[h5] keys: {list(f.keys())}")
        for candidate in ("input_image", "images", "image", "input_images", "data"):
            if candidate in f:
                key = candidate
                break
        else:
            # Auto-detect by shape
            key = None
            for k in f.keys():
                ds = f[k]
                if isinstance(ds, h5py.Dataset) and len(ds.shape) == 4:
                    key = k
                    break
            if key is None:
                print("[h5] could not find image dataset", file=sys.stderr)
                return 0

        ds = f[key]
        total = ds.shape[0]
        indices = np.linspace(0, total - 1, min(n, total), dtype=int)
        print(f"[h5] extracting {len(indices)} of {total} from key '{key}' (shape {ds.shape})")

        for i, idx in enumerate(indices):
            arr = ds[idx]
            if arr.shape[0] == 3 and arr.ndim == 3:  # (C, H, W) → (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (arr * 255 if arr.max() <= 1.0 else arr).astype(np.uint8)
            save_image(Image.fromarray(arr), f"fashiongen_{idx:06d}")
            saved += 1
            if (i + 1) % 20 == 0:
                print(f"[h5]   {i + 1}/{len(indices)}")
    return saved


# ─────────────────────────── HuggingFace dataset ───────────────────────────

# Marqo/fashion-product-images-small is a curated set of e-commerce style fashion photos.
# We stream it (no full download) and pull the first N samples.
DEFAULT_HF_DATASET = "ceyda/fashion-products-small"


def extract_from_hf(dataset_id: str, n: int) -> int:
    from datasets import load_dataset

    print(f"[hf] streaming {dataset_id} (first {n} samples)…")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    saved = 0
    for i, row in enumerate(ds):
        if saved >= n:
            break
        # Try common image keys
        img = None
        for key in ("image", "img", "Image", "picture"):
            if key in row and row[key] is not None:
                v = row[key]
                if isinstance(v, Image.Image):
                    img = v
                elif isinstance(v, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(v))
                elif isinstance(v, dict) and "bytes" in v:
                    img = Image.open(io.BytesIO(v["bytes"]))
                break
        if img is None:
            continue
        try:
            label_parts = []
            for lk in ("articleType", "subCategory", "masterCategory", "label", "productDisplayName"):
                if lk in row and row[lk]:
                    label_parts.append(str(row[lk]).replace(" ", "_").lower())
                    break
            label = label_parts[0] if label_parts else "fashion"
            save_image(img, f"hf_{saved:03d}_{label}")
            saved += 1
            if saved % 10 == 0:
                print(f"[hf]   {saved}/{n}")
        except Exception as e:
            print(f"[hf]   skip row {i}: {e}")
    return saved


# ─────────────────────────── Stitch fallback ───────────────────────────

STITCH_URLS = [
    ("https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=1024", "tweed_jacket"),
    ("https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=1024", "emerald_silk_dress"),
    ("https://images.unsplash.com/photo-1551028719-00167b16eac5?w=1024", "leather_jacket_detail"),
    ("https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=1024", "structured_trench"),
    ("https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=1024", "yellow_summer_dress"),
    ("https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=1024", "midnight_silk_slip"),
]


def extract_from_stitch() -> int:
    import urllib.request

    saved = 0
    for url, name in STITCH_URLS:
        try:
            print(f"[stitch] {name} …")
            req = urllib.request.Request(url, headers={"User-Agent": "atelier/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            save_image(Image.open(io.BytesIO(data)), f"stitch_{name}")
            saved += 1
        except Exception as e:
            print(f"[stitch]   failed {name}: {e}")
    return saved


# ─────────────────────────── Main ───────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, default=None, help="Local FashionGen H5 file")
    parser.add_argument("--hf", action="store_true", help="Use HuggingFace fashion dataset (default if no other source)")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET, help="HF dataset id")
    parser.add_argument("--stitch", action="store_true", help="Download the 6 Unsplash URLs from Stitch's mockups")
    parser.add_argument("-n", type=int, default=60, help="Target number of samples")
    parser.add_argument("--clear", action="store_true", help="Wipe samples/ first")
    args = parser.parse_args()

    if args.clear:
        for p in SAMPLES_DIR.glob("*"):
            if p.is_file():
                p.unlink()
        print(f"[clear] wiped {SAMPLES_DIR}")

    total = 0
    if args.h5:
        total += extract_from_h5(args.h5, args.n)
    elif args.stitch and not args.hf:
        total += extract_from_stitch()
    else:
        # default
        total += extract_from_hf(args.hf_dataset, args.n)

    print(f"\n✅ saved {total} samples to {SAMPLES_DIR}")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
