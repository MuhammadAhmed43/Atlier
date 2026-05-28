"""One-shot fetcher: download a curated set of Unsplash fashion images into samples/."""
import io
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
TARGET = 512

# Direct Unsplash image IDs scraped from /t/fashion-beauty and /s/photos/clothing-product
# plus the 6 already-baked-in IDs from Stitch's mockups.
IMAGE_IDS = [
    # Fashion-beauty topic page
    ("model_a", "1779911915323-006b1029133b"),
    ("model_b", "1778301981932-3a047a01eacd"),
    ("model_c", "1778991044742-993ad3f21f19"),
    ("model_d", "1778242921088-b57193c76233"),
    ("model_e", "1778242922564-2d41b033aa92"),
    ("model_f", "1778516631278-fdccc88ba98a"),
    ("model_g", "1778516631416-2694b5921b36"),
    ("model_h", "1778403283539-9e67927cfda2"),
    ("model_i", "1779040622687-42bb00790c67"),
    ("model_j", "1777223130640-d0e29c171358"),
    ("model_k", "1776843370483-4f793bc14739"),
    ("model_l", "1764592358977-a181a6e71af4"),
    ("model_m", "1769451741805-65e173d9a084"),
    ("model_n", "1774804818922-ed70eebce881"),
    ("model_o", "1775592230963-fe488fd06dd7"),
    ("model_p", "1775831726736-3369f057988a"),
    ("model_q", "1775831726700-774b7d48faee"),
    ("model_r", "1775831726875-e3ca6f3ed2f8"),
    ("model_s", "1774804819708-61c4481d193d"),
    ("model_t", "1773932546144-b67f585c7585"),
    # Clothing-product search
    ("garment_a", "1489987707025-afc232f7ea0f"),
    ("garment_b", "1544441893-675973e31985"),
    ("garment_c", "1495121605193-b116b5b9c5fe"),
    ("garment_d", "1625698311031-f0dd15be5144"),
    ("garment_e", "1479064555552-3ef4979f8908"),
    ("garment_f", "1485125639709-a60c3a500bf1"),
    ("garment_g", "1613461920867-9ea115fee900"),
    ("garment_h", "1416339698674-4f118dd3388b"),
    ("garment_i", "1576188973526-0e5d7047b0cf"),
    ("garment_j", "1520923179278-ee25e25e09e4"),
    ("garment_k", "1578932750294-f5075e85f44a"),
    ("garment_l", "1586363104862-3a5e2ab60d99"),
    ("garment_m", "1739169585911-1ad4376bf85e"),
    ("garment_n", "1616761512547-ea151d8a56d5"),
    ("garment_o", "1685875018148-6ac6d41b7c4e"),
    # Stitch mockup IDs (verified)
    ("stitch_tweed",       "1591047139829-d91aecb6caea"),
    ("stitch_silk_green",  "1576566588028-4147f3842f27"),
    ("stitch_leather",     "1551028719-00167b16eac5"),
    ("stitch_trench",      "1539109136881-3be0616acf4b"),
    ("stitch_yellow",      "1515886657613-9f3515b0c78f"),
    ("stitch_silk_dark",   "1496747611176-843222e1e57c"),
]


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 atelier/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save_image(raw: bytes, name: str) -> bool:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        img = img.resize((TARGET, TARGET), Image.LANCZOS)
        img.save(SAMPLES_DIR / f"{name}.png", "PNG", optimize=True)
        return True
    except Exception as e:
        print(f"  ! save failed for {name}: {e}")
        return False


def main():
    saved = 0
    failed = 0
    print(f"Fetching {len(IMAGE_IDS)} images to {SAMPLES_DIR}\n")
    for name, id_ in IMAGE_IDS:
        url = f"https://images.unsplash.com/photo-{id_}?w=1024&fit=crop&auto=format"
        print(f"  [{saved + failed + 1:02d}/{len(IMAGE_IDS)}] {name} ...", end=" ", flush=True)
        try:
            raw = http_get(url)
            if save_image(raw, f"unsplash_{name}"):
                saved += 1
                print("ok")
            else:
                failed += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1
        time.sleep(0.1)

    print(f"\n>> Saved {saved}, failed {failed}, total in samples/: {len(list(SAMPLES_DIR.glob('*')))}")
    if saved == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
