#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import glob
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

# Prefer huggingface datasets because it handles Image feature nicely.
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError(
        "This script requires `datasets`. Install with: pip install datasets pyarrow pillow tqdm"
    ) from e


def _find_parquets(data_dir: str) -> Dict[str, List[str]]:
    split_patterns = {
        "train": "train-*.parquet",
        "validation": "validation-*.parquet",
        "test": "test-*.parquet",
    }
    out: Dict[str, List[str]] = {}
    for split, pat in split_patterns.items():
        paths = sorted(glob.glob(os.path.join(data_dir, pat)))
        if not paths:
            raise FileNotFoundError(f"No parquet files found for split={split} using pattern {pat} under {data_dir}")
        out[split] = paths
    return out


def _pil_to_jpeg_bytes(img: Image.Image, jpeg_quality: int = 95) -> bytes:
    # Some PathVQA images are CMYK; convert to RGB for jpg saving.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # keep grayscale as L; JPEG supports it, but convert to RGB if you prefer
        pass

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def _get_image_bytes(image_field: Any) -> Tuple[bytes, Image.Image]:
    """
    Handles common representations:
    - PIL.Image.Image (most common when datasets decodes Image feature)
    - dict with 'bytes' (some datasets store as {'bytes':..., 'path':...})
    - bytes directly
    """
    if isinstance(image_field, Image.Image):
        img = image_field
        b = _pil_to_jpeg_bytes(img)
        return b, img

    if isinstance(image_field, dict):
        # datasets Image feature sometimes gives {"path":..., "bytes":...} when not decoded
        if "bytes" in image_field and image_field["bytes"] is not None:
            raw = image_field["bytes"]
            img = Image.open(io.BytesIO(raw))
            b = _pil_to_jpeg_bytes(img)
            return b, img
        if "path" in image_field and image_field["path"]:
            img = Image.open(image_field["path"])
            b = _pil_to_jpeg_bytes(img)
            return b, img

    if isinstance(image_field, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_field))
        b = _pil_to_jpeg_bytes(img)
        return b, img

    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_split(
    ds_split,
    images_dir: str,
    image_prefix: str,
    jpeg_quality: int,
    existing_hash_to_name: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """
    Returns:
      all_samples_json, yesno_samples_json, num_saved_images, num_total_samples
    """
    all_samples: List[Dict[str, Any]] = []
    yesno_samples: List[Dict[str, Any]] = []

    num_saved = 0
    num_total = len(ds_split)

    for ex in tqdm(ds_split, total=num_total, desc="extract"):
        # read fields
        q = ex.get("question")
        a = ex.get("answer")

        if q is None or a is None:
            raise KeyError(f"Missing 'question' or 'answer' in example keys={list(ex.keys())}")

        # image extraction
        image_field = ex.get("image")
        if image_field is None:
            raise KeyError(f"Missing 'image' in example keys={list(ex.keys())}")

        jpeg_bytes, _ = _get_image_bytes(image_field)
        h = _md5_hex(jpeg_bytes)

        if h in existing_hash_to_name:
            fname = existing_hash_to_name[h]
        else:
            # stable filename
            fname = f"{image_prefix}_{h[:16]}.jpg"
            out_path = os.path.join(images_dir, fname)
            # Write atomically-ish: write if not exists
            if not os.path.exists(out_path):
                with open(out_path, "wb") as wf:
                    wf.write(jpeg_bytes)
                num_saved += 1
            existing_hash_to_name[h] = fname

        img_path = os.path.join(images_dir, fname)

        # build json entries
        all_samples.append(
            {
                "question": str(q),
                "answer": str(a),
                "image_path": img_path,
            }
        )

        a_norm = str(a).strip().lower()
        if a_norm in ("yes", "no"):
            yesno_samples.append(
                {
                    "question": str(q),
                    "label": a_norm,
                    "image_path": img_path,
                }
            )

    return all_samples, yesno_samples, num_saved, num_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="PathVQA data dir containing parquet files and images/")
    ap.add_argument("--images_dir", type=str, default=None, help="Where to store extracted images (default: <data_dir>/images)")
    ap.add_argument("--out_dir", type=str, default=None, help="Where to store json outputs (default: <data_dir>)")
    ap.add_argument("--image_prefix", type=str, default="pathvqa", help="Prefix for saved image filenames")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    images_dir = os.path.abspath(args.images_dir) if args.images_dir else os.path.join(data_dir, "images")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir

    _ensure_dir(images_dir)
    _ensure_dir(out_dir)

    split_files = _find_parquets(data_dir)

    # load_dataset can take list of files for each split
    ds = load_dataset(
        "parquet",
        data_files={
            "train": split_files["train"],
            "validation": split_files["validation"],
            "test": split_files["test"],
        },
    )

    hash_to_name: Dict[str, str] = {}

    # train
    train_all, train_yesno, saved_train, n_train = process_split(
        ds["train"], images_dir, args.image_prefix, args.jpeg_quality, hash_to_name
    )
    _dump_json(os.path.join(out_dir, "pathvqa_train.json"), train_all)
    _dump_json(os.path.join(out_dir, "pathvqa_train_yesno.json"), train_yesno)

    # valid
    valid_all, valid_yesno, saved_valid, n_valid = process_split(
        ds["validation"], images_dir, args.image_prefix, args.jpeg_quality, hash_to_name
    )
    _dump_json(os.path.join(out_dir, "pathvqa_valid.json"), valid_all)
    _dump_json(os.path.join(out_dir, "pathvqa_valid_yesno.json"), valid_yesno)

    # test
    test_all, test_yesno, saved_test, n_test = process_split(
        ds["test"], images_dir, args.image_prefix, args.jpeg_quality, hash_to_name
    )
    _dump_json(os.path.join(out_dir, "pathvqa_test.json"), test_all)
    _dump_json(os.path.join(out_dir, "pathvqa_test_yesno.json"), test_yesno)

    print("Done.")
    print(f"Split sizes: train={n_train}, valid={n_valid}, test={n_test}")
    print(f"New images saved (dedup by md5): {saved_train + saved_valid + saved_test}")
    print(f"Images directory: {images_dir}")
    print(f"JSON output directory: {out_dir}")


if __name__ == "__main__":
    main()

