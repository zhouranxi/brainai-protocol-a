#!/usr/bin/env python3
"""Convert official cifar-10-python tarball to ImageFolder layout.

Target layout (matches school server): DATA/cifar_full/{train,test}/{0..9}/*.jpg
Source layout inside tar: cifar-10-batches-py/{data_batch_1..5, test_batch}
  each a pickled dict with 'data' (bytes, N*3*32*32) and 'labels' (bytes, N).
Only the test split is used by P19 extract_features, but both are built.
"""
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(os.environ.get("BRAINAI_DATA_ROOT", "/home/lance/p19/data"))
TAR = DATA / "cifar-10-python.tar.gz"
OUT = DATA / "cifar_full"


def _write_split(tf, prefix, split, n_batches, out_dir):
    """Read n_batches pickle files, split images into label subdirs as jpg."""
    for label in range(10):
        (out_dir / str(label)).mkdir(parents=True, exist_ok=True)
    n = 0
    for b in range(1, n_batches + 1):
        name = f"{prefix}/data_batch_{b}" if split == "train" else f"{prefix}/test_batch"
        blob = tf.extractfile(name)
        d = pickle.load(blob, encoding="bytes")
        imgs = np.asarray(d[b"data"], dtype="uint8").reshape(-1, 3, 32, 32)
        labels = np.asarray(d[b"labels"], dtype="uint8").reshape(-1)
        for i in range(imgs.shape[0]):
            y = int(labels[i])
            rgb = imgs[i].transpose(1, 2, 0)  # (32,32,3)
            Image.fromarray(rgb).save(
                out_dir / str(y) / f"c10_{split}_{b}_{i}.jpg", quality=95)
            n += 1
    print(f"{split}: {n} images -> {out_dir}", flush=True)


def main():
    if not TAR.exists():
        print("MISSING", TAR)
        sys.exit(1)
    with tarfile.open(TAR) as tf:
        p = "cifar-10-batches-py"
        _write_split(tf, p, "train", 5, OUT / "train")
        _write_split(tf, p, "test", 1, OUT / "test")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
