"""
utils/bossbase_loader.py
------------------------
Utility for loading grayscale images from the BOSSBase 1.01 dataset.

BOSSBase 1.01 consists of 10,000 grayscale PGM images (256 × 256 pixels).
Download: http://agents.fel.cvut.cz/stegodata/BossBase-1.01-cover.tar.gz

Expected directory layout
-------------------------
data/
└── bossbase/
    ├── 1.pgm
    ├── 2.pgma
    ├── ...
    └── 10000.pgm
"""

import os
import glob
import random
import numpy as np
import cv2

# Supported image extensions in BOSSBase (PGM) and common alternatives
_SUPPORTED_EXT = (".pgm", ".png", ".jpg", ".jpeg", ".bmp")


def _find_images(root: str) -> list[str]:
    """Return a sorted list of all supported image paths under `root`."""
    paths = []
    for ext in _SUPPORTED_EXT:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(root, f"*{ext.upper()}")))
    return sorted(set(paths))


def load_image(path: str, img_shape: tuple = (256, 256)) -> np.ndarray:
    """
    Load a single grayscale image and resize to `img_shape`.

    Parameters
    ----------
    path      : str    full path to the image file
    img_shape : tuple  (height, width) — default (256, 256)

    Returns
    -------
    img : np.ndarray  shape (H, W), dtype uint8
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.shape != img_shape:
        img = cv2.resize(img, (img_shape[1], img_shape[0]),
                         interpolation=cv2.INTER_AREA)
    return img


class BOSSBaseLoader:
    """
    Iterates over BOSSBase images in order or randomly.

    Parameters
    ----------
    data_dir  : str    path to the folder containing the .pgm files
    img_shape : tuple  target (H, W) — default (256, 256)
    shuffle   : bool   randomise order on each epoch (default False)
    """

    def __init__(self,
                 data_dir:  str,
                 img_shape: tuple = (256, 256),
                 shuffle:   bool  = False):
        self.data_dir  = data_dir
        self.img_shape = img_shape
        self.shuffle   = shuffle

        self._paths = _find_images(data_dir)
        if not self._paths:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'.\n"
                f"Please place BOSSBase 1.01 .pgm files there, or set\n"
                f"  data_dir='data/bossbase/'  in main.py."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of images in the dataset."""
        return len(self._paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, index: int) -> tuple[np.ndarray, str]:
        """
        Load the image at position `index`.

        Returns
        -------
        (img, filename)
        """
        path = self._paths[index]
        return load_image(path, self.img_shape), os.path.basename(path)

    def get_random(self) -> tuple[np.ndarray, str]:
        """Load a randomly chosen image."""
        path = random.choice(self._paths)
        return load_image(path, self.img_shape), os.path.basename(path)

    def __iter__(self):
        paths = self._paths.copy()
        if self.shuffle:
            random.shuffle(paths)
        for path in paths:
            yield load_image(path, self.img_shape), os.path.basename(path)

    def __len__(self) -> int:
        return self.size