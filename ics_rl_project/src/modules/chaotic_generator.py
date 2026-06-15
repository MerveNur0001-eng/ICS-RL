# src/modules/chaotic_generator.py
import numpy as np


class LogisticMapGenerator:
    """
    Logistic Map tabanlı Kaotik Aday Piksel Üreteci.

    Formül: x_{n+1} = r * x_n * (1 - x_n)

    Tasarım raporuna göre:
    - Kaotik dizi, piksel indislerini sıralamak için anahtar olarak kullanılır
    - Bu sayede: duplicate yok, uniform spatial coverage var,
      aynı x0 ile her zaman aynı set üretilir

    Parameters
    x0 : float   Gizli anahtar, (0, 1) aralığında olmalı
    r  : float   Büyüme oranı, 3.99 kaotik davranış garantiler
    """

    def __init__(self, x0: float, r: float = 3.99):
        if not (0.0 < x0 < 1.0):
            raise ValueError(f"x0 must be in (0, 1), got {x0}")
        self.x0 = x0
        self.r  = r

    def _generate_chaotic_sequence(self, length: int) -> np.ndarray:
        seq = np.empty(length, dtype=np.float64)
        x = self.x0
        for i in range(length):
            x = self.r * x * (1.0 - x)
            seq[i] = x
        return seq

    def generate_candidates(self,
                            img_shape:    tuple,
                            n_candidates: int = 15000) -> np.ndarray:
        h, w          = img_shape
        total_pixels  = h * w  

        if n_candidates > total_pixels:
            raise ValueError(
                f"n_candidates ({n_candidates}) > total pixels ({total_pixels})"
            )

        chaotic_seq   = self._generate_chaotic_sequence(total_pixels)
        chaotic_order = np.argsort(chaotic_seq)          
        selected      = chaotic_order[:n_candidates]    

        rows = (selected // w).astype(np.int32)
        cols = (selected  % w).astype(np.int32)
        return np.stack([rows, cols], axis=1)          

    def get_mask(self,
                 img_shape:    tuple,
                 n_candidates: int = 30000) -> np.ndarray:
        candidates        = self.generate_candidates(img_shape, n_candidates)
        mask              = np.zeros(img_shape, dtype=np.float32)
        mask[candidates[:, 0], candidates[:, 1]] = 1.0
        return mask