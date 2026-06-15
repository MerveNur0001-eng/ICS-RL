# src/modules/texture_extractor.py
import numpy as np
import cv2


class TextureFeatureExtractor:
    """
    Per-piksel texture saliency map hesaplar.

    Üç lokal özelliği birleştirir:
      1. Local variance   — patch içindeki yoğunluk yayılımı
      2. Sobel gradient   — kenar gücü / yüksek frekans içeriği
      3. Laplacian        — lokal karmaşıklık için hızlı ikinci türev proxy'si

    Tasarım raporuna göre (Section 3.3):
    - Her harita [0,1]'e normalize edilir
    - Üçü ortalaması alınarak tek saliency map elde edilir
    - Yüksek değer → texture-rich bölge → LSB gömme için daha güvenli

    Parameters
    ----------
    patch_size : int
        Varyans hesabında kullanılan kare komşuluk boyutu (default 7).
    """

    def __init__(self, patch_size: int = 7):
        self.patch_size = patch_size



    def compute_local_variance(self, img_f: np.ndarray) -> np.ndarray:
        mean    = cv2.blur(img_f,      (self.patch_size, self.patch_size))
        sq_mean = cv2.blur(img_f ** 2, (self.patch_size, self.patch_size))
        return np.maximum(sq_mean - mean ** 2, 0.0)

    def compute_sobel_gradient(self, img_f: np.ndarray) -> np.ndarray:
        
        grad_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    def compute_fast_texture_score(self, img_f: np.ndarray) -> np.ndarray:
        return np.abs(cv2.Laplacian(img_f, cv2.CV_32F))

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def get_texture_saliency_map(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32) / 255.0

        var  = self.compute_local_variance(img_f)
        grad = self.compute_sobel_gradient(img_f)
        tex  = self.compute_fast_texture_score(img_f)

        return (
            self._normalize(var) +
            self._normalize(grad) +
            self._normalize(tex)
        ) / 3.0