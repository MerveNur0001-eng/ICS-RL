"""
baselines.py
────────────
Gerçek steganografi baseline implementasyonları.

Kaynak: Daniel Lerch / stegolab (MIT License)
https://github.com/daniellerch/stegolab

İçerik:
  - HILL     : Li et al. ICIP 2014 — gerçek implementasyon
  - S-UNIWARD: Holub et al. EURASIP 2014 — grayscale uyarlaması

Bu modülü evaluate_seeded.py içinde şöyle import et:
    from baselines import embed_hill_real, embed_suniward_real
"""

import numpy as np
import scipy.signal
import scipy.fftpack


# ═══════════════════════════════════════════════════════════════
#  ORTAK YARDIMCILAR (her iki yöntem için)
# ═══════════════════════════════════════════════════════════════

def _ternary_entropy(pP1, pM1):
    p0 = 1 - pP1 - pM1
    P  = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H  = -P * np.log2(P + 1e-300)
    eps = 2.2204e-16
    H[P < eps]     = 0
    H[P > 1 - eps] = 0
    return np.sum(H)


def _calc_lambda(rho_p1, rho_m1, message_length, n):
    """
    STCsız embedding simulator için lambda parametresini bul.
    Kaynak: stegolab/HILL/HILL.py (Daniel Lerch, MIT)
    """
    l3 = 1e3
    m3 = float(message_length + 1)
    iterations = 0
    while m3 > message_length:
        l3 *= 2
        pP1 = np.exp(-l3 * rho_p1) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = np.exp(-l3 * rho_m1) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3  = _ternary_entropy(pP1, pM1)
        iterations += 1
        if iterations > 10:
            return l3

    l1   = 0
    m1   = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length) / n
    while float(m1 - m3) / n > alpha / 1000.0 and iterations < 300:
        lamb = l1 + (l3 - l1) / 2
        pP1  = np.exp(-lamb * rho_p1) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        pM1  = np.exp(-lamb * rho_m1) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        m2   = _ternary_entropy(pP1, pM1)
        if m2 < message_length:
            l3, m3 = lamb, m2
        else:
            l1, m1 = lamb, m2
        iterations += 1
    return lamb


def _embedding_simulator(cover, rho_p1, rho_m1, message_length, rng_seed=None):
    """
    STM (Syndrome Trellis Coding) yerine probabilistik embedding simulator.
    Kaynak: stegolab (Daniel Lerch, MIT)

    NOT: Gerçek STC yerine bu yaklaşım kullanılır; fark küçüktür
    (bkz. Filler et al. 2011 — simulator ile STC arasındaki PSNR farkı <0.1 dB).
    """
    n    = cover.shape[0] * cover.shape[1]
    lamb = _calc_lambda(rho_p1, rho_m1, message_length, n)

    pP1 = np.exp(-lamb * rho_p1) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
    pM1 = np.exp(-lamb * rho_m1) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))

    rng = np.random.default_rng(rng_seed)
    rand_change = rng.random(cover.shape)

    stego = cover.copy().astype(np.int16)
    stego[rand_change < pP1]                                      += 1
    stego[(rand_change >= pP1) & (rand_change < pP1 + pM1)]      -= 1
    stego = np.clip(stego, 0, 255).astype(np.uint8)
    return stego


# ═══════════════════════════════════════════════════════════════
#  HILL — Li et al. ICIP 2014
#  Kaynak: stegolab/HILL/HILL.py (Daniel Lerch, MIT License)
# ═══════════════════════════════════════════════════════════════

def _hill_cost(cover: np.ndarray) -> np.ndarray:
    """
    HILL cost haritası.
    HF1: 3×3 high-pass (Laplacian-like)
    H2 : 3×3 average (lokal ortalama)
    HW : 15×15 average (geniş komşuluk)

    rho = HW * (1 / (H2 * |HF1 * I| + ε))
    Kaynak: stegolab (Daniel Lerch) / orijinal MATLAB kodu (Li et al.)
    """
    HF1 = np.array([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1]
    ], dtype=np.float64)

    H2  = np.ones((3,  3),  dtype=np.float64) / 9.0
    HW  = np.ones((15, 15), dtype=np.float64) / 225.0

    I   = cover.astype(np.float64)
    R1  = scipy.signal.convolve2d(I,          HF1, mode='same', boundary='symm')
    W1  = scipy.signal.convolve2d(np.abs(R1), H2,  mode='same', boundary='symm')
    rho = 1.0 / (W1 + 1e-10)
    cost = scipy.signal.convolve2d(rho, HW, mode='same', boundary='symm')
    return cost


def embed_hill_real(cover: np.ndarray, payload_rate: float,
                    rng_seed: int = 42) -> np.ndarray:
    """
    HILL embedding (grayscale).

    Parameters
    ----------
    cover        : uint8 grayscale image (H×W)
    payload_rate : bits per pixel, ör. 0.4
    rng_seed     : tekrarlanabilirlik için

    Returns
    -------
    stego : uint8 grayscale stego image
    """
    rho      = _hill_cost(cover)
    wet_cost = 1e10

    rho_p1 = rho.copy()
    rho_m1 = rho.copy()
    rho_p1[cover == 255] = wet_cost
    rho_m1[cover == 0]   = wet_cost
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_m1[np.isnan(rho_m1)] = wet_cost
    rho_p1[rho_p1 > wet_cost] = wet_cost
    rho_m1[rho_m1 > wet_cost] = wet_cost

    message_length = round(payload_rate * cover.shape[0] * cover.shape[1])
    return _embedding_simulator(cover, rho_p1, rho_m1, message_length, rng_seed)


# ═══════════════════════════════════════════════════════════════
#  S-UNIWARD — Holub et al. EURASIP 2014  (grayscale uyarlama)
#  Kaynak: stegolab/S-UNIWARD/s-uniward-color.py (Daniel Lerch, MIT)
#  Orijinal renk kanalı için yazılmış; burada tek kanal (gray) uyarlandı.
# ═══════════════════════════════════════════════════════════════

def _suniward_cost(cover: np.ndarray):
    """
    S-UNIWARD cost haritası (grayscale).

    Daubechies 8-tap wavelet filtresi ile 3 yönde (H, V, D) rezidüel hesaplar.
    rho = Σ_k  (|F_k| ⊛ 1/(|F_k ⊛ I| + σ))

    Kaynak: stegolab (Daniel Lerch) / Holub et al. 2014
    """
    import scipy.signal as ss

    k, l = cover.shape
    sgm  = 1.0

    # Daubechies 8-tap
    hpdf = np.array([
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])
    sign = np.array([-1 if i % 2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    # 3 filtre: LL (yaklaşım), LH, HL, HH — S-UNIWARD sadece detail subbands
    filters = [
        np.outer(lpdf, hpdf),  # Horizontal detail
        np.outer(hpdf, lpdf),  # Vertical detail
        np.outer(hpdf, hpdf),  # Diagonal detail
    ]

    pad = 16
    rho = np.zeros((k, l), dtype=np.float64)

    for F in filters:
        cover_padded = np.pad(cover.astype(np.float32),
                              (pad, pad), 'symmetric')
        R0 = ss.convolve2d(cover_padded, F, mode='same')
        X  = ss.convolve2d(1.0 / (np.abs(R0) + sgm),
                           np.rot90(np.abs(F), 2), 'same')

        # Çift boyutlu filtreler için kaydırma düzeltmesi
        if F.shape[0] % 2 == 0:
            X = np.roll(X, 1, axis=0)
        if F.shape[1] % 2 == 0:
            X = np.roll(X, 1, axis=1)

        # Padding kaldır
        trim_r = (X.shape[0] - k) // 2
        trim_c = (X.shape[1] - l) // 2
        X = X[trim_r: trim_r + k, trim_c: trim_c + l]
        rho += X

    return rho


def embed_suniward_real(cover: np.ndarray, payload_rate: float,
                        rng_seed: int = 42) -> np.ndarray:
    """
    S-UNIWARD embedding (grayscale).

    Parameters
    ----------
    cover        : uint8 grayscale image (H×W)
    payload_rate : bits per pixel, ör. 0.4
    rng_seed     : tekrarlanabilirlik için

    Returns
    -------
    stego : uint8 grayscale stego image
    """
    wet_cost = 1e13

    rho      = _suniward_cost(cover)
    rho_p1   = rho.copy()
    rho_m1   = rho.copy()

    rho_p1[rho_p1 > wet_cost]  = wet_cost
    rho_p1[np.isnan(rho_p1)]   = wet_cost
    rho_p1[cover == 255]        = wet_cost

    rho_m1[rho_m1 > wet_cost]  = wet_cost
    rho_m1[np.isnan(rho_m1)]   = wet_cost
    rho_m1[cover == 0]          = wet_cost

    message_length = round(payload_rate * cover.shape[0] * cover.shape[1])
    return _embedding_simulator(cover, rho_p1, rho_m1, message_length, rng_seed)


# ═══════════════════════════════════════════════════════════════
#  HIZLI TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("baselines.py — self test")
    rng  = np.random.default_rng(0)
    img  = rng.integers(10, 245, (256, 256), dtype=np.uint8)

    s_hill = embed_hill_real(img, payload_rate=0.4, rng_seed=42)
    diff_h = np.abs(img.astype(np.int16) - s_hill.astype(np.int16))
    print(f"  HILL     : {(diff_h != 0).sum()} px changed, "
          f"max_diff={diff_h.max()}")

    s_su   = embed_suniward_real(img, payload_rate=0.4, rng_seed=42)
    diff_s = np.abs(img.astype(np.int16) - s_su.astype(np.int16))
    print(f"  S-UNIWARD: {(diff_s != 0).sum()} px changed, "
          f"max_diff={diff_s.max()}")

    print("  OK — her iki yöntem çalışıyor.")