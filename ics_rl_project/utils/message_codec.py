# src/utils/message_codec.py
"""
Mesaj kodlama / çözme modülü.

Tang et al. (2020) uyarlaması:
  - Gerçek UTF-8 mesaj bit dizisine çevrilir
  - 32-bit uzunluk header gömülür (çıkarımda kaç bit okunacağı bilinir)
  - LSB embedding ile piksel LSB'leri değiştirilir
  - Aynı chaotic koordinatlarla mesaj geri çıkarılır

Kullanım:
    codec = MessageCodec(max_bits=26214)
    bits  = codec.encode("gizli mesaj")
    text  = codec.decode(bits)
"""

import struct
import numpy as np


class MessageCodec:
    """
    UTF-8 metin ↔ bit dizisi dönüşümü.

    max_bits : embedding kapasitesi (örn. 0.4 bpp × 256² = 26214)
    """

    HEADER_BITS = 32   # 4-byte uzunluk header

    def __init__(self, max_bits: int = 26214):
        self.max_bits      = max_bits
        self.max_msg_bytes = (max_bits - self.HEADER_BITS) // 8

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Metni bit dizisine çevirir.

        Yapı: [32-bit uzunluk header] + [mesaj bitleri]
        Toplam ≤ max_bits.

        Returns
        -------
        bits : np.ndarray  dtype=uint8, değerler {0, 1}
        """
        raw = text.encode("utf-8")
        if len(raw) > self.max_msg_bytes:
            raw = raw[: self.max_msg_bytes]   # kapasiteyi aşıyorsa kes

        header = struct.pack(">I", len(raw))  # 4-byte big-endian uzunluk
        payload = header + raw
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        return bits.astype(np.uint8)

    def decode(self, bits: np.ndarray) -> str:
        """
        Bit dizisini metne çevirir.

        İlk 32 bit uzunluk header'ı okur, sonra mesajı reconstruct eder.
        """
        if len(bits) < self.HEADER_BITS:
            return ""

        length_bytes = np.packbits(bits[: self.HEADER_BITS])
        msg_len      = struct.unpack(">I", bytes(length_bytes))[0]

        end_bit = self.HEADER_BITS + msg_len * 8
        if end_bit > len(bits):
            return ""           # mesaj kesilmiş

        msg_bits  = bits[self.HEADER_BITS: end_bit]
        msg_bytes = np.packbits(msg_bits)
        try:
            return bytes(msg_bytes[:msg_len]).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(msg_bytes[:msg_len]).decode("utf-8", errors="replace")

    # ── Embed ────────────────────────────────────────────────────────────────

    def embed_into_pixels(self,
                          cover:    np.ndarray,
                          bits:     np.ndarray,
                          rows:     np.ndarray,
                          cols:     np.ndarray) -> np.ndarray:
        """
        Cover görüntüye bit dizisini LSB ile gömer.

        Parametreler
        ------------
        cover : (H, W) uint8
        bits  : (N,)   uint8, değerler {0, 1}
        rows  : (N,)   int32  — hedef satır koordinatları
        cols  : (N,)   int32  — hedef sütun koordinatları

        Returns
        -------
        stego : (H, W) uint8
        """
        n_embed = min(len(bits), len(rows))
        stego   = cover.copy().astype(np.int16)

        r = rows[:n_embed]
        c = cols[:n_embed]
        b = bits[:n_embed].astype(np.int16)

        # LSB embedding: son biti sıfırla (& 0xFE), sonra yeni bit koy (| b)
        stego[r, c] = (stego[r, c] & np.int16(0xFE)) | b

        return np.clip(stego, 0, 255).astype(np.uint8)

    def extract_from_pixels(self,
                             stego: np.ndarray,
                             rows:  np.ndarray,
                             cols:  np.ndarray,
                             n_bits: int) -> np.ndarray:
        """
        Stego görüntüden LSB'leri çıkarır.

        Returns
        -------
        bits : (n_bits,) uint8
        """
        n = min(n_bits, len(rows))
        bits = np.zeros(n_bits, dtype=np.uint8)
        bits[:n] = (stego[rows[:n], cols[:n]] & 1).astype(np.uint8)
        return bits