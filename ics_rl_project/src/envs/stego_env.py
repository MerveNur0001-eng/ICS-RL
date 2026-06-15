"""
StegoEnv v8 — STABLE ROLLOUT + EMBED CONTROL

Problems from v7:
  ✗ steps_per_image=8 → too few steps on the same image, GAE still weak
    → changed to steps_per_image=12 (48-step rollout = 4 images × 12 steps)
  ✗ embed_ratio definition inconsistent (sometimes bits/capacity, sometimes bits/n_msg)
    → Standard: embed_ratio = bits_embedded / n_msg_bits (0..1)
  ✗ Chaotic coordinate order was recalculated on every reset
    → Calculate once, store (already in v7, preserved in v8)
  ✗ _build_obs texture normalization was missing for some images
    → Normalized during preload (already in v7)
  ✗ terminated=True but next obs was still the same image
    → Now cover is returned at each step instead of stego (fresh start)
    → Each step is independent, GAE bootstrap=0 is correct
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.modules.chaotic_generator import LogisticMapGenerator
from src.modules.texture_extractor import TextureFeatureExtractor
from src.envs.steganography_env    import SteganographyEnv
from utils.message_codec           import MessageCodec
from utils.bossbase_loader         import BOSSBaseLoader


DEFAULT_MESSAGE = (
    "ICS-RL Steganography System - Intelligent Chaotic Steganography "
    "using Reinforcement Learning. Logistic map (x0=secret, r=3.99) "
    "selects candidate pixels; PPO agent embeds real payload via LSB. "
    "This hidden message demonstrates real steganographic embedding "
    "as in Tang et al. 2020 and Ogras 2019. Beykoz University CME6405."
)


class StegoEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self,
                 loader,
                 chaotic_gen,
                 texture_ext,
                 reward_env,
                 img_shape=(256, 256),
                 n_candidates=10_000,
                 payload_bits=6553,
                 message=DEFAULT_MESSAGE,
                 steps_per_image=12):

        self.loader          = loader
        self.chaotic_gen     = chaotic_gen
        self.texture_ext     = texture_ext
        self.reward_env      = reward_env
        self.img_shape       = img_shape
        self.H, self.W       = img_shape
        self.n_candidates    = n_candidates
        self.payload_bits    = payload_bits
        self.steps_per_image = steps_per_image

        self.codec     = MessageCodec(max_bits=payload_bits)
        self.message   = message
        self._msg_bits = self.codec.encode(message)

        n_msg = len(self._msg_bits)
        pct   = n_msg / payload_bits * 100
        print(f"[StegoEnv v8] Message        : {len(message)} characters")
        print(f"[StegoEnv v8] Bit sequence   : {n_msg} bits (capacity: {payload_bits})")
        print(f"[StegoEnv v8] Fill ratio     : {pct:.1f}%")
        print(f"[StegoEnv v8] Image mode     : switch every {steps_per_image} steps")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, self.H, self.W),
            dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.H * self.W)

        self._mask = self.chaotic_gen.get_mask(self.img_shape, self.n_candidates)
        self._chaotic_rows, self._chaotic_cols = self._build_ordered_coords()

        self._cover      = None
        self._texture    = None
        self._img_index  = 0
        self._step_count = 0

        self._preload_images()

    def _preload_images(self):
        n = self.loader.size
        print(f"[StegoEnv v8] Preloading {n} images...")
        self._images, self._textures = [], []
        for i in range(n):
            img, _ = self.loader.get(i)
            tex    = self.texture_ext.get_texture_saliency_map(img)
            t_min, t_max = tex.min(), tex.max()
            if t_max - t_min > 1e-8:
                tex = (tex - t_min) / (t_max - t_min)
            self._images.append(img)
            self._textures.append(tex)
        print(f"[StegoEnv v8] Preload complete.")

    def _build_ordered_coords(self):
        h, w  = self.img_shape
        total = h * w
        x0, r = self.chaotic_gen.x0, self.chaotic_gen.r
        seq   = np.empty(total, dtype=np.float64)
        x = x0
        for i in range(total):
            x = r * x * (1.0 - x)
            seq[i] = x
        order    = np.argsort(seq)
        selected = order[:self.n_candidates]
        rows = (selected // w).astype(np.int32)
        cols = (selected  % w).astype(np.int32)
        return rows, cols

    def _load_next_image(self):
        idx = self._img_index % len(self._images)
        self._img_index += 1
        self._cover      = self._images[idx]
        self._texture    = self._textures[idx]
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._cover is None or self._step_count >= self.steps_per_image:
            self._load_next_image()
        return self._build_obs(self._cover), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        action_map     = action.reshape(self.H, self.W).astype(np.float32)
        action_map     = action_map * self._mask

        selected_flat  = action_map.flatten() == 1.0
        sel            = selected_flat[self._chaotic_rows * self.W + self._chaotic_cols]
        rows_arr       = self._chaotic_rows[sel]
        cols_arr       = self._chaotic_cols[sel]

        n_selected = len(rows_arr)
        n_msg_bits = len(self._msg_bits)

        stego = self.codec.embed_into_pixels(
            self._cover, self._msg_bits, rows_arr, cols_arr
        )

        bits_embedded = min(n_selected, n_msg_bits)
        embed_ratio   = bits_embedded / n_msg_bits

        cover_t      = torch.from_numpy(self._cover).unsqueeze(0)
        stego_t      = torch.from_numpy(stego).unsqueeze(0)
        texture_t    = torch.from_numpy(self._texture).unsqueeze(0)
        action_map_t = torch.from_numpy(action_map).unsqueeze(0)

        reward_t, breakdown = self.reward_env.compute_reward(
            cover_t, stego_t, texture_t,
            embed_ratio=embed_ratio,
            action_map=action_map_t
        )
        total_reward = float(reward_t.mean().item())

        diff       = self._cover.astype(np.float64) - stego.astype(np.float64)
        mse        = float(np.mean(diff ** 2))
        psnr       = (20.0 * np.log10(255.0 / (np.sqrt(mse) + 1e-10))
                      if mse > 1e-10 else 100.0)
        n_changed  = int((diff != 0).sum())
        pct_change = n_changed / self._cover.size * 100

        info = {
            "n_selected"   : n_selected,
            "n_msg_bits"   : n_msg_bits,
            "bits_embedded": bits_embedded,
            "embed_ratio"  : embed_ratio,
            "n_candidates" : int(self._mask.sum()),
            "mse"          : mse,
            "psnr"         : psnr,
            "n_changed"    : n_changed,
            "pct_change"   : pct_change,
            **breakdown,
        }
        terminated = (self._step_count >= self.steps_per_image)

        return self._build_obs(stego), total_reward, terminated, False, info

    def _build_obs(self, img: np.ndarray) -> np.ndarray:
        return np.stack([
            img.astype(np.float32) / 255.0,
            self._mask,
            self._texture,
        ], axis=0)

    def render(self): pass
    def close(self):  pass

    def get_message(self):
        return self.message

    def extract_from_stego(self, stego):
        bits = self.codec.extract_from_pixels(
            stego, self._chaotic_rows, self._chaotic_cols,
            n_bits=len(self._msg_bits)
        )
        return self.codec.decode(bits)