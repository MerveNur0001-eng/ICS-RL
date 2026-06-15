"""
REWARD ENV v15 — TEXTURE + EMA + REWARD FIX

Problems from v14 and their solutions:
─────────────────────────────────
✗ r_texture constantly ~-0.93 (too negative)
  REASON: tanh(4*(tex_mean - 0.5)) formula gives the agent a
          constant negative signal when tex_mean < 0.5.
          Agent only changes 1 bit via LSB → texture map average
          stays low → always -0.93
  → FIX 1: When computing texture score of selected pixels,
    use chaotic coordinates DIRECTLY instead of action_map.
  → FIX 2: Remove baseline — r_texture = tanh(2*tex_mean)
    [0,1] → [0, tanh(2)] ≈ [0, 0.96], never negative.
  → FIX 3: w3=0.35 too high, reduce to 0.15; w1=0.70→0.75

✗ EMA alpha=0.9 too high → p_detect signal lags 10+ steps behind
  → alpha=0.7 (faster update)
  → warm-up 100→50 (faster activation)

✗ capacity_penalty_coef=0.7 too aggressive at the start
  → reduce to 0.3 (will also be 0.3 in train.py)

✗ sparse_bonus only for embed_ratio < 0.30 → meaningless
  → REMOVED (capacity_penalty already handles this)
"""

import torch
import torch.nn as nn
import numpy as np


class _SRNetProxy(nn.Module):
    """Lightweight SRM filter-based detector."""
    _SRM_KERNELS = torch.tensor([
        [[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]],
        [[ 0, -1,  0], [ 0,  2,  0], [ 0, -1,  0]],
        [[-1,  0,  1], [ 0,  0,  0], [ 1,  0, -1]],
    ], dtype=torch.float32).unsqueeze(1) / 4.0

    def __init__(self):
        super().__init__()
        self.srm = nn.Conv2d(1, 3, 3, padding=1, bias=False)
        with torch.no_grad():
            self.srm.weight.copy_(self._SRM_KERNELS)
        self.srm.weight.requires_grad = False
        self.clf = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(self.srm(x))


# ─────────────────────────────────────────────────────────────
#  STEGANOGRAPHY REWARD ENV
# ─────────────────────────────────────────────────────────────
class SteganographyEnv:
    def __init__(self,
                 srnet_model_path:        str   = None,
                 w1:                      float = 0.50,
                 w2:                      float = 0.35,
                 w3:                      float = 0.15,
                 target_embed_ratio:      float = 0.80,
                 capacity_penalty_coef:   float = 0.30,
                 sparse_threshold:        float = 0.30,
                 sparse_bonus_val:        float = 0.0,
                 high_embed_threshold:    float = 1.10,
                 high_embed_penalty:      float = 0.0,
                 use_psnr_detect:         bool  = False,
                 detect_ema_alpha:        float = 0.7):

        if abs(w1 + w2 + w3 - 1.0) > 1e-6:
            raise ValueError(f"w1+w2+w3 = {w1+w2+w3:.4f} ≠ 1.0")

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.target_embed_ratio    = target_embed_ratio
        self.capacity_penalty_coef = capacity_penalty_coef
        self.detect_ema_alpha      = detect_ema_alpha

        self._ema_p_detect = 0.5
        self._step_count   = 0
        self._warmup_steps = 50
        self.srnet, self._using_proxy = self._load_srnet(srnet_model_path)
        self.srnet.eval()
        for p in self.srnet.parameters():
            p.requires_grad = False

        mode = "proxy (SRM)" if self._using_proxy else f"SRNet ← {srnet_model_path}"
        print(f"[SteganographyEnv v15] Detector          : {mode}")
        print(f"[SteganographyEnv v15] Weights           : w1={w1} w2={w2} w3={w3}")
        print(f"[SteganographyEnv v15] Capacity target   : embed_ratio ≥ {target_embed_ratio}")
        print(f"[SteganographyEnv v15] Capacity penalty  : coef={capacity_penalty_coef}")
        print(f"[SteganographyEnv v15] EMA alpha         : {detect_ema_alpha} (reduced)")
        print(f"[SteganographyEnv v15] Warm-up steps     : {self._warmup_steps}")

    def _load_srnet(self, model_path):
        if model_path is None:
            print("[SteganographyEnv v15] srnet_model_path=None → proxy")
            return _SRNetProxy(), True
        try:
            from src.models.srnet import SRNet
            model = SRNet()
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"[SteganographyEnv v15] ✓ Real SRNet loaded: {model_path}")
            return model, False
        except Exception as e:
            print(f"[SteganographyEnv v15] SRNet could not be loaded ({e}) → proxy.")
            return _SRNetProxy(), True

    def _detection_reward(self, cover, stego):
        with torch.no_grad():
            inp = stego.unsqueeze(1).float() / 255.0
            logits = self.srnet(inp)
            probs = torch.softmax(logits, dim=1)
            p_detect = probs[:, 1]

        r_detect = 1.0 - 2.0 * torch.abs(p_detect - 0.5)

        if self._step_count < self._warmup_steps:
            warmup_scale = self._step_count / self._warmup_steps
            r_detect = r_detect * warmup_scale

        return r_detect, p_detect

    @staticmethod
    def _distortion_reward(cover: torch.Tensor, stego: torch.Tensor):
        """SSIM → r_distort = ssim - 1 ∈ [-1, 0]"""
        c  = cover.float()
        s  = stego.float()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_c   = c.mean(dim=(1, 2), keepdim=True)
        mu_s   = s.mean(dim=(1, 2), keepdim=True)
        sig_cc = ((c - mu_c) ** 2).mean(dim=(1, 2))
        sig_ss = ((s - mu_s) ** 2).mean(dim=(1, 2))
        sig_cs = ((c - mu_c) * (s - mu_s)).mean(dim=(1, 2))

        ssim = ((2 * mu_c.squeeze() * mu_s.squeeze() + c1) *
                (2 * sig_cs + c2)) / \
               ((mu_c.squeeze() ** 2 + mu_s.squeeze() ** 2 + c1) *
                (sig_cc + sig_ss + c2) + 1e-8)

        ssim      = torch.clamp(ssim, 0.0, 1.0)
        r_distort = ssim - 1.0
        return r_distort, ssim

    @staticmethod
    def _texture_reward(texture_map: torch.Tensor,
                        action_map:  torch.Tensor,
                        cover:       torch.Tensor,
                        stego:       torch.Tensor):

        if action_map is not None:
            wmap = action_map.float()
        else:
            wmap = (cover.float() != stego.float()).float()

        n_sel    = wmap.sum(dim=(1, 2))
        tex_sum  = (wmap * texture_map).sum(dim=(1, 2))
        tex_mean = tex_sum / (n_sel + 1e-8)

        r_texture = torch.tanh(2.0 * tex_mean)

        return r_texture, tex_mean

    def _capacity_penalty(self, embed_ratio: float):
        shortfall = max(0.0, self.target_embed_ratio - embed_ratio)
        penalty = -self.capacity_penalty_coef * (shortfall ** 2) * 5.0
        return penalty

    def compute_reward(self,
                       cover:       torch.Tensor,
                       stego:       torch.Tensor,
                       texture_map: torch.Tensor,
                       embed_ratio: float        = 1.0,
                       action_map:  torch.Tensor = None):
        """
        cover, stego  : (B, H, W) uint8
        texture_map   : (B, H, W) float32 [0,1]
        action_map    : (B, H, W) float32 {0,1}
        embed_ratio   : bits_embedded / n_msg_bits ∈ [0, 1]
        """
        self._step_count += 1

        r_detect, p_detect  = self._detection_reward(cover, stego)
        r_distort, ssim     = self._distortion_reward(cover, stego)
        r_texture, tex_mean = self._texture_reward(texture_map, action_map, cover, stego)

        weighted = (self.w1 * r_detect
                  + self.w2 * r_distort
                  + self.w3 * r_texture)

        r_cap   = self._capacity_penalty(embed_ratio)
        r_cap_t = torch.full_like(r_detect, r_cap)

        total = weighted + r_cap_t

        n_changed = int(float((cover.float() != stego.float()).sum().item()))
        mse       = float(torch.mean((cover.float() - stego.float()) ** 2).item())
        psnr      = (20.0 * np.log10(255.0 / (np.sqrt(mse) + 1e-10))
                     if mse > 1e-10 else 100.0)

        breakdown = {
            "r_detect"        : float(r_detect.mean().item()),
            "r_distort"       : float(r_distort.mean().item()),
            "r_texture"       : float(r_texture.mean().item()),
            "p_detect"        : float(p_detect.mean().item()),
            "ssim"            : float(ssim.mean().item()),
            "tex_mean"        : float(tex_mean.mean().item()),
            "embed_ratio"     : float(embed_ratio),
            "capacity_penalty": float(r_cap),
            "sparse_bonus"    : 0.0,
            "mse"             : mse,
            "psnr"            : psnr,
            "n_changed"       : n_changed,
            "warmup_scale"    : min(1.0, self._step_count / self._warmup_steps),
        }

        return total, breakdown