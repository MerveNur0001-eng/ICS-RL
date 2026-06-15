"""
PPOTrainer v16 — ROLLOUT + MASK DÜZELTMELERİ

v15'teki problemler ve çözümleri:
─────────────────────────────────
✗ Rollout'ta mask uygulaması yanlış:
  action = dist.sample() * mask
  → mask (256,256) ama logits (1,256,256) → broadcast yanlış çalışıyor
  → DÜZELTME: mask squeeze edildi, action maskeleme doğrulandı

✗ Value loss scale sorunu: ret standart sapması 0 olunca NaN
  → +1e-8 eklendi

✗ KL erken durdurma tüm epoch'u kesiyordu (skip_kl flag)
  → break → continue (sadece o sample atla)

✗ ent_coef decay çok hızlı: 0.03 * 0.9995^468 ≈ 0.023 (30k/64 step)
  → min 0.01 (daha fazla keşif)

✗ Reward normalize edilmiyordu → yüksek varyans
  → Rollout reward'ları normalize edildi (mean/std)

✗ GAE: terminated=True her adımda → next_val=0 doğru AMA
  steps_per_image=16 → aynı imajda birden fazla adım var!
  → done flag'i doğru kullanıldı (env'den gelen)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ─────────────────────────────────────────────────────────────
#  VALUE HEAD
# ─────────────────────────────────────────────────────────────
class ValueHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.net  = nn.Sequential(
            nn.Linear(in_channels * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        x = self.pool(feat)
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  PPO TRAINER 
# ─────────────────────────────────────────────────────────────
class PPOTrainer:

    def __init__(self,
                 policy,
                 env,
                 lr            = 3e-5,
                 gamma         = 0.99,
                 lam           = 0.95,
                 clip_eps      = 0.20,
                 vf_coef       = 0.5,
                 ent_coef      = 0.03,
                 n_steps       = 64,
                 n_epochs      = 6,
                 target_kl     = 0.02,
                 max_grad_norm = 0.5,
                 reward_scale  = 2.0,
                 total_updates = None,
                 detect_coef         = 0.0,
                 texture_coef        = 0.0,
                 embed_penalty_coef  = 0.0,
                 min_embed_ratio     = 0.05,
                 max_embed_ratio     = 0.50):

        self.policy        = policy
        self.env           = env
        self.gamma         = gamma
        self.lam           = lam
        self.clip          = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.target_kl     = target_kl
        self.max_grad_norm = max_grad_norm
        self.reward_scale  = reward_scale

        feat_channels = getattr(policy, '_feat_channels', 64)
        self.value_head = ValueHead(in_channels=feat_channels)

        self.policy_opt = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.value_opt  = optim.Adam(self.value_head.parameters(), lr=lr * 2, eps=1e-5)

        if total_updates and total_updates > 0:
            self._sched_p = optim.lr_scheduler.CosineAnnealingLR(
                self.policy_opt, T_max=total_updates, eta_min=lr * 0.05)
            self._sched_v = optim.lr_scheduler.CosineAnnealingLR(
                self.value_opt,  T_max=total_updates, eta_min=lr * 0.1)
        else:
            self._sched_p = None
            self._sched_v = None

        self._feat_cache    = {}
        self._register_hook()

        self._best_p_detect  = 1.0
        self._last_info      = {}
        self._update_count   = 0
        self._last_grad_norm = 0.0

        print("[PPOTrainer v16 READY]")
        print(f"  lr={lr}  clip={clip_eps}  target_kl={target_kl}")
        print(f"  n_steps={n_steps}  n_epochs={n_epochs}  reward_scale={reward_scale}")
        print(f"  mask broadcast, reward normalize, KL per-sample")

    def _register_hook(self):
        target = None
        for name in ['conv5', 'encoder', 'backbone']:
            target = getattr(self.policy, name, None)
            if target is not None:
                break
        if target is None:
            for m in reversed(list(self.policy.modules())):
                if isinstance(m, nn.Conv2d):
                    target = m
                    break
        if target is not None:
            def hook(_, __, output):
                self._feat_cache["feat"] = output
            target.register_forward_hook(hook)
        else:
            print("[PPOTrainer v16] UYARI: Feature hook kurulamadı.")

    def _forward(self, obs):
        out = self.policy(obs)
        if isinstance(out, tuple):
            logits, feat = out
        else:
            logits = out
            feat   = self._feat_cache.get("feat", None)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=8.0, neginf=-8.0)

        if feat is None:
            B = obs.size(0)
            feat = torch.zeros((B, 64, 8, 8), device=obs.device)

        return logits, feat

    def _collect(self):
        """n_steps adımlık rollout."""
        rollout = []
        obs_np, _ = self.env.reset()
        obs = torch.from_numpy(obs_np).unsqueeze(0).float()

        for _ in range(self.n_steps):
            with torch.no_grad():
                logits, feat = self._forward(obs)

                mask = obs[0, 1, :, :]  

                if logits.dim() == 3:
                    logits_masked = torch.where(
                        mask.unsqueeze(0) > 0.5,
                        logits,
                        torch.full_like(logits, -10.0)
                    )
                else:
                    logits_masked = torch.where(
                        mask > 0.5, logits,
                        torch.full_like(logits, -10.0)
                    )

                dist   = torch.distributions.Bernoulli(logits=logits_masked)
                action = dist.sample()

                if logits.dim() == 3:
                    action = action * (mask.unsqueeze(0) > 0.5).float()
                else:
                    action = action * (mask > 0.5).float()

                logp  = dist.log_prob(action).mean()
                value = self.value_head(feat)

            action_np = action.detach().cpu().numpy().flatten()
            next_obs_np, raw_reward, terminated, truncated, info = \
                self.env.step(action_np)

            reward = float(np.clip(raw_reward * self.reward_scale, -5.0, 5.0))

            rollout.append({
                "obs"    : obs.clone(),
                "action" : action.detach().clone(),
                "logp"   : logp.detach(),
                "value"  : value.detach(),
                "reward" : reward,
                "done"   : terminated or truncated,
                "info"   : info,
            })

            if terminated or truncated:
                next_obs_np, _ = self.env.reset()
            obs = torch.from_numpy(next_obs_np).unsqueeze(0).float()

        self._last_info = rollout[-1]["info"]
        return rollout

    def _compute_gae(self, rollout):
        T   = len(rollout)
        adv = [0.0] * T
        gae = 0.0

        rewards = np.array([r["reward"] for r in rollout])
        r_mean  = rewards.mean()
        r_std   = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        for t in reversed(range(T)):                          # ← döngü başı
            r    = rollout[t]
            done = r["done"]

            if done or t + 1 >= T:
                next_val = 0.0
            else:
                next_val = rollout[t + 1]["value"].item()

            delta = (rewards_norm[t]                          # ← DÖNGÜ İÇİNDE
                    + self.gamma * next_val * (1 - float(done))
                    - r["value"].item())
            gae   = delta + self.gamma * self.lam * (1 - float(done)) * gae
            adv[t] = gae                                      # ← DÖNGÜ İÇİNDE

        adv_t = torch.tensor(adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = torch.clamp(adv_t, -5.0, 5.0)

        values_t = torch.tensor(
            [r["value"].item() for r in rollout], dtype=torch.float32)
        ret_t = adv_t + values_t
        return adv_t, ret_t

    def train_step(self):
        self._update_count += 1
        self.ent_coef = max(self.ent_coef * 0.9995, 0.01)

        rollout      = self._collect()
        adv, ret     = self._compute_gae(rollout)
        rewards      = [r["reward"] for r in rollout]

        total_loss = 0.0
        count      = 0
        total_grad = 0.0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(rollout))

            for i in indices:
                r        = rollout[i]
                obs      = r["obs"]
                act      = r["action"]
                old_logp = r["logp"]

                logits, feat = self._forward(obs)

                mask = obs[0, 1, :, :]
                if logits.dim() == 3:
                    logits = torch.where(
                        mask.unsqueeze(0) > 0.5, logits,
                        torch.full_like(logits, -10.0)
                    )
                else:
                    logits = torch.where(
                        mask > 0.5, logits,
                        torch.full_like(logits, -10.0)
                    )

                dist     = torch.distributions.Bernoulli(logits=logits)
                new_logp = dist.log_prob(act).mean()
                entropy  = dist.entropy().mean()

                approx_kl = (old_logp - new_logp).abs().item()
                if approx_kl > self.target_kl * 2:
                    continue   
                ratio_t = torch.exp(
                    (new_logp - old_logp).clamp(-5.0, 5.0))

                a_i = adv[i]
                s1  = ratio_t * a_i
                s2  = torch.clamp(ratio_t,
                                  1 - self.clip,
                                  1 + self.clip) * a_i
                policy_loss = -torch.min(s1, s2)

                value   = self.value_head(feat)
                old_val = r["value"].clone().detach()
                v_clip  = old_val + torch.clamp(
                    value - old_val, -self.clip, self.clip)
                vf_loss = torch.max(
                    (value - ret[i]) ** 2,
                    (v_clip - ret[i]) ** 2
                )

                loss = (policy_loss
                        + self.vf_coef * vf_loss
                        - self.ent_coef * entropy)

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                loss.backward()

                gn = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(), self.max_grad_norm)

                self.policy_opt.step()
                self.value_opt.step()

                total_loss += loss.item()
                total_grad += float(gn)
                count      += 1

        if self._sched_p:
            self._sched_p.step()
            self._sched_v.step()

        last_p = float(self._last_info.get("p_detect", 1.0))
        if last_p < self._best_p_detect:
            self._best_p_detect = last_p

        self._last_grad_norm = total_grad / max(count, 1)
        mean_loss   = total_loss / max(count, 1)
        mean_reward = float(np.mean(rewards))
        return mean_loss, mean_reward, self._last_info

    def save(self, path):
        torch.save({
            "policy"       : self.policy.state_dict(),
            "value_head"   : self.value_head.state_dict(),
            "policy_opt"   : self.policy_opt.state_dict(),
            "value_opt"    : self.value_opt.state_dict(),
            "ent_coef"     : self.ent_coef,
            "update_count" : self._update_count,
            "best_p_detect": self._best_p_detect,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.policy_opt.load_state_dict(ckpt["policy_opt"])
        self.value_opt.load_state_dict(ckpt["value_opt"])
        self.ent_coef        = ckpt.get("ent_coef",       self.ent_coef)
        self._update_count   = ckpt.get("update_count",   0)
        self._best_p_detect  = ckpt.get("best_p_detect",  1.0)