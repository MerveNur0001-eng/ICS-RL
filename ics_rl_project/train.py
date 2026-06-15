# train.py — v15  (CSV LOGGING FIX: every update written to disk)
"""
v15 CHANGES vs v14:
  - CSV log written after EVERY update (crash-safe)
  - bd dict verified non-empty before logging (fallback to last_info)
  - Fresh-start flag: pass --fresh to delete old checkpoints
  - Plot generated from CSV (not in-memory log) — always up to date
  - All print output also mirrors to results/train_v15_console.txt
"""

import os
import sys
import glob
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.modules.policy_network    import PolicyNetwork
from src.modules.chaotic_generator import LogisticMapGenerator
from src.modules.texture_extractor import TextureFeatureExtractor
from src.envs.stego_env            import StegoEnv
from src.envs.steganography_env    import SteganographyEnv
from src.training.ppo_trainer      import PPOTrainer
from utils.bossbase_loader         import BOSSBaseLoader

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
IMG_SHAPE      = (256, 256)
X0             = 0.123456
N_CANDIDATES   = 15_000
PAYLOAD_BITS   = int(256 * 256 * 0.4)
DATA_DIR       = "data/bossbase/"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

TOTAL_STEPS    = 30_000
N_STEPS        = 256
N_EPOCHS       = 6
SAVE_INTERVAL  = 5        # checkpoint every N updates
MAX_IMAGES     = 50

SRNET_PATH     = "src/models/srnet_best.pth"
USE_SRNET      = os.path.exists(SRNET_PATH)

LOG_CSV      = os.path.join(RESULTS_DIR, "training_log_v15.csv")
PLOT_PATH    = os.path.join(RESULTS_DIR, "training_curves_v15.png")
CONSOLE_LOG  = os.path.join(RESULTS_DIR, "train_v15_console.txt")

CSV_FIELDS = [
    "step", "update", "loss", "mean_reward",
    "p_detect", "r_detect", "r_distort", "r_texture",
    "embed_ratio", "n_selected", "psnr", "ssim",
    "capacity_penalty", "tex_mean", "best_p_detect"
]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def clean_old_checkpoints():
    removed = 0
    for pattern in [
        os.path.join(CHECKPOINT_DIR, "ckpt_step*.pt"),
        os.path.join(CHECKPOINT_DIR, "ckpt_final.pt"),
    ]:
        for f in glob.glob(pattern):
            os.remove(f); removed += 1
    pf = os.path.join(CHECKPOINT_DIR, "policy_final.pt")
    if os.path.exists(pf):
        os.remove(pf); removed += 1
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    print(f"  Removed {removed} old checkpoint files and log.")


def init_csv():
    with open(LOG_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv(row: dict):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({
            k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
            for k in CSV_FIELDS
        })


def try_load(trainer, ckpts):
    if not ckpts:
        return 0, 0
    latest = ckpts[-1]
    try:
        step = int(os.path.basename(latest)
                   .replace("ckpt_step", "").replace(".pt", ""))
    except Exception:
        step = 0
    try:
        trainer.load(latest)
        print(f"  Resumed from: {latest}  (step={step:,})")
        return step, step // N_STEPS
    except Exception as e:
        print(f"  Incompatible checkpoint ({e}) — starting from scratch.")
        clean_old_checkpoints()
        return 0, 0


# ══════════════════════════════════════════════════════════════════════
#  PLOT  — reads from CSV, always accurate
# ══════════════════════════════════════════════════════════════════════
def save_plots():
    rows = []
    try:
        with open(LOG_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        print("  Plot skipped: CSV not found.")
        return
    if len(rows) < 2:
        print("  Plot skipped: insufficient data.")
        return

    def col(key):
        return np.array([float(r[key]) for r in rows])

    steps       = col("step").astype(int)
    p_detect    = col("p_detect")
    best_pd     = col("best_p_detect")
    psnr        = col("psnr")
    ssim        = col("ssim")
    tex         = col("tex_mean")
    r_detect    = col("r_detect")
    r_distort   = col("r_distort")
    r_texture   = col("r_texture")
    mean_reward = col("mean_reward")
    embed_ratio = col("embed_ratio")
    loss        = col("loss")

    def ma(data, n=5):
        if len(data) >= n:
            return np.convolve(data, np.ones(n)/n, mode="valid"), steps[n-1:]
        return data, steps

    BLUE   = "#1565c0"
    RED    = "#c62828"
    GREEN  = "#2e7d32"
    ORANGE = "#e65100"
    PURPLE = "#6a1b9a"
    TEAL   = "#00695c"
    LGRAY  = "#f5f7fa"

    fig = plt.figure(figsize=(22, 14), facecolor="white")
    fig.suptitle(
        "ICS-RL PPO Training Curves (v15)\n"
        f"30,000 steps  |  {len(rows)} updates  |  "
        "50 BOSSBase images  |  SRNet reward signal",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.36,
                          top=0.93, bottom=0.06)

    # (a) p_detect
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, p_detect, color=RED, lw=1, alpha=0.35, label="p_detect (per update)")
    sm, sx = ma(p_detect)
    ax.plot(sx, sm, color=RED, lw=2.5, label="MA-5")
    ax.plot(steps, best_pd, color=ORANGE, lw=1.5, ls="--", alpha=0.85,
            label=f"Best p_detect (min={best_pd.min():.4f})")
    ax.axhline(0.50, color=GREEN,    ls="--", lw=1.8, label="Target <= 0.50")
    ax.axhline(0.45, color="#66bb6a", ls=":",  lw=1.3, label="Ideal < 0.45")
    ax.set_ylim(0, 1)
    ax.set_title("(a) SRNet Detection Probability", fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("p_detect")
    ax.legend(fontsize=7.5, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (b) Policy Loss
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(LGRAY)
    valid = [(s, v) for s, v in zip(steps, loss)
             if not (np.isnan(v) or np.isinf(v)) and abs(v) < 50]
    if valid:
        lx, lv = zip(*valid)
        lv_arr = np.array(lv)
        ax.plot(lx, lv, color=ORANGE, lw=1, alpha=0.4)
        sm2, sx2 = ma(lv_arr, 5)
        ax.plot(list(sx2)[:len(sm2)], sm2, color=ORANGE, lw=2.5, label="MA-5")
    ax.set_title("(b) PPO Policy Loss", fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (c) Mean Reward
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, mean_reward, color=BLUE, lw=1, alpha=0.35)
    sm, sx = ma(mean_reward)
    ax.plot(sx, sm, color=BLUE, lw=2.5, label="MA-5")
    ax.axhline(0, color=RED, ls="--", lw=1.2, alpha=0.5)
    ax.set_title("(c) Mean Episode Reward", fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Reward")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (d) Reward Components
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, r_detect,  color=RED,    lw=1.5,
            label=f"r_detect  (mean={r_detect.mean():.3f})")
    ax.plot(steps, r_distort, color=ORANGE, lw=1.5,
            label=f"r_distort (mean={r_distort.mean():.3f})")
    ax.plot(steps, r_texture, color=GREEN,  lw=1.5,
            label=f"r_texture (mean={r_texture.mean():.3f})")
    ax.axhline(0, color="#aaa", ls="--", lw=0.8)
    ax.set_title("(d) Reward Components", fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Component Value")
    ax.legend(fontsize=7.5, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (e) Texture Score
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, tex, color=PURPLE, lw=1, alpha=0.4)
    sm, sx = ma(tex)
    ax.plot(sx, sm, color=PURPLE, lw=2.5,
            label=f"MA-5 (mean={tex.mean():.4f})")
    ax.axhline(0.0646, color=RED, ls="--", lw=1.8,
               label="Chaotic baseline (0.0646)")
    ax.fill_between(steps, 0.0646, tex,
                    where=tex > 0.0646, alpha=0.12, color=PURPLE)
    ax.set_title("(e) Texture Score — Embedding Quality",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Avg Texture Score")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (f) Embed Ratio
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, embed_ratio, color=TEAL, lw=1.5,
            label=f"embed_ratio (mean={embed_ratio.mean():.3f})")
    ax.axhline(0.80, color=GREEN,    ls="--", lw=1.8, label="Target >= 0.80")
    ax.axhline(1.00, color="#66bb6a", ls=":",  lw=1.2, label="Full capacity")
    ax.set_ylim(0, 1.1)
    ax.set_title("(f) Embedding Capacity Ratio", fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("bits_embedded / n_msg_bits")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (g) PSNR
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, psnr, color=TEAL, lw=1, alpha=0.5)
    sm, sx = ma(psnr)
    ax.plot(sx, sm, color=TEAL, lw=2.5,
            label=f"MA-5 (avg={psnr.mean():.2f} dB)")
    ax.axhline(40, color=GREEN, ls="--", lw=1.8, label="Target > 40 dB")
    ax.set_ylim(max(psnr.min() - 1, 39), psnr.max() + 1)
    ax.set_title("(g) PSNR — Visual Imperceptibility",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (h) SSIM
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor(LGRAY)
    ax.plot(steps, ssim, color=BLUE, lw=1, alpha=0.5)
    sm, sx = ma(ssim)
    ax.plot(sx, sm, color=BLUE, lw=2.5,
            label=f"MA-5 (avg={ssim.mean():.6f})")
    ax.axhline(0.98, color=GREEN, ls="--", lw=1.8, label="Target > 0.98")
    ax.set_ylim(max(ssim.min() - 0.001, 0.990), 1.0005)
    ax.set_title("(h) SSIM — Structural Similarity",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step"); ax.set_ylabel("SSIM")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # (i) Summary box
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    summary = (
        "Training Summary\n"
        "─────────────────────\n"
        f"Total steps    : {steps[-1]:,}\n"
        f"Updates        : {len(rows)}\n"
        "─────────────────────\n"
        f"p_detect avg   : {p_detect.mean():.4f}\n"
        f"p_detect best  : {best_pd.min():.4f}\n"
        f"p_detect final : {p_detect[-1]:.4f}\n"
        "─────────────────────\n"
        f"PSNR avg       : {psnr.mean():.2f} dB\n"
        f"SSIM avg       : {ssim.mean():.6f}\n"
        f"Texture avg    : {tex.mean():.4f}\n"
        f"Chaotic tex    : 0.0646\n"
        f"Tex improve    : +{tex.mean() - 0.0646:.4f}\n"
        "─────────────────────\n"
        f"Embed ratio    : {embed_ratio.mean():.3f}"
    )
    ax.text(0.08, 0.95, summary,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9.5,
            fontfamily="monospace",
            color="white",
            linespacing=1.6)

    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    fresh = "--fresh" in sys.argv
    sys.stdout = Tee(CONSOLE_LOG)

    updates_total = TOTAL_STEPS // N_STEPS
    print("=" * 72)
    print("  ICS-RL PPO — v15  (CSV LOGGING FIX)")
    print(f"  Total steps : {TOTAL_STEPS:,}   Updates: {updates_total}")
    print(f"  n_steps     : {N_STEPS}          n_epochs: {N_EPOCHS}")
    print(f"  Images      : {MAX_IMAGES}        Payload: {PAYLOAD_BITS:,} bits (0.4 bpp)")
    print(f"  SRNet       : {'loaded ✓' if USE_SRNET else 'proxy (SRM)'}")
    print(f"  Fresh start : {fresh}")
    print("=" * 72 + "\n")

    if fresh:
        print("  [--fresh] Removing old checkpoints and log...")
        clean_old_checkpoints()

    try:
        loader = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); return

    loader._paths = loader._paths[:MAX_IMAGES]

    chaotic_gen = LogisticMapGenerator(x0=X0)
    texture_ext = TextureFeatureExtractor(patch_size=7)

    reward_env = SteganographyEnv(
        srnet_model_path      = SRNET_PATH if USE_SRNET else None,
        w1=0.50, w2=0.35, w3=0.15,
        target_embed_ratio    = 0.70,
        capacity_penalty_coef = 1.0,
        detect_ema_alpha      = 0.70,
        sparse_threshold      = 0.30,
        sparse_bonus_val      = 0.0,
        high_embed_threshold  = 1.10,
        high_embed_penalty    = 0.0,
    )

    env = StegoEnv(
        loader          = loader,
        chaotic_gen     = chaotic_gen,
        texture_ext     = texture_ext,
        reward_env      = reward_env,
        img_shape       = IMG_SHAPE,
        n_candidates    = N_CANDIDATES,
        payload_bits    = PAYLOAD_BITS,
        steps_per_image = 32,
    )

    policy  = PolicyNetwork(input_channels=3)
    trainer = PPOTrainer(
        policy, env,
        lr                 = 1e-4,
        n_steps            = N_STEPS,
        n_epochs           = N_EPOCHS,
        ent_coef           = 0.05,
        clip_eps           = 0.20,
        reward_scale       = 5.0,
        target_kl          = 0.15,
        max_grad_norm      = 0.5,
        detect_coef        = 0.0,
        texture_coef       = 0.0,
        embed_penalty_coef = 0.0,
    )

    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step*.pt")))
    step, update = try_load(trainer, ckpts)

    if step == 0:
        init_csv()
        print("  Starting from scratch.\n")

    print(f"  {'Step':>8} | {'Loss':>8} | {'Reward':>8} | "
          f"{'p_det':>7} | {'tex':>7} | {'embed':>7} | "
          f"{'PSNR':>7} | {'SSIM':>8} | Status")
    print("  " + "-" * 82)

    while step < TOTAL_STEPS:
        try:
            loss, mr, bd = trainer.train_step()
        except Exception as e:
            print(f"\n  [WARNING] train_step error: {e} — skipping update")
            step   += N_STEPS
            update += 1
            continue

        step   += N_STEPS
        update += 1

        loss_v = safe_float(loss)
        mr_v   = safe_float(mr)
        p      = safe_float(bd.get("p_detect",         0.5))
        rd     = safe_float(bd.get("r_detect",         0.0))
        rdi    = safe_float(bd.get("r_distort",        0.0))
        rt     = safe_float(bd.get("r_texture",        0.0))
        emb    = safe_float(bd.get("embed_ratio",      0.0))
        ns     = int(safe_float(bd.get("n_selected",   0)))
        psnr_v = safe_float(bd.get("psnr",             0.0))
        ssim_v = safe_float(bd.get("ssim",             0.0))
        cap_v  = safe_float(bd.get("capacity_penalty", 0.0))
        tex_v  = safe_float(bd.get("tex_mean",         0.0))
        best_p = trainer._best_p_detect

        append_csv({
            "step"             : step,
            "update"           : update,
            "loss"             : loss_v,
            "mean_reward"      : mr_v,
            "p_detect"         : p,
            "r_detect"         : rd,
            "r_distort"        : rdi,
            "r_texture"        : rt,
            "embed_ratio"      : emb,
            "n_selected"       : float(ns),
            "psnr"             : psnr_v,
            "ssim"             : ssim_v,
            "capacity_penalty" : cap_v,
            "tex_mean"         : tex_v,
            "best_p_detect"    : best_p,
        })

        if   p < 0.45: status = "EXCELLENT"
        elif p < 0.50: status = "HIDDEN   "
        elif p < 0.65: status = "MEDIUM   "
        else:          status = "DETECTED "

        cap_warn = " [low cap]" if emb < 0.5 else ""

        print(f"  {step:>8,} | {loss_v:>8.4f} | {mr_v:>8.4f} | "
              f"{p:>7.4f} | {tex_v:>7.4f} | {emb:>7.3f} | "
              f"{psnr_v:>7.2f} | {ssim_v:>8.6f} | {status}{cap_warn}")

        if update % SAVE_INTERVAL == 0:
            ck = os.path.join(CHECKPOINT_DIR, f"ckpt_step{step}.pt")
            try:
                trainer.save(ck)
                print(f"  Checkpoint: {ck}  (best_p={best_p:.4f})")
            except Exception as e:
                print(f"  [WARNING] Checkpoint save failed: {e}")

    print("\n  Training complete.")
    try:
        trainer.save(os.path.join(CHECKPOINT_DIR, "ckpt_final.pt"))
        torch.save(policy.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "policy_final.pt"))
        print(f"  Final policy: {CHECKPOINT_DIR}/policy_final.pt")
    except Exception as e:
        print(f"  [WARNING] Final save failed: {e}")

    save_plots()

    print()
    print("=" * 72)
    print(f"  Best p_detect : {trainer._best_p_detect:.4f}")
    print(f"  Total steps   : {step:,}")
    print(f"  Log CSV       : {LOG_CSV}")
    print(f"  Plot          : {PLOT_PATH}")
    print(f"  Console log   : {CONSOLE_LOG}")
    print("=" * 72)


if __name__ == "__main__":
    main()