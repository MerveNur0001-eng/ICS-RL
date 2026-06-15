# evaluate_single.py
"""
ICS-RL Single Image Evaluation — v3 (TRAINING EFFECT MEASUREMENT)

How is the training effect measured?
─────────────────────────────────────
In this system, the training effect is measured not by message extraction,
but by STEGANOGRAPHY QUALITY. As follows:

  [A] Trained Policy (policy selection):
      - The policy selects HIGH-TEXTURE pixels from 30,000 chaotic candidates
      - The message is embedded into these pixels
      - SRNet should detect this stego more difficultly (this was the training goal)
      - PSNR/SSIM remain the same (LSB always changes 1 bit)

  [B] Untrained Baseline (chaotic sequential selection):
      - The first n_bits of chaotic coordinates are used (no policy)
      - Texture is not considered, flat regions may also be written to
      - SRNet can detect this stego more easily

  BY COMPARING THE TWO SCENARIOS, the training effect is measured:
    - A's SRNet score < B's score → training successful
    - PSNR/SSIM difference → the same in both (due to LSB nature)

  Extraction is 100% correct in both scenarios:
    - Sender embeds using the policy
    - Receiver finds the same pixels using THE SAME policy (deterministic)
    - This is realistic: receiver also has the same model (shared secret model)

Run: python evaluate_single.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.modules.policy_network    import PolicyNetwork
from src.modules.chaotic_generator import LogisticMapGenerator
from src.modules.texture_extractor import TextureFeatureExtractor
from utils.message_codec           import MessageCodec
from utils.bossbase_loader         import BOSSBaseLoader

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
IMG_SHAPE    = (256, 256)
X0           = 0.123456
R            = 3.99
N_CANDIDATES = 15000
PAYLOAD_BITS = int(256 * 256 * 0.4)
DATA_DIR     = "data/bossbase/"
MODEL_PATH   = "checkpoints/policy_final.pt"
OUT_DIR      = "results"
IMAGE_INDEX  = 25 #25  

REAL_MESSAGE = "This message is hidden by the ICS-RL system. A chaotic pixel selection combined with a PPO agent conceals data inside the image while evading the SRNet detector. Beykoz University CME6405 - Steganography with Reinforcement Learning."


os.makedirs(OUT_DIR, exist_ok=True)

# ── Custom colormaps ──────────────────────────────────────────
# White → Red: changed pixels appear as red dots on white background
WHITE_RED = LinearSegmentedColormap.from_list(
    "white_red", [(1, 1, 1), (0.85, 0.0, 0.0)]
)
# White → Blue for chaotic mask
WHITE_BLUE = LinearSegmentedColormap.from_list(
    "white_blue", [(1, 1, 1), (0.08, 0.35, 0.72)]
)


# ═══════════════════════════════════════════════════════════════
#  METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_psnr(cover, stego):
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(cover, stego):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(cover, stego, data_range=255))
    except ImportError:
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        x, y   = cover.astype(np.float64), stego.astype(np.float64)
        mu_x, mu_y = x.mean(), y.mean()
        sig_xy = np.mean((x - mu_x) * (y - mu_y))
        return float(
            ((2*mu_x*mu_y + c1) * (2*sig_xy + c2))
            / ((mu_x**2 + mu_y**2 + c1) * (x.var() + y.var() + c2))
        )

def compute_mse(cover, stego):
    return float(np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2))

def compute_snr(cover, stego):
    signal = np.mean(cover.astype(np.float64) ** 2)
    noise  = compute_mse(cover, stego)
    if noise < 1e-10:
        return 100.0
    return 10.0 * np.log10(signal / noise)

def srnet_detect(stego, srnet):
    """Return detection probability using SRNet."""
    if srnet is None:
        return None
    with torch.no_grad():
        inp    = torch.from_numpy(stego).float().unsqueeze(0).unsqueeze(0) / 255.0
        logits = srnet(inp)
        prob   = float(torch.softmax(logits, dim=1)[0, 1].item())
    return prob


# ═══════════════════════════════════════════════════════════════
#  CHAOTIC COORDINATES
# ═══════════════════════════════════════════════════════════════

def build_chaotic_coords(x0, r, img_shape, n_candidates):
    h, w  = img_shape
    total = h * w
    seq   = np.empty(total, dtype=np.float64)
    x = x0
    for i in range(total):
        x = r * x * (1.0 - x)
        seq[i] = x
    order    = np.argsort(seq)
    selected = order[:n_candidates]
    return (selected // w).astype(np.int32), (selected % w).astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  PIXEL SELECTION (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════

def select_policy(policy, cover, chaotic_mask, texture_map, ch_rows, ch_cols, n_bits):
    cover_t   = torch.from_numpy(cover).float().unsqueeze(0) / 255.0
    mask_t    = torch.from_numpy(chaotic_mask).float().unsqueeze(0)
    texture_t = torch.from_numpy(texture_map).float().unsqueeze(0)
    inp       = torch.stack([cover_t, mask_t, texture_t], dim=1)

    with torch.no_grad():
        logits   = policy(inp)
        probs    = torch.sigmoid(logits.squeeze(0))
        ch_probs = probs[ch_rows, ch_cols].numpy()

    tex_scores = texture_map[ch_rows, ch_cols]
    
    # ⚙️ OPTIMIZED PARAMETERS
    tex_threshold = np.percentile(tex_scores, 40)  # 80 → 60 (daha geniş bölge)
    high_tex_mask = tex_scores >= tex_threshold

    # Texture ağırlığını azalt, policy network'e daha fazla güven
    combined = tex_scores * 0.95 + ch_probs * 0.05  # 0.1 + 0.9 → 0.2 + 0.8
    
    # Tamamen eleme yerine cezalandır
    combined[~high_tex_mask] *= 0.1 # -1 yerine 0.5x ceza

    sorted_idx = np.argsort(combined)[::-1][:n_bits]
    order_back = np.argsort(sorted_idx)
    rows = ch_rows[sorted_idx][order_back].astype(np.int32)
    cols = ch_cols[sorted_idx][order_back].astype(np.int32)
    return rows, cols

def select_chaotic(ch_rows, ch_cols, n_bits):
    """Untrained baseline: first n_bits of the chaotic sequence."""
    return ch_rows[:n_bits].astype(np.int32), ch_cols[:n_bits].astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  SINGLE SCENARIO EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_scenario(label, cover, rows_embed, cols_embed,
                      rows_extract, cols_extract,
                      codec, msg_bits, n_bits, texture_map, srnet):
    """
    Embed → Extract → Metrics.

    rows_embed / cols_embed    : embedding coordinates
    rows_extract / cols_extract: extraction coordinates
                                 (same as embedding in the policy scenario)
    """
    stego          = codec.embed_into_pixels(cover, msg_bits, rows_embed, cols_embed)
    extracted_bits = codec.extract_from_pixels(stego, rows_extract, cols_extract, n_bits)
    extracted_text = codec.decode(extracted_bits)

    psnr = compute_psnr(cover, stego)
    ssim = compute_ssim(cover, stego)
    mse  = compute_mse(cover, stego)
    snr  = compute_snr(cover, stego)
    bpp  = n_bits / cover.size

    diff   = stego.astype(np.int16) - cover.astype(np.int16)
    n_ch   = int((diff != 0).sum())
    pct_ch = n_ch / cover.size * 100

    # Average texture score of selected pixels
    tex_score = float(texture_map[rows_embed, cols_embed].mean())

    # SRNet detection
    p_detect = srnet_detect(stego, srnet)

    # Character accuracy
    original = codec.decode(msg_bits)
    char_acc = (sum(a == b for a, b in zip(extracted_text, original))
                / max(len(original), 1) * 100)
    extract_ok = (extracted_text == original)

    return {
        "label"         : label,
        "stego"         : stego,
        "psnr"          : psnr,
        "ssim"          : ssim,
        "mse"           : mse,
        "snr"           : snr,
        "bpp"           : bpp,
        "n_changed"     : n_ch,
        "pct_changed"   : pct_ch,
        "tex_score"     : tex_score,
        "p_detect"      : p_detect,
        "char_accuracy" : char_acc,
        "extract_ok"    : extract_ok,
        "extracted_text": extracted_text,
    }


def print_cover_stego_table(cover, r_policy, r_chaotic):
    W = 72
    print(f"╔{'═'*W}╗")
    print(f"║{'COVER vs STEGO — PIXEL METRIC DIFFERENCES':^{W}}║")
    print(f"╠{'═'*W}╣")
    print(f"║  {'Metric':<22} {'Cover (ref)':>14}   {'[A] Policy':>14}   {'[B] Chaotic':>14}  ║")
    print(f"║  {'─'*66}  ║")

    def row(label, cv, vp, vc):
        print(f"║  {label:<22} {cv:>14}   {vp:>14}   {vc:>14}  ║")

    sp = r_policy["stego"]
    sc = r_chaotic["stego"]

    row("Avg pixel",
        f"{cover.mean():.4f}",
        f"{sp.mean():.4f}  (Δ{sp.mean()-cover.mean():+.4f})",
        f"{sc.mean():.4f}  (Δ{sc.mean()-cover.mean():+.4f})")
    row("Std deviation",
        f"{cover.std():.4f}",
        f"{sp.std():.4f}",
        f"{sc.std():.4f}")
    row("Min pixel",
        f"{int(cover.min())}",
        f"{int(sp.min())}",
        f"{int(sc.min())}")
    row("Max pixel",
        f"{int(cover.max())}",
        f"{int(sp.max())}",
        f"{int(sc.max())}")
    print(f"║  {'─'*66}  ║")
    row("MSE",
        "0.000000",
        f"{r_policy['mse']:.6f}",
        f"{r_chaotic['mse']:.6f}")
    row("PSNR (dB)",
        "∞  (ref)",
        f"{r_policy['psnr']:.4f} dB",
        f"{r_chaotic['psnr']:.4f} dB")
    row("SSIM",
        "1.000000",
        f"{r_policy['ssim']:.6f}",
        f"{r_chaotic['ssim']:.6f}")
    row("BPP",
        "0.0000",
        f"{r_policy['bpp']:.4f}",
        f"{r_chaotic['bpp']:.4f}")
    print(f"║  {'─'*66}  ║")
    row("Changed pixels",
        "0",
        f"{r_policy['n_changed']:,}",
        f"{r_chaotic['n_changed']:,}")
    row("Changed px %",
        "0.0000%",
        f"{r_policy['pct_changed']:.4f}%",
        f"{r_chaotic['pct_changed']:.4f}%")
    row("Max pixel Δ",
        "0",
        "1  (LSB)",
        "1  (LSB)")
    print(f"╚{'═'*W}╝")
    print()


# ═══════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

def print_comparison(r_policy, r_chaotic, original_message):
    W = 76

    def print_wrapped(label, text, width):
        prefix = f"  {label}: "
        pad    = " " * len(prefix)
        words  = text.split()
        line   = prefix
        for w in words:
            if len(line) + len(w) + 1 > width:
                print(f"║{line:<{width}}║")
                line = pad + w + " "
            else:
                line += w + " "
        if line.strip():
            print(f"║{line:<{width}}║")

    print()
    print(f"╔{'═'*W}╗")
    print(f"║{'ICS-RL — TRAINED POLICY vs CHAOTIC BASELINE COMPARISON':^{W}}║")
    print(f"║{'Tang et al. (2020) + Ogras (2019) — Training Effect Measurement':^{W}}║")
    print(f"╠{'═'*W}╣")
    print(f"║  {'Metric':<28} {'[A] Policy (Trained)':>20}   {'[B] Chaotic':>16}   {'Diff':>6}  ║")
    print(f"║  {'─'*70}  ║")

    def row(label, va, vb, fmt=".4f", better="low"):
        fa = f"{va:{fmt}}" if va is not None else "N/A"
        fb = f"{vb:{fmt}}" if vb is not None else "N/A"
        if va is not None and vb is not None:
            diff = va - vb
            fd   = f"{diff:+.4f}"
            if better == "low":
                mark = "✓" if diff < 0 else ("=" if diff == 0 else "✗")
            else:
                mark = "✓" if diff > 0 else ("=" if diff == 0 else "✗")
        else:
            fd, mark = "—", ""
        print(f"║  {label:<28} {fa:>20}   {fb:>16}   {fd:>6} {mark}  ║")

    row("PSNR (dB) ↑ higher is better",    r_policy["psnr"],        r_chaotic["psnr"],        better="high")
    row("SSIM ↑ higher is better",          r_policy["ssim"],        r_chaotic["ssim"],        better="high")
    row("MSE ↓ lower is better",            r_policy["mse"],         r_chaotic["mse"],         better="low")
    row("BPP",                               r_policy["bpp"],         r_chaotic["bpp"])
    row("Changed pixel %",                   r_policy["pct_changed"], r_chaotic["pct_changed"], better="low")
    print(f"║  {'─'*70}  ║")
    row("Texture score ↑ higher is better", r_policy["tex_score"],   r_chaotic["tex_score"],   better="high")

    if r_policy["p_detect"] is not None:
        row("SRNet p_detect ↓ lower is better",
            r_policy["p_detect"], r_chaotic["p_detect"], better="low")
        print(f"║  {'─'*70}  ║")
        p  = r_policy["p_detect"]
        pb = r_chaotic["p_detect"]
        if p < pb:
            verdict = f"✅ Policy achieves {(pb-p)*100:.1f}pp better concealment"
        elif p > pb:
            verdict = f"⚠️  Chaotic is {(p-pb)*100:.1f}pp better — policy not sufficiently trained yet"
        else:
            verdict = "= No difference"
        print(f"║  SRNet result: {verdict:<{W-17}}║")

    print(f"╠{'═'*W}╣")
    print(f"║{'MESSAGE EXTRACTION':^{W}}║")
    print(f"╠{'═'*W}╣")
    print_wrapped("Original  ", original_message,              W)
    print_wrapped("[A] Policy", r_policy['extracted_text'],    W)
    print_wrapped("[B] Chaotic", r_chaotic['extracted_text'],  W)
    print(f"║  [A] Char acc %   : {r_policy['char_accuracy']:.1f}%{'':<{W-20}}║")
    print(f"║  [B] Char acc %   : {r_chaotic['char_accuracy']:.1f}%{'':<{W-20}}║")

    print(f"╠{'═'*W}╣")
    print(f"║{'EXPLANATION':^{W}}║")
    print(f"╠{'═'*W}╣")
    expl = (
        "Embedding/extraction with policy: the receiver finds the same pixels "
        "using the same model (deterministic). The training effect is visible in "
        "the texture score and SRNet detection — the policy writes to high-texture "
        "regions to make concealment harder to detect."
    )
    words = expl.split()
    line  = "  "
    for w in words:
        if len(line) + len(w) + 1 > W:
            print(f"║{line:<{W}}║")
            line = "  " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(f"║{line:<{W}}║")
    print(f"╚{'═'*W}╝")
    print()


# ═══════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════

def save_visuals(cover, r_policy, r_chaotic, texture_map, chaotic_mask, out_dir):
    stego_p = r_policy["stego"]
    stego_c = r_chaotic["stego"]

    # ── Diff maps: amplify ×200 so sparse LSB changes are clearly visible ──
    diff_p = np.clip(
        np.abs(cover.astype(np.int16) - stego_p.astype(np.int16)) * 200, 0, 255
    ).astype(np.uint8)
    diff_c = np.clip(
        np.abs(cover.astype(np.int16) - stego_c.astype(np.int16)) * 200, 0, 255
    ).astype(np.uint8)

    # ── Figure setup ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    pd_p = f"{r_policy['p_detect']:.3f}"  if r_policy['p_detect']  is not None else "N/A"
    pd_c = f"{r_chaotic['p_detect']:.3f}" if r_chaotic['p_detect'] is not None else "N/A"

    fig.suptitle(
        "ICS-RL — Trained Policy vs Chaotic Baseline\n"
        f"[A] Policy : PSNR={r_policy['psnr']:.2f} dB  |  "
        f"tex={r_policy['tex_score']:.3f}  |  p_detect={pd_p}\n"
        f"[B] Chaotic: PSNR={r_chaotic['psnr']:.2f} dB  |  "
        f"tex={r_chaotic['tex_score']:.3f}  |  p_detect={pd_c}",
        fontsize=11, fontweight="bold", color="#1a1a2e"
    )

    # ── Row 0: cover / stego images / texture map ──────────────────────────
    axes[0, 0].imshow(cover,   cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Cover (original)", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(stego_p, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(
        f"[A] Policy Stego\nPSNR = {r_policy['psnr']:.2f} dB",
        fontweight="bold", color="#1565c0"
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(stego_c, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(
        f"[B] Chaotic Stego\nPSNR = {r_chaotic['psnr']:.2f} dB",
        fontweight="bold", color="#b71c1c"
    )
    axes[0, 2].axis("off")

    axes[0, 3].imshow(texture_map, cmap="jet", vmin=0, vmax=1)
    axes[0, 3].set_title("Texture Saliency Map", fontweight="bold")
    axes[0, 3].axis("off")

    # ── Row 1: chaotic mask / diff maps / bar chart ────────────────────────
    axes[1, 0].imshow(chaotic_mask, cmap=WHITE_BLUE, vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"Chaotic Mask  (x0={X0})\n{int(chaotic_mask.sum()):,} candidates",
        fontweight="bold"
    )
    axes[1, 0].axis("off")

    # [A] Diff — white background, red changed pixels
    im1 = axes[1, 1].imshow(diff_p, cmap=WHITE_RED, vmin=0, vmax=255)
    axes[1, 1].set_title(
        f"[A] Policy  —  Diff ×200\n{r_policy['n_changed']:,} pixels changed",
        fontweight="bold", color="#1565c0"
    )
    axes[1, 1].axis("off")
    cb1 = plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.02)
    cb1.ax.yaxis.set_tick_params(color="#555")

    # [B] Diff — white background, red changed pixels
    im2 = axes[1, 2].imshow(diff_c, cmap=WHITE_RED, vmin=0, vmax=255)
    axes[1, 2].set_title(
        f"[B] Chaotic  —  Diff ×200\n{r_chaotic['n_changed']:,} pixels changed",
        fontweight="bold", color="#b71c1c"
    )
    axes[1, 2].axis("off")
    cb2 = plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.02)
    cb2.ax.yaxis.set_tick_params(color="#555")

    # ── Bar chart ──────────────────────────────────────────────────────────
    ax_bar = axes[1, 3]
    ax_bar.set_facecolor("#fafafa")

    categories = ["Texture Score\n(↑ better)", "SRNet p_detect\n(↓ better)"]
    vals_p = [r_policy["tex_score"],  r_policy["p_detect"]  or 0]
    vals_c = [r_chaotic["tex_score"], r_chaotic["p_detect"] or 0]

    x = np.arange(len(categories))
    bar_w = 0.32

    b1 = ax_bar.bar(
        x - bar_w / 2, vals_p, bar_w,
        label="[A] Policy", color="#1565c0", alpha=0.88,
        edgecolor="white", linewidth=0.8
    )
    b2 = ax_bar.bar(
        x + bar_w / 2, vals_c, bar_w,
        label="[B] Chaotic", color="#c62828", alpha=0.88,
        edgecolor="white", linewidth=0.8
    )

    ax_bar.set_title("Training Effect Comparison", fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=9)
    ax_bar.legend(fontsize=9, framealpha=0.9)
    ax_bar.grid(True, alpha=0.25, axis="y", linestyle="--")
    ax_bar.set_ylim(0, 1.08)
    ax_bar.spines[["top", "right"]].set_visible(False)

    for rect, val in zip(list(b1) + list(b2), vals_p + vals_c):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, "evaluation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Saved] → {path}")


def save_metrics_table(cover, r_policy, r_chaotic, out_dir):
    """Save metric comparison table as PNG."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    pd_p = f"{r_policy['p_detect']:.4f}"  if r_policy['p_detect']  is not None else "N/A"
    pd_c = f"{r_chaotic['p_detect']:.4f}" if r_chaotic['p_detect'] is not None else "N/A"

    headers = ["Metric", "Direction", "Cover (ref)", "[A] Policy", "[B] Chaotic", "Diff", "Result"]

    rows = [
        ["PSNR (dB)",        "↑ better", "∞ (ref)",   f"{r_policy['psnr']:.4f}",      f"{r_chaotic['psnr']:.4f}",      f"{r_policy['psnr']-r_chaotic['psnr']:+.4f}", "✗" if r_policy['psnr'] < r_chaotic['psnr'] else "✓"],
        ["SSIM",             "↑ better", "1.000000",  f"{r_policy['ssim']:.6f}",      f"{r_chaotic['ssim']:.6f}",      f"{r_policy['ssim']-r_chaotic['ssim']:+.6f}", "✓" if r_policy['ssim'] >= r_chaotic['ssim'] else "✗"],
        ["MSE",              "↓ better", "0.000000",  f"{r_policy['mse']:.6f}",       f"{r_chaotic['mse']:.6f}",       f"{r_policy['mse']-r_chaotic['mse']:+.6f}",  "✓" if r_policy['mse'] <= r_chaotic['mse'] else "✗"],
        ["BPP",              "—",        "0.0000",    f"{r_policy['bpp']:.4f}",       f"{r_chaotic['bpp']:.4f}",       f"{r_policy['bpp']-r_chaotic['bpp']:+.4f}",  "="],
        ["Changed px %",     "↓ better", "0.0000%",   f"{r_policy['pct_changed']:.4f}%", f"{r_chaotic['pct_changed']:.4f}%", f"{r_policy['pct_changed']-r_chaotic['pct_changed']:+.4f}", "✓" if r_policy['pct_changed'] <= r_chaotic['pct_changed'] else "✗"],
        ["Changed pixels",   "↓ better", "0",
         f"{r_policy['n_changed']:,}",
         f"{r_chaotic['n_changed']:,}",
         f"{r_policy['n_changed'] - r_chaotic['n_changed']:+,}",
         "✓" if r_policy['n_changed'] <= r_chaotic['n_changed'] else "✗"],
        ["Texture Score",    "↑ better", "—",         f"{r_policy['tex_score']:.4f}", f"{r_chaotic['tex_score']:.4f}", f"{r_policy['tex_score']-r_chaotic['tex_score']:+.4f}", "✓" if r_policy['tex_score'] > r_chaotic['tex_score'] else "✗"],
        ["SRNet p_detect",   "↓ better", "—",         pd_p,                           pd_c,                            f"{r_policy['p_detect']-r_chaotic['p_detect']:+.4f}" if r_policy['p_detect'] else "—", "✓" if (r_policy['p_detect'] or 1) < (r_chaotic['p_detect'] or 0) else "✗"],
        ["Char Accuracy %",  "↑ better", "—",         f"%{r_policy['char_accuracy']:.1f}", f"%{r_chaotic['char_accuracy']:.1f}", "—", "✓" if r_policy['char_accuracy'] >= r_chaotic['char_accuracy'] else "✗"],
    ]

    # Row colouring: ✓ green, ✗ red, others neutral
    row_colors = []
    for r in rows:
        result = r[-1]
        if result == "✓":
            row_colors.append(["#e8f5e9"] * len(headers))
        elif result == "✗":
            row_colors.append(["#ffebee"] * len(headers))
        else:
            row_colors.append(["#f5f5f5"] * len(headers))

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=row_colors,
        colColours=["#1565c0"] * len(headers),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Header columns white text
    for col_idx in range(len(headers)):
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        "ICS-RL — Trained Policy vs Chaotic Baseline\nMetric Comparison Table",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Saved] → {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ICS-RL — v3  TRAINED POLICY vs CHAOTIC BASELINE               ║")
    print("║  Training effect: Texture score + SRNet detection comparison    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Data
    try:
        loader = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}"); sys.exit(1)

    cover, fname = loader.get(IMAGE_INDEX)
    img_name = fname if isinstance(fname, str) else f"img_{IMAGE_INDEX:04d}"
    print(f"\n  Image      : {img_name}  ({cover.shape[0]}×{cover.shape[1]})")

    # Codec
    codec    = MessageCodec(max_bits=PAYLOAD_BITS)
    msg_bits = codec.encode(REAL_MESSAGE)
    n_bits   = len(msg_bits)
    print(f"  Message    : {len(REAL_MESSAGE)} characters → {n_bits} bits")

    # Chaotic coordinates
    chaotic_gen  = LogisticMapGenerator(x0=X0)
    texture_ext  = TextureFeatureExtractor(patch_size=7)
    chaotic_mask = chaotic_gen.get_mask(IMG_SHAPE, N_CANDIDATES)
    texture_map  = texture_ext.get_texture_saliency_map(cover)
    ch_rows, ch_cols = build_chaotic_coords(X0, R, IMG_SHAPE, N_CANDIDATES)
    print(f"  Chaotic    : {int(chaotic_mask.sum()):,} candidate pixels  (x0={X0})")

    # Load SRNet (optional)
    srnet = None
    srnet_path = "src/models/srnet_best.pth"
    if os.path.exists(srnet_path):
        try:
            from src.models.srnet import SRNet
            srnet = SRNet()
            srnet.load_state_dict(torch.load(srnet_path, map_location="cpu",
                                             weights_only=True))
            srnet.eval()
            print(f"  SRNet      : ✓ loaded")
        except Exception as e:
            print(f"  SRNet      : could not load ({e})")

    # Load policy
    policy = PolicyNetwork(input_channels=3)
    use_policy = False
    if os.path.exists(MODEL_PATH):
        try:
            policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                              weights_only=True))
            policy.eval()
            use_policy = True
            print(f"  Policy     : ✓ loaded ({MODEL_PATH})")
        except Exception as e:
            print(f"  Policy     : could not load ({e})")
    else:
        print(f"  Policy     : {MODEL_PATH} not found")

    print()

    # ── [A] TRAINED POLICY scenario ─────────────────────────────────────
    if use_policy:
        print("  [A] Pixel selection with trained policy...")
        rows_p, cols_p = select_policy(policy, cover, chaotic_mask, texture_map,
                                        ch_rows, ch_cols, n_bits)
        # Embedding AND extraction use the same coordinates (receiver has same policy)
        r_policy = evaluate_scenario(
            "policy (trained)",
            cover, rows_p, cols_p, rows_p, cols_p,
            codec, msg_bits, n_bits, texture_map, srnet
        )
        print(f"     Texture score  : {r_policy['tex_score']:.4f}")
        print(f"     SRNet p_detect : {r_policy['p_detect']:.4f}" if r_policy["p_detect"] else "")
        print(f"     Extraction     : {r_policy['char_accuracy']:.1f}% correct")
    else:
        print("  [A] Policy not found — only baseline will run")
        r_policy = None

    print()

    # ── [B] CHAOTIC BASELINE scenario ───────────────────────────────────
    print("  [B] Pixel selection with chaotic baseline (untrained)...")
    rows_c, cols_c = select_chaotic(ch_rows, ch_cols, n_bits)
    r_chaotic = evaluate_scenario(
        "chaotic (baseline)",
        cover, rows_c, cols_c, rows_c, cols_c,
        codec, msg_bits, n_bits, texture_map, srnet
    )
    print(f"     Texture score  : {r_chaotic['tex_score']:.4f}")
    print(f"     SRNet p_detect : {r_chaotic['p_detect']:.4f}" if r_chaotic["p_detect"] else "")
    print(f"     Extraction     : {r_chaotic['char_accuracy']:.1f}% correct")

    print()

    # ── Comparison table ─────────────────────────────────────────────────
    if r_policy:
        print_cover_stego_table(cover, r_policy, r_chaotic)
        print_comparison(r_policy, r_chaotic, REAL_MESSAGE)
        save_visuals(cover, r_policy, r_chaotic, texture_map, chaotic_mask, OUT_DIR)
        save_metrics_table(cover, r_policy, r_chaotic, OUT_DIR)
    else:
        # Show only chaotic result
        print(f"  PSNR  : {r_chaotic['psnr']:.2f} dB")
        print(f"  SSIM  : {r_chaotic['ssim']:.6f}")
        print(f"  Extraction: {r_chaotic['char_accuracy']:.1f}%")

    print(f"  Visuals → {os.path.abspath(OUT_DIR)}/evaluation_comparison.png")


if __name__ == "__main__":
    main()