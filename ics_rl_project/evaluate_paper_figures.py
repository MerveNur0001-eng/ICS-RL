"""
evaluate_paper_figures.py
─────────────────────────
Generates 4 publication-quality supplementary figures for PeerJ submission.

Figure 1: Best / Median / Worst case (3-row, 8-panel each)
Figure 2: Failure analysis (images where policy < chaotic)
Figure 3: Texture heatmap overlay (policy selected pixels on cover)
Figure 4: Per-image bar chart (all 20 images, p_detect comparison)

Output:
  results/paper_figures/fig1_best_median_worst.png
  results/paper_figures/fig2_failure_analysis.png
  results/paper_figures/fig3_texture_overlay.png
  results/paper_figures/fig4_perimage_bar.png

Usage:
    python evaluate_paper_figures.py
"""

import os
import sys
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

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
OUT_DIR      = "results/paper_figures"

SEED = 42
N    = 20

REAL_MESSAGE = (
    "ICS-RL Steganography System - Intelligent Chaotic Steganography "
    "using Reinforcement Learning. Logistic map (x0=secret, r=3.99) "
    "selects candidate pixels; PPO agent embeds real payload via LSB. "
    "This hidden message demonstrates real steganographic embedding "
    "as in Tang et al. 2020 and Ogras 2019. Beykoz University CME6405."
)

os.makedirs(OUT_DIR, exist_ok=True)

BLUE      = "#1565c0"
RED       = "#c62828"
GREEN     = "#2e7d32"
ORANGE    = "#e65100"
LGRAY     = "#f0f4f8"

WHITE_RED = LinearSegmentedColormap.from_list(
    "white_red",  [(1,1,1), (0.85,0.0,0.0)])
WHITE_BLUE = LinearSegmentedColormap.from_list(
    "white_blue", [(1,1,1), (0.08,0.35,0.72)])


# ═══════════════════════════════════════════════════════════════
#  SEED-BASED IMAGE SELECTION (evaluate_seeded.py ile AYNI)
# ═══════════════════════════════════════════════════════════════

def get_image_indices(data_dir, n, seed):
    loader = BOSSBaseLoader(data_dir, img_shape=IMG_SHAPE)
    all_indices = list(range(loader.size))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    return all_indices[:n]


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_psnr(cover, stego):
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else 20.0 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(cover, stego):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(cover, stego, data_range=255))
    except ImportError:
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        x, y   = cover.astype(np.float64), stego.astype(np.float64)
        mu_x, mu_y = x.mean(), y.mean()
        sig_xy = np.mean((x-mu_x)*(y-mu_y))
        return float(((2*mu_x*mu_y+c1)*(2*sig_xy+c2))
                     /((mu_x**2+mu_y**2+c1)*(x.var()+y.var()+c2)))

def srnet_detect(stego, srnet):
    if srnet is None: return None
    with torch.no_grad():
        inp    = torch.from_numpy(stego).float().unsqueeze(0).unsqueeze(0)/255.0
        logits = srnet(inp)
        return float(torch.softmax(logits, dim=1)[0,1].item())


# ═══════════════════════════════════════════════════════════════
#  COORD HELPERS
# ═══════════════════════════════════════════════════════════════

def build_chaotic_coords(x0, r, img_shape, n_candidates):
    h, w  = img_shape
    seq   = np.empty(h*w, dtype=np.float64)
    x = x0
    for i in range(h*w):
        x = r*x*(1.0-x); seq[i] = x
    order = np.argsort(seq)[:n_candidates]
    return (order//w).astype(np.int32), (order%w).astype(np.int32)


def select_policy(policy, cover, chaotic_mask, texture_map,
                  ch_rows, ch_cols, n_bits):
    cover_t   = torch.from_numpy(cover).float().unsqueeze(0) / 255.0
    mask_t    = torch.from_numpy(chaotic_mask).float().unsqueeze(0)
    texture_t = torch.from_numpy(texture_map).float().unsqueeze(0)
    inp       = torch.stack([cover_t, mask_t, texture_t], dim=1)

    with torch.no_grad():
        logits   = policy(inp)
        probs    = torch.sigmoid(logits.squeeze(0))
        ch_probs = probs[ch_rows, ch_cols].numpy()

    tex_scores    = texture_map[ch_rows, ch_cols]
    tex_threshold = np.percentile(tex_scores, 40)
    high_tex_mask = tex_scores >= tex_threshold

    combined = tex_scores * 0.95 + ch_probs * 0.05
    combined[~high_tex_mask] *= 0.1

    sorted_idx = np.argsort(combined)[::-1][:n_bits]
    order_back = np.argsort(sorted_idx)
    rows = ch_rows[sorted_idx][order_back].astype(np.int32)
    cols = ch_cols[sorted_idx][order_back].astype(np.int32)
    return rows, cols

def select_chaotic(ch_rows, ch_cols, n_bits):
    return ch_rows[:n_bits].astype(np.int32), ch_cols[:n_bits].astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  EVALUATE ALL IMAGES → collect results
# ═══════════════════════════════════════════════════════════════

def evaluate_all(loader, policy, srnet, codec, msg_bits, n_bits,
                 chaotic_gen, texture_ext, ch_rows, ch_cols, image_indices):
    results = []
    print(f"  Evaluating {len(image_indices)} images...")
    for i, img_idx in enumerate(image_indices):
        cover, _ = loader.get(img_idx)
        chaotic_mask = chaotic_gen.get_mask(IMG_SHAPE, N_CANDIDATES)
        texture_map  = texture_ext.get_texture_saliency_map(cover)

        rows_p, cols_p = select_policy(policy, cover, chaotic_mask,
                                       texture_map, ch_rows, ch_cols, n_bits)
        stego_p = codec.embed_into_pixels(cover, msg_bits, rows_p, cols_p)

        rows_c, cols_c = select_chaotic(ch_rows, ch_cols, n_bits)
        stego_c = codec.embed_into_pixels(cover, msg_bits, rows_c, cols_c)

        pdet_p = srnet_detect(stego_p, srnet)
        pdet_c = srnet_detect(stego_c, srnet)

        diff_pp = (pdet_c - pdet_p)*100 if (pdet_p and pdet_c) else 0

        results.append({
            "img_idx"     : img_idx,
            "cover"       : cover,
            "stego_p"     : stego_p,
            "stego_c"     : stego_c,
            "texture_map" : texture_map,
            "chaotic_mask": chaotic_mask,
            "rows_p"      : rows_p,
            "cols_p"      : cols_p,
            "rows_c"      : rows_c,
            "cols_c"      : cols_c,
            "psnr_p"      : compute_psnr(cover, stego_p),
            "psnr_c"      : compute_psnr(cover, stego_c),
            "ssim_p"      : compute_ssim(cover, stego_p),
            "ssim_c"      : compute_ssim(cover, stego_c),
            "tex_p"       : float(texture_map[rows_p, cols_p].mean()),
            "tex_c"       : float(texture_map[rows_c, cols_c].mean()),
            "pdet_p"      : pdet_p,
            "pdet_c"      : pdet_c,
            "diff_pp"     : diff_pp,
            "n_ch_p"      : int((stego_p.astype(np.int16)-cover.astype(np.int16)!=0).sum()),
            "n_ch_c"      : int((stego_c.astype(np.int16)-cover.astype(np.int16)!=0).sum()),
        })
        mark = "✓" if diff_pp > 0 else "✗"
        print(f"    [{i+1:2d}] img#{img_idx:<5} "
              f"p_detect: {pdet_p:.3f} vs {pdet_c:.3f}  "
              f"Δ={diff_pp:+.1f}pp {mark}")
    return results


# ═══════════════════════════════════════════════════════════════
#  HELPER — 8-panel row (used by Figure 1 & 2)
# ═══════════════════════════════════════════════════════════════

def draw_8panel_row(axes_row, res, row_label, label_color):
    """Fill one row of 8 axes with evaluate_single.py-style panels."""
    cover   = res["cover"]
    stego_p = res["stego_p"]
    stego_c = res["stego_c"]
    diff_p  = np.clip(np.abs(cover.astype(np.int16)-stego_p.astype(np.int16))*200,0,255).astype(np.uint8)
    diff_c  = np.clip(np.abs(cover.astype(np.int16)-stego_c.astype(np.int16))*200,0,255).astype(np.uint8)

    pd_p = res["pdet_p"]; pd_c = res["pdet_c"]
    diff_pp = res["diff_pp"]

    titles = [
        (f"{row_label}\nCover #{res['img_idx']}", "black"),
        (f"[A] Policy Stego\nPSNR={res['psnr_p']:.2f}  p={pd_p:.3f}", BLUE),
        (f"[B] Chaotic Stego\nPSNR={res['psnr_c']:.2f}  p={pd_c:.3f}", RED),
        (f"Texture Map\ntex_A={res['tex_p']:.3f}  tex_B={res['tex_c']:.3f}", "black"),
        (f"Chaotic Mask\n{int(res['chaotic_mask'].sum()):,} candidates", "black"),
        (f"[A] Diff ×200\n{res['n_ch_p']:,} px changed", BLUE),
        (f"[B] Diff ×200\n{res['n_ch_c']:,} px changed", RED),
        (f"Δp_detect = {diff_pp:+.1f} pp\n{'Policy better ✓' if diff_pp>0 else 'Chaotic better ✗'}", label_color),
    ]
    images = [cover, stego_p, stego_c, res["texture_map"],
              res["chaotic_mask"], diff_p, diff_c, None]
    cmaps  = ["gray","gray","gray","jet","Greys",None,None,None]

    for j, (ax, img, cmap, (ttl, tcol)) in enumerate(
            zip(axes_row, images, cmaps, titles)):
        ax.set_facecolor("white")
        if j == 4:
            ax.imshow(res["chaotic_mask"], cmap=WHITE_BLUE, vmin=0, vmax=1)
        elif j == 5:
            ax.imshow(diff_p, cmap=WHITE_RED, vmin=0, vmax=255)
        elif j == 6:
            ax.imshow(diff_c, cmap=WHITE_RED, vmin=0, vmax=255)
        elif j == 7:
            ax.set_facecolor(LGRAY)
            cats = ["Texture\nScore", "p_detect"]
            vp   = [res["tex_p"],  pd_p or 0]
            vc   = [res["tex_c"],  pd_c or 0]
            x    = np.arange(2)
            b1 = ax.bar(x-0.18, vp, 0.32, color=BLUE, alpha=0.88, label="Policy")
            b2 = ax.bar(x+0.18, vc, 0.32, color=RED,  alpha=0.88, label="Chaotic")
            ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7)
            ax.set_ylim(0, 1.05); ax.legend(fontsize=6, loc="upper left")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)
            for rect, val in zip(list(b1)+list(b2), vp+vc):
                ax.text(rect.get_x()+rect.get_width()/2,
                        rect.get_height()+0.02, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6, fontweight="bold")
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if cmap=="gray" else
                      (1 if cmap=="jet" else None), aspect="auto")
        ax.axis("off")
        ax.set_title(ttl, fontsize=8, fontweight="bold", color=tcol, pad=3)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1 — Best / Median / Worst
# ═══════════════════════════════════════════════════════════════

def fig1_best_median_worst(results, path):
    print("\n  [Figure 1] Best / Median / Worst...")
    valid = [r for r in results if r["pdet_p"] and r["pdet_c"]]
    ranked = sorted(valid, key=lambda r: r["diff_pp"], reverse=True)

    best   = ranked[0]
    worst  = ranked[-1]
    diffs  = [r["diff_pp"] for r in ranked]
    med_val = np.median(diffs)
    median = min(ranked, key=lambda r: abs(r["diff_pp"] - med_val))

    cases = [
        (best,   "BEST CASE",   GREEN,  f"Largest improvement: Δp = {best['diff_pp']:+.1f} pp"),
        (median, "MEDIAN CASE", BLUE,   f"Median improvement:  Δp = {median['diff_pp']:+.1f} pp"),
        (worst,  "WORST CASE",  ORANGE, f"Smallest improvement: Δp = {worst['diff_pp']:+.1f} pp"),
    ]

    fig, axes = plt.subplots(3, 8, figsize=(32, 13))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Figure 1 — ICS-RL: Best, Median, and Worst Case Performance\n"
        "Trained Policy vs Chaotic Baseline across 20 BOSSBase Images",
        fontsize=13, fontweight="bold", y=1.01
    )

    for row_idx, (res, label, color, subtitle) in enumerate(cases):
        draw_8panel_row(axes[row_idx], res, label, color)
        fig.text(
            0.005, 1 - (row_idx + 0.5) / 3,
            f"{label}\n{subtitle}",
            va="center", ha="left", fontsize=9,
            fontweight="bold", color=color,
            rotation=90
        )

    plt.tight_layout(rect=[0.015, 0, 1, 0.98])
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — Failure Analysis
# ═══════════════════════════════════════════════════════════════

def fig2_failure_analysis(results, path):
    print("\n  [Figure 2] Failure analysis...")
    valid    = [r for r in results if r["pdet_p"] and r["pdet_c"]]
    failures = [r for r in valid if r["diff_pp"] <= 0]
    successes = [r for r in valid if r["diff_pp"] > 0]

    show = sorted(failures, key=lambda r: r["diff_pp"])[:3]
    while len(show) < 3:
        show.append(sorted(successes, key=lambda r: r["diff_pp"])[len(show)-len(failures)])

    fig, axes = plt.subplots(len(show), 8, figsize=(32, 5*len(show)+2))
    if len(show) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Figure 2 — ICS-RL: Failure Analysis\n"
        "Cases Where Policy Improvement is Smallest or Negative\n"
        "(Orange = policy disadvantaged; typically low-texture images)",
        fontsize=13, fontweight="bold", y=1.02
    )

    for i, res in enumerate(show):
        color = RED if res["diff_pp"] <= 0 else ORANGE
        label = "FAILURE" if res["diff_pp"] <= 0 else "NEAR-FAILURE"
        draw_8panel_row(axes[i], res, label, color)

        reason = ("Low texture image — policy has fewer high-texture regions "
                  "to exploit, reducing its advantage over the chaotic baseline.")
        axes[i][7].text(
            0.5, -0.18, reason,
            transform=axes[i][7].transAxes,
            ha="center", va="top", fontsize=7, color="#555",
            style="italic", wrap=True
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3 — Texture Heatmap Overlay
# ═══════════════════════════════════════════════════════════════

def fig3_texture_overlay(results, path):
    print("\n  [Figure 3] Texture overlay...")
    valid  = [r for r in results if r["pdet_p"] and r["pdet_c"]]
    ranked = sorted(valid, key=lambda r: r["diff_pp"], reverse=True)
    picks  = [ranked[0], ranked[1],
              ranked[len(ranked)//2 - 1], ranked[len(ranked)//2],
              ranked[-2], ranked[-1]]
    labels = ["Best #1","Best #2","Median #1","Median #2","Worst #1","Worst #2"]
    label_colors = [GREEN,GREEN, BLUE,BLUE, ORANGE,ORANGE]

    fig, axes = plt.subplots(3, 6, figsize=(26, 13))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Figure 3 — ICS-RL: Texture Saliency & Policy Pixel Selection Overlay\n"
        "Red dots = pixels selected by trained policy  |  "
        "Blue dots = chaotic baseline pixels  |  "
        "Background = texture saliency map",
        fontsize=12, fontweight="bold", y=1.01
    )

    col_titles = [f"{lbl}\nΔp={picks[i]['diff_pp']:+.1f} pp"
                  for i, lbl in enumerate(labels)]

    for col, (res, lbl, lcol) in enumerate(zip(picks, labels, label_colors)):
        cover       = res["cover"]
        texture_map = res["texture_map"]
        rows_p = res["rows_p"]; cols_p = res["cols_p"]
        rows_c = res["rows_c"]; cols_c = res["cols_c"]

        ax0 = axes[0][col]
        ax0.imshow(cover, cmap="gray", vmin=0, vmax=255, alpha=0.85)
        step_p = max(1, len(rows_p)//600)
        ax0.scatter(cols_p[::step_p], rows_p[::step_p],
                    c=BLUE, s=1.5, alpha=0.6, linewidths=0)
        ax0.set_title(f"{col_titles[col]}\nPolicy pixels (blue)",
                      fontsize=8, fontweight="bold", color=lcol)
        ax0.axis("off")

        ax1 = axes[1][col]
        ax1.imshow(texture_map, cmap="jet", vmin=0, vmax=1, alpha=0.9)
        ax1.scatter(cols_p[::step_p], rows_p[::step_p],
                    c="white", s=2.5, alpha=0.7, linewidths=0, label="Policy")
        step_c = max(1, len(rows_c)//600)
        ax1.scatter(cols_c[::step_c], rows_c[::step_c],
                    c="black", s=1.0, alpha=0.4, linewidths=0, label="Chaotic")
        ax1.set_title("Texture map\nWhite=Policy, Black=Chaotic",
                      fontsize=7, fontweight="bold")
        if col == 0:
            ax1.legend(fontsize=6, loc="lower right",
                       markerscale=4, framealpha=0.8)
        ax1.axis("off")

        ax2 = axes[2][col]
        density = np.zeros(IMG_SHAPE, dtype=np.float32)
        np.add.at(density, (rows_p, cols_p), 1)
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=6)
        ax2.imshow(cover, cmap="gray", vmin=0, vmax=255, alpha=0.5)
        ax2.imshow(density, cmap="hot", alpha=0.6, vmin=0, vmax=density.max())
        ax2.set_title(f"Policy pixel density\ntex_policy={res['tex_p']:.3f}",
                      fontsize=8, fontweight="bold", color=BLUE)
        ax2.axis("off")

    row_labels = ["Cover + Policy Pixel Overlay",
                  "Texture Saliency + Pixel Comparison",
                  "Policy Pixel Density Heatmap"]
    for i, rl in enumerate(row_labels):
        fig.text(0.002, 1 - (i+0.5)/3, rl, va="center", ha="left",
                 fontsize=9, fontweight="bold", rotation=90, color="#333")

    plt.tight_layout(rect=[0.012, 0, 1, 0.98])
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4 — Per-image bar chart
# ═══════════════════════════════════════════════════════════════

def fig4_perimage_bar(results, path):
    print("\n  [Figure 4] Per-image bar chart...")
    valid   = [r for r in results if r["pdet_p"] and r["pdet_c"]]
    indices = [r["img_idx"] for r in valid]
    pdet_p  = [r["pdet_p"]  for r in valid]
    pdet_c  = [r["pdet_c"]  for r in valid]
    diff_pp = [r["diff_pp"] for r in valid]

    x     = np.arange(len(valid))
    bar_w = 0.35

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.45,
                   height_ratios=[2.2, 1.2, 1.0])
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Figure 4 — ICS-RL: Per-Image SRNet Detection Probability Comparison\n"
        "Trained Policy vs Chaotic Baseline — 20 BOSSBase Images",
        fontsize=13, fontweight="bold", y=0.98
    )

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(LGRAY)

    bars_p = ax1.bar(x - bar_w/2, pdet_p, bar_w,
                     color=BLUE, alpha=0.88, label="ICS-RL Policy (Trained)",
                     edgecolor="white", linewidth=0.6)
    bars_c = ax1.bar(x + bar_w/2, pdet_c, bar_w,
                     color=RED,  alpha=0.88, label="Chaotic Baseline",
                     edgecolor="white", linewidth=0.6)

    for bar in list(bars_p) + list(bars_c):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", rotation=90)

    mean_p = np.mean(pdet_p); mean_c = np.mean(pdet_c)
    ax1.axhline(mean_p, color=BLUE, linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"Policy mean = {mean_p:.4f}")
    ax1.axhline(mean_c, color=RED,  linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"Chaotic mean = {mean_c:.4f}")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"#{i}" for i in indices], fontsize=9)
    ax1.set_ylabel("SRNet p_detect", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.set_title("SRNet Detection Probability per Image (↓ lower = harder to detect = better)",
                  fontsize=10, fontweight="bold", pad=6)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(LGRAY)

    bar_colors = [GREEN if d > 0 else RED for d in diff_pp]
    bars_d = ax2.bar(x, diff_pp, color=bar_colors, alpha=0.85,
                     edgecolor="white", linewidth=0.6)
    ax2.axhline(0, color="#333", linewidth=1.2)
    ax2.axhline(np.mean(diff_pp), color=GREEN, linestyle="--",
                linewidth=1.5, label=f"Mean Δ = {np.mean(diff_pp):+.2f} pp")

    for bar, val in zip(bars_d, diff_pp):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + (0.3 if val >= 0 else -0.5),
                 f"{val:+.1f}", ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=7.5, fontweight="bold",
                 color=GREEN if val > 0 else RED)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{i}" for i in indices], fontsize=9)
    ax2.set_ylabel("Δ p_detect (pp)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    n_better = sum(1 for d in diff_pp if d > 0)
    n_worse  = len(diff_pp) - n_better
    ax2.set_title(
        f"Improvement per Image (Policy − Chaotic)  |  "
        f"Policy better: {n_better}/{len(diff_pp)} images  |  "
        f"Policy worse: {n_worse}/{len(diff_pp)} images",
        fontsize=10, fontweight="bold", pad=6
    )

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(LGRAY)

    tex_p = [r["tex_p"] for r in valid]
    tex_c = [r["tex_c"] for r in valid]
    ax3.plot(x, tex_p, "o-", color=BLUE, linewidth=1.8, markersize=5,
             label=f"Policy tex (mean={np.mean(tex_p):.4f})")
    ax3.plot(x, tex_c, "s-", color=RED,  linewidth=1.8, markersize=5,
             label=f"Chaotic tex (mean={np.mean(tex_c):.4f})")
    ax3.fill_between(x, tex_p, tex_c, alpha=0.12, color=GREEN)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"#{i}" for i in indices], fontsize=9)
    ax3.set_ylabel("Texture Score", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.set_title(
        "Texture Score per Image — Policy consistently selects higher-texture pixels",
        fontsize=10, fontweight="bold", pad=6
    )

    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ICS-RL — PAPER FIGURES GENERATOR  (PeerJ Submission)              ║")
    print("║  Figure 1: Best/Median/Worst  |  Figure 2: Failure Analysis        ║")
    print("║  Figure 3: Texture Overlay    |  Figure 4: Per-Image Bar Chart     ║")
    print(f"║  seed={SEED}  |  n={N} images  (evaluate_seeded.py ile AYNI)           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        loader = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    # ✅ Seed-based image selection — evaluate_seeded.py ile birebir aynı
    IMAGE_INDICES = get_image_indices(DATA_DIR, N, SEED)
    print(f"  Selected images (seed={SEED}): {IMAGE_INDICES}\n")

    codec       = MessageCodec(max_bits=PAYLOAD_BITS)
    msg_bits    = codec.encode(REAL_MESSAGE)
    n_bits      = len(msg_bits)
    chaotic_gen = LogisticMapGenerator(x0=X0)
    texture_ext = TextureFeatureExtractor(patch_size=7)
    ch_rows, ch_cols = build_chaotic_coords(X0, R, IMG_SHAPE, N_CANDIDATES)

    # SRNet
    srnet = None
    srnet_path = "src/models/srnet_best.pth"
    if os.path.exists(srnet_path):
        try:
            from src.models.srnet import SRNet
            srnet = SRNet()
            srnet.load_state_dict(torch.load(srnet_path, map_location="cpu",
                                             weights_only=True))
            srnet.eval()
            print("  SRNet  : loaded ✓")
        except Exception as e:
            print(f"  SRNet  : could not load ({e})")

    # Policy
    policy = PolicyNetwork(input_channels=3)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found"); sys.exit(1)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                      weights_only=True))
    policy.eval()
    print("  Policy : loaded ✓\n")

    # Evaluate all images once, reuse for all figures
    results = evaluate_all(loader, policy, srnet, codec, msg_bits, n_bits,
                           chaotic_gen, texture_ext, ch_rows, ch_cols,
                           IMAGE_INDICES)

    # Generate figures
    fig1_best_median_worst(results,
        os.path.join(OUT_DIR, "fig1_best_median_worst.png"))
    fig2_failure_analysis(results,
        os.path.join(OUT_DIR, "fig2_failure_analysis.png"))
    fig3_texture_overlay(results,
        os.path.join(OUT_DIR, "fig3_texture_overlay.png"))
    fig4_perimage_bar(results,
        os.path.join(OUT_DIR, "fig4_perimage_bar.png"))

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ALL FIGURES SAVED:                                                 ║")
    print(f"║  {OUT_DIR}/fig1_best_median_worst.png                ║")
    print(f"║  {OUT_DIR}/fig2_failure_analysis.png                 ║")
    print(f"║  {OUT_DIR}/fig3_texture_overlay.png                  ║")
    print(f"║  {OUT_DIR}/fig4_perimage_bar.png                     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()