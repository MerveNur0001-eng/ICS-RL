"""
evaluate_seeded.py
──────────────────
ICS-RL: Seed-tabanlı rastgele görüntü seçimi ile değerlendirme.

Baseline Yöntemler:
  - Chaotic LSB  : Logistic map tabanlı naive LSB
  - S-UNIWARD    : Wavelet domain cost + STC embedding
  - HILL         : Higher-order residual tabanlı adaptif cost

Çıktılar:
  results/seeded/seed{SEED}_selected_images.csv
  results/seeded/seed{SEED}_per_image_results.csv
  results/seeded/seed{SEED}_summary.txt
  results/seeded/seed{SEED}_comparison_grid.png
  results/seeded/seed{SEED}_boxplot.png
  results/seeded/seed{SEED}_table.png
  results/seeded/seed{SEED}_baseline_radar.png

Kullanım:
    python evaluate_seeded.py
    python evaluate_seeded.py --seed 99 --n 20
"""

import argparse
import csv
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src.modules.policy_network    import PolicyNetwork
from src.modules.chaotic_generator import LogisticMapGenerator
from src.modules.texture_extractor import TextureFeatureExtractor
from utils.message_codec           import MessageCodec
from utils.bossbase_loader         import BOSSBaseLoader

from baselines import embed_hill_real, embed_suniward_real

# ═══════════════════════════════════════════════════════════════
#  VARSAYILAN AYARLAR
# ═══════════════════════════════════════════════════════════════
DEFAULT_SEED   = 42
DEFAULT_N      = 150
IMG_SHAPE      = (256, 256)
X0             = 0.123456
R              = 3.99
N_CANDIDATES   = 15_000
PAYLOAD_BITS   = int(256 * 256 * 0.4)
DATA_DIR       = "data/bossbase/"
MODEL_PATH     = "checkpoints/policy_final.pt"

REAL_MESSAGE = (
    "ICS-RL Steganography System - Intelligent Chaotic Steganography "
    "using Reinforcement Learning. Logistic map (x0=secret, r=3.99) "
    "selects candidate pixels; PPO agent embeds real payload via LSB. "
    "This hidden message demonstrates real steganographic embedding "
    "as in Tang et al. 2020 and Ogras 2019. Beykoz University CME6405."
)

BLUE   = "#1565c0"   # ICS-RL Policy
RED    = "#c62828"   # Chaotic LSB
GREEN  = "#2e7d32"   # HILL
ORANGE = "#e65100"   # S-UNIWARD
LGRAY  = "#f0f4f8"

METHOD_COLORS = {
    "policy":   BLUE,
    "chaotic":  RED,
    "hill":     GREEN,
    "suniward": ORANGE,
}
METHOD_LABELS = {
    "policy":   "ICS-RL (Ours)",
    "chaotic":  "Chaotic LSB",
    "hill":     "HILL",
    "suniward": "S-UNIWARD",
}

WHITE_RED  = LinearSegmentedColormap.from_list("white_red",  [(1,1,1),(0.85,0,0)])
WHITE_BLUE = LinearSegmentedColormap.from_list("white_blue", [(1,1,1),(0.08,0.35,0.72)])


# ═══════════════════════════════════════════════════════════════
#  ARG PARSE
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="ICS-RL seeded evaluation with baselines")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--n",    type=int, default=DEFAULT_N,
                   help=f"Number of images to evaluate (default: {DEFAULT_N})")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  SEED'Lİ GÖRÜNTÜ SEÇİMİ
# ═══════════════════════════════════════════════════════════════
def select_images_seeded(data_dir: str, n: int, seed: int):
    loader = BOSSBaseLoader(data_dir, img_shape=IMG_SHAPE)
    total  = loader.total_images()
    all_indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    return all_indices[:n]


# ═══════════════════════════════════════════════════════════════
#  METRİKLER
# ═══════════════════════════════════════════════════════════════
def compute_psnr(cover, stego):
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else 20*np.log10(255/np.sqrt(mse))

def compute_ssim(cover, stego):
    try:
        from skimage.metrics import structural_similarity as _ssim
        return float(_ssim(cover, stego, data_range=255))
    except ImportError:
        c1,c2 = (0.01*255)**2,(0.03*255)**2
        x,y   = cover.astype(np.float64), stego.astype(np.float64)
        mx,my = x.mean(), y.mean()
        cov   = np.mean((x-mx)*(y-my))
        return float(((2*mx*my+c1)*(2*cov+c2))/((mx**2+my**2+c1)*(x.var()+y.var()+c2)))

def srnet_detect(stego, srnet):
    if srnet is None:
        return None
    with torch.no_grad():
        inp = torch.from_numpy(stego).float().unsqueeze(0).unsqueeze(0)/255.0
        return float(torch.softmax(srnet(inp), dim=1)[0,1].item())


# ═══════════════════════════════════════════════════════════════
#  KAOTİK KOORDİNATLAR
# ═══════════════════════════════════════════════════════════════
def build_chaotic_coords(x0, r, img_shape, n_cand):
    h,w  = img_shape
    seq  = np.empty(h*w, dtype=np.float64)
    x = x0
    for i in range(h*w):
        x = r*x*(1-x); seq[i] = x
    order = np.argsort(seq)[:n_cand]
    return (order//w).astype(np.int32), (order%w).astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  BASELINE: S-UNIWARD + HILL
#  Kaynak: Daniel Lerch / stegolab (MIT License)
# ═══════════════════════════════════════════════════════════════

def _embed_suniward_wrapper(cover, payload_rate, tex_map, seed=42):
    stego = embed_suniward_real(cover, payload_rate=payload_rate, rng_seed=seed)
    diff  = np.abs(cover.astype(np.int16) - stego.astype(np.int16)) != 0
    rows, cols = np.where(diff)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    if len(rows) == 0:
        rows = np.array([0], dtype=np.int32)
        cols = np.array([0], dtype=np.int32)
    return stego, rows, cols


def _embed_hill_wrapper(cover, payload_rate, tex_map, seed=42):
    stego = embed_hill_real(cover, payload_rate=payload_rate, rng_seed=seed)
    diff  = np.abs(cover.astype(np.int16) - stego.astype(np.int16)) != 0
    rows, cols = np.where(diff)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    if len(rows) == 0:
        rows = np.array([0], dtype=np.int32)
        cols = np.array([0], dtype=np.int32)
    return stego, rows, cols


# ═══════════════════════════════════════════════════════════════
#  PİKSEL SEÇİMİ
# ═══════════════════════════════════════════════════════════════
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
#  TEK GÖRÜNTÜ DEĞERLENDİRME — 4 YÖNTEM
# ═══════════════════════════════════════════════════════════════
def evaluate_one(img_idx, cover, policy, srnet,
                 codec, msg_bits, n_bits,
                 chaotic_gen, texture_ext, ch_rows, ch_cols):
    mask    = chaotic_gen.get_mask(IMG_SHAPE, N_CANDIDATES)
    tex_map = texture_ext.get_texture_saliency_map(cover)

    # 1. ICS-RL Policy
    rp, cp = select_policy(policy, cover, mask, tex_map, ch_rows, ch_cols, n_bits)
    stego_p = codec.embed_into_pixels(cover, msg_bits, rp, cp)

    # 2. Chaotic LSB
    rc, cc  = select_chaotic(ch_rows, ch_cols, n_bits)
    stego_c = codec.embed_into_pixels(cover, msg_bits, rc, cc)

    # 3. S-UNIWARD
    payload_rate = n_bits / (IMG_SHAPE[0] * IMG_SHAPE[1])
    stego_su, rsu, csu = _embed_suniward_wrapper(cover, payload_rate, tex_map, seed=img_idx)

    # 4. HILL
    stego_h, rh, ch_ = _embed_hill_wrapper(cover, payload_rate, tex_map, seed=img_idx)

    def metrics(stego, rows, cols):
        psnr = compute_psnr(cover, stego)
        ssim = compute_ssim(cover, stego)
        pdet = srnet_detect(stego, srnet)
        tex  = float(tex_map[rows, cols].mean())
        orig = codec.decode(msg_bits)
        ext  = codec.decode(codec.extract_from_pixels(stego, rows, cols, n_bits))
        acc  = sum(a==b for a,b in zip(ext, orig)) / max(len(orig),1) * 100
        n_ch = int((stego.astype(np.int16) - cover.astype(np.int16) != 0).sum())
        return dict(psnr=psnr, ssim=ssim, pdet=pdet, tex=tex, acc=acc, n_ch=n_ch)

    m_p  = metrics(stego_p,  rp,  cp)
    m_c  = metrics(stego_c,  rc,  cc)
    m_su = metrics(stego_su, rsu, csu)
    m_h  = metrics(stego_h,  rh,  ch_)

    return dict(
        img_idx=img_idx, cover=cover, tex_map=tex_map, chaotic_mask=mask,
        stego_p=stego_p, rows_p=rp, cols_p=cp,
        **{f"p_{k}":  v for k, v in m_p.items()},
        stego_c=stego_c, rows_c=rc, cols_c=cc,
        **{f"c_{k}":  v for k, v in m_c.items()},
        stego_su=stego_su, rows_su=rsu, cols_su=csu,
        **{f"su_{k}": v for k, v in m_su.items()},
        stego_h=stego_h, rows_h=rh, cols_h=ch_,
        **{f"h_{k}":  v for k, v in m_h.items()},
    )


# ═══════════════════════════════════════════════════════════════
#  CSV KAYDET
# ═══════════════════════════════════════════════════════════════
def save_results_csv(results, path):
    methods = ["p", "c", "su", "h"]
    fields  = ["img_idx"]
    for m in methods:
        fields += [f"{m}_psnr", f"{m}_ssim", f"{m}_tex", f"{m}_pdet", f"{m}_acc", f"{m}_n_ch"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {"img_idx": r["img_idx"]}
            for m in methods:
                row[f"{m}_psnr"] = f"{r[f'{m}_psnr']:.4f}"
                row[f"{m}_ssim"] = f"{r[f'{m}_ssim']:.6f}"
                row[f"{m}_tex"]  = f"{r[f'{m}_tex']:.4f}"
                row[f"{m}_pdet"] = f"{r[f'{m}_pdet']:.4f}" if r[f"{m}_pdet"] else "N/A"
                row[f"{m}_acc"]  = f"{r[f'{m}_acc']:.2f}"
                row[f"{m}_n_ch"] = r[f"{m}_n_ch"]
            w.writerow(row)
    print(f"  [CSV]  → {path}")


# ═══════════════════════════════════════════════════════════════
#  SEÇİLEN GÖRÜNTÜLER CSV
# ═══════════════════════════════════════════════════════════════
def save_selected_csv(indices, seed, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "img_idx", "filename", "seed"])
        for i, idx in enumerate(indices, 1):
            w.writerow([i, idx, f"{idx:04d}.pgm", seed])


# ═══════════════════════════════════════════════════════════════
#  ÖZET + T-TEST
# ═══════════════════════════════════════════════════════════════
def compute_summary_and_tests(results):
    methods = ["p", "c", "su", "h"]
    summary = {}
    for m in methods:
        for k in ["psnr", "ssim", "tex", "pdet"]:
            key = f"{m}_{k}"
            a = np.array([r[key] for r in results if r[key] is not None], dtype=np.float64)
            if len(a):
                summary[key] = dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(), n=len(a))

    tests = {}
    for baseline in ["c", "su", "h"]:
        bname = {"c":"Chaotic","su":"S-UNIWARD","h":"HILL"}[baseline]
        for metric in ["pdet", "psnr", "ssim", "tex"]:
            pairs = []
            for r in results:
                pv = r.get(f"p_{metric}")
                bv = r.get(f"{baseline}_{metric}")
                if pv is not None and bv is not None:
                    pairs.append((pv, bv))
            if len(pairs) >= 2:
                pa = np.array([p[0] for p in pairs])
                ba = np.array([p[1] for p in pairs])
                t, p = stats.ttest_rel(pa, ba)
                key = f"ICSvs{bname}_{metric}"
                tests[key] = dict(
                    t=t, p=p,
                    better="lower" if metric == "pdet" else "higher",
                    diff_mean=(pa - ba).mean(),
                    sig=p < 0.05,
                    baseline=bname, metric=metric, n=len(pairs)
                )
    return summary, tests


def save_summary_txt(summary, tests, n_images, seed, path):
    W = 80
    lines = [
        "="*W,
        "ICS-RL — SEED-BASED MULTI-IMAGE EVALUATION".center(W),
        f"seed={seed}  |  n={n_images} randomly selected BOSSBase images".center(W),
        "="*W, "",
        "METRIC SUMMARY (mean ± std)".center(W),
        "-"*W,
        f"{'Metric':<22} {'ICS-RL(Ours)':>17} {'Chaotic':>15} {'S-UNIWARD':>15} {'HILL':>15}",
        "-"*W,
    ]

    def s(key, d=4):
        if key not in summary: return "   N/A   "
        v = summary[key]
        return f"{v['mean']:.{d}f}±{v['std']:.{d}f}"

    for lbl, metric, d in [
        ("SRNet p_detect ↓", "pdet", 4),
        ("PSNR (dB) ↑",      "psnr", 2),
        ("SSIM ↑",           "ssim", 4),
        ("Texture ↑",        "tex",  4),
    ]:
        lines.append(
            f"  {lbl:<20} {s(f'p_{metric}',d):>17} {s(f'c_{metric}',d):>15} "
            f"{s(f'su_{metric}',d):>15} {s(f'h_{metric}',d):>15}"
        )

    lines += ["", "PAIRED T-TEST: ICS-RL vs EACH BASELINE".center(W), "-"*W]
    for baseline in ["Chaotic", "S-UNIWARD", "HILL"]:
        key = f"ICSvs{baseline}_pdet"
        if key in tests:
            t = tests[key]
            sig = "SIGNIFICANT ✓" if t["sig"] else "n.s."
            better_lbl = "ICS-RL better" if (t["diff_mean"] < 0) else f"{baseline} better"
            lines.append(f"  vs {baseline:<12} t={t['t']:+.3f}  p={t['p']:.4f}  ({sig})  [{better_lbl}]")

    lines += ["", "="*W]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [TXT]  → {path}")
    print("\n  " + "\n  ".join(lines))


# ═══════════════════════════════════════════════════════════════
#  KARŞILAŞTIRMA GRİD — 4 yöntem (5 sütun: Cover + 4)
# ═══════════════════════════════════════════════════════════════
def draw_comparison_grid(results, seed, path):
    n    = len(results)
    COLS = 5  # Cover, ICS-RL, Chaotic, S-UNIWARD, HILL
    ROWS = n

    fig = plt.figure(figsize=(COLS * 3.2, ROWS * 3.6))
    fig.patch.set_facecolor("#fafafa")

    col_titles = ["Cover", "ICS-RL\n(Ours)", "Chaotic\nLSB", "S-UNIWARD", "HILL"]
    col_colors = ["black", BLUE, RED, ORANGE, GREEN]

    outer = gridspec.GridSpec(ROWS, COLS, figure=fig, hspace=0.08, wspace=0.05)

    for row_i, res in enumerate(results):
        images = [res["cover"], res["stego_p"], res["stego_c"], res["stego_su"], res["stego_h"]]
        psnrs  = [None, res["p_psnr"], res["c_psnr"], res["su_psnr"], res["h_psnr"]]
        pdets  = [None, res["p_pdet"], res["c_pdet"], res["su_pdet"], res["h_pdet"]]

        for col_i, (img, psnr, pdet) in enumerate(zip(images, psnrs, pdets)):
            ax = fig.add_subplot(outer[row_i, col_i])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=8, fontweight="bold",
                             color=col_colors[col_i], pad=3)
            if col_i == 0:
                ax.set_ylabel(f"#{res['img_idx']}", fontsize=7,
                              rotation=0, labelpad=28, va="center")
            if psnr is not None:
                pdet_str = f"pd={pdet:.2f}" if pdet else ""
                ax.set_xlabel(f"PSNR={psnr:.1f}\n{pdet_str}",
                              fontsize=6, color=col_colors[col_i], labelpad=1)

    fig.suptitle(
        f"ICS-RL vs Baselines — seed={seed}, {n} BOSSBase images\n"
        f"Columns: Cover | ICS-RL (Ours) | Chaotic LSB | S-UNIWARD | HILL",
        fontsize=10, fontweight="bold", y=1.003
    )
    plt.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"  [GRID] → {path}")


# ═══════════════════════════════════════════════════════════════
#  KUTU GRAFİK — 4 YÖNTEM
# ═══════════════════════════════════════════════════════════════
def save_boxplot(results, seed, path):
    methods     = ["p", "c", "su", "h"]
    labels      = ["ICS-RL\n(Ours)", "Chaotic\nLSB", "S-UNIWARD", "HILL"]
    colors      = [BLUE, RED, ORANGE, GREEN]
    metrics_cfg = [
        ("SRNet p_detect\n(↓ better)", "pdet"),
        ("PSNR (dB)\n(↑ better)",      "psnr"),
        ("SSIM\n(↑ better)",           "ssim"),
        ("Texture Score\n(↑ better)",  "tex"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"ICS-RL vs Baselines: S-UNIWARD, HILL, Chaotic LSB\n"
        f"seed={seed}, n={len(results)} randomly selected BOSSBase images",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, (title, metric) in zip(axes, metrics_cfg):
        ax.set_facecolor(LGRAY)
        data = []
        for m in methods:
            vals = np.array([r[f"{m}_{metric}"] for r in results
                             if r[f"{m}_{metric}"] is not None], dtype=np.float64)
            data.append(vals)

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5), widths=0.45)
        for box, col in zip(bp["boxes"], colors):
            box.set_facecolor(col); box.set_alpha(0.82)

        for xi, (vals, col) in enumerate(zip(data, colors), 1):
            ax.scatter(np.random.normal(xi, 0.06, len(vals)),
                       vals, alpha=0.50, s=22, color=col, zorder=3)
            ax.plot(xi, vals.mean(), "D", color="white", markersize=7,
                    markeredgecolor=col, markeredgewidth=1.5, zorder=5)
            ax.annotate(f"μ={vals.mean():.4f}", xy=(xi, vals.mean()),
                        xytext=(xi+0.12, vals.mean()), fontsize=7, color=col,
                        fontweight="bold", arrowprops=dict(arrowstyle="-", color=col, lw=0.7))

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [BOX]  → {path}")


# ═══════════════════════════════════════════════════════════════
#  TABLO FİGÜRÜ — 4 YÖNTEM
# ═══════════════════════════════════════════════════════════════
def save_table_figure(summary, tests, n_images, seed, path):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white"); ax.axis("off")

    def s(key, d=4):
        if key not in summary: return "N/A"
        v = summary[key]
        return f"{v['mean']:.{d}f}±{v['std']:.{d}f}"

    def pstr(baseline, metric):
        bname = {"c":"Chaotic","su":"S-UNIWARD","h":"HILL"}[baseline]
        key   = f"ICSvs{bname}_{metric}"
        if key not in tests: return "—"
        p = tests[key]["p"]
        return ("p<0.001" if p<0.001 else "p<0.01" if p<0.01
                else f"p={p:.3f}*" if p<0.05 else f"p={p:.3f}")

    def best(metric, better="lower"):
        candidates = ["p","c","su","h"]
        vals = {m: summary[f"{m}_{metric}"]["mean"] for m in candidates
                if f"{m}_{metric}" in summary}
        if not vals: return "N/A"
        fn = min if better=="lower" else max
        best_val = fn(vals.values())
        if abs(vals.get("p", float("inf") if better=="lower" else float("-inf")) - best_val) < 1e-6:
            return "ICS-RL (Ours)"
        best_m = fn(vals, key=vals.get)
        return METHOD_LABELS.get(
            {"p":"policy","c":"chaotic","su":"suniward","h":"hill"}[best_m], best_m)

    col_headers = [
        "Metric", "ICS-RL\n(Ours)", "Chaotic LSB",
        "S-UNIWARD", "HILL", "Best Method",
        "Paired t (vs ICS-RL)"
    ]

    data = [
        ["SRNet p_detect ↓",
         s("p_pdet"), s("c_pdet"), s("su_pdet"), s("h_pdet"),
         best("pdet", "lower"),
         " / ".join([pstr(b,"pdet") for b in ["c","su","h"]])],
        ["PSNR (dB) ↑",
         s("p_psnr",2), s("c_psnr",2), s("su_psnr",2), s("h_psnr",2),
         best("psnr", "higher"),
         " / ".join([pstr(b,"psnr") for b in ["c","su","h"]])],
        ["SSIM ↑",
         s("p_ssim"), s("c_ssim"), s("su_ssim"), s("h_ssim"),
         best("ssim", "higher"),
         " / ".join([pstr(b,"ssim") for b in ["c","su","h"]])],
        ["Texture ↑",
         s("p_tex"), s("c_tex"), s("su_tex"), s("h_tex"),
         best("tex", "higher"),
         " / ".join([pstr(b,"tex") for b in ["c","su","h"]])],
    ]

    row_colors = [
        ["#e3f2fd"] * 7,
        ["#f1f8e9"] * 7,
        ["#f1f8e9"] * 7,
        ["#fff8e1"] * 7,
    ]

    tbl = ax.table(cellText=data, colLabels=col_headers,
                   cellColours=row_colors,
                   colColours=["#1565c0"] * 7,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 2.8)
    for col_idx in range(7):
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        f"Table 1: ICS-RL vs Baselines (S-UNIWARD, HILL, Chaotic LSB)\n"
        f"seed={seed}, n={n_images} randomly selected BOSSBase images  |  "
        f"Primary metric: SRNet p_detect ↓  |  t-test columns: Chaotic / S-UNIWARD / HILL",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [TBL]  → {path}")


# ═══════════════════════════════════════════════════════════════
#  RADAR CHART — 4 YÖNTEM
# ═══════════════════════════════════════════════════════════════
def save_radar_chart(summary, seed, n_images, path):
    methods    = ["p", "c", "su", "h"]
    labels     = ["ICS-RL\n(Ours)", "Chaotic\nLSB", "S-UNIWARD", "HILL"]
    colors     = [BLUE, RED, ORANGE, GREEN]
    categories = ["Undetectability\n(1-p_det)", "PSNR", "SSIM", "Texture"]
    N = len(categories)

    raw = {}
    for m in methods:
        raw[m] = [
            1.0 - summary.get(f"{m}_pdet", {}).get("mean", 0.5),
            summary.get(f"{m}_psnr", {}).get("mean", 30.0),
            summary.get(f"{m}_ssim", {}).get("mean", 0.9),
            summary.get(f"{m}_tex",  {}).get("mean", 0.5),
        ]

    all_vals = np.array(list(raw.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    normed = {}
    for m in methods:
        v = np.array(raw[m])
        normed[m] = (v - mins) / (maxs - mins + 1e-10)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    for m, label, color in zip(methods, labels, colors):
        vals = normed[m].tolist() + normed[m].tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=8, color="gray")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Radar: ICS-RL vs Baselines\n"
        f"seed={seed}, n={n_images} — normalized [0→1], higher=better",
        fontsize=11, fontweight="bold", pad=18
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [RADAR]→ {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    args    = parse_args()
    SEED    = args.seed
    N       = args.n
    OUT_DIR = "results/seeded"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  ICS-RL — SEED-BASED EVALUATION WITH BASELINES                         ║")
    print(f"║  seed={SEED}  |  n={N} images  |  methods: ICS-RL, Chaotic, S-UNIWARD, HILL    ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝\n")

    try:
        loader = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    try:
        image_indices = select_images_seeded(DATA_DIR, N, SEED)
    except AttributeError:
        import glob
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pgm")) +
                       glob.glob(os.path.join(DATA_DIR, "*.png")) +
                       glob.glob(os.path.join(DATA_DIR, "*.jpg")))
        total = len(files)
        if total == 0:
            print(f"ERROR: {DATA_DIR} içinde görüntü bulunamadı"); sys.exit(1)
        rng = random.Random(SEED)
        all_idx = list(range(total))
        rng.shuffle(all_idx)
        image_indices = sorted(all_idx[:N])

    print(f"  Seçilen indeksler: {image_indices}\n")

    codec       = MessageCodec(max_bits=PAYLOAD_BITS)
    msg_bits    = codec.encode(REAL_MESSAGE)
    n_bits      = len(msg_bits)
    chaotic_gen = LogisticMapGenerator(x0=X0)
    texture_ext = TextureFeatureExtractor(patch_size=7)
    ch_rows, ch_cols = build_chaotic_coords(X0, R, IMG_SHAPE, N_CANDIDATES)

    srnet = None
    for srnet_path in ["src/models/srnet_best.pth", "checkpoints/srnet_best.pth"]:
        if os.path.exists(srnet_path):
            try:
                from src.models.srnet import SRNet
                srnet = SRNet()
                srnet.load_state_dict(torch.load(srnet_path, map_location="cpu", weights_only=True))
                srnet.eval()
                print(f"  SRNet    : loaded ✓  ({srnet_path})")
            except Exception as e:
                print(f"  SRNet    : yüklenemedi ({e})")
            break
    if srnet is None:
        print("  SRNet    : bulunamadı — p_detect N/A olacak")

    policy = PolicyNetwork(input_channels=3)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} bulunamadı"); sys.exit(1)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    policy.eval()
    print(f"  Policy   : loaded ✓  ({MODEL_PATH})\n")

    results = []
    t0 = time.time()
    header = (f"  {'Img':>4}  {'ICS-RL':>7}  {'Chaotic':>7}  {'S-UNI':>7}  {'HILL':>7}  "
              f"{'PSNR_p':>7}  {'tex_p':>6}  Best")
    print(header)
    print("  " + "─" * (len(header)-2))

    for i, img_idx in enumerate(image_indices):
        cover, _ = loader.get(img_idx)
        res = evaluate_one(
            img_idx, cover, policy, srnet,
            codec, msg_bits, n_bits,
            chaotic_gen, texture_ext, ch_rows, ch_cols
        )
        results.append(res)

        pdet_vals = {
            "P":  res["p_pdet"],
            "C":  res["c_pdet"],
            "SU": res["su_pdet"],
            "H":  res["h_pdet"],
        }
        valid = {k: v for k, v in pdet_vals.items() if v is not None}
        best  = min(valid, key=valid.get) if valid else "?"

        def pfmt(v): return f"{v:.4f}" if v is not None else " N/A  "
        print(
            f"  [{i+1:2d}] #{img_idx:<5} "
            f"{pfmt(res['p_pdet'])}  {pfmt(res['c_pdet'])}  "
            f"{pfmt(res['su_pdet'])}  {pfmt(res['h_pdet'])}  "
            f"{res['p_psnr']:>6.2f}  {res['p_tex']:>5.3f}  [{best}]"
        )

    elapsed = time.time() - t0
    print(f"\n  Tamamlandı — {elapsed:.1f}s\n")

    pf = lambda name: os.path.join(OUT_DIR, f"seed{SEED}_{name}")

    save_selected_csv(image_indices, SEED, pf("selected_images.csv"))
    save_results_csv(results, pf("per_image_results.csv"))

    summary, tests = compute_summary_and_tests(results)
    save_summary_txt(summary, tests, N, SEED, pf("summary.txt"))
    draw_comparison_grid(results, SEED, pf("comparison_grid.png"))
    save_boxplot(results, SEED, pf("boxplot.png"))
    save_table_figure(summary, tests, N, SEED, pf("table.png"))
    save_radar_chart(summary, SEED, N, pf("baseline_radar.png"))

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  ÇIKTILAR:                                                              ║")
    for fname in [
        f"seed{SEED}_selected_images.csv",
        f"seed{SEED}_per_image_results.csv",
        f"seed{SEED}_summary.txt",
        f"seed{SEED}_comparison_grid.png",
        f"seed{SEED}_boxplot.png",
        f"seed{SEED}_table.png",
        f"seed{SEED}_baseline_radar.png",
    ]:
        full = os.path.join(OUT_DIR, fname)
        print(f"║  {full:<72}║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Tekrar çalıştır: python evaluate_seeded.py --seed {SEED} --n {N}")


if __name__ == "__main__":
    main()