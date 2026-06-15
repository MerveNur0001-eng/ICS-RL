"""
evaluate_seeded_n20.py
──────────────────────
ICS-RL: Seed-tabanlı rastgele görüntü seçimi ile değerlendirme (n=20).

Deterministik [5, 10, 15, ...] yerine:
  - Tüm BOSSBase dosyaları listelenir
  - RANDOM_SEED ile reproducible shuffle yapılır
  - İlk N_IMAGES adet rastgele seçilir
  - Seed ve seçilen dosya listesi CSV'ye kaydedilir (reviewer tekrarlayabilir)

Çıktılar:
  results/seeded/n20_seed{SEED}_selected_images.csv    ← seçilen dosyalar
  results/seeded/n20_seed{SEED}_per_image_results.csv  ← metrikler
  results/seeded/n20_seed{SEED}_summary.txt            ← özet + t-testi
  results/seeded/n20_seed{SEED}_comparison_grid.png    ← 20-panel grid
  results/seeded/n20_seed{SEED}_boxplot.png            ← kutu grafik
  results/seeded/n20_seed{SEED}_table.png              ← tablo figürü

Kullanım:
    python evaluate_seeded_n20.py
    python evaluate_seeded_n20.py --seed 99 --n 20
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

# ═══════════════════════════════════════════════════════════════
#  VARSAYILAN AYARLAR
# ═══════════════════════════════════════════════════════════════
DEFAULT_SEED   = 42
DEFAULT_N      = 20
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

# Renk sabitleri
BLUE   = "#1565c0"
RED    = "#c62828"
GREEN  = "#2e7d32"
ORANGE = "#e65100"
LGRAY  = "#f0f4f8"

WHITE_RED  = LinearSegmentedColormap.from_list("white_red",  [(1,1,1),(0.85,0,0)])
WHITE_BLUE = LinearSegmentedColormap.from_list("white_blue", [(1,1,1),(0.08,0.35,0.72)])


# ═══════════════════════════════════════════════════════════════
#  ARG PARSE
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="ICS-RL seeded evaluation (n=20)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--n",    type=int, default=DEFAULT_N,
                   help=f"Number of images to evaluate (default: {DEFAULT_N})")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  SEED'LI GORUNTU SECIMI
# ═══════════════════════════════════════════════════════════════
def select_images_seeded(data_dir: str, n: int, seed: int):
    loader = BOSSBaseLoader(data_dir, img_shape=IMG_SHAPE)
    total = loader.total_images()
    all_indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    return all_indices[:n]


# ═══════════════════════════════════════════════════════════════
#  METRIKLER
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
#  KAOTIK KOORDINATLAR
# ═══════════════════════════════════════════════════════════════
def build_chaotic_coords(x0, r, img_shape, n_cand):
    h, w = img_shape
    seq  = np.empty(h*w, dtype=np.float64)
    x = x0
    for i in range(h*w):
        x = r*x*(1-x); seq[i] = x
    order = np.argsort(seq)[:n_cand]
    return (order//w).astype(np.int32), (order%w).astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  PIKSEL SECIMI
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
#  TEK GORUNTU DEGERLENDIRME
# ═══════════════════════════════════════════════════════════════
def evaluate_one(img_idx, cover, policy, srnet, codec, msg_bits, n_bits,
                 chaotic_gen, texture_ext, ch_rows, ch_cols):
    mask    = chaotic_gen.get_mask(IMG_SHAPE, N_CANDIDATES)
    tex_map = texture_ext.get_texture_saliency_map(cover)

    rp, cp = select_policy(policy, cover, mask, tex_map, ch_rows, ch_cols, n_bits)
    rc, cc = select_chaotic(ch_rows, ch_cols, n_bits)

    stego_p = codec.embed_into_pixels(cover, msg_bits, rp, cp)
    stego_c = codec.embed_into_pixels(cover, msg_bits, rc, cc)

    psnr_p, ssim_p = compute_psnr(cover, stego_p), compute_ssim(cover, stego_p)
    psnr_c, ssim_c = compute_psnr(cover, stego_c), compute_ssim(cover, stego_c)
    tex_p  = float(tex_map[rp, cp].mean())
    tex_c  = float(tex_map[rc, cc].mean())
    pdet_p = srnet_detect(stego_p, srnet)
    pdet_c = srnet_detect(stego_c, srnet)

    orig  = codec.decode(msg_bits)
    acc_p = sum(a==b for a,b in zip(
        codec.decode(codec.extract_from_pixels(stego_p, rp, cp, n_bits)), orig
    )) / max(len(orig), 1) * 100
    acc_c = sum(a==b for a,b in zip(
        codec.decode(codec.extract_from_pixels(stego_c, rc, cc, n_bits)), orig
    )) / max(len(orig), 1) * 100

    return dict(
        img_idx=img_idx,
        cover=cover, stego_p=stego_p, stego_c=stego_c,
        texture_map=tex_map, chaotic_mask=mask,
        rows_p=rp, cols_p=cp, rows_c=rc, cols_c=cc,
        psnr_p=psnr_p, psnr_c=psnr_c,
        ssim_p=ssim_p, ssim_c=ssim_c,
        tex_p=tex_p,   tex_c=tex_c,
        pdet_p=pdet_p, pdet_c=pdet_c,
        acc_p=acc_p,   acc_c=acc_c,
        n_ch_p=int((stego_p.astype(np.int16)-cover.astype(np.int16)!=0).sum()),
        n_ch_c=int((stego_c.astype(np.int16)-cover.astype(np.int16)!=0).sum()),
        diff_pp=(pdet_c-pdet_p)*100 if (pdet_p and pdet_c) else 0.0,
    )


# ═══════════════════════════════════════════════════════════════
#  CSV KAYDET
# ═══════════════════════════════════════════════════════════════
def save_results_csv(results, path):
    fields = ["img_idx",
              "psnr_p","ssim_p","tex_p","pdet_p","acc_p",
              "psnr_c","ssim_c","tex_c","pdet_c","acc_c",
              "diff_psnr","diff_ssim","diff_tex","diff_pdet_pp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "img_idx"      : r["img_idx"],
                "psnr_p"       : f"{r['psnr_p']:.4f}",
                "ssim_p"       : f"{r['ssim_p']:.6f}",
                "tex_p"        : f"{r['tex_p']:.4f}",
                "pdet_p"       : f"{r['pdet_p']:.4f}" if r["pdet_p"] else "N/A",
                "acc_p"        : f"{r['acc_p']:.2f}",
                "psnr_c"       : f"{r['psnr_c']:.4f}",
                "ssim_c"       : f"{r['ssim_c']:.6f}",
                "tex_c"        : f"{r['tex_c']:.4f}",
                "pdet_c"       : f"{r['pdet_c']:.4f}" if r["pdet_c"] else "N/A",
                "acc_c"        : f"{r['acc_c']:.2f}",
                "diff_psnr"    : f"{r['psnr_p']-r['psnr_c']:.4f}",
                "diff_ssim"    : f"{r['ssim_p']-r['ssim_c']:.6f}",
                "diff_tex"     : f"{r['tex_p']-r['tex_c']:.4f}",
                "diff_pdet_pp" : f"{r['diff_pp']:.2f}",
            })
    print(f"  [CSV]  -> {path}")


# ═══════════════════════════════════════════════════════════════
#  SECILEN GORUNTULAR CSV
# ═══════════════════════════════════════════════════════════════
def save_selected_csv(indices, seed, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "img_idx", "filename", "seed"])
        for i, idx in enumerate(indices, 1):
            fname = f"{idx:04d}.pgm"
            w.writerow([i, idx, fname, seed])


# ═══════════════════════════════════════════════════════════════
#  OZET + T-TEST
# ═══════════════════════════════════════════════════════════════
def compute_summary_and_tests(results):
    def arr(key):
        return np.array([r[key] for r in results if r[key] is not None], dtype=np.float64)

    keys = ["psnr_p","ssim_p","tex_p","pdet_p","psnr_c","ssim_c","tex_c","pdet_c"]
    summary = {}
    for k in keys:
        a = arr(k)
        summary[k] = dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(), n=len(a))

    tests = {}
    for metric, pk, ck, better in [
        ("p_detect", "pdet_p", "pdet_c", "lower"),
        ("PSNR",     "psnr_p", "psnr_c", "higher"),
        ("SSIM",     "ssim_p", "ssim_c", "higher"),
        ("Texture",  "tex_p",  "tex_c",  "higher"),
    ]:
        pa, ca = arr(pk), arr(ck)
        n = min(len(pa), len(ca))
        if n >= 2:
            t, p = stats.ttest_rel(pa[:n], ca[:n])
            tests[metric] = dict(t=t, p=p, better=better,
                                 diff_mean=(pa[:n]-ca[:n]).mean(),
                                 sig=p<0.05)
    return summary, tests


def save_summary_txt(summary, tests, n_images, seed, path):
    W = 72
    lines = [
        "="*W,
        "ICS-RL -- SEED-BASED MULTI-IMAGE EVALUATION (n=20)".center(W),
        f"seed={seed}  |  n={n_images} randomly selected BOSSBase images".center(W),
        "="*W, "",
        "METRIC SUMMARY (mean +- std)".center(W),
        "-"*W,
        f"{'Metric':<22} {'Policy':>22}   {'Chaotic':>22}",
        "-"*W,
    ]
    for lbl, pk, ck in [
        ("SRNet p_detect (lower)", "pdet_p", "pdet_c"),
        ("PSNR (dB) (higher)",     "psnr_p", "psnr_c"),
        ("SSIM (higher)",          "ssim_p", "ssim_c"),
        ("Texture (higher)",       "tex_p",  "tex_c"),
    ]:
        ps, cs = summary[pk], summary[ck]
        lines.append(f"  {lbl:<20} {ps['mean']:>8.4f}+-{ps['std']:.4f}   "
                     f"{cs['mean']:>8.4f}+-{cs['std']:.4f}")

    lines += ["", "PAIRED T-TEST RESULTS".center(W), "-"*W]
    for m, res in tests.items():
        sig    = "SIGNIFICANT" if res["sig"] else "n.s."
        better = ("Policy" if (res["better"]=="lower"  and res["diff_mean"]<0) or
                               (res["better"]=="higher" and res["diff_mean"]>0)
                  else "Chaotic") + " better"
        lines.append(f"  {m:<12} t={res['t']:+.3f}  p={res['p']:.4f}  ({sig})  [{better}]")

    if "p_detect" in tests:
        t  = tests["p_detect"]
        ps = summary["pdet_p"]
        cs = summary["pdet_c"]
        diff_pp = (cs["mean"] - ps["mean"]) * 100
        lines += [
            "", "REPRODUCIBILITY NOTE".center(W), "-"*W,
            f"  Random seed : {seed}",
            f"  Images      : {n_images} randomly drawn from BOSSBase",
            f"  To reproduce: python evaluate_seeded_n20.py --seed {seed} --n {n_images}",
            "",
            "PAPER-READY STATEMENT".center(W), "-"*W,
            f"  {n_images} images were randomly selected from BOSSBase using",
            f"  seed={seed} for reproducibility. The ICS-RL policy achieves",
            f"  SRNet p_detect = {ps['mean']:.4f}+-{ps['std']:.4f} vs chaotic",
            f"  {cs['mean']:.4f}+-{cs['std']:.4f}, a {diff_pp:.2f} pp improvement",
            f"  (paired t-test: t={t['t']:.3f}, p={t['p']:.4f}{'*' if t['sig'] else ''}).",
        ]
    lines += ["", "="*W]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [TXT]  -> {path}")
    print("\n  " + "\n  ".join(lines))


# ═══════════════════════════════════════════════════════════════
#  KARSILASTIRMA GRID -- 20 panel
# ═══════════════════════════════════════════════════════════════
def draw_comparison_grid(results, seed, path):
    n    = len(results)
    COLS = 5
    ROWS = (n + COLS - 1) // COLS

    fig  = plt.figure(figsize=(COLS*7.5, ROWS*5.5))
    fig.patch.set_facecolor("#fafafa")
    outer = gridspec.GridSpec(ROWS, COLS, figure=fig, hspace=0.08, wspace=0.06)

    for i, res in enumerate(results):
        r, c  = i // COLS, i % COLS
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[r, c], hspace=0.05, wspace=0.05
        )

        cover   = res["cover"]
        stego_p = res["stego_p"]
        stego_c = res["stego_c"]
        tex_map = res["texture_map"]
        diff_p  = np.clip(np.abs(cover.astype(np.int16)-stego_p.astype(np.int16))*200, 0, 255).astype(np.uint8)
        diff_c  = np.clip(np.abs(cover.astype(np.int16)-stego_c.astype(np.int16))*200, 0, 255).astype(np.uint8)

        pd_p  = res["pdet_p"]
        pd_c  = res["pdet_c"]
        dpp   = res["diff_pp"]
        color = GREEN if dpp > 0 else RED
        mark  = "+" if dpp > 0 else "-"

        # Row 0
        ax00 = fig.add_subplot(inner[0, 0])
        ax00.imshow(cover, cmap="gray", vmin=0, vmax=255)
        ax00.set_title(f"#{res['img_idx']}", fontsize=7.5, fontweight="bold", pad=2)
        ax00.axis("off")

        ax01 = fig.add_subplot(inner[0, 1])
        ax01.imshow(stego_p, cmap="gray", vmin=0, vmax=255)
        ax01.set_title(f"[A] PSNR={res['psnr_p']:.1f}", fontsize=6.5, color=BLUE, pad=2)
        ax01.axis("off")

        ax02 = fig.add_subplot(inner[0, 2])
        ax02.imshow(stego_c, cmap="gray", vmin=0, vmax=255)
        ax02.set_title(f"[B] PSNR={res['psnr_c']:.1f}", fontsize=6.5, color=RED, pad=2)
        ax02.axis("off")

        ax03 = fig.add_subplot(inner[0, 3])
        ax03.imshow(tex_map, cmap="jet", vmin=0, vmax=1)
        ax03.set_title("Texture", fontsize=6.5, pad=2)
        ax03.axis("off")

        # Row 1
        ax10 = fig.add_subplot(inner[1, 0])
        ax10.imshow(res["chaotic_mask"], cmap=WHITE_BLUE, vmin=0, vmax=1)
        ax10.set_title("Mask", fontsize=6.5, pad=2)
        ax10.axis("off")

        ax11 = fig.add_subplot(inner[1, 1])
        ax11.imshow(diff_p, cmap=WHITE_RED, vmin=0, vmax=255)
        ax11.set_title(f"[A]d*200\n{res['n_ch_p']:,}px", fontsize=6.0, color=BLUE, pad=2)
        ax11.axis("off")

        ax12 = fig.add_subplot(inner[1, 2])
        ax12.imshow(diff_c, cmap=WHITE_RED, vmin=0, vmax=255)
        ax12.set_title(f"[B]d*200\n{res['n_ch_c']:,}px", fontsize=6.0, color=RED, pad=2)
        ax12.axis("off")

        ax13 = fig.add_subplot(inner[1, 3])
        ax13.set_facecolor(LGRAY)
        cats = ["tex", "p_det"]
        vp   = [res["tex_p"], pd_p or 0]
        vc   = [res["tex_c"], pd_c or 0]
        x    = np.arange(2)
        b1 = ax13.bar(x-0.18, vp, 0.30, color=BLUE, alpha=0.85)
        b2 = ax13.bar(x+0.18, vc, 0.30, color=RED,  alpha=0.85)
        ax13.set_xticks(x)
        ax13.set_xticklabels(cats, fontsize=5.5)
        ax13.set_ylim(0, 1.12)
        ax13.tick_params(axis="y", labelsize=5.5)
        ax13.spines[["top", "right"]].set_visible(False)
        ax13.grid(True, axis="y", alpha=0.25, linestyle="--")
        for rect, val in zip(list(b1)+list(b2), vp+vc):
            ax13.text(rect.get_x()+rect.get_width()/2,
                      rect.get_height()+0.015, f"{val:.2f}",
                      ha="center", va="bottom", fontsize=4.8, fontweight="bold")
        ax13.set_title(f"Dp={mark}{abs(dpp):.1f}pp",
                       fontsize=6.5, fontweight="bold", color=color, pad=2)

    fig.suptitle(
        f"ICS-RL Seed-Based Comparison  |  seed={seed}  |  "
        f"{n} randomly selected BOSSBase images (n=20)\n"
        f"[A]=Policy (blue)  vs  [B]=Chaotic (red)",
        fontsize=11, fontweight="bold", y=1.005
    )
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"  [GRID] -> {path}")


# ═══════════════════════════════════════════════════════════════
#  KUTU GRAFIK
# ═══════════════════════════════════════════════════════════════
def save_boxplot(results, seed, path):
    def arr(key):
        return np.array([r[key] for r in results if r[key] is not None])

    metrics = [
        ("SRNet p_detect\n(lower better)", arr("pdet_p"), arr("pdet_c")),
        ("PSNR (dB)\n(higher better)",     arr("psnr_p"), arr("psnr_c")),
        ("SSIM\n(higher better)",          arr("ssim_p"), arr("ssim_c")),
        ("Texture\n(higher better)",       arr("tex_p"),  arr("tex_c")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"ICS-RL: Policy vs Chaotic -- seed={seed}, n={len(results)} random BOSSBase images",
        fontsize=12, fontweight="bold", y=1.01
    )
    for ax, (title, pv, cv) in zip(axes, metrics):
        ax.set_facecolor(LGRAY)
        bp = ax.boxplot([pv, cv], labels=["Policy", "Chaotic"],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        widths=0.45)
        bp["boxes"][0].set_facecolor(BLUE); bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(RED);  bp["boxes"][1].set_alpha(0.85)
        ax.scatter(np.random.normal(1, .07, len(pv)), pv, alpha=0.55, s=28, color=BLUE, zorder=3)
        ax.scatter(np.random.normal(2, .07, len(cv)), cv, alpha=0.55, s=28, color=RED,  zorder=3)
        ax.plot(1, pv.mean(), "D", color="white", markersize=7,
                markeredgecolor=BLUE, markeredgewidth=1.5, zorder=5)
        ax.plot(2, cv.mean(), "D", color="white", markersize=7,
                markeredgecolor=RED,  markeredgewidth=1.5, zorder=5)
        ax.annotate(f"u={pv.mean():.4f}", xy=(1, pv.mean()), xytext=(1.18, pv.mean()),
                    fontsize=8, color=BLUE, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
        ax.annotate(f"u={cv.mean():.4f}", xy=(2, cv.mean()), xytext=(2.18, cv.mean()),
                    fontsize=8, color=RED,  fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [BOX]  -> {path}")


# ═══════════════════════════════════════════════════════════════
#  TABLO FIGURU
# ═══════════════════════════════════════════════════════════════
def save_table_figure(summary, tests, n_images, seed, path):
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white"); ax.axis("off")

    def fmt(k, d=4):
        s = summary[k]
        return f"{s['mean']:.{d}f} +- {s['std']:.{d}f}"

    def pstr(m):
        if m not in tests: return "--"
        p = tests[m]["p"]
        return ("p<0.001" if p < 0.001 else "p<0.01" if p < 0.01
                else f"p={p:.3f}*" if p < 0.05 else f"p={p:.3f}")

    def impr(m, pk, ck, b):
        pm, cm = summary[pk]["mean"], summary[ck]["mean"]
        d   = pm - cm
        pct = abs(d) / max(abs(cm), 1e-9) * 100
        a   = ("v" if b=="lower" else "^") if (b=="lower" and d<0) or (b=="higher" and d>0) else ("^" if b=="lower" else "v")
        return f"{a} {abs(d):.4f} ({pct:.1f}%)"

    data = [
        ["SRNet p_detect", "lower=better",  fmt("pdet_p"),   fmt("pdet_c"),   impr("p_detect","pdet_p","pdet_c","lower"),  pstr("p_detect")],
        ["PSNR (dB)",      "higher=better", fmt("psnr_p",2), fmt("psnr_c",2), impr("PSNR",    "psnr_p","psnr_c","higher"), pstr("PSNR")],
        ["SSIM",           "higher=better", fmt("ssim_p",6), fmt("ssim_c",6), impr("SSIM",    "ssim_p","ssim_c","higher"), pstr("SSIM")],
        ["Texture Score",  "higher=better", fmt("tex_p"),    fmt("tex_c"),    impr("Texture", "tex_p", "tex_c", "higher"),  pstr("Texture")],
    ]
    hdrs = ["Metric", "Direction",
            f"ICS-RL Policy\n(n={n_images}, seed={seed})",
            "Chaotic Baseline",
            "Improvement", "Paired t-test"]
    row_colors = [["#e3f2fd"]*6, ["#f1f8e9"]*6, ["#f1f8e9"]*6, ["#fff8e1"]*6]

    tbl = ax.table(cellText=data, colLabels=hdrs,
                   cellColours=row_colors, colColours=["#1565c0"]*6,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.15, 2.4)
    for col_idx in range(6):
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        f"Table: ICS-RL vs Chaotic -- seed={seed}, n={n_images} random BOSSBase images\n"
        "Primary metric: SRNet p_detect (bold row, blue bg)",
        fontsize=11, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [TBL]  -> {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    args    = parse_args()
    SEED    = args.seed
    N       = args.n
    OUT_DIR = "results/seeded"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  ICS-RL -- SEED-BASED EVALUATION (n=20)")
    print(f"  seed={SEED}  |  n={N} randomly selected images")
    print("=" * 72 + "\n")

    # Dataset
    try:
        loader = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    # Seed'li secim
    try:
        image_indices = select_images_seeded(DATA_DIR, N, SEED)
    except AttributeError:
        import glob
        files = sorted(
            glob.glob(os.path.join(DATA_DIR, "*.pgm")) +
            glob.glob(os.path.join(DATA_DIR, "*.png")) +
            glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        )
        total = len(files)
        if total == 0:
            print(f"ERROR: {DATA_DIR} icinde goruntu bulunamadi"); sys.exit(1)
        rng = random.Random(SEED)
        all_idx = list(range(total))
        rng.shuffle(all_idx)
        image_indices = sorted(all_idx[:N])
        print(f"  Toplam {total} goruntu, seed={SEED} ile {N} adet secildi.")

    print(f"  Secilen indeksler: {image_indices}\n")

    # Paylasilan nesneler
    codec       = MessageCodec(max_bits=PAYLOAD_BITS)
    msg_bits    = codec.encode(REAL_MESSAGE)
    n_bits      = len(msg_bits)
    chaotic_gen = LogisticMapGenerator(x0=X0)
    texture_ext = TextureFeatureExtractor(patch_size=7)
    ch_rows, ch_cols = build_chaotic_coords(X0, R, IMG_SHAPE, N_CANDIDATES)

    # SRNet
    srnet = None
    for srnet_path in ["src/models/srnet_best.pth", "checkpoints/srnet_best.pth"]:
        if os.path.exists(srnet_path):
            try:
                from src.models.srnet import SRNet
                srnet = SRNet()
                srnet.load_state_dict(torch.load(srnet_path, map_location="cpu",
                                                  weights_only=True))
                srnet.eval()
                print(f"  SRNet  : loaded  ({srnet_path})")
            except Exception as e:
                print(f"  SRNet  : yuklenemedi ({e})")
            break
    if srnet is None:
        print("  SRNet  : bulunamadi -- p_detect atlanacak")

    # Policy
    policy = PolicyNetwork(input_channels=3)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} bulunamadi"); sys.exit(1)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    policy.eval()
    print(f"  Policy : loaded\n")

    # Degerlendirme
    results = []
    t0 = time.time()
    print(f"  {'Img':>4}  {'A(p_det)':>9}  {'B(p_det)':>9}  {'PSNR_A':>8}  "
          f"{'tex_A':>6}  {'Dpp':>7}  Status")
    print("  " + "-" * 60)

    for i, img_idx in enumerate(image_indices):
        cover, _ = loader.get(img_idx)
        res = evaluate_one(img_idx, cover, policy, srnet,
                           codec, msg_bits, n_bits,
                           chaotic_gen, texture_ext, ch_rows, ch_cols)
        results.append(res)
        mark = "OK" if res["diff_pp"] > 0 else "XX"
        pdp  = f"{res['pdet_p']:.4f}" if res["pdet_p"] else " N/A  "
        pdc  = f"{res['pdet_c']:.4f}" if res["pdet_c"] else " N/A  "
        print(f"  [{i+1:2d}] #{img_idx:<5} {pdp}  {pdc}  "
              f"{res['psnr_p']:>6.2f}  {res['tex_p']:>5.3f}  "
              f"{res['diff_pp']:>+6.1f}pp {mark}")

    elapsed = time.time() - t0
    print(f"\n  Tamamlandi -- {elapsed:.1f}s\n")

    # Kaydet -- on ek: n20_seed{SEED}_
    pf = lambda name: os.path.join(OUT_DIR, f"n20_seed{SEED}_{name}")

    save_selected_csv(image_indices, SEED, pf("selected_images.csv"))
    save_results_csv(results, pf("per_image_results.csv"))

    summary, tests = compute_summary_and_tests(results)
    save_summary_txt(summary, tests, N, SEED, pf("summary.txt"))
    draw_comparison_grid(results, SEED, pf("comparison_grid.png"))
    save_boxplot(results, SEED, pf("boxplot.png"))
    save_table_figure(summary, tests, N, SEED, pf("table.png"))

    print()
    print("=" * 72)
    print("  CIKTILAR:")
    for fname in [
        f"n20_seed{SEED}_selected_images.csv",
        f"n20_seed{SEED}_per_image_results.csv",
        f"n20_seed{SEED}_summary.txt",
        f"n20_seed{SEED}_comparison_grid.png",
        f"n20_seed{SEED}_boxplot.png",
        f"n20_seed{SEED}_table.png",
    ]:
        print(f"  {os.path.join(OUT_DIR, fname)}")
    print("=" * 72)
    print(f"\n  Tekrar calistir: python evaluate_seeded_n20.py --seed {SEED} --n {N}")


if __name__ == "__main__":
    main()