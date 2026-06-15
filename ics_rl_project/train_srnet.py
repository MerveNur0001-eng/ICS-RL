"""
train_srnet.py — v8 ROOT CAUSE FIX + RESUME + LOG + TARGET STOP
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.models.srnet      import SRNet
from utils.bossbase_loader import BOSSBaseLoader

DATA_DIR       = "data/bossbase/"
SRNET_SAVE_DIR = "src/models/"
IMG_SHAPE      = (256, 256)
BATCH_SIZE     = 16
EPOCHS         = 40
MAX_IMAGES     = 1000
TRAIN_RATIO    = 0.8
LR             = 1e-4
PAYLOAD        = 0.25

CHAOTIC_X0_BASE  = 0.123456
CHAOTIC_R        = 3.99
COORD_JITTER_STD = 0.15
RANDOM_COORD_PROB= 0.80

IDEAL_MAX      = 0.75
EARLY_STOP_PAT = 10
TARGET_VAL_ACC = 0.72

os.makedirs(SRNET_SAVE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)
_H, _W   = IMG_SHAPE
_N_BITS  = int(_H * _W * PAYLOAD)


def _build_chaotic_coords(h, w, n_bits, x0, r=CHAOTIC_R):
    total = h * w
    seq   = np.empty(total, dtype=np.float64)
    x     = x0
    for i in range(total):
        x = r * x * (1.0 - x)
        seq[i] = x
    order    = np.argsort(seq)
    selected = order[:n_bits]
    rows = (selected // w).astype(np.int32)
    cols = (selected  % w).astype(np.int32)
    return rows, cols


class StegoDataset(Dataset):
    def __init__(self, cover_indices, stego_indices, loader, augment=True):
        self.n_bits  = _N_BITS
        self.augment = augment

        total = len(cover_indices) + len(stego_indices)
        print(f"  Loading images ({total} total = "
              f"{len(cover_indices)} cover + {len(stego_indices)} stego)...")

        self.cover_tensors = []
        for i, idx in enumerate(cover_indices):
            img, _ = loader.get(idx)
            t = torch.from_numpy(img.astype(np.float32) / 255.0)
            self.cover_tensors.append(t)

        self.stego_tensors = []
        for i, idx in enumerate(stego_indices):
            img, _ = loader.get(idx)
            t = torch.from_numpy(img.astype(np.float32) / 255.0)
            self.stego_tensors.append(t)

        print(f"  Loading complete.\n")

    def __len__(self):
        return len(self.cover_tensors) + len(self.stego_tensors)

    def _embed_stego(self, img_t):
        img_arr = (img_t * 255).byte().numpy()

        if np.random.random() < RANDOM_COORD_PROB:
            flat_idx = np.random.choice(_H * _W, self.n_bits, replace=False)
            rows = (flat_idx // _W).astype(np.int32)
            cols = (flat_idx  % _W).astype(np.int32)
        else:
            jitter = np.random.normal(0, COORD_JITTER_STD)
            x0_new = np.clip(CHAOTIC_X0_BASE + jitter, 0.01, 0.99)
            rows, cols = _build_chaotic_coords(_H, _W, self.n_bits, x0_new)

        bits = np.random.randint(0, 2, size=self.n_bits, dtype=np.uint8)
        img_arr[rows, cols] = (img_arr[rows, cols] & np.uint8(0xFE)) | bits
        return torch.from_numpy(img_arr.astype(np.float32) / 255.0)

    def __getitem__(self, idx):
        if idx < len(self.cover_tensors):
            img_t = self.cover_tensors[idx].clone()
            label = 0
        else:
            stego_idx = idx - len(self.cover_tensors)
            img_t = self._embed_stego(self.stego_tensors[stego_idx].clone())
            label = 1

        return img_t.unsqueeze(0), torch.tensor(label, dtype=torch.long)


def train_srnet():
    log_path    = "results/srnet_v8_log.txt"
    save_path   = os.path.join(SRNET_SAVE_DIR, "srnet_best.pth")
    resume_path = os.path.join(SRNET_SAVE_DIR, "srnet_checkpoint.pth")

    loader  = BOSSBaseLoader(DATA_DIR, img_shape=IMG_SHAPE)
    n_total = min(loader.size, MAX_IMAGES)

    half        = n_total // 2
    all_indices = list(range(n_total))
    np.random.shuffle(all_indices)

    cover_all = all_indices[:half]
    stego_all = all_indices[half:]

    n_cover_train = int(len(cover_all) * TRAIN_RATIO)
    n_stego_train = int(len(stego_all) * TRAIN_RATIO)

    cover_train = cover_all[:n_cover_train]
    cover_val   = cover_all[n_cover_train:]
    stego_train = stego_all[:n_stego_train]
    stego_val   = stego_all[n_stego_train:]

    print(f"Data split:")
    print(f"  Train → {len(cover_train)} cover + {len(stego_train)} stego "
          f"= {len(cover_train)+len(stego_train)} samples")
    print(f"  Val   → {len(cover_val)} cover + {len(stego_val)} stego "
          f"= {len(cover_val)+len(stego_val)} samples\n")

    train_ds = StegoDataset(cover_train, stego_train, loader, augment=True)
    val_ds   = StegoDataset(cover_val,   stego_val,   loader, augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = SRNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler      = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc   = 0.0
    patience_count = 0
    history        = {"train_acc": [], "val_acc": [], "train_loss": []}
    start_epoch    = 1

    # ── RESUME ──────────────────────────────────────────────────
    if os.path.exists(resume_path):
        print(f"  Checkpoint found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_acc   = ckpt["best_val_acc"]
        start_epoch    = ckpt["epoch"] + 1
        history        = ckpt["history"]
        patience_count = ckpt.get("patience_count", 0)
        print(f"  Resuming from epoch {ckpt['epoch']} "
              f"(best_val={best_val_acc:.4f})\n")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,lr,loss,train_acc,val_acc,status\n")
        print(f"  Starting from scratch...\n")

    print(f"Device: {device}")
    print(f"Target: training stops automatically when val_acc >= {TARGET_VAL_ACC}")
    print(f"{'Epoch':>6} | {'LR':>8} | {'Loss':>7} | {'Train':>7} | {'Val':>7} | Status")
    print("-" * 65)

    target_reached = False

    for epoch in range(start_epoch, EPOCHS + 1):

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_loss    += loss.item() * imgs.size(0)
            train_total   += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        avg_loss  = train_loss    / train_total
        cur_lr    = optimizer.param_groups[0]['lr']

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)

        if val_acc > IDEAL_MAX:
            status = "MEMORISING"
            patience_count += 1
        elif val_acc < 0.52:
            status = "BELOW CHANCE"
            patience_count = 0
        elif val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            if val_acc >= TARGET_VAL_ACC:
                status = f"TARGET REACHED ({val_acc:.4f} >= {TARGET_VAL_ACC})"
                target_reached = True
            else:
                status = "SAVED"
        else:
            status = ""
            patience_count = 0

        print(f"{epoch:>5}/{EPOCHS} | {cur_lr:.2e} | {avg_loss:.4f} | "
              f"{train_acc:.4f} | {val_acc:.4f} | {status}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{cur_lr:.2e},{avg_loss:.4f},"
                    f"{train_acc:.4f},{val_acc:.4f},{status}\n")

        torch.save({
            "epoch"         : epoch,
            "model"         : model.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            "best_val_acc"  : best_val_acc,
            "history"       : history,
            "patience_count": patience_count,
        }, resume_path)

        scheduler.step()

        if target_reached:
            print(f"\n  ✓ Target val_acc={TARGET_VAL_ACC} reached — training stopped.")
            print(f"  Model saved: {save_path}")
            break

        if patience_count >= EARLY_STOP_PAT:
            print(f"\n  !! Early stopping: memorisation for {EARLY_STOP_PAT} epochs !!")
            break

    print(f"\nComplete. Best val_acc: {best_val_acc:.4f}")
    print(f"Log: {log_path}")

    if best_val_acc == 0.0:
        torch.save(model.state_dict(), save_path)
        print("(No model found in ideal range — last model saved)")

    _print_advice(best_val_acc)
    _save_plot(history)
    return history


def _print_advice(best_val_acc):
    print()
    if best_val_acc >= TARGET_VAL_ACC:
        print(f"  SRNet EXCELLENT ✓  — target {TARGET_VAL_ACC} reached")
    elif best_val_acc >= 0.63:
        print("  SRNet READY ✓  — proceed to RL training")
    elif best_val_acc >= 0.52:
        print("  SRNet weak but usable — try more epochs")
    else:
        print("  Below chance level — check DATA_DIR")


def _save_plot(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(history["train_acc"])
        x = range(1, n + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("SRNet v8 — Cover/Stego Split", fontsize=12)

        axes[0].plot(x, history["train_acc"], label="Train", marker="o", markersize=3)
        axes[0].plot(x, history["val_acc"],   label="Val",   marker="s", markersize=3)
        axes[0].axhline(0.50, color="red",    linestyle="--", alpha=0.5, label="Chance (0.50)")
        axes[0].axhline(0.63, color="green",  linestyle=":",  alpha=0.7, label="Min (0.63)")
        axes[0].axhline(0.72, color="blue",   linestyle="-",  alpha=0.7, label="Target (0.72)")
        axes[0].axhline(0.75, color="orange", linestyle=":",  alpha=0.7, label="Max (0.75)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)
        axes[0].set_ylim(0.3, 1.0)

        axes[1].plot(x, history["train_loss"], color="orange", marker="o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].grid(True)

        plt.tight_layout()
        out = "results/srnet_v8_training.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {out}")
    except Exception as e:
        print(f"Plot could not be saved: {e}")


if __name__ == "__main__":
    train_srnet()