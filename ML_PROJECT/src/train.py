import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import timm


# ============================================================
# CONFIG (tuned for your ~8:1 imbalance and RTX 3060)
# ============================================================

@dataclass
class CFG:
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8

    use_amp: bool = True
    grad_clip: float = 1.0

    # FAST hybrid search (so it doesn't feel stuck)
    pop_size: int = 4
    search_iters: int = 3
    warmup_epochs: int = 1      # 1 epoch per candidate (fast)
    final_epochs: int = 25
    patience: int = 6

    # search ranges
    lr_min: float = 8e-6
    lr_max: float = 3e-4
    wd_min: float = 1e-6
    wd_max: float = 2e-3
    drop_min: float = 0.0
    drop_max: float = 0.35
    ls_min: float = 0.0
    ls_max: float = 0.12

    # Extra malignant emphasis scale (important for 8:1)
    pos_scale_min: float = 1.0
    pos_scale_max: float = 3.0

    seed: int = 42

    save_model_name: str = "best_efficientnet_b0.pth"
    save_hp_name: str = "best_hparams.json"


CFG = CFG()


# ============================================================
# SAFE PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("=" * 70)
print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_DIR   :", DATA_DIR)
print("MODEL_DIR  :", MODEL_DIR)
print("=" * 70)

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Missing train folder: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Missing test folder: {TEST_DIR}")


# ============================================================
# SEED
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(CFG.seed)


# ============================================================
# DEVICE (ERROR-SAFE)
# ============================================================

# Prevent HF downloads from stalling forever (optional safety)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# If CUDA is broken, torch.cuda.is_available() may be False even with GPU present.
# This is environment/driver related, not code.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    # CPU safety to avoid "Killed" (OOM) when CUDA is unavailable.
    # This does NOT change your training logic; it only prevents crashes.
    CFG.batch_size = min(CFG.batch_size, 16)
    CFG.num_workers = 0
    CFG.use_amp = False
    print("⚠ CUDA not available -> CPU mode safety applied:")
    print("   batch_size =", CFG.batch_size)
    print("   num_workers =", CFG.num_workers)
    print("   use_amp =", CFG.use_amp)


# ============================================================
# TRANSFORMS
# ============================================================

train_tf = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
    transforms.RandomResizedCrop(CFG.img_size, scale=(0.80, 1.0)),
    transforms.ToTensor(),
])

eval_tf = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
])


# ============================================================
# DATASETS
# ============================================================

def build_datasets():
    train_full = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    if os.path.isdir(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0:
        val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
        print("VAL folder found -> using provided val split.")
        return train_full, val_ds, test_ds

    print("No VAL folder found -> creating val split from TRAIN (10%).")
    n = len(train_full)
    idx = np.arange(n)
    np.random.shuffle(idx)
    v = max(1, int(0.10 * n))
    val_idx = idx[:v]
    tr_idx = idx[v:]

    train_ds = Subset(train_full, tr_idx.tolist())
    val_base = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
    val_ds = Subset(val_base, val_idx.tolist())
    return train_ds, val_ds, test_ds


def get_targets(ds) -> List[int]:
    if isinstance(ds, datasets.ImageFolder):
        return [y for _, y in ds.samples]
    if isinstance(ds, Subset):
        base = ds.dataset
        if not isinstance(base, datasets.ImageFolder):
            raise TypeError("Subset must wrap ImageFolder.")
        return [base.samples[i][1] for i in ds.indices]
    raise TypeError("Unsupported dataset type.")


def make_sampler(train_ds):
    targets = np.array(get_targets(train_ds), dtype=np.int64)
    counts = np.bincount(targets, minlength=2).astype(np.float64)

    class_w = 1.0 / np.clip(counts, 1.0, None)
    sample_w = class_w[targets]

    print("Train class counts (0=benign,1=malignant):", counts.tolist())
    print("Sampler class weights:", class_w.tolist())

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True
    )
    return sampler, counts


# ============================================================
# MODEL
# ============================================================

def create_model(dropout: float):
    m = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=2,
        drop_rate=float(dropout),
    )
    return m.to(device)


# ============================================================
# METRICS (no sklearn)
# ============================================================

@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
    p = torch.cat(preds_all).numpy()
    t = torch.cat(labels_all).numpy()

    tn = int(((p == 0) & (t == 0)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tp = int(((p == 1) & (t == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return {"acc": acc, "prec_mal": prec, "rec_mal": rec, "f1_mal": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs):
    model.train()
    run_loss = 0.0
    correct = 0
    total = 0

    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if CFG.use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if CFG.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

        run_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if bi % 30 == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = run_loss / max(1, total)
            avg_acc = correct / max(1, total)
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"[Epoch {epoch}/{total_epochs}] Batch {bi}/{len(loader)} "
                      f"loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr:.3g} mem={mem:.2f}GB")
            else:
                print(f"[Epoch {epoch}/{total_epochs}] Batch {bi}/{len(loader)} "
                      f"loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr:.3g}")

    return (run_loss / max(1, total)), (correct / max(1, total))


def fit(train_loader, val_loader, counts, hp, epochs, save_path):
    model = create_model(hp["dropout"])

    base = (counts.sum() / np.clip(counts, 1.0, None))
    base = base / base.mean()
    base[1] *= float(hp["pos_scale"])
    class_weight = torch.tensor(base, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=float(hp["label_smoothing"])
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"])
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=float(hp["lr"]) * 0.05
    )

    # FIX: remove FutureWarning by using torch.amp.GradScaler for CUDA
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=CFG.use_amp)
    else:
        scaler = None  # not used in CPU path

    best_f1 = -1.0
    bad = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ep, epochs)
        val_m = eval_loader(model, val_loader)
        scheduler.step(ep + 0.0)

        print(f"Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_acc={val_m['acc']:.4f} val_f1_mal={val_m['f1_mal']:.4f} "
              f"val_rec_mal={val_m['rec_mal']:.4f}")

        if val_m["f1_mal"] > best_f1 + 1e-4:
            best_f1 = val_m["f1_mal"]
            bad = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model -> {save_path} (best val_f1_mal={best_f1:.4f})")
        else:
            bad += 1
            print(f"No improvement: {bad}/{CFG.patience}")
            if bad >= CFG.patience:
                print("🛑 Early stopping.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_f1


# ============================================================
# HYBRID PSO + GA (FAST)
# ============================================================

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def logu(lo, hi):
    r = random.random()
    return math.exp(math.log(lo) + r * (math.log(hi) - math.log(lo)))

def init_pop(n):
    return [{
        "lr": logu(CFG.lr_min, CFG.lr_max),
        "weight_decay": logu(CFG.wd_min, CFG.wd_max),
        "dropout": random.uniform(CFG.drop_min, CFG.drop_max),
        "label_smoothing": random.uniform(CFG.ls_min, CFG.ls_max),
        "pos_scale": random.uniform(CFG.pos_scale_min, CFG.pos_scale_max),
    } for _ in range(n)]

def hp_to_vec(hp):
    return np.array([math.log(hp["lr"]), math.log(hp["weight_decay"]),
                     hp["dropout"], hp["label_smoothing"], hp["pos_scale"]], dtype=np.float64)

def vec_to_hp(v):
    return {
        "lr": clip(math.exp(v[0]), CFG.lr_min, CFG.lr_max),
        "weight_decay": clip(math.exp(v[1]), CFG.wd_min, CFG.wd_max),
        "dropout": clip(v[2], CFG.drop_min, CFG.drop_max),
        "label_smoothing": clip(v[3], CFG.ls_min, CFG.ls_max),
        "pos_scale": clip(v[4], CFG.pos_scale_min, CFG.pos_scale_max),
    }

def mutate(hp, p=0.45):
    out = dict(hp)
    if random.random() < p: out["lr"] = logu(CFG.lr_min, CFG.lr_max)
    if random.random() < p: out["weight_decay"] = logu(CFG.wd_min, CFG.wd_max)
    if random.random() < p: out["dropout"] = clip(out["dropout"] + random.uniform(-0.08, 0.08), CFG.drop_min, CFG.drop_max)
    if random.random() < p: out["label_smoothing"] = clip(out["label_smoothing"] + random.uniform(-0.03, 0.03), CFG.ls_min, CFG.ls_max)
    if random.random() < p: out["pos_scale"] = clip(out["pos_scale"] + random.uniform(-0.25, 0.25), CFG.pos_scale_min, CFG.pos_scale_max)
    return out

def crossover(a, b):
    return {k: (a[k] if random.random() < 0.5 else b[k]) for k in a.keys()}

def hybrid_search(train_loader, val_loader, counts):
    print("\n" + "=" * 70)
    print("HYBRID SEARCH (PSO+GA) - FAST MODE")
    print("=" * 70)

    pop = init_pop(CFG.pop_size)
    X = np.stack([hp_to_vec(h) for h in pop], axis=0)
    V = np.zeros_like(X)

    pbest = pop[:]
    pbest_s = np.full((CFG.pop_size,), -1e9, dtype=np.float64)
    gbest = None
    gbest_s = -1e9

    w, c1, c2 = 0.6, 1.2, 1.2

    tmp_path = os.path.join(MODEL_DIR, "_tmp.pth")

    for it in range(1, CFG.search_iters + 1):
        print(f"\n--- Iteration {it}/{CFG.search_iters} ---")

        for i in range(CFG.pop_size):
            hp = vec_to_hp(X[i])
            print(f"Candidate {i+1}/{CFG.pop_size}: {hp}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            _, score = fit(train_loader, val_loader, counts, hp, CFG.warmup_epochs, tmp_path)
            print(f" -> score (val_f1_mal) = {score:.4f}")

            if score > pbest_s[i]:
                pbest_s[i] = score
                pbest[i] = hp
            if score > gbest_s:
                gbest_s = score
                gbest = hp

        print(f"🔥 Best so far: {gbest_s:.4f} | {gbest}")

        P = np.stack([hp_to_vec(h) for h in pbest], axis=0)
        G = hp_to_vec(gbest)

        for i in range(CFG.pop_size):
            r1 = np.random.rand(X.shape[1])
            r2 = np.random.rand(X.shape[1])
            V[i] = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (G - X[i])
            X[i] = X[i] + V[i]

            X[i][0] = clip(X[i][0], math.log(CFG.lr_min), math.log(CFG.lr_max))
            X[i][1] = clip(X[i][1], math.log(CFG.wd_min), math.log(CFG.wd_max))
            X[i][2] = clip(X[i][2], CFG.drop_min, CFG.drop_max)
            X[i][3] = clip(X[i][3], CFG.ls_min, CFG.ls_max)
            X[i][4] = clip(X[i][4], CFG.pos_scale_min, CFG.pos_scale_max)

        k = max(2, CFG.pop_size // 2)
        top_idx = np.argsort(-pbest_s)[:k].tolist()
        elites = [pbest[j] for j in top_idx]

        new_pop = elites[:]
        while len(new_pop) < CFG.pop_size:
            a, b = random.sample(elites, 2)
            child = mutate(crossover(a, b), p=0.5)
            new_pop.append(child)

        X = np.stack([hp_to_vec(h) for h in new_pop], axis=0)

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    return gbest, float(gbest_s)


# ============================================================
# MAIN
# ============================================================

def main():
    train_ds, val_ds, test_ds = build_datasets()
    sampler, counts = make_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, sampler=sampler,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda")
    )

    best_hp, best_score = hybrid_search(train_loader, val_loader, counts)

    hp_path = os.path.join(MODEL_DIR, CFG.save_hp_name)
    with open(hp_path, "w") as f:
        json.dump({"best_hp": best_hp, "best_val_f1_mal": best_score}, f, indent=2)
    print(f"\n✅ Best hyperparameters saved -> {hp_path}")

    best_model_path = os.path.join(MODEL_DIR, CFG.save_model_name)
    print("\n" + "=" * 70)
    print("FINAL TRAINING with best hyperparameters")
    print("Best HP:", best_hp)
    print("=" * 70)

    model, best_f1 = fit(train_loader, val_loader, counts, best_hp, CFG.final_epochs, best_model_path)

    test_m = eval_loader(model, test_loader)
    print("\n" + "=" * 70)
    print("TEST EVALUATION (0=benign, 1=malignant)")
    print("=" * 70)
    print(f"Accuracy        : {test_m['acc']:.4f}")
    print(f"Precision (mal) : {test_m['prec_mal']:.4f}")
    print(f"Recall (mal)    : {test_m['rec_mal']:.4f}")
    print(f"F1 (mal)        : {test_m['f1_mal']:.4f}")
    print("Confusion [[TN FP],[FN TP]]:")
    print([[test_m["tn"], test_m["fp"]], [test_m["fn"], test_m["tp"]]])
    print("=" * 70)
    print(f"✅ Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()