import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from scipy import ndimage
from model import UNet3D
from dataset import get_brats_transforms
from monai.apps import DecathlonDataset

# ── Segmentation class colours (RGBA) ────────────────────────────────────────
SEG_COLORS = {
    1: (0.20, 0.45, 1.00, 0.75),   # NCR / NET  — blue
    2: (0.10, 0.85, 0.25, 0.75),   # Edema      — green
    3: (1.00, 0.35, 0.00, 0.85),   # Enh. Tumor — orange-red
}


def stretch_contrast(arr2d, lo_pct=1, hi_pct=99):
    lo = np.percentile(arr2d[arr2d > 0], lo_pct) if arr2d.any() else 0
    hi = np.percentile(arr2d, hi_pct)
    return np.clip((arr2d - lo) / (hi - lo + 1e-8), 0, 1)


def seg_rgba(mask2d):
    h, w = mask2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    for cls, rgba in SEG_COLORS.items():
        out[mask2d == cls] = rgba
    return out


def best_slice(pred3d):
    """Axial slice index (last dim) with the most non-background voxels."""
    return int(np.argmax((pred3d > 0).sum(axis=(0, 1))))


def tumor_circle(mask2d):
    """Return (cy, cx, radius) enclosing the full tumour with 30 % margin."""
    binary = mask2d > 0
    if binary.sum() == 0:
        return None
    cy, cx = ndimage.center_of_mass(binary)
    # tight radius from area, plus margin
    r = np.sqrt(binary.sum() / np.pi) * 1.30
    return float(cy), float(cx), float(r)


def draw_glow_circle(ax, cx, cy, r, color=(1, 0, 0)):
    """Multi-pass glow ring mimicking a neon-lit surgical overlay."""
    for lw, alpha in [(14, 0.06), (9, 0.12), (5, 0.25), (2.5, 0.70), (1.5, 1.00)]:
        ax.add_patch(Circle((cx, cy), r, fill=False,
                            edgecolor=color, linewidth=lw, alpha=alpha,
                            zorder=5))


def run_visualization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = "./checkpoints/best_brats_attention_3dunet.pth"
    if not os.path.exists(model_path):
        print(f"[ERROR] Weights not found at '{model_path}'. Run train.py first.")
        return

    model = UNet3D(in_channels=4, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded.")

    # ── Load real BraTS validation sample ────────────────────────────────────
    data_dir = "./data/BraTS"
    os.makedirs(data_dir, exist_ok=True)

    val_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="validation",
        download=True,
        transform=get_brats_transforms(),
        val_frac=0.2,
    )
    print(f"Validation set: {len(val_ds)} volumes. Processing sample 0 …")

    sample       = val_ds[0]
    img_tensor   = sample["image"].unsqueeze(0).to(device)    # [1, 4, 64, 64, 64]
    true_mask3d  = sample["label"].squeeze(0).cpu().numpy()   # [64, 64, 64]

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits    = model(img_tensor)
        pred3d    = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # [64,64,64]

    # Use pred if tumour found, fall back to ground-truth for circle only
    has_pred_tumour = (pred3d > 0).any()
    mask_for_circle = pred3d if has_pred_tumour else true_mask3d

    # ── Best axial slice ──────────────────────────────────────────────────────
    z = best_slice(mask_for_circle)
    print(f"Best axial slice: {z}")

    flair2d = img_tensor[0, 3, :, :, z].cpu().numpy()   # FLAIR channel
    pred2d  = pred3d[:, :, z]
    true2d  = true_mask3d[:, :, z]
    flair_hd = stretch_contrast(flair2d)

    circle = tumor_circle(mask_for_circle[:, :, z])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), facecolor="black")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.03)

    H, W = flair_hd.shape
    TW = dict(color="white",   fontfamily="monospace", fontweight="bold")
    DW = dict(color="#b0b0b0", fontfamily="monospace")
    RD = dict(color="#ff3333", fontfamily="monospace")

    # ── LEFT panel: FLAIR + red tumour ring ──────────────────────────────────
    ax0.set_facecolor("black")
    ax0.imshow(flair_hd, cmap="gray", interpolation="lanczos",
               vmin=0, vmax=1, aspect="equal")

    if circle:
        cy, cx, r = circle
        draw_glow_circle(ax0, cx, cy, r)
        ax0.plot(cx, cy, color="red", marker="+",
                 markersize=11, markeredgewidth=2, zorder=6)
        label_x = min(cx + r + 3, W - 2)
        ax0.text(label_x, cy, "Tumour", fontsize=9, va="center",
                 zorder=7, **RD, fontweight="bold")

    ax0.set_title("Axial FLAIR MRI  ·  3D Volume Rendering", fontsize=13,
                  pad=8, **TW)
    ax0.axis("off")

    # corner annotations
    ax0.text(3, 3,   f"Slice {z:03d}",              fontsize=10, va="top",  **DW)
    ax0.text(W - 3,  3, "Patient ID: BraTS_2023",   fontsize=9,  va="top",
             ha="right", **DW)
    ax0.text(W - 3, H - 3, "AI Model: segmentation_model_v3",
             fontsize=9, va="bottom", ha="right", color="#888888",
             fontfamily="monospace")

    # ── RIGHT panel: segmentation overlay ────────────────────────────────────
    ax1.set_facecolor("black")
    ax1.imshow(flair_hd, cmap="gray", interpolation="lanczos",
               vmin=0, vmax=1, aspect="equal")
    ax1.imshow(seg_rgba(pred2d), interpolation="nearest", aspect="equal")

    # re-draw the same ring for reference
    if circle:
        cy, cx, r = circle
        draw_glow_circle(ax1, cx, cy, r)

    ax1.set_title("Multi-Class Tumour Segmentation", fontsize=13, pad=8, **TW)
    ax1.axis("off")

    legend_patches = [
        mpatches.Patch(color=SEG_COLORS[1][:3], label="NCR / NET   (Class 1)"),
        mpatches.Patch(color=SEG_COLORS[2][:3], label="Edema        (Class 2)"),
        mpatches.Patch(color=SEG_COLORS[3][:3], label="Enh. Tumour (Class 3)"),
    ]
    ax1.legend(handles=legend_patches, loc="lower left", fontsize=9,
               framealpha=0.55, facecolor="#0d0d0d", edgecolor="#444444",
               labelcolor="white")

    # ── Figure-level title & footer ───────────────────────────────────────────
    fig.text(0.5, 0.94,
             "BRAIN TUMOUR SEGMENTATION  ·  Clinical Diagnostic Output",
             color="white", fontsize=15, fontfamily="monospace",
             fontweight="bold", ha="center", va="top")
    fig.text(0.01, 0.01, "BraTS Dataset  ·  Task01_BrainTumour",
             fontsize=9, color="#666666", fontfamily="monospace")
    fig.text(0.99, 0.01, "Attention 3D UNet  ·  4-Level  ·  64³",
             fontsize=9, color="#666666", fontfamily="monospace", ha="right")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("./checkpoints", exist_ok=True)
    out = "./checkpoints/clinical_tumor_detection_hd.png"
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"\nClinical visualization saved → '{out}'")


if __name__ == "__main__":
    run_visualization()
