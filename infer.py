import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D
from dataset import get_brats_transforms
from monai.apps import DecathlonDataset


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Real BraTS Inference on Device: {device}")

    # 1. Load trained model weights
    model = UNet3D(in_channels=4, num_classes=4).to(device)
    model_path = "./checkpoints/best_brats_attention_3dunet.pth"

    if not os.path.exists(model_path):
        print(f"Error: Trained model weights '{model_path}' not found!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Trained Model Weights Loaded Successfully!")

    # 2. Load a real validation sample from DecathlonDataset
    data_dir = "./data/BraTS"
    val_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="validation",
        download=False,
        transform=get_brats_transforms(),
        val_frac=0.2,
    )
    print(f"Validation set: {len(val_ds)} volumes. Running inference on sample 0...")

    sample     = val_ds[0]
    image      = sample["image"].unsqueeze(0).to(device)   # [1, 4, 64, 64, 64]
    true_mask  = sample["label"].squeeze(0).cpu().numpy()  # [64, 64, 64]

    # 3. Predict tumor regions
    with torch.no_grad():
        output_logits = model(image)
        pred_mask = torch.argmax(output_logits, dim=1).squeeze(0).cpu().numpy()  # [64, 64, 64]

    # 4. Visualise middle slice: MRI scan | ground truth | prediction
    slice_idx = pred_mask.shape[-1] // 2
    flair_channel = image[0, 3, :, :, slice_idx].cpu().numpy()  # FLAIR is channel 3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("MRI Scan (FLAIR Channel)")
    axes[0].imshow(flair_channel, cmap="gray")
    axes[0].axis("off")

    axes[1].set_title("Ground Truth Mask")
    axes[1].imshow(true_mask[:, :, slice_idx], cmap="jet", vmin=0, vmax=3)
    axes[1].axis("off")

    axes[2].set_title("AI Predicted Tumor Segmentation")
    axes[2].imshow(pred_mask[:, :, slice_idx], cmap="jet", vmin=0, vmax=3)
    axes[2].axis("off")

    plt.tight_layout()
    output_plot = "./checkpoints/real_brats_inference_result.png"
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Inference complete! Saved to '{output_plot}'")


if __name__ == "__main__":
    run_inference()
