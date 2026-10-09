import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import UNet3D
from losses import HybridDiceFocalLoss
from dataset import get_brats_transforms
from tqdm import tqdm
from monai.apps import DecathlonDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Real BraTS Training on Device: {device}")

    data_dir = "./data/BraTS"
    os.makedirs(data_dir, exist_ok=True)

    print("\nReal BraTS (Task01_BrainTumour) Dataset is Loading Now...")
    try:
        train_ds = DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="training",
            download=True,
            transform=get_brats_transforms(),
            val_frac=0.2,
        )
        val_ds = DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="validation",
            download=False,
            transform=get_brats_transforms(),
            val_frac=0.2,
        )
        train_loader = DataLoader(
            train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True
        )
        print(f"Dataset Loaded! Train: {len(train_ds)} | Val: {len(val_ds)} volumes")
    except Exception as e:
        print(f"Dataset Loading Exception: {e}")
        return

    epochs = 100
    model = UNet3D(in_channels=4, num_classes=4).to(device)
    criterion = HybridDiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nStarting 4-Level Attention 3D UNet Training for {epochs} Epochs...\n")

    for epoch in range(epochs):
        # --- Training Pass ---
        model.train()
        running_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train"):
            images = batch_data["image"].to(device)
            masks  = batch_data["label"].to(device, dtype=torch.long)
            if masks.dim() == 5 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss, d_loss, f_loss = criterion(outputs, masks)

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Pass ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  "):
                images = batch_data["image"].to(device)
                masks  = batch_data["label"].to(device, dtype=torch.long)
                if masks.dim() == 5 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        scheduler.step()

        print(
            f"-> Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_brats_attention_3dunet.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   [+] Best Model Saved (Val Loss: {best_val_loss:.4f})")

    print("\n" + "=" * 60)
    print("REAL BraTS DATASET TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
