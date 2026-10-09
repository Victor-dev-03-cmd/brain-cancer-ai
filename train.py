import os
import torch
from torch.utils.data import DataLoader
from model import UNet3D
from losses import HybridDiceFocalLoss
from dataset import RealBraTSDataset, get_brats_transforms
from tqdm import tqdm
from monai.apps import DecathlonDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Real BraTS Training on Device: {device}")

    # 1. Download & Load Real BraTS Task01 Dataset via MONAI
    data_dir = "./data/BraTS"
    os.makedirs(data_dir, exist_ok=True)
    
    print("\nReal BraTS (Task01_BrainTumour) Dataset லோட் செய்யப்படுகிறது...")
    try:
        train_ds = DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="training",
            download=True,
            transform=get_brats_transforms(),
            val_frac=0.2,
        )
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
        print(f"Dataset Successfully Loaded! Total Training Volumes: {len(train_ds)}")
    except Exception as e:
        print(f"Dataset Loading Exception: {e}")
        return

    # 2. Model, Loss, Optimizer & AMP Scaler
    model = UNet3D(in_channels=4, num_classes=3).to(device)
    criterion = HybridDiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 3. Training Loop (100 Epochs)
    epochs = 100
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    print(f"\nStarting Attention 3D UNet Real Training for {epochs} Epochs...\n")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images, masks = images.to(device), masks.to(device)
            
            # Squeeze mask channel if necessary
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

        avg_loss = running_loss / len(train_loader)
        print(f"-> Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Save Best Model Weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, "best_brats_attention_3dunet.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   [+] Best Model Saved with Loss: {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("REAL BraTS DATASET TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
