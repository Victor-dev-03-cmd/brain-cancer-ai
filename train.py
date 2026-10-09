import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import UNet3D
from losses import HybridDiceFocalLoss
from dataset import BrainTumorDataset
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing High-Performance Training on Device: {device}")

    # 1. Dataset & Dataloader
    dataset = BrainTumorDataset(num_samples=20, spatial_size=(64, 64, 64))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Model, Loss, Optimizer & AMP Scaler Setup
    model = UNet3D(in_channels=4, num_classes=3).to(device)
    criterion = HybridDiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 3. Training Loop
    epochs = 3
    print("Starting Attention 3D UNet Model Training...")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # AMP Mixed Precision Forward Pass
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss, d_loss, f_loss = criterion(outputs, masks)

            # Backward Pass with Scaler
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {running_loss / len(train_loader):.4f}")

    # 4. Save Trained Checkpoint
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, "brain_cancer_3dunet_v1.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nUpgraded Attention Model Weights saved to '{model_save_path}'")

if __name__ == "__main__":
    main()
