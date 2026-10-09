import os
import torch
from torch.utils.data import DataLoader
from model import UNet3D
from losses import HybridDiceFocalLoss
from dataset import BrainTumorDataset
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training on Device: {device}")

    # 1. Dataset & Dataloader
    dataset = BrainTumorDataset(num_samples=20, spatial_size=(64, 64, 64))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Model & Loss Setup
    model = UNet3D(in_channels=4, num_classes=3).to(device)
    criterion = HybridDiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Training Loop
    epochs = 3
    print("Starting Model Training...")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, d_loss, f_loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {running_loss / len(train_loader):.4f}")

    # 4. Save Trained Model Checkpoint locally
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, "brain_cancer_3dunet_v1.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel Weights saved to '{model_save_path}'")

if __name__ == "__main__":
    main()
