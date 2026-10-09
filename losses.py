import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for 3D Segmentation
    Measures overlap between Predicted Mask and Ground Truth Mask.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply Softmax to get class probabilities
        probs = F.softmax(logits, dim=1)
        
        # Flatten tensors for calculation: [Batch, Classes, Voxels]
        num_classes = logits.shape[1]
        probs = probs.view(probs.shape[0], num_classes, -1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3)
        targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        
        intersection = torch.sum(probs * targets_one_hot, dim=-1)
        cardinality = torch.sum(probs + targets_one_hot, dim=-1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


class FocalLoss(nn.Module):
    """
    Focal Loss to handle extreme class imbalance (Small Tumors)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


class HybridDiceFocalLoss(nn.Module):
    """
    Combines Dice Loss + Focal Loss for World-Class Segmentation
    """
    def __init__(self, weight_dice=1.0, weight_focal=1.0):
        super(HybridDiceFocalLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.w_dice = weight_dice
        self.w_focal = weight_focal

    def forward(self, logits, targets):
        l_dice = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)
        total_loss = (self.w_dice * l_dice) + (self.w_focal * l_focal)
        return total_loss, l_dice, l_focal


# Local Testing
if __name__ == "__main__":
    # Test Prediction: [Batch=1, Classes=3, Depth=32, Height=32, Width=32]
    dummy_logits = torch.randn(1, 3, 32, 32, 32)
    # Test Target Mask: [Batch=1, Depth=32, Height=32, Width=32] with class indices 0, 1, 2
    dummy_targets = torch.randint(0, 3, (1, 32, 32, 32), dtype=torch.long)
    
    criterion = HybridDiceFocalLoss()
    total_loss, dice_loss, focal_loss = criterion(dummy_logits, dummy_targets)
    
    print("=" * 55)
    print("  HYBRID LOSS FUNCTION CREATED SUCCESSFULLY!  ")
    print("=" * 55)
    print(f"Total Hybrid Loss : {total_loss.item():.4f}")
    print(f"  ├── Dice Loss   : {dice_loss.item():.4f}")
    print(f"  └── Focal Loss  : {focal_loss.item():.4f}")
    print("Status            : Mathematical Objective Ready!")
    print("=" * 55)
