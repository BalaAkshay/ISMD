import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.custom_unet_1 import CustomUNet


class UNetMiningDataset(Dataset):

    def __init__(self, patches_dir, annotations_file):
        self.patches_dir = patches_dir
        self.annotations_file = annotations_file
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.patch_files = []
        for root, dirs, files in os.walk(patches_dir):
            for file in files:
                if file.endswith('.npy'):
                    relative_path = os.path.relpath(os.path.join(root, file), patches_dir)
                    annotation_key = relative_path.replace(os.path.sep, '/')
                    
                    if annotation_key in self.annotations:
                        self.patch_files.append({
                            'path': os.path.join(root, file),
                            'label': self.annotations[annotation_key]['label'],
                            'key': annotation_key
                        })
        
        print(f"Loaded {len(self.patch_files)} patches for U-Net training")
        if len(self.patch_files) > 0:
            mining_count = sum(1 for item in self.patch_files if item['label'] == 1)
            print(f"Mining: {mining_count}, Non-mining: {len(self.patch_files) - mining_count}")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        item = self.patch_files[idx]
        
        patch_data = np.load(item['path'])
        image_tensor = torch.from_numpy(patch_data.astype(np.float32))
        
        if item['label'] == 1:
            ndvi = patch_data[3, :, :]
            ndwi = patch_data[4, :, :]
            mndwi = patch_data[5, :, :]
            
            bare_soil = (ndvi < 0.2) & (mndwi < 0.3)
            water_pits = (mndwi > 0.2) | (ndwi > 0.1)
            disturbed = (ndvi < 0.1)
            
            rough_mask = bare_soil | water_pits | disturbed
            
            from scipy import ndimage
            
            rough_mask = ndimage.binary_fill_holes(rough_mask)
            rough_mask = ndimage.binary_erosion(rough_mask, iterations=1)
            rough_mask = ndimage.binary_dilation(rough_mask, iterations=2)
            
            mask = rough_mask.astype(np.float32)
            
            if mask.sum() < 100:  
                mask = np.ones((256, 256), dtype=np.float32) * 0.5  
        else:
            mask = np.zeros((256, 256), dtype=np.float32)
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor


class UNetTrainer:
    
    def __init__(self, learning_rate=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        
        self.model = CustomUNet(n_channels=6, n_classes=1).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        
        print(f"U-Net initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, patches_dir, annotations_file, batch_size=8):

        print(f"Looking for data in:")
        print(f"  - Patches: {os.path.abspath(patches_dir)}")
        print(f"  - Annotations: {os.path.abspath(annotations_file)}")
        
        if not os.path.exists(patches_dir):
            print(f"Patches directory not found: {patches_dir}")
            return None, None
        
        if not os.path.exists(annotations_file):
            print(f"Annotations file not found: {annotations_file}")
            return None, None
        
        full_dataset = UNetMiningDataset(patches_dir, annotations_file)
        
        if len(full_dataset) == 0:
            print("No patches found!")
            return None, None
        
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data prepared:")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Batch size: {batch_size}")
        
        return self.train_loader, self.val_loader
    
    def calculate_iou(self, pred_mask, true_mask, threshold=0.5):

        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
        true_binary = true_mask
        
        intersection = (pred_binary * true_binary).sum()
        union = pred_binary.sum() + true_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    def train_epoch(self):
        
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            batch_iou = self.calculate_iou(outputs, masks)
            running_iou += batch_iou
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, IoU: {batch_iou:.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_iou = running_iou / len(self.train_loader)
        
        return epoch_loss, epoch_iou
    
    def validate_epoch(self):
        
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                running_iou += self.calculate_iou(outputs, masks)
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_iou = running_iou / len(self.val_loader)
        
        return epoch_loss, epoch_iou
    
    def train(self, epochs=20, patience=5):
        
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            print("Cannot train - data not loaded properly")
            return
        
        print(f"\n{'='*50}")
        print(f"Starting U-Net training for {epochs} epochs...")
        print(f"{'='*50}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            train_loss, train_iou = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_ious.append(train_iou)
            
            val_loss, val_iou = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("unet_mining_model.pth")
                print(f"\n New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        self.plot_training_history()
        print(f"\n{'='*50}")
        print(f"Training completed! Best val loss: {best_val_loss:.4f}")
        print(f"{'='*50}\n")
    
    def save_model(self, filename):

        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
        }, model_path)
        
        print(f"Model saved to: {model_path}")
    
    def plot_training_history(self):

        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.train_ious, label='Train IoU', marker='o')
        ax2.plot(self.val_ious, label='Val IoU', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('Training & Validation IoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = "models/unet_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
        plt.close()


def start_unet_training():

    print("\n" + "="*60)
    print("U-Net Change Detection Training")
    print("="*60 + "\n")
    
    trainer = UNetTrainer(learning_rate=1e-4)
    
    patches_dir = "data/processed/patches_all"
    annotations_file = "data/annotations/improved_labels.json"
    
    success = trainer.prepare_data(patches_dir, annotations_file, batch_size=8)
    
    if success:
        trainer.train(epochs=20, patience=5)
    else:
        print("Failed to load data. Cannot start training.")


if __name__ == "__main__":
    start_unet_training()
