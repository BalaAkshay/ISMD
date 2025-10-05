import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Define the model directly
class MiningClassifier(nn.Module):
    """Simpler CNN classifier for mining detection"""
    def __init__(self, in_channels=6, num_classes=1):
        super(MiningClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleMiningDataset(Dataset):
    """Simple dataset that definitely works"""
    def __init__(self, patches_dir, annotations_file):
        self.patches_dir = patches_dir
        self.annotations_file = annotations_file
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Find all patch files
        self.patch_files = []
        for root, dirs, files in os.walk(patches_dir):
            for file in files:
                if file.endswith('.npy'):
                    # Create the key used in annotations
                    relative_path = os.path.relpath(os.path.join(root, file), patches_dir)
                    annotation_key = relative_path.replace(os.path.sep, '/')
                    
                    if annotation_key in self.annotations:
                        self.patch_files.append({
                            'path': os.path.join(root, file),
                            'label': self.annotations[annotation_key]['label'],
                            'key': annotation_key
                        })
        
        print(f"Loaded {len(self.patch_files)} patches")
        if len(self.patch_files) > 0:
            mining_count = sum(1 for item in self.patch_files if item['label'] == 1)
            print(f"Mining: {mining_count}, Non-mining: {len(self.patch_files) - mining_count}")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        item = self.patch_files[idx]
        
        # Load patch
        patch_data = np.load(item['path'])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(patch_data.astype(np.float32))
        label_tensor = torch.tensor(item['label'], dtype=torch.float32)
        
        return image_tensor, label_tensor

class WorkingMiningTrainer:
    def __init__(self, learning_rate=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        
        # Use simpler classifier
        self.model = MiningClassifier(in_channels=6, num_classes=1).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Custom model initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, patches_dir, annotations_file, batch_size=16):
        """Prepare data loaders - SIMPLE AND RELIABLE"""
        print(f"Looking for data in:")
        print(f"  - Patches: {os.path.abspath(patches_dir)}")
        print(f"  - Annotations: {os.path.abspath(annotations_file)}")
        
        # Check if files exist
        if not os.path.exists(patches_dir):
            print(f"Patches directory not found: {patches_dir}")
            return None, None
        
        if not os.path.exists(annotations_file):
            print(f"Annotations file not found: {annotations_file}")
            return None, None
        
        # Create dataset
        full_dataset = SimpleMiningDataset(patches_dir, annotations_file)
        
        if len(full_dataset) == 0:
            print("No patches found! Checking directory structure...")
            self._debug_directory_structure(patches_dir, annotations_file)
            return None, None
        
        # Split into train/validation
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data prepared:")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Batch size: {batch_size}")
        
        return self.train_loader, self.val_loader
    
    def _debug_directory_structure(self, patches_dir, annotations_file):
        """Debug why no patches are found"""
        print("\n DEBUGGING DIRECTORY STRUCTURE:")
        
        # Check patches directory
        if os.path.exists(patches_dir):
            print(f"Patches directory exists")
            patch_count = 0
            for root, dirs, files in os.walk(patches_dir):
                npy_files = [f for f in files if f.endswith('.npy')]
                if npy_files:
                    print(f"  {root}: {len(npy_files)} .npy files")
                    patch_count += len(npy_files)
                    if len(npy_files) > 0:
                        print(f"    Sample files: {npy_files[:3]}")
            print(f"  Total .npy files found: {patch_count}")
        else:
            print(f"Patches directory does not exist: {patches_dir}")
        
        # Check annotations file
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            print(f"Annotations file exists with {len(annotations)} entries")
            print(f"  Sample keys: {list(annotations.keys())[:3]}")
        else:
            print(f"Annotations file does not exist: {annotations_file}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            if batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs.squeeze(), targets)
                
                running_loss += loss.item()
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def train(self, epochs=30, patience=7):
        """Full training loop"""
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            print("Cannot train - data not loaded properly")
            return
        
        print(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("working_mining_model.pth")
                print(f"💾 New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        self.plot_training_history()
        print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, model_path)
        
        print(f"Model saved to: {model_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "models/training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

def start_working_training():
    """Start training with the working dataset"""
    trainer = WorkingMiningTrainer()
    
    # Prepare data - use direct paths
    patches_dir = "data/processed/patches_all"
    annotations_file = "data/annotations/improved_labels.json"
    
    success = trainer.prepare_data(patches_dir, annotations_file, batch_size=16)
    
    if success:
        # Start training
        trainer.train(epochs=30, patience=7)
    else:
        print("Failed to load data. Cannot start training.")

if __name__ == "__main__":
    start_working_training()