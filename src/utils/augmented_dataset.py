import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.augmentations import get_training_augmentations

class AugmentedMiningDataset(Dataset):
    def __init__(self, patches_dir, annotations_file, transform=None, is_training=True):
        self.patches_dir = patches_dir
        self.is_training = is_training
        
        # Set transform - use training augmentations if training, else None
        if is_training:
            self.transform = transform if transform else get_training_augmentations()
        else:
            self.transform = None
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get all patch files with labels
        self.patch_files = [f for f in os.listdir(patches_dir) 
                           if f.endswith('.npy') and f in self.annotations]
        
        print(f"Loaded {len(self.patch_files)} patches for {'training' if is_training else 'validation'}")
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]
        patch_path = os.path.join(self.patches_dir, patch_file)
        
        # Load patch data
        patch_data = np.load(patch_path)  # Shape: (6, 256, 256)
        
        # Get label
        label = self.annotations[patch_file]
        
        # Apply augmentation if training and transform is available
        if self.is_training and self.transform is not None:
            # Convert to HWC format for albumentations (256, 256, 6)
            image_np = patch_data.transpose(1, 2, 0)
            
            # Apply augmentation
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
            
            # Convert back to CHW format (6, 256, 256)
            patch_data = image_np.transpose(2, 0, 1)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(patch_data.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image_tensor, label_tensor
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        labels = [self.annotations[f] for f in self.patch_files]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)

def test_augmented_dataset():
    """Test the augmented dataset"""
    patches_dir = "data/processed/patches"
    annotations_file = "data/annotations/automated_labels.json"
    
    if os.path.exists(annotations_file) and os.path.exists(patches_dir):
        dataset = AugmentedMiningDataset(patches_dir, annotations_file, is_training=True)
        print(f"Dataset size: {len(dataset)}")
        
        # Show first sample
        image, label = dataset[0]
        print(f"Image shape: {image.shape}, Label: {label}")
        
        # Show class distribution
        labels = [dataset.annotations[f] for f in dataset.patch_files]
        print(f"Class distribution: {np.bincount(labels)}")
        
        return dataset
    else:
        print("Required files not found!")
        print(f"Annotations exists: {os.path.exists(annotations_file)}")
        print(f"Patches dir exists: {os.path.exists(patches_dir)}")
        return None

if __name__ == "__main__":
    test_augmented_dataset()