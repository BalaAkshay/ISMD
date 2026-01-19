import torch
import numpy as np
import json
import os
from src.inference.unet_inference import UNetInference

def find_mining_patches(annotations_file, patches_dir, num_samples=5):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    mining_patches = []
    non_mining_patches = []
    
    for key, data in annotations.items():
        patch_path = os.path.join(patches_dir, key)
        if os.path.exists(patch_path):
            if data['label'] == 1:
                mining_patches.append(patch_path)
            else:
                non_mining_patches.append(patch_path)
    
    print(f"Found {len(mining_patches)} mining patches")
    print(f"Found {len(non_mining_patches)} non-mining patches")
    
    return mining_patches[:num_samples], non_mining_patches[:num_samples]


def main():
    print("\n" + "="*70)
    print("Testing U-Net on MINING vs NON-MINING Patches")
    print("="*70 + "\n")
    
    model_path = "models/unet_mining_model.pth"
    annotations_file = "data/annotations/improved_labels.json"
    patches_dir = "data/processed/patches_all"
    
    inferencer = UNetInference(model_path)
    
    mining_patches, non_mining_patches = find_mining_patches(
        annotations_file, patches_dir, num_samples=3
    )
    
    print("\n" + "="*70)
    print("MINING PATCHES (Expected high probability)")
    print("="*70)
    
    output_dir = "outputs/unet_predictions/mining_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, patch_path in enumerate(mining_patches):
        print(f"\nMining Patch {i+1}: {os.path.basename(patch_path)}")
        patch_data = np.load(patch_path)
        
        save_path = os.path.join(output_dir, f"mining_{i+1}.png")
        change_map, binary_mask = inferencer.visualize_prediction(
            patch_data, save_path=save_path, threshold=0.5
        )
        
        change_pct = (binary_mask.sum() / binary_mask.size) * 100
        print(f"Changed pixels: {change_pct:.2f}%")
        print(f"Mean probability: {change_map.mean():.4f}")
        print(f"Max probability: {change_map.max():.4f}")
    
    print("\n" + "="*70)
    print("NON-MINING PATCHES (Expected low probability)")
    print("="*70)
    
    output_dir = "outputs/unet_predictions/non_mining_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, patch_path in enumerate(non_mining_patches):
        print(f"\nNon-Mining Patch {i+1}: {os.path.basename(patch_path)}")
        patch_data = np.load(patch_path)
        
        save_path = os.path.join(output_dir, f"non_mining_{i+1}.png")
        change_map, binary_mask = inferencer.visualize_prediction(
            patch_data, save_path=save_path, threshold=0.5
        )
        
        change_pct = (binary_mask.sum() / binary_mask.size) * 100
        print(f"Changed pixels: {change_pct:.2f}%")
        print(f"Mean probability: {change_map.mean():.4f}")
        print(f"Max probability: {change_map.max():.4f}")
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
