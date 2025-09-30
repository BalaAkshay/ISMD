# src/utils/change_detection.py
import numpy as np
import os
from pathlib import Path

def generate_change_labels(patches_dir, output_dir):
    """Generate automatic labels based on temporal changes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group patches by location (extract coordinates from filename)
    patch_groups = {}
    
    for patch_file in os.listdir(patches_dir):
        if patch_file.endswith('.npy'):
            # Extract coordinates from filename: S2_Yamuna_2023_01_patch_0_256.npy
            parts = patch_file.split('_')
            location_key = f"{parts[-2]}_{parts[-1].replace('.npy', '')}"  # "0_256"
            
            if location_key not in patch_groups:
                patch_groups[location_key] = []
            
            patch_groups[location_key].append(patch_file)
    
    # For each location, compare consecutive months to detect changes
    change_labels = {}
    
    for location, patches in patch_groups.items():
        patches.sort()  # Sort by date
        if len(patches) >= 2:
            # Compare first and last month for that location
            patch1 = np.load(os.path.join(patches_dir, patches[0]))
            patch2 = np.load(os.path.join(patches_dir, patches[-1]))
            
            # Calculate change magnitude using NDWI difference
            ndwi_change = np.abs(patch2[4] - patch1[4])  # Band 4 is NDWI
            
            # Label as "change" if significant NDWI difference
            change_score = np.mean(ndwi_change)
            change_labels[location] = 1 if change_score > 0.1 else 0  # Threshold
    
    return change_labels