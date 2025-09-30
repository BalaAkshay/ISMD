# src/utils/rule_based_labeling.py
import numpy as np
import os

def rule_based_labels(patches_dir):
    """Use domain knowledge rules to generate labels"""
    labels = {}
    
    for patch_file in os.listdir(patches_dir):
        if patch_file.endswith('.npy'):
            patch_data = np.load(os.path.join(patches_dir, patch_file))
            
            # Rule 1: High NDWI variance might indicate disturbance
            ndwi_variance = np.var(patch_data[4])
            
            # Rule 2: Unusual spectral patterns
            red_mean = np.mean(patch_data[0])  # Red band
            nir_mean = np.mean(patch_data[3])  # NIR band
            
            # Rule 3: Edge density (potential for machinery signatures)
            from scipy import ndimage
            ndwi_edges = ndimage.sobel(patch_data[4])
            edge_density = np.mean(np.abs(ndwi_edges))
            
            # Combine rules
            mining_score = (ndwi_variance * 0.4 + 
                          edge_density * 0.3 + 
                          abs(red_mean - nir_mean) * 0.3)
            
            labels[patch_file] = 1 if mining_score > 0.05 else 0
    
    return labels