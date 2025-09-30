# src/utils/anomaly_detection.py
import numpy as np
import os
from sklearn.ensemble import IsolationForest

def detect_spectral_anomalies(patches_dir):
    """Use unsupervised learning to find anomalous patches"""
    patches_data = []
    patch_files = []
    
    # Load all patches
    for patch_file in os.listdir(patches_dir):
        if patch_file.endswith('.npy'):
            patch_data = np.load(os.path.join(patches_dir, patch_file))
            # Use spectral features for anomaly detection
            spectral_features = [
                np.mean(patch_data[4]),  # NDWI mean
                np.std(patch_data[4]),   # NDWI std
                np.mean(patch_data[5]),  # MNDWI mean
                np.std(patch_data[5]),   # MNDWI std
            ]
            patches_data.append(spectral_features)
            patch_files.append(patch_file)
    
    # Use Isolation Forest for anomaly detection
    clf = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(patches_data)
    
    # Convert to binary labels (-1=anomaly→1, 1=normal→0)
    anomaly_labels = {patch_files[i]: 1 if labels[i] == -1 else 0 
                     for i in range(len(patch_files))}
    
    return anomaly_labels