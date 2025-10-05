import os
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
import albumentations as A

class PatchExtractor:
    def __init__(self, patch_size=256, stride=256):
        self.patch_size = patch_size
        self.stride = stride
        
    def extract_patches_from_tiff(self, tiff_path, output_dir):
        """Extract patches from a GeoTIFF file"""
        os.makedirs(output_dir, exist_ok=True)
        
        with rasterio.open(tiff_path) as src:
            height, width = src.height, src.width
            patches = []
            metadata = []
            
            for i in range(0, height - self.patch_size + 1, self.stride):
                for j in range(0, width - self.patch_size + 1, self.stride):
                    # Extract patch
                    window = Window(j, i, self.patch_size, self.patch_size)
                    patch_data = src.read(window=window)
                    
                    # Skip patches with too many zeros (clouds/no data)
                    if np.mean(patch_data == 0) < 0.3:  # Less than 30% zeros
                        patch_filename = f"{os.path.basename(tiff_path).replace('.tif', '')}_patch_{i}_{j}.npy"
                        patch_path = os.path.join(output_dir, patch_filename)
                        
                        # Save patch
                        np.save(patch_path, patch_data)
                        patches.append(patch_path)
                        
                        # Store metadata
                        metadata.append({
                            'file_path': patch_path,
                            'coordinates': (i, j),
                            'original_file': tiff_path
                        })
            
            return patches, metadata

def test_patch_extraction():
    """Test the patch extraction on one file"""
    extractor = PatchExtractor(patch_size=256, stride=256)
    sample_file = "data/raw/sentinal2/S2_Yamuna_2023_01.tif"
    output_dir = "data/processed/patches"
    
    if os.path.exists(sample_file):
        patches, metadata = extractor.extract_patches_from_tiff(sample_file, output_dir)
        print(f"Extracted {len(patches)} patches from {sample_file}")
        return patches
    else:
        print("Sample file not found - run data pipeline first")
        return []

if __name__ == "__main__":
    test_patch_extraction()