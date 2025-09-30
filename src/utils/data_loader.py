import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SatelliteImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        with rasterio.open(img_path) as src:
            image = src.read()  # Read all bands
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'filename': self.image_files[idx]
            }
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, metadata
    
    def visualize_sample(self, idx, band_indices=[3, 2, 1]):  # Default to RGB
        image, metadata = self[idx]
        
        plt.figure(figsize=(12, 4))
        
        # RGB composite
        plt.subplot(1, 3, 1)
        rgb = np.stack([image[band_indices[0]], 
                       image[band_indices[1]], 
                       image[band_indices[2]]], axis=-1)
        rgb = np.clip(rgb * 3, 0, 1)  # Enhance brightness
        plt.imshow(rgb)
        plt.title(f"RGB - {metadata['filename']}")
        plt.axis('off')
        
        # NDWI
        plt.subplot(1, 3, 2)
        ndwi = image[4]  # Assuming NDWI is band 4
        plt.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
        plt.title("NDWI (Water Index)")
        plt.axis('off')
        plt.colorbar()
        
        # MNDWI
        plt.subplot(1, 3, 3)
        mndwi = image[5]  # Assuming MNDWI is band 5
        plt.imshow(mndwi, cmap='RdYlBu', vmin=-1, vmax=1)
        plt.title("MNDWI (Modified Water Index)")
        plt.axis('off')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

def test_data_loader():
    """Test the data loader once files are downloaded"""
    data_dir = "../../data/raw/sentinal2"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        dataset = SatelliteImageDataset(data_dir)
        print(f"Dataset loaded with {len(dataset)} images")
        dataset.visualize_sample(0)
    else:
        print("Download satellite images first")