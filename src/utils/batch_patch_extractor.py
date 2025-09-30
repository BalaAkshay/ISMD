import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import concurrent.futures

class BatchPatchExtractor:
    def __init__(self, patch_size=256, stride=256, max_workers=4):
        self.patch_size = patch_size
        self.stride = stride
        self.max_workers = max_workers
        
    def process_single_image(self, tiff_path, output_dir):
        """Extract patches from a single GeoTIFF"""
        image_name = Path(tiff_path).stem
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        patches = []
        metadata = []
        
        try:
            with rasterio.open(tiff_path) as src:
                height, width = src.height, src.width
                
                for i in range(0, height - self.patch_size + 1, self.stride):
                    for j in range(0, width - self.patch_size + 1, self.stride):
                        window = Window(j, i, self.patch_size, self.patch_size)
                        patch_data = src.read(window=window)
                        
                        # Skip patches with too many zeros (clouds/no data)
                        if np.mean(patch_data == 0) < 0.3:
                            patch_filename = f"patch_{i}_{j}.npy"
                            patch_path = os.path.join(image_output_dir, patch_filename)
                            
                            np.save(patch_path, patch_data)
                            patches.append(patch_path)
                            
                            metadata.append({
                                'file_path': patch_path,
                                'coordinates': (i, j),
                                'original_file': tiff_path,
                                'image_name': image_name
                            })
            
            print(f"✅ {image_name}: Extracted {len(patches)} patches")
            return patches, metadata
            
        except Exception as e:
            print(f"❌ {image_name}: Error - {e}")
            return [], []
    
    def process_all_images(self, input_dir, output_dir):
        """Process all GeoTIFF files in the input directory"""
        tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.endswith('.tif') or f.endswith('.tiff')]
        
        print(f"📁 Found {len(tiff_files)} GeoTIFF files to process")
        
        all_patches = []
        all_metadata = []
        
        # Use parallel processing for faster extraction
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for tiff_file in tiff_files:
                future = executor.submit(self.process_single_image, tiff_file, output_dir)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                patches, metadata = future.result()
                all_patches.extend(patches)
                all_metadata.extend(metadata)
        
        print(f"🎉 Total patches extracted: {len(all_patches)}")
        return all_patches, all_metadata

def run_batch_extraction():
    """Run batch patch extraction on all satellite images"""
    input_dir = "data/raw/sentinal2"
    output_dir = "data/processed/patches_all"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    extractor = BatchPatchExtractor(patch_size=256, stride=256, max_workers=4)
    patches, metadata = extractor.process_all_images(input_dir, output_dir)
    
    # Save metadata
    import json
    metadata_file = os.path.join(output_dir, "extraction_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved to: {metadata_file}")
    return patches

if __name__ == "__main__":
    run_batch_extraction()