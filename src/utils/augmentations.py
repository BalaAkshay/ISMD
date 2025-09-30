import albumentations as A
import numpy as np

def get_training_augmentations():
    """Geometric augmentations only (as recommended by Gemini)"""
    return A.Compose([
        # Safe geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        
        # Safe pixel-level transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.3
        ),
        
        # Add some noise to improve robustness
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ])

def get_validation_augmentations():
    """Just basic processing for validation"""
    return A.Compose([])

# Test the augmentation
if __name__ == "__main__":
    aug = get_training_augmentations()
    print("✅ Augmentation pipeline ready:")
    print(aug)