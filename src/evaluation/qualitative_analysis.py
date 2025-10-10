import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os
import json
import sys
from pathlib import Path

# Add the src directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.fixed_trainer import MiningClassifier, SimpleMiningDataset

class QualitativeAnalyzer:
    def __init__(self, model_path="models/working_mining_model.pth", threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        print(f"🔍 Qualitative Analysis on: {self.device} (Threshold: {threshold})")
        
        # Load model
        self.model = MiningClassifier(in_channels=6, num_classes=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Load data
        self.dataset = SimpleMiningDataset("data/processed/patches_all", "data/annotations/improved_labels.json")
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True)  # Single samples for analysis
    
    def visualize_predictions(self, num_samples=12):
        """Visualize random predictions with ground truth"""
        print(f"🎨 Visualizing {num_samples} random predictions...")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.loader):
                if idx >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probability = torch.sigmoid(output.squeeze()).item()
                prediction = 1 if probability > self.threshold else 0
                true_label = target.item()
                
                # Convert tensor to numpy for visualization
                image_data = data.cpu().numpy()[0]  # Shape: (6, 256, 256)
                
                # Create RGB composite (Bands 3,2,1 - but our bands are different)
                # Assuming bands: [0:B4(Red), 1:B3(Green), 2:B2(Blue), 3:NDVI, 4:NDWI, 5:MNDWI]
                rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)  # R, G, B
                rgb = np.clip(rgb * 3, 0, 1)  # Enhance brightness
                
                # NDWI for water detection
                ndwi = image_data[4]  # Band 4 is NDWI
                
                # MNDWI for improved water detection
                mndwi = image_data[5]  # Band 5 is MNDWI
                
                # Plot
                ax = axes[idx]
                
                # RGB
                ax.imshow(rgb)
                
                # Add title with prediction info
                status = "✅ CORRECT" if prediction == true_label else "❌ WRONG"
                confidence = probability if prediction == 1 else (1 - probability)
                title = f"True: {'Mining' if true_label == 1 else 'No Mining'}\n"
                title += f"Pred: {'Mining' if prediction == 1 else 'No Mining'}\n"
                title += f"Conf: {confidence:.3f}\n{status}"
                
                ax.set_title(title, fontsize=10, 
                           color='green' if prediction == true_label else 'red',
                           fontweight='bold')
                ax.axis('off')
                
                # Add bounding box for mining predictions
                if prediction == 1:
                    # Add red box for mining predictions
                    rect = plt.Rectangle((0, 0), 255, 255, linewidth=4, 
                                       edgecolor='red', facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                
                if prediction == true_label:
                    correct_predictions += 1
                total_samples += 1
        
        plt.tight_layout()
        plt.savefig('models/qualitative_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"📊 Sample Accuracy: {accuracy:.3f} ({correct_predictions}/{total_samples} correct)")
    
    def analyze_false_positives(self, num_samples=8):
        """Analyze false positive predictions (model says mining but it's not)"""
        print(f"🔍 Analyzing False Positives...")
        
        false_positives = []
        
        with torch.no_grad():
            for data, target in self.loader:
                if len(false_positives) >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probability = torch.sigmoid(output.squeeze()).item()
                prediction = 1 if probability > self.threshold else 0
                true_label = target.item()
                
                if prediction == 1 and true_label == 0:  # False positive
                    false_positives.append({
                        'data': data.cpu().numpy()[0],
                        'probability': probability,
                        'true_label': true_label
                    })
        
        if not false_positives:
            print("🎉 No false positives found!")
            return
        
        # Plot false positives
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, fp in enumerate(false_positives[:8]):
            image_data = fp['data']
            
            # RGB composite
            rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)
            rgb = np.clip(rgb * 3, 0, 1)
            
            # NDWI
            ndwi = image_data[4]
            
            ax = axes[idx]
            ax.imshow(rgb)
            ax.set_title(f'False Positive\nConf: {fp["probability"]:.3f}\n(True: No Mining)', 
                        color='red', fontweight='bold')
            ax.axis('off')
            
            # Add red box
            rect = plt.Rectangle((0, 0), 255, 255, linewidth=4, 
                               edgecolor='red', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
        
        # Remove empty subplots
        for idx in range(len(false_positives), 8):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/false_positives_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Found {len(false_positives)} false positives")
    
    def analyze_false_negatives(self, num_samples=8):
        """Analyze false negative predictions (model misses mining)"""
        print(f"🔍 Analyzing False Negatives...")
        
        false_negatives = []
        
        with torch.no_grad():
            for data, target in self.loader:
                if len(false_negatives) >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probability = torch.sigmoid(output.squeeze()).item()
                prediction = 1 if probability > self.threshold else 0
                true_label = target.item()
                
                if prediction == 0 and true_label == 1:  # False negative
                    false_negatives.append({
                        'data': data.cpu().numpy()[0],
                        'probability': probability,
                        'true_label': true_label
                    })
        
        if not false_negatives:
            print("🎉 No false negatives found!")
            return
        
        # Plot false negatives
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, fn in enumerate(false_negatives[:8]):
            image_data = fn['data']
            
            # RGB composite
            rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)
            rgb = np.clip(rgb * 3, 0, 1)
            
            # NDWI
            ndwi = image_data[4]
            
            ax = axes[idx]
            ax.imshow(rgb)
            ax.set_title(f'False Negative\nConf: {fn["probability"]:.3f}\n(True: Mining)', 
                        color='orange', fontweight='bold')
            ax.axis('off')
            
            # Add orange box (missed mining)
            rect = plt.Rectangle((0, 0), 255, 255, linewidth=4, 
                               edgecolor='orange', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
        
        # Remove empty subplots
        for idx in range(len(false_negatives), 8):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/false_negatives_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Found {len(false_negatives)} false negatives")
    
    def analyze_high_confidence_cases(self, num_samples=6):
        """Analyze cases where model is very confident (both correct and incorrect)"""
        print(f"🔍 Analyzing High-Confidence Cases...")
        
        high_confidence_cases = []
        
        with torch.no_grad():
            for data, target in self.loader:
                if len(high_confidence_cases) >= num_samples * 2:  # Get both correct and incorrect
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probability = torch.sigmoid(output.squeeze()).item()
                prediction = 1 if probability > self.threshold else 0
                true_label = target.item()
                
                confidence = probability if prediction == 1 else (1 - probability)
                
                if confidence > 0.9:  # Very confident
                    high_confidence_cases.append({
                        'data': data.cpu().numpy()[0],
                        'probability': probability,
                        'prediction': prediction,
                        'true_label': true_label,
                        'confidence': confidence,
                        'correct': prediction == true_label
                    })
        
        if not high_confidence_cases:
            print("No high-confidence cases found with confidence > 0.9")
            return
        
        # Separate correct and incorrect
        correct_high_conf = [c for c in high_confidence_cases if c['correct']]
        incorrect_high_conf = [c for c in high_confidence_cases if not c['correct']]
        
        # Plot
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # Correct high-confidence cases
        for idx, case in enumerate(correct_high_conf[:6]):
            image_data = case['data']
            rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)
            rgb = np.clip(rgb * 3, 0, 1)
            
            ax = axes[0, idx]
            ax.imshow(rgb)
            ax.set_title(f'✅ High Conf Correct\nTrue: {"Mining" if case["true_label"] == 1 else "No Mining"}\nConf: {case["confidence"]:.3f}', 
                        color='green', fontweight='bold', fontsize=9)
            ax.axis('off')
        
        # Incorrect high-confidence cases
        for idx, case in enumerate(incorrect_high_conf[:6]):
            image_data = case['data']
            rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)
            rgb = np.clip(rgb * 3, 0, 1)
            
            ax = axes[1, idx]
            ax.imshow(rgb)
            ax.set_title(f'❌ High Conf Wrong\nTrue: {"Mining" if case["true_label"] == 1 else "No Mining"}\nConf: {case["confidence"]:.3f}', 
                        color='red', fontweight='bold', fontsize=9)
            ax.axis('off')
        
        # Remove empty subplots
        for idx in range(len(correct_high_conf), 6):
            axes[0, idx].axis('off')
        for idx in range(len(incorrect_high_conf), 6):
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/high_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Found {len(correct_high_conf)} correct high-confidence cases")
        print(f"📊 Found {len(incorrect_high_conf)} incorrect high-confidence cases")
    
    def spectral_analysis_of_errors(self):
        """Analyze spectral characteristics of errors vs correct predictions"""
        print(f"📊 Analyzing Spectral Patterns...")
        
        mining_correct = []
        mining_errors = []
        non_mining_correct = []
        non_mining_errors = []
        
        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probability = torch.sigmoid(output.squeeze()).item()
                prediction = 1 if probability > self.threshold else 0
                true_label = target.item()
                
                image_data = data.cpu().numpy()[0]
                correct = prediction == true_label
                
                # Extract spectral features
                features = {
                    'mean_ndwi': np.mean(image_data[4]),  # NDWI mean
                    'std_ndwi': np.std(image_data[4]),    # NDWI std
                    'mean_mndwi': np.mean(image_data[5]), # MNDWI mean
                    'mean_red': np.mean(image_data[0]),   # Red band mean
                }
                
                if true_label == 1:  # Actual mining
                    if correct:
                        mining_correct.append(features)
                    else:
                        mining_errors.append(features)
                else:  # Actual non-mining
                    if correct:
                        non_mining_correct.append(features)
                    else:
                        non_mining_errors.append(features)
        
        # Create summary
        print(f"\n📈 SPECTRAL ANALYSIS SUMMARY:")
        print(f"Mining - Correct: {len(mining_correct)}, Errors: {len(mining_errors)}")
        print(f"Non-Mining - Correct: {len(non_mining_correct)}, Errors: {len(non_mining_errors)}")
        
        # Plot spectral comparisons
        if mining_correct and mining_errors:
            self.plot_spectral_comparison(mining_correct, mining_errors, "Mining Cases")
        if non_mining_correct and non_mining_errors:
            self.plot_spectral_comparison(non_mining_correct, non_mining_errors, "Non-Mining Cases")
    
    def plot_spectral_comparison(self, correct_cases, error_cases, title):
        """Plot spectral feature comparison between correct and error cases"""
        features = ['mean_ndwi', 'std_ndwi', 'mean_mndwi', 'mean_red']
        feature_names = ['NDWI Mean', 'NDWI Std', 'MNDWI Mean', 'Red Band Mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, (feature, name) in enumerate(zip(features, feature_names)):
            correct_vals = [case[feature] for case in correct_cases]
            error_vals = [case[feature] for case in error_cases]
            
            # Box plot
            data = [correct_vals, error_vals]
            axes[idx].boxplot(data, labels=['Correct', 'Error'])
            axes[idx].set_title(f'{name} - {title}')
            axes[idx].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f'models/spectral_analysis_{title.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Run with optimal threshold (you can change this)
    analyzer = QualitativeAnalyzer(threshold=0.3)
    
    print("🚀 STARTING COMPREHENSIVE QUALITATIVE ANALYSIS")
    print("=" * 60)
    
    # 1. Visualize random predictions
    analyzer.visualize_predictions(num_samples=12)
    
    # 2. Analyze false positives
    analyzer.analyze_false_positives(num_samples=8)
    
    # 3. Analyze false negatives  
    analyzer.analyze_false_negatives(num_samples=8)
    
    # 4. Analyze high-confidence cases
    analyzer.analyze_high_confidence_cases(num_samples=6)
    
    # 5. Spectral analysis
    analyzer.spectral_analysis_of_errors()
    
    print("\n🎉 QUALITATIVE ANALYSIS COMPLETED!")
    print("💾 Check the 'models' folder for generated visualizations")

if __name__ == "__main__":
    main()