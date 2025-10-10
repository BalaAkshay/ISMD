import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
import os
import json
import sys

# Add the src directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.fixed_trainer import MiningClassifier, SimpleMiningDataset

class ThresholdOptimizer:
    def __init__(self, model_path="models/working_mining_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Threshold Optimization on: {self.device}")
        
        # Load model
        self.model = MiningClassifier(in_channels=6, num_classes=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print(f"✅ Model loaded from: {model_path}")
        
        # Load data
        self.dataset = SimpleMiningDataset("data/processed/patches_all", "data/annotations/improved_labels.json")
    
    def get_predictions_with_probabilities(self):
        """Get all predictions with probabilities"""
        from torch.utils.data import DataLoader
        
        all_labels = []
        all_probabilities = []
        
        loader = DataLoader(self.dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probabilities = torch.sigmoid(outputs.squeeze())
                
                all_labels.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, labels, predictions):
        """Calculate precision, recall, f1 for given predictions"""
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def optimize_threshold(self):
        """Find optimal classification threshold"""
        print("🎯 Starting Threshold Optimization...")
        
        # Get predictions with probabilities
        labels, probabilities = self.get_predictions_with_probabilities()
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.02)
        results = []
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            metrics = self.calculate_metrics(labels, predictions)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        # Find optimal thresholds
        best_f1 = max(results, key=lambda x: x['f1'])
        best_balanced = min(results, key=lambda x: abs(x['precision'] - x['recall']))
        high_recall = max([r for r in results if r['precision'] > 0.7], key=lambda x: x['recall'])
        
        print(f"\n📊 OPTIMAL THRESHOLDS FOUND:")
        print(f"🎯 Best F1-Score: {best_f1['threshold']:.2f}")
        print(f"   F1: {best_f1['f1']:.3f}, Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}")
        
        print(f"⚖️  Most Balanced: {best_balanced['threshold']:.2f}")
        print(f"   Precision: {best_balanced['precision']:.3f}, Recall: {best_balanced['recall']:.3f}")
        
        print(f"🔍 High Recall: {high_recall['threshold']:.2f}")
        print(f"   Precision: {high_recall['precision']:.3f}, Recall: {high_recall['recall']:.3f}")
        
        # Plot results
        self.plot_threshold_analysis(results, best_f1, best_balanced, high_recall)
        
        return results, best_f1, best_balanced, high_recall
    
    def plot_threshold_analysis(self, results, best_f1, best_balanced, high_recall):
        """Plot threshold optimization results"""
        thresholds = [r['threshold'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Precision, Recall, F1
        ax1.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        ax1.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        ax1.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
        ax1.axvline(best_f1['threshold'], color='green', linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1["threshold"]:.2f})')
        ax1.axvline(best_balanced['threshold'], color='orange', linestyle='--', alpha=0.7, label=f'Balanced ({best_balanced["threshold"]:.2f})')
        ax1.axvline(high_recall['threshold'], color='purple', linestyle='--', alpha=0.7, label=f'High Recall ({high_recall["threshold"]:.2f})')
        ax1.set_xlabel('Classification Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision, Recall, and F1-Score vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2.plot(thresholds, accuracies, 'purple', linewidth=2)
        ax2.axvline(best_f1['threshold'], color='green', linestyle='--', alpha=0.7)
        ax2.axvline(best_balanced['threshold'], color='orange', linestyle='--', alpha=0.7)
        ax2.axvline(high_recall['threshold'], color='purple', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Classification Threshold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precision-Recall trade-off
        ax3.plot(recalls, precisions, 'black', linewidth=2)
        ax3.scatter(best_f1['recall'], best_f1['precision'], color='green', s=100, label=f'Best F1', zorder=5)
        ax3.scatter(best_balanced['recall'], best_balanced['precision'], color='orange', s=100, label=f'Balanced', zorder=5)
        ax3.scatter(high_recall['recall'], high_recall['precision'], color='purple', s=100, label=f'High Recall', zorder=5)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison table
        ax4.axis('off')
        comparison_data = [
            ['Threshold', 'Best F1', 'Balanced', 'High Recall'],
            ['Value', f"{best_f1['threshold']:.2f}", f"{best_balanced['threshold']:.2f}", f"{high_recall['threshold']:.2f}"],
            ['Precision', f"{best_f1['precision']:.3f}", f"{best_balanced['precision']:.3f}", f"{high_recall['precision']:.3f}"],
            ['Recall', f"{best_f1['recall']:.3f}", f"{best_balanced['recall']:.3f}", f"{high_recall['recall']:.3f}"],
            ['F1-Score', f"{best_f1['f1']:.3f}", f"{best_balanced['f1']:.3f}", f"{high_recall['f1']:.3f}"],
            ['Accuracy', f"{best_f1['accuracy']:.3f}", f"{best_balanced['accuracy']:.3f}", f"{high_recall['accuracy']:.3f}"]
        ]
        
        table = ax4.table(cellText=comparison_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Threshold Comparison')
        
        plt.tight_layout()
        plt.savefig('models/threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Threshold analysis saved to: models/threshold_optimization.png")
    
    def test_optimal_threshold(self, threshold=0.3):
        """Test a specific threshold and show results"""
        labels, probabilities = self.get_predictions_with_probabilities()
        predictions = (probabilities > threshold).astype(int)
        metrics = self.calculate_metrics(labels, predictions)
        
        print(f"\n🧪 TESTING THRESHOLD: {threshold:.2f}")
        print("=" * 40)
        print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"F1-Score: {metrics['f1']:.3f} ({metrics['f1']*100:.1f}%)")
        print(f"\nConfusion Matrix:")
        print(f"True Positives: {metrics['tp']}")
        print(f"False Positives: {metrics['fp']}")
        print(f"False Negatives: {metrics['fn']}")
        print(f"True Negatives: {metrics['tn']}")
        
        return metrics

def main():
    optimizer = ThresholdOptimizer()
    
    # Run full optimization
    results, best_f1, best_balanced, high_recall = optimizer.optimize_threshold()
    
    # Test current threshold (0.5)
    print("\n" + "="*50)
    print("📊 CURRENT THRESHOLD (0.5) PERFORMANCE:")
    print("="*50)
    optimizer.test_optimal_threshold(0.5)
    
    # Test recommended threshold
    print("\n" + "="*50)
    print("🚀 RECOMMENDED THRESHOLD PERFORMANCE:")
    print("="*50)
    optimizer.test_optimal_threshold(best_f1['threshold'])
    
    print(f"\n🎯 RECOMMENDATION: Use threshold {best_f1['threshold']:.2f} for optimal F1-score")
    print(f"   This improves recall from ~67% to ~{best_f1['recall']*100:.1f}% while maintaining good precision")

if __name__ == "__main__":
    main()