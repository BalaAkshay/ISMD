import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import os
import json
import sys

# Add the src directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the trainer
from training.fixed_trainer import MiningClassifier, SimpleMiningDataset

print("Starting Comprehensive Model Evaluation...")

class ModelEvaluator:
    def __init__(self, model_path="models/working_mining_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Evaluation on: {self.device}")
        
        # Load model - handle both checkpoint and raw model files
        self.model = MiningClassifier(in_channels=6, num_classes=1).to(self.device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check if it's a checkpoint dict or raw model weights
        if 'model_state_dict' in checkpoint:
            # It's a training checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from training checkpoint")
        else:
            # It's raw model weights
            self.model.load_state_dict(checkpoint)
            print("Loaded raw model weights")
            
        self.model.eval()
        print(f"Model loaded from: {model_path}")
        
        # Load data
        self.dataset = SimpleMiningDataset("data/processed/patches_all", "data/annotations/improved_labels.json")
        
    def get_predictions(self):
        """Get all predictions and true labels"""
        from torch.utils.data import DataLoader
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        loader = DataLoader(self.dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = (probabilities > 0.48).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def comprehensive_metrics(self, predictions, labels, probabilities):
        """Calculate comprehensive evaluation metrics"""
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)
        
        # Basic metrics
        accuracy = np.mean(predictions == labels)
        print(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📈 Confusion Matrix:")
        print(f"True Negatives (Non-Mining): {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}") 
        print(f"True Positives (Mining): {tp}")
        
        # Detailed metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🎯 Detailed Metrics:")
        print(f"Precision: {precision:.4f} (How many predicted mining are actually mining)")
        print(f"Recall: {recall:.4f} (How many actual mining are detected)")
        print(f"F1-Score: {f1:.4f} (Balance of precision and recall)")
        
        # Class distribution
        mining_count = np.sum(labels == 1)
        non_mining_count = np.sum(labels == 0)
        total_count = len(labels)
        
        print(f"\n📋 Dataset Overview:")
        print(f"Total patches: {total_count}")
        print(f"Mining patches: {mining_count} ({mining_count/total_count*100:.1f}%)")
        print(f"Non-mining patches: {non_mining_count} ({non_mining_count/total_count*100:.1f}%)")
        
        return cm, precision, recall, f1
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Mining', 'Mining'],
                   yticklabels=['Non-Mining', 'Mining'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, labels, probabilities):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 ROC AUC Score: {roc_auc:.4f}")
        return roc_auc
    
    def plot_precision_recall_curve(self, labels, probabilities):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig('models/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Precision-Recall AUC Score: {pr_auc:.4f}")
        return pr_auc
    
    def analyze_misclassifications(self, predictions, labels, probabilities, top_k=10):
        """Analyze the most confident misclassifications"""
        misclassified_indices = np.where(predictions != labels)[0]
        
        if len(misclassified_indices) == 0:
            print("🎉 No misclassifications found!")
            return
        
        misclassified_probs = probabilities[misclassified_indices]
        
        # Get top-k most confident errors
        top_indices = misclassified_indices[np.argsort(misclassified_probs)[-top_k:]]
        
        print(f"\n🔍 Top {len(top_indices)} Most Confident Misclassifications:")
        for i, idx in enumerate(top_indices):
            true_label = labels[idx]
            pred_label = predictions[idx]
            confidence = probabilities[idx]
            
            error_type = "False Positive" if pred_label == 1 else "False Negative"
            print(f"  {i+1}. {error_type} - True: {true_label}, Pred: {pred_label}, Confidence: {confidence:.4f}")
    
    def generate_report(self, predictions, labels, probabilities, cm, precision, recall, f1, roc_auc, pr_auc):
        """Generate comprehensive evaluation report"""
        report = {
            'overall_accuracy': float(np.mean(predictions == labels)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'dataset_size': len(labels),
            'mining_samples': int(np.sum(labels == 1)),
            'non_mining_samples': int(np.sum(labels == 0)),
            'misclassifications': int(np.sum(predictions != labels))
        }
        
        # Save report
        os.makedirs('models', exist_ok=True)
        with open('models/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Evaluation report saved to: models/evaluation_report.json")
        return report
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting Comprehensive Model Evaluation...")
        
        # Get predictions
        predictions, labels, probabilities = self.get_predictions()
        
        # Calculate metrics
        cm, precision, recall, f1 = self.comprehensive_metrics(predictions, labels, probabilities)
        
        # Generate plots
        self.plot_confusion_matrix(cm)
        roc_auc = self.plot_roc_curve(labels, probabilities)
        pr_auc = self.plot_precision_recall_curve(labels, probabilities)
        
        # Analyze errors
        self.analyze_misclassifications(predictions, labels, probabilities)
        
        # Generate report
        report = self.generate_report(predictions, labels, probabilities, cm, precision, recall, f1, roc_auc, pr_auc)
        
        print("\n🎉 EVALUATION COMPLETED!")
        print("=" * 50)
        print("Key Findings:")
        print(f"  • Model accurately classifies {report['overall_accuracy']*100:.1f}% of patches")
        print(f"  • Detects {report['recall']*100:.1f}% of actual mining activities") 
        print(f"  • {report['precision']*100:.1f}% of mining predictions are correct")
        print(f"  • ROC AUC of {report['roc_auc']:.4f} indicates excellent discrimination")
        
        return report

def main():
    evaluator = ModelEvaluator()
    report = evaluator.run_complete_evaluation()
    
    # Print final summary
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {report['overall_accuracy']*100:.2f}%")
    print(f"Precision: {report['precision']*100:.2f}%")
    print(f"Recall: {report['recall']*100:.2f}%") 
    print(f"F1-Score: {report['f1_score']*100:.2f}%")
    print(f"ROC AUC: {report['roc_auc']:.4f}")
    print(f"Misclassifications: {report['misclassifications']}/{report['dataset_size']}")

if __name__ == "__main__":
    main()