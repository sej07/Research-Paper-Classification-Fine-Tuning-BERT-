"""
Evaluation script for BERT document classification.
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from config import config
from data_loader import load_processed_data, create_dataloaders
from model import BertClassifier


def load_best_model():
    """Load the best saved model."""
    print("Loading best model...")
    
    model = BertClassifier(num_classes=config.NUM_CLASSES)
    
    checkpoint = torch.load(f'{config.MODELS_DIR}/best_model.pth', 
                           map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"Model loaded! (Validation accuracy was: {checkpoint['val_acc']:.4f})")
    
    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and collect predictions.
    
    Returns:
        tuple: (all_predictions, all_labels, all_logits)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print(f"\n Evaluating on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_logits)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CATEGORY_NAMES,
                yticklabels=config.CATEGORY_NAMES)
    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_accuracy(y_true, y_pred, save_path):
    """Plot per-class accuracy."""
    accuracies = []
    
    for i in range(config.NUM_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(config.NUM_CLASSES), accuracies, color='skyblue', edgecolor='navy')
    
    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if accuracies[i] >= 0.8:
            bar.set_color('green')
        elif accuracies[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Category Accuracy on Test Set', fontsize=16, pad=20)
    plt.xticks(range(config.NUM_CLASSES), config.CATEGORY_NAMES, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Average: {np.mean(accuracies):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to: {save_path}")
    plt.close()


def save_evaluation_results(y_true, y_pred, accuracy, report_dict, save_dir):
    """Save evaluation results as JSON."""
    results = {
        'test_accuracy': float(accuracy),
        'total_samples': len(y_true),
        'per_class_metrics': report_dict,
        'category_names': config.CATEGORY_NAMES
    }
    
    results_path = f"{save_dir}/test_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation results saved to: {results_path}")


def main():
    """Main evaluation function."""
    print("MODEL EVALUATION ON TEST SET")
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load tokenizer
    print("\n Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load data
    print("Loading test data...")
    _, _, test_processed = load_processed_data()
    
    # Create test dataloader directly
    print("Creating test dataloader...")
    from torch.utils.data import DataLoader
    from data_loader import ArxivDataset
    
    test_dataset = ArxivDataset(
        abstracts=test_processed['abstract'],
        labels=test_processed['label'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Test loader created: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Load model
    model = load_best_model()
    
    # Evaluate
    
    predictions, labels, logits = evaluate_model(model, test_loader, config.DEVICE)
    
    # Calculate metrics
    accuracy = (predictions == labels).sum() / len(labels)
    
    print("TEST SET RESULTS")
    print(f"\n Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total test samples: {len(labels)}")
    
    # Classification report
    print("\n" )
    print("DETAILED CLASSIFICATION REPORT")
    
    report = classification_report(
        labels, 
        predictions, 
        target_names=config.CATEGORY_NAMES,
        digits=4
    )
    print(report)
    
    # Get report as dictionary for saving
    report_dict = classification_report(
        labels, 
        predictions, 
        target_names=config.CATEGORY_NAMES,
        output_dict=True
    )
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        labels, 
        predictions, 
        f'{config.RESULTS_DIR}/confusion_matrix.png'
    )
    
    # Plot per-class accuracy
    plot_per_class_accuracy(
        labels, 
        predictions, 
        f'{config.RESULTS_DIR}/per_class_accuracy.png'
    )
    
    # Save results
    save_evaluation_results(
        labels, 
        predictions, 
        accuracy, 
        report_dict, 
        config.RESULTS_DIR
    )
    
    print("EVALUATION COMPLETE!")
    print(f"\nResults saved in: {config.RESULTS_DIR}/")
    print("  - confusion_matrix.png")
    print("  - per_class_accuracy.png")
    print("  - test_evaluation.json")


if __name__ == "__main__":
    main()