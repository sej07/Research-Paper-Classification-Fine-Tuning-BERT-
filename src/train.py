import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os 
import json
from datetime import datetime
from config import config
from data_loader import load_processed_data, create_dataloaders
from model import create_model

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"Random seed set to {seed}")

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    print(f"\nStarting training on {len(train_loader)} batches...")
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct_predictions / total_samples
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    print(f"Training complete: Loss={avg_loss:.4f}, Acc={accuracy:.4f}\n")
    
    return avg_loss, accuracy
    
def eval_epoch(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nStarting validation on {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct_predictions / total_samples
                print(f"  Batch [{batch_idx + 1}/{len(val_loader)}] "
                      f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    print(f"Validation complete: Loss={avg_loss:.4f}, Acc={accuracy:.4f}\n")
    
    return avg_loss, accuracy

def train(model, train_loader, val_loader, num_epochs=3):
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0

    print("Starting Training")
    print(f"Total epochs: {num_epochs}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config.WARMUP_STEPS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Device: {config.DEVICE}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE
        )
        
        val_loss, val_acc = eval_epoch(model, val_loader, config.DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"\nNew best validation accuracy! Saving model")
            
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f'{config.MODELS_DIR}/best_model.pth')
            
            print(f"Model saved! Best Val Acc: {best_val_acc:.4f}")
    
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history

def save_results(history):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{config.RESULTS_DIR}/training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")

def main():
    print("Training BERT document classifier")

    set_seed(config.SEED)

    print("\n Loading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    print("Tokenizer loaded")
    
    # Load data
    print("\nLoading processed data")
    train_processed, val_processed, test_processed = load_processed_data()
    print(" Data loaded")
    
    # Create dataloaders
    print("\nCreating dataloaders")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_processed,
        val_processed,
        test_processed,
        tokenizer,
        batch_size=config.BATCH_SIZE
    )
    
    print("\n Creating model")
    model = create_model(num_classes=config.NUM_CLASSES)
    
    history = train(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS)
    save_results(history)
    
    print("\nDone")

if __name__ == "__main__":
    main()

