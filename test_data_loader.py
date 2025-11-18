import sys
sys.path.append('src')  

from transformers import BertTokenizer
from data_loader import load_processed_data, create_dataloaders
from config import config

print("TESTING DATA LOADER")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
print(f"Tokenizer loaded: {config.MODEL_NAME}")

# Load processed data
print("\n Loading processed data...")
train_processed, val_processed, test_processed = load_processed_data()
print(f"Train: {len(train_processed['abstract'])} samples")
print(f"Val: {len(val_processed['abstract'])} samples")
print(f"Test: {len(test_processed['abstract'])} samples")

# Create dataloaders
print("\nCreating dataloaders...")
train_loader, val_loader, test_loader = create_dataloaders(
    train_processed, 
    val_processed, 
    test_processed,
    tokenizer,
    batch_size=config.BATCH_SIZE
)

# Test by getting one batch
print("\nTesting one batch from train_loader...")
batch = next(iter(train_loader))
print(f"  input_ids shape: {batch['input_ids'].shape}")
print(f"  attention_mask shape: {batch['attention_mask'].shape}")
print(f"  labels shape: {batch['label'].shape}")

print("\nALL TESTS PASSED! Data loader is working correctly!")
print(f"Device: {config.DEVICE}")