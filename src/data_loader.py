import re
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from config import config

def extract_abstract(text):
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    
    if not abstract_match:
        return text[:500].strip()
    
    start_idx = abstract_match.end()
    
    # Skip whitespace and potential numbering like "1."
    text_after_abstract = text[start_idx:start_idx+50]
    skip_match = re.match(r'\s*\d+\.\s*', text_after_abstract)
    if skip_match:
        start_idx += skip_match.end()
    
    # End patterns
    end_patterns = [
        r'\n\s*\d+\s*\.?\s*\n',
        r'\n\s*\d+\s+[A-Z][A-Za-z\s]+\n',
        r'\n\s*[IVX]+\.\s+[A-Z]',
        r'\n\s*∗+\s*\n',
        r'\n\s*†+\s*\n',
        r'\n\s*1\s+Introduction',
        r'\n\s*Introduction\s*$',
        r'\n\s*INTRODUCTION\s*$',
        r'\n\s*Keywords?:',
        r'\n\s*Key\s+words?:',
        r'\n\s*Categories and Subject',
    ]
    
    search_text = text[start_idx:start_idx+8000]
    end_idx = len(search_text)
    
    for pattern in end_patterns:
        match = re.search(pattern, search_text)
        if match:
            potential_end = match.start()
            if potential_end < end_idx and potential_end > 50:
                end_idx = potential_end
    
    abstract = search_text[:end_idx].strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    abstract = re.sub(r'[∗†\d]+\s*$', '', abstract)
    
    # Length validation
    if len(abstract) < 50:
        abstract = text[:1000].strip()
        abstract = re.sub(r'\s+', ' ', abstract)
    elif len(abstract) > 4000:
        abstract = abstract[:4000].strip()
    
    return abstract

def download_and_process_dataset():
    print("Downloading arXiv dataset")
    dataset = load_dataset("ccdv/arxiv-classification")
    
    print("Processing abstracts.")
    
    def process_split(dataset_split):
        """Process a single split of the dataset."""
        processed_data = {
            'abstract': [],
            'label': []
        }
        
        for i, example in enumerate(dataset_split):
            abstract = extract_abstract(example['text'])
            processed_data['abstract'].append(abstract)
            processed_data['label'].append(example['label'])
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1} papers...")
        
        return processed_data
    
    train_processed = process_split(dataset['train'])
    val_processed = process_split(dataset['validation'])
    test_processed = process_split(dataset['test'])
    
    return train_processed, val_processed, test_processed

def save_processed_data(train_processed, val_processed, test_processed):
    import os
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    with open(f'{config.PROCESSED_DATA_DIR}/train_processed.pkl', 'wb') as f:
        pickle.dump(train_processed, f)
    
    with open(f'{config.PROCESSED_DATA_DIR}/val_processed.pkl', 'wb') as f:
        pickle.dump(val_processed, f)
    
    with open(f'{config.PROCESSED_DATA_DIR}/test_processed.pkl', 'wb') as f:
        pickle.dump(test_processed, f)
    
    print("Processed data saved!")

def load_processed_data():
    with open(f'{config.PROCESSED_DATA_DIR}/train_processed.pkl', 'rb') as f:
        train_processed = pickle.load(f)
    
    with open(f'{config.PROCESSED_DATA_DIR}/val_processed.pkl', 'rb') as f:
        val_processed = pickle.load(f)
    
    with open(f'{config.PROCESSED_DATA_DIR}/test_processed.pkl', 'rb') as f:
        test_processed = pickle.load(f)
    
    return train_processed, val_processed, test_processed

class ArxivDataset(Dataset):
    def __init__(self, abstracts, labels, tokenizer, max_length=512):
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.abstracts)
    
    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            abstract,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(train_processed, val_processed, test_processed, tokenizer, batch_size=16):
    # Create datasets
    train_dataset = ArxivDataset(
        abstracts=train_processed['abstract'],
        labels=train_processed['label'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = ArxivDataset(
        abstracts=val_processed['abstract'],
        labels=val_processed['label'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_dataset = ArxivDataset(
        abstracts=test_processed['abstract'],
        labels=test_processed['label'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader