import torch 
import torch.nn as nn
from transformers import BertModel
from config import config

class BertClassifier(nn.Module):
    def __init__(self, num_classes= 11, dropout = 0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask= attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def create_model(num_classes = 11):
        print("Creating BERT classifier")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Number of classes: {num_classes}")
        model = BertClassifier(num_classes= num_classes)
        model = model.to(config.DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Model size: ~{total_params * 4 / 1024 ** 2:.1f} MB")
        print("Model created")
        return model
    
if __name__ == "__main__":
    print("Testing")
    model = BertClassifier.create_model(num_classes = config.NUM_CLASSES)
    batch_size = 4
    seq_length = 512
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(config.DEVICE)
    dummy_attention_mask = torch.ones(batch_size, seq_length).to(config.DEVICE)
    print(f"  Input shape: {dummy_input_ids.shape}")
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: [{batch_size}, {config.NUM_CLASSES}]")
    
    if logits.shape == (batch_size, config.NUM_CLASSES):
        print("\nModel test PASSED!")
    else:
        print("\nModel test FAILED!")
