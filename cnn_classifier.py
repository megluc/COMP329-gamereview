#!/usr/bin/env python3
"""
CNN Classifier for Game Review Sentiment Analysis
Binary classification of game reviews using Convolutional Neural Network

To modify settings, edit the 'config' dictionary in the main() function.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============== Data Loading ==============
def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution: {train_df['user_suggestion'].value_counts().to_dict()}")
    return train_df, test_df


# ============== Text Preprocessing ==============
def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s\.!\?\'\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============== Vocabulary ==============
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_count = {}
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        
        idx = 2
        for word, count in self.word_count.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def encode(self, text, max_len):
        """Encode text to indices"""
        tokens = text.split()
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        if len(indices) >= max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (max_len - len(indices))
        
        return indices


# ============== Dataset ==============
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels=None, vocab=None, max_len=256):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        encoded = self.vocab.encode(review, self.max_len)
        
        if self.labels is not None:
            return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)
        return torch.tensor(encoded, dtype=torch.long)


# ============== CNN Model ==============
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.transpose(1, 2)
        
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# ============== Training Functions ==============
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        texts, labels = batch
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(predictions) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def predict(model, loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                texts = batch[0]
            else:
                texts = batch
            texts = texts.to(device)
            outputs = model(texts).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
            predictions.extend(preds)
    
    return predictions


# ============== Main ==============
def main():
    # Configuration - adjust these values as needed
    # To change settings, simply edit the values below and run the script
    config = {
        'train_path': 'train.csv',          # Path to training data
        'test_path': 'test.csv',            # Path to test data
        'output_path': 'cnn_predictions.csv', # Where to save predictions
        'epochs': 10,                        # Number of training epochs
        'batch_size': 32,                   # Batch size for training
        'embed_dim': 64,                    # Word embedding dimension
        'num_filters': 50,                  # Filters per convolution size
        'max_len': 128,                     # Maximum sequence length
        'min_freq': 5,                      # Minimum word frequency for vocab
        'filter_sizes': [2, 3, 4],          # Convolution filter sizes
        'sample_size': 0.2                  # Fraction of data to use (0.1 = 10%)
    }
    
    print("CNN Classifier Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    print("=" * 50)
    print("Loading data...")
    train_df, test_df = load_data(config['train_path'], config['test_path'])
    
    # Sample data if requested
    if config['sample_size'] < 1.0:
        print(f"\nSampling {config['sample_size']*100:.1f}% of the data...")
        train_sample_size = int(len(train_df) * config['sample_size'])
        test_sample_size = int(len(test_df) * config['sample_size'])
        
        train_df = train_df.sample(n=train_sample_size, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Sampled training samples: {len(train_df)}")
        print(f"Sampled test samples: {len(test_df)}")
    
    # Preprocess
    print("\nPreprocessing text...")
    train_df['clean_review'] = train_df['user_review'].apply(preprocess_text)
    test_df['clean_review'] = test_df['user_review'].apply(preprocess_text)
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=config['min_freq'])
    vocab.build_vocab(train_df['clean_review'].tolist() + test_df['clean_review'].tolist())
    
    # Prepare data
    X = train_df['clean_review'].tolist()
    y = train_df['user_suggestion'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, vocab, config['max_len'])
    val_dataset = ReviewDataset(X_val, y_val, vocab, config['max_len'])
    test_dataset = ReviewDataset(test_df['clean_review'].tolist(), None, vocab, config['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model
    model = TextCNN(
        vocab_size=len(vocab.word2idx),
        embed_dim=config['embed_dim'],
        num_filters=config['num_filters'],
        filter_sizes=config['filter_sizes'],
        output_dim=1,
        dropout=0.5
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Filter sizes: {config['filter_sizes']}")
    print(f"Embedding dim: {config['embed_dim']}, Filters per size: {config['num_filters']}")
    print(f"Vocab size: {len(vocab.word2idx)}, Max length: {config['max_len']}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        print()
    
    # Load best model and predict
    print("=" * 50)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_state)
    
    predictions = predict(model, test_loader, device)
    print(f"Total predictions: {len(predictions)}")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    # Save results
    submission = pd.DataFrame({
        'review_id': test_df['review_id'],
        'user_suggestion': predictions
    })
    submission.to_csv(config['output_path'], index=False)
    print(f"\nPredictions saved to '{config['output_path']}'")
    print(submission.head(10))


if __name__ == '__main__':
    main()