#!/usr/bin/env python3
"""
CNN Classifier with Fast Sampling + Progress Printing
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ============== Data Loading (FAST SAMPLING) ==============
def load_data(train_path, test_path, sample_frac=0.5):
    print(f"Loading {sample_frac*100:.1f}% of training data...")

    start = time.time()
    train_df = pd.read_csv(
        train_path,
        skiprows=lambda i: i > 0 and random.random() > sample_frac
    )
    print(f"Training loaded in {time.time() - start:.2f}s")

    print(f"Loading {sample_frac*100:.1f}% of test data...")

    start = time.time()
    test_df = pd.read_csv(
        test_path,
        skiprows=lambda i: i > 0 and random.random() > sample_frac
    )
    print(f"Test loaded in {time.time() - start:.2f}s")

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df


# ============== Preprocessing with Progress ==============
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s\.!\?\'\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_column(df, column_name):
    print(f"Preprocessing {column_name}...")
    processed = []
    start = time.time()

    for i, text in enumerate(df[column_name]):
        processed.append(preprocess_text(text))

        if (i + 1) % 10000 == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)} ({(i+1)/len(df)*100:.1f}%) | "
                  f"{time.time() - start:.1f}s")

    return processed


# ============== Vocabulary ==============
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.word_count = {}

    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1

        idx = 2
        for word, count in self.word_count.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1

        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, text, max_len):
        tokens = text.split()
        indices = [self.word2idx.get(token, 1) for token in tokens]

        if len(indices) >= max_len:
            return indices[:max_len]
        return indices + [0] * (max_len - len(indices))


# ============== Dataset ==============
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels=None, vocab=None, max_len=150):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.reviews[idx], self.max_len)
        if self.labels is not None:
            return torch.tensor(encoded), torch.tensor(self.labels[idx], dtype=torch.float)
        return torch.tensor(encoded)


# ============== CNN Model ==============
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)

        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]

        x = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(x)


# ============== Training ==============
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for i, (texts, labels) in enumerate(loader):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(texts).squeeze(1)
        loss = criterion(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(preds) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(f"  Batch {i+1}/{len(loader)} "
                  f"({(i+1)/len(loader)*100:.1f}%) | "
                  f"Loss: {total_loss/(i+1):.4f} | "
                  f"Acc: {correct/total:.4f}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            preds = model(texts).squeeze(1)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            predicted = (torch.sigmoid(preds) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# ============== Main ==============
def main():

    config = {
        'train_path': 'train.csv',
        'test_path': 'test.csv',
        'sample_frac': 0.5,
        'epochs': 5,
        'batch_size': 64,
        'embed_dim': 100,
        'num_filters': 100,
        'max_len': 150,
        'min_freq': 2,
        'filter_sizes': [2, 3, 4]
    }

    train_df, test_df = load_data(
        config['train_path'],
        config['test_path'],
        config['sample_frac']
    )

    """
    print("\nLabel distribution (full sampled dataset):")
    print(train_df['user_suggestion'].value_counts())
    print("\nProportions:")
    print(train_df['user_suggestion'].value_counts(normalize=True))
    """

    # Preprocess
    train_df['clean'] = preprocess_column(train_df, 'user_review')
    test_df['clean'] = preprocess_column(test_df, 'user_review')

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['clean'],
        train_df['user_suggestion'],
        test_size=0.2,
        stratify=train_df['user_suggestion'],
        random_state=42
    )

    # Vocabulary
    vocab = Vocabulary(config['min_freq'])
    vocab.build_vocab(X_train.tolist())

    # Datasets
    train_ds = ReviewDataset(X_train.tolist(), y_train.values, vocab, config['max_len'])
    val_ds = ReviewDataset(X_val.tolist(), y_val.values, vocab, config['max_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Model
    model = TextCNN(len(vocab.word2idx),
                    config['embed_dim'],
                    config['num_filters'],
                    config['filter_sizes']).to(device)

    # Class imbalance
    pos_weight = torch.tensor([
        len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    ]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    best_acc = 0

    for epoch in range(config['epochs']):
        print(f"\n===== Epoch {epoch+1}/{config['epochs']} =====")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f" Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc

    print("\nBest Validation Accuracy:", best_acc)


if __name__ == "__main__":
    main()