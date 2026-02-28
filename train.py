import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import re
import os
import urllib.request
import tarfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ===================== 1. Basic Configuration and Utility Functions =====================
# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Set plot style
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Device configuration: GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================== 2. Data Loading and Preprocessing =====================
# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    # Clean text: remove special characters, numbers, convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().strip()
    # Tokenization + stopword removal + stemming
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# Download IMDB dataset
def download_imdb(data_dir='./imdb_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    
    # Download only if not exists
    if not os.path.exists(tar_path):
        print("Downloading IMDB dataset (about 80MB), only needed for first run...")
        urllib.request.urlretrieve(url, tar_path)
    
    # Extract tar file
    if not os.path.exists(os.path.join(data_dir, 'aclImdb')):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
    return os.path.join(data_dir, 'aclImdb')

# Custom IMDB Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=500):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Convert text to index sequence
        tokens = self.texts[idx]
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        # Fixed length: pad with <pad> if shorter, truncate if longer
        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices), torch.tensor(self.labels[idx], dtype=torch.float)

# Build vocabulary and save to file (核心新增功能)
def build_vocab(texts, max_size=25000, save_path='vocab.txt'):
    """
    Build vocabulary from training texts and save to file
    :param texts: List of tokenized texts from training set
    :param max_size: Maximum vocabulary size
    :param save_path: Path to save vocabulary file
    :return: Vocabulary dictionary
    """
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    
    # Build vocabulary with special tokens
    vocab = {'<pad>': 0, '<unk>': 1}  # <pad>: padding, <unk>: unknown words
    for idx, (token, _) in enumerate(counter.most_common(max_size-2), 2):
        vocab[token] = idx
    
    # Save vocabulary to file (for inference)
    with open(save_path, 'w', encoding='utf-8') as f:
        for token, idx in vocab.items():
            f.write(f"{token}\t{idx}\n")  # Format: token \t index
    print(f"Vocabulary saved to {save_path}, size: {len(vocab)}")
    
    return vocab

# Load complete dataset and collect statistics
def load_imdb_data(batch_size=64, max_len=500, vocab_save_path='vocab.txt'):
    # Download and read raw data
    imdb_dir = download_imdb()
    texts = []
    labels = []
    text_lengths = []  # For text length distribution
    
    # Load training set
    for label in ['pos', 'neg']:
        dir_path = os.path.join(imdb_dir, 'train', label)
        for fname in os.listdir(dir_path):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                    processed = preprocess_text(f.read())
                    texts.append(processed)
                    labels.append(1 if label == 'pos' else 0)
                    text_lengths.append(len(processed))
    
    # Load test set
    test_texts = []
    test_labels = []
    for label in ['pos', 'neg']:
        dir_path = os.path.join(imdb_dir, 'test', label)
        for fname in os.listdir(dir_path):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                    test_texts.append(preprocess_text(f.read()))
                    test_labels.append(1 if label == 'pos' else 0)
    
    # Split training/validation set (9:1)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED
    )
    
    # Build and save vocabulary (only from training set)
    vocab = build_vocab(train_texts, save_path=vocab_save_path)
    vocab_size = len(vocab)
    pad_idx = vocab['<pad>']
    unk_idx = vocab['<unk>']
    
    # Build Dataset and DataLoader
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_len)
    valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Print dataset statistics
    print(f"Dataset loaded successfully:")
    print(f"  - Training set: {len(train_dataset)} samples")
    print(f"  - Validation set: {len(valid_dataset)} samples")
    print(f"  - Test set: {len(test_dataset)} samples")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Average text length: {sum(text_lengths)/len(text_lengths):.2f} tokens")
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'vocab_size': vocab_size,
        'pad_idx': pad_idx,
        'unk_idx': unk_idx,
        'vocab': vocab,
        'text_lengths': text_lengths,  # For visualization
        'train_texts': train_texts     # For word frequency analysis
    }

# ===================== 3. Model Definitions =====================
# TextCNN Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Multi-size convolutional kernels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # Dropout (prevent overfitting)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch_size, seq_len] → add channel dim [batch_size, 1, seq_len, emb_dim]
        embedded = self.embedding(text).unsqueeze(1)
        # Convolution + ReLU → [batch_size, n_filters, seq_len - fs + 1, 1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # Max pooling → [batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # Concatenate pooling results + Dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        # Classification output (sigmoid for binary classification)
        return torch.sigmoid(self.fc(cat))

# TextLSTM Model
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0)
        # Fully connected layer (double hidden dim if bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch_size, seq_len] → [seq_len, batch_size] (adapt to LSTM input format)
        text = text.permute(1, 0)
        # Embedding layer + Dropout → [seq_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(text))
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate last layer hidden states of bidirectional LSTM
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        # Classification output
        return torch.sigmoid(self.fc(hidden))

# ===================== 4. Training and Evaluation Functions =====================
# Calculate accuracy
def calculate_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Training function
def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()  # Enable training mode (Dropout active)
    
    for text, labels in loader:
        text = text.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()  # Zero gradients
        preds = model(text).squeeze(1)  # Squeeze dimension: [batch_size,1] → [batch_size]
        loss = criterion(preds, labels)  # Calculate loss
        acc = calculate_accuracy(preds, labels)  # Calculate accuracy
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)

# Evaluation function (no gradient calculation)
def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []
    model.eval()  # Enable evaluation mode (Dropout inactive)
    
    with torch.no_grad():  # Disable gradient calculation
        for text, labels in loader:
            text = text.to(device)
            labels = labels.to(device)
            
            preds = model(text).squeeze(1)
            loss = criterion(preds, labels)
            acc = calculate_accuracy(preds, labels)
            
            # Save prediction results (for Precision/Recall/F1 calculation)
            rounded_preds = torch.round(preds).cpu()
            labels_cpu = labels.cpu()
            all_preds.extend(rounded_preds.tolist())
            all_labels.extend(labels_cpu.tolist())
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    # Calculate evaluation metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss / len(loader), epoch_acc / len(loader), precision, recall, f1

# Training process wrapper
def train_model(model, train_loader, valid_loader, optimizer, criterion, n_epochs, model_name):
    best_loss = float('inf')
    train_times = []  # Record training time per epoch
    # Record training process (for visualization)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    print(f"\nStart training {model_name} model...")
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Training + Validation
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, _, _, _ = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        train_times.append(epoch_time)
        epoch_mins = int(epoch_time / 60)
        epoch_secs = int(epoch_time % 60)
        
        # Save best model (minimum validation loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'{model_name}_best.pt')
            print(f"  -> New best model saved (valid loss: {valid_loss:.3f})")
        
        # Record results
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Print log
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'  Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.3f}')
        print(f'  Valid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_acc:.3f}')
    
    # Return training history
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs,
        'best_loss': best_loss,
        'train_times': train_times
    }

# ===================== 5. Enhanced Visualization Functions =====================
# 1. Plot text length distribution
def plot_text_length_distribution(lengths, save_path='text_length_dist.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(sum(lengths)/len(lengths), color='red', linestyle='--', label=f'Average: {sum(lengths)/len(lengths):.2f}')
    plt.axvline(500, color='orange', linestyle='--', label='Max length (500)')
    plt.title('Distribution of Text Lengths (Token Count)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 2. Plot word frequency distribution (top 20 words)
def plot_word_frequency(texts, save_path='word_freq.png'):
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    top_words = counter.most_common(20)
    
    plt.figure(figsize=(12, 6))
    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]
    
    sns.barplot(x=counts, y=words, palette='viridis')
    plt.title('Top 20 Most Frequent Words in Training Set')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 3. Plot training curves (loss + accuracy)
def plot_train_curves(cnn_history, lstm_history, n_epochs, save_path='train_curves.png'):
    epochs = list(range(1, n_epochs+1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    axes[0].plot(epochs, cnn_history['train_losses'], label='CNN Train Loss', marker='o')
    axes[0].plot(epochs, cnn_history['valid_losses'], label='CNN Valid Loss', marker='s')
    axes[0].plot(epochs, lstm_history['train_losses'], label='LSTM Train Loss', marker='^')
    axes[0].plot(epochs, lstm_history['valid_losses'], label='LSTM Valid Loss', marker='*')
    axes[0].set_title('Training/Validation Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, cnn_history['train_accs'], label='CNN Train Accuracy', marker='o')
    axes[1].plot(epochs, cnn_history['valid_accs'], label='CNN Valid Accuracy', marker='s')
    axes[1].plot(epochs, lstm_history['train_accs'], label='LSTM Train Accuracy', marker='^')
    axes[1].plot(epochs, lstm_history['valid_accs'], label='LSTM Valid Accuracy', marker='*')
    axes[1].set_title('Training/Validation Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 4. Plot confusion matrix
def plot_confusion_matrix(model, loader, model_name, save_path=None):
    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for text, labels in loader:
            text = text.to(device)
            labels = labels.to(device)
            preds = model(text).squeeze(1)
            rounded_preds = torch.round(preds).cpu()
            labels_cpu = labels.cpu()
            all_preds.extend(rounded_preds.tolist())
            all_labels.extend(labels_cpu.tolist())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    save_path = f'{model_name}_confusion_matrix.png' if save_path is None else save_path
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print(f'\n{model_name} Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

# 5. Plot model training time comparison
def plot_training_time(cnn_history, lstm_history, n_epochs, save_path='train_time.png'):
    epochs = list(range(1, n_epochs+1))
    cnn_times = cnn_history['train_times']
    lstm_times = lstm_history['train_times']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cnn_times, label='CNN Training Time', marker='o', color='blue')
    plt.plot(epochs, lstm_times, label='LSTM Training Time', marker='s', color='red')
    plt.axhline(sum(cnn_times)/len(cnn_times), color='blue', linestyle='--', label=f'CNN Avg: {sum(cnn_times)/len(cnn_times):.2f}s')
    plt.axhline(sum(lstm_times)/len(lstm_times), color='red', linestyle='--', label=f'LSTM Avg: {sum(lstm_times)/len(lstm_times):.2f}s')
    
    plt.title('Training Time per Epoch (CNN vs LSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 6. Plot model performance heatmap
def plot_performance_heatmap(cnn_metrics, lstm_metrics, save_path='performance_heatmap.png'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    data = {
        'CNN': [cnn_metrics['acc'], cnn_metrics['pre'], cnn_metrics['rec'], cnn_metrics['f1']],
        'LSTM': [lstm_metrics['acc'], lstm_metrics['pre'], lstm_metrics['rec'], lstm_metrics['f1']]
    }
    
    plt.figure(figsize=(8, 4))
    sns.heatmap([data['CNN'], data['LSTM']], annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=metrics, yticklabels=['CNN', 'LSTM'])
    plt.title('Model Performance Heatmap (Test Set)')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 7. Plot model performance comparison bar chart
def plot_model_comparison(cnn_metrics, lstm_metrics, save_path='model_comparison.png'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_vals = [cnn_metrics['acc'], cnn_metrics['pre'], cnn_metrics['rec'], cnn_metrics['f1']]
    lstm_vals = [lstm_metrics['acc'], lstm_metrics['pre'], lstm_metrics['rec'], lstm_metrics['f1']]
    
    x = list(range(len(metrics)))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar([i - width/2 for i in x], cnn_vals, width, label='CNN', color='skyblue')
    rects2 = ax.bar([i + width/2 for i in x], lstm_vals, width, label='LSTM', color='lightcoral')
    
    # Add value labels
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    ax.set_title('CNN vs LSTM Performance Comparison (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ===================== 6. Main Training Flow =====================
if __name__ == '__main__':
    # -------------------------- Hyperparameter Configuration --------------------------
    BATCH_SIZE = 64
    MAX_LEN = 500
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    N_EPOCHS = 10
    
    # CNN hyperparameters
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    
    # LSTM hyperparameters
    HIDDEN_DIM = 128
    N_LAYERS = 2
    BIDIRECTIONAL = True
    
    # Vocabulary save path (关键配置)
    VOCAB_SAVE_PATH = 'vocab.txt'
    
    # -------------------------- Step 1: Load Data and Save Vocabulary --------------------------
    data_config = load_imdb_data(
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN, 
        vocab_save_path=VOCAB_SAVE_PATH  # 自动保存词汇表
    )
    
    # -------------------------- Step 2: Data Visualization --------------------------
    print("\n=== Dataset Visualization ===")
    # Plot text length distribution
    plot_text_length_distribution(data_config['text_lengths'])
    # Plot word frequency
    plot_word_frequency(data_config['train_texts'])
    
    # -------------------------- Step 3: Initialize Models --------------------------
    # CNN model
    cnn_model = TextCNN(
        vocab_size=data_config['vocab_size'],
        embedding_dim=EMBEDDING_DIM,
        n_filters=N_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        pad_idx=data_config['pad_idx']
    ).to(device)
    
    # LSTM model
    lstm_model = TextLSTM(
        vocab_size=data_config['vocab_size'],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=data_config['pad_idx']
    ).to(device)
    
    # Initialize embedding layer: set PAD/UNK vectors to zero
    cnn_model.embedding.weight.data[data_config['unk_idx']] = torch.zeros(EMBEDDING_DIM)
    cnn_model.embedding.weight.data[data_config['pad_idx']] = torch.zeros(EMBEDDING_DIM)
    lstm_model.embedding.weight.data[data_config['unk_idx']] = torch.zeros(EMBEDDING_DIM)
    lstm_model.embedding.weight.data[data_config['pad_idx']] = torch.zeros(EMBEDDING_DIM)
    
    # -------------------------- Step 4: Configure Optimizers and Loss Function --------------------------
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss().to(device)
    
    # -------------------------- Step 5: Train Models --------------------------
    cnn_history = train_model(
        model=cnn_model,
        train_loader=data_config['train_loader'],
        valid_loader=data_config['valid_loader'],
        optimizer=cnn_optimizer,
        criterion=criterion,
        n_epochs=N_EPOCHS,
        model_name='TextCNN'
    )
    
    lstm_history = train_model(
        model=lstm_model,
        train_loader=data_config['train_loader'],
        valid_loader=data_config['valid_loader'],
        optimizer=lstm_optimizer,
        criterion=criterion,
        n_epochs=N_EPOCHS,
        model_name='TextLSTM'
    )
    
    # -------------------------- Step 6: Test Best Models --------------------------
    # Load best CNN model
    cnn_model.load_state_dict(torch.load('TextCNN_best.pt'))
    cnn_test_loss, cnn_test_acc, cnn_test_pre, cnn_test_rec, cnn_test_f1 = evaluate(
        cnn_model, data_config['test_loader'], criterion
    )
    
    # Load best LSTM model
    lstm_model.load_state_dict(torch.load('TextLSTM_best.pt'))
    lstm_test_loss, lstm_test_acc, lstm_test_pre, lstm_test_rec, lstm_test_f1 = evaluate(
        lstm_model, data_config['test_loader'], criterion
    )
    
    # Print test results
    print("\n" + "="*60)
    print("Final Test Set Results:")
    print("="*60)
    print(f"TextCNN | Loss: {cnn_test_loss:.3f} | Accuracy: {cnn_test_acc:.3f} | Precision: {cnn_test_pre:.3f} | Recall: {cnn_test_rec:.3f} | F1: {cnn_test_f1:.3f}")
    print(f"TextLSTM | Loss: {lstm_test_loss:.3f} | Accuracy: {lstm_test_acc:.3f} | Precision: {lstm_test_pre:.3f} | Recall: {lstm_test_rec:.3f} | F1: {lstm_test_f1:.3f}")
    
    # -------------------------- Step 7: Comprehensive Visualization --------------------------
    print("\n=== Model Visualization ===")
    # Training curves
    plot_train_curves(cnn_history, lstm_history, N_EPOCHS)
    # Training time comparison
    plot_training_time(cnn_history, lstm_history, N_EPOCHS)
    # Confusion matrices
    plot_confusion_matrix(cnn_model, data_config['test_loader'], 'TextCNN')
    plot_confusion_matrix(lstm_model, data_config['test_loader'], 'TextLSTM')
    # Performance heatmap
    cnn_metrics = {'acc': cnn_test_acc, 'pre': cnn_test_pre, 'rec': cnn_test_rec, 'f1': cnn_test_f1}
    lstm_metrics = {'acc': lstm_test_acc, 'pre': lstm_test_pre, 'rec': lstm_test_rec, 'f1': lstm_test_f1}
    plot_performance_heatmap(cnn_metrics, lstm_metrics)
    # Performance comparison bar chart
    plot_model_comparison(cnn_metrics, lstm_metrics)
    
    # -------------------------- Final Output --------------------------
    print("\n=== Training Complete ===")
    print(f"✅ Vocabulary saved to: {VOCAB_SAVE_PATH}")
    print(f"✅ Best CNN model saved to: TextCNN_best.pt")
    print(f"✅ Best LSTM model saved to: TextLSTM_best.pt")
    print(f"✅ All visualization plots saved to current directory")
