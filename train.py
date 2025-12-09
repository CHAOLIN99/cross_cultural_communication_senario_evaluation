import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ===========================
# 1. CONFIGURATION
# ===========================
class Config:
    MODEL_NAME = "openai-community/gpt2"
    DATASET_PATH = "scenarios.csv"  # Update with your actual path
    EMBEDDING_CACHE = "embeddings_cache.pt"
    
    # Training parameters
    BATCH_SIZE = 16
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DROPOUT = 0.3
    
    # Data split
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

config = Config()
print(f"Using device: {config.DEVICE}")

# ===========================
# 2. LOAD AND SPLIT DATASET
# ===========================
def load_and_split_data(csv_path):
    """
    Load dataset and split into train/val/test sets.
    
    Expected CSV columns:
    - 'scenario_text' or 'text': the scenario description
    - 'label' or 'violation': binary label (0=no violation, 1=violation)
    """
    print(f"\nLoading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Identify text and label columns (adjust if your columns have different names)
    text_col = 'scenario_text' if 'scenario_text' in df.columns else 'text'
    label_col = 'label' if 'label' in df.columns else 'violation'
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=config.RANDOM_SEED,
        stratify=df[label_col]
    )
    
    # Second split: 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=temp_df[label_col]
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df, text_col, label_col

# ===========================
# 3. EXTRACT EMBEDDINGS
# ===========================
def extract_embeddings(texts, model, tokenizer, device):
    """
    Extract embeddings from GPT-2 model.
    Uses the last token's hidden state as the embedding.
    """
    print(f"\nExtracting embeddings for {len(texts)} samples...")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting embeddings"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Get model outputs
            outputs = model(**inputs, output_hidden_states=True)
            
            # Use last token's hidden state as embedding
            # Shape: (batch_size, seq_len, hidden_dim)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get last token embedding
            embedding = last_hidden_state[0, -1, :].cpu()
            embeddings.append(embedding)
    
    return torch.stack(embeddings)

def get_or_compute_embeddings(train_df, val_df, test_df, text_col, cache_path):
    """
    Load cached embeddings or compute them if cache doesn't exist.
    """
    try:
        print(f"\nTrying to load cached embeddings from {cache_path}...")
        cache = torch.load(cache_path)
        print("Successfully loaded cached embeddings!")
        return cache['train'], cache['val'], cache['test']
    except FileNotFoundError:
        print("Cache not found. Computing embeddings...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Extract embeddings
        train_embeddings = extract_embeddings(
            train_df[text_col].tolist(), model, tokenizer, config.DEVICE
        )
        val_embeddings = extract_embeddings(
            val_df[text_col].tolist(), model, tokenizer, config.DEVICE
        )
        test_embeddings = extract_embeddings(
            test_df[text_col].tolist(), model, tokenizer, config.DEVICE
        )
        
        # Cache embeddings
        print(f"\nSaving embeddings to {cache_path}...")
        torch.save({
            'train': train_embeddings,
            'val': val_embeddings,
            'test': test_embeddings
        }, cache_path)
        
        return train_embeddings, val_embeddings, test_embeddings

# ===========================
# 4. MLP PROBE MODEL
# ===========================
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLPProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)

# ===========================
# 5. TRAINING FUNCTION
# ===========================
def train_model(model, train_loader, val_loader, config):
    """
    Train the MLP probe with early stopping.
    """
    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(embeddings)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_probe.pt')
            patience_counter = 0
            print(f"  → New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_accuracies

# ===========================
# 6. EVALUATION FUNCTION
# ===========================
def evaluate_model(model, test_loader, config):
    """
    Evaluate model on test set and compute all metrics.
    """
    model.load_state_dict(torch.load('best_probe.pt'))
    model.to(config.DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(config.DEVICE)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    # Calculate specificity (True Negative Rate)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    return results, all_preds, all_labels

# ===========================
# 7. VISUALIZATION
# ===========================
def plot_training_curves(train_losses, val_accuracies):
    """Plot training loss and validation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\nTraining curves saved to 'training_curves.png'")

def plot_confusion_matrix(cm, classes=['No Violation', 'Violation']):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to 'confusion_matrix.png'")

# ===========================
# 8. MAIN EXECUTION
# ===========================
def main():
    print("="*60)
    print("Cross-Cultural Norm Violation Detection - MLP Probe")
    print("="*60)
    
    # Load and split data
    train_df, val_df, test_df, text_col, label_col = load_and_split_data(config.DATASET_PATH)
    
    # Get or compute embeddings
    train_embeddings, val_embeddings, test_embeddings = get_or_compute_embeddings(
        train_df, val_df, test_df, text_col, config.EMBEDDING_CACHE
    )
    
    print(f"\nEmbedding dimensions: {train_embeddings.shape[1]}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        train_embeddings,
        torch.tensor(train_df[label_col].values, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        val_embeddings,
        torch.tensor(val_df[label_col].values, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        test_embeddings,
        torch.tensor(test_df[label_col].values, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    input_dim = train_embeddings.shape[1]
    probe = MLPProbe(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
    )
    
    print(f"\nModel architecture:")
    print(probe)
    print(f"\nTotal parameters: {sum(p.numel() for p in probe.parameters())}")
    
    # Train model
    train_losses, val_accuracies = train_model(probe, train_loader, val_loader, config)
    
    # Plot training curves
    plot_training_curves(train_losses, val_accuracies)
    
    # Evaluate on test set
    results, predictions, true_labels = evaluate_model(probe, test_loader, config)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print(f"F1 Score:    {results['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Save results to JSON
    results_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in results.items() if k != 'confusion_matrix'}
    results_to_save['confusion_matrix'] = results['confusion_matrix'].tolist()
    
    with open('results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print("\nResults saved to 'results.json'")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()