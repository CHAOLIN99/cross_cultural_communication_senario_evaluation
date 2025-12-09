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
    
    # Training parameters (auto-adjusted based on dataset size)
    BATCH_SIZE = 16  # Will be adjusted if dataset is small
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DROPOUT = 0.3
    
    # Data split ratios (configurable)
    TRAIN_RATIO = 0.70  # 70% for training
    VAL_RATIO = 0.15    # 15% for validation
    TEST_RATIO = 0.15   # 15% for testing
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    @classmethod
    def adjust_for_dataset_size(cls, dataset_size):
        """
        Automatically adjust hyperparameters based on dataset size.
        """
        print(f"\n📊 Auto-adjusting hyperparameters for dataset size: {dataset_size}")
        
        # Adjust batch size for small datasets
        if dataset_size < 100:
            cls.BATCH_SIZE = 4
            print(f"  → Batch size: {cls.BATCH_SIZE} (small dataset)")
        elif dataset_size < 300:
            cls.BATCH_SIZE = 8
            print(f"  → Batch size: {cls.BATCH_SIZE} (medium dataset)")
        else:
            cls.BATCH_SIZE = 16
            print(f"  → Batch size: {cls.BATCH_SIZE} (large dataset)")
        
        # Adjust epochs for very small datasets
        if dataset_size < 50:
            cls.NUM_EPOCHS = 100
            print(f"  → Epochs: {cls.NUM_EPOCHS} (more training for small dataset)")
        
        # Adjust hidden dimension for large datasets
        if dataset_size > 2000:
            cls.HIDDEN_DIM = 256
            print(f"  → Hidden dim: {cls.HIDDEN_DIM} (larger network for big dataset)")

config = Config()
print(f"Using device: {config.DEVICE}")

# ===========================
# 2. LOAD AND SPLIT DATASET
# ===========================
def explore_dataset(df, text_col, label_col):
    """
    Print detailed statistics about the dataset.
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    # Basic stats
    print(f"\nTotal samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    # Show demographic columns if available
    demographic_cols = ['gender1', 'gender2', 'nationality1', 'nationality2', 
                       'age_group1', 'age_group2']
    available_demographics = [col for col in demographic_cols if col in df.columns]
    
    if available_demographics:
        print(f"\nDemographic features available: {available_demographics}")
        
        # Show unique values for key demographics
        if 'nationality1' in df.columns:
            nationalities = pd.concat([df['nationality1'], df['nationality2']]).unique()
            print(f"  Nationalities: {len(nationalities)} unique ({', '.join(nationalities[:5])}...)")
        
        if 'gender1' in df.columns:
            genders = df['gender1'].value_counts()
            print(f"  Gender distribution (sender):")
            for gender, count in genders.items():
                print(f"    {gender}: {count} ({count/len(df)*100:.1f}%)")
    
    # Label distribution
    print(f"\nLabel distribution:")
    label_counts = df[label_col].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        label_name = "Violation" if label == 1 else "No Violation"
        print(f"  {label_name} ({label}): {count} samples ({percentage:.1f}%)")
    
    # Class balance check
    if len(label_counts) == 2:
        imbalance_ratio = label_counts.max() / label_counts.min()
        if imbalance_ratio > 1.5:
            print(f"  ⚠️  Dataset is imbalanced (ratio: {imbalance_ratio:.2f}:1)")
        else:
            print(f"  ✓ Dataset is balanced (ratio: {imbalance_ratio:.2f}:1)")
    
    # Text length statistics
    text_lengths = df[text_col].str.len()
    print(f"\nText length statistics:")
    print(f"  Mean: {text_lengths.mean():.0f} characters")
    print(f"  Median: {text_lengths.median():.0f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")
    
    # Check for demographic placeholders
    sample_text = df[text_col].iloc[0]
    has_placeholders = '<<gender' in sample_text or '<<nationality' in sample_text
    
    if has_placeholders:
        print(f"\n✓ Scenarios use placeholder format: <<gender1>> (<<nationality1>>)")
    
    print("\nSample scenarios:")
    print("-" * 60)
    # Show one violation and one non-violation example
    for label in [0, 1]:
        sample = df[df[label_col] == label].iloc[0] if len(df[df[label_col] == label]) > 0 else None
        if sample is not None:
            label_name = "VIOLATION" if label == 1 else "NO VIOLATION"
            print(f"\nExample {label_name}:")
            scenario_text = sample[text_col]
            print(scenario_text[:350] if len(scenario_text) > 350 else scenario_text)
            if len(scenario_text) > 350:
                print("... (truncated)")
            
            # Show demographics if available
            if 'nationality1' in df.columns:
                print(f"Demographics: {sample['gender1']} ({sample['nationality1']}) ↔ "
                      f"{sample['gender2']} ({sample['nationality2']})")
    print("-" * 60)
    print("="*60)

def load_and_split_data(csv_path):
    """
    Load dataset and split into train/val/test sets.
    
    Expected CSV structure (Weng et al. format):
    - 'scenario': the conversation with demographic placeholders
    - 'result': 'Yes' or 'No' indicating violation
    - Additional columns: gender1, gender2, nationality1, nationality2, etc.
    """
    print(f"\nLoading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Available columns: {df.columns.tolist()}")
    
    # Identify text column (try common variations)
    text_col = None
    for col in ['scenario', 'scenario_text', 'text', 'conversation']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
    
    # Identify label column and convert to binary
    label_col = None
    for col in ['result', 'violation', 'label', 'has_violation', 'is_violation']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
    
    # Convert 'Yes'/'No' to binary labels (1/0)
    if df[label_col].dtype == 'object':
        print(f"\nConverting '{label_col}' from text to binary labels...")
        # Map Yes->1 (violation), No->0 (no violation)
        label_map = {'Yes': 1, 'yes': 1, 'YES': 1, 
                     'No': 0, 'no': 0, 'NO': 0}
        df['binary_label'] = df[label_col].map(label_map)
        
        # Check for unmapped values
        unmapped = df['binary_label'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} labels could not be mapped. Unique values: {df[label_col].unique()}")
            df = df.dropna(subset=['binary_label'])
        
        label_col = 'binary_label'
    
    print(f"\nUsing text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst scenario preview:")
    print(f"{df[text_col].iloc[0][:300]}...")
    print(f"\nLabel distribution:\n{df[label_col].value_counts()}")
    
    # Explore the dataset
    explore_dataset(df, text_col, label_col)
    
    # Check if we have enough samples for stratification
    min_class_count = df[label_col].value_counts().min()
    print(f"\nSmallest class has {min_class_count} samples")
    
    # Calculate minimum samples needed for stratified split
    # We need at least 2 samples per class in each split
    min_samples_needed = 2 / config.TEST_SIZE  # ~14 samples minimum
    
    if min_class_count < min_samples_needed:
        print(f"⚠️  Warning: Small dataset. Stratification may not be possible.")
        print(f"   Proceeding with random split (no stratification)...")
        stratify_param = None
    else:
        stratify_param = df[label_col]
        print("✓ Sufficient samples for stratified split")
    
    # First split: 70% train, 30% temp
    try:
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=config.RANDOM_SEED,
            stratify=stratify_param
        )
        
        # Second split: 15% val, 15% test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=config.RANDOM_SEED,
            stratify=temp_df[label_col] if stratify_param is not None else None
        )
    except ValueError as e:
        print(f"⚠️  Stratification failed: {e}")
        print("   Falling back to random split...")
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=config.RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=config.RANDOM_SEED
        )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify label distribution in each split
    print(f"\nLabel distribution across splits:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        label_dist = split_df[label_col].value_counts(normalize=True)
        print(f"  {split_name}:")
        for label, pct in label_dist.items():
            label_name = "Violation" if label == 1 else "No Violation"
            print(f"    {label_name}: {pct*100:.1f}%")
    
    return train_df, val_df, test_df, text_col, label_col

# ===========================
# 3. EXTRACT EMBEDDINGS
# ===========================
def extract_embeddings(texts, model, tokenizer, device):
    """
    Extract embeddings from GPT-2 model.
    Uses the last token's hidden state as the embedding.
    Handles datasets of any size with batch processing.
    """
    print(f"\nExtracting embeddings for {len(texts)} samples...")
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 8  # Process 8 texts at a time to manage memory
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
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
            
            # Get last token embedding for each item in batch
            for j in range(last_hidden_state.size(0)):
                embedding = last_hidden_state[j, -1, :].cpu()
                embeddings.append(embedding)
    
    print(f"✓ Extracted {len(embeddings)} embeddings")
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
    print("Supports datasets of any size with automatic optimization")
    print("="*60)
    
    # Load and split data
    train_df, val_df, test_df, text_col, label_col = load_and_split_data(config.DATASET_PATH)
    
    # Auto-adjust hyperparameters based on dataset size
    config.adjust_for_dataset_size(len(train_df) + len(val_df) + len(test_df))
    
    # Get or compute embeddings
    train_embeddings, val_embeddings, test_embeddings = get_or_compute_embeddings(
        train_df, val_df, test_df, text_col, config.EMBEDDING_CACHE
    )
    
    print(f"\n✓ Embedding dimensions: {train_embeddings.shape[1]}")
    print(f"✓ Training samples: {train_embeddings.shape[0]}")
    print(f"✓ Validation samples: {val_embeddings.shape[0]}")
    print(f"✓ Test samples: {test_embeddings.shape[0]}")
    
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