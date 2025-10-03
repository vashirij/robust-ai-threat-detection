"""
cicids_mlp_adv.py
Workflow:
 - Load CICIDS2017 CSV(s)
 - Preprocess (numeric features, impute, scale)
 - Train a small MLP (PyTorch)
 - Evaluate baseline
 - Craft FGSM & PGD attacks
 - Evaluate on adversarial examples
 - Optionally perform adversarial training
 - Report metrics: Accuracy, False Negative Rate (FNR), Attack Success Rate
"""

import os, glob, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import dataclasses
from typing import List, Tuple

# -------------------------
# Config with Hyperparameter Management
# -------------------------
import dataclasses
from typing import List, Tuple

@dataclasses.dataclass
class Config:
    """Centralized configuration management"""
    # Data
    data_dir: str = "data/CICIDS2017"
    csv_glob: str = "*.csv"
    test_size: float = 0.2
    val_size: float = 0.1  # From remaining training data
    sample_frac: float = 0.1
    
    # Model
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    
    # Training
    batch_size: int = 256
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    
    # Attacks
    eps_fgsm: float = 0.05
    eps_pgd: float = 0.05
    pgd_steps: int = 8
    pgd_step_size: float = 0.01
    
    # System
    random_seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

# Default configuration
config = Config()

# Legacy constants for backward compatibility
DATA_DIR = config.data_dir
CSV_GLOB = config.csv_glob
RANDOM_SEED = config.random_seed
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LR = config.lr
DEVICE = torch.device(config.device)
EPS_FGSM = config.eps_fgsm
EPS_PGD = config.eps_pgd
PGD_STEPS = config.pgd_steps
PGD_STEP_SIZE = config.pgd_step_size

# -------------------------
# 1) Data loading
# -------------------------
def load_cicids(csv_dir_or_files):
    """Load one or more CICIDS2017 CSVs into a DataFrame"""
    if isinstance(csv_dir_or_files, str) and os.path.isdir(csv_dir_or_files):
        files = glob.glob(os.path.join(csv_dir_or_files, CSV_GLOB))
    elif isinstance(csv_dir_or_files, (list, tuple)):
        files = csv_dir_or_files
    else:
        files = [csv_dir_or_files]

    if not files:
        raise RuntimeError("No CICIDS files found.")

    dfs = []
    for f in sorted(files):
        print("Loading:", f)
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)

def preprocess_cicids(df, sample_frac=None):
    """Keep numeric features, impute NaN, scale, map labels to {0,1}"""
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED)

    if " Label" not in df.columns:
        raise RuntimeError("No ' Label' column in data.")

    drop_cols = ["Flow ID","Source IP","Destination IP","Timestamp","StartTime","EndTime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    y = (df[" Label"].astype(str) != "BENIGN").astype(int).values
    X = df.drop(columns=[" Label"])

    # Clean data: handle infinity and very large values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Remove constant columns
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    # Select only numeric columns and handle any remaining extreme values
    numeric = X.select_dtypes(include=[np.number])
    
    # Clip extreme values to prevent scaling issues
    numeric = numeric.clip(lower=np.percentile(numeric.values, 0.1), 
                          upper=np.percentile(numeric.values, 99.9))
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(numeric.values)
    return Xs, y, scaler, numeric.columns.tolist()

# -------------------------
# 2) MLP model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128,64], dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

# -------------------------
# 3) Train / Eval with Best Practices
# -------------------------
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def train_epoch(model, loader, optim, device):
    model.train(); total=0
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device).float()
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward(); optim.step()
        total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

def validate_epoch(model, loader, device):
    """Validation without gradient computation"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def train_model_with_validation(model, train_loader, val_loader, device, epochs=15, lr=1e-3, patience=7):
    """Train model with validation and early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Added L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        # Validation
        val_loss = validate_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return train_losses, val_losses, optimizer

def eval_model(model, loader, device, threshold=0.5):
    model.eval(); ys=[]; preds=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            preds += (probs>=threshold).astype(int).tolist()
            ys += yb.numpy().tolist()
    ys=np.array(ys); preds=np.array(preds)
    acc = accuracy_score(ys,preds)
    cm = confusion_matrix(ys,preds)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    fnr = fn/(fn+tp+1e-12)
    return {"acc":acc,"fnr":fnr,"cm":cm}

# -------------------------
# 4) Attacks
# -------------------------
def fgsm_attack(model,x,y,eps):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(model(x_adv), y.float())
    loss.backward()
    x_adv = x_adv + eps*torch.sign(x_adv.grad)
    return x_adv.detach()

def pgd_attack(model,x,y,eps,step_size,steps):
    x0=x.clone().detach()
    x_adv=x0+torch.empty_like(x0).uniform_(-eps,eps)
    x_adv.requires_grad_(True)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(model(x_adv), y.float())
        model.zero_grad(); 
        if x_adv.grad is not None: x_adv.grad.zero_()
        loss.backward()
        x_adv = x_adv + step_size*torch.sign(x_adv.grad)
        x_adv = torch.min(torch.max(x_adv,x0-eps),x0+eps).detach().requires_grad_(True)
    return x_adv.detach()

def make_adv_loader(model,loader,attack_fn,device,**kwargs):
    xb_list=[]; y_list=[]
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        adv = attack_fn(model,xb,yb,**kwargs).cpu()
        xb_list.append(adv); y_list.append(yb.cpu())
    return DataLoader(TensorDataset(torch.cat(xb_list),torch.cat(y_list)),
                      batch_size=loader.batch_size,shuffle=False)

# -------------------------
# 5) Defense Mechanisms
# -------------------------
def adversarial_training_epoch(model, clean_loader, device, eps=0.3, alpha=0.5):
    """Train model with mix of clean and adversarial examples"""
    model.train()
    total_loss = 0
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for xb, yb in clean_loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        
        # Generate adversarial examples
        x_adv = fgsm_attack(model, xb, yb, eps)
        
        # Mix clean and adversarial examples
        mixed_x = torch.cat([xb, x_adv], dim=0)
        mixed_y = torch.cat([yb, yb], dim=0)
        
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(mixed_x), mixed_y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * mixed_x.size(0)
    
    return total_loss / (len(clean_loader.dataset) * 2)

def train_adversarial_model(model, train_loader, device, epochs=15, eps=0.3):
    """Train model with adversarial training"""
    print("Training with adversarial examples...")
    for epoch in range(epochs):
        loss = adversarial_training_epoch(model, train_loader, device, eps)
        if (epoch + 1) % 5 == 0:
            print(f"Adversarial Epoch {epoch + 1} | loss {loss:.4f}")
    return model

class EnsembleDefense:
    """Ensemble defense using multiple models"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, x, device, threshold=0.5):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                x_device = x.to(device)
                pred = torch.sigmoid(model(x_device)).cpu().numpy()
                predictions.append(pred)
        
        # Average predictions across models
        avg_pred = np.mean(predictions, axis=0)
        return (avg_pred >= threshold).astype(int)
    
    def evaluate(self, loader, device):
        all_preds = []
        all_labels = []
        
        for xb, yb in loader:
            preds = self.predict(xb, device)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fnr = fn / (fn + tp + 1e-12)
        
        return {"acc": acc, "fnr": fnr, "cm": cm}

def feature_squeezing_defense(X, bit_depth=8):
    """Input preprocessing defense through feature squeezing"""
    # Normalize to [0, 1] range
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-12)
    
    # Reduce bit depth
    levels = 2 ** bit_depth
    X_squeezed = np.round(X_norm * (levels - 1)) / (levels - 1)
    
    # Scale back to original range
    X_squeezed = X_squeezed * (X.max() - X.min()) + X.min()
    
    return X_squeezed

def enhanced_evaluation_metrics(y_true, y_pred, y_probs=None):
    """Comprehensive evaluation metrics for threat detection"""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fnr = fn / (fn + tp + 1e-12)  # False Negative Rate
    fpr = fp / (fp + tn + 1e-12)  # False Positive Rate
    
    # F1 Score and Specificity
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    
    # Attack success rate (for adversarial evaluation)
    attack_success_rate = np.mean(y_pred != y_true)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "fnr": fnr,
        "fpr": fpr,
        "attack_success_rate": attack_success_rate,
        "confusion_matrix": cm
    }
    
    return metrics

class ModelCheckpoint:
    """Save and load model checkpoints"""
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, epoch, loss, filename=None):
        if filename is None:
            filename = f"model_epoch_{epoch}_loss_{loss:.4f}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'config': dataclasses.asdict(config)
        }
        
        # Only save optimizer state if optimizer is provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint_data, filepath)
        print(f"Model saved: {filepath}")
        return filepath
    
    def load(self, filepath, model, optimizer=None):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

class ExperimentLogger:
    """Simple experiment tracking"""
    def __init__(self, log_dir="experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.results = {}
    
    def log_experiment(self, experiment_name, config, results):
        """Log experiment configuration and results"""
        log_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': dataclasses.asdict(config),
            'results': results
        }
        
        log_file = os.path.join(self.log_dir, f"{experiment_name}_{int(time.time())}.json")
        
        # Simple JSON-like logging
        with open(log_file, 'w') as f:
            import json
            f.write(json.dumps(log_data, indent=2, default=str))
        
        print(f"Experiment logged: {log_file}")
        return log_file

# -------------------------
# 6) Main pipeline
# -------------------------
def run_comprehensive_pipeline(csv_source=DATA_DIR, sample_frac=0.2, use_best_practices=True):
    """Enhanced pipeline with ML best practices and comprehensive evaluation"""
    print("=" * 80)
    print("ROBUST AI THREAT DETECTION - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Initialize experiment tracking
    experiment_logger = ExperimentLogger()
    checkpoint_manager = ModelCheckpoint()
    
    # Set all random seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Load and preprocess data
    df = load_cicids(csv_source)
    X, y, scaler, features = preprocess_cicids(df, sample_frac)
    print(f"Data: {X.shape}, Labels: {np.bincount(y)}")
    print(f"Class distribution: Benign: {np.sum(y==0)}, Malicious: {np.sum(y==1)}")

    # Enhanced data splitting with validation set
    if use_best_practices:
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_seed, stratify=y
        )
        # Second split: train vs validation
        val_size_adjusted = config.val_size / (1 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=config.random_seed, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    else:
        # Legacy split for compatibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=config.random_seed, stratify=y
        )
        X_val, y_val = None, None

    # Convert to tensors
    tX_train = torch.tensor(X_train, dtype=torch.float32)
    ty_train = torch.tensor(y_train, dtype=torch.long)
    tX_test = torch.tensor(X_test, dtype=torch.float32)
    ty_test = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(tX_train, ty_train), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(tX_test, ty_test), batch_size=config.batch_size)
    
    if use_best_practices and X_val is not None:
        tX_val = torch.tensor(X_val, dtype=torch.float32)
        ty_val = torch.tensor(y_val, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(tX_val, ty_val), batch_size=config.batch_size)
    else:
        val_loader = None

    results = {}
    
    # ========== BASELINE MODEL WITH BEST PRACTICES ==========
    print("\n" + "=" * 50)
    print("1. TRAINING BASELINE MODEL WITH BEST PRACTICES")
    print("=" * 50)
    
    baseline_model = MLP(X.shape[1], hidden_dims=config.hidden_dims, dropout=config.dropout).to(DEVICE)
    
    if use_best_practices and val_loader:
        print("Training with validation set and early stopping...")
        train_losses, val_losses, optimizer = train_model_with_validation(
            baseline_model, train_loader, val_loader, DEVICE, 
            epochs=config.epochs, lr=config.lr, patience=config.patience
        )
        # Save best model
        checkpoint_manager.save(baseline_model, optimizer, config.epochs, val_losses[-1], "baseline_model.pt")
    else:
        print("Training with legacy method...")
        optim = torch.optim.Adam(baseline_model.parameters(), lr=config.lr)
        for ep in range(config.epochs):
            loss = train_epoch(baseline_model, train_loader, optim, DEVICE)
            if (ep + 1) % 5 == 0:
                print(f"Epoch {ep + 1} | loss {loss:.4f}")

    # Evaluate baseline with enhanced metrics
    print("\n--- Baseline Model Performance ---")
    base_stats = eval_model(baseline_model, test_loader, DEVICE)
    print(f"Baseline Accuracy: {base_stats['acc']:.4f}")
    print(f"Baseline FNR: {base_stats['fnr']:.4f}")
    results['baseline'] = base_stats

    # ========== ADVERSARIAL ATTACKS ON BASELINE ==========
    print("\n" + "=" * 50)
    print("2. ADVERSARIAL ATTACKS ON BASELINE")
    print("=" * 50)
    
    # FGSM Attack
    print("--- FGSM Attack ---")
    fgsm_loader = make_adv_loader(baseline_model, test_loader, fgsm_attack, DEVICE, eps=EPS_FGSM)
    fgsm_stats = eval_model(baseline_model, fgsm_loader, DEVICE)
    print(f"FGSM Accuracy: {fgsm_stats['acc']:.4f}")
    print(f"FGSM Attack Success Rate: {1 - fgsm_stats['acc']:.4f}")
    results['fgsm_baseline'] = fgsm_stats

    # PGD Attack
    print("--- PGD Attack ---")
    pgd_loader = make_adv_loader(baseline_model, test_loader, pgd_attack, DEVICE,
                                eps=EPS_PGD, step_size=PGD_STEP_SIZE, steps=PGD_STEPS)
    pgd_stats = eval_model(baseline_model, pgd_loader, DEVICE)
    print(f"PGD Accuracy: {pgd_stats['acc']:.4f}")
    print(f"PGD Attack Success Rate: {1 - pgd_stats['acc']:.4f}")
    results['pgd_baseline'] = pgd_stats

    # ========== DEFENSE MECHANISMS ==========
    print("\n" + "=" * 50)
    print("3. IMPLEMENTING DEFENSE MECHANISMS")
    print("=" * 50)
    
    # DEFENSE 1: Adversarial Training
    print("--- Defense 1: Adversarial Training ---")
    adv_trained_model = MLP(X.shape[1]).to(DEVICE)
    train_adversarial_model(adv_trained_model, train_loader, DEVICE, epochs=EPOCHS)
    
    # Test adversarial trained model
    adv_clean_stats = eval_model(adv_trained_model, test_loader, DEVICE)
    adv_fgsm_stats = eval_model(adv_trained_model, fgsm_loader, DEVICE)
    adv_pgd_stats = eval_model(adv_trained_model, pgd_loader, DEVICE)
    
    print(f"Adv Training - Clean Accuracy: {adv_clean_stats['acc']:.4f}")
    print(f"Adv Training - FGSM Accuracy: {adv_fgsm_stats['acc']:.4f}")
    print(f"Adv Training - PGD Accuracy: {adv_pgd_stats['acc']:.4f}")
    
    results['adversarial_training'] = {
        'clean': adv_clean_stats,
        'fgsm': adv_fgsm_stats,
        'pgd': adv_pgd_stats
    }

    # DEFENSE 2: Ensemble Method
    print("\n--- Defense 2: Ensemble Method ---")
    ensemble_models = []
    for i in range(3):  # Create 3 different models
        model = MLP(X.shape[1], hidden_dims=[128+i*32, 64+i*16]).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        for ep in range(EPOCHS):
            train_epoch(model, train_loader, optim, DEVICE)
        ensemble_models.append(model)
    
    ensemble_defense = EnsembleDefense(ensemble_models)
    
    ensemble_clean_stats = ensemble_defense.evaluate(test_loader, DEVICE)
    ensemble_fgsm_stats = ensemble_defense.evaluate(fgsm_loader, DEVICE)
    ensemble_pgd_stats = ensemble_defense.evaluate(pgd_loader, DEVICE)
    
    print(f"Ensemble - Clean Accuracy: {ensemble_clean_stats['acc']:.4f}")
    print(f"Ensemble - FGSM Accuracy: {ensemble_fgsm_stats['acc']:.4f}")
    print(f"Ensemble - PGD Accuracy: {ensemble_pgd_stats['acc']:.4f}")
    
    results['ensemble'] = {
        'clean': ensemble_clean_stats,
        'fgsm': ensemble_fgsm_stats,
        'pgd': ensemble_pgd_stats
    }

    # DEFENSE 3: Feature Squeezing
    print("\n--- Defense 3: Feature Squeezing ---")
    X_squeezed = feature_squeezing_defense(X_test, bit_depth=6)
    tX_squeezed = torch.tensor(X_squeezed, dtype=torch.float32)
    squeezed_loader = DataLoader(TensorDataset(tX_squeezed, ty_test), batch_size=config.batch_size)
    
    squeezed_stats = eval_model(baseline_model, squeezed_loader, DEVICE)
    print(f"Feature Squeezing - Clean Accuracy: {squeezed_stats['acc']:.4f}")
    
    results['feature_squeezing'] = {'clean': squeezed_stats}

    # ========== COMPARATIVE ANALYSIS ==========
    print("\n" + "=" * 50)
    print("4. COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    print("\n--- Defense Effectiveness Summary ---")
    print("Method\t\t\tClean Acc\tFGSM Acc\tPGD Acc\t\tRobustness Gain")
    print("-" * 80)
    
    baseline_acc = base_stats['acc']
    fgsm_acc = fgsm_stats['acc']
    pgd_acc = pgd_stats['acc']
    
    print(f"Baseline\t\t{baseline_acc:.4f}\t\t{fgsm_acc:.4f}\t\t{pgd_acc:.4f}\t\t-")
    
    adv_fgsm_gain = adv_fgsm_stats['acc'] - fgsm_acc
    adv_pgd_gain = adv_pgd_stats['acc'] - pgd_acc
    print(f"Adversarial Training\t{adv_clean_stats['acc']:.4f}\t\t{adv_fgsm_stats['acc']:.4f}\t\t{adv_pgd_stats['acc']:.4f}\t\t+{(adv_fgsm_gain + adv_pgd_gain)/2:.4f}")
    
    ens_fgsm_gain = ensemble_fgsm_stats['acc'] - fgsm_acc
    ens_pgd_gain = ensemble_pgd_stats['acc'] - pgd_acc
    print(f"Ensemble\t\t{ensemble_clean_stats['acc']:.4f}\t\t{ensemble_fgsm_stats['acc']:.4f}\t\t{ensemble_pgd_stats['acc']:.4f}\t\t+{(ens_fgsm_gain + ens_pgd_gain)/2:.4f}")
    
    print(f"Feature Squeezing\t{squeezed_stats['acc']:.4f}\t\tN/A\t\tN/A\t\tN/A")

    # Log experiment results
    experiment_logger.log_experiment("comprehensive_robustness_evaluation", config, results)
    
    return results

# Legacy function for backward compatibility
def run_pipeline(csv_source=DATA_DIR, sample_frac=0.2):
    df = load_cicids(csv_source)
    X, y, scaler, features = preprocess_cicids(df, sample_frac)
    print("Data:", X.shape, "Labels:", np.bincount(y))

    # split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    tXtr, tytr = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    tXte, tyte = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(tXtr, tytr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(tXte, tyte), batch_size=BATCH_SIZE)

    # train
    model = MLP(X.shape[1]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, optim, DEVICE)
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1} | loss {loss:.4f}")

    # eval baseline
    base_stats = eval_model(model, test_loader, DEVICE)
    print("Baseline:", base_stats)

    # FGSM
    adv_loader = make_adv_loader(model, test_loader, fgsm_attack, DEVICE, eps=EPS_FGSM)
    fgsm_stats = eval_model(model, adv_loader, DEVICE)
    print("FGSM:", fgsm_stats)

    # PGD
    adv_loader_pgd = make_adv_loader(model, test_loader, pgd_attack, DEVICE,
                                   eps=EPS_PGD, step_size=PGD_STEP_SIZE, steps=PGD_STEPS)
    pgd_stats = eval_model(model, adv_loader_pgd, DEVICE)
    print("PGD:", pgd_stats)

    return {"baseline": base_stats, "fgsm": fgsm_stats, "pgd": pgd_stats}

if __name__ == "__main__":
    # Run comprehensive evaluation with all defense mechanisms
    results = run_comprehensive_pipeline(csv_source=DATA_DIR, sample_frac=0.1)
    
    # Save results for analysis
    print("\n" + "=" * 50)
    print("RESULTS SAVED - Ready for further analysis")
    print("=" * 50)
