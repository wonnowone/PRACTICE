import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional
from transformers import BertConfig, BertForSequenceClassification


def epoch_time(start_time: float, end_time: float) -> tuple:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(model: nn.Module) -> None:

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def load_fmri_data(dataset, parcel_type, phenotypes_to_include, target_phenotype, categorical_features):
    """
    Load and combine fMRI data and phenotype data for all subjects.

    Args:
        dataset (str): Type of dataset to use (ABCD, UKB).
        parcel_type (str): Parcellation type for each subject's fMRI data (HCP, HCP180, Schaefer).
        phenotypes_to_include (list): Phenotype columns to include as input features.
        target_phenotype (str): Phenotype column to use as the target.
        categorical_features (list): Columns in phenotypes_to_include that are categorical.

    Returns:
        X (list of tensors): List of input tensors (fMRI + phenotypes) for all subjects.
        y (list): List of target labels for all subjects.
    """

    if dataset == "ABCD":
        phenotypes = pd.read_csv("ABCD/ABCD_phenotype_total.csv")
        phenotypes = phenotypes[phenotypes_to_include + ["subjectkey", target_phenotype]].dropna()
        subject_ids = phenotypes["subjectkey"].values
    elif dataset == "UKB":
        phenotypes = pd.read_csv("UKB/UKB_phenotype_gps_fluidint.csv")
        phenotypes = phenotypes[phenotypes_to_include + ["eid", target_phenotype]].dropna()
        subject_ids = phenotypes["eid"].values

    continuous_features = [col for col in phenotypes_to_include if col not in categorical_features]
    
    phenotypes[continuous_features] = (phenotypes[continuous_features] - phenotypes[continuous_features].mean()) / phenotypes[continuous_features].std()
    input_phenotypes = phenotypes[phenotypes_to_include].values
    target_labels = phenotypes[target_phenotype].values

    X, y = [], []
    valid_subject_count = 0
    
    for i, subject_id in enumerate(subject_ids):
        if dataset == "ABCD":
            if parcel_type == "HCP":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_sub-{subject_id}.npy"
            elif parcel_type == "HCP180":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_180_sub-{subject_id}.npy"
            elif parcel_type == "Schaefer":
                fmri_path = f"ABCD/sub-{subject_id}/schaefer_sub-{subject_id}.npy"
        elif dataset == "UKB":
            if parcel_type == "HCP":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type == "HCP180":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type == "Schaefer":
                fmri_path = f"UKB/{subject_id}/schaefer_400Parcels_17Networks_{subject_id}.npy"
        if not os.path.exists(fmri_path):
            continue
        fmri_data = np.load(fmri_path)
        if dataset == "UKB" and fmri_data.shape[0] > 363:
            start_idx = (fmri_data.shape[0] - 363) // 2
            fmri_data = fmri_data[start_idx:start_idx + 363]
        fmri_mean = fmri_data.mean(axis=0)
        fmri_std = fmri_data.std(axis=0)
        
        fmri_std[fmri_std == 0] = 1e-8
        
        fmri_data = (fmri_data - fmri_mean) / fmri_std
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
        phenotype_tensor = torch.tensor(input_phenotypes[i], dtype=torch.float32).repeat(fmri_tensor.shape[0], 1)
        combined_features = torch.cat((fmri_tensor, phenotype_tensor), dim=1)
        
        X.append(combined_features)
        y.append(target_labels[i])
        valid_subject_count += 1
    print(f"Final sample size (number of subjects): {valid_subject_count}")
    return X, y

def split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device, binary, stratify=True):
    """
    Split data into train, validation, and test sets and create DataLoaders with sliding windows.

    Args:
        X (list of tensors): Input data for all subjects (time-series data combined with phenotypes).
        y (list): Target labels for all subjects.
        batch_size (int): Batch size for DataLoader.
        sequence_length (int): Length of each input sequence.
        device (torch.device): Device to use (CPU or GPU).
        stratify (bool): Whether to stratify the split by target labels.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
    """
    def create_sequences(data, labels):
        """
        Create fixed-length sequences for each subject using a sliding window.

        Args:
            data (list of tensors): Time-series data for all subjects.
            labels (list): Target labels for all subjects.

        Returns:
            sequences (list of tensors): Sequences of shape (sequence_length, feature_dim).
            sequence_labels (list): Labels for each sequence.
        """

        sequences, sequence_labels = [], []
        for subject_data, label in zip(data, labels):
            num_time_points = subject_data.shape[0]
            for start in range(0, num_time_points - sequence_length + 1, sequence_length):
                seq = subject_data[start:start + sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)
        return sequences, sequence_labels

    y = np.array(y)
    stratify_labels = y if stratify else None
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.3, stratify=stratify_labels, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, stratify=temp_y if stratify else None, random_state=42)
    
    train_sequences, train_sequence_labels = create_sequences(train_X, train_y)
    val_sequences, val_sequence_labels = create_sequences(val_X, val_y)
    test_sequences, test_sequence_labels = create_sequences(test_X, test_y)
    
    def create_dataloader(sequences, labels):
        x_tensors = [seq.to(device) for seq in sequences]
        y_tensor = torch.tensor(labels, dtype=torch.float32 if binary else torch.long, device=device)
        dataset = TensorDataset(torch.stack(x_tensors), y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return (create_dataloader(train_sequences, train_sequence_labels),
            create_dataloader(val_sequences, val_sequence_labels),
            create_dataloader(test_sequences, test_sequence_labels))


def create_model(hyperparams: dict, device: torch.device, sequence_length: int, feature_dim: int) -> torch.nn.Module:

    config = BertConfig(
        vocab_size=30522,  
        hidden_size=feature_dim,
        num_hidden_layers=hyperparams["ansatz_layers"],
        num_attention_heads=hyperparams["degree"],
        intermediate_size=4 * feature_dim,
        hidden_dropout_prob=hyperparams["dropout"],
        attention_probs_dropout_prob=hyperparams["dropout"],
        max_position_embeddings=sequence_length,
        type_vocab_size=2,
        initializer_range=0.02,
    )
    model = BertForSequenceClassification(config)
    return model

def train_epoch(
    model: torch.nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
):
    model.train()
    epoch_loss = 0
    y_true, y_scores = [], []  # For AUC calculation
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item()
        
        # Convert predictions to probabilities if using BCEWithLogitsLoss
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            yhat = torch.sigmoid(yhat)
            
        # Ensure correct format for AUC calculation
        y_true.extend(y.cpu().numpy().astype(int))  # Convert to int for binary labels
        y_scores.extend(yhat.detach().cpu().squeeze().numpy())
    
    try:
        # Convert to numpy arrays and ensure proper shape
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Check if we have binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        print(f"y_true shape: {np.shape(y_true)}, unique values: {np.unique(y_true)}")
        print(f"y_scores shape: {np.shape(y_scores)}, range: [{np.min(y_scores)}, {np.max(y_scores)}]")
        auc = None
            
    return epoch_loss / len(iterator), auc

def evaluate(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            
            # Convert predictions to probabilities if using BCEWithLogitsLoss
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                yhat = torch.sigmoid(yhat)
                
            # Ensure correct format for AUC calculation
            y_true.extend(y.cpu().numpy().astype(int))  # Convert to int for binary labels
            y_scores.extend(yhat.cpu().squeeze().numpy())
    
    try:
        # Convert to numpy arrays and ensure proper shape
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Check if we have binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        print(f"y_true shape: {np.shape(y_true)}, unique values: {np.unique(y_true)}")
        print(f"y_scores shape: {np.shape(y_scores)}, range: [{np.min(y_scores)}, {np.max(y_scores)}]")
        auc = None
        
    return epoch_loss / len(iterator), auc


def train_epoch_multiclass(
    model: torch.nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
):
    model.train()
    epoch_loss = 0
    y_true, y_pred_classes = [], []  # For accuracy calculation

    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)

        loss = criterion(yhat.squeeze(), y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

        # Collect predictions and true labels for accuracy
        _, predicted_classes = torch.max(yhat, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred_classes.extend(predicted_classes.cpu().numpy())

    # Calculate accuracy
    # acc = accuracy_score(y_true, y_pred_classes)         
    try:
    # Train the model and compute roc_auc_score
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        auc = None  # Or any fallback value           
        
    return epoch_loss / len(iterator), acc


def evaluate_multiclass(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0
    y_true, y_pred_classes = [], []  # For accuracy calculation
    
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()

            # Collect predictions and true labels for accuracy
            _, predicted_classes = torch.max(yhat, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred_classes.extend(predicted_classes.cpu().numpy())

    # acc = accuracy_score(y_true, y_pred_classes)         
    try:
    # Train the model and compute roc_auc_score
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        auc = None  # Or any fallback value   
        
    return epoch_loss / len(iterator), acc

def train_cycle(model: nn.Module, hyperparams: dict, device: torch.device, 
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, tuning_set):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    
    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hyperparams["restart_epochs"])
  
    criterion = nn.CrossEntropyLoss() if not hyperparams["binary"] else nn.BCEWithLogitsLoss()
    
    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, hyperparams["max_grad_norm"], scheduler)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = f"BERT_{hyperparams['target']}_{hyperparams['seed']}_{tuning_set}.pt"
            torch.save(model.state_dict(), save_path)
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}  Accuracy: {train_acc:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}  Accuracy: {valid_acc:.3f}")
    save_path = f"BERT_{hyperparams['target']}_{hyperparams['seed']}_{tuning_set}.pt"
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.3f}  Accuracy: {test_acc:.3f}")
    return test_loss, test_acc

def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparams = {
         "qubits": 6,                    
         "ansatz_layers": 2,             
         "degree": 4,                   
         "dropout": 0.1,
         "optimizer": "Adam",
         "lr": 0.001,
         "wd": 1e-4,
         "eps": 1e-8,
         "lr_sched": "cos",
         "restart_epochs": 30000,
         "epochs": 10,
         "max_grad_norm": 1.0,
         "binary": False,               
         "output_dim": 2,              
         "target": "sex",
         "parcel_type": "HCP",
         "dataset": "UKB",
         "input_phenotype": [],          
         "input_categorical": []
    }
    hyperparams["seed"] = 2024


    X, y = load_fmri_data(hyperparams["dataset"], hyperparams["parcel_type"],
                          hyperparams["input_phenotype"], hyperparams["target"],
                          hyperparams["input_categorical"])
    batch_size = 16
    sequence_length = 363
    train_loader, val_loader, test_loader = split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device, hyperparams["binary"])
    feature_dim = train_loader.dataset[0][0].shape[-1]
    

    model = create_model(hyperparams, device, sequence_length, feature_dim)
    init_weights(model)
    model = model.to(device)
    
    tuning_set = "default"
    test_loss, test_acc = train_cycle(model, hyperparams, device, train_loader, val_loader, test_loader, tuning_set)
