import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score, accuracy_score


class ClassicalLSTMTimeSeries(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)       
        last_output = lstm_out[:, -1, :] 
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)    
        return logits


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(model: nn.Module) -> None:
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(_init_weights)


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


def create_model(hyperparams: dict[str, Any], device: torch.device, sequence_length: int, feature_dim: int) -> torch.nn.Module:
    hidden_dim = hyperparams.get("hidden_dim", 128)
    num_layers = hyperparams.get("num_layers", 2)
    output_dim = hyperparams["output_dim"]
    dropout = hyperparams["dropout"]
    model = ClassicalLSTMTimeSeries(feature_dim, hidden_dim, num_layers, output_dim, dropout)
    return model


def train_epoch(model: torch.nn.Module, iterator: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion, clip: float, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]):
    model.train()
    epoch_loss = 0
    y_true, y_scores = [], []  
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item()
        
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            yhat = torch.sigmoid(yhat)
            
        y_true.extend(y.cpu().numpy().astype(int))
        y_scores.extend(yhat.detach().cpu().squeeze().numpy())
    
    try:
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        auc = None
            
    return epoch_loss / len(iterator), accuracy_score(y_true, y_scores)

def evaluate(model: torch.nn.Module, iterator: DataLoader, criterion):
    model.eval()
    epoch_loss = 0
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                yhat = torch.sigmoid(yhat)
                
            y_true.extend(y.cpu().numpy().astype(int))
            y_scores.extend(yhat.cpu().squeeze().numpy())
    
    try:
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            print(f"Warning: Found {len(unique_labels)} classes. AUC requires binary classification.")
            auc = None
        else:
            auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        auc = None
        
    return epoch_loss / len(iterator), accuracy_score(y_true, y_scores)

def train_epoch_multiclass(model: torch.nn.Module, iterator: DataLoader, optimizer: torch.optim.Optimizer, 
                           criterion, clip: float, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]):
    model.train()
    epoch_loss = 0
    y_true, y_pred_classes = [], []
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item()
        _, predicted_classes = torch.max(yhat, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred_classes.extend(predicted_classes.cpu().numpy())
    try:
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        acc = None
    return epoch_loss / len(iterator), acc

def evaluate_multiclass(model: torch.nn.Module, iterator: DataLoader, criterion):
    model.eval()
    epoch_loss = 0
    y_true, y_pred_classes = [], []
    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            _, predicted_classes = torch.max(yhat, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred_classes.extend(predicted_classes.cpu().numpy())
    try:
        acc = accuracy_score(y_true, y_pred_classes)
    except ValueError as e:
        print(f"Skipping hyperparameter set due to error: {e}")
        acc = None
    return epoch_loss / len(iterator), acc

def train_cycle(model: torch.nn.Module, hyperparams: dict[str, Any], device: torch.device, 
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, tuning_set):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
        
    scheduler = None
    if hyperparams["lr_sched"] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hyperparams["restart_epochs"])

    if hyperparams["binary"]==True:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_metrics, valid_metrics, test_metrics = [], [], []
    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()
        if hyperparams["binary"]==True:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, hyperparams["max_grad_norm"], scheduler)
            valid_loss, valid_acc = evaluate(model, val_loader, criterion)
        else:
            train_loss, train_acc = train_epoch_multiclass(model, train_loader, optimizer, criterion, hyperparams["max_grad_norm"], scheduler)
            valid_loss, valid_acc = evaluate_multiclass(model, val_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if hyperparams.get("model_dir") is None:
                torch.save(model.state_dict(), f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/classical_best_time_series_model.pt")
            else:
                torch.save(model.state_dict(), f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/classical_{hyperparams['model_dir']}_{hyperparams['seed']}_{tuning_set}.pt")
        train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc})
        valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss, 'valid_acc': valid_acc})
        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}  AUC/Accuracy: {train_acc}")
        print(f"\t Val. Loss: {valid_loss:.3f}  AUC/Accuracy: {valid_acc}")
    if hyperparams.get("model_dir") is None:
        model.load_state_dict(torch.load(f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/classical_best_time_series_model.pt", weights_only=True))
    else:
        model.load_state_dict(torch.load(f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/classical_{hyperparams['model_dir']}_{hyperparams['seed']}_{tuning_set}.pt", weights_only=True))
    if hyperparams["binary"]==True:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
    else:
        test_loss, test_acc = evaluate_multiclass(model, test_loader, criterion)
    test_metrics.append({'epoch': hyperparams['epochs'], 'test_loss': test_loss, 'test_acc': test_acc})
    print(f"Test Loss: {test_loss:.3f}  AUC/Accuracy: {test_acc}")
    metrics = []
    for epoch in range(hyperparams['epochs']):
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics[epoch]['train_loss'],
            'train_acc': train_metrics[epoch]['train_acc'],
            'valid_loss': valid_metrics[epoch]['valid_loss'],
            'valid_acc': valid_metrics[epoch]['valid_acc'],
            'test_loss': test_metrics[0]['test_loss'],
            'test_acc': test_metrics[0]['test_acc'],
        })
    metrics_df = pd.DataFrame(metrics)
    csv_filename = f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/classical_{hyperparams['target']}_{hyperparams['seed']}_{tuning_set}.csv"
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")
    return test_loss, test_acc

def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def get_train_evaluate_class(device: torch.device, tuning_set) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> tuple:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        seed(parameterization["seed"])
        sequence_length = 363
        X, y = load_fmri_data(
            parameterization["dataset"],
            parameterization["parcel_type"],
            parameterization["input_phenotype"],
            parameterization["target"],
            parameterization["input_categorical"],
        )
        train_loader, val_loader, test_loader = split_and_prepare_dataloaders(
            X, y,
            parameterization["batch_size"],
            sequence_length,
            device,
            parameterization["binary"],
        )
        feature_dim = train_loader.dataset[0][0].shape[-1]
        model = create_model(parameterization, device, sequence_length, feature_dim)
        init_weights(model)
        model = model.to(device)
        test_loss, test_acc = train_cycle(
            model,
            parameterization,
            device,
            train_loader,
            val_loader,
            test_loader,
            tuning_set,
        )
        return test_loss, test_acc
    return train_evaluate


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = {
         "model_type": "LSTM",  
         "hidden_dim": 128,
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
    batch_size = 16
    sequence_length = 363
    X, y = load_fmri_data(hyperparams["dataset"], hyperparams["parcel_type"],
                          hyperparams["input_phenotype"], hyperparams["target"],
                          hyperparams["input_categorical"])
    train_loader, val_loader, test_loader = split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device, hyperparams["binary"])
    feature_dim = train_loader.dataset[0][0].shape[-1]
    model = create_model(hyperparams, device, sequence_length, feature_dim)
    init_weights(model)
    model = model.to(device)
    tuning_set = "default"
    test_loss, test_acc = train_cycle(model, hyperparams, device, train_loader, val_loader, test_loader, tuning_set)
