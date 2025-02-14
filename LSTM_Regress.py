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
from sklearn.metrics import mean_absolute_error


class ClassicalLSTMTimeSeriesRegress(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int = 1,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)       
        last_output = lstm_out[:, -1, :] 
        last_output = self.dropout(last_output)
        output = self.fc(last_output)    
        return output

def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(model: nn.Module) -> None:
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
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

    continuous_features.append(target_phenotype)
    
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

def split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device):
    """
    Split data into train, validation, and test sets and create DataLoaders with sliding windows.

    Args:
        X (list of tensors): Input data for all subjects (time-series data combined with phenotypes).
        y (list): Target labels for all subjects.
        batch_size (int): Batch size for DataLoader.
        sequence_length (int): Length of each input sequence.
        device (torch.device): Device to use (CPU or GPU).

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
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
    
    train_sequences, train_sequence_labels = create_sequences(train_X, train_y)
    val_sequences, val_sequence_labels = create_sequences(val_X, val_y)
    test_sequences, test_sequence_labels = create_sequences(test_X, test_y)
    
    def create_dataloader(sequences, labels):
        x_tensors = [seq.to(device) for seq in sequences]
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        dataset = TensorDataset(torch.stack(x_tensors), y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return (create_dataloader(train_sequences, train_sequence_labels),
            create_dataloader(val_sequences, val_sequence_labels),
            create_dataloader(test_sequences, test_sequence_labels))


def create_model(hyperparams: dict[str, Any], device: torch.device, sequence_length: int, feature_dim: int) -> torch.nn.Module:
    hidden_dim = hyperparams.get("hidden_dim", 128)
    num_layers = hyperparams.get("num_layers", 2)
    output_dim = hyperparams.get("output_dim", 1)  
    dropout = hyperparams["dropout"]
    model = ClassicalLSTMTimeSeriesRegress(feature_dim, hidden_dim, num_layers, output_dim, dropout)
    return model


def train_epoch_regress(model: torch.nn.Module, iterator: DataLoader, optimizer: torch.optim.Optimizer, 
                        criterion, clip: float, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []
    for x, y in tqdm(iterator, desc="Training"):
        optimizer.zero_grad()
        yhat = model(x)  # 출력: (batch_size, 1)
        loss = criterion(yhat.squeeze(), y)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item()
        all_preds.extend(yhat.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return epoch_loss / len(iterator), mae

def evaluate_regress(model: torch.nn.Module, iterator: DataLoader, criterion):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(iterator, desc="Evaluating"):
            yhat = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()
            all_preds.extend(yhat.detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return epoch_loss / len(iterator), mae

def train_cycle_regress(model: torch.nn.Module, hyperparams: dict[str, Any], device: torch.device, 
                        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, tuning_set="default"):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"], eps=hyperparams["eps"])
    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hyperparams["restart_epochs"])
    
    if hyperparams["lossfunction"] == "MAE":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    best_valid_loss = float("inf")
    
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()
        train_loss, train_mae = train_epoch_regress(model, train_loader, optimizer, criterion, hyperparams["max_grad_norm"], scheduler)
        valid_loss, valid_mae = evaluate_regress(model, val_loader, criterion)
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = f"LSTM_{hyperparams['target']}_{hyperparams['lossfunction']}_{hyperparams['seed']}_{tuning_set}.pt"
            torch.save(model.state_dict(), save_path)
        print(f"Epoch: {epoch+1:02} | Time: {mins}m {secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}  Train MAE: {train_mae:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}  Val. MAE: {valid_mae:.3f}")
    
    save_path = f"LSTM_{hyperparams['target']}_{hyperparams['lossfunction']}_{hyperparams['seed']}_{tuning_set}.pt"
    model.load_state_dict(torch.load(save_path))
    test_loss, test_mae = evaluate_regress(model, test_loader, criterion)
    
    print(f"Test Loss: {test_loss:.3f}  Test MAE: {test_mae:.3f}")
    return test_loss, test_mae

def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def get_train_evaluate_regress(device: torch.device, tuning_set) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
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
        )
        feature_dim = train_loader.dataset[0][0].shape[-1]
        model = create_model(parameterization, device, sequence_length, feature_dim)
        init_weights(model)
        model = model.to(device)
        test_loss = train_cycle_regress(model, parameterization, device, train_loader, val_loader, test_loader, tuning_set)
        return test_loss
    return train_evaluate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = {
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
         "lossfunction": "MSE",  
         "output_dim": 1,        
         "target": "nihtbx_fluidcomp_uncorrected",
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
    train_loader, val_loader, test_loader = split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device)
    feature_dim = train_loader.dataset[0][0].shape[-1]
    model = create_model(hyperparams, device, sequence_length, feature_dim)
    init_weights(model)
    model = model.to(device)
    tuning_set = "default_regress"
    test_loss, test_mae = train_cycle_regress(model, hyperparams, device, train_loader, val_loader, test_loader, tuning_set)
