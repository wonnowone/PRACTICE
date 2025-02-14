import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from tqdm import tqdm
from typing import Callable, Any, Tuple

def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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
      
        X.append(fmri_data)
        y.append(target_labels[i])
        valid_subject_count += 1
    
    print(f"Final sample size (number of subjects): {valid_subject_count}")
    return X, y


def aggregate_features(X):

    aggregated = []
    for data in X:
        mean_vals = data.mean(axis=0)
        std_vals = data.std(axis=0)
        aggregated.append(np.concatenate([mean_vals, std_vals]))
    return np.array(aggregated)


def prepare_datasets(X, y):
    X_agg = aggregate_features(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_agg, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return dtrain, dval, dtest


def train_xgboost_cycle(dtrain, dval, params, num_round=1000, early_stopping_rounds=20):
    evals = [(dtrain, "train"), (dval, "eval")]
    bst = xgb.train(params, dtrain, num_round, evals, early_stopping_rounds=early_stopping_rounds)
    return bst

def evaluate_xgboost(bst, dtest):
    preds = bst.predict(dtest)
    y_test = dtest.get_label()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"Test RMSE: {rmse:.3f}  Test MAE: {mae:.3f}")
    return rmse, mae


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def get_train_evaluate_regress_xgb(tuning_set) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> tuple:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        seed(parameterization["seed"])
        
        
        X, y = load_fmri_data(
            parameterization["dataset"],
            parameterization["parcel_type"],
            parameterization["input_phenotype"],
            parameterization["target"],
            parameterization["input_categorical"],
        )
        
        dtrain, dval, dtest = prepare_datasets(X, y)
        
     
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": parameterization.get("eta", 0.01),
            "max_depth": parameterization.get("max_depth", 6),
        }
        num_round = parameterization.get("num_round", 1000)
        early_stopping_rounds = parameterization.get("early_stopping_rounds", 20)
       
        bst = train_xgboost_cycle(dtrain, dval, params, num_round=num_round, early_stopping_rounds=early_stopping_rounds)
     
        rmse, mae = evaluate_xgboost(bst, dtest)
        return rmse, mae
    return train_evaluate


if __name__ == "__main__":
  
    parameterization = {
        "seed": 2024,
        "dataset": "UKB",
        "parcel_type": "HCP",
        "input_phenotype": [],  
        "input_categorical": [],
        "target": "nihtbx_fluidcomp_uncorrected",  
        "eta": 0.01,
        "max_depth": 6,
        "num_round": 1000,
        "early_stopping_rounds": 20,
    }
    tuning_set = "default_xgb"
    train_evaluate = get_train_evaluate_regress_xgb(tuning_set)
    rmse, mae = train_evaluate(parameterization)
