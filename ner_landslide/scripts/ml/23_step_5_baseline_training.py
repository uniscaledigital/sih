import os
import time
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_metrics(y_true, y_prob, y_pred):
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': spec,
        'brier_score': brier
    }

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "prototype")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results", "prototype")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)
    
    TRAIN_CSV = os.path.join(FINAL_DIR, "modelA_train_final.csv")
    VAL_CSV = os.path.join(FINAL_DIR, "modelA_validation_final.csv")
    
    REP_TXT = os.path.join(REP_DIR, "phase5_baseline_training.txt")
    
    # Load data
    logging.info("Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV)
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    
    logging.info("Loading validation data...")
    val_df = pd.read_csv(VAL_CSV)
    X_val = val_df.drop(columns=['label'])
    y_val = val_df['label']
    
    # 1. Logistic Regression
    logging.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    start_time = time.time()
    lr.fit(X_train, y_train)
    lr_train_time = time.time() - start_time
    
    lr_y_prob = lr.predict_proba(X_val)[:, 1]
    lr_y_pred = lr.predict(X_val)
    lr_metrics = calculate_metrics(y_val, lr_y_prob, lr_y_pred)
    
    # 2. Random Forest
    logging.info("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    rf_y_prob = rf.predict_proba(X_val)[:, 1]
    rf_y_pred = rf.predict(X_val)
    rf_metrics = calculate_metrics(y_val, rf_y_prob, rf_y_pred)
    
    # Save models
    logging.info("Saving models...")
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression_baseline.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest_baseline.joblib"))
    
    # Save predictions
    val_preds_df = pd.DataFrame({
        'true_label': y_val,
        'lr_prob': lr_y_prob,
        'lr_pred': lr_y_pred,
        'rf_prob': rf_y_prob,
        'rf_pred': rf_y_pred
    })
    val_preds_df.to_csv(os.path.join(RESULTS_DIR, "validation_predictions.csv"), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {'model': 'Logistic Regression', 'train_time_s': lr_train_time, **lr_metrics},
        {'model': 'Random Forest', 'train_time_s': rf_train_time, **rf_metrics}
    ])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "baseline_metrics.csv"), index=False)
    
    # Determine better model (by ROC-AUC)
    better_model = "Random Forest" if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else "Logistic Regression"
    
    # Report
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5 — BASELINE MODEL TRAINING\n")
        f.write("=================================\n\n")
        
        f.write("Logistic Regression Metrics (Validation):\n")
        for k, v in lr_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write(f"- Training Time: {lr_train_time:.2f} seconds\n\n")
        
        f.write("Random Forest Metrics (Validation):\n")
        for k, v in rf_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write(f"- Training Time: {rf_train_time:.2f} seconds\n\n")
        
        f.write(f"Better Model (by ROC-AUC): {better_model}\n\n")
        
        f.write("Warnings/Errors: None\n")
        f.write("STOP.\n")
        
    print("Baseline training completed successfully.")

if __name__ == "__main__":
    main()
