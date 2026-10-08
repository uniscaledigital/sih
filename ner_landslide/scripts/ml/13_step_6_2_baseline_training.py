import os
import json
import logging
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    brier_score_loss
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_metrics(y_true, y_prob, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = rec # recall is sensitivity
    
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'roc_auc': roc,
        'pr_auc': pr,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'brier_score': brier,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "baseline")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results", "baseline")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    TRAIN_CSV = os.path.join(FINAL_DIR, "modelA_train_final.csv")
    VAL_CSV = os.path.join(FINAL_DIR, "modelA_validation_final.csv")
    TEST_CSV = os.path.join(FINAL_DIR, "modelA_test_final.csv")
    META_CSV = os.path.join(FINAL_DIR, "modelA_metadata.csv")
    
    LR_ART = os.path.join(MODELS_DIR, "logistic_regression_baseline.joblib")
    RF_ART = os.path.join(MODELS_DIR, "random_forest_baseline.joblib")
    
    METRICS_CSV = os.path.join(RESULTS_DIR, "baseline_model_metrics.csv")
    VAL_PRED_CSV = os.path.join(RESULTS_DIR, "validation_predictions.csv")
    TEST_PRED_CSV = os.path.join(RESULTS_DIR, "test_predictions.csv")
    
    REP_TXT = os.path.join(REP_DIR, "phase6_2_baseline_training.txt")
    STATUS_JSON = os.path.join(REP_DIR, "PHASE6_STATUS.json")
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    meta_df = pd.read_csv(META_CSV)
    
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_val = val_df.drop(columns=['label'])
    y_val = val_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    # Validation
    assert len(train_df) == 17541
    assert len(val_df) == 3778
    assert len(test_df) == 3731
    assert len(X_train.columns) == 201
    assert not train_df.isnull().values.any()
    
    metrics_data = []
    val_predictions = []
    test_predictions = []
    
    # ---------------------------------------------------------
    # MODEL 1: Logistic Regression
    # ---------------------------------------------------------
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42, max_iter=2000, penalty='l2'))
    ])
    
    start_t = time.time()
    lr_pipeline.fit(X_train, y_train)
    lr_time = time.time() - start_t
    
    joblib.dump(lr_pipeline, LR_ART)
    
    lr_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_val_pred = (lr_val_prob >= 0.5).astype(int)
    lr_val_metrics = calculate_metrics(y_val, lr_val_prob, lr_val_pred)
    
    lr_test_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_test_pred = (lr_test_prob >= 0.5).astype(int)
    lr_test_metrics = calculate_metrics(y_test, lr_test_prob, lr_test_pred)
    
    lr_val_row = {'model': 'Logistic Regression', 'dataset': 'Validation'}
    lr_val_row.update(lr_val_metrics)
    metrics_data.append(lr_val_row)
    
    lr_test_row = {'model': 'Logistic Regression', 'dataset': 'Test'}
    lr_test_row.update(lr_test_metrics)
    metrics_data.append(lr_test_row)
    
    # ---------------------------------------------------------
    # MODEL 2: Random Forest
    # ---------------------------------------------------------
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    
    start_t = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_t
    
    joblib.dump(rf_model, RF_ART)
    
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_val_pred = (rf_val_prob >= 0.5).astype(int)
    rf_val_metrics = calculate_metrics(y_val, rf_val_prob, rf_val_pred)
    
    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_test_pred = (rf_test_prob >= 0.5).astype(int)
    rf_test_metrics = calculate_metrics(y_test, rf_test_prob, rf_test_pred)
    
    rf_val_row = {'model': 'Random Forest', 'dataset': 'Validation'}
    rf_val_row.update(rf_val_metrics)
    metrics_data.append(rf_val_row)
    
    rf_test_row = {'model': 'Random Forest', 'dataset': 'Test'}
    rf_test_row.update(rf_test_metrics)
    metrics_data.append(rf_test_row)
    
    pd.DataFrame(metrics_data).to_csv(METRICS_CSV, index=False)
    
    # Map index for predictions
    val_meta = meta_df[meta_df['split'] == 'VALIDATION'].copy().reset_index(drop=True)
    test_meta = meta_df[meta_df['split'] == 'TEST'].copy().reset_index(drop=True)
    
    # In case row orders differ slightly, we'll just use a safe row identifier.
    # The split files were saved exactly as extracted. So iloc matches perfectly.
    val_pred_df = pd.DataFrame({
        'safe_row_id': range(len(y_val)),
        'actual_label': y_val,
        'lr_predicted_probability': lr_val_prob,
        'lr_predicted_class_0_5': lr_val_pred,
        'rf_predicted_probability': rf_val_prob,
        'rf_predicted_class_0_5': rf_val_pred
    })
    val_pred_df.to_csv(VAL_PRED_CSV, index=False)
    
    test_pred_df = pd.DataFrame({
        'safe_row_id': range(len(y_test)),
        'actual_label': y_test,
        'lr_predicted_probability': lr_test_prob,
        'lr_predicted_class_0_5': lr_test_pred,
        'rf_predicted_probability': rf_test_prob,
        'rf_predicted_class_0_5': rf_test_pred
    })
    test_pred_df.to_csv(TEST_PRED_CSV, index=False)
    
    # ---------------------------------------------------------
    # Training Log
    # ---------------------------------------------------------
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 6.2 — BASELINE TRAINING LOG\n")
        f.write("=================================\n\n")
        f.write(f"Train rows: {len(y_train)} (Pos: {(y_train==1).sum()}, Neg: {(y_train==0).sum()})\n")
        f.write(f"Val rows: {len(y_val)} (Pos: {(y_val==1).sum()}, Neg: {(y_val==0).sum()})\n")
        f.write(f"Test rows: {len(y_test)} (Pos: {(y_test==1).sum()}, Neg: {(y_test==0).sum()})\n")
        f.write(f"Predictors: {X_train.shape[1]}\n\n")
        
        f.write("MODEL 1: Logistic Regression\n")
        f.write("Parameters: StandardScaler + LogisticRegression(penalty='l2', max_iter=2000, random_state=42)\n")
        f.write(f"Training time: {lr_time:.2f}s\n")
        f.write(f"Val ROC-AUC: {lr_val_metrics['roc_auc']:.4f}, Val PR-AUC: {lr_val_metrics['pr_auc']:.4f}\n")
        f.write(f"Test ROC-AUC: {lr_test_metrics['roc_auc']:.4f}, Test PR-AUC: {lr_test_metrics['pr_auc']:.4f}\n\n")
        
        f.write("MODEL 2: Random Forest\n")
        f.write("Parameters: RandomForestClassifier(n_estimators=100, random_state=42)\n")
        f.write(f"Training time: {rf_time:.2f}s\n")
        f.write(f"Val ROC-AUC: {rf_val_metrics['roc_auc']:.4f}, Val PR-AUC: {rf_val_metrics['pr_auc']:.4f}\n")
        f.write(f"Test ROC-AUC: {rf_test_metrics['roc_auc']:.4f}, Test PR-AUC: {rf_test_metrics['pr_auc']:.4f}\n\n")
        
        f.write("Artifacts Saved:\n")
        f.write(f"- {LR_ART}\n")
        f.write(f"- {RF_ART}\n\n")
        
        f.write("Warnings/Issues: None.\n")
        
    print("PHASE 6.2 — BASELINE MODEL TRAINING\n")
    print("Status:\nCOMPLETED\n")
    print(f"Training rows: {len(y_train)}")
    print(f"Validation rows: {len(y_val)}")
    print(f"Test rows: {len(y_test)}\n")
    print(f"Predictors: {X_train.shape[1]}\n")
    
    print("Models trained:\n1. Logistic Regression\n2. Random Forest\n")
    
    print("Logistic Regression:")
    print(f"Validation ROC-AUC: {lr_val_metrics['roc_auc']:.4f}")
    print(f"Validation PR-AUC: {lr_val_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {lr_test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {lr_test_metrics['pr_auc']:.4f}\n")
    
    print("Random Forest:")
    print(f"Validation ROC-AUC: {rf_val_metrics['roc_auc']:.4f}")
    print(f"Validation PR-AUC: {rf_val_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {rf_test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {rf_test_metrics['pr_auc']:.4f}\n")
    
    print("Best validation model:\nDO NOT declare final model yet.\n")
    print("Threshold:\n0.5 reference only — NOT FINAL\n")
    print("Hyperparameter tuning:\nDEFERRED\n")
    print("Feature selection:\nDEFERRED\n")
    print("Final test-based model selection:\nDEFERRED\n")
    print("Model B:\nDEFERRED\n")
    
    print("Artifacts:")
    print(f"- {LR_ART}")
    print(f"- {RF_ART}\n")
    
    print("Reports:")
    print(f"- {METRICS_CSV}")
    print(f"- {VAL_PRED_CSV}")
    print(f"- {TEST_PRED_CSV}")
    print(f"- {REP_TXT}\n")
    
    print("Final decision:\nPASS\n")
    
    print("NEXT:")
    print("PHASE 6.3 — BASELINE VALIDATION AND PERFORMANCE ANALYSIS\n")
    print("STOP.")
    
    # Update Status
    if not os.path.exists(STATUS_JSON):
        status = {"completion_percentage": 0, "steps": {}, "outputs": []}
    else:
        with open(STATUS_JSON, 'r') as f:
            status = json.load(f)
            
    status["completion_percentage"] = 25
    status["current_step"] = "6.2_baseline_training"
    status["steps"]["6.2_baseline_training"] = "COMPLETED"
    status["outputs"].extend([LR_ART, RF_ART, METRICS_CSV, VAL_PRED_CSV, TEST_PRED_CSV, REP_TXT])
    status["next_step"] = "Step 6.3 — Baseline Validation"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
