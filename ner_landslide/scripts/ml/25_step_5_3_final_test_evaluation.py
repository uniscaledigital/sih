import os
import logging
import pandas as pd
import numpy as np
import joblib

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

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "prototype")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results", "prototype")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)
    
    TEST_CSV = os.path.join(FINAL_DIR, "modelA_test_final.csv")
    MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_baseline.joblib")
    PREDS_CSV = os.path.join(RESULTS_DIR, "final_test_predictions.csv")
    REP_TXT = os.path.join(REP_DIR, "phase5_3_final_test_evaluation.txt")
    
    logging.info("Loading test dataset...")
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    logging.info("Loading Random Forest model...")
    rf = joblib.load(MODEL_PATH)
    
    logging.info("Predicting on test set...")
    rf_prob = rf.predict_proba(X_test)[:, 1]
    
    threshold = 0.50
    rf_pred = (rf_prob >= threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, rf_prob)
    pr_auc = average_precision_score(y_test, rf_prob)
    acc = accuracy_score(y_test, rf_pred)
    prec = precision_score(y_test, rf_pred, zero_division=0)
    rec = recall_score(y_test, rf_pred, zero_division=0)
    f1 = f1_score(y_test, rf_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, rf_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    brier = brier_score_loss(y_test, rf_prob)
    
    val_roc_auc = 0.6758
    val_pr_auc = 0.4634
    
    roc_diff = roc_auc - val_roc_auc
    pr_diff = pr_auc - val_pr_auc
    
    # Assessment
    # Typically, a drop of more than 0.05 indicates severe overfitting.
    if abs(roc_diff) < 0.05 and abs(pr_diff) < 0.05:
        generalization = "Acceptable (Performance stable compared to validation)"
        decision = "PASS"
    elif roc_diff < -0.05 or pr_diff < -0.05:
        generalization = f"Overfitting Detected (Test ROC dropped by {abs(roc_diff):.4f})"
        decision = "PASS WITH LIMITATIONS"
    else:
        generalization = "Unexpectedly higher on test set"
        decision = "PASS WITH LIMITATIONS"
        
    # Save predictions
    preds_df = pd.DataFrame({
        'true_label': y_test,
        'rf_prob': rf_prob,
        'rf_pred_050': rf_pred
    })
    preds_df.to_csv(PREDS_CSV, index=False)
    
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5.3 — FINAL TEST EVALUATION\n")
        f.write("=================================\n\n")
        f.write(f"TEST METRICS (Random Forest @ 0.50):\n")
        f.write(f"- ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"- PR-AUC: {pr_auc:.4f}\n")
        f.write(f"- Precision: {prec:.4f}\n")
        f.write(f"- Recall: {rec:.4f}\n")
        f.write(f"- F1-score: {f1:.4f}\n")
        f.write(f"- Specificity: {spec:.4f}\n")
        f.write(f"- Accuracy: {acc:.4f}\n")
        f.write(f"- Brier score: {brier:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives: {int(tp)}\n")
        f.write(f"True Negatives: {int(tn)}\n")
        f.write(f"False Positives: {int(fp)}\n")
        f.write(f"False Negatives: {int(fn)}\n\n")
        
        f.write("VALIDATION VS TEST COMPARISON:\n")
        f.write(f"Validation ROC-AUC: {val_roc_auc:.4f}  |  Test ROC-AUC: {roc_auc:.4f}  |  Diff: {roc_diff:+.4f}\n")
        f.write(f"Validation PR-AUC: {val_pr_auc:.4f}  |  Test PR-AUC: {pr_auc:.4f}  |  Diff: {pr_diff:+.4f}\n\n")
        
        f.write(f"Generalization: {generalization}\n\n")
        f.write(f"FINAL DECISION: {decision}\n\n")
        f.write("STOP.\n")
        
    logging.info("Test evaluation completed.")

if __name__ == "__main__":
    main()
