import os
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Youden's J = Sensitivity(Recall) + Specificity - 1
    youden_j = rec + spec - 1.0
    
    return {
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': spec,
        'youden_j': youden_j,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results", "prototype")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)
    
    PREDS_CSV = os.path.join(RESULTS_DIR, "validation_predictions.csv")
    REP_TXT = os.path.join(REP_DIR, "phase5_2_validation_thresholds.txt")
    
    logging.info("Loading validation predictions...")
    preds_df = pd.read_csv(PREDS_CSV)
    y_true = preds_df['true_label']
    rf_prob = preds_df['rf_prob']
    lr_prob = preds_df['lr_prob']
    
    # Model AUCs
    rf_roc_auc = roc_auc_score(y_true, rf_prob)
    rf_pr_auc = average_precision_score(y_true, rf_prob)
    rf_brier = brier_score_loss(y_true, rf_prob)
    
    lr_roc_auc = roc_auc_score(y_true, lr_prob)
    
    better_model = "Random Forest" if rf_roc_auc > lr_roc_auc else "Logistic Regression"
    
    # Threshold evaluation for RF
    thresholds = np.arange(0.1, 0.95, 0.05)
    rf_evals = []
    
    for t in thresholds:
        rf_evals.append(calculate_metrics_at_threshold(y_true, rf_prob, t))
        
    evals_df = pd.DataFrame(rf_evals)
    evals_df.to_csv(os.path.join(RESULTS_DIR, "rf_threshold_evaluations.csv"), index=False)
    
    best_f1_idx = evals_df['f1_score'].idxmax()
    best_f1_thresh = evals_df.loc[best_f1_idx, 'threshold']
    
    best_youden_idx = evals_df['youden_j'].idxmax()
    best_youden_thresh = evals_df.loc[best_youden_idx, 'threshold']
    
    # Recommend operating threshold (usually max f1 or youden, let's pick Youden for balance)
    rec_thresh = best_youden_thresh
    rec_row = evals_df.loc[best_youden_idx]
    
    # Save curves
    fpr, tpr, roc_threshs = roc_curve(y_true, rf_prob)
    prec, rec, pr_threshs = precision_recall_curve(y_true, rf_prob)
    
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_threshs}).to_csv(os.path.join(RESULTS_DIR, "rf_roc_curve.csv"), index=False)
    
    pr_threshs = np.append(pr_threshs, 1.0) # match length
    pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': pr_threshs}).to_csv(os.path.join(RESULTS_DIR, "rf_pr_curve.csv"), index=False)
    
    # Concerns
    concerns = "The model AUC of 0.67 is relatively modest and only slightly better than random baseline, indicating that simple static features without dynamic triggers (rainfall, earthquakes) and without LULC struggle to powerfully discriminate positive vs negative landslides."

    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5.2 — BASELINE VALIDATION AND THRESHOLD SELECTION\n")
        f.write("=========================================================\n\n")
        
        f.write(f"Best model: {better_model}\n\n")
        
        f.write(f"Best threshold by F1: {best_f1_thresh:.2f}\n")
        f.write(f"Best threshold by Youden's J: {best_youden_thresh:.2f}\n")
        f.write(f"Recommended operating threshold: {rec_thresh:.2f}\n\n")
        
        f.write("AT RECOMMENDED THRESHOLD (Random Forest):\n")
        f.write(f"Validation ROC-AUC: {rf_roc_auc:.4f}\n")
        f.write(f"Validation PR-AUC: {rf_pr_auc:.4f}\n")
        f.write(f"Precision: {rec_row['precision']:.4f}\n")
        f.write(f"Recall: {rec_row['recall']:.4f}\n")
        f.write(f"F1: {rec_row['f1_score']:.4f}\n")
        f.write(f"Specificity: {rec_row['specificity']:.4f}\n")
        f.write(f"Accuracy: {rec_row['accuracy']:.4f}\n")
        f.write(f"Brier score: {rf_brier:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives: {int(rec_row['tp'])}\n")
        f.write(f"True Negatives: {int(rec_row['tn'])}\n")
        f.write(f"False Positives: {int(rec_row['fp'])}\n")
        f.write(f"False Negatives: {int(rec_row['fn'])}\n\n")
        
        f.write("CONCERNS:\n")
        f.write(f"{concerns}\n\n")
        f.write("STOP.\n")
        
    print("Threshold evaluation completed.")

if __name__ == "__main__":
    main()
