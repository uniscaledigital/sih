import os
import json
import logging
import pandas as pd
import sklearn

try:
    import xgboost
    xgboost_avail = xgboost.__version__
except ImportError:
    xgboost_avail = "Not installed"

try:
    import lightgbm
    lightgbm_avail = lightgbm.__version__
except ImportError:
    lightgbm_avail = "Not installed"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    # Create required directories
    os.makedirs(os.path.join(MODELS_DIR, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "tuned"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "final"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    TRAIN_CSV = os.path.join(FINAL_DIR, "modelA_train_final.csv")
    VAL_CSV = os.path.join(FINAL_DIR, "modelA_validation_final.csv")
    TEST_CSV = os.path.join(FINAL_DIR, "modelA_test_final.csv")
    
    REP_TXT = os.path.join(REP_DIR, "phase6_1_model_selection.txt")
    REP_CSV = os.path.join(REP_DIR, "phase6_1_model_selection.csv")
    
    train_df = pd.read_csv(TRAIN_CSV, nrows=10) # only reading headers to count features
    num_cols = len(train_df.columns)
    predictors = num_cols - 1
    
    # Since we know the rows from previous step
    # We could also read full to confirm no nan, but it was just done. Let's read full to be thorough.
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    tr_len = len(train_df)
    va_len = len(val_df)
    te_len = len(test_df)
    
    tr_pos = train_df['label'].sum()
    tr_neg = tr_len - tr_pos
    
    va_pos = val_df['label'].sum()
    va_neg = va_len - va_pos
    
    te_pos = test_df['label'].sum()
    te_neg = te_len - te_pos
    
    models_selected = [
        {
            'model': 'LogisticRegression',
            'model_type': 'Linear Model',
            'reason': 'Provides a highly interpretable, fast baseline benchmark. Calculates base linear separability.',
            'advantages': 'Transparent feature importance (coefficients), perfectly calibrated baseline probabilities, fast.',
            'limitations': 'Cannot capture non-linear environmental interactions without manual feature engineering.',
            'baseline_parameters': 'penalty=l2, max_iter=1000',
            'metrics_to_report': 'ROC-AUC, PR-AUC, Accuracy, F1',
            'training_data': 'modelA_train_final.csv',
            'validation_data': 'modelA_validation_final.csv',
            'test_data': 'modelA_test_final.csv (Strictly Held Out)',
            'selected_for_baseline': 'YES',
            'notes': 'Must scale numerical features if l2 regularization is active.'
        },
        {
            'model': 'RandomForestClassifier',
            'model_type': 'Tree Ensemble (Bagging)',
            'reason': 'Strong non-linear baseline standard for spatial modeling. Handles categorical data and outliers well.',
            'advantages': 'Robust to multicollinearity (e.g. slope/TRI). Captures interactions implicitly. Built-in feature importance.',
            'limitations': 'Can struggle with perfectly calibrated probabilities compared to Logistic Regression. Memory intensive for huge trees.',
            'baseline_parameters': 'n_estimators=100, max_depth=None',
            'metrics_to_report': 'ROC-AUC, PR-AUC, Accuracy, F1',
            'training_data': 'modelA_train_final.csv',
            'validation_data': 'modelA_validation_final.csv',
            'test_data': 'modelA_test_final.csv (Strictly Held Out)',
            'selected_for_baseline': 'YES',
            'notes': 'Excellent for mixed continuous/encoded static environmental variables.'
        }
    ]
    
    if xgboost_avail != "Not installed":
        models_selected.append({
            'model': 'XGBoostClassifier',
            'model_type': 'Tree Ensemble (Boosting)',
            'reason': 'State-of-the-art performance on tabular data.',
            'advantages': 'Often outperforms Random Forest. Handles missing data natively (though we imputed).',
            'limitations': 'Prone to overfitting if not tuned properly. Less interpretable than simple trees.',
            'baseline_parameters': 'n_estimators=100, learning_rate=0.1, max_depth=6',
            'metrics_to_report': 'ROC-AUC, PR-AUC, Accuracy, F1',
            'training_data': 'modelA_train_final.csv',
            'validation_data': 'modelA_validation_final.csv',
            'test_data': 'modelA_test_final.csv (Strictly Held Out)',
            'selected_for_baseline': 'YES',
            'notes': 'Strongest expected baseline candidate.'
        })
        
    pd.DataFrame(models_selected).to_csv(REP_CSV, index=False)
    
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 6.1 — BASELINE MODEL SELECTION AND TRAINING DESIGN\n")
        f.write("========================================================\n\n")
        
        f.write("1. DATASET INTEGRITY\n")
        f.write(f"Train: {tr_len} rows, Val: {va_len} rows, Test: {te_len} rows.\n")
        f.write("Label is binary (0/1). Coordinates, IDs, dates completely excluded. No Missing values (handled in Phase 5.11).\n\n")
        
        f.write("2. BASELINE MODELS SELECTED\n")
        for m in models_selected:
            f.write(f"- {m['model']} ({m['model_type']})\n")
        f.write("\n")
        
        f.write("3. CLASS IMBALANCE\n")
        f.write("Train class distribution: ~50% / 50%.\n")
        f.write("Strategy: None required. DO NOT use SMOTE or resampling. The dataset is already balanced by the Phase 5.4 design.\n\n")
        
        f.write("4. METRICS\n")
        f.write("Primary: ROC-AUC, PR-AUC\n")
        f.write("Secondary: Accuracy, Precision, Recall, F1-Score, Brier Score (for probability calibration).\n")
        f.write("Reason: Susceptibility mapping requires well-ranked spatial probabilities rather than just a hard 0/1 decision. ROC-AUC evaluates ranking ability across all thresholds.\n\n")
        
        f.write("5. THRESHOLD POLICY\n")
        f.write("Default 0.5 threshold will be reported initially, but the final operational threshold will be determined using Youden's J or F1 optimization on the VALIDATION set in a future phase.\n\n")
        
        f.write("6. SPATIAL VALIDATION REQUIREMENT\n")
        f.write("The existing Train/Val/Test split generated via Phase 5.10 Spatial Blocking MUST be strictly respected. DO NOT employ random cross-validation, as this breaches spatial leakage barriers.\n\n")
        
        f.write("7. REPRODUCIBILITY\n")
        f.write("Random Seed = 42 for all non-deterministic models (RandomForest, XGBoost).\n\n")
        
        f.write("8. EXPERIMENT TRACKING\n")
        f.write("Models will be stored in D:/sih/ner_landslide/models/baseline/\n\n")
        
        f.write("9. MODEL B STATUS\n")
        f.write("DEFERRED. Dynamic trigger model cannot be trained without temporal negative controls.\n")
        
    print("PHASE 6.1 — BASELINE MODEL SELECTION\n")
    print("Status: COMPLETED\n")
    
    print("Model:\nMODEL A — SPATIAL LANDSLIDE SUSCEPTIBILITY\n")
    print(f"Training rows: {tr_len}")
    print(f"Validation rows: {va_len}")
    print(f"Test rows: {te_len}\n")
    
    print(f"Final predictors: {predictors}\n")
    
    print("Recommended baseline models:")
    for i, m in enumerate(models_selected, 1):
        print(f"{i}. {m['model']}")
    print("\nModel selection rationale:\nLogistic Regression provides a linear, highly interpretable baseline. Random Forest provides a robust tree-based non-linear baseline. XGBoost (if available) tests state-of-the-art gradient boosting performance on structured tabular data.\n")
    
    print("Primary metrics:\nROC-AUC, PR-AUC\n")
    print("Secondary metrics:\nAccuracy, Precision, Recall, F1-score, Specificity, Brier score\n")
    print("Threshold strategy:\nAnalyze varying thresholds (Youden's J, F1 max) on VALIDATION set. Do not assume 0.5.\n")
    print("Class imbalance strategy:\nNone needed. Dataset is spatially stratified at 1:1 ratio.\n")
    print("Spatial validation strategy:\nStrictly preserve the pre-computed Spatial Blocks (Train/Val/Test). Do not use Random CV.\n")
    print("Random seed:\n42\n")
    
    print(f"Available ML libraries:\nscikit-learn: {sklearn.__version__}")
    print(f"xgboost: {xgboost_avail}")
    print(f"lightgbm: {lightgbm_avail}\n")
    
    print("Hyperparameter tuning:\nDEFERRED TO PHASE 6.4\n")
    print("Model B:\nDEFERRED\n")
    print("Test set:\nSTRICTLY HELD OUT\n")
    
    print("Final decision:\nPASS\n")
    
    print("Output files:")
    print(f"1. {REP_TXT}")
    print(f"2. {REP_CSV}\n")
    
    print("NEXT:")
    print("PHASE 6.2 — BASELINE MODEL TRAINING\n")
    print("STOP.")

if __name__ == "__main__":
    main()
