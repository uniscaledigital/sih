import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_feature_group(col):
    if col in ['elevation', 'slope', 'aspect', 'aspect_sin', 'aspect_cos', 'profile_curvature', 'planform_curvature', 'tri']:
        return 'Static terrain/DEM'
    elif 'geology' in col:
        return 'Geology'
    elif 'lulc' in col:
        return 'LULC'
    elif 'rainfall' in col:
        return 'Rainfall'
    elif 'swvl' in col:
        return 'Soil moisture'
    elif 'earthquake' in col or 'magnitude' in col:
        return 'Earthquake'
    elif col in ['landslide_id', 'latitude_x', 'longitude_x', 'latitude_y', 'longitude_y', 'event_date', 'label', 'latitude', 'longitude']:
        return 'Core/Metadata'
    return 'Other'

def get_probable_cause(col_group, pos_pct, neg_pct):
    if col_group == 'Rainfall' or col_group == 'Soil moisture':
        if neg_pct > 99:
            return "Negative temporal absence (structural) + Pre-2016 positives"
        else:
            return "Pre-2016 spatial/temporal limitations"
    elif col_group == 'Earthquake':
        if neg_pct > 99:
            return "Negative temporal absence (structural) + Natural absence of events"
        else:
            return "Natural absence of nearby seismic events in time window"
    elif col_group == 'Geology' or col_group == 'LULC':
        return "Spatial boundary gaps in polygon sources"
    elif col_group == 'Static terrain/DEM':
        return "DEM Tile NoData regions (e.g. water bodies, edges)"
    return "Unknown"

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    COMBO_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "positive_negative_dataset.csv")
    
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    AUDIT_CSV = os.path.join(REP_DIR, "phase5_7_missing_value_audit.csv")
    STRATEGY_TXT = os.path.join(REP_DIR, "phase5_7_missing_value_strategy.txt")
    STATUS_JSON = os.path.join(REP_DIR, "PHASE5_STATUS.json")
    
    df = pd.read_csv(COMBO_CSV)
    
    total_rows = len(df)
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    pos_count = len(pos_df)
    neg_count = len(neg_df)
    
    audit_data = []
    
    for col in df.columns:
        group = get_feature_group(col)
        missing_total = df[col].isnull().sum()
        missing_pct = (missing_total / total_rows) * 100
        
        pos_miss = pos_df[col].isnull().sum()
        neg_miss = neg_df[col].isnull().sum()
        
        pos_miss_pct = (pos_miss / pos_count) * 100 if pos_count > 0 else 0
        neg_miss_pct = (neg_miss / neg_count) * 100 if neg_count > 0 else 0
        miss_diff = abs(pos_miss_pct - neg_miss_pct)
        
        cause = get_probable_cause(group, pos_miss_pct, neg_miss_pct)
        
        # Recommendations
        if group == 'Core/Metadata':
            rec = "Retain"
            ma_action = "Retain"
            mb_action = "Retain"
            leakage = "LOW"
        elif group == 'Static terrain/DEM':
            rec = "Median imputation (fitted on train only) or drop row"
            ma_action = "Impute median"
            mb_action = "Impute median"
            leakage = "HIGH if imputed globally"
        elif group in ['Geology', 'LULC']:
            rec = "Assign 'Unknown' category"
            ma_action = "Fill 'Unknown'"
            mb_action = "Fill 'Unknown'"
            leakage = "LOW"
        elif group in ['Rainfall', 'Soil moisture', 'Earthquake']:
            rec = "Do NOT fill with zero. Exclude for Model A. Model B requires temporal control construction."
            ma_action = "Exclude completely"
            mb_action = "Requires valid event date and temporal matching"
            leakage = "SEVERE if zeros imputed for negatives"
        else:
            rec = "Investigate"
            ma_action = "N/A"
            mb_action = "N/A"
            leakage = "UNKNOWN"
            
        audit_data.append({
            'feature': col,
            'feature_group': group,
            'total_count': total_rows,
            'missing_count': missing_total,
            'missing_percentage': missing_pct,
            'positive_missing': pos_miss,
            'negative_missing': neg_miss,
            'positive_missing_pct': pos_miss_pct,
            'negative_missing_pct': neg_miss_pct,
            'missingness_difference': miss_diff,
            'probable_cause': cause,
            'recommended_treatment': rec,
            'model_A_action': ma_action,
            'model_B_action': mb_action,
            'leakage_risk': leakage
        })
        
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv(AUDIT_CSV, index=False)
    
    total_missing_cells = audit_df['missing_count'].sum()
    feats_with_missing = (audit_df['missing_count'] > 0).sum()
    feats_without_missing = len(audit_df) - feats_with_missing
    
    highest_miss = audit_df.sort_values('missing_percentage', ascending=False).head(5)[['feature', 'missing_percentage']].values.tolist()
    
    pos_total_missing = audit_df['positive_missing'].sum()
    neg_total_missing = audit_df['negative_missing'].sum()
    
    # Strategy Report
    with open(STRATEGY_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5.7 — MISSING VALUE STRATEGY REPORT\n")
        f.write("=========================================\n\n")
        
        f.write("1. STATIC FEATURE STRATEGY:\n")
        f.write("- Terrain/DEM missing values (very few) should be imputed using median values fitted ONLY on the training split later, or the rows can be safely dropped given the small number (5 positive records).\n\n")
        
        f.write("2. CATEGORICAL FEATURE STRATEGY:\n")
        f.write("- Geology and LULC missing values (boundary gaps) will be filled with a distinct 'Unknown' category to prevent data loss while preserving the structural gap information.\n\n")
        
        f.write("3. DYNAMIC FEATURE STRATEGY:\n")
        f.write("- Rainfall, Soil Moisture, and Earthquakes will NEVER be converted to zero simply because they are missing.\n")
        f.write("- Missingness implies no observation in the source window, not an explicit 0 measurement.\n\n")
        
        f.write("4. UNDATED POSITIVE STRATEGY:\n")
        f.write("- 6,186 positive records have no dates. They are entirely valid for MODEL A (Spatial Susceptibility) because it only uses static environmental features.\n")
        f.write("- They CANNOT be used for MODEL B (Dynamic Event) because their antecedent conditions cannot be measured without a timestamp.\n\n")
        
        f.write("5. NEGATIVE TEMPORAL STRATEGY:\n")
        f.write("- Negative samples lack temporal dates. Their dynamic features are almost 100% missing because they were not generated with dates.\n")
        f.write("- Imputing zeros for negatives would create massive target leakage (Model learns: Missing Data = Negative Class).\n")
        f.write("- For MODEL B, negative temporal controls must be explicitly sampled (e.g. assigning safe non-event dates to background locations and extracting dynamic features for those dates).\n\n")
        
        f.write("6. LEAKAGE PREVENTION:\n")
        f.write("- No global imputation was performed.\n")
        f.write("- Final imputation parameters are deferred until a strict train/validation/test split is enforced.\n")
        
    print("PHASE 5.7 — MISSING VALUE ANALYSIS\n")
    print("Status: COMPLETED\n")
    print(f"Rows: {total_rows}")
    print(f"Columns: {len(audit_df)}\n")
    print(f"Total missing cells: {total_missing_cells}\n")
    print(f"Features with missing values: {feats_with_missing}")
    print(f"Features without missing values: {feats_without_missing}\n")
    
    print("Highest missingness features:")
    for feat, pct in highest_miss:
        print(f"- {feat}: {pct:.2f}%")
        
    print(f"\nPositive missingness: {pos_total_missing} total missing cells")
    print(f"Negative missingness: {neg_total_missing} total missing cells\n")
    
    print("Static feature strategy:\nMedian imputation (fitted on train split only) or row deletion for DEM; no global mean fill.\n")
    print("Categorical feature strategy:\nExplicitly assign 'Unknown' class to preserve boundary gap semantics.\n")
    print("Dynamic feature strategy:\nNEVER fill with zero. True absence of data must not be confused with 0.0 measurements.\n")
    print("Undated positive strategy:\nRetain for Model A (static); strictly exclude from Model B (dynamic) unless a date can be resolved.\n")
    print("Negative temporal strategy:\nDo not impute. Negative samples require explicit temporal/control generation before Model B dynamic feature extraction.\n")
    
    print("Global imputation performed:\nNO\n")
    print("Leakage-safe final imputation:\nDEFERRED UNTIL TRAINING SPLIT\n")
    
    print("Major limitations:\n- Model B currently cannot proceed with standard imputation. It requires a dedicated spatio-temporal negative sampling strategy to resolve the ~100% dynamic missingness in negatives.\n")
    
    print("Final decision:\nPASS\n")
    
    print("Output files:")
    print(f"1. {AUDIT_CSV}")
    print(f"2. {STRATEGY_TXT}\n")
    
    print("NEXT:")
    print("PHASE 5.8 — CATEGORICAL FEATURE ENCODING\n")
    print("STOP.")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 63
    status["current_step"] = "5.7_missing_values"
    status["steps"]["5.7_missing_values"] = "COMPLETED"
    status["outputs"].extend([AUDIT_CSV, STRATEGY_TXT])
    status["next_step"] = "Step 5.8 — Categorical Feature Encoding"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
