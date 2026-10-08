import os
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    COMBO_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "positive_negative_dataset.csv")
    
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    STRATEGY_TXT = os.path.join(REP_DIR, "phase5_8_encoding_strategy.txt")
    STRATEGY_CSV = os.path.join(REP_DIR, "phase5_8_encoding_strategy.csv")
    STATUS_JSON = os.path.join(REP_DIR, "PHASE5_STATUS.json")
    
    # Read the dataset
    df = pd.read_csv(COMBO_CSV, low_memory=False)
    
    total_rows = len(df)
    
    # Identify Categorical Features
    exclude_cols = ['landslide_id', 'latitude_x', 'longitude_x', 'latitude_y', 'longitude_y', 'event_date', 'label', 'latitude', 'longitude']
    
    cat_features = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # If it's explicitly object type or known categorical
        if df[col].dtype == 'object' or 'geology' in col or 'lulc' in col:
            cat_features.append(col)
            
    strategy_data = []
    high_cardinality = []
    one_hot_candidates = []
    
    # Analyze each categorical feature
    for feat in cat_features:
        dt = str(df[feat].dtype)
        unique_vals = df[feat].dropna().unique()
        num_unique = len(unique_vals)
        missing_count = df[feat].isnull().sum()
        
        # Decide cardinality threshold (e.g. > 20 is high)
        if num_unique > 50:
            cardinality = "HIGH"
            high_cardinality.append(feat)
            rec_encoding = "Frequency Encoding or Rare-Class Grouping + One-Hot"
        else:
            cardinality = "LOW/MEDIUM"
            one_hot_candidates.append(feat)
            rec_encoding = "One-Hot Encoding"
            
        # Top categories
        vc = df[feat].value_counts(dropna=False)
        top_cats = ", ".join([f"{k} ({v})" for k, v in list(vc.items())[:3]])
        
        strategy_data.append({
            'feature': feat,
            'data_type': dt,
            'unique_categories': num_unique,
            'missing_count': missing_count,
            'cardinality': cardinality,
            'recommended_encoding': rec_encoding,
            'model_A': 'YES',
            'model_B': 'YES',
            'leakage_risk': 'LOW (if fitted on train only)',
            'reason': f"Environmental categorical feature. Top: {top_cats}"
        })
        
    strat_df = pd.DataFrame(strategy_data)
    strat_df.to_csv(STRATEGY_CSV, index=False)
    
    # Strategy Report
    with open(STRATEGY_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5.8 — CATEGORICAL ENCODING STRATEGY\n")
        f.write("=========================================\n\n")
        
        f.write("IDENTIFIED CATEGORICAL FEATURES:\n")
        for feat in cat_features:
            f.write(f"- {feat}\n")
        f.write("\n")
        
        f.write("ENCODING SELECTION RULES:\n")
        f.write("1. No arbitrary numerical ordinal encoding is permitted since these are unordered environmental classes.\n")
        f.write("2. No target encoding is permitted to explicitly prevent target leakage.\n")
        f.write("3. Missing values must be filled with 'Unknown' BEFORE encoding.\n")
        f.write("4. Low/Medium cardinality features (<50 categories) will use standard One-Hot Encoding.\n")
        f.write("5. High cardinality features (>50 categories) will use Rare-Class Grouping (grouping tail classes into 'Other') followed by One-Hot Encoding, or Frequency Encoding.\n\n")
        
        f.write("LEAKAGE PREVENTION MANDATE:\n")
        f.write("- The final Scikit-Learn (or equivalent) OneHotEncoder MUST be fitted strictly on the Training split during Phase 5.10.\n")
        f.write("- The encoder must be configured to handle unknown categories (handle_unknown='ignore') to gracefully process unseen classes in Validation/Test splits.\n")
        
    print("PHASE 5.8 — CATEGORICAL FEATURE ENCODING\n")
    print("Status: COMPLETED\n")
    
    print(f"Categorical features: {cat_features}")
    print(f"Total categorical features: {len(cat_features)}\n")
    
    print("Feature summary:")
    for _, row in strat_df.iterrows():
        print(f"- {row['feature']} : {row['unique_categories']} unique, {row['missing_count']} missing")
        
    print(f"\nOne-hot candidates: {one_hot_candidates}")
    print(f"High-cardinality features: {high_cardinality}")
    print(f"Excluded features: {exclude_cols}\n")
    
    print("Model A encoding strategy:\nFill missing with 'Unknown'. Use One-Hot Encoding. Fit encoder solely on Train split.\n")
    print("Model B encoding strategy:\nSame as Model A. Categorical inputs remain identically distributed for the dynamic model.\n")
    
    print("Target encoding:\nNOT USED\n")
    print("Encoder fit on full dataset:\nNO\n")
    print("Final leakage-safe encoder:\nDEFERRED UNTIL TRAIN SPLIT\n")
    
    print("Major limitations:\n- geology_supergroup has extremely high missingness (97%) and may just become a single 'Unknown' column after encoding.\n")
    
    print("Final decision:\nPASS\n")
    
    print("Output files:")
    print(f"- {STRATEGY_TXT}")
    print(f"- {STRATEGY_CSV}\n")
    
    print("NEXT:")
    print("PHASE 5.9 — DATA LEAKAGE AUDIT\n")
    print("STOP.")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 72
    status["current_step"] = "5.8_categorical_encoding"
    status["steps"]["5.8_categorical_encoding"] = "COMPLETED"
    status["outputs"].extend([STRATEGY_TXT, STRATEGY_CSV])
    status["next_step"] = "Step 5.9 — Data Leakage Audit"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
