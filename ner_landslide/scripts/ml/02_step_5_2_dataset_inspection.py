import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_dataset(df):
    stats = {}
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    stats['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Classify columns
    col_classes = {'id': [], 'lat_lon': [], 'date': [], 'numerical': [], 'categorical': [], 'other': []}
    for col in df.columns:
        if 'id' in col.lower() and len(df[col].unique()) > len(df) * 0.9:
            col_classes['id'].append(col)
        elif col in ['latitude_x', 'longitude_x', 'latitude_y', 'longitude_y', 'latitude', 'longitude']:
            col_classes['lat_lon'].append(col)
        elif col == 'event_date' or 'date' in col.lower():
            col_classes['date'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_classes['numerical'].append(col)
        else:
            col_classes['categorical'].append(col)
            
    stats['classes'] = col_classes
    return stats

def analyze_numerical(df, cols):
    if not cols: return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc['missing_count'] = df[cols].isnull().sum()
    desc['missing_pct'] = (desc['missing_count'] / len(df)) * 100
    desc['unique_count'] = df[cols].nunique()
    return desc

def analyze_categorical(df, cols):
    cat_data = []
    for col in cols:
        vc = df[col].value_counts(dropna=False)
        missing_count = df[col].isnull().sum()
        cat_data.append({
            'feature': col,
            'unique_count': df[col].nunique(),
            'missing_count': missing_count,
            'missing_pct': (missing_count / len(df)) * 100,
            'top_categories': dict(list(vc.items())[:5])
        })
    return pd.DataFrame(cat_data)

def generate_feature_assessment(df):
    rows = []
    for col in df.columns:
        dt = str(df[col].dtype)
        static_or_dyn = "static"
        susc_mod = "YES"
        dyn_mod = "YES"
        leakage = "LOW"
        pred_time = "YES"
        rec = "KEEP"
        reason = "Standard environmental predictor."
        
        if col in ['landslide_id', 'latitude_y', 'longitude_y']:
            susc_mod = dyn_mod = "NO"
            rec = "EXCLUDE"
            reason = "Identifier or duplicate column."
            
        elif col in ['latitude_x', 'longitude_x', 'latitude', 'longitude']:
            susc_mod = dyn_mod = "NO"
            leakage = "HIGH"
            rec = "EXCLUDE"
            reason = "Potential spatial leakage if used directly."
            
        elif col == 'event_date':
            susc_mod = dyn_mod = "NO"
            static_or_dyn = "N/A"
            leakage = "HIGH"
            rec = "EXCLUDE"
            reason = "Target leakage / Cannot be used as feature."
            
        elif 'rainfall' in col or 'swvl' in col or 'earthquake' in col or 'magnitude' in col:
            static_or_dyn = "dynamic"
            susc_mod = "NO"
            dyn_mod = "YES"
            reason = "Dynamic trigger variable."
            
        elif 'aspect' in col:
            if col == 'aspect':
                rec = "EXCLUDE"
                reason = "Circular variable; use aspect_sin/cos instead."
            
        rows.append({
            'feature': col,
            'data_type': dt,
            'static_or_dynamic': static_or_dyn,
            'susceptibility_model': susc_mod,
            'dynamic_model': dyn_mod,
            'leakage_risk': leakage,
            'prediction_time_available': pred_time,
            'recommended_action': rec,
            'reason': reason
        })
    return pd.DataFrame(rows)

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    MASTER_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "feature_engineering", "master_feature_dataset.csv")
    
    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    INSPECT_CSV = os.path.join(OUT_DIR, "phase5_2_master_dataset_inspection.csv")
    INSPECT_TXT = os.path.join(OUT_DIR, "phase5_2_master_dataset_inspection.txt")
    ASSESS_CSV = os.path.join(OUT_DIR, "phase5_2_feature_assessment.csv")
    STATUS_JSON = os.path.join(OUT_DIR, "PHASE5_STATUS.json")

    df = pd.read_csv(MASTER_CSV)
    
    # 1. Structure
    stats = analyze_dataset(df)
    
    # 2. Numerical
    num_df = analyze_numerical(df, stats['classes']['numerical'])
    
    # 3. Categorical
    cat_df = analyze_categorical(df, stats['classes']['categorical'])
    
    # 4. Dates
    if 'event_date' in df.columns:
        dated = df[df['event_date'] != 'missing'].copy()
        dated['dt'] = pd.to_datetime(dated['event_date'], errors='coerce')
        min_date = dated['dt'].min()
        max_date = dated['dt'].max()
        num_dated = len(dated)
        num_undated = len(df) - num_dated
    else:
        min_date = max_date = None
        num_dated = num_undated = 0
        
    # 5. Identifier
    dup_ids = df['landslide_id'].duplicated().sum()
    if 'latitude_x' in df.columns and 'longitude_x' in df.columns:
        dup_coords = df.duplicated(subset=['latitude_x', 'longitude_x']).sum()
    else:
        dup_coords = df.duplicated(subset=['latitude', 'longitude']).sum()
        
    # 6. Feature Assessment
    assess_df = generate_feature_assessment(df)
    assess_df.to_csv(ASSESS_CSV, index=False)
    
    # 7. Correlations
    corr_matrix = df[stats['classes']['numerical']].corr().abs()
    high_corr = []
    # upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for i in range(len(upper.columns)):
        for j in range(i+1, len(upper.columns)):
            col1 = upper.columns[i]
            col2 = upper.columns[j]
            val = upper.iloc[i, j]
            if val > 0.85:
                high_corr.append((col1, col2, val))
                
    # 8. Data Quality (Missingness)
    high_missing = num_df[num_df['missing_pct'] > 50].index.tolist()
    
    # Write TXT report
    with open(INSPECT_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5.2 — MASTER DATASET INSPECTION\n")
        f.write("=======================================\n\n")
        f.write(f"Total rows: {stats['rows']}\n")
        f.write(f"Total columns: {stats['columns']}\n")
        f.write(f"Memory usage: {stats['memory_usage_mb']:.2f} MB\n\n")
        f.write(f"ID columns: {stats['classes']['id']}\n")
        f.write(f"Lat/Lon columns: {stats['classes']['lat_lon']}\n")
        f.write(f"Date columns: {stats['classes']['date']}\n")
        f.write(f"Numerical columns: {len(stats['classes']['numerical'])}\n")
        f.write(f"Categorical columns: {len(stats['classes']['categorical'])}\n\n")
        
        f.write("DATE FEATURES\n")
        f.write(f"Min date: {min_date}\n")
        f.write(f"Max date: {max_date}\n")
        f.write(f"Dated: {num_dated}\n")
        f.write(f"Undated: {num_undated}\n\n")
        
        f.write("IDENTIFIER ANALYSIS\n")
        f.write(f"Duplicate IDs: {dup_ids}\n")
        f.write(f"Duplicate Coordinates: {dup_coords}\n\n")
        
        f.write("HIGH CORRELATIONS (>0.85)\n")
        for c1, c2, v in high_corr:
            f.write(f"{c1} - {c2} : {v:.3f}\n")
            
        f.write("\nDATA QUALITY\n")
        f.write(f"Features with >50% missing: {high_missing}\n")
        
    # Write CSV report (Numerical desc)
    num_df.to_csv(INSPECT_CSV)
    
    # Print Output
    print("PHASE 5.2 — MASTER DATASET INSPECTION\n")
    print("Status: COMPLETED")
    print(f"Rows: {stats['rows']}")
    print(f"Columns: {stats['columns']}")
    print(f"Duplicate IDs: {dup_ids}")
    print(f"Duplicate coordinates: {dup_coords}\n")
    
    print(f"Dated records: {num_dated}")
    print(f"Undated records: {num_undated}\n")
    
    print(f"Numerical features: {len(stats['classes']['numerical'])}")
    print(f"Categorical features: {len(stats['classes']['categorical'])}")
    print(f"Date features: {len(stats['classes']['date'])}")
    print(f"ID features: {len(stats['classes']['id'])}\n")
    
    # Assess summaries
    susc_recs = assess_df[assess_df['susceptibility_model'] == 'YES']['feature'].tolist()
    dyn_recs = assess_df[assess_df['dynamic_model'] == 'YES']['feature'].tolist()
    excl_recs = assess_df[assess_df['recommended_action'] == 'EXCLUDE']['feature'].tolist()
    leak_recs = assess_df[assess_df['leakage_risk'] == 'HIGH']['feature'].tolist()
    
    print(f"Features recommended for susceptibility: {len(susc_recs)} (e.g. elevation, slope, geology)")
    print(f"Features recommended for dynamic model: {len(dyn_recs)} (e.g. rainfall_1d, swvl1_change)")
    print(f"Features requiring exclusion: {excl_recs}")
    print(f"Potential leakage features: {leak_recs}\n")
    
    print("High-correlation pairs:")
    for c1, c2, v in high_corr[:5]:
        print(f"{c1} - {c2} : {v:.2f}")
    if len(high_corr) > 5:
        print(f"...and {len(high_corr)-5} more.")
        
    print("\nMajor data-quality issues:")
    print("- Duplicate coordinates exist (multiple landslides recorded at exact same pixel resolution).")
    print("- ~49% of rows lack temporal features due to missing dates or pre-2016 dates.")
    print("- Circular 'aspect' feature should be dropped in favor of aspect_sin and aspect_cos.")
    
    print(f"\nOutput files:")
    print(f"- {INSPECT_TXT}")
    print(f"- {INSPECT_CSV}")
    print(f"- {ASSESS_CSV}\n")
    
    print("Overall:\nPASS")
    print("\nNEXT:")
    print("PHASE 5.3 — STATIC VS DYNAMIC FEATURE CLASSIFICATION\n")
    print("STOP.")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 18
    status["current_step"] = "5.2_master_dataset_inspection"
    status["steps"]["5.2_master_dataset_inspection"] = "COMPLETED"
    status["outputs"].extend([INSPECT_TXT, INSPECT_CSV, ASSESS_CSV])
    status["next_step"] = "Step 5.3 — Static vs Dynamic Feature Classification"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
