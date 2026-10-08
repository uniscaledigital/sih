import os
import glob
import json
import logging
import math
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_aspect(window):
    if window.shape != (3, 3) or np.isnan(window).any():
        return np.nan, np.nan
    z = window
    dz_dx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0]))
    dz_dy = ((z[2, 0] + 2*z[2, 1] + z[2, 2]) - (z[0, 0] + 2*z[0, 1] + z[0, 2]))
    
    if dz_dx != 0:
        aspect_rad = math.atan2(dz_dy, -dz_dx)
        aspect_deg = math.degrees(aspect_rad)
        if aspect_deg < 0:
            aspect_deg += 360.0
    else:
        if dz_dy > 0:
            aspect_deg = 90.0
        elif dz_dy < 0:
            aspect_deg = 270.0
        else:
            aspect_deg = -1.0 
            
    aspect_sin = math.sin(math.radians(aspect_deg)) if aspect_deg != -1 else np.nan
    aspect_cos = math.cos(math.radians(aspect_deg)) if aspect_deg != -1 else np.nan
    return aspect_sin, aspect_cos

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    POS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "feature_engineering", "master_feature_dataset.csv")
    NEG_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "negative_samples_india_valid.csv")
    
    DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
    GEO_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "geology", "NGDR_Geology_2M.geojsonl")
    
    FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final")
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "artifacts")
    REP_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
    
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)
    
    TRAIN_CSV = os.path.join(FINAL_DIR, "modelA_train_final.csv")
    VAL_CSV = os.path.join(FINAL_DIR, "modelA_validation_final.csv")
    TEST_CSV = os.path.join(FINAL_DIR, "modelA_test_final.csv")
    PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "modelA_preprocessor.joblib")
    REP_TXT = os.path.join(REP_DIR, "phase5_fast_final_ml_dataset_report.txt")
    
    # 1. Load Positives
    logging.info("Loading positives...")
    pos_df = pd.read_csv(POS_CSV, low_memory=False)
    pos_df['label'] = 1
    if 'latitude_x' in pos_df.columns:
        pos_df.rename(columns={'latitude_x': 'latitude', 'longitude_x': 'longitude'}, inplace=True)
        
    pos_cols = ['latitude', 'longitude', 'elevation', 'slope', 'tri', 'aspect_sin', 'aspect_cos',
                'geology_age', 'geology_stratigraphy', 'geology_stratigraphy_new', 'geology_group', 'geology_supergroup', 'label']
    pos_df = pos_df[pos_cols].copy()
    
    # 2. Load Negatives
    logging.info("Loading negatives...")
    neg_df = pd.read_csv(NEG_CSV)
    neg_df['label'] = 0
    neg_cols = ['latitude', 'longitude', 'elevation', 'slope', 'tri', 'label']
    neg_df = neg_df[neg_cols].copy()
    
    # 3. Extract missing DEM features (aspect) for negatives
    logging.info("Extracting DEM features for negatives...")
    neg_df['aspect_sin'] = np.nan
    neg_df['aspect_cos'] = np.nan
    
    dem_files = glob.glob(os.path.join(DEM_DIR, "*.tif"))
    for f in dem_files:
        with rasterio.open(f) as src:
            bounds = src.bounds
            mask = (neg_df['latitude'] >= bounds.bottom) & (neg_df['latitude'] <= bounds.top) & \
                   (neg_df['longitude'] >= bounds.left) & (neg_df['longitude'] <= bounds.right) & \
                   (neg_df['aspect_sin'].isna())
            pts = neg_df[mask]
            if len(pts) > 0:
                for idx, row in pts.iterrows():
                    py, px = src.index(row['longitude'], row['latitude'])
                    try:
                        window = src.read(1, window=rasterio.windows.Window(px-1, py-1, 3, 3)).astype(float)
                        if src.nodata is not None:
                            window[window == src.nodata] = np.nan
                        window[window == -255.0] = np.nan
                        a_sin, a_cos = calculate_aspect(window)
                        neg_df.at[idx, 'aspect_sin'] = a_sin
                        neg_df.at[idx, 'aspect_cos'] = a_cos
                    except:
                        pass

    # 4. Extract Geology features for negatives
    logging.info("Extracting Geology for negatives...")
    geo_polygons = []
    geo_props = []
    with open(GEO_FILE, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                geo_polygons.append(shape(obj['geometry']))
                geo_props.append(obj.get('properties', {}))
            except:
                pass
                
    geo_tree = STRtree(geo_polygons)
    
    neg_df['geology_age'] = pd.Series(dtype='object')
    neg_df['geology_stratigraphy'] = pd.Series(dtype='object')
    neg_df['geology_stratigraphy_new'] = pd.Series(dtype='object')
    neg_df['geology_group'] = pd.Series(dtype='object')
    neg_df['geology_supergroup'] = pd.Series(dtype='object')
    
    for idx, row in neg_df.iterrows():
        pt = Point(row['longitude'], row['latitude'])
        candidates = geo_tree.query(pt)
        for c_idx in candidates:
            if geo_polygons[c_idx].contains(pt):
                props = geo_props[c_idx]
                neg_df.at[idx, 'geology_age'] = props.get('age')
                neg_df.at[idx, 'geology_stratigraphy'] = props.get('stratigrap')
                neg_df.at[idx, 'geology_stratigraphy_new'] = props.get('stratigrap_new')
                neg_df.at[idx, 'geology_group'] = props.get('group_')
                neg_df.at[idx, 'geology_supergroup'] = props.get('supergroup')
                break

    # Combine datasets
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    # 5. Missing-value analysis & Drop features with excessive missingness
    logging.info("Missing value analysis...")
    missing_counts = combined_df.isna().sum()
    missing_pcts = (missing_counts / len(combined_df)) * 100
    
    # Let's say excessive is > 10%
    excessive_thresh = 10.0
    cols_to_drop = missing_pcts[missing_pcts > excessive_thresh].index.tolist()
    logging.info(f"Dropping cols due to missingness > {excessive_thresh}%: {cols_to_drop}")
    combined_df.drop(columns=cols_to_drop, inplace=True)
    
    # Ensure predictor lists
    num_cols = ['elevation', 'slope', 'tri', 'aspect_sin', 'aspect_cos']
    cat_cols = ['geology_age', 'geology_stratigraphy', 'geology_stratigraphy_new', 'geology_group', 'geology_supergroup']
    
    num_cols = [c for c in num_cols if c in combined_df.columns]
    cat_cols = [c for c in cat_cols if c in combined_df.columns]
    
    missing_before = combined_df[num_cols + cat_cols].isna().sum().sum()
    
    # Spatial Split (70-15-15)
    logging.info("Spatial Splitting...")
    # Create spatial block id
    combined_df['lat_bin'] = pd.cut(combined_df['latitude'], bins=np.arange(0, 40, 0.5), labels=False)
    combined_df['lon_bin'] = pd.cut(combined_df['longitude'], bins=np.arange(60, 100, 0.5), labels=False)
    combined_df['spatial_block_id'] = combined_df['lat_bin'].astype(str) + "_" + combined_df['lon_bin'].astype(str)
    
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(splitter.split(combined_df, groups=combined_df['spatial_block_id']))
    
    train_df = combined_df.iloc[train_idx].copy()
    temp_df = combined_df.iloc[temp_idx].copy()
    
    val_splitter = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(val_splitter.split(temp_df, groups=temp_df['spatial_block_id']))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    # Check spatial leakage
    train_blocks = set(train_df['spatial_block_id'])
    val_blocks = set(val_df['spatial_block_id'])
    test_blocks = set(test_df['spatial_block_id'])
    
    leakage = train_blocks.intersection(val_blocks) or train_blocks.intersection(test_blocks) or val_blocks.intersection(test_blocks)
    spatial_leakage_status = "FAIL" if leakage else "PASS"
    
    # Prepare Preprocessor
    logging.info("Fitting Preprocessor...")
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
        
    X_train = train_df[num_cols + cat_cols]
    y_train = train_df['label']
    
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df['label']
    
    X_test = test_df[num_cols + cat_cols]
    y_test = test_df['label']
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    joblib.dump(preprocessor, PREPROC_PATH)
    
    # Get feature names after one hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    
    # Create final dataframes
    train_final = pd.DataFrame(X_train_processed, columns=all_feature_names)
    train_final['label'] = y_train.values
    
    val_final = pd.DataFrame(X_val_processed, columns=all_feature_names)
    val_final['label'] = y_val.values
    
    test_final = pd.DataFrame(X_test_processed, columns=all_feature_names)
    test_final['label'] = y_test.values
    
    # Check Missingness after
    missing_after = train_final.isna().sum().sum() + val_final.isna().sum().sum() + test_final.isna().sum().sum()
    
    # Validations
    coords_overlap = (cKDTree(pos_df[['latitude', 'longitude']].values).query(neg_df[['latitude', 'longitude']].values)[0] == 0).sum()
    dups = combined_df.duplicated(subset=['latitude', 'longitude']).sum()
    coord_leakage_status = "FAIL" if dups > 0 or coords_overlap > 0 else "PASS"
    
    forbidden = ['latitude', 'longitude', 'landslide_id', 'event_date']
    target_leakage_status = "FAIL" if any(f in all_feature_names for f in forbidden + ['label']) else "PASS"
    
    train_final.to_csv(TRAIN_CSV, index=False)
    val_final.to_csv(VAL_CSV, index=False)
    test_final.to_csv(TEST_CSV, index=False)
    
    # Generate Report
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5 — FAST FINAL ML DATASET REPORT\n")
        f.write("=======================================\n\n")
        
        f.write(f"Total rows: {len(combined_df)}\n")
        f.write(f"Train: {len(train_final)}\n")
        f.write(f"Validation: {len(val_final)}\n")
        f.write(f"Test: {len(test_final)}\n\n")
        
        f.write(f"Final predictor count: {len(all_feature_names)}\n")
        f.write(f"Numerical predictors: {len(num_cols)} ({', '.join(num_cols)})\n")
        f.write(f"Categorical predictors: {len(cat_cols)} ({', '.join(cat_cols)})\n\n")
        
        f.write(f"Dropped features: {', '.join(cols_to_drop) if cols_to_drop else 'None'}\n")
        f.write("Explicitly skipped: LULC\n\n")
        
        f.write(f"Missing values before preprocessing: {missing_before}\n")
        f.write(f"Missing values after preprocessing: {missing_after}\n\n")
        
        f.write(f"Spatial leakage: {spatial_leakage_status}\n")
        f.write(f"Coordinate leakage: {coord_leakage_status}\n")
        f.write(f"Target leakage: {target_leakage_status}\n\n")
        
        pos_ratio = (y_train.sum() + y_val.sum() + y_test.sum()) / len(combined_df)
        f.write(f"Class balance: Positive ratio = {pos_ratio:.4f} (Target ~ 0.5000)\n\n")
        
        final_decision = "PASS" if all([spatial_leakage_status == "PASS", coord_leakage_status == "PASS", target_leakage_status == "PASS"]) else "FAIL"
        f.write(f"Final decision: {final_decision}\n")
        f.write("STOP.\n")
        
    print("Fast final dataset built successfully.")

if __name__ == "__main__":
    main()
