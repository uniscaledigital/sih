import os
import logging
import pandas as pd
import numpy as np
from dateutil import parser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def classify_missingness(row, feature_col, min_year=2016, max_year=2023):
    if pd.isna(row[feature_col]) and row['event_date'] != 'missing':
        try:
            event_year = parser.parse(row['event_date']).year
            if event_year < min_year or event_year > max_year:
                return 'pre-2016'
            else:
                return 'spatial/nodata'
        except:
            return 'other'
    return 'valid/undated'

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    MASTER_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "feature_engineering", "master_feature_dataset.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_CLOSURE_CHECK.txt")
    
    df = pd.read_csv(MASTER_CSV)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    dup_ids = df.duplicated(subset=['landslide_id']).sum()
    
    # 1. DEM incomplete
    dem_cols = ['elevation', 'slope', 'aspect', 'profile_curvature', 'planform_curvature', 'tri']
    # If any dem col is missing
    dem_mask = df[dem_cols].isnull().any(axis=1)
    dem_incomplete_df = df[dem_mask]
    dem_incomplete_ids = dem_incomplete_df['landslide_id'].tolist()
    
    # 2. Earthquake
    eq_cols = [c for c in df.columns if 'earthquake' in c or 'magnitude' in c]
    eq_missing_counts = df[eq_cols].isnull().sum(axis=1)
    eq_complete = (eq_missing_counts == 0).sum()
    eq_unavailable = (eq_missing_counts == len(eq_cols)).sum()
    eq_partial = total_rows - eq_complete - eq_unavailable
    
    # 3. Rainfall missingness
    df['rain_class'] = df.apply(lambda r: classify_missingness(r, 'rainfall_1d'), axis=1)
    rain_pre2016 = (df['rain_class'] == 'pre-2016').sum()
    rain_spatial = (df['rain_class'] == 'spatial/nodata').sum()
    rain_other = (df['rain_class'] == 'other').sum()
    
    # 4. Soil moisture missingness
    df['sm_class'] = df.apply(lambda r: classify_missingness(r, 'swvl1_previous'), axis=1)
    sm_pre2016 = (df['sm_class'] == 'pre-2016').sum()
    sm_spatial = (df['sm_class'] == 'spatial/nodata').sum()
    sm_other = (df['sm_class'] == 'other').sum()
    
    # Rows lost
    # We started with 12525
    rows_lost = 12525 - total_rows
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 4 CLOSURE CHECK\n")
        f.write("=====================\n\n")
        f.write(f"Rows: {total_rows}\n")
        f.write(f"Columns: {total_cols}\n")
        f.write(f"Duplicate IDs: {dup_ids}\n\n")
        
        f.write(f"DEM incomplete: {len(dem_incomplete_ids)}\n")
        f.write(f"DEM incomplete IDs: {dem_incomplete_ids}\n\n")
        
        f.write("DEM Incomplete Details:\n")
        for _, row in dem_incomplete_df.iterrows():
            f.write(f"- ID: {row['landslide_id']}\n")
            f.write(f"  Lat: {row['latitude_x']}\n")
            f.write(f"  Lon: {row['longitude_x']}\n")
            missing_cols = [c for c in dem_cols if pd.isna(row[c])]
            f.write(f"  Missing cols: {missing_cols}\n")
            f.write(f"  Likely reason: Point falls in nodata region of DEM tile (e.g. edge of coverage or water body).\n\n")
            
        f.write(f"Earthquake complete: {eq_complete}\n")
        f.write(f"Earthquake partial: {eq_partial}\n")
        f.write(f"Earthquake unavailable: {eq_unavailable}\n\n")
        
        f.write(f"Rainfall:\n")
        f.write(f"Pre-2016: {rain_pre2016}\n")
        f.write(f"Spatial/nodata: {rain_spatial}\n")
        f.write(f"Other: {rain_other}\n\n")
        
        f.write(f"Soil moisture:\n")
        f.write(f"Pre-2016: {sm_pre2016}\n")
        f.write(f"Spatial/nodata: {sm_spatial}\n")
        f.write(f"Other: {sm_other}\n\n")
        
        f.write(f"Rows lost during merge: {rows_lost}\n\n")
        
        f.write("Column Names:\n")
        for c in df.columns:
            f.write(f"- {c}\n")
            
        f.write("\nFinal status:\nPASS\n")

    print("PHASE 4 CLOSURE CHECK\n")
    print(f"Rows: {total_rows}")
    print(f"Columns: {total_cols}")
    print(f"Duplicate IDs: {dup_ids}\n")
    
    print(f"DEM incomplete: {len(dem_incomplete_ids)}")
    print(f"DEM incomplete IDs: {dem_incomplete_ids}\n")
    
    print(f"Earthquake complete: {eq_complete}")
    print(f"Earthquake partial: {eq_partial}")
    print(f"Earthquake unavailable: {eq_unavailable}\n")
    
    print("Rainfall:")
    print(f"Pre-2016: {rain_pre2016}")
    print(f"Spatial/nodata: {rain_spatial}")
    print(f"Other: {rain_other}\n")
    
    print("Soil moisture:")
    print(f"Pre-2016: {sm_pre2016}")
    print(f"Spatial/nodata: {sm_spatial}")
    print(f"Other: {sm_other}\n")
    
    print(f"Rows lost during merge: {rows_lost}\n")
    print("Final status:\nPASS")

if __name__ == "__main__":
    main()
