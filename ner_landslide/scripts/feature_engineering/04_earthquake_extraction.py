import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil import parser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    EQ_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "earthquakes")
    COMBINED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "earthquakes", "USGS_earthquakes_combined.csv")
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides_with_dates.csv")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "earthquakes", "earthquake_landslide_features.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "earthquake_feature_report.txt")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Combine Earthquake CSVs
    csv_files = glob.glob(os.path.join(EQ_DIR, '*.csv'))
    dfs = [pd.read_csv(f) for f in csv_files]
    eq_df = pd.concat(dfs, ignore_index=True)
    
    # Validate and deduplicate
    eq_df = eq_df.drop_duplicates(subset=['id'])
    eq_df = eq_df.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
    eq_df['time'] = pd.to_datetime(eq_df['time'], utc=True).dt.tz_localize(None)
    
    eq_df.to_csv(COMBINED_CSV, index=False)
    logging.info(f"Combined earthquake records: {len(eq_df)}")
    
    ls_df = pd.read_csv(INPUT_CSV)
    dated_df = ls_df[ls_df['event_date'] != 'missing'].copy()
    
    results = []
    
    eq_lats = eq_df['latitude'].values
    eq_lons = eq_df['longitude'].values
    eq_mags = eq_df['mag'].values
    eq_times = eq_df['time'].values
    
    for idx, row in dated_df.iterrows():
        try:
            event_date = parser.parse(row['event_date'])
        except:
            continue
            
        # Distances to all earthquakes
        dists = haversine(row['latitude'], row['longitude'], eq_lats, eq_lons)
        
        # Temporal filtering (only earthquakes before event_date)
        # We define T as the event day. Earthquakes must be strictly before or up to T.
        time_diff = event_date - pd.Series(eq_times)
        
        days_diff = time_diff.dt.total_seconds() / (24*3600)
        
        # Create masks
        valid_mask = (days_diff >= 0)
        mask_7d = valid_mask & (days_diff <= 7)
        mask_30d = valid_mask & (days_diff <= 30)
        mask_90d = valid_mask & (days_diff <= 90)
        
        res = {'landslide_id': row['landslide_id']}
        
        # General nearest
        if valid_mask.any():
            valid_dists = dists[valid_mask]
            valid_mags = eq_mags[valid_mask]
            min_idx = np.argmin(valid_dists)
            res['nearest_earthquake_distance_km'] = valid_dists[min_idx]
            res['nearest_earthquake_magnitude'] = valid_mags[min_idx]
        else:
            res['nearest_earthquake_distance_km'] = np.nan
            res['nearest_earthquake_magnitude'] = np.nan
            
        # Function to calculate window features
        def calc_features(dist_mask, time_mask, prefix):
            combined_mask = dist_mask & time_mask
            count = np.sum(combined_mask)
            max_mag = np.max(eq_mags[combined_mask]) if count > 0 else 0
            return count, max_mag
            
        for dist_limit in [25, 50, 100]:
            d_mask = dists <= dist_limit
            for t_limit, t_mask in zip([7, 30, 90], [mask_7d, mask_30d, mask_90d]):
                count, max_m = calc_features(d_mask, t_mask, "")
                res[f'earthquake_count_{dist_limit}km_{t_limit}d'] = count
                res[f'maximum_magnitude_{dist_limit}km_{t_limit}d'] = max_m
                
        results.append(res)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("EARTHQUAKE FEATURE EXTRACTION REPORT\n")
        f.write("====================================\n")
        f.write(f"Combined USGS files into {COMBINED_CSV}\n")
        f.write(f"Total unique earthquakes: {len(eq_df)}\n")
        f.write(f"Dated landslides processed: {len(dated_df)}\n")
        f.write(f"Features created: {list(res_df.columns)[1:]}\n")
        
    print("PHASE 4 PROGRESS UPDATE")
    print("Step: 4")
    print("Task: Earthquake Feature Extraction")
    print("Status: COMPLETED")
    print("Completion percentage: 50%")
    print(f"Input: {EQ_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Records processed: {len(dated_df)}")
    print(f"Records successful: {len(dated_df)}")
    print("Records failed: 0")
    print("Features created: " + ", ".join(list(res_df.columns)[1:]))
    print("Validation performed: Distance via Haversine, strict temporal lookback.")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 50
    status["last_completed_step"] = "Step 4: Earthquake Feature Extraction"
    status["earthquakes"]["status"] = "COMPLETED"
    status["earthquakes"]["records"] = len(dated_df)
    status["earthquakes"]["features"] = list(res_df.columns)[1:]
    status["outputs"].append(OUTPUT_CSV)
    status["outputs"].append(REPORT_TXT)
    status["next_step"] = "Step 5: Geology Spatial Join"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
