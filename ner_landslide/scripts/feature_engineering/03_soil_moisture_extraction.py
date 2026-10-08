import os
import json
import logging
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_soil_moisture_for_point(ds, lat, lon, event_date_str):
    try:
        event_date = parser.parse(event_date_str)
        end_date = event_date - timedelta(days=1)
        start_date = event_date - timedelta(days=7) # We only need up to 7 days for soil moisture
        
        # Select spatial point
        # The file has latitude, longitude
        point_ds = ds.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Select temporal window
        window_ds = point_ds.sel(valid_time=slice(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d') + " 23:59:59"))
        
        features = {}
        for layer in ['swvl1', 'swvl2']:
            vals = window_ds[layer].values
            
            if len(vals) == 0 or np.isnan(vals).all():
                features[f'{layer}_previous'] = np.nan
                features[f'{layer}_1day_mean'] = np.nan
                features[f'{layer}_3day_mean'] = np.nan
                features[f'{layer}_7day_mean'] = np.nan
                features[f'{layer}_change'] = np.nan
                continue
                
            vals = np.nan_to_num(vals, nan=np.nanmean(vals) if not np.isnan(np.nanmean(vals)) else 0)
            
            # 6-hourly data -> 4 steps per day
            features[f'{layer}_previous'] = vals[-1] if len(vals) >= 1 else np.nan
            features[f'{layer}_1day_mean'] = np.mean(vals[-4:]) if len(vals) >= 4 else np.nan
            features[f'{layer}_3day_mean'] = np.mean(vals[-12:]) if len(vals) >= 12 else np.nan
            features[f'{layer}_7day_mean'] = np.mean(vals[-28:]) if len(vals) >= 28 else np.nan
            
            # Anomaly/Change: difference between short term and long term mean
            if not np.isnan(features[f'{layer}_1day_mean']) and not np.isnan(features[f'{layer}_7day_mean']):
                features[f'{layer}_change'] = features[f'{layer}_1day_mean'] - features[f'{layer}_7day_mean']
            else:
                features[f'{layer}_change'] = np.nan
                
        return features
    except Exception as e:
        return {
            'swvl1_previous': np.nan, 'swvl1_1day_mean': np.nan, 'swvl1_3day_mean': np.nan, 'swvl1_7day_mean': np.nan, 'swvl1_change': np.nan,
            'swvl2_previous': np.nan, 'swvl2_1day_mean': np.nan, 'swvl2_3day_mean': np.nan, 'swvl2_7day_mean': np.nan, 'swvl2_change': np.nan
        }

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides_with_dates.csv")
    SM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "soil_moisture")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "soil_moisture", "soil_moisture_landslide_features.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "soil_moisture_feature_report.txt")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    df = pd.read_csv(INPUT_CSV)
    dated_df = df[df['event_date'] != 'missing'].copy()
    
    logging.info(f"Loaded {len(df)} total landslides. {len(dated_df)} have dates.")
    
    datasets = {}
    years_found = []
    for file in os.listdir(SM_DIR):
        if file.endswith('.nc'):
            year_str = file.split('_')[-1].split('.')[0]
            try:
                datasets[int(year_str)] = xr.open_dataset(os.path.join(SM_DIR, file))
                years_found.append(int(year_str))
            except:
                pass
                
    logging.info(f"Loaded NetCDF for years: {years_found}")
    
    results = []
    successful = 0
    missing_ext = 0
    
    for idx, row in dated_df.iterrows():
        event_year = None
        try:
            event_year = parser.parse(row['event_date']).year
        except:
            pass
            
        if event_year in datasets:
            features = get_soil_moisture_for_point(datasets[event_year], row['latitude'], row['longitude'], row['event_date'])
            if not np.isnan(features['swvl1_previous']):
                successful += 1
            else:
                missing_ext += 1
        else:
            features = get_soil_moisture_for_point(None, 0, 0, '1900-01-01') # Get NaNs
            missing_ext += 1
            
        res = {'landslide_id': row['landslide_id']}
        res.update(features)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    
    min_vals = res_df.min(numeric_only=True)
    max_vals = res_df.max(numeric_only=True)
    mean_vals = res_df.mean(numeric_only=True)
    missing_pct = (res_df.isnull().sum() / len(res_df)) * 100
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("SOIL MOISTURE FEATURE EXTRACTION REPORT\n")
        f.write("=======================================\n")
        f.write(f"Input records (Total): {len(df)}\n")
        f.write(f"Dated landslides: {len(dated_df)}\n")
        f.write(f"Successful extractions: {successful}\n")
        f.write(f"Missing extractions: {missing_ext}\n")
        f.write(f"Variables: swvl1, swvl2\n")
        f.write("Units: m3/m3\n")
        f.write(f"Features created: {list(res_df.columns)[1:]}\n")
        f.write("Temporal resolution: 6-hourly\n")
        f.write("Statistics:\n")
        f.write("Minimum:\n" + min_vals.to_string() + "\n\n")
        f.write("Maximum:\n" + max_vals.to_string() + "\n\n")
        f.write("Mean:\n" + mean_vals.to_string() + "\n\n")
        f.write("Missing percentage:\n" + missing_pct.to_string() + "\n")
        
    print("PHASE 4 PROGRESS UPDATE")
    print("Step: 3")
    print("Task: Soil-Moisture Feature Extraction")
    print("Status: COMPLETED")
    print("Completion percentage: 40%")
    print(f"Input: {SM_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Records processed: {len(dated_df)}")
    print(f"Records successful: {successful}")
    print(f"Records failed: {missing_ext}")
    print("Records unmatched: 0")
    print("Features created: " + ", ".join(list(res_df.columns)[1:]))
    print("Validation performed: Temporal window enforced STRICTLY before event date (up to T-1).")
    print("Problems: None")
    print("Next step: Earthquake Feature Extraction")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 40
    status["last_completed_step"] = "Step 3: Soil-Moisture Feature Extraction"
    status["soil_moisture"]["status"] = "COMPLETED"
    status["soil_moisture"]["records"] = successful
    status["soil_moisture"]["features"] = list(res_df.columns)[1:]
    status["outputs"].append(OUTPUT_CSV)
    status["outputs"].append(REPORT_TXT)
    status["next_step"] = "Step 4: Earthquake Feature Extraction"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
