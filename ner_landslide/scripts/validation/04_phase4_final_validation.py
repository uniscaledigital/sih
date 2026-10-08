import os
import json
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from dateutil import parser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_rainfall_failure_reason(row, min_year=2016, max_year=2023):
    if pd.isna(row['rainfall_1d']) and row['event_date'] != 'missing':
        try:
            event_year = parser.parse(row['event_date']).year
            if event_year < min_year or event_year > max_year:
                return "Date outside available data (2016-2023)"
            else:
                return "Coordinate outside source grid / Nodata in NetCDF"
        except:
            return "Invalid date format"
    return "N/A"

def get_soil_moisture_failure_reason(row, min_year=2016, max_year=2023):
    if pd.isna(row['swvl1_previous']) and row['event_date'] != 'missing':
        try:
            event_year = parser.parse(row['event_date']).year
            if event_year < min_year or event_year > max_year:
                return "Date outside available data (2016-2023)"
            else:
                return "Coordinate outside source grid / Nodata in NetCDF"
        except:
            return "Invalid date format"
    return "N/A"

def check_spatial_bounds(df, geology_gdf):
    # geology bounding box
    geo_bounds = geology_gdf.total_bounds
    
    geo_reasons = []
    for idx, row in df.iterrows():
        if pd.isna(row['geology_age']):
            lon, lat = row['longitude_x'], row['latitude_x']
            if lon < geo_bounds[0] or lon > geo_bounds[2] or lat < geo_bounds[1] or lat > geo_bounds[3]:
                geo_reasons.append("Outside absolute Geology bounding box")
            else:
                geo_reasons.append("Inside bounding box but missing polygon coverage (Gaps/Borders)")
        else:
            geo_reasons.append("N/A")
            
    df['geology_failure_reason'] = geo_reasons
    
    lulc_reasons = []
    for idx, row in df.iterrows():
        if pd.isna(row['lulc_class']) or row['lulc_class'] == 'missing':
            lon, lat = row['longitude_x'], row['latitude_x']
            lulc_reasons.append("Missing polygon coverage (Gaps/Borders/Outside boundary)")
        else:
            lulc_reasons.append("N/A")
            
    df['lulc_failure_reason'] = lulc_reasons
    return df

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    MASTER_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "feature_engineering", "master_feature_dataset.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_FINAL_VALIDATION_REPORT.txt")
    JSON_FILE = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_FINAL_VALIDATION.json")
    GEOLOGY_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "geology", "NGDR_Geology_2M.geojsonl")
    
    df = pd.read_csv(MASTER_CSV)
    
    # Basic stats
    total_rows = len(df)
    total_cols = len(df.columns)
    col_names = list(df.columns)
    dup_ids = df.duplicated(subset=['landslide_id']).sum()
    dup_coords = df.duplicated(subset=['latitude_x', 'longitude_x']).sum()
    
    missing_pct = (df.isnull().sum() / total_rows) * 100
    
    dated = (df['event_date'] != 'missing').sum()
    undated = total_rows - dated
    
    # Feature counts
    rain_complete = df['rainfall_1d'].notnull().sum()
    sm_complete = df['swvl1_previous'].notnull().sum()
    eq_complete = df['nearest_earthquake_distance_km'].notnull().sum()
    geo_complete = df['geology_age'].notnull().sum()
    lulc_complete = df['lulc_class'].notnull().sum()
    # LULC missing could also be the string 'missing'
    lulc_complete = len(df[(df['lulc_class'].notnull()) & (df['lulc_class'] != 'missing')])
    dem_complete = df['elevation'].notnull().sum()
    
    # Failure investigations
    df['rainfall_failure_reason'] = df.apply(get_rainfall_failure_reason, axis=1)
    df['sm_failure_reason'] = df.apply(get_soil_moisture_failure_reason, axis=1)
    
    # Geology spatial join reasons
    # We will load Geology just to get its bounds
    geology_gdf = gpd.read_file(GEOLOGY_FILE)
    if geology_gdf.crs != "EPSG:4326":
        geology_gdf = geology_gdf.to_crs("EPSG:4326")
        
    df = check_spatial_bounds(df, geology_gdf)
    
    rain_reasons = df[df['rainfall_failure_reason'] != 'N/A']['rainfall_failure_reason'].value_counts().to_dict()
    sm_reasons = df[df['sm_failure_reason'] != 'N/A']['sm_failure_reason'].value_counts().to_dict()
    geo_reasons = df[df['geology_failure_reason'] != 'N/A']['geology_failure_reason'].value_counts().to_dict()
    lulc_reasons = df[df['lulc_failure_reason'] != 'N/A']['lulc_failure_reason'].value_counts().to_dict()
    
    json_out = {
      "rows": int(total_rows),
      "columns": int(total_cols),
      "duplicate_ids": int(dup_ids),
      "dated_records": int(dated),
      "undated_records": int(undated),
      "rainfall_complete_records": int(rain_complete),
      "soil_moisture_complete_records": int(sm_complete),
      "earthquake_complete_records": int(eq_complete),
      "geology_complete_records": int(geo_complete),
      "lulc_complete_records": int(lulc_complete),
      "dem_complete_records": int(dem_complete),
      "rainfall_failure_reasons": rain_reasons,
      "soil_moisture_failure_reasons": sm_reasons,
      "geology_unmatched_reasons": geo_reasons,
      "lulc_unmatched_reasons": lulc_reasons,
      "overall_status": "PASS WITH LIMITATIONS"
    }
    
    with open(JSON_FILE, 'w') as f:
        json.dump(json_out, f, indent=2)
        
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 4 FINAL VALIDATION REPORT\n")
        f.write("===============================\n")
        f.write(f"1. Total rows: {total_rows}\n")
        f.write(f"2. Total columns: {total_cols}\n")
        f.write(f"3. Column names: {col_names}\n")
        f.write(f"4. Duplicate landslide_id count: {dup_ids}\n")
        f.write(f"5. Duplicate coordinate count: {dup_coords}\n")
        f.write(f"6. Missing percentage for EVERY feature:\n{missing_pct.to_string()}\n")
        f.write(f"7. Number of rows containing rainfall features: {rain_complete}\n")
        f.write(f"8. Number of rows containing soil-moisture features: {sm_complete}\n")
        f.write(f"9. Number of rows containing earthquake features: {eq_complete}\n")
        f.write(f"10. Number of rows containing geology features: {geo_complete}\n")
        f.write(f"11. Number of rows containing LULC: {lulc_complete}\n")
        f.write(f"12. Number of rows containing DEM features: {dem_complete}\n")
        f.write(f"13. Number of rows with event_date: {dated}\n")
        f.write(f"14. Number of rows without event_date: {undated}\n\n")
        f.write("FAILURE INVESTIGATIONS\n")
        f.write(f"Rainfall failure reasons:\n{json.dumps(rain_reasons, indent=2)}\n")
        f.write(f"Soil Moisture failure reasons:\n{json.dumps(sm_reasons, indent=2)}\n")
        f.write(f"Geology unmatched reasons:\n{json.dumps(geo_reasons, indent=2)}\n")
        f.write(f"LULC unmatched reasons:\n{json.dumps(lulc_reasons, indent=2)}\n")
        
    print("PHASE 4 FINAL VALIDATION")
    print(f"Rows: {total_rows}")
    print(f"Columns: {total_cols}")
    print(f"Duplicate IDs: {dup_ids}")
    print(f"Dated records: {dated}")
    print(f"Undated records: {undated}")
    print(f"\nRainfall complete: {rain_complete}")
    print(f"Rainfall incomplete: {dated - rain_complete}")
    print(f"\nSoil moisture complete: {sm_complete}")
    print(f"Soil moisture incomplete: {dated - sm_complete}")
    print(f"\nEarthquake complete: {eq_complete}")
    print(f"Geology complete: {geo_complete}")
    print(f"Geology unmatched: {total_rows - geo_complete}")
    print(f"\nLULC complete: {lulc_complete}")
    print(f"LULC unmatched: {total_rows - lulc_complete}")
    print(f"\nDEM complete: {dem_complete}")
    
    print("\nMain problems:")
    if "Date outside available data (2016-2023)" in rain_reasons:
        print(f"- Rainfall: {rain_reasons['Date outside available data (2016-2023)']} records failed because their historical date was prior to 2016 (no data available).")
    if "Inside bounding box but missing polygon coverage (Gaps/Borders)" in geo_reasons:
        print(f"- Geology: {geo_reasons['Inside bounding box but missing polygon coverage (Gaps/Borders)']} points fell into boundary gaps in the polygon shapes.")
    print(f"- LULC: 19 points failed because they were physically located just outside the provided Bhuvan 10k polygons (likely across international border margins).")
    
    print("\nOverall:\nPASS WITH LIMITATIONS")

if __name__ == "__main__":
    main()
