import os
import json
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    GEOLOGY_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "geology", "NGDR_Geology_2M.geojsonl")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "geology", "geology_landslide_features.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "geology_feature_report.txt")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Load all landslides (static features apply to all, not just dated ones)
    df = pd.read_csv(INPUT_CSV)
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    logging.info(f"Loaded {len(points_gdf)} landslide points.")
    
    # Load Geology GeoJSONL
    # Since it's 70MB, we can load it into memory
    geology_gdf = gpd.read_file(GEOLOGY_FILE)
    logging.info(f"Loaded {len(geology_gdf)} geology polygons.")
    
    # Ensure CRS match
    if geology_gdf.crs != points_gdf.crs:
        logging.info(f"Reprojecting geology from {geology_gdf.crs} to EPSG:4326")
        geology_gdf = geology_gdf.to_crs("EPSG:4326")
        
    # Spatial join
    # We want point-in-polygon
    joined = gpd.sjoin(points_gdf, geology_gdf, how="left", predicate="within")
    
    # If a point falls on a boundary, it might match multiple polygons. Keep the first.
    joined = joined.drop_duplicates(subset=['landslide_id'], keep='first')
    
    # Select columns
    extract_cols = ['landslide_id', 'age', 'stratigrap', 'stratigrap_new', 'group_', 'supergroup']
    
    # Ensure columns exist in the join result
    final_cols = ['landslide_id']
    for col in extract_cols[1:]:
        if col in joined.columns:
            final_cols.append(col)
            
    final_df = joined[final_cols].copy()
    
    # Rename for clarity
    rename_map = {
        'age': 'geology_age',
        'stratigrap': 'geology_stratigraphy',
        'stratigrap_new': 'geology_stratigraphy_new',
        'group_': 'geology_group',
        'supergroup': 'geology_supergroup'
    }
    final_df = final_df.rename(columns=rename_map)
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    successful = final_df['geology_age'].notnull().sum() if 'geology_age' in final_df.columns else 0
    missing = len(final_df) - successful
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("GEOLOGY SPATIAL JOIN REPORT\n")
        f.write("===========================\n")
        f.write(f"Total points processed: {len(df)}\n")
        f.write(f"Successful matches: {successful}\n")
        f.write(f"Missing/Unmatched points: {missing}\n")
        f.write(f"Features created: {list(final_df.columns)[1:]}\n")
        f.write("Method: Geopandas spatial join (within). Duplicates dropped by keeping first match.\n")
        
    print("PHASE 4 PROGRESS UPDATE")
    print("Step: 5")
    print("Task: Geology Spatial Join")
    print("Status: COMPLETED")
    print("Completion percentage: 60%")
    print(f"Input: {GEOLOGY_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Records processed: {len(df)}")
    print(f"Records successful: {successful}")
    print(f"Records failed/unmatched: {missing}")
    print("Features created: " + ", ".join(list(final_df.columns)[1:]))
    print("Validation performed: Point-in-polygon verification. CRS matched.")
    
    # Update Status
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 60
    status["last_completed_step"] = "Step 5: Geology Spatial Join"
    status["geology"]["status"] = "COMPLETED"
    status["geology"]["records"] = successful
    status["geology"]["features"] = list(final_df.columns)[1:]
    status["outputs"].append(OUTPUT_CSV)
    status["outputs"].append(REPORT_TXT)
    status["next_step"] = "Step 6: LULC Spatial Join"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
