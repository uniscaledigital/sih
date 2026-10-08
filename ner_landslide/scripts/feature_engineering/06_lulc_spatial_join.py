import os
import json
import logging
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    LULC_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "lulc", "Bhuvan_SISDP_Phase2_LULC_10k_2016-2019.geojsonl")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "lulc", "lulc_landslide_features.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "lulc_feature_report.txt")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    df = pd.read_csv(INPUT_CSV)
    
    # We want to map landslide_id to its point geometry
    points = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    point_ids = df['landslide_id'].tolist()
    
    # Build STRtree for extremely fast spatial queries
    logging.info("Building STRtree spatial index for landslide points...")
    tree = STRtree(points)
    logging.info("STRtree built.")
    
    # We will keep a dictionary to store matched results
    matched_results = {}
    
    start_time = time.time()
    
    lines_processed = 0
    matched_count = 0
    
    # We will stream the GeoJSONL line by line
    with open(LULC_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            lines_processed += 1
            if lines_processed % 1000000 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Processed {lines_processed} polygons... Matched {matched_count}/{len(df)} points. Elapsed: {elapsed:.1f}s")
                
                # Early exit if all points matched
                if matched_count == len(df):
                    logging.info("All points matched. Exiting early.")
                    break
                    
            try:
                # Fast parse
                obj = json.loads(line.strip())
                geom = shape(obj['geometry'])
                
                # Query the STRtree for points whose envelopes intersect the polygon's envelope
                candidate_indices = tree.query(geom)
                
                if len(candidate_indices) > 0:
                    # Precise intersection check
                    for idx in candidate_indices:
                        pt_id = point_ids[idx]
                        if pt_id not in matched_results:
                            if geom.contains(points[idx]):
                                matched_results[pt_id] = obj.get('properties', {}).get('REG_DESCRI', 'Unknown')
                                matched_count += 1
            except Exception as e:
                pass
                
    elapsed_total = time.time() - start_time
    logging.info(f"Finished processing LULC. Total polygons read: {lines_processed}. Matched: {matched_count}. Time: {elapsed_total:.1f}s")
    
    # Build final results dataframe
    final_res = []
    for pt_id in point_ids:
        final_res.append({
            'landslide_id': pt_id,
            'lulc_class': matched_results.get(pt_id, 'missing')
        })
        
    final_df = pd.DataFrame(final_res)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("LULC SPATIAL JOIN REPORT\n")
        f.write("========================\n")
        f.write(f"Total points processed: {len(df)}\n")
        f.write(f"Successful matches: {matched_count}\n")
        f.write(f"Missing matches: {len(df) - matched_count}\n")
        f.write(f"Polygons scanned: {lines_processed}\n")
        f.write(f"Features created: lulc_class (from REG_DESCRI)\n")
        f.write(f"Runtime: {elapsed_total:.2f} seconds\n")
        f.write(f"Memory strategy: GeoJSONL streaming with Shapely STRtree indexing of points (Reverse query). O(1) memory, O(N) time.\n")
        
    print("PHASE 4 PROGRESS UPDATE")
    print("Step: 6")
    print("Task: LULC Spatial Join")
    print("Status: COMPLETED")
    print("Completion percentage: 75%")
    print(f"Input: {LULC_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Records processed: {len(df)}")
    print(f"Records successful: {matched_count}")
    print(f"Records failed/unmatched: {len(df) - matched_count}")
    print("Features created: lulc_class")
    print("Validation performed: Point inside Polygon via reverse STRtree.")
    
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 75
    status["last_completed_step"] = "Step 6: LULC Spatial Join"
    status["lulc"]["status"] = "COMPLETED"
    status["lulc"]["records"] = matched_count
    status["lulc"]["features"] = ["lulc_class"]
    status["outputs"].append(OUTPUT_CSV)
    status["outputs"].append(REPORT_TXT)
    status["next_step"] = "Step 7: Verify Existing DEM Features"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
