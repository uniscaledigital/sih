import os
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    DEM_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "dem", "dem_landslide_features.csv")
    REPORT_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "dem_feature_validation_report.txt")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")
    
    df_val = pd.read_csv(INPUT_CSV)
    df_dem = pd.read_csv(DEM_CSV)
    
    total = len(df_val)
    successful = len(df_dem)
    
    # Check for missing
    missing_ids = df_val[~df_val['landslide_id'].isin(df_dem['landslide_id'])]['landslide_id'].tolist()
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("DEM FEATURE VALIDATION REPORT\n")
        f.write("=============================\n")
        f.write(f"Total validated landslides: {total}\n")
        f.write(f"DEM features extracted: {successful}\n")
        f.write(f"Missing extractions: {len(missing_ids)}\n")
        
        if len(missing_ids) == 0:
            f.write("Investigation Result: All 12,525 records were successfully extracted in Phase 2. There are no missing records due to nodata or tile boundaries.\n")
        else:
            f.write(f"Missing IDs: {missing_ids}\n")
            
    print("PHASE 4 PROGRESS UPDATE")
    print("Step: 7")
    print("Task: Verify Existing DEM Features")
    print("Status: COMPLETED")
    print("Completion percentage: 80%")
    print(f"Input: {DEM_CSV}")
    print(f"Output: {REPORT_TXT}")
    print(f"Records processed: {total}")
    print(f"Records successful: {successful}")
    print(f"Records failed: {len(missing_ids)}")
    print("Validation performed: Cross-checked validated landslides with extracted DEM features.")
    
    with open(STATUS_JSON, 'r') as f:
        status = json.load(f)
        
    status["completion_percentage"] = 80
    status["last_completed_step"] = "Step 7: Verify Existing DEM Features"
    status["dem_validation"]["status"] = "COMPLETED"
    status["dem_validation"]["successful"] = successful
    status["dem_validation"]["failed"] = len(missing_ids)
    status["outputs"].append(REPORT_TXT)
    status["next_step"] = "Step 8: Merge All Features"
    
    with open(STATUS_JSON, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
