import os
import json

PROJECT_ROOT = "D:/sih/ner_landslide"
STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE4_STATUS.json")

status = {
  "phase": "Phase 4",
  "status": "IN_PROGRESS",
  "last_completed_step": "Step 7: Verify Existing DEM Features",
  "completion_percentage": 80,
  "event_dates": {
    "status": "COMPLETED",
    "total": 12525,
    "with_dates": 6339,
    "without_dates": 6186
  },
  "rainfall": {
    "status": "COMPLETED",
    "records": 2330,
    "features": ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d", "maximum_daily_rainfall", "mean_daily_rainfall"]
  },
  "soil_moisture": {
    "status": "COMPLETED",
    "records": 2356,
    "features": ["swvl1_previous", "swvl1_1day_mean", "swvl1_3day_mean", "swvl1_7day_mean", "swvl1_change", "swvl2_previous", "swvl2_1day_mean", "swvl2_3day_mean", "swvl2_7day_mean", "swvl2_change"]
  },
  "earthquakes": {
    "status": "COMPLETED",
    "records": 6339,
    "features": ["nearest_earthquake_distance_km", "nearest_earthquake_magnitude", "earthquake_count_25km_7d", "maximum_magnitude_25km_7d", "earthquake_count_25km_30d", "maximum_magnitude_25km_30d", "earthquake_count_25km_90d", "maximum_magnitude_25km_90d", "earthquake_count_50km_7d", "maximum_magnitude_50km_7d", "earthquake_count_50km_30d", "maximum_magnitude_50km_30d", "earthquake_count_50km_90d", "maximum_magnitude_50km_90d", "earthquake_count_100km_7d", "maximum_magnitude_100km_7d", "earthquake_count_100km_30d", "maximum_magnitude_100km_30d", "earthquake_count_100km_90d", "maximum_magnitude_100km_90d"]
  },
  "geology": {
    "status": "COMPLETED",
    "records": 12425,
    "features": ["geology_age", "geology_stratigraphy", "geology_stratigraphy_new", "geology_group", "geology_supergroup"]
  },
  "lulc": {
    "status": "IN_PROGRESS",
    "records": 0,
    "features": []
  },
  "dem_validation": {
    "status": "COMPLETED",
    "successful": 12525,
    "failed": 0
  },
  "master_dataset": {
    "status": "NOT_STARTED",
    "rows": 0,
    "columns": 0
  },
  "feature_engineering": {
    "status": "NOT_STARTED",
    "derived_features": []
  },
  "quality_control": {
    "missing_value_analysis": "NOT_STARTED",
    "outlier_analysis": "NOT_STARTED",
    "correlation_analysis": "NOT_STARTED"
  },
  "outputs": [],
  "problems": ["6186 records lacked explicit dates in the PDF."],
  "next_step": "Step 8: Merge All Features"
}

with open(STATUS_JSON, 'w') as f:
    json.dump(status, f, indent=2)
