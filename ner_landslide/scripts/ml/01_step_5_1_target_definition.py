import os
import json

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    TXT_REPORT = os.path.join(PROJECT_ROOT, "outputs", "reports", "phase5_1_target_definition.txt")
    JSON_REPORT = os.path.join(PROJECT_ROOT, "outputs", "reports", "phase5_1_target_definition.json")
    STATUS_JSON = os.path.join(PROJECT_ROOT, "outputs", "reports", "PHASE5_STATUS.json")

    # 1. Generate text report
    report_text = """PHASE 5.1 — TARGET DEFINITION

1. WHAT THE MODEL PREDICTS:
The project encompasses two related but distinct concepts: Susceptibility and Early Warning. Because 6,186 of our positive records lack explicit dates (meaning we cannot map historical rainfall/soil moisture/earthquakes to them), the project mathematically demands TWO ML targets for two separate models:
- Target A (Susceptibility): Binary probability that a given spatial point is historically prone to landslides based purely on static environmental conditions.
- Target B (Dynamic Event/Early Warning): Binary probability that a landslide will trigger on a specific date given antecedent static AND dynamic conditions.

2. CLASS DEFINITIONS:
- Class 1 (Positive): Verified historical landslide occurrence location.
- Class 0 (Negative): Background non-landslide location (to be generated in Step 5.4 via spatial/environmental constraints).

3. SUSCEPTIBILITY VS EVENT PREDICTION:
- BOTH are required. 
- The Susceptibility model will use all 12,525 positive samples with static features (DEM, Geology, LULC).
- The Event model will use the 6,339 dated positive samples with both static and dynamic features (Rainfall, Soil Moisture, Earthquakes).

4. EARLY-WARNING RELATIONSHIP:
The Susceptibility model creates the baseline spatial hazard map. The Event model acts as the dynamic overlay, integrating rolling 1D/3D/7D/15D/30D rainfall and soil moisture anomalies at T-1 to generate operational early-warning alerts for high-susceptibility zones.

5. PREDICTION TIME AVAILABILITY:
At operational prediction time (T), we only have access to data up to T-1. All our dynamic features were strictly engineered to stop at T-1 (e.g. antecedent rainfall, previous day's soil moisture, earthquakes strictly before the event), ensuring zero temporal leakage.

6. POTENTIAL TARGET LEAKAGE:
- Future weather data implicitly bleeding into historical averages.
- Using coordinates (latitude/longitude) directly as features.
- Spatially correlated data (random train/test splits that place points 5 meters apart into different sets, causing spatial data leakage).
"""

    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 2. Generate JSON report
    json_data = {
      "task_type": "Binary Classification (Dual-Model Framework)",
      "target_name": "landslide_occurrence",
      "positive_class": "1 (Verified historical landslide)",
      "negative_class": "0 (Background non-landslide sample)",
      "prediction_definition": "Probability of landslide occurrence given environmental factors",
      "susceptibility_or_event_prediction": "Both (Requires separate models due to undated records)",
      "early_warning_relation": "Event model drives dynamic temporal early-warning; Susceptibility provides baseline spatial hazard.",
      "potential_leakage_variables": [
        "latitude", 
        "longitude",
        "future_dynamic_measurements"
      ],
      "recommended_target_strategy": "Dual binary classification. Target A = Spatial Susceptibility. Target B = Dynamic Event Trigger.",
      "confidence": "High",
      "limitations": [
        "6,186 records cannot be used for the Dynamic Event Trigger model due to missing dates.",
        "Requires scientifically rigorous negative sampling to balance both spatial and spatio-temporal domains."
      ]
    }

    with open(JSON_REPORT, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    # 3. Update PHASE5_STATUS.json
    status = {
      "phase": "Phase 5",
      "status": "IN_PROGRESS",
      "completion_percentage": 9,
      "current_step": "5.1_target_definition",
      "steps": {
        "5.1_target_definition": "COMPLETED",
        "5.2_master_dataset_inspection": "NOT_STARTED",
        "5.3_static_dynamic_classification": "NOT_STARTED",
        "5.4_negative_sampling": "NOT_STARTED",
        "5.5_sampling_validation": "NOT_STARTED",
        "5.6_positive_negative_dataset": "NOT_STARTED",
        "5.7_missing_values": "NOT_STARTED",
        "5.8_categorical_encoding": "NOT_STARTED",
        "5.9_leakage_audit": "NOT_STARTED",
        "5.10_train_validation_test_split": "NOT_STARTED",
        "5.11_final_ml_dataset": "NOT_STARTED"
      },
      "rows": 0,
      "columns": 0,
      "positive_samples": 0,
      "negative_samples": 0,
      "features": [],
      "removed_features": [],
      "problems": [],
      "outputs": [
          TXT_REPORT,
          JSON_REPORT
      ],
      "next_step": "Step 5.2 — Inspect Master Dataset"
    }

    with open(STATUS_JSON, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)

    # 4. Final Terminal Output
    print("PHASE 5.1 — TARGET DEFINITION\n")
    print("Status: COMPLETED")
    print("Task type: Binary Classification (Dual-Model Framework)")
    print("Target: landslide_occurrence (1/0)")
    print("Positive class: 1 (Verified historical landslide)")
    print("Negative class: 0 (Background non-landslide sample)")
    print("Prediction meaning: Probability of landslide occurrence given environmental factors\n")
    print("Susceptibility model:\nYES\n")
    print("Event/trigger model:\nYES\n")
    print("Early-warning relationship:\nEvent model drives dynamic temporal early-warning; Susceptibility provides baseline spatial hazard.\n")
    print("Potential leakage variables:\n- latitude\n- longitude\n- Spatially correlated neighbors in random splits\n")
    print("Recommended strategy:\nDual binary classification. Target A = Spatial Susceptibility (all 12k records). Target B = Dynamic Event Trigger (6k dated records).\n")
    print("Limitations:\n- 6,186 records cannot be used for the Event model due to missing dates.\n")
    print(f"Output report: {TXT_REPORT}")
    print(f"Output JSON: {JSON_REPORT}\n")
    print("NEXT:")
    print("PHASE 5.2 — MASTER DATASET INSPECTION\n")
    print("STOP.")

if __name__ == "__main__":
    main()
