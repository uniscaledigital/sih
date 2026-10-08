import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_landslides(input_csv, output_csv):
    logging.info(f"Loading raw landslide data from {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error(f"Failed to read input CSV: {e}")
        return

    initial_count = len(df)
    logging.info(f"Initial record count: {initial_count}")

    # Standardize column names if needed
    lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)

    if not lat_col or not lon_col:
        logging.error("Could not find latitude or longitude columns.")
        return

    # Filter to NER bounding box (Lat: 21-30 N, Lon: 88-98 E)
    df_filtered = df[(df[lat_col] >= 21) & (df[lat_col] <= 30) & 
                     (df[lon_col] >= 88) & (df[lon_col] <= 98)].copy()
    
    ner_count = len(df_filtered)
    logging.info(f"Records inside NER bounding box: {ner_count}")

    # Drop duplicate coordinates
    df_filtered = df_filtered.drop_duplicates(subset=[lat_col, lon_col])
    final_count = len(df_filtered)
    logging.info(f"Records after dropping duplicates: {final_count}")

    # Generate stable unique identifiers
    # Sort by latitude and longitude to ensure deterministic ordering before ID generation
    df_filtered = df_filtered.sort_values(by=[lat_col, lon_col]).reset_index(drop=True)
    df_filtered.insert(0, 'landslide_id', [f"LS_{i:05d}" for i in range(1, final_count + 1)])

    # Save to output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    logging.info(f"Validated dataset saved to {output_csv}")

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "landslides", "landslide_coordinates.csv")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    
    validate_landslides(INPUT_CSV, OUTPUT_CSV)
