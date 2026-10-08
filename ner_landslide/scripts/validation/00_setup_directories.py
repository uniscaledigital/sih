import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def setup_directories(base_dir):
    directories = [
        "data/processed/landslides",
        "data/processed/dem",
        "data/processed/rainfall",
        "data/processed/soil_moisture",
        "data/processed/earthquakes",
        "data/processed/geology",
        "data/processed/lulc",
        "data/final/training_dataset",
        "data/final/prediction_dataset",
        "scripts/validation",
        "scripts/dem",
        "scripts/rainfall",
        "scripts/soil_moisture",
        "scripts/earthquakes",
        "scripts/geology",
        "scripts/lulc",
        "scripts/feature_engineering",
        "scripts/modeling",
        "models/trained",
        "models/scalers",
        "outputs/maps",
        "outputs/figures",
        "outputs/predictions",
        "outputs/reports",
        "notebooks",
        "logs"
    ]

    for d in directories:
        dir_path = os.path.join(base_dir, d)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
        else:
            logging.info(f"Directory already exists: {dir_path}")

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    setup_directories(PROJECT_ROOT)
    logging.info("Directory setup complete.")
