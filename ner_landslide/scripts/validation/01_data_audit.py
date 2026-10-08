import os
import sys
import glob
import json
import logging
from datetime import datetime
import pandas as pd
import rasterio
import geopandas as gpd
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def write_section_header(f, title):
    f.write(f"\n{'='*50}\n")
    f.write(f"{title}\n")
    f.write(f"{'='*50}\n")

def check_environment(f):
    write_section_header(f, "1. SOFTWARE ENVIRONMENT")
    f.write(f"Python Version: {sys.version}\n")
    f.write(f"pandas: {pd.__version__}\n")
    f.write(f"rasterio: {rasterio.__version__}\n")
    f.write(f"geopandas: {gpd.__version__}\n")
    f.write(f"xarray: {xr.__version__}\n")
    logging.info("Environment check complete.")

def check_landslides(f, data_dir):
    write_section_header(f, "2. LANDSLIDE DATA AUDIT")
    csv_path = os.path.join(data_dir, 'landslides', 'landslide_coordinates.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            f.write(f"File: {csv_path}\n")
            f.write(f"Row count: {len(df)}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Missing values:\n{df.isnull().sum().to_string()}\n")
            
            # NER bounding box check: Lat 21-30, Lon 88-98
            # Assuming columns might be 'latitude', 'longitude' or 'lat', 'lon'
            lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                valid_lat = df[(df[lat_col] >= 21) & (df[lat_col] <= 30)]
                valid_lon = df[(df[lon_col] >= 88) & (df[lon_col] <= 98)]
                valid_both = df[(df[lat_col] >= 21) & (df[lat_col] <= 30) & (df[lon_col] >= 88) & (df[lon_col] <= 98)]
                f.write(f"Valid Latitudes (21-30 N): {len(valid_lat)} / {len(df)}\n")
                f.write(f"Valid Longitudes (88-98 E): {len(valid_lon)} / {len(df)}\n")
                f.write(f"Inside NER bounding box: {len(valid_both)} / {len(df)}\n")
                
            duplicates = df.duplicated(subset=[lat_col, lon_col]) if (lat_col and lon_col) else df.duplicated()
            f.write(f"Duplicate records: {duplicates.sum()}\n")
        except Exception as e:
            f.write(f"Error reading landslide CSV: {e}\n")
    else:
        f.write("Landslide CSV not found.\n")
    logging.info("Landslide audit complete.")

def check_dem(f, data_dir):
    write_section_header(f, "3. DEM DATA AUDIT")
    dem_dir = os.path.join(data_dir, 'dem')
    tiles = glob.glob(os.path.join(dem_dir, '*.tif'))
    f.write(f"Found {len(tiles)} GeoTIFF tiles.\n")
    
    if tiles:
        f.write("Summary of first 5 tiles (or fewer):\n")
        for tile in tiles[:5]:
            try:
                with rasterio.open(tile) as src:
                    f.write(f"- File: {os.path.basename(tile)}\n")
                    f.write(f"  CRS: {src.crs}\n")
                    f.write(f"  Resolution: {src.res}\n")
                    f.write(f"  Bounds: {src.bounds}\n")
                    f.write(f"  Dtype: {src.dtypes}\n")
                    f.write(f"  Nodata: {src.nodata}\n")
            except Exception as e:
                f.write(f"  Error reading tile {os.path.basename(tile)}: {e}\n")
    logging.info("DEM audit complete.")

def check_netcdf(f, data_dir, category):
    write_section_header(f, f"4. {category.upper()} DATA AUDIT")
    target_dir = os.path.join(data_dir, category)
    files = glob.glob(os.path.join(target_dir, '*.nc'))
    f.write(f"Found {len(files)} NetCDF files.\n")
    
    if files:
        sample_file = files[0]
        try:
            f.write(f"Inspecting sample file: {os.path.basename(sample_file)}\n")
            ds = xr.open_dataset(sample_file)
            f.write(f"Dimensions: {dict(ds.dims)}\n")
            f.write(f"Coordinates: {list(ds.coords.keys())}\n")
            f.write(f"Data Variables: {list(ds.data_vars.keys())}\n")
            f.write(f"Attributes (subset): { {k: v for i, (k, v) in enumerate(ds.attrs.items()) if i < 5} }\n")
            ds.close()
        except Exception as e:
            f.write(f"Error reading NetCDF file: {e}\n")
    logging.info(f"{category.upper()} audit complete.")

def check_earthquakes(f, data_dir):
    write_section_header(f, "5. EARTHQUAKE DATA AUDIT")
    eq_dir = os.path.join(data_dir, 'earthquakes')
    csv_files = glob.glob(os.path.join(eq_dir, '*.csv'))
    f.write(f"Found {len(csv_files)} earthquake CSV files.\n")
    
    total_records = 0
    try:
        if csv_files:
            dfs = []
            for cf in csv_files:
                df_temp = pd.read_csv(cf)
                dfs.append(df_temp)
            combined_df = pd.concat(dfs, ignore_index=True)
            total_records = len(combined_df)
            f.write(f"Total combined row count: {total_records}\n")
            f.write(f"Columns: {list(combined_df.columns)}\n")
            f.write(f"Missing values:\n{combined_df.isnull().sum().to_string()}\n")
            f.write(f"Duplicate records: {combined_df.duplicated().sum()}\n")
            if 'time' in combined_df.columns:
                f.write(f"Time parsing test (first 5 records):\n{combined_df['time'].head().to_string()}\n")
    except Exception as e:
        f.write(f"Error reading earthquake data: {e}\n")
    logging.info("Earthquake audit complete.")

def check_geojsonl(f, file_path, category):
    write_section_header(f, f"6. {category.upper()} DATA AUDIT")
    f.write(f"File: {file_path}\n")
    
    if os.path.exists(file_path):
        f.write(f"File size: {os.path.getsize(file_path) / (1024*1024*1024):.2f} GB\n")
        try:
            # Read first few lines instead of whole file
            records = 0
            sample_properties = None
            sample_geometry_type = None
            
            with open(file_path, 'r', encoding='utf-8') as geo_f:
                for line in geo_f:
                    if records == 0:
                        try:
                            obj = json.loads(line.strip())
                            sample_properties = list(obj.get('properties', {}).keys())
                            sample_geometry_type = obj.get('geometry', {}).get('type')
                        except json.JSONDecodeError:
                            pass
                    records += 1
                    
                    # For huge files like LULC, don't read entirely unless we want a full count.
                    # We will read 10,000 lines max for the audit to save time, unless it's geology.
                    if records >= 100000:
                        f.write(f"Counted up to {records} records (stopped for time limit).\n")
                        break
                else:
                    f.write(f"Total record count: {records}\n")
                    
            f.write(f"Sample Geometry Type: {sample_geometry_type}\n")
            f.write(f"Sample Properties: {sample_properties}\n")
        except Exception as e:
            f.write(f"Error inspecting {category}: {e}\n")
    else:
        f.write(f"File not found: {file_path}\n")
    logging.info(f"{category.upper()} audit complete.")

def run_audit(base_dir, output_file):
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"DATA AUDIT REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        check_environment(f)
        check_landslides(f, raw_data_dir)
        check_dem(f, raw_data_dir)
        check_netcdf(f, raw_data_dir, 'rainfall')
        check_netcdf(f, raw_data_dir, 'soil_moisture')
        check_earthquakes(f, raw_data_dir)
        
        geology_file = os.path.join(raw_data_dir, 'geology', 'NGDR_Geology_2M.geojsonl')
        check_geojsonl(f, geology_file, 'Geology')
        
        lulc_file = os.path.join(raw_data_dir, 'lulc', 'Bhuvan_SISDP_Phase2_LULC_10k_2016-2019.geojsonl')
        check_geojsonl(f, lulc_file, 'LULC')
        
    logging.info(f"Data audit report generated at: {output_file}")

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "reports", "data_audit_report.txt")
    run_audit(PROJECT_ROOT, OUTPUT_FILE)
