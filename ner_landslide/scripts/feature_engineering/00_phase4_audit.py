import os
import glob
import logging
import json
import pandas as pd
import xarray as xr
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def write_section_header(f, title):
    f.write(f"\n{'='*70}\n")
    f.write(f"{title}\n")
    f.write(f"{'='*70}\n")

def check_netcdf(f, data_dir, category):
    write_section_header(f, f"1. {category.upper()} DATA AUDIT")
    target_dir = os.path.join(data_dir, category)
    files = sorted(glob.glob(os.path.join(target_dir, '*.nc')))
    f.write(f"File Count: {len(files)}\n")
    if len(files) <= 5:
        f.write(f"Files: {[os.path.basename(x) for x in files]}\n")
    else:
        f.write(f"Files (first 5): {[os.path.basename(x) for x in files[:5]]}\n")
    
    if files:
        sample_file = files[0]
        try:
            ds = xr.open_dataset(sample_file)
            f.write(f"\nDetailed Inspection of: {os.path.basename(sample_file)}\n")
            f.write(f"Dimensions: {dict(ds.sizes)}\n")
            f.write(f"Variables: {list(ds.data_vars.keys())}\n")
            
            for var_name in ds.data_vars:
                var_da = ds[var_name]
                f.write(f"  -> Variable '{var_name}' attributes:\n")
                f.write(f"     Units: {var_da.attrs.get('units', 'Unknown')}\n")
                f.write(f"     FillValue/Missing: {var_da.attrs.get('_FillValue', 'Unknown')}\n")
            
            # Check temporal extent
            if 'time' in ds.coords or 'valid_time' in ds.coords:
                t_var = 'time' if 'time' in ds.coords else 'valid_time'
                t_vals = ds[t_var].values
                f.write(f"Temporal extent ({t_var}): {t_vals.min()} to {t_vals.max()}\n")
            
            # Check spatial extent
            lat_var = 'lat' if 'lat' in ds.coords else 'latitude'
            lon_var = 'lon' if 'lon' in ds.coords else 'longitude'
            if lat_var in ds.coords and lon_var in ds.coords:
                lat_vals = ds[lat_var].values
                lon_vals = ds[lon_var].values
                f.write(f"Spatial extent ({lat_var}): {lat_vals.min()} to {lat_vals.max()}\n")
                f.write(f"Spatial extent ({lon_var}): {lon_vals.min()} to {lon_vals.max()}\n")
                f.write(f"Spatial resolution: Lat ~{abs(lat_vals[1] - lat_vals[0]):.4f} deg, Lon ~{abs(lon_vals[1] - lon_vals[0]):.4f} deg\n")
            ds.close()
        except Exception as e:
            f.write(f"Error inspecting NetCDF: {e}\n")

def check_earthquakes(f, data_dir):
    write_section_header(f, "2. EARTHQUAKE DATA AUDIT")
    eq_dir = os.path.join(data_dir, 'earthquakes')
    csv_files = glob.glob(os.path.join(eq_dir, '*.csv'))
    
    dfs = []
    for cf in csv_files:
        dfs.append(pd.read_csv(cf))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        f.write(f"Total Combined Records: {len(df)}\n")
        f.write(f"Unique Earthquake IDs: {df['id'].nunique()}\n")
        f.write(f"Duplicate IDs: {len(df) - df['id'].nunique()}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Magnitude range: {df['mag'].min()} to {df['mag'].max()}\n")
        f.write(f"Temporal extent (time string limits): {df['time'].min()} to {df['time'].max()}\n")
        f.write(f"Missing values summary:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}\n")
    else:
        f.write("No earthquake CSVs found.\n")

def check_geojsonl(f, file_path, category, sample_limit=1000):
    write_section_header(f, f"3. {category.upper()} DATA AUDIT")
    f.write(f"File: {os.path.basename(file_path)}\n")
    if not os.path.exists(file_path):
        f.write("File not found.\n")
        return
        
    f.write(f"File Size: {os.path.getsize(file_path) / (1024**3):.2f} GB\n")
    
    records = 0
    props = set()
    geom_types = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as geo_f:
            for line in geo_f:
                if records < sample_limit:
                    try:
                        obj = json.loads(line.strip())
                        props.update(obj.get('properties', {}).keys())
                        geom_types.add(obj.get('geometry', {}).get('type'))
                    except:
                        pass
                records += 1
        
        f.write(f"Total Record Count: {records}\n")
        f.write(f"Geometry Types (from sample): {geom_types}\n")
        f.write(f"Properties (from sample): {list(props)}\n")
    except Exception as e:
        f.write(f"Error inspecting GeoJSONL: {e}\n")

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "reports", "phase4_source_audit.txt")
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("PHASE 4 SOURCE AUDIT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        check_netcdf(f, raw_data_dir, 'rainfall')
        check_netcdf(f, raw_data_dir, 'soil_moisture')
        check_earthquakes(f, raw_data_dir)
        
        geology_file = os.path.join(raw_data_dir, 'geology', 'NGDR_Geology_2M.geojsonl')
        check_geojsonl(f, geology_file, 'Geology')
        
        lulc_file = os.path.join(raw_data_dir, 'lulc', 'Bhuvan_SISDP_Phase2_LULC_10k_2016-2019.geojsonl')
        # We don't read the whole 70GB file line by line just for the audit unless we have to, 
        # but counting lines in a 70GB file takes ~5-10 minutes.
        f.write("\nNOTE: LULC counting may be skipped if it's too large, falling back to sample count.\n")
        check_geojsonl(f, lulc_file, 'LULC', sample_limit=100)
    
    logging.info(f"Audit saved to {OUTPUT_FILE}")
