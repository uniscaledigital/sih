import os
import glob
import logging
import math
import numpy as np
import pandas as pd
import rasterio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_topographic_features(window, lat, cell_size_deg):
    """
    Calculates topographic features from a 3x3 elevation window.
    window: 3x3 numpy array of elevations (meters)
    lat: latitude in degrees (used to scale x-distance)
    cell_size_deg: cell size in degrees (e.g., 0.0002777...)
    
    Returns: (elevation, slope_deg, aspect_deg, prof_curv, plan_curv, tri)
    """
    if window.shape != (3, 3) or np.isnan(window).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    z = window
    elevation = z[1, 1]

    # Constants for distance conversions
    # 1 degree of latitude is roughly 111,120 meters
    dy = cell_size_deg * 111120.0 
    # 1 degree of longitude varies by cosine of latitude
    dx = cell_size_deg * 111120.0 * math.cos(math.radians(lat))

    # Horn's formula for slope and aspect
    dz_dx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8 * dx)
    dz_dy = ((z[2, 0] + 2*z[2, 1] + z[2, 2]) - (z[0, 0] + 2*z[0, 1] + z[0, 2])) / (8 * dy)
    
    # Slope
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)
    
    # Aspect
    if dz_dx != 0:
        aspect_rad = math.atan2(dz_dy, -dz_dx)
        aspect_deg = math.degrees(aspect_rad)
        if aspect_deg < 0:
            aspect_deg += 360.0
    else:
        if dz_dy > 0:
            aspect_deg = 90.0
        elif dz_dy < 0:
            aspect_deg = 270.0
        else:
            aspect_deg = -1.0 # Flat

    # Curvature (Zevenbergen & Thorne method)
    D = ((z[1, 0] + z[1, 2]) / 2 - z[1, 1]) / (dx**2)
    E = ((z[0, 1] + z[2, 1]) / 2 - z[1, 1]) / (dy**2)
    F = (-z[0, 0] + z[0, 2] + z[2, 0] - z[2, 2]) / (4 * dx * dy)
    G = (-z[1, 0] + z[1, 2]) / (2 * dx)
    H = (z[0, 1] - z[2, 1]) / (2 * dy)
    
    p = G**2 + H**2
    if p > 0:
        prof_curv = -2 * (D * G**2 + E * H**2 + F * G * H) / p
        plan_curv = 2 * (D * H**2 + E * G**2 - F * G * H) / p
    else:
        prof_curv = 0.0
        plan_curv = 0.0

    # Terrain Ruggedness Index (TRI) - Riley et al. 1999
    tri = np.sqrt(np.sum((z - elevation)**2))
    
    return elevation, slope_deg, aspect_deg, prof_curv, plan_curv, tri

def extract_dem_features(landslides_csv, dem_dir, output_csv):
    logging.info(f"Loading landslide coordinates from {landslides_csv}")
    df = pd.read_csv(landslides_csv)
    
    # Identify lat/lon columns
    lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
    
    if not lat_col or not lon_col:
        logging.error("Could not find latitude or longitude columns.")
        return

    # Initialize feature columns
    df['elevation'] = np.nan
    df['slope'] = np.nan
    df['aspect'] = np.nan
    df['profile_curvature'] = np.nan
    df['planform_curvature'] = np.nan
    df['tri'] = np.nan

    tiles = glob.glob(os.path.join(dem_dir, '*.tif'))
    logging.info(f"Found {len(tiles)} DEM tiles.")

    points_processed = 0
    
    # Iterate over DEM tiles
    for tile_idx, tile_path in enumerate(tiles):
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
            
            # Find points that fall within this tile's bounds (with a small buffer to allow 3x3 window)
            cell_size = src.res[0] # assuming square pixels (dx == dy)
            minx, miny, maxx, maxy = bounds.left + cell_size, bounds.bottom + cell_size, bounds.right - cell_size, bounds.top - cell_size
            
            # Mask of points in this tile that haven't been processed yet
            mask = (df[lon_col] >= minx) & (df[lon_col] <= maxx) & \
                   (df[lat_col] >= miny) & (df[lat_col] <= maxy) & \
                   (df['elevation'].isna())
                   
            pts_in_tile = df[mask]
            
            if len(pts_in_tile) > 0:
                logging.info(f"Tile {tile_idx+1}/{len(tiles)} ({os.path.basename(tile_path)}): Processing {len(pts_in_tile)} points")
                
                # Load entire tile into memory (usually ~25MB for 3600x3600 float32/int16)
                arr = src.read(1).astype(float)
                
                # Handle nodata
                nodata = src.nodata
                # The audit showed some nodata is -255.0 or None. Let's explicitly catch negatives if needed.
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                arr[arr == -255.0] = np.nan
                
                for idx, row in pts_in_tile.iterrows():
                    lon = row[lon_col]
                    lat = row[lat_col]
                    
                    r, c = src.index(lon, lat)
                    
                    # Ensure 3x3 window is fully within array bounds
                    if 1 <= r < arr.shape[0] - 1 and 1 <= c < arr.shape[1] - 1:
                        window = arr[r-1:r+2, c-1:c+2]
                        elev, slope, aspect, pcurv, plcurv, tri = calculate_topographic_features(window, lat, cell_size)
                        
                        df.at[idx, 'elevation'] = elev
                        df.at[idx, 'slope'] = slope
                        df.at[idx, 'aspect'] = aspect
                        df.at[idx, 'profile_curvature'] = pcurv
                        df.at[idx, 'planform_curvature'] = plcurv
                        df.at[idx, 'tri'] = tri
                        
                        points_processed += 1
                        
                        if points_processed % 2000 == 0:
                            logging.info(f"Processed {points_processed} points...")

    # Log summary
    missing = df['elevation'].isna().sum()
    logging.info(f"Finished. Extracted features for {points_processed} points. Missing: {missing}")
    
    if missing > 0:
        logging.warning("Some points did not match any DEM tile or fell on nodata edges.")
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"DEM features saved to {output_csv}")

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "dem", "dem_landslide_features.csv")
    
    extract_dem_features(INPUT_CSV, DEM_DIR, OUTPUT_CSV)
