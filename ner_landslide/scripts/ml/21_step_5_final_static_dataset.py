import os
import glob
import json
import logging
import math
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_topographic_features(window, lat, res):
    if window.shape != (3, 3) or np.isnan(window).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    z = window
    elevation = z[1, 1]
    
    dy = res * 111120.0 
    dx = res * 111120.0 * math.cos(math.radians(lat))
    
    dz_dx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8 * dx)
    dz_dy = ((z[2, 0] + 2*z[2, 1] + z[2, 2]) - (z[0, 0] + 2*z[0, 1] + z[0, 2])) / (8 * dy)
    
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope = math.degrees(slope_rad)
    
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
            aspect_deg = -1.0 
            
    aspect_sin = math.sin(math.radians(aspect_deg)) if aspect_deg != -1 else np.nan
    aspect_cos = math.cos(math.radians(aspect_deg)) if aspect_deg != -1 else np.nan
    
    tri = np.sqrt(np.sum((z - elevation)**2))
    
    return elevation, slope, tri, aspect_deg, aspect_sin, aspect_cos

def main():
    PROJECT_ROOT = "D:/sih/ner_landslide"
    POS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "feature_engineering", "master_feature_dataset.csv")
    NEG_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "negative_samples_india_valid.csv")
    
    DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
    GEO_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "geology", "NGDR_Geology_2M.geojsonl")
    LULC_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "lulc", "Bhuvan_SISDP_Phase2_LULC_10k_2016-2019.geojsonl")
    
    OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "final_static_dataset.csv")
    REP_TXT = os.path.join(PROJECT_ROOT, "outputs", "reports", "phase5_final_static_dataset_report.txt")
    
    # 1. Load Data
    logging.info("Loading dataset...")
    pos_df = pd.read_csv(POS_CSV, low_memory=False)
    pos_df['label'] = 1
    # standardize coords
    if 'latitude_x' in pos_df.columns:
        pos_df.rename(columns={'latitude_x': 'latitude', 'longitude_x': 'longitude'}, inplace=True)
    
    neg_df = pd.read_csv(NEG_CSV)
    neg_df['label'] = 0
    if 'landslide_id' not in neg_df.columns and 'negative_id' in neg_df.columns:
        neg_df.rename(columns={'negative_id': 'landslide_id'}, inplace=True)

    # 2. Extract DEM features for negatives (we need aspect)
    logging.info("Extracting DEM features for negatives...")
    neg_df['aspect_sin'] = np.nan
    neg_df['aspect_cos'] = np.nan
    neg_df['aspect'] = np.nan
    # We already have elevation, slope, tri in neg_df, but let's recompute aspect quickly
    
    dem_files = glob.glob(os.path.join(DEM_DIR, "*.tif"))
    for f in dem_files:
        with rasterio.open(f) as src:
            bounds = src.bounds
            res = src.res[0]
            mask = (neg_df['latitude'] >= bounds.bottom) & (neg_df['latitude'] <= bounds.top) & \
                   (neg_df['longitude'] >= bounds.left) & (neg_df['longitude'] <= bounds.right) & \
                   (neg_df['aspect'].isna())
            pts = neg_df[mask]
            
            if len(pts) > 0:
                for idx, row in pts.iterrows():
                    py, px = src.index(row['longitude'], row['latitude'])
                    try:
                        window = src.read(1, window=rasterio.windows.Window(px-1, py-1, 3, 3)).astype(float)
                        if src.nodata is not None:
                            window[window == src.nodata] = np.nan
                        window[window == -255.0] = np.nan
                        
                        _, _, _, aspect, aspect_sin, aspect_cos = calculate_topographic_features(window, row['latitude'], res)
                        neg_df.at[idx, 'aspect'] = aspect
                        neg_df.at[idx, 'aspect_sin'] = aspect_sin
                        neg_df.at[idx, 'aspect_cos'] = aspect_cos
                    except:
                        pass
    dem_matched = neg_df['aspect'].notna().sum()
    logging.info(f"DEM extraction matched {dem_matched}/{len(neg_df)}")

    # 3. Extract Geology for negatives
    logging.info("Extracting Geology for negatives...")
    geo_polygons = []
    geo_props = []
    with open(GEO_FILE, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                geo_polygons.append(shape(obj['geometry']))
                geo_props.append(obj.get('properties', {}))
            except:
                pass
                
    geo_tree = STRtree(geo_polygons)
    
    neg_df['geology_age'] = pd.Series(dtype='object')
    neg_df['geology_stratigraphy'] = pd.Series(dtype='object')
    neg_df['geology_stratigraphy_new'] = pd.Series(dtype='object')
    neg_df['geology_group'] = pd.Series(dtype='object')
    neg_df['geology_supergroup'] = pd.Series(dtype='object')
    
    geo_matched = 0
    for idx, row in neg_df.iterrows():
        pt = Point(row['longitude'], row['latitude'])
        candidates = geo_tree.query(pt)
        for c_idx in candidates:
            if geo_polygons[c_idx].contains(pt):
                props = geo_props[c_idx]
                neg_df.at[idx, 'geology_age'] = props.get('age')
                neg_df.at[idx, 'geology_stratigraphy'] = props.get('stratigrap')
                neg_df.at[idx, 'geology_stratigraphy_new'] = props.get('stratigrap_new')
                neg_df.at[idx, 'geology_group'] = props.get('group_')
                neg_df.at[idx, 'geology_supergroup'] = props.get('supergroup')
                geo_matched += 1
                break
                
    logging.info(f"Geology matched {geo_matched}/{len(neg_df)}")
    
    # 4. Extract LULC for negatives (Single pass)
    logging.info("Extracting LULC for negatives... (Will take approx 40 minutes)")
    neg_points = [Point(lon, lat) for lon, lat in zip(neg_df['longitude'], neg_df['latitude'])]
    neg_tree = STRtree(neg_points)
    
    lulc_matches = {i: np.nan for i in range(len(neg_points))}
    lulc_matched_count = 0
    lines_read = 0
    
    with open(LULC_FILE, 'r') as f:
        for line in f:
            lines_read += 1
            if lines_read % 1000000 == 0:
                logging.info(f"LULC: Processed {lines_read} polygons... Matched {lulc_matched_count}/{len(neg_df)}")
                if lulc_matched_count == len(neg_df):
                    break
            try:
                obj = json.loads(line.strip())
                geom = shape(obj['geometry'])
                candidates = neg_tree.query(geom)
                if len(candidates) > 0:
                    for c_idx in candidates:
                        if pd.isna(lulc_matches[c_idx]):
                            if geom.contains(neg_points[c_idx]):
                                lulc_matches[c_idx] = obj.get('properties', {}).get('REG_DESCRI', 'Unknown')
                                lulc_matched_count += 1
            except:
                pass
                
    neg_df['lulc_class'] = [lulc_matches[i] for i in range(len(neg_points))]
    logging.info(f"LULC matched {lulc_matched_count}/{len(neg_df)}")
    
    # Combine
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    combined_df.to_csv(OUT_CSV, index=False)
    
    # Validations
    p_df = combined_df[combined_df['label'] == 1]
    n_df = combined_df[combined_df['label'] == 0]
    
    # Missingness comparison
    def get_missing_pct(df_sub, col):
        return (df_sub[col].isna().sum() / len(df_sub)) * 100
        
    p_geo_miss = get_missing_pct(p_df, 'geology_age')
    n_geo_miss = get_missing_pct(n_df, 'geology_age')
    
    p_lulc_miss = get_missing_pct(p_df, 'lulc_class')
    n_lulc_miss = get_missing_pct(n_df, 'lulc_class')
    
    missingness_resolved = "YES" if (abs(p_geo_miss - n_geo_miss) < 5 and abs(p_lulc_miss - n_lulc_miss) < 5) else "NO"
    
    # Environmental Balance
    def calc_cohens_d(s1, s2):
        s1 = s1.dropna()
        s2 = s2.dropna()
        pool_std = np.sqrt((s1.var() + s2.var()) / 2)
        return abs(s1.mean() - s2.mean()) / pool_std if pool_std > 0 else 0
        
    d_slope = calc_cohens_d(p_df['slope'], n_df['slope'])
    d_elev = calc_cohens_d(p_df['elevation'], n_df['elevation'])
    d_tri = calc_cohens_d(p_df['tri'], n_df['tri'])
    
    env_pass = "PASS" if (d_slope < 1.0 and d_elev < 1.0) else "PASS WITH LIMITATIONS"
    
    # Distance / overlap
    pos_coords = p_df[['latitude', 'longitude']].values
    neg_coords = n_df[['latitude', 'longitude']].values
    p_tree = cKDTree(pos_coords)
    dists, _ = p_tree.query(neg_coords, k=1)
    
    overlaps = (dists == 0).sum()
    dups = combined_df.duplicated(subset=['latitude', 'longitude']).sum()
    
    spat_pass = "PASS" if overlaps == 0 and dups == 0 else "FAIL"
    final_pass = "PASS" if env_pass.startswith("PASS") and spat_pass == "PASS" and missingness_resolved == "YES" else "PASS WITH LIMITATIONS"
    if overlaps > 0 or dups > 0:
        final_pass = "FAIL"
        
    # Write report
    with open(REP_TXT, 'w', encoding='utf-8') as f:
        f.write("PHASE 5 — REBUILD FINAL ML DATASET\n")
        f.write("==================================\n\n")
        
        f.write(f"Total rows: {len(combined_df)}\n")
        f.write(f"Positives: {len(p_df)}\n")
        f.write(f"Negatives: {len(n_df)}\n")
        f.write(f"Total features: {len(combined_df.columns)}\n\n")
        
        f.write("EXTRACTION RESULTS (NEGATIVES ONLY):\n")
        f.write(f"DEM matched: {dem_matched}\n")
        f.write(f"Geology matched: {geo_matched} / Unmatched: {len(n_df) - geo_matched}\n")
        f.write(f"LULC matched: {lulc_matched_count} / Unmatched: {len(n_df) - lulc_matched_count}\n\n")
        
        f.write("MISSINGNESS COMPARISON (POS vs NEG):\n")
        f.write(f"Positive Geology Missing: {p_geo_miss:.2f}%\n")
        f.write(f"Negative Geology Missing: {n_geo_miss:.2f}%\n")
        f.write(f"Positive LULC Missing: {p_lulc_miss:.2f}%\n")
        f.write(f"Negative LULC Missing: {n_lulc_miss:.2f}%\n")
        f.write(f"Previous artifact resolved: {missingness_resolved}\n\n")
        
        f.write("ENVIRONMENTAL BALANCE:\n")
        f.write(f"Slope Cohen's d: {d_slope:.3f}\n")
        f.write(f"Elevation Cohen's d: {d_elev:.3f}\n")
        f.write(f"TRI Cohen's d: {d_tri:.3f}\n\n")
        
        f.write("SPATIAL VALIDATION:\n")
        f.write(f"Positive/Negative Overlaps: {overlaps}\n")
        f.write(f"Duplicate Coordinates: {dups}\n\n")
        
        f.write(f"FINAL DECISION:\n{final_pass}\n\n")
        f.write("STOP.\n")
        
    print("Done building dataset and extracting features.")

if __name__ == "__main__":
    main()
