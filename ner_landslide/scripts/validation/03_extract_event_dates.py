import os
import re
import logging
import pandas as pd
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_dates_from_pdf(pdf_path, df):
    logging.info(f"Opening PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF: {e}")
        return df

    # Regex to match dates like "17 May 2016" or "17/05/2016"
    date_pattern = re.compile(r'\b(\d{1,2}[\s/-]+[a-zA-Z]{3,9}[\s/-]+\d{4}|\d{1,2}[\s/-]+\d{1,2}[\s/-]+\d{4})\b')
    
    # We will just extract all text and build a mapping of rough coordinate strings to dates.
    # Since we don't know the exact format, we look for lat/lon numbers in the same block as a date.
    
    coord_to_date = {}
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4] # Text content of the block
            dates = date_pattern.findall(text)
            if dates:
                # Find any float numbers that look like lat/lon
                # Lat: 21-30, Lon: 88-98
                floats = re.findall(r'\b(\d{2}\.\d+)\b', text)
                for f in floats:
                    # Map the float to the first date found in this block
                    coord_to_date[f] = dates[0]
                    
    doc.close()
    
    logging.info(f"Extracted {len(coord_to_date)} potential coordinate-date mappings from PDF.")
    
    # Assign event dates based on string matching of latitude/longitude
    event_dates = []
    matched = 0
    
    # Identify lat/lon columns
    lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
    
    for idx, row in df.iterrows():
        lat_str = str(round(row[lat_col], 4))
        lon_str = str(round(row[lon_col], 4))
        
        # Try to find a match
        date_found = None
        for k, v in coord_to_date.items():
            if k in lat_str or k in lon_str or lat_str in k or lon_str in k:
                date_found = v
                break
                
        if date_found:
            event_dates.append(date_found)
            matched += 1
        else:
            event_dates.append('missing')
            
    df['event_date'] = event_dates
    logging.info(f"Successfully matched dates for {matched} out of {len(df)} records.")
    return df

if __name__ == "__main__":
    PROJECT_ROOT = "D:/sih/ner_landslide"
    INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides.csv")
    PDF_PATH = "C:/Users/kumar/Downloads/files/landslide_report.pdf"
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landslides", "validated_landslides_with_dates.csv")
    
    logging.info("Starting date extraction...")
    df = pd.read_csv(INPUT_CSV)
    df_with_dates = extract_dates_from_pdf(PDF_PATH, df)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_with_dates.to_csv(OUTPUT_CSV, index=False)
    
    # Summary
    total = len(df_with_dates)
    matched = len(df_with_dates[df_with_dates['event_date'] != 'missing'])
    missing = total - matched
    
    logging.info("--- Extraction Summary ---")
    logging.info(f"Total landslides: {total}")
    logging.info(f"Landslides with reliable dates: {matched}")
    logging.info(f"Landslides without reliable dates: {missing}")
    logging.info(f"Percentage with dates: {(matched/total)*100:.2f}%")
