import os
import requests
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# ==========================================
#               CONFIGURATION
# ==========================================
CONFIG = {
    # --------------------------------------------------------
    # TIME WINDOW SETUP
    # --------------------------------------------------------
    # Format: 'YYYY-MM-DD'
    'START_DATE': '2006-1-1',  
    'END_DATE':   '2026-1-1',  
    
    # --------------------------------------------------------
    # DATA SELECTION
    # --------------------------------------------------------
    # 'Lsurf': Surface data (Rain, Temp, Wind 10m, Pressure)
    # 'L-pall': All pressure levels (Warning: Very large files)
    'DATA_TYPE': 'Lsurf',
    
    # [NEW] Filter for Forecast Hours
    # 'FH00-15': Downloads only short-term (Best for stitching/forcing)
    # 'FH16-33': Downloads only long-range
    # None     : Downloads ALL files
    'FORECAST_FILTER': 'FH00-15',
    
    # --------------------------------------------------------
    # OUTPUT SETTINGS
    # --------------------------------------------------------
    'OUTPUT_DIR': './msm_data',
    
    # --------------------------------------------------------
    # SYSTEM SETTINGS
    # --------------------------------------------------------
    'BASE_URL': 'http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/',
    'DELAY': 1.0  # Seconds between requests to be polite
}

# ==========================================
#           CORE FUNCTIONS
# ==========================================

def get_date_range(start_str, end_str):
    """Generates a list of datetime objects between start and end."""
    if start_str is None:
        print("WARNING: START_DATE is None. Defaulting to 2006-03-01.")
        time.sleep(5)
        start_date = datetime.date(2006, 3, 1)
    else:
        start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()

    if end_str is None:
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

    delta = end_date - start_date
    if delta.days < 0:
        raise ValueError("END_DATE is before START_DATE.")
        
    for i in range(delta.days + 1):
        yield start_date + datetime.timedelta(days=i)

def find_files_for_date(target_date, data_type, forecast_filter):
    """
    Scrapes the RISH directory for a specific date to find filenames.
    Applies filters for Data Type (Lsurf) and Forecast (FH00-15).
    """
    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    day = target_date.strftime("%d")
    
    dir_url = f"{CONFIG['BASE_URL']}{year}/{month}/{day}/"
    
    try:
        response = requests.get(dir_url, timeout=10)
        if response.status_code == 404:
            print(f"  [!] No data found for {target_date} (404)")
            return []
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Error accessing {dir_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    file_urls = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if not href:
            continue
            
        # ------------------------------------------------------
        #                  FILTERING LOGIC
        # ------------------------------------------------------
        # 1. Must be a binary file (.bin)
        # 2. Must be MSM model data
        # 3. Must match data type (e.g., Lsurf)
        if href.endswith('.bin') and 'MSM' in href and data_type in href:
            
            # 4. [NEW] Check Forecast Filter (FH00-15)
            if forecast_filter and forecast_filter not in href:
                continue
                
            full_url = urljoin(dir_url, href)
            file_urls.append(full_url)
            
    return sorted(list(set(file_urls)))

def download_file(url, save_dir):
    """Downloads a single file to save_dir."""
    filename = url.split('/')[-1]
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        print(f"    [Skip] {filename} already exists.")
        return

    print(f"    [Down] Downloading {filename}...", end='', flush=True)
    
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(" Done.")
    except Exception as e:
        print(f" Failed! ({e})")
        if os.path.exists(save_path):
            os.remove(save_path)

# ==========================================
#               MAIN LOOP
# ==========================================
def main():
    print(f"--- JMA-MSM Short-Term Downloader ---")
    print(f"Target: {CONFIG['DATA_TYPE']}")
    print(f"Filter: {CONFIG['FORECAST_FILTER']}")
    print(f"Output: {CONFIG['OUTPUT_DIR']}")
    
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    try:
        dates = list(get_date_range(CONFIG['START_DATE'], CONFIG['END_DATE']))
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print("-" * 30)

    for current_date in dates:
        print(f"Processing {current_date}...")
        
        # Create Yearly Folder (e.g., ./msm_data/2011)
        year_folder = current_date.strftime("%Y")
        save_dir = os.path.join(CONFIG['OUTPUT_DIR'], year_folder)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"  [Info] Created new yearly folder: {year_folder}")

        # Find files with the specific filters
        urls = find_files_for_date(current_date, CONFIG['DATA_TYPE'], CONFIG['FORECAST_FILTER'])
        
        if not urls:
            print(f"  [x] No matching files found for {current_date}.")
            continue
            
        print(f"  Found {len(urls)} files (Short-term only).")
        
        # Download
        for url in urls:
            download_file(url, save_dir)
            time.sleep(CONFIG['DELAY'])

    print("-" * 30)
    print("Download Complete.")

if __name__ == "__main__":
    main()
