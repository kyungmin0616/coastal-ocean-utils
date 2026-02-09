import os
import requests
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import sys

# ==========================================
#               CONFIGURATION
# ==========================================
CONFIG = {
    'START_DATE': '2011-12-31',  
    'END_DATE':   '2012-01-01',  
    'OUTPUT_DIR': './jma_data',
    'BASE_URL': 'http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/',
    'DELAY': 1.0 
}

# ==========================================
#           CORE FUNCTIONS
# ==========================================
def get_date_range(start_str, end_str):
    start = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    delta = end - start
    for i in range(delta.days + 1):
        yield start + datetime.timedelta(days=i)

def find_files(target_date, model_type):
    """
    Finds files for a specific model (MSM or GSM).
    """
    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    day = target_date.strftime("%d")
    dir_url = f"{CONFIG['BASE_URL']}{year}/{month}/{day}/"
    
    try:
        r = requests.get(dir_url, timeout=10)
        if r.status_code == 404: return []
    except: return []

    soup = BeautifulSoup(r.text, 'html.parser')
    files = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if not href or not href.endswith('.bin'): continue
        
        # FILTER LOGIC
        if model_type == 'MSM':
            # Target: MSM Surface Short-term (Wind/Pres/Rain)
            if 'MSM' in href and 'Lsurf' in href and 'FH00-15' in href:
                files.append(urljoin(dir_url, href))
                
        elif model_type == 'GSM':
            # Target: GSM Surface Short-term (Radiation)
            # GSM files usually have 'FD0000-0312' or similar patterns
            if 'GSM' in href and 'Lsurf' in href and 'FD0000' in href:
                files.append(urljoin(dir_url, href))
            
    return sorted(list(set(files)))

def download(url, save_dir):
    filename = url.split('/')[-1]
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        print(f"    [Skip] {filename}")
        return
    print(f"    [Down] {filename}...", end='', flush=True)
    try:
        with requests.get(url, stream=True) as r:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(16384): f.write(chunk)
        print(" Done.")
    except:
        print(" Failed.")

# ==========================================
#               MAIN
# ==========================================
if __name__ == "__main__":
    dates = list(get_date_range(CONFIG['START_DATE'], CONFIG['END_DATE']))
    
    for d in dates:
        print(f"\nProcessing {d}...")
        year = d.strftime("%Y")
        
        # 1. Download MSM (Wind/Air)
        msm_dir = os.path.join(CONFIG['OUTPUT_DIR'], 'MSM', year)
        os.makedirs(msm_dir, exist_ok=True)
        for u in find_files(d, 'MSM'): download(u, msm_dir)
        
        # 2. Download GSM (Radiation)
        gsm_dir = os.path.join(CONFIG['OUTPUT_DIR'], 'GSM', year)
        os.makedirs(gsm_dir, exist_ok=True)
        for u in find_files(d, 'GSM'): download(u, gsm_dir)
