
import requests
import json
import os

URL = "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"
SAVE_PATH = "data/stereoset_dev.json"

def download_stereoset():
    print(f"Downloading StereoSet from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        # The official format is:
        # { "version": "...", "data": { "intersentence": [...], "intrasentence": [...] } }
        # My current reader expects a list of examples.
        # I should verify this and maybe flatten it or update the reader.
        
        print("Keys in downloaded data:", data.keys())
        if 'data' in data:
            print("Keys in 'data':", data['data'].keys())
            inter = len(data['data'].get('intersentence', []))
            intra = len(data['data'].get('intrasentence', []))
            print(f"Counts: intersentence={inter}, intrasentence={intra}")
            
            # Save the raw file first
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {SAVE_PATH}")
            
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_stereoset()
