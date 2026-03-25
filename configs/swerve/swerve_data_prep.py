import os
import requests
import io

import bs4
import pandas as pd

base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(base_dir, "..", "..", "data", "raw", "SWERVE")

# set event
event = "2024-05-10"

# GitHub API URL for directory contents
api_url = f"https://api.github.com/repos/lucywilkerson/SWERVE-{event}/contents/data_processed/sites"
# base URL for raw file access
base_file_url = f"https://raw.githubusercontent.com/lucywilkerson/SWERVE-{event}/main/data_processed/sites"
base_info_url = f"https://raw.githubusercontent.com/lucywilkerson/SWERVE/main/info/{event}/nearest_b_sites.csv"
    
# Create empty log file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f:
    pass

def xprint(msg):
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")

def download_file(url, dest_folder):
    file_name = os.path.join(dest_folder, url.split("/")[-1])
    if os.path.exists(file_name):
        print(f"  File already exists; not re-downloading: {os.path.abspath(file_name)}")
        return
    os.makedirs(dest_folder, exist_ok=True)
    print(f"  Getting {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"  Saved: {os.path.abspath(file_name)}")

# Get list of folders from GitHub API
response = requests.get(api_url)
response.raise_for_status()
data = response.json()

# Get valid site ids from info file
response = requests.get(base_info_url)
response.raise_for_status()
info_df = pd.read_csv(io.StringIO(response.text))
valid_sites = set(info_df[['gic_sid', 'nearest_b_sid']].values.ravel())

# Extract folder names and download files
for item in data:
    if item["type"] == "dir":
        sid = item["name"]
        if sid not in {str(site).lower().replace(' ', '') for site in valid_sites}:
            continue
        if sid and not sid.startswith("."):
            files_dir = os.path.join(base_dir, sid.lower().replace(' ', ''))
            if not os.path.exists(files_dir):
                os.makedirs(files_dir, exist_ok=True)
            # Construct the raw download URL for the file
            file_url = base_file_url + f"/{sid.lower().replace(' ', '')}/data/_all.pkl"
            download_file(file_url, files_dir)

# Create dataframes for each site
for gic_sid in set(info_df['gic_sid']):
    b_sid = info_df.loc[info_df['gic_sid'] == gic_sid, 'nearest_b_sid'].values[0]
    gic_file = os.path.join(base_dir, gic_sid.lower().replace(' ', ''), "_all.pkl")
    b_file = os.path.join(base_dir, b_sid.lower().replace(' ', ''), "_all.pkl")
    if os.path.exists(gic_file) and os.path.exists(b_file):
        gic_data = pd.read_pickle(gic_file)['GIC']['measured']
        b_data = pd.read_pickle(b_file)['B']['measured']
        # Extract modified data
        gic = next(iter(gic_data.values()))['modified']
        b = next(iter(b_data.values()))['modified']
        # Align timestamps 
        common_times = pd.Index(gic['time']).intersection(pd.Index(b['time']))
        # Create combined dataframe
        site_df = pd.DataFrame({
            'datetime': common_times,
            'gic': pd.Series(gic['data'].flatten(), index=gic['time']).reindex(common_times).values,
            'bx': pd.Series(b['data'][:, 0], index=b['time']).reindex(common_times).values,
            'by': pd.Series(b['data'][:, 1], index=b['time']).reindex(common_times).values,
            'bz': pd.Series(b['data'][:, 2], index=b['time']).reindex(common_times).values
        })
        # Save df to CSV
        file_name = f"{gic_sid.lower().replace(' ', '')}_gic_b.csv"
        site_df.to_csv(base_dir + f"/{gic_sid.lower().replace(' ', '')}/{file_name}", index=False)
        print(f"Saved combined data for {gic_sid} and {b_sid} to {os.path.join(base_dir, gic_sid.lower().replace(' ', ''), file_name)}.")
    else:
        raise FileNotFoundError(f"Missing file for site {gic_sid} or {b_sid}")