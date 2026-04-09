import io
import os
import requests
import functools

import pandas as pd

base = "https://raw.githubusercontent.com/lucywilkerson"

def siteid_lc(site):
  return str(site).lower().replace(' ', '')


def get_ids(event):
    info_df = get_info(event)
    gic_sids = info_df['gic_sid'].tolist()
    for idx, gic_sid in enumerate(gic_sids):
        gic_sids[idx] = siteid_lc(gic_sid)
    return gic_sids


@functools.lru_cache(maxsize=None)
def get_info(event):

    base_info_url = f"{base}/SWERVE/main/info/{event}/nearest_b_sites.csv"

    # Get info to find the nearest B site for the requested GIC site
    response = requests.get(base_info_url)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def swerve_data_download(event, site):

    info_df = get_info(event)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, "..", "..", "data", "raw", "SWERVE")

    # base URL for raw file access
    base_file_url = f"{base}/SWERVE-{event}/main/data_processed/sites"

    gic_sid = siteid_lc(site)

    match = info_df[info_df['gic_sid'].str.lower().str.replace(' ', '') == gic_sid]
    if match.empty:
        raise ValueError(f"Site '{site}' not found.")

    b_sid = match['nearest_b_sid'].values[0].lower().replace(' ', '')

    # Download the two needed files
    gic_url = f"{base_file_url}/{gic_sid}/data/_all.pkl"
    gic_dir = os.path.join(base_dir, gic_sid)
    try:
        gic_file = _download_file(gic_url, gic_dir)
    except requests.HTTPError as e:
        raise ValueError(f"Error downloading GIC file for site '{site}' from {gic_url}: {e}")

    b_url = f"{base_file_url}/{b_sid}/data/_all.pkl"
    b_dir = os.path.join(base_dir, b_sid)
    try:
        b_file = _download_file(b_url, b_dir)
    except requests.HTTPError as e:
        raise ValueError(f"Error downloading B file for site '{site}' from {b_url}: {e}")

    return gic_file, b_file


def _download_file(url, dest_dir):
    file_name = os.path.join(dest_dir, url.split("/")[-1])
    if os.path.exists(file_name):
        print(f"  File already exists; not re-downloading: {os.path.abspath(file_name)}")
        return os.path.abspath(file_name)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  Getting {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"  Saved: {os.path.abspath(file_name)}")
    return os.path.abspath(file_name)
