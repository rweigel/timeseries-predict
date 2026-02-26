import os
import requests

import bs4
import pandas

all = True # False => only d/l cluster1 files

base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join("..", "..", "data", "raw", "satellite-b")
files_dir = os.path.join(base_dir, "files")

# URL of the directory
url = "https://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/"
url += "empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/"


if not os.path.exists(files_dir):
    os.makedirs(files_dir, exist_ok=True)

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
        print(f"  File already exists; not re-downloading: {file_name}")
        return
    os.makedirs(dest_folder, exist_ok=True)
    print(f"  Getting {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"  Saved: {file_name}")


# Get list of files from directory listing
response = requests.get(url)
response.raise_for_status()
soup = bs4.BeautifulSoup(response.content, "html.parser")
links = soup.find_all("a")

# Extract   links for the files
files = [link.get("href") for link in links if link.get("href").endswith(".dat")]

# Download each file
for file in files:
    if not all and not file.startswith("cluster1"):
        # Only d/l cluster1 files
        continue
    file_url = url + file
    download_file(file_url, files_dir)
    if not all:
        print("File download complete, all=False")
        break

print("Downloading complete")

## Create pickle files

# List all files in download directory
files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f)) and f.endswith(".dat")]

xprint(f"Saving DataFrames for {len(files)} file(s)")

# Extract column names from first file
first_file = os.path.join(files_dir, files[0])
names = pandas.DataFrame(pandas.read_csv(first_file, nrows=0, delimiter=',')).columns

# Process each file in directory
for i, file in enumerate(files):
    file = os.path.join(files_dir, file)
    xprint(f"  Processing file {i+1}/{len(files)}: {file}")

    # Read the data, skip first row, whitespace as delimiter
    data = pandas.read_csv(file, skiprows=1, header=None, delimiter=r'\s+')

    # Convert to DataFrame, assign column names
    df = pandas.DataFrame(data)
    df.columns = names

    # Save DataFrame as pickle file
    pkl_filename = file.replace(".dat", ".pkl")
    df.to_pickle(pkl_filename)

    xprint(f"  Saved DataFrame to {pkl_filename}")

xprint("All files pickled and saved")