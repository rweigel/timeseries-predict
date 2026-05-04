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

def xzy2rtp(xyz):
    import numpy as np
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
    return np.column_stack((r, theta, phi))


## Create pickle files

# List all files in download directory
files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f)) and f.endswith(".dat")]

xprint(f"Saving DataFrames for {len(files)} file(s)")

column_info = {
    'epoch': {
        'id': 'epoch',
        'label': 'epoch',
        'unit': 'ms'
    },
    'doy': {
        'id': 'doy',
        'label': 'doy',
        'latex': r'\mathrm{DOY}'
    },
    'year': {
        'id': 'year',
        'label': 'year'
    },
    'month': {
        'id': 'month',
        'label': 'month'
    },
    'day': {
        'id': 'day',
        'label': 'day'
    },
    'hour': {
        'id': 'hour',
        'label': 'hour'
    },
    'minute': {
        'id': 'minute',
        'label': 'minute'
    },
    'second': {
        'id': 'second',
        'label': 'second'
    },
    'count': {
        'id': 'count',
        'label': 'count'
    },
    'x[km]': {
        'id': 'x',
        'label': 'x',
        'latex': r'$x$',
        'unit': 'km',
        'frame': 'GSM'
    },
    'y[km]': {
        'id': 'y',
        'label': 'y',
        'latex': r'$y$',
        'unit': 'km',
        'frame': 'GSM'
    },
    'z[km]': {
        'id': 'z',
        'label': 'z',
        'latex': r'$z$',
        'unit': 'km',
        'frame': 'GSM'
    },
    'bx[nT]': {
        'id': 'dbx',
        'label': 'dbx',
        'latex': r'$\Delta B_x$',
        'unit': 'nT'
    },
    'by[nT]': {
        'id': 'dby',
        'label': 'dby',
        'latex': r'$\Delta B_y$',
        'unit': 'nT'
    },
    'bz[nT]': {
        'id': 'dbz',
        'label': 'dbz',
        'latex': r'$\Delta B_z$',
        'unit': 'nT'
    },
    'sdbx[nT]': {
        'id': 'sdbx',
        'label': 'sdbx',
        'latex': r'$\sigma_{\Delta B_x}$',
        'unit': 'nT'
    },
    'sdby[nT]': {
        'id': 'sdby',
        'label': 'sdby',
        'latex': r'$\sigma_{\Delta B_y}$',
        'unit': 'nT'
    },
    'sdbz[nT]': {
        'id': 'sdbz',
        'label': 'sdbz',
        'latex': r'$\sigma_{\Delta B_z}$',
        'unit': 'nT'
    },
    'imfbx[nT]': {
        'id': 'imfbx',
        'label': 'imfbx',
        'latex': r'$B_x^{\mathrm{IMF}}$',
        'unit': 'nT',
        'frame': 'GSM'
    },
    'imfby[nT]': {
        'id': 'imfby',
        'label': 'imfby',
        'latex': r'$B_y^{\mathrm{IMF}}$',
        'unit': 'nT',
        'frame': 'GSM'
    },
    'imfbz[nT]': {
        'id': 'imfbz',
        'label': 'imfbz',
        'latex': r'$B_z^{\mathrm{IMF}}$',
        'unit': 'nT',
        'frame': 'GSM'
    },
    'nsw[1/cm^3]': {
        'id': 'nsw',
        'label': 'nsw',
        'latex': r'$n_{\mathrm{SW}}$',
        'unit': '$cm^{-3}$'
    },
    'vsw[km/s]': {
        'id': 'vsw',
        'label': 'vsw',
        'latex': r'$v_{\mathrm{SW}}$',
        'unit': 'km/s'
    },
    'ae[nT]': {
        'id': 'ae',
        'label': 'ae',
        'latex': r'$\mathrm{AE}$',
        'unit': 'nT'
    },
    'al[nT]': {
        'id': 'al',
        'label': 'al',
        'latex': r'$\mathrm{AL}$',
        'unit': 'nT'
    },
    'au[nT]': {
        'id': 'au',
        'label': 'au',
        'latex': r'$\mathrm{AU}$',
        'unit': 'nT'
    },
    'symh[nT]': {
        'id': 'symh',
        'label': 'symh',
        'latex': r'\mathrm{SYM}-\mathrm{H}',
        'unit': 'nT'
    },
    'symh*[nT]': {
        'id': 'symhstar',
        'label': 'symh*',
        'latex': r'\mathrm{SYM}-{\mathrm{H^{*}}',
        'unit': 'nT'
    },
    'ey[mV/m]': {
        'id': 'ey',
        'label': 'ey',
        'latex': r'$E_y^{\mathrm{IMF}}$',
        'unit': 'mV/m',
        'frame': 'GSM'
    },
    '<symh*>[nT]': {
        'id': 'symhstar_avg',
        'label': '<symh*>',
        'latex': r'$\langle \mathrm{SYM-H}^{*} \rangle$',
        'unit': 'nT'
    },
    '<dsymh*_dt>[nT/h]': {
        'id': 'dsymhstar_dt_avg',
        'label': '<dsymh*/dt>',
        'latex': r'$\langle\frac{d\mathrm{SYM-H}^{*}}{dt}\rangle$',
        'unit': 'nT/h'
    },
    '<ey>[mV/m]': {
        'id': 'ey_avg',
        'label': '<ey>',
        'latex': r'$\langle E_y^{\mathrm{IMF}}\rangle$',
        'unit': 'mV/m',
        'frame': 'GSM'
    },
    'datetime': {
        'id': 'datetime',
        'label': 'datetime'
    },
    'r[km]': {
        'id': 'r',
        'label': 'r',
        'latex': r'$r$',
        'unit': 'km'
    },
    'theta[deg]': {
        'id': 'theta',
        'label': 'theta',
        'latex': r'$\theta$',
        'unit': 'deg'
    },
    'phi[deg]': {
        'id': 'phi',
        'label': 'phi',
        'latex': r'$\phi$',
        'unit': 'deg'
    }
}

# Extract column names from first file
first_file = os.path.join(files_dir, files[0])
names = pandas.DataFrame(pandas.read_csv(first_file, nrows=0, delimiter=',')).columns

# Labels for Cartesian position columns
position_cart = ["x[km]", "y[km]", "z[km]"]

# Labels for derived columns used to convert from Cartesian to spherical
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]

# Process each file in directory
for i, file in enumerate(files):
    file = os.path.join(files_dir, file)
    xprint(f"  Processing file {i+1}/{len(files)}: {file}")

    # Read the data, skip first row, whitespace as delimiter
    data = pandas.read_csv(file, skiprows=1, header=None, delimiter=r'\s+')

    # Convert to DataFrame, assign column names
    df = pandas.DataFrame(data)
    df.columns = names

    cartesian = df[position_cart].to_numpy()
    spherical = xzy2rtp(cartesian)

    # Add spherical coordinates to the DataFrame
    xprint(f"    Adding spherical coordinates columns: {position_sph}")
    for i, col in enumerate(position_sph):
      df[col] = spherical[:, i]

    ymdhms = df[['year', 'month', 'day', 'hour', 'minute', 'second']]
    xprint(f"    Adding datetime column")
    df['datetime'] = pandas.to_datetime(ymdhms)

    xprint(f"    adding df.attrs['column_info']")
    df.attrs['column_info'] = column_info

    # Save DataFrame as pickle file
    pkl_filename = file.replace(".dat", ".pkl")
    df.to_pickle(pkl_filename)

    xprint(f"  Saved DataFrame to {pkl_filename}")

xprint("All files pickled and saved")