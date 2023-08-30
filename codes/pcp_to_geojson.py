import progressbar
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from pydap.client import open_url
from pydap.cas.urs import setup_session
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs
import tropycal.tracks as tracks
import tropycal
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import tkinter as tk
from tkinter import ttk
import tropycal.tracks as tracks


password = pd.read_csv('/rhome/akumar/password.txt').columns[0]
cmap = plt.cm.jet
bounds = np.array([0, 32, 64, 83, 96, 113, 137, 150])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"

basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)

hurdat_selected = basin.get_storm(('HARVEY', 2017))
hurdat_xarray = hurdat_selected.to_xarray()

hurdat = hurdat_xarray.isel(time=np.where(np.isin(np.array([int(dates.strftime('%H')) for dates in hurdat_selected.date]), [0, 6, 12, 18]))[0])


pcp_stacked, lons_stacked, lats_stacked, pcp_stacked1, lons_stacked1, lats_stacked1, time_values = [
], [], [], [], [], [], []

infile = "https://disc2.gesdisc.eosdis.nasa.gov/opendap/TRMM_RT/TRMM_3B42RT.7/2018/283/3B42RT.2018101018.7.nc4"

session = setup_session('ankurk017', password, check_url=infile)
pcp_all = []
for time_index in progressbar.progressbar(np.arange(1, len(hurdat.time), 3)):
    #hurdat_time = hurdat.to_xarray().isel(time=time_index)
    hurdat_time = hurdat.isel(time=time_index)

    time_value = hurdat_time.time.values
    time_datetime = pd.to_datetime(time_value)

    julian_date = time_datetime.timetuple().tm_yday - \
        1 if time_datetime.timetuple().tm_hour == 0 else time_datetime.timetuple().tm_yday

    year = time_datetime.year
    month = time_datetime.month
    day = time_datetime.day

    formatted_time_value = time_datetime.strftime('%Y%m%d%H')

    infile = f"https://disc2.gesdisc.eosdis.nasa.gov/opendap/TRMM_RT/TRMM_3B42RT.7/{year}/{julian_date:03d}/3B42RT.{formatted_time_value}.7.nc4"

    gpm = open_url(infile, session=session)

    dataset = xr.open_dataset(xr.backends.PydapDataStore(gpm))

    ROI = [hurdat_time['lon'].values-5, hurdat_time['lon'].values +
           5, hurdat_time['lat'].values-5, hurdat_time['lat'].values+5]

    lon = dataset['lon']
    lat = dataset['lat']
    lon_id = np.where(np.logical_and(lon > ROI[0], lon < ROI[1]))[0]
    lat_id = np.where(np.logical_and(lat > ROI[2], lat < ROI[3]))[0]

    pcp_domain = dataset['precipitation']
    pcp = pcp_domain.isel(lon=lon_id, lat=lat_id)
    lons, lats = np.meshgrid(pcp.lon, pcp.lat)
    pcp_stacked.append(pcp)
    lons_stacked.append(lons)
    lats_stacked.append(lats)

    ROI = [-160, -60, -50, 50]

    lon_id = np.where(np.logical_and(lon > ROI[0], lon < ROI[1]))[0]
    lat_id = np.where(np.logical_and(lat > ROI[2], lat < ROI[3]))[0]

    pcp = pcp_domain.isel(lon=lon_id, lat=lat_id)
    lons, lats = np.meshgrid(pcp.lon, pcp.lat)
    pcp_stacked1.append(pcp)
    lons_stacked1.append(lons)
    lats_stacked1.append(lats)
    time_values.append(datetime.strptime(
        dataset.attrs['NominalTime'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%dT%H'))


    arr = np.array((lons_stacked1[-1].ravel(), lats_stacked1[-1].ravel(), pcp_stacked1[-1].values.ravel())).T
    new_column = np.full((arr.shape[0], 1), time_values[-1], dtype='object')
    pcp_all.append(np.hstack((arr, new_column)))


df  = pd.DataFrame(np.vstack((pcp_all)), columns=['lons', 'lat', 'precipitation', 'date'])
df.to_csv('Harvey_precipitation.csv', index=False)
geometry = [Point(xy) for xy in zip(df['lons'], df['lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

output_filename = 'Harvey_precipitation.geojson'
geo_df.to_file(output_filename, driver='GeoJSON')





















