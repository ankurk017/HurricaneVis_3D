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


import tkinter as tk
from tkinter import ttk
import tropycal.tracks as tracks


def coast_init():
    target_projection = ccrs.PlateCarree()
    feature = cartopy.feature.NaturalEarthFeature(
        'physical', 'coastline', '110m')
    geoms = feature.geometries()
    geoms = [target_projection.project_geometry(geom, feature.crs)
             for geom in geoms]
    paths = list(itertools.chain.from_iterable(geos_to_path(geom)
                 for geom in geoms))
    segments = []
    for path in paths:
        vertices = [vertex for vertex, _ in path.iter_segments()]
        vertices = np.asarray(vertices)
        segments.append(vertices)
    lc = LineCollection(segments, color='black', linewidth=0.5)
    return lc

def get_storm_names(year):
    storm_names = basin.get_season(year).to_dataframe()["name"].values
    return storm_names

def create_gui_and_get_hurdat():
    # Function to update the selected storm name based on user's selection
    def update_storm_name(event):
        selected_year = int(year_var.get())
        storm_names = get_storm_names(selected_year)
        storm_name_combo["values"] = storm_names
        storm_name_combo.current(0)  # Set the default value

    # Function to retrieve and display the selected storm's data
    def show_storm_data():
        global stored_hurdat  # Access the global variable
        
        selected_storm_name = storm_name_var.get()
        selected_year = int(year_var.get())
        stored_hurdat = basin.get_storm((selected_storm_name[1:-1], selected_year))
        
        # Close the GUI after displaying the storm data
        root.destroy()

    # Create the main GUI window
    root = tk.Tk()
    root.title("Tropical Storm Selector")

    # Year selection
    year_label = ttk.Label(root, text="Select Year:")
    year_label.pack()
    year_var = tk.StringVar()
    year_combo = ttk.Combobox(root, textvariable=year_var)
    year_combo["values"] = list(range(1851, 2023))  # Adjust the range as needed
    year_combo.bind("<<ComboboxSelected>>", update_storm_name)
    year_combo.pack()

    # Storm name selection
    storm_name_label = ttk.Label(root, text="Select Storm Name:")
    storm_name_label.pack()
    storm_name_var = tk.StringVar()
    storm_name_combo = ttk.Combobox(root, textvariable=storm_name_var)
    storm_name_combo.pack()

    # Show storm data button
    show_data_button = ttk.Button(root, text="Click here to select", command=show_storm_data)
    show_data_button.pack()

    # Start the GUI event loop
    root.mainloop()

    # The stored_hurdat variable will contain the hurdat data after the GUI is closed
    return stored_hurdat



password = pd.read_csv('/rhome/akumar/password.txt').columns[0]
cmap = plt.cm.jet
bounds = np.array([0, 32, 64, 83, 96, 113, 137, 150])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"

basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)

hurdat_selected = create_gui_and_get_hurdat()

fsdhfvdhb


hurdat_xarray = hurdat_selected.to_xarray()

hurdat = hurdat_xarray.isel(time=np.where(np.isin(np.array([int(dates.strftime('%H')) for dates in hurdat_selected.date]), [0, 6, 12, 18]))[0])




pcp_stacked, lons_stacked, lats_stacked, pcp_stacked1, lons_stacked1, lats_stacked1, time_values = [
], [], [], [], [], [], []

infile = "https://disc2.gesdisc.eosdis.nasa.gov/opendap/TRMM_RT/TRMM_3B42RT.7/2018/283/3B42RT.2018101018.7.nc4"

session = setup_session('ankurk017', password, check_url=infile)

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
        dataset.attrs['NominalTime'], '%Y-%m-%dT%H:%M:%SZ'))
convert = lambda input: np.array([datetime.datetime.utcfromtimestamp(pd.Timestamp(time_str.values).timestamp()) for time_str in input])


def update_frame(time_index1):
    ax.clear()
#    axins.clear()
#    ax2.clear()
    ax.add_collection3d(coast_init())
    precip_hurr = pcp_stacked[time_index1].values

    ax.plot_surface(lons_stacked[time_index1], lats_stacked[time_index1],
                    precip_hurr,  cmap='gist_ncar', lw=0.5, rstride=1, cstride=1)
    ax.plot(hurdat["lon"], hurdat["lat"], "k")
    ax.plot(hurdat["lon"], hurdat["lat"], "k")
    scatter = ax.scatter(hurdat["lon"], hurdat["lat"],
                         c=hurdat["vmax"], cmap=cmap, norm=norm)
    precip_data = pcp_stacked1[time_index1].values
    precip_data[precip_data < 0.5] = np.nan
    cset = ax.contourf(lons_stacked1[time_index1][0, :], lats_stacked1[time_index1][:, 1], precip_data,
                       zdir='z', cmap='jet', offset=-1)

    ax.set_xlabel('Longitudes')
    ax.set_ylabel('Latitudes')
    ax.set_zlabel('Precipitation (mm/hr)')
    ax.set_xlim((-100, hurdat['lon'].max()+10))
    ax.set_zlim((0, 20))
    ax.set_title(time_values[time_index1])
    print(time_values[time_index1])
    current_timeid = np.where(convert(hurdat['time']) == time_values[time_index1])[0][0]
    axins.plot(hurdat['time'][current_timeid], hurdat['vmax']
               [current_timeid], marker='d', color='tab:blue')
    ax2.plot(hurdat['time'][current_timeid], hurdat['mslp']
             [current_timeid], marker='d', color='tab:red')
    #ax.set_title(str(hurdat['date'][current_timeid]))
#    plt.title(str(hurdat['time'][current_timeid]))



fig = plt.figure(figsize=(15, 10))
ax = Axes3D(fig, xlim=[-120, -60], ylim=[-10, 50],
            zlim=[0, pcp.max().values/2])

axins = inset_axes(ax, width="35%", height="25%", loc='upper left')
# axins.plot(hurdat['date'], hurdat['mslp'], color=color, marker='d')

color = 'tab:blue'
axins.plot(hurdat['time'], hurdat['vmax'], color=color)
axins.set_ylabel('Maximum Sustained Wind Speed (kt)', color=color)
axins.set_xlabel('Date (YYYY-MM-DD)', color='k')
axins.set_yticks(np.arange(0, 150, 30))
axins.tick_params(axis='y', labelcolor=color)

ax2 = axins.twinx()
color = 'tab:red'
ax2.plot(hurdat['time'], hurdat['mslp'], color=color)
ax2.set_ylabel('Minimum Sea Pressure Level (hPa)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
axins.tick_params(axis='x', which='major', rotation=30, color=color)
axins.grid()

datetime_array = hurdat['time'][::15]
axins.set_xticks(datetime_array)
#formatted_dates = [dt.strftime('%Y-%m-%d') for dt in datetime_array]
#axins.set_xticklabels(formatted_dates, rotation=45)
#axins.set_xlabel('Time')
#axins.set_ylabel('Value')
#axins.set_title('Timeseries')

animation = FuncAnimation(
    fig, update_frame, frames=len(pcp_stacked), interval=500)

plt.show()



