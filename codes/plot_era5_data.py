import xarray as xr
import matplotlib.pyplot as plt
import tropycal.tracks as tracks
import cartopy.crs as ccrs
import numpy as np
from coast import plot_coast

plt.rcParams.update({"font.size": 17, "font.weight": "bold"})


def plot_era(time_id=1, level=1000, var_name='d', output_dir='/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN/'):
    # Select the time slice from the dataset A
    A_cropped = A.isel(time=time_id)

    # Extract the ERA time as a string
    era_time = str(A_cropped.time.values)[:13]

    # Find the corresponding time index in hurdat_time
#    hurdat_time_id = np.where(hurdat_time == era_time)[0][0]

    # Get latitude and longitude from hurdat dataset
#    hurdat_lon = hurdat.isel(time=hurdat_time_id).lon.values
#    hurdat_lat = hurdat.isel(time=hurdat_time_id).lat.values

    # Crop A_cropped to the specified latitude and longitude box and select the desired level
#    A_cropped_hurr = A_cropped.sel(longitude=slice(hurdat_lon - box, hurdat_lon + box), 
#                                   latitude=slice(hurdat_lat + box, hurdat_lat - box)).sel(level=level, method='nearest')
    A_cropped_hurr = A_cropped.sel(level=level, method='nearest')
    # Create title and figure name based on variable name
    print(var_name)
    if 'Wind' in var_name:
        print('WInd')
        title_name = f"Wind Speed at {str(A_cropped_hurr.level.values)} hPa | {str(A_cropped_hurr.time.values)[:13]}"
        fig_name = f'{hurr_name}_ws_{level}_{era_time}.jpeg'
    else:
        title_name = f"{A_cropped_hurr[var_name].attrs['long_name']} at {str(A_cropped_hurr.level.values)} hPa | {str(A_cropped_hurr.time.values)[:13]}"
        fig_name = f'{hurr_name}_{var_name}_{level}_{era_time}.jpeg'

    # Create a figure and plot the variable on a map
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    A_cropped_hurr[var_name].plot(ax=ax, cbar_kwargs={'shrink': 0.8})
    
    # Add coastlines to the plot
    plot_coast(ax)
    
    # Set the plot title
    ax.set_title(title_name, weight='bold')
    
    # Save the figure to the specified output directory
    print(output_dir + fig_name)
    plt.savefig(output_dir + fig_name, dpi=400)
    plt.close()    
    return None



sfc = '/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_sfc.nc'
prs = '/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_prs.nc'

output_dir = '/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN/'
url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
box = 5


hurr_name = 'IAN'
hurr_year = 2022
#basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)
#hurdat =  basin.get_storm((hurr_name, hurr_year)).to_xarray()
#hurdat_time = np.array([str(val)[:13] for val in hurdat.time.values])



A= xr.open_dataset(prs)
A['Wind Speed (m/s)'] = np.sqrt(A['u']**2 + A['v']**2)

for time_id in range(A['time'].shape[0]):
        for level in (100,  200,  300,  400,  500,  600,  650,  700,  750,  800,  850,  900, 950, 1000):
            for var_name in list(A.variables)[4:]:
                print(time_id, level, var_name)
                plot_era(time_id = time_id, level = level, var_name = var_name)
















