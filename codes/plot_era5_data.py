import xarray as xr
import matplotlib.pyplot as plt
import tropycal.tracks as tracks
import cartopy.crs as ccrs
import numpy as np
from coast import plot_coast

plt.rcParams.update({"font.size": 17, "font.weight": "bold"})


def plot_era(
    dataset,
    time_id=1,
    level=1000,
    var_name="d", plot_winds=False,
    output_dir="/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN/",
):

    if "level" in list(dataset.coords):
        input_data_type = "pressure"
    else:
        input_data_type = "surface"

    A_cropped = dataset.isel(time=time_id)

    era_time = str(A_cropped.time.values)[:13]

    hurdat_time_id = np.where(hurdat_time == era_time)[0][0]

    hurdat_lon = hurdat.isel(time=hurdat_time_id).lon.values
    hurdat_lat = hurdat.isel(time=hurdat_time_id).lat.values

    A_cropped_hurr = A_cropped.sel(
        longitude=slice(hurdat_lon - box, hurdat_lon + box),
        latitude=slice(hurdat_lat + box, hurdat_lat - box),
    )

    if input_data_type == "pressure":
        print("Pressure data")
        A_cropped_hurr = A_cropped_hurr.sel(level=level, method="nearest")

    if "Wind" in var_name:
        if input_data_type == "pressure":
            title_name = f"Wind Speed at {str(A_cropped_hurr.level.values)} hPa | {str(A_cropped_hurr.time.values)[:13]}"
        else:
            title_name = f"Wind Speed at 10 m | {str(A_cropped_hurr.time.values)[:13]}"
        fig_name = f"{hurr_name}_ws_{level}_{era_time}_wind{int(plot_winds)}.jpeg"
    else:
        if input_data_type == "pressure":
            title_name = f"{A_cropped_hurr[var_name].attrs['long_name']} at {str(A_cropped_hurr.level.values)} hPa | {str(A_cropped_hurr.time.values)[:13]}"
        else:
            title_name = f"{A_cropped_hurr[var_name].attrs['long_name']} | {str(A_cropped_hurr.time.values)[:13]}"
        fig_name = f"{hurr_name}_{var_name}_{level}_{era_time}_wind{int(plot_winds)}.jpeg"

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    A_cropped_hurr[var_name].plot(ax=ax, cbar_kwargs={"shrink": 0.8})

    if plot_winds:
        if input_data_type == "pressure":
            ax.quiver(
                A_cropped_hurr["longitude"][::3],
                A_cropped_hurr["latitude"][::3],
                A_cropped_hurr["u"][::3, ::3],
                A_cropped_hurr["v"][::3, ::3],
            )
        else:
            ax.quiver(
                A_cropped_hurr["longitude"][::3],
                A_cropped_hurr["latitude"][::3],
                A_cropped_hurr["u10"][::3, ::3],
                A_cropped_hurr["v10"][::3, ::3],
            )

    plot_coast(ax)

    ax.set_title(title_name, weight="bold")

    print(output_dir + fig_name)
    plt.savefig(output_dir + fig_name, dpi=400)
    plt.close()
    return None


sfc = "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_sfc.nc"
prs = "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_prs.nc"

output_dir = (
    "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN/"
)
# url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
url = "/rhome/akumar/hurdat2-1980-2022-042723.txt"

box = 5


hurr_name = "IAN"
hurr_year = 2022
basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)
hurdat = basin.get_storm((hurr_name, hurr_year)).to_xarray()
hurdat_time = np.array([str(val)[:13] for val in hurdat.time.values])

era5_prs = xr.open_dataset(prs)
era5_prs["Wind Speed (m/s)"] = np.sqrt(era5_prs["u"] ** 2 + era5_prs["v"] ** 2)

for time_id in range(era5_prs["time"].shape[0]):
    for level in list(era5_prs.level.values):
        for var_name in list(era5_prs.variables)[4:]:
            print(time_id, level, var_name)
            plot_era(dataset=era5_prs, time_id=time_id, level=level, var_name=var_name, plot_winds = False)
            plot_era(dataset=era5_prs, time_id=time_id, level=level, var_name=var_name, plot_winds = True)


# for surface variables

era5_sfc = xr.open_dataset(sfc)
era5_sfc["Wind Speed (m/s)"] = np.sqrt(era5_sfc["u10"] ** 2 + era5_sfc["v10"] ** 2)

for time_id in range(era5_sfc["time"].shape[0]):
    #for var_name in list(era5_sfc.variables)[4:]:
    for var_name in list(era5_sfc.variables)[3:4]:
        print(time_id, level, var_name)
        plot_era(dataset=era5_sfc, time_id=time_id, level=level, var_name=var_name, plot_winds = False)
        plot_era(dataset=era5_sfc, time_id=time_id, level=level, var_name=var_name, plot_winds = True)
