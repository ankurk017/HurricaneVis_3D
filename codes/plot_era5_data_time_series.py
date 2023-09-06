import xarray as xr
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import tropycal.tracks as tracks
import cartopy.crs as ccrs
import numpy as np
from coast import plot_coast
import re

plt.rcParams.update({"font.size": 17, "font.weight": "bold"})


def era_var_mean(
    dataset,
    time_id=1,
    level=1000,
    var_name="d",
):

    if "level" in list(dataset.coords):
        input_data_type = "pressure"
    else:
        input_data_type = "surface"

    A_cropped = dataset[var_name].isel(time=time_id)

    era_time = str(A_cropped.time.values)[:13]

    hurdat_time_id = np.where(hurdat_time == era_time)[0][0]

    hurdat_lon = hurdat.isel(time=hurdat_time_id).lon.values
    hurdat_lat = hurdat.isel(time=hurdat_time_id).lat.values

    A_cropped_hurr = A_cropped.sel(
        longitude=slice(hurdat_lon - box, hurdat_lon + box),
        latitude=slice(hurdat_lat + box, hurdat_lat - box),
    ).mean(dim=["longitude", "latitude"], keep_attrs=True)

    if input_data_type == "pressure":
        A_cropped_hurr = A_cropped_hurr.sel(level=level, method="nearest")
    return A_cropped_hurr


def get_title_name(
    A_cropped_hurr,
    time_id=1,
    level=1000,
    var_name="d",
    output_dir="/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN/",
):
    if "level" in list(A_cropped_hurr.coords):
        input_data_type = "pressure"
    else:
        input_data_type = "surface"

    if "Wind" in var_name:
        if input_data_type == "pressure":
            title_name = f"Wind Speed at {str(A_cropped_hurr.level.values)} hPa "
        else:
            title_name = f"Wind Speed at 10 m  "
        fig_name = f"TS_{hurr_name}_ws_{level}.html"
    else:
        if input_data_type == "pressure":
            title_name = f"{A_cropped_hurr.attrs['long_name']} at {str(A_cropped_hurr.level.values)} hPa "
        else:
            title_name = f"{A_cropped_hurr.attrs['long_name']} "
        fig_name = f"TS_{hurr_name}_{var_name}_{level}.html"

    return title_name, fig_name


def plot_xarray_time_series(
    output_var,
    fontsize=14,
    figure_size=(800, 400),
    output_file="output_plot.html",
    title="None",
):
    # Extract attributes
    units = output_var.attrs.get("units", "m/s")

    standard_name = output_var.attrs.get("standard_name", None)
    if standard_name is None:
        if "Wind" in era5_var.name:
            standard_name = era5_var.name
        else:
            standard_name = output_var.long_name

    long_name = re.sub("_", " ", standard_name)
    # Create a Plotly figure
    fig = px.line(x=output_var.time, y=output_var, labels={"x": "Time", "y": units})
    fig.update_layout(
        # title=f'Time Series of {long_name}',
        title=title,
        xaxis_title="Time",
        yaxis_title=f"{long_name} ({units})",
        font=dict(size=18, family="Arial, bold"),
        autosize=False,
        width=figure_size[0],  # Set custom figure width
        height=figure_size[1],  # Set custom figure height
    )
    # Save the figure as an HTML file
    pio.write_html(fig, output_file)
    plt.close()


sfc = "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_sfc.nc"
prs = "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/ERA5/dataset/ERA/ERA_IAN_v2_prs.nc"

output_dir = "/nas/rstor/akumar/USA/IMPACT/AGU2023_3d_Hurricane/HurricaneVis_3D/figures/IAN_time_series/"
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

for level in list(era5_prs.level.values):
    for var_name in list(era5_prs.variables)[4:]:
        var_ts = []
        for time_id in range(era5_prs["time"].shape[0]):
            print(time_id, level, var_name)
            var_ts.append(
                era_var_mean(
                    dataset=era5_prs,
                    time_id=time_id,
                    level=level,
                    var_name=var_name,
                )
            )
            fig_names = get_title_name(
                var_ts[-1],
                time_id=time_id,
                level=level,
                var_name=var_name,
            )
            print(fig_names)
        era5_var = xr.concat(var_ts, dim="time")
        plot_xarray_time_series(
            era5_var,
            fontsize=18,
            output_file=output_dir + fig_names[1],
            title=fig_names[0],
        )

era5_sfc = xr.open_dataset(sfc)
era5_sfc["Wind Speed (m/s)"] = np.sqrt(era5_sfc["u10"] ** 2 + era5_sfc["v10"] ** 2)

for var_name in list(era5_sfc.variables)[4:]:
    var_ts = []
    for time_id in range(era5_sfc["time"].shape[0]):
        print(time_id, level, var_name)
        var_ts.append(
            era_var_mean(
                dataset=era5_sfc,
                time_id=time_id,
                level=level,
                var_name=var_name,
            )
        )
        fig_names = get_title_name(
            var_ts[-1],
            time_id=time_id,
            level=level,
            var_name=var_name,
        )
        print(fig_names)
    era5_var = xr.concat(var_ts, dim="time")
    plot_xarray_time_series(
        era5_var, fontsize=18, output_file=output_dir + fig_names[1], title=fig_names[0]
    )



