import tropycal.tracks as ttracks
import pandas as pd
import numpy as np
from datetime import datetime

url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
basin = ttracks.TrackDataset(basin="north_atlantic", atlantic_url=url)

hurdat_track = []
for hurdat_year in np.arange(1980, 2021):
    print(hurdat_year)
    hurdat_hurricanes = basin.get_season(hurdat_year).to_dataframe()["name"].values
    hurdat_valid_hurricanes = hurdat_hurricanes[hurdat_hurricanes != "UNNAMED"]

    tracks = [
        basin.get_storm((storm_name, hurdat_year)).to_xarray()
        for storm_name in hurdat_valid_hurricanes
    ]

    for hurr_id in range(len(tracks)):
        for time_id in range(tracks[hurr_id]["time"].shape[0]):
            track_crop = tracks[hurr_id].isel(time=time_id)
            name = track_crop.attrs["name"]

            timestamp = datetime.strptime(
                str(track_crop.time.values)[:13], "%Y-%m-%dT%H"
            )
            year = timestamp.strftime("%Y")
            month = timestamp.strftime("%m")
            day = timestamp.strftime("%d")
            hour = timestamp.strftime("%H")
            latitude = float(track_crop["lat"].values)
            longitude = float(track_crop["lon"].values)
            intensity_pressure = float(track_crop["mslp"].values)
            intensity_windspeed = float(track_crop["vmax"].values)
            hurdat_track.append(
                np.array(
                    (
                        name,
                        year,
                        month,
                        day,
                        hour,
                        latitude,
                        longitude,
                        intensity_pressure,
                        intensity_windspeed,
                    )
                )
            )


hurdat_csv = pd.DataFrame(
    np.array(hurdat_track),
    columns=[
        "Name",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Lat",
        "Lon",
        "Intensity_MSLP",
        "Intensity_WS",
    ],
)

mask = ~hurdat_csv.apply(lambda row: row.str.contains("nan")).any(axis=1)
hurdat_csv = hurdat_csv[mask]
hurdat_csv.to_csv('HURDAT.csv', index=False)








