import tropycal.tracks as ttracks
import pandas as pd
import numpy as np
from datetime import datetime
import math


def get_hurricane_category(wind_speed):
    if wind_speed <= 38:
        return "TD"
    elif wind_speed <= 73:
        return "TS"
    elif wind_speed < 96:
        return "Cat1"
    elif wind_speed < 111:
        return "Cat2"
    elif wind_speed < 130:
        return "Cat3"
    elif wind_speed < 157:
        return "Cat4"
    else:
        return "Cat5"


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Distance in kilometers
    return distance


def get_new_points(current_lat, current_lon, next_lat, next_lon, distance):
    bearing = math.atan2(
        math.radians(next_lon - current_lon),
        math.log(math.tan(math.radians(next_lat / 2 + 45))),
    )
    left_bearing = bearing + math.radians(90)
    right_bearing = bearing - math.radians(90)

    left_lat = math.degrees(
        math.asin(
            math.sin(math.radians(current_lat)) * math.cos(distance / 6371)
            + math.cos(math.radians(current_lat))
            * math.sin(distance / 6371)
            * math.cos(left_bearing)
        )
    )
    left_lon = math.degrees(
        math.radians(current_lon)
        + math.atan2(
            math.sin(left_bearing)
            * math.sin(distance / 6371)
            * math.cos(math.radians(current_lat)),
            math.cos(distance / 6371)
            - math.sin(math.radians(current_lat)) * math.sin(math.radians(left_lat)),
        )
    )

    right_lat = math.degrees(
        math.asin(
            math.sin(math.radians(current_lat)) * math.cos(distance / 6371)
            + math.cos(math.radians(current_lat))
            * math.sin(distance / 6371)
            * math.cos(right_bearing)
        )
    )
    right_lon = math.degrees(
        math.radians(current_lon)
        + math.atan2(
            math.sin(right_bearing)
            * math.sin(distance / 6371)
            * math.cos(math.radians(current_lat)),
            math.cos(distance / 6371)
            - math.sin(math.radians(current_lat)) * math.sin(math.radians(right_lat)),
        )
    )

    return (left_lat, left_lon), (right_lat, right_lon)


def get_new_points_new(current_lat, current_lon, next_lat, next_lon, distance):
    R = 6371  # Earth's radius in kilometers

    # Convert coordinates to radians
    current_lat_rad = math.radians(current_lat)
    current_lon_rad = math.radians(current_lon)
    next_lat_rad = math.radians(next_lat)
    next_lon_rad = math.radians(next_lon)

    # Calculate initial bearing from current to next point
    dlon = next_lon_rad - current_lon_rad
    y = math.sin(dlon) * math.cos(next_lat_rad)
    x = math.cos(current_lat_rad) * math.sin(next_lat_rad) - math.sin(
        current_lat_rad
    ) * math.cos(next_lat_rad) * math.cos(dlon)
    initial_bearing = math.atan2(y, x)

    # Calculate left and right points based on initial bearing
    left_bearing = initial_bearing + math.radians(90)
    right_bearing = initial_bearing - math.radians(90)

    # Calculate left and right points' coordinates
    left_lat_rad = math.asin(
        math.sin(current_lat_rad) * math.cos(distance / R)
        + math.cos(current_lat_rad) * math.sin(distance / R) * math.cos(left_bearing)
    )
    left_lon_rad = current_lon_rad + math.atan2(
        math.sin(left_bearing) * math.sin(distance / R) * math.cos(current_lat_rad),
        math.cos(distance / R) - math.sin(current_lat_rad) * math.sin(left_lat_rad),
    )
    right_lat_rad = math.asin(
        math.sin(current_lat_rad) * math.cos(distance / R)
        + math.cos(current_lat_rad) * math.sin(distance / R) * math.cos(right_bearing)
    )
    right_lon_rad = current_lon_rad + math.atan2(
        math.sin(right_bearing) * math.sin(distance / R) * math.cos(current_lat_rad),
        math.cos(distance / R) - math.sin(current_lat_rad) * math.sin(right_lat_rad),
    )

    left_lat = math.degrees(left_lat_rad)
    left_lon = math.degrees(left_lon_rad)
    right_lat = math.degrees(right_lat_rad)
    right_lon = math.degrees(right_lon_rad)

    return (left_lat, left_lon), (right_lat, right_lon)


def get_lr_lonlats(lons_in, lats_in):
    lons = np.append(lons_in, lons_in[-1])
    lats = np.append(lats_in, lats_in[-1])
    newlonslats = [
        get_new_points_new(
            lats[index], lons[index], lats[index + 1], lons[index + 1], 500
        )
        for index in range(lons.shape[0] - 1)
    ]
    right = np.round(np.array(newlonslats)[:, 0, :], 2)
    left = np.round(np.array(newlonslats)[:, 1, :], 2)
    return right[:, 0], right[:, 1], left[:, 0], left[:, 1]


url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
basin = ttracks.TrackDataset(basin="north_atlantic", atlantic_url=url)
new_hurdat_track = []
for hurdat_year in np.arange(1980, 2021):
    print(hurdat_year)
    hurdat_hurricanes = basin.get_season(hurdat_year).to_dataframe()["name"].values
    hurdat_valid_hurricanes = hurdat_hurricanes[hurdat_hurricanes != "UNNAMED"]

    tracks = [
        basin.get_storm((storm_name, hurdat_year)).to_xarray()
        for storm_name in hurdat_valid_hurricanes
    ]

    for hurr_id in range(len(tracks)):
        hurdat_track = []
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
            intensity_cat = get_hurricane_category(intensity_windspeed)

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
                        intensity_cat,
                    )
                )
            )

        right_lat, right_lon, left_lat, left_lon = get_lr_lonlats(
            np.array(hurdat_track)[:, 5].astype(float),
            np.array(hurdat_track)[:, 6].astype(float),
        )
        new_hurdat_track.append(
            np.hstack(
                (
                    np.array(hurdat_track),
                    right_lat.reshape(-1, 1),
                    right_lon.reshape(-1, 1),
                    left_lat.reshape(-1, 1),
                    left_lon.reshape(-1, 1),
                )
            )
        )


hurdat_csv = pd.DataFrame(
    np.vstack(np.array(new_hurdat_track, dtype="object")),
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
        "Intensity_Cat",
        "right_lat",
        "right_lon",
        "left_lat",
        "left_lon",
    ],
)

mask = ~hurdat_csv.apply(lambda row: row.str.contains("nan")).any(axis=1)
hurdat_csv = hurdat_csv[mask]
hurdat_csv.to_csv("HURDAT.csv", index=False)
