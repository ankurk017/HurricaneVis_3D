import numpy as np
import datetime
from scipy.interpolate import griddata
from netCDF4 import Dataset
from pyproj import Proj
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
from datetime import datetime, timedelta
import boto3
from datetime import datetime

import tropycal.tracks as tracks

from osgeo import gdal
from osgeo import osr

import os
import pyproj


def generate_lon_lat_range(cen_lon, cen_lat, box_size_deg=10):
    half_box_size_deg = box_size_deg / 2

    min_lon = cen_lon - half_box_size_deg
    max_lon = cen_lon + half_box_size_deg
    min_lat = cen_lat - half_box_size_deg
    max_lat = cen_lat + half_box_size_deg

    lon_range = np.arange(min_lon, max_lon + 0.01, 0.025)
    lat_range = np.arange(min_lat, max_lat + 0.01, 0.025)

    return lon_range, lat_range


def create_goes_imagery(fname, location, output_filename):

    if "s3://" in fname:
        C = read_nc_file_from_s3(fname)
    else:
        C = xr.open_dataset(fname)

    scan_start = datetime.strptime(C.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")

    scan_end = datetime.strptime(C.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")

    file_created = datetime.strptime(C.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

    midpoint = str(C["t"].data)[:-8]
    scan_mid = datetime.strptime(midpoint, "%Y-%m-%dT%H:%M:%S.%f")

    R = C["CMI_C02"].data
    G = C["CMI_C03"].data
    B = C["CMI_C01"].data
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    gamma = 2.2
    R = np.power(R, 1 / gamma)
    G = np.power(G, 1 / gamma)
    B = np.power(B, 1 / gamma)

    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)

    RGB_veggie = np.dstack([R, G, B])

    RGB = np.dstack([R, G_true, B])

    sat_h = C["goes_imager_projection"].perspective_point_height
    sat_lon = C["goes_imager_projection"].longitude_of_projection_origin
    sat_sweep = C["goes_imager_projection"].sweep_angle_axis
    semi_major = C["goes_imager_projection"].semi_major_axis
    semi_minor = C["goes_imager_projection"].semi_minor_axis

    x = C["x"][:] * sat_h
    y = C["y"][:] * sat_h

    globe = ccrs.Globe(semimajor_axis=semi_major, semiminor_axis=semi_minor)
    geos = ccrs.Geostationary(
        central_longitude=sat_lon, satellite_height=sat_h, globe=globe
    )

    central_lon = C["goes_imager_projection"].attrs["longitude_of_projection_origin"]

    # proj = ccrs.PlateCarree(central_longitude=central_lon, globe=globe)
    proj = ccrs.PlateCarree()

    xx, yy = np.meshgrid(x.values, y.values)
    latlon = proj.transform_points(geos, xx, yy)

    lats = latlon[:, :, 1]
    lons = latlon[:, :, 0]

    lon_range, lat_range = generate_lon_lat_range(
        location["cen_lon"], location["cen_lat"], box_size_deg=12
    )
    lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)

    r_interp = griddata(
        (xx.ravel(), yy.ravel()),
        RGB[:, :, 0].ravel(),
        (lon_mesh.ravel(), lat_mesh.ravel()),
    )
    g_interp = griddata(
        (xx.ravel(), yy.ravel()),
        RGB[:, :, 0].ravel(),
        (lon_mesh.ravel(), lat_mesh.ravel()),
    )
    b_interp = griddata(
        (xx.ravel(), yy.ravel()),
        RGB[:, :, 0].ravel(),
        (lon_mesh.ravel(), lat_mesh.ravel()),
    )

    interp = griddata(
        (lons.ravel(), lats.ravel()),
        np.array((RGB[:, :, 0].ravel(), RGB[:, :, 1].ravel(), RGB[:, :, 2].ravel())).T,
        (lon_mesh.ravel(), lat_mesh.ravel()),
    )

    rgb_cropped = (
        np.stack(
            (
                interp[:, 0].reshape(lon_mesh.shape),
                interp[:, 1].reshape(lon_mesh.shape),
                interp[:, 2].reshape(lon_mesh.shape),
            ),
            axis=2,
        )
        * 255
    ).astype(int)

    image_size = rgb_cropped[:, :, 0].shape
    lat = lat_range
    lon = lon_range

    nx = image_size[1]
    ny = image_size[0]
    xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    dst_ds = gdal.GetDriverByName("GTiff").Create(
        output_filename, nx, ny, 3, gdal.GDT_Byte
    )

    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())

    dst_ds.GetRasterBand(1).WriteArray(rgb_cropped[:, :, 0])
    dst_ds.GetRasterBand(2).WriteArray(rgb_cropped[:, :, 1])
    dst_ds.GetRasterBand(3).WriteArray(rgb_cropped[:, :, 2])

    dst_ds.FlushCache()
    dst_ds = None


def read_nc_file_from_s3(s3filename):
    bucket_name = s3filename.split("/")[2]
    file_name = "/".join(s3filename.split("/")[3:])

    local_file_path = "/tmp/" + file_name.split("/")[-1]

    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
    except ClientError as e:
        print(f"An error occurred while checking the object: {e}")
    #    return None

    try:
        s3.download_file(bucket_name, file_name, local_file_path)
    except ClientError as e:
        print(f"An error occurred while downloading the file: {e}")
    #   return None

    ds = xr.open_dataset(local_file_path)

    os.remove(local_file_path)

    return ds


import warnings



def get_goes_s3_url(timestamp_datetime: datetime) -> str:
    s3 = boto3.client("s3")
    bucket_name = "noaa-goes16"
    prefix = f'ABI-L2-MCMIPC/{timestamp_datetime.strftime("%Y")}/{str(timestamp_datetime.timetuple().tm_yday).zfill(3)}/{timestamp_datetime.strftime("%H")}/'
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=100)
    urls = [f"s3://{bucket_name}/{obj['Key']}" for obj in response["Contents"]]
    goes_file = urls[0]
    return goes_file


warnings.filterwarnings("ignore", category=DeprecationWarning, module="boto3.compat")


#fname = "/rhome/akumar/Downloads/OR_ABI-L2-MCMIPC-M3_G16_s20172371457173_e20172371459546_c20172371500046.nc"

url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)

hurr_name = 'HARVEY'
storm = basin.get_storm((hurr_name, 2017))

for time_ids in np.arange(1, 55):
 timestamp = storm['date'][time_ids]
 location = {"cen_lon": storm['lon'][55], "cen_lat": storm['lon'][55]}
 
 output_filename = f"/rtmp/akumar/GEOS/goes_{hurr_name}_{timestamp.strftime('%Y%m%d%H')}.tiff"
 
 fname = get_goes_s3_url(timestamp)
 create_goes_imagery(fname, location, output_filename)

















