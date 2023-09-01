import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime, timedelta


from osgeo import gdal
from osgeo import osr

import xarray as xr
import os
import pyproj
fname = '/rhome/akumar/Downloads/OR_ABI-L2-MCMIPC-M3_G16_s20172371457173_e20172371459546_c20172371500046.nc' 

for index in range(1):
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
import numpy as np

def generate_lon_lat_range(cen_lon, cen_lat, box_size_deg=10):
    half_box_size_deg = box_size_deg / 2
    
    min_lon = cen_lon - half_box_size_deg
    max_lon = cen_lon + half_box_size_deg
    min_lat = cen_lat - half_box_size_deg
    max_lat = cen_lat + half_box_size_deg
    
    lon_range = np.arange(min_lon, max_lon + 0.01, 0.025)  # Adjust step size as needed
    lat_range = np.arange(min_lat, max_lat + 0.01, 0.025)  # Adjust step size as needed
    
    return lon_range, lat_range




central_lon=C['goes_imager_projection'].attrs['longitude_of_projection_origin']

#proj = ccrs.PlateCarree(central_longitude=central_lon, globe=globe)
proj = ccrs.PlateCarree()


xx, yy = np.meshgrid(x.values, y.values)
latlon=proj.transform_points(geos, xx, yy)

lats = latlon[:,:,1]
lons = latlon[:,:,0]

cen_lon = -96.05
cen_lat = 27.82

from scipy.interpolate import griddata

lon_range, lat_range = generate_lon_lat_range(cen_lon, cen_lat, box_size_deg=12)
lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)

r_interp = griddata((xx.ravel(), yy.ravel()), RGB[:, :, 0].ravel(), (lon_mesh.ravel(), lat_mesh.ravel()))
g_interp = griddata((xx.ravel(), yy.ravel()), RGB[:, :, 0].ravel(), (lon_mesh.ravel(), lat_mesh.ravel()))
b_interp = griddata((xx.ravel(), yy.ravel()), RGB[:, :, 0].ravel(), (lon_mesh.ravel(), lat_mesh.ravel()))

interp = griddata((lons.ravel(), lats.ravel()), np.array((RGB[:, :, 0].ravel(), RGB[:, :, 1].ravel(), RGB[:, :, 2].ravel())).T, (lon_mesh.ravel(), lat_mesh.ravel()))

rgb_cropped = (np.stack((interp[:, 0].reshape(lon_mesh.shape), interp[:, 1].reshape(lon_mesh.shape), interp[:, 2].reshape(lon_mesh.shape)), axis=2)*255).astype(int)

image_size = rgb_cropped[:,:,0].shape
lat = lat_range
lon = lon_range
output_filename = 'goes_hurricane.tiff'

nx = image_size[1]
ny = image_size[0]
xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)
geotransform = (xmin, xres, 0, ymax, 0, -yres)

dst_ds = gdal.GetDriverByName("GTiff").Create(output_filename, nx, ny, 3, gdal.GDT_Byte)

dst_ds.SetGeoTransform(geotransform)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
dst_ds.SetProjection(srs.ExportToWkt())

dst_ds.GetRasterBand(1).WriteArray(rgb_cropped[:, :, 0])
dst_ds.GetRasterBand(2).WriteArray(rgb_cropped[:, :, 1])
dst_ds.GetRasterBand(3).WriteArray(rgb_cropped[:, :, 2])

dst_ds.FlushCache()
dst_ds = None

#write_to_COG(output_filename)




"""
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)

ax.imshow(rgb_cropped, extent=(lon_range.min(), lon_range.max(), lat_range.min(), lat_range.max()))
ax.coastlines(resolution="50m", color="black", linewidth=2)
ax.add_feature(ccrs.cartopy.feature.STATES)
plt.savefig('RGB_test.jpeg')
plt.show()

"""









