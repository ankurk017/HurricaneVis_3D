import numpy as np
import xarray as xr
import copy
import numpy as np

from osgeo import gdal, osr
import pyproj
from scipy.interpolate import interp2d

from scipy.interpolate import griddata
import pandas as pd
from shapely.geometry import Point

import geopandas as gpd

year = '2001'
modis_ndvi_file = f"/nas/rstor/akumar/USA/PhD/Objective01/Hurricane_Harvey/update_geog/NDVI/NDVI_{year}.tif"

ds = gdal.Open(modis_ndvi_file)
obsdat = ds.ReadAsArray()
gt = ds.GetGeoTransform()
width = ds.RasterXSize
height = ds.RasterYSize

minx = gt[0]
miny = gt[3] + width * gt[4] + height * gt[5]
maxx = gt[0] + width * gt[1] + height * gt[2]
maxy = gt[3]

# Getting Projection
proj = ds.GetProjection()
inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)
#projcs = inproj.GetAuthorityCode("PROJCS")
projcs = inproj.GetAuthorityCode('GEOGCS')

# Transforming coordinates
xcord = np.array([minx, minx, maxx, maxx])
ycord = np.array([miny, maxy, miny, maxy])

proj_convert = pyproj.Transformer.from_crs(int(projcs), 4326, always_xy=True)

cords = np.vstack(
    [
        proj_convert.transform(xcord[index], ycord[index])
        for index in range(xcord.shape[0])
    ]
)
lon_start = cords[0, 0]
lon_end = cords[2, 0]
lat_start = cords[0, 1]
lat_end = cords[1, 1]
print(np.round(np.array([lon_start, lon_end, lat_start, lat_end]), 2))

longitudes = np.linspace(lon_start, lon_end, obsdat.shape[1])
latitudes = np.linspace(lat_start, lat_end, obsdat.shape[0])
scale_factor = 0.0001
ndvi_obs = np.flip(obsdat, 0) * scale_factor

ndvi_obs = ndvi_obs

houston_loc = (-95.3698, 29.7604)

bounding_box = (houston_loc[0]-1.5, houston_loc[0]+1.5, houston_loc[1]-1.5, houston_loc[1]+1.5)

min_lon, max_lon, min_lat, max_lat = bounding_box

lon_indices = np.where(np.logical_and(longitudes >= min_lon, longitudes <= max_lon))[0]
lat_indices = np.where(np.logical_and(latitudes >= min_lat, latitudes <= max_lat))[0]

lon_cropped = longitudes[slice(lon_indices[0], lon_indices[-1])]
lat_cropped = latitudes[slice(lat_indices[0], lat_indices[-1])]
ndvi_cropped_pre = ndvi_obs[slice(lat_indices[0], lat_indices[-1]), slice(lon_indices[0], lon_indices[-1])]

lon_mesh, lat_mesh = np.meshgrid(lon_cropped, lat_cropped)


year = '2020'
modis_ndvi_file = f"/nas/rstor/akumar/USA/PhD/Objective01/Hurricane_Harvey/update_geog/NDVI/NDVI_{year}.tif"

ds = gdal.Open(modis_ndvi_file)
obsdat = ds.ReadAsArray()
gt = ds.GetGeoTransform()
width = ds.RasterXSize
height = ds.RasterYSize

minx = gt[0]
miny = gt[3] + width * gt[4] + height * gt[5]
maxx = gt[0] + width * gt[1] + height * gt[2]
maxy = gt[3]

# Getting Projection
proj = ds.GetProjection()
inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)
#projcs = inproj.GetAuthorityCode("PROJCS")
projcs = inproj.GetAuthorityCode('GEOGCS')

# Transforming coordinates
xcord = np.array([minx, minx, maxx, maxx])
ycord = np.array([miny, maxy, miny, maxy])

proj_convert = pyproj.Transformer.from_crs(int(projcs), 4326, always_xy=True)

cords = np.vstack(
    [
        proj_convert.transform(xcord[index], ycord[index])
        for index in range(xcord.shape[0])
    ]
)
lon_start = cords[0, 0]
lon_end = cords[2, 0]
lat_start = cords[0, 1]
lat_end = cords[1, 1]
print(np.round(np.array([lon_start, lon_end, lat_start, lat_end]), 2))

longitudes = np.linspace(lon_start, lon_end, obsdat.shape[1])
latitudes = np.linspace(lat_start, lat_end, obsdat.shape[0])
scale_factor = 0.0001
ndvi_obs = np.flip(obsdat, 0) * scale_factor

ndvi_obs = ndvi_obs

houston_loc = (-95.3698, 29.7604)

bounding_box = (houston_loc[0]-1.5, houston_loc[0]+1.5, houston_loc[1]-1.5, houston_loc[1]+1.5)

min_lon, max_lon, min_lat, max_lat = bounding_box

lon_indices = np.where(np.logical_and(longitudes >= min_lon, longitudes <= max_lon))[0]
lat_indices = np.where(np.logical_and(latitudes >= min_lat, latitudes <= max_lat))[0]

lon_cropped = longitudes[slice(lon_indices[0], lon_indices[-1])]
lat_cropped = latitudes[slice(lat_indices[0], lat_indices[-1])]
ndvi_cropped_post = ndvi_obs[slice(lat_indices[0], lat_indices[-1]), slice(lon_indices[0], lon_indices[-1])]

lon_mesh, lat_mesh = np.meshgrid(lon_cropped, lat_cropped)



df  = pd.DataFrame(np.array((lon_mesh.ravel(), lat_mesh.ravel(), (ndvi_cropped_post-ndvi_cropped_pre).ravel())).T, columns=['lons', 'lat', 'NDVI_diff'])
geometry = [Point(xy) for xy in zip(df['lons'], df['lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

output_filename = 'NDVI_diff.geojson'
geo_df.to_file(output_filename, driver='GeoJSON')





