import rasterio as rio
from rasterio.warp import Resampling, calculate_default_transform
from shapely.geometry import box
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

def process_file(project_id):
    image_path = 'uploads/' + str(project_id)
    OUTPUT = 'uploads/' + str(project_id) + '/tiles'
    INPUTS = [image_path]
    TILESIZE = 512
    #Loop through each raster file present
    print('Starting tile creation')
    #if tiles folder does not exist then create it
    os.makedirs(OUTPUT, exist_ok=True)
    INPUT = Path(INPUTS[0])
    OUTPUT = Path(OUTPUT)
    
    for raster_file in INPUT.glob('*.tif'):
        raster_file = str(raster_file)
        raster_id = raster_file.split('/')[-1].replace('.tif', '')
        with rio.open(raster_file, 'r') as raster:
            geometry = box(*raster.bounds)
            bounds = geometry.bounds

            #Create bounds for each tiles by dividng the bigger bounds
            xmin = bounds[0]
            xmax = bounds[2]
            ymin = bounds[1]
            ymax = bounds[3]
            x_extent = xmax - xmin
            y_extent = ymax - ymin
            tile_size = [TILESIZE * raster.transform[0], -TILESIZE * raster.transform[4]]
            x_steps = np.ceil(x_extent / tile_size[1])
            y_steps = np.ceil(y_extent / tile_size[0])
            x_mins = np.arange(xmin, xmin + tile_size[1] * x_steps, tile_size[1])
            y_mins = np.arange(ymin, ymin + tile_size[0] * y_steps, tile_size[0])
            tile_bounds = [
                (i, j, i + tile_size[1], j + tile_size[0])
                for i in x_mins for j in y_mins if geometry.intersects(
                box(*(i, j, i + tile_size[1], j + tile_size[0])))
            ]

            src_crs = raster.crs
            dest_crs = src_crs

            print(f'Generating {len(tile_bounds)} tiles for {raster_id} ')

            for tb in tqdm(tile_bounds, total=len(tile_bounds)):
                #window for each tile
                window = rio.windows.from_bounds(*tb, transform=raster.transform, width=TILESIZE, height=TILESIZE)
                window = window.round_lengths(op='ceil', pixel_precision=1)
                tile = raster.read(
                        window=window,
                        indexes=list(range(1, raster.count + 1)),
                        boundless=True,
                        fill_value=raster.nodata)

                dst_transform, width, height = calculate_default_transform(
                                            src_crs, dest_crs,
                                                raster.width, raster.height, *tb,
                                                dst_height=TILESIZE,
                                                dst_width=TILESIZE)
                #update destination profile
                profile = raster.profile
                profile.update(width=TILESIZE,
                            height=TILESIZE,
                            crs=dest_crs,
                            count=tile.shape[0],
                            transform=dst_transform)

                #save the tile to tiles folder
                dest_file_name = f"{raster_id}"
                dest_file_name += f"_{np.round(profile['transform'][2], 3)}"
                dest_file_name += f"_{np.round(profile['transform'][5], 3)}.tif"
                dest_path = str(OUTPUT/dest_file_name)
                with rio.open(dest_path, 'w', **profile) as dest:
                    for band in range(1, profile['count'] + 1):
                        dest.write(tile[band-1, :, :], band)