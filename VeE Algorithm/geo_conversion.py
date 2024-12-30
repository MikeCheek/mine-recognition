import os
import random
from osgeo import gdal, osr
from geopy.distance import geodesic


def convert_images_to_tiff(src_folder, dest_folder):
    # Generate random coordinates around Paris (in km)
    paris_coords = (48.8566, 2.3522)
    radius = 50
    os.makedirs(dest_folder, exist_ok=True)
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_file in image_files:
        # Open the image
        src_path = os.path.join(src_folder, img_file)
        dest_path = os.path.join(dest_folder, os.path.splitext(img_file)[0] + ".tiff")

        # Generate random coordinates for the top-left corner
        random_angle_tl = random.uniform(0, 360)
        random_distance_tl = random.uniform(0, radius)
        top_left_coords = geodesic(kilometers=random_distance_tl).destination(paris_coords, random_angle_tl)
        top_left_latitude, top_left_longitude = top_left_coords.latitude, top_left_coords.longitude

        # Generate random coordinates for the bottom-right corner
        random_angle_br = random.uniform(0, 360)
        random_distance_br = random.uniform(0, radius)
        bottom_right_coords = geodesic(kilometers=random_distance_br).destination(paris_coords, random_angle_br)
        bottom_right_latitude, bottom_right_longitude = bottom_right_coords.latitude, bottom_right_coords.longitude

        # Open the source image using GDAL
        src_ds = gdal.Open(src_path)
        if src_ds is None:
            print(f"Failed to open {src_path}")
            continue

        # Get the dimensions of the image
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize

        # Calculate pixel size based on geographic coordinates
        pixel_width = (bottom_right_longitude - top_left_longitude) / width
        pixel_height = (top_left_latitude - bottom_right_latitude) / height

        # Create a new GeoTIFF dataset
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(dest_path, width, height, src_ds.RasterCount, gdal.GDT_Byte)

        if dst_ds is None:
            print(f"Failed to create {dest_path}")
            continue

        # Copy image data from source to destination
        for band in range(1, src_ds.RasterCount + 1):
            src_band = src_ds.GetRasterBand(band)
            dst_band = dst_ds.GetRasterBand(band)
            dst_band.WriteArray(src_band.ReadAsArray())

        # Set the GeoTransform and Projection
        geotransform = [
            top_left_longitude, pixel_width, 0,  # Top-left longitude, pixel width, rotation
            top_left_latitude, 0, -pixel_height  # Top-left latitude, rotation, -pixel height
        ]
        dst_ds.SetGeoTransform(geotransform)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84 coordinate system
        dst_ds.SetProjection(srs.ExportToWkt())

        # Close datasets
        src_ds = None
        dst_ds = None

    print(f"Conversion completed. GeoTIFF images saved in: {dest_folder}")

# Example usage:
# convert_images_to_tiff("path/to/source/folder", "path/to/destination/folder")
