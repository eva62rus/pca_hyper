from osgeo import gdal
import os
import numpy as np


DATA_DIR = './img/'
FILE_EXT = '.tiff'


def get_raster_shape(raster):
    return raster.RasterYSize, raster.RasterXSize, raster.RasterCount


def get_band(raster, band_num):
    return raster.GetRasterBand(band_num).ReadAsArray()


def get_bands_from_raster(raster):
    rows, cols, dims = get_raster_shape(raster)
    out_img = np.zeros((rows, cols, dims)).astype(np.uint16)
    for i in range(1, dims):
        out_img[:, :, i] = get_band(raster, i)
    return out_img



def load_data_from_dir(path):
    file_names = os.listdir(path)
    out_set = []
    for file_name in file_names:
        if file_name[-len(FILE_EXT):] == FILE_EXT:
            raster = gdal.Open(path + file_name)
            out_set.append(get_bands_from_raster(raster))
    return out_set 


def main():
    images = load_data_from_dir(DATA_DIR)
    print(len(images))
    for img in images:
        print(img.shape)
    

if __name__ == '__main__':
    main()

