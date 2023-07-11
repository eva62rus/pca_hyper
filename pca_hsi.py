from osgeo import gdal
import os
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DATA_DIR = './img/'
FILE_EXT = '.tiff'
N_COMPONENTS = 2



def time_counter(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        f = func(*args, **kwargs)
        end = datetime.now() - start
        print(f'time of executing: {end.total_seconds()}')
        return f
    return wrapper


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



def print_info(image_set, text=None):
    if text is not None:
        print(text)
    print(f'Image count: {len(image_set)}')
    for image in image_set:
        print(image.shape)



def reshape_img(img):
    return img.reshape(img.shape[0] * img.shape[1], -1)



def concatenate_images(img_set):
    pca_input = None
    for img in img_set:
        if pca_input is None:
            pca_input = img
        else:
            pca_input = np.vstack((pca_input, img))
    return pca_input



def plot_eigen_values(eigen_values, npoints):
    vals = eigen_values[:npoints]
    x = np.arange(0, len(vals), 1)
    plt.plot(x, vals, marker = 'o')
    plt.show()
    


def plot_projected_data(projected_data, title=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1])
    plt.legend(handles=scatter.legend_elements()[0], loc="upper left")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if title is not None:
        plt.title(title)
    plt.show()



def skl_pca(pca_input):
    pca = PCA(n_components=N_COMPONENTS)
    return pca.fit_transform(pca_input)



@time_counter
def main():
    # загрузка изображений
    images = load_data_from_dir(DATA_DIR)
    print_info(images, 'Source images:')
    # изменение размерности к [rows * cols, dimensions]
    for i in range(len(images)):
        images[i] = reshape_img(images[i])
    print_info(images, 'Images after reshape:')
    # объединение данных в один массив
    pca_input = concatenate_images(images)
    print(f'pca input shape: {pca_input.shape}')
    # вычисление ковариационной матрицы
    covariance_matrix = np.cov(pca_input.T)
    # вычисление собственных значений и 
    # собственных векторов ковариационной матрицы
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    # сортировка собственных значений 
    # для ранжирования собственных векторов
    indeces = np.arange(0, len(eigen_values), 1)
    indeces = [x for _, x in sorted(zip(eigen_values, indeces))]
    indeces = indeces[::-1]
    # eigen_values1 = eigen_values[indeces]
    # plot_eigen_values(eigen_values1, 10)
    eigen_vectors1 = eigen_vectors[:, indeces] 
    # извлечение N_COMPONENTS главных компонент
    eigen_vectors1 = eigen_vectors1[:, :N_COMPONENTS]
    # Матрица проекции собственных векторов
    projection_matrix = (eigen_vectors.T[:][:N_COMPONENTS]).T
    print(f'Projection matrix of eigen vectors: {projection_matrix.shape}')
    print(projection_matrix)
    with open('Projection_matrix.txt', 'w') as f:
        f.write(str(projection_matrix))
    # Проецирование исходных данных
    pca_out = pca_input.dot(projection_matrix)
    print(pca_out.shape)
    plot_projected_data(pca_out, 'numpy version')
    # Проверка с sklearn
    pca = PCA(2)
    pca_out_1 = pca.fit_transform(pca_input)
    plot_projected_data(pca_out_1, 'sklearn version')



if __name__ == '__main__':
    main()

