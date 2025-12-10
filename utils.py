from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import osmnx as ox
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import warnings


roads_filepath = "data/tl_2024_44_prisecroads/tl_2024_44_prisecroads.shp"
population_filepath = "data/tl_2024_44_tabblock20/tl_2024_44_tabblock20.shp"

class ProxysDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :, :, :], self.Y[index, :, :]
    
    def check_data_shapes(self):
        print("self.X: ", self.X.shape)
        print("self.Y: ", self.Y.shape)
        
class PredictionsDataset(Dataset):
    def __init__(self, X):
        super().__init__()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

def get_prediction_dataloader() -> DataLoader:
    print("Loading Data")
    roads, population, landuse, irradiance = load_data()
    print("Creating Prediction Data")
    dataset = PredictionsDataset(create_prediction_data(roads, population, landuse, irradiance))
    return DataLoader(dataset, batch_size=1, shuffle=False)

def create_prediction_data(roads, population, landuse, irradiance) -> np.ndarray:
    X = []
    for i in range(irradiance.shape[0]):
        sample = np.concatenate([roads[None, :, :], population[None, :, :], landuse, irradiance[i, None, :, :]], axis=0)
        X.append(sample)
    return np.array(X)

def get_proxys_dataloader() -> tuple[DataLoader, DataLoader]:
    print("Loading Data")
    roads, population, landuse, irradiance = load_data()
    print("Creating Training Data")
    X = create_training_data(roads, population, landuse, irradiance, size_reduction=5)
    Y = irradiance
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

    train_dataset = ProxysDataset(X_train, Y_train)
    val_dataset = ProxysDataset(X_val, Y_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    return train_dataloader, val_dataloader


def create_training_data(roads, population, landuse, irradiance, size_reduction):
    X = []
    for i in range(irradiance.shape[0]):
        target_shape = roads.shape

        reduced_roads = roads[::size_reduction, ::size_reduction]
        reduced_population = population[::size_reduction, ::size_reduction]
        reduced_landuse = landuse[:, ::size_reduction, ::size_reduction]
        reduced_irradiance = irradiance[i, ::size_reduction, ::size_reduction]

        zoom_factors = (target_shape[0] / reduced_roads.shape[0], target_shape[1] / reduced_roads.shape[1])

        roads_upscaled = zoom(reduced_roads, zoom_factors, order=0)
        population_upscaled = zoom(reduced_population, zoom_factors, order=0)
        landuse_upscaled = np.zeros_like(landuse)
        for i in range(reduced_landuse.shape[0]):
            landuse_upscaled[i] = zoom(reduced_landuse[i, :, :], zoom_factors, order=0)
        irradiance_upscaled = zoom(reduced_irradiance, zoom_factors, order=0)
        
        sample = np.concatenate([roads_upscaled[None, :, :], population_upscaled[None, :, :], landuse_upscaled, irradiance_upscaled[None, :, :]], axis=0)
        X.append(sample)

    return np.array(X)

'''
Loads in irradiance data from VIIRS directory. Returns a list of datasets, all of 
which contain irradiance data for different days of the year. 
'''
def open_irradiance_data() -> rasterio.io.DatasetReader:
    datasets = []

    for i in range (1, 11):
        path = ''
        if i < 10:
            path = 'data/VIIRS/VNP46A2.A202400' + str(i) + '.h10v04.h5'
        else:
            path = 'data/VIIRS/VNP46A2.A20240' + str(i) + '.h10v04.h5'
        
        # ignore warning from rasterio about non georeferenced data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            path = rasterio.open(path).subdatasets[2]
            
        datasets.append(rasterio.open(path))
    
    return datasets

'''
Trims the irradiance data to use the same bounds as the roads data. Irradiance tiles
are much larger than Rhode Island.
'''
def trim_irradiance_data(irradiance_data: rasterio.io.DatasetReader, bounds: tuple, roads_bounds) -> np.ndarray:
    shapey, shapex = irradiance_data.shape
    X = np.linspace(bounds[0], bounds[2], shapex)
    Y = np.linspace(bounds[3], bounds[1], shapey)
    
    minx_indx = np.argmin(np.abs(X - roads_bounds[0]))
    maxx_indx = np.argmin(np.abs(X - roads_bounds[2]))
    miny_indx = np.argmin(np.abs(Y - roads_bounds[1]))
    maxy_indx = np.argmin(np.abs(Y - roads_bounds[3]))

    return irradiance_data.read(1)[maxy_indx:miny_indx, minx_indx:maxx_indx]


'''
Reshapes irradiance raster to the target_shape by nearest neighbor upscaling. 
Resolution remains ~500m but the number of total pixels now matches finer resolution
proxy datasets. 
'''
def reshape_irradiance(irradiance_raster: np.ndarray, target_shape: tuple) -> np.ndarray:
    zoom_factors = (target_shape[0] / irradiance_raster.shape[0], 
                    target_shape[1] / irradiance_raster.shape[1])
    
    irradiance_raster_resampled = zoom(irradiance_raster, zoom_factors, order=0)
    
    return irradiance_raster_resampled

'''
Loads in and processes all proxy datasets to be later used for model training.
'''
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roads_gdf = gpd.read_file(roads_filepath)
    population_gdf = gpd.read_file(population_filepath)
    land_use_gdf = get_landuse_gdf()
    irradiance_data = open_irradiance_data()

    irradiance_bounds = (irradiance_data[0].bounds.left, irradiance_data[0].bounds.bottom, irradiance_data[0].bounds.right, irradiance_data[0].bounds.top)

    irradiance_rasters_raw = []
    for ds in irradiance_data:
        irradiance_rasters_raw.append(trim_irradiance_data(ds, irradiance_bounds, roads_gdf.total_bounds))

    gdf_pop = population_gdf.to_crs("EPSG:26919")
    gdf_roads = roads_gdf.to_crs("EPSG:26919")
    gdf_landuse = land_use_gdf.to_crs("EPSG:26919")  

    roads_bounds = gdf_roads.total_bounds
    population_bounds = gdf_pop.total_bounds
    landuse_bounds = gdf_landuse.total_bounds

    pixel_size = 100

    roads_raster = rasterize_roads(roads_bounds[0], roads_bounds[1], roads_bounds[2], roads_bounds[3], gdf_roads, pixel_size=pixel_size)
    population_raster = rasterize_population(population_bounds[0], population_bounds[1], population_bounds[2], population_bounds[3], gdf_pop, pixel_size=pixel_size)
    landuse_rasters_raw = rasterize_landuse(landuse_bounds[0], landuse_bounds[1], landuse_bounds[2], landuse_bounds[3], gdf_landuse, pixel_size=pixel_size)

    cells_off_x_min_pop = int(abs(roads_bounds[0] - population_bounds[0]) / pixel_size + 1)
    cells_off_x_max_pop = int(abs(roads_bounds[2] - population_bounds[2]) / pixel_size)
    cells_off_y_min_pop = int(abs(roads_bounds[1] - population_bounds[1]) / pixel_size)
    cells_off_y_max_pop = int(abs(roads_bounds[3] - population_bounds[3]) / pixel_size)

    cells_off_x_min_landuse = int(abs(roads_bounds[0] - landuse_bounds[0]) / pixel_size + 1)
    cells_off_x_max_landuse = int(abs(roads_bounds[2] - landuse_bounds[2]) / pixel_size)
    cells_off_y_min_landuse = int(abs(roads_bounds[1] - landuse_bounds[1]) / pixel_size + 1)
    cells_off_y_max_landuse = int(abs(roads_bounds[3] - landuse_bounds[3]) / pixel_size)
    
    population_raster = population_raster[cells_off_y_max_pop:-cells_off_y_min_pop, cells_off_x_max_pop:-cells_off_x_min_pop]

    landuse_rasters = []
    for raster in landuse_rasters_raw:
        landuse_rasters.append(raster[cells_off_y_max_landuse:-cells_off_y_min_landuse, cells_off_x_max_landuse:-cells_off_x_min_landuse])
    
    irradiance_rasters = []
    for raster in irradiance_rasters_raw:
        irradiance_rasters.append(reshape_irradiance(raster, population_raster.shape))
    
    for ds in irradiance_data:
        ds.close()

    return roads_raster, population_raster, np.array(landuse_rasters), np.array(irradiance_rasters)


'''
Converts roads GeoDataFrame into a numpy ndarray raster.
'''
def rasterize_roads(minx, miny, maxx, maxy, gdf: gpd.GeoDataFrame, pixel_size: int) -> np.ndarray:
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    shapes = ((geom, 1) for geom in gdf.geometry)

    roads_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=int,
    )
    
    return roads_raster


'''
Converts population GeoDataFrame into a numpy ndarray raster.
'''
def rasterize_population(minx, miny, maxx, maxy, gdf: gpd.GeoDataFrame, pixel_size: int) -> np.ndarray:
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['POP20']))

    population_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        dtype=int
    )

    return population_raster

'''
Converts landuse GeoDataFrame into a list of numpy ndarray rasters, one for each landuse type.
'''
def rasterize_landuse(minx, miny, maxx, maxy, gdf: gpd.GeoDataFrame, pixel_size: int) -> tuple:
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    landuse_rasters_raw = []
    landuse_types = []

    for landuse in gdf.landuse.unique():
        shapes = ((geom, 1) for geom in gdf[gdf.landuse == landuse].geometry)

        rastered = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            dtype=int
        )

        if np.sum(rastered) > 0:
            landuse_rasters_raw.append(rastered)
            landuse_types.append(landuse)

    return landuse_rasters_raw

'''
Loads landuse data from open street maps for the state of Rhode Island.
'''
def get_landuse_gdf() -> gpd.GeoDataFrame:
    place = "Rhode Island, USA"
    tags ={'landuse':True}
    
    gdf_landuse=ox.features_from_place(place, tags=tags)

    return gdf_landuse