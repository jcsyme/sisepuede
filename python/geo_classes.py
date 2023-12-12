###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY AND SUPPORT DATA PIPELINE
###

import geo_functions as gf
import logging
from model_attributes import *
import numpy as np
import pandas as pd
import rioxarray as rx
import support_functions as sf



class Grid:
    """
    Get information about the grid that is implied by the input dataframe
        derived from a GEOTIFF
        
    Initialization Arguments
    ------------------------
    - df_in: input data frame derived from GeoTiff or input path to read GeoTiff
    
    Keyword Arguments
    -----------------
    - decimals: number of decimals to use for rounding
    """
    
    def __init__(self,
        df_in: Union[pd.DataFrame, str],
        decimals: int = 8,
    ):
        
        # must initialize grid first, then order
        self._initialize_grid(df_in, decimals,)
        self._initialize_coords()
        
    
    
    
    #############################
    #    INITIALIZE ELEMENTS    #
    #############################
    
    def _initialize_coords(self,
        decimals: Union[int, None] = None,
    ) -> None:
        """
        Initialize grid elements. Sets the following properties:
            
            self.bounds_x
            self.bounds_y
            self.centroids_x
            self.centroids_y
            self.delta_x
            self.delta_y
            self.orientation_x
            self.orientation_y
            self.x_max
            self.x_min
            self.y_max
            self.y_min
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - decimals: number of decimals to use for rounding to infer grid width
            (defaults to self.coordinate_accuracy)
        """

        decimals = (
            self.coordinate_accuracy 
            if not isinstance(decimals, int) 
            else max(decimals, 0)
        )


        ##  X AXIS
        
        x_centroids = self.centroids_x
        x_delta = self.get_delta(x_centroids, decimals)
        x_min = x_centroids.min() - x_delta/2
        x_max = x_centroids.max() + x_delta/2
        
        x_bounds = self.get_boundaries_from_centroids_delta(x_centroids, x_delta)
        x_orientation = self.get_orientation(x_centroids)
        
        
        ##  Y AXIS
        
        y_centroids = self.centroids_y
        y_delta = self.get_delta(y_centroids, decimals)
        y_min = y_centroids.min() - y_delta/2
        y_max = y_centroids.max() + y_delta/2
        
        y_bounds = self.get_boundaries_from_centroids_delta(y_centroids, y_delta)
        y_orientation = self.get_orientation(y_centroids)
        
    
        ##  SET PROPERTIES
        
        self.bounds_x = x_bounds
        self.bounds_y = y_bounds
        self.delta_x = x_delta
        self.delta_y = y_delta
        self.orientation_x = x_orientation
        self.orientation_y = y_orientation
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        
        return None
    
    
    
    def _initialize_grid(self,
        data_in: Union[pd.DataFrame, 'xarray.DataArray', str],
        decimals: int,
    ) -> None:
        """
        Initialize the grid using a file path or a grid data frame. Sets the
            following properties:
            
            * self.coordinate_accuracy 
            * self.data
            * self.shape
        
        Function Arguments
        ------------------
        - data_in: input dataset used to initialize the grid
        - decimals: number of decimals to use for rounding to infer grid width

        """
        # check types
        no_error_q = isinstance(data_in, str)
        no_error_q |= isinstance(data_in, pd.DataFrame)
        no_error_q |= isinstance(data_in, rx.raster_array.xarray.DataArray)

        if not no_error_q:
            raise RuntimeError(f"Error in Grid: Invalid type '{type(data_in)}' of data_in specified. Must be str or pandas.DataFrame.")

        # if string, try reading as file
        if isinstance(data_in, str):
            if not os.path.exists(data_in):
                raise RuntimeError(f"Error initializing Grid: path {data_in} not found.")

            rx_array = rx.open_rasterio(data_in)
            centroids_x = rx_array[0].x.to_numpy()
            centroids_y = rx_array[0].y.to_numpy()
            data = rx_array[0].to_numpy()
            

        if isinstance(data_in, rx.raster_array.xarray.DataArray):
            centroids_x = data_in.x.to_numpy()
            centroids_y = data_in.y.to_numpy()
            data = data_in.to_numpy()


        if isinstance(data_in, pd.DataFrame):
            centroids_x = np.array(data_in.columns).round(decimals = decimals)
            centroids_y = np.array(data_in.index).round(decimals = decimals)
            data = data_in.to_numpy()

        dims = (len(centroids_y), len(centroids_x))
        
        ##  SET PROPERTIES

        self.centroids_x = centroids_x
        self.centroids_y = centroids_y
        self.coordinate_accuracy = decimals
        self.data = data
        self.shape = dims
        
        return None
    
    
    
    
    ###########################
    #    SUPPORT FUNCTIONS    #
    ###########################
    
    def get_boundaries_from_centroids_delta(self,
        centroids: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        """
        Using a vector of centroids and a delta, get the axis boundaries for grid
        """
        # set grid boundaries
        adj = (
            -delta
            if (self.get_orientation(centroids) == "increasing")
            else delta
        )
        bounds = centroids.copy() + adj/2
        bounds = np.insert(bounds, len(bounds), bounds[-1] - adj)
        
        return bounds
    
    
    
    def get_delta(self,
        vec_centroids: np.ndarray,
        decimals: int,
    ) -> float:
        """
        Get the grid delta
        """

        delta = set(vec_centroids[1:] - vec_centroids[0:-1])
        delta = np.array(list(delta)).round(decimals = decimals)
        delta = list(set(delta))[0]
        delta = np.abs(delta)

        return delta
    
    
    
    def get_index_from_bound(self,
        bound: Union[float, int],
        axis: str,
    ) -> Union[int, None]:
        """
        Get the index in self.centroids_AXIS (AXIS = x or y) for the grid cell 
            containing the bound. If the bound is itself a grid cell boundary,
            returns the index of the upper- or left-most cell. 
        
        Returns None if an invalid type is entered for bound or -999 if a valid
            value for bound is entered but it falls outside the grid.
        
        Function Arguments
        ------------------
        - bound: bound to find cell index for
        - axis: "x" or "y"

        Keyword Arguments
        -----------------
        
        """
        
        # check input type
        if not sf.isnumber(bound):
            return None

        # get some vals
        bounds = self.bounds_x if (axis == "x") else self.bounds_y
        centroids = self.centroids_x if (axis == "x") else self.centroids_y
        delta = self.delta_x if (axis == "x") else self.delta_y
        orientation = self.orientation_x if (axis == "x") else self.orientation_y
        
        
        ##  CHECK SOME SPECIAL CASES
        
        # check for case when outside boundaries
        if (bound < bounds.min()) | (bound > bounds.max()):
            return -999
        
        # check for specification of actual boundary
        w = np.where(bound == bounds)[0]
        if len(w) != 0:
            ind = min(w)
            ind -= 1 if (ind == len(bounds) - 1) else 0
            return ind
        
        
        ##  GET BOX BOX 
         
        ind = (
            np.where(bound > bounds)[0][-1]
            if orientation == "increasing"
            else np.where(bound < bounds)[0][-1]
        )
        
        
        return ind
    
    
    
    def get_orientation(self,
        centroids: np.ndarray,
    ) -> str:
        """
        Determine if centroids increase or decrease
        """
        
        out = (
            "increasing"
            if (centroids[0] == centroids.min())
            else "decreasing"
        )
        
        return out






class GriddedDataset:
    """
    Group all data on the same grid into one dataset
    
    Initialization Arguments
    ------------------------
    - dict_datasets: dictionary mapping a string (name) to RioXArray containing
        gridded data
    - key_indexing_grid: string giving the key in dict_datasets to use for indexing
        regions. This grid is used to calculate dimensions, coordinates, and areas
        of cells.
        
    Optional Arguments
    ------------------
    
    """
    
    def __init__(self,
        dict_datasets: Dict[str, 'xarray.DataArray'],
        key_indexing_grid: str,
        logger: Union[logging.Logger, None] = None,
    ):
        
        self.logger = logger
        
        self._initialize_datasets(
            dict_datasets,
            key_indexing_grid,
        )
    
    
    
    
    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
    def _initialize_datasets(self,
        dict_datasets: Dict[str, 'xarray.DataArray'],    
        key_indexing_grid: str,
    ) -> None:
        """
        Initialize grid index. Sets the following properties
        
            * self.array_index
            * self.cell_areas
            * self.coords_x
            * self.coords_y
            * self.shape
            * Also assigns elements dict_datasets to properties
            
        NOTE: it is much faster to specify a numpy array here that is
            derived from a grid. An xarray.DataArray
            
        Function Arguments
        ------------------
        - dict_datasets: dictionary mapping a string (name) to RioXArray containing
            gridded data
        """
        
        ##  DATA VERIFICATION
        
        # 1. check that dictionary is specified
        if not isinstance(dict_datasets, dict):
            tp = str(type(dict_datasets))
            
            msg = f"""
            Error instantiating indexing GriddedDataset: invalid type '{tp}' entered 
            for dict_datasets. dict_datasets must be of type 'dict'
            """
            
            self._log(msg, type_log = "error")
            
            raise ValueError(msg)
            
        
        # 2. check that elements in dictionary are properly specified
        dict_datasets = dict(
            (k, v) for k, v in dict_datasets.items()
            if (
                isinstance(v, rx.raster_array.xarray.DataArray)
                & isinstance(k, str)
            )
            
        )
        
        if len(dict_datasets) == 0:
            msg = f"""
            Error instantiating indexing GriddedDataset: no valid entries found in 
                dict_datasets.
            """
            
            self._log(msg, type_log = "error")
            
            raise ValueError(msg)
            
            
        # 3. verify base key is present in dataset
        if key_indexing_grid not in dict_datasets.keys():
            tp = str(type(dict_datasets))
            
            msg = f"""
            Error instantiating indexing GriddedDataset: invalid type '{tp}' entered 
            for dict_datasets. dict_datasets must be of type 'dict'
            """
            
            self._log(msg, type_log = "error")
            
            raise ValueError(msg)
            
        
        ##  GET PROPERTIES
        
        grid_base = dict_datasets.get(key_indexing_grid)
        shape = grid_base.shape
        
        array_index = grid_base[0].to_numpy()
        coords_x = grid_base[0].x.to_numpy()
        coords_y = grid_base[0].y.to_numpy()
        
        cell_areas = gf.get_rioxarray_row_areas(grid_base)
        
            
        ##  SET PROPERTIES
        
        self.array_index = array_index
        self.cell_areas = cell_areas
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.shape = shape
        
        # set from data
        for k, v in dict_datasets.items():
            
            # set dataset
            setattr(self, str(k), v)
            
            if k == key_indexing_grid:
                continue
                
            # set array
            setattr(self, f"array_{k}", v[0].to_numpy())
            
        return None
        
    
    
    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See
            ?sf._optional_log for more information.

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments 
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)
        
        return None





class GridFeature:
    """
    Extract a feature
    
    Initialization Arguments
    ------------------------
    - grid: support_classes.Grid object to extract from
    - feature: value to extract
    """
    
    def __init__(self,
        grid: Grid,
        feature: Union[int, float, str],
    ):
        
        self._initialize_feature(grid, feature)
        
        
    
    
    def _initialize_feature(self,
        grid: Grid,
        feature: Union[int, float, str],
    ) -> None:
        """
        Initialize the feature index and some other information. Sets the
            following properties:
            
            self.feature
            self.feature_index
                NOTE: this index is oriented as (x, y), flipped from the numpy 
                array default of (row, col))
        """
        
        # initialize properties
        self.feature = None
        self.feature_index = None
        
        # check
        w = np.where(grid.data == feature)
        if len(w[0]) == 0:
            return None
        
        
        # modify feature specification - 
        self.feature = feature
        self.feature_index = (w[1], w[0])
        
        return None
        
        


    
    