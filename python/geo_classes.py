###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY AND SUPPORT DATA PIPELINE
###

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
        self._initialize_grid(df_in)
        self._initialize_coords(decimals)
        
    
    
    
    #############################
    #    INITIALIZE ELEMENTS    #
    #############################
    
    def _initialize_coords(self,
        decimals: int,
        df_in: Union[pd.DataFrame, None] = None,
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
        - decimals: number of decimals to use for rounding to infer grid width

        Keyword Arguments
        -----------------
        - df_in: optional grid data frame to specify. If None, uses self.data
        """

        df_in = self.data if not isinstance(df_in, pd.DataFrame) else df_in


        ##  X AXIS
        
        x_centroids = np.array(df_in.columns)
        x_delta = self.get_delta(x_centroids, decimals)
        x_min = x_centroids.min() - x_delta/2
        x_max = x_centroids.max() + x_delta/2
        
        x_bounds = self.get_boundaries_from_centroids_delta(x_centroids, x_delta)
        x_orientation = self.get_orientation(x_centroids)
        
        
        ##  Y AXIS
        
        y_centroids = np.array(df_in.index).round(decimals = decimals)
        y_delta = self.get_delta(y_centroids, decimals)
        y_min = y_centroids.min() - y_delta/2
        y_max = y_centroids.max() + y_delta/2
        
        y_bounds = self.get_boundaries_from_centroids_delta(y_centroids, y_delta)
        y_orientation = self.get_orientation(y_centroids)
        
    
        ##  SET PROPERTIES
        
        self.bounds_x = x_bounds
        self.bounds_y = y_bounds
        self.centroids_x = x_centroids
        self.centroids_y = y_centroids
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
        data_in: Union[pd.DataFrame, str],
    ) -> None:
        """
        Initialize the grid using a file path or a grid data frame. Sets the
            following properties:
            
            self.data
            self.shape
        """
        # check types
        is_str = isinstance(data_in, str)
        is_df = isinstance(data_in, pd.DataFrame)
        if not (is_str | is_df):
            raise RuntimeError(f"Error in Grid: Invalid type '{type(data_in)}' of data_in specified. Must be str or pandas.DataFrame.")

        # if string, try reading as file
        if is_str:
            if not os.path.exists(data_in):
                raise RuntimeError(f"Error initializing Grid: path {data_in} not found.")

            rx_array = rx.open_rasterio(data_in)
            data_in = rx_array[0].to_pandas()   

        
        ##  SET PROPERTIES

        self.data = data_in
        self.shape = data_in.shape
        
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



    
    