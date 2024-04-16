# imports
import geopandas as gpd
import numpy as np
import os, os.path
import pandas as pd
import support_classes as sc
import support_functions as sf
import time
import warnings
from typing import *




###########################
#    PRIMARY FUNCTIONS    #
###########################

def check_xarray_grid_equivalence(
    xarray_1: 'xarray.DataArray',
    xarray_2: 'xarray.DataArray',
    round_equality: int = 6,
) -> bool:
    """
    Check whether or not two gridded datasets are equivalent, or, probably 
        equivalent.
        
    Function Arguments
    ------------------
    - xarray_1: first rio xarray to check
    - xarray_2: second to compare
    
    Keyword Arguments
    -----------------
    - round_equality: number of digits to round lat/lon to to check bounds
    """
    
    b1 = np.round(xarray_1.rio.bounds(), decimals = round_equality)
    b2 = np.round(xarray_2.rio.bounds(), decimals = round_equality)
    
    # conditions for equality
    equivalent = all(b1 == b2)
    equivalent &= (xarray_1.rio.height == xarray_2.rio.height)
    equivalent &= (xarray_1.rio.width == xarray_2.rio.width)
    
    return equivalent



def cell_area_from_grid(
    lat_0: float,
    lat_1: float,
    lon_0: float,
    lon_1: float,
    angle_type: str = "degree",
    radius: float =  6371.0072, # km
) -> float:
    """
    Calculate the area of a grid cell based on lat/lon bounding points.

    https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9671.2010.01200.x
    
    Function Arguments
    ------------------
    - lat_0: lower latitude
    - lat_1: higher latitude
    - lon_0: lower longitude
    - lon_1: higher longitude
    
    Keyword Arguments
    -----------------
    - angle_type: "degree" or "radian"
    - radius: authlalic radius (in units of interest) of earth. 
        * source: https://en.wikipedia.org/wiki/Earth_radius
    """
    
    # scale 
    scalar = np.pi/180.0 if (angle_type == "degree") else 1.0
    lat_0 *= scalar
    lat_1 *= scalar
    lon_0 *= scalar
    lon_1 *= scalar

    # implement formula for generic grid area (derived from surface integral)
    area = (radius**2)*(lon_1 - lon_0)*(np.sin(lat_1) - np.sin(lat_0))
    
    return area



def get_low_res_indices_for_higher_res(
    grid_low: 'geo_classes.Grid',
    grid_high: 'geo_classes.Grid',
) -> Union[pd.DataFrame, Tuple[np.array]]:
    """
    Map gridded elements from a higher resolution data frame to a lower 
        resolution data frame
        
    Function Arguments
    ------------------
    - grid_low: support_classes.Grid (row indexed by latitude, columns are
        longitude) containing gridded data at a lower resolution
    - grid_high: support_classes.Grid (row indexed by latitude, columns are 
        longitude) containing gridded data at a higher resolution
    
    Keyword Arguments
    -----------------
    """
    
    ##  INITIALIZATION

    # check orientations
    return_none = (grid_low.orientation_x != grid_high.orientation_x)
    return_none |= (grid_low.orientation_y != grid_low.orientation_y)
    if return_none:
        return None

    
    # get bounds ands starting indices
    dict_bounds_inds = get_shared_bounds_and_indices(grid_high, grid_low)
    x_min, x_max, y_min, y_max = dict_bounds_inds.get("bounds")
    ind_x_high, ind_x_low, ind_y_high, ind_y_low = dict_bounds_inds.get("inds")
    
    
    ##  ITERATE OVER HIGH RESOLUTION INDICES
    
    inds_low_x_by_high_res = np.zeros(len(grid_high.bounds_x))
    inds_low_y_by_high_res = np.zeros(len(grid_high.bounds_y))
    
    # get indices, for each centroid in the high resolution grid, of closes low-res match (x-axis)
    inds_low_x_by_high_res = iterate_high_to_low(
        grid_high.centroids_x,
        grid_low.centroids_x,
        grid_high.delta_x,
        grid_low.delta_x,
        ind_x_high,
        ind_x_low,
        grid_high.orientation_x,
    )
    # for y-axis
    inds_low_y_by_high_res = iterate_high_to_low(
        grid_high.centroids_y,
        grid_low.centroids_y,
        grid_high.delta_y,
        grid_low.delta_y,
        ind_y_high,
        ind_y_low,
        grid_high.orientation_y,
    )
    
    # return indices
    tup_out = inds_low_x_by_high_res, inds_low_y_by_high_res
    
    return tup_out



def get_overlay_bounds(
    grid_1: 'geo_classes.Grid',
    grid_2: 'geo_classes.Grid',
) -> Tuple:
    """
    Return bounds to iterate over from two grids
    
        (x_min, x_max, y_min, y_max)
    """
    
    x_max = min(grid_1.x_max, grid_2.x_max)
    x_min = max(grid_1.x_min, grid_2.x_min)

    y_max = min(grid_1.y_max, grid_2.y_max)
    y_min = max(grid_1.y_min, grid_2.y_min)
    
    tup_out = (x_min, x_max, y_min, y_max)

    return tup_out

    
    
def get_rioxarray_row_areas(
    rx_array: 'xarray.DataArray',
    decimals: int = 8,
) -> np.ndarray:
    """
    Using the rx_array, generate areas of each grid cell by row. Note, in a
        regular grid, the areas are the same for each row (latitude band)
        
    Function Arguments
    ------------------
    - rx_array: input RioXArray containing gridded information (from NetCDF or 
        GeoTIF)
    
    Keyword Arguments
    -----------------
    - decimals: number of digits to use to determine degree width
    """
    # get widths - y
    y = rx_array.y.to_numpy()
    delta_y = np.unique(np.round(y[1:] - y[0:-1], decimals = decimals))
    bounds_y = np.append(y - delta_y/2, y[-1] + delta_y/2)

    # x
    x = rx_array.x.to_numpy()
    delta_x = np.unique(np.round(x[1:] - x[0:-1], decimals = decimals))
    bounds_x = np.append(x - delta_x/2, x[-1] + delta_x/2)

    # areas on every row are the same; only need to iterate over rows (latitude) since grid is uniform
    area_by_row = np.array(
        [
            cell_area_from_grid(
                bounds_y[i],
                bounds_y[i - 1],
                bounds_x[0],
                bounds_x[1]
            )
            for i in range(1, len(bounds_y))
        ]
    )
        
    return area_by_row



def get_shared_bounds_and_indices(
    grid_1: 'geo_classes.Grid',
    grid_2: 'geo_classes.Grid',
) -> Dict[str, Tuple]:
    """
    For two grids, determine minimal boundaries within the range of
        both grids. Returns a dictionary with tuples:
        
        dict_out["bounds"] = x_min, x_max, y_min, y_max
        dict_out["inds"] = ind_x_1, ind_x_2, ind_y_1, ind_y_2
    """
    # initialize output dictionary
    dict_out = {}
    
    # get overlay bounds and update dict
    bounds = get_overlay_bounds(grid_1, grid_2)
    x_min, x_max, y_min, y_max = bounds
    dict_out.update({"bounds": bounds})
    
    # get starting indices - 1 and 2 will have same orientation
    ind_x_1 = (
        grid_1.get_index_from_bound(x_min, "x")
        if grid_1.orientation_x == "increasing"
        else grid_1.get_index_from_bound(x_max, "x")
    )
    
    ind_x_2 = (
        grid_2.get_index_from_bound(x_min, "x")
        if grid_2.orientation_x == "increasing"
        else grid_2.get_index_from_bound(x_max, "x")
    )
    
    ind_y_1 = (
        grid_1.get_index_from_bound(y_min, "y")
        if grid_1.orientation_y == "increasing"
        else grid_1.get_index_from_bound(y_max, "y")
    )
    
    ind_y_2 = (
        grid_2.get_index_from_bound(y_min, "y")
        if grid_2.orientation_y == "increasing"
        else grid_2.get_index_from_bound(y_max, "y")
    )
    
    tup_inds = (ind_x_1, ind_x_2, ind_y_1, ind_y_2)
    dict_out.update({"inds": tup_inds})
    
    
    return dict_out


    
def iterate_high_to_low(
    centroids_high_res: np.ndarray,
    centroids_low_res: np.ndarray,
    delta_high: float,
    delta_low: float,
    ind_0_high: int,
    ind_0_low: int,
    orientation: str,
) -> np.ndarray:
    """
    Map elements in the lower-resolution grid to the higher-resolution 
        grid. Returns a numpy array with the length of centroids_high_res,
        with each element the index in the associated axis of the lower 
        resolution grid to use.

    Function Arguments
    ------------------
    - centroids_high_res: axis centroids for the higher-resolution grid
    - centroids_low_res: axis centroids for the lower-resolution grid
    - delta_high: grid square width for higher-resolution grid
    - delta_low: grid square width for lower-resolution grid
    - ind_0_high: starting index for the higher-resolution grid
    - ind_0_low: starting index for the low-resolution grid
    - orientation: "increasing" or "decreasing". High and res grid must 
        have same orientation
    """

    inds_low_x_by_high_res = -np.ones(len(centroids_high_res)).astype(int)

    # set sign for delta
    dec_q = (orientation == "decreasing")

    # start by iterating over the 
    i = ind_0_high
    j = ind_0_low

    while (i < len(centroids_high_res)) & (j < len(centroids_low_res)):

        d_to_lr_cur = np.abs(centroids_high_res[i] - centroids_low_res[j])
        add = 1 if (j < len(centroids_low_res) - 1) else 0
        d_to_lr_next = np.abs(centroids_high_res[i] - centroids_low_res[j + add])

        # if closest to current cell in low-res, keep; otherwise, move to next
        j += 0 if (d_to_lr_cur <= d_to_lr_next) else 1

        # assign output
        inds_low_x_by_high_res[i] = j

        i += 1

    return inds_low_x_by_high_res


