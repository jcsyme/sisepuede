import geo_classes as gc
import geo_functions as gf
import importlib
import itertools
import logging
import model_afolu as mafl
import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import random
import rioxarray as rx
import setup_analysis as sa
import sisepuede_data_api as api
import statistics
import support_classes as sc
import support_functions as sf
import time
import warnings
from attribute_table import AttributeTable
from typing import *



#####################
###               ###
###    CLASSES    ###
###               ###
#####################

class RegionalGriddedData:
    """
    Initialize a regional gridded dataset. Includes specialized operations for
        datasets within a region, including:

        * generating transition matrices from one dataset to the next

        
    Initialization Arguments
    ------------------------
    - region: region name, three digit ISO code, or numeric ISO code used 
        to initialize region and search indexing array
    - indexing_geo_array: indexing array with grid entries storing country
        isos (numeric). Used to extract data from other entries on the same
        grid. Can be a GriddedDataset
    - regions: sc.Regions support class used to access region information
    
    Optional Arguments
    ------------------

    """
    
    
    def __init__(self,
        region: Union[int, str],
        indexing_geo_array: Union['xarray.DataArray', np.ndarray, gc.GriddedDataset],
        regions: sc.Regions,
    ):
        
        self._initialize_region(
            region,
            regions,
        )
        
        self._initialize_grid_indexing_array(
            indexing_geo_array,
        )
        
        
        
    
        
    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
    def _initialize_grid_indexing_array(self,
        indexing_geo_array: Union['xarray.DataArray', np.ndarray],
    ) -> None:
        """
        Initialize grid index. Sets the following properties
        
            * self.gridded_dataset
            * self.region_grid_indices
            
        NOTE: it is much faster to specify a numpy array here that is
            derived from a grid. An xarray.DataArray
            
        Function Arguments
        ------------------
        - indexing_geo_array: indexing array with grid entries storing country
            isos (numeric). Used to extract data from other entries on the same
            grid
        """
        
        gridded_dataset = None
        
        # update indexing_geo_array based on input types to be a Numpy array
        if isinstance(indexing_geo_array, rx.raster_array.xarray.DataArray):
            indexing_geo_array = indexing_geo_array.to_numpy();
            indexing_geo_array = indexing_geo_array[0]
            
        if isinstance(indexing_geo_array, gc.GriddedDataset): 
            gridded_dataset = indexing_geo_array
            indexing_geo_array = indexing_geo_array.array_index

        # check final type and throw error if type of indexing_geo_array is invalid
        if not isinstance(indexing_geo_array, np.ndarray):
            tp = str(type(indexing_geo_array))
            
            msg = f"""
            Error instantiating indexing geo array in RegionalTransitionMatrix: invalid type
            '{tp}' entered for indexing_geo_array. The array must be of type xarray.DataArray
            or numpy.ndarray.
            """
            
            raise ValueError(msg)
            
        
        dims = indexing_geo_array.shape
        w = np.where(indexing_geo_array == self.iso_numeric);
        
        if len(w[0]) == 0:
            msg = f"""
            Error instantiating indexing geo array in RegionalTransitionMatrix: region 
            '{self.region}' not found in the indexing array. Transition matrices cannot be 
            calculated without indexing.
            """
            raise RuntimeError(msg)
        
        
        ##  SET PROPERTIES
        
        self.gridded_dataset = gridded_dataset
        self.region_grid_indices = w
        self.shape = dims
        
        return None
    
        
        
    def _initialize_region(self,
        region: Union[int, str],
        regions: sc.Regions,
    ) -> None:
        """
        Initialize regional characteristics. Sets the following properties:
        
            * self.region
            * self.iso
            * self.iso_numeric
        
        Function Arguments
        ------------------
        - region: region name, three digit ISO code, or numeric ISO code used 
            to initialize region and search indexing array
        - regions: sc.Regions support class used to access region information

        Keyword Arguments
        -----------------
        """
        
        # check region input 
        region_name = regions.return_region_or_iso(
            region,
            return_none_on_missing = True,
            return_type = "region",
            try_iso_numeric_as_string = True,
        )
        
        if region_name is None:
            msg = f"Error instantiating RegionalTransitionMatrix: region '{region}' not found."
            raise ValueError(msg)
    
        
        # continue by getting codes
        iso = regions.return_region_or_iso(
            region_name,
            return_type = "iso",
        )
        
        iso_numeric = regions.return_region_or_iso(
            region_name,
            return_type = "iso_numeric",
        )
        
        
        ##  SET PROPERTIES

        self.iso = iso
        self.iso_numeric = iso_numeric
        self.region = region_name
        
        return None
    
    
    
    ########################
    #    BASE FUNCTIONS    #
    ########################
    
    def get_regional_array_subset(self,
        array: np.ndarray,
    ) -> Union[np.ndarray, None]:
        """
        Retrieve region indices from `array` 
        """
        array_out = None
        
        if isinstance(array, np.ndarray):
            if len(array.dims) == 2:
                array_out = array[self.region_grid_indices]
        
        return array_out
    
    
    
    def get_regional_array(self,
        attr: str,
    ) -> Union[np.ndarray, None]:
        """
        Try to retrieve array associated with attribute attr
        """
        
        #
        arr = getattr(self.gridded_dataset, attr, None)
        if not isinstance(arr, np.ndarray):
            return None
        
        # if cell_areas, retrieve only one dimension
        inds = (
            self.region_grid_indices[0]
            if attr == "cell_areas"
            else self.region_grid_indices
        )
        
        arr_out = arr[inds]
        
        return arr_out
    


    def get_transition_data_frame(self,
        array_0: Union[np.ndarray, str],
        array_1: Union[np.ndarray, str],
        array_areas: Union[np.ndarray, None] = None,
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_probability_transition: str = "p_ij",
        include_pij: bool = True,
        return_aggregated: bool = True,
    ) -> Union[pd.DataFrame, None]:
        """
        Return a data frame that 
        
        Function Arguments
        ------------------
        - array_0: array at time t = 0 or string denoting GriddedDataset table to pull
        - array_1: array at time t = 1 or string denoting GriddedDataset table to pull
        - array_areas: array of areas of the cells or, if None, pulls cell areas for 
            transition matrix
        
        Keyword Arguments
        -----------------
        - field_array_0: field storing the category in t = 0 array
        - field_array_1: field storing the category in t = 1 array
        - field_area: field storing area
        - field_area_total_0: field storing the total area of the outbound category 
            (that associated with field_array_0)
        - field_probability_transition: field storing estimated probability of 
            land use category transition
        - include_pij: include the probability estimate?
        - return_aggregated: return transitions aggregated by edge type? If False, r
            returns a Data Frame with all transition grid cells
        """
        
        # check inputs 
        return_none = False
        
        array_0 = (
            self.get_regional_array(array_0)
            if isinstance(array_0, str)
            else array_0
        )
        
        array_1 = (
            self.get_regional_array(array_1)
            if isinstance(array_1, str)
            else array_1
        )
        
        array_areas = (
            self.get_regional_array("cell_areas")
            if array_areas is None
            else array_areas
        )
        
        return_none |= not isinstance(array_0, np.ndarray)
        return_none |= not isinstance(array_1, np.ndarray)
        return_none |= not isinstance(array_areas, np.ndarray)

        # if arrays are defined well, check they have the correct shape
        shapes = (
            set({
                array_0.shape,
                array_1.shape,
                array_areas.shape,
            })
            if not return_none
            else set({})
        )

        if (len(shapes) != 1) | return_none:
            return None
        
        
        
        ##  CONSTRUCT DATA FRAMES
        
        # long data frame of areas and land use types at time 0 and 1
        df = pd.DataFrame(
            {
                field_array_0: array_0,
                field_array_1: array_1,
                field_area: array_areas,
            }
        )
        
        # get the area at time 0 of each type
        df_area_0 = (
            sf.simple_df_agg(
                df.drop([field_array_1], axis = 1),
                [field_array_0],
                {field_area: "sum"}
            )
            .rename(columns = {field_area: field_area_total_0})
        )
        
        if not return_aggregated:
            return df
        
        # collapse to estimate transition probabilities
        df_collapsed = sf.simple_df_agg(
            df, 
            [field_array_0, field_array_1],
            {field_area: "sum"},
        )
        
        df_collapsed = pd.merge(
            df_collapsed, 
            df_area_0,
            how = "left",
        )
        
        if include_pij:
            vec_probs = np.array(df_collapsed[field_area])/np.array(df_collapsed[field_area_total_0])
            df_collapsed[field_probability_transition] = vec_probs
        
        return df_collapsed
            



