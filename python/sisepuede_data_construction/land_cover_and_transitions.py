# IMPORTS

import geopandas as gpd
import itertools
import logging
import numpy as np
import os, os.path
import pandas as pd
import random
import re
import rioxarray as rx
import statistics
import sys
import time
from typing import *

dir_data_construction = os.path.dirname(os.path.realpath(__file__))
dir_python = os.path.dirname(dir_data_construction)
(
    sys.path.append(dir_python)
    if dir_python not in sys.path
    else None
)
import geo_classes as gc
import geo_functions as gf
import model_afolu as mafl
import model_attributes as ma
import regional_gridded_data as rgd
import setup_analysis as sa
import sisepuede_data_api as api
import support_classes as sc
import support_functions as sf



class LUCTransitionGenerator:
    """
    A class for generating land use transitions

    Copernicus data does not include information on mangroves. Use this module 
        to split out areas of mangroves from wetlands based on land use 
        prevalence
        
    
    Initialization Arguments
    ------------------------
    - classes_copernicus_forest_high_density: ESA Copernicus land use classes 
        associated with high density tree cover
    - classes_copernicus_forest_low_density: ESA Copernicus land use classes 
        associated with low density tree cover
    - dict_copernicus_to_sisepuede: dictionary mapping ES Copernicus land use
        classes to SISEPUEDE land use classes 
    - model_afolu: AFOLU model, used to access model_attributes and variables
    
    Optional Arguments
    ------------------
    - df_areas_base: areas of land use type; used for splitting mangroves from 
        wetlands (if specified). Should be a single row or series
    - field_array_0: field storing the category in t = 0 array
    - field_array_1: field storing the category in t = 1 array
    - field_area: field storing area
    - field_area_total_0: field storing the total area of the outbound category 
        (that associated with field_array_0)
    - field_area_total_0_ssp: field storing the total area of the outbound 
        SISEPUEDE category (that associated with field_category_i)
    - field_probability_transition: field storing estimated probability of 
        land use category transition
    - logger: optional logger object OR path to a log file
    - luc_copernicus_herbaceous_wetland: copernicus identifier for herbaceous 
        wetland, which stores mangroves as well
    - luc_copernicus_herbaceous_wetland_new: copernicus identifier used to 
        replace wetland to wetland transition (grassland is default)
    - luc_copernicus_mangroves_dummy: dummy number used to represent mangroves 
        in eventual conversion. If None, will auto-generate an index
    - regional_gridded_data: optional RegionalGriddedData dataset available to 
        use for setting region informatio and accessing key datasets
    - str_luc_dataset_prepend: dataset name prepended to year used as dataset
        base name, e.g., f"{str_luc_dataset_prepend}_{year}" would be entered
        as input to regional gridded data (datasets then have an additional 
        prepenage added)
        * NOTE: must contain the following fields: 
            - sc.regions.key
            - sc.TimePeriods.field_year
    - verify_df_areas_base: verify the structure of df_areas_base. If False, 
        will ignore failure to initialize and set self.df_areas_base = None
        (meaning that mangroves cannot be split from wetlands)
    """


    def __init__(self,
        classes_copernicus_forest_high_density: List[str],
        classes_copernicus_forest_low_density: List[str],
        dict_copernicus_to_sisepuede: dict,
        model_afolu: mafl.AFOLU,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_area_total_0_ssp: str = "area_ssp_i",
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
        logger: Union[str, None] = None,
        luc_copernicus_mangroves_dummy: Union[int, None] = None,
        luc_copernicus_herbaceous_wetland: int = 90,
        luc_copernicus_herbaceous_wetland_new: int = 30,
        regional_gridded_data: Union[rgd.RegionalGriddedData] = None,
        str_luc_dataset_prepend: str = "luc",
        verify_df_areas_base: bool = True,
    ):

        self._initialize_logger(logger,)

        self._initialize_general_properties(
            field_area = field_area,
            field_area_total_0 = field_area_total_0,
            field_area_total_0_ssp = field_area_total_0_ssp,
            field_array_0 = field_array_0,
            field_array_1 = field_array_1,
            field_category_i = field_category_i,
            field_category_j = field_category_j,
            field_probability_transition = field_probability_transition,
            str_luc_dataset_prepend = str_luc_dataset_prepend,
        )

        self._initialize_attributes(
            model_afolu,
        )

        self._initialize_lucs(
            classes_copernicus_forest_high_density,
            classes_copernicus_forest_low_density,
            dict_copernicus_to_sisepuede,
            luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy,
            luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
            luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new,
        )

        self._initialize_regional_data(
            df_areas_base = df_areas_base,
            regional_gridded_data = regional_gridded_data,
            stop_on_error = verify_df_areas_base,
        )


        return None
    


    ########################
    #    INITIALIZATION    #
    ########################

    def _initialize_areas_base(self,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
        region: Union[str, None] = None,
        stop_on_error: bool = True,
    ) -> None:
        """
        Initialize the data frame containing baseline areas for SISEPUEDE. 
            Verifies field names and the presence of the correct indices.

            * self.df_areas_base

        Function Arguments
        ------------------
        - df_areas_base: areas of land use type; used for splitting mangroves 
            from wetlands (if specified). Should be a single row or series. 
        - region: optional region to pass
        - stop_on_error: throw an error if the area data frame can't be 
            instantiated properly. Default is True since Coprernicus data may
            require df_areas_base to fill all categories.
        """

        # initialize and check input specification
        self.df_areas_base = None
        if not isinstance(df_areas_base, pd.DataFrame):
            return None


        ##  VERIFY DATAFRAME

        # try adding region or iso 
        df_areas = df_areas_base.copy()
        df_areas = self.regions.add_region_or_iso_field(df_areas)

        # establish required fields
        fields_req = [
            self.regions.key,
            self.time_periods.field_year
        ]
        fields_req += self.model_attributes.build_varlist(
            None,
            self.model_afolu.modvar_lndu_area_by_cat
        )

        if not set(fields_req).issubset(df_areas.columns):
            msg = sf.print_setdiff(fields_req, list(df_areas.columns))
            msg = f"Error instantiating df_areas_base in LUCTransitionGenerator: missing fields {msg}"
            self._log(msg, type_log = "warning")

            if stop_on_error:
                raise KeyError(msg)
            
            return None

        # filter on region
        df_areas = (
            (
                df_areas[
                    df_areas[self.regions.key].isin([region])
                ]
                .reset_index(drop = True)
            )
            if isinstance(region, str)
            else df_areas
        ) 

        # reduce data frame and set 
        df_areas = None if (len(df_areas) == 0) else df_areas[fields_req]

        
        ##  SET PROPERTIES

        self.df_areas_base = df_areas
        

        return None
        


    def _initialize_attributes(self,
        model_afolu: mafl.AFOLU,
    ) -> None:
        """
        Initialize attributes and models used throughout. Sets the following 
            properties:

            * self.model_afolu
            * self.model_attributes
            * self.regions
            * self.time_periods
        """

        if not isinstance(model_afolu, mafl.AFOLU):
            tp = str(type(model_afolu))
            msg = f"Invalid type '{tp}' found for model_afolu in LUCTransitionGenerator. model_afolu must be of type `mafl.AFOLU`"

            raise ValueError(msg)

        model_attributes = model_afolu.model_attributes
        attr_lndu = model_attributes.get_attribute_table(
            model_attributes.subsec_name_lndu
        )

        regions = sc.Regions(model_attributes)
        time_periods = sc.TimePeriods(model_attributes) 


        ##  SET PROPERTIES

        self.attr_lndu = attr_lndu
        self.model_afolu = model_afolu
        self.model_attributes = model_attributes
        self.regions = regions
        self.time_periods = time_periods

        return None



    def _initialize_general_properties(self,
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_area_total_0_ssp: str = "area_ssp_i",
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
        str_luc_dataset_prepend: str = "luc",
    ) -> None:
        """
        Initialize key properties, including the following values:

            * self.field_area
            * self.field_area_total_0
            * self.field_area_total_0_ssp
            * self.field_array_0
            * self.field_array_1
            * self.field_category_i
            * self.field_category_j
            * self.field_probability_transition
            * self.str_luc_dataset_prepend
        """


        ##  SET PROPERTIES

        self.field_area = field_area
        self.field_area_total_0 = field_area_total_0
        self.field_area_total_0_ssp = field_area_total_0_ssp
        self.field_array_0 = field_array_0
        self.field_array_1 = field_array_1
        self.field_category_i = field_category_i
        self.field_category_j = field_category_j
        self.field_probability_transition = field_probability_transition
        self.str_luc_dataset_prepend = str_luc_dataset_prepend

        return None
    


    def _initialize_logger(self,
        logger: Union[logging.Logger, str, None] = None,
        namespace: str = "land_cover_and_transitions.LUCTransitionGenerator",
    ) -> None:
        """
        Initialize the logger object. Sets the following properties:

            * self.logger
            

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - logger: logger object OR path to file to log
        - namespace: namespace for the logger
        """

        logger = (
            sf.setup_logger(
                fn_out = logger,
                namespace = namespace,
            )
            if isinstance(logger, str)
            else logger
        )

        self.logger = logger

        return None
    


    def _initialize_lucs(self,
        classes_copernicus_forest_high_density: List[str],
        classes_copernicus_forest_low_density: List[str],
        dict_copernicus_to_sisepuede: dict,
        luc_copernicus_herbaceous_wetland: int = 90,
        luc_copernicus_herbaceous_wetland_new: int = 30,
        luc_copernicus_mangroves_dummy: Union[int, None] = None,
    ) -> None:
        """
        Initialize attributes and models used throughout. Sets the following 
            properties:

            * self.classes_copernicus_forest_high_density
            * self.classes_copernicus_forest_low_density
            * self.dict_copernicus_to_sisepuede
            * self.luc_copernicus_herbaceous_wetland
            * self.luc_copernicus_herbaceous_wetland_new
            * self.luc_copernicus_mangroves_dummy
        """

        ##  CHECK INPUTS

        if not isinstance(dict_copernicus_to_sisepuede, dict):
            tp = str(type(dict_copernicus_to_sisepuede))
            msg = f"Invalid type '{tp}' found for dict_copernicus_to_sisepuede in LUCTransitionGenerator. model_afolu must be of type `dict`"

            self._log(msg, type_log = "error")
            raise ValueError(msg)

        if not sf.islistlike(classes_copernicus_forest_high_density):
            msg = f"Invalid specification of `classes_copernicus_forest_high_density` in LUCTransitionGenerator. Must be a list, iterator, or np.ndarray"
            raise ValueError(msg)

        if not sf.islistlike(classes_copernicus_forest_low_density):
            msg = f"Invalid specification of `classes_copernicus_forest_low_density` in LUCTransitionGenerator. Must be a list, iterator, or np.ndarray"
            raise ValueError(msg)


        # update dictionary
        dict_copernicus_to_sisepuede = dict(
            (k, v) for k, v in dict_copernicus_to_sisepuede.items()
            if (
                (v in self.attr_lndu.key_values)
                & isinstance(k, int)
                & (k not in classes_copernicus_forest_high_density) # drop forest categories 
                & (k not in classes_copernicus_forest_low_density)
            )
        )
        
        # verify that special LUCs are contained in the dictionary
        lucs_special = [
            luc_copernicus_herbaceous_wetland,
            luc_copernicus_herbaceous_wetland_new,
        ]

        for luc in lucs_special:
            if luc not in dict_copernicus_to_sisepuede.keys():
                msg = f"Invalid copernicus category {luc} in LUCTransitionGenerator: land use category not found in dict_copernicus_to_sisepuede."
                self._log(msg, type_log = "error")
                raise ValueError(msg) 

        # check specification of mangroves and update dictionary
        luc_copernicus_mangroves_dummy = (
            max(dict_copernicus_to_sisepuede.keys()) + 1
            if not isinstance(luc_copernicus_mangroves_dummy, int)
            else luc_copernicus_mangroves_dummy
        )
        dict_copernicus_to_sisepuede.update({
            luc_copernicus_mangroves_dummy: self.model_afolu.cat_lndu_fstm
        })
        

        ##  SET PROPERTIES

        self.classes_copernicus_forest_high_density = classes_copernicus_forest_high_density
        self.classes_copernicus_forest_low_density = classes_copernicus_forest_low_density
        self.dict_copernicus_to_sisepuede = dict_copernicus_to_sisepuede
        self.luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland
        self.luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new
        self.luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy

        return None
    


    def _initialize_regional_data(self,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
        regional_gridded_data: Union[rgd.RegionalGriddedData] = None,
        stop_on_error: bool = True,
    ) -> None:
        """
        Initialize any regional data. Sets the following properties:

            * self.df_areas_base
            * self.regex_luc_data
            * self.region
            * self.regional_gridded_data
            

        Function Arguments
        ------------------
        - df_areas_base: areas of land use type; used for splitting mangroves 
            from wetlands (if specified). Should be a single row or series. 
        - regional_gridded_data: optional RegionalGriddedData dataset available 
            to use for setting region informatio and accessing key datasets
        - stop_on_error: throw an error if the area data frame can't be 
            instantiated properly. Default is True since Coprernicus data may
            require df_areas_base to fill all categories.
        """

        # first, try to initialize the regional gridded dataset
        regional_gridded_data = (
            None
            if not isinstance(regional_gridded_data, rgd.RegionalGriddedData)
            else regional_gridded_data
        )

        # get dependent properties
        region = None if (regional_gridded_data is None) else regional_gridded_data.region
        regex_luc_data = self.get_regex_luc_dataset(regional_gridded_data)
        years_luc_available = self.get_available_luc_years(
            regional_gridded_data,
            regex_luc_data,
        )

        # next, pass this regional data to initialize areas_base
        self._initialize_areas_base(
            df_areas_base = df_areas_base,
            region = region,
            stop_on_error = stop_on_error,
        )


        ##  SET PROPERTIES

        self.regex_luc_data = regex_luc_data
        self.region = region
        self.regional_gridded_data = regional_gridded_data
        self.years_luc_available = years_luc_available

        return None

    

    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See ?sf._optional_log for more information

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



    def get_available_luc_years(self,
        regional_gridded_data: Union[rgd.RegionalGriddedData, None],
        regex_luc_data: re.Pattern, 
    ) -> Union[List[int], None]:
        """
        Using regional_gridded_data, try to get available LUC years

        Function Arguments
        ------------------
        - regional_gridded_data: regional gridded dataset to pass
        - regex_luc_data: regex used to match LUC data
        """

        if not isinstance(regional_gridded_data, rgd.RegionalGriddedData):
            return None

        years = [
            int(regex_luc_data.match(x).groups()[0]) 
            for x in regional_gridded_data.gridded_dataset.all_datasets
            if regex_luc_data.match(x) is not None
        ]

        years = (
            sorted(years)
            if len(years) > 0
            else None
        )

        return years


        
    def get_regex_luc_dataset(self,
        regional_gridded_data: Union[rgd.RegionalGriddedData] = None,
        dataset_prepend_default: str = "array",
    ) -> re.Pattern:
        """
        Get the regular expression for searching for available data

        Function Arguments
        ------------------
        - regional_gridded_data: optional RegionalGriddedData dataset available 
            to use for setting region informatio and accessing key datasets
        """

        regex_luc_data = (
            regional_gridded_data.gridded_dataset.str_prepend_array_dataset
            if isinstance(regional_gridded_data, rgd.RegionalGriddedData)
            else "array"
        )
        regex_luc_data = f"{regex_luc_data}_{self.str_luc_dataset_prepend}"
        regex_luc_data = regex = re.compile(f"{regex_luc_data}_(\d*$)")

        return regex_luc_data





    ###########################
    #    PRIMARY FUNCTIONS    #
    ###########################

    def build_transition_df(self,
        df_areas_base: Union[pd.DataFrame, None] = None,
        max_projection_devation_from_mean: Union[float, None] = None,
        regional_gridded_data: Union[rgd.RegionalGriddedData, None] = None,
        years_luc: Union[List[int], np.array, range, None] = None,
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """

        Function Arguments
        ------------------


        Keyword Arguments
        -----------------
        - df_areas_base: base areas (should include rows for each base year) used to 
            split wetlands into mangroves
        - max_projection_devation_from_mean: optional deviation from mean to specify
        - regional_gridded_data: RegionalTransitionMatrix containing GriddedData 
            divided into land use arrays. The arrays must be accessile using
            
            regional_gridded_data.get_regional_array(f"{dataset_prepend}_{year}")
        - years_luc: base years (output transition) to build dataset for
        **kwargs: passed to build_transition_df_for_year()
        """
        
        ##  INITIALIZATION AND CHECKS
        
        # some key checks
        years_luc = [years_luc] if isinstance(years_luc, int) else years_luc
        years_luc = self.years_luc_available if (years_luc is None) else years_luc
        regional_gridded_data = (
            self.regional_gridded_data
            if not isinstance(regional_gridded_data, rgd.RegionalGriddedData)
            else regional_gridded_data
        )

        # verify input types
        return_none = not isinstance(regional_gridded_data, rgd.RegionalGriddedData)
        return_none |= not sf.islistlike(years_luc)
        if return_none:
            return None
        
        # get some other key features 
        df_areas_region = self.get_areas_base(
            df = df_areas_base,
            region = regional_gridded_data.region,
        )
        split_mangroves = isinstance(df_areas_region, pd.DataFrame)
        years_luc = [x for x in self.years_luc_available if x in years_luc]

        # initialize key attributes
        n = self.attr_lndu.n_key_values
        field_region = self.regions.key
        field_year = self.time_periods.field_year

        
        ##  ITERATE OVER YEARS TO BUILD TRANSITION COMPONENTS
        
        args, kwa = sf.get_args(self.build_transition_df_for_year)
        kwargs_build_transition = dict((k, v) for k in kwargs if k in kwa)
        
        df_out = []

        for yr in years_luc[0:-1]:
            
            # filter area and try to generate the component
            df_area_cur = (
                df_areas_region[
                    df_areas_region[field_year].isin([yr])
                ]
                if split_mangroves
                else df_areas_region
            )
            
            try:
                df_component = self.build_transition_df_for_year(
                    yr,
                    df_areas_base = df_area_cur,
                    dict_fields_add = {
                        field_region: regional_gridded_data.region,
                        field_year: yr
                    },
                    regional_gridded_data = regional_gridded_data,
                    **kwargs_build_transition
                )
            
            except Exception as e:
                # errors here
                self._log(f"Error in build_transition_df: {e}", type_log = "error")
                continue
            
            df_out.append(df_component)

        df_out = (
            pd.concat(df_out, axis = 0)
            .reset_index(drop = True)
        )
        
        
        ##  PROJECT FORWARD ONE TIME STEP AND NORMALIZE
        
        df_transitions_new = self.model_attributes.get_standard_variables(
            df_out,
            self.model_afolu.modvar_lndu_prob_transition,
            return_type = "data_frame",
        )

        vec_new = sf.project_from_array(
            np.array(df_transitions_new),
            max_deviation_from_mean = 5*10**(-4),
        )
        
        df_transitions_new = pd.concat(
            [
                df_transitions_new,
                pd.DataFrame(
                    [vec_new], 
                    columns = df_transitions_new.columns,
                )
            ],
            axis = 0
        )
            
        # next, convert to transition matrices and clean/normalize rows
        mats = self.model_afolu.get_markov_matrices(
            df_transitions_new,
            get_emission_factors = False,
            n_tp = len(df_transitions_new),
        )

        mats = np.array(
            [sf.clean_row_stochastic_matrix(x) for x in mats]
        )

        global mm
        mm = mats

        
        ##  CONVERT BACK TO A DATA FRAME, THEN MERGE TO ALL TIME PERIODS AND FILL DOWN

        df_transitions_new = []
        years_new = []
        for i, mat in enumerate(mats):
            # skip entries that have all the mass on the diagonal
            if np.diag(mat).sum() == n:
                continue
                
            
            df = self.model_afolu.format_transition_matrix_as_input_dataframe(
                mat,
                exclude_time_period = True,
                field_key = "drop",
            )

            df_transitions_new.append(df)
            years_new.append(years_luc[i])

        # check length & return None if nothing is found--otherwise, build df
        if len(df_transitions_new) == 0:
            self._log(
                f"No valid transitions found in region '{regional_gridded_data.region}'. Skipping...",
                type_log = "warning"
            )
            return None

        df_out = pd.concat(df_transitions_new, axis = 0)
        df_out[field_region] = regional_gridded_data.region
        df_out[field_year] = years_new

        
        ##  INITIALIZE ALL YEARS AND MERGE IN

        df_all_years = (
            self.time_periods
            .get_time_period_df(include_year = True)
            .drop([self.time_periods.field_time_period], axis = 1)
        )
        df_all_years[field_region] = regional_gridded_data.region
        
        df_out = (
            pd.merge(
                df_all_years,
                df_out,
                how = "left",
            )
            .interpolate(method = "linear")
            .interpolate(method = "bfill")
            .interpolate(method = "ffill")
        )

        return df_out



    def build_transition_df_for_year(self,
        year: int,
        df_areas_base: Union[pd.DataFrame, None] = None,
        dict_fields_add: Union[dict, None] = None,
        regional_gridded_data: Union[rgd.RegionalGriddedData, None] = None,
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """
        Construct a dataframe from the land use change datasets from year `year` 
            to `year + 1`
            
        Returns None on errors or misspecification of input data (e.g., invalid 
            year or type)
        
        
        Function Arguments
        ------------------
        - year: base year (output transition)
        
        
        Keyword Arguments
        -----------------
        - dataset_prepend: dataset prependage in RegionalTransitionMatrix used 
            to access land use classification data
        - df_areas_base: areas of land use type; used for splitting mangroves 
            from wetlands (if specified). Should be a single row or series
        - dict_fields_add: optional fields to add to output dataframe. 
            Dictionary maps new column to value
        - regional_gridded_data: RegionalTransitionMatrix containing 
            GriddedData divided into land use arrays. The arrays must be 
            accessible using 
            regional_gridded_data.get_regional_array(f"{dataset_prepend}_{year}"

            If None, defaults to self.regional_gridded_data

        **kwargs: passed to self.add_data_frame_fields_from_dict
        """
        
        ##  CHECK INPUT TYPES
        regional_gridded_data = (
            self.regional_gridded_data
            if not isinstance(regional_gridded_data, rgd.RegionalGriddedData)
            else regional_gridded_data
        )
        if not isinstance(regional_gridded_data, rgd.RegionalGriddedData):
            return None
        
        arr_0 = self.get_luc_dataset(regional_gridded_data, year)
        arr_1 = self.get_luc_dataset(regional_gridded_data, year + 1)
        vec_areas = regional_gridded_data.get_cell_areas()

        if (arr_0 is None) | (arr_1 is None) | (vec_areas is None):
            return None
        
        df_areas_base = self.get_areas_base(
            df = df_areas_base,
            year = year,
        )
        

        ##  GET DATAFRAME AND MODIFY
        
        df = regional_gridded_data.get_transition_data_frame(
            arr_0,
            arr_1,
            vec_areas,
            field_area = self.field_area,
            field_area_total_0 = self.field_area_total_0,
            field_array_0 = self.field_array_0,
            field_array_1 = self.field_array_1,
            field_probability_transition = self.field_probability_transition,
            include_pij = False,
        )
        
        
        # call transition matrix
        df_transition = self.convert_transition_classes_to_sisepuede(
            df,
            df_areas_base = df_areas_base,
        )
        
        mat = self.model_afolu.get_transition_matrix_from_long_df(
            df_transition, 
            self.field_category_i,
            self.field_category_j,
            self.field_probability_transition,
        ) 

        global dtn
        dtn = df_transition

        df_out = self.model_afolu.format_transition_matrix_as_input_dataframe(
            mat,
        )
        
        if isinstance(dict_fields_add, dict):
            df_out = sf.add_data_frame_fields_from_dict(
                df_out,
                dict_fields_add,
                **kwargs
            )

        
        return df_out



    def convert_herbaceous_vegetation_to_related_class(self,
        df_transition: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Herbaceous wetlands (HW) includes a number of categories that are 
            included elsewhere in SISEPUEDE v1.0; in general, it likely 
            corresponds best with grassland. Notably, transitions into wetlands 
            tend to be the highest off-diagonal probabilities, which is likely 
            unrealistic. 
        
        To account for this, we assume that HWs are best accounted for by the 
            correspondence class, and we eliminate most dynamics. 
            
            
        Function Arguments
        ------------------
        - df_transition: data frame containing aggregated transitions with 
            Copernicus classes.
        
        Keyword Arguments
        -----------------
        """

        # set some properties
        cat_mangrove = self.model_afolu.cat_lndu_fstm
        dict_copernicus_to_sisepuede = self.dict_copernicus_to_sisepuede
        model_afolu = self.model_afolu
        model_attributes = self.model_attributes

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        
        # land use classes
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new

        
        # first, get dictionary mapping array_0 to area_luc_0
        dict_state_0_to_area_total = sf.build_dict(
            df_transition[[field_array_0, field_area_total_0]]
        )
        
        # initialize new columns
        vec_new_0 = list(df_transition[field_array_0])
        vec_new_1 = list(df_transition[field_array_1])
        vec_new_array_total = list(df_transition[field_area_total_0])
        
        # initialize list of output classes that wetlands have been converted to
        output_edges_converted_to_wetlands = []
        
        for i, row in df_transition.iterrows():
            
            state_0 = int(row[field_array_0])
            state_1 = int(row[field_array_1])
            states = [state_0, state_1]
            
            add_area = True
            area = float(row[field_area])
            
            if luc_copernicus_herbaceous_wetland not in states:
                continue
                
                
            # get new state 
            state = (
                luc_copernicus_herbaceous_wetland_new
                if state_0 == state_1
                else [x for x in states if x != luc_copernicus_herbaceous_wetland][0]
            )
                
            # if new state was previously the inbound class, we have to add the area to total for outbound
            dict_state_0_to_area_total[state] += (
                area 
                if ((state == state_1) | (state_0 == state_1)) 
                else 0.0
            )   
                
            vec_new_0[i] = state
            vec_new_1[i] = state
        

        # update data frames
        df_transition[field_array_0] = vec_new_0
        df_transition[field_array_1] = vec_new_1
        df_transition[field_area_total_0] = (
            df_transition[field_array_0]
            .replace(dict_state_0_to_area_total)
        )
        
        # re-aggregate
        df_transition = sf.simple_df_agg(
            df_transition,
            [field_array_0, field_array_1],
            {
                field_area_total_0: "first",
                field_area: "sum",
            }
        )
        
        return df_transition

        

    def convert_transition_classes_to_sisepuede(self,
        df_transition: pd.DataFrame,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Convert a transition data frame to SISEPUEDE classes. Contains 
        
        Function Arguments
        ------------------
        - df_transition: transition data frame (generated by 
            get_transition_data_frame()) that includes copernicus land use 
            classes
        
        Keyword Arguments
        -----------------
        - df_areas_base: areas of land use type; used for splitting mangroves 
            from wetlands (if specified). Should be a single row or series
        """

        ##  INITIALIZE SOME INTERNAL VARIABLES

        dict_copernicus_to_sisepuede = self.dict_copernicus_to_sisepuede
        model_afolu = self.model_afolu

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_area_total_0_ssp = self.field_area_total_0_ssp
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        field_category_i = self.field_category_i
        field_category_j = self.field_category_j
        field_probability_transition = self.field_probability_transition

        # land use classes
        classes_copernicus_forest_high_density = self.classes_copernicus_forest_high_density
        classes_copernicus_forest_low_density = self.classes_copernicus_forest_low_density
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new
        

        ##  PRE-PROCESS

        df_areas_base = self.get_areas_base(df = df_areas_base, )

        # split out mangroves before proceeding
        df_transition_out = self.split_mangroves_from_herbaceous_wetlands(
            df_transition,
            df_areas_base = df_areas_base,
        ) 
        
        # try converting wetlnds to new class (if specified)
        df_transition_out = (
            self.convert_herbaceous_vegetation_to_related_class(df_transition_out,)
            if isinstance(luc_copernicus_herbaceous_wetland_new, int)
            else df_transition_out
        )
        

        vec_sisepuede_0 = []
        vec_sisepuede_1 = []
        
        # iterate over rows
        for i, row in df_transition_out.iterrows():
            
            class_0 = int(row[field_array_0])
            class_1 = int(row[field_array_1])
            
            class_ssp_0 = dict_copernicus_to_sisepuede.get(class_0)
            class_ssp_1 = dict_copernicus_to_sisepuede.get(class_1)
            
            if (class_ssp_0 is not None) & (class_ssp_1 is not None):
                # update
                vec_sisepuede_0.append(class_ssp_0)
                vec_sisepuede_1.append(class_ssp_1)
            
            elif (class_ssp_0 is not None) and (class_ssp_1 is None):
                # case where a non-forest class transitions into a forest class
                vec_sisepuede_0.append(class_ssp_0)
                vec_sisepuede_1.append(model_afolu.cat_lndu_fsts)
            
            elif (class_ssp_0 is None) and (class_ssp_1 is not None):
                # case where a forest class transitions into a non-forest class
                (
                    vec_sisepuede_0.append(model_afolu.cat_lndu_fstp)
                    if class_0 in classes_copernicus_forest_high_density
                    else vec_sisepuede_0.append(model_afolu.cat_lndu_fsts)
                )
                vec_sisepuede_1.append(class_ssp_1)
            
            # next two cases occur where forests transition into each other
            elif class_0 in classes_copernicus_forest_high_density:
                
                # assume primary transitions into primary
                vec_sisepuede_0.append(model_afolu.cat_lndu_fstp)
                vec_sisepuede_1.append(model_afolu.cat_lndu_fstp)
            else: 

                vec_sisepuede_0.append(model_afolu.cat_lndu_fsts)
                (
                    vec_sisepuede_1.append(model_afolu.cat_lndu_fstp)
                    if class_1 in classes_copernicus_forest_high_density
                    else vec_sisepuede_1.append(model_afolu.cat_lndu_fsts)
                )
        
        # initialize output data frame
        df_out = df_transition_out.copy()
        df_out[field_category_i] = vec_sisepuede_0
        df_out[field_category_j] = vec_sisepuede_1
        
        # group areas and collapse
        df_area_0 = (
            sf.simple_df_agg(
                df_out[
                    [field_array_0, field_area_total_0, field_category_i]
                ]
                .drop_duplicates()
                .drop([field_array_0], axis = 1),
                [field_category_i],
                {
                    field_area_total_0: "sum"
                }
            )
            .rename(columns = {field_area_total_0: field_area_total_0_ssp})
        )
        
        # merge in areas
        df_out = pd.merge(
            df_out,
            df_area_0,
            how = "left"
        )
        
        # next, aggregate over types
        dict_agg = {
            field_area: "sum",
            field_area_total_0_ssp: "first"
        }
        df_out = sf.simple_df_agg(
            df_out[[field_category_i, field_category_j, field_area, field_area_total_0_ssp]],
            [field_category_i, field_category_j],
            dict_agg
        )
        
        # add probabilities
        vec_probs = np.array(df_out[field_area])/np.array(df_out[field_area_total_0_ssp])
        vec_probs = np.nan_to_num(vec_probs, nan = 1.0, posinf = 1.0)
        df_out[field_probability_transition] = vec_probs
        
        return df_out
    


    def get_areas_base(self,
        df: Union[pd.DataFrame, None] = None,
        region: Union[str, None] = None,
        series_from_single: bool = True,
        year: Union[int, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieve data frame of areas if available

        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - df: input data frame to search
        - region: region to retrieve data for (region name). If None, tries to
            retrieve from self.region
        - series_from_single: if only one row is found, return as a series?
        - year: optional specification of year to use for data generation
        """
        
        df = df.to_frame().transpose() if isinstance(df, pd.Series) else df
        df = self.df_areas_base if not isinstance(df, pd.DataFrame) else df
        if df is None:
            return None

        
        # if region/time period are not found, will take first row
        first_q = False

        # check if region is defined
        try:
            df_out = (
                df[df[self.regions.key].isin([region])]
                if isinstance(region, str)
                else df
            )
        except Exception as e:
            self._log(
                f"Error trying to retrieve region '{region}' in get_areas_base: {e}", 
                type_log = "error"
            )
            first_q = True

        # try getting time period
        try:
            df_out = (
                df_out[
                    df_out[self.time_periods.field_year].isin([year])
                ]
                if isinstance(year, int)
                else df_out
            )

        except Exception as e:
            self._log(
                f"Error trying to retrieve {self.time_periods.field_year} = {year} in get_areas_base: {e}", 
                type_log = "error"
            )
            first_q = True

        # reset the index and conver to None if no valid records are found
        df_out.reset_index(drop = True, inplace = True)
        df_out = None if (len(df_out) == 0) else df_out 
    
        if isinstance(df_out, pd.DataFrame):
            df_out = df_out.iloc[0:1] if first_q else df_out
            df_out = (
                df_out.iloc[0] 
                if (series_from_single & (len(df_out) == 1)) 
                else df_out
            )


        return df_out



    def get_luc_dataset(self,
        regional_gridded_data: rgd.RegionalGriddedData,
        year: int,
        delim: str = "_",
        return_name_only: bool = False,
    ) -> Union[np.ndarray, str, None]:
        """
        Get the Copernicus Land Use Classification dataset from regional 
            gridded data. 

        Returns None if invalid inputs are found.

        Function Arguments
        ------------------
        - regional_gridded_data: rgd.RegionalGriddedData storing the region's
            land use classification arrays
        - year: year to retrieve data for
        
        Keyword Arguments
        -----------------
        - return_name_only: set to true to return the name of the dataset 
            instead of the dataset itself
        """

        # checks on initialization
        return_none = not isinstance(regional_gridded_data, rgd.RegionalGriddedData)
        return_none |= not isinstance(year, int)
        if return_none:
            return None

        # build dataset by joining elements using the delimter
        delim = "_" if not isinstance(delim, str) else delim

        dataset_name = [
            regional_gridded_data.gridded_dataset.str_prepend_array_dataset,
            self.str_luc_dataset_prepend,
            str(year)
        ]

        dataset_name = delim.join(dataset_name)

        # return name only if desired; otherwise, get full dataset
        out = (
            dataset_name 
            if return_name_only
            else regional_gridded_data.get_regional_array(dataset_name)
        )

        return out
        


    def split_mangroves_from_herbaceous_wetlands(self,
        df_transition: pd.DataFrame,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Copernicus data does not include information on mangroves. Use this 
            function to split out areas of mangroves from wetlands based on land 
            use prevalence in the base time period.
        
        Function Arguments
        ------------------
        - df_transition: transition data frame (generated by 
            get_transition_data_frame()) that includes Copernicus land use 
            classes

        Keyword Arguments
        -----------------
        - df_areas_base: areas in base time period (0) used to allocate wetlands
        """
        
        ##  INITIALIZE PROPERTIES

        # set some properties
        cat_mangrove = self.model_afolu.cat_lndu_fstm
        dict_copernicus_to_sisepuede = self.dict_copernicus_to_sisepuede
        model_afolu = self.model_afolu
        model_attributes = self.model_attributes

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        field_category_i = self.field_category_i
        field_category_j = self.field_category_j
        field_probability_transition = self.field_probability_transition

        # land use classes
        luc_copernicus_mangroves_dummy = self.luc_copernicus_mangroves_dummy
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new


        ##  WETLAND AND MANGROVE SPLIT

        df_areas_base = self.get_areas_base(df = df_areas_base, )
        if not isinstance(df_areas_base, pd.Series):
            return df_transition
        
            
        ##  GET BASE FRACTIONS
        
        # set categories
        cat_mangrove = (
            model_afolu.cat_lnfu_fstm
            if cat_mangrove not in self.attr_lndu.key_values
            else cat_mangrove
        )
        
        cat_primary = dict_copernicus_to_sisepuede.get(luc_copernicus_herbaceous_wetland)
        cat_primary = (
            model_afolu.cat_lnfu_grass
            if cat_primary is None
            else cat_primary
        )

        fld_mng = model_attributes.build_varlist(
            None,
            model_afolu.modvar_lndu_area_by_cat,
            restrict_to_category_values = [cat_mangrove]
        )[0]
        fld_wet = model_attributes.build_varlist(
            None,
            model_afolu.modvar_lndu_area_by_cat,
            restrict_to_category_values = [cat_primary]
        )[0]

        area_mng = df_areas_base.get(fld_mng)
        area_wet = df_areas_base.get(fld_wet)

        denom = area_mng + area_wet
        frac_mng = np.nan_to_num(area_mng/denom, 0.0)
        frac_wet = np.nan_to_num(area_wet/denom, 0.0)

        dict_mult = {
            cat_mangrove: frac_mng,
            cat_primary: frac_wet,
        }

        
        ##  SPLIT AND CREATE NEW DFS
        
        # split wetlands out
        field_mult = "mult"
        df_out_a = df_transition[
            ~df_transition[field_array_0].isin([luc_copernicus_herbaceous_wetland])
            & ~df_transition[field_array_1].isin([luc_copernicus_herbaceous_wetland])
        ]
        df_out_b = df_transition[
            df_transition[field_array_0].isin([luc_copernicus_herbaceous_wetland])
            | df_transition[field_array_1].isin([luc_copernicus_herbaceous_wetland])
        ]

        
        # START CASES HERE
        # initialize new dataframe with cases that don't involve wetlands
        # then, go on case by case basis
        
        df_new = [df_out_a]
        
        
        ##  CASE 1: transitioning out of wetland into non-wetland
        
        df_out_b_1 = df_out_b[
            df_out_b[field_array_0].isin([luc_copernicus_herbaceous_wetland])
            & ~df_out_b[field_array_1].isin([luc_copernicus_herbaceous_wetland])
        ]
        
        # mangroves data frame
        df_out_b_1_mng = df_out_b_1.copy()
        df_out_b_1_mng[field_area] = np.array(df_out_b_1_mng[field_area])*frac_mng
        df_out_b_1_mng[field_area_total_0] = np.array(df_out_b_1_mng[field_area_total_0])*frac_mng
        df_out_b_1_mng[field_array_0] = luc_copernicus_mangroves_dummy

        # wetlands data frame
        df_out_b_1_wet = df_out_b_1.copy()
        df_out_b_1_wet[field_area] = np.array(df_out_b_1_wet[field_area])*frac_wet
        df_out_b_1_wet[field_area_total_0] = np.array(df_out_b_1_wet[field_area_total_0])*frac_wet
        
        df_new.extend([df_out_b_1_mng, df_out_b_1_wet])
        
        
        ##  CASE 2: transitioning out of non-wetland into wetland
            
        df_out_b_2 = df_out_b[
            ~df_out_b[field_array_0].isin([luc_copernicus_herbaceous_wetland])
            & df_out_b[field_array_1].isin([luc_copernicus_herbaceous_wetland])
        ]
        
        # mangroves data frame
        df_out_b_2_mng = df_out_b_2.copy()
        df_out_b_2_mng[field_area] = np.array(df_out_b_2_mng[field_area])*frac_mng
        df_out_b_2_mng[field_array_1] = luc_copernicus_mangroves_dummy
        
        # wetlands data frame
        df_out_b_2_wet = df_out_b_2.copy()
        df_out_b_2_wet[field_area] = np.array(df_out_b_2_wet[field_area])*frac_wet
        
        df_new.extend([df_out_b_2_mng, df_out_b_2_wet])

        
        ##  CASE 3: from wetland into wetland
        
        df_out_b_3 = df_out_b[
            df_out_b[field_array_0].isin([luc_copernicus_herbaceous_wetland])
            & df_out_b[field_array_1].isin([luc_copernicus_herbaceous_wetland])
        ]
        # use this fraction as an estimate of m -> m and w -> w
        frac_remaining = np.nan_to_num(
            float(df_out_b_3[field_area])/
            float(df_out_b_3[field_area_total_0]),
            0.0
        )
        
        # 3a. mangroves -> mangroves data frame
        df_out_b_3_mng_to_mng = df_out_b_3.copy()
        df_out_b_3_mng_to_mng[field_area] = np.array(df_out_b_3_mng_to_mng[field_area])*frac_mng*frac_remaining
        df_out_b_3_mng_to_mng[field_area_total_0] = np.array(df_out_b_3_mng_to_mng[field_area_total_0])*frac_mng
        df_out_b_3_mng_to_mng[field_array_0] = luc_copernicus_mangroves_dummy
        df_out_b_3_mng_to_mng[field_array_1] = luc_copernicus_mangroves_dummy
        
        # 3b. mangroves -> wetlands data frame
        df_out_b_3_mng_to_wet = df_out_b_3.copy()
        df_out_b_3_mng_to_wet[field_area] = np.array(df_out_b_3_mng_to_wet[field_area])*frac_mng*(1 - frac_remaining)
        df_out_b_3_mng_to_wet[field_area_total_0] = np.array(df_out_b_3_mng_to_wet[field_area_total_0])*frac_mng
        df_out_b_3_mng_to_wet[field_array_0] = luc_copernicus_mangroves_dummy
        
        # 3c. wetlands -> mangroves data frame
        df_out_b_3_wet_to_mng = df_out_b_3.copy()
        df_out_b_3_wet_to_mng[field_area] = np.array(df_out_b_3_wet_to_mng[field_area])*frac_wet*(1 - frac_remaining)
        df_out_b_3_wet_to_mng[field_area_total_0] = np.array(df_out_b_3_wet_to_mng[field_area_total_0])*frac_wet
        df_out_b_3_wet_to_mng[field_array_1] = luc_copernicus_mangroves_dummy

        # 3d. wetlands -> wetlands data frame
        df_out_b_3_wet_to_wet = df_out_b_3.copy()
        df_out_b_3_wet_to_wet[field_area] = np.array(df_out_b_3_wet_to_wet[field_area])*frac_wet*frac_remaining
        df_out_b_3_wet_to_wet[field_area_total_0] = np.array(df_out_b_3_wet_to_wet[field_area_total_0])*frac_wet
        
        df_new.extend(
            [
                df_out_b_3_mng_to_mng,
                df_out_b_3_mng_to_wet,
                df_out_b_3_wet_to_mng,
                df_out_b_3_wet_to_wet
            ]
        )
        
        # finally, concatenate
        df_new = (
            pd.concat(df_new, axis = 0)
            .sort_values(by = [field_array_0, field_array_1])
            .reset_index(drop = True)
        )
        
        return df_new




###########################
###                     ###
###    KEY FUNCTIONS    ###
###                     ###
###########################

def build_transition_dataframe(
    dataset: gc.GriddedDataset,
    all_isos_numeric: Union[List[int], np.ndarray],
    classes_copernicus_forest_high_density: List[str],
    classes_copernicus_forest_low_density: List[str],
    dict_copernicus_to_sisepuede: dict,
    model_afolu: mafl.AFOLU,
    df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
    logger: Union[logging.Logger, None] = None,
    luc_copernicus_herbaceous_wetland: int = 90,
    luc_copernicus_herbaceous_wetland_new: int = 30,
    regions: Union[sc.Regions, None] = None,
    time_periods: Union[sc.TimePeriods, None] = None,
) -> Union[pd.DataFrame, None]:
    """
    Build a DataFrame of all transitions for all available ISO Numeric codes.
    
    Function Arguments
    ------------------
    - dataset: GriddedData containing LUC data
    - all_isos_numeric: list-like object specifying region values to build data 
        for
    - classes_copernicus_forest_high_density: ESA Copernicus land use classes 
        associated with high density tree cover
    - classes_copernicus_forest_low_density: ESA Copernicus land use classes 
        associated with low density tree cover
    - dict_copernicus_to_sisepuede: dictionary mapping ES Copernicus land use
        classes to SISEPUEDE land use classes 
    - model_afolu: AFOLU model, used to access model_attributes and variables
    
    Keyword Arguments
    -----------------
    - df_areas_base: areas of land use type; used for splitting mangroves from 
        wetlands (if specified). Should be a single row or series
    - logger: optional logging object 
    - luc_copernicus_herbaceous_wetland: copernicus identifier for herbaceous 
        wetland, which stores mangroves as well
    - luc_copernicus_herbaceous_wetland_new: copernicus identifier used to 
        replace wetland to wetland transition (grassland is default)
    - regions: optional sc.Regions object to pass
    - time_periods: optional sc.TimePeriods object to pass
    """

    ##  INITIALIZATION AND CHECKS

    return_none = not sf.islistlike(all_isos_numeric)
    return_none |= not isinstance(dataset, gc.GriddedDataset)
    return_none |= not isinstance(model_afolu, mafl.AFOLU)
    if return_none:
        return None

    # some key classes for managing regions/time
    regions = (
        sc.Regions(model_afolu.model_attributes)
        if not isinstance(regions, sc.Regions)
        else regions
    )

    time_periods = (
        sc.TimePeriods(model_afolu.model_attributes)
        if not isinstance(time_periods, sc.TimePeriods)
        else time_periods
    )
    
    # build the transition generator
    transition_generator = LUCTransitionGenerator(
        classes_copernicus_forest_high_density,
        classes_copernicus_forest_low_density,
        dict_copernicus_to_sisepuede,
        model_afolu,
        df_areas_base = df_areas_base, # df_area
        logger = logger,
        luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
        luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new,
        regional_gridded_data = None,
    )
    
    # initialize output database and time
    df_out = []
    t0_outer = time.time()

    ##  ITERATE OVER INCLUDED ISOS

    
    for iso in all_isos_numeric:
        
        # time the region
        t0_inner = time.time()

        # log start
        region_name = regions.return_region_or_iso(iso, return_type = "region",)
        msg = f"Starting region '{region_name}'"
        transition_generator._log(msg, type_log = "info")

        try:
            # get gridded data and try to update the generator
            q_rgd = rgd.RegionalGriddedData(
                iso, 
                dataset,
                regions,
            )
            
            transition_generator._initialize_regional_data(
                df_areas_base = df_areas_base,
                regional_gridded_data = q_rgd,
            )
            
        except Exception as e:
            msg = f"Construction of regional gridded data for region '{iso}' failed: {e}"
            transition_generator._log(msg, type_log = "warning")
            continue
    
        
        # try building the data frame
        try: 
            df = transition_generator.build_transition_df()
            
        except Exception as e:
            msg = f"Construction of transition data frame for region '{iso}' failed: {e}"
            transition_generator._log(msg, type_log = "warning")
            continue
        
        # log the iteration
        t_elapsed_inner = sf.get_time_elapsed(t0_inner)
        t_elapsed_outer = sf.get_time_elapsed(t0_outer)
        msg = f"Completed region '{q_rgd.region}' in {t_elapsed_inner} seconds (total time elapsed = {t_elapsed_outer})"
        transition_generator._log(msg, type_log = "info")

        df_out.append(df)
    
    df_out = [x for x in df_out if (x is not None)]
    
    df_out = (
        (
            pd.concat(df_out, axis = 0)
            .reset_index(drop = True)
        )
        if len(df_out) > 0
        else None
    )
    
    # add ISO and drop region, interpolate a
    df_out = (
        regions.add_region_or_iso_field(
            df_out
        )
        .drop([regions.key], axis = 1)
        .dropna()
    )
    
    # sort fields
    fields_id = [regions.field_iso, time_periods.field_year]
    fields_dat = sorted([x for x in df_out.columns if x not in fields_id])
    df_out = df_out[fields_id + fields_dat]

    t_elapsed_outer = sf.get_time_elapsed(t0_outer)
    msg = f"Completed construction of transitions in {t_elapsed_outer} seconds"
    transition_generator._log(msg, type_log = "info")

    return df_out


   
    
