# IMPORTS

import geopandas as gpd
import itertools
import logging
import numpy as np
import os, os.path
import pandas as pd
import random
import rioxarray as rx
import statistics
import sys
import time

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
    - model_attributes: ma.ModelAttributes object necessary for organizing 
        categories
    
    Optional Arguments
    ------------------
    - df_areas_base: areas of land use type; used for splitting mangroves from 
        wetlands (if specified). Should be a single row or series
    - dict_copernicus_to_sisepuede: dictionary mapping copernicus LUC to 
        SISEPUEDE LUC. Used to set primary target for herbaceous wetlands.
    - field_array_0: field storing the category in t = 0 array
    - field_array_1: field storing the category in t = 1 array
    - field_area: field storing area
    - field_area_total_0: field storing the total area of the outbound category 
        (that associated with field_array_0)
    - field_probability_transition: field storing estimated probability of 
        land use category transition
    - luc_copernicus_herbaceous_wetland: copernicus identifier for herbaceous 
        wetland, which stores mangroves as well
    - luc_copernicus_mangroves_dummy: dummy number used to represent mangroves 
        in eventual conversion. If None, will auto-generate an index
    - model_afolu: AFOLU model for property access
    - model_attrbutes: ModelAttributes object for variable and method access
    """


    def __init__(self,
        dict_copernicus_to_sisepuede: dict,
        model_afolu: mafl.AFOLU,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
        logger: Union[logging.Logger, None] = None,
        luc_copernicus_mangroves_dummy: Union[int, None] = None,
        luc_copernicus_herbaceous_wetland: int = 90,
        luc_copernicus_herbaceous_wetland_new: int = 30,
    ):

        self.logger = logger

        self._initialize_general_properties(
            field_area = field_area,
            field_area_total_0 = field_area_total_0,
            field_array_0 = field_array_0,
            field_array_1 = field_array_1,
            field_category_i = field_category_i,
            field_category_j = field_category_j,
            field_probability_transition = field_probability_transition,
        )
        self._initialize_attributes(
            model_afolu,
        )
        self._initialize_lucs(
            dict_copernicus_to_sisepuede,
            luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy,
            luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
            luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new,
        )

        return None
    


    ########################
    #    INITIALIZATION    #
    ########################

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

        attr_lndu = model_attributes.get_attribute_table(
            model_attributes.subsec_name_lndu
        )
        regions = sc.Regions(model_afolu.model_attributes)
        time_periods = sc.TimePeriods(model_afolu.model_attributes) 


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
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
    ) -> None:
        """
        Initialize key properties, including the following values:

            * self.field_area
            * self.field_area_total_0
            * self.field_array_0
            * self.field_array_1
            * self.field_category_i
            * self.field_category_j
            * self.field_probability_transition
        """


        ##  SET PROPERTIES

        self.field_area = field_area
        self.field_area_total_0 = field_area_total_0
        self.field_array_0 = field_array_0
        self.field_array_1 = field_array_1
        self.field_category_i = field_category_i
        self.field_category_j = field_category_j
        self.field_probability_transition = field_probability_transition

        return None
    


    def _initialize_lucs(self,
        dict_copernicus_to_sisepuede: dict,
        luc_copernicus_herbaceous_wetland: int = 90,
        luc_copernicus_herbaceous_wetland_new: int = 30,
        luc_copernicus_mangroves_dummy: Union[int, None] = None,
    ) -> None:
        """
        Initialize attributes and models used throughout. Sets the following 
            properties:

            * self.dict_copernicus_to_sisepuede
            * self.luc_copernicus_herbaceous_wetland
            * self.luc_copernicus_herbaceous_wetland_new
            * self.luc_copernicus_mangroves_dummy
        """

        if not isinstance(dict_copernicus_to_sisepuede, dict):
            tp = str(type(dict_copernicus_to_sisepuede))
            msg = f"Invalid type '{tp}' found for dict_copernicus_to_sisepuede in LUCTransitionGenerator. model_afolu must be of type `dict`"

            self._log(msg, type_log = "error")
            raise ValueError(msg)

        # update dictionary
        dict_copernicus_to_sisepuede = dict(
            (k, v) for k, v in dict_copernicus_to_sisepuede.items()
            if (v in attr_lndu.key_values)
            and isinstance(k, int)
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
            luc_copernicus_mangroves_dummy: model_afolu.cat_lndu_fstm
        })
        

        ##  SET PROPERTIES

        self.dict_copernicus_to_sisepuede = dict_copernicus_to_sisepuede
        self.luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland
        self.luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new
        self.luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy

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




    ###########################
    #    PRIMARY FUNCTIONS    #
    ###########################

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
        model_attrbutes = self.model_attributes

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        field_category_i = self.field_category_i
        field_category_i = self.field_category_j
        field_probability_transition = self.field_probability_transition

        # land use classes
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new


        ##  WETLAND AND MANGROVE SPLIT

        df_areas_base = (
            self.df_areas_base.iloc[0]
            if isinstance(df_areas_base, pd.DataFrame)
            else df_areas_base
        )
        
        
        if not isinstance(df_areas_base, pd.Series):
            return df_transition
        
        attr_lndu = sa.model_attributes.get_attribute_table(
            sa.model_attributes.subsec_name_lndu
        )
        
            

        ##  GET BASE FRACTIONS
        
        # set categories
        cat_mangrove = (
            model_afolu.cat_lnfu_fstm
            if cat_mangrove not in attr_lndu.key_values
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
        model_attrbutes = self.model_attributes

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

        

    def convert_transition_classes_to_sisepuede(
        df_transition: pd.DataFrame,
        classes_copernicus_forest_high_density: List[int] = classes_copernicus_forest_high_density,
        classes_copernicus_forest_low_density: List[int] = classes_copernicus_forest_low_density,
        df_areas_base: Union[pd.DataFrame, pd.Series, None] = None,
        dict_copernicus_to_sisepuede: dict = dict_copernicus_to_sisepuede,
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_area_total_0_ssp: str = "area_ssp_i",
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
        luc_copernicus_herbaceous_wetland: int = ind_herb_wetlands,
        luc_copernicus_herbaceous_wetland_new: Union[int, None] = ind_grassland,
        luc_copernicus_mangroves_dummy: int = ind_mangrove_dummy,
        model_afolu: mafl.AFOLU = model_afolu,
        model_attributes: ma.ModelAttributes = sa.model_attributes,
    ) -> Union[pd.DataFrame, None]:
        """
        Convert a transition data frame to SISEPUEDE classes. Contains 
        
        Function Arguments
        ------------------
        - df_transition: transition data frame (generated by get_transition_data_frame()) 
            that includes copernicus land use classes
        
        Keyword Arguments
        -----------------
        - classes_copernicus_forest_high_density: ESA Copernicus land use classes associated 
            with high density tree cover
        - classes_copernicus_forest_low_density: ESA Copernicus land use classes associated 
            with low density tree cover
        - df_areas_base: areas of land use type; used for splitting mangroves from 
            wetlands (if specified). Should be a single row or series
        - dict_copernicus_to_sisepuede: dictionary mapping
        - field_array_0: field storing the category in t = 0 array
        - field_array_1: field storing the category in t = 1 array
        - field_area: field storing area
        - field_area_total_0: field storing the total area of the outbound category 
            (that associated with field_array_0)
        - field_area_total_0_ssp: field storing the total area of the outbound SISEPUEDE
            category (that associated with field_category_i)
        - field_probability_transition: field storing estimated probability of 
            land use category transition
        - luc_copernicus_herbaceous_wetland: copernicus identifier for herbaceous wetland, 
            which stores mangroves as well
        - luc_copernicus_herbaceous_wetland_new: optional replacement for herbaceous wetland
        - luc_copernicus_mangroves_dummy: dummy number (must be in 
            dict_copernicus_to_sisepuede) used to represent mangroves in eventual 
            conversion
        - model_afolu: AFOLU model for property access
        - model_attrbutes: ModelAttributes object for variable and method access
        """
        
        # split out mangroves before proceeding
        df_transition_out = split_mangroves_from_herbaceous_wetlands(
            df_transition,
            df_areas_base = df_areas_base,
            dict_copernicus_to_sisepuede = dict_copernicus_to_sisepuede,
            field_array_0 = field_array_0,
            field_array_1 = field_array_1,
            field_area = field_area,
            field_area_total_0 = field_area_total_0,
            field_category_i = field_category_i,
            field_category_j = field_category_j,
            field_probability_transition = field_probability_transition,
            luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
            luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy,
            model_afolu = model_afolu,
            model_attributes = sa.model_attributes,
        ) 
        
        if isinstance(luc_copernicus_herbaceous_wetland_new, int):
            df_transition_out = convert_herbaceous_vegetation_to_related_class(
                df_transition_out,
                field_array_0 = field_array_0,
                field_array_1 = field_array_1,
                field_area = field_area,
                field_area_total_0 = field_area_total_0,
                luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
                luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new,
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
        df_out[field_probability_transition] = vec_probs
        
        return df_out



    def build_transition_df_for_year(
        transition_matrix: rgd.RegionalGriddedData,
        year: int,
        classes_copernicus_forest_high_density: List[int] = classes_copernicus_forest_high_density,
        classes_copernicus_forest_low_density: List[int] = classes_copernicus_forest_low_density,
        dataset_prepend: str = "array_luc",
        df_areas_base: Union[pd.DataFrame, None] = None,
        dict_copernicus_to_sisepuede: dict = dict_copernicus_to_sisepuede,
        dict_fields_add: Union[dict, None] = None,
        field_array_0: str = "array_0",
        field_array_1: str = "array_1",
        field_area: str = "area",
        field_area_total_0: str = "area_luc_0",
        field_area_total_0_ssp: str = "area_ssp_i",
        field_category_i: str = "cat_lndu_i",
        field_category_j: str = "cat_lndu_j",
        field_probability_transition: str = "p_ij",
        luc_copernicus_herbaceous_wetland: int = ind_herb_wetlands,
        luc_copernicus_herbaceous_wetland_new: Union[int, None] = ind_grassland,
        luc_copernicus_mangroves_dummy: int = ind_mangrove_dummy,
        model_afolu: mafl.AFOLU = model_afolu,
        model_attributes: ma.ModelAttributes = sa.model_attributes,
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """
        Construct a dataframe from the land use change datasets from year `year` to
            `year + 1`
            
        Returns None on errors or misspecification of input data (e.g., invalid year
            or type)
        
        
        Function Arguments
        ------------------
        - transition_matrix: RegionalTransitionMatrix containing GriddedData divided
            into land use arrays. The arrays must be accessile using
            transition_matrix.get_regional_array(f"{dataset_prepend}_{year}")
        - year: base year (output transition)
        
        
        Keyword Arguments
        -----------------
        - classes_copernicus_forest_high_density: ESA Copernicus land use classes associated 
            with high density tree cover
        - classes_copernicus_forest_low_density: ESA Copernicus land use classes associated 
            with low density tree cover
        - dataset_prepend: dataset prependage in RegionalTransitionMatrix used to 
            access land use classification data
        - df_areas_base: areas of land use type; used for splitting mangroves from 
            wetlands (if specified). Should be a single row or series
        - dict_copernicus_to_sisepuede: dictionary mapping
        - dict_fields_add: optional fields to add to output dataframe. Dictionary maps new
            column to value
        - field_array_0: field storing the category in t = 0 array
        - field_array_1: field storing the category in t = 1 array
        - field_area: field storing area
        - field_area_total_0: field storing the total area of the outbound category 
            (that associated with field_array_0)
        - field_area_total_0_ssp: field storing the total area of the outbound SISEPUEDE
            category (that associated with field_category_i)
        - field_probability_transition: field storing estimated probability of 
            land use category transition
        - luc_copernicus_herbaceous_wetland: copernicus identifier for herbaceous wetland, 
            which stores mangroves as well
        - luc_copernicus_herbaceous_wetland_new: optional replacement for herbaceous wetland
        - luc_copernicus_mangroves_dummy: dummy number (must be in 
            dict_copernicus_to_sisepuede) used to represent mangroves in eventual 
            conversion
        - model_afolu: AFOLU model for property access
        - model_attrbutes: ModelAttributes object for variable and method access
        **kwargs: passed to self.add_data_frame_fields_from_dict
        """
        
        
        ##  CHECKS AND INITIALIZATION
        
        if not isinstance(transition_matrix, rgd.RegionalGriddedData):
            return None
        
        arr_0 = transition_matrix.get_regional_array(f"{dataset_prepend}_{year}")
        arr_1 = transition_matrix.get_regional_array(f"{dataset_prepend}_{year + 1}")
        vec_areas = transition_matrix.get_regional_array("cell_areas")
        
        if (arr_0 is None) | (arr_1 is None) | (vec_areas is None):
            return None
        
        
        ##  GET DATAFRAME AND MODIFY
        
        df = transition_matrix.get_transition_data_frame(
            arr_0,
            arr_1,
            vec_areas,
            include_pij = False,
        )
        
        
        # call transition matrix
        df_transition = convert_transition_classes_to_sisepuede(
            df,
            df_areas_base = df_areas_base,
            classes_copernicus_forest_high_density = classes_copernicus_forest_high_density,
            classes_copernicus_forest_low_density = classes_copernicus_forest_low_density,
            dict_copernicus_to_sisepuede = dict_copernicus_to_sisepuede,
            field_array_0 = field_array_0,
            field_array_1 = field_array_1,
            field_area = field_area,
            field_area_total_0 = field_area_total_0,
            field_category_i = field_category_i,
            field_category_j = field_category_j,
            field_probability_transition = field_probability_transition,
            luc_copernicus_herbaceous_wetland = luc_copernicus_herbaceous_wetland,
            luc_copernicus_herbaceous_wetland_new = luc_copernicus_herbaceous_wetland_new,
            luc_copernicus_mangroves_dummy = luc_copernicus_mangroves_dummy,
            model_afolu = model_afolu,
            model_attributes = model_attributes,
        )
        
        mat = model_afolu.get_transition_matrix_from_long_df(
            df_transition, 
            field_category_i,
            field_category_j,
            field_probability_transition,
        ) 

        df_out = model_afolu.format_transition_matrix_as_input_dataframe(
            mat,
        )
        
        if isinstance(dict_fields_add, dict):

            df_out = sf.add_data_frame_fields_from_dict(
                df_out,
                dict_fields_add,
                **kwargs
            )
            
        
        return df_out
