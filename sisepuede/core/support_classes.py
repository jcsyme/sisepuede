###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY AND SUPPORT DATA PIPELINE
###

import geopy.distance
import numpy as np
import pathlib
import pandas as pd
import re


import sisepuede.utilities._toolbox as sf
from typing import *


##  SOME ERROR CLASSES

class InvalidRegion(Exception):
    pass

class InvalidRegionGroup(Exception):
    pass

class InvalidTimePeriod(Exception):
    pass

class YAMLConfigurationKeyError(Exception):
    pass



##########################
#    GLOBAL VARIABLES    #
##########################

# fields
_FIELD_DAY = "day"
_FIELD_MONTH = "month"
_FIELD_YEAR = "year"


# module UUID
_MODULE_UUID = "6B7410BF-A491-42F9-B904-5AB526C74180"

# return types
_RETURN_TYPE_ISO = "iso"
_RETURN_TYPE_ISO_NUMERIC = "iso_numeric"
_RETURN_TYPE_REGION = "region"




##################################
#    INITIALIZATION FUNCTIONS    #
##################################


class Regions:
    """Leverage some simple region actions based on model attributes. Supports
        the following actions for data:

        * Aggregation by World Bank global region
        * Finding the closest region (by population centroid)
        * Shared replacement dictionaries (IEA/WB/UN)
        * And more

    The Regions class is designed to provide convenient support for batch 
        integration of global and regional datasets into the SISEPUEDE 
        framework.


    Initialization Arguments
    ------------------------
    model_attributes : ModelAttributes
        ModelAttributes object used to coordinate attributes and variables

    Optional Arguments
    ------------------
    regex_region_groups : Union[re.Pattern, None]
        Optional regular expression used to identify region groups in the 
        attribute table
    """

    def __init__(self,
        model_attributes: 'ModelAttributes',
        regex_region_groups: Union[re.Pattern, None] = re.compile("(.*)_region$"),
    ):

        self._initialize_region_properties(
            model_attributes,
            regex_region_groups = regex_region_groups,
        )

        # initialize some default data source properties
        self._initialize_defaults_iea()
        self._initialize_generic_dict()
        self._initialize_uuid()

        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_defaults_iea(self,
    ) -> None:
        """
        Sets the following default properties, associated with fields in IEA
            data tables:

            * self.dict_iea_countries_lc_to_regions
            * self.field_iea_balance
            * self.field_iea_country
            * self.field_iea_product
            * self.field_iea_time
            * self.field_iea_unit
            * self.field_iea_value
        """

        self.dict_iea_countries_lc_to_regions = {
            "chinese_taipei": "taiwan",
            "czech_republic": "czechia",
            "hong_kong_(china)": "hong_kong",
            "korea": "republic_of_korea",
            "people's_republic_of_china": "china",
            "republic_of_north_macedonia": "north_macedonia",
            "republic_of_turkiye": "turkey",
            "republic_of_türkiye": "turkey",
            "slovak_republic": "slovakia",
            "united_states": "united_states_of_america",
        }

        self.field_iea_balance = "Balance"
        self.field_iea_country = "Country"
        self.field_iea_product = "Product"
        self.field_iea_time = "Time"
        self.field_iea_unit = "Unit"
        self.field_iea_value = "Value"     

        return None



    def _initialize_generic_dict(self,
    ) -> None:
        """
        Sets the following default properties, associated with different generic
            country specifications:

            * self.dict_generic_countries_to_regions
        """

        dict_generic_countries_to_regions = {
            "laos": "lao",
            "luxemburg": "luxembourg",
            "swaziland": "eswatini",
            "uk": "united_kingdom",
            "usa": "united_states_of_america",
            "us": "united_states_of_america",
            "vietnam": "viet_nam"
        }

        self.dict_generic_countries_to_regions = dict_generic_countries_to_regions

        return None

        

    def _initialize_region_properties(self,
        model_attributes: 'ModelAttributes',
        regex_region_groups: Union[re.Pattern, None] = re.compile("(.*)_region$"),
    ) -> None:
        """
        Set the following properties:

            * self.all_isos
            * self.all_isos_numeric
            * self.all_region
            * self.all_wb_regions
            * self.attributes
            * self.dict_iso_to_region
            * self.dict_iso_numeric_to_region
            * self.dict_region_to_iso
            * self.dict_region_to_iso_numeric
            * self.dict_region_to_wb_region
            * self.dict_wb_region_to_region
            * self.field_iso
            * self.field_iso_two
            * self.field_iso_numeric
            * self.field_lat
            * self.field_lon
            * self.field_wb_global_region
            * self.key
            * self.regex_superregion
            * self.valid_region_groups
            * self.valid_return_types


        """
        # some fields
        field_iso = "iso_alpha_3"
        field_iso_two = "iso_alpha_2"
        field_iso_numeric = "iso_numeric"
        field_lat = "latitude_population_centroid_2020"
        field_lon = "longitude_population_centroid_2020"
        field_un_global_region = "un_region"
        field_wb_global_region = "world_bank_global_region"

        # set attributes and some dictionaries
        attributes = model_attributes.get_other_attribute_table("region")

        # initialize ISO dictionaries
        dict_iso_to_region = attributes.field_maps.get(f"{field_iso}_to_{attributes.key}")
        dict_iso_numeric_to_region = attributes.field_maps.get(f"{field_iso_numeric}_to_{attributes.key}")
        dict_region_to_iso = attributes.field_maps.get(f"{attributes.key}_to_{field_iso}")
        dict_region_to_iso_numeric = attributes.field_maps.get(f"{attributes.key}_to_{field_iso_numeric}")
    
        # check numeric codes
        dict_iso_numeric_to_region = dict((int(k), v) for k, v in dict_iso_numeric_to_region.items())
        dict_region_to_iso_numeric = dict((k, int(v)) for k, v in dict_region_to_iso_numeric.items())

        # set up some sets
        all_isos = sorted(list(dict_iso_to_region.keys()))
        all_isos_numeric = sorted(list(dict_iso_numeric_to_region.keys()))


        ##  GET VALID REGION GROUPS

        region_groupings = [
            regex_region_groups.match(x).groups()[0] 
            for x in attributes.table.columns
            if regex_region_groups.match(x) is not None
        ]


        ##  OTHER REGIONAL CODES

        # WorldBank region dictionaries
        dict_wb_region_to_region = sf.group_df_as_dict(
            attributes.table,
            [field_wb_global_region],
            fields_out_set = attributes.key
        )
        dict_region_to_wb_region = attributes.field_maps.get(f"{attributes.key}_to_{field_wb_global_region}")
        all_wb_regions = sorted(list(dict_wb_region_to_region.keys()))

        # UN regions
        dict_un_region_to_region = sf.group_df_as_dict(
            attributes.table,
            [field_un_global_region],
            fields_out_set = attributes.key
        )
        dict_region_to_un_region = attributes.field_maps.get(f"{attributes.key}_to_{field_un_global_region}")
        all_un_regions = sorted(list(dict_region_to_un_region.keys()))

        
        ##  SET VALID RETURN TYPES

        valid_return_types = [
            _RETURN_TYPE_ISO,
            _RETURN_TYPE_ISO_NUMERIC,
            _RETURN_TYPE_REGION
        ]

        ##  SET PROPERTIES

        self.all_isos = all_isos
        self.all_isos_numeric = all_isos_numeric
        self.all_regions = attributes.key_values
        self.all_un_regions = all_un_regions
        self.all_wb_regions = all_wb_regions
        self.attributes = attributes
        self.dict_iso_to_region = dict_iso_to_region
        self.dict_iso_numeric_to_region = dict_iso_numeric_to_region
        self.dict_region_to_iso = dict_region_to_iso
        self.dict_region_to_iso_numeric = dict_region_to_iso_numeric
        self.dict_region_to_un_region = dict_region_to_un_region
        self.dict_region_to_wb_region = dict_region_to_wb_region
        self.dict_un_region_to_region = dict_un_region_to_region
        self.dict_wb_region_to_region = dict_wb_region_to_region
        self.field_iso = field_iso
        self.field_iso_two = field_iso_two
        self.field_iso_numeric = field_iso_numeric
        self.field_lat = field_lat
        self.field_lon = field_lon
        self.field_wb_global_region = field_wb_global_region
        self.key = attributes.key
        self.regex_region_groups = regex_region_groups
        self.region_groupings = region_groupings
        self.valid_return_types = valid_return_types

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID. Sets the following properties:

            * self.is_regions
            * self._uuid
        """

        self.is_regions = True
        self._uuid = _MODULE_UUID

        return None




    ########################
    #    CORE FUNCTIONS    #
    ########################


    #
    #
    #
 
    """
    USE THIS AS A TEMPLATE TO BUILD FUNCTION FOR MISSING VARIABLES
    def add_missing_probs(
        df_afolu_probs: pd.DataFrame,
        model_afolu: mafl.AFOLU,
        regions: sc.Regions,
        time_periods: sc.TimePeriods,
        field_iso: str,
        region_grouping: str = "un_sub",
    ) -> Union[pd.DataFrame, None]:
        
        "DOCSTRING HERE"

        regions_missing = [
            x for x in regions.all_isos
            if x not in df_afolu_probs[field_iso].unique()
        ]
        
        print(regions_missing)
        if len(regions_missing) == 0:
            return None
        
        
        fields = (
            model_afolu
            .model_attributes
            .build_variable_fields(
                model_afolu.modvar_lndu_prob_transition
            )
        )
        
        # generate by region
        df_agg_by_region = regions.aggregate_df_by_region_group(
            df_afolu_probs[
                [field_iso, time_periods.field_time_period] + fields
            ]
            .rename(
                columns = {field_iso: regions.field_iso}
            ),
            region_grouping,
            [regions.field_iso, time_periods.field_time_period],
            dict((x, "mean") for x in fields),
            field_merge = regions.field_iso,
        )
        
        # group and convert to dictionary
        field_grouping = regions.get_region_group_field(region_grouping)
        dict_agg_by_region = sf.group_df_as_dict(
            df_agg_by_region,
            [field_grouping],
        )
        
        
        ##  BUILD OUTPUT TABLE
        
        df_out = []
        
        for x in regions_missing:
            
            rg = regions.get_region_group(x, region_grouping)
            
            if rg not in dict_agg_by_region.keys():
                print(f"Region group '{rg}' not found for region {x}: skipping")
                continue

            df_tmp = (
                dict_agg_by_region
                .get(rg)
                .copy()
                .drop([field_grouping], axis = 1, )
            )
            
            df_tmp[regions.key] = x
            df_out.append(df_tmp)

        df_out = (
            pd.concat(df_out)
            .reset_index(drop = True)
        )
        
        
        return df_out
        
    """

    def add_region_or_iso_field(self,
        df: pd.DataFrame,
        field_iso: Union[str, None] = None,
        field_region: Union[str, None] = None,
        **kwargs,
    ) -> Union[str, None]:
        """Return region for region entered as region or ISO.

        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame to which to add region or iso field

        Keyword Arguments
        -----------------
        field_iso: str
            Field in df storing ISO code OR field to add with ISO code (if not 
            in df)
        field_region: str
            Field in df storing region OR field to add with region (if not in 
            df)
        **kwargs
            Passed to return_region_or_iso
        """

        field_iso = (
            self.field_iso
            if not isinstance(field_iso, str)
            else field_iso
        )
        field_region = (
            self.key
            if not isinstance(field_region, str)
            else field_region
        )

        no_action = (field_iso not in df.columns) & (field_region not in df.columns)
        no_action |= (field_iso in df.columns) & (field_region in df.columns)
        if no_action:
            return df
        
        # get fields and apply switch
        field = field_region if (field_region in df.columns) else field_iso
        field_new = field_iso if (field_region in df.columns) else field_region
        return_type = "iso" if (field_region in df.columns) else "region"

        df_out = df.copy()
        vec = list(
            df_out[field]
            .apply(
                self.return_region_or_iso, 
                return_type = return_type, 
                **kwargs
            )
        )
        df_out[field_new] = vec

        return df_out
    


    def aggregate_df_by_region_group(self,
        df_in: pd.DataFrame,
        region_grouping: str,
        fields_group: List[str],
        dict_agg: Dict[str, str],
        extract_regions: Union[str, List[str], None] = None,
        field_merge: Union[str, None] = None,
        input_type: str = "auto",
        stop_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        Get a regional average across ISOs for related to the applicable region
            group. 

        Function Arguments
        ------------------
        - df_in: input data frame
        - region_grouping: either "un" or "world_bank". The group used to group
            the regions in the data frame
        - fields_group: fields to group on (excluding region)
        - dict_agg: aggregation dictionary to use 

        Keyword Arguments
        -----------------
        - extract_regions: optional specification of a region for which a region
            grouping will be extracted from the aggregation. str, list of 
            strings, or None
        - field_merge: field to merge on. If None, defaults to Regions.field_iso
        - stop_on_error: stop execution on an error? Otherwise, returns original
            data frame.
        """
        
        ##  INITIALIZATION AND CHECKS

        # check merge field
        field_merge = self.field_iso if (field_merge is None) else field_merge
        if field_merge not in df_in.columns:
            if stop_on_error:
                valid_regions = sf.format_print_list(self.region_groupings)
                msg = f"""Region merge field {field_merge} not found in df_in in aggregate_df_by_region_group(). 
                
                Check to ensure the field is specified correctly.
                """
                raise KeyError(msg)
            
            return df_in


        # check the region grouping and return the original data frame if it's invalid
        # (raises an error if stop_on_error = True)
        region_grouping = self.check_region_grouping(
            region_grouping,
            stop_on_error = stop_on_error,
        )
        if region_grouping is None:
            return df_in


        # get desired regions from the attribute table
        field = self.get_region_group_field(region_grouping)
        tab = self.attributes.table
        all_region_groups = list(tab[field])
        
        if extract_regions is not None:
            extract_regions = (
                [extract_regions] 
                if isinstance(extract_regions, str) 
                else extract_regions
            )

            extract_regions = (
                []
                if not sf.islistlike(extract_regions)
                else [x for x in extract_regions if x in all_region_groups]
            )

            if len(extract_regions) > 0:
                tab = (
                    self.attributes.table[
                        self.attributes.table[field].isin(extract_regions)
                    ]
                    .reset_index(drop = True)
                )


        # merge on applicable field,
        df_filt = pd.merge(
            df_in,
            tab[[field_merge, field]],
            how = "inner",
        )

        dict_agg_pass = dict((k, v) for k, v in dict_agg.items())


        ##  CHECK GROUPING/AGG DICTIONARY

        if field not in fields_group:
            fields_group.append(field)
            dict_agg_pass.update({field: "first"})

        if field_merge in fields_group:
            fields_group = [x for x in fields_group if x != field_merge]
        
        # drop field_merge from dictionary if present
        if field_merge in dict_agg_pass.keys():
            dict_agg_pass.pop(field_merge)
            
        # get aggregation
        df_filt = sf.simple_df_agg(
            df_filt.drop([field_merge], axis = 1), 
            fields_group,
            dict_agg
        )
        
        return df_filt
    


    def check_region_grouping(self,
        region_grouping: str,
        stop_on_error: bool = True,
    ) -> Union[None, str]:
        """
        Check whether a region grouping is valid. If an error is not raised,
            returns the region grouping if it is valid or None otherwise.
        """
        if region_grouping not in self.region_groupings:

            if stop_on_error:
                msg = f"""
                Invalid region grouping '{region_grouping}' specified: valid 
                region groupings are defined in Regions.attributes and include
                {self.region_groupings}
                """
                raise InvalidRegionGroup(msg)

            return None

        return region_grouping



    def clean_region(self,
        region: str,
        force_to_string: bool = False,
    ) -> str:
        """
        Clean the region name. Set `force_to_string` to True to force inputs
            to take type str
        """

        if not (isinstance(region, str) | force_to_string):
            return region

        dict_repl = dict((x, "_") for x in ["-", " "])

        out = str(region).strip().lower()
        out = sf.str_replace(out, dict_repl)

        return out

    

    def convert_region_codes(self,
        df: pd.DataFrame,
        field_codes: str,
        input_code_type: str,
        output_code_type: str,
        merge_type: str = "inner",
        overwrite: bool = False,
        replace: bool = True,
    ) -> Union[pd.DataFrame, None]:
        """Convert region code types. Valid input and output code types include:
        
        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame containing the field to convert
        field_codes : str
            Field name containg codes to convert
        input_code_type : str
            Input code type to convert from. Valid input codes include:
            
            * "fao_region_code":
                FAO region code (integer)
            * "iso":
                ISO 3 alphanumeric code
            * "iso_numeric":
                ISO numeric (integer) code
            * "region": 
                Region name

        output_code_type : str
            Output code type to convert to. Valid output codes include:

            * "fao_region_code":
                FAO region code (integer)
            * "iso":
                ISO 3 alphanumeric code
            * "iso_numeric":
                ISO numeric (integer) code
            * "region": 
                Region name
            * "un_region":
                UN Region name (non-injective, image only)
            * "un_sub_region":
                UN Region name (non-injective, image only)
            * "world_bank_global_region": 
                World Bank region code
                
            
        Keyword Arguments
        -----------------
        merge_type : str
            "inner" or "outer". If outer, can   NAs
        overwrite : bool
            Overwrite existing output field if present?
        replace : bool
            Replace the field containing the region codes? If False, keeps both 
            the old (input) and new (output) fields
        """

        ##  CHECKS AND INIT

        # check that field_codes is in the dataframe
        if field_codes not in df.columns:
            return df

        # map the input/output type to the field in the attribute table
        dict_type_to_attribute_field = {
            "fao_region_code": "fao_area_code",
            _RETURN_TYPE_ISO: self.field_iso,
            _RETURN_TYPE_ISO_NUMERIC: self.field_iso_numeric,
            _RETURN_TYPE_REGION: self.key,
        }

        dict_update = [self.get_region_group_field(x) for x in self.region_groupings]
        dict_type_to_attribute_field.update(dict((x, x) for x in dict_update))

        # next, get the fields from the attribute table
        field_input = dict_type_to_attribute_field.get(input_code_type)
        field_output = dict_type_to_attribute_field.get(output_code_type)

        # check type specifications - start with inputs
        if field_input is None:
            valid_inputs_print = sf.format_print_list(list(dict_type_to_attribute_field.keys()))
            msg = f"Invalid input type '{input_code_type}' in convert_region_codes(). Valid inputs are {valid_inputs_print}."
            raise TypeError(msg)

        # check outputs
        if field_output is None:
            valid_outputs_print = sf.format_print_list(list(dict_type_to_attribute_field.keys()))
            msg = f"Invalid output type '{output_code_type}' in convert_region_codes(). Valid outputs are {valid_outputs_print}."
            raise TypeError(msg)

        # then check output overwriting and drop the existing field if necessary
        if (field_output in df.columns):
            if not overwrite:
                return df
            
            df.drop(
                [field_output], 
                axis = 1,
                inplace = True,
            )
            

        ##  BUILD OUTPUT TABLE

        df_merge = (
            self.attributes
            .table[[field_input, field_output]]
            .copy()
            .rename(
                columns = {field_input: field_codes}
            )
        )
        
        df = sf.merge_replace(
            df,
            df_merge,
            merge_type = merge_type,
            replace = replace,
        )
        
        
        return df
    


    def extract_from_df(self,
        df: pd.DataFrame,
        regions: Union[int, str, List[int], List[str], None],
        field_regions: str,
    ) -> Union[str, None]:
        """Extract 
        

        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame containing rows to filter on
        region : Union[int, str, List[int], List[str], None]
            Regions (name, ISO, or ISO numeric) to extract from a data frame. If
            None, tries to extract all rows associated with a valid region.
        field_regions : str
            Field storing regions index. If None, tries in order of hieracrhy:
            (1) self.key:               sees if the regions key is present first
            (2) self.field_iso          second, checks for field_iso
            (3) self.field_iso_numeric: finally, looks for field iso_numeric

            If the input field is misspecified or if no field is present,
            returns df.

        
        Keyword Arguments
        -----------------
        missing_flag : float
            Flag indicating a missing value
        regions_valid : Union[List[str], None]
            Optional list of regions to restrict search to. If None, searches 
            through all regions specified in attr_region
        type_input : str
            Input region type. Either "region" or "iso"
        type_return : 
            Return type. Either "region" or "iso"
        """

        ##  INITIALIZATION AND CHECKS

        if not isinstance(df, pd.DataFrame):
            return df
        
        field_regions = self.get_hierarchical_regions_field(
            df,
            field_regions = field_regions, 
        )

        if field_regions is None:
            return df
        
        regions = self.get_valid_regions(regions, )
        

        ##  NEXT, LOOK FOR VALID REGIONS

        inds = self.return_region_or_iso(
            df[field_regions].to_numpy(),
            return_none_on_missing = True,
        )
        df_out = df[[x in regions for x in inds]]

        return df_out



            


    def fill_missing_regions(self,
        df: pd.DataFrame,
        fields_data: List[str],
        dict_method: dict,
        region_spec: str,
        field_region: Union[str, None] = None,
        regions_fill: Union[List[str], None] = None,
    ) -> Union[pd.DataFrame, None]:
        """Fill in data for regions that are missing using a 
        
        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame with regions to fill
        fields_data : List[str]
            Data fields to fill
        dict_method : dict
            Fill method information. Keys are a method while values are 
            dictionaries that map parameters to values. Options are:
            * "grouping_average": Use a regional grouping average. Requires the
                following parameters:
                * "regional_grouping": grouping method to use
            * "analog_population_center": Use an analog from the nearest 2020 
                population center.
        region_spec : str
            Regional specification; must be one of
            * "iso": use ISO codes, available in regions.all_iso
            * "iso_numeric": use ISO numeric codes, available in 
                regions.all_isos_numeric
            * "region": use the region identifier, available as 
                regions.all_regions
            
        Keyword Arguments
        -----------------
        field_region : Union[str, None]
            Optional specification of region field to use. If None, uses the 
            field associated with region_spec.
        regions_fill : Union[List[str], None]
            Optional specification of regions to fill. If None, defaults to all 
            regions availabile in regions.all_regions.
        """

        ##  INITIAlIZATION AND CHECKS
        
        # get region specification--default to region
        region_spec = (
            _RETURN_TYPE_REGION
            if region_spec not in self.valid_return_types
            else region_spec
        )

        # check dict method specification
        ret_df = not isinstance(dict_method, dict)
        ret_df |= (len(dict_method) != 1) if not ret_df else ret_df
        if ret_df:
            msg = "Invalid specification of dict_method in fill_missing_regions(): must be a dictionary of length 1."
            warnings.warn(msg)
            return df

        # check method
        method = list(dict_method.keys())[0]
        valid_methods = [
            "grouping_average", 
            "analog_population_center",
        ]

        if method not in valid_methods:
            msg = sf.format_print_list(valid_methods)
            msg = f"""Invalid method '{method}' specified in fill_missing_regions(): valid 
            methods are {msg}. Returning original data frame.
            """
            return df

        
        # SET REGION INFO; option to set region field as something else

        field_region = None if not isinstance(field_region, str) else field_region

        if region_spec == _RETURN_TYPE_ISO:
                field_region = self.field_iso if field_region is None else field_region
                regions_space = self.all_isos
        elif region_spec == _RETURN_TYPE_ISO_NUMERIC:
                field_region = self.field_iso_numeric if field_region is None else field_region
                regions_space = self.all_isos_numeric
        elif region_spec == _RETURN_TYPE_REGION:
                field_region = self.key if field_region is None else field_region
                regions_space = self.all_regions

        # raise an error if the region field is not found
        if field_region not in df.columns:
            msg = f"Error in fill_missing_regions: region field {field_region} not found in the input data frame."
            raise KeyError(msg)

        fields_index = [x for x in df.columns if x not in [field_region] + fields_data]

        # initialize regions that are not in the space
        regions_missing = [
            x for x in regions_space
            if x not in df[field_region].unique()
        ]

        # regions to actually fill in
        regions_fill = (
            regions_missing
            if not sf.islistlike(regions_fill)
            else [x for x in regions_fill if x in regions_missing]
        )
        
        if len(regions_fill) == 0:
            return df


        ##  NOW, BASED ON THE REPLACEMENT TYPE, CALL A SUPPORTING FUNCTION

        # initialize as df
        df_out = df

        if method == "grouping_average":
            
            # get the region grouping
            dict_cur = dict_method.get(method)
            grp = dict_cur.get("region_grouping")
        
            if grp is None:
                msg = f"""
                Invalid specification of method '{method}' in 
                fill_missing_regions(): parameter "region_grouping" not found.
                """
                warnings.warn(msg)
                return df
            
            # otherwise, fill using the aggregate
            df_out = self.fill_missing_regions_using_grouping_aggregate(
                df,
                grp,
                field_region,
                fields_data,
                fields_index,
                regions_fill,
                return_region_type = region_spec,
                aggregation_method = "mean",
            )


        elif method == "analog_population_center":
            print("ADD analog_population_center")


        return df_out



    def fill_missing_regions_using_grouping_aggregate(self,
        df: pd.DataFrame,
        region_grouping: str,
        field_region: Union[str, None],
        fields_data: List[str],
        fields_index: Union[List[str], None],
        regions_fill: List[str],
        return_region_type: str,
        aggregation_method: str = "mean",
    ) -> Union[pd.DataFrame, None]:
        """Fill in data for regions that are missing using a grouping average. 
            NOTE: includes little generic checking. Support function for 
            fill_missing_regions()

        
        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame with regions to fill
        region_grouping : str
            Grouping to use for filling in reginal mean
        field_region : str
            Optional specification of region field to use. If None, uses the 
            field associated with region_spec.
        fields_data : List[str]
            Data fields to fill
        fields_index : Union[List[str], None]
            Optional fields to specify as index. If None, defaults to those in
            ModelAttributes
        regions_fill : List[str]
            Optional specification of regions to fill. If None, defaults to all 
            regions availabile in regions.all_regions.
        return_region_type : str
            Return "region", "iso", or "iso_numeric". Should align with field 
            types in field_region

        Keyword Arguments
        -----------------
        aggregation_method : 
            Aggregation function used to combine across regions. Passed to 
            pd.groupby.aggregate. "mean", "sum", etc are acceptable. 
        """

        ##  INITIALIZATION AND CHECKS

        # check data fields
        fields_data = [x for x in fields_data if x in df.columns]
        if len(fields_data) == 0:
            return df

        # check the region grouping
        if region_grouping not in self.region_groupings:
            msg = f"Error in fill_missing_regions_grouping_average: invalid region grouping '{region_grouping}'."
            raise InvalidRegionGroup(msg)

        # next, get region groupings that need to be retrieved 
        all_group_vals = set(
            [
                self.get_region_group(region, region_grouping)
                for region in regions_fill
            ]
        )
        all_group_vals = sorted(list(all_group_vals))

        # get regions as key values
        regions_fill_all = [
            self.return_region_or_iso(x, return_type = _RETURN_TYPE_REGION, )
            for x in regions_fill
        ]
        field_rg = self.get_region_group_field(region_grouping)
        
        # get attribute table, then convert to dictionary mapping region group to regions
        dict_rg_to_regions = self.attributes.table
        dict_rg_to_regions = dict_rg_to_regions[
            dict_rg_to_regions[field_rg]
            .isin(all_group_vals)
        ][[self.key, field_rg]]

        # initialize the dictionary of regions to region means
        dict_rg_to_region_means = sf.group_df_as_dict(
            dict_rg_to_regions,
            fields_group = [field_rg],
            fields_out_set = [self.key]
        )


        ##  NEXT, BUILD REGIONAL AGGREGATE

        field_tmp = "REGION_TMP_ADD"
        df[field_tmp] = [
            self.return_region_or_iso(x, return_type = _RETURN_TYPE_REGION, )
            for x in list(df[field_region])
        ]

        # init dictionary and map over region groupings to combine data
        dict_rg_to_aggs = {}
        for k, v in dict_rg_to_region_means.items():

            # merge to regions, drop na, and calculate aggregate 
            df_cur = (
                pd.merge(
                    v.rename(columns = {self.key: field_tmp}),
                    df,
                    how = "inner",
                )
                .drop(
                    columns = [field_tmp, field_region]
                )
                .dropna()
            )

            dict_agg = dict((x, "first") for x in fields_index)
            dict_agg.update(dict((x, aggregation_method) for x in fields_data))

            # get the means
            df_out = sf.simple_df_agg(
                df_cur,
                fields_index,
                dict_agg,
            )

            dict_rg_to_aggs.update({k: df_out})


        # drop temporary merge column
        df.drop(
            columns = field_tmp, 
            inplace = True,
        )

        
        ##  ITERATE OVER REGIONS AND ASSIGN AGGREGATE

        df_out = [df]

        for region in regions_fill_all:
            
            # get region grouping and aggregate info
            grouping = self.get_region_group(region, region_grouping)
            df_rg = dict_rg_to_aggs.get(grouping)
            if df_rg is None:
                continue

            df_update = df_rg.copy()
            df_update[field_region] = self.return_region_or_iso(
                region,
                return_type = return_region_type,
            )

            df_out.append(df_update[df.columns])

        
        # concatenate and return
        df_out = (
            pd.concat(
                df_out,
                axis = 0
            )
            .reset_index(drop = True)
        )


        return df_out


    
    def get_closest_region(self,
        region: Union[int, str],
        missing_flag: float = -999,
        regions_valid: Union[List[str], None] = None,
        type_input: str = _RETURN_TYPE_REGION,
        type_return: str = _RETURN_TYPE_REGION,
    ) -> Union[str, None]:
        """Based on latitude/longitude of population centers, find the closest 
            neighboring region.
        

        Function Arguments
        ------------------
        region : Union[int, str]
            Region (name, ISO, or ISO numeric) to search for closest neighbor
        attr_region : attribute table for regions
        
        Keyword Arguments
        -----------------
        missing_flag : float
            Flag indicating a missing value
        regions_valid : Union[List[str], None]
            Optional list of regions to restrict search to. If None, searches 
            through all regions specified in attr_region
        type_input : str
            Input region type. Either "region" or "iso"
        type_return : 
            Return type. Either "region" or "iso"
        """

        ##  INITIALIZATION
        attr_region = self.attributes
        type_return = _RETURN_TYPE_REGION if (type_return not in self.valid_return_types) else type_return
        type_input = _RETURN_TYPE_REGION if (type_input not in self.valid_return_types) else type_input
        
        # check region/lat/lon
        #region = self.dict_iso_to_region.get(region) if (type_input == "iso") else region
        region = self.return_region_or_iso(
            region, 
            return_none_on_missing = True,
            return_type = _RETURN_TYPE_REGION, 
        )
        region = region if (region in attr_region.key_values) else None
        coords = self.get_coordinates(region)
        
        # return None if one of the dimensions is missing
        if (coords is None) or (region is None):
            return None

        lat, lon = coords
        
        
        ##  FILTER TABLE TO VALID REGIONS AND BUILD DISTANCE FUNCTION
        
        if (regions_valid is None):
            regions_valid = attr_region.key_values 
        else:
            regions_valid = [
                self.return_region_or_iso(x, return_type = _RETURN_TYPE_REGION, )
                for x in regions_valid
            ]
            regions_valid = [x for x in attr_region.key_values if x in regions_valid]
            
            
        df_regions = (
            attr_region.table[
                attr_region.table[attr_region.key].isin(regions_valid)
            ]
            .copy()
            .reset_index(drop = True)
        )
        
        # function to apply
        def f(
            tup: Tuple[float, float]
        ) -> float:
            y, x = tuple(tup)
            
            out = (
                -1.0
                if (min(y, lat) < -90) or (max(y, lat) > 90) or (min(x, lon) < -180) or (max(x, lon) > 180)
                else geopy.distance.geodesic((lat, lon), (y, x)).km
            )
            
            return out
        

        ##  FINALLY, APPLY DISTANCES AND FILTER

        vec_dists = np.array(
            df_regions[[self.field_lat, self.field_lon]]
            .apply(f, raw = True, axis = 1)
        )
        valid_dists = vec_dists[vec_dists > 0.0]
        out = None
        
        if len(valid_dists) > 0:

            m = min(vec_dists)
            w = np.where(vec_dists == m)[0]

            out = (
                list(df_regions[attr_region.key])[w[0]]
                if len(w) > 0
                else None
            )

            out = self.return_region_or_iso(
                out,
                return_type = type_return,
            )


        return out



    def get_coordinates(self,
        region: Union[int, str, None],
    ) -> Union[Tuple[float, float], None]:
        """Return the latitude, longitude coordinates of the population centroid 
            of region `region`. `region` can be entered as a region (one of the 
            self.attributes.key_values) or the ISO3 code. If neither is found, 
            returns None

        Function Arguments
        ------------------
        region_str : Union[int, str, None]
            Region specification; either region name, ISO Alpha 3 code, or ISO 
            Numeric code can be entered
        """
        
        dict_region_to_lat = self.attributes.field_maps.get(
            f"{self.attributes.key}_to_{self.field_lat}"
        )
        dict_region_to_lon = self.attributes.field_maps.get(
            f"{self.attributes.key}_to_{self.field_lon}"
        )

        # check region
        region = self.return_region_or_iso(
            region, 
            return_type = _RETURN_TYPE_REGION, 
        )
        
        if region is None:
            return None

        # if valid, get coordinates
        tuple_out = (dict_region_to_lat.get(region), dict_region_to_lon.get(region))

        return tuple_out
    


    def get_hierarchical_regions_field(self,
        df: pd.DataFrame,
        field_regions: Union[str, None] = None,
    ) -> Union[str, None]:
        """See if field_regions is present, or retrieves a hierarchical field. 
            If field_regions is None, searches for one of the following (in 
            order) and returns the first found. 

            (1) self.key:               sees if the regions key is present first
            (2) self.field_iso          second, checks for field_iso
            (3) self.field_iso_numeric: finally, looks for field iso_numeric

        Returns None if no valid field is found. 


        Function Arguments
        ------------------
        df : pd.DataFrame
            DataFrame containing rows to filter on
        field_regions : str
            Field storing regions index. If None, retuns one of the following if
            present, with priority of return as noted above.
        """

        # if string, return based on presence in df
        if isinstance(field_regions, str):
            out = field_regions if field_regions in df.columns else None
            return out

        # initialize as None and return the field if it is found
        out = None
        fields_ordered = [
            self.key,
            self.field_iso,
            self.field_iso_numeric
        ]

        for fld in fields_ordered:
            if fld in df.columns: 
                out = fld
                break
        
        return out
    


    def get_region_group(self,
        region: Union[int, str], 
        region_grouping: str,
        stop_on_error: bool = False,
    ) -> Union[str, None]:
        """Get a regional grouping for a region by type

        Function Arguments
        ------------------
        region : Union[int, str]
            region name, ISO Alpha 3, or ISO numeric code
        region_grouping : str
            Either region grouping specified in the attribute table. Can 
            include:
            * ipcc_estimated_afolu
            * ipcc_ar5
            * un
            * un_sub
            * world_bank_global

        Keyword Arguments
        -----------------
        stop_on_error : bool
            Stop if the region grouping isn't found?
        """
        
        ##  INITIALIZATION AND CHECKS

        # 
        if region_grouping not in self.region_groupings:
            if stop_on_error:
                valid_regions = sf.format_print_list(self.region_groupings)
                msg = f"""Invalid region grouping '{region_grouping}' specified in 
                aggregate_df_by_region_group(): valid regions are {self.region_groupings}
                """
                raise InvalidRegionGroup(msg)
            return None

        # get it as a key, then use attribute function
        region = self.return_region_or_iso(
            region,
            return_type = _RETURN_TYPE_REGION,
        )

        out = self.attributes.get_attribute(
            region, 
            self.get_region_group_field(region_grouping),
        )

        return out
    


    def get_region_group_field(self,
        region_grouping: str,
    ) -> Union[str, None]:
        """Get a regional grouping field in the regions attribute table. Returns
            None if no match is found.

        Function Arguments
        ------------------
        region_grouping : str
            Element of self.region_groupings

        Keyword Arguments
        -----------------
        """

        if region_grouping not in self.region_groupings:
            return None

        field = None
        
        # search over fields
        for fld in self.attributes.table.columns:
            match = self.regex_region_groups.match(fld)
            if match is None:
                continue

            if match.groups()[0] == region_grouping:
                field = fld
                break

        return field
    


    def get_region_name(self,
        region: Union[str, int], 
    ) -> pd.DataFrame:
        """Get a region name (not-cleaned)

        Function Arguments
        ------------------
        region : Union[str, int]
            Region, ISO Alpha 3, or ISO numeric code

        Keyword Arguments
        -----------------
        """

        # get it as a key, then use attribute function
        region = self.return_region_or_iso(
            region,
            return_type = "region",
        )

        out = self.attributes.get_attribute(
            region, 
            "category_name",
        )

        return out



    def get_regions_df(self,
        include_iso: bool = False,
        include_region_wb_key: bool = False,
        regions: Union[List[str], None] = None,
        regions_wb: Union[List[str], None] = None,    
    ) -> Union[pd.DataFrame, None]:
        """Initialize a data frame of regions. Returns None if no valid regions
            are specified. 
        
        Keyword Arguments
        -----------------
        include_iso : bool
            Include iso code?
        include_region_wb_key : bool
            If `regions_wb == True`, then set to True to keep the region_wb key 
            as a column
        regions : Union[List[str], None]
            List of regions to use to build the data frame
        regions_wb : Union[List[str], None]
            Optional list of world bank regions to use to filter
        
        """
        
        # check regions and WB regions
        regions = self.get_valid_regions(regions)
        if regions is None:
            return None

        regions_wb = [regions_wb] if isinstance(regions_wb, str) else regions_wb

        # initialize output df and check if global regions should be added
        df_out = pd.DataFrame({self.key: regions})
        if sf.islistlike(regions_wb):

            df_out[self.field_wb_global_region] = (
                df_out[self.key]
                .replace(self.dict_region_to_wb_region)
            )

            df_out = (
                df_out[
                    df_out[self.field_wb_global_region].isin(regions_wb)
                ]
                .reset_index(drop = True)
            )

            (
                df_out.drop([self.field_wb_global_region], axis = 1, inplace = True)
                if not include_region_wb_key
                else None
            )

        if include_iso:
            df_out[self.field_iso] = df_out[self.key].apply(self.dict_region_to_iso.get)

        return df_out


    
    def get_valid_regions(self,
        regions: Union[int, str, List[int], List[str], None],
    ) -> Union[List[str], None]:
        """Enter a list (or list-like object) iteratable of regions, a single
            region, or None (to return all valid regions), and return a list of
            valid regions.

        Function Arguments
        ------------------
        regions : Union[int, str, List[int], List[str], None]
            List-like object of regions, a string specifying a region, or None 
            (to return all valid regions)
        """
        regions = [regions] if isinstance(regions, (int, str, )) else regions
        regions = self.all_regions if (regions is None) else regions

        regions = (
            [
                self.return_region_or_iso(
                    x,
                    return_none_on_missing = True,
                    return_type = _RETURN_TYPE_REGION, 
                ) 
                for x in regions
            ]
            if sf.islistlike(regions)
            else None
        )

        # sort
        regions = (
            None 
            if (len(regions) == 0) 
            else [x for x in self.all_regions if x in regions]
        )

        return regions



    def get_un_region(self,
        region: str,
        input_region: str = "region",
    ) -> Union[str, None]:
        """Retrieve the UN global region associated with region (roughly a 
            continent). Often used for assigning regional averages. Use 
            input_region = "iso" to convert from an iso code.
        """
        out = self.get_region_group(region, "un")

        return out



    def get_world_bank_region(self,
        region: str,
        input_region: str = "region",
    ) -> Union[str, None]:
        """Retrieve the World Bank global region associated with region. Often 
            used for assigning regional averages. Use input_region = "iso" to 
            convert from an iso code.
        """
        out = self.get_region_group(region, "world_bank_global")

        return out



    def return_region_or_iso(self,
        region: Union[int, str, List[int], List[str]],
        clean_inputs: bool = False,
        return_none_on_missing: bool = False,
        return_type: str = _RETURN_TYPE_REGION,
        try_iso_numeric_as_string: bool = True,
    ) -> Union[str, None]:
        """Return region for region entered as region or ISO. 
        
        * Returns None if `region` is specified improperly, or, if 
            `return_none_on_missing = True`, the region is properly specified
            but not found.

        Function Arguments
        ------------------
        region : Union[int, str, List[str]]
            Region, iso 3-digit alpha code, or iso numeric code in individual or
            list-like form

        Keyword Arguments
        -----------------
        clean_inputs : bool
            Try to clean the input region first
        return_none_on_missing : bool 
            Set to True to return None if input region `region` is not found
        return_type : str
            "region" or "iso" or "iso_numeric". Will return: 
            * Region if set to "region" or 
            * ISO Alpha 3 if set to "iso"
            * ISO Numeric (integer) if set to "iso_numeric"
        try_iso_numeric_as_string : 
            If a region is not found in any of the three inputs (region, iso, 
            iso_numeris), will check to see if it is an iso_numeric code entered 
            as a string.
        """
        return_type = (
            _RETURN_TYPE_REGION
            if (return_type not in self.valid_return_types) 
            else return_type
        )
      
        # set up to allow for listlike or string
        element_input = isinstance(region, str) | isinstance(region, int)
        region = [region] if element_input else region
        region = list(region) if sf.islistlike(region) else None
        if region is None:
            return None

        # check for input type
        for i, r in enumerate(region):
            
            # init and cleaning
            input_type = None
            r = self.clean_region(r) if clean_inputs else r

            # check region against input type
            if r in self.all_regions:
                input_type = _RETURN_TYPE_REGION
            elif r in self.all_isos:
                input_type = _RETURN_TYPE_ISO
            elif try_iso_numeric_as_string & isinstance(r, str):
                try:
                    r = int(r)
                except:
                    None
                
            if (r in self.all_isos_numeric) & (input_type is None):
                input_type = _RETURN_TYPE_ISO_NUMERIC
            
            # if input not found, continue
            if input_type is None:
                region[i] = None if return_none_on_missing else r
                continue

            
            # otherwise, retrieve in terms of key - if r is a region, then the dictionary will not contain it
            region_full = (
                self.dict_iso_to_region.get(r)
                if input_type == _RETURN_TYPE_ISO
                else self.dict_iso_numeric_to_region.get(r, r)
            )

            # based on return type, get output
            out = region_full
            if return_type == _RETURN_TYPE_ISO:
                out = self.dict_region_to_iso.get(region_full)
            elif return_type == _RETURN_TYPE_ISO_NUMERIC:
                out = self.dict_region_to_iso_numeric.get(region_full)
        
            region[i] = out
            

        region = region[0] if element_input else region

        return region



    ##  DATA SOURCE-SPECIFIC MODIFICATIONS

    def data_func_iea_get_isos_from_countries(self,
        df_in: Union[pd.DataFrame, List, np.ndarray, str],
        field_country: Union[str, None] = None,
        return_modified_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Map IEA countries in field_country to ISO codes contained in 
            df_in[field_country]. If field_country is None, defaults to 
            self.field_iea_country.

        Function Arguments
        ------------------
        - df_in: input data frame containing field country (if None, uses 
            self.field_iea_country) OR list/np.ndarray or input country strings
            OR string

        Keyword Arguments
        -----------------
        - field_country: field in df_in used to identify IEA countries if df_in
            is a DataFrame
        - return_modified_df: if True and df_in is a DataFrame, will return a 
            DataFrame modified to include the iso field
        """
       
        field_country = self.field_iea_country if (field_country is None) else field_country
        vec_iso = (
            list(df_in[field_country]) 
            if isinstance(df_in, pd.DataFrame) 
            else (
                [df_in] if isinstance(df_in, str) else df_in
            )
        )

        vec_iso = [self.clean_region(x) for x in vec_iso]
        vec_iso = [self.dict_iea_countries_lc_to_regions.get(x, x) for x in vec_iso]
        vec_iso = [self.dict_region_to_iso.get(x, x) for x in vec_iso]
        
        out = np.array(vec_iso).astype(str)
        if isinstance(df_in, pd.DataFrame) & return_modified_df:
            df_in[self.field_iso] = vec_iso
            out = df_in

        return out
    


    def data_func_try_isos_from_countries(self,
        df_in: Union[pd.DataFrame, List, np.ndarray, str],
        field_country: Union[str, None] = None,
        missing_iso_flag: Union[str, None] = None,
        return_modified_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Map countries in field_country to ISO codes contained in 
            df_in[field_country]. If field_country is None, defaults to 
            self.key.

        Function Arguments
        ------------------
        - df_in: input data frame containing field country (if None, uses 
            self.key) OR list/np.ndarray or input country strings OR string

        Keyword Arguments
        -----------------
        - field_country: field in df_in used to identify IEA countries if df_in
            is a DataFrame
        - missing_iso_flag: if is None, will leave regions as input values if 
            not found. Otherwise, uses flag
        - return_modified_df: if True and df_in is a DataFrame, will return a 
            DataFrame modified to include the iso field
        """
        
        # check country field
        field_country = self.key if (field_country is None) else field_country
        if isinstance(df_in, pd.DataFrame):
            if field_country not in df_in.columns:
                return df_in

        # initialize vector of isos
        vec_iso = (
            list(df_in[field_country]) 
            if isinstance(df_in, pd.DataFrame) 
            else (
                [df_in] if isinstance(df_in, str) else df_in
            )
        )
        missing_iso_flag = (
            str(missing_iso_flag)
            if missing_iso_flag is not None
            else None
        )

        # try to build
        for i, region_base in enumerate(vec_iso):
            
            region = self.clean_region(region_base)

            # set a hierarchy
            region_full = (
                region
                if region in self.dict_region_to_iso.keys()
                else None
            )
            region_full = (
                self.dict_iea_countries_lc_to_regions.get(region)
                if region_full is None
                else region_full
            )
            region_full = (
                self.dict_generic_countries_to_regions.get(region)
                if region_full is None
                else region_full
            )
            
            # get default flag, and check if region is entered as a valid ISO
            flag = (
                region_base 
                if (missing_iso_flag is None) 
                else missing_iso_flag
            )
            flag = (
                region.upper()
                if region.upper() in self.dict_region_to_iso.values()
                else flag
            )

            iso = self.dict_region_to_iso.get(region_full, flag)
            vec_iso[i] = iso

        # convert to array and modify data frame if specified
        out = np.array(vec_iso).astype(str)
        if isinstance(df_in, pd.DataFrame) & return_modified_df:
            df_in[self.field_iso] = vec_iso
            out = df_in

        return out





class TimePeriods:
    """
    Leverage some simple time period actions based on a model attributes. The 
        TimePeriods class provides a seamless method for converting years to 
        time periods in SISEPUEDE and can be expanded to integrate months (if
        modeling at that scale).
    """
    def __init__(self,
        model_attributes: 'ModelAttributes'
    ):

        self._initialize_time_properties(model_attributes, )
        self._initialize_uuid()

        return None
    


    def _initialize_time_properties(self,
        model_attributes: 'ModelAttributes',
        field_year: str = _FIELD_YEAR,
    ) -> None:
        """
        Set the following properties:

            * self.all_time_periods
            * self.all_years
            * self.attributes
            * self.dict_time_period_to_year
            * self.dict_year_to_time_period
            * self.field_time_period
            * self.field_year
            * self.min_year
        """

        attributes = model_attributes.get_dimensional_attribute_table(model_attributes.dim_time_period)
        dict_year_to_time_period = attributes.field_maps.get(f"{field_year}_to_{attributes.key}")
        dict_time_period_to_year = attributes.field_maps.get(f"{attributes.key}_to_{field_year}")
        
        all_time_periods = attributes.key_values
        all_years = sorted(list(set([dict_time_period_to_year.get(x) for x in all_time_periods])))
        year_min, year_max = min(all_years), max(all_years)


        self.all_time_periods = all_time_periods
        self.all_years = all_years
        self.attributes = attributes
        self.dict_time_period_to_year = dict_time_period_to_year
        self.dict_year_to_time_period = dict_year_to_time_period
        self.field_time_period = attributes.key
        self.field_year = field_year
        self.year_max = year_max
        self.year_min = year_min

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID. Sets the following properties:

            * self.is_time_periods
            * self._uuid
        """

        self.is_time_periods = True
        self._uuid = _MODULE_UUID

        return None


    

    ########################
    #    CORE FUNCTIONS    #
    ########################

    def get_closest_time_period(self,
        t: int,
    ) -> int:
        """
        Return the closest available time period in time periods. If t is in the 
            time periods, returns t
        """

        if t in self.all_time_periods:
            return t

        # get distances and find first that is closest
        t_dists = np.abs(np.array(self.all_time_periods) - t)
        pos = np.argmin(t_dists)

        t_out = self.all_time_periods[pos]

        return t_out



    def get_closest_year(self,
        y: int,
    ) -> int:
        """
        Return the closest available year in time periods. If y is in the time 
            periods, returns y
        """

        if y in self.all_years:
            return y

        # get distances and find first that is closest
        y_dists = np.abs(np.array(self.all_years) - y)
        pos = np.argmin(y_dists)

        y_out = self.all_years[pos]

        return y_out


    def get_time_period_df(self,
        include_year: bool = False,
        time_periods: Union[List[int], None] = None,
        years: Union[List[int], None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Initialize a data frame of regions. Returns None if no valid regions
            are specified. 
        
        NOTE: returns none if both time_periods & years are specified

        Keyword Arguments
        -----------------
        - include_year: include year?
        - time_periods: list of time periods to include
        - years: list of years to base off of
        """
        
        # default to returning None
        return_none = sf.islistlike(time_periods) & sf.islistlike(years)

        # check time period specification
        time_periods = (
            self.all_time_periods 
            if not sf.islistlike(time_periods)
            else [x for x in self.all_time_periods if x in time_periods]
        )
        return_none |= (len(time_periods) == 0)
        
        # check years
        years = (
            self.all_years 
            if not sf.islistlike(years)
            else [x for x in self.all_years if x in years]
        )
        return_none |= (len(years) == 0)

        if return_none:
            return None

        # initialize output df and check if global regions should be added
        df_out = pd.DataFrame({self.field_time_period: time_periods})
        df_out = self.tps_to_years(df_out)
        df_out = (
            df_out[
                df_out[self.field_time_period].isin(time_periods)
                & df_out[self.field_year].isin(years)
            ]
            .sort_values(by = [self.field_time_period])
            .reset_index(drop = True)
        )

        (
            df_out.drop([self.field_year], axis = 1, inplace = True)
            if not include_year
            else None
        )

        return df_out



    def tp_to_year(self,
        time_period: int,
    ) -> int:
        """
        Convert time period to a year. If time_period is numeric, uses closest
            integer; otherwise, returns None
        """
        time_period = (
            time_period if isinstance(time_period, int) else (
                int(np.round(time_period))
                if isinstance(time_period, float)
                else None
            )     
        )

        if time_period is None:
            return None

        out = self.dict_time_period_to_year.get(
            time_period,
            time_period + self.year_min
        )

        return out


    
    def tps_to_years(self,
        vec_tps: Union[List, np.ndarray, pd.DataFrame, pd.Series],
        field_time_period: Union[str, None] = None,
        field_year: Union[str, None] = None,
    ) -> np.ndarray:
        """
        Convert a vector of years to time periods. 

        Function Arguments
        ------------------
        - vec_tps: List-like input including time periods to convert to years; 
            if DataFrame, will write to field_year (if None, default to
            self.field_year) and look for field_time_period (source time 
            periods, defaults to self.field_time_period)

        Keyword Arguments
        -----------------
        - field_time_period: optional specification of a field to store time 
            period. Only used if vec_years is a DataFrame.
        - field_year: optional specification of a field containing years. Only 
            used if vec_years is a DataFrame.
        """

        df_q = isinstance(vec_tps, pd.DataFrame)
        # check input if data frame
        if df_q:
            
            field_time_period = self.field_time_period if (field_time_period is None) else field_time_period
            field_year = self.field_year if (field_year is None) else field_year
            if field_time_period not in vec_tps.columns:
                return None

            vec = list(vec_tps[field_time_period])

        else:
            vec = list(vec_tps)

        out = np.array([self.tp_to_year(x) for x in vec])

        if df_q:
            df_out = vec_tps.copy()
            df_out[field_year] = out
            out = df_out

        return out
    


    def year_to_tp(self,
        year: int,
    ) -> Union[int, None]:
        """
        Convert a year to a time period. If year is numeric, uses closest
            integer; otherwise, returns None
        """
        year = (
            year if sf.isnumber(year, integer = True) else (
                int(np.round(year))
                if sf.isnumber(year)
                else None
            )     
        )

        if year is None:
            return None

        out = self.dict_year_to_time_period.get(
            year,
            year - self.year_min
        )

        return out



    def years_to_tps(self,
        vec_years: Union[List, np.ndarray, pd.DataFrame, pd.Series],
        field_time_period: Union[str, None] = None,
        field_year: Union[str, None] = None,
    ) -> np.ndarray:
        """
        Convert a vector of years to time periods. 

        Function Arguments
        ------------------
        - vec_years: List-like input including years to convert to time period;
            if DataFrame, will write to field_time_period (if None, default to
            self.field_time_period) and look for field_year (source years,
            defaults to self.field_year)

        Keyword Arguments
        -----------------
        - field_time_period: optional specification of a field to store time 
            period. Only used if vec_years is a DataFrame.
        - field_year: optional specification of a field containing years. Only 
            used if vec_years is a DataFrame.
        """

        df_q = isinstance(vec_years, pd.DataFrame)
        # check input if data frame
        if df_q:

            field_time_period = self.field_time_period if (field_time_period is None) else field_time_period
            field_year = self.field_year if (field_year is None) else field_year
            if field_year not in vec_years.columns:
                return None

            vec = list(vec_years[field_year])
        else:
            vec = list(vec_years)

        out = np.array([self.year_to_tp(x) for x in vec])

        if df_q:
            df_out = vec_years.copy()
            df_out[field_time_period] = out
            out = df_out

        return out
        


class YAMLConfiguration:
    """Initialize a configuration from a YAML file. 

    Initialization Arguments
    ------------------------
    fp : Union[dict, str, pathlib.Path]
        File path to YAML file to read in OR dictionary
    """
    def __init__(self,
        fp: Union[dict, str, pathlib.Path],
    ) -> None:
        
        self._initialize_data(fp)
        self._initialize_uuid()

        return None
        
    
        
    def _initialize_data(self,
        fp: Union[dict, str, pathlib.Path],
    ) -> None:
        """Read the yaml dictionary. Sets the following properties:
        
            * self.dict_yaml
            * self.path
        """
        
        if isinstance(fp, str) | isinstance(fp, pathlib.Path):
            try:
                dict_yaml = sf.read_yaml(str(fp), munchify_dict = False)
            except Exception as e:
                raise RuntimeError(f"Error initializing YAML dictionary in yaml_config: {e}")
        
        elif isinstance(fp, dict):
            dict_yaml = fp
            fp = None

        else: 
            tp = str(type(fp))
            msg = f"Invalid type '{tp}' specified for YAMLConfiguration: fp must be of type 'str' or 'dict'."
            raise TypeError(msg)


        ##  SET PROPERTIES

        self.dict_yaml = dict_yaml
        self.path = fp
        
        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """Initialize the UUID. Sets the following properties:
            
            * self.is_yaml_configuration
            * self._uuid
        """

        self.is_yaml_configuration = True
        self._uuid = _MODULE_UUID

        return None
            


    def get(self,
        key: str,
        delim: str = ".",
        return_on_none: Any = None,
        stop_on_missing: bool = False,
    ) -> Any:
        """Allow for recursive retrieval of dictionary values. Nested keys are 
            stored using delimiters.

        Function Arguments
        ------------------
        key : 
            key that represents YAML nesting. Levels are seperated by delim, 
                e.g., to access

                dict_yaml.get("level_1").get("level_2")

                use 

                YAMLConfig.get("level_1.level_2")

        Keyword Arguments
        -----------------
        delim : str
            Delimeter to use in get
        return_on_none : Any
            Optional value to return on missing value
        stop_on_missing : bool
            Option to stop if not found
        """
        
        return_none = not (isinstance(key, str) & isinstance(delim, str))
        return_none |= not isinstance(self.dict_yaml, dict)
        if return_none:
            return None
        
        # split keys into path and initialize value
        key_full = key
        keys_nested = key.split(delim)
        value = self.dict_yaml
        
        for key in keys_nested:
            
            value = value.get(key)
    
            if not isinstance(value, dict):
                break
        
        # return value?
        if (key == keys_nested[-1]) & (value is not None):
            return value
        
        if stop_on_missing:
            raise YAMLConfigurationKeyError(f"Configuration key '{key_full}' not found in the YAML Configuration file.")
        
        # previous code------
        # value = (
        #     value
        #     if (key == keys_nested[-1]) & (value is not None)
        #     else return_on_none
        # )

        return return_on_none
        



###################################
###                             ###
###    SOME SIMPLE FUNCTIONS    ###
###                             ###
###################################

def is_regions(
    obj: Any,
) -> bool:
    """
    check if obj is a Regions object
    """

    out = hasattr(obj, "is_regions")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out




def is_time_periods(
    obj: Any,
) -> bool:
    """
    check if obj is a TimePeriods object
    """

    out = hasattr(obj, "is_time_periods")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out




def is_yaml_configuration(
    obj: Any,
) -> bool:
    """
    check if obj is a Regions object
    """

    out = hasattr(obj, "is_yaml_configuration")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out






    
    
    