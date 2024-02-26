###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY AND SUPPORT DATA PIPELINE
###

import geopy.distance
from model_attributes import *
import numpy as np
import pandas as pd
import support_functions as sf



class Regions:
    """
    Leverage some simple region actions based on model attributes. Supports the
        following actions for data:

        * Aggregation by World Bank global region
        * Finding the closest region (by population centroid)
        * Shared replacement dictionaries (IEA/WB/UN)
        * And more

    The Regions class is designed to provide convenient support for batch 
        integration of global and regional datasets into the SISEPUEDE 
        framework.
    """
    def __init__(self,
        model_attributes: ModelAttributes,
    ):

        self._initialize_region_properties(model_attributes)

        # initialize some default data source properties
        self._initialize_defaults_iea()
        self._initialize_generic_dict()

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
        model_attributes: ModelAttributes,
        field_year: str = "year",
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

        return None




    ########################
    #    CORE FUNCTIONS    #
    ########################

    # 
    def aggregate_df_by_wb_global_region(self,
        df_in: pd.DataFrame,
        global_wb_region: str,
        fields_group: List[str],
        dict_agg: Dict[str, str],
        field_iso: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Get a regional average (for WB global region) across ISOs for which
            production averages are available in df_in

        Function Arguments
        ------------------
        - df_in: input data frame
        - global_wb_region: World Bank global region to aggregate df_in to
        - fields_group: fields to group on (excluding region)
        - dict_agg: aggregation dictionary to use 

        Keyword Arguments
        -----------------
        - field_iso: field containing the ISO code. If None, defaults to 
            self.field_iso
        """
        
        field_iso = self.field_iso if (field_iso is None) else field_iso
        if global_wb_region not in self.all_wb_regions:
            return df_in

        regions_wb = [
            self.dict_region_to_iso.get(x) 
            for x in self.dict_wb_region_to_region.get(global_wb_region)
        ]
        df_filt = df_in[df_in[field_iso].isin(regions_wb)]
        
        # get aggregation
        df_filt = sf.simple_df_agg(
            df_filt, 
            fields_group,
            dict_agg
        )
        
        return df_filt
    


    def get_regions_df(self,
        include_iso: bool = False,
        include_region_wb_key: bool = False,
        regions: Union[List[str], None] = None,
        regions_wb: Union[List[str], None] = None,    
    ) -> Union[pd.DataFrame, None]:
        """
        Initialize a data frame of regions. Returns None if no valid regions
            are specified. 
        
        Keyword Arguments
        -----------------
        - include_iso: include iso code?
        - include_region_wb_key: if `regions_wb == True`, then set to True to
            keep the region_wb key as a column
        - regions: list of regions to use to build the data frame
        - regions_wb: optional list of world bank regions to use to filter
        - use_iso: 
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


    
    def get_closest_region(self,
        region: str,
        missing_flag: float = -999,
        regions_valid: Union[List[str], None] = None,
        type_input: str = "region",
        type_return: str = "region",
    ) -> Union[str, None]:
        """
        Based on latitude/longitude of population centers, find the 
            closest neighboring region.
        

        Function Arguments
        ------------------
        - region: region to search for closest neighbor
        - attr_region: attribute table for regions
        
        Keyword Arguments
        -----------------
        - field_iso: iso field in attr_regin
        - field_lat: field storing latitude
        - field_lon: field storing longitude
        - missing_flag: flag indicating a missing value
        - regions_valid: optional list of regions to restrict search to. If None,
            searches through all regions specified in attr_region
        - type_input: input region type. Either "region" or "iso"
        - type_return: return type. Either "region" or "iso"
        """
        
        ##  INITIALIZATION
        attr_region = self.attributes
        type_return = "region" if (type_return not in ["region", "iso"]) else type_return
        type_input = "region" if (type_input not in ["region", "iso"]) else type_input
        
        # check region/lat/lon
        region = self.dict_iso_to_region.get(region) if (type_input == "iso") else region
        region = region if (region in attr_region.key_values) else None
        coords = self.get_coordinates(region)
        
        # return None if one of the dimensions is missing
        if (coords is None) or (region is None):
            return None

        lat, lon = coords
        
        
        ##  FILTER TABLE AND APPLY DISTANCES
        
        if (regions_valid is None):
            regions_valid = attr_region.key_values 
        else:
            regions_valid = (
                [x for x in attr_region.key_values if x in (regions_valid)]
                if type_input == "region"
                else [x for x in attr_region.key_values if self.dict_region_to_iso.get(x) in (regions_valid)]
            )
            
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
            out = self.dict_region_to_iso.get(out) if (type_return == "iso") else out


        return out



    def get_coordinates(self,
        region: Union[str, None],
    ) -> Union[Tuple[float, float], None]:
        """
        Return the latitude, longitude coordinates of the population centroid of
            region `region`. `region` can be entered as a region (one of the 
            self.attributes.key_values) or the ISO3 code. If neither is found, 
            returns None

        Function Arguments
        ------------------
        - region_str: region string; either region or ISO can be entered
        """
        
        dict_region_to_lat = self.attributes.field_maps.get(f"{self.attributes.key}_to_{self.field_lat}")
        dict_region_to_lon = self.attributes.field_maps.get(f"{self.attributes.key}_to_{self.field_lon}")

        # check region
        region = (
            self.dict_iso_to_region.get(region)
            if region not in self.all_regions
            else region
        )

        if region is None:
            return None

        # if valid, get coordinates
        tuple_out = (dict_region_to_lat.get(region), dict_region_to_lon.get(region))

        return tuple_out


    
    def get_valid_regions(self,
        regions: Union[List[str], str, None],
    ) -> Union[List[str], None]:
        """
        Enter a list (or list-like object) iteratable of regions, a single
            region, or None (to return all valid regions), and return a list of
            valid regions.

        Function Arguments
        ------------------
        - regions: list-like object of regions, a string specifying a region, or
            None (to return all valid regions)
        """
        regions = [regions] if isinstance(regions, str) else regions
        regions = self.all_regions if (regions is None) else regions
        regions = (
            [x for x in self.all_regions if x in regions]
            if sf.islistlike(regions)
            else None
        )
        regions = None if (len(regions) == 0) else regions

        return regions



    def get_un_region(self,
        region: str
    ) -> Union[str, None]:
        """
        Retrieve the UN global region associated with region (roughly a 
            continent). Often used for assigning regional averages.
        """
        region = self.return_region_or_iso(region, return_type = "region")
        out = self.dict_region_to_un_region.get(region)

        return out



    def get_world_bank_region(self,
        region: str
    ) -> Union[str, None]:
        """
        Retrieve the World Bank global region associated with region. Often used 
            for assigning regional averages.
        """
        region = self.return_region_or_iso(region, return_type = "region")
        out = self.dict_region_to_wb_region.get(region)

        return out



    def return_region_or_iso(self,
        region: Union[int, str, List[str]],
        clean_inputs: bool = False,
        return_none_on_missing: bool = False,
        return_type: str = "region",
        try_iso_numeric_as_string: bool = True,
    ) -> Union[str, None]:
        """
        Return region for region entered as region or ISO. 
        
        * Returns None if `region` is specified improperly, or, if 
            `return_none_on_missing = True`, the region is properly specified
            but not found.

        Function Arguments
        ------------------
        - region: region, iso 3-digit alpha code, or iso numeric code

        Keyword Arguments
        -----------------
        - clean_inputs: try to clean the input region first
        - return_none_on_missing: set to True to return None if input region 
            `region` is not found
        - return_type: "region" or "iso". Will return: 
            * Region if set to "region" or 
            * ISO Alpha 3 if set to "iso"
            * ISO Numeric (integer) if set to "iso_numeric"
        - try_iso_numeric_as_string: if a region is not found in any of the 
            three inputs (region, iso, iso_numeris), will check to see if it is 
            an iso_numeric code entered as a string.
        """
        return_type = (
            "region" 
            if (return_type not in ["region", "iso", "iso_numeric"]) 
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
                input_type = "region"
            elif r in self.all_isos:
                input_type = "iso"
            elif try_iso_numeric_as_string & isinstance(r, str):
                try:
                    r = int(r)
                except:
                    None
                
            if (r in self.all_isos_numeric) & (input_type is None):
                input_type = "iso_numeric"
            
            # if input not found, continue
            if input_type is None:
                region[i] = None if return_none_on_missing else r
                continue

            
            # otherwise, retrieve in terms of key - if r is a region, then the dictionary will not contain it
            region_full = (
                self.dict_iso_to_region.get(r)
                if input_type == "iso"
                else self.dict_iso_numeric_to_region.get(r, r)
            )

            # based on return type, get output
            out = region_full
            if return_type == "iso":
                out = self.dict_region_to_iso.get(region_full)
            elif return_type == "iso_numeric":
                out = self.dict_region_to_iso_numeric.get(region_full)
        
            region[i] = out
            

        """
        if (return_type == "region"):
            dict_retrieve = self.dict_iso_to_region 

        dict_retrieve = (
            self.dict_iso_to_region 
            if (return_type == "region") 
            else self.dict_region_to_iso
        )
        all_vals = (
            self.all_regions 
            if (return_type == "region") 
            else self.all_isos
        )


        region = [
            (self.clean_region(x) if clean_inputs else x) 
            for x in region
        ]
        
        # check region
        region = [
            (
                dict_retrieve.get(x)
                if x not in all_vals
                else x
            )
            for x in region
        ]
        """;

        region = region[0] if element_input else region

        return region



    def add_region_or_iso_field(self,
        df: pd.DataFrame,
        field_iso: Union[str, None] = None,
        field_region: Union[str, None] = None,
        **kwargs,
    ) -> Union[str, None]:
        """
        Return region for region entered as region or ISO.

        Function Arguments
        ------------------
        - df: DataFrame to which to add region or iso field

        Keyword Arguments
        -----------------
        - field_iso: field in df storing ISO code OR field to add with ISO code
            (if not in df)
        - field_region: field in df storing region OR field to add with region
            (if not in df)
        - **kwargs: passed to return_region_or_iso
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
            else Nonea
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
        model_attributes: ModelAttributes
    ):

        self._initialize_time_properties(model_attributes)

        return None
    


    def _initialize_time_properties(self,
        model_attributes: ModelAttributes,
        field_year: str = "year",
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






class Transformation:
    """
    Create a Transformation class to support construction in sectoral 
        transformations. 

    Initialization Arguments
    ------------------------
    - code: strategy code associated with the transformation. Must be defined in 
        attr_strategy.table[field_strategy_code]
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    - attr_strategy: AttributeTable usd to define strategies from 
        ModelAttributes

    Keyword Arguments
    -----------------
    - field_strategy_code: field in attr_strategy.table containing the strategy
        codes
    - field_strategy_name: field in attr_strategy.table containing the strategy
        name
    """
    
    def __init__(self,
        code: str,
        func: Union[Callable, List[Callable]],
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str = "strategy_code",
        field_strategy_name: str = "strategy",
    ):
        
        self._initialize_function(func)
        self._initialize_code(
            code, 
            attr_strategy, 
            field_strategy_code,
            field_strategy_name
        )
        
    
    
    def __call__(self,
        *args,
        **kwargs
    ) -> Any:
        
        val = self.function(
            *args,
            strat = self.id,
            **kwargs
        )

        return val
    




    def _initialize_code(self,
        code: str,
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str,
        field_strategy_name: str,
    ) -> None:
        """
        Initialize the transformation name. Sets the following
            properties:

            * self.baseline 
                - bool indicating whether or not it represents the baseline 
                    strategy
            * self.code
            * self.id
            * self.name
        """
        
        # initialize and check code/id num
        id_num = (
            attr_strategy.field_maps.get(f"{field_strategy_code}_to_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        id_num = id_num.get(code) if (id_num is not None) else -1

        if id_num is None:
            raise ValueError(f"Invalid strategy code '{code}' specified in support_classes.Transformation: strategy not found.")

        id_num = id_num if (id_num is not None) else -1

        # initialize and check name/id num
        name = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_{field_strategy_name}")
            if attr_strategy is not None
            else None
        )
        name = name.get(id_num) if (name is not None) else ""

        # check baseline
        baseline = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_baseline_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        baseline = (baseline.get(id_num, 0) == 1)


        ##  set properties

        self.baseline = bool(baseline)
        self.code = str(code)
        self.id = int(id_num)
        self.name = str(name)
        
        return None

    
    
    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None


        if isinstance(func, list):

            func = [x for x in func if callable(x)]

            if len(func) > 0:  
                
                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformation function for {self.name}
                    """
                    out = None
                    if len(args) > 0:
                        out = (
                            args[0].copy() 
                            if isinstance(args[0], pd.DataFrame) | isinstance(args[0], np.ndarray)
                            else args[0]
                        )

                    for f in func:
                        out = f(out, **kwargs)

                    return out

                function = function_out
                function_list = func

        elif callable(func):
            function = func
            function_list = [func]


        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None
        


class YAMLConfiguration:
    """
    Initialize a configuration from a YAML file. 

    Initialization Arguments
    ------------------------
    - fp: file path to YAML file to read in
    """
    def __init__(self,
        fp: str,
    ):
        
        self._initialize_data(fp)

        return None
        
    
        
    def _initialize_data(self,
        fp: str,
    ) -> None:
        """
        Read the yaml dictionary. Sets the following properties:
        
            * self.dict_yaml
            * self.path
        """
        
        try:
            dict_yaml = sf.read_yaml(fp, munchify_dict = False)
        except Exception as e:
            raise RuntimeError(f"Error initializing YAML dictionary in yaml_config: {e}")
        

        ##  SET PROPERTIES

        self.dict_yaml = dict_yaml
        self.path = fp
        
        return None
            


    def get(self,
        key: str,
        delim: str = ".",
    ) -> Any:
        """
        Allow for recursive retrieval of dictionary values. Nested keys
            are stored using delimiters.

        Function Arguments
        ------------------
        - key: key that represents YAML nesting. Levels are seperated by delim, 
            e.g., to access

            dict_yaml.get("level_1").get("level_2")

            use 

            YAMLConfig.get("level_1.level_2")

        Keyword Arguments
        -----------------
        - delim: delimeter to use in get
        """
        
        return_none = not (isinstance(key, str) & isinstance(delim, str))
        return_none |= not isinstance(self.dict_yaml, dict)
        if return_none:
            return None
        
        # split keys into path and initialize value
        keys_nested = key.split(delim)
        value = self.dict_yaml
        
        for key in keys_nested:
            
            value = value.get(key)
    
            if not isinstance(value, dict):
                break
        
        value = (
            value
            if key == keys_nested[-1]
            else None
        )

        return value
        



    
    
    