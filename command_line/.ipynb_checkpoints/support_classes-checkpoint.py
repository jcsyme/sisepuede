###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY 
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
        * and more
    """
    def __init__(self,
        model_attributes: ModelAttributes
    ):

        self._initialize_region_properties(model_attributes)

        # initialize some default data source properties
        self._initialize_defaults_iea()

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
            "czech_republic": "czechia",
            "korea": "republic_of_korea",
            "people's_republic_of_china": "china",
            "republic_of_turkiye": "turkey",
            "slovak_republic": "slovakia",
            "united_states": "united_states_of_america"
        }

        self.field_iea_balance = "Balance"
        self.field_iea_country = "Country"
        self.field_iea_product = "Product"
        self.field_iea_time = "Time"
        self.field_iea_unit = "Unit"
        self.field_iea_value = "Value"     

        return None

        

    def _initialize_region_properties(self,
        model_attributes: ModelAttributes,
        field_year: str = "year",
    ) -> None:
        """
        Set the following properties:

            * self.all_isos
            * self.all_region
            * self.all_wb_regions
            * self.attributes
            * self.dict_iso_to_region
            * self.dict_region_to_iso
            * self.dict_region_to_wb_region
            * self.dict_wb_region_to_region
            * self.field_iso
            * self.field_lat
            * self.field_lon
            * self.field_wb_global_region
            * self.key

        """
        # some fields
        field_iso = "iso_alpha_3"
        field_lat = "latitude_population_centroid_2020"
        field_lon = "longitude_population_centroid_2020"
        field_wb_global_region = "world_bank_global_region"

        # set attributes and some dictionaries
        attributes = model_attributes.dict_attributes.get(f"{model_attributes.dim_region}")

        # initialize ISO dictionaries
        dict_region_to_iso = attributes.field_maps.get(f"{attributes.key}_to_{field_iso}")
        dict_iso_to_region = attributes.field_maps.get(f"{field_iso}_to_{attributes.key}")
        all_isos = sorted(list(dict_iso_to_region.keys()))

        # WorldBank region dictionaries
        dict_wb_region_to_region = sf.group_df_as_dict(
            attributes.table,
            [field_wb_global_region],
            fields_out_set = attributes.key
        )
        dict_region_to_wb_region = attributes.field_maps.get(f"{attributes.key}_to_{field_wb_global_region}")
        all_wb_regions = sorted(list(dict_wb_region_to_region.keys()))

    
        # assign as properties
        self.all_isos = all_isos
        self.all_regions = attributes.key_values
        self.all_wb_regions = all_wb_regions
        self.attributes = attributes
        self.dict_region_to_iso = dict_region_to_iso
        self.dict_iso_to_region = dict_iso_to_region
        self.dict_region_to_wb_region = dict_region_to_wb_region
        self.dict_wb_region_to_region = dict_wb_region_to_region
        self.field_iso = field_iso
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
        region: str,
        return_type: str = "region",
    ) -> Union[str, None]:
        """
        Return region for region entered as region or ISO.

        Function Arguments
        ------------------
        - region: region or iso code

        Keyword Arguments
        -----------------
        return_type: "region" or "iso". Will return a region if set to "region" 
            or ISO if set to "iso"
        """
        return_type = "region" if (return_type not in ["region", "iso"]) else return_type
        dict_retrieve = self.dict_iso_to_region if (return_type == "region") else self.dict_region_to_iso
        all_vals = self.all_regions if (return_type == "region") else self.all_isos

        # check region
        region = (
            dict_retrieve.get(region)
            if region not in all_vals
            else region
        )

        return region




    ##  DATA SOURCE-SPECIFIC MODIFICATIONS

    def data_func_iea_get_isos_from_countries(self,
        df_in: Union[pd.DataFrame, List, np.ndarray, str],
        field_country: Union[str, None] = None,
    ) -> np.ndarray:
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
        """
       
        field_country = self.field_iea_country if (field_country is None) else field_country
        vec_iso = (
            list(df_in[field_country]) 
            if isinstance(df_in, pd.DataFrame) 
            else (
                [df_in] if isinstance(df_in, str) else df_in
            )
        )

        vec_iso = [x.lower().replace(" ", "_") for x in vec_iso]
        vec_iso = [self.dict_iea_countries_lc_to_regions.get(x, x) for x in vec_iso]
        vec_iso = [self.dict_region_to_iso.get(x, x) for x in vec_iso]
        
        return np.array(vec_iso).astype(str)







class TimePeriods:
    """
    Leverage some simple time period actions based on a model attributes
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

            * self.attributes
            * self.dict_time_period_to_year
            * self.dict_year_to_time_period
            * self.field_year
            * self.min_year
        """

        attributes = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
        dict_year_to_time_period = attributes.field_maps.get(f"{field_year}_to_{attributes.key}")
        dict_time_period_to_year = attributes.field_maps.get(f"{attributes.key}_to_{field_year}")
        
        year_max = max((dict_year_to_time_period.keys()))
        year_min = min((dict_year_to_time_period.keys()))

        self.attributes = attributes
        self.dict_time_period_to_year = dict_time_period_to_year
        self.dict_year_to_time_period = dict_year_to_time_period
        self.field_year = field_year
        self.year_max = year_max
        self.year_min = year_min

        return None


    def year_to_tp(self,
        year: int,
    ) -> int:
        """
        Convert a year to a time period
        """
        out = self.dict_year_to_time_period.get(
            year,
            year - self.year_min
        )

        return out
