
import datetime
import geopy.distance
import numpy as np
import pandas as pd
import pytz
import suntime
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.utilities._toolbox as sf



####################################
###                              ###
###    BUILD SOME REGION INFO    ###
###                              ###
####################################


class region_solar:
    """
    Generate solar sunrise-sunset by time of year and use to develop diurnal 
        irradiance scalars for capacity factors and demand scalars. 

    Initialization Arguments
    ------------------------
    - iso: ISO Alpha 3 code
    - lat: representative latitude to use for the region
    - lon: representative longitude to use for the region

    Notes
    -----
    - Leverages the suntime package (https://pypi.org/project/suntime/) to 
        generate sun up/sun down.
    - Baseline Assumption about 3.5 from 0.5 hr after sunrise to peak, then 2 hr 
        from peak to 0.5 before sunset
        * https://www.researchgate.net/figure/The-start-stop-and-peak-of-solar-generation-red-approximates-the-time-of-sunrise_fig1_310821721
    """
    
    def __init__(self,
        iso: str,
        lat_pop: float,
        lon_pop: float
    ):
        self.iso = iso
        self.lat_pop = lat_pop
        self.lon_pop = lon_pop
        
        self._init_sun()
        self._init_timezone()
        
    


    def _init_sun(self,
    ) -> None:
        """
        Initialize the suntime.Sun time object. Sets the following properties:
        
            * self.sun

        NOTE: self.sun.get_sunrise_time() and self.sun.get_sunset_time() return
            times in UTC; if using outside of the region class, must conver to 
            the proper time zone. 
        """
        self.sun = suntime.Sun(
            self.lat_pop, 
            self.lon_pop
        )

        return None



    def _init_timezone(self,
        lat: Union[float, None] = None,
        lon: Union[float, None] = None,
    ) -> None:
        """
        Initialize timezone objects. Sets the following properties:

            * self.timezone_finder
            * self.timezone_pytz_tzfile
            * self.timezone_str

        Keyword Arguments
        -----------------
        - lat: optional specification of latitude. If None, default to 
            self.lat_pop
        - lon: optional specification of longitude. If None, default to 
            self.lon_pop
        """

        lat = self.lat_pop if (lat is None) else lat
        lon = self.lon_pop if (lon is None) else lon

        timezone_finder = tzf.TimezoneFinder()
        timezone_str = timezone_finder.timezone_at(
            lat = lat,
            lng = lon
        )

        # get the pytz timezone object, which includes a UTC offset
        timezone_pytz_tzfile = pytz.timezone(timezone_str)

        self.timezone_finder = timezone_finder
        self.timezone_pytz_tzfile = timezone_pytz_tzfile
        self.timezone_str = timezone_str

        return None



    ###########################
    #    SUPPORT FUNCTIONS    #
    ###########################

    def get_datetime(self,
        *args,
        direction: str = "utc_to_native"
    ) -> datetime.datetime:
        """
        Return date time that is shifted to the correct time zone. 

        Function Arguments
        ------------------
        - *args: passed to datetime.datetime
        - direction: convert from utc to native time zone ("utc_to_native") or 
            native time zone to utc ("native_to_utc")
        """
        dt = datetime.datetime(*args)
        delta = self.timezone_pytz_tzfile.utcoffset(dt)
        
        direction = "utc_to_native" if (direction not in ["utc_to_native", "native_to_utc"]) else direction
        sign = -1 if (direction == "native_to_utc") else 1

        # convert date time as UTC
        dt_out = dt + sign*delta

        return dt_out




    ########################
    #    CORE FUNCTIONS    #
    ########################

    def build_solar_cf_seasonal_component_by_hour(self,
        attributes: Union[ModelAttributes, None],
        attr_hour: Union[AttributeTable, None] = None,
        attr_time_period: Union[AttributeTable, None] = None,
        attr_ts_group_1: Union[AttributeTable, None] = None,
        explode_by_year: bool = False,
    ) -> pd.DataFrame:
        """
        Build a data frame, wide by ts_group_1 (months/seasons) and long by hour, 
            that gives the seasonality component for the capacity factor 

        Function Arguments
        ------------------
        - attributes: ModelAttributes object used to extract default 
        
        Keyword Arguments
        -----------------
        - attr_hour: Hour attribute table
        - attr_time_period: Time Period attribute table (NemoMod)
        - attr_ts_group_1: Time Slice Group 1 attribute table (NemoMod)
        - explode_by_year: if True, returns a data frame long by year. If False,
            uses average over years specified in attr_time_period
        """

        return_none = (attributes is None)
        return_none &= (attr_hour is None)
        return_none &= (attr_time_period is None)
        return_none &= (attr_ts_group_1 is None)

        return_none |= (self.lat_pop is None) | (self.lon_pop is None)

        if return_none:
            return None

        # get defaults from model attributes
        attr_time_period = (
            attributes.get_dimensional_attribute_table(attributes.dim_time_period)
            if (attributes is not None) and (attr_time_period is None)
            else attr_time_period
        )
        attr_ts_group_1 = (
            attributes.get_other_attribute_table(f"ts_group_1")
            if (attributes is not None) and (attr_ts_group_1 is None)
            else attr_ts_group_1
        )
        attr_hour = (
            attributes.get_other_attribute_table(f"hour")
            if (attributes is not None) and (attr_hour is None)
            else attr_hour
        )
        
        # get set of years and time_slice_group_1 elements
        field_year = "year"
        dict_tp_to_year = attr_time_period.field_maps.get(f"{attr_time_period.key}_to_{field_year}")
        dict_tsg1_to_months = attr_ts_group_1.field_maps.get(f"{attr_ts_group_1.key}_to_months")
        dict_tsg1_to_months = dict((k, [int(x) for x in v.split("|")]) for k, v in dict_tsg1_to_months.items())


        ##  ITERATE OVER TIME PERIODS TO GENERATE SUNSET/SUNRISE TIMES BY 

        n_months = 12
        hrs = np.arange(24).astype(int)
        yrs = [dict_tp_to_year.get(x) for x in attr_time_period.key_values]

        df_base = pd.DataFrame({attr_hour.key: list(range(24))})
        df_base = (
            sf.explode_merge(pd.DataFrame({field_year: yrs}), df_base)
            if explode_by_year
            else df_base
        )

        
        for tsg in dict_tsg1_to_months.keys():
            
            # initialize average and iterator for indices
            vec_col = np.zeros(len(df_base))
            i = 0

            for tp in attr_time_period.key_values:

                year = dict_tp_to_year.get(tp)

                mos = dict_tsg1_to_months.get(tsg)
                dpm = [sf.days_per_month((year, x)) for x in mos]
                midpoint_days = np.sum(dpm)/2
                
                mo_base = n_months if (min([x%n_months for x in mos]) == 0) else min(mos)
                yr_base = year - 1 if (min([x%n_months for x in mos]) == 0) else year
                
                date_base = datetime.date(yr_base, mo_base, 1)
                date_base += datetime.timedelta(days = midpoint_days)

                # build the curve
                vec_seasonality = self.build_solar_seasonality_curve(
                    date_base, 
                    return_type = "array"
                )

                inds = (
                    (i*len(hrs), (i + 1)*len(hrs))
                    if explode_by_year
                    else (0, len(hrs))
                )
                vec_col[inds[0]:inds[1]] += vec_seasonality

                i += 1
            
            vec_col /= (1 if explode_by_year else len(yrs))

            df_base[tsg] = vec_col

        df_base = (
            pd.merge(
                df_base,
                attr_hour.table[[attr_hour.key, "hour_group"]],
                how = "left"
            )
            .sort_values(by = [attr_hour.key])
            .reset_index(drop = True)
        )

        return df_base



    def build_solar_seasonality_curve(self,
        date_obj: datetime.date,
        h_shift_sunrise_to_first: float = 0.5,
        h_shift_last_to_sunet: float = 0.5,
        n_hrs_rise: int = 3.5,
        n_hrs_set: int = 2.0,
        **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Build a curve of hourly solar irradience factors

        Function Arguments
        ------------------
        - date_obj: datetime.date day object (datetime.date(Y, M, D))
        
        Assumes that if days are less than n_hrs_rise + n_hrs_set, then 100% 
            solar irradiance will not be met.

        NOTE: See https://www.researchgate.net/figure/The-start-stop-and-peak-of-solar-generation-red-approximates-the-time-of-sunrise_fig1_310821721
            for rough approximation of 3.5 hours to rise, 2.0 to set. 
            
        Keyword Arguments
        -----------------
        - h_shift_sunrise_to_first: assumed time difference between sunrise and
            first generation (in baseline assumed to be 30 minutes)
        - h_shift_last_to_sunet: assumed time difference between last generation
            and true sunset (in baseline assumed to be 30 minutes)
        - n_hrs_rise: time from 0% irradiance to 100%. Assumed to be 3.5 hours
        - n_hrs_set: time from 100% irradiance to 0%. Assumed to be 2 hours
        - **kwargs passed to 
            batch_data_support_regions.build_solar_seasonality_curve()

            Include:

            - h_shift: number of hours to shift the output hour to; e.g., if 
                h_shift == 0.5, then the factor is set to the center of the hour 
                (e.g., hour 6 would be associated with the factor at 06:30). Limited 
                to closed interval [0, 0.99]
            - n_hrs_per_day: number of hours per day
            - return_type: output type. Acceptable values are:
                * "array": vector of factors for hours 
                    0, 1, 2, ..., n_hrs_per_day -1
                * "data_frame": data frame long by hour
                    * fields are "hour" and "factor"
                * "data_frame_with_minutes" data frame long by hour and minute
                    * fields are "hour", "minute", and "factor"
        """
        dt_sunrise = self.sun.get_sunrise_time(date_obj)
        dt_sunset = self.sun.get_sunset_time(date_obj)

        h_0 = self.get_datetime(
            dt_sunrise.year, 
            dt_sunrise.month, 
            dt_sunrise.day, 
            dt_sunrise.hour,
            direction = "utc_to_native"
        ).hour
        h_1 = self.get_datetime(
            dt_sunset.year, 
            dt_sunset.month, 
            dt_sunset.day, 
            dt_sunset.hour,
            direction = "utc_to_native"
        ).hour

        # shift generation if valid
        if h_0 + h_shift_sunrise_to_first < h_1 - h_shift_last_to_sunet:
            h_0 += h_shift_sunrise_to_first
            h_1 -= h_shift_last_to_sunet

        # build output
        out_val = build_solar_seasonality_curve(
            h_0,
            h_1,
            n_hrs_rise,
            n_hrs_set,
            **kwargs
        )

        return out_val



##########################
#    SHARED FUNCTIONS    #
##########################

# estimate max solar irradiance factor curve -- could be better set to be based off of solar arc
# see https://en.wikipedia.org/wiki/Sunrise_equation for potential improvement
def build_solar_seasonality_curve(
    h_0: int,
    h_1: int,
    n_hrs_rise: int,
    n_hrs_set: int,
    h_shift: float = 0.5,
    n_hrs_per_day: int = 24,
    return_type: str = "array" 
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Build a curve of hourly solar irradience factors based on:
    
        - h_0: time of sunrise
        - h_1: time of sunset
        - n_hrs_rise: time from 0% irradiance to 100%
        - n_hrs_set: time from 100% irradiance to 0%
    
    Assumes that if days are less than n_hrs_rise + n_hrs_set, then 100% solar 
        irradiance will not be met.
        
    Keyword Arguments
    -----------------
    - h_shift: number of hours to shift the output hour to; e.g., if 
        h_shift == 0.5, then the factor is set to the center of the hour (e.g., 
        hour 6 would be associated with the factor at 06:30). Limited to closed 
        interval [0, 0.99]
    - n_hrs_per_day: number of hours per day
    - return_type: output type. Acceptable values are:
        * "array": vector of factors for hours 
            0, 1, 2, ..., n_hrs_per_day -1
        * "data_frame": data frame long by hour
            * fields are "hour" and "factor"
        * "data_frame_with_minutes" data frame long by hour and minute
            * fields are "hour", "minute", and "factor"
    """
    
    # hours organization
    h_0 = min(h_0, h_1)
    h_1 = max(h_0, h_1)
    h_shift = float(sf.vec_bounds(h_shift, (0.0, 0.99)))
    delta_h = h_1 - h_0
    mins_per_hour = 60
    
    # set lower bound on rise/set length
    n_hrs_rise = max(n_hrs_rise, 1)
    n_hrs_set = max(n_hrs_set, 1)
    
    # some estimators
    total_hrs_rise_set = n_hrs_rise + n_hrs_set
    ratio_rise_to_total = n_hrs_rise/total_hrs_rise_set
    ratio_set_to_total = n_hrs_set/total_hrs_rise_set
    
    # get slopes
    m_0 = 1/n_hrs_rise
    m_1 = -1/n_hrs_set
    
    # value at hypothetical triangle peak, hours, and vector out initialization
    h_peak = h_0 + ratio_rise_to_total*delta_h
    value_at_peak = ratio_rise_to_total*delta_h*m_0
    hrs = list(range(n_hrs_per_day))
    vec_out = []
    
    # iterate over hours
    for h in hrs:
        h += h_shift
        if (h <= h_0) or (h >= h_1):
            vec_out.append(0)
        else:
            (
                vec_out.append(min(1, (h - h_0)*m_0))
                if h <= h_peak
                else vec_out.append(min(1, value_at_peak + (h - h_peak)*m_1))
            )

    
    if return_type in ["data_frame", "data_frame_with_minutes"]:
        
        field_factor = "factor"
        field_hr = "hour"
        field_min = "minute"
        
        # initialize output as dictionary
        vec_out = {field_hr: hrs, field_factor: vec_out}
        (
            vec_out.update({field_min: h_shift*mins_per_hour}) 
            if (return_type == "data_frame_with_minutes") 
            else None
        )
        
        # convert to dataframe
        vec_out = pd.DataFrame(vec_out)
        
    return vec_out



def get_closest_region(
    region: str,
    attr_region: AttributeTable, 
    field_iso: str = "iso_alpha_3",
    field_lat: str = "latitude_population_centroid_2020",
    field_lon: str = "longitude_population_centroid_2020",
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
    
    type_return = "region" if (type_return not in ["region", "iso"]) else type_return
    type_input = "region" if (type_input not in ["region", "iso"]) else type_input
    
    # get some dictionaries
    dict_region_to_lat = attr_region.field_maps.get(f"{attr_region.key}_to_{field_lat}")
    dict_region_to_lon = attr_region.field_maps.get(f"{attr_region.key}_to_{field_lon}")
    dict_iso_to_region = attr_region.field_maps.get(f"{field_iso}_to_{attr_region.key}")
    dict_region_to_iso = attr_region.field_maps.get(f"{attr_region.key}_to_{field_iso}")
    
    # check region/lat/lon
    region = dict_iso_to_region.get(region) if (type_input == "iso") else region
    region = region if (region in attr_region.key_values) else None
    lat, lon = dict_region_to_lat.get(region), dict_region_to_lon.get(region)
    
    # return None if one of the dimensions is missing
    if (lat is None) or (lon is None) or (region is None):
        return None
    
    
    ##  FILTER TABLE AND APPLY DISTANCES
    
    if (regions_valid is None):
        regions_valid = attr_region.key_values 
    else:
        regions_valid = (
            [x for x in attr_region.key_values if x in (regions_valid)]
            if type_input == "region"
            else [x for x in attr_region.key_values if dict_region_to_iso.get(x) in (regions_valid)]
        )
        
    df_regions = attr_region.table[
        attr_region.table[attr_region.key].isin(regions_valid)
    ].copy().reset_index(drop = True)
    
    # function to apply
    def f(tup: Tuple[float, float]) -> float:
        y, x = tuple(tup)
        
        out = (
            -1.0
            if (min(y, lat) < -90) or (max(y, lat) > 90) or (min(x, lon) < -180) or (max(x, lon) > 180)
            else geopy.distance.geodesic((lat, lon), (y, x)).km
        )
        
        return out
    

    vec_dists = np.array(df_regions[[field_lat, field_lon]].apply(f, raw = True, axis = 1))
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
        out = dict_region_to_iso.get(out) if (type_return == "iso") else out


    return out

