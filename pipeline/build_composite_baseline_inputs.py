"""
Build composite inputs using raw data from a number of sources, including the
    SISEPUEDE data repository, batch data repository, fake data, and more.
"""
import numpy as np
import pandas as pd
from typing import *


import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf



########################
#    SOME FUNCTIONS    #
########################

def project_model_variable_using_growth_rate(
    df_data: pd.DataFrame,
    growth_rate_by_time_period: float,
    time_periods_hist: List[int],
    modvar: str,
    model_attributes: ma.ModelAttributes,
    add_region: bool = False,
    field_iso: Union[str, None] = None,
    field_region: Union[str, None] = None,
    input_regions_only: bool = False,
    regions: Union[sc.Regions, None] = None,
    time_periods: Union[sc.TimePeriods, None] = None,
    **kwargs,
) -> Union[pd.DataFrame, None]:
    """
    Project a model variable
    
    Function Arguments
    ------------------
    - df_data: DataFrame containing input data 
    - growth_rate_by_time_period: growth rate to apply to post-historical
        time periods
    - time_periods_hist: time periods containing historical data (not projected 
        over)
    - modvar: model variable to project
    - model_attributes: ModelAttributes object spcifying 
    
    Keyword Arguments
    -----------------
    - add_region: add region field?
    - field_iso: field to use as ISO string
    - field_region: field storing regions/countries
    - input_regions_only: only run for regions associated with input file?
    - regions: optional support_classes.Regions object to pass
    - time_periods: optional support_classes.TimePeriods object to pass
    - **kwargs: passed to project_from_growth_rate()
    """
    
    # do some checks
    return_none = not isinstance(df_data, pd.DataFrame)
    return_none |= not sf.islistlike(time_periods_hist)
    return_none |= (model_attributes.get_variable(modvar) is None)
    if return_none:
        return None

    ##  INITIALIZATION

    regions = (
        sc.Regions(model_attributes)
        if not isinstance(regions, sc.Regions)
        else regions
    )
    time_periods = (
        sc.TimePeriods(model_attributes)
        if not isinstance(time_periods, sc.TimePeriods)
        else time_periods
    )

    # how to deal with regions that are missing?
    missing_regions_process = (
        None
        if input_regions_only
        else "substitute_closest"
    )


    ##  GET FIELDS AND BUILD FROM GROWTH RATE

    # initialize fields to project over
    flds = [field_iso]
    (
        flds.extend([time_periods.field_time_period])
        if (time_periods.field_time_period in df_data.columns)
        else None
    )
    flds += model_attributes.build_variable_fields(modvar)

    # return None if any required fields are not in the input dataframe
    s1 = set(flds) - set([time_periods.field_time_period])
    if not s1.issubset(set(df_data.columns)):
        return None
    
    
    # call project_from_growth_rate
    df_return = project_from_growth_rate(
        (
            df_data[
                df_data[time_periods.field_time_period].isin(time_periods_hist)
            ][flds]
            .reset_index(drop = True)
        ), 
        model_attributes,
        modvar,
        growth_rate_by_time_period,
        field_iso = field_iso,
        missing_regions_process = missing_regions_process,
        **kwargs
    )
    
    # add region to output
    if add_region:
        df_return = regions.add_region_or_iso_field(
            df_return,
            field_iso = field_iso,
            field_region = field_region,
        )
    
    return df_return



def project_from_growth_rate(
    df_hist: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    modvar: str,# = model_afolu.modvar_agrc_yf,
    growth_rate_by_time_period: float,# = 0.016,
    bounds: Union[Tuple, None] = None,
    field_iso: Union[str, None] = None,
    field_year: Union[str, None] = None,
    max_deviation_from_mean: float = 0.2,
    missing_regions_process: Union[str, None] = "substitute_closest",
    n_tp_lookback_max: Union[int, None] = None,
    regions: Union[sc.Regions, None] = None,
    time_periods: Union[sc.TimePeriods, None] = None,
) -> Union[pd.DataFrame, None]:
    """
    Apply an annual growth rate of growth_rate_by_time_period to historical 
        values specified in df_hist
        
    Function Arguments
    ------------------
    - df_hist: data frame containing historical yield factors
    - model_attributes: model attribute object for variable access
    - modvar: model variable containing yield factors
    - growth_rate_by_time_period: growth rate per time period

    Keyword Arguments
    -----------------
    - bounds: optional tuple specifying bounds for the projection. If None 
        (default), no bounds are established
    - field_iso: field in df_hist containing the iso code
    - field_year: field in df_hist containing the year
    - max_deviation_from_mean: maximum deviation from observed historical mean 
        allowed in first projection period (as fraction)
    - missing_regions_process: process for filling in data for regions that are
        missing. Options include:
        * None: ignore missing regions
        * "substitute_closest": using population centroids, substitute value 
            using closes region for which data are available
    - n_tp_lookback_max: number of lookback time periods to use for mean in 
        first projected time period
    - regions: optional support_classes.Regions object to pass
    - time_periods: optional support_classes.TimePeriods object to pass
    """
    
    regions = (
        sc.Regions(model_attributes)
        if not isinstance(regions, sc.Regions)
        else regions
    )
    time_periods = (
        sc.TimePeriods(model_attributes)
        if not isinstance(time_periods, sc.TimePeriods)
        else time_periods
    )

    field_iso = (
        regions.field_iso
        if not isinstance(field_iso, str)
        else field_iso
    )
    field_year = (
        time_periods.field_year
        if not isinstance(field_year, str)
        else field_year
    )
    field_time_period = time_periods.field_time_period
    return_none = not (field_iso in df_hist.columns)

    # check if time periods need to be added
    df_hist_out = (
        time_periods.years_to_tps(
            df_hist
            .rename(
                columns = {field_year: time_periods.field_year}
            )
            .copy()
        )
        if field_time_period not in df_hist.columns
        else df_hist.copy()
    )
    return_none |= (df_hist_out is None)
    
    if return_none:
        return None

    time_periods_all = range(
        min(min(time_periods.all_time_periods), min(df_hist[field_time_period])),
        max(max(time_periods.all_time_periods), max(df_hist[field_time_period])) + 1
    )

    # get
    df_all = sf.explode_merge(
        regions.get_regions_df(include_iso = True),
        pd.DataFrame({time_periods.field_time_period: time_periods_all})
    )
    all_isos_defined = set(df_hist[field_iso])
    isos_in_hist = [x for x in regions.all_isos if x in all_isos_defined]
    isos_not_in_hist = [x for x in regions.all_isos if x not in all_isos_defined]
    
    # update
    if missing_regions_process is not None:

        if missing_regions_process == "substitute_closest":

            df_hist_append = [df_hist_out]

            for iso in isos_not_in_hist:
                iso_sub = regions.get_closest_region(
                    iso,
                    regions_valid = all_isos_defined,
                    type_input = "iso",
                    type_return = "iso",
                ) 
                df_copy = df_hist[
                    df_hist[field_iso].isin([iso_sub])
                ].copy()
                df_copy[field_iso] = iso
                df_hist_append.append(df_copy)

            df_hist_out = pd.concat(df_hist_append, axis = 0)
            

    # group by iso
    dfg = df_hist_out.groupby([field_iso])
    df_out = []

    for iso, df in dfg:
        
        iso = iso[0] if isinstance(iso, tuple) else iso

        fields = model_attributes.build_variable_fields(modvar)
        arr_cur = np.array(df[fields])
        
        # project first period after historical based on trend/mean 
        vec_base_proj = sf.project_from_array(
            arr_cur, 
            max_deviation_from_mean = max_deviation_from_mean,
            max_lookback = n_tp_lookback_max,
        )
        
        tp_hist_max = max(list(df[field_time_period]))
        tp_proj_max = max(time_periods_all)
        tp_proj = np.arange(tp_proj_max - tp_hist_max)

        vec_rates = (1 + growth_rate_by_time_period)**tp_proj
        vec_rates = np.outer(vec_rates, np.ones(arr_cur.shape[1]))
        vec_rates *= vec_base_proj

        vec_rates = (
            sf.vec_bounds(vec_rates, bounds)
            if isinstance(bounds, tuple)
            else vec_rates
        )
        
        df_append = pd.DataFrame(vec_rates, columns = fields)
        df_append[field_time_period] = tp_hist_max + tp_proj + 1
        df_append[field_iso] = iso
        
        df_out.append(
            pd.concat(
                [
                    df,
                    df_append[df.columns]
                ]
            )
            .reset_index(drop = True)
        )
        
    df_out = (
        pd.concat(df_out, axis = 0)
        .reset_index(drop = True)
    )
    
    return df_out