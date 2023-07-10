"""
Build composite inputs using raw data from a number of sources, including the
    SISEPUEDE data repository, batch data repository, fake data, and more.
"""

import batch_data_support_general as bds_gen
import model_attributes as ma
import model_afolu as mafl
import model_ippu as mi
import model_circular_economy as mc
import model_electricity as ml
import model_energy as me
import model_socioeconomic as se
import setup_analysis as sa
import sisepuede_data_api as api
import support_classes as sc
import support_functions as sf
import numpy as np
import os, os.path
import pandas as pd
import time
from typing import *


########################
#    SOME FUNCTIONS    #
########################


def project_from_growth_rate(
    df_hist: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    modvar: str,# = model_afolu.modvar_agrc_yf,
    growth_rate_by_time_period: float,# = 0.016,
    field_iso: Union[str, None] = None,
    field_year: Union[str, None] = None,
    max_deviation_from_mean: float = 0.2,
    missing_regions_process: Union[str, None] = "substitute_closest",
    n_tp_lookback_max: int = 5,
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
    - field_iso: field in df_hist containing the iso code
    - field_year: field in df_hist containing the year
    - max_deviation_from_mean: maximum deviation from observed historical mean 
        allowed in first projection period (as fraction)
    - missing_regions_process: process for filling in data for regions that are
        missing. Options include:
        * None: ignore missing regions
        * "substitute_closest": using population centroids, substitute value 
            using closes region for which data are available
    - n_tp_lookback_max: number of lookback time periods to use for mean in first 
        projected time period
    """
    
    regions = sc.Regions(model_attributes)
    time_periods = sc.TimePeriods(model_attributes)

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
        
        fields = model_attributes.build_varlist(
            None, 
            modvar,
        )
        arr_cur = np.array(df[fields])
        
        # project first period after historical based on trend/mean 
        vec_base_proj = sf.project_from_array(
            arr_cur, 
            max_deviation_from_mean = max_deviation_from_mean,
        )
        
        tp_hist_max = max(list(df[field_time_period]))
        tp_proj_max = max(time_periods_all)
        tp_proj = np.arange(tp_proj_max - tp_hist_max)
        vec_rates = (1 + growth_rate_by_time_period)**tp_proj
        vec_rates = np.outer(vec_rates, np.ones(arr_cur.shape[1]))
        vec_rates *= vec_base_proj
        
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