###
###    LIBRARY TO STANDARDIZE SOME PROJECTION METHODS
###

import pandas as pd
from typing import *


import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf


############################
#    PROJECTION METHODS    #
############################

def repeat_value(
    df_to_project: pd.DataFrame,
    time_periods_project: List[int],
    n_time_periods_lookback: int,
    method: str,
    field_time_period: str,
    fields_data: List[str],
    fields_group: Union[List[str], None] = None,
    include_historical: bool = True,
    max_error: Union['Real', None] = None,
) -> pd.DataFrame:
    """Using a historical band (lookback) of recent data, to generate a value 
        and repeat going forward. Options include

        * use a mean (including the last value) over `n_time_periods_lookback`
        * regress over the lookback to identify a target value (useful if 
            capturing a trend), then repeat


    Function Arguments
    ------------------
    df_to_project : pd.DataFrame
        DataFrame to project forth using historical repeat
    time_periods_project : List[int]
        Time periods to project to; must exclude historical data.
    method : str
        One of the following:
        * "linear_regression"
        * "mean"
    n_time_periods_lookback : int
        Number of time periods to use for mean. If 1, uses last available year; 
        if 2, uses mean of last 2 years, etc.
    field_time_period : str
        Field storing time index
    fields_data : List[str]
        Fields to project

    Keyword Arguments
    -----------------
    fields_group : Union[List[str], None]
        Optional specification of grouping fields
    include_historical : bool
        Return a data frame with the historical data included.
    max_error : Union['Real', None]
        Maximum deviation, as error, from mean allowed. Can be used to ensure 
        regression projections--which may be derived from noisy data--do not 
        unreasonably exceed historical means based on short-term trends
    """

    ##  INITIALIZATION AND CHECKS

    if len(time_periods_project) == 0:
        return df_project
    
    method = "mean" if (method not in ["mean", "linear_regression"]) else method
    n_time_periods_lookback = max(n_time_periods_lookback, 1)

    # initialize fields
    fields_group = [] if not sf.islistlike(fields_group) else list(fields_group)
    fields_check = fields_group + [field_time_period] + fields_data

    # check whether the fields are present
    sf.check_fields(
        df_to_project, 
        fields_data,
    )

    
    ##  GROUP AND ITERATE

    df_group = (
        df_to_project.groupby(fields_group)
        if len(fields_group) > 0
        else [(None, df_to_project)]
    )

    df_out = []

    for grouping, df in df_group:
        
        # get available time periods 
        all_tp_available = sorted(list(df[field_time_period].unique()))
        tp_max_available = max(all_tp_available)

        tps_proj = [x for x in time_periods_project if x not in all_tp_available]
        tp_max_proj = max(tps_proj)
        n_lookback = min(n_time_periods_lookback, len(all_tp_available))
        n_proj = len(tps_proj)

        # if to_time_period is in the 
        if n_proj == 0:
            df.append(
                df[
                    df[field_time_period] <= tp_max_available
                ]
                .reset_index(drop = True)
            )

            continue

        # values to repeat
        n_reps = tp_max_proj - tp_max_available
        
        df_rep = (
            df
            .sort_values(by = [field_time_period])
            .get(fields_data)
            .iloc[-n_lookback:]
            .reset_index(drop = True)
        )

        if method == "mean":
            df_rep = (
                df_rep
                .rolling(n_lookback)
                .mean()
                .dropna()
            )

        elif method == "linear_regression":
            
            arr = sf.project_from_array(
                df_rep.to_numpy(),
                max_deviation_from_mean = max_error,
            )

            df_rep = pd.DataFrame(arr[None,...], columns = fields_data)


        # tactic for projections--repeat the data frame from the base to the 
        # max future, then filter out any tps that are not needed
        df_rep = sf.repeat_df(
            df_rep, 
            n_reps
        )

        df_rep[field_time_period] = list(range(tp_max_available + 1, tp_max_proj + 1))

        df_rep = (
            df_rep[
                df_rep[field_time_period].isin(tps_proj)
            ]
        )

        # add in group info if needed
        if grouping is not None:
            for k, field in enumerate(fields_group):
                df_rep[field] = grouping[k]

        # finally, concat with historical df if needed
        if include_historical:
            df_rep = (
                pd.concat(
                    [
                        df,
                        df_rep[df.columns]
                    ],
                    axis = 0
                )
            )
        
        df_out.append(df_rep)

    df_out = pd.concat(df_out).reset_index(drop = True)

    return df_out








