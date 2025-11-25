import numpy as np
import pandas as pd
import sisepuede.calibration._error_functions as ef
import sisepuede.core.support_classes as sc
import sisepuede.manager.sisepuede_models as sm
import sisepuede.utilities._toolbox as sf
from typing import *



##########################
#    GLOBAL VARIABLES    #
##########################

# error classes
class UndefinedErrorFunction(Exception):
    pass




_ERROR_FUNCTIONS = ef.ErrorFunctions()






###########################
#    PRIMARY FUNCTIONS    #
###########################

def scale_inputs_single_value(
    df_in: pd.DataFrame,
    fields_input: List[str],
    fields_output: List[str],
    target_value: Union[Dict[int, float], int],
    models: 'SISEPUEDEModels',
    error_function: str = "proportional_deviation",
    include_energy_production: bool = False,
    threshold: float = 0.000001,
    **kwargs,
) -> pd.DataFrame:
    """Scale inputs to align output (sum) with target

    Function Arguments
    ------------------
    df_in  pd.DataFrame
        Input DataFrame containing fields_input. Fed to models
    fields_input : List[str]
        Input fields to scale
    fields_output:  List[str]
        Output fields to calibrate to
    target_value : Dict[int, float]
        Dictionary mapping a target time period to a target value. 
        If an integer, assumes only time_period 0.
    models : SISEPUEDEModels
        SISEPUEDEModels object used for calibration
        
    Keyword Arguments
    -----------------
    error_function : str
        Error function to use
    include_energy_production : bool
        Run with EnergyProduction model? Only set to True if calibrating 
        EnergyProduction variables.
    threshold : float
        Acceptance threshold for acceptance of candidate
    **kwargs :
        Passed to error function 
    """

    ##  INITIALIZATION
    
    matt = models.model_attributes
    field_time_period = matt.dim_time_period

    # check input
    if not sm.is_sisepuede_models(models):
        tp = str(type(models))
        raise TypeError(f"Unable to scale inputs: invalid type '{tp}' specified for models. Must be a SISEPUEDEModels object.")

    # format the target value as a dict
    if sf.isnumber(target_value):
        key = matt.get_baseline_scenario_id(field_time_period, )
        target_value = {key: target_value, }

    validated_tv, msg = sf.check_type(target_value, dict, write_message = True, )
    if not validated_tv:
        raise TypeError(f"Unable to scale inputs: {msg}")

    # extract info
    time_period_target = list(target_value.keys())[0]
    target = target_value.get(time_period_target)

    df_candidate = df_in.copy()


    ##  START PROCESS

    # get the time period
    vec_time_periods = df_candidate[field_time_period].to_numpy()
    ind = np.where(vec_time_periods == time_period_target)[0]
    if len(ind) == 0:
        raise RuntimeError(f"Time period {time_period_target} not found in the DataFrame.")
        
    ind = ind[0]
    
    # project the model onces
    df_projected_initial = models.project(
        df_candidate,
        include_electricity_in_energy = include_energy_production,
    )

    # get index to compare
    model_sum_unadjusted = df_projected_initial[fields_output].iloc[ind].sum()
    scalar = (
        target/model_sum_unadjusted
        if model_sum_unadjusted != 0
        else 0
    )
    df_candidate[fields_input] = df_candidate[fields_input]*scalar

    # project again and recalculate
    df_projected_final = models.project(
        df_candidate,
        include_electricity_in_energy = include_energy_production,
    )
    model_sum_adjusted = df_projected_final[fields_output].iloc[ind].sum()


    ##  CALCULATE ERROR
    
    error = _ERROR_FUNCTIONS(
        error_function,
        target,
        model_sum_adjusted,
        **kwargs,
    )

    if error > threshold:
        raise RuntimeError(f"Unable to rescale inputs: error {error} exceeds acceptance threshold {threshold}.")

    return df_candidate


