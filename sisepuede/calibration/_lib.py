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
    return_projected: bool = False,
    threshold: float = 0.000001,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
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
    return_projected : bool
        Return the projected data frame as well. If True, returns a tuple

        (
            df_candidate,
            df_projected,
        )

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


    out = (
        (df_candidate, df_projected_final, )
        if return_projected
        else df_candidate
    )

    return out



def shift_fuels_based_on_single_point(
    df_input: pd.DataFrame,
    subsector: str,
    fuel_targ: str,
    fuels_shift_out: List[str],
    ind_tp: int,
    scalar_full: float,
    dict_fuel_to_modvars_by_subsec: Dict[str, Dict],
    model_attributes: 'ModelAttributes',
    cats_iter: Union[List[str], None] = None,
    key_fuel_fraction: str = "fuel_fraction",
) -> pd.DataFrame:
    """Using a single point in time comparison for target mixes, shift fuels 
        from one source to another.

    Function Arguments
    ------------------
    df_input : pd.DataFrame
        DataFrame of input rajectories. Returns a candidate *copy* of this 
        DataFrame
    subsector : str
        Subsector (energy consumption) to work in 
    fuel_targ : str
        Target fuel to apply scalar to
    fuels_shift_out : List[str]
        Fuels to shift out of to hit target
    ind_tp : int
        Row index associated with the time period to target
    scalar_full : float
        Scalar to apply to fuel_targ in time stored at ind_tp
    dict_fuel_to_modvars_by_subsec : Dict[str, Dict]
        Dictionary mapping each subsector to the dictionary that maps each fuel 
        to ModelVariables associated with different components (e.g., fuel 
        fraction, etc.)

        e.g., 

        dict_fuel_to_modvars_by_subsec = {
            matt.subsec_name_inen: EnergyConsumption.dict_inen_fuel_categories_to_fuel_variables,
            matt.subsec_name_scoe: EnergyConsumption.get_scoe_dict_fuel_categories_to_fuel_variables()[0],
            matt.subsec_name_trns: EnergyConsumption.dict_trns_fuel_categories_to_fuel_variables,
        }
    model_attributes : ModelAttributes
        ModelAttributes object used for variable management

    Keyword Arguments
    -----------------
    cats_iter : Union[List[str], None]
        Optional list of categories to iterate over. If None, will use all
        available in subsector
    key_fuel_fraction : str
        Key in dict_fuel_to_modvars_by_subsec.get(subector).get(_CAT_ENFU)
    """

    ##  INITIALIZE

    df_candidate = df_input.copy()
    
    # loop over INEN vars to shift
    cat_element = model_attributes.get_subsector_attribute(
        subsector, 
        "pycategory_primary_element",
    )
    dict_specs_by_cat = {}

    # subsector attribute table
    attr = model_attributes.get_attribute_table(subsector, )

    # some fuel modvars
    dict_fuel_to_modvar_by_type = dict_fuel_to_modvars_by_subsec.get(subsector, )
    if dict_fuel_to_modvar_by_type is None:
        raise RuntimeError(f"No fuel to modvar dictionary found for subsector '{subsector}'")

    
    # get dictionary mapping--if not found, the fuel is not associated with the subsector
    dict_fuel_to_fuel_fraction = dict_fuel_to_modvar_by_type.get(fuel_targ, )
    if dict_fuel_to_fuel_fraction is None:
        return df_candidate
        
    modvar_target = model_attributes.get_variable(
        dict_fuel_to_fuel_fraction.get(key_fuel_fraction, )
    )


    ##  ITERATE BY CAT

    cats_iter = (
        attr.key_values
        if not sf.islistlike(cats_iter)
        else [x for x in attr.key_values if x in cats_iter]
    )

    for cat in cats_iter:

        # get the target (electricity) field
        field_target = modvar_target.build_fields(category_restrictions = cat, )
        if field_target is None: continue


        ##  GET TOTAL IN ELEC
    
        vec_target_original = df_candidate[field_target].to_numpy()
        vec_target_new = sf.vec_bounds(vec_target_original*scalar_full, (0, 1))

        scalar = np.nan_to_num(
            vec_target_new/vec_target_original,
            nan = 1.0,
            posinf = 1.0,
        )[ind_tp]

        # get mass that needs to be shifted
        vec_mass_shift = vec_target_original*(scalar - 1)
        

        ##  GET TOTAL AVAIL

        arr_mass = np.zeros(
            (
                df_candidate.shape[0],
                len(fuels_shift_out, )
            )
        )

        #print(dict_fuel_to_modvar_by_type)
        
        # iterate over each output fuel to retrieve how much is available
        for i, fuel in enumerate(fuels_shift_out, ):

            # try getting subdicts for fuel--pass if not defined
            v = dict_fuel_to_modvar_by_type.get(fuel, )
            if v is None: continue

            # try getting the fuel fraction modvar
            modvar = model_attributes.get_variable(v.get(key_fuel_fraction))
            if modvar is None: continue

            # get current total
            field_cur = modvar.build_fields(category_restrictions = cat, )
            if field_cur is None: continue 

            arr_mass[:, i] = df_candidate[field_cur].to_numpy().copy()


        # cap shift at available
        vec_mass_cur = arr_mass.sum(axis = 1)
        vec_mass_cur = sf.vec_bounds(
            vec_mass_shift,
            [(-vec_target_original[i], x) for i, x in enumerate(vec_mass_cur)]
        )
        
        # convert to allocation 
        arr_mass = sf.check_row_sums(
            arr_mass,
            thresh_correction = None,
        )
        arr_mass_shift = sf.do_array_mult(
            arr_mass,
            vec_mass_cur,
        )

   
        # now, execute the shift by iterating over "output" fuels
        vec_mass_shift_to_target = 0.0
        for i, fuel in enumerate(fuels_shift_out, ):
            
            # try getting subdicts for fuel--pass if not defined
            v = dict_fuel_to_modvar_by_type.get(fuel, )
            if v is None: continue

            # try getting the fuel fraction modvar
            modvar = model_attributes.get_variable(v.get(key_fuel_fraction))
            if modvar is None: continue

            # get current total
            field_cur = modvar.build_fields(category_restrictions = cat, )
            if field_cur is None: continue

            vec_shift_cur = arr_mass_shift[:, i]
            vec_mass_shift_to_target += vec_shift_cur
            
            df_candidate[field_cur] = df_candidate[field_cur].to_numpy() - vec_shift_cur
        
         
        # update elec
        df_candidate[field_target] = df_candidate[field_target].to_numpy() + vec_mass_shift_to_target

    return df_candidate