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
class SimplexBoundsError(Exception):
    pass

class UndefinedErrorFunction(Exception):
    pass




_ERROR_FUNCTIONS = ef.ErrorFunctions()


_TSSS_TYPE_FRACTION_OF_SOURCE = "fraction_of_source"
_TSSS_TYPE_VECTOR_SCALAR = "scalar_vectors"


class TimeSeriesSimplexShifter:
    """Shift fractions in a time series.

    In all descriptions below, let

        * T:        number of time periods
        * n:        simplex dimension (number of categories)

    Allows for shifting of fractions from a source group to a target group 
        based on one of the following types:

        * fraction_of_source:
            Shift a fraction of source mass into target mass. If specifying,
            must specify target mass allocations for shifted mass. If None is 
            provided, will allocate proportionally to target mass.

            REQUIRED ARGUMENTS
            ------------------
            * source fractions                  # fraction of source mass 
                                                #   categories to be shifted
            * target allocations                # allocation fractions for
                                                #   target mass categories
                                                #   receiving shifted mass
        * scalar_vectors:
            Using a vector of scaling, adjust source mass categories and 
            allocated to target fuels accordingly.

            REQUIRED ARGUMENTS
            ------------------
            * dict_source_vector_scalars        # dictionary mapping source 
                                                #   indices (column) to scalars
                                                #   to apply. If a scalar is a
                                                #   single number, then that 
                                                #   number is applied uniformly.
                                                #   If it is a vector, it is 
                                                #   used to scale the source 
                                                #   category.
            * arr_target_allocations            # array (T x n) mapping target 
                                                #   indices to allocated share.
                                                #   Row sums must equal 1. If 
                                                #   None, then all non-source
                                                #   categories are targeted. 

        

        
    Initialization Arguments
    ------------------------
    array_base : np.narray (T x n)
        Baseline array of
    """

    def __init__(self,
        array_base: np.ndarray,
        thresh_correction: float = 1e-6,
    ) -> None:

        self._initialize_array_base(
            array_base,
            thresh_correction,
        )

        return None
    

    def _initialize_array_base(self,
        array_base: np.ndarray,
        thresh_correction: float,
    ) -> None:
        """Initialize the base array of share vectors
        """
        
        # check that everything is positive
        if array_base.min() < 0:
            raise ValueError(f"Values for time series array_base in TimeSeriesSimplexShifter cannot be negative.")

        # verify sums
        array_base = sf.check_row_sums(
            array_base, 
            msg_pass = "array_base in TimeSeriesSimplexShifter",
            thresh_correction = thresh_correction,
        )


        ##  SET PROPERTIES

        self.T = array_base.shape[0]
        self.n = array_base.shape[1]
        self.array_base = array_base

        return None



    def _initialize_magnitude_types(self,
    ) -> None:
        """Initialize magnitude types that can be passed.
        """

        magnitude_types = [
            _TSSS_TYPE_VECTOR_SCALAR
        ]

        
        ##  SET PROPERTIES

        self.magnitude_type_scalar_vector = _TSSS_TYPE_VECTOR_SCALAR
        self.magnitude_types = magnitude_types

        return None
    

    
    #####################################################################
    #    END-STATE FUNCTIONS                                            #
    #    -------------------------------                                #
    #    - functions that estimate the end-states for sources and/or    #     
    #       targets, depending on type.                                 #
    #                                                                   #
    #####################################################################

    def _get_end_state_vectors_sources_sv(self,
        dict_source_vector_scalars: Dict[int, np.ndarray],
        normalize_exceedance: bool = True,
        stop_on_bounds_error: bool = False, 
    ) -> Union[np.ndarray, None]:
        """Get the end-state vectors for SOURCE vectors in scalar_vectors shift
            type. Returns an array of size T x n where only SOURCE vectors are 
            set. All others are ignored.

        
        Function Arguments
        ------------------
        dict_source_vector_scalars : Dict[int, np.ndarray]
            Dictionary mapping a column index to a scalar vector to apply to 
            that index.

        Keyword Arguments
        -----------------
        normalize_exceedance : bool 
            Normalize total targets that exceed one? 
            * True:     If end states that are scaled exceed 1, they will be 
                        normalized to ensure that the simplex is preserved
            * False:    If end states exceed 1 in total, an error is thrown.  
        stop_on_bounds_error : bool
            Stop if negative scalars are found? If False, skips.
        """

        if not isinstance(dict_source_vector_scalars, dict):
            return None
        

        # continue 
        arr_out = np.zeros(self.array_base.shape, )

        for k, v in dict_source_vector_scalars.items():

            # conditions that need to be met--index
            skip = not sf.isnumber(k, integer = True, )
            skip |= ((k < 0) | (k >= self.n)) if not skip else skip            
            if skip: continue

            # conditions that need to be met--specification
            val = v*np.ones(self.T, ) if sf.isnumber(v) else v
            skip = not isinstance(val, np.ndarray)
            skip |= (val.shape != (self.T, )) if not skip else skip
            if skip: continue

            # check scalars
            if val.min() < 0: 
                if stop_on_bounds_error:
                    raise SimplexBoundsError(f"Unable to apply scalar at index {k}: negative scalars found.")
                continue

            # apply to column
            arr_out[:, k] = self.array_base[:, k]*val
        
        
        # check 
        if normalize_exceedance:
            arr_out = np.nan_to_num(
                sf.vector_limiter(arr_out, (0, 1)),
                nan = 0.0,
                posinf = 0.0,
            )
        
        else:
            vec_sum = arr_out.sum(axis = 1)
            w = np.where((vec_sum > 1) | (vec_sum < 0))[0]

            if len(w) > 0:
                msg = f"Some target mass fractions are either > 1 or < 0. Check the scalars provided or allow normalize_exceedance."
                raise SimplexBoundsError(msg)
        

        return arr_out











###########################
#    PRIMARY FUNCTIONS    #
###########################

def _clean_shift_into_field(
    vec: np.ndarray,
    effective_zero: float,
) -> np.ndarray:
    """Clean a target, shifted field.
    """
    
    vec_out = np.nan_to_num(
        vec,
        nan = 0.0,
        posinf = 0.0,
    )

    vec_out[np.abs(vec_out) < effective_zero] = 0.0

    return vec_out



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


"""
_DICT_SUBSEC_TO_DICT_FUEL_TO_MODVAR = {
    matt.subsec_name_inen: model_afolu.model_enercons.dict_inen_fuel_categories_to_fuel_variables,
    matt.subsec_name_scoe: model_afolu.model_enercons.get_scoe_dict_fuel_categories_to_fuel_variables()[0],
    matt.subsec_name_trns: model_afolu.model_enercons.dict_trns_fuel_categories_to_fuel_variables,
}
"""


def get_modvar_from_fuel(
    dict_fuel_to_modvar_by_type: Dict[str, Dict],
    fuel: str,
    model_attributes: 'ModelAttributes',
    key_fuel_fraction: str = "fuel_fraction",
) -> Union['ModelVariable', None]:
    """Get the ModelVariable associated with a fuel fraction
    """
    # get dictionary mapping--if not found, the fuel is not associated with the subsector
    dict_fuel_to_fuel_fraction = dict_fuel_to_modvar_by_type.get(fuel, )
    if dict_fuel_to_fuel_fraction is None:
        return None
        
    modvar_target = model_attributes.get_variable(
        dict_fuel_to_fuel_fraction.get(key_fuel_fraction, )
    )

    return modvar_target



def scale_fuel_fraction_from_vector(
    df_input: pd.DataFrame,
    subsector: str,
    dict_fuels_targ: Dict[str, Union[np.ndarray, float]],
    fuels_scale_response: Union[List[str], None],
    dict_fuel_to_modvars_by_subsec: Dict[str, Dict],
    model_attributes: 'ModelAttributes',
    cats_iter: Union[List[str], None] = None,
    effective_zero: float = 10.0**(-8.0),
    key_fuel_fraction: str = "fuel_fraction",
    vec_mass_ledger: Union[np.ndarray, None] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Using a vector of scalars for each time period, scale the fraction
        associated with a given fuel. Will scale all other fuels associated
        with fuels_scale_response uniformly. If None, all other fuels are 
        adjusted.

    Returns a tuple of the form:
        (
            df_candidate,       # DataFrame with fuels that have been shifted
            vec_mass_ledger,    # Vector with available mass after scaling
        )
        
    Function Arguments
    ------------------
    df_input : pd.DataFrame
        DataFrame of input rajectories. Returns a candidate *copy* of this 
        DataFrame
    subsector : str
        Subsector (energy consumption) to work in 
    dict_fuels_targ : Dict[str, Union[np.ndarray, float]]
        Dictionary mapping fuels to scalars, which are specified either as a
        float (same value applied to all time periods) or a vector (one value
        for each time period)
    fuels_shift_out : Union[List[str], None]
        Fuels to shift out of to hit target. If None, will shift out of all
        fuels not specified in the target set.
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
    effective_zero : float
        Value below which numbers are assumed to be 0
    key_fuel_fraction : str
        Key in dict_fuel_to_modvars_by_subsec.get(subector).get(_CAT_ENFU)
    vec_mass_ledger : Union[np.ndarray, None]
        Optional "ledger" for keeping track of mass available. Can be used as a
        passthrough to allow iterative shifting away from fuels
    """

    ##  INITIALIZE

    df_candidate = df_input.copy()
    
    # loop over subsector categories for which shift is made
    cat_element = model_attributes.get_subsector_attribute(
        subsector, 
        "pycategory_primary_element",
    )
    dict_specs_by_cat = {}

    # get attribute tables
    attr = model_attributes.get_attribute_table(subsector, )
    attr_enfu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_enfu, 
    )

    # some fuel modvars
    dict_fuel_to_modvar_by_type = dict_fuel_to_modvars_by_subsec.get(subsector, )
    if dict_fuel_to_modvar_by_type is None:
        raise RuntimeError(f"No fuel to modvar dictionary found for subsector '{subsector}'")


    # get fuels that can respond
    fuels_scale_response = (
        [x for x in attr_enfu.key_values if x in fuels_scale_response]
        if sf.islistlike(fuels_scale_response)
        else attr_enfu.key_values
    )
    if len(fuels_scale_response) == 0:
        return df_candidate
    
    fuels_stable_try = [
        x for x in attr_enfu.key_values if x not in fuels_scale_response + [fuel_targ]
    ]

    # get dictionary mapping--if not found, the fuel is not associated with the subsector
    modvar_target = get_modvar_from_fuel(
        dict_fuel_to_modvar_by_type,
        fuel_targ,
        model_attributes,
        key_fuel_fraction = key_fuel_fraction,
    )
    if modvar_target is None:
        return df_candidate



    ##  ITERATE BY CAT

    cats_iter = ( 
        attr.key_values
        if not sf.islistlike(cats_iter)
        else [x for x in attr.key_values if x in cats_iter]
    )

    for cat in cats_iter:

        # get the target field
        field_target = modvar_target.build_fields(
            category_restrictions = cat, 
        )
        if field_target is None: continue


        ##  GET TOTAL IN TARGET

        # need to make sure the new target does not exceed 1
        vec_target_original = df_candidate[field_target].to_numpy()
        vec_target_new = np.clip(
            vec_target_original*vec_scalars, 
            (0, 1),
        )


        ##  GET TOTAL MASS AVAIL TO SHIFT IN

        arr_mass = np.zeros(
            (
                df_candidate.shape[0],
                attr_enfu.n_key_values
            )
        )
        
        # indices for different groups
        inds_mass_stable = []
        inds_mass_scalable = []

        # iterate over each output fuel to retrieve how much is available
        for i, fuel in enumerate(attr_enfu.key_values, ):
            
            # get variable
            modvar_cur_frac = get_modvar_from_fuel(
                dict_fuel_to_modvar_by_type,
                fuel,
                model_attributes,
                key_fuel_fraction = key_fuel_fraction,
            )
            if modvar_cur_frac is None: continue

            # field and data
            field_frac_cur = modvar_cur_frac.build_fields(category_restrictions = cat, )
            arr_mass[:, i] = df_candidate[field_frac_cur].to_numpy()
            
            if fuel in fuels_stable_try: 
                inds_mass_stable.append(i)
                continue

            if fuel in fuels_scale_response:
                inds_mass_scalable.append(i)
                

        ##  CHECK STABLE FUELS TO SEE IF THEY NEED TO BE MADE SCALABLE

        # check the target to see if, with stable, it will exceed 1
        vec_target_new_with_stable_try = vec_target_new + arr_mass[:, inds_mass_stable].sum(axis = 1)
        vec_target_new_with_stable_try_clipped = np.clip(
            vec_target_new_with_stable_try, 0, 1,
        )

        # if so, just set stable classes to be scalable
        eps = np.abs(vec_target_new_with_stable_try_clipped - vec_target_new_with_stable_try).max()
        if eps > 0.0000001:
            inds_mass_scalable += inds_mass_stable 
            inds_mass_stable = []

        #




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
        arr_mass = np.nan_to_num(
            arr_mass,
            nan = 0.0,
            posinf = 0.0,
        )
        
        global ams
        ams = arr_mass.copy()

        arr_mass_shift = sf.do_array_mult(
            arr_mass,
            vec_mass_cur,
        )


        

        ##  EXECUTE THE SHIFT BY ITERATING OVER "OUT" FUELS

        
        #ams = arr_mass_shift.copy()
        w = np.where(np.isnan(arr_mass_shift))[0]
        if len(w) > 0:
            raise RuntimeError("Done")

        vec_mass_shift_to_target = np.zeros(arr_mass_shift.shape[0])

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
            
            # convert nans and small values
            df_candidate[field_cur] = _clean_shift_into_field(
                df_candidate[field_cur].to_numpy() - vec_shift_cur,
                effective_zero,
            )

        
        df_candidate[field_target] = _clean_shift_into_field(
            df_candidate[field_target].to_numpy() + vec_mass_shift_to_target,
            effective_zero,
        )
        
    return df_candidate







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
    effective_zero: float = 10.0**(-8.0),
    key_fuel_fraction: str = "fuel_fraction",
    vec_mass_ledger: Union[np.ndarray, None] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Using a single point in time comparison for target mixes, shift fuels 
        from one source to another.

    Returns a tuple of the form:
        (
            df_candidate,       # DataFrame with fuels that have been shifted
            vec_mass_ledger,    # Vector with available mass after scaling
        )
        
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
    effective_zero : float
        Value below which numbers are assumed to be 0
    key_fuel_fraction : str
        Key in dict_fuel_to_modvars_by_subsec.get(subector).get(_CAT_ENFU)
    vec_mass_ledger : Union[np.ndarray, None]
        Optional "ledger" for keeping track of mass available. Can be used as a
        passthrough to allow iterative shifting away from fuels
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


        ##  GET TOTAL IN TARGET

        # need to make sure the new target does not exceed 1
        vec_target_original = df_candidate[field_target].to_numpy()
        vec_target_new = sf.vec_bounds(vec_target_original*scalar_full, (0, 1))

        # get adjusted scalar and convert to mass that is to be shifted
        scalar = np.nan_to_num(
            vec_target_new/vec_target_original,
            nan = 1.0,
            posinf = 1.0,
        )[ind_tp]

        vec_mass_shift = vec_target_original*(scalar - 1)
        

        ##  GET TOTAL MASS AVAIL TO SHIFT IN

        arr_mass = np.zeros(
            (
                df_candidate.shape[0],
                len(fuels_shift_out, )
            )
        )
        
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
        arr_mass = np.nan_to_num(
            arr_mass,
            nan = 0.0,
            posinf = 0.0,
        )
        
        global ams
        ams = arr_mass.copy()

        arr_mass_shift = sf.do_array_mult(
            arr_mass,
            vec_mass_cur,
        )


        

        ##  EXECUTE THE SHIFT BY ITERATING OVER "OUT" FUELS

        
        #ams = arr_mass_shift.copy()
        w = np.where(np.isnan(arr_mass_shift))[0]
        if len(w) > 0:
            raise RuntimeError("Done")

        vec_mass_shift_to_target = np.zeros(arr_mass_shift.shape[0])

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
            
            # convert nans and small values
            df_candidate[field_cur] = _clean_shift_into_field(
                df_candidate[field_cur].to_numpy() - vec_shift_cur,
                effective_zero,
            )

        
        df_candidate[field_target] = _clean_shift_into_field(
            df_candidate[field_target].to_numpy() + vec_mass_shift_to_target,
            effective_zero,
        )
        
    return df_candidate



