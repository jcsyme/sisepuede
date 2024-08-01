import model_attributes as ma
import model_ippu as mi
import model_socioeconomic as se
import numpy as np
import pandas as pd
import support_functions as sf
import transformations_base_general as tbg
from typing import *




##################################
###                            ###
###    IPPU TRANSFORMATIONS    ###
###                            ###
##################################

##############
#    IPPU    #
##############

def transformation_ippu_scale_emission_factor(
    df_input: pd.DataFrame,
    dict_magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_ippu: Union[mi.IPPU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Scale emission factors for gasses. See description of `dict_magnitude` for
        information on various ways to modify gasses.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_magnitude: scalar applied to gas emission factors by final time 
        period (e.g., to reduce the emission factor by 90%, enter 0.1). Keys can
        be specified in any of the three ways:

        * keys are gasses that map to scalar values: If using this method,
            only applies the transformation to gasses specified as keys in
            dict_magnitude
        * keys are collections of gasses specified in 
            `IPPU.dict_fc_ef_modvars_by_type` (derived from cat_industry
            attribute table):
            
            If using this approach, the following keys are accepted:

                + hfcs: all HFC gasses
                + none: all gasses that are non-fluorinated compounds
                + other_fcs: all fluorinated compounds that are not HFCs or
                    PFCs. Incudes SF6 and NF3
                + pfcs: all PFCs

        * keys are model variables: enter the emission factor model variable 
            as a key associated with the scalar as a value

    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to (applies to all 
        variables/gasses entered)
    - field_region: field in df_input that specifies the region
    - model_ippu: optional IPPU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    ##  INITIAlIZATION

    if not isinstance(dict_magnitude, dict):
        # LOGGING
        return df_input
    attr_gas = model_attributes.get_other_attribute_table("emission_gas").attribute_table
    attr_ippu = model_attributes.get_attribute_table(model_attributes.subsec_name_ippu)

    # initialize IPPU model and dictionary of emission factors by FC specification
    model_ippu = (
        mi.IPPU(model_attributes) 
        if model_ippu is None
        else model_ippu
    )
    dict_fc_type_to_modvars = model_ippu.dict_fc_ef_modvars_by_type

    # set model variables as a dictionary
    dict_transform = {}

    for k, v in dict_magnitude.items():
        
        modvars = None
        if k in attr_gas.key_values:
            modvars = model_ippu.dict_gas_to_fc_ef_modvars.get(k)    
        elif k in dict_fc_type_to_modvars.keys():
            modvars = dict_fc_type_to_modvars.get(k)
        elif k in model_attributes.all_variables:
            modvars = [k]

        # skip if no valid model variables are found
        if modvars is None:
            continue

        for modvar in modvars:
            cats_modvar = model_attributes.get_variable_categories(modvar)
            cats = (
                (
                    [x for x in attr_ippu.key_values if (x in categories) & (x in cats_modvar)]
                    if (categories is not None)
                    else attr_ippu.key_values
                )
                if cats_modvar is not None
                else None
            )

            dict_transform.update({
                modvar: {
                    "bounds": (0, np.inf),
                    "categories": cats,
                    "magnitude": float(sf.vec_bounds(v, (0.0, np.inf))),
                    "magnitude_type": "baseline_scalar",
                    "time_period_baseline": tbg.get_time_period(model_attributes, "max"),
                    "vec_ramp": vec_ramp
                }
            })
   

    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        dict_transform,
        **kwargs
    )


    return df_out



def transformation_ippu_reduce_demand(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_ippu: Union[mi.IPPU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Demand" transformations.

    NOTE: THIS IS CURRENTLY INCOMPLETE AND REQUIRES ADDITIONAL INTEGRATION
        WITH SUPPLY-SIDE IMPACTS OF DECREASED WASTE (LESS CONSUMER CONSUMPTION)


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying decrease as proprtion of final value (e.g.,
        a 30% reduction is entered as 0.3) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional IPPU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    attr_ippu = model_attributes.get_attribute_table(model_attributes.subsec_name_ippu)
    bounds = (0, 1)

    model_ippu = (
        mi.IPPU(model_attributes) 
        if model_ippu is None
        else model_ippu
    )
    
    modvar = model_ippu.modvar_ippu_scalar_production

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 - v, bounds)))
            for k, v in magnitude.items()
        )
    )
    
    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        bounds = bounds,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out