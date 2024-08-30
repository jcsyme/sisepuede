import sisepuede.core.model_attributes as ma
import sisepuede.models.afolu as mafl
import sisepuede.models.ippu as mi
import sisepuede.models.circular_economy as mc
import sisepuede.models.energy_production as ml
import sisepuede.models.energy_consumption as me
import sisepuede.models.socioeconomic as se
import numpy as np
import pandas as pd
import sisepuede.utilities._toolbox as sf
import transformations_base_general as tbg
from typing import *




##############################################
###                                        ###
###    CIRCULAR ECONOMY TRANSFORMATIONS    ###
###                                        ###
##############################################

##############
#    TRWW    #
##############

def transformation_trww_increase_gas_capture(
    df_input: pd.DataFrame,
    dict_magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase biogas capture at aerobic and anaerobic wastewater treatment 
        facilities.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_magnitude: dictionary mapping categories (TRWW) to viable capture 
        fractions by the final time period OR a float. If float, applies the 
        value uniformly to all available categories.
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # check category specification
    categories = model_attributes.get_variable_categories(model_circecon.modvar_trww_rf_biogas_recovered)
    

    # check dict magnitude
    if isinstance(dict_magnitude, dict):

        df_out = df_input.copy()

        for cat, v in dict_magnitude.items():

            if (cat not in categories) or not sf.isnumber(v):
                continue

            # call general transformation
            df_out = tbg.transformation_general(
                df_out,
                model_attributes,
                {
                    model_circecon.modvar_trww_rf_biogas_recovered: {
                        "bounds": (0, 1),
                        "categories": [cat],
                        "magnitude": float(sf.vec_bounds(v, (0.0, 1.0))),
                        "magnitude_type": "final_value_floor",
                        "vec_ramp": vec_ramp
                    }
                },
                **kwargs
            )
        

    elif sf.isnumber(dict_magnitude):
   
        df_out = tbg.transformation_general(
            df_input,
            model_attributes,
            {
                model_circecon.modvar_trww_rf_biogas_recovered: {
                    "bounds": (0, 1),
                    "categories": categories,
                    "magnitude": float(sf.vec_bounds(dict_magnitude, (0.0, 1.0))),
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_ramp
                }
            },
            **kwargs
        )


    return df_out



def transformation_trww_increase_septic_compliance(
    df_input: pd.DataFrame,
    dict_magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase septic sludge compliance; produces more sludge, which can be used
        for fertilizer or treated similarly to any other solid waste.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_magnitude: dictionary mapping categories (TRWW) to viable capture 
        fractions by the final time period OR a float. If float, applies the 
        value uniformly to all available categories.
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # check category specification
    categories = model_attributes.get_variable_categories(model_circecon.modvar_trww_septic_sludge_compliance)
    

    # check dict magnitude
    if isinstance(dict_magnitude, dict):

        df_out = df_input.copy()

        for cat, v in dict_magnitude.items():

            if (cat not in categories) or not sf.isnumber(v):
                continue

            # call general transformation
            df_out = tbg.transformation_general(
                df_out,
                model_attributes,
                {
                    model_circecon.modvar_trww_septic_sludge_compliance: {
                        "bounds": (0, 1),
                        "categories": [cat],
                        "magnitude": float(sf.vec_bounds(v, (0.0, 1.0))),
                        "magnitude_type": "final_value_floor",
                        "vec_ramp": vec_ramp
                    }
                },
                **kwargs
            )
        

    elif sf.isnumber(dict_magnitude):
   
        df_out = tbg.transformation_general(
            df_input,
            model_attributes,
            {
                model_circecon.modvar_trww_septic_sludge_compliance: {
                    "bounds": (0, 1),
                    "categories": categories,
                    "magnitude": float(sf.vec_bounds(dict_magnitude, (0.0, 1.0))),
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_ramp
                }
            },
            **kwargs
        )


    return df_out



##############
#    WALI    #
##############

def transformation_wali_improve_sanitation(
    df_input: pd.DataFrame,
    category: str,
    dict_magnitude: Dict[str, float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Improve Sanitation" transformations for wastewater treatment.
        Use `category` to set for urban, rural, and industrial. Specify target
        magnitudes using dict_magnitude (see below)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - category: WALI (Liquid Waste) category to apply transformation for
    - dict_magnitude: target allocation, across TRWW (Wastewater Treatment) 
        categories (categories are keys), of treatment as total fraction. 
        * E.g., to acheive 80% of treatment from advanced anaerobic and 10% from
            scondary aerobic by the final time period, the following dictionary 
            would be specified:

            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_anaerobic": 0.1
            }

    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    ##  INITIALIZATION

    # check wali category
    category = model_attributes.get_valid_categories(
        [category],
        model_attributes.subsec_name_wali
    )
    # check model variable category specifications
    categories_trww = model_attributes.get_valid_categories(
        list(dict_magnitude.keys()),
        model_attributes.subsec_name_trww
    )
    if (categories_trww is None) or (category is None):
        # LOGGING
        return df_input
    
    category = category[0]

    # check dict magnitude
    dict_magnitude = dict(
        (k, float(sf.vec_bounds(v, (0.0, 1.0)))) 
        for k, v in dict_magnitude.items() 
        if k in categories_trww
        and sf.isnumber(v)
    )
    magnitude = sum(dict_magnitude.values())
    if magnitude > 1:
        # LOGGING
        return df_input

    # get model if needed
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )


    ##  BUILD TRANSFORMATION DICTIONARY AND CALL GENERAL

    # set of all modvars to use as source + iteration initialization
    dict_cats_to_modvars = model_circecon.dict_trww_categories_to_wali_fraction_variables
    dict_transformations = {}
    modvars = []

    for cat, v in dict_cats_to_modvars.items():
    
        modvar = v.get("treatment_fraction")

        # update model variable domain for transfer
        (
            modvars.append(modvar)
            if modvar is not None
            else None
        )

        # get model variables to use as target
        mag = dict_magnitude.get(cat)
        (
            dict_transformations.update({
                modvar: mag/magnitude
            })
            if mag is not None
            else None
        )


    # build allocation
    df_out = tbg.transformation_general_shift_fractions_from_modvars(
        df_input,
        magnitude,
        modvars,
        dict_transformations,
        vec_ramp,
        model_attributes,
        categories = [category],
        **kwargs,
    )

    return df_out



##############
#    WASO    #
##############

def transformation_waso_decrease_municipal_waste(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Decrease Municipal Waste" transformations.

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
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # do some initializtion
    bounds = (0, 1)
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )
    modvar = model_circecon.modvar_waso_waste_per_capita_scalar

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



def transformation_waso_decrease_municipal_waste_base(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Decrease Municipal Waste" transformations.

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
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    attr_waso = model_attributes.get_attribute_table(model_attributes.subsec_name_waso)
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # call to general transformation differs based on whether magnitude is a number or a dictionary
    if sf.isnumber(magnitude):
        
        magnitude = float(sf.vec_bounds(1 - magnitude, (0.0, 1.0)))

        # check category specification
        categories = model_attributes.get_valid_categories(
            categories,
            model_attributes.subsec_name_waso
        )
        if categories is None:
            # LOGGING
            return df_input
            
        # apply same magnitude to all categories
        df_out = tbg.transformation_general(
            df_input,
            model_attributes,
            {
                model_circecon.modvar_waso_waste_per_capita_scalar: {
                    "bounds": (0, 1),
                    "categories": categories,
                    "magnitude": magnitude,
                    "magnitude_type": "baseline_scalar",
                    "vec_ramp": vec_ramp
                }
            },
            **kwargs
        )

    if isinstance(magnitude, dict):

        # invert the dictionary map
        dict_rev = sf.reverse_dict(magnitude, allow_multi_keys = True)
        df_out = df_input.copy()

        # iterate over separately defined magnitudes
        for mag, cats in dict_rev.items():
            
            cats = [cats] if (not isinstance(cats, list)) else cats

            # check categories
            cats = model_attributes.get_valid_categories(
                cats,
                model_attributes.subsec_name_waso
            )
        
            if cats is None:
                continue

            mag = float(sf.vec_bounds(1 - mag, (0.0, 1.0)))

            # call general transformation
            df_out = tbg.transformation_general(
                df_out,
                model_attributes,
                {
                    model_circecon.modvar_waso_waste_per_capita_scalar: {
                        "bounds": (0, 1),
                        "categories": cats,
                        "magnitude": mag,
                        "magnitude_type": "baseline_scalar",
                        "vec_ramp": vec_ramp
                    }
                },
                **kwargs
            )
    

    return df_out



def transformation_waso_increase_anaerobic_treatment_and_composting(
    df_input: pd.DataFrame,
    magnitude_biogas: float,
    magnitude_compost: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    rebalance_fractions: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Composting" and "Increase Biogas" transformations.

    NOTE: These are contained in one function because they interact with each
        other; the value of is restricted to (interval notation)
        
            magnitude_biogas + magnitude_compost \element [0, 1]


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude_biogas: proportion of organic solid waste that is treated using
        anaerobic treatment
    - magnitude_compost: proportion of organic solid waste that is treated using
        compost
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - rebalance_fractions: rebalance magnitude_compost and magnitude_biogas if 
        they exceed one?
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    # check total that is specified
    m_total = magnitude_biogas + magnitude_compost
    if (m_total > 1) and rebalance_fractions:
        magnitude_biogas /= m_total
        magnitude_compost /= m_total
        m_total = 1

    if (m_total > 1) | (m_total < 0):
        # LOGGING
        return df_input
    
    # get attribute table, CircularEconomy model for variables, and check categories
    attr_waso = model_attributes.get_attribute_table(model_attributes.subsec_name_waso)
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_waso
    )
    if categories is None:
        # LOGGING
        return df_input

    
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            # biogas
            model_circecon.modvar_waso_frac_biogas: {
                "bounds": (0, 1),
                "categories": categories,
                "magnitude": magnitude_biogas,
                "magnitude_type": "final_value_floor",
                "vec_ramp": vec_ramp
            },
            # compost
            model_circecon.modvar_waso_frac_compost: {
                "bounds": (0, 1),
                "categories": categories,
                "magnitude": magnitude_compost,
                "magnitude_type": "final_value_floor",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out



def transformation_waso_increase_energy_from_biogas(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase use of captured biogas for energy (APPLIES TO LANDFILLS ONLY AT 
        MOMENT)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary with keys for "landfill" mapping to the proportion 
        of gas generated at landfills (respectively) that is collective for
        energy use.If float, applies to all available biogas collection 
        groupings.

        NOTE: Set up as dictionary to allow for future expansion to include
            biogas from anaerobic treatment for energy

    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )
    

    ##  BUILD TRANSFORMATION FOR BIOGAS/LANDFILL

    dict_key_to_modvar = {
        "landfill": model_circecon.modvar_waso_frac_landfill_gas_ch4_to_energy
    }

    dict_transformation = {}
    for key, modvar in dict_key_to_modvar.items():
        # get the current magnitude of gas capture
        mag = (
            magnitude.get(key)
            if isinstance(magnitude, dict)
            else (magnitude if sf.isnumber(magnitude) else None)
        )

        (
            dict_transformation.update(
                {
                    modvar: {
                        "bounds": (0, 1),
                        "magnitude": mag,
                        "magnitude_type": "final_value_floor",
                        "vec_ramp": vec_ramp
                    }
                }
            )
            if mag is not None
            else None
        )


    # call general transformation
    df_out = (
        tbg.transformation_general(
            df_input,
            model_attributes,
            dict_transformation,
            **kwargs
        )
        if len(dict_transformation) > 0
        else df_input
    )

    return df_out



def transformation_waso_increase_energy_from_incineration(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase gas capture at anaerobic treatment and landfill facilities.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary with keys for "isw" or "msw" that map to the
        proportion of incinerated ISW/MSW that is captured for energy OR float.
        If float, applies to all available incinerated groupings
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )
    

    ##  BUILD TRANSFORMATION FOR BIOGAS/LANDFILL

    dict_key_to_modvar = {
        "isw": model_circecon.modvar_waso_frac_recovered_for_energy_incineration_isw,
        "msw": model_circecon.modvar_waso_frac_recovered_for_energy_incineration_msw,
    }

    dict_transformation = {}
    for key, modvar in dict_key_to_modvar.items():
        # get the current magnitude of gas capture
        mag = (
            magnitude.get(key)
            if isinstance(magnitude, dict)
            else (magnitude if sf.isnumber(magnitude) else None)
        )

        (
            dict_transformation.update(
                {
                    modvar: {
                        "bounds": (0, 1),
                        "magnitude": mag,
                        "magnitude_type": "final_value_floor",
                        "vec_ramp": vec_ramp
                    }
                }
            )
            if mag is not None
            else None
        )


    # call general transformation
    df_out = (
        tbg.transformation_general(
            df_input,
            model_attributes,
            dict_transformation,
            **kwargs
        )
        if len(dict_transformation) > 0
        else df_input
    )

    return df_out



def transformation_waso_increase_gas_capture(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase gas capture at anaerobic treatment and landfill facilities.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary with keys for "landfill" or "biogas" that map to the
        proportion of landfill and/or biogas (respectively) captured OR float.
        If float, applies to all available gas captures
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )


    ##  BUILD TRANSFORMATION FOR BIOGAS/LANDFILL

    dict_key_to_modvar = {
        "biogas": model_circecon.modvar_waso_rf_biogas,
        "landfill": model_circecon.modvar_waso_rf_landfill_gas_recovered,
    }

    dict_transformation = {}
    for key, modvar in dict_key_to_modvar.items():
        # get the current magnitude of gas capture
        mag = (
            magnitude.get(key)
            if isinstance(magnitude, dict)
            else (magnitude if sf.isnumber(magnitude) else None)
        )

        (
            dict_transformation.update(
                {
                    modvar: {
                        "bounds": (0, 1),
                        "magnitude": mag,
                        "magnitude_type": "final_value_floor",
                        "vec_ramp": vec_ramp
                    }
                }
            )
            if mag is not None
            else None
        )


    # call general transformation
    df_out = (
        tbg.transformation_general(
            df_input,
            model_attributes,
            dict_transformation,
            **kwargs
        )
        if len(dict_transformation) > 0
        else df_input
    )

    return df_out



def transformation_waso_increase_landfilling(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Landfilling" transformation (all non-recycled,
        non-biogas, and non-compost waste ends up in landfills)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportion of waste in landfills by final time period 
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # setup model variables, don't switch out of incineration
    modvars_domain_ignore = [
        model_circecon.modvar_waso_frac_nonrecycled_incineration
    ]
    modvars_domain = [
        x for x in model_circecon.modvars_waso_frac_non_recyled_pathways
        if x not in modvars_domain_ignore
    ]

    df_out = tbg.transformation_general_shift_fractions_from_modvars(
        df_input,
        magnitude,
        modvars_domain,
        {
            model_circecon.modvar_waso_frac_nonrecycled_landfill: 1.0
        },
        vec_ramp,
        model_attributes,
        **kwargs,
    )

    return df_out



def transformation_waso_increase_recycling(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Recycling" transformation (affects industrial 
        production in integrated environment)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportion of recyclable solid waste that is recycled by 
        final time period
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_waso
    )
    if categories is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_circecon.modvar_waso_frac_recycled: {
                "bounds": (0, 1),
                "categories": categories,
                "magnitude": magnitude,
                "magnitude_type": "final_value_floor",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



