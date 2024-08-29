
import numpy as np
import pandas as pd
from typing import *


import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.models.afolu as mafl
import sisepuede.transformers.lib.general as tbg
import sisepuede.utilities.support_functions as sf




###################################
###                             ###
###    AFOLU TRANSFORMATIONS    ###
###                             ###
###################################

##############
#    AGRC    #
##############

def transformation_agrc_improve_crop_residue_management(
    df_input: pd.DataFrame,
    magnitude_burned: float,
    magnitude_removed: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    rescale: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Improve Crop Residue Management" transformation.

    NOTE: 0 <= (magnitude_burned + magnitude_removed) <= 1


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude_burned: total fraction of crops burned by final time period
    - magnitude_removed: total fraction of crops removed and used elsewhere by
        final time period
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional specification of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - rescale: rescale magnitude_burned na magnitude_removed if they exceed 1
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )

    magnitude_burned = (
        float(sf.vec_bounds(magnitude_burned, (0.0, 1.0)))
        if sf.isnumber(magnitude_burned)
        else None
    )

    magnitude_removed = (
        float(sf.vec_bounds(magnitude_removed, (0.0, 1.0)))
        if sf.isnumber(magnitude_removed)
        else None
    )

    # rescale
    if (magnitude_removed + magnitude_burned > 1):
        if not rescale:
            # logging
            return df_input

        scalar_div = max(1, magnitude_removed + magnitude_burned)
        magnitude_burned /= scalar_div
        magnitude_removed /= scalar_div

    # check categories
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_agrc
    )
    if categories is None:
        # LOGGING
        return df_input


    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_afolu.modvar_agrc_frac_residues_burned: {
                "bounds": (0, 1),
                "categories": categories,
                "magnitude": magnitude_burned,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            },

            model_afolu.modvar_agrc_frac_residues_removed: {
                "bounds": (0, 1),
                "categories": categories,
                "magnitude": magnitude_removed,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out



def transformation_agrc_improve_rice_management(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Improve Rice Management" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional reduction, by final time period, in methane 
        emitted from rice production--e.g., to reduce methane from rice by 
        30%, enter 0.3
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, (0.0, 1.0)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_afolu.modvar_agrc_ef_ch4: {
                "bounds": (0, np.inf),
                "categories": ["rice"],
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_agrc_increase_crop_productivity(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Crop Productivity" transformation (increases yield
        factors)


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying increase as proprtion of final value (e.g.,
        a 30% increase is entered as 0.3) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)HEREHERE
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_agrc_yf
    bounds = (0, np.inf)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 + magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 + v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_agrc,
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out



def transformation_agrc_increase_no_till(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Set a new floor for F_MG (as described in in V4 Equation 2.25 (2019R)). Used
        to Implement the "Expand Conservation Agriculture" transformation, which 
        reduces losses of soil organic carbon through no-till. Can be 
        implemented in cropland and grassland. 


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying target value of F_{MG} (e.g. per Table 5.5 in 
        V4, Chapter 5 [Croplands], no-till can increase F_{MG} to 1.1 under 
        certain conditions) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_agrc_frac_no_till
    bounds = (0.0, np.inf)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_lndu
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general - set as floor (don't want to make it worse)
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "final_value_floor",
        **kwargs
    )

    return df_out



def transformation_agrc_reduce_supply_chain_losses(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Supply Chain Losses" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional minimum reduction, from final time period, in 
        supply chain losses--e.g., to reduce supply chain losses by 30%, enter 
        0.3
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, (0.0, 1.0)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_afolu.modvar_agrc_frac_production_lost: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out





##############
#    FRST    #
##############

def transformation_frst_reduce_deforestation(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Reduce deforestion by stopping transitions out of forest land use 
        categories.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying target fraction of forests that remain forests 
        (applied to all forest categories)
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    - **kwargs: passed to 
        transformation_support_lndu_transition_to_category_targets_single_region()
    """
    
    # initialization
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )
    
    magnitude = (
        float(sf.vec_bounds(magnitude, (0.0, 1.0)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    
    ##  SETUP DICTIONARY--shift 
    cat_frst_primary = model_afolu.cat_frst_prim
    cat_lndu_frst_primary = model_afolu.dict_cats_frst_to_cats_lndu.get(cat_frst_primary)
    cats_lndu_forest = [
        x for x in attr_lndu.key_values
        if x in model_afolu.dict_cats_lndu_to_cats_frst.keys()
    ]

    # instantiate as preservation of forests
    dict_magnitude = dict(
        (
            (x, x), 
            {
                "magnitude": magnitude,
                "magnitude_type": "final_value_floor"
            }
        )
        for x in cats_lndu_forest
    )

    # add on preservation of cross-forest flows
    dict_magnitude.update(
        dict(
            (
                (x, y),
                {
                    "magnitude": 1,
                    "magnitude_type": "baseline_scalar"
                }
            )
            for x in cats_lndu_forest
            for y in cats_lndu_forest
            if (x != y)
            and (x != cat_lndu_frst_primary)   
        )
    )


    # get region
    field_region_def = "nation"
    region_default = "DEFAULT"
    field_region = kwargs.get("field_region", field_region_def)
    field_region = field_region_def if (field_region is None) else field_region
    #(
    #    field_region_def
    #    if "field_region" not in kwargs.keys()
    #    else kwargs.get("field_region")
    #)


    # organize and group
    df_group = df_input.copy()
    use_fake_region = (field_region not in df_group.columns)
    if use_fake_region:
        df_group[field_region] = region_default
    df_group = df_group.groupby([field_region])
    
    df_out = None
    i = 0
    
    for region, df in df_group:
        
        region = region[0] if isinstance(region, tuple) else region

        # PER-REGION MODS HERE 
        df_trans = transformation_support_lndu_specify_transitions(
            df,
            dict_magnitude,
            vec_ramp,
            model_attributes,
            model_afolu = model_afolu,
            **kwargs,
        )

        if df_out is None:
            df_out = [df_trans for k in range(len(df_group))]
        else:
            
            df_out[i] = df_trans
            
        i += 1
        

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    (
        df_out.drop([field_region], axis = 1, inplace = True)
        if use_fake_region
        else None
    )
    
    return df_out




def transformation_frst_increase_reforestation(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_inflow_restriction: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Reduce deforestion by stopping transitions out of forest land use 
        categories.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying increase in reforestation as a fraction of 
        final value
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - cats_inflow_restriction: optional specification of inflow categories used
        to restrict transitions into secondary forest
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    - **kwargs: passed to 
        transformation_support_lndu_transition_to_category_targets_single_region()
    """
    
    # build
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu,
    )
    
    magnitude = (
        float(sf.vec_bounds(magnitude, (0.0, np.inf)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    
    ##  SETUP DICTIONARY--shift 
    cat_fsts = model_afolu.cat_lndu_fsts
    cat_crop = model_afolu.cat_lndu_crop
    cat_grsl = model_afolu.cat_lndu_grass

    cats_ir = (#[cat_crop, cat_grsl]
        [x for x in attr_lndu.key_values if x in cats_inflow_restriction and x != cat_fsts]
        if sf.islistlike(cats_inflow_restriction)
        else [x for x in attr_lndu.key_values if x != cat_fsts]

    )
    
    dict_magnitude = {
        cat_fsts: {
            "categories_inflow_restrict": cats_ir,
            "magnitude": magnitude + 1,
            "magnitude_type": "baseline_scalar"
        }
    }
    
    cats_stable = [
        model_afolu.cat_lndu_stlm,
        model_afolu.cat_lndu_wetl,
        model_afolu.cat_lndu_fstp,
    ]
    
    # get region
    field_region_def = "nation"
    region_default = "DEFAULT"
    field_region = kwargs.get("field_region", field_region_def)
    field_region = field_region_def if (field_region is None) else field_region
    
    # organize and group
    df_group = df_input.copy()
    use_fake_region = (field_region not in df_group.columns)
    if use_fake_region:
        df_group[field_region] = region_default
    df_group = df_group.groupby([field_region])
    
    df_out = None
    i = 0


    for region, df in df_group:
        
        region = region[0] if isinstance(region, tuple) else region

        # PER-REGION MODS HERE 
        tup = transformation_support_lndu_transition_to_category_targets_single_region(
            df,
            dict_magnitude,
            vec_ramp,
            model_attributes,
            cats_stable = cats_stable,
            model_afolu = model_afolu,
            #**kwargs
        )

        
        if df_out is None:
            df_out = [tup[0] for k in range(len(df_group))]
        else:
            df_out[i] = tup[0]
            
        i += 1
        

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    (
        df_out.drop([field_region], axis = 1, inplace = True)
        if use_fake_region
        else None
    )
    
    return df_out






##############
#    LNDU    #
##############

def transformation_support_lndu_check_ltct_magnitude_dictionary(
    dict_magnitudes: Dict[str, Dict[str, Any]],
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    pasture_key: str = "pasture_tba",
) -> Union[None]:
    """
    Support function for 
        transformation_support_lndu_transition_to_category_targets_single_region
    
    Checks to verify that dict_magnitudes is properly specified. 

    NOTE: pasture_key and model_afolu.cat_lndu_grass cannot both be specified


    Function Arguments
    ------------------
    - dict_magnitudes: dictionary mapping land use categories to fraction 
        information. Should take the following form:

        {
            category: {
                "magnitude_type": magnitude_type,
                "magnitude": value,
                "categories_inflow_restrict": [cat_restrict_0, cat_restrict_1, ...],
                "categories_scalar_reference": cat_reference,
            }
        }

        NOTE: keys "categories_scalar_reference" and "categories_restrict" only
            required if 

                magnitude_type in [
                    "transfer_value", 
                    "transfer_value_scalar"
                ]

            * "categories_scalar_reference": gives a reference category for use
                in applying a scalar (REQUIRED for transfer value scalar) or a
                list of reference categories
            * "categories_inflow_restrict": optional list of inflow classes to 
                restrict transition scaling to. If None, defaults to all 
                non-zero inflow edges available in the transition matrix
    - model_attributes: ModelAttributes object used to call strategies/
        variables

    Keyword Arguments
    -----------------
    - model_afolu: optional AFOLU object to pass for variable access
    - pasture_key: key in dict_magnitude used to specify changes in pastures
    """

    model_afolu = (
        mafl.AFOLU(model_attributes)
        if model_afolu is None
        else model_afolu
    )

    time_periods = sc.TimePeriods(model_attributes)

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    # valid specifications of magnitude type
    magnitude_types_valid = [
        "add_from_reference_scalar",
        "baseline_scalar",
        "final_value",
        "final_value_ceiling",
        "final_value_floor"
    ]


    ##  CHECK SPECIFICATION DICTIONARY

    # pasture and grassland cannot be both specified
    if set([pasture_key, model_afolu.cat_lndu_grass]).issubset(set(dict_magnitudes.keys())):
        return None


    cats_all = attr_lndu.key_values + [pasture_key]
    dict_magnitude_cleaned = {}

    for cat in cats_all:
    
        # default verified to true; set to false if any condition is not met
        verified_cat = True
        dict_magnitude_cur = dict_magnitudes.get(cat)
        if dict_magnitude_cur is None:
            continue
        
        # check magnitude type
        magnitude_type = dict_magnitude_cur.get("magnitude_type")
        verified_cat &= (magnitude_type in magnitude_types_valid)
        
        # check magnitude
        magnitude = dict_magnitude_cur.get("magnitude")
        verified_cat &= sf.isnumber(magnitude)

        # reference magnitudes for scaling can be provided, including as pasture_key (hence use of cats_all)
        categories_scalar_reference = dict_magnitude_cur.get("categories_scalar_reference")
        categories_scalar_reference = (
            [categories_scalar_reference] 
            if isinstance(categories_scalar_reference, str) 
            else categories_scalar_reference
        )
        categories_scalar_reference = (
            [
                x for x in categories_scalar_reference 
                if x in cats_all
                and x != cat
            ]
            if sf.islistlike(categories_scalar_reference)
            else None
        )
        if categories_scalar_reference is not None:
            categories_scalar_reference = (
                None
                if len(categories_scalar_reference) == 0
                else categories_scalar_reference
            )

        # category restrictions for transition inflow edges
        categories_inflow_restrict = dict_magnitude_cur.get("categories_inflow_restrict")
        categories_inflow_restrict = (
            None
            if not sf.islistlike(categories_inflow_restrict)
            else [x for x in attr_lndu.key_values if x in categories_inflow_restrict]
        )
        if categories_inflow_restrict is not None:
            categories_inflow_restrict = (
                None 
                if (len(categories_inflow_restrict) == 0) 
                else categories_inflow_restrict
            )

        # check for time period as baseline
        tp_baseline = dict_magnitude_cur.get("time_period_baseline")
        tp_baseline = max(time_periods.all_time_periods) if (tp_baseline not in time_periods.all_time_periods) else tp_baseline

        if verified_cat:
            dict_magnitude_cleaned.update({
                cat: {
                    "categories_inflow_restrict": categories_inflow_restrict,
                    "categories_scalar_reference": categories_scalar_reference,
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
                    "tp_baseline": tp_baseline,
                }
            })

    return dict_magnitude_cleaned



def transformation_support_lndu_check_pasture_magnitude(
    magnitude: float,
    area_grassland: float,
    pasture_fraction: float, 
    magnitude_type: str,
    model_afolu: mafl.AFOLU,
    max_change_allocated_to_pasture_frac_adjustment: float = 0.0,
) -> Tuple[float, float]:
    """
    Some transformations may operate on pastures rather than grasslands as a
        whole--use this function to express pasture fractions in terms of 
        grassland fractions.  

    Returns a tuple of the form

        (magnitude_grassland, pasture_fraction)

        where `magnitude_grassland` is the magnitude applied to grasslands and
        `pasture_fraction` is the pasture fraction. If `pasture_fraction` is 
        None, then no change occurs.

    Function Arguments
    ------------------
    - magnitude: magnitude of tranformation 
    - area_grassland: area of grassland
    - pasture_fraction: fraction of grassland that is used as pasture
    - magntitude_type: valid type of magnitude, used to modify calculation of 
        fraction
    - model_afolu: AFOLU class used to access model attributes, properties, and
        methods

    Keyword Arguments
    -----------------
    - max_change_allocated_to_pasture_frac_adjustment: maximum allowable 
        fraction of changes that can be allocated to the pasture fraction 
        adjustments (e.g., silvopasture might rely on shifting existing 
        pastures to secondary forests rather than grassland as a whole)
    """
    model_attributes = model_afolu.model_attributes
    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    pasture_fraction = float(sf.vec_bounds(pasture_fraction, (0.0, 1.0)))
    area_pasture = pasture_fraction*area_grassland
    area_grassland_no_pasture = area_grassland - pasture_fraction


    ##  START BY GETTING TARGET AREA

    dict_apn = {
        "baseline_scalar": area_pasture*magnitude,
        "final_value": magnitude,
        "final_value_ceiling": min(magnitude, area_pasture),
        "final_value_floor": max(magnitude, area_pasture)
    }
    area_pasture_new = dict_apn.get(magnitude_type)
    if area_pasture_new is None:
        return None, None


    ##  MODIFY BASED ON max_increase_in_pasture_fraction

    """
    Calculate the pasture fraciton and total grassland area using components:
        a. Bound 0 for grassland area (area_grassland_b0), which occurs if
            pasture fraction is unchanged
        b. Bound 1 for grassland area (area_grassland_b1), which occurs at 
            greatest change in pasture fraction
        c. Calculate grassland area as 
        
            (
                b0(1 - max_change_allocated_to_pasture_frac_adjustment) 
                + b1*max_change_allocated_to_pasture_frac_adjustment
            )
    """
    area_pasture_delta = area_pasture_new - area_pasture
    area_grassland_no_pasture = area_grassland - area_pasture

    # under this bound, ensure that preserving the fraction does not erode natural grasslands
    area_grassland_b0 = max(
        area_grassland + area_pasture_delta/pasture_fraction,
        area_grassland_no_pasture + area_pasture_new
    )
    area_grassland_b1 = max(area_grassland, area_pasture + area_pasture_delta)
    
    # calculate new grassland area
    area_grassland_new = area_grassland_b1*max_change_allocated_to_pasture_frac_adjustment
    area_grassland_new += area_grassland_b0*(1 - max_change_allocated_to_pasture_frac_adjustment)
    pasture_fraction_new = (area_pasture + area_pasture_delta)/area_grassland_new

    if pasture_fraction == pasture_fraction_new:
        pasture_fraction_new = None

    out = (area_grassland_new, pasture_fraction_new)

    return out



def transformation_support_lndu_get_adjusted_fractions_from_transition_w_natural_grassland(
    arr_land_use_prevalence_out_no_intervention: np.ndarray,
    arr_land_use_prevalence_out_with_intervention: np.ndarray,
    vec_lndu_pasture_frac_no_intervention: np.ndarray,
    vec_lvst_scalar_carrying_capacity_no_intervention: np.ndarray,
    model_afolu: mafl.AFOLU,
    min_frac_grassland_pasture: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get adjusted fractions for pasture + adjusted carrying capacities (used
        primarily in Silvopasture modeling) if preserving fixed natural 
        grassland fraction assumptions. 

    Returns a two-ple of the following form:

        (vec_lndu_pasture_frac_new, vec_lndu_carrying_capacity_new)


    Function Arguments
    ------------------
    - arr_land_use_prevalence_out_no_intervention: array giving land use prevalence
        (wide by class, long by time period) WITHOUT the transformation of land use
        classes
    - arr_land_use_prevalence_out_with_intervention: array giving land use prevalence
        (wide by class, long by time period) WITH the transformation of land use
        classes
    - vec_lndu_pasture_frac_no_intervention: vector of pasture fractions of grassland
        without intervention (dim = (n_tp, ))
    - vec_lvst_scalar_carrying_capacity_no_intervention: vector of carrying capacity 
        scalars w/o intervention (dim = (n_tp, ))
    - model_afolu: AFOLU model class to pass
    
    Keyword Arguments
    -----------------
    - min_frac_grassland_pasture: minimum fraction of grassland that must be preserved
        as pasture
    """
    
    # get some baseline values
    vec_lndu_prevalence_grass_base = arr_land_use_prevalence_out_no_intervention[:, model_afolu.ind_lndu_grass]
    vec_lndu_prevalence_pasture_base = vec_lndu_prevalence_grass_base*vec_lndu_pasture_frac_no_intervention
    vec_lndu_prevalence_natural_grassland_base = vec_lndu_prevalence_grass_base - vec_lndu_prevalence_pasture_base
    
    # calculate carrying target for new pasture
    vec_lndu_carrying_target_base = vec_lndu_prevalence_pasture_base*vec_lvst_scalar_carrying_capacity_no_intervention
    
    # now, get the area associated with pasture now and generate the scalar
    vec_lndu_prevalence_grass_intervention = arr_land_use_prevalence_out_with_intervention[:, model_afolu.ind_lndu_grass]
    vec_lndu_prevalence_pasture_intervention = sf.vec_bounds(vec_lndu_prevalence_grass_intervention - vec_lndu_prevalence_natural_grassland_base, (min_frac_grassland_pasture, 1.0))
    vec_lndu_carrying_capacity_new = vec_lndu_carrying_target_base/vec_lndu_prevalence_pasture_intervention
    
    
    # get new pasture fraction
    vec_lndu_pasture_frac_new = np.nan_to_num(
        vec_lndu_prevalence_pasture_intervention/vec_lndu_prevalence_grass_intervention,
        nan = 0.0, 
        posinf = 0.0,
    )
    
    out = (vec_lndu_pasture_frac_new, vec_lndu_carrying_capacity_new)

    return out



def transformation_support_lndu_get_adjusted_pasture_fraction(
    prevalence_vector_0: float,
    prevalence_vector_1: float,
    frac_pasture_0: float,
    cat_target: str,
    model_afolu: mafl.AFOLU,
    cat_pasture: str = "grasslands",
    frac_target_containing_shift: float = 1,
    scalar_base_grassland: Union[float, None] = None, 
) -> Union[float, None]:
    """
    Using previous grassland area, previous pasture fraction, current grassland 
        area, and land use category to which pasture was shifted, calculate 
        calculate the change in pasture fraction.

    Returns tuple of the form 

        (frac_pasture_1, ratio_pasture_frac_cur_to_orig)
        

    Function Arguments
    ------------------
    - prevalence_vector_0: original (unadjusted) vector of land use prevalence
    - prevalence_vector_1: current (adjusted) vector of land use prevalence
    - frac_pasture_0: original (unadjusted) grassland pasture fraction
    - cat_target: target category to which pasture was shifted
    - model_afolu: AFOLU class used to access model attributes, properties, and
        methods
    
    Keyword Arguments
    -----------------
    - cat_pasture: category which pastures are contained in
    - frac_target_containing_shift: fraction of increase in transfer target land
        use containing pasture
    - scalar_base_grassland: optional scalar to use to represent changes to base
        grassland area. If None, remains unchanged
    """
    
    # some initialization and checkes
    model_attributes = model_afolu.model_attributes
    attr_lndu = model_attributes.get_attribute_table(model_attributes.subsec_name_lndu)
    frac_pasture_0 = float(sf.vec_bounds(frac_pasture_0, (0.0, 1.0)))
    scalar_base_grassland = (
        float(sf.vec_bounds(scalar_base_grassland, (0.0, np.inf)))
        if sf.isnumber(scalar_base_grassland)
        else 1.0
    )

    ind_grass = attr_lndu.get_key_value_index(cat_pasture)
    ind_target = attr_lndu.get_key_value_index(cat_target)
    if ind_target is None:
        return None

    # calculate revisions to pasture fractions
    vec_lndu_transferred = prevalence_vector_1 - prevalence_vector_0
    area_past_transferred = min(
        -vec_lndu_transferred[ind_grass],
        vec_lndu_transferred[ind_target]*frac_target_containing_shift
    )

    prevalence_grass_cur = prevalence_vector_1[ind_grass]
    prevalence_grass_orig = prevalence_vector_0[ind_grass]

    # get base grassland area (non-pasture) and provide for optional scalar
    prevalence_pasture_orig = prevalence_grass_orig*frac_pasture_0
    prevalence_grass_no_pasture_orig = prevalence_grass_orig - prevalence_pasture_orig
    prevalence_grass_no_pasture_orig *= scalar_base_grassland

    # assume no change in baseline grass area unless absolutely necessary
    prevalence_pasture_cur = prevalence_grass_cur - prevalence_grass_no_pasture_orig
    frac_pasture_current = prevalence_pasture_cur/prevalence_grass_cur
    ratio_pasture_frac_cur_to_orig = prevalence_pasture_cur/prevalence_pasture_orig

    out = frac_pasture_current, ratio_pasture_frac_cur_to_orig
   
    return out




def transformation_support_lndu_specify_transitions(
    df_input: pd.DataFrame,
    dict_magnitude: Dict[str, Dict[str, Any]],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    strategy_id: Union[int, None] = None,
    **kwargs
 ) -> pd.DataFrame:
    """
    Modify transition probabilities directly based on (i, j) (row, col) 
        specification.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_magnitude: dictionary mapping a pair of categories (source, target) 
        to a magnitude, either a 

        {
            (category_i, category_j): {
                "magnitude_type": magnitude_type,
                "magnitude": value
            }
        }

        Valid values for "magnitude_type" include

        * "baseline_scalar": multiply baseline value by magnitude
        * "final_value": magnitude is a final value
        * "final_value_ceiling": magnitude is the lesser of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            smaller
        * "final_value_floor": magnitude is the greater of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            greater

        NOTE: caution should be taken to not overuse this; transition matrices
            can be chaotic, and modifying too many target categories may cause 
            strange behavior. 

    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation


    Keyword Arguments
    -----------------
    - cats_stable: optional set of categories to preserve with stable transition
        probabilities *out* of the categori
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    ##  SOME INITIALIZATION 

    model_afolu = (
        mafl.AFOLU(model_attributes)
        if model_afolu is None
        else model_afolu
    )

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    mt_valid = [
        "baseline_scalar",
        "final_value",
        "final_value_ceiling",
        "final_value_floor"
    ]

    # check dictionary
    return_input = True
    if isinstance(dict_magnitude, dict):

        dict_magnitude_clean = {}
        keys = list(dict_magnitude.keys())

        for k in keys:
            v = dict_magnitude.get(k)

            keep_q = True
            keep_q &= isinstance(k, tuple)
            keep_q &= (len(k) == 2) if keep_q else False
            keep_q &= (
                (k[0] in attr_lndu.key_values) & (k[1] in attr_lndu.key_values) 
                if keep_q 
                else False
            )

            # check values
            keep_q &= isinstance(v, dict)
            if keep_q:
                keep_q &= (v.get("magnitude_type") in mt_valid)
                keep_q &= sf.isnumber(v.get("magnitude"))

            # format and update
            if keep_q:
                k_new = (
                    attr_lndu.get_key_value_index(k[0]), 
                    attr_lndu.get_key_value_index(k[1])
                )

                dict_magnitude_clean.update({k_new: v})
        
        # no modifcations can be made if the dictionary is incorrectly specified
        return_input = (len(dict_magnitude_clean) == 0)
        
    if return_input:
        # LOGGING
        return df_input

    n_tp = len(df_input)
    vec_ramp = sf.vec_bounds(vec_ramp, (0.0, 1.0))


    # get transition matrices and emission factors
    qs, efs = model_afolu.get_markov_matrices(
        df_input, 
        len(df_input)
    )

    # next, format magnitude vectors
    dict_adjs = {}
    for key, val in dict_magnitude_clean.items():
        i, j = key
        magnitude = val.get("magnitude")
        magnitude_type = val.get("magnitude_type")

        # get the trajectory of probabilities
        vec_traj_probs = np.array([q[i, j] for q in qs])
        
        # convert to a value, which will then be converted back to a scalar later
        if magnitude_type == "baseline_scalar":
            mag = vec_traj_probs[-1]*magnitude
        elif magnitude_type == "final_value":
            mag = magnitude
        elif magnitude_type == "final_value_ceiling":
            mag = min(magnitude, vec_traj_probs[-1])
        elif magnitude_type == "final_value_floor":
            mag = max(magnitude, vec_traj_probs[-1])
        
        # next, get the difference vector, update 
        mag = float(sf.vec_bounds(mag, (0, 1)))
        vec_diff = vec_ramp*(mag - vec_traj_probs)
        vec_new_probs = vec_traj_probs + vec_diff
        
        dict_adjs.update({key: vec_new_probs})


    # update transition probability matrices on the fluy
    for t, q in enumerate(qs):
        
        dict_adj = {}

        for key, vec in dict_adjs.items():

            i, j = key
            mag_target = vec[t]
            mag_cur = q[i, j]
            scalar = np.nan_to_num(mag_target/mag_cur, nan = 1.0, posinf = 1.0)

            dict_adj.update({key: scalar})

        qs[t] = model_afolu.adjust_transition_matrix(q, dict_adj)


    # convert to input format and overwrite in output data
    df_out = model_afolu.format_transition_matrix_as_input_dataframe(qs)
    df_out = sf.match_df_to_target_df(
        df_input,
        df_out,
        [model_attributes.dim_time_period]
    )
    
    # add strategy id if called for
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    return df_out



def transformation_support_lndu_transition_to_category_targets_single_region(
    df_input: pd.DataFrame,
    dict_magnitude: Dict[str, Dict[str, Any]],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_stable: Union[List[str], None] = None,
    max_change_allocated_to_pasture_frac_adjustment: float = 0.0,
    max_value: float = 0.9,
    min_value: float = 0.0,
    model_afolu: Union[mafl.AFOLU, None] = None,
    pasture_key: str = "pasture_tba",
    strategy_id: Union[int, None] = None,
    **kwargs
 ) -> pd.DataFrame:
    """
    Modify transition probabilities to acheieve targets for land use categories.


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_magnitude: dictionary mapping land use categories to fraction 
        information. Should take the following form:

        {
            category: {
                "magnitude_type": magnitude_type,
                "magnitude": value,
                "categories_inflow_restrict": [cat_restrict_0, cat_restrict_1, ...],
                "categories_scalar_reference": cat_reference,
            }
        }

        Valid values for "magnitude_type" include

        * "add_from_reference_scalar": add value based on a scalar applied to 
            magnitude from another category (specified as 
            `categories_scalar_reference`). E.g., to increase secondary 
            forests by 30% of pasture area, use

            "forests_secondary": {
                "categories_scalar_reference": "pasture_tba"
                "magnitude": 0.3,
                "magnitude_type": "add_from_reference_scalar",
            } 

            NOTE: pasture_tba is the pasture_key, not a land use class

        * "baseline_scalar": multiply baseline value by magnitude
        * "final_value": magnitude is the final value for the variable to take 
            (achieved in accordance with vec_ramp)
        * "final_value_ceiling": magnitude is the lesser of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            smaller
        * "final_value_floor": magnitude is the greater of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            greater

        NOTE: keys "categories_scalar_reference" and "categories_restrict" only
            required if 

                magnitude_type in [
                    "transfer_value", 
                    "transfer_value_scalar"
                ]

            * "categories_scalar_reference": gives a reference category for use
                in applying a scalar (REQUIRED for transfer value scalar) or a
                list of reference categories
            * "categories_inflow_restrict": optional list of inflow classes to 
                restrict transition scaling to. If None, defaults to all 
                non-zero inflow edges available in the transition matrix

        NOTE: caution should be taken to not overuse this; transition matrices
            can be chaotic, and modifying too many target categories may cause 
            strange behavior. 

    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation


    Keyword Arguments
    -----------------
    - cats_stable: optional set of categories to preserve with stable transition
        probabilities *out* of the categori
    - field_region: field in df_input that specifies the region
    - max_change_allocated_to_pasture_frac_adjustment: passed to 
        transformation_support_lndu_check_pasture_magnitude; represents maximum 
        fraction of pasture-specific transformations that can be allocated to an
        adjustment in the pasture fraction
    - max_value: maximum value in final time period that any land use class can 
        take
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    model_afolu = (
        mafl.AFOLU(model_attributes)
        if model_afolu is None
        else model_afolu
    )

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    if not isinstance(dict_magnitude, dict):
        return df_input

    # check dictionary
    dict_magnitude = transformation_support_lndu_check_ltct_magnitude_dictionary(
        dict_magnitude,
        model_attributes,
        model_afolu,
        pasture_key = pasture_key,
    )
    n_tp = len(df_input)

    # some pasture/grass info
    grass_is_pasture_q = pasture_key in dict_magnitude.keys()
    dict_ptg = {pasture_key: model_afolu.cat_lndu_grass}
    
    # get indices for categories
    cats_to_modify = sorted([dict_ptg.get(x, x) for x in dict_magnitude.keys()])
    inds_to_modify = [
        (
            attr_lndu.get_key_value_index(x)
            if x != pasture_key
            else attr_lndu.get_key_value_index(model_afolu.cat_lndu_grass)
        ) 
        for x in cats_to_modify
    ]
    


    """
    REPEAT MODEL PROJECTIONS OF LAND USE UNDER LURF = 0 (specified transitions)
        * Calculate fractions going forward
        * Determine appropriate target magnitudes (based on magnitude type)
        * Use model_afolu.adjust_transition_matrix() to scale columns up/down
    """

    ##  1. GET COMPONENTS USED FOR LAND USE PROJECTION

    # get the initial distribution of land and pasture fraction
    vec_lndu_initial_frac = model_attributes.extract_model_variable(#
        df_input, 
        model_afolu.modvar_lndu_initial_frac, 
        return_type = "array_base",
    )[0]

    vec_lndu_pasture_frac = model_attributes.extract_model_variable(#
        df_input, 
        model_afolu.modvar_lndu_frac_grassland_that_is_pasture, 
        return_type = "array_base",
    )

    # determine when to initialize the scaling
    ind_first_nz = np.where(vec_ramp > 0)[0][0]
    ind_first_full_impl = np.where(vec_ramp == 1)[0][0]
    ind_last_zero = ind_first_nz - 1

    # get transition matrices and emission factors
    qs, efs = model_afolu.get_markov_matrices(
        df_input, 
        len(df_input)
    )


    ##  2. IF magnitude_type IS RELATIVE TO BASE VALUE, NEED TO KNOW BASELINE VALUES
    
    # project land use over all time periods and get final fractions without intervention
    arr_emissions_conv, arr_land_use, arrs_land_conv = model_afolu.project_land_use(
        vec_lndu_initial_frac,
        qs,
        efs, 
    )
    # prevalence in final time period without any adjustment
    vec_lndu_final_frac_unadj = arr_land_use[-1, :] #NOTE: might need to base on ind_first_full_impl

    # inialize output array
    arr_land_use_prevalence_out_no_intervention = arr_land_use.copy()
    arr_land_use_prevalence_out_with_intervention = np.ones(arr_land_use.shape)


    # 2a. for "baseline_scalar" that include a reference, get magnitudes

    dict_magnitude_overwrite = {}

    for i, cat in enumerate(cats_to_modify):

        dict_cur = dict_magnitude.get(cat)

        categories_scalar_reference = dict_cur.get("categories_scalar_reference")
        magnitude_type = dict_cur.get("magnitude_type")
        valid_mt = (magnitude_type in ["baseline_scalar", "add_from_reference_scalar"])
       
        # only apply this "unfolding" of magnitudes if using a scalar based approach
        if (categories_scalar_reference is not None) & valid_mt:

            magnitude = dict_cur.get("magnitude")
            magnitude_type = dict_cur.get("magnitude_type")
            tp_baseline = dict_cur.get("tp_baseline")
            tp_baseline_ind = np.where(
                np.array(df_input[model_attributes.dim_time_period]) == tp_baseline
            )[0][0]

            # inds for source (transfering out)
            ind = inds_to_modify[i]
            val_unadj_source = arr_land_use[tp_baseline_ind, ind]
            val_unadj_reference = 0.0

            # loop over reference categories and apply magnitude
            for csr in categories_scalar_reference:

                cat_ref = dict_ptg.get(csr, csr)
                ind_reference = attr_lndu.get_key_value_index(cat_ref)
            
                val_unadj_reference_cur = arr_land_use[tp_baseline_ind, ind_reference]
                val_unadj_reference_cur *= (
                    vec_lndu_pasture_frac[tp_baseline_ind]
                    if (cat_ref == model_afolu.cat_lndu_grass) & (csr == pasture_key)
                    else 1.0
                )

                val_unadj_reference += val_unadj_reference_cur

            # will convert to "baseline_scalar" approach
            magnitude_new = (magnitude*val_unadj_reference)/val_unadj_source
            magnitude_new += (1 if (magnitude_type == "add_from_reference_scalar") else 0)
            
            dict_ow = {}
            for k in dict_cur.keys():
                val = (dict_cur.get(k) if (k != "magnitude") else magnitude_new)
                val = (val if (k != "magnitude_type") else "baseline_scalar")
                dict_ow.update({k: val})

            dict_magnitude_overwrite.update({cat: dict_ow})

    # update
    dict_magnitude.update(dict_magnitude_overwrite)


    # next, setup magnitude as a target value 
    for i, cat in enumerate(cats_to_modify):
        
        dict_cur = dict_magnitude.get(cat)

        categories_target = dict_cur.get("categories_target")
        magnitude = dict_cur.get("magnitude")
        magnitude_type = dict_cur.get("magnitude_type")
        tp_baseline = dict_cur.get("tp_baseline")

        ind = inds_to_modify[i]
        val_unadj = vec_lndu_final_frac_unadj[ind]
        
        mag_new = magnitude
        if magnitude_type == "baseline_scalar":
            mag_new = magnitude*val_unadj
        elif magnitude_type == "final_value_ceiling":
            mag_new = min(magnitude, val_unadj)
        elif magnitude_type == "final_value_floor":
            mag_new = max(magnitude, val_unadj)
        
        # bound to prevent excessive fractions
        sup = max(val_unadj, max_value)
        mag_new = sf.vec_bounds(mag_new, (min_value, sup))
        dict_cur.update({"magnitude": mag_new})


    ##  3. CONTINUE WITH PROJECTION AND ADJUSTMENT OF PROBS DURING vec_implementation_ramp NON-ZERO YEARS

    # run forward to final period before ramp for all associated with no change 
    arr_emissions_conv, arr_land_use, arrs_land_conv = model_afolu.project_land_use(
        vec_lndu_initial_frac,
        qs[0:ind_first_nz],
        efs, 
        n_tp = ind_last_zero,
    )
    vec_lndu_final_virnz_frac_unadj = arr_land_use[-1, :] # at time ind_first_nz - 1
    arr_land_use_prevalence_out_with_intervention[0:ind_last_zero] = arr_land_use # land use up until final time period


    ##  4. PREPARE TRANSITION MATRIX FOR MODIFICATION

    # initialize adjustment dictionary
    dict_adj = {}

    # check unadjusted final period fractions
    n_tp_scale = n_tp - ind_first_nz - 1
    fracs_unadj_first_effect_tp = np.dot(vec_lndu_final_virnz_frac_unadj, qs[ind_first_nz - 1])
    fracs_unadj_first_effect_tp = np.dot(fracs_unadj_first_effect_tp, qs[ind_first_nz])[inds_to_modify]
    fracs_target_final_tp = np.array([dict_magnitude.get(x).get("magnitude") for x in cats_to_modify])
    """
    OPTION FOR EXPANSION: SPECIFY NON-LINEAR TARGETS (READ OFF OF vec_implementation_ramp)

    df_tp = df_input[[model_attributes.dim_time_period]].copy()
    
    df_tmp = df_tp.iloc[0:ind_first_nz].copy()
    df_tmp = pd.concat(
        [
            df_tmp.reset_index(drop = True), 
            pd.DataFrame(
                arr_land_use[:, inds_to_modify],
                columns = cats_to_modify
            )
        ],
        axis = 1
    )

    df_append = {
        model_attributes.dim_time_period: [int(df_input[model_attributes.dim_time_period].iloc[-1])]
    }
    df_append.update(
        dict(
            (cats_to_modify: [])
        )
    )


    df_tmp = pd.DataFrame({
        time_periods.field_time_period: [0, 1, 2, 3, 4, 5, 35],
        #"val": [0.3, 0.31, 0.32, 0.3275, 0.3325, 0.335, 0.335]
        "val": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.335]
    })
    df_tmp = pd.merge(df_tp, df_tmp, how = "left")
    ?df_tmp.interpolate
    df_tmp.interpolate(method = "linear", order = 2).plot(x = "time_period")
    """
    arr_target_shares = (fracs_target_final_tp - fracs_unadj_first_effect_tp)/n_tp_scale
    arr_target_shares = np.outer(np.arange(1, n_tp_scale + 1), arr_target_shares)
    arr_target_shares += fracs_unadj_first_effect_tp

    # verify and implement stable output transition categories
    cats_ignore = []
    cats_stable = (
        [x for x in attr_lndu.key_values if x in cats_stable]
        if sf.islistlike(cats_stable)
        else []
    )

    for cat_stable in cats_stable:
        ind_lndu_stable = attr_lndu.get_key_value_index(cat_stable)
        for cat in attr_lndu.key_values:
            ind = attr_lndu.get_key_value_index(cat)
            if cat not in cats_ignore:
                dict_adj.update({(ind_lndu_stable, ind): 1})
    

    
    ##  5. FINALLY, ADJUST TRANSITION MATRICES AND OVERWRITE

    """
    Process:

    a. Start with land use prevalance at time ind_first_nz 
        i. estimated as prevalence at x_{ind_first_nz - 1}Q_{ind_first_nz - 1}
    b. Next, project forward from time t to t+1 and get adjustment to columnar
        inflows (scalars_to_adj)
    c. Use `model_afolu.get_lndu_scalar_max_out_states` to get true positional 
        scalars. This accounts for states that might "max out" (as determined by
        model_afolu.mask_lndu_max_out_states), or reach 100% or 0% probability 
        during the scaling process.
    d. Then, with the scalars obtained, adjust the matrix using 
        model_afolu.adjust_transition_matrix
    """;

    x = np.dot(vec_lndu_final_virnz_frac_unadj, qs[ind_first_nz - 1])
    arr_land_use_prevalence_out_with_intervention[ind_first_nz - 1, :] = x
    inds_iter = list(range(ind_first_nz, n_tp))    

    for ind_row, i in enumerate(inds_iter):

        # in first iteation, this is projected prevalence at ind_first_nz + 1
        x_next_unadj = np.dot(x, qs[i]) 
        ind_row = min(ind_row, arr_target_shares.shape[0] - 1)
        scalars_to_adj = np.nan_to_num(
            arr_target_shares[ind_row, :]/x_next_unadj[inds_to_modify],
            0.0,
            posinf = 0.0
        )

        for j, z in enumerate(inds_to_modify):

            scalar = scalars_to_adj[j]
            cat_to_modify = cats_to_modify[j]
            cats_inflow_restrict = dict_magnitude.get(cat_to_modify)
            
            # specify max-out mask (if explicit, set )
            lmo = None
            if cats_inflow_restrict is not None:
                cats_inflow_restrict = cats_inflow_restrict.get("categories_inflow_restrict")
                lmo = "decrease_and_increase"

            # get states to "max out" (scale as much as possible before shifting to other states)
            mask_lndu_max_out_states = model_afolu.get_lndu_scalar_max_out_states(
                scalar, 
                cats_max_out = cats_inflow_restrict,
                lmo_approach = lmo,
            )

            
            scalar_lndu_cur = model_afolu.get_matrix_column_scalar(
                qs[i][:, z],
                scalar,
                x,
                mask_max_out_states = mask_lndu_max_out_states,
                max_iter = 100,
            )


            dict_adj.update(
                dict(
                    ((ind, z), s) for ind, s in enumerate(scalar_lndu_cur)
                )
            )

        qs[i] = model_afolu.adjust_transition_matrix(qs[i], dict_adj)
        x = np.dot(x, qs[i])

        arr_land_use_prevalence_out_with_intervention[i, :] = x

        

    
    # convert to input format and overwrite in output data
    df_out = model_afolu.format_transition_matrix_as_input_dataframe(qs)
    df_out = sf.match_df_to_target_df(
        df_input,
        df_out,
        [model_attributes.dim_time_period]
    )
    
    # add strategy id if called for
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    # return:
    #  output data frame
    #  vector of previous land use outcome
    #  vector of current land use outcome
    #  original pasture land use fraction vector
    tup_out = (
        df_out, 
        vec_lndu_final_frac_unadj, 
        x, 
        vec_lndu_pasture_frac, 
        arr_land_use_prevalence_out_no_intervention, 
        arr_land_use_prevalence_out_with_intervention
    )

    return tup_out



def transformation_lndu_increase_silvopasture(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Increase the use of silvopasture by shifting pastures to secondary forest. 
        NOTE: This trnsformation relies on modifying transition matrices, which
        can compound some minor numerical errors in the crude implementation 
        taken here. Final area prevalences may not reflect precise shifts.


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying fraction of pasture to shift to silvopasture
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    - **kwargs: passed to 
        transformation_support_lndu_transition_to_category_targets_single_region()
    """
    
    # build
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    magnitude = (
        float(sf.vec_bounds(magnitude, (0.0, 1.0)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    
    ##  SETUP DICTIONARY--shift 
    cat_fsts = model_afolu.cat_lndu_fsts
    cat_grsl = model_afolu.cat_lndu_grass
    key_pastures = "pasture_tba"
    dict_magnitude = {
        cat_fsts: {
            "categories_scalar_reference": [key_pastures],
            "categories_inflow_restrict": [cat_grsl],
            "magnitude": magnitude,
            "magnitude_type": "add_from_reference_scalar"
        }
    }
    
    cats_stable = [
        model_afolu.cat_lndu_othr,
        model_afolu.cat_lndu_stlm,
        model_afolu.cat_lndu_wetl,
    ]
    
    # get region
    field_region_def = "nation"
    region_default = "DEFAULT"
    field_region = kwargs.get("field_region", field_region_def)
    field_region = field_region_def if (field_region is None) else field_region
    #(
    #    field_region_def
    #    if "field_region" not in kwargs.keys()
    #    else kwargs.get("field_region")
    #)
    
    # organize and group
    df_group = df_input.copy()
    use_fake_region = (field_region not in df_group.columns)
    if use_fake_region:
        df_group[field_region] = region_default
    df_group = df_group.groupby([field_region])
    
    df_out = None
    i = 0
    
    global arr_land_use_prevalence_out_no_intervention
    global arr_land_use_prevalence_out_with_intervention
    global vec_lndu_pasture_frac

    for region, df in df_group:
        
        region = region[0] if isinstance(region, tuple) else region
        
        # PER-REGION MODS HERE 
        (
            df_trans, 
            vec_lndu_prevalence_0, 
            vec_lndu_prevalence_1,
            vec_lndu_pasture_frac,
            arr_land_use_prevalence_out_no_intervention, 
            arr_land_use_prevalence_out_with_intervention,
        ) = transformation_support_lndu_transition_to_category_targets_single_region(
            df,
            dict_magnitude,
            vec_ramp,
            model_attributes,
            cats_stable = cats_stable,
            max_change_allocated_to_pasture_frac_adjustment = 0.0,
            model_afolu = model_afolu,
            pasture_key = key_pastures,
            **kwargs
        )


        # get pasture fractions and carrying capacities
        vec_lvst_carrying_capcity_scalar = model_attributes.extract_model_variable(#
            df,
            model_afolu.modvar_lvst_carrying_capacity_scalar,
            return_type = "array_base",
        )


        (
            vec_lndu_pasture_frac_new, 
            vec_lndu_carrying_capacity_new
        ) = transformation_support_lndu_get_adjusted_fractions_from_transition_w_natural_grassland(
            arr_land_use_prevalence_out_no_intervention,
            arr_land_use_prevalence_out_with_intervention,
            vec_lndu_pasture_frac,
            vec_lvst_carrying_capcity_scalar,
            model_afolu,
        )

        
        """
        # get changes to pasture fraction and associated ratio of increase for lvst carrying capacity
        frac_past_new, scalar_carrying_capacity_inv = transformation_support_lndu_get_adjusted_pasture_fraction(
            vec_lndu_prevalence_0,
            vec_lndu_prevalence_1,
            vec_lndu_pasture_frac[-1], 
            cat_fsts,
            model_afolu
        )

        """;
        
        # apply general transformation to set pasture fraction and use to scale lvst
        df_trans = tbg.transformation_general(
            df_trans,
            model_attributes,
            {
                model_afolu.modvar_lvst_carrying_capacity_scalar: {
                    "bounds": (0.0, np.inf),
                    "magnitude": vec_lndu_carrying_capacity_new,#1/scalar_carrying_capacity_inv,
                    "magnitude_type": "vector_specification",#"baseline_scalar",
                    "vec_ramp": vec_ramp
                },

                model_afolu.modvar_lndu_frac_grassland_that_is_pasture: {
                    "bounds": (0.0, 1.0),
                    "magnitude": vec_lndu_pasture_frac_new,
                    "magnitude_type": "vector_specification",
                    "vec_ramp": vec_ramp
                }
            },
            **kwargs
        )
        
        if df_out is None:
            df_out = [df_trans for k in range(len(df_group))]
        else:
            df_out[i] = df_trans
            
        i += 1
        

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    (
        df_out.drop([field_region], axis = 1, inplace = True)
        if use_fake_region
        else None
    )
    
    return df_out



##############
#    LSMM    #
##############

def transformation_lsmm_improve_manure_management(
    df_input: pd.DataFrame,
    dict_lsmm_magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories_lvst: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Improve Livestock Manure Management" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_lsmm_magnitude: dictionary mapping LSMM categories to target
        fractions (total x must be 0 <= x <= 1.0)
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories_lvst: optional livestock categories to specify (will apply 
        fractions only to those categories)
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    # check dict of lsmm magnitudes
    categories_lsmm = model_attributes.get_valid_categories(None, model_attributes.subsec_name_lsmm)
    dict_lsmm_magnitude = dict(
        (k, float(sf.vec_bounds(v, (0.0, 1.0)))) 
        for k, v in dict_lsmm_magnitude.items() 
        if k in categories_lsmm
        and sf.isnumber(v)
    )
    
    magnitude = sum(dict_lsmm_magnitude.values())
    if magnitude > 1:
        # LOGGING
        return df_input
    
    # check lvst categories
    categories_lvst = model_attributes.get_valid_categories(
        categories_lvst,
        model_attributes.subsec_name_lvst
    )

    #modvars = model_afolu.modvar_list_lvst_mm_fractions
    
    
    # set of all modvars to use as source + iteration initialization
    dict_cats_to_modvars = model_afolu.dict_lsmm_categories_to_lvst_fraction_variables
    dict_transformations = {}
    modvars = []

    for cat, v in dict_cats_to_modvars.items():
    
        modvar = v.get("mm_fraction")

        # update model variable domain for transfer
        (
            modvars.append(modvar)
            if modvar is not None
            else None
        )

        # get model variables to use as target
        mag = dict_lsmm_magnitude.get(cat)
        (
            dict_transformations.update({
                modvar: mag/magnitude
            })
            if mag is not None
            else None
        )
        
    
    df_out = tbg.transformation_general_shift_fractions_from_modvars(
        df_input,
        magnitude,
        modvars,
        dict_transformations,
        vec_ramp,
        model_attributes,
        categories = categories_lvst,
        **kwargs,
    )

    return df_out



##############
#    LVST    #
##############

def transformation_lvst_increase_productivity(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Livestock Productivity" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional increase, by final time period, in livestock
        carrying capacity per area of managed grassland--e.g., to increase 
        productivity by 30%, enter 0.3. 
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    magnitude = (
        float(sf.vec_bounds(1 + magnitude, (0.0, np.inf)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_afolu.modvar_lvst_carrying_capacity_scalar: {
                "bounds": (0.0, np.inf),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out



def transformation_lvst_reduce_enteric_fermentation(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Enteric Fermentation" transformation.


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
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_lvst_ef_ch4_ef
    bounds = (0, 1)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 - v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_lvst
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out



##############
#    SOIL    #
##############

def transformation_soil_reduce_excess_fertilizer(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Excess Fertilizer" transformation. Can be used to
        reduce excess N from fertilizer or reduce liming. See `magnitude` in 
        function arguments for information on dictionary specification.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary with keys for "fertilizer_n" and "lime" mapping 
        to proportional reductions in the per unit application of fertilizer N 
        and lime, respectively. If float, applies to fertilizer N and lime 
        uniformly.
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )


    ##  BUILD TRANSFORMATION FOR BIOGAS/LANDFILL

    dict_key_to_modvar = {
        "fertilizer_n": model_afolu.modvar_soil_demscalar_fertilizer,
        "lime": model_afolu.modvar_soil_demscalar_liming,
    }

    dict_transformation = {}
    for key, modvar in dict_key_to_modvar.items():
        # get the current magnitude of gas capture
        mag = (
            magnitude.get(key)
            if isinstance(magnitude, dict)
            else (magnitude if sf.isnumber(magnitude) else None)
        )

        mag = (
            float(sf.vec_bounds(1 - mag, (0, 1)))
            if mag is not None
            else None
        )

        (
            dict_transformation.update(
                {
                    modvar: {
                        "bounds": None,
                        "magnitude": mag,
                        "magnitude_type": "baseline_scalar",
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