import sisepuede.core.model_attributes as ma
import sisepuede.models.afolu as mafl
import sisepuede.models.ippu as mi
import sisepuede.models.circular_economy as mc
import sisepuede.models.energy_production as ml
import sisepuede.models.energy_consumption as me
import sisepuede.models.socioeconomic as se
import numpy as np
import pandas as pd
import sisepuede.utilities.support_functions as sf
from transformations_base_general import *
from typing import *






####################################
###                              ###
###    ENERGY TRANSFORMATIONS    ###
###                              ###
####################################


###########################
#    SUPPORT FUNCTIONS    #
###########################

def get_renewable_categories_from_inputs_final_tp(
    df_input: pd.DataFrame,
    model_electricity: ml.EnergyProduction,
) -> Union[Dict[str, float], None]:
    """
    Get renewable categories from the input DataFrame. Returns a
        dictionary mapping each renewble category to its fraction of 
        production considered renewable in the final time period. If an
        error occurs, returns None.
        
    
    Function Arguments
    ------------------
    - df_input: input data frame containing 
        model_electricity.modvar_entc_nemomod_renewable_tag_technology
    - model_electricity: ElectricEnercy model used to obtain variables
        and call model attributes
    
    
    Keyword Arguments
    -----------------
    """
    
    model_attributes = model_electricity.model_attributes
    modvar = model_electricity.modvar_entc_nemomod_renewable_tag_technology
    subsec = model_attributes.get_variable_subsector(modvar)
    attr = model_electricity.model_attributes.get_attribute_table(subsec)
    
    try:
        arr_entc_tag_renewable = model_attributes.extract_model_variable(#
            df_input,
            modvar,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

    except Exception as e:
        return None
    
    
    ##  INITIALIZE OUTPUT DICTIONARY AND GET VALUES
    
    dict_out = {}
    
    w = np.where(arr_entc_tag_renewable[-1] > 0)[0]
    if len(w) > 0:
        keys = np.array(attr.key_values)[w]
        vals = arr_entc_tag_renewable[-1, w]
        dict_out = dict(zip(keys, vals))
    
    return dict_out









##########################################
#    CARBON CAPTURE AND SEQUESTRATION    #
##########################################

def transformation_ccsq_increase_direct_air_capture(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase direct air capture" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of direct air capture in final time period
        * IMPORTANT: entered in MT
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    # get conversion units
    units = model_attributes.get_variable_characteristic(
        model_energy.modvar_ccsq_total_sequestration,
        model_attributes.varchar_str_unit_mass
    )
    scalar = model_attributes.get_mass_equivalent("mt", units)
    categories = ["direct_air_capture"]

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_ccsq_total_sequestration: {
                "bounds": (0, np.inf),
                "categories": categories,
                "magnitude": magnitude*scalar,
                "magnitude_type": "final_value",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out






###########################################
#    ENERGY TECHNOLOGY TRANSFORMATIONS    #
###########################################

def transformation_entc_change_msp_max(
    df_input: pd.DataFrame,
    dict_cat_to_vector: Dict[str, float],
    model_electricity: ml.EnergyProduction,
    drop_flag: Union[int, float, None] = None,
    vec_ramp: Union[np.ndarray, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement a transformation for the baseline to resolve constraint
        conflicts between TotalTechnologyAnnualActivityUpperLimit/
        TotalTechnologyAnnualActivityLowerLimit if MinShareProduction is 
        Specified. 

    This transformation will turn on the MSP Max method in EnergyProduction,
        which will cap electric production (for a given technology) at the 
        value estimated for the last non-engaged time period. 
        
    E.g., suppose a technology has the following estimated electricity 
        production (estimated endogenously and excluding demands for ENTC) 
        and associated value of msp_max (stored in the "Maximum Production 
        Increase Fraction to Satisfy MinShareProduction Electricity" 
        SISEPUEDE model variable):

        time_period     est. production     msp_max
                        implied by MSP     
        -----------     ---------------     -------
        0               10                  -999
        1               10.5                -999
        2               11                  -999
        3               11.5                -999
        4               12                  0
        .
        .
        .
        n - 2           23                  0
        n - 1           23.1                0

        Then the MSP for this technology would be adjusted to never exceed 
        the value of 11.5, which was found at time_period 3. msp_max = 0
        means that a 0% increase is allowable in the MSP passed to NemoMod,
        so the specified MSP trajectory (which is passed to NemoMod) is 
        adjusted to reflect this change.
    
    NOTE: Only the *first value* after that last non-specified time period
        affects this variable. Using the above table as an example, entering 
        0 in time_period 4 and 1 in time_period 5 means that 0 is used for 
        all time_periods on and after 4.
    
    NOTE: both dict_cat_to_vector and vec_ramp cannot be None. If both are None,
        returns df_input

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_cat_to_vector: dictionary mapping a technology category to two an 
        input vector. The vector uses a drop flag to (generally -999) to 
        identify time periods that are not subject to an MSP Max Prod; other 
        values greater than 0 are used to identify the maximum deviation 
        from the *last time period with a non-drop flag*, entered as a 
        proportion.
    - model_electricity: Electricity and Fuel Production model used to call 
        variables

    Keyword Arguments
    -----------------
    - drop_flag: value in 
        model_electricity.modvar_entc_max_elec_prod_increase_for_msp used to 
        signal the presence of no constraint. Defaults to 
        model_electricity.drop_flag_tech_capacities if None
    - vec_ramp: ramp vec used for implementation
        * NOTE: if dict_cat_to_vector, will defaulto cap hydro based on the 
            implementation schedule. If both 
    - **kwargs: passed to ade.transformations_general()
    """
    if (not isinstance(dict_cat_to_vector, dict)) & (not sf.islistlike(vec_ramp)):
        return df_input

    # initialize some key components
    drop_flag = model_electricity.drop_flag_tech_capacities if not sf.isnumber(drop_flag) else drop_flag
    modvar_msp_max = model_electricity.modvar_entc_max_elec_prod_increase_for_msp

    # check for variables and initialize fields_check as drops
    model_attributes = model_electricity.model_attributes
    fields_check = model_attributes.build_variable_fields(modvar_msp_max)
    df_out = df_input.copy()
    df_out[fields_check] = drop_flag
    

    # default specification is to cap hydro (no hydropower growth)
    vec_msp = (
        np.array([(drop_flag if (x == 0) else 0) for x in vec_ramp])
        if vec_ramp is not None
        else None
    )
    dict_cat_to_vector = (
        {
            "pp_hydropower": vec_msp
        }
        if not isinstance(dict_cat_to_vector, dict)
        else dict_cat_to_vector
    )

    for cat, vec in dict_cat_to_vector.items():
        
        dict_trans = {
            modvar_msp_max: {
                "bounds": (drop_flag, np.inf),
                "categories": [cat],
                "magnitude": vec,
                "magnitude_type": "vector_specification",
                "vec_ramp": vec
            }
        }
        
        # call general transformation
        df_out = transformation_general(
            df_out,
            model_attributes,
            dict_trans,
            **kwargs
        )

    return df_out



def transformation_entc_hydrogen_electrolysis(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    cats_to_apply: List[str] = ["fp_hydrogen_electrolysis"],
    cats_response: List[str] = [
        "fp_hydrogen_gasification", 
        "fp_hydrogen_reformation"
    ],
    field_region: str = "nation",
    **kwargs
    ) -> pd.DataFrame:
    """
    Implement the "Green hydrogen" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of target value
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - cats_to_apply: hydrogen production categories to apply magnitude to
    - cats_response: hydrogen production categories that respond to the target
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr_entc = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)
    attr_enfu = model_attributes.get_attribute_table(model_attributes.subsec_name_enfu)
    dict_tech_info = model_electricity.get_tech_info_dict(attribute_technology = attr_entc)
    modvar_msp = model_electricity.modvar_entc_nemomod_min_share_production

    cats_to_apply = [x for x in attr_entc.key_values if x in cats_to_apply]
    cats_response = [x for x in attr_entc.key_values if x in cats_response]

    if len(cats_to_apply) == 0:
        return df_input


    #vec_implementation_ramp_short = sf.vec_bounds(vec_ramp/min(vec_ramp[vec_ramp != 0]), (0, 1))
    vec_implementation_ramp_short = sf.vec_bounds(vec_ramp*2, (0, 1))

    # 
    df_transformed = transformation_general(
        df_input,
        model_attributes, 
        {
            modvar_msp: {
                "bounds": (0, 1),
                "categories": cats_to_apply,
                "magnitude": magnitude,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        **kwargs
    )

    # adjust other hydrogen shares
    if len(cats_response) > 0:

        vars_appplied = model_attributes.build_variable_fields(
            modvar_msp,
            restrict_to_category_values = cats_to_apply,
        )

        vars_respond = model_attributes.build_variable_fields(
            modvar_msp,
            restrict_to_category_values = cats_response,
        )

        arr_maintain = np.array(df_input[vars_appplied])
        vec_total_maintain = arr_maintain.sum(axis = 1)
        arr_adjust = np.array(df_input[vars_respond])
        vec_total_adjust = arr_adjust.sum(axis = 1)

        # if the total does not exceed 1, div by 1; if it does, divide by that total
        vec_exceed_one = sf.vec_bounds(vec_total_maintain + vec_total_adjust - 1, (0.0, np.inf))
        vec_scalar_adj = np.nan_to_num((vec_total_adjust - vec_exceed_one)/vec_total_adjust, 0.0, posinf = 0.0)

        arr_adjust = sf.do_array_mult(arr_adjust, vec_scalar_adj)

        for i, field in enumerate(vars_respond):
            df_transformed[field] = arr_adjust[:, i]

    return df_transformed



def transformation_entc_increase_efficiency_of_electricity_production(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    bounds: Tuple = (0, 0.9),
    field_region: str = "nation",
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase efficiency of electricity production" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase to apply, as additive factor, to
        energy technology efficiency factors
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - bounds: optional bounds on the efficiency. Default is maximum of 90%
        efficiency
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # only apply to generation techs
    categories = model_attributes.filter_keys_by_attribute(
        model_attributes.subsec_name_entc,
        {
            "power_plant": 1
        }
    )

    # iterate over categories to modify output data frame -- will use to copy into new variables
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_electricity.modvar_entc_efficiency_factor_technology: {
                "bounds": bounds,
                "categories": categories,
                "magnitude": magnitude,
                "magnitude_type": "baseline_additive",
                "vec_ramp": vec_ramp,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        **kwargs
    )

    return df_out



def transformation_entc_increase_renewables(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    field_region: str = "nation",
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase renewables" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase to apply to renewable energy minimum
        installed capacity. Entered as a scalar--for example, to double from
        existing (or planned) residual capacity, enter 2.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)

    # initialize output and categories
    df_out = df_input.copy()
    categories = model_attributes.filter_keys_by_attribute(
        model_attributes.subsec_name_entc,
        {
            "renewable_energy_technology": 1
        }
    )

    # source fields to target fields
    fields_source = model_attributes.build_variable_fields(
        model_electricity.modvar_entc_nemomod_residual_capacity,
        restrict_to_category_values = categories
    )
    fields_target = model_attributes.build_variable_fields(
        model_electricity.modvar_entc_nemomod_total_annual_min_capacity,
        restrict_to_category_values = categories
    )
    dict_source_to_target = dict(zip(fields_source, fields_target))

    # iterate over categories to modify output data frame -- will use to copy into new variables
    df_out_source = transformation_general(
        df_input,
        model_attributes,
        {
            model_electricity.modvar_entc_nemomod_residual_capacity: {
                "bounds": (0, np.inf),
                "categories": categories,
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        **kwargs
    )

    # copy into df_out (leave actual fields unchange)
    fields_ind = [field_region, model_attributes.dim_time_period]
    for k in fields_source:
        dict_update = sf.build_dict(
            df_out_source[fields_ind + [k]],
            dims = (2, 1)
        )

        new_col = sf.df_to_tuples(df_out[fields_ind])
        new_col = [dict_update.get(x) for x in new_col]
        df_out[dict_source_to_target.get(k)] = new_col

    return df_out



def transformation_entc_least_cost_solution(
    df_input: pd.DataFrame,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    drop_flag: Union[int, float, None] = None,
    field_region: str = "nation",
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "% of electricity is generated by renewables in 2050" 
        transformation. Applies to both renewable (true renewable) and fossil
        fuel (fake renewable) transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - drop_flag: value in 
        model_electricity.modvar_entc_max_elec_prod_increase_for_msp used to 
        signal the presence of no constraint. Defaults to 
        model_electricity.drop_flag_tech_capacities if None
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # initialize some key elements
    attr_entc = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)
    attr_enfu = model_attributes.get_attribute_table(model_attributes.subsec_name_enfu)
    drop_flag = model_electricity.drop_flag_tech_capacities if not sf.isnumber(drop_flag) else drop_flag
    dict_tech_info = model_electricity.get_tech_info_dict()
    fields_to_drop_flag = model_attributes.build_variable_fields(
        model_electricity.modvar_entc_max_elec_prod_increase_for_msp
    )
    
    # specify ramp vec t
    # this one will eliminate MSP and REMT immediately
    # 
    #vec_implementation_ramp_short = sf.vec_bounds(vec_ramp/min(vec_ramp[vec_ramp != 0]), (0, 1))
    vec_implementation_ramp_short = sf.vec_bounds(vec_ramp*2, (0, 1))
    
    
    # check for variables and initialize fields_check as drops
    df_transformed = df_input.copy()
    df_transformed[fields_to_drop_flag] = drop_flag

    # 
    df_transformed = transformation_general(
        df_transformed,
        model_attributes, 
        {
            model_electricity.modvar_entc_nemomod_min_share_production: {
                "bounds": (0, 1),
                "categories": dict_tech_info.get("all_techs_pp"),
                "magnitude": 0.0,
                "magnitude_type": "final_value",
                "vec_ramp": vec_implementation_ramp_short,
                "time_period_baseline": get_time_period(model_attributes, "max")
            },

            model_electricity.modvar_enfu_nemomod_renewable_production_target: {
                "bounds": (0, 1),
                "categories": [model_electricity.cat_enfu_elec],
                "magnitude": 0.0,
                "magnitude_type": "final_value",
                "vec_ramp": vec_implementation_ramp_short,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        **kwargs
    )

    return df_transformed



def transformation_entc_reduce_cost_of_renewables(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce cost of renewables" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: fractional scalar of reduction applied to final period. For
        example, to reduce costs by 30% by the final time period, enter 0.3
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # apply to capital, fixed, and variable for now
    modvars = [
        model_electricity.modvar_entc_nemomod_capital_cost,
        model_electricity.modvar_entc_nemomod_fixed_cost,
        model_electricity.modvar_entc_nemomod_variable_cost
    ]

    # get categories
    categories = model_attributes.filter_keys_by_attribute(
        model_attributes.subsec_name_entc,
        {
            "renewable_energy_technology": 1
        }
    )

    # setup dictionaries
    dict_base = {
        "bounds": (0, np.inf),
        "categories": categories,
        "magnitude": 1 - magnitude,
        "magnitude_type": "baseline_scalar",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }
    dict_run = dict((modvar, dict_base) for modvar in modvars)


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        **kwargs
    )

    return df_out



def transformation_entc_renewable_target(
    df_input: pd.DataFrame,
    magnitude_target: Union[float, str],
    vec_ramp: np.ndarray,
    model_electricity: ml.EnergyProduction,
    cats_entc_hydrogen: List[str] = [
        "fp_hydrogen_electrolysis",
        "fp_hydrogen_gasification",
        "fp_hydrogen_reformation"
    ],
    dict_cats_entc_max_investment: Union[Dict[str, np.ndarray], None] = None,
    drop_flag: Union[int, float, None] = None,
    factor_vec_ramp_msp: Union[float, int, None] = None,
    field_region: str = "nation",
    fuel_elec: Union[str, None] = None,
    fuel_hydg: str = "fuel_hydrogen",
    include_target: bool = True,
    magnitude_as_floor: bool = False,
    magnitude_renewables: Union[Dict[str, float], float, None] = None,
    scale_non_renewables_to_match_surplus_msp: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "% of electricity is generated by renewables in 2050" 
        transformation. Applies to both renewable (true renewable) and fossil
        fuel (fake renewable) transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude_target: magnitude of target to hit by 2050   OR   optional str 
        "VEC_FIRST_RAMP", which will set the magnitude to the mix of renewable
        capacity at the first time period where VEC_FIRST_RAMP != 0
    - model_electricity: EnergyProduction model used to define variables. Also 
        defines the internal model_atrributes variable to preserve logical
        consistency
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - cats_entc_hydrogen: categories used to produce hydrogen (may be subject to
        renewable energy targets)
    - dict_cats_entc_max_investment: dictionary of categories to place a cap on
        maximum investment for. Each key maps to a dictionary with two elements;
        one is a vector of values to use for the cap (-999 is used to implement
        no maximum--key "vec"), and the other is the type of vector, which can
        be either
            * "value" (use raw values) or 
            * "scalar" (scalar applied to maximum residual capacity over periods
                where vec_ramp = 0

        The dictionary should take the following form
        
            dict_cats_entc_max_investment = {
                cat_entc_1: {
                    "vec": [m_0, m_1, ... , m_{t - 1}],
                    "type": "value"
                },
                ...
            }

        where `cat_entc_i` is a category, `vec` gives values as numpy vector, and
        `type` gives the type of the time series.
    - drop_flag: value in 
        model_electricity.modvar_entc_max_elec_prod_increase_for_msp used to 
        signal the presence of no constraint. Defaults to 
        model_electricity.drop_flag_tech_capacities if None
    - factor_vec_ramp_msp: factor used to accelerate the rate at which
        MinShareProduction declines to 0 for non-renewable energy technologies.
        If None, defaults to 1.5 (1 is the same rate).
    - field_region: field in df_input that specifies the region
    - fuel_elec: $CAT-TECHNOLOGY$ category specifying electricity. If None, 
        defaults to model_electricity.cat_enfu_fuel
    - include_target: if True, sets the renewable target (Default). If False, 
        will only manipulate minimum shares of production. Should only be used
        in conjunction with `scale_non_renewables_to_match_surplus_msp = True`
    - magnitude_as_floor: if True, will not allow any renewables to decline in 
        magnitude unless the dictionary `magnitude_renewables` forces some 
        renewables to make up a higher share
    - magnitude_renewables: Dict mapping renewable categories to target minimum
        shares of production by the final time period OR float giving a uniform 
        value to apply to all renewable categories (as defined in 
        `cats_renewable`). If None, the renewable minimum share of production 
        for each renewable category is kept stable as soon as vec_remp != 0.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - scale_non_renewables_to_match_surplus_msp: if True, will scale MSP from
        non-renewable sources to match the surplus, where surplus is calculated
        as 
            surplus = max(MSP_0 - T, 0)

        where R is the original total of all MSPs and T is the total renewable
        target. If False, MSPs are set to 0 for all non-renewable targets.

    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    ##  INITIALIZATION

    # initialize some key elements
    model_attributes = model_electricity.model_attributes
    attr_entc = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)
    attr_enfu = model_attributes.get_attribute_table(model_attributes.subsec_name_enfu)
    
    # some info for drops related to MSP Max Prod increase relative to base (model_electricity.modvar_entc_max_elec_prod_increase_for_msp)
    drop_flag = model_electricity.drop_flag_tech_capacities if not sf.isnumber(drop_flag) else drop_flag
    fields_to_drop_flag = model_attributes.build_variable_fields(
        model_electricity.modvar_entc_max_elec_prod_increase_for_msp
    )

    # get fuel categories and technology dictionary
    fuel_elec = model_electricity.cat_enfu_elec if (fuel_elec not in attr_enfu.key_values) else fuel_elec
    fuel_hydg = "fuel_hydrogen" if (fuel_hydg not in attr_enfu.key_values) else fuel_hydg
    dict_tech_info = model_electricity.get_tech_info_dict()


    ##  CHECK SPECIFICATION OF INVESTMENT CAPS

    if isinstance(dict_cats_entc_max_investment, dict):
        keys_check = list(dict_cats_entc_max_investment.keys())

        for cat in keys_check:
            # check specification of tech
            keep_q = (cat in attr_entc.key_values)

            # check that value is a dictionary
            val = dict_cats_entc_max_investment.get(cat)
            keep_q = keep_q & isinstance(val, dict)

            # check that vectors are properly specified
            if keep_q:
                keep_q = keep_q & isinstance(dict_cats_entc_max_investment.get(cat).get("vec"), np.ndarray)
                keep_q = keep_q & (dict_cats_entc_max_investment.get(cat).get("type") in ["value", "scalar"])

            # remove the key if invalid
            if not keep_q:
                del dict_cats_entc_max_investment[cat]

    dict_cats_entc_max_investment = (
        None 
        if not isinstance(dict_cats_entc_max_investment, dict) 
        else (
            None
            if len(dict_cats_entc_max_investment) == 0
            else dict_cats_entc_max_investment
        )
    )


    ##  ITERATE OVER REGIONS TO APPLY TRANSFORMATION

    dict_vec_entc_msp_final_period = {} # must be available by region
    dfs = df_input.groupby(field_region)
    df_out = []

    # set some temporary fields used below
    field_total_mass_drops = "TMPFIELD_TOTAL_MASS_DROPS"
    field_total_mass_original = "TMPFIELD_TOTAL_MASS_ORIGINAL"
    field_total_surplus = "TMPFIELD_SURPLUS"

    for region, df in dfs:
        
        region = region[0] if isinstance(region, tuple) else region

        # init the magnitude for the region
        magnitude = magnitude_target

        # get renewable categories from input data frame; if none are specified, append and return (nothing to transform)
        dict_cats_renewable_all = get_renewable_categories_from_inputs_final_tp(
            df, 
            model_electricity,
        )

        if len(dict_cats_renewable_all) == 0:
            df_out.append(df)
            continue
        
        # only focus on electricity generation
        cats_renewable = [
            x for x in attr_entc.key_values 
            if x in dict_cats_renewable_all.keys()
            and x in dict_tech_info.get("all_techs_pp")
        ]
        inds_renewable = [attr_entc.get_key_value_index(x) for x in cats_renewable]
        inds_elec = [attr_entc.get_key_value_index(x) for x in dict_tech_info.get("all_techs_pp")]

        # check and clean renewable production target specification
        magnitude_renewables = (
            dict(
                (k, min(max(v, 0.0), 1.0)) 
                for k, v in magnitude_renewables.items() 
                if (k in cats_renewable) 
                and (isinstance(v, float) or isinstance(v, int)) 
            ) 
            if isinstance(magnitude_renewables, dict)
            else (
                None
                if not (isinstance(magnitude_renewables, float) or isinstance(magnitude_renewables, int)) 
                else min(max(magnitude_renewables, 0.0), 1.0)
            )
        )
        
    
        ##  INITIALIZE OUTPUT AND LOOP OVER REGION GROUPS

        # technologies to reduce MinShareProduction values to 0 and those to *avoid* reducing to 0
        cats_entc_drop = [x for x in dict_tech_info.get("all_techs_pp") if x not in cats_renewable]
        cats_entc_no_drop = [x for x in dict_tech_info.get("all_techs_pp") if x not in cats_entc_drop]
        inds_entc_drop = [attr_entc.get_key_value_index(x) for x in cats_entc_drop]
        inds_entc_no_drop = [attr_entc.get_key_value_index(x) for x in cats_entc_no_drop]

        # first index where vec_ramp starts to deviate from zero
        ind_vec_ramp_first_zero_deviation = np.where(np.array(vec_ramp) != 0)[0][0]



        ##################################################
        #    1. IMPLEMENT RENEWABLE GENERATION TARGET    #
        ##################################################

        # setup renewable specification
        for cat in attr_entc.key_values:
            field = model_attributes.build_variable_fields(
                model_electricity.modvar_entc_nemomod_renewable_tag_technology,
                restrict_to_category_values = cat
            )

            if isinstance(field, str):
                df[field] = int(cat in cats_renewable)

        # make an adjustment too identify the magnitude if it hasn't been dealt with yet
        if magnitude == "VEC_FIRST_RAMP":

            arr_entc_residual_capacity = model_attributes.extract_model_variable(#
                df,
                model_electricity.modvar_entc_nemomod_residual_capacity,
                expand_to_all_cats = True,
                return_type = "array_base",
            )

            magnitude = arr_entc_residual_capacity[ind_vec_ramp_first_zero_deviation, inds_renewable].sum()
            magnitude /= arr_entc_residual_capacity[ind_vec_ramp_first_zero_deviation, :].sum()


        # get the current minimum share of production (does not change when df_transformed is assigned below, since that only modifies the renewable target)
        arr_entc_min_share_production = model_attributes.extract_model_variable(#
            df,
            model_electricity.modvar_entc_nemomod_min_share_production,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        vec_entc_msp_total_mass_original = (
            arr_entc_min_share_production[:, inds_entc_drop + inds_entc_no_drop]
            .sum(axis = 1)
        )
        vec_entc_msp_final_period = arr_entc_min_share_production[-1, :]
        dict_vec_entc_msp_final_period.update({region: vec_entc_msp_final_period})


        # add the total original mass to the dataframe so it can be accessed later (and dropped)
        df[field_total_mass_original] = vec_entc_msp_total_mass_original


        # next, if setting the magnitude as a floor, don't allow renewables to decline--set the total magnitude based categorically (note that MSP floors can override these in modeling)
        magnitude_renewables_by_region = magnitude_renewables

        if magnitude_as_floor:

            mag_base = 0.0
            magnitude_renewables_by_region = {}
            
            for i, cat in enumerate(cats_renewable):
                ind = inds_renewable[i]

                msp_min_specified = magnitude_renewables.get(cat, 0.0)
                cat_target = max(msp_min_specified, vec_entc_msp_final_period[ind])
                mag_base += cat_target

                magnitude_renewables_by_region.update({cat: cat_target})


            # get magnitude and scale
            magnitude = max(mag_base, magnitude)

            if (magnitude > 1):
                magnitude_renewables_by_region = dict((k, v/magnitude) for k, v in magnitude_renewables_by_region.items())

            if scale_non_renewables_to_match_surplus_msp:
                v_total = sum(list(magnitude_renewables_by_region.values()))
                magnitude_renewables_by_region = dict(
                    (k, np.nan_to_num(magnitude*v/v_total, 0.0))
                    for k, v in magnitude_renewables_by_region.items()
                )


        # iterate over categories to modify output data frame -- will use to copy into new variables
        df_transformed = (
            transformation_general(
                df,
                model_attributes, 
                {
                    model_electricity.modvar_enfu_nemomod_renewable_production_target: {
                        "bounds": (0, 1),
                        "categories": [model_electricity.cat_enfu_elec],
                        "magnitude": magnitude,
                        "magnitude_type": "final_value",
                        "vec_ramp": vec_ramp,
                        "time_period_baseline": get_time_period(model_attributes, "max")
                    }
                },
                field_region = field_region,
                **kwargs
            )
            if include_target
            else df
        )

        
        #########################################################
        #    2. ADD IN MINIMUM SHARES FOR CERTAIN RENEWABLES    #
        #########################################################

        """
        - NOTE: renewables should not decline as MSP diminishes and increase
            again as renewable targets increase. This step ensures that 
            renewables stay stable and are scaled appropriately with the 
            renewable generation target.

            However, some renewables may need to decline to meet specified 
            shares. E.g., if hydropower is 70% of power, and the future calls
            for 40% to come from solar, wind, and geothermal, hydropower should
            decline by 10%. 


        Here, get the target total magnitude of MSP of renewables and a scalar
            to apply to them to ensure they do not exceed REMinProductionTarget,
            then verify sum and scale if necessary; 
        
            * if magnitude > total_magnitude_msp_renewables, returns a scalar of 
                1 (no modifiation necessary)
            * if magnitude < total_magnitude_msp_renewables, returns scalar to 
                apply to specified MSP to ensure does not exceed magnitude
        """;

        # get the total magnitude of MSP of renewables that are specified in the dictionary, not the target total
        total_magnitude_msp_renewables = (
            (
                np.array(list(magnitude_renewables_by_region.values())).sum() 
                if isinstance(magnitude_renewables_by_region, dict)
                else magnitude_renewables_by_region*len(cats_renewable)
            ) 
            if magnitude_renewables_by_region is not None 
            else arr_entc_min_share_production[ind_vec_ramp_first_zero_deviation, inds_renewable].sum()
        )
        scalar_renewables_div = min(magnitude/total_magnitude_msp_renewables, 1.0)

        
        if isinstance(magnitude_renewables_by_region, dict):

            # apply to each category - slightly slower
            for cat, mag in magnitude_renewables_by_region.items():

                df_transformed = transformation_general(
                    df_transformed,
                    model_attributes, 
                    {
                        model_electricity.modvar_entc_nemomod_min_share_production: {
                            "bounds": (0, 1),
                            "categories": [cat],
                            "magnitude": mag*scalar_renewables_div,
                            "magnitude_type": "final_value",
                            "vec_ramp": vec_ramp,
                            "time_period_baseline": get_time_period(model_attributes, "max")
                        }
                    },
                    field_region = field_region,
                    **kwargs
                )
        

        elif isinstance(magnitude_renewables_by_region, float):
            
            # apply to all categories at once
            df_transformed = transformation_general(
                    df_transformed,
                    model_attributes, 
                    {
                        model_electricity.modvar_entc_nemomod_min_share_production: {
                            "bounds": (0, 1),
                            "categories": cats_entc_no_drop,
                            "magnitude": magnitude_renewables_by_region*scalar_renewables_div,
                            "magnitude_type": "final_value",
                            "vec_ramp": vec_ramp,
                            "time_period_baseline": get_time_period(model_attributes, "max")
                        }
                    },
                    field_region = field_region,
                    **kwargs
                )

        else:
            
            # maintain final value
            for cat in cats_entc_no_drop:

                ind_cat_cur = attr_entc.get_key_value_index(cat)

                df_transformed = transformation_general(
                    df_transformed,
                    model_attributes, 
                    {
                        model_electricity.modvar_entc_nemomod_min_share_production: {
                            "bounds": (0, 1),
                            "categories": [cat],
                            "magnitude": arr_entc_min_share_production[ind_vec_ramp_first_zero_deviation, ind_cat_cur]*scalar_renewables_div,
                            "magnitude_type": "final_value",
                            "vec_ramp": vec_ramp,
                            "time_period_baseline": get_time_period(model_attributes, "max")
                        }
                    },
                    field_region = field_region,
                    **kwargs
                )


        #############################################################################################
        #    3. CHECK FOR SUM OF TOTAL RENEWABLES AND ADJUST NON-SPECIFIED DOWNWARD IF NECESSARY    #
        #############################################################################################

        cats_renewable_unspecified = (
            [x for x in cats_renewable if x not in magnitude_renewables_by_region.keys()] 
            if isinstance(magnitude_renewables_by_region, dict)
            else None
        ) 
        cats_renewable_unspecified = (
            None 
            if (len(cats_renewable_unspecified) == 0) 
            else cats_renewable_unspecified
        )
        inds_renewable_unspecified = (
            [attr_entc.get_key_value_index(x) for x in cats_renewable_unspecified]
            if cats_renewable_unspecified is not None
            else None
        )
        inds_renewable_specified = (
            [x for x in inds_renewable if x not in inds_renewable_unspecified]
            if inds_renewable_unspecified is not None
            else inds_renewable
        )


        ##  3A: MODIFY FRACTIONS FOR UNSPECIFIED/SPECIFIED RENEWABLES (IN ORDER)

        # refresh and get totals
        arr_entc_min_share_production = model_attributes.extract_model_variable(#
            df_transformed,
            model_electricity.modvar_entc_nemomod_min_share_production,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        
        # get total MSP for renewables + total for unspecified/specified
        vec_entc_total_msp_renewables = arr_entc_min_share_production[:, inds_renewable].sum(axis = 1)
        vec_entc_total_msp_renewables_unspecified = (
            arr_entc_min_share_production[:, inds_renewable_unspecified].sum(axis = 1)
            if inds_renewable_unspecified is not None
            else np.zeros(len(arr_entc_min_share_production))
        )

        vec_entc_total_msp_renewables_specified = vec_entc_total_msp_renewables - vec_entc_total_msp_renewables_unspecified
        vec_entc_total_msp_renewable_cap = sf.vec_bounds(vec_entc_total_msp_renewables, (0, 1.0))
        

        if cats_renewable_unspecified is not None:
            
            ##  3.A.I FIRST, MODIFY SPECIFIED RENEWABLES TO PREVENT EXCEEDING 1

            # bound total MSP, then scale specified categories if necesary
            vec_entc_msp_renewable_specified_cap = sf.vec_bounds(
                vec_entc_total_msp_renewables_specified, 
                (0, magnitude)
            )
            
            vec_entc_scale_msp_renewable_specified = np.nan_to_num(
                vec_entc_msp_renewable_specified_cap/vec_entc_total_msp_renewables_specified,
                1.0,
            )

            # get the fields to scale
            fields_scale = model_attributes.build_variable_fields(
                model_electricity.modvar_entc_nemomod_min_share_production,
                restrict_to_category_values = list(magnitude_renewables_by_region.keys())
            )
            
            for field in fields_scale:
                df_transformed[field] = np.array(df_transformed[field])*vec_entc_scale_msp_renewable_specified


            ##  3.A.II NEXT, SCALE UNSPECIFIED RENEWABLES IF NECESSARY
            
            # refresh and get totals
            arr_entc_min_share_production = model_attributes.extract_model_variable(#
                df_transformed,
                model_electricity.modvar_entc_nemomod_min_share_production,
                expand_to_all_cats = True,
                return_type = "array_base",
            )
            
            # get total MSP for renewables + total for unspecified/specified
            vec_entc_total_msp_renewables = arr_entc_min_share_production[:, inds_renewable].sum(axis = 1)
            vec_entc_total_msp_renewables_unspecified = (
                arr_entc_min_share_production[:, inds_renewable_unspecified].sum(axis = 1)
                if inds_renewable_unspecified is not None
                else np.zeros(len(arr_entc_min_share_production))
            )
            vec_entc_total_msp_renewables_specified = vec_entc_total_msp_renewables - vec_entc_total_msp_renewables_unspecified
            vec_entc_total_msp_renewable_cap = sf.vec_bounds(vec_entc_total_msp_renewables, (0, magnitude))

            # scale unspecfied categories downward
            vec_scale_match_unspecified = vec_entc_total_msp_renewable_cap - vec_entc_total_msp_renewables_specified
            vec_scale_unspecified = np.nan_to_num(
                vec_scale_match_unspecified/vec_entc_total_msp_renewables_unspecified,
                posinf = 0.0
            )

            # get the fields to scale (unspecified)
            fields_scale = model_attributes.build_variable_fields(
                model_electricity.modvar_entc_nemomod_min_share_production,
                restrict_to_category_values = cats_renewable_unspecified
            )

            for field in fields_scale:
                df_transformed[field] = np.array(df_transformed[field])*vec_scale_unspecified

        

        #################################################
        #    4. ADD IN MAXIMUM TECHNOLOGY INVESTMENT    #
        #################################################

        if (dict_cats_entc_max_investment is not None):
            
            arr_entc_max_investment = model_attributes.extract_model_variable(#
                df_transformed,
                model_electricity.modvar_entc_nemomod_total_annual_max_capacity_investment,
                expand_to_all_cats = True,
                return_type = "array_base",
            )

            # get maximum residual capacities by technology
            vec_entc_max_capacites = model_attributes.extract_model_variable(#
                df_transformed,
                model_electricity.modvar_entc_nemomod_residual_capacity,
                expand_to_all_cats = True,
                return_type = "array_base",
            )
            vec_entc_max_capacites = np.max(vec_entc_max_capacites, axis = 0)

            # iterate over categories
            for cat in dict_cats_entc_max_investment.keys():
                
                dict_cur = dict_cats_entc_max_investment.get(cat)

                vec_repl = dict_cur.get("vec").copy()
                type_repl = dict_cur.get("type")
                
                if len(vec_repl) == len(arr_entc_max_investment):

                    # get category index
                    ind_repl = attr_entc.get_key_value_index(cat)

                    # clean and prepareinput vector
                    np.put(vec_repl, np.where(vec_repl < 0)[0], model_electricity.drop_flag_tech_capacities)
                    w = np.where(vec_repl != model_electricity.drop_flag_tech_capacities)
                    
                    vals_new = vec_repl[w]*vec_entc_max_capacites[ind_repl]
                    np.put(vec_repl, w, vals_new) if (type_repl == "scalar") else None

                    # overwrite if valid
                    arr_entc_max_investment[:, ind_repl] = vec_repl

            
            arr_entc_max_investment = model_attributes.array_to_df(
                arr_entc_max_investment, 
                model_electricity.modvar_entc_nemomod_total_annual_max_capacity_investment,
                reduce_from_all_cats_to_specified_cats = True
            )

            # overwrite in df_transformed
            for fld in arr_entc_max_investment.columns:
                if fld in df_transformed.columns:
                    df_transformed[fld] = np.array(arr_entc_max_investment[fld])

        df_out.append(df_transformed)


    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)



    ###############################################################
    #    5. MODIFY ANY SPECIFIED MINSHARE PRODUCTION SPECIFIED    #
    ###############################################################

    # very aggressive, turns off any MSP as soon as a target goes above 0
    # vec_implementation_ramp_short = sf.vec_bounds(vec_ramp/min(vec_ramp[vec_ramp != 0]), (0, 1))
    factor_vec_ramp_msp = (
        1.25 
        if not (isinstance(factor_vec_ramp_msp, float) or isinstance(factor_vec_ramp_msp, int)) 
        else max(1.0, factor_vec_ramp_msp)
    )
    vec_implementation_ramp_short = sf.vec_bounds(vec_ramp*factor_vec_ramp_msp, (0.0, 1.0))

    # if scaling MSPs, calculate here
    if scale_non_renewables_to_match_surplus_msp:

        # get current status of minimum share of production after transforming
        arr_entc_min_share_production = model_attributes.extract_model_variable(
            df_out,
            model_electricity.modvar_entc_nemomod_min_share_production,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # get total for categories that were specified
        vec_entc_msp_total_mass_drops = arr_entc_min_share_production[:, inds_entc_drop].sum(axis = 1)
        vec_entc_msp_total_mass_no_drops = arr_entc_min_share_production[:, inds_entc_no_drop].sum(axis = 1)
        vec_entc_msp_total_mass_original = np.array(df_out[field_total_mass_original])
        vec_entc_msp_surplus = sf.vec_bounds(
            vec_entc_msp_total_mass_original - vec_entc_msp_total_mass_no_drops, 
            (0.0, 1.0)
        )
        
        # add temporary fields (defined above)
        df_out[field_total_mass_drops] = vec_entc_msp_total_mass_drops
        df_out[field_total_surplus] = vec_entc_msp_surplus

        # get the current totals of fields to drop, then and scale to match total surplus
        fields_drop = model_attributes.build_variable_fields(
            model_electricity.modvar_entc_nemomod_min_share_production,
            restrict_to_category_values = cats_entc_drop
        )

        vec_entc_fields_to_scale = np.array(df_out[fields_drop]).sum(axis = 1)
        vec_entc_scalar_drops_to_surplus = np.nan_to_num(
            vec_entc_msp_surplus/vec_entc_fields_to_scale,
            0.0
        )

        for field in fields_drop:
            df_out[field] = np.array(df_out[field])*vec_entc_scalar_drops_to_surplus
        

        # group by regions 
        dfg = df_out.groupby([field_region])
        df_new = []
        
        for region, df_cur in dfg:
            
            region = region[0] if isinstance(region, tuple) else region

            # get original total MSP accounted for 
            vec_entc_msp_final_period = dict_vec_entc_msp_final_period.get(region)
            if vec_entc_msp_final_period is None:
                msg = f"Error in transformation_entc_renewable_target(): no final time period MSP found for region {region}."
                raise RuntimeError(msg)

            total_msp_original_drops = vec_entc_msp_final_period[inds_entc_drop].sum()

            # get vectors
            vec_entc_msp_total_mass_cur = np.array(df_cur[field_total_mass_original])
            vec_entc_msp_total_mass_drops_cur = np.array(df_cur[field_total_mass_drops])
            vec_entc_msp_surplus_cur = np.array(df_cur[field_total_surplus])

            # scale the surplus -- mix between original vector and target vector, which will have same ceiling
            for i, cat in enumerate(cats_entc_drop):
                
                field_cat = model_attributes.build_variable_fields(
                    model_electricity.modvar_entc_nemomod_min_share_production,
                    restrict_to_category_values = cat
                )

                # get target for MSP + current 
                ind = inds_entc_drop[i]
                vec_target = vec_entc_msp_surplus_cur*vec_entc_msp_final_period[ind]/total_msp_original_drops
                vec_cur = np.array(df_cur[field_cat])

                vec_new = (1.0 - vec_implementation_ramp_short)*vec_cur
                vec_new += vec_implementation_ramp_short*vec_target

                df_cur[field_cat] = vec_new

            df_new.append(df_cur)

        # drop temporary fields
        df_out = (
            pd.concat(df_new, axis = 0)
            .drop([field_total_mass_original, field_total_mass_drops, field_total_surplus], axis = 1)
            .reset_index(drop = True)
        )


    else:

        df_out = transformation_general(
            df_out,
            model_attributes,
            {
                model_electricity.modvar_entc_nemomod_min_share_production: {
                    "bounds": (0, 1),
                    "categories": cats_entc_drop,
                    "magnitude": 0.0,
                    "magnitude_type": "final_value",
                    "vec_ramp": vec_implementation_ramp_short,
                    "time_period_baseline": get_time_period(model_attributes, "max")
                }
            },
            field_region = field_region,
            **kwargs
        )

        # drop the total MSP mass column (unused)
        (
            df_out.drop([field_total_mass_original], axis = 1, inplace = True)
            if field_total_mass_original in df_out.columns
            else None
        )



    #####################################################################
    #    6. BOUND EVERYTHING (CAN APPLY TO ALL REGIONS AT SAME TIME)    #
    #####################################################################

    arr_entc_min_share_production = model_attributes.extract_model_variable(
        df_out,
        model_electricity.modvar_entc_nemomod_min_share_production,
        expand_to_all_cats = True,
        return_type = "array_base",
    )

    vec_entc_total_msp_total = arr_entc_min_share_production[:, inds_elec].sum(axis = 1)
    vec_entc_msp_specified_cap = sf.vec_bounds(vec_entc_total_msp_total, (0, 1.0))
    vec_entc_scale_msp = vec_entc_msp_specified_cap/vec_entc_total_msp_total

    # get the fields to scale
    fields_scale = model_attributes.build_variable_fields(
        model_electricity.modvar_entc_nemomod_min_share_production,
        restrict_to_category_values = dict_tech_info.get("all_techs_pp"),
    )
    
    for field in fields_scale:
        df_out[field] = np.array(df_out[field])*vec_entc_scale_msp


    return df_out



def transformation_entc_specify_transmission_losses(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    field_region: str = "nation",
    magnitude_type: str = "final_value",
    min_loss: Union[float, None] = 0.02,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce transmission losses" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of transmission loss in final time period as:
        * scalar (if `magnitude_type == "basline_scalar"`)
        * final value (if `magnitude_type == "final_value"`)
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - magnitude_type: string 
    - min_loss: minimum feasible transmission loss. If None, transmission losses
        can be reduced to 0
    - **kwargs: passed to transformation_general()
    """

    magnitude_type = (
        "baseline_scalar" 
        if (magnitude_type not in ["baseline_scalar", "final_value"]) 
        else magnitude_type
    )

    var_bound = model_attributes.build_variable_fields(
        model_electricity.modvar_enfu_transmission_loss_frac_electricity,
        restrict_to_category_values = model_electricity.cat_enfu_elec,
    )

    if magnitude_type == "basline_scalar":
        # call general transformation
        df_out = transformation_general(
            df_input,
            model_attributes,
            {
                model_electricity.modvar_enfu_transmission_loss_frac_electricity: {
                    "bounds": (0, 1),
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
                    "time_period_baseline": get_time_period(model_attributes, "max"),
                    "vec_ramp": vec_ramp
                }
            },
            field_region = field_region,
            **kwargs
        )

    # group by region to ensure the specified value is not higher than the actual value
    elif magnitude_type == "final_value":

        df_out_list = []
        dfs_out = df_input.groupby([field_region])

        for i, df in dfs_out:
            
            i = i[0] if isinstance(i, tuple) else i

            val_final = float(
                df[
                    df[model_attributes.dim_time_period] == get_time_period(model_attributes, "max")
                ][var_bound]
            )
            magnitude = min(magnitude, val_final)

            df_trns = transformation_general(
                df,
                model_attributes,
                {
                    model_electricity.modvar_enfu_transmission_loss_frac_electricity: {
                        "bounds": (0, 1),
                        "magnitude": magnitude,
                        "magnitude_type": "final_value",
                        "time_period_baseline": get_time_period(model_attributes, "max"),
                        "vec_ramp": vec_ramp
                    }
                },
                field_region = field_region,
                **kwargs
            )

            df_out_list.append(df_trns)
        
        df_out = pd.concat(df_out_list, axis = 0).reset_index(drop = True)


    # bound losses
    if isinstance(min_loss, float) or isinstance(min_loss, int):
        min_loss = max(min_loss, 0.0)
        df_out[var_bound] = sf.vec_bounds(np.array(df_out[var_bound]), (min_loss, 1.0))

    return df_out



def transformation_entc_retire_fossil_fuel_early(
    df_input: pd.DataFrame,
    dict_categories_to_vec_ramp: Dict[str, np.ndarray],
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.EnergyProduction,
    magnitude: float = 0.0,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Production Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_categories_to_vec_ramp: dictionary mapping categories to ramp vector
        to use for retirement
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: EnergyProduction model used to define variables

    Keyword Arguments
    -----------------
    - magnitude: final magnitude of generation capacity.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)

    # initialize output
    df_out = df_input.copy()

    # iterate over categories to modify output data frame
    for cat in dict_categories_to_vec_ramp.keys():
        if cat in attr.key_values:
            vec_ramp = dict_categories_to_vec_ramp.get(cat)

            if isinstance(vec_ramp, np.ndarray):
                df_out = transformation_general(
                    df_out,
                    model_attributes,
                    {
                        model_electricity.modvar_entc_nemomod_residual_capacity: {
                            "bounds": (0, np.inf),
                            "categories": [cat],
                            "magnitude": 0.0,
                            "magnitude_type": "final_value",
                            "vec_ramp": vec_ramp,
                            "time_period_baseline": get_time_period(model_attributes, "max")
                        }
                    },
                    **kwargs
                )
    return df_out




############################################
#    FUGITIVE EMISSIONS TRANSFORMATIONS    #
############################################

def transformation_fgtv_maximize_flaring(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: me.EnergyConsumption,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Flaring" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to time 0
        (interpreted as a magnitude change, not a scalar change).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_fgtv_frac_non_fugitive_flared: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out



def transformation_fgtv_reduce_leaks(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Leaks" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: fractional magnitude of reduction in leaks relative to time 0
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_fgtv_frac_reduction_fugitive_leaks: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar_diff_reduction",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out






###########################################
#    INDUSTRIAL ENERGY TRANSFORMATIONS    #
###########################################

def transformation_inen_maximize_energy_efficiency(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Energy Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period.
            * Interpreted as a magnitude increase in the industrial efficiency
                factor--e.g., to increase the efficiency factor by 0.3 by the
                final time period, enter 0.3.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_enfu_efficiency_factor_industrial_energy: {
                "bounds": (0, np.inf),
                "magnitude": magnitude,
                "magnitude_type": "baseline_additive",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_inen_maximize_production_efficiency(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Production Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional reduction in demand relative to final time period
        (to reduce production energy demand by 30% by the final time period,
        enter magnitude = 0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_inen_demscalar: {
                "bounds": (0, 1),
                "magnitude": 1 - magnitude,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_inen_shift_modvars(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    dict_modvar_specs: Union[Dict[str, float], None] = None,
    field_region: str = "nation",
    magnitude_relative_to_baseline: bool = False,
    model_energy: Union[me.EnergyConsumption, None] = None,
    regions_apply: Union[List[str], None] = None,
    return_modvars_only: bool = False,
    strategy_id: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Implement fuel switch transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: target magnitude of fuel mixture
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: INEN categories to apply transformation to
    - dict_modvar_specs: dictionary of targets modvars to shift into (assumes
        that will take from others). Maps from modvar to fraction of magnitude.
        Sum of values must == 1.
    - field_region: field in df_input that specifies the region
    - magnitude_relative_to_baseline: apply the magnitude relative to baseline?
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - return_modvars_only: return the model variables that define fuel fractions
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # model variables to explore
    model_energy = (
        me.EnergyConsumption(model_attributes) 
        if not isinstance(model_energy, me.EnergyConsumption) 
        else model_energy
    )
    modvars = model_energy.modvars_inen_list_fuel_fraction
    if return_modvars_only:
        return modvars

    # dertivative vars (alphabetical)
    all_regions = sorted(list(set(df_input[field_region])))
    attr_inen = model_attributes.get_attribute_table(model_attributes.subsec_name_inen)
    attr_time_period = model_attributes.get_dimensional_attribute_table(model_attributes.dim_time_period)

    df_out = []
    regions_apply = (
        all_regions 
        if (regions_apply is None) 
        else [x for x in regions_apply if x in all_regions]
    )

    dict_modvar_specs_def = {model_energy.modvar_inen_frac_en_electricity: 1}
    dict_modvar_specs = dict_modvar_specs_def if not isinstance(dict_modvar_specs, dict) else dict_modvar_specs
    dict_modvar_specs = dict(
        (k, v) for k, v in dict_modvar_specs.items() 
        if (k in modvars) and (isinstance(v, int) or isinstance(v, float))
    )
    dict_modvar_specs = dict_modvar_specs_def if (sum(list(dict_modvar_specs.values())) != 1.0) else dict_modvar_specs

    # get model variables and filter categories
    modvars_source = [x for x in modvars if x not in dict_modvar_specs.keys()]
    modvars_target = [x for x in modvars if x in dict_modvar_specs.keys()]
    cats_all = [set(model_attributes.get_variable_categories(x)) for x in modvars_target]
    cats_all = set.intersection(*cats_all)

    categories = (
        [x for x in attr_inen.key_values if x in categories]
        if isinstance(categories, list)
        else attr_inen.key_values
    )
    cats_all = (
        set(cats_all) & set(categories)
        if len(categories) > 0
        else None
    )

    if cats_all is None:
        return df_input

    subsec = model_attributes.subsec_name_inen


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = (
            df_input[
                df_input[field_region] == region
            ]
            .sort_values(by = [model_attributes.dim_time_period])
            .reset_index(drop = True)
        )
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        if region in regions_apply:
            for cat in cats_all:

                fields = [
                    model_attributes.build_variable_fields(
                        x,
                        restrict_to_category_values = cat
                    ) for x in modvars_target
                ]

                vec_initial_vals = np.array(df_in[fields].iloc[0]).astype(float)
                val_initial_target = vec_initial_vals.sum() if magnitude_relative_to_baseline else 0.0
                vec_initial_distribution = np.nan_to_num(vec_initial_vals/vec_initial_vals.sum(), 1.0, posinf = 1.0)

                # get the current total value of fractions
                vec_final_vals = np.array(df_in[fields].iloc[n_tp - 1]).astype(float)
                val_final_target = sum(vec_final_vals)

                target_value = float(sf.vec_bounds(magnitude + val_initial_target, (0.0, 1.0)))#*dict_modvar_specs.get(modvar_target)
                scale_non_elec = np.nan_to_num((1 - target_value)/(1 - val_final_target), 0.0, posinf = 0.0)

                target_distribution = magnitude*np.array([dict_modvar_specs.get(x) for x in modvars_target]) + val_initial_target*vec_initial_distribution
                target_distribution /= max(magnitude + val_initial_target, 1.0) 
                target_distribution = np.nan_to_num(target_distribution, 0.0, posinf = 0.0)

                dict_target_distribution = dict((x, target_distribution[i]) for i, x in enumerate(modvars_target))

                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_variable_fields(
                        modvar,
                        restrict_to_category_values = cat,
                    )

                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (
                        np.nan_to_num(val_final, 0.0, posinf = 0.0)*scale_non_elec 
                        if (modvar not in modvars_target) 
                        else dict_target_distribution.get(modvar)
                    )
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
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





##############################
#    SCOE TRANSFORMATIONS    #
##############################

def transformation_scoe_electrify_category_to_target(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_elec: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.EnergyConsumption, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement the "Switch to electricity for heat" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of final proportion of heat energy that is
        electrified for each category
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = (
        me.EnergyConsumption(model_attributes) 
        if not isinstance(model_energy, me.EnergyConsumption) 
        else model_energy
    )
    all_regions = sorted(list(set(df_input[field_region])))
    
    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.get_dimensional_attribute_table(model_attributes.dim_time_period)
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]
    subsec = model_attributes.subsec_name_scoe

    # model variables to explore
    modvars = [
        model_energy.modvar_scoe_frac_heat_en_coal,
        model_energy.modvar_scoe_frac_heat_en_diesel,
        model_energy.modvar_scoe_frac_heat_en_electricity,
        model_energy.modvar_scoe_frac_heat_en_gasoline,
        model_energy.modvar_scoe_frac_heat_en_hydrogen,
        model_energy.modvar_scoe_frac_heat_en_kerosene,
        model_energy.modvar_scoe_frac_heat_en_natural_gas,
        model_energy.modvar_scoe_frac_heat_en_hgl,
        model_energy.modvar_scoe_frac_heat_en_solid_biomass
    ]


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        # get electric categories and build dictionary of target values
        cats_elec_all = model_attributes.get_variable_categories(model_energy.modvar_scoe_frac_heat_en_electricity)
        cats_elec = [x for x in cats_elec_all if x in cats_elec] if isinstance(cats_elec, list) else cats_elec_all
        dict_targets_final_tp = dict((x, magnitude) for x in cats_elec)


        if region in regions_apply:
            for cat in dict_targets_final_tp.keys():
                field_elec = model_attributes.build_variable_fields(
                    model_energy.modvar_scoe_frac_heat_en_electricity,
                    restrict_to_category_values = cat,
                )

                val_final_elec = float(df_in[field_elec].iloc[n_tp - 1])
                target_value = min(max(dict_targets_final_tp.get(cat) + val_final_elec, 0), 1)
                scale_non_elec = 1 - target_value

                # get model variables that need to be adjusted
                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_variable_fields(
                        modvar,
                        restrict_to_category_values = cat,
                    )

                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (val_final/(1 - val_final_elec))*scale_non_elec if (field_cur != field_elec) else target_value
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
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



def transformation_scoe_increase_energy_efficiency_heat(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase efficiency of fuel for heat" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as an additive change to the baseline value--e.g.,
        an increase of 0.25 in the efficiency factor relative to the final time
        period is entered as 0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    dict_base = {
        "bounds": (0, 1),
        "magnitude": magnitude,
        "magnitude_type": "baseline_additive",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }

    modvars = [
        model_energy.modvar_scoe_efficiency_fact_heat_en_coal,
        model_energy.modvar_scoe_efficiency_fact_heat_en_diesel,
        #model_energy.modvar_scoe_efficiency_fact_heat_en_electricity,
        model_energy.modvar_scoe_efficiency_fact_heat_en_gasoline,
        model_energy.modvar_scoe_efficiency_fact_heat_en_hydrogen,
        model_energy.modvar_scoe_efficiency_fact_heat_en_kerosene,
        model_energy.modvar_scoe_efficiency_fact_heat_en_natural_gas,
        model_energy.modvar_scoe_efficiency_fact_heat_en_hgl,
        model_energy.modvar_scoe_efficiency_fact_heat_en_solid_biomass
    ]

    dict_run = dict((modvar, dict_base) for modvar in modvars)


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        **kwargs
    )

    return df_out



def transformation_scoe_reduce_demand_for_appliance_energy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase appliance efficiency" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in electric energy demand relative to 
        final time period (interpreted as an proportional scalar--e.g., a 30% 
        retuction in  electric energy demand in the final time period is entered 
        as 0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_scoe_demscalar_elec_energy_demand : {
                "bounds": (0, np.inf),
                "magnitude": float(sf.vec_bounds(1 - magnitude, (0, np.inf))),
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }

        },
        **kwargs
    )

    return df_out



def transformation_scoe_reduce_demand_for_heat_energy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce demand for heat energy" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in heat energy demand relative final
        time period (interpreted as an proportional scalar--e.g.,
        an 30% in heat energy demand in the final time period is entered as
        0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_scoe_demscalar_heat_energy_demand : {
                "bounds": (0, np.inf),
                "magnitude": float(sf.vec_bounds(1 - magnitude, (0, np.inf))),
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out





###############################################
#    TRANSPORTATION DEMAND TRANSFORMATIONS    #
###############################################

def transformation_trde_reduce_demand(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in demand relative to the final time
        period (interprted as a scalar change; i.e., enter 0.25 to reduce demand
        by 25% by the final time period).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trde_demand_scalar: {
                "magnitude": 1 - magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )

    return df_out



########################################
#    TRANSPORTATION TRANSFORMATIONS    #
########################################

def transformation_trns_fuel_shift_to_target(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    baseline_period: str = "final",
    categories: Union[List[str], None] = None,
    dict_modvar_specs: Union[Dict[str, float], None] = None,
    field_region: str = "nation",
    magnitude_type: str = "baseline_additive",
    model_energy: Union[me.EnergyConsumption, None] = None,
    modvars_source: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    return_modvars_only: bool = False,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement fuel switch transformations in Transportation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: target magnitude of fuel mixture. See keyword argument
        `magnitude_type` below for more information on how the magnitude is 
        specified
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - baseline_period: string specifying "final" or "initial". If "final", 
        the baseline period used to determine the shift is the final time 
        period. If initial, uses the first time period.
    - categories: TRNS categories to apply transformation to
    - dict_modvar_specs: dictionary of targets modvars to shift into (assumes
        that will take from others). Maps from modvar to fraction of magnitude.
        Sum of values must == 1.
    - field_region: field in df_input that specifies the region
    - magnitude_type: type of magnitude to use. Valid types include
            * "baseline_additive": add the magnitude to the baseline
            * "baseline_scalar": multiply baseline value by magnitude
            * "final_value": magnitude is the final value for the variable to
                take (achieved in accordance with vec_ramp)
            * "transfer_scalar": apply scalar to outbound categories to
                calculate final transfer magnitude. 
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - modvars_source: optional list of fuel fraction model variables to use as a
        source for transfering to target fuel. NOTE: must be specified within 
        this function as a model variable.
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - return_modvars_only: return the model variables that define fuel fractions
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = (
        me.EnergyConsumption(model_attributes) 
        if not isinstance(model_energy, me.EnergyConsumption) 
        else model_energy
    )
    all_regions = sorted(list(set(df_input[field_region])))

    # dertivative vars (alphabetical)
    attr_trns = model_attributes.get_attribute_table("Transportation")
    attr_time_period = model_attributes.get_dimensional_attribute_table(model_attributes.dim_time_period)

    df_out = []
    regions_apply = (
        all_regions 
        if (regions_apply is None) 
        else [x for x in regions_apply if x in all_regions]
    )

    # model variables to explore--should pull these from attributes HEREHERE
    modvars = [
        model_energy.modvar_trns_fuel_fraction_biofuels,
        model_energy.modvar_trns_fuel_fraction_diesel,
        model_energy.modvar_trns_fuel_fraction_electricity,
        model_energy.modvar_trns_fuel_fraction_gasoline,
        model_energy.modvar_trns_fuel_fraction_hydrogen,
        model_energy.modvar_trns_fuel_fraction_kerosene,
        model_energy.modvar_trns_fuel_fraction_natural_gas
    ]

    if return_modvars_only:
        return modvars

    dict_modvar_specs_def = {model_energy.modvar_inen_frac_en_electricity: 1}
    dict_modvar_specs = dict_modvar_specs_def if not isinstance(dict_modvar_specs, dict) else dict_modvar_specs
    dict_modvar_specs = dict((k, v) for k, v in dict_modvar_specs.items() if (k in modvars) and (isinstance(v, int) or isinstance(v, float)))
    dict_modvar_specs = dict_modvar_specs_def if (sum(list(dict_modvar_specs.values())) != 1.0) else dict_modvar_specs

    modvars_source = modvars if (modvars_source is None) else [x for x in modvars_source if x in modvars]
    modvars_source = [x for x in modvars_source if (x not in dict_modvar_specs.keys())] 
    modvars_target = [x for x in modvars if x in dict_modvar_specs.keys()]
    cats_all = [set(model_attributes.get_variable_categories(x)) for x in modvars_target]
    cats_all = set.intersection(*cats_all)
    cats_all = [x for x in cats_all if x in categories]
   
    # set some parameters
    subsec = model_attributes.subsec_name_trns
    magnitude_relative_to_baseline = (
        magnitude_type in [
            "baseline_scalar", 
            "baseline_additive",
            "transfer_scalar"
        ]
    )


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = (
            df_input[
                df_input[field_region] == region
            ]
            .sort_values(by = [model_attributes.dim_time_period])
            .reset_index(drop = True)
        )
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        if region in regions_apply:
            for cat in cats_all:
                
                # initialize model variables that are adjusted
                modvars_adjust = []
                fields_source = []
                fields_target = []

                for modvar in modvars_source + modvars_target:

                    cats_valid = model_attributes.get_variable_categories(modvar)
                    if cat not in cats_valid:
                        continue

                    modvars_adjust.append(modvar) 

                    if (modvar in modvars_target):
                        fields_target.append(
                            model_attributes.build_variable_fields(
                                modvar,
                                restrict_to_category_values = cat,
                            )
                        )
                    
                    if (modvar in modvars_source):
                        fields_source.append(
                            model_attributes.build_variable_fields(
                                modvar,
                                restrict_to_category_values = cat,
                            )
                        )


                # get some baseline values
                tp_baseline = (n_tp - 1) if (baseline_period == "final") else 0
                vec_target_baseline_total = np.array(df_in[fields_target]).astype(float).sum(axis = 1)
                vec_initial_vals = np.array(df_in[fields_target].iloc[tp_baseline]).astype(float)
                vec_initial_distribution = np.nan_to_num(vec_initial_vals/vec_initial_vals.sum(), 1.0, posinf = 1.0)

                # set magnitude
                magnitude_shift = vec_initial_vals.sum()*magnitude if (magnitude_type in ["baseline_scalar"]) else magnitude
                if magnitude_type == "transfer_scalar":

                    magnitude_new = 0.0

                    for modvar in modvars_source:
                        if cat in model_attributes.get_variable_categories(modvar):
                            field_cur = model_attributes.build_variable_fields(
                                modvar,
                                restrict_to_category_values = cat
                            )
                            
                            magnitude_new += float(df_in[field_cur].iloc[tp_baseline])*magnitude

                    magnitude_shift = magnitude_new


                # get the current total value of fractions
                """
                vec_final_vals = np.array(df_in[fields_target].iloc[n_tp - 1]).astype(float)
                val_final_target = sum(vec_final_vals)
                target_shift = float(sf.vec_bounds(magnitude_shift + val_initial_target, (0.0, 1.0)))
                scale_non_elec = np.nan_to_num((1 - target_shift)/(1 - val_final_target), 0.0, posinf = 0.0)
                
                vec_target_baseline = np.array(df_in[fields_target]).astype(float).sum(axis = 1)
                vec_bounds = np.array(df_in[fields_target + fields_source]).astype(float).sum(axis = 1)
                vec_target_shift = float(sf.vec_bounds(magnitude_shift + val_initial_target, (0.0, 1.0)))*vec_ramp
                scale_non_elec = np.nan_to_num((vec_bounds - vec_target_shift)/(vec_bounds - vec_target_baseline), 0.0, posinf = 0.0)

                target_distribution = magnitude_shift*np.array([dict_modvar_specs.get(x) for x in modvars_target]) + val_initial_target*vec_initial_distribution
                target_distribution /= max(magnitude_shift + val_initial_target, 1.0) 
                target_distribution = np.nan_to_num(target_distribution, 0.0, posinf = 0.0)

                dict_target_distribution = dict((x, target_distribution[i]) for i, x in enumerate(modvars_target))
                """
                vec_bounds = np.array(df_in[fields_target + fields_source]).astype(float).sum(axis = 1)

                vec_magnitude_base = (
                    vec_target_baseline_total 
                    if magnitude_relative_to_baseline 
                    else np.zeros(len(vec_target_baseline_total))
                )
                val_initial_target = vec_magnitude_base[tp_baseline]
                vec_target_with_ramp = sf.vec_bounds(magnitude_shift*vec_ramp + vec_magnitude_base, (0.0, 1.0))
                scale_non_elec = np.nan_to_num(
                    (vec_bounds - vec_target_with_ramp)/(vec_bounds - vec_target_baseline_total), 
                    0.0, 
                    posinf = 0.0
                )

                target_distribution = magnitude_shift*np.array([dict_modvar_specs.get(x) for x in modvars_target]) 
                target_distribution += val_initial_target*vec_initial_distribution
                target_distribution /= max(magnitude_shift + val_initial_target, 1.0) 
                target_distribution = np.nan_to_num(target_distribution, 0.0, posinf = 0.0)

                dict_target_distribution = dict((x, target_distribution[i]) for i, x in enumerate(modvars_target))

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    
                    field_cur = model_attributes.build_variable_fields(
                        modvar,
                        restrict_to_category_values = cat,
                    )

                    """
                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (
                        np.nan_to_num(val_final, 0.0, posinf = 0.0)*scale_non_elec 
                        if (modvar not in modvars_target) 
                        else dict_target_distribution.get(modvar)#magnitude_shift*dict_modvar_specs.get(modvar)
                    )
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old
                    """
                    # NOTE: scale_non_elec (defined above) includes the mix that is (1 - vec_ramp) if all modvars are specified
                    vec_old = np.array(df_in[field_cur])
                    vec_new = (
                        np.nan_to_num(vec_old*scale_non_elec, 0.0, posinf = 0.0) 
                        if (modvar not in modvars_target) 
                        else dict_target_distribution.get(modvar)*vec_ramp + (1 - vec_ramp)*vec_old
                    )

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
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



def transformation_trns_electrify_category_to_target_old(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_elec: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.EnergyConsumption, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement the "Electrify light duty road transport" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of final proportion of light duty transport that is
        electrified.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = (
        me.EnergyConsumption(model_attributes) 
        if not isinstance(model_energy, me.EnergyConsumption) 
        else model_energy
    )
    all_regions = sorted(list(set(df_input[field_region])))

    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.get_dimensional_attribute_table(model_attributes.dim_time_period)
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]

    # model variables to explore
    modvars = [
        model_energy.modvar_trns_fuel_fraction_biofuels,
        model_energy.modvar_trns_fuel_fraction_diesel,
        model_energy.modvar_trns_fuel_fraction_electricity,
        model_energy.modvar_trns_fuel_fraction_gasoline,
        model_energy.modvar_trns_fuel_fraction_hydrogen,
        model_energy.modvar_trns_fuel_fraction_kerosene,
        model_energy.modvar_trns_fuel_fraction_natural_gas
    ]


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        # get electric categories and build dictionary of target values
        cats_elec_all = model_attributes.get_variable_categories(model_energy.modvar_trns_fuel_fraction_electricity)
        cats_elec = [x for x in cats_elec_all if x in cats_elec] if isinstance(cats_elec, list) else cats_elec_all
        dict_targets_final_tp = dict((x, magnitude) for x in cats_elec)


        if region in regions_apply:
            for cat in dict_targets_final_tp.keys():

                field_elec = model_attributes.build_variable_fields(
                    model_energy.modvar_trns_fuel_fraction_electricity,
                    restrict_to_category_values = cat,
                )

                target_value = dict_targets_final_tp.get(cat)
                scale_non_elec = 1 - target_value

                val_final_elec = float(df_in[field_elec].iloc[n_tp - 1])

                # get model variables that need to be adjusted
                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_variable_fields(
                        modvar,
                        restrict_to_category_values = cat,
                    )

                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (
                        np.nan_to_num(
                            (val_final/(1 - val_final_elec))*scale_non_elec,
                            0.0, 
                            posinf = 0.0
                        ) 
                        if (field_cur != field_elec) 
                        else target_value
                    )

                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
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



def transformation_trns_increase_energy_efficiency_electric(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as a scalar change to the baseline value--e.g., a
        25% increase relative to the final time period value is entered as
        0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trns_electrical_efficiency: {
                "magnitude": 1 + magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_trns_increase_energy_efficiency_non_electric(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as a scalar change to the baseline value--e.g., a
        25% increase relative to the final time period value is entered as
        0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    dict_base = {
        "magnitude": 1 + magnitude,
        "magnitude_type": "baseline_scalar",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }

    modvars = [
        model_energy.modvar_trns_fuel_efficiency_biofuels,
        model_energy.modvar_trns_fuel_efficiency_diesel,
        model_energy.modvar_trns_fuel_efficiency_gasoline,
        model_energy.modvar_trns_fuel_efficiency_hgl,
        model_energy.modvar_trns_fuel_efficiency_hydrogen,
        model_energy.modvar_trns_fuel_efficiency_kerosene,
        model_energy.modvar_trns_fuel_efficiency_natural_gas
    ]

    dict_run = dict((modvar, dict_base) for modvar in modvars)

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        **kwargs
    )

    return df_out



def transformation_trns_increase_vehicle_occupancy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: List[str] = ["road_light"],
    model_energy: Union[me.EnergyConsumption, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in vehicle occupancy for private light
        vehicles relative to final time period (interpreted as a scalar change
        to the baseline value--e.g., a 25% increase relative to the final time
        period value is entered as 0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional EnergyConsumption object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trns_average_passenger_occupancy: {
                "categories": categories,
                "magnitude": 1 + magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out
