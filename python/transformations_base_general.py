import model_attributes as ma
import model_afolu as mafl
import model_ippu as mi
import model_circular_economy as mc
import model_electricity as ml
import model_energy as me
import model_socioeconomic as se
import numpy as np
import pandas as pd
import support_classes as sc
import support_functions as sf
from typing import *






###############################
###                         ###
###    GENERIC FUNCTIONS    ###
###                         ###
###############################

def get_time_period(
    model_attributes: ma.ModelAttributes,
    return_type: str = "max",
) -> int:
    """
    Get max or min time period using model_attributes. Set return_type = "max"
        for the maximum time period or return_type = "min" for the minimum time
        period.
    """
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    return_val = min(attr_time_period.key_values) if (return_type == "min") else max(attr_time_period.key_values)

    return return_val



def prepare_demand_scalars(
    df_input: pd.DataFrame,
    modvars: Union[List[str], str, None],
    model_attributes: ma.ModelAttributes,
    key_region: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Setup so that all input demand scalars are normalized to have value 1 during 
        the initial time period

    Function Arguments
    ------------------
    - df_input: input DataFrame to prepare
    - modvars: list of model variables (or single model variable) to apply the 
        modification to
    - model_attributes: model attributes object used for accessing fields, time
        periods, and region in formation

    Keyword Arguments
    -----------------
    - key_region: optional specification of key region. If None, defaults to
        model_attributes.dim_region
    """
    # check model variable specification
    modvars = (
        [modvars]
        if isinstance(modvars, str)
        else (
            list(modvars)
            if sf.islistlike(modvars)
            else None
        )
    )

    if modvars is None:
        return df_input


    # group by region
    key_region = (
        model_attributes.dim_region
        if key_region is None
        else key_region
    )
    df_out = (
        [df for x, df in df_input.groupby([key_region])]
        if key_region in df_input.columns
        else None
    )
    if df_out is None:
        return df_input

    #  initialize time periods
    time_periods = sc.TimePeriods(model_attributes)
    tp_min = min(time_periods.all_time_periods)


    # loop over each region to normalize to 1 at first time period
    for i, df in enumerate(df_out):
        
        row = df[
            df[time_periods.field_time_period].isin([tp_min]) 
        ]

        for modvar in modvars:
            
            fields = model_attributes.build_varlist(None, modvar)

            vec_base = np.array(row[fields].iloc[0]).astype(float)
            df[fields] = np.nan_to_num(
                np.array(df[fields])/vec_base, 
                1.0
            )

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

    return df_out



def transformation_general(
    df_input: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    dict_modvar_specs: Dict[str, Dict[str, str]],
    field_region: str = "nation",
    regions_apply: Union[List[str], None] = None,
    strategy_id: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Generalized function to implement some common transformations. Many other
        transformation functions are wrappers for this function.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_modvar_specs: dictionary mapping model variable to some
        characteristics:

        REQUIRED KEYS
        -------------
        * "magnitude": magnitude of change to apply by final
        * "magnitude_type": type of magnitude to use. Valid types include
            * "baseline_additive": add the magnitude to the baseline
            * "baseline_scalar": multiply baseline value by magnitude
            * "baseline_scalar_diff_reduction": reduce the difference between
                the value in the baseline time period and the upper bound (NOTE:
                requires specification of bounds to work) by magnitude
            * "final_value": magnitude is the final value for the variable to
                take (achieved in accordance with vec_ramp)
            * "final_value_ceiling": magnitude is the lesser of (a) the existing 
                final value for the variable to take (achieved in accordance 
                with vec_ramp) or (b) the existing specified final value,
                whichever is smaller
            * "final_value_floor": magnitude is the greater of (a) the existing 
                final value for the variable to take (achieved in accordance 
                with vec_ramp) or (b) the existing specified final value,
                whichever is greater
            * "transfer_value": transfer value from categories to other
                categories. Must specify "categories_source" &
                "categories_target" in dict_modvar_specs. See description below
                in OPTIONAL for information on specifying this.
            * "transfer_scalar_value": transfer value from categories to other
                categories based on a scalar. Must specify "categories_source" &
                "categories_target" in dict_modvar_specs. See description below
                in OPTIONAL for information on specifying this.
            * "transfer_value_to_acheieve_magnitude": transfer value from
                categories to other categories to acheive a target magnitude.
                Must specify "categories_source" & "categories_target" in
                dict_modvar_specs. See description below in OPTIONAL for
                information on specifying this.
            * "vector_specification": simply enter a vector to use for region
        * "vec_ramp": implementation ramp vector to use for the variable

        OPTIONAL
        --------
        * "bounds": optional specification of bounds to use on final change
        * "categories": optional category restrictions to use
        * "categories_source" & "categories_target": must be specified together
            and only valid with the "transfer_value" or
            "transfer_value_to_acheieve_magnitude" magnitude_types. Transfers
            some quantity from categories specified within "categories_source"
            to categories "categories_target". "categories_target" is a
            dictionary of target categories mapping to proportions of the
            magnitude to receive.

            For example,

                {
                    "magnitude" = 0.8,
                    "categories_source" = ["cat_1", "cat_2", "cat_3"],
                    "categories_target" = {"cat_4": 0.7, "cat_5": 0.3}
                }

            will distribute 0.8 from categories 1, 2, and 3 to 4 and 5, giving
            0.56 to cat_4 and 0.24 to cat_5. In general, the source distribution
            is proportional to the source categories' implied pmf at the final
            time period.

        * "time_period_baseline": time period to use as baseline for change if
            magnitude_type in ["baseline_additive", "baseline_scalar"]

        EXAMPLE
        -------
        * The dictionary should take the following form:

        {
            modvar_0: {
                "magnitude": 0.5,
                "magnitude_type": "final_value",
                "vec_ramp": np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0]),
                "bounds": (0, 1),    # optional
                "time_period_change": 0    # optional
            },
            modvar_1: ...
        }

    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - field_region: field in df_input that specifies the region
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    ##  INITIAlIZATION

    # core vars (ordered)
    all_regions = sorted(list(set(df_input[field_region])))
 
    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]

    # valid specifications of magnitude type
    magnitude_types_valid = [
        "baseline_additive",
        "baseline_scalar",
        "baseline_scalar_diff_reduction",
        "final_value",
        "final_value_ceiling",
        "final_value_floor",
        "transfer_value",
        "transfer_value_scalar",
        "transfer_value_to_acheieve_magnitude",
        "vector_specification"
    ]


    ##  CHECK SPECIFICATION DICTIONARY

    modvars = sorted([x for x in dict_modvar_specs.keys() if x in model_attributes.all_variables])
    dict_modvar_specs_clean = {}

    for modvar in modvars:
        # default verified to true; set to false if any condition is not met
        verified_modvar = True
        dict_modvar_specs_cur = dict_modvar_specs.get(modvar)
        
        # check magnitude type
        magnitude_type = dict_modvar_specs_cur.get("magnitude_type")
        verified_modvar &= (magnitude_type in magnitude_types_valid)
        
        # check magnitude
        magnitude = dict_modvar_specs_cur.get("magnitude")
        verified_modvar = (
            sf.isnumber(magnitude)
            if magnitude_type not in ["vector_specification"]
            else (
                all([sf.isnumber(x) for x in magnitude])
                if isinstance(magnitude, list) or isinstance(magnitude, np.ndarray)
                else False
            )
        ) & verified_modvar
        
        # check ramp vector
        vec_ramp = dict_modvar_specs_cur.get("vec_ramp")
        vec_ramp = np.array(vec_ramp) if isinstance(vec_ramp, list) else vec_ramp
        verified_modvar &= isinstance(vec_ramp, np.ndarray)
        
        # check for bounds
        bounds = dict_modvar_specs_cur.get("bounds")
        bounds = None if not (isinstance(bounds, tuple) and len(bounds) == 2) else bounds
        verified_modvar = ((bounds is not None) & verified_modvar) if (magnitude_type == "baseline_scalar_diff_reduction") else verified_modvar
        
        # check for categories
        categories = dict_modvar_specs_cur.get("categories")
        categories = None if not isinstance(categories, list) else categories

        # check for source/target categories
        categories_source = dict_modvar_specs_cur.get("categories_source")
        categories_source = None if not isinstance(categories_source, list) else categories_source
        categories_target = dict_modvar_specs_cur.get("categories_target")
        categories_target = (
            dict((k, v) for k, v in categories_target.items() if sf.isnumber(v)) 
            if isinstance(categories_target, dict) 
            else None
        )
        # check special case
        verified_modvar = (
            ((categories_source is not None) & verified_modvar) 
            if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) 
            else verified_modvar
        )
        verified_modvar = (
            ((categories_target is not None) & verified_modvar) 
            if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) 
            else verified_modvar
        )

        # check for time period as baseline
        tp_baseline = dict_modvar_specs_cur.get("time_period_baseline")
        tp_baseline = max(attr_time_period.key_values) if (tp_baseline not in attr_time_period.key_values) else tp_baseline

        

        ## IF VERIFIED, ADD TO CLEANED DICTIONARY

        if verified_modvar:

            subsector = model_attributes.dict_model_variable_to_subsector.get(modvar)

            # check categories against subsector
            if (categories is not None) or (categories_source is not None) or (categories_target is not None):
                pycat = model_attributes.get_subsector_attribute(subsector, "pycategory_primary")
                attr = model_attributes.dict_attributes.get(pycat)

                if (categories is not None):
                    categories = [x for x in categories if x in attr.key_values]
                    categories = None if (len(categories) == 0) else categories

                if (categories_source is not None):
                    categories_source = [x for x in categories_source if x in attr.key_values]
                    categories_source = None if (len(categories_source) == 0) else categories_source

                if (categories_target is not None):
                    categories_target = dict((k, v) for k, v in categories_target.items() if k in attr.key_values)
                    categories_target = None if (sum(list(categories_target.values())) != 1.0) else categories_target

            vector_targets_ordered = [x for x in attr.key_values if x in categories_target.keys()] if isinstance(categories_target, dict) else None


            dict_modvar_specs_clean.update({
                modvar: {
                    "bounds": bounds,
                    "categories": categories,
                    "categories_source": categories_source,
                    "categories_target": categories_target,
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
                    "subsector": subsector,
                    "tp_baseline": tp_baseline,
                    "vec_ramp": vec_ramp,
                    "vector_targets_ordered": vector_targets_ordered
                }
            })

    modvars = sorted(list(dict_modvar_specs_clean.keys()))

    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION
    
    for region in all_regions:
        df_in = df_input[df_input[ field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        if region in regions_apply:
            
            for modvar in modvars:

                dict_cur = dict_modvar_specs_clean.get(modvar)

                # get components
                bounds = dict_cur.get("bounds")
                categories = dict_cur.get("categories")
                categories_source = dict_cur.get("categories_source")
                categories_target = dict_cur.get("categories_target")
                magnitude = dict_cur.get("magnitude")
                magnitude_type = dict_cur.get("magnitude_type")
                mix_to_transform = (magnitude_type not in ["vector_specification"])
                tp_baseline = dict_cur.get("tp_baseline")
                vec_ramp = dict_cur.get("vec_ramp")
                vector_targets_ordered = dict_cur.get("vector_targets_ordered")
                ind_tp_baseline = (
                    vec_tp.index(tp_baseline) 
                    if (
                        magnitude_type in [
                            "baseline_scalar", 
                            "baseline_additive",
                            "baseline_scalar_diff_reduction",
                            "transfer_value_scalar"
                        ]
                    ) 
                    else None
                )

                # set fields
                fields_adjust = model_attributes.build_varlist(
                    dict_cur.get("subsector"),
                    modvar,
                    restrict_to_category_values = categories
                )
                fields_adjust_source = None
                fields_adjust_target = None

                if (categories_source is not None) and (categories_target is not None):

                    fields_adjust = None

                    fields_adjust_source = model_attributes.build_varlist(
                        dict_cur.get("subsector"),
                        modvar,
                        restrict_to_category_values = categories_source
                    )

                    fields_adjust_target = model_attributes.build_varlist(
                        dict_cur.get("subsector"),
                        modvar,
                        restrict_to_category_values = sorted(list(categories_target.keys()))
                    )


                ##  DO MIXING

                if magnitude_type in ["transfer_value", "transfer_value_scalar", "transfer_value_to_acheieve_magnitude"]:

                    # TRANSFER OF MAGNITUDE BETWEEN CATEGORIES

                    # get baseline values
                    arr_base_source = np.array(df_in_new[fields_adjust_source])
                    arr_base_target = np.array(df_in_new[fields_adjust_target])
                    sum_preservation = np.sum(np.array(df_in_new[fields_adjust_source + fields_adjust_target]), axis = 1)
                    
                    # modify magnitude if set as scalar or if it is set as a final value threshold
                    magnitude = (
                        sum(magnitude * arr_base_source[ind_tp_baseline, :])
                        if (magnitude_type == "transfer_value_scalar")
                        else magnitude
                    )
                    
                    # get value of target in baseline and magnitude to transfer
                    vec_target_initial = arr_base_target[tp_baseline, :]
                    total_target_initial = sum(vec_target_initial) if (magnitude_type == "transfer_value_to_acheieve_magnitude") else 0
                    magnitude_transfer = magnitude - total_target_initial

                    # get distribution to transfer--check that it does not violate bounds if specified
                    vec_source_initial = arr_base_source[tp_baseline, :]
                    vec_distribution_transfer = np.nan_to_num(vec_source_initial/sum(vec_source_initial), 0.0)
                    vec_transfer = magnitude_transfer*vec_distribution_transfer

                    vec_source_new = sf.vec_bounds(vec_source_initial - vec_transfer, bounds)
                    vec_transfer = (vec_source_initial - vec_source_new) if (max(np.abs(vec_source_new - vec_transfer)) > 0) else vec_transfer
                    magnitude_transfer = sum(vec_transfer)

                    # new target vector - note that these are both ordered properly according to category
                    vec_target = magnitude_transfer*np.array([categories_target.get(x) for x in vector_targets_ordered])
                    arr_new_source = np.outer(
                        np.ones(arr_base_source.shape[0]),
                        vec_source_new
                    )
                    arr_new_target = arr_base_target + np.outer(
                        np.ones(arr_base_target.shape[0]),
                        vec_target
                    )

                    arr_base = np.concatenate([arr_base_source, arr_base_target], axis = 1)
                    arr_final = np.concatenate([arr_new_source, arr_new_target], axis = 1)
                    fields_adjust = fields_adjust_source + fields_adjust_target

                elif magnitude_type in ["vector_specification"]:
                    
                    # CASE WHERE VECTOR IS EXPLICITLY SPECIFIED - specificy everything as float

                    arr_final = np.array(df_in_new[fields_adjust]).astype(float)
                    for j, field in enumerate(fields_adjust):
                        arr_final[:, j] = np.array(magnitude).astype(float)  

                else:

                    # CASE WITH MOST STANDARD MODIFICATIONS
                    
                    arr_base = np.array(df_in_new[fields_adjust])
                    arr_base = sf.vec_bounds(arr_base, bounds) if (bounds is not None) else arr_base

                    # the final value depends on the magnitude type
                    if magnitude_type == "baseline_scalar":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            magnitude * arr_base[ind_tp_baseline, :]
                        )

                    elif magnitude_type == "baseline_scalar_diff_reduction":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            arr_base[ind_tp_baseline, :] + magnitude * (bounds[1] - arr_base[ind_tp_baseline, :])
                        )

                    elif magnitude_type == "baseline_additive":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            arr_base[ind_tp_baseline, :]
                        ) + magnitude

                    elif magnitude_type == "final_value":
                        arr_final = magnitude * np.ones(arr_base.shape)

                    elif magnitude_type == "final_value_ceiling":
                        arr_final = sf.vec_bounds(
                            arr_base,
                            (0, magnitude)
                        )
                    
                    elif magnitude_type == "final_value_floor":
                        arr_final = sf.vec_bounds(
                            arr_base,
                            (magnitude, np.inf)
                        )

                # check if bounds need to be applied
                arr_final = sf.vec_bounds(arr_final, bounds) if (bounds is not None) else arr_final
                
                if mix_to_transform:
                    vec_ramp = sf.vec_bounds(vec_ramp, (0, 1))
                    arr_transform = sf.do_array_mult(arr_base, 1 - vec_ramp)
                    arr_transform += sf.do_array_mult(arr_final, vec_ramp)
                else:
                    arr_transform = arr_final

                # update dataframe if needed
                for fld in enumerate(fields_adjust):
                    i, fld = fld
                    df_in_new[fld] = arr_transform[:, i]

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



def transformation_general_shift_fractions_from_modvars(
    df_input: pd.DataFrame,
    magnitude: float,
    modvars: List[str],
    dict_modvar_specs: Dict[str, float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    field_region: str = "nation",
    magnitude_relative_to_baseline: bool = False,
    preserve_modvar_domain_sum: bool = True,
    regions_apply: Union[List[str], None] = None,
    strategy_id: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Implement fractional swap transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: target magnitude of mixture (summed across target categories)
    - modvars: list of model variables used to constitute fractions
    - dict_modvar_specs: dictionary of targets modvars to shift into (assumes
        that will take from others). Maps from modvar to fraction of magnitude.
        Sum of values must == 1.
    - vec_ramp: ramp vec used for implementation
    - model_attributes: ModelAttributes object used to call strategies/variables

    Keyword Arguments
    -----------------
    - categories: categories to apply transformation to
    - field_region: field in df_input that specifies the region
    - magnitude_relative_to_baseline: apply the magnitude relative to baseline?
    - preserve_modvar_domain_sum: preserve sum of modvars observed in data? If 
        false, caps at one
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # check modvars
    subsec = set([model_attributes.get_variable_subsector(x) for x in modvars])
    if len(subsec) > 1:
        return df_input

    subsec = list(subsec)[0]

    # dertivative vars (alphabetical)
    all_regions = sorted(list(set(df_input[field_region])))
    attr_subsec = model_attributes.get_attribute_table(subsec)
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]

    # check the modvar specification dictionary (for targets)
    dict_modvar_specs = dict(
        (k, v) for k, v in dict_modvar_specs.items() 
        if (k in modvars) 
        and (isinstance(v, int) or isinstance(v, float))
    )

    # return the original DataFrame if the allocation among target variables is incorrect
    if (sum(list(dict_modvar_specs.values())) != 1.0):
        return df_input

    # get model variables and filter categories
    modvars_source = [x for x in modvars if x not in dict_modvar_specs.keys()]
    modvars_target = [x for x in modvars if x in dict_modvar_specs.keys()]
    
    
    ##  CATEGORIES CHECK
    ##    - categories can either be all none or all not-none

    categories = (
        attr_subsec.key_values
        if not sf.islistlike(categories)
        else [x for x in attr_subsec.key_values if x in categories]
    )

    cats_all_init = []
    any_none = False
    all_none = True
    for x in modvars_target:
        cats = model_attributes.get_variable_categories(x)
        all_none = (cats is None) & all_none
        any_none = (cats is None) | any_none
        (
            cats_all_init.append(set(cats)) 
            if cats is not None
            else None
        )

    if (any_none & (not all_none)) | (categories is None):
        # LOGGING
        return df_input

    # set up all categories
    cats_all = [None]
    if not all_none:
        cats_all = set.intersection(*cats_all_init)
        cats_all = (
            set(cats_all) & set(categories)
            if len(categories) > 0
            else None
        )
    
    else:
        # check sources
        all_none_source = True
        for x in modvars_source:
            cats = model_attributes.get_variable_categories(x)
            all_none_source = (cats is None) & all_none_source
        
        if not all_none_source:
            # LOGGING
            return df_input


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    df_out = []
    df_in_grouped = df_input.groupby([field_region])

    for region, df_in in df_in_grouped:

        # return baseline df if not in applicable regions
        if region not in regions_apply:
            df_out.append(df_in)
            continue

        # prep dataframe
        df_in = (
            df_in
            .sort_values(by = [model_attributes.dim_time_period])
            .reset_index(drop = True)
        )
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        for cat in cats_all:

            restriction = [cat] if (cat is not None) else None
            fields = [
                model_attributes.build_varlist(
                    None,
                    x,
                    restrict_to_category_values = restriction
                )[0] for x in modvars_target
            ]
            fields_source = [
                model_attributes.build_varlist(
                    None,
                    x,
                    restrict_to_category_values = restriction
                )[0] for x in modvars_source
            ]
            
            # values at first time period, initial total of target columns, and associated pmf
            vec_initial_vals = np.array(df_in[fields].iloc[0]).astype(float)
            val_initial_target = vec_initial_vals.sum() if magnitude_relative_to_baseline else 0.0
            vec_initial_distribution = np.nan_to_num(vec_initial_vals/vec_initial_vals.sum(), 1.0, posinf = 1.0)

            # get the current total value of fractions
            vec_final_vals = np.array(df_in[fields].iloc[n_tp - 1]).astype(float)
            val_final_target = sum(vec_final_vals)
            val_final_domain = np.array(df_in[fields + fields_source].iloc[n_tp - 1]).sum()
            target_supremum = min(val_final_domain, 1.0) if preserve_modvar_domain_sum else 1.0

            target_value = float(sf.vec_bounds(magnitude + val_initial_target, (0.0, target_supremum)))#*dict_modvar_specs.get(modvar_target)
            magnitude_adj = target_value - val_initial_target
            scale_non_elec = np.nan_to_num((target_supremum - target_value)/(target_supremum - val_final_target), 0.0, posinf = 0.0)
            # 
            target_distribution = magnitude_adj*np.array([dict_modvar_specs.get(x) for x in modvars_target]) + val_initial_target*vec_initial_distribution
            target_distribution /= max(magnitude_adj + val_initial_target, 1.0) 
            target_distribution = np.nan_to_num(target_distribution, 0.0, posinf = 0.0)

            dict_target_distribution = dict((x, target_distribution[i]) for i, x in enumerate(modvars_target))

            # update source modvars to adjust if any are legitimate
            modvars_adjust = []
            for modvar in modvars:
                
                cats = model_attributes.get_variable_categories(modvar)
                append_q = (
                    cat in cats
                    if not all_none
                    else (cats is None)
                )
                modvars_adjust.append(modvar) if append_q else None
                
            # loop over adjustment variables to build new trajectories
            for modvar in modvars_adjust:
                
                restriction = [cat] if (cat is not None) else None
                field_cur = model_attributes.build_varlist(
                    subsec,
                    modvar,
                    restrict_to_category_values = restriction,
                )[0]

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



def transformation_general_with_magnitude_differential_by_cat(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    modvar: str,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    bounds: Union[Tuple, None] = None,
    categories: Union[List[str], None] = None,
    magnitude_type: str = "baseline_scalar",
    **kwargs
) -> pd.DataFrame:
    """
    Implement a transformation_general transformation with optional difference
        in specification of magnitude, either as a number or a dictionary of
        categories that are mapped to magnitudes.


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying magnitude  OR  dictionary mapping individual 
        categories to magnitude (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - modvar: model variable that is adjusted
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - bounds: optional bounds to set on the magnitude (uniformly applied across
        categories)
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - magnitude_type: see ?transformation_general for more information. Default 
        is "baseline_scalar"
    - **kwargs: passed to transformation_general
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    subsec = model_attributes.get_variable_subsector(modvar, throw_error_q = False)
    if subsec is None:
        return None
    attr = model_attributes.get_attribute_table(subsec)

    # call to general transformation differs based on whether magnitude is a number or a dictionary
    if sf.isnumber(magnitude):
        
        magnitude = float(sf.vec_bounds(magnitude, bounds))

        # check category specification
        categories = model_attributes.get_valid_categories(categories, subsec)
        if categories is None:
            # LOGGING
            return df_input
            
        # apply same magnitude to all categories
        df_out = transformation_general(
            df_input,
            model_attributes,
            {
                modvar: {
                    "bounds": bounds,
                    "categories": categories,
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
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
            
            # check categories
            cats = [cats] if (not isinstance(cats, list)) else cats
            cats = model_attributes.get_valid_categories(cats, subsec) 
            if cats is None:
                continue

            mag = float(sf.vec_bounds(mag, bounds))

            # call general transformation
            df_out = transformation_general(
                df_out,
                model_attributes,
                {
                    modvar: {
                        "bounds": bounds,
                        "categories": cats,
                        "magnitude": mag,
                        "magnitude_type": magnitude_type,
                        "vec_ramp": vec_ramp
                    }
                },
                **kwargs
            )
    

    return df_out

