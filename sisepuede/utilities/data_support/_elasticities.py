import numpy as np
import pandas as pd
import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.models.socioeconomic as se
import sisepuede.utilities._toolbox as sf
from typing import *



#########################
#    BEGIN FUNCTIONS    #
#########################


def exogenous_demands_to_sispeuede_ies(
    df_inputs: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    modvar_demand: Union[str, 'ModelVariable'],
    modvar_driver: Union[str, 'ModelVariable'],
    modvar_elasticity: Union[str, 'ModelVariable'],
    modvar_scalar_demand: Union[str, 'ModelVariable'],
    time_period_projection_first: int,
    cat_driver: Union[str, None] = None,
    elasticity_bounds: Union[Tuple[float, float], None] = None,
    elasticity_default: float = 1.0,
    field_iso: Union[str, None] = None,
    fill_missing_se: Union[float, int, None] = None,
    max_dev_from_mean: Union[float, int, None] = None,
    model_socioeconomic: Union[se.Socioeconomic, None] = None,
    stop_on_error: bool = False,
    sup_elast_magnitude: Union[float, int, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build [I]nitial demands (or production), [E]lasticities, [S]calars from 
        exogenous specification of input variables. Formats raw vectors for 
        SISEPUEDE. Returns a three-ple of data frames:
        
        (
            df_demscalar,
            df_elasticities,
            df_prodinit
        )
        
    These data frames are designed for input into SISEPUEDE templates.
    
    NOTE: To avoid projection and only use inputs that are specified (i.e., 
        preserve exogenous production projections), enter the final time period 
        defined in the model_attributes time_period attribute table as 
        time_period_projection_first. This will calculate elasticities 
        associated with each production projection.
        
    
    For example, can be used to calculate elasticities, historical production 
        sorted in IPPU.modvar_ippu_prod_qty_init
    
    Function Arguments
    ------------------
    df_inputs : pd.DataFrame
        DataFrame containing variables modvar_demand and modvar_driver
    model_attributes : ModelAttributes
        ModelAttributes object used for variables/time periods/etc.    
    modvar_demand : Union[str, ModelVariable]
        SISEPUEDE model variable used to store demands (that are elastic)
    modvar_driver : Union[str, ModelVariable]
        SISEPUEDE model variable to which demands (or production) are elastic 
        (e.g., GDP)
    modvar_elasticity : Union[str, ModelVariable]
        SISEPUEDE model variable for elasticities--must be in the same subsector 
        as modvar_demand
    modvar_scalar_demand : Union[str, ModelVariable]
        SISEPUEDE model variable for demand scalars--must be in the same 
        subsector as modvar_demand
    time_period_projection_first : int
        First time period to use as historical (data dependent)
    
    Keyword Arguments
    -----------------
    cat_driver : Union[str, None]
        Optional specification of a cateogory to associate with the driver 
        variable. Only applicable if modvar_driver is associated with categories
    elasticity_bounds : Union[Tuple[float, float], None]
        Bounds to use to prevent extreme average elasticities. If None, no 
        bounds are applied
    elasticity_default : float
        Default elasticity to use if invalid elasticities are found
    field_iso : Union[str, None]
        Field in df_inputs containing ISO codes. If None, defaults to 
        Regions.field_iso (see support_classses.Regions)
    fill_missing_se : Union[float, int, None]
        Optional value to use to fill missing fields required for
        Socioeconomic.project() 
        * NOTE :    Use with caution; if any required variables not present in 
                    df_inputs, this will fill fields used to define those 
                    missing variables with the value in fill_missing_se.
    max_dev_from_mean :  Union[float, int, None]
        Maximum devation from the mean to allow in projections
    model_socioeconomic : Union[Socioeconomic, None]
        Optional socioeconomic model to pass
    stop_on_error : bool
        * True:     Stops if any required columns are missing
        * False:    Attempts to solve for all categories possible
    sup_elast_magnitude :  Union[float, int, None]
        Supremum for the magnitude of an elasticity; prevent wild swings or 
        crashing to 0. If None, no supremum is applied

    """
    
    ##  CHECKS

    # run some checks - check that the driver is not associated with categories
    cats_driver = model_attributes.get_variable_categories(modvar_driver)
    if cats_driver is not None:
        cat_driver = (
            cats_driver[0] 
            if ((cat_driver is None) and (len(cats_driver) == 0)) 
            else cat_driver
        )

        if cat_driver not in cats_driver:
            # WARNINGS/LOGGING
            return None
    else:
        cat_driver = None

    # check if driver requires socioeconomic run
    model_socioeconomic = (
        se.Socioeconomic(model_attributes) 
        if not se.is_sisepuede_model_socioeconomic(model_socioeconomic, )
        else model_socioeconomic
    )

    # get the variables
    modvar_demand = model_attributes.get_variable(modvar_demand, )
    modvar_driver = model_attributes.get_variable(modvar_driver, )
    modvar_elasticity = model_attributes.get_variable(modvar_elasticity, )
    modvar_scalar_demand = model_attributes.get_variable(modvar_scalar_demand, )
   
    # check subsector equivalience
    subsec_demand = model_attributes.get_variable_subsector(modvar_demand)
    subsec_elasticity = model_attributes.get_variable_subsector(modvar_elasticity)
    subsec_scalar = model_attributes.get_variable_subsector(modvar_scalar_demand)
    if len(set([subsec_demand, subsec_elasticity, subsec_scalar])) > 1:
        # WARNINGS/LOGGING
        return None

    attribute = model_attributes.get_attribute_table(subsec_demand, )
    
    # verify some analytical parameters
    elasticity_bounds = None if not isinstance(elasticity_bounds, tuple) else elasticity_bounds
    elasticity_default = (
        1.0 
        if not sf.isnumber(elasticity_default)
        else elasticity_default
    )
    max_dev_from_mean = (
        np.inf 
        if not sf.isnumber(max_dev_from_mean)
        else np.abs(max_dev_from_mean)
    )
    sup_elast_magnitude = (
        np.inf
        if not sf.isnumber(sup_elast_magnitude)
        else sup_elast_magnitude
    )
    
    
    ##  INITIALIZATION
    
    # initialize time period and region information
    time_periods = sc.TimePeriods(model_attributes)
    regions = sc.Regions(model_attributes)
    
    # some basic init
    
    field_iso = regions.field_iso if not isinstance(field_iso, str) else field_iso
    field_year = time_periods.field_year
    
    # initialize year info + add another projected historical year to avoid odd growth behavior
    year_0_project = time_periods.tp_to_year(time_period_projection_first)
    years_historical = [x for x in list(df_inputs[field_year].unique()) if (x < year_0_project)]
    years_historical_with_proj = years_historical + [max(years_historical)]
    year_target = time_periods.year_max
    
    # some data frames for years
    years_full = years_historical.copy()
    years_full += list(range(max(years_historical) + 1, year_target + 1))
    df_years_full_base = pd.DataFrame({field_year: years_full})


    # get some fields
    field_driver = model_attributes.build_variable_fields(
        modvar_driver, 
        restrict_to_category_values = cat_driver,
    )
    field_driver = field_driver[0] if isinstance(field_driver, list) else field_driver

    # get categories that are shared between demand and elast
    cats_shared = get_shared_categories(
        model_attributes, 
        modvar_demand, 
        modvar_elasticity,
        modvar_scalar_demand,
    )
    cats_shared = [x for x in attribute.key_values if x in cats_shared]
    
    # reduce to only those that are available
    cats_dem = []
    for cat in cats_shared:
        field_dem = modvar_demand.build_fields(category_restrictions = cat, )
        field_scalar = modvar_scalar_demand.build_fields(category_restrictions = cat, )
        
        # verify they're all present
        if not set([field_dem, field_scalar]).issubset(set(df_inputs.columns)):
            if not stop_on_error: continue
            raise RuntimeError(f"One or more fields for category '{cat}' were not found in the input DataFrame.")

        # append to list of categories
        cats_dem.append(cat)

    
    # nothing happens if no categories were found
    if len(cats_dem) == 0:
        return None

    # otherwise, set fields
    fields_dem = modvar_demand.build_fields(category_restrictions = cats_dem)
    fields_elast = modvar_elasticity.build_fields(category_restrictions = cats_dem, )
    fields_scalars = modvar_scalar_demand.build_fields(category_restrictions = cats_dem, )

    
    ##  PREPARE TO BUILD
    
    df_elasticities = []
    df_prodinit = []
    df_demscalar = []
    
    df_inputs_by_iso = (
        df_inputs[
            df_inputs[field_iso].isin(regions.all_isos)
        ]
        .reset_index(drop = True)
        .groupby([field_iso])
    )

    global dfi
    dfi = df_inputs
    
    # maximum magntiude of elasticity to include in means

    for i, df in df_inputs_by_iso:
        
        i = i[0] if isinstance(i, tuple) else i

        # set a data frame for the driver; can be pulled from socioeconomic
        if modvar_driver in model_socioeconomic.output_model_variables:
            
            if fill_missing_se is not None:
                # check if fields need to be filled
                fields_fill = [x for x in model_socioeconomic.required_variables if (x not in df_inputs.columns)]
                fields_dim_fill = [x for x in fields_fill if x in model_socioeconomic.required_dimensions]
                fill_missing_se_doa = int(np.round(fill_missing_se))

                df[fields_fill] = fill_missing_se
                df[fields_dim_fill] = fill_missing_se_doa

            df_se = model_socioeconomic.project(
                df.reset_index(drop = True), 
                ignore_time_periods = True,
                project_for_internal = False
            )

            fields_add = [x for x in df_se.columns if x not in df.columns]
            for fld in fields_add:
                df[fld] = np.array(df_se[fld])

            df.dropna(inplace = True, subset = [field_driver] + fields_dem)

        if len(df) == 0:
            continue


        # get historical, then add dummy year
        df_hist = (
            df[
                df[field_year].isin(years_historical)
            ]
            .sort_values(by = [field_year])
            .reset_index(drop = True)
        )

        df_years = pd.DataFrame({field_year: years_historical})
        df_hist = pd.merge(df_years, df_hist, how = "left")
        df_hist.interpolate(inplace = True)
        df_hist.dropna(inplace = True, how = "any")

        # get production array and and project first non-historical time period from regression
        arr_dem = np.array(df_hist[fields_dem])
        
        global arr_dem_out
        arr_dem_out = arr_dem

        arr_dem_proj_first_post_historical = sf.project_from_array(
            arr_dem, 
            max_deviation_from_mean = max_dev_from_mean
        )

        # append projected (first time period post-historical) to df_hist
        df_hist_append = {
            field_year: [max(years_historical) + 1],
            field_iso: [i]
        }
        df_hist_append.update(
            dict(
                (fields_dem[i], [arr_dem_proj_first_post_historical[i]]) 
                for i in range(len(fields_dem))
            )
        )
        df_hist = (
            pd.concat(
                [
                    df_hist, 
                    pd.DataFrame(df_hist_append)
                ],
                axis = 0
            )
            .reset_index(drop = True)
        )

        # overwrite driver since the final value is missing
        df_hist = sf.match_df_to_target_df(
            df_hist.drop(columns = [field_driver], ), #axis = 1),
            df[[field_year, field_iso, field_driver]],
            fields_index = [field_year, field_iso],
            overwrite_only = False
        )

        # get change in driver
        vec_driver = np.array(df_hist[field_driver])
        vec_driver_change = vec_driver[1:]/vec_driver[0:-1] - 1

        # 1. update arr_dem, change in production, and esl -
        # 2. get indices of zeros (used in demscalar)
        # 3. convert 0s to small numbers to allow elast to function
        arr_dem = np.array(df_hist[fields_dem])
        w_dem_zero = np.where(arr_dem == 0.0)
        arr_dem = sf.zeros_to_small(arr_dem, axis = 0)
        arr_dem_change = arr_dem[1:]/arr_dem[0:-1] - 1

        # get elasticity of production to driver
        arr_elast = sf.do_array_mult(arr_dem_change, 1/vec_driver_change)
        arr_elast = np.nan_to_num(arr_elast, nan = 1.0, posinf = 1.0, neginf = -1.0)

        df_elast = pd.DataFrame(arr_elast, columns = fields_elast)
        df_hist = pd.concat(
            [
                df_hist.drop(columns = [x for x in fields_elast if x in df_hist.columns], ), 
                df_elast
            ], 
            axis = 1,
        )
        df_hist = df_hist[df_hist[field_year].isin(years_historical)].reset_index(drop = True)

        vec_means = np.zeros(arr_elast.shape[1])
        vec_medians = np.zeros(arr_elast.shape[1])
        vec_targs = np.zeros(arr_elast.shape[1])
        
        # set targets for final time period based on mean/median
        for j in range(len(vec_means)):

            # drop extreme elasticities and get mean/median--also, drop elasticity associated with the projection
            vec_cur = arr_elast[0:-1, j]
            vec_cur = vec_cur[np.where((vec_cur > min(vec_cur)) & (vec_cur < max(vec_cur)))]
            vec_cur = vec_cur[np.where(np.abs(vec_cur) <= sup_elast_magnitude)]
            mu = np.mean(vec_cur) if (len(vec_cur) > 0) else 1.0
            med = np.median(vec_cur) if (len(vec_cur) > 0) else 1.0

            # bound negative mean and median elasticities at low number
            vec_means[j] = float(sf.vec_bounds(mu, elasticity_bounds)) if not np.isnan(mu) else 1.0
            vec_medians[j] = float(sf.vec_bounds(med, elasticity_bounds)) if not np.isnan(med) else 1.0

            m0 = min(vec_cur) if (len(vec_cur) > 0) else 1.0
            m1 = max(vec_cur) if (len(vec_cur) > 0) else 1.0

            if m0 > 0:
                vec_targs[j] = 1.0 if (mu > 1) else mu
            elif m1 < 0:
                vec_targs[j] = -1.0 if (mu < -1) else mu
            else:
                vec_targs[j] = elasticity_default



        ##  BUILD DATA FRAME FOR FUTURE YEARS
        
        df_years_full = df_years_full_base.copy()
        
        df_full = {
            field_year: [max(list(df_hist[field_year])) + 1, year_target], 
            field_iso: [i, i]
        }
        df_full.update(
            dict(
                (fields_elast[j], [vec_medians[j], vec_targs[j]]) 
                for j in range(len(fields_elast))
            )
        )
        
        global DFH
        global DFF
        DFH = df_hist.copy()
        DFF = pd.DataFrame(df_full)

        df_full_check = df_full.copy()
        df_full = (
            pd.concat([df_hist, pd.DataFrame(df_full)], axis = 0)
            .dropna(subset = fields_elast)
            .drop(columns = [field_iso, field_driver] + fields_dem, )
        )

        df_full = (
            pd.merge(df_years_full, df_full, how = "left", on = [field_year])
            .interpolate()
            .sort_values(by = [field_year])
            .reset_index(drop = True)
        )
        df_full[field_iso] = i
        df_full = df_full[[field_year, field_iso] + fields_elast]
        
        
        ##  BUILD INITIAL PRODUCTION

        df_years_full[field_iso] = i
        """
        df_prod_0 = {field_iso: [i]}
        df_prod_0.update(
            dict(
                (field, [arr_dem[0, i]]) 
                for i, field in enumerate(fields_dem)
            )
        )
        """;
        df_prod_0 = (
            df[
                df[field_year].isin(list(years_historical) + [max(years_historical) + 1])
            ][[field_iso, field_year] + fields_dem]
            .sort_values(by = [field_year])
            .reset_index(drop = True)
        )
        df_prod_0 = (
            pd.merge(
                df_years_full, 
                df_prod_0,
                #pd.DataFrame(df_prod_0),
                how = "left"
            )
            .interpolate(method = "ffill")
        )
        df_prod_0 = df_prod_0[[field_year, field_iso] + fields_dem]

        
        ##  BUILD DEMAND SCALARS
        
        # call w_dem_zero from above, use to set demand scalars
        arr_scalars = np.ones(arr_dem.shape)
        arr_scalars[w_dem_zero] = 0.0

        shape_append = len(df_prod_0) - arr_dem.shape[0]
        shape_append = (shape_append, arr_scalars.shape[1])
        
        arr_scalars = (
            np.concatenate([arr_scalars, np.ones(shape_append)], axis = 0)
            if shape_append[0] > 0
            else arr_scalars
        )

        df_scalars = pd.DataFrame(
            arr_scalars,
            columns = fields_scalars
        )

        # add identifiers
        df_scalars = pd.concat(
            [
                df_prod_0[[field_year, field_iso]].reset_index(drop = True),
                df_scalars
            ],
            axis = 1
        )
        df_scalars = df_scalars[[field_year, field_iso] + fields_scalars]

        df_demscalar.append(df_scalars)
        df_elasticities.append(df_full)
        df_prodinit.append(df_prod_0)


    df_demscalar = pd.concat(df_demscalar, axis = 0).reset_index(drop = True)
    df_elasticities = pd.concat(df_elasticities, axis = 0).reset_index(drop = True)
    df_prodinit = pd.concat(df_prodinit, axis = 0).reset_index(drop = True)
    
    return df_demscalar, df_elasticities, df_prodinit



def get_shared_categories(
    model_attributes: 'ModelAttributes',
    *modvars,
    force_dict: bool = False,
    **kwargs,
) -> List[str]:
    """Get shared categories from model variables. Set force_dict to True to 
        force a dictionary if there's only one dimension; otherwise, returns a 
    """
    
    dict_keys = {}

    for modvar in modvars:

        # try retrieving
        modvar = model_attributes.get_variable(modvar)
        if modvar is None:
            continue

            
        dict_cur = modvar.dict_category_keys
        for k, v in dict_cur.items():

            # if in the dictionary, update the set
            if k in dict_keys.keys():
                s = dict_keys.get(k) & set(v)
                dict_keys.update({k: s, })
                continue
            
            # otherwise, init
            dict_keys.update({k: set(v), })
    
    if (len(dict_keys) == 1) and not force_dict:
        k = list(dict_keys.keys())[0]
        dict_keys = dict_keys.get(k)
    
    return dict_keys