# 
#  COMMAND LINE RUNTIME FOR SISEPUEDE
#
# NOTE NOTE: SHOULD START WITH JULIA_NUM_THREADS=XX TO SET THREADS ACCESSIBLE TO JULIA WHEN RUNNING NemoMod
#  
#
import argparse
import os, os.path
import pathlib
from typing import *
import warnings

from sisepuede.core.model_attributes import ModelAttributes
import sisepuede.core.support_classes as sc
import sisepuede.manager.sisepuede_examples as sxl
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.manager.sisepuede as ssp
import sisepuede.transformers as trf
import sisepuede.utilities._toolbox as sf




############################
#    SOME ERROR CLASSES    #
############################

class InvalidStrategyDirectory(Exception):
    pass



###################################
#    SOME SUPPORTING FUNCTIONS    #
###################################

def get_dimensional_dict(
    args: Dict,
    model_attributes: ModelAttributes,
    cl_key_design: str = "keys_design",
    cl_key_future: str = "keys_future",
    cl_key_primary: str = "keys_primary",
    cl_key_strategy: str = "keys_strategy",
) -> Union[Dict, None]:
    """
    From args, get dimensional keys. Use model_attributes to assign.
    """
    
    if not isinstance(args, dict):
        return None
    

    # initialize dictionary mapping command-line key to sisepuede key
    dict_key_map = {
        cl_key_design: model_attributes.dim_design_id,
        cl_key_future: model_attributes.dim_future_id,
        cl_key_primary: model_attributes.dim_primary_id,
        cl_key_strategy: model_attributes.dim_strategy_id
    }

    # initialize dictionary
    dict_out = {}

    if args.get(cl_key_primary) is not None:
        key = dict_key_map.get(cl_key_primary)
        vals = sf.get_dimensional_values(args.get(cl_key_primary), key)
        (
            dict_out.update({key: vals})
            if vals is not None
            else None
        )

    else:
        for k in [cl_key_design, cl_key_future, cl_key_strategy]:
            key = dict_key_map.get(k)
            vals = sf.get_dimensional_values(args.get(k), key)
            (
                dict_out.update({key: vals})
                if vals is not None
                else None
            )

    dict_out =(
        None
        if len(dict_out) == 0
        else dict_out
    )

    return dict_out



def get_file_struct_and_regions(
    args: dict,
) -> Tuple[sfs.SISEPUEDEFileStructure, sc.Regions]:
    """Retrieve the SISEPUEDEFileStructure and Regions objects
    """
    # get file structure to activate model attributes before instantiating SISEPUEDE
    attribute_time_period = args.get("attribute_time_period")
    if isinstance(attribute_time_period, str):
        try:
            attribute_time_period = pathlib.Path(attribute_time_period)
        except Exception as e:
            raise RuntimeError(f"Unable to read attribute table from path '{attribute_time_period}': {e}")
    
    # get the file structure and regions
    file_struct = sfs.SISEPUEDEFileStructure(attribute_time_period = attribute_time_period, )
    regions_obj = sc.Regions(file_struct.model_attributes, )

    out = (file_struct, regions_obj, )

    return out



def get_models(
    models: str,
    model_attributes: ModelAttributes,
    delim: str = ",",
) -> Union[List[str], str, None]:
    """
    Return list of regions to run. Splits argument with delimiter `delim`
    """
    # if invalid, return a list of length 0
    if not isinstance(models, str):
        return None

    # fed to SISEPUEDE; if None, then it will instantiate all regions
    if models.upper() == "ALL":
        return None

    attr_sector = model_attributes.get_sector_attribute_table()
    valid_models = attr_sector.key_values + list(attr_sector.table["sector"])

    models = [
        x for x in models.split(delim)
        if x in valid_models
    ]

    return models



def get_regions(
    region_args: str,
    regions_obj: sc.Regions,
    delim: str = ",",
) -> Union[List[str], str, None]:
    """
    Return list of regions to run. Splits argument with delimiter `delim`
    """
    # if invalid, return a list of length 0
    if not isinstance(region_args, str):
        return []

    # fed to SISEPUEDE; if None, then it will instantiate all regions
    if region_args == "ALLREGIONS":
        return None

    regions = sf.get_dimensional_values(
        region_args, 
        regions_obj.key,
        delim = delim,
        return_type = str,
    )
    regions = [
        regions_obj.return_region_or_iso(x, return_type = "region")
        for x in regions
    ]

    regions = [x for x in regions if x is not None]

    return regions



def get_strategies(
    path_strategies: Union[str, None] = None,
    stop_on_error: bool = False, 
) -> Union[str, None]:
    """Attempt to read transformations directory

    Function Arguments
    ------------------

    Keyword Arguments
    -----------------
    path_strategies : Union[str, None]
        Location to check for strategies. If does not exist or invalidly 
        specified, defaults to None.
    stop_on_error : bool
        If True, will stop procession of the program. Otherwise, defaults to
        None
    """

    ##  CHECK VALIDITY OF INPUTS

    # behavior if specified as default
    if path_strategies is None:
        return None
    
    # check validity of input
    try:
        path_strategies = pathlib.Path(path_strategies)
    except Exception as e:
        msg = f"Unable to create path from '{path_strategies}': {e}"
        if stop_on_error:
            raise InvalidStrategyDirectory(msg)
    
        warnings.warn(msg)

        return None
    
    # verify the directory exists
    if not path_strategies.is_dir():
        msg = f"Unable to instantiate strategies at '{path_strategies}' not found."
        if stop_on_error:
            raise InvalidStrategyDirectory(msg)
        
        warnings.warn(msg)

        return None

    
    ##  SETUP A STRATEGIES OBJECT USING VALID DUMMY DATA--WE ONLY NEED THE POINTER + ATTRIBUTE TABLE

    # next, use a dummy data frame to instantiate values; want to get the strategy attribute + templates from
    examples = sxl.SISEPUEDEExamples()
    df_examples = examples("input_data_frame")

    # set up transformers/transformations
    transformers = trf.Transformers({}, df_input = df_examples,)
    transformations = trf.Transformations(
        path_strategies,
        transformers = transformers,
    )

    # get strategies and attribute (includes verification steps etc.)
    # ONLY USED FOR ATTRIBUTE TABLE AND --EXISTING-- TEMPLATE BUILDS
    strategies = trf.Strategies(
        transformations,
        export_path = "transformations",
    )

    return strategies



def parse_arguments(
) -> dict:
    
    f"""Command line utility to run SImulating SEctoral Pathways and Uncertainty 
        Exploration for DEcarbonization (SISEPUEDE)

    Required arguments:
    -------------------
    --regions           Comma-delimitted list of regions to run. Regions can be 
                            entered as region names or ISO codes. E.g., 
                            BRA,CHL,MEX is acceptable, as is 
                            brazil,chile,mexico. 
                    
                            * To run all available regions (may take a 
                                significant amount of time), use ALLREGIONS.
    

    At Least ONE of the following arguments is required
    ---------------------------------------------------
    NOTE: if specifiying keys_primary and any of keys_design, keys_future, 
        or keys_strategy, key_primary will take precedence. Otherwise, runs all 
        scenarios that match *all* specifications of design, future, and/or 
        strategy (i.e., intersectional set logic).

    NOTE: Keys that are specified in a file must include the following rules:
        * If specifying as a file, the file must include each id on a new line
        * Non-numeric values are skipped
        * If specifying in a CSV, will try to read from the appropriate key

    --keys_design       Comma-delimited list of design ids to run  OR  a valid 
                            file path to a text file containing id numbers to 
                            run. Must contain key SISEPUEDE.key_design, and ids 
                            must be defined in the attribute_design_id.csv 
                            ModelAttributes source file.

    --keys_future       Comma-delimited list of future ids to run  OR  a valid 
                            file path to a text file containing id numbers to 
                            run. Must contain key SISEPUEDE.key_future, and ids 
                            x must be 0 <= x <= n_trials (0 is a null trial)

    --keys_primary      Comma-delimited list of primary ids to run  OR  a valid 
                            file path to a text file containing id numbers to 
                            run. 

    --keys_strategy     Comma-delimited list of strategy ids to run  OR  a valid 
                            file path to a text file containing id numbers to 
                            run. Must contain key SISEPUEDE.key_strategy, and 
                            ids x must be defined in the 
                            attribute_strategy_id.csv ModelAttributes source 
                            file  --OR--  the strategy attribute file defined in
                            the transformations-dir that is specified (if
                            specified)

    Optional arguments:
    -------------------
    --attribute-time-period     Optional specification of a path to an attribute 
                                table for time periods. 
    --id                    Optional AnalysisID name to pass. If not specified, 
                                sets at runtime.
    --max-solve-attempts    Maximum number of attempts to solve a problem. Only
                                retries if emissions values indicate potential 
                                numerical issues with the solution.
    --n-trials              Number of Latin Hypercube trials to run (number of 
                                futures, which represent exogenous uncertainties 
                                and/or lever effect uncertainties)
    --random-seed           Optional random seed to specify. If not specifed, 
                                defaults to configuration specifation. Enter a 
                                negative number to generate on the fly.

    --save-inputs           Include this flag to save inputs as well as outputs.
                                If not specified, inputs are not saved.

    --strategies-dir         Optional directory storing tranformations and 
                                strategies specification to upload. If None, 
                                defaults to definitions in SISEPUEDE package.

    --try-exogenous-xl-types    Include this flag to attempt to read in 
                                exogenous specification of Xs and LS. Reads from 
                                SISEPUEDE.file_struct.fp_variable_specification_xl_types. 
                                If None, infers XL types based on inputs. 
    """
    parser = argparse.ArgumentParser(
        description = ""
    )


    ##  REQUIRED ARGUMENTS

    # regions to run
    msg_hlp_regions = f"""
    Comma-delimited list of regions or ISO codes to run. Required argument. To
    run all available regions, set to ALLREGIONS
    """
    parser.add_argument(
        "--regions",
        help = msg_hlp_regions,
    )


    ##  AT LEAST ONE MUST BE SPECIFIED

    # design keys
    msg_hlp_key_design = f"""
    Comma-delimited list of design keys to read OR file path to table 
        containing list of key values (as rows)
    """
    parser.add_argument(
        "--keys-design",
        default = None,
        help = msg_hlp_key_design,
        type = str,
    )


    # future keys
    msg_hlp_key_future = f"""
    Comma-delimited list of future keys to read OR file path to table 
        containing list of key values (as rows). Only those that are <= n_trials
        will be included.
    """
    parser.add_argument(
        "--keys-future",
        default = None,
        help = msg_hlp_key_future,
        type = str,
    )


    # primary keys
    msg_hlp_key_primary = f"""
    Comma-delimited list of primary keys to read OR file path to table 
        containing list of key values (as rows)
    """
    parser.add_argument(
        "--keys-primary",
        default = None,
        help = msg_hlp_key_primary,
        type = str,
    )


    # strategy keys
    msg_hlp_key_strategy = f"""
    Comma-delimited list of strategy keys to read OR file path to table 
    containing list of key values (as rows)
    """
    parser.add_argument(
        "--keys-strategy",
        default = None,
        help = msg_hlp_key_strategy,
        type = str,
    )



    ##################################
    #    OPTIONAL ARGUMENTS/FLAGS    #
    ##################################

    # optional attribute_time_period to pass
    msg_attribute_time_period = f"""
    Optional specification of a file path to a time period attribute table in a 
    CSV file.
    """
    parser.add_argument(
        "--attribute-time-period",
        type = str,
        help = msg_attribute_time_period,
        default = None,
    )


    # optional database type specification
    msg_hlp_db_type = f"""
    Optional specification of output database type (str). Default is sqlite. 
    Acceptable options are "csv" and "sqlite"
    """
    parser.add_argument(
        "--database-type",
        type = str,
        help = msg_hlp_db_type,
        default = "sqlite",
    )


    # optional flag to *exclude* electricity model 
    msg_hlp_exclude_fuel_prod = f"""
    Exclude the fuel production (NemoMod) model from runs. 
    """
    parser.add_argument(
        "--exclude-fuel-production",
        action = "store_true",
        help = msg_hlp_exclude_fuel_prod,
    )


    # optional AnalysisID string to pass
    msg_hlp_id = f"""
    Optional id to pass on instantiation.
    """
    parser.add_argument(
        "--id",
        default = None,
        help = msg_hlp_id,
        type = str,
    )


    # optional flag for models to include
    msg_hlp_max_solve_attempts = f"""
    Maximum number of times to attempt solving a problem due to numerical 
    instability or solve issues. Default is 2.
    """
    parser.add_argument(
        "--max-solve-attempts",
        type = int,
        help = msg_hlp_max_solve_attempts,
        default = 2,
    )


    # optional flag for models to include
    msg_hlp_models = f"""
    Optional flag used to specify models to run. Possible values include 'All' 
    (run all models [default]) or any comma-delimited combination of the 
    following: 
        
        * AFOLU, AF, or af
        * CircularEconomy, CE, or ce 
        * Energy, EN, or en
        * IPPU, IP, or ip

    NOTE: to turn off running the fuel production (electricity) model, use the 
        flag `--exclude-fuel-production`
    """
    parser.add_argument(
        "--models",
        type = str,
        help = msg_hlp_models,
        default = "All"
    )


    # optional flag for number of trials
    msg_hlp_n_trials = f"""
    Specify the number of Latin Hypercube trials to run (number of futures, 
    which represent exogenous uncertainties and/or lever effect uncertainties). 
    If unspecified, defaults to configuration value.
    """
    parser.add_argument(
        "--n-trials",
        help = msg_hlp_n_trials,
        type = int,
        default = None,
    )


    # optional random seed 
    msg_hlp_random_seed = f"""
    Optional random seed to specify for runs. If not specified, defaults to 
    configuration random seed. 
        
        * NOTE: To choose one randomly, set to a negative number.
    """
    parser.add_argument(
        "--random-seed",
        type = int,
        help = msg_hlp_random_seed,
        default = None
    )


    # optional save inputs
    msg_hlp_save_inputs = f"""
    Include the --save-inputs flag to save off inputs to the 
    SISEPUEDEOutputDatabase. In general, model inputs are not saved off to 
    reduce space requirements and can generally be accessed using
    
        (
            SISEPUEDE
            .experimental_manager
            .dict_future_trajectories
            .get(region)
            .generate_future_from_lhs_vector()
        )
    """
    parser.add_argument(
        "--save-inputs",
        action = "store_true",
        help = msg_hlp_save_inputs,
    )


    # optional random seed 
    msg_hlp_strategies_dir = f"""
    Optional directory of transformations that have been defined, including 
    strategies, to upload. If not specified, defaults to SISEPUEDE defaults.
    """
    parser.add_argument(
        "--strategies-dir",
        type = str,
        help = msg_hlp_strategies_dir,
        default = None,
    )


    # optional attempt to read exogenous XL types
    msg_hlp_try_exogenous_xl_types = f"""
    Include the --try-exogenous-xl-types flag to try to read exogenous XL types 
    for variable specifications (as fed to SamplingUnit). Reads from 
    SISEPUEDE.file_struct.fp_variable_specification_xl_types. If None, infers XL 
    types based on inputs. 

        NOTE: "X" or "L" cannot be specified for any inferred XL types. 
        * If a variable is inferred as an X, it can be exogenously specified as
            either an "X" or an "L" (Xs can be treated as Ls)
        * If a variable is inferred as an L, it can only be specified as an L;
            this is because uncertainty exploration will fail otherwise.
    """
    parser.add_argument(
        "--try-exogenous-xl-types",
        action = "store_true",
        help = msg_hlp_try_exogenous_xl_types,
    )

    

    if False:
        # optional input csv
        msg_hlp_from_input = f"""
        Path to an input CSV, long by time_period, that contains required input 
            variables. 
            
        NOTE: use with caution. This is approach is not-well checked
            and current implementation of error handling may be insufficient.
        """
        parser.add_argument(
            "--from-input",
            type = str,
            help = msg_hlp_from_input,
            default = None,
        )


        # optional output csv
        msg_hlp_from_input = f"""
        Path to an output CSV, long by time_period. Only used it --from-input is 
            specified.
        """
        parser.add_argument(
            "--to-output",
            type = str,
            help = msg_hlp_from_input,
            default = None,
        )


    ##  PARSE ARGUMENTS

    parsed_args = parser.parse_args()

    # Since defaults are env vars, still need to checking to make sure its passed
    errors = []
    if parsed_args.regions is None:
        errors.append(
            "Missing --regions argument. Use --regions ALLREGIONS to run all available regions."
        )
    
    if errors:
        raise ValueError(f"Missing arguments detected: {sf.format_print_list(errors)}")

    # json args over-write specified args
    parsed_args_as_dict = vars(parsed_args)

    return parsed_args_as_dict





def main(
    args: dict,
) -> None:
    """SImulation of SEctoral Pathways and Uncertaintay Exploration for 
        DEcarbonization (SISEPUEDE)
    
    Copyright (C) 2023-2025 James Syme
    
    This program comes with ABSOLUTELY NO WARRANTY. This is free software, and 
    you are welcome to redistribute it under certain conditions. See LICENSE.md 
    for the conditions. 

    MIT LICENSE:

    """

    warnings.filterwarnings("ignore")

    # get file structure to activate model attributes before instantiating SISEPUEDE
    file_struct, regions_obj = get_file_struct_and_regions(args, )


    ##  1. GET INPUTS

    # models
    models = get_models(args.get("models"), file_struct.model_attributes)
    matt = file_struct.model_attributes

    # regions
    regions_run = get_regions(
        args.get("regions"),
        regions_obj,
        delim = ",",
    )

    # call a strategies object?
    path_strategies = args.get("strategies_dir")
    strategies = get_strategies(
        path_strategies = path_strategies,
        stop_on_error = True,
    )

    # scenario information
    dict_scenarios = get_dimensional_dict(args, matt, )
    if dict_scenarios is None:
        msg = f"""No valid dimensional subsets or scenarios were specified. Ensure that 
            either primary_id OR any combination of design_id, future_id, and/or
            straetgy_id are specified.
        """
        raise RuntimeError(msg)
        return None

    # additional parameters
    db_type = args.get("database_type")
    db_type = "sqlite" if (db_type not in ["sqlite", "csv"]) else db_type
    id_str = args.get("id")

    # a number of additional key checks
    include_fuel_prod = (not args.get("exclude_fuel_production"))
    max_solve_attempts = args.get("max_solve_attempts")
    n_trials = args.get("n_trials")
    random_seed = args.get("random_seed")
    save_inputs = args.get("save_inputs")
    try_xl_types = args.get("try_exogenous_xl_types")

    # checks
    return_none = (regions_run is None)
    return_none |= ((len(dict_scenarios) == 0) if not return_none else False)
    return_none |= (dict_scenarios is None)

    if return_none:
        raise RuntimeError("Invalid specification of regions, input dimensions. Check arguments and try again")
        return None


    ##  2. INITIALIZE SISEPUEDE AND RUN

    sisepuede = ssp.SISEPUEDE(
        "calibrated",
        attribute_time_period = matt.get_dimensional_attribute_table(matt.dim_time_period, ),
        db_type = db_type,
        id_str = id_str,
        n_trials = n_trials,
        random_seed = random_seed,
        regions = regions_run,
        strategies = strategies,
        try_exogenous_xl_types_in_variable_specification = try_xl_types,
    )

    dict_primaries_complete = sisepuede(
        dict_scenarios,
        chunk_size = 2,
        check_results = True,
        include_electricity_in_energy = include_fuel_prod,
        max_attempts = max_solve_attempts,
        reinitialize_output_table_on_verification_failure = True,
        save_inputs = save_inputs,
    )

    sisepuede._log(
        f"SISEPUEDE run '{sisepuede.id}' complete.", 
        type_log = "info"
    )

    return 0

    

if __name__ == "__main__":

    args = parse_arguments()
        
    main(args)
    
