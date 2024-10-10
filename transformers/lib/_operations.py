
from typing import *
import copy
import inspect
import pandas as pd
import pathlib

from sisepuede.core.attribute_table import AttributeTable
import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf
import sisepuede.transformers.strategies as st
import sisepuede.transformers.transformations as trn
import sisepuede.transformers.transformers as trs




#####################################
###                               ###
###    COMBINE TRANSFORMATIONS    ###
###                               ###
#####################################


def build_default_general_config_dict(
    transformers: trs.Transformers,
) -> dict:
    """
    Build the default general configuration dictionary for a new transformation
        definition directory.

    """
    ##  GENERAL

    dict_out = trs.get_dict_config_default()
    dict_out = dict_out.dict_yaml

    # add the implementation vector

    dict_vir = spawn_args_dict(
        transformers.build_implementation_ramp_vector,
    )

    # clean tuples
    dict_vir = dict(
        (k, list(v) if isinstance(v, tuple) else v)
        for k, v in dict_vir.items()
    )

    dict_vir = {
        trs._DICT_KEYS.get("vec_implementation_ramp"): dict_vir,
    }

    dict_vir2 = copy.deepcopy(dict_vir)

    # update general dictionary
    (
        dict_out
        .get(trs._DICT_KEYS.get("general"))
        .update(dict_vir)
    )

    
    ##  BASELINE

    dict_update = spawn_args_dict(
        transformers._trfunc_baseline,
        args_ignore = [
            "n_tp_ramp",
            "tp_0_ramp",
            "vec_implementation_ramp",
            "strat"
        ],
    )

    (
        dict_out
        .get(trs._DICT_KEYS.get("baseline"))
        .update(dict_update)
    )

    (
        dict_out
        .get(trs._DICT_KEYS.get("baseline"))
        .update(dict_vir2)
    )


    return dict_out



def build_default_strategies(
    transformations: trn.Transformations,
    baseline_strategy_id: int = 0,
    code_cross_sector: str = "PFLO",
    delim: str = "|",
    factor_sectoral_multiplier: int = 1000,
    nm_strat_all: str = "ALL",
    **kwargs,
) -> pd.DataFrame:
    """
    Build a default strategy definition table based on transformations. 
        
    The default table includes:
    
        * Baseline 
        * Each transformation alone
        * Sector combinations. For each sector, the table will look for 
            any transformations associated with a relevant subsector 
            code (e.g., TX:SUBSEC:CODE_HERE)
        * All transformations
        
    NOTE: Users should take care to modify the "ALL" strategy in the 
        presence of of multiple transformations that use the same 
        transformer code.
        
        If multiple transformations share the same code, then the default
        strategie_definitions file will pick the first transformation that
        uses that code.
    
    
    Function Arguments
    ------------------
    - transformations: Transformations object used to build default 
        strategies
    
    Keyword Arguments
    -----------------
    - baseline_strategy_id: baseline strategy id in default
    - code_cross_sector: code used for cross-sector
    - delim: delimter used to divide transformations in transformation 
        specification
    - factor_sectoral_multiplier: factor that is applied to sector indices
        to build strategy ids
    - nm_strat_all: name for all strategy
    """
    
    ##  SOME INITIALIZATION
    
    attr_sector = (
        transformations
        .transformers
        .model_attributes
        .get_sector_attribute_table()
    )
        
    key_baseline = "baseline"
    
    
    ##  COLLECT COMPOSITE DEFAULTS
    
    # first, identify a code for each tranformer
    dict_transformation_codes_by_transformer_code = transformations.get_transformation_codes_by_transformer_code()
    all_transformation_codes = []
    
    for k, v in dict_transformation_codes_by_transformer_code.items():
        
        # skip any transformers with no associated transformation codes
        if (len(v) == 0) | (k == transformations.transformers.code_baseline):
            continue
            
        v_sorted = sorted(v)
        all_transformation_codes.append(v_sorted[0])
    
    
    ##  NEXT, GENERATE SECTOR-SPECIFIC COMPOSITES
    
    dict_transformer_codes_by_sector = (
        transformations
        .transformers
        .get_transformer_codes_by_sector()
    )
    
    # - dict_transformation_codes_by_sector stores
    # - dict_transformation_codes_by_sector_all is used for ordering the singleton strategies
    dict_transformation_codes_by_sector = {}
    dict_transformation_codes_by_sector_all = {
        key_baseline: [transformations.code_baseline],
    }
    
    
    
    # get transformation codes that will be kept in each sector-specific subset
    # - look over all transformer codes by sector
    for k, v in dict_transformer_codes_by_sector.items():
        transformation_codes = []
        transformation_codes_all = []
        
        for w in v:
            codes_tr = dict_transformation_codes_by_transformer_code.get(w)
            if codes_tr is None:
                continue
            
            # choose transformations associated with the transformer code that are in all
            for j in codes_tr:
                
                transformation_codes_all.append(j)
                if j in all_transformation_codes:
                    transformation_codes.append(j)
        
        dict_transformation_codes_by_sector.update({k: sorted(transformation_codes), })
        dict_transformation_codes_by_sector_all.update({k: sorted(transformation_codes_all), })
    
    
    ##  FINALLY, START BUILDING TABLE
    
    # get some fields
    field_strategy_id = transformations.transformers.model_attributes.dim_strategy_id
    
    field_description = kwargs.get(
        "field_description", 
        st._DICT_KEYS.get("description"), 
    )
    field_strategy_code = kwargs.get(
        "field_strategy_code", 
        st._DICT_KEYS.get("code"), 
    )
    field_strategy_name = kwargs.get(
        "field_strategy_name", 
        st._DICT_KEYS.get("name"), 
    )
    field_transformation_specification = kwargs.get(
        "field_transformation_specification", 
        st._DICT_KEYS.get("transformation_specification"), 
    )
    
    
    # initialize columns
    
    vec_codes = []
    vec_descriptions = []
    vec_ids = []
    vec_names = []
    vec_transformation_specifications = []
    
    
    # ORDER KEYS
    
    keys_sector = [
        attr_sector.get_attribute(k, "sector")
        for k in attr_sector.key_values
    ]
    keys = ["baseline"] + keys_sector + ["other"]
    
    sid_factor = 0
    
    # iterate over each of the groupings
    for k in keys:
        
        # initialize the strategy id
        strat_id = sid_factor*factor_sectoral_multiplier + baseline_strategy_id
        
        # singletons
        codes_single = dict_transformation_codes_by_sector_all.get(k)
        for code in codes_single:
            
            # get the transformation and new strategy code and name
            tr_cur = transformations.get_transformation(code)
            code_strat, name_strat = _cn_default_strategy_singleton(tr_cur, )
            
            # update vectors
            vec_codes.append(code_strat)
            vec_descriptions.append(tr_cur if isinstance(tr_cur, str) else "")
            vec_ids.append(strat_id)
            vec_names.append(name_strat)
            vec_transformation_specifications.append(code)
            
            strat_id += 1
        
        
        # next, add sector composite
        if k in keys_sector:
            
            codes = dict_transformation_codes_by_sector.get(k)
            
            if len(codes) > 0:
                description = f"All (unique by transformer) {k} transformations"
                code_strat, name_strat = _cn_default_strategy_sector_composite(
                    k,
                    attr_sector = attr_sector,
                    nm_strat_all = nm_strat_all,
                )

                ts_code = delim.join(dict_transformation_codes_by_sector.get(k))

                # update vectors
                vec_codes.append(code_strat)
                vec_descriptions.append(description)
                vec_ids.append(strat_id)
                vec_names.append(name_strat)
                vec_transformation_specifications.append(ts_code)

                strat_id += 1
        
        sid_factor += 1
        
        
    # finally, build all actions
    #strat_id = sid_factor*factor_sectoral_multiplier + baseline_strategy_id
    description = f"All actions (unique by transformer)"
    code_strat = f"{code_cross_sector}:{nm_strat_all}"
    name_strat = "All Actions"
    ts_code = delim.join(all_transformation_codes)

    # update vectors
    vec_codes.append(code_strat)
    vec_descriptions.append(description)
    vec_ids.append(strat_id)
    vec_names.append(name_strat)
    vec_transformation_specifications.append(ts_code)
        
        
    
    ##  BUILD THE OUTPUT DATA FRAME
   
    df_out = {
        field_strategy_id: vec_ids,
        field_strategy_code: vec_codes,
        field_strategy_name: vec_names,
        field_description: vec_descriptions,
        field_transformation_specification: vec_transformation_specifications, 
    }
    
    df_out = pd.DataFrame(df_out)
    
    return df_out



def build_default_transformation_config_dict(
    transformer: trs.Transformer,
    dict_code_prepenage_map: dict = {
        trs._MODULE_CODE_SIGNATURE: trn._MODULE_CODE_SIGNATURE,
    },
    prependage_name: str = "Default Value - ",
) -> Tuple[dict, str]:
    """
    Build a dictionary for a configuration file for a Transformation. Returns
        a tuple with a dictionary and file name

    Function Arguments
    ------------------
    - transformer_code: string Transformer code used to spawn the new 
        Transformation config using defaults

    Keyword Arguments
    -----------------
    - dict_code_prepenage_map: dictionary mapping Transformer code signature to 
        Transformation code signature
    - prependage_name: prendage added to transformer names to create 
        transformation names
    """

    # get some keys for TRANSFORMATION definitions
    key_citations = trn._DICT_KEYS.get("citations")
    key_description = trn._DICT_KEYS.get("description")
    key_identifiers = trn._DICT_KEYS.get("identifiers")
    key_transformation_code = trn._DICT_KEYS.get("code")
    key_transformation_name = trn._DICT_KEYS.get("name")
    key_transformer = trn._DICT_KEYS.get("transformer")
    key_parameters = trn._DICT_KEYS.get("parameters")

    # get the new code and name
    code_new = sf.str_replace(
        str(transformer.code),
        dict_code_prepenage_map,
    )

    # name adds a prependage
    name_new = (
        f"{prependage_name}{transformer.name}"
        if isinstance(prependage_name, str)
        else transformer.name
    )
    

    ##  BUILD DICTIONARIES AND FILE NAME

    dict_parameters = spawn_args_dict(
        transformer,
        args_ignore = [
            "df_input",
            "strat"
        ]
    )

    # wrap 
    dict_out = {
        key_citations: transformer.citations,
        key_description: sf.yaml_quoted(transformer.description),
        key_identifiers: {
            key_transformation_code: sf.yaml_quoted(code_new),
            key_transformation_name: sf.yaml_quoted(name_new),
        },
        key_parameters: dict_parameters,
        key_transformer: sf.yaml_quoted(transformer.code)
    }

    file_name_out = code_to_file_name(transformer.code, )

    # return a tuple
    out = (dict_out, file_name_out, )

    return out



def _cn_default_strategy_sector_composite(
    sector_name: str,
    attr_sector: Union[AttributeTable, None] = None,
    nm_strat_all: str = "ALL",
) -> Union[Tuple[str, str], None]:
    """
    Generate a code/name combination for naming a sectoral composite
        strategy.
    """
    
    # get the attribute table
    attr_sector = (
        (
            transformations
            .transformer
            .model_attributes
            .get_sector_attribute_table()
        )
        if attr_sector is None
        else attr_sector
    )
    
    dict_map = attr_sector.field_maps.get(f"sector_to_{attr_sector.key}")
    
    # get the abbreviation for the code
    sector_abv = dict_map.get(sector_name)
    sector_abv = sector_abv if sector_abv is not None else sector_name
    sector_abv = ma.clean_schema(sector_abv).upper()
    
    # set the code
    code = f"{sector_abv}:{nm_strat_all}"
    name = f"Sectoral Composite - {sector_name}"
    
    out = (code, name)
    
    return out



def _cn_default_strategy_singleton(
    transformation: trn.Transformation, 
) -> Union[Tuple[str, str], None]:
    """
    Generate a code/name combination for naming a singleton strategy 
        based on a transformation.
    """
    
    if not trn.is_transformation(transformation):
        return None
    
    # modify code/name
    code = (
        transformation
        .code
        .replace(
            f"{trn._MODULE_CODE_SIGNATURE}:",
            ""
        )
    )
    
    name = (
        f"Strategy {transformation.code}"
        if transformation.name is None
        else f"Singleton - {transformation.name}"
    )
    
    out = (code, name)
    
    return out



def code_to_file_name(
    code: str,
    prependage: str = trn._TRANSFORMATION_REGEX_FLAG_PREPEND,
) -> str:
    """
    Convert a transformer `code` to a file name 
    """
    
    out = ma.clean_schema(
        code
        .replace(trs._MODULE_CODE_SIGNATURE, "")
        .strip(":")
        .replace(":", "_")
    )
    
    out = f"{prependage}_{out}.yaml"
    
    return out



def instantiate_default_transformations(
    transformers: trs.Transformers,
    path_transformations: Union[str, pathlib.Path],
    export_transformations: bool = True,
    fn_citations: str = trn._DICT_FILE_NAME_DEFAULTS.get("citations"),
    fn_config_general: str = trn._DICT_FILE_NAME_DEFAULTS.get("config_general"),
    fn_strategy_definitions: str = st._DICT_FILE_NAME_DEFAULTS.get("strategy_definitions"),
    mk_path: bool = True,
    return_dict: bool = False,
    **kwargs,
) -> Union[dict, None]:
    """
    Instantiate default transformations configuration files and strategy
        definition file. 
        
    Function Arguments
    ------------------
    - path_transformations: output directory where default transformations are 
        to be spawned
        
    Keyword Arguments
    -----------------
    - export_transformations: build the output directory and export?
    - fn_citations: file name of Bibtex file in dir_init containing optional 
        citations to provide
    - fn_config_general: file name of the general configuration file in dir_init
    - mk_path: make the path of the direcetory if it doesn't exist? Only applies 
        if `export_transformations == True`
    - return_dict: if True, will return the dictionary of outputs
    - **kwargs: passed to build_default_strategies()  
    """
    
    ##  VERIFY INPUTS
    
    # check transformers
    if not trs.is_transformers(transformers):
        tp = str(type(transformers))
        msg = f"Invalid input type '{tp}' specified for transformers. Must be a Transformers object."
        raise RuntimeError(msg)
        
    # check input path
    try:
        path_transformations = pathlib.Path(path_transformations)
        
    except Exception as e:
        
        msg = f"Unable to instantiate transformation export directory '{path_transformations}': {e}"
        transformers._log(msg, type_log = "error", )
        raise RuntimeError(msg)
    
    # build it if it doesn't exist?
    if (not path_transformations.exists()) & export_transformations:
        if not mk_path:
            msg = f"Export directory '{path_transformations}' does not exist. Set 'mk_path = True' to build the directory in this case."
            transformers._log(msg, type_log = "error", )
            raise RuntimeError(msg)
        
        # build
        os.makedirs(str(path_transformations), exist_ok = True, )
            
    
    ##  BUILD OUTPUT FILES 
    
    # get default general configuration
    dict_default_general = build_default_general_config_dict(transformers, )
    
    dict_transformations = {}
    
    for code in transformers.all_transformers:
        
        # ignore the baseline; that's set in the general config
        if code == transformers.code_baseline:
            continue
        
        # get the transformer, file name, and dictionary
        transformer = transformers.get_transformer(code)
        dict_export = build_default_transformation_config_dict(transformer, ) # fn, dict
        
        # add to output dict
        dict_transformations.update({code: dict_export, })
    

    
    ##  EXPORT?
    
    if export_transformations:
        
        # write general
        path_config = path_transformations.joinpath(fn_config_general)
        sf._write_yaml(
            dict_default_general,
            path_config,
        )
        
        # write citations (empty file)
        path_citations = path_transformations.joinpath(fn_citations)
        with open(path_citations, "w+") as fp:
            fp.writelines(["\n"])
        
        # write transformations
        for k, v in dict_transformations.items():
            sf._write_yaml(
                v[0],
                path_transformations.joinpath(v[1]),
            )
            

        ##  GENERATE STRATEGY DEFINITION FROM TRANSFORMATIONS
        
        transformations = trn.Transformations(
            path_transformations,
            transformers = transformers,
        )
        # generate the output default strategy definitions file
        df_strategy_def = build_default_strategies(
            transformations,
            **kwargs,
        )

        path_strategy_def = path_transformations.joinpath(fn_strategy_definitions, )
        df_strategy_def.to_csv(
            str(path_strategy_def),
            index = None,
            encoding = "UTF-8",
        )

        
    if not return_dict:
        return None
        
        
    # temp
    return dict_transformations



def spawn_args_dict(
    transformer: Union[callable, trs.Transformer],
    args_ignore: Union[List[str], None] = None,
    include_kwargs: bool = True,
) -> Union[dict, None]:
    """
    Read a transformer object and convert optional and kwargs to
        a dictionary. If the transformer argument is not a Transformer
        object, the function returns None. 
        
    Function Arguments
    ------------------
    - transformer: a Transformer object from which to build the argument 
        dictionary OR a function
    
    Keyword Arguments
    -----------------
    - args_ignore: Optional specification of arguments to ignore from the 
        dictionary
    - include_kwargs: include kwargs only?  If False, will ignore keyword
        arguments
    """
    
    # if not a transformer, return None
    if not trs.is_transformer(transformer) | callable(transformer):
        return None
    
    
    ##  GET FULL ARGSPEC AND BUILD DICTIONARY
    
    full_arg_spec = (
        inspect.getfullargspec(transformer.function)
        if trs.is_transformer(transformer)
        else inspect.getfullargspec(transformer)
    )
    
    args = full_arg_spec.args
    dflts = full_arg_spec.defaults
    n = len(dflts)
    
    # initialize the output dictionary
    dict_out = dict(zip(args[-n:], dflts))
    
    
    # check if keyword arguments should be added
    check_kwargs = include_kwargs
    check_kwargs &= len(full_arg_spec.kwonlyargs) > 0
    check_kwargs &= sf.islistlike(full_arg_spec.kwonlydefaults)
    
    if check_kwargs:
        dict_out.update(
            dict(
                zip(
                    full_arg_spec.kwonlyargs,
                    full_arg_spec.kwonlydefaults
                )
            )
        )
        
    
    ##  FINALLY, DROP UNWANTED KEYS
    
    if sf.islistlike(args_ignore):
        for k in args_ignore:
            dict_out.pop(k) if (k in dict_out.keys()) else None
                
    return dict_out
    
    
    




