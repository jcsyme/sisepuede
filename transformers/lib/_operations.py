
from typing import *
import copy
import inspect
import pandas as pd


import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf
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

    dict_out = {
        key_citations: transformer.citations,
        key_description: transformer.description,
        key_identifiers: {
            key_transformation_code: code_new,
            key_transformation_name: name_new,
        },
        key_parameters: dict_parameters,
    }

    file_name_out = code_to_file_name(transformer.code, )

    # return a tuple
    out = (dict_out, file_name_out, )

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
    
    
    




