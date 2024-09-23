import logging
import numpy as np
import pandas as pd
import re
import time
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf




_MODULE_UUID = "D3BC5456-5BB7-4F7A-8799-AFE0A44C3FFA" 



#####################################
###                               ###
###    BEGIN CLASS DEFINITIONS    ###
###                               ###
#####################################

class Strategy:
    """
    A collection of transformations

    Initialization Arguments
    ------------------------
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))
    """

    def __init__(self,
    ) -> None:

        return None
    


    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
        overwrite_docstr: bool = True,
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None

        if isinstance(func, list):

            func = [x for x in func if callable(x)]

            if len(func) > 0:  
                
                overwrite_docstr &= (len(func) == 1)

                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformer function for {self.name}
                    """
                    out = None
                    if len(args) > 0:
                        out = (
                            args[0].copy() 
                            if isinstance(args[0], pd.DataFrame) | isinstance(args[0], np.ndarray)
                            else args[0]
                        )

                    for f in func:
                        out = f(out, **kwargs)

                    return out

                function = function_out
                function_list = func
            
            else:
                overwrite_docstr = False

        elif callable(func):
            function = func
            function_list = [func]

        
        # overwrite doc?
        if overwrite_docstr:
            self.__doc__ = function_list[0].__doc__ 

        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None




class Transformer:
    """
    Create a Transformation class to support construction in sectoral 
        transformations. 

    Initialization Arguments
    ------------------------
    - code: transformer code used to map the transformer to the attribute table. 
        Must be defined in attr_transfomers.table[attr_transfomers.key]
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    - attr_transformers: AttributeTable usd to define transformers from 
        ModelAttributes

    Keyword Arguments
    -----------------
    - code_baseline: transformer code that stores the baseline code, which is 
        applied to raw data.
    - field_transformer_id: field in attr_transfomer.table containing the
        transformer index
    - field_transformer_name: field in attr_transfomer.table containing the
        transformer name
    - overwrite_docstr: overwrite the docstring if there's only one function?
    """
    
    def __init__(self,
        code: str,
        func: Callable,
        attr_transfomer: Union[AttributeTable, None],
        code_baseline: str = "TX:BASE",
        field_transformer_id: str = "transformer_id",
        field_transformer_name: str = "transformer",
        overwrite_docstr: bool = True,
    ) -> None:

        self._initialize_function(
            func, 
            overwrite_docstr,
        )
        self._initialize_code(
            code, 
            code_baseline,
            attr_transfomer, 
            field_transformer_id,
            field_transformer_name,
        )

        self._initialize_properties()

        return None
        

    
    def __call__(self,
        *args,
        **kwargs
    ) -> Any:
        
        val = self.function(
            *args,
            # strat = self.id,
            **kwargs
        )

        return val



    def _initialize_code(self,
        code: str,
        code_baseline: str,
        attr_transfomer: Union[AttributeTable, None],
        field_transformer_id: str,
        field_transformer_name: str,
    ) -> None:
        """
        Initialize transfomer identifiers, including the code (key), name, and
            ID. Sets the following properties:

            * self.baseline
            * self.code
            * self.id
            * self.name
        """
        
        # check code
        if code not in attr_transfomer.key_values:
            raise KeyError(f"Invalid Transformer code '{code}': code not found in attribute table.")

        # initialize and check code/id num
        id_num = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{field_transformer_id}")
            if attr_transfomer is not None
            else None
        )
        id_num = id_num.get(code) if (id_num is not None) else -1


        # initialize and check name/id num
        name = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{field_transformer_name}")
            if attr_transfomer is not None
            else None
        )
        name = name.get(code) if (name is not None) else ""


        # check baseline
        baseline = (code == code_baseline)


        ##  SET PROPERTIES

        self.baseline = bool(baseline)
        self.code = str(code)
        self.id = int(id_num)
        self.name = str(name)
        
        return None

    
    
    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
        overwrite_docstr: bool = True,
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None

        if isinstance(func, list):

            func = [x for x in func if callable(x)]

            if len(func) > 0:  
                
                overwrite_docstr &= (len(func) == 1)

                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformer function for {self.name}
                    """
                    out = None
                    if len(args) > 0:
                        out = (
                            args[0].copy() 
                            if isinstance(args[0], pd.DataFrame) | isinstance(args[0], np.ndarray)
                            else args[0]
                        )

                    for f in func:
                        out = f(out, **kwargs)

                    return out

                function = function_out
                function_list = func
            
            else:
                overwrite_docstr = False

        elif callable(func):
            function = func
            function_list = [func]

        
        # overwrite doc?
        if overwrite_docstr:
            self.__doc__ = function_list[0].__doc__ 

        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None
    


    def _initialize_properties(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_transformer
            * self.uuid
        """

        self.is_transformer = True
        self.uuid = _MODULE_UUID

        return None






class Transformation:
    """
    Parameterization class for Transformer. Used to vary implementations
        of Transfomers.
    

    Initialization Arguments
    ------------------------
    - config: specification of configuration dictionary used to map parameters
        to Transformer. Can be:

        * dict: configuration dictionary
        * str: file path to configuration file to read
        * YAMLConfiguration: existing YAMLConfiguration

    - transformers: Transformers object used to validate input parameters and 
        call function

    Optional Arguments
    ------------------
    - **kwargs: Optional keyword arguments can include the following elements
        

        ##  Configuration Keys

        * key_citations
        * key_description
        * key_identifiers
        * key_parameters
        * key_transformation_code
        * key_transformation_name
        * key_transformer

    """
    
    def __init__(self,
        config: Union[dict, str, sc.YAMLConfiguration],
        transformers: Transformer,
        **kwargs,
    ) -> None:

        self._initialize_keys(
            **kwargs,
        )

        self._initialize_config(
            config,
            transformers,
        )

        self._initialize_identifiers()
        self._initialize_function(transformers, )
        
        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
    def _initialize_config(self,
        config: Union[dict, str, sc.YAMLConfiguration],
        transformers: Transformer,
    ) -> None:
        """
        Set the configuration used to parameterize the transformer as well as
            any derivative properties. Sets the following properties:

            * self.config
            * self.dict_parameters
            * self.transformer_code
        """
        
        ##  CHECK CONFIG TYPE

        config = (
            sc.YAMLConfiguration(config, )
            if isinstance(config, dict) | isinstance(config, str)
            else config
        )

        if not sc.is_yaml_configuration(config):
            tp = str(type(config))
            msg = f"Invalid type '{tp}' specified for config in Transformation: must be of type dict, str, or YAMLConfiguration"

            raise TypeError(msg)


        ##  VERIFY KEYS

        # verify top-level keys
        sf.check_keys(
            config.dict_yaml, 
            [
                self.key_identifiers,
                self.key_parameters,
                self.key_transformer
            ]
        )

        # check that a transformation code is specified
        sf.check_keys(
            config.get(self.key_identifiers),
            [
                self.key_transformation_code
            ]
        )


        ##  CHECK THE TRANFORMER CODE AND GET PARAMETERS

        transformer_code = config.get(self.key_transformer)
        if transformer_code not in transformers.all_transformers:
            msg = f"Transformer code '{transformer_code}' not found in the Transformers. The Transformation cannot be instantiated."
            raise KeyError(msg)


        # check parameter specification
        dict_parameters = self.get_parameters_dict(config, transformers, )
        

        ##  SET PROPERTIES

        self.config = config
        self.dict_parameters = dict_parameters
        self.transformer_code = transformer_code

        return None
    


    def _initialize_function(self,
        transformers: Transformer,
    ) -> None:
        """
        Assign the transformer function with configuration-specified keyword
            arguments. Sets the following properties:

            * self.function
        """

        transformer = transformers.get_transformer(self.transformer_code, )
        
        # build the output function
        def func(
            *args
        ):
            out = transformer.function(*args, **self.dict_parameters, )

            return out
        
        # update docstrings
        func.__doc__ = transformer.function.__doc__
        self.__doc__ = transformer.function.__doc__


        ##  SET PARAMETERS
        
        self.function = func
        
        return None



    def _initialize_identifiers(self,
    ) -> None:
        """
        Set transformation code and, optionally, transformation name. Sets the
            following properties:

            * self.code
            * self.name
        """

        code = self.config.get(self.key_yc_trasformation_code)
        name = self.config.get(self.key_yc_trasformation_name)

        
        ##  SET PROPERTIES

        self.code = code
        self.name = name

        return None


    
    def _initialize_keys(self,
        **kwargs,
    ) -> None:
        """
        Set the optional and required keys used to specify a transformation.
            Can use keyword arguments to set keys.
        """

        # set some shortcut codes 

        key_identifiers = kwargs.get("key_identifiers", "identifiers")
        key_transformation_code = kwargs.get("key_transformation_code", "transformation_code")
        key_transformation_name = kwargs.get("key_transformation_name", "transformation_name")

        key_yc_trasformation_code = f"{key_identifiers}.{key_transformation_code}"
        key_yc_trasformation_name = f"{key_identifiers}.{key_transformation_name}"


        ##  SET PARAMETERS

        self.key_citations = kwargs.get("key_citations", "citations")
        self.key_description = kwargs.get("key_description", "description")
        self.key_identifiers = key_identifiers
        self.key_parameters = kwargs.get("key_parameters", "parameters")
        self.key_transformation_code = key_transformation_code
        self.key_transformation_name = key_transformation_name
        self.key_transformer = kwargs.get("key_transformer", "transformer")
        self.key_yc_trasformation_code = key_yc_trasformation_code
        self.key_yc_trasformation_name = key_yc_trasformation_name

        return None
    


    
    


    def get_parameters_dict(self,
        config: sc.YAMLConfiguration,
        transformers: Transformers,
    ) -> None:
        """
        Get the parameters dictionary associated with the specified Transformer.
            Keeps only keys associated with valid default and keyword arguments 
            to the Transformer function.
        """

        # try retrieving dictionary; if not a dict, conver to empty dict
        dict_parameters = config.get(self.key_parameters)
        if not isinstance(dict_parameters, dict):
            dict_parameters = {}

        # get transformer
        transformer_code = config.get(self.key_transformer)
        transformer = transformers.get_transformer(transformer_code)
        if not is_transformer(transformer, ):
            raise RuntimeError(f"Invalid transformation '{transformation_code}' found in Transformers")

        # get arguments to the function 
        _, keywords = sf.get_args(
            transformer.function, 
            include_defaults = True,
        )
        dict_parameters = dict(
            (k, v) for (k, v) in dict_parameters.items() if k in keywords
        )

        return dict_parameters

"""
citations:
  - xyz
  - xbm
identifiers: 
  transformation_code: "TX:TRNS:SHIFT_FUEL_MEDIUM_DUTY"
  transformation_name: "This one for Inidia 123"
description:
  "blah blah blah"
parameters:
  categories:
  - road_heavy_freight
  - road_heavy_regional
  - public
  dict_allocation_fuels_target: null
  fuels_source:
  - fuel_diesel
  - fuel_gas
  magnitude: 0.7
  vec_implementation_ramp:
    n_tp_ramp: 14
    tp_0_ramp: 14
transformer:
"""



class Transformations:
    """
    Build a collection of parameters used to construct transformations. The 
        Transformations class searches a specified directory to ingest two types
        of files:

        (1) Transformation configuration files, which define transformations as 
            parameterizations of exiting Transformers. By default, these files 
            should follow the

            `transformation_TEXTHERE.yaml` pattern (written as a regular
                expression as "transformation_(.\D*).yaml")
            
            though this can be modified ysing the 
            `regex_transformation_template` keyword argument.

            Each transformation configuration file **must** include the `codes`
            key at the top level (level 0) in addition to `transformation` and 
            `transformer` codes at level 1. 

            `codes.transformation`: unique code for the transformation; should 
                be wrapped in double quotes in the YAML configuration file. 
                Additionally, codes should follow a convention; for example,

                "TX:AGRC:INC_CONSERVATION_AGRICULTURE_FULL"




                citations:
                - xyz
                - xbm
                codes: 
                transformation: "TX:TRNS:SHIFT_FUEL_MEDIUM_DUTY"
                transformer: "TFR:TRNS:SHIFT_FUEL_MEDIUM_DUTY"
                description:
                "Description of transformer here"
                parameters:
                categories:
                    - road_heavy_freight
                    - road_heavy_regional
                    - public
                dict_allocation_fuels_target: null
                fuels_source:
                    - fuel_diesel
                    - fuel_gas
                magnitude: 0.7
                vec_implementation_ramp:

            
  


        (2) A strategy definition table, which provides strategy names and IDs
            for combinations of transformations. By default, this file is called

            `strategy_definitions.csv`
            
            though this can be modified ysing the `fn_strategy_definition` 
            keyword argument.
        

    
    Initialization Arguments    
    ------------------------


    Optional Arguments
    ------------------
    """

    def __init__(self,
        dir_init: str,
        regex_transformation_config: re.Pattern = re.compile("transformation_(.\D*).yaml"),
        **kwargs,
    ) -> None:
        
        return None

    
    
    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################





########################
#    SOME FUNCTIONS    #
########################

def is_transformer(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Transformer
    """
    out = hasattr(obj, "is_transformer")
    uuid = getattr(obj, "uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out









