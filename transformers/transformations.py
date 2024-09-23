import logging
import numpy as np
import pandas as pd
import pathlib
import re
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.core.support_classes as sc
import sisepuede.transformers.transformers as trs
import sisepuede.utilities._toolbox as sf




_MODULE_UUID = "5FF5362F-3DE2-4A58-9CB8-01CB851D3CDC" 



###################################
#    START WITH TRANSFORMATION    #
###################################

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
        transformers: trs.Transformer,
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
        transformers: trs.Transformer,
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
        transformers: trs.Transformer,
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
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_transformation
            * self.uuid
        """

        self.is_transformation = True
        self.uuid = _MODULE_UUID

        return None
    


    def get_parameters_dict(self,
        config: sc.YAMLConfiguration,
        transformers: trs.Transformers,
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
        if not trs.is_transformer(transformer, ):
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






#######################################
#    COLLECTION OF TRANSFORMATIONS    #
#######################################

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

            The strategy definition table *must* include the following columns
        

    
    Initialization Arguments    
    ------------------------
    - dir_init: directory containing configuration files, 
    - transformers: Transformers object used to validate input parameters and 
        call function

    Optional Arguments
    ------------------
    - fn_citations: name of Bibtex file in dir_init containing optional 
        citations to provide
    - fn_strategy_definition: name of strategy definiton 
    """

    def __init__(self,
        dir_init: Union[str, pathlib.Path],
        transformers: trs.Transformers,
        fn_citations: str = "citations.bib",
        fn_strategy_definition: str = "stratgy_definitions.csv",
        regex_transformation_config: re.Pattern = re.compile("transformation_(.\D*).yaml"),
        **kwargs,
    ) -> None:
        
        self._initialize_transformations(
            dir_init,
            fn_strategy_definition,
            regex_transformation_config,
        )

        return None
    
    
    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
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



    def _initialize_transformations(self,
        dir_init: Union[str, pathlib.Path],
        fn_strategy_definition: str,
        regex_transformation_config: re.Pattern,
    ) -> None:
        """
        Initialize transformations provided in directory dir_init
        """

        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the following properties:
        
            * self.is_transformations
            * self.uuid
        """

        self.is_transformations = True
        self.uuid = _MODULE_UUID
        
        return None


    
    def get_files(self,
        dir_init: Union[str, pathlib.Path],
        fn_strategy_definition: str,
        regex_transformation_config: re.Pattern,
    ) -> Dict[str, List[str]]:
        """
        Retrieve transformation configuration files and the strategy definition
            files.
        """






########################
#    SOME FUNCTIONS    #
########################

def is_transformation(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Transformation
    """
    out = hasattr(obj, "is_transformation")
    uuid = getattr(obj, "uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out



def is_transformations(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Transformations
    """
    out = hasattr(obj, "is_transformations")
    uuid = getattr(obj, "uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out
        
