import logging
import numpy as np
import pandas as pd
import pathlib
import re
import warnings
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
        id_num = self.config.get(self.key_yc_trasformation_id)
        name = self.config.get(self.key_yc_trasformation_name)

        
        ##  SET PROPERTIES

        self.code = code
        self.id_num = id_num
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
        key_transformation_id = kwargs.get("key_transformation_code", "transformation_id")
        key_transformation_name = kwargs.get("key_transformation_name", "transformation_name")

        key_yc_trasformation_code = f"{key_identifiers}.{key_transformation_code}"
        key_yc_trasformation_id = f"{key_identifiers}.{key_transformation_id}"
        key_yc_trasformation_name = f"{key_identifiers}.{key_transformation_name}"


        ##  SET PARAMETERS

        self.key_citations = kwargs.get("key_citations", "citations")
        self.key_description = kwargs.get("key_description", "description")
        self.key_identifiers = key_identifiers
        self.key_parameters = kwargs.get("key_parameters", "parameters")
        self.key_transformation_code = key_transformation_code
        self.key_transformation_id = key_transformation_id
        self.key_transformation_name = key_transformation_name
        self.key_transformer = kwargs.get("key_transformer", "transformer")
        self.key_yc_trasformation_code = key_yc_trasformation_code
        self.key_yc_trasformation_id = key_yc_trasformation_id
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
        Transformations class searches a specified directory to ingest three 
        required file types and a fourth optional type:

        (1) General configuration file (by default `config_general.yaml`). This
            file is used to specify some general parameters used across 
            transformations, including categories that are subject to high heat.
            Additionally, this configuration is used to specify information
            about the baseline, including whether or not to include a non-zero
            Land Use Reallocation Factor (LURF).

            To revert to defaults, leave this file empty.

        (2) Transformation configuration files, which define transformations as 
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


            (3) A Bibtex citation file. This citation file can be used to 
                supplement default citations found in the SISEPUEDE 
                transformations. SISEPUEDE default data citations are stored in 
                the SISEPUEDE Data Pipeline repository (HEREHERE LINK).
        
            

    
    Initialization Arguments    
    ------------------------
    - dir_init: directory containing configuration files, 
    - transformers: Transformers object used to validate input parameters and 
        call function

    Optional Arguments
    ------------------
    - fn_citations: file name of Bibtex file in dir_init containing optional 
        citations to provide
    - fn_config_general: file name of the general configuration file in dir_init
    - regex_transformation_config: regular expression used to match 
        transformation configuration files
    - stop_on_error: throw an error if a transformation fails? Otherwise, will
        skip transformation configuration files that fail. 
    - transformers: optional existing Transformers object. If None is available,
        initializes one;
        NOTE: If a transformers object is NOT specified, then you must include
            the following keywords to generate dataframes of inputs. 

            * df_input: the input dataframe of base SISEPUEDE inputs
        
        Additionally, "field_region" can be included if the region field differs
        from `model_attributes.dim_region`
    """

    def __init__(self,
        dir_init: Union[str, pathlib.Path],
        fn_citations: str = "citations.bib",
        fn_config_general: str = "config_general.yaml",
        logger: Union[logging.Logger, None] = None,
        regex_transformation_config: re.Pattern = re.compile("transformation_(.\D*).yaml"),
        stop_on_error: bool = True,
        transformers: Union[trs.Transformers, None] = None,
        **kwargs,
    ) -> None:
        
        self.logger = logger

        self._initialize_keys(
            **kwargs,
        )
        self.initialize_config(
            dir_init,
            fn_citations,
            fn_config_general,
            regex_transformation_config,
        )
        self._initialize_citations()

        # initialize transformation components
        self._initialize_transformers(
            transformers,
            **kwargs,
        )

        self._initialize_transformations(
            **kwargs,
        )

        return None
    


    def __call__(self,
        tranformation_name: str,
        **kwargs
    ) -> pd.DataFrame:

        transformation = self.get_transformation(tranformation_name, )
        out = transformation(**kwargs, )

        return out


    

    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_citations(self,
    ) -> None:
        """
        Initialize citations. Sets the following properties:

        """

        # get the file path
        fp_citations = self.dict_paths.get(
            self.key_path_citations,
        )

        warngins.warn(f"NOTE: citations mechanism in Transformations needs to be set. See _initialize_citations()")

        return None



    def _initialize_config(self,
        dir_init: Union[str, pathlib.Path],
        fn_citations: str,
        fn_config_general: str,
        regex_transformation_config: re.Pattern,
    ) -> None:
        """
        Initialize the general configuration file and the dictionary of file
            paths. Sets the following properties:

            * self.config
            * self.dict_paths
            * self.dir_init
        """

        # get the files
        dict_paths = self.get_files(
            dir_init,
            fn_citations,
            fn_config_general,
            regex_transformation_config,
        )

        # read configuration
        config = sc.YAMLConfiguration(
            dict_paths.get(self.key_path_config_general, )
        )

        dir_init = pathlib.Path(dir_init)


        ##  SET PARAMETERS

        self.config = config
        self.dict_paths = dict_paths
        self.dir_init = dir_init

        return None
    


    def _initialize_keys(self,
        **kwargs,
    ) -> None:
        """
        Set the optional and required keys used to specify a transformation.
            Can use keyword arguments to set keys.
        """

        # set some shortcut codes 



        ##  SET PARAMETERS
        """
        self.key_citations = kwargs.get("key_citations", "citations")
        self.key_description = kwargs.get("key_description", "description")
        self.key_identifiers = key_identifiers
        self.key_parameters = kwargs.get("key_parameters", "parameters")
        self.key_transformation_code = key_transformation_code
        self.key_transformation_name = key_transformation_name
        self.key_transformer = kwargs.get("key_transformer", "transformer")
        self.key_yc_trasformation_code = key_yc_trasformation_code
        self.key_yc_trasformation_name = key_yc_trasformation_name
        """

        self.key_trconfig_description = "description"
        self.key_path_citations = "citations"
        self.key_path_config_general = "config_general"
        self.key_path_transformations = "transformations"

        return None
    

    
    def _initialize_transformations(self,
        stop_on_error: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the transformer used to build transformations. 
        """

        ## INIT

        # get files to iterate over
        files_transformation_build = self.dict_paths.get(
            self.key_path_transformations,
        )

        # initialize dictionary of transformations 
        # - dictionary mapping transformation codes to transformations
        dict_all_transformations = {}
        dict_transformation_code_to_attributes = {}


        # iterate over available files 
        for fp in files_transformation_build:

            try:

                # try building the transformation and verify the code
                transformation = trn.Transformation(
                    fp,
                    transformers,
                )

                if transformation.code in all_transformations.keys():
                    fp_existing = dict_transformation_code_to_fp.get(transformation.code)
                    raise KeyError(f"Transformation code {transformation.code} already specified in file '{fp_existing}'.")

            except Exception as e:
                msg = f"Transformation configuration file at path '{fp}' failed: {e}"
                if stop_on_error:
                    raise RuntimeError(msg)

                # warn and skip if there's a problem and we're not stopping
                self._log(msg, type_log = "warning", )
                continue
            
            # otherwise, update dictionaries
            dict_all_transformations.update({transformation.code: transformation})
            dict_transformation_code_to_fp.update({transformation.code: {"path": fp}})
        
        
        ##  NEXT, CLEAN THE ATTRIBUTES BY ASSIGNING AN ID AND NAME

        # sort by default by code
        id_def = 1
        all_transformation_codes = sorted(dict_all_transformations.keys())
        ids_defined = [
            x.id_num for x in dict_all_transformations.values() 
            if self.validate_idnum(x)
        ]
        nms_defined = [
            x.name for x in dict_all_transformations.values()
            if x is not None
        ]

        # iterate over codes to check set of ids 
        for code in all_transformation_codes:

            transformation = dict_all_transformations.get(code)
            id_num = transformation.id_num
            name = transformation.name

            if id_num is None:
                id_num = (max(ids_defined) + 1) if (len(ids_defined) > 0) else id_def
                # here, add the ID number

            continue

        dict_transformation_code_to_fp.update({transformation.code: dict_attributes})

        
        ##  SET PROPERTIES

        self.all_transformation_codes = all_transformation_codes
        self.dict_all_transformations = dict_all_transformations
        self.dict_transformation_code_to_attributes = dict_transformation_code_to_attributes

        return None
        


    def _initialize_transformers(self,
        transformers: Union[trs.Transformers, None] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the transformer used to build transformations. 
        """

        # check inputs
        if not trs.is_transformers(transformers):
            transformers = trs.Transformers(
                self.config.dict_yaml,
                df_input = kwargs.get("df_input"),
                field_region = kwargs.get("field_region"),
                #logger = self.logger,
            )
        
        else:
            # update the configuration 
            transformers._initialize_config(
                self.config.dict_yaml,
            )


        ##  SET PROPERTIES

        self.transformers = transformers

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
        fn_citations: str,
        fn_config_general: str,
        regex_transformation_config: re.Pattern,
    ) -> Dict:
        """
        Retrieve transformation configuration files and the general 
            configuration file.
        """
        
        ##  VERITY PATHS

        # try path
        try:
            path_init = pathlib.Path(dir_init)
        
        except Exception as e:
            msg = f"Unable to set path from dir_init = '{dir_init}'. Check the specification of dir_init in Transformations."
            raise RuntimeError(msg)

        # check that it exists
        if not path_init.exist():
            msg = f"Unable to initialize Transformations: path '{path_init}' does not exist."
            raise RuntimeError(msg)


        ##  CHECK REQUIRED FILES

        # initialize output dictionary
        dict_out = {}

        # check config
        path_config_general = path_init.join(fn_config_general)
        if not path_config_general.exists():
            msg = f"""General configuration file '{fn_config_general}' not found 
            in path '{path_init}'. Cannot proceed. To use default configuration, 
            create a blank file in the directory and name it 
            '{fn_config_general}'.
            """
            raise RuntimeError(msg)

        dict_out.update({self.key_path_config_general: path_config_general})


        # check citation (optional, so no error message)
        path_citations = path_init.join(fn_citations)
        if not path_citations.exists():
            path_citations = None

        dict_out.update({self.key_path_citations: path_citations})


        # look for transformation files - iterate over all files, then build a list of paths
        fps_transformation = [
            path_init.join(x) for x in os.listdir(path_init)
            if regex_transformation_config.match(x) is not None
        ]

        if len(fps_transformation) == 0:
            warnings.warn(f"No valid Transformation configuration files were found in '{path_init}'.")
        
        dict_out.update({self.key_path_transformations: fps_transformation})

        
        # return

        return dict_out
    


    def validate_idnum(self,
        id_num: Any,
    ) -> bool:
        """
        Check if an id_num is valid
        """

        if not sf.isnumber(id_num, integer = True):
            return False
        
        out = (id_num > 0)

        return out
    


    ########################
    #    CORE FUNCTIONS    #
    ########################

    def get_transformation(self,
        transformation: Union[int, str, None],
        return_code: bool = False,
    ) -> None:
        """
        Get `transformer` based on transformer code, id, or name
            
        Function Arguments
        ------------------
        - transformer: transformer_id, transformer name, or transformer code to 
            use to retrieve sc.Trasnformation object
            
        Keyword Arguments
        ------------------
        - field_transformer_code: field in transformer attribute table 
            containing the transformer code
        - field_transformer_name: field in transformer attribute table 
            containing the transformer name
        - return_code: set to True to return the transformer code only
        """

        # skip these types
        is_int = sf.isnumber(transformer, integer = True)
        return_none = not is_int
        return_none &= not isinstance(transformer, str)
        if return_none:
            return None

        # Transformer objects are tied to the attribute table, so these field maps work
        dict_id_to_code = self.attribute_transformer_code.field_maps.get(
            f"{field_transformer_id}_to_{self.attribute_transformer_code.key}"
        )
        dict_name_to_code = self.attribute_transformer_code.field_maps.get(
            f"{field_transformer_name}_to_{self.attribute_transformer_code.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(transformer, str):
            code = (
                transformer
                if transformer in self.attribute_transformer_code.key_values
                else dict_name_to_code.get(transformer)
            )
        
        elif is_int:
            code = dict_id_to_code.get(transformer)

        # check returns
        if code is None:
            return None

        if return_code:
            return code


        out = self.dict_transformers.get(code)
        
        return out

    


    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See ?sf._optional_log for more information

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)






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
        
