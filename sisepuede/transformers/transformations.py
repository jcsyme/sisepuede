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



_MODULE_CODE_SIGNATURE = "TX"
_MODULE_UUID = "5FF5362F-3DE2-4A58-9CB8-01CB851D3CDC" 
_TRANSFORMATION_REGEX_FLAG_PREPEND = "transformation"


###################################
#    START WITH TRANSFORMATION    #
###################################


# specify some default file names
_DICT_FILE_NAME_DEFAULTS = {
    "citations": "citations.bib",
    "config_general": "config_general.yaml"
}

# specify some default keys for configuration 
_DICT_KEYS = {
    "citations": "citations",
    "code": "transformation_code",
    "config_general": "config_general",
    "description": "description",
    "id": "transformation_id",
    "identifiers": "identifiers",
    "name": "transformation_name",
    "parameters": "parameters",
    "path": "path",
    "transformations": "transformations",
    "transformer": "transformer",
}







class Transformation:
    """Parameterization class for Transformer. Used to vary implementations of Transfomers. A Transformation reads parameters from a configuration file, an exiting YAMLConfiguration object, or an existing dictionary to allow users the ability to explore different magnitudes, timing, categories, or other parameterizations of a ``Transformer``.
    

    Parameters
    ----------
    config : Union[dict, str, sc.YAMLConfiguration]
        specification of configuration dictionary used to map parameters to Transformer. Can be:

        * dict: configuration dictionary
        * str: file path to configuration file to read
        * YAMLConfiguration: existing YAMLConfiguration

    transformers : trs.Transformer
        Transformers object used to validate input parameters and call function

    **kwargs:
        Optional keyword arguments, which can include the following elements
        
        * key_citations
        * key_description
        * key_identifiers
        * key_parameters
        * key_transformation_code
        * key_transformation_name
        * key_transformer
    
    Returns
    -------
    ``Transformation`` class
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

        self._initialize_uuid()
        
        return None
    


    def __call__(self,
        df_input: Union[pd.DataFrame, None] = None,
    ) -> pd.DataFrame:

        out = self.function(df_input = df_input, )
        
        return out




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
            if isinstance(config, dict) | isinstance(config, str) | isinstance(config, pathlib.Path)
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
            *args,
            df_input: Union[pd.DataFrame, None] = None,
            strat: Union[int, None] = None,
        ):
            out = transformer.function(
                *args, 
                df_input = df_input,
                strat = strat,
                **self.dict_parameters, 
            )

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
            
            * self.citations
            * self.code
            * self.description
            * self.id_num
            * self.name
        """

        citations = self.config.get(self.key_citations)
        code = self.config.get(self.key_yc_trasformation_code)
        description = self.config.get(self.key_description)
        id_num = None # initialize as None, can overwrite later
        name = self.config.get(self.key_yc_trasformation_name)

        
        ##  SET PROPERTIES

        self.citations = citations
        self.code = code
        self.description = description
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

        key_identifiers = kwargs.get("key_identifiers", _DICT_KEYS.get("identifiers"))
        key_transformation_code = kwargs.get("key_transformation_code", _DICT_KEYS.get("code"))
        key_transformation_name = kwargs.get("key_transformation_name", _DICT_KEYS.get("name"))

        key_yc_trasformation_code = f"{key_identifiers}.{key_transformation_code}"
        key_yc_trasformation_name = f"{key_identifiers}.{key_transformation_name}"


        ##  SET PARAMETERS

        self.key_citations = kwargs.get("key_citations", _DICT_KEYS.get("citations"))
        self.key_description = kwargs.get("key_description", _DICT_KEYS.get("description"))
        self.key_identifiers = key_identifiers
        self.key_parameters = kwargs.get("key_parameters", _DICT_KEYS.get("parameters"))
        self.key_transformation_code = key_transformation_code
        self.key_transformation_id = _DICT_KEYS.get("id")
        self.key_transformation_name = key_transformation_name
        self.key_transformer = kwargs.get("key_transformer", _DICT_KEYS.get("transformer"))
        self.key_yc_trasformation_code = key_yc_trasformation_code
        self.key_yc_trasformation_name = key_yc_trasformation_name

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_transformation
            * self._uuid
        """

        self.is_transformation = True
        self._uuid = _MODULE_UUID

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
            raise RuntimeError(f"Invalid transformation '{transformer_code}' found in Transformers")

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
    """Build a collection of parameters used to construct transformations. The 
        ``Transformations`` class searches a specified directory to ingest three 
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
        
            

    
    Parameters    
    ----------
    dir_init: Union[str, pathlib.Path]
        Directory containing configuration files
    baseline_id : int
        id used for baseline, or transformation that is applied to raw data. All other transformations in the attribute table are increased from this id.
    fn_citations : str
        file name of Bibtex file in dir_init containing optional citations to provide
    fn_config_general : str
        file name of the general configuration file in dir_init
    logger : Union[logging.Logger, None]
        Optional logger object to pass
    regex_transformation_config : re.Pattern
        regular expression used to match transformation configuration files
    stop_on_error : bool
        throw an error if a transformation fails? Otherwise, will skip transformation configuration files that fail. 
    transformers : Union[trs.Transformers, None]
        optional existing Transformers object. If None is available, initializes one.

        NOTE: If a transformers object is NOT specified (i.e., if transformers is None), then you must include the following keywords to generate dataframes of inputs. 

            * `df_input`: the input dataframe of base SISEPUEDE inputs
        
        Additionally, "field_region" can be included if the region field differs from `model_attributes.dim_region`
    """

    def __init__(self,
        dir_init: Union[str, pathlib.Path],
        baseline_id: int = 0,
        fn_citations: str = _DICT_FILE_NAME_DEFAULTS.get("citations"),
        fn_config_general: str = _DICT_FILE_NAME_DEFAULTS.get("config_general"),
        logger: Union[logging.Logger, None] = None,
        regex_transformation_config: re.Pattern = re.compile(f"{_TRANSFORMATION_REGEX_FLAG_PREPEND}_(.\w*).yaml"),
        stop_on_error: bool = True,
        transformers: Union[trs.Transformers, None] = None,
        **kwargs,
    ) -> None:
        
        self.logger = logger

        self._initialize_keys(
            **kwargs,
        )
        self._initialize_config(
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
            baseline_id = baseline_id,
            **kwargs,
        )
        self._initialize_uuid()

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
        """Initialize citations. Sets the following properties:
        """

        # get the file path
        fp_citations = self.dict_paths.get(
            self.key_path_citations,
        )

        warnings.warn(f"NOTE: citations mechanism in Transformations needs to be set. See _initialize_citations()")

        return None



    def _initialize_config(self,
        dir_init: Union[str, pathlib.Path],
        fn_citations: str,
        fn_config_general: str,
        regex_transformation_config: re.Pattern,
    ) -> None:
        """Initialize the general configuration file and the dictionary of file
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
            str(dict_paths.get(self.key_path_config_general, ))
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
        """Set the optional and required keys used to specify a transformation.
            Can use keyword arguments to set keys.
        """

        # set some shortcut codes 



        ##  SET PARAMETERS

        self.key_trconfig_description = _DICT_KEYS.get("description")
        self.key_path = _DICT_KEYS.get("path")
        self.key_path_citations = _DICT_KEYS.get("citations")
        self.key_path_config_general = _DICT_KEYS.get("config_general")
        self.key_path_transformations = _DICT_KEYS.get("transformations")

        return None



    def _initialize_transformations(self,
        baseline_id: int = 0,
        default_nm_prepend: str = "Transformation",
        stop_on_error: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the transformer used to build transformations. 

        Keyword Arguments
        ------------------
        baseline_id : int
            id used for baseline, or transformation that is applied to raw data. 
            All other transformations in the attribute table are increased from 
            this id.
        default_nm_prepend : str
            String prepended to the id to generate names for transformations 
            that have invalid names specified
        stop_on_error : bool
            Stop if a transformation fails? If False, logs and skips the failed 
            transformation
        """

        ## INIT

        # get files to iterate over
        files_transformation_build = self.dict_paths.get(
            self.key_path_transformations,
        )

        # initialize dictionary of transformations 
        # - dictionary mapping transformation codes to transformations
        # - get baseline transformation first

        transformation_baseline = self.get_transformation_baseline()

        dict_all_transformations = {
            transformation_baseline.code: transformation_baseline,
        }
        dict_transformation_code_to_fp = {
            transformation_baseline.code: None,
        }

        # iterate over available files 
        for fp in files_transformation_build:

            try:
                
                # try building the transformation and verify the code
                transformation = Transformation(
                    fp,
                    self.transformers,
                )

                if transformation.code in dict_all_transformations.keys():
                    fp_existing = (
                        dict_transformation_code_to_fp
                        .get(transformation.code)
                    )
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
            dict_transformation_code_to_fp.update({transformation.code: str(fp)})
        

        # build the attribute table
        attribute_transformation, dict_fields = self.build_attribute_table(
            transformation_baseline.code,
            dict_all_transformations,
            dict_transformation_code_to_fp,
            baseline_id = baseline_id,
            default_nm_prepend = default_nm_prepend,
        )

        all_transformation_codes = attribute_transformation.key_values
        

        ##  SET PROPERTIES

        self.attribute_transformation = attribute_transformation
        self.all_transformation_codes = all_transformation_codes
        self.code_baseline = transformation_baseline.code
        self.dict_transformations = dict_all_transformations
        self.field_attr_code = dict_fields.get("field_code")
        self.field_attr_citation = dict_fields.get("field_citation")
        self.field_attr_description = dict_fields.get("field_desc")
        self.field_attr_id = dict_fields.get("field_id")
        self.field_attr_name = dict_fields.get("field_name")
        self.field_attr_path = dict_fields.get("field_path")

        return None
        


    def _initialize_transformers(self,
        transformers: Union[trs.Transformers, None] = None,
        **kwargs,
    ) -> None:
        """Initialize the transformer used to build transformations.     
        """

        # check inputs
        if not trs.is_transformers(transformers):
            transformers = trs.Transformers(
                self.config.dict_yaml,
                attr_time_period = kwargs.get("attr_time_period"),
                df_input = kwargs.get("df_input"),
                field_region = kwargs.get("field_region"),
                #logger = self.logger,
            )
        
        else:
            # update the configuration, ramp, and then update the data
            transformers._initialize_config(
                self.config.dict_yaml,
                transformers.code_baseline,
            )

            transformers._initialize_ramp()

            transformers._initialize_baseline_inputs(
                transformers.inputs_raw,
            )


        ##  SET PROPERTIES

        self.transformers = transformers

        return None



    def _initialize_uuid(self,
    ) -> None:
        """Initialize the following properties:
        
            * self.is_transformations
            * self._uuid
        """

        self.is_transformations = True
        self._uuid = _MODULE_UUID
        
        return None
    


    def build_attribute_table(self,
        baseline_code: str,
        dict_all_transformations: Dict[str, Transformation],
        dict_transformation_code_to_fp: Dict[str, pathlib.Path],
        baseline_id: int = 0,
        baseline_in_dict: bool = True,
        default_nm_prepend: str = "Transformation",
    ) -> Union[AttributeTable, dict]:
        """Build the transformation attribute table. Returns the attribute table
            plus a dictionary of field names.
        """

        ##  NEXT, BUILD THE ATTRIBUTE TABLEs

        # sort by default by code
        baseline_id = (
            max(baseline_id, 0) 
            if sf.isnumber(baseline_id, integer = True)
            else 0
        )
        id_def = baseline_id if baseline_in_dict else baseline_id + 1#

        # sort to ensure that baseline is first
        all_transformation_codes_base = sorted(dict_all_transformations.keys())
        all_transformation_codes = [baseline_code] 
        all_transformation_codes += [
            x for x in all_transformation_codes_base 
            if x != all_transformation_codes[0]
        ]


        nms_defined = []

        # initialize dictionaries and information for attribute table
        dict_fields = {}
        dict_table = {
            self.key_path: [],
        }

        # fields will be updated iteratively
        key_code = None
        field_citations = None
        field_desc = None
        field_id = None
        field_name = None
        field_path = None
        

        # iterate over codes to assign 
        for code in all_transformation_codes:
            
            # get transformation
            transformation = dict_all_transformations.get(code)


            ##  CODE (ATTRIBUTE KEY)

            if key_code is None:
                key_code = transformation.key_transformation_code
                dict_table.update({key_code: []})

            dict_fields.update({"field_code": key_code})
            dict_table.get(key_code).append(code) 


            ##  ID
            
            if field_id is None:
                field_id = transformation.key_transformation_id
                dict_table.update({field_id: []})
            
            dict_fields.update({"field_id": field_id})
            dict_table.get(field_id).append(id_def)
            transformation.id_num = id_def


            ##  NAME

            # check name
            name = transformation.name
            if name in nms_defined:
                name_new = f"{default_nm_prepend} {id_def}"
                self._log(
                    f"Name '{name}' for transformation code '{code}' already taken: assigning name {name_new}",
                    type_log = "warning",
                )

                name = name_new
                

            # update name
            nms_defined.append(name)
            transformation.name = name

            if field_name is None:
                field_name = transformation.key_transformation_name
                dict_table.update({field_name: []})

            dict_fields.update({"field_name": field_name})
            dict_table.get(field_name).append(name)


            ##  DESCRIPTION 

            desc = transformation.config.get(transformation.key_description)

            if field_desc is None:
                field_desc = transformation.key_description
                dict_table.update({field_desc: []})

            dict_fields.update({"field_desc": field_desc})
            dict_table.get(field_desc).append(desc)
            

            ##  CITATIONS

            citations = transformation.config.get(transformation.key_citations)
            citations = (
                "|".join(citations)
                if isinstance(citations, list)
                else str(citations)
            )

            if field_citations is None:
                field_citations = transformation.key_citations
                dict_table.update({field_citations: []})

            dict_fields.update({"field_citations": field_citations})
            dict_table.get(field_citations).append(citations)
            

            ##  FILE PATH

            fp = str(dict_transformation_code_to_fp.get(code))

            if field_path is None:
                field_path = self.key_path
                dict_table.update({field_path: []})

            dict_fields.update({"field_path": field_path})
            dict_table.get(field_path).append(fp)


            # next iterationm
            id_def += 1



        # build the table
        attribute_return = pd.DataFrame(dict_table)
        attribute_return = AttributeTable(
            attribute_return[
                [
                    field_id,
                    key_code,
                    field_name,
                    field_desc,
                    field_citations,
                    field_path
                ]
            ],
            key_code,
        )

        out = (attribute_return, dict_fields)

        return out
    

    
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
        if not path_init.exists():
            msg = f"Unable to initialize Transformations: path '{path_init}' does not exist."
            raise RuntimeError(msg)


        ##  CHECK REQUIRED FILES

        # initialize output dictionary
        dict_out = {}

        # check config
        path_config_general = path_init.joinpath(fn_config_general)
        if not path_config_general.exists():
            msg = f"""General configuration file '{fn_config_general}' not found 
            in path '{path_init}'. Cannot proceed. To use default configuration, 
            create a blank file in the directory and name it 
            '{fn_config_general}'.
            """
            raise RuntimeError(msg)

        dict_out.update({self.key_path_config_general: path_config_general})


        # check citation (optional, so no error message)
        path_citations = path_init.joinpath(fn_citations)
        if not path_citations.exists():
            path_citations = None

        dict_out.update({self.key_path_citations: path_citations})


        # look for transformation files - iterate over all files, then build a list of paths
        fps_transformation = [
            path_init.joinpath(x) for x in os.listdir(path_init)
            if regex_transformation_config.match(x) is not None
        ]

        if len(fps_transformation) == 0:
            warnings.warn(f"No valid Transformation configuration files were found in '{path_init}'.")
        
        dict_out.update({self.key_path_transformations: fps_transformation})

        
        # return

        return dict_out
    


    def get_transformation_baseline(self,
    ) -> None:
        """
        Build the baseline Transformation
        """

        # get codes
        key_code = _DICT_KEYS.get("code")
        key_id = _DICT_KEYS.get("identifiers")
        key_name = _DICT_KEYS.get("name")

        # set the default to be the transformers base with the new Transformations signature
        code_def = (
            self
            .transformers
            .code_baseline
            .replace(
                trs._MODULE_CODE_SIGNATURE,
                _MODULE_CODE_SIGNATURE,
            )
        )
        code = f"{self.transformers.key_config_baseline}.{key_id}.{key_code}"
        code = self.config.get(
            code, 
            return_on_none = code_def, 
        )

        name = f"{self.transformers.key_config_baseline}.{key_id}.{key_name}"
        name = self.config.get(name)


        ##  BUILD TRANSFORMATION

        dict_tr = {
            key_id: {
                key_code: code,
                key_name: name,
            },
            "parameters": {},
            "transformer": self.transformers.code_baseline,
        }

        trfmn = Transformation(
            dict_tr,
            self.transformers
        )


        return trfmn
    


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
            use to retrieve Trasnformation object
            
        Keyword Arguments
        ------------------
        - return_code: set to True to return the transformer code only
        """

        # skip these types
        is_int = sf.isnumber(transformation, integer = True)
        return_none = not is_int
        return_none &= not isinstance(transformation, str)
        if return_none:
            return None

        # Transformer objects are tied to the attribute table, so these field maps work
        dict_id_to_code = self.attribute_transformation.field_maps.get(
            f"{self.field_attr_id}_to_{self.attribute_transformation.key}"
        )
        dict_name_to_code = self.attribute_transformation.field_maps.get(
            f"{self.field_attr_name}_to_{self.attribute_transformation.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(transformation, str):
            code = (
                transformation
                if transformation in self.attribute_transformation.key_values
                else dict_name_to_code.get(transformation)
            )
        
        elif is_int:
            code = dict_id_to_code.get(transformation)

        # check returns
        if code is None:
            return None

        if return_code:
            return code


        out = self.dict_transformations.get(code)
        
        return out
    


    def get_transformations_variable_fields(self,
        error_thresh: float = 10**(-3), 
        field_sample_group: str = "sample_group",
        field_transformation_code: str = "transformation_code",
        field_variable: str = "variable",
        field_variable_field: str = "variable_field",
        include_all_variable_fields_by_modvar: bool = False, 
        include_sample_group: bool = True,
        transformations_include: Union[List[Transformation], None] = None,
    ) -> pd.DataFrame:
        """Map transformer code to variable fields and variable that are 
            modified by the transformer. Creates a DataFrame map with the 
            following columns:

            field_sample_group:         Optional field storing a sample group.
                                        Sample groups are defined by all 
                                        variable fields that share transformer 
                                        codes. In general, a variable is only 
                                        affected by one transformer, but there 
                                        are some cases of overlap. 
            field_transformation_code:  Field storing the transformation code.
            field_variable:             Field storing the SISEPUEDE variable 
                                        name associated with the variable field 
                                        that responds to the transformer
            field_variable_field:       Field storing the variable field that 
                                        responds to the transformer specified in 
                                        field_transformer_code.

        Function Arguments
        ------------------


        Keyword Arguments
        -----------------
        error_thresh : float
            Threshold for determining equality; values with a normalized error 
            less than this (i.e., eps < |1 - x_0/x_i| for x_0 baseline and x_i 
            transformed) will be considered equal
        field_sample_group : str
            Field storing the sample group
        field_transformation_code : str
            Field storing the transformer code
        field_variable : str
            Field name for SISEPUEDE variable
        field_variable_field : str
            Field name for SISEPUEDE variable field
        include_all_variable_fields_by_modvar : bool
            * true:     include all the variable fields associated with model 
                        variables
            * false:    include onlny the variable fields that vary
        include_sample_group : bool
            Include the sample group in the specification?
        transformations_include : Union[List[Transformation], None]
            Optional list of transformations to include. Useful when setting an
            experiment that varies transformations within a strategy to preserve
            fields in a transformation in a single sample group.
        """

        # initialize the output dataframe
        df_out = []
        
        # baseline to which others are compared
        df_base = self.get_transformation_baseline()
        df_base = df_base()

        matt = self.transformers.model_attributes
        fields_compare = matt.all_variable_fields_input

        # get codes
        transformations_iterate = (
            [self.get_transformer(x) for x in self.all_transformation_codes]
            if not sf.islistlike(transformations_include)
            else transformations_include
        )

        # iterate through codes
        for transformation in transformations_iterate:
            # skip baseline
            if transformation.code == self.code_baseline: continue

            # get current transformer and run
            df_cur = transformation()

            vec_dist = (
                np.abs(
                    1 - np.nan_to_num(
                        df_cur[fields_compare]/df_base[fields_compare],
                        nan = 1.0,
                        posinf = 0.0,
                    )
                )
                .max(axis = 0, )
            )
            
            w = np.where(vec_dist >= error_thresh)[0]

            # nothing add
            if len(w) == 0: continue
            

            # otherwise, add to rows
            fields_change = [fields_compare[x] for x in w]
            modvars = [
                matt.get_variable(
                    matt.dict_variable_fields_to_model_variables.get(x)
                )
                for x in fields_change
            ]
            modvar_names = [x.name for x in modvars]

            # if not including all fields by modvar, build the DataFrame and move on
            if not include_all_variable_fields_by_modvar: 
                df_out_cur = pd.DataFrame(
                    {
                        field_transformation_code: [transformation.code for x in range(len(fields_change))],
                        field_variable: modvar_names,
                        field_variable_field: fields_change
                    }
                )
                df_out.append(df_out_cur, )

                continue
            
            # otherwise, use the available names to build a dataframe
            mvs = sorted(list(set(modvar_names)))
            
            for mv in mvs:
                mv = matt.get_variable(mv)
                df_out_cur = pd.DataFrame(
                    {
                        field_transformation_code: [transformation.code for x in mv.fields],
                        field_variable: [mv.name for x in mv.fields],
                        field_variable_field: mv.fields
                    }
                )

                df_out.append(df_out_cur, )

        # create output dataframe
        df_out = sf._concat_df(df_out, )
        
        
        ##  INCLUDE THE SAMPLE GROUP?

        if not include_sample_group:
            return df_out
        
        # initialize
        sample_group = 1
        df_out_with_group = []
        
        while df_out.shape[0] > 0:
            df_in_group, df_out = trs.extract_variable_field_group(
                df_out, 
                field_code = field_transformation_code,
                field_sample_group = field_sample_group,
                sample_group = sample_group,
            )

            df_out_with_group.append(df_in_group, )
            sample_group += 1

        df_out_with_group = sf._concat_df(df_out_with_group, )

        return df_out_with_group
    


    def get_transformation_codes_by_transformer_code(self,
        include_missing_transformers: bool = False,
    ) -> dict: 
        """
        Build a dictionary of all transformation codes associated with available
            transformer codes. Set `include_missing_transformers = True` to 
            include transformers that are not associated with any 
            Transformations.
        """
        
        dict_out = dict(
            (x, self.get_transformation(x).transformer_code)
            for x in self.all_transformation_codes
        )
        
        dict_out = sf.reverse_dict(
            dict_out,
            allow_multi_keys = True,
            force_list_values = True,
        )

        # 
        if include_missing_transformers:
            dict_update = dict(
                (x, []) for x in self.transformers.all_transformers
                if x not in dict_out.keys()
            )

            dict_out.update(dict_update)
        
        return dict_out
    


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
    uuid = getattr(obj, "_uuid", None)

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
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out
        
