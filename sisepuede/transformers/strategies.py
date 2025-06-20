import logging
import numpy as np
import pandas as pd
import re
import shutil
import time
from typing import *

from copy import deepcopy
from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.core.support_classes as sc
import sisepuede.data_management.ingestion as ing
import sisepuede.transformers.transformations as trn
import sisepuede.utilities._toolbox as sf




_MODULE_UUID = "BEE85F6C-6BA3-4382-9A8C-5C2D8691DE4E" 



# specify some default file names
_DICT_FILE_NAME_DEFAULTS = {
    "strategy_definitions": "strategy_definitions.csv",
}

_DICT_KEYS = {
    "baseline": "baseline_strategy_id",
    "code": "strategy_code",
    "description": "description",
    "name": "strategy",
    "transformation_specification": "transformation_specification"
}

_FLAG_EXPORT_TO_TRANSFORMATIONS = "transformations"


#####################################
###                               ###
###    BEGIN CLASS DEFINITIONS    ###
###                               ###
#####################################

class Strategy:
    """
    A collection of transformations. The Strategy code 

                
    Initialization Arguments
    ------------------------
    - strategy_id: id number for the strategy
    - transformation_codes: string denoting the trasformation code (or ids) to 
        use to implement the strategy. Transformations are combined as a 
        `delim`-delimited string or list of strings (which can be delimited) or 
        integers.

        E.g.,

        * "TX:AGRC:TEST_1|TX:ENTC:TEST_2|TX:LSMM:MANURE_MANAGEMENT|51"

        * ["TX:AGRC:TEST_1", "TX:ENTC:TEST_2", "TX:LSMM:MANURE_MANAGEMENT", 51]

        * ["TX:AGRC:TEST_1"|"TX:ENTC:TEST_2", TX:LSMM:MANURE_MANAGEMENT", 51]

        are all valid inputs.

    Optional Arguments
    ------------------
    - delim: delimiter used to split codes
    - dict_attributes: optional dictionary of attributes to assign to the
        Strategy. 
    - prebuild: prebuild the data frame? If True, will call the function and 
        store it in the `table` property
    """

    def __init__(self,
        strategy_id: int,
        transformation_codes: Union[str, List[Union[str, int]]],
        transformations: trn.Transformations,
        delim: str = "|",
        dict_attributes: Union[Dict[str, Any], None] = None,
        prebuild: bool = True,
    ) -> None:

        self._initialize_identifiers(
            strategy_id,
            transformations,
            dict_attributes = dict_attributes,
        )

        self._initialize_function(
            transformation_codes,
            transformations,
            delim = delim,
        )

        self._initialize_table(
            prebuild,
        )

        self._initialize_uuid()
        
        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> pd.DataFrame:

        if (self.table is not None) and ("df_input" not in kwargs.keys()):
            out = self.table

        else:
            out = self.function(*args, **kwargs)
            if "df_input" not in kwargs.keys():
                self.table = out # update the table if running with defaults

        return out





    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_function(self,
        transformation_codes: Union[str, List[Union[str, int]]],
        transformations: trn.Transformations,
        delim: str = "|",
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.delimiter_transformation_codes
            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None

        ##  initialize transformations 

        func = self.get_transformation_list(
            transformation_codes,
            transformations,
            delim = delim,
        )


        ##  INITIALIZE AS A LIST 

        func = [x.function for x in func]

        if len(func) > 0:  
            
            # define a dummy function and assign
            def function_out(
                **kwargs
            ) -> Any:
                f"""
                Composite Transformer function for {self.name}
                """
                # out = None
                out = kwargs.get("df_input")
                for f in func:
                    out = f(
                        df_input = out, 
                        strat = self.id_num, 
                    )

                return out

            function = function_out
            function_list = func
        
        else:
            
            def function_out(
                **kwargs
            ) -> pd.DataFrame:

                out = transformations.get_transformation(
                    transformations.code_baseline,
                )

                out = out.function(
                    df_input = kwargs.get("df_input"),
                    strat = self.id_num,
                )
                
                return out

            
            function = function_out
            function_list = [function_out]




        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.delimiter_transformation_codes = delim
        self.function = function
        self.function_list = function_list
        
        return None
    


    def _initialize_identifiers(self,
        strategy_id: int,
        transformations: trn.Transformations,
        dict_attributes: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """
        Set transformation code and, optionally, transformation name. Sets the
            following properties:

            self.attributes: 
                dictionary containing optional attributes
            self.code: 
                optional strategy code
            self.id_num 
                strategy id number, base of all SISEPUEDE indexing
            self.name   
                strategy name 
            self.key_strategy
                strategy key (used in attribute table)
            self.field_description = field_description
            self.field_strategy_code = field_strategy_code
            self.field_strategy_name

        
        Function Arguments
        ------------------


        Keyword Arguments
        -----------------
        - **kwargs: can be used to pass

            * field_description: optional decription to pass
            * field_strategy_code: optional code to pass for the strategy
            * field_strategy_name: name of the strategy
        """
        # verify that the id is correctly input 
        if not sf.isnumber(strategy_id, integer = True, ):
            tp = str(type(strategy_id))
            msg = f"Invalid type '{tp}' specified for strategy_id in Strategy: must be integer."
            raise ValueError(msg)
            
        key_strategy = (
            transformations
            .transformers
            .model_attributes
            .dim_strategy_id
        )

        # set some fields
        field_description = "description"
        field_strategy_code = kwargs.get(
            "field_strategy_code",
            _DICT_KEYS.get("code")
        )

        field_strategy_name = kwargs.get(
            "field_strategy_name",
            _DICT_KEYS.get("name")
        )

        # get identifiers--
        self.code = kwargs.get("strategy_code")
        id_num = strategy_id
        self.name = kwargs.get("strategy_name")

        # set any attributes specified in the dictionary--can pass name and code here
        if isinstance(dict_attributes, dict):
            for k, v in dict_attributes.items():
                attr_try = getattr(self, k, None)
                if attr_try is not None:
                    continue

                setattr(self, str(k), v)

        
        ##  SET PROPERTIES

        self.id_num = id_num
        self.key_strategy = key_strategy
        self.field_description = field_description
        self.field_strategy_code = field_strategy_code
        self.field_strategy_name = field_strategy_name

        return None
    


    def _initialize_table(self,
        prebuild: bool,
        **kwargs,
    ) -> None:
        """
        Initialize the `table` property
        """

        table = (
            self.function(**kwargs, )
            if prebuild
            else None
        )

        self.table = table

        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_strategy
            * self._uuid
        """

        self.is_strategy = True
        self._uuid = _MODULE_UUID

        return None
    


    def get_transformation_list(self,
        code_specification: Union[int, str, List[Union[str, int]]],
        transformations: trn.Transformations,
        delim: str = "|",
        stop_on_error: bool = False,
    ) -> List[str]:
        """
        Get a list of codes to try to read. 

        Function Arguments
        ------------------
        - code_specification: 
            * string: either individual code of delim-delimited string of codes
            * integer: a strategy
            * list: list of string or integer values as above
        - transformations: tranformations object used to access Transformation
            objects

        Keyword Arguments
        -----------------
        - delim: optional delimiter to split codes
        - stop_on_error: if False, returns empty list if code_specification is
            invalid
        """
        
        # verify input type
        if sf.islistlike(code_specification):
            code_specification = list(code_specification)
        
        else:
            # throw an error if problem with code_specification
            error_q = not isinstance(code_specification, str) 
            error_q &= not sf.isnumber(code_specification, integer = True, )

            if error_q:
                if stop_on_error:
                    tp = str(code_specification)
                    raise ValueError(f"Invalid input type '{tp}' for code_specification in get_code_specification()")
                else:
                    return []

            
        codes = []
        
        # check type - if string, split; if no delimiter is found, return it
        if isinstance(code_specification, str):
            code_specification = [code_specification.split(delim)]
            
        elif isinstance(code_specification, int):
            code_specification = [[code_specification]]
        
        elif isinstance(code_specification, list):
            code_specification = [
                x.split(delim) if isinstance(x, str) else [x]
                for x in code_specification
            ]
        
        # sum it and get transformations
        code_specification = sum(code_specification, [])
        list_transformations = []

        # get valid codes
        for code in code_specification:
            code = transformations.get_transformation(
                code,
                return_code = True,
            )
            if code is None:
                continue
                
            list_transformations.append(code)
        
        # sort by key values 
        list_transformations = [
            transformations.get_transformation(x) 
            for x in transformations.attribute_transformation.key_values
            if x in list_transformations
        ]

        return list_transformations





class Strategies:
    """
    A collection of Strategy objects. Coordinate strategies, build an attribute
        table, test builds, and generate templates and hash_ids for build. 



        `strategy_definitions.csv`
        
        though this can be modified ysing the `fn_strategy_definition` 
        keyword argument.

        The strategy definition table *must* include the following columns:

            - baseline_strategy: binary field used to denote whether or not
                the strategy is the baseline. Must apply to one and only one
                strategy
            - strategy_id: This field should be the same as 
                transformers.model_attributes.dim_strategy_id. It is often
                useful to specify a strategy_id (an unique strategy key) 
                in ways that align with subsectors. 
                
                For example, strategies that are associated with AFOLU might 
                occupy 1000-1999, Circular Economy 2000-2999, ... , and 
                cross sector strategies 5000-5999. Howevrer, if not 
                specified, IDs will automatically be assigned. IDs are the
                default mechanism for telling SISEPUEDE which experiments to
                run. 
            - transformation_specification: this field is used to defin

        The following columns are optional:

            - strategy_code: optional code to specify
            - strategy_name: The strategy name is useful for display and
                tracking/automated reporting, but it is not used within the
                SISEPUEDE architecture.
            
            
    Initialization Arguments
    ------------------------
    - transformations: the Transformations object used to define strategies. All
        strategies that are defined in the `transformation_specification` field
        must be defined in the Transformations object. 

    Optional Arguments
    ------------------
    - baseline_id: optional specification of an ID as baseline. Default is 0.
    - export_path: optional export path specification.
        - If "transformations", writes to 
            os.path.join(transformations.dir_init, "templates")
        - If pathlib.Path, writes to that directory (must be a directory)
        - If None, exports to SISEPUEDE default (in ref)
    - fn_strategy_definition: file name of strategy definiton file in 
        transformations.dir_init *OR* pathlib.Path giving a full path to a
        strategy definitions CSV file.
    - logger: optional logger to use
    - prebuild: prebuild the tables?
    - stop_on_error: stop if a strategy fails to initialize
    - **kwargs: can be used to pass 
    """

    def __init__(self,
        transformations: trn.Transformations,
        baseline_id: int = 0,
        export_path: Union[str, pathlib.Path, None] = "transformations",
        fn_strategy_definition: Union[str, pathlib.Path] = _DICT_FILE_NAME_DEFAULTS.get("strategy_definitions"),
        logger: Union[logging.Logger, None] = None,
        prebuild: bool = True,
        stop_on_error: bool = False,
        **kwargs,
    ) -> None:

        self.logger = logger
        
        ##  FIRST STAGE -- BUILD TRANSFORMATIONS AND STRATEGIES

        self._initialize_transformations(
            transformations,
        )
        self._initialize_fields(
            **kwargs,
        )
        self._initialize_strategies(
            fn_strategy_definition,
            baseline_id,
            prebuild = prebuild,
            stop_on_error = stop_on_error,
        )

        
        ##  SECOND STAGE, REQUIRED FOR TEMPLATIZATION

        self._initialize_file_structure()
        self._update_model_attributes()
        self._initialize_base_input_database(
            export_path = export_path,
        )
        self._initialize_templates()

        self._initialize_uuid()
        
        return None
    



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def get_export_path(self,
        export_path: Union[str, pathlib.Path, None] = _FLAG_EXPORT_TO_TRANSFORMATIONS,
        flag_export_to_transformations: str = _FLAG_EXPORT_TO_TRANSFORMATIONS,
        subdir_default: str = "templates",
    ) -> str:
        """
        Retrieve the export path
        """
        
        # get SISEPUEDE defaults
        dir_templates = (
            self
            .file_struct
            .dict_data_mode_to_template_directory
            .get("calibrated")
        )

        dir_templates_demo = (
            self
            .file_struct
            .dict_data_mode_to_template_directory
            .get("demo")
        )

        
        ##  DIFFERENT ACTIONS BASED ON TYPE

        if isinstance(export_path, str):
            if export_path == flag_export_to_transformations:
                # turn into pathlib.Path
                export_path = self.transformations.dir_init.joinpath(subdir_default)

        if isinstance(export_path, pathlib.Path):
            if not export_path.exists():
                if ("." not in export_path.stem):
                    export_path.mkdir()
            
            if export_path.is_dir():
                dir_templates = str(export_path.joinpath("calibrated"))
        

        ##  RETURN DIRS

        out = (
            dir_templates,
            dir_templates_demo
        )

        return out



    def _initialize_base_input_database(self,
        export_path: Union[str, pathlib.Path, None] = None,
        regions: Union[List[str], None] = None,
        use_demo_template_on_missing: bool = True,
    ) -> None:
        f"""
        Initialize the BaseInputDatabase class used to construct future
            trajectories. Initializes the following properties:

            * self.base_input_database
            * self.base_input_database_demo
            * self.baseline_strategy
            * self.regions


        Keyword Arguments
        ------------------
        - export_path: optional export path specification.
            - If None, exports to SISEPUEDE default (in ref)
            - If "{_FLAG_EXPORT_TO_TRANSFORMATIONS}", writes to 
                os.path.join(transformations.dir_init, "templates")
            - If pathlib.Path, writes to that directory (must be a directory)
        - regions: list of regions to run experiment for
            * If None, will attempt to initialize all regions defined in
                ModelAttributes
        - use_demo_template_on_missing: tries to instantiate a blank template if
            a template for a target region is missing. 
        """

        self._log("Initializing BaseInputDatabase", type_log = "info")

        # get regions
        regions = (
            self.transformations.transformers.regions
            if not sf.islistlike(regions)
            else [
                x for x in self.transformations.transformers.regions_manager.all_regions
                if x in regions
            ]
        )

        # template directories
        dir_templates, dir_templates_demo = self.get_export_path(
            export_path = export_path,
        )

        
        # trying building for demo
        try:
            region_val = (
                regions[0]
                if len(regions) > 0
                else None
            )

            base_input_database_demo = ing.BaseInputDatabase(
                dir_templates_demo,
                self.model_attributes,
                region_val,
                demo_q = True,
                logger = self.logger,
            )

        except Exception as e:
            msg = f"Error initializing BaseInputDatabase for demo -- {e}"
            self._log(msg, type_log = "error")
            raise RuntimeError(msg)


        # try building base input database for all
        # first, try to copy templates (if necessary and desired) fom demo to new region
        self._try_template_init_from_demo(
            base_input_database_demo,
            regions,
            fp_templates_target = dir_templates,
            try_instantiating_templates = use_demo_template_on_missing,
        )


        try:
            base_input_database = ing.BaseInputDatabase(
                dir_templates,
                self.model_attributes,
                regions,
                create_export_dir = False,
                demo_q = False,
                logger = self.logger
            )

        except Exception as e:
            
            # first, try building 

            msg = f"Error initializing BaseInputDatabase -- {e}"
            self._log(msg, type_log = "error")
            raise RuntimeError(msg)

        
        ##  SET PROPERTIES

        self.base_input_database = base_input_database
        self.base_input_database_demo = base_input_database_demo
        self.dir_templates = dir_templates
        self.dir_templates_demo = dir_templates_demo
        self.regions = base_input_database.regions

        return None



    def _initialize_fields(self,
        **kwargs,
    ) -> None:
        """
        Sets the following other properties:

            * self.key_strategy
            * self.field_baseline_strategy
            * self.field_description
            * self.field_strategy_code
            * self.field_strategy_name
            * self.field_transformation_specification
        """

        # check from keyword arguments
        field_baseline = kwargs.get("field_baseline_strategy", _DICT_KEYS.get("baseline"))
        field_description = kwargs.get("field_description", _DICT_KEYS.get("description"))
        field_strategy_code = kwargs.get("field_strategy_code", _DICT_KEYS.get("code"))
        field_strategy_name = kwargs.get("field_strategy_name", _DICT_KEYS.get("name"))
        field_transformation_specification = kwargs.get(
            "field_transformation_specification", 
            _DICT_KEYS.get("transformation_specification"),
        )
        
        # set some keys
        key_region = self.model_attributes.dim_region
        key_strategy = self.model_attributes.dim_strategy_id
        key_time_period = self.model_attributes.dim_time_period


        ##  SET PROPERTIES
        
        self.field_baseline_strategy = field_baseline
        self.field_description = field_description
        self.field_strategy_code = field_strategy_code
        self.field_strategy_name = field_strategy_name
        self.field_transformation_specification = field_transformation_specification
        self.key_region = key_region
        self.key_strategy = key_strategy
        self.key_time_period = key_time_period

        return None
    


    def _initialize_file_structure(self,
    ) -> None:
        """
        Intialize the SISEPUEDEFileStructure object and model_attributes object.
            Initializes the following properties:

            * self.file_struct

            Used to access demo template paths.

        """

        # set shortcut
        self.file_struct = (
            self
            .transformations
            .transformers
            .file_struct
        )

        return None
    
    

    def _initialize_strategies(self,
        fn_strategy_definition: Union[str, pathlib.Path],
        baseline_id: int,
        prebuild: bool = True,
        stop_on_error: bool = False,
    ) -> None:
        """
        Initialize the strategy objects and the attribute table. Sets the 
            following properties:

            * self.all_strategies
            * self.attribute_table
            * self.baseline_id
            * self.dict_strategies
            * self.path_strategy_definition
        """
        # get the path to the strategy definition and the derived attribute table
        # - NOTE: the get_attribute_table() will check the baseline_id, so it can be set as a property
        path = self.get_strategy_definition_path(fn_strategy_definition, )
        attribute_table = self.get_attribute_table(
            path, 
            baseline_id,
        )

        
        ##  ITERATE OVER THE ATTRIBUTE TABLE TO BUILD STRATEGIES

        dict_strategies = {}

        for i, row in attribute_table.table.iterrows():
            
            # get components
            id_num = row.get(self.key_strategy)
            code = row.get(self.field_strategy_code)
            desc = row.get(self.field_description)
            name = row.get(self.field_strategy_name)
            tspec = row.get(self.field_transformation_specification)

            try:
                strat = Strategy(
                    id_num,
                    tspec,
                    self.transformations,
                    dict_attributes = {
                        "code": code,
                        "description": desc,
                        "name": name,
                        "transformation_specification": tspec,
                    },
                    prebuild = prebuild,
                )
            
            except Exception as e:
                msg = f"Strategy {id_num} (name '{name}') failed with exception: {e}"
                if stop_on_error:
                    raise RuntimeError(msg)

            dict_strategies.update({id_num: strat})

        all_strategies = sorted(list(dict_strategies.keys()))
        all_strategies_non_baseline = [x for x in all_strategies if x != baseline_id]

        ##  SET PROPERTIES

        self.all_strategies = all_strategies
        self.all_strategies_non_baseline = all_strategies_non_baseline
        self.attribute_table = attribute_table
        self.baseline_id = baseline_id
        self.dict_strategies = dict_strategies
        self.path_strategy_definition = path

        return None



    def _initialize_templates(self,
    ) -> None:
        """
        Initialize sectoral templates. Sets the following properties:

            * self.dict_sectoral_templates
            * self.input_template
        """

        # initialize some components
        input_template = ing.InputTemplate(
            None,
            self.model_attributes
        )
        attr_sector = self.model_attributes.get_sector_attribute_table()
        dict_sectoral_templates = {}


        for sector in self.model_attributes.all_sectors:
            
            # get baseline "demo" template, use for ranges
            fp_read = self.base_input_database_demo.get_template_path(
                self.regions[0], 
                sector
            )

            df_template = pd.read_excel(
                fp_read, 
                sheet_name = input_template.name_sheet_from_index(input_template.baseline_strategy)
            )

            # extract key fields (do not need time periods)
            fields_ext = [x for x in input_template.list_fields_required_base]
            fields_ext += [
                x for x in df_template.columns 
                if input_template.regex_template_max.match(str(x)) is not None
            ]
            fields_ext += [
                x for x in df_template.columns 
                if input_template.regex_template_min.match(str(x)) is not None
            ]

            df_template = df_template[fields_ext].drop_duplicates()

            dict_sectoral_templates.update({sector: df_template})

        self.dict_sectoral_templates = dict_sectoral_templates
        self.input_template = input_template

        return None

    

    def _initialize_transformations(self,
        transformations: trn.Transformations,
    ) -> None:
        """
        Sets the following other properties:

            * self.model_attributes
            * self.transformations
        """

        if not trn.is_transformations(transformations, ):
            tp = str(transformations)
            msg = f"Invalid type '{tp}' input for `transformations` in Strategies: must be of class Transformations"
            raise RuntimeError(msg)


        ##  set properties

        self.model_attributes = transformations.transformers.model_attributes # shortcut
        self.transformations = transformations

        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_strategies
            * self._uuid
        """

        self.is_strategies = True
        self._uuid = _MODULE_UUID

        return None
    


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
    


    def _update_model_attributes(self,
    ) -> None:
        """
        Update shared ModelAttributes, which will need to access single 
            attribute_strategy defined herein. Updates the following properties:

            * self.file_struct.model_attributes
            * self.model_attributess
            * self.transformations.transformers.model_attributes
        """

        # update dictionary in model attributes object
        model_attributes = self.model_attributes
        (
            model_attributes
            .dict_attributes
            .get(model_attributes.attribute_group_key_dim)
            .update(
                {
                    model_attributes.dim_strategy_id: self.attribute_table
                }
            )
        )

        
        ##  UPDATE ATTRIBUTES

        self.file_struct.model_attributes = model_attributes
        self.model_attributes = model_attributes
        self.transformations.transformers.file_struct.model_attributes = model_attributes
        self.transformations.transformers.model_attributes = model_attributes

        return None
    


    
    ##########################################
    #    INITIALIZATION SUPPORT FUNCTIONS    #
    ##########################################

    def _try_template_init_from_demo(self,
        base_input_database_demo: Union[ing.BaseInputDatabase, None],
        regions: List[str],
        fp_templates_target: Union[str, None] = None,
        try_instantiating_templates: bool = True,
    ) -> None:
        """
        Try initializing templates for each sector using the demo templates. 

        Function Arguments
        ------------------
        - base_input_database_demo: Base input database for demo. If None, tries
            to call self.base_input_database_demo
        - regions: list of regions to init for. If None, tries self.regions


        Keyword Arguments
        -----------------
        - fp_templates_target: optional path to templates to try. Need to 
            specify for regions if base_input_database_demo is being used to 
            build paths for region templates
        - try_instantiating_templates: base functionality; if True, tries
            initializing new templates from demo
        """
        
        ##  SOME INITIALIZATION

        try:
            base_input_database_demo = (
                self.base_input_database_demo
                if base_input_database_demo is None
                else base_input_database_demo
            )

            regions = self.regions if (regions is None) else regions

        except Exception as e:
            return None

        if not try_instantiating_templates:
            return None


        ##  ITERATE OVER REGIONS TO BUILD TEMPLATES IF NEEDED

        for region in regions:
            for sector in self.model_attributes.all_sectors:

                # try to build regional path
                fp_region = base_input_database_demo.get_template_path(
                    region,
                    sector,
                    create_export_dir = True,
                    demo_q = False,
                    fp_templates = fp_templates_target, 
                )

                if os.path.exists(fp_region):
                    continue

                # otherwise, get demo path and copy over
                fp_demo = base_input_database_demo.get_template_path(
                    None,
                    sector,
                )
                
                shutil.copyfile(
                    fp_demo,
                    fp_region
                )
        
        return None



    def get_attribute_table(self,
        fp: Union[str, pathlib.Path],
        baseline_id: int,
    ) -> None:
        """
        Read in the strategy definition file and build an attribute table
        """

        path = str(fp) if isinstance(fp, pathlib.Path) else fp
            
        try:
            df = pd.read_csv(path)
        
        except Exception as e:
            raise RuntimeError(f"Error reading strategies definition table from '{fp}': {e}")


        ##  CHECK FIELDS

        fields_req = [
            self.key_strategy,
            self.field_transformation_specification
        ]

        # check fields; if missing, an error will be thrown
        sf.check_fields(
            df,
            fields_req,
            msg_prepend = "Fields required for Strategy definiton table "
        )

        
        ##  SPECIFY BASELINE INFO

        # verify that the baseline id is in the keys
        if baseline_id not in df[self.key_strategy].unique():
            msg = f"Baseline {self.key_strategy} = {baseline_id} not found in attribute table. Check that the baseline is defined"
            raise KeyError(msg)
        
        # set baseline field (ignore if it exists already)
        df[self.field_baseline_strategy] = [
            int(x == baseline_id) 
            for x in list(df[self.key_strategy])
        ]
            

        # next, build attribute table and check the specification of baseline strategy
        attribute_table = AttributeTable(
            df,
            self.key_strategy,
            []
        )

        return attribute_table
    


    def get_strategy_definition_path(self,
        path_specification: Union[str, pathlib.Path],
    ) -> pathlib.Path:
        """
        Get the path to strategy definition file
        """

        # check specification of path
        if isinstance(path_specification, str):
            path = pathlib.Path(self.transformations.dir_init)
            path_specification = path.joinpath(path_specification)
        
        if not isinstance(path_specification, pathlib.Path):
            tp = str(type(path_specification))
            msg = f"Invalid fn_strategy_definition specification in Strategies: input must be of type str or pathlib.Path"
            raise ValueError(msg)
        
        # verify that path exists
        if not path_specification.exists():
            msg = f"Invalid fn_strategy_definition specification in Strategies: strategy definition table '{path_specification}' not found."
            raise RuntimeError(msg)

        return path_specification
    


    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def build_strategies_long(self,
        df_input: Union[pd.DataFrame, None] = None,
        include_base_df: bool = True,
        strategies: Union[List[str], List[int], None] = None,
    ) -> pd.DataFrame:
        """
        Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
        - df_input: baseline (untransformed) data frame to use to build 
            strategies. Must contain self.key_region and 
            self.model_attributes.dim_time_period in columns. If None, defaults
            to self.baseline_inputs
        - include_base_df: include df_input in the output DataFrame? If False,
            only includes strategies associated with transformation 
        - strategies: strategies to build for. Can be a mixture of strategy_ids
            and names. If None, runs all available. 
        """

        # INITIALIZE STRATEGIES TO LOOP OVER

        strategies = (
            self.all_strategies_non_baseline
            if not sf.islistlike(strategies)
            else [x for x in self.all_strategies_non_baseline if x in strategies]
        )

        strategies = [self.get_strategy(x) for x in strategies]
        strategies = sorted([x.id_num for x in strategies if x is not None])
        n = len(strategies)


        # LOOP TO BUILD
        
        t0 = time.time()
        self._log(
            f"Strategies.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        # initialize baseline
        strat_baseline = self.get_strategy(self.baseline_id, )
        df_out = (
            strat_baseline(df_input = df_input, )
            if df_input is not None
            else strat_baseline()
        )

        if df_out is None:
            return None

        # initialize to overwrite dataframes
        iter_shift = int(include_base_df)
        df_out = [df_out for x in range(len(strategies) + iter_shift)]

        for i, strat in enumerate(strategies):
            t0_cur = time.time()
            strategy = self.get_strategy(strat)

            if strategy is not None:
                try:
                    df_out[i + iter_shift] = strategy(df_input = df_input, )
                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built strategy {self.key_strategy} = {strategy.id_num} ('{strategy.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )

                except Exception as e: 
                    df_out[i + 1] = None
                    self._log(
                        f"\tError trying to build strategy {self.key_strategy} = {strategy.id_num}: {e}",
                        type_log = "error"
                    )
            else:
                df_out[i + iter_shift] = None
                self._log(
                    f"\tStrategy {self.key_strategy} not found. Skipping...",
                    type_log = "warning"
                )

        # concatenate, log time elapsed and completion
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"Strategies.build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out
    


    def build_strategies_to_templates(self,
        df_base_trajectories: Union[pd.DataFrame, None] = None,
        df_exogenous_strategies: Union[pd.DataFrame, None] = None,
        regions: Union[List[str], None] = None,
        replace_template: bool = False,
        return_q: bool = False,
        sectors: Union[List[str], str] = None,
        strategies: Union[List[str], List[int], None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        df_base_trajectories : Union[pd.DataFrame, None]
            baseline (untransformed) data frame to use to build strategies. Must 
            contain self.key_region and model_attributes.dim_time_period in 
            columns. If None, defaults to self.baseline_inputs
        df_exogenous_strategies : Union[pd.DataFrame, None]
            Optional exogenous strategies to pass. Must contain self.key_region 
            and model_attributes.dim_time_period in columns. If None, no action 
            is taken. 
        regions : Union[List[str], None]
            Optional list of regions to build strategies for. If None, defaults 
            to all defined.
        replace_template : bool
            Replace template if it exists? If False, tries to overwrite existing 
            sheets.
        return_q : bool
            Return an output dictionary? If True, will return all templates in 
            the form of a dictionary. Otherwise, writes to output path implied 
            by SISEPUEDEFileStructure
        sectors : Union[List[str], str]
            Optional sectors to specify for export. If None, will export all.
        strategies : Union[List[str], List[int], None]
            Strategies to build for. Can be a mixture of strategy_ids and names. 
            If None, runs all available. 
        **kwargs
            Passed to self.input_template.template_from_inputs(). Notable 
            keyword arguments include:

            - df_trajgroup: optional dataframe mapping each field variable to 
                trajectory groups. (default None)
                * Must contain field_subsector, field_variable, and 
                    field_variable_trajectory_group as fields
                * Overrides include_simplex_group_as_trajgroup if specified and 
                    conflicts occur
		    - include_simplex_group_as_trajgroup: default to include simplex 
                group from attributes as trajectory group? (default True)
        """

        # INITIALIZE STRATEGIES TO LOOP OVER
        
        model_attributes = self.model_attributes

        # initialize attributes and other basic variables
        attr_sector = model_attributes.get_sector_attribute_table()
        attr_strat = model_attributes.get_dimensional_attribute_table(
            model_attributes.dim_strategy_id
        )

        fields_var_all = model_attributes.build_variable_dataframe_by_sector(
            None, 
            include_time_periods = False,
        )
        fields_var_all = list(fields_var_all["variable_field"])
        
        
        # initialize baseline dataframe
        strat_baseline = self.get_strategy(self.baseline_id, )
        df_out = strat_baseline(df_input = df_base_trajectories, )
        return_none = df_out is None
        
        
        # get input component and add baseline strategy marker
        fields_sort = (
            [
                self.attribute_table.key, 
                self.key_time_period
            ] 
            if (self.attribute_table.key in df_out.columns) 
            else [self.key_time_period]
        )
        
        
        # check regions
        regions = (
            [x for x in self.regions if x in regions]
            if sf.islistlike(regions)
            else self.regions
        )
        n_r = len(regions)
        return_none |= (len(regions) == 0)
        
        
        # check sectors
        sectors = (
            [x for x in sectors if x in model_attributes.all_sectors]
            if sf.islistlike(sectors)
            else model_attributes.all_sectors
        )
        return_none |= (len(sectors) == 0)
        
        
        # check strategies HEREHERE

        if not sf.islistlike(strategies):
            strategies = self.all_strategies_non_baseline
        
        else:
            strategies_out = []
            for x in strategies:
                x = self.get_strategy_id(x)

                continue_q = x is None
                continue_q |= (x == self.baseline_id) if not continue_q else continue_q
                if continue_q:
                    continue
                    
                strategies_out.append(x)

            strategies = [x for x in self.all_strategies_non_baseline if x in strategies_out]


        n = len(strategies)
        
        if return_none:
            return None
        
        
        # LOOP TO BUILD

        t0 = time.time()
        self._log(
            f"Starting build of {n} strategies in {n_r} regions...",
            type_log = "info"
        )
        
        
        # initialize to overwrite dataframes
        dict_sectors = dict((s, {}) for s in attr_sector.key_values)
        dict_write = dict((r, dict_sectors) for r in regions)
        df_out_grouped = df_out.groupby([self.key_region])

        # try getting exogenous strategies grouped by region (as dict) -- if fails, will return None
        dict_exogenous_grouped = self.check_exogenous_strategies(df_exogenous_strategies)
    

        # iterate over regions
        for r, df in df_out_grouped:
            
            r = r[0] if isinstance(r, tuple) else r
            if r not in regions:
                continue
            
            self._log(f"Starting build for region {r}", type_log = "info")
            
            df_out_list = [df, df.copy()]
            dict_cur = (
                dict_write.get(r)
                if return_q
                else dict_sectors.copy()
            )
                

            ##  ITERATE OVER FUNCTIONAL TRANSFORMATIONS

            for i, strat in enumerate(strategies):

                t0_cur = time.time()
                strategy = self.get_strategy(strat)

                if strategy is None:
                    self._log(
                        f"\tStrategy {self.key_strategy} not found: check that a Strategy object has been defined associated with the id.",
                        type_log = "warning"
                    )
                    continue


                # build strategies (baseline and alternative)
                try:
                    df_out_list[1] = strategy(df_input = df, )

                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built Strategy {self.key_strategy} = {strategy.id_num} ('{strategy.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )
                    skip_q = False
                    
                except Exception as e: 
                    df_out[i + 1] = None
                    self._log(
                        f"\tError trying to build strategy {self.key_strategy} = {strategy.id_num}: {e}",
                        type_log = "error"
                    )
                    continue
                
                # turn into input to template builder 
                df_cur = pd.concat(df_out_list, axis = 0).reset_index(drop = True)
                fields_drop = [x for x in df_cur.columns if x not in fields_var_all + fields_sort]
                df_cur = (
                    df_cur
                    .drop(fields_drop, axis = 1)
                    .sort_values(by = fields_sort)
                    .reset_index(drop = True)
                )

                # verify that strategy ids are being passed properly--this ios sometimes a problem with transformer base functions
                if len(df_cur[self.key_strategy].unique()) == 1:    
                    msg = f"""Error trying to build strategy {self.key_strategy} = {strategy.id_num}: 
                    At least one transformer function (or transformer baselib function) is not properly 
                    associated with a strategy_id. Check the functions and rebuiild.
                    """
                    self._log(msg, type_log = "error", )

                    continue

                """
                Needed for troubleshooting sometimes:

                global dc
                global dc2

                if strat == 1006:
                    dc = df_cur.copy() 
                elif strat == 1007:
                    dc2 = df_cur.copy()
                """;

                # split the current transformation into 
                dict_cur = self.build_templates_dictionary_from_current_transformation(
                    df_cur,
                    attr_sector,
                    dict_cur,
                    **kwargs
                )
                

            ##  ADD ANY ADDITIONAL, EXOGENOUSLY DEFINED STRATEGIES?

            if isinstance(dict_exogenous_grouped, dict):
                
                self._log(
                    f"Starting integration of exogenous strategies in region '{r}",
                    type_log = "info"
                )

                df_cur_grouped = dict_exogenous_grouped.get(r)
                if df_cur_grouped is not None:

                    # ignore strategies that are defined functionally
                    df_cur_grouped = (
                        df_cur_grouped[
                            ~df_cur_grouped[self.key_strategy].isin(strategies)
                        ]
                        .groupby([self.key_strategy])
                    )

                    for strat, df_exog in df_cur_grouped:
                        
                        strat = strat[0] if isinstance(strat, tuple) else strat

                        df_cur = None
                        # try to concatenate current strategy
                        try:
                            df_out_list[1] = df_exog
                            df_cur = (
                                pd.concat(
                                    df_out_list, 
                                    axis = 0,
                                )
                                .reset_index(drop = True)
                            )

                            self._log(
                                f"\tSuccessfully integrated exogenous strategy {self.key_strategy} = {strat}.",
                                type_log = "info"
                            )
                            
                        except Exception as e: 
                            df_out[i + 1] = None

                            self._log(
                                f"\tError trying to integrate exogenous strategy {self.key_strategy} = {strat}: {e}",
                                type_log = "error"
                            )

                            continue
                    
                        # drop unnecessary fields and prepare for templatization
                        fields_drop = [x for x in df_cur.columns if x not in fields_var_all + fields_sort]
                        df_cur = (
                            df_cur
                            .drop(fields_drop, axis = 1)
                            .sort_values(by = fields_sort)
                            .reset_index(drop = True)
                        )

                        # split the current exogenous transformation up
                        dict_cur = self.build_templates_dictionary_from_current_transformation(
                            df_cur,
                            attr_sector,
                            dict_cur,
                            **kwargs
                        )

            
            ##  EXPORT FILES?

            if not return_q:
                for sector_abv in attr_sector.key_values:
                    
                    sector = (
                        attr_sector
                        .field_maps.get(f"{attr_sector.key}_to_sector")
                        .get(sector_abv)
                    )
        
                    if sector not in sectors:
                        continue
                        
                     # get path and write output   
                    fp_write = self.base_input_database.get_template_path(
                        r, 
                        sector,
                        create_export_dir = True
                    )

                    self._log(
                        f"Exporting '{sector_abv}' template in region {r} to {fp_write}", 
                        type_log = "info"
                    )

                    sf.dict_to_excel(
                        fp_write,
                        dict_cur.get(sector_abv),
                        replace_file = replace_template,
                    )
                    
            
            self._log(f"Templates build for region {r} complete.\n\n", type_log = "info")

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"Integrated transformations build complete in {t_elapse} seconds.",
            type_log = "info"
        )
        
        return_val = dict_write if return_q else 0

        return return_val
    


    def build_templates_dictionary_from_current_transformation(self,
        df_cur: pd.DataFrame,
        attr_sector: AttributeTable,
        dict_update: dict,
        **kwargs
    ) -> Union[Dict, None]:
        """
        Support function for build_strategies_to_templates(); 

        Function Arguments
        ------------------
        - df_cur: data frame that represents a transformation
        - attr_sector: sector attribute table
        - dict_update: dictionary to update

        Keyword Arguments
        -----------------
        **kwargs: passed to self.input_template.template_from_inputs()
        """

        for sector_abv in attr_sector.key_values:

            # get sector name
            sector = (
                attr_sector
                .field_maps
                .get(f"{attr_sector.key}_to_sector")
                .get(sector_abv)
            )

            df_template = self.dict_sectoral_templates.get(sector)

            # build template
            dict_write_cur = self.input_template.template_from_inputs(
                df_cur,
                df_template,
                sector_abv,
                **kwargs
            )

            dict_update[sector_abv].update(dict_write_cur)

        return dict_update
    


    def build_tornado_strategies(self,
        strategy_base: Union[int, str, None],
        code_prepend: str = "TORNADO",
        code_prependages_skip: Union[str, List[str]] = "PFLO",
        delim: Union[str, None] = None,
        delim_code: str = ":",
        max_length_intersection: Union[int, None] = 0, 
        strategy_stress: Union[int, str, None] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, None]:
        """Build strategies designed by adding transformations 1-by-1 to a 
            strategy

        Function Arguments
        ------------------
        strategy_base : Union[int, str]
            Strategy to add to the base. If None, defaults to base strategy
        
        Keyword Arguments
        -----------------
        code_prepend : str
            Code to prepend to new strategies
        delim : str 
            Delimiter used to split transformation specifications
        delim_code : str
            Delimiter used to track transformation code hierarchy 
        max_length_intersection : Union[int, None]
            Optional specification of maximum size of intersection between 
            strategy_base and potential strategies
        strategy_stress : Union[int, str, None]
            Optional strategy to decompose into tornado; if None, will build
            tornado design for each defined transformation
        """

        ##  INITIALIZATION
        
        # get the strategy and transformation codes asociated with it
        strategy_base = self.baseline_id if (strategy_base is None) else strategy_base
        strat = self.get_strategy(strategy_base, )
        if strat is None:
            return None

        code_base = strat.code
        id_base = strat.id_num

        # get code prependages to skip
        code_prependages_skip = (
            [code_prependages_skip]
            if isinstance(code_prependages_skip, str)
            else code_prependages_skip
        )
        code_prependages_skip = (
            [] 
            if not isinstance(code_prependages_skip, list)
            else code_prependages_skip
        )

        # get transformations included in the base strategy
        transformations_deconstruct = strat.get_transformation_list(
            strat.transformation_specification,
            self.transformations,
        )
        codes_base = sorted([x.code for x in transformations_deconstruct])

        delim = (
            strat.delimiter_transformation_codes
            if not isinstance(delim, str)
            else delim
        )

        
        # set transformation codes to iterate over
        codes_iter = self.transformations.all_transformation_codes

        # restrict to one strategy's set?
        strategy_stress = self.get_strategy(strategy_stress, )
        if is_strategy(strategy_stress, ):
            transformations_restrict = strategy_stress.get_transformation_list(
                strategy_stress.transformation_specification,
                self.transformations,
            )

            # filter down 
            transformations_restrict = [x.code for x in transformations_restrict]
            codes_iter = [x for x in transformations_restrict if x in codes_iter]
        
        
        ##  GET STRATEGY CODES TO ITERATE OVER

        keys = self.attribute_table.key_values
        id_new = max(self.attribute_table.key_values)

        codes_new = []
        ids_new = []
        names_new = []
        specs_new = []
        
        # set the maximum size of the intersection
        max_length_intersection = np.inf if max_length_intersection is None else max_length_intersection
        max_length_intersection = (
            0 
            if not sf.isnumber(max_length_intersection, integer = True) 
            else max_length_intersection
        )

        # 
        for code in codes_iter:
            
            transformation = self.transformations.get_transformation(code)
            
            # get the strategy and run some checks
            skip = (code == self.transformations.code_baseline)
            skip |= transformation in transformations_deconstruct
            
            if skip: continue

            # setup the spcification
            spec_new = (
                [strat.transformation_specification, code]
                if id_base != self.baseline_id
                else [code]
            )
            spec_new = delim.join(spec_new)
            
            # new code and id
            code_new = f"{code_prepend}{delim_code}{code}"
            name_new = f"Add {code} to {strat.name}"
            id_new += 1
            
            codes_new.append(code_new, )
            ids_new.append(id_new, )
            names_new.append(name_new, )
            specs_new.append(spec_new, )


        # build output datadrame
        df_out = pd.DataFrame(
            {
                self.attribute_table.key: ids_new,
                self.field_baseline_strategy: np.zeros(len(specs_new)),
                self.field_description: ["" for x in specs_new],
                self.field_strategy_code: codes_new,
                self.field_strategy_name: names_new,
                self.field_transformation_specification: specs_new,
            }
        )

        df_out = df_out[self.attribute_table.table.columns]

        return df_out
    


    def build_whirlpool_strategies(self,
        strategy: Union[int, str, None],
        code_prepend: str = "WHIRLPOOL",
        delim: Union[str, None] = None,
        ids: Union[None, List[int]] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, None]:
        """Build strategies designed by removing transformations 1-by-1 from a 
            strategy (whirlpool--kind of the inverse of a tornado)

        Function Arguments
        ------------------
        strategy : Union[int, str, None]
            Strategy to remove transformations from
        
        Keyword Arguments
        -----------------
        code_prepend : str
            Code to prepend to the transformation codes to create new strategy
            code.
        delim : Union[str, None]
            Delimiter used to split transformation specifications
        ids : Union[None, List[int]]
            Optional specification of IDs. 
            * int:          Specify the base id explicitly. Will take the 
                            maximum between this value and max(existing_ids) + 1
            * List[int]:    Specify ids explicitly. Must be of correct length.
            * None:         Automatically start at 1 above the highest defined 
                            strategy id.
        """
        # get the strategy and transformation codes asociated with it
        strat = self.get_strategy(strategy, )
        if strat is None:
            return None
        
        transformations_deconstruct = strat.get_transformation_list(
            strat.transformation_specification,
            self.transformations,
        )
        codes = sorted([x.code for x in transformations_deconstruct])

        delim = (
            strat.delimiter_transformation_codes
            if not isinstance(delim, str)
            else delim
        )


        ##  START BUILDING FIELDS
        
        trans_specs = []
        trans_code = []
        trans_name = []
        
        tab = self.attribute_table.table
        all_codes = list(tab[self.field_strategy_code].unique())
        all_names = list(tab[self.field_strategy_name].unique())

        for i, code in enumerate(codes):
            
            # verify that the new code and name are unique
            code_new = f"{code_prepend}:{code}"
            name_new = f"Remove {code} from {strat.name}"

            continue_q = (code_new in all_codes + trans_code) 
            continue_q |= (name_new in all_names + trans_name)
            if continue_q:
                continue
            
            # otherwise, remove codes one by one
            if i == 0:
                codes_cur = codes[i+1:]
        
            elif i == len(codes) - 1:
                codes_cur = codes[0:-1]
        
            else:
                codes_cur = codes[0:i] + codes[i + 1:]
        
            trans_specs.append(delim.join(codes_cur))
            trans_code.append(code_new)
            trans_name.append(name_new)


        ##  BUILD IDS

        keys = self.attribute_table.key_values
        max_id = max(self.attribute_table.key_values)

        build_ids = not sf.islistlike(ids)
        if not build_ids:
            ids = [x for x in ids if x not in keys]
            build_ids = len(ids) != len(trans_specs)
        elif sf.isnumber(ids, integer = True, ):
            max_id = max(max_id, ids - 1, )
                

        ids = (
            list(range(max_id + 1, max_id + len(trans_specs) + 1))
            if build_ids
            else ids
        )


        ##  BUILD OUTPUT TABLE

        df_out = pd.DataFrame(
            {
                self.attribute_table.key: ids,
                self.field_baseline_strategy: np.zeros(len(trans_specs)),
                self.field_description: ["" for x in trans_specs],
                self.field_strategy_code: trans_code,
                self.field_strategy_name: trans_name,
                self.field_transformation_specification: trans_specs,
            }
        )

        df_out = df_out[self.attribute_table.table.columns]
                
        return df_out
    


    def check_exogenous_strategies(self,
        df_exogenous_strategies: pd.DataFrame,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """Check df_exogenous_strategies for build_strategies_to_templates(). If 
            df_exogenous_strategies is valid, will return a dictionary mapping
            a region (key) to a dataframe containing exogenous strategies.

            Otherwise, returns None. 
        """
        # check if exogenous strategies are being passed
        dict_exogenous_grouped = None
        return_none = False

        if isinstance(df_exogenous_strategies, pd.DataFrame):
            
            # check fields
            if self.key_region not in df_exogenous_strategies.columns:
                self._log(
                    f"Region key '{self.key_region}' missing from df_exogenous_strategies. Strategies will not be read from df_exogenous_strategies.",
                    type_log = "warning",
                )
                return_none = True
            
            if self.key_strategy not in df_exogenous_strategies.columns:
                self._log(
                    f"Strategy key '{self.key_strategy}' missing from df_exogenous_strategies. Strategies will not be read from df_exogenous_strategies.",
                    type_log = "warning",
                )
                return_none = True

            
            if return_none:
                return None

            # next, try to group the dataframe
            try:
                dict_exogenous_grouped = sf.group_df_as_dict(
                    df_exogenous_strategies,
                    [self.key_region],
                )

            except Exception as e:
                self._log(
                    f"Error trying to group df_exogenous_strategies in check_exogenous_strategies(): {e}.",
                    type_log = "error",
                )

        # return output
        return dict_exogenous_grouped
    


    def get_strategy(self,
        strategy: Union[int, str, None],
        return_code: bool = False,
    ) -> None:
        """Get `strategy` based on strategy code, id, or name
            
        Function Arguments
        ------------------
        strategy : Union[int, str, None]
            strategy_id, strategy name, or strategy code to use to retrieve 
            Strategy object
            
        Keyword Arguments
        ------------------
        return_code : bool
            Set to True to return the transformer code only
        """

        # skip these types
        is_int = sf.isnumber(strategy, integer = True)
        return_none = not is_int
        return_none &= not isinstance(strategy, str)
        if return_none:
            return None

        # Transformer objects are tied to the attribute table, so these field maps work
        dict_code_to_id = self.attribute_table.field_maps.get(
            f"{self.field_strategy_code}_to_{self.attribute_table.key}"
        )
        dict_name_to_id = self.attribute_table.field_maps.get(
            f"{self.field_strategy_name}_to_{self.attribute_table.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(strategy, str):
            code = (
                dict_code_to_id.get(strategy)
                if strategy in dict_code_to_id.keys()
                else dict_name_to_id.get(strategy)
            )
        
        elif is_int:
            code = strategy

        # check returns
        if return_code | (code is None): 
            return code
        
        out = deepcopy(self.dict_strategies.get(code))

        return out
    


    def get_strategy_id(self,
        strategy_specification: Union[int, str, None],
    ) -> Union[int, None]:
        """
        Get the strategy id based on strategy code, id, or name
            
        Function Arguments
        ------------------
        - strategy_specification: strategy_id, strategy name, or strategy code 
            to use to retrieve ID
            
        Keyword Arguments
        ------------------
        """

        strat = self.get_strategy(strategy_specification)
        out = None if (strat is None) else strat.id_num

        return out

    





########################
#    SOME FUNCTIONS    #
########################

def is_strategies(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Strategies
    """
    out = hasattr(obj, "is_strategies")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out



def is_strategy(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Strategy
    """
    out = hasattr(obj, "is_strategy")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out

















