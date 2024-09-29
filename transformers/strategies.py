import logging
import numpy as np
import pandas as pd
import re
import time
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.core.support_classes as sc
import sisepuede.transformers.transformations as trn
import sisepuede.utilities._toolbox as sf




_MODULE_UUID = "D3BC5456-5BB7-4F7A-8799-AFE0A44C3FFA" 



_DICT_KEYS = {
    "code": "strategy_code",
    "description": "description",
    "name": "strategy_name",
    "transformation_specification": "transformation_specification"
}


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

        if self.table is not None:
            out = self.table

        else:
            out = self.function(*args, **kwargs)
            self.table = out # update the table

        return out




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_function(self,
        transformation_codes: Union[str, List[Union[str, int]]],
        transformations: trn.Transformations,
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None

        ##  initialize transformations 

        func = self.get_transformation_list(
            transformation_codes,
            transformations,
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
                out = None
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
                    strat = self.id_num,
                )
                
                return out

            
            function = function_out
            function_list = [function_out]




        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
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
            
        key_strategy = transformations.transformers.model_attributes.dim_strategy_id

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

        # get identifiers
        code = kwargs.get("strategy_code")
        id_num = strategy_id
        name = kwargs.get("strategy_name")

        # set any attributes specified in the dictionary
        if isinstance(dict_attributes, dict):
            for k, v in dict_attributes.items():
                if hasattr(self, k):
                    continue

                setattr(self, str(k), v)

        
        ##  SET PROPERTIES

        self.code = code
        self.id_num = id_num
        self.name = name
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
            * self.uuid
        """

        self.is_strategy = True
        self.uuid = _MODULE_UUID

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
        fn_strategy_definition: Union[str, pathlib.Path] = "strategy_definitions.csv",
        logger: Union[logging.Logger, None] = None,
        prebuild: bool = True,
        stop_on_error: bool = False,
        **kwargs,
    ) -> None:

        self.logger = logger
        
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

        self._initialize_uuid()
        
        return None
    



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_fields(self,
        **kwargs,
    ) -> None:
        """
        Sets the following other properties:

            * self.key_strategy
            * self.field_code
            * self.field_description
            * self.field_id
            * self.field_name
            * self.field_transformation_specification
        """

        # check from keyword arguments
        field_description = kwargs.get("field_description", _DICT_KEYS.get("description"))
        field_strategy_code = kwargs.get("field_strategy_code", _DICT_KEYS.get("code"))
        field_strategy_name = kwargs.get("field_strategy_name", _DICT_KEYS.get("name"))
        field_transformation_specification = kwargs.get(
            "field_transformation_specification", 
            _DICT_KEYS.get("transformation_specification"),
        )
        

        key_strategy = (
            self
            .transformations
            .transformers
            .model_attributes
            .dim_strategy_id
        )


        ##  SET PROPERTIES
        
        self.field_description = field_description
        self.field_strategy_code = field_strategy_code
        self.field_strategy_name = field_strategy_name
        self.field_transformation_specification = field_transformation_specification
        self.key_strategy = key_strategy

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
        

        ##  SET PROPERTIES

        self.all_strategies = all_strategies
        self.attribute_table = attribute_table
        self.baseline_id = baseline_id
        self.dict_strategies = dict_strategies
        self.path_strategy_definition = path

        return None


    

    def _initialize_transformations(self,
        transformations: trn.Transformations,
    ) -> None:
        """
        Sets the following other properties:

            * self.transformations
        """

        if not trn.is_transformations(transformations, ):
            tp = str(transformations)
            msg = f"Invalid type '{tp}' input for `transformations` in Strategies: must be of class Transformations"
            raise RuntimeError(msg)

        
        ##  set properties

        self.transformations = transformations
        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_strategies
            * self.uuid
        """

        self.is_strategies = True
        self.uuid = _MODULE_UUID

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

    

    
    ##########################################
    #    INITIALIZATION SUPPORT FUNCTIONS    #
    ##########################################

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

        # next, build attribute table and check the specification of baseline strategy
        attribute_table = AttributeTable(
            df,
            self.key_strategy,
            []
        )

        # verify that the baseline id is in the keys
        if baseline_id not in attribute_table.key_values:
            msg = f"Baseline {self.key_strategy} = {baseline_id} not found in attribute table. Check that the baseline is defined"
            raise KeyError(msg)

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
    uuid = getattr(obj, "uuid", None)

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
    uuid = getattr(obj, "uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out

















