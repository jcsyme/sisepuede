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
 
    """

    def __init__(self,
        strategy_id: int,
        transformation_codes: Union[str, List[Union[str, int]]],
        transformations: trn.Transformations,
        delim: str = "|",
    ) -> None:

        self._initialize_identifiers(
            strategy_id,
            transformations,
        )
        self._initialize_function(
            transformation_codes,
            transformations,
        )
        self._initialize_uuid()
        
        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> pd.DataFrame:

        out = self.function(*args, **kwargs)

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
                    out = f(df_input = out, strat = self.id_num, )#**kwargs)

                return out

            function = function_out
            function_list = func
        
        else:

            def function_out(
                *args, 
                **kwargs
            ) -> pd.DataFrame:

                out = transformations.transformers.baseline_inputs
                
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
        **kwargs,
    ) -> None:
        """
        Set transformation code and, optionally, transformation name. Sets the
            following properties:

            self.code: 
                optional strategy code
            self.id_num 
                strategy id number, base of all SISEPUEDE indexing
            self.name   
                strategy name 
            self.key_strategy
                strategy key (used in attribute table)
            self.field_baseline_strategy 
            self.field_description = field_description
            self.field_strategy_code = field_strategy_code
            self.field_strategy_name

        
        Function Arguments
        ------------------


        Keyword Arguments
        -----------------
        - **kwargs: can be used to pass

            * description: optional decription to pass
            * strategy_code: optional code to pass for the strategy
            * strategy_name: name of the strategy
        """
        # verify that the id is correctly input 
        if not sf.isnumber(strategy_id, integer = True, ):
            tp = str(type(strategy_id))
            msg = f"Invalid type '{tp}' specified for strategy_id in Strategy: must be integer."
            raise ValueError(msg)
            
        key_strategy = transformations.transformers.model_attributes.dim_strategy_id

        # set some fields
        field_baseline_strategy = "baseline_strategy"
        field_description = "description"
        field_strategy_code = key_strategy.replace("_id", "code")
        field_strategy_name = key_strategy.replace("_id", "name")

        # get identifiers
        code = kwargs.get("strategy_code")
        id_num = strategy_id
        name = kwargs.get("strategy_name")

        
        ##  SET PROPERTIES

        self.code = code
        self.id_num = id_num
        self.name = name
        self.key_strategy = key_strategy
        self.field_baseline_strategy = field_baseline_strategy
        self.field_description = field_description
        self.field_strategy_code = field_strategy_code
        self.field_strategy_name = field_strategy_name

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
        
        print("he1")
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

            - strategy_code
            - transformation_specification

        The following columns are optional:

            - strategy_name: The strategy name is useful for display and
                tracking/automated reporting, but it is not used within the
                SISEPUEDE architecture.
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

                
    Initialization Arguments
    ------------------------
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    Optional Arguments
    ------------------
    - fn_strategy_definition: file name of strategy definiton file in 
        transformations.dir_init *OR* pathlib.Path giving a full path to a
        strategy definitions CSV file.
 
    """

    def __init__(self,
        transformations: trn.Transformations,
        fn_strategy_definition: str = "stratgy_definitions.csv",
    ) -> None:

        
        self._initialize_uuid()
        
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

















