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
    A collection of transformations

    A strategy definition table, which provides strategy names and IDs
        for combinations of transformations. By default, this file is called

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

















