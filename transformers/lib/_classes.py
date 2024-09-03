import logging
import numpy as np
import pandas as pd
import time
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import *
import sisepuede.utilities._toolbox as sf
 



class Strategy:
    """
    A collection of transformations
    """

    def __init__(self,
    ) -> None:

        return None
    


class Transformer:
    """
    Create a Transformation class to support construction in sectoral 
        transformations. 

    Initialization Arguments
    ------------------------
    - code: strategy code associated with the transformation. Must be defined in 
        attr_strategy.table[field_strategy_code]
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    - attr_strategy: AttributeTable usd to define strategies from 
        ModelAttributes

    Keyword Arguments
    -----------------
    - field_strategy_code: field in attr_strategy.table containing the strategy
        codes
    - field_strategy_name: field in attr_strategy.table containing the strategy
        name
    """
    
    def __init__(self,
        code: str,
        func: Union[Callable, List[Callable]],
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str = "strategy_code",
        field_strategy_name: str = "strategy",
    ) -> None:
        
        self._initialize_function(func)
        self._initialize_code(
            code, 
            attr_strategy, 
            field_strategy_code,
            field_strategy_name,
        )

        return None
        
    
    
    def __call__(self,
        *args,
        **kwargs
    ) -> Any:
        
        val = self.function(
            *args,
            strat = self.id,
            **kwargs
        )

        return val




    def _initialize_code(self,
        code: str,
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str,
        field_strategy_name: str,
    ) -> None:
        """
        Initialize the transformation name. Sets the following
            properties:

            * self.baseline 
                - bool indicating whether or not it represents the baseline 
                    strategy
            * self.code
            * self.id
            * self.name
        """
        
        # initialize and check code/id num
        id_num = (
            attr_strategy.field_maps.get(f"{field_strategy_code}_to_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        id_num = id_num.get(code) if (id_num is not None) else -1

        if id_num is None:
            raise ValueError(f"Invalid strategy code '{code}' specified in support_classes.Transformation: strategy not found.")

        id_num = id_num if (id_num is not None) else -1

        # initialize and check name/id num
        name = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_{field_strategy_name}")
            if attr_strategy is not None
            else None
        )
        name = name.get(id_num) if (name is not None) else ""

        # check baseline
        baseline = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_baseline_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        baseline = (baseline.get(id_num, 0) == 1)


        ##  set properties

        self.baseline = bool(baseline)
        self.code = str(code)
        self.id = int(id_num)
        self.name = str(name)
        
        return None

    
    
    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
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
                
                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformation function for {self.name}
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

        elif callable(func):
            function = func
            function_list = [func]


        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None



class Transformers:
    """
    Super class for the collections of transformers
    """

    def __init__(self,
    ):

        return None








