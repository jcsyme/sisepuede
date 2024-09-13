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
    """
    
    def __init__(self,
        code: str,
        func: Union[Callable, List[Callable]],
        attr_transfomer: Union[AttributeTable, None],
        code_baseline: str = "TX:BASE",
        field_transformer_id: str = "transformer_id",
        field_transformer_name: str = "transformer",
    ) -> None:

        self._initialize_function(func)
        self._initialize_code(
            code, 
            code_baseline,
            attr_transfomer, 
            field_transformer_id,
            field_transformer_name,
        )

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

        elif callable(func):
            function = func
            function_list = [func]


        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None










