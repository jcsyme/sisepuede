import numpy as np
import sisepuede.utilities._toolbox as sf
from typing import *




##########################
#    GLOBAL VARIABLES    #
##########################

# error classes
class UndefinedErrorFunction(Exception):
    pass



#######################
#    PRIMARY CLASS    #
#######################

class ErrorFunctions:
    """Collection of error functions to use in calibration.
    """

    def __init__(self,
    ) -> None:

        self._initialize_error_functions()

        return None


    def __call__(self,
        name: str,
        *args,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """Call error function with name name, passing args and kwargs to the
            error function.
        """

        func = self.get_error_function(name, )
        out = func(*args, **kwargs, )

        return out
    


    def _initialize_error_functions(self,
    ) -> None:
        """Initialize the error functions that are available and valid
        """
        
        dict_error_functions = {
            "proportional_deviation": self._error_proportional_deviation,
        }

        error_functions = sorted(
            list(
                dict_error_functions.keys()
            )
        )

        
        ##  SET PROPERTIES

        self.dict_error_functions = dict_error_functions
        self.error_functions = error_functions

        return None


    
    def _error_proportional_deviation(self,
        target: Union[float, np.ndarray],
        current: Union[float, np.ndarray],
        return_type: str = "vector",
        signed: bool = False, 
    ) -> float:
        """Calculate the proportional deviation:
        
        np.abs((current - target)/target)

        Function Arguments
        ------------------
        target : Union[float, np.ndarray]
            Target number or vector to align to
        current : Union[float, np.ndarray]
            Current value

        Keyword Arguments
        -----------------
        return_type : str
            If target and current are both numbers, then the function returns
            the original number. Otherwise, this keyword argument governs the
            return type (i.e.g, if either target or current are vectors). Note 
            that is signed = True, then min error could have the maximum
            magnitude, and vis-versa.

            * "max":        Return the max error
            * "min":        Return the min error
            * "vector":     Return the vector of inputs. If target and current
                            are both numbers, returns the original number.
            
        signed : bool
            If False, takes absolute value of error. Otherwise, retains 
            directionality (<0 => current < target, >0 => current > target)
        """
        out = (current - target)/target
        if not signed:
            out = np.abs(out)

        if return_type == "vector":
            return out
    
        out = (
            out.max() if return_type == "max" else out.min()
        )
    
        return out



    def get_error_function(self,
        name: str,
        stop_on_missing: bool = True, 
    ) -> callable:
        """Get an error function
        """

        func = self.dict_error_functions.get(name)
        if (func is None) and stop_on_missing:
            raise UndefinedErrorFunction("Function '{name}' not found")
        
        return func