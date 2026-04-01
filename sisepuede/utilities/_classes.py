import numpy as np
import pandas as pd
import sisepuede.utilities._toolbox as sf
from typing import *


##########################
#    GLOBAL VARIABLES    #
##########################

# error classes
class SimplexBoundsError(Exception):
    pass

# TimeSeriesSimplexShifter types
_TSSS_TYPE_FRACTION_OF_SOURCE = "fraction_of_source"
_TSSS_TYPE_VECTOR_SCALAR = "scalar_vectors"





#######################
#    BUILD CLASSES    #
#######################


class TimeSeriesSimplexShifter:
    """Shift fractions in a time series.

    In all descriptions below, let

        * T:        number of time periods
        * n:        simplex dimension (number of categories)

    Allows for shifting of fractions from a source group to a target group 
        based on one of the following types:

        * fraction_of_source:
            Shift a fraction of source mass into target mass. If specifying,
            must specify target mass allocations for shifted mass. If None is 
            provided, will allocate proportionally to target mass.

            REQUIRED ARGUMENTS
            ------------------
            * source fractions                  # fraction of source mass 
                                                #   categories to be shifted
            * target allocations                # allocation fractions for
                                                #   target mass categories
                                                #   receiving shifted mass
        * scalar_vectors:
            Using a vector of scaling, adjust source mass categories and 
            allocated to target fuels accordingly.

            REQUIRED ARGUMENTS
            ------------------
            * dict_source_vector_scalars        # dictionary mapping source 
                                                #   indices (column) to scalars
                                                #   to apply. If a scalar is a
                                                #   single number, then that 
                                                #   number is applied uniformly.
                                                #   If it is a vector, it is 
                                                #   used to scale the source 
                                                #   category.
            * arr_target_allocations            # array (T x n) mapping target 
                                                #   indices to allocated share.
                                                #   Row sums must equal 1. If 
                                                #   None, then all non-source
                                                #   categories are targeted. 

        

        
    Initialization Arguments
    ------------------------
    array_base : np.narray (T x n)
        Baseline array of
    """

    def __init__(self,
        array_base: np.ndarray,
        thresh_correction: float = 1e-6,
    ) -> None:

        self._initialize_array_base(
            array_base,
            thresh_correction,
        )

        return None
    

    def _initialize_array_base(self,
        array_base: np.ndarray,
        thresh_correction: float,
    ) -> None:
        """Initialize the base array of share vectors
        """
        
        # check that everything is positive
        if array_base.min() < 0:
            raise ValueError(f"Values for time series array_base in TimeSeriesSimplexShifter cannot be negative.")

        # verify sums
        array_base = sf.check_row_sums(
            array_base, 
            msg_pass = "array_base in TimeSeriesSimplexShifter",
            thresh_correction = thresh_correction,
        )


        ##  SET PROPERTIES

        self.T = array_base.shape[0]
        self.n = array_base.shape[1]
        self.array_base = array_base

        return None



    def _initialize_magnitude_types(self,
    ) -> None:
        """Initialize magnitude types that can be passed.
        """

        magnitude_types = [
            _TSSS_TYPE_VECTOR_SCALAR
        ]

        
        ##  SET PROPERTIES

        self.magnitude_type_scalar_vector = _TSSS_TYPE_VECTOR_SCALAR
        self.magnitude_types = magnitude_types

        return None
    

    

    ################################
    #    VERIFICATION FUNCTIONS    #
    ################################

    def _check_arr_target_allocations(self,
        arr_target_allocations: np.ndarray,
    ) -> None:
        """Check the allocation array for type and shape
        """
        # check type
        if not isinstance(arr_target_allocations, np.ndarray):
            tp = str(type(arr_target_allocations))
            raise TypeError(f"Invalid type '{tp}' specified: must be numpy array.")

        # check shape
        if arr_target_allocations.shape != self.array_base.shape:
            msg = f"""Invalid shape {arr_target_allocations.shape} specified for arr_target_allocations:
            Must have shape {self.array_base.shape}"""
            raise RuntimeError(msg)

        # check min value
        if arr_target_allocations.min() < 0:
            msg = arr_target_allocations.min()
            msg = f"arr_target_allocations has minimum {msg} < 0. All values must be >= 0."
            raise ValueError(msg)

        return None
    



    #####################################################################
    #    END-STATE FUNCTIONS                                            #
    #    -------------------------------                                #
    #    - functions that estimate the end-states for sources and/or    #     
    #       targets, depending on type.                                 #
    #                                                                   #
    #####################################################################

    def _filter_dict_source_vector_scalars(self,
        dict_source_vector_scalars: Dict[int, Union[float, np.ndarray]],
        stop_on_bounds_error: bool = False,
        unsafe: bool = False,
    ) -> Dict[int, np.ndarray]:
        """Filter the dict_source_vector_scalars dictionary to make sure it 
            maps an integer to a numpy array. 
            
            * If `stop_on_bounds_error`, will throw an error if any scalars 
                are < 0. 
            * If `unsafe`, will simply pass dict_source_vector_scalars.
        """

        if unsafe:
            return dict_source_vector_scalars
        

        dict_out = {}

        for k, v in dict_source_vector_scalars.items():

            # conditions that need to be met--index
            skip = not sf.isnumber(k, integer = True, )
            skip |= ((k < 0) | (k >= self.n)) if not skip else skip            
            if skip: continue

            # conditions that need to be met--specification
            vec = v*np.ones(self.T, ) if sf.isnumber(v) else v
            skip = not isinstance(vec, np.ndarray)
            skip |= (vec.shape != (self.T, )) if not skip else skip
            if skip: continue

            # check scalars
            if vec.min() < 0: 
                if stop_on_bounds_error:
                    raise SimplexBoundsError(f"Unable to apply scalar at index {k}: negative scalars found.")
                continue

            dict_out.update({k: vec, })

        return dict_out
    


    def _get_end_state_vectors_sources_sv(self,
        dict_source_vector_scalars: Dict[int, np.ndarray],
        normalize_exceedance: bool = True,
        pass_dict_svs_unsafe: bool = False,
        stop_on_bounds_error: bool = False, 
    ) -> Union[np.ndarray, None]:
        """Get the end-state vectors for SOURCE vectors in scalar_vectors shift
            type. Returns an array of size T x n where only SOURCE vectors are 
            set. All others are ignored.
        
        Function Arguments
        ------------------
        dict_source_vector_scalars : Dict[int, np.ndarray]
            Dictionary mapping a column index to a scalar vector to apply to 
            that index.

        Keyword Arguments
        -----------------
        normalize_exceedance : bool 
            Normalize total targets that exceed one? 
            * True:     If end states that are scaled exceed 1, they will be 
                        normalized to ensure that the simplex is preserved
            * False:    If end states exceed 1 in total, an error is thrown.  
        pass_dict_svs_unsafe : bool
            * True:     Pass `unsafe` to _filter_dict_source_vector_scalars();
                        only should be done if the dictionary is known to have 
                        been checked previously
            * False:    Verify dictionary elements
        stop_on_bounds_error : bool
            Stop if negative scalars are found? If False, skips.
        """

        if not isinstance(dict_source_vector_scalars, dict):
            return None
        

        # initialize array and filter the dictionary 
        arr_out = np.zeros(self.array_base.shape, )
        dict_svs = self._filter_dict_source_vector_scalars(
            dict_source_vector_scalars,
            stop_on_bounds_error = stop_on_bounds_error,
            unsafe = pass_dict_svs_unsafe,
        )
        
        # apply scalars to each specified column; the output of 
        #   _filter... is a dictionary mapping an index to a numpy array
        for k, v in dict_svs.items():
            arr_out[:, k] = self.array_base[:, k]*v
        
        
        ##  RETURN AND FINAL CHECKS

        # check 
        if normalize_exceedance:
            arr_out = np.nan_to_num(
                sf.vector_limiter(arr_out, (0, 1)),
                nan = 0.0,
                posinf = 0.0,
            )

            return arr_out

        # otherwise, check sums and throw an error if any issues arise
        vec_sum = arr_out.sum(axis = 1)
        w = np.where((vec_sum > 1) | (vec_sum < 0))[0]
        if len(w) > 0:
            msg = f"Some target mass fractions are either > 1 or < 0. Check the scalars provided or allow normalize_exceedance."
            raise SimplexBoundsError(msg)
        

        return arr_out
    


    def _get_allocation_vectors_targets_sv(self,
        arr_target_allocations: np.ndarray,
        dict_source_vector_scalars: Dict[int, np.ndarray],
        allow_self_shifts: bool = False,
        pass_dict_svs_unsafe: bool = False,
        stop_on_bounds_error: bool = False,
    ) -> np.ndarray:
        """Get allocation vectors for targets in scalar shift. Checks 
            specification
        
        Function Arguments
        ------------------
        arr_target_allocations : np.ndarray
            Numpy array (T x n) giving shares for target shift. 
        dict_source_vector_scalars : Dict[int, np.ndarray]
            Dictionary mapping a column index to a scalar vector to apply to 
            that index.

        Keyword Arguments
        -----------------
        allow_self_shifts : bool
            Allow the allocation target to include shifts into itself? 
            * True:     Shifts out can be partially allocated back into source 
                        categories.
            * False:    Any specified shifts into source categories are
                        eliminated and allocations are re-normalized
        pass_dict_svs_unsafe : bool
            * True:     Pass `unsafe` to _filter_dict_source_vector_scalars();
                        only should be done if the dictionary is known to have 
                        been checked previously
            * False:    Verify dictionary elements
        stop_on_bounds_error : bool
            Stop if negative scalars are found? If False, skips.
        """

        # check the array specification
        self._check_arr_target_allocations(arr_target_allocations, )

        # get source vectors scalings
        dict_svs = self._filter_dict_source_vector_scalars(
            dict_source_vector_scalars,
            stop_on_bounds_error = stop_on_bounds_error,
            unsafe = pass_dict_svs_unsafe,
        )
        arr_out = arr_target_allocations.copy()

        # check
        if not allow_self_shifts:
            arr_out[:, list(dict_svs.keys())] = 0
        

        # since _check_arr_target_allocations ensures no negative numbers
        # we can assume that all are >= 0
        vec_sums = arr_out.sum(axis = 1 , )
        w = np.where(vec_sums == 0, )[0]
        arr_out[w, :] = 1                   # will renormalize below; creates homogenous
        
        # update again 
        if (len(w) > 0) and not allow_self_shifts:
            arr_out[:, list(dict_svs.keys())] = 0

        # normalize
        arr_out = sf.check_row_sums(
            arr_out,
            thresh_correction = None, 
        )

        return arr_out
    



    #################################
    #    MASS-SHIFTING FUNCTIONS    #
    #################################

    def shift_mass_scalar_vectors(self,
        arr_target_allocations: np.ndarray,
        dict_vec_scalars: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Perform the `scalar_vectors` shift

        Function Arguments
        ------------------
        arr_target_allocations : np.ndarray
            Numpy array (T x n) giving shares for target shift. 
        dict_source_vector_scalars : Dict[int, np.ndarray]
            Dictionary mapping a column index to a scalar vector to apply to 
            that index.
        """

        ##  INITIALIZATION

        # check the vector of scalars
        dict_vec_scalars = self._filter_dict_source_vector_scalars(
            dict_vec_scalars,
        )

        # get end states (after shift)
        arr_end_states = self._get_end_state_vectors_sources_sv(
            dict_vec_scalars,
            pass_dict_svs_unsafe = True,    # This dictionary has already been checked
        )

        # get allocations for shift out
        arr_target_allocations = self._get_allocation_vectors_targets_sv(
            arr_target_allocations,
            dict_vec_scalars,
            pass_dict_svs_unsafe = True,    # This dictionary has already been checked
        )
        

        ##  PERFORM SHIFT
        
        arr_shifted = self.array_base.copy()
        arr_target_allocations_tr = arr_target_allocations.transpose()
        
        for k, v in dict_vec_scalars.items():
        
            vec_shift = self.array_base[:, k] - arr_end_states[:, k]
        
            # update 
            arr_shifted += (arr_target_allocations_tr*vec_shift).transpose()
            arr_shifted[:, k] = arr_end_states[:, k].copy()

        return arr_shifted



