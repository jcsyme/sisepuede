"""Set up some classes for extraction. Helps clean up arrays in each model 
    class. In general, extraction from input DataFrames is performed here for
    base classes, and additional support arrays can be stored here.
"""

import numpy as np
import pandas as pd
from typing import *

import sisepuede.core.model_attributes as ma 
from sisepuede.core.model_variable import *



#################################
###                           ###
###    BUILD ARRAY CLASSES    ###
###                           ###
#################################

class ArraysLNDU(ma.SubsectorArraysCollection):
    """Store arrays for LNDU Calculations
    """

    def __init__(self,
        df_trajectories: pd.DataFrame,
        model_attributes: 'ModelAttributes',
        **kwargs,
    ) -> None:

        # get the subsector name
        subsec_name = model_attributes.subsec_name_lndu

        # initialize core properties
        super().__init__(
            model_attributes,
            subsec_name,
            **kwargs,
        )

        self._initialize_arrays(df_trajectories, )
        
        return None

    

    def _initialize_arrays(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize LNDU arrays that are carried through
        """
        

        # pasture yield, average
        self.get_modvar_array(
            df_trajectories,
            self.modvar_lndu_yf_pasture_avg,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        # pasture yield, supremum
        self.get_modvar_array(
            df_trajectories,
            self.modvar_lndu_yf_pasture_sup,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        # vegetarian dietary exchange scalar
        self.get_modvar_array(
            df_trajectories,
            self.modvar_lndu_vdes,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        return None







