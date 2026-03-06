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
        

        return None


    def _initialize_arrays_lndu_(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize land use...
        """

        return None






