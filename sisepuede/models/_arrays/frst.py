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

class ArraysFRST(ma.SubsectorArraysCollection):
    """Store arrays for FRST Calculations
    """

    def __init__(self,
        df_trajectories: pd.DataFrame,
        model_attributes: 'ModelAttributes',
        **kwargs,
    ) -> None:

        # get the subsector name
        subsec_name = model_attributes.subsec_name_frst

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
        """Initialize FRST arrays that are carried through
        """

        # emission factor--CH4 from methane emission
        self.get_modvar_array(
            df_trajectories,
            self.modvar_frst_ef_ch4,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
            set_property = True,
        )

        # fraction of forest C available from land use conversions away from forests
        self.get_modvar_array(
            df_trajectories,
            self.modvar_frst_frac_c_converted_available,
            set_property = True,
            var_bounds = (0, 1),
        )

        # fraction of c per dry matter in forests
        self.get_modvar_array(
            df_trajectories,
            self.modvar_frst_frac_c_per_dm,
            override_vector_for_single_mv_q = False, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # priority fraction for removals from young forests
        self.get_modvar_array(
            df_trajectories,
            self.modvar_frst_bcl_frac_rmv_priority_yf,
            override_vector_for_single_mv_q = False, 
            set_property = True,
            var_bounds = (0, 1),
        )
       

        return None






