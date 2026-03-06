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

class ArraysAGRC(ma.SubsectorArraysCollection):
    """Store arrays for AGRC Calculations
    """

    def __init__(self,
        df_trajectories: pd.DataFrame,
        model_attributes: 'ModelAttributes',
        **kwargs,
    ) -> None:

        # get the subsector name
        subsec_name = model_attributes.subsec_name_agrc

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
        """Initialize AGRC arrays that are carried through
        """

        self._initialize_arrays_agrc_demand_and_trade(df_trajectories, )
        self._initialize_arrays_agrc_yield(df_trajectories, )

        return None



    def _initialize_arrays_agrc_demand_and_trade(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize agriculture demand, export, import, and production related 
            arrays.
        """

        # elasticities of AGRC demand to gdp/gapita
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_elas_crop_demand_income,
            set_property = True,
        )

        # exports, unadjusted 
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_equivalent_exports,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # fraction animal field
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_animal_feed,
            set_property = True,
            var_bounds = (0, 1),
        )

        # import fraction
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_demand_imported,
            set_property = True,
            var_bounds = (0, 1),
        )

        # vegetarian dietary exchange scalar
        self.get_modvar_array(
            df_trajectories,
            self.modvar_lndu_vdes,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )


        return None



    def _initialize_arrays_agrc_yield(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize agriculture demand, export, import, and production related 
            arrays.
        """

        # yield factors
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_yf,
            set_property = True,
        )
        
        return None






