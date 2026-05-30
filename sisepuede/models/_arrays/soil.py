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

class ArraysSOIL(ma.SubsectorArraysCollection):
    """Store arrays for AGRC Calculations
    """

    def __init__(self,
        df_trajectories: pd.DataFrame,
        model_attributes: 'ModelAttributes',
        **kwargs,
    ) -> None:

        # get the subsector name
        subsec_name = model_attributes.subsec_name_soil

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
        """Initialize SOIL arrays that are carried through
        """
        
        self._initialize_arrays_standard(
            df_trajectories, 
        )

        return None



    def _initialize_arrays_standard(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize SOIL arrays that are carried through
        """

        # fertilizer N demand scalar (can be used to increase/decrease)
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_demscalar_fertilizer,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # liming demand scalar (can be used to increase/decrease)
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_demscalar_liming,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # EF1 N2O emission factor, organic fertilizer
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef1_n_managed_soils_org_fert,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )
        
        # EF1 N2O emission factor for rice
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef1_n_managed_soils_rice,
            set_property = True,
        )

        # EF1 N2O emission factor, synthetic fertilizer
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef1_n_managed_soils_syn_fert,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # EF2 N2O emission factor in organic soils
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef2_n_organic_soils,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # EF3 N emission factor for pasture/range/paddock
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef3_n_prp,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # EF4 N emission factor for volatilisation
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef4_n_volatilisation,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # EF5 N emission factor for leaching
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef5_n_leaching,
            set_property = True,
        )

        # EF for C - dolomite
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef_c_liming_dolomite,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # EF for C - limestone
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef_c_liming_limestone,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # EF for C - urea
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ef_c_urea,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # Initial synthetic fertilizer use demand
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_fertuse_init_synthetic,
            set_property = True,
        )

        # fraction of N lost to leaching
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_frac_n_lost_leaching,
            set_property = True,
            var_bounds = (0, 1),
        )

        # fraction of N lost to volatilisation of organic N
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_frac_n_lost_volatilisation_on,
            override_vector_for_single_mv_q = False, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # fraction of N lost to volatilisation of synthetic N, non-urea
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea,
            override_vector_for_single_mv_q = False, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # fraction of N lost to volatilisation of synethetic N, urea
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_frac_n_lost_volatilisation_sn_urea,
            override_vector_for_single_mv_q = False, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # Fraction of synthetic fertilizer from Urea
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_frac_synethic_fertilizer_urea,
            set_property = True,
            var_bounds = (0, 1), 
        )

        # organic C stocks
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_organic_c_stocks,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            set_property = True,
        )

        # initial quantity of dolomite use
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_qtyinit_liming_dolomite,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # initial quantity of limestone use
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_qtyinit_liming_limestone,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # ratio of C to N in SOC
        self.get_modvar_array(
            df_trajectories,
            self.modvar_soil_ratio_c_to_n_soil_organic_matter,
            set_property = True,
        )

        return None





