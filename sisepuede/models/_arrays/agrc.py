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

        self._initialize_arrays_residue_removal_pathways(
            df_trajectories, 
        )

        self._initialize_arrays_standard(
            df_trajectories, 
        )

        return None



    def _initialize_arrays_residue_removal_pathways(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize livestock diet related arrays.
        """

        ##  GET DIETARY BOUNDS
        
        modvars_agrc_residue_pathways = [
            self.modvar_agrc_frac_residues_burned,
            self.modvar_agrc_frac_residues_removed_for_energy,
            self.modvar_agrc_frac_residues_removed_for_feed
        ]

        if not isinstance(df_trajectories, pd.DataFrame):

            ##  BUILD EMPTY DEFAULTS

            dict_residue_pathways = dict(
                (
                    k,
                    self.get_modvar_array(
                        df_trajectories,
                        k,
                        expand_to_all_cats = True,
                    )
                )
                for k in modvars_agrc_residue_pathways
            )
        
        else:
            # get maximum bound variables
            dict_residue_pathways = (
                self
                .model_attributes
                .get_multivariables_with_bounded_sum_by_category(
                    df_trajectories,
                    modvars_agrc_residue_pathways,
                    1.0,
                    force_sum_equality = False,
                    msg_append = "in assigning dietary fraction bound variables.",
                )
            )

        
        # set properties for these arrays
        for k, v in dict_residue_pathways.items():
            name_property = self.get_property_name_array(k, )
            setattr(self, name_property, v, )
        
        self.dict_residue_pathways = dict_residue_pathways

        return None
    


    def _initialize_arrays_standard(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize AGRC arrays that are carried through
        """

        # residue production, bagasse
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_bagasse_yield_factor,
            expand_to_all_cats = True,
            set_property = True,
        )

        # combustion factor for residues
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_combustion_factor,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # emission factor--ch4, residue burning
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ef_ch4_burning,
            return_type = "array_units_corrected",
            set_property = True,
        )

        # emission factor--ch4, crop decomposition
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ef_ch4_crop_decomposition,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
            set_property = True,
        )

        # emission factor--co2 from biomass
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ef_co2_biomass,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
            set_property = True,
        )

        # emission factor--n2o, residue burning
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ef_n2o_burning,
            return_type = "array_units_corrected",
            set_property = True,
        )

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
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # fraction animal field
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_animal_feed,
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, 1),
        )

        # import fraction
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_demand_imported,
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, 1),
        )

        # dry matter fraction of crops
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_dry_matter_in_crop,
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, 1),
        )

        # fracion of each crop type under no-till
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_no_till,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, 1),
        )

        # production fraction lost
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_frac_production_lost,
            set_property = True,
            var_bounds = (0, 1),
        )

        # gravimetric energy density of different residue types
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ged_residues,
            expand_to_all_cats = True,
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # N content of above-ground residues
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_n_content_of_above_ground_residues,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
            var_bounds = (0, 1),
        )

        # N content of below-ground residues
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_n_content_of_below_ground_residues,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
            var_bounds = (0, 1),
        )
        
        # ratio of below-ground biomass to above-ground biomass
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
            var_bounds = (0, np.inf),
        )

        # residue production regression b
        self.get_modvar_array(
            df_trajectories, 
            self.modvar_agrc_regression_b_above_ground_residue, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # residue production regression m
        self.get_modvar_array(
            df_trajectories, 
            self.modvar_agrc_regression_m_above_ground_residue, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            set_property = True,
        )

        # residue final use -- burned
        self.get_modvar_array(
            None,
            self.modvar_agrc_residue_final_use_burned,
            expand_to_all_cats = True,
            set_property = True,
        )

        # residue final use -- energy
        self.get_modvar_array(
            None,
            self.modvar_agrc_residue_final_use_energy,
            expand_to_all_cats = True,
            set_property = True,
        )

        # residue final use -- feed
        self.get_modvar_array(
            None,
            self.modvar_agrc_residue_final_use_feed,
            expand_to_all_cats = True,
            set_property = True,
        )

        # residue final use -- field
        self.get_modvar_array(
            None,
            self.modvar_agrc_residue_final_use_field,
            expand_to_all_cats = True,
            set_property = True,
        )

        # yield factors
        self.get_modvar_array(
            df_trajectories,
            self.modvar_agrc_yf,
            set_property = True,
        )


        return None





