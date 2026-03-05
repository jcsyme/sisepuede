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

class ArraysLVST(ma.SubsectorArraysCollection):
    """Store arrays for AGRC Calculations
    """

    def __init__(self,
        df_trajectories: pd.DataFrame,
        model_attributes: 'ModelAttributes',
        **kwargs,
    ) -> None:

        # get the subsector name
        subsec_name = model_attributes.subsec_name_lvst

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
        """Initialize LVST arrays that are carried through
        """
        
        self._initialize_arrays_lvst_diet(df_trajectories, )
        

        return None


    def _initialize_arrays_lvst_diet(self,
        df_trajectories: pd.DataFrame,
    ) -> None:
        """Initialize livestock diet related arrays
        """

        ##  GET DIETARY BOUNDS
        
        modvars_lvst_diets_ordered = [
            self.modvar_lvst_frac_diet_max_from_crop_residues,
            self.modvar_lvst_frac_diet_max_from_crops_cereals,
            self.modvar_lvst_frac_diet_max_from_crops_non_cereals,
            self.modvar_lvst_frac_diet_max_from_pastures,
            self.modvar_lvst_frac_diet_min_from_crop_residues,
            self.modvar_lvst_frac_diet_min_from_crops_cereals,
            self.modvar_lvst_frac_diet_min_from_crops_non_cereals,
            self.modvar_lvst_frac_diet_min_from_pastures
        ]

        if not isinstance(df_trajectories, pd.DataFrame):

            ##  BUILD EMPTY DEFAULTS

            dict_var_bounds = dict(
                (
                    k,
                    self.get_modvar_array(
                        df_trajectories,
                        k,
                        expand_to_all_cats = True,
                    )
                )
                for k in modvars_lvst_diets_ordered
            )
        
        else:
            # get maximum bound variables
            dict_var_bounds = (
                self
                .model_attributes
                .get_multivariables_with_bounded_sum_by_category(
                    df_trajectories,
                    modvars_lvst_diets_ordered[0:4],
                    4.0,
                    expand_to_all_cats = True,
                    msg_append = "in assigning dietary fraction bound variables.",
                )
            )

            # get minimium bound variables (must be  <=1 to allow for a solution, preferably << 1)
            dict_var_bounds.update(
                self
                .model_attributes
                .get_multivariables_with_bounded_sum_by_category(
                    df_trajectories,
                    modvars_lvst_diets_ordered[4:],
                    1.0,
                    expand_to_all_cats = True,
                    msg_append = "in assigning dietary fraction bound variables.",
                )
            )

        
        # set properties for these arrays
        for k, v in dict_var_bounds.items():
            name_property = self.get_property_name_array(k, )
            setattr(self, name_property, v, )
        

        ##  GET FEED TO MASS RATIOS

        arr_lvst_factor_feed_to_mass = self.get_modvar_array(
            df_trajectories,
            self.modvar_lvst_factor_feed_to_mass,
            expand_to_all_cats = True,
            var_bounds = (0, 1), 
        )


        # get name and set
        name_property = self.get_property_name_array(
            self.modvar_lvst_factor_feed_to_mass,
        )
        setattr(self, name_property, arr_lvst_factor_feed_to_mass, )

        return None

        
        





