####################################
    #    AGRC TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_agrc_decrease_exports(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.5,
        magnitude_type: str = "baseline_scalar",
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Decrease Exports" AGRC transformer on input DataFrame df_input (reduce by 50%)

        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of decrease in exports. If using the default value of `magnitude_type == "scalar"`, this magnitude will scale the final time value downward by this factor. 
            NOTE: If magnitude_type changes, then the behavior of the trasnformation will change.
        magnitude_type: str
            Type of magnitude, as specified in `transformers.lib.general.transformations_general`. See `?transformers.lib.general.transformations_general` for more information on the specification of magnitude_type for general transformer values. 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.5)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_afolu.modvar_agrc_equivalent_exports: {
                    "bounds": (0.0, 1.0),
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
                    "vec_ramp": self.vec_implementation_ramp
                },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def _trfunc_agrc_expand_conservation_agriculture(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_categories_to_magnitude: Union[Dict[str, float], None] = None,
        magnitude_burned: float = 0.0,
        magnitude_removed: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Expand Conservation Agriculture" AGRC transformer on input DataFrame df_input. 
            
        NOTE: Sets a new floor for F_MG (as described in in V4 Equation 2.25 (2019R)) to reduce losses of soil organic carbon through no-till in cropland + reduces removals and burning of crop residues, increasing residue covers on fields.
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_categories_to_magnitude: Union[Dict[str, float], None]
            Conservation agriculture is practically applied to only select crop types. Use the dictionary to map SISEPUEDE crop categories to target implementation magnitudes.
            * If None, maps to the following dictionary:

                {
                    "cereals": 0.8,
                    "fibers": 0.8,
                    "other_annual": 0.8,
                    "pulses": 0.5,
                    "tubers": 0.5,
                    "vegetables_and_vines": 0.5,
                }
                
        magnitude_burned: float
            Target fraction of residues that are burned
        magnitude_removed: float
            Maximum fraction of residues that are removed
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        # specify dictionary
        dict_categories_to_magnitude = (
            {
                "cereals": 0.8,
                "fibers": 0.8,
                "other_annual": 0.8,
                "pulses": 0.5,
                "tubers": 0.5,
                "vegetables_and_vines": 0.5,
            }
            if not isinstance(dict_categories_to_magnitude, dict)
            else dict_categories_to_magnitude
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # COMBINES SEVERAL COMPONENTS - NO TILL + REDUCTIONS IN RESIDUE REMOVAL AND BURNING
        
        # 1. increase no till
        df_out = tba.transformation_agrc_increase_no_till(
            df_input,
            dict_categories_to_magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        # 2. reduce burning and removals
        df_out = tbg.transformation_general(
            df_out,
            self.model_attributes,
            {
                self.model_afolu.modvar_agrc_frac_residues_burned: {
                    "bounds": (0.0, 1.0),
                    "magnitude": magnitude_burned,
                    "magnitude_type": "final_value",
                    "vec_ramp": vec_implementation_ramp
                },

                self.model_afolu.modvar_agrc_frac_residues_removed: {
                    "bounds": (0.0, 1.0),
                    "magnitude": magnitude_removed,
                    "magnitude_type": "final_value_ceiling",
                    "vec_ramp": vec_implementation_ramp
                },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def _trfunc_agrc_improve_crop_residue_management(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_burned: float = 0.0,
        magnitude_removed: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Crop Management" AGRC transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude_burned: float
            Target fraction of residues that are burned
        magnitude_removed: float
            Maximum fraction of residues that are removed
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude_burned = (
            0.0 
            if not isinstance(magnitude_burned, float) 
            else magnitude_burned
        )
        magnitude_removed = (
            0.0 
            if not isinstance(magnitude_removed, float) 
            else magnitude_removed
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

       
        df_out = tba.transformation_agrc_improve_crop_residue_management(
            df_input,
            magnitude_burned, # stop burning crops
            magnitude_removed, #remove 95%
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_agrc_improve_rice_management(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.45,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Rice Management" AGRC transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Minimum target fraction of rice production under improved management.
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.45)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )
        
        df_out = tba.transformation_agrc_improve_rice_management(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out


    
    def _trfunc_agrc_increase_crop_productivity(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, Dict[str, float]] = 0.2,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Crop Productivity" AGRC transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: Union[float, Dict[str, float]]
            Magnitude of productivity increase to apply to crops (e.g., a 20% increase is entered as 0.2); can be specified as one of the following
            * float: apply a single scalar increase to productivity for all crops
            * dict: specify crop productivity increases individually, with each key being a crop and the associated value being the productivity increase
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = (
            0.2 
            if not (isinstance(magnitude, float) | isinstance(magnitude, dict)) 
            else magnitude
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tba.transformation_agrc_increase_crop_productivity(
            df_input,
            magnitude, # can be specified as dictionary to affect different crops differently 
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_agrc_reduce_supply_chain_losses(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Supply Chain Losses" AGRC transformer on input DataFrame df_input. 
        
        Parameters
        ----------        
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of reduction in supply chain losses. Specified as a fraction (e.g., a 30% reduction is specified as 0.3)
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.3)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tba.transformation_agrc_reduce_supply_chain_losses(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out


    
    ###########################################
    #    FRST (LNDU) TRANSFORMER FUNCTIONS    #
    ###########################################

    def _trfunc_lndu_increase_reforestation(self,
        df_input: Union[pd.DataFrame, None] = None,
        cats_inflow_restriction: Union[List[str], None] = ["croplands", "other"],
        magnitude: float = 0.2,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Reforestation" FRST transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        cats_inflow_restriction: Union[List[str], None]
            LNDU categories to allow to transition into secondary forest; don't specify categories that cannot be reforested 
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional increase in secondary forest area to specify; for example, a 10% increase in secondary forests is specified as 0.1
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.2)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tba.transformation_frst_increase_reforestation(
            df_input, 
            magnitude, # double forests INDIA
            vec_implementation_ramp,
            self.model_attributes,
            cats_inflow_restriction = cats_inflow_restriction, # SET FOR INDIA--NEED A BETTER WAY TO DETERMINE
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_lndu_stop_deforestation(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.99999,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Stop Deforestation" FRST transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of final primary forest transition probability
            into itself; higher magnitudes indicate less deforestation.
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.99999)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        ##  BEGIN modify ramp to be a binary/start in another year HEREHERE - TEMP  ##
        vec_ramp = np.array(
            [float(int(x > 0)) for x in vec_implementation_ramp]
        )
        w = np.where(vec_ramp == 1)[0][0]
        vec_ramp = np.array(
            [
                float(sf.vec_bounds((x - (w - 1))/5, (0, 1))) # start in 2040
                for x in range(len(vec_implementation_ramp))
            ]
        )
        ##  END ##

        df_out = tba.transformation_frst_reduce_deforestation(
            df_input,
            magnitude,
            vec_ramp,#self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    ####################################
    #    LNDU TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_lndu_expand_silvopasture(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.1,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Increase the use of silvopasture by shifting pastures to secondary forest. 
            
        NOTE: This transformer relies on modifying transition matrices, which can compound some minor numerical errors in the crude implementation taken here. Final area prevalences may not reflect get_matrix_column_scalarget_matrix_column_scalarprecise shifts.
        
        Parameters
        ---------- 
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            magnitude of increase in fraction of pastures subject to 
            silvopasture
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )
        

        df_out = tba.transformation_lndu_increase_silvopasture(
            df_input,
            magnitude, # CHANGEDFORINDIA - ORIG 0.1
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    
    

    def _trfunc_lndu_expand_sustainable_grazing(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Expand Sustainable Grazing" LNDU transformer on input DataFrame df_input. 
            
        NOTE: Sets a new floor for F_MG (as described in in V4 Equation 2.25 (2019R)) through improved grassland management (grasslands).
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fraction of pastures subject to improved pasture management. This value acts as a floor, so that if the existing value is greater than is specified by the transformation, the existing value will be maintained. 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.95)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_afolu.modvar_lndu_frac_pastures_improved: {
                    "bounds": (0.0, 1.0),
                    "magnitude": magnitude,
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def _trfunc_lndu_integrated_transitions(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_deforestation: Union[float, None] = None,
        magnitude_silvopasture: Union[float, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Increase the use of silvopasture by shifting pastures to secondary forest AND reduce deforestation. Sets orderering of these transformations for bundles.
            
        NOTE: This transformer relies on modifying transition matrices, which can compound some minor numerical errors in the crude implementation taken here. Final area prevalences may not reflect precise shifts.
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude_deforestation: float
            Magnitude to apply to deforestation (transition probability from primary forest into self). If None, uses default.
        magnitude_silvopasture: float
            Magnitude passed to silvopasture transformation. If None, uses silvopasture magnitude default. 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # silvopasture must come first
        df_out = self._trfunc_lndu_expand_silvopasture(
            df_input,
            magnitude = magnitude_silvopasture,
            strat = strat,
            vec_implementation_ramp = vec_implementation_ramp,
        )
        # then deforestation
        df_out = self._trfunc_lndu_stop_deforestation(
            df_out,
            magnitude = magnitude_deforestation,
            strat = strat,
            vec_implementation_ramp = vec_implementation_ramp,
        )
        
        return df_out


    
    def _trfunc_lndu_reallocate_land(self,
        df_input: Union[pd.DataFrame, None] = None,
        force: bool = False,
        magnitude: Union[float, None] = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement land use reallocation factor as a ramp.
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        force: bool
            If the baseline includes LURF > 0, then this transformer will not work by default to avoid double-implementation. Set force = True to force the transformer to further modify the LURF
        magnitude: float
            Land use reallocation factor value with implementation
            ramp vector
        strat: int
            Optional strategy index to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """

        # check input dataframe
        df_out = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.5)
    
        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # if baseline includes LURF, don't modify unless forced to do so
        if (not self.baseline_with_lurf) | force:
            df_out = tbg.transformation_general(
                df_out,
                self.model_attributes,
                {
                    self.model_afolu.modvar_lndu_reallocation_factor: {
                        "bounds": (0.0, 1),
                        "magnitude": magnitude,
                        "magnitude_type": "final_value",
                        "vec_ramp": vec_implementation_ramp
                    }
                },
                field_region = self.key_region,
                strategy_id = strat,
            )

        return df_out


        

    ####################################
    #    LSMM TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_lsmm_improve_manure_management_cattle_pigs(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_lsmm_pathways: Union[dict, None] = None,
        strat: Union[int, None] = None,
        vec_cats_lvst: Union[List[str], None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Livestock Manure Management for Cattle and Pigs" transformer on the input DataFrame.
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_lsmm_pathways: 
            Dictionary allocating treatment to LSMM categories as fractional targets (must sum to <= 1). If None, defaults to

            dict_lsmm_pathways = {
                "anaerobic_digester": 0.59375, # 0.625*0.95,
                "composting": 0.11875, # 0.125*0.95,
                "daily_spread": 0.2375, # 0.25*0.95,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_cats_lvst: Union[List[str], None]
            LVST categories receiving treatment in this transformation. Default (if None) is

            [
                "cattle_dairy",
                "cattle_nondairy",
                "pigs"
            ]

        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # allocation across manure management options
        frac_managed = 0.95
        dict_lsmm_pathways = (
            {
                "anaerobic_digester": 0.625*frac_managed,
                "composting": 0.125*frac_managed,
                "daily_spread": 0.25*frac_managed,
                #"solid_storage": 0.125*frac_managed
            }
            if not isinstance(dict_lsmm_pathways, dict)
            else self.model_attributes.get_valid_categories_dict(
                dict_lsmm_pathways,
                self.model_attributes.subsec_name_lsmm,
            )
        )
        
        # get categories to apply management paradigm to
        vec_lvst_cats = (
            [
                "cattle_dairy",
                "cattle_nondairy",
                "pigs",
            ]
            if not sf.islistlike(vec_cats_lvst)
            else self.model_attributes.get_valid_categories(
                list(vec_cats_lvst),
                self.model_attributes.subsec_name_lvst
            )
        )

        # 
        df_out = tba.transformation_lsmm_improve_manure_management(
            df_input,
            dict_lsmm_pathways,
            vec_implementation_ramp,
            self.model_attributes,
            categories_lvst = vec_lvst_cats,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_lsmm_improve_manure_management_other(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_lsmm_pathways: Union[dict, None] = None,
        strat: Union[int, None] = None,
        vec_cats_lvst: Union[List[str], None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Livestock Manure Management for Other Animals" LSMM transformer on the input DataFrame.
        
        Parameters
        ----------  
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_lsmm_pathways: Union[dict, None]
            Dictionary allocating treatment to LSMM categories as fractional targets (must sum to <= 1). If None, defaults to

            dict_lsmm_pathways = {
                "anaerobic_digester": 0.475, # 0.5*0.95,
                "composting": 0.2375, # 0.25*0.95,
                "dry_lot": 0.11875, # 0.125*0.95,
                "daily_spread": 0.11875, # 0.125*0.95,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_cats_lvst: Union[List[str], None]
            LVST categories receiving treatment in this transformation. Default (if None) is
            [
                "buffalo",
                "goats",
                "horses",
                "mules",
                "sheep",
            ]

        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # allocation across manure management options
        frac_managed = 0.95
        dict_lsmm_pathways = (
            {
                "anaerobic_digester": 0.50*frac_managed,
                "composting": 0.25*frac_managed,
                "dry_lot": 0.125*frac_managed,
                "daily_spread": 0.125*frac_managed,
            }
            if not isinstance(dict_lsmm_pathways, dict)
            else self.model_attributes.get_valid_categories_dict(
                dict_lsmm_pathways,
                self.model_attributes.subsec_name_lsmm,
            )
        )
        
        # get categories to apply management paradigm to
        vec_lvst_cats = (
            [
                "buffalo",
                "goats",
                "horses",
                "mules",
                "sheep",
            ]
            if not sf.islistlike(vec_cats_lvst)
            else self.model_attributes.get_valid_categories(
                list(vec_cats_lvst),
                self.model_attributes.subsec_name_lvst
            )
        )



        df_out = tba.transformation_lsmm_improve_manure_management(
            df_input,
            dict_lsmm_pathways,
            vec_implementation_ramp,
            self.model_attributes,
            categories_lvst = vec_lvst_cats,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        return df_out
    


    def _trfunc_lsmm_improve_manure_management_poultry(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_lsmm_pathways: Union[dict, None] = None,
        strat: Union[int, None] = None,
        vec_cats_lvst: Union[List[str], None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Livestock Manure Management for Poultry" LSMM transformer on the input DataFrame.
        
        Parameters
        ----------   
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_lsmm_pathways: Union[dict, None]
            Dictionary allocating treatment to LSMM categories as fractional targets (must sum to <= 1). If None, defaults to
            dict_lsmm_pathways = {
                "anaerobic_digester": 0.475, # 0.5*0.95,
                "poultry_manure": 0.475, # 0.5*0.95,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_cats_lvst: Union[List[str], None]
            LVST categories receiving treatment in this transformation. Default (if None) is
            [
                "chickens",
            ]

        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # allocation across manure management options
        frac_managed = 0.95
        dict_lsmm_pathways = (
            {
                "anaerobic_digester": 0.50*frac_managed,
                "poultry_manure": 0.5*frac_managed,
            }
            if not isinstance(dict_lsmm_pathways, dict)
            else self.model_attributes.get_valid_categories_dict(
                dict_lsmm_pathways,
                self.model_attributes.subsec_name_lsmm,
            )
        )
        
        # get categories to apply management paradigm to
        vec_lvst_cats = (
            [
                "chickens"
            ]
            if not sf.islistlike(vec_cats_lvst)
            else self.model_attributes.get_valid_categories(
                list(vec_cats_lvst),
                self.model_attributes.subsec_name_lvst
            )
        )
        

        df_out = tba.transformation_lsmm_improve_manure_management(
            df_input,
            dict_lsmm_pathways,
            vec_implementation_ramp,
            self.model_attributes,
            categories_lvst = vec_lvst_cats,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_lsmm_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, None] = 0.9,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Biogas Capture at Anaerobic Decomposition 
            Facilities" transformer on the input DataFrame.
        
        Parameters
        ----------
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]

        
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            target minimum fraction of biogas that is captured at
            anerobic decomposition facilities.
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.9)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )
            
        # update the biogas recovery factor
        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_afolu.modvar_lsmm_rf_biogas: {
                    "bounds": (0.0, 1),
                    "magnitude": magnitude, # CHANGEDFORINDIA 0.9
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    ####################################
    #    LVST TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_lvst_decrease_exports(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, None] = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Decrease Exports" LVST transformer on input DataFrame df_input (reduce by 50%)
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fractional reduction in exports applied directly to time
            periods (reaches 100% implementation when ramp reaches 1)
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.5)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_afolu.modvar_lvst_equivalent_exports: {
                    "bounds": (0.0, np.inf),
                    "magnitude": magnitude,
                    "magnitude_type": "baseline_scalar",
                    "vec_ramp": vec_implementation_ramp
                },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def _trfunc_lvst_increase_productivity(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, None] = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Livestock Productivity" LVST transformer on 
            input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fractional increase in productivity applied to carrying
            capcity for livestock
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.3)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tba.transformation_lvst_increase_productivity(
            df_input,
            magnitude, # CHANGEDFORINDIA 0.2
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    def _trfunc_lvst_reduce_enteric_fermentation(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_lvst_reductions: Union[dict, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Enteric Fermentation" LVST transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        - dict_lvst_reductions: dictionary allocating mapping livestock 
            categories to associated reductions in enteric fermentation. If 
            None, defaults to

            {
                "buffalo": 0.4,
                "cattle_dairy": 0.4,
                "cattle_nondairy": 0.4,
                "goats": 0.56,
                "sheep": 0.56
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        dict_lvst_reductions = (
            {
                "buffalo": 0.4, # CHANGEDFORINDIA 0.4
                "cattle_dairy": 0.4, # CHANGEDFORINDIA 0.4
                "cattle_nondairy": 0.4, # CHANGEDFORINDIA 0.4
                "goats": 0.56,
                "sheep": 0.56
            }
            if not isinstance(dict_lvst_reductions, dict)
            else dict_lvst_reductions

        )

        
        df_out = tba.transformation_lvst_reduce_enteric_fermentation(
            df_input,
            dict_lvst_reductions,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    ####################################
    #    SOIL TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_soil_reduce_excess_fertilizer(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, None] = 0.2,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Excess Fertilizer" SOIL transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fractional reduction in fertilier N to achieve in
            accordane with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.2)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tba.transformation_soil_reduce_excess_fertilizer(
            df_input,
            {
                "fertilizer_n": magnitude,
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    


    def _trfunc_soil_reduce_excess_lime(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, None] = 0.2,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Excess Liming" SOIL transformer on input DataFrame df_input. 
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fractional reduction in lime application to achieve in
            accordane with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # set the magnitude in case of none
        magnitude = self.bounded_real_magnitude(magnitude, 0.2)

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tba.transformation_soil_reduce_excess_fertilizer(
            df_input,
            {
                "lime": magnitude,
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    



    ####################################################
    ###                                              ###
    ###    CIRCULAR ECONOMY TRANSFORMER FUNCTIONS    ###
    ###                                              ###
    ####################################################

    ####################################
    #    TRWW TRANSFORMER FUNCTIONS    #
    ####################################
    
    def _trfunc_trww_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.85,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Biogas Capture at Wastewater Treatment Plants" 
            TRWW transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final magnitude of biogas capture at TRWW facilties.
            NOTE: If specified as a float, the same value applies to both 
                landfill and biogas. Specify as a dictionary to specifiy 
                different capture fractions by TRWW technology, e.g., 
                
                magnitude = {
                    "treated_advanced_anaerobic": 0.85, 
                    "treated_secondary_anaerobic": 0.5
                }
                
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbc.transformation_trww_increase_gas_capture(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out


    
    def _trfunc_trww_increase_septic_compliance(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.9,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Compliance" TRWW transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final magnitude of compliance at septic tanks that are
            installed
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbc.transformation_trww_increase_septic_compliance(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out



    ####################################
    #    WALI TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_wali_improve_sanitation_industrial(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_magnitude: Union[Dict[str, float], None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Industrial Sanitation" WALI transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_magnitude: Union[Dict[str, float], None]
            Target allocation, across TRWW (Wastewater Treatment) categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% from scondary aerobic by the final time period, the following dictionary would be specified:
            
            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_anaerobic": 0.1
            }

            If None, defaults to:

            {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_aerobic": 0.1,
                "treated_secondary_anaerobic": 0.1,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        ##  CHECK DICTIONARY

        if not isinstance(dict_magnitude, dict):
            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_aerobic": 0.1,
                "treated_secondary_anaerobic": 0.1,
            }
        
        dict_magnitude = self.model_attributes.get_valid_categories_dict(
            dict_magnitude,
            self.model_attributes.subsec_name_trww,
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_industrial",
            dict_magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat,
        )


        return df_out



    def _trfunc_wali_improve_sanitation_rural(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_magnitude: Union[Dict[str, float], None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Rural Sanitation" WALI transformer on input DataFrame df_input

        Parameters
        ---------- 
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_magnitude: Union[Dict[str, float], None]
            Target allocation, across TRWW (Wastewater Treatment) categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% from scondary aerobic by the final time period, the following dictionary would be specified:

            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_anaerobic": 0.1
            }

            If None, defaults to:

            {
                "treated_septic": 1.0,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        ##  CHECK DICTIONARY

        if not isinstance(dict_magnitude, dict):
            dict_magnitude = {
                "treated_septic": 1.0, 
            }
        

        dict_magnitude = self.model_attributes.get_valid_categories_dict(
            dict_magnitude,
            self.model_attributes.subsec_name_trww,
        )


        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_domestic_rural",
            dict_magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat,
        )


        return df_out



    def _trfunc_wali_improve_sanitation_urban(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_magnitude: Union[Dict[str, float], None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Improve Urban Sanitation" WALI transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_magnitude: Union[Dict[str, float], None]
            Target allocation, across TRWW (Wastewater Treatment) categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% from scondary aerobic by the final time period, the following dictionary would be specified:

            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_anaerobic": 0.1
            }

            If None, defaults to:

            {
                "treated_advanced_aerobic": 0.3,
                "treated_advanced_anaerobic": 0.3,
                "treated_secondary_aerobic": 0.2,
                "treated_secondary_anaerobic": 0.2,
            }

        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        ##  CHECK DICTIONARY

        if not isinstance(dict_magnitude, dict):
            dict_magnitude = {
                "treated_advanced_aerobic": 0.3,
                "treated_advanced_anaerobic": 0.3,
                "treated_secondary_aerobic": 0.2,
                "treated_secondary_anaerobic": 0.2,
            }
        
        dict_magnitude = self.model_attributes.get_valid_categories_dict(
            dict_magnitude,
            self.model_attributes.subsec_name_trww,
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_domestic_urban",
            dict_magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat,
        )

        return df_out



    ####################################
    #    WASO TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_waso_decrease_food_waste(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Decrease Municipal Solid Waste" WASO transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            reduction in food waste sent to munipal solid waste
            treatment stream
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # get categories and dictionary to specify parameters (move to config eventually)
        categories = (
            self.model_attributes
            .get_attribute_table(
                self.model_attributes.subsec_name_waso
            )
            .key_values
        )

        #dict_specify = dict((x, 0.25) for x in categories)
        #dict_specify.update({"food": 0.3})
        dict_specify = {
            "food": min(max(magnitude, 0.0), 1.0),
        }

        df_out = tbc.transformation_waso_decrease_municipal_waste(
            df_input,
            dict_specify,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out



    def _trfunc_waso_increase_anaerobic_treatment_and_composting(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_biogas: float = 0.475,
        magnitude_compost: float = 0.475,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Anaerobic Treatment and Composting" WASO 
            transformer on input DataFrame df_input. 

        Note that 0 <= magnitude_biogas + magnitude_compost should be <= 1; 
            if they exceed 1, they will be scaled proportionally to sum to 1
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        - magnitude_biogas: proportion of organic solid waste that is treated 
            using anaerobic treatment
        - magnitude_compost: proportion of organic solid waste that is treated 
            using compost
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbc.transformation_waso_increase_anaerobic_treatment_and_composting(
            df_input,
            magnitude_biogas,
            magnitude_compost,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out


    
    def _trfunc_waso_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, Dict[str, float]] = 0.85,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Biogas Capture at Anaerobic Treatment Facilities
            and Landfills" WASO transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final magnitude of biogas capture at landfill and anaerobic
            digestion facilties.
            NOTE: If specified as a float, the same value applies to both 
                landfill and biogas. Specify as a dictionary to specifiy 
                different capture fractions by WASO technology, e.g., 
                
                magnitude = {
                    "landfill": 0.85, 
                    "biogas": 0.5
                }

        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbc.transformation_waso_increase_gas_capture(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out


    
    def _trfunc_waso_increase_energy_from_biogas(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.85,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Energy from Captured Biogas" WASO transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final magnitude of energy use from captured biogas. 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = min(max(magnitude, 0.0), 1.0)

        df_out = tbc.transformation_waso_increase_energy_from_biogas(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out
    


    def _trfunc_waso_increase_energy_from_incineration(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.85,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Energy from Solid Waste Incineration" WASO transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final magnitude of waste that is incinerated that is
            recovered for energy use
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = min(max(magnitude, 0.0), 1.0)

        df_out = tbc.transformation_waso_increase_energy_from_incineration(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out


    
    def _trfunc_waso_increase_landfilling(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 1.0,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Landfilling" WASO transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fraction of non-recycled solid waste (including composting 
            and anaerobic digestion) sent to landfills
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = min(max(magnitude, 0.0), 1.0)

        df_out = tbc.transformation_waso_increase_landfilling(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out



    def _trfunc_waso_increase_recycling(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Recycling" WASO transformer on input DataFrame df_input

        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            magnitude of recylables that are recycled  
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.95, )

        df_out = tbc.transformation_waso_increase_recycling(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circular_economy,
            strategy_id = strat
        )

        return df_out
    



    ##########################################
    ###                                    ###
    ###    ENERGY TRANSFORMER FUNCTIONS    ###
    ###                                    ###
    ##########################################


    ####################################
    #    CCSQ TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_ccsq_increase_air_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: Union[float, int] = 50,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Direct Air Capture" CCSQ transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            final total, in MT, of direct air capture capacity 
            installed at 100% implementation
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_strat_cur = tbe.transformation_ccsq_increase_direct_air_capture(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    ####################################
    #    ENTC TRANSFORMER FUNCTIONS    #
    ####################################
    
    def _trfunc_entc_clean_hydrogen(self,
        categories_source: Union[List[str], None] = None,
        categories_target: Union[List[str], None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement "green hydrogen" transformer requirements by forcing at least 95% (or `magnitude`) of hydrogen production to come from electrolysis.
        
        Parameters
        ----------
        categories_source: Union[List[str], None]
            Hydrogen-producing technology categories that are reduced in response to increases in green hydrogen. 
            * If None, defaults to 
                [
                    "fp_hydrogen_gasification", 
                    "fp_hydrogen_reformation",
                    "fp_hydrogen_reformation_ccs"
                ]

        categories_target: Union[List[str], None]
            Hydrogen-producing technology categories that are considered green; they will produce `magnitude` of hydrogen by 100% implementation. 
            * If None, defaults to 
                [
                    "fp_hydrogen_electrolysis"
                ]

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            target fraction of hydrogen from clean (categories_source)
            sources. In general, this is electrolysis
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.95, )

        ##  VERIFY CATEGORIES

        categories_source = (
            [
                "fp_hydrogen_gasification", 
                "fp_hydrogen_reformation",
                "fp_hydrogen_reformation_ccs"
            ]
            if not sf.islistlike(categories_source)
            else self.model_attributes.get_valid_categories(
                categories_source,
                self.model_attributes.subsec_name_entc,
            )
        )

        categories_target = (
            ["fp_hydrogen_electrolysis"]
            if not sf.islistlike(categories_target)
            else self.model_attributes.get_valid_categories(
                categories_target,
                self.model_attributes.subsec_name_entc,
            )
        )


        ##  RUN STRATEGY

        df_strat_cur = tbe.transformation_entc_clean_hydrogen(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            self.model_enerprod,
            cats_to_apply = categories_target,
            cats_response = categories_source,
            field_region = self.key_region,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_entc_least_cost(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Least Cost" ENTC transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_strat_cur = tbe.transformation_entc_least_cost_solution(
            df_input,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enerprod = self.model_enerprod,
            strategy_id = strat,
        )

        return df_strat_cur



    def _trfunc_entc_reduce_transmission_losses(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.06,
        magnitude_type: str = "final_value", # behavior here is a ceiling
        min_loss: Union[float, None] = 0.02,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Transmission Losses" ENTC transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            magnitude of transmission loss in final time; behavior 
            depends on `magnitude_type`
        magnitude_type: str
            * scalar (if `magnitude_type == "basline_scalar"`)
            * final value (if `magnitude_type == "final_value"`)
            * final value ceiling (if `magnitude_type == "final_value_ceiling"`)
        min_loss: float
            Minimum feasible transmission loss in the system
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_strat_cur = tbe.transformation_entc_specify_transmission_losses(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            self.model_enerprod,
            field_region = self.key_region,
            magnitude_type = magnitude_type,
            min_loss = min_loss,
            strategy_id = strat,
        )

        return df_strat_cur



    def _trfunc_entc_renewables_target(self,
        df_input: Union[pd.DataFrame, None] = None,
        categories_entc_max_investment_ramp: Union[List[str], None] = None,
        categories_entc_renewable: Union[List[str], None] = None,
        dict_entc_renewable_target_msp: Union[Dict[str, float], None] = {},
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "renewables target" transformer (shared repeatability), which sets a Minimum Share of Production from renewable energy as a target.
        
        Parameters
        ----------
        categories_entc_max_investment_ramp: Union[List[str], None]
            Categories to cap investments in
        categories_entc_renewable: Union[List[str], None]
            Power plant technologies to consider as renewable energy sources
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        dict_entc_renewable_target_msp: Union[Dict[str, float], None]
            Optional dictionary mapping renewable ENTC categories to MSP fractions. Can be used to ensure some minimum contribution of certain renewables--e.g.,

                        {
                            "pp_hydropower": 0.1,
                            "pp_solar": 0.15
                        }

            will ensure that hydropower is at least 10% of the mix and solar is at least 15%. 
        magnitude: 
            Minimum target fraction of electricity produced from renewable sources by 100% implementation
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.95)


        # get max investment ramp categories 
        cats_entc_max_investment_ramp = self.get_entc_cats_max_investment_ramp(
            cats_entc_max_investment_ramp = categories_entc_max_investment_ramp,
        )

        # renewable categories
        categories_entc_renewable = self.get_entc_cats_renewable(
            categories_entc_renewable, 
        )

        
        # dictionary mapping to target minimum shares of production
        dict_entc_renewable_target_msp = self.get_entc_dict_renewable_target_msp(
            cats_renewable = categories_entc_renewable,
            dict_entc_renewable_target_msp = dict_entc_renewable_target_msp,
        )

        # characteristics for MSP ramp 
        (
            dict_entc_renewable_target_cats_max_investment,
            vec_implementation_ramp,
            vec_implementation_ramp_renewable_cap,
            vec_msp_resolution_cap,
        ) = self.get_vectors_for_ramp_and_cap(
            categories_entc_max_investment_ramp = categories_entc_max_investment_ramp,
            vec_implementation_ramp = vec_implementation_ramp,
        )


        # finally, implement target
        df_strat_cur = tbe.transformation_entc_renewable_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_enerprod,
            dict_cats_entc_max_investment = dict_entc_renewable_target_cats_max_investment,
            field_region = self.key_region,
            magnitude_renewables = dict_entc_renewable_target_msp,
            strategy_id = strat,
        )

        return df_strat_cur



    



    ####################################
    #    FGTV TRANSFORMER FUNCTIONS    #
    ####################################
        
    def _trfunc_fgtv_maximize_flaring(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.8,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Maximize Flaring" FGTV transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            Fraction of vented methane that is flared.
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # verify magnitude
        magnitude = self.bounded_real_magnitude(magnitude, 0.8)

        
        df_strat_cur = tbe.transformation_fgtv_maximize_flaring(
            df_input,
            magnitude, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_fgtv_minimize_leaks(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.8,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Minimize Leaks" FGTV transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            Fraction of leaky sources (pipelines, storage, etc) that are fixed
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # verify magnitude
        magnitude = self.bounded_real_magnitude(magnitude, 0.8)

        
        df_strat_cur = tbe.transformation_fgtv_reduce_leaks(
            df_input,
            magnitude, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    ####################################
    #    INEN TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_inen_fuel_switch_low_and_high_temp(self,
        df_input: Union[pd.DataFrame, None] = None,
        frac_high_given_high: Union[float, dict, None] = None,
        frac_switchable: float = 0.9, 
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Fuel switch low-temp thermal processes to industrial heat pumps" or/and "Fuel switch medium and high-temp thermal processes to hydrogen and electricity" INEN transformations on input DataFrame df_input (note: these must be combined in a new function instead of as a composition due to the electricity shift in high-heat categories)
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        frac_high_given_high: Union[float, dict, None]
            In high heat categories, fraction of heat demand that is high (NOTE: needs to be switched to per industry). 
            * If specified as a float, this is applied to all high heat categories
            * If specified as None, uses the following dictionary as a default:
                (TEMP: from https://www.sciencedirect.com/science/article/pii/S0360544222018175?via%3Dihub#bib34 [see sainz_et_al_2022])

                {
                    "cement": 0.88, # use non-metallic minerals
                    "chemicals": 0.5, 
                    "glass": 0.88, 
                    "lime_and_carbonite": 0.88, 
                    "metals": 0.92,
                    "paper": 0.18, 
                }

        frac_switchable: float
            Fraction of demand that can be switched 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )
        

        dict_frac_high_given_high_def, cats_inen_low_med_heat = self.get_inen_parameters()

        # calculate some fractions
        if frac_high_given_high is None:
            frac_high_given_high = dict_frac_high_given_high_def
        
        elif sf.isnumber(frac_high_given_high):
            frac_high_given_high = self.bounded_real_magnitude(frac_high_given_high, 0.5)
            frac_high_given_high = dict(
                (k, frac_high_given_high) for k in dict_frac_high_given_high_def.keys()
            )

        # do some checks
        frac_high_given_high = self.model_attributes.get_valid_categories_dict(
            frac_high_given_high,
            self.model_attributes.subsec_name_inen,
        )


        # iterate over each high-heat industrial case
        df_out = df_input.copy()

        for (cat, frac) in frac_high_given_high.items():

            frac_low_given_high = 1.0 - frac
            frac_switchable = self.bounded_real_magnitude(frac_switchable, 0.9)

            frac_inen_low_temp_elec_given_high = frac_switchable*frac_low_given_high
            frac_inen_high_temp_elec_hydg = frac_switchable*frac
            
            # set up fractions 
            frac_shift_hh_elec = frac_inen_low_temp_elec_given_high + frac_inen_high_temp_elec_hydg/2
            frac_shift_hh_elec /= frac_switchable

            frac_shift_hh_hydrogen = frac_inen_high_temp_elec_hydg/2
            frac_shift_hh_hydrogen /= frac_switchable

            # HIGH HEAT CATS ONLY
            # Fuel switch high-temp thermal processes + Fuel switch low-temp thermal processes to industrial heat pumps
            df_out = tbe.transformation_inen_shift_modvars(
                df_out,
                frac_switchable,
                vec_implementation_ramp, 
                self.model_attributes,
                categories = [cat],
                dict_modvar_specs = {
                    self.model_enercons.modvar_inen_frac_en_electricity: frac_shift_hh_elec,
                    self.model_enercons.modvar_inen_frac_en_hydrogen: frac_shift_hh_hydrogen,
                },
                field_region = self.key_region,
                model_enercons = self.model_enercons,
                strategy_id = strat
            )


        # LOW HEAT CATS ONLY
        # + Fuel switch low-temp thermal processes to industrial heat pumps
        df_out = tbe.transformation_inen_shift_modvars(
            df_out,
            frac_switchable,
            vec_implementation_ramp, 
            self.model_attributes,
            categories = cats_inen_low_med_heat,
            dict_modvar_specs = {
                self.model_enercons.modvar_inen_frac_en_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_out



    def _trfunc_inen_maximize_efficiency_energy(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Maximize Industrial Energy Efficiency" INEN 
            transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            Magnitude of energy efficiency increase (applied to industrial efficiency factor)
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.3)
        
        df_strat_cur = tbe.transformation_inen_maximize_energy_efficiency(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_inen_maximize_efficiency_production(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.4,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Maximize Industrial Production Efficiency" INEN transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of energy efficiency increase (applied to industrial production factor)
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.4)
        
        df_strat_cur = tbe.transformation_inen_maximize_production_efficiency(
            df_input,
            magnitude, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    ####################################
    #    SCOE TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_scoe_fuel_switch_electrify(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Switch to electricity for heat using heat pumps, electric stoves, etc." INEN transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magntiude of fraction of heat energy that is electrified
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.95)
        
        df_strat_cur = tbe.transformation_scoe_electrify_category_to_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_scoe_reduce_heat_energy_demand(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce end-use demand for heat energy by improving building shell" SCOE transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Reduction in heat energy demand, driven by retrofitting and changes in use
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.5)
        
        df_strat_cur = tbe.transformation_scoe_reduce_demand_for_heat_energy(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_scoe_increase_applicance_efficiency(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase appliance efficiency" SCOE transformer on 
            input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional increase in applieance energy efficiency
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.5)

        df_strat_cur = tbe.transformation_scoe_reduce_demand_for_appliance_energy(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    ####################################
    #    TRDE TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_trde_reduce_demand(self,
        df_input: pd.DataFrame = None,
        magnitude: float = 0.25,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce Demand" TRDE transformer on input DataFrame
            df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional reduction in transportation demand applied to all transportation categories.
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.25)

        df_out = tbe.transformation_trde_reduce_demand(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out

    

    ####################################
    #    TRNS TRANSFORMER FUNCTIONS    #
    ####################################

    def _trfunc_trns_electrify_road_light_duty(self,
        categories: List[str] = ["road_light"],
        dict_fuel_allocation: Union[dict, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.7,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Electrify Light-Duty" TRNS transformer on input DataFrame df_input
        
        Parameters
        ----------
        categories: List[str]
            Transportation categories to include; defaults to "road_light"
        dict_fuel_allocation: Union[dict, None]
            Optional dictionary defining fractional allocation of fuels in fuel switch. If undefined, defaults to
                {
                    "fuel_electricity": 1.0
                }
            
            NOTE: keys must be valid TRNS fuels and values in the dictionary 
            must sum to 1.

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fraction of light duty vehicles electrified 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # bound the magnitude and check categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.7)
        categories = self.model_attributes.get_valid_categories(
            categories,
            self.model_attributes.subsec_name_trns
        )

        # check the specification of the fuel allocation dictionary
        dict_modvar_specs = self.check_trns_fuel_switch_allocation_dict(
            dict_fuel_allocation,
            {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            }
        )


        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = dict_modvar_specs,
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def _trfunc_trns_electrify_rail(self,
        categories: List[str] = ["rail_freight", "rail_passenger"],
        dict_fuel_allocation: Union[dict, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.25,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Electrify Rail" TRNS transformer on input DataFrame
            df_input
        
        Parameters
        ----------
        categories: List[str]
            Transportation categories to include; defaults to ["rail_freight", "rail_passenger"]
        dict_fuel_allocation: Union[dict, None]
            Optional dictionary dictionary defining fractional allocation of fuels in fuel switch. If undefined, defaults to
                {
                    "fuel_electricity": 1.0
                }
            
            NOTE: keys must be valid TRNS fuels and values in the dictionary 
            must sum to 1.

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            fraction of light duty vehicles electrified 
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # bound the magnitude and check categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.25)
        categories = self.model_attributes.get_valid_categories(
            categories,
            self.model_attributes.subsec_name_trns
        )

        # check the specification of the fuel allocation dictionary
        dict_modvar_specs = self.check_trns_fuel_switch_allocation_dict(
            dict_fuel_allocation,
            {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            }
        )

        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = dict_modvar_specs,
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    def _trfunc_trns_fuel_switch_maritime(self,
        categories: List[str] = ["water_borne"],
        dict_allocation_fuels_target: Union[dict, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        fuels_source: List[str] = ["fuel_diesel", "fuel_gasoline"],
        magnitude: float = 0.7,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Fuel-Swich Maritime" TRNS transformer on input DataFrame df_input. By default, transfers mangitude to hydrogen from 
            gasoline and diesel; e.g., with magnitude = 0.7, then 70% of diesel 
            and gas demand are transfered to fuels in fuels_target. The rest of
            the fuel demand is then transferred to electricity. 
        
        Parameters
        ----------
        categories: List[str]
            Transportation categories to include; defaults to ["water_borne"]
        dict_allocation_fuels_target: Union[dict, None]
            Optional dictionary allocating target fuels. If None, defaults to
            {
                "fuel_hydrogen": 1.0,
            }

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        fuels_source: List[str]
            Fuels to transfer out; for F the percentage of TRNS demand met by fuels in fuels source, M*F (M = magtnitude) is transferred to fuels defined in dict_allocation_fuels_target
        magnitude: float
            Fraction of fuels_source (gas and diesel, e.g.) that is shifted to target fuels fuels_target (hydrogen is default, can include ammonia) for categories. Note, remaining is shifted to electricity
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        ##  CHECKS AND INIT

        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # bound the magnitude and check categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.7)
        categories = self.model_attributes.get_valid_categories(
            categories,
            self.model_attributes.subsec_name_trns
        )

        # check the specification of the fuel allocation dictionary
        dict_modvar_specs = self.check_trns_fuel_switch_allocation_dict(
            dict_allocation_fuels_target,
            {
                self.model_enercons.modvar_trns_fuel_fraction_hydrogen: 1.0
            }
        )
        
        # get fuel source modvars
        fuels_source = self.model_attributes.get_valid_categories(
            fuels_source,
            self.model_attributes.subsec_name_enfu
        )
        modvars_source = [
            (
                self.model_enercons
                .dict_trns_fuel_categories_to_fuel_variables
                .get(x)
                .get("fuel_fraction")
            )
            for x in fuels_source
        ]

        
        ##  RUN TRANSFORMATION IN TWO STAGES

        # transfer magnitude (frac) of source fuels to fuels_target
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = dict_modvar_specs,
            field_region = self.key_region,
            modvars_source = modvars_source,
            #[
            #    self.model_enercons.modvar_trns_fuel_fraction_diesel,
            #    self.model_enercons.modvar_trns_fuel_fraction_gasoline
            #],
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        # transfer remaining diesel + gasoline to electricity
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_out,
            1.0,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            modvars_source = modvars_source,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    def _trfunc_trns_fuel_switch_road_medium_duty(self,
        categories: List[str] = [
            "road_heavy_freight", 
            "road_heavy_regional", 
            "public"
        ],
        dict_allocation_fuels_target: Union[dict, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        fuels_source: List[str] = ["fuel_diesel", "fuel_gasoline"],
        magnitude: float = 0.7,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Fuel-Switch Medium Duty" TRNS transformer on input DataFrame df_input. By default, transfers mangitude to electricity from gasoline and diesel; e.g., with magnitude = 0.7, then 70% of diesel and gas demand are transfered to fuels in fuels_target. The rest of the fuel demand is then transferred to hydrogen. 
        
        Parameters
        ----------
        categories: List[str]
            Transportation categories to include; defaults to 
            [
                "road_heavy_freight", 
                "road_heavy_regional", 
                "public"
            ]
        dict_allocation_fuels_target: Union[dict, None]
            Optional dictionary allocating target fuels. If None, defaults to
            {
                "fuel_electricity": 1.0,
            }

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        fuels_source: List[str]
            Fuels to transfer out; for F the percentage of TRNS demand met by fuels in fuels source, M*F (M = magtnitude) is transferred to fuels defined in dict_allocation_fuels_target
        magnitude: float
            Fraction of fuels_source (gas and diesel, e.g.) that shifted to target fuels fuels_target (hydrogen is default, can include ammonia). Note, remaining is shifted to electricity
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        ##  CHECKS AND INIT

        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # bound the magnitude and check categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.7)
        categories = self.model_attributes.get_valid_categories(
            categories,
            self.model_attributes.subsec_name_trns
        )

        # check the specification of the fuel allocation dictionary
        dict_modvar_specs = self.check_trns_fuel_switch_allocation_dict(
            dict_allocation_fuels_target,
            {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            }
        )
        
        # get fuel source modvars
        fuels_source = self.model_attributes.get_valid_categories(
            fuels_source,
            self.model_attributes.subsec_name_enfu
        )
        modvars_source = [
            (
                self.model_enercons
                .dict_trns_fuel_categories_to_fuel_variables
                .get(x)
                .get("fuel_fraction")
            )
            for x in fuels_source
        ]


        ##  DO STAGED IMPLEMENTATION

        # transfer 70% of diesel + gasoline to electricity
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = dict_modvar_specs,
            field_region = self.key_region,
            modvars_source = modvars_source,
            #modvars_source = [
            #    self.model_enercons.modvar_trns_fuel_fraction_diesel,
            #    self.model_enercons.modvar_trns_fuel_fraction_gasoline
            #],
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        # transfer remaining diesel + gasoline to hydrogen
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_out,
            1.0,
            vec_implementation_ramp,
            self.model_attributes,
            categories = categories,
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            field_region = self.key_region,
            modvars_source = modvars_source,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
    
        return df_out
    

    
    def _trfunc_trns_increase_efficiency_electric(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.25,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Electric Efficiency" TRNS transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Increase the efficiency of electric vehicales by this proportion
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.25)
        
        df_out = tbe.transformation_trns_increase_energy_efficiency_electric(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_increase_efficiency_non_electric(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.25,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Non-Electric Efficiency" TRNS transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Increase the efficiency of non-electric vehicales by this proportion
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.25)

        df_out = tbe.transformation_trns_increase_energy_efficiency_non_electric(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_increase_occupancy_light_duty(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.25,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Increase Vehicle Occupancy" TRNS transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Increase the occupancy rate of light duty vehicles by this proporiton
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.25)
        
        df_out = tbe.transformation_trns_increase_vehicle_occupancy(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_mode_shift_freight(self,
        categories_out: List[str] = ["aviation", "road_heavy_freight"],
        dict_categories_target: Union[Dict[str, float], None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.2,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Mode Shift Freight" TRNS transformer on input DataFrame df_input. By Default, transfer 20% of aviation and road heavy freight to rail freight.
        
        Parameters
        ----------
        categories_out: List[str] 
            Categories to shift out of 
        dict_categories_target: Union[Dict[str, float], None]
            Dictionary mapping target categories to proportional allocation of mode mass. If None, defaults to
            {
                "rail_freight": 1.0
            }

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of mode mass to shift out of cats_out
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """

        ##  CHECKS AND INIT

        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # check magnitude and categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.2)
        categories_out = self.model_attributes.get_valid_categories(
            categories_out,
            self.model_attributes.subsec_name_trns,
        )

        # check the target dictionary
        dict_categories_target_out = self.check_trns_tech_allocation_dict(
            dict_categories_target,
            {
                "rail_freight": 1.0
            }
        )
        
        df_out = tbe.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_freight: {
                    "bounds": (0, 1),
                    "magnitude": magnitude,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": categories_out,
                    "categories_target": dict_categories_target_out,
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_mode_shift_public_private(self,
        categories_out: List[str] = ["road_light"],
        dict_categories_target: Union[Dict[str, float], None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Mode Shift Passenger Vehicles to Others" TRNS 
            transformer on input DataFrame df_input
        
        Parameters
        ----------
        categories_out: List[str] 
            Categories to shift out of
        dict_categories_target: Union[Dict[str, float], None]
            Dictionary mapping target categories to proportional allocation of mode mass. If None, defaults to
            {
                "human_powered": (1/6),
                "powered_bikes": (2/6),
                "public": 0.5
            }

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Magnitude of mode mass to shift out of cats_out
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # check magnitude and categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.3)
        categories_out = self.model_attributes.get_valid_categories(
            categories_out,
            self.model_attributes.subsec_name_trns,
        )

        # check the target dictionary
        dict_categories_target_out = self.check_trns_tech_allocation_dict(
            dict_categories_target,
            {
                "human_powered": (1/6),
                "powered_bikes": (2/6),
                "public": 0.5
            }
        )


        df_out = tbe.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_public_private: {
                    "bounds": (0, 1),
                    "magnitude": magnitude,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": categories_out,
                    "categories_target": dict_categories_target_out,
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
        )
        
        return df_out
    
    

    def _trfunc_trns_mode_shift_regional(self,
        dict_categories_out: Dict[str, float] = {
            "aviation": 0.1,
            "road_light": 0.2,
        },
        dict_categories_target: Union[Dict[str, float], None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Mode Shift Regional Travel" TRNS transformer on input DataFrame df_input
        
        Parameters
        ----------
        dict_categories_out: Dict[str, float]
            Dictionary mapping categories to shift out of to the magnitude of the outward shift
        dict_categories_target: Union[Dict[str, float], None]
            Dictionary mapping target categories to proportional allocation of mode mass. If None, defaults to
            {
                "road_heavy_regional": 1.0
            }

        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        # check the target dictionary
        dict_categories_target_out = self.check_trns_tech_allocation_dict(
            dict_categories_target,
            {
                "road_heavy_regional": 1.0,
            }
        )
        
        dict_categories_out = self.check_trns_tech_allocation_dict(
            dict_categories_out,
            {
                "aviation": 0.1,
                "road_light": 0.2,
            },
            sum_check = "leq",
        )
        

        ##  APPLY THE TRANSFORMATION(S) ITERATIVELY

        df_out = df_input.copy()

        for (cat, mag) in dict_categories_out.items():
   
            df_out = tbe.transformation_general(
                df_out,
                self.model_attributes,
                {
                    self.model_enercons.modvar_trns_modeshare_regional: {
                        "bounds": (0, 1),
                        "magnitude": mag,
                        "magnitude_type": "transfer_value_scalar",
                        "categories_source": [cat],
                        "categories_target": dict_categories_target_out,
                        "vec_ramp": vec_implementation_ramp
                    }
                },
                field_region = self.key_region,
                strategy_id = strat
            )


        
        return df_out
    



    ########################################
    ###                                  ###
    ###    IPPU TRANSFORMER FUNCTIONS    ###
    ###                                  ###
    ########################################

    def _trfunc_ippu_reduce_cement_clinker(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduce cement clinker" IPPU transformer on input DataFrame df_input. Implements a cap on the fraction of cement that is produced using clinker (magnitude)
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            fraction of cement producd using clinker
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.5)
        
        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_ippu.modvar_ippu_clinker_fraction_cement: {
                    "bounds": (0, 1),
                    "magnitude": magnitude,
                    "magnitude_type": "final_value_ceiling",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_ippu_reduce_demand(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.3,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Demand Management" IPPU transformer on input DataFrame df_input. Reduces industrial production.
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional reduction in demand in accordance with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.3)

        df_out = tbi.transformation_ippu_reduce_demand(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_ippu = self.model_ippu,
            strategy_id = strat
        )

        return df_out


    
    def _trfunc_ippu_reduce_hfcs(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.1,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduces HFCs" IPPU transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional reduction in HFC emissions in accordance with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

        df_out = tbi.transformation_ippu_scale_emission_factor(
            df_input,
            {"hfc": magnitude}, # applies to all HFC emission factors
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_ippu = self.model_ippu,
            strategy_id = strat,
        )        

        return df_out
    


    def _trfunc_ippu_reduce_n2o(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.1,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduces N2O" IPPU transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional reduction in IPPU N2O emissions in accordance with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

        df_out = tbi.transformation_ippu_scale_emission_factor(
            df_input,
            {
                self.model_ippu.modvar_ippu_ef_n2o_per_gdp_process : magnitude,
                self.model_ippu.modvar_ippu_ef_n2o_per_prod_process : magnitude,
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_ippu = self.model_ippu,
            strategy_id = strat,
        )        

        return df_out


    
    def _trfunc_ippu_reduce_other_fcs(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.1,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduces Other FCs" IPPU transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: 
            Fractional reduction in IPPU other FC emissions in accordance with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

        df_out = tbi.transformation_ippu_scale_emission_factor(
            df_input,
            {"other_fc": magnitude}, # applies to all Other Fluorinated Compound emission factors
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_ippu = self.model_ippu,
            strategy_id = strat,
        )        

        return df_out



    def _trfunc_ippu_reduce_pfcs(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.1,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Implement the "Reduces Other FCs" IPPU transformer on input DataFrame df_input
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude: float
            Fractional reduction in IPPU other FC emissions in 
            accordance with vec_implementation_ramp
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

        df_out = tbi.transformation_ippu_scale_emission_factor(
            df_input,
            {"pfc": magnitude}, # applies to all PFC emission factors
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_ippu = self.model_ippu,
            strategy_id = strat,
        )        

        return df_out




    ##################################################
    ###                                            ###
    ###    CROSS-SECTORAL TRANSFORMER FUNCTIONS    ###
    ###                                            ###
    ##################################################

    def _trfunc_pflo_healthier_diets(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_red_meat: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Healthier Diets" transformer on input DataFrame df_input (affects GNRL and and AGRC [NOTE: AGRC component currently not implemented]).
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        magnitude_red_meat: float
            Final period maximum fraction of per capita red meat consumption relative to baseline (e.g., 0.5 means that people eat 50% as much red meat as they would have without the intervention)
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )

        
        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat: {
                    "bounds": (0, 1),
                    "magnitude": magnitude_red_meat,
                    "magnitude_type": "final_value_ceiling",
                    "vec_ramp": vec_implementation_ramp
                },

                # TEMPORARY UNTIL A DEMAND SCALAR CAN BE ADDED IN
                # self.model_afolu.modvar_agrc_elas_crop_demand_income: {
                #    "bounds": (-2, 2),
                #    "categories": ["sugar_cane"],
                #    "magnitude": -0.2,
                #    "magnitude_type": "final_value_ceiling",
                #    "vec_ramp": vec_implementation_ramp
                # },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def _trfunc_pflo_industrial_ccs(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """Implement the "Industrial Point of Capture" transformer on input DataFrame df_input (affects IPPU and INEN).
        
        Parameters
        ----------
        df_input: pd.DataFrame
            Optional data frame containing trajectories to modify
        strat: int
            Optional strategy value to specify for the transformation
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None]
            Optional vector or dictionary specifying the implementation scalar ramp for the transformation. If None, defaults to a uniform ramp that starts at the time specified in the configuration.
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        dict_magnitude_eff = None
        dict_magnitude_prev = {
            "cement": 0.8,
            "chemicals": 0.8,
            "metals": 0.8,
            "plastic": 0.8,
        }

        # increase prevalence of capture
        df_out = tbs.transformation_mlti_industrial_carbon_capture(
            df_input,
            dict_magnitude_eff,
            dict_magnitude_prev,
            vec_implementation_ramp,
            self.model_attributes,
            model_ippu = self.model_ippu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out