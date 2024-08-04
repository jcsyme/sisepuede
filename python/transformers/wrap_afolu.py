from attribute_table import AttributeTable
import ingestion as ing
import logging
import model_afolu as mafl
import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import setup_analysis as sa
from sisepuede_file_structure import *
import support_classes as sc
import support_functions as sf
import time
import transformers.lib.afolu as tba
import transformers.lib.general as tbg
from typing import Union
import warnings



class TransformationsAFOLU:
    """
    Build energy transformations using general transformations defined in
        auxiliary_definitions_transformations. Wraps more general forms from 
        auxiliary_definitions_transformations into functions and classes
        with shared ramps, paramters, and build functionality.

    NOTE: To update transformations, users need to follow three steps:

        1. Use or create a function in auxiliarty_definitions_transformations, 
            which modifies an input DataFrame, using the ModelAttributes object
            and any approprite SISEPUEDE models. 

        2. If the transformation is not the composition of existing 
            transformation functions, create a transformation definition 
            function using the following functional template:

             def transformation_sabv_###(
                df_input: pd.DataFrame,
                strat: Union[int, None] = None,
                **kwargs
             ) -> pd.DataFrame:
                #DOCSTRING
                ...
                return df_out

            This function is where parameterizations are defined (can be passed 
                through dict_config too if desired, but not done here)

            If using the composition of functions, can leverage the 
            sc.Transformation composition functionality, which lets the user
            enter lists of functions (see ?sc.Transformation for more 
            information)

        3. Finally, define the Transformation object using the 
            `sc.Transformation` class, which connects the function to the 
            Strategy name in attribute_strategy_id, assigns an id, and 
            simplifies the organization and running of strategies. 


    Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and
		coordination
    - dict_config: configuration dictionary used to pass parameters to 
        transformations. See ?TransformationEnergy._initialize_parameters() for
        more information on requirements.
    - dir_jl: location of Julia directory containing Julia environment and 
        support modules
    - fp_nemomod_reference_files: directory housing reference files called by
		NemoMod when running electricity model. Required to access data in 
        ElectricEnergy. Needs the following CSVs:

        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
    - baseline_with_plur: set to True to let the baseline include partial land
        use reallocation. 
        * NOTE: If True, then transformation_lndu_reallocate_land() 
            has no effect.
        * NOTE: this is set in both `self.transformation_af_baseline()` and
            `self.transformation_lndu_reallocate_land()` separately
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
		used in Julia NemoMod Electricity model
        * If None, defaults to a temporary path sql database
    - logger: optional logger object
    - model_afolu: optional AFOLU object to pass for property and method 
        access
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes 
            argument are the same.
    """
    
    def __init__(self,
        model_attributes: ma.ModelAttributes,
        dict_config: Dict,
        baseline_with_plur: bool = False,
        field_region: Union[str, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
		logger: Union[logging.Logger, None] = None,
        model_afolu: Union[mafl.AFOLU, None] = None,
    ):

        self.logger = logger

        self._initialize_attributes(
            field_region, 
            model_attributes,
            baseline_with_plur = baseline_with_plur,
        )
        self._initialize_config(
            dict_config = dict_config,
        )
        self._initialize_models(
            model_afolu = model_afolu,
        )
        self._initialize_parameters(
            dict_config = dict_config,
        )
        self._initialize_ramp()
        self._initialize_baseline_inputs(
            df_input,
        )
        self._initialize_transformations()

        return None




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def get_ramp_characteristics(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Get parameters for the implementation of transformations. Returns a 
            tuple with the following elements:

            (
                n_tp_ramp,
                vir_renewable_cap_delta_frac,
                vir_renewable_cap_max_frac,
                year_0_ramp, 
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires those keys in dict_config to set. If not found, will set
            the following defaults:
                * year_0_ramp: 9th year (10th time period)
                * n_tp_ramp: n_tp - t0_ramp - 1 (ramps to 1 at final time 
                    period)

        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_entc_renewable: list of categories to tag as renewable 
                for the Renewable Targets transformation.
        """

        dict_config = self.config if not isinstance(dict_config, dict) else dict_config
        n_tp = len(self.time_periods.all_time_periods)

        # get first year of non-baseline
        default_year = self.time_periods.all_years[min(9, n_tp - 1)]
        year_0_ramp = dict_config.get(self.key_config_year_0_ramp)
        year_0_ramp = (
            self.time_periods.all_years[default_year] 
            if not sf.isnumber(year_0_ramp, integer = True)
            else year_0_ramp
        )

        # shift by 2--1 to account for baseline having no uncertainty, 1 for py reindexing
        default_n_tp_ramp = n_tp - self.time_periods.year_to_tp(year_0_ramp) - 1
        n_tp_ramp = dict_config.get(self.key_config_n_tp_ramp)
        n_tp_ramp = (
            default_n_tp_ramp
            if not sf.isnumber(n_tp_ramp, integer = True)
            else n_tp_ramp
        )

        tup_out = (
            n_tp_ramp,
            year_0_ramp, 
        )

        return tup_out



    def _initialize_attributes(self,
        field_region: Union[str, None],
        model_attributes: ma.ModelAttributes,
        baseline_with_plur: bool = False,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.baseline_strategy
            * self.baseline_with_plur
            * self.key_region
            * self.key_strategy
            * self.model_attributes
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in TransformationsIPPU")

        # get strategy attribute, baseline strategy, and some fields
        attribute_strategy = model_attributes.get_dimensional_attribute_table(
            model_attributes.dim_strategy_id
        )

        baseline_strategy = int(
            attribute_strategy.table[
                attribute_strategy.table["baseline_strategy_id"] == 1
            ][attribute_strategy.key].iloc[0]
        )

        field_region = (
            model_attributes.dim_region 
            if (field_region is None) 
            else field_region
        )

        # set some useful classes
        time_periods = sc.TimePeriods(model_attributes)
        regions = sc.Regions(model_attributes)

        # set baseline with plur
        baseline_with_plur = (
            False 
            if not isinstance(baseline_with_plur, bool) 
            else baseline_with_plur
        )

        ##  SET PROPERTIES
        
        self.attribute_strategy = attribute_strategy
        self.baseline_strategy = baseline_strategy
        self.baseline_with_plur = baseline_with_plur
        self.key_region = field_region
        self.key_strategy = attribute_strategy.key
        self.model_attributes = model_attributes
        self.regions = regions
        self.time_periods = time_periods

        return None
    


    def _initialize_baseline_inputs(self,
        df_inputs: Union[pd.DataFrame, None],
        baseline_with_plur: bool = False,
    ) -> None:
        """
        Initialize the baseline inputs dataframe based on the initialization 
            value of df_inputs. It not initialied, sets as None. Sets the 
            following properties:

            * self.baseline_inputs

        Keyword Arguments
        -----------------
        - baseline_with_plur: set to True to let the baseline include partial 
            land use reallocation. 
            * NOTE: If True, then transformation_lndu_reallocate_land() 
                has no effect.
            * NOTE: this is set in both `self.transformation_af_baseline()` and
                `self.transformation_lndu_reallocate_land()` separately
        """

        baseline_inputs = (
            self.transformation_af_baseline(
                df_inputs,
                strat = self.baseline_strategy,
            ) 
            if isinstance(df_inputs, pd.DataFrame) 
            else None
        )

        self.baseline_inputs = baseline_inputs

        return None



    def _initialize_config(self,
        dict_config: Union[Dict[str, Any], None],
    ) -> None:
        """
        Define the configuration dictionary and paramter keys. Sets the 
            following properties:

            * self.config (configuration dictionary)
            * self.key_* (keys)
            
        Function Arguments
        ------------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Can include the following keys:

            * "categories_entc_max_investment_ramp": list of categories to apply
                self.vec_implementation_ramp_renewable_cap to with a maximum
                investment cap (implemented *after* turning on renewable target)
            * "categories_entc_renewable": list of categories to tag as 
                renewable for the Renewable Targets transformation (sets 
                self.cats_renewable)
            * "dict_entc_renewable_target_msp": optional dictionary mapping 
                renewable ENTC categories to MSP fractions to use in the 
                Renewable Targets trasnsformationl. Can be used to ensure some
                minimum contribution of certain renewables--e.g.,

                    {
                        "pp_hydropower": 0.1,
                        "pp_solar": 0.15
                    }

                will ensure that hydropower is at least 10% of the mix and solar
                is at least 15%. 

            * "n_tp_ramp": number of time periods to use to ramp up. If None or
                not specified, builds to full implementation by the final time
                period
            * "vir_renewable_cap_delta_frac": change (applied downward from 
                "vir_renewable_cap_max_frac") in cap for for new technology
                capacities available to build in time period while transitioning
                to renewable capacties. Default is 0.01 (will decline by 1% each
                time period after "year_0_ramp")
            * "vir_renewable_cap_max_frac": cap for for new technology 
                capacities available to build in time period while transitioning
                to renewable capacties; entered as a fraction of estimated
                capacity in "year_0_ramp". Default is 0.05
            * "year_0_ramp": last year with no diversion from baseline strategy
                (baseline for implementation ramp)
        """

        dict_config = {} if not isinstance(dict_config, dict) else dict_config

        # set parameters
        self.config = dict_config

        self.key_config_cats_entc_max_investment_ramp = "categories_entc_max_investment_ramp"
        self.key_config_cats_entc_renewable = "categories_entc_renewable"
        self.key_config_cats_inen_high_heat = "categories_inen_high_heat",
        self.key_config_dict_entc_renewable_target_msp = "dict_entc_renewable_target_msp"
        self.key_config_frac_inen_high_temp_elec_hydg = "frac_inen_low_temp_elec"
        self.key_config_frac_inen_low_temp_elec = "frac_inen_low_temp_elec"
        self.key_config_n_tp_ramp = "n_tp_ramp"
        self.key_config_vir_renewable_cap_delta_frac = "vir_renewable_cap_delta_frac"
        self.key_config_vir_renewable_cap_max_frac = "vir_renewable_cap_max_frac"
        self.key_config_year_0_ramp = "year_0_ramp" 

        return None



    def _initialize_models(self,
        model_afolu: Union[mafl.AFOLU, None] = None,
    ) -> None:
        """
        Define model objects for use in variable access and base estimates.

        Keyword Arguments
        -----------------
        - model_afolu: optional AFOLU object to pass for property and method 
            access
            * NOTE: if passing, ensure that the ModelAttributes objects used to 
                instantiate the model + what is passed to the model_attributes 
                argument are the same.
        """

        model_afolu = (
            mafl.AFOLU(self.model_attributes)
            if model_afolu is None
            else model_afolu
        )

        self.model_afolu = model_afolu

        return None


    
    def _initialize_parameters(self,
        dict_config: Union[Dict[str, Any], None] = None,
    ) -> None:
        """
        Define key parameters for transformation. For keys needed to initialize
            and define these parameters, see ?self._initialize_config
      
        """

        dict_config = self.config if not isinstance(dict_config, dict) else dict_config

        # get parameters from configuration dictionary
        (
            n_tp_ramp,
            year_0_ramp
        ) = self.get_ramp_characteristics()


        ##  SET PROPERTIES

        self.n_tp_ramp = n_tp_ramp
        self.year_0_ramp = year_0_ramp

        return None
    


    def _initialize_ramp(self,
    ) -> None: 
        """
        Initialize the ramp vector for implementing transformations. Sets the 
            following properties:

            * self.vec_implementation_ramp
        """
        
        vec_implementation_ramp = self.build_implementation_ramp_vector()
        
        ##  SET PROPERTIES
        self.vec_implementation_ramp = vec_implementation_ramp

        return None



    def _initialize_transformations(self,
    ) -> None:
        """
        Initialize all sc.Transformation objects used to manage the construction
            of transformations. Note that each transformation == a strategy.

        NOTE: This is the key function mapping each function to a transformation
            name.
            
        Sets the following properties:

            * self.all_transformations
            * self.all_transformations_non_baseline
            * self.dict_transformations
            * self.transformation_id_baseline
            * self.transformation_***
        """

        attr_strategy = self.attribute_strategy
        all_transformations = []
        dict_transformations = {}


        ##################
        #    BASELINE    #
        ##################

        self.baseline = sc.Transformation(
            "BASE", 
            self.transformation_af_baseline, 
            attr_strategy
        )
        all_transformations.append(self.baseline)


        
        ######################################
        #    AFOLU TRANSFORMATION BUNDLES    #
        ######################################

        self.af_all = sc.Transformation(
            "AF:ALL", 
            [
                #self.transformation_agrc_decrease_exports,
                self.transformation_agrc_expand_conservation_agriculture,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses,
                # self.transformation_lndu_integrated_transitions replaces:
                #   self.transformation_lndu_expand_silvopasture,
                self.transformation_lndu_stop_deforestation,
                #self.transformation_lndu_integrated_transitions,
                self.transformation_lndu_increase_reforestation, #INDIA - must come AFTER silvopasture
                self.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformation_lsmm_improve_manure_management_other,
                self.transformation_lsmm_improve_manure_management_poultry,
                self.transformation_lsmm_increase_biogas_capture,
                self.transformation_lvst_decrease_exports,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
                self.transformation_soil_reduce_excess_fertilizer,
                self.transformation_soil_reduce_excess_lime
            ],
            attr_strategy
        )
        all_transformations.append(self.af_all)


        self.af_all_with_partial_reallocation = sc.Transformation(
            "AF:ALL_PLUR", 
            [
                #self.transformation_agrc_decrease_exports,
                self.transformation_agrc_expand_conservation_agriculture,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses,
                # self.transformation_lndu_integrated_transitions replaces:
                #   self.transformation_lndu_expand_silvopasture,
                self.transformation_lndu_stop_deforestation,
                #self.transformation_lndu_integrated_transitions,
                self.transformation_lndu_increase_reforestation, #INDIA - must come AFTER silvopasture
                self.transformation_lndu_reallocate_land,
                self.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformation_lsmm_improve_manure_management_other,
                self.transformation_lsmm_improve_manure_management_poultry,
                self.transformation_lsmm_increase_biogas_capture,
                self.transformation_lvst_decrease_exports,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
                self.transformation_soil_reduce_excess_fertilizer,
                self.transformation_soil_reduce_excess_lime
            ],
            attr_strategy
        )
        all_transformations.append(self.af_all_with_partial_reallocation)


        self.af_all_with_deforestation_and_partial_reallocation = sc.Transformation(
            "AF:ALL_NO_STOPPING_DEFORESTATION_PLUR", 
            [
                #self.transformation_agrc_decrease_exports,
                self.transformation_agrc_expand_conservation_agriculture,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses,
                # self.transformation_lndu_integrated_transitions replaces:
                self.transformation_lndu_expand_silvopasture,
                #   self.transformation_lndu_stop_deforestation
                self.transformation_lndu_increase_reforestation, #INDIA - must come AFTER silvopasture
                self.transformation_lndu_reallocate_land,
                self.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformation_lsmm_improve_manure_management_other,
                self.transformation_lsmm_improve_manure_management_poultry,
                self.transformation_lsmm_increase_biogas_capture,
                self.transformation_lvst_decrease_exports,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
                self.transformation_soil_reduce_excess_fertilizer,
                self.transformation_soil_reduce_excess_lime
            ],
            attr_strategy
        )
        all_transformations.append(self.af_all_with_deforestation_and_partial_reallocation)

        """
        self.af_all_no_lvst_export_reduction_with_partial_reallocation = sc.Transformation(
            "AF:ALL_NO_LVST_EXPORT_REDUCTION_PLUR", 
            [
                #self.transformation_agrc_decrease_exports,
                self.transformation_agrc_expand_conservation_agriculture,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses,
                # self.transformation_lndu_integrated_transitions replaces:
                #   self.transformation_lndu_expand_silvopasture,
                #   self.transformation_lndu_stop_deforestation
                self.transformation_lndu_integrated_transitions,
                self.transformation_lndu_reallocate_land,
                self.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformation_lsmm_improve_manure_management_other,
                self.transformation_lsmm_improve_manure_management_poultry,
                self.transformation_lsmm_increase_biogas_capture,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
                self.transformation_soil_reduce_excess_fertilizer,
                self.transformation_soil_reduce_excess_lime
            ],
            attr_strategy
        )
        all_transformations.append(self.af_all_no_lvst_export_reduction_with_partial_reallocation)
        """;
         


        ##############################
        #    AGRC TRANSFORMATIONS    #
        ##############################

        self.agrc_all = sc.Transformation(
            "AGRC:ALL", 
            [
                #self.transformation_agrc_decrease_exports,
                #self.transformation_agrc_improve_crop_residue_management,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses
            ],
            attr_strategy
        )
        all_transformations.append(self.agrc_all)


        self.agrc_all_with_partial_reallocation = sc.Transformation(
            "AGRC:ALL_PLUR", 
            [
                #self.transformation_agrc_decrease_exports,
                #self.transformation_agrc_improve_crop_residue_management,
                self.transformation_agrc_improve_rice_management,
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_agrc_reduce_supply_chain_losses,
                self.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.agrc_all_with_partial_reallocation)
        

        self.agrc_improve_rice_management = sc.Transformation(
            "AGRC:DEC_CH4_RICE", 
            self.transformation_agrc_improve_rice_management,
            attr_strategy
        )
        all_transformations.append(self.agrc_improve_rice_management)


        self.agrc_decrease_exports = sc.Transformation(
            "AGRC:DEC_EXPORTS", 
            self.transformation_agrc_decrease_exports,
            attr_strategy
        )
        all_transformations.append(self.agrc_decrease_exports)


        self.agrc_expand_conservation_agriculture = sc.Transformation(
            "AGRC:INC_CONSERVATION_AGRICULTURE", 
            self.transformation_agrc_expand_conservation_agriculture,
            attr_strategy
        )
        all_transformations.append(self.agrc_expand_conservation_agriculture)


        self.agrc_increase_crop_productivity = sc.Transformation(
            "AGRC:INC_PRODUCTIVITY", 
            self.transformation_agrc_increase_crop_productivity,
            attr_strategy
        )
        all_transformations.append(self.agrc_increase_crop_productivity)


        self.agrc_increase_crop_productivity_with_partial_reallocation = sc.Transformation(
            "AGRC:INC_PRODUCTIVITY_PLUR", 
            [
                self.transformation_agrc_increase_crop_productivity,
                self.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.agrc_increase_crop_productivity_with_partial_reallocation)


        self.agrc_reduce_supply_chain_losses = sc.Transformation(
            "AGRC:DEC_LOSSES_SUPPLY_CHAIN", 
            self.transformation_agrc_reduce_supply_chain_losses,
            attr_strategy
        )
        all_transformations.append(self.agrc_reduce_supply_chain_losses)


        self.agrc_reduce_supply_chain_losses_with_partial_reallocation = sc.Transformation(
            "AGRC:DEC_LOSSES_SUPPLY_CHAIN_PLUR", 
            [
                self.transformation_agrc_reduce_supply_chain_losses,
                self.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.agrc_reduce_supply_chain_losses_with_partial_reallocation)
        


        ##############################
        #    FRST TRANSFORMATIONS    #
        ##############################

        


        ##############################
        #    LNDU TRANSFORMATIONS    #
        ##############################

        self.lndu_expand_silvopasture = sc.Transformation(
            "LNDU:INC_SILVOPASTURE", 
            self.transformation_lndu_expand_silvopasture,
            attr_strategy
        )
        all_transformations.append(self.lndu_expand_silvopasture)


        self.lndu_expand_sustainable_grazing = sc.Transformation(
            "LNDU:INC_SUSTAINABLE_GRAZING", 
            self.transformation_lndu_expand_sustainable_grazing,
            attr_strategy
        )
        all_transformations.append(self.lndu_expand_sustainable_grazing)


        self.lndu_increase_reforestation = sc.Transformation(
            "LNDU:INC_REFORESTATION", 
            self.transformation_lndu_increase_reforestation,
            attr_strategy
        )
        all_transformations.append(self.lndu_increase_reforestation)


        self.lndu_partial_reallocation = sc.Transformation(
            "LNDU:PLUR", 
            self.transformation_lndu_reallocate_land,
            attr_strategy
        )
        all_transformations.append(self.lndu_partial_reallocation)


        self.lndu_expand_silvopasture_with_partial_reallocation = sc.Transformation(
            "LNDU:INC_SILVOPASTURE_PLUR", 
            [
                self.transformation_lndu_expand_silvopasture,
                self.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.lndu_expand_silvopasture_with_partial_reallocation)


        self.lndu_stop_deforestation = sc.Transformation(
            "LNDU:DEC_DEFORESTATION", 
            self.transformation_lndu_stop_deforestation,
            attr_strategy
        )
        all_transformations.append(self.lndu_stop_deforestation)



        ##############################
        #    LSMM TRANSFORMATIONS    #
        ##############################

        self.lsmm_bundle_manure_management = sc.Transformation(
            "LSMM:BUNDLE_INC_MANAGEMENT", 
            [
                self.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformation_lsmm_improve_manure_management_other,
                self.transformation_lsmm_improve_manure_management_poultry,
                self.transformation_lsmm_increase_biogas_capture,
            ],
            attr_strategy
        )
        all_transformations.append(self.lsmm_bundle_manure_management)


        self.lsmm_improve_manure_management_cattle_pigs = sc.Transformation(
            "LSMM:INC_MANAGEMENT_CATTLE_PIGS", 
            self.transformation_lsmm_improve_manure_management_cattle_pigs,
            attr_strategy
        )
        all_transformations.append(self.lsmm_improve_manure_management_cattle_pigs)


        self.lsmm_improve_manure_management_other = sc.Transformation(
            "LSMM:INC_MANAGEMENT_OTHER", 
            self.transformation_lsmm_improve_manure_management_other,
            attr_strategy
        )
        all_transformations.append(self.lsmm_improve_manure_management_other)
        

        self.lsmm_improve_manure_management_poultry = sc.Transformation(
            "LSMM:INC_MANAGEMENT_POULTRY", 
            self.transformation_lsmm_improve_manure_management_poultry,
            attr_strategy
        )
        all_transformations.append(self.lsmm_improve_manure_management_poultry)


        self.lsmm_increase_biogas_capture = sc.Transformation(
            "LSMM:INC_CAPTURE_BIOGAS", 
            self.transformation_lsmm_increase_biogas_capture,
            attr_strategy
        )
        all_transformations.append(self.lsmm_increase_biogas_capture)

        

        ##############################
        #    LVST TRANSFORMATIONS    #
        ##############################
      
        self.lvst_all = sc.Transformation(
            "LVST:ALL", 
            [
                self.transformation_lvst_decrease_exports,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
            ],
            attr_strategy
        )
        all_transformations.append(self.lvst_all)


        self.lvst_all_with_partial_reallocation = sc.Transformation(
            "LVST:ALL_PLUR", 
            [
                self.transformation_lvst_decrease_exports,
                self.transformation_lvst_increase_productivity,
                self.transformation_lvst_reduce_enteric_fermentation,
                self.transformation_lndu_reallocate_land,
            ],
            attr_strategy
        )
        all_transformations.append(self.lvst_all_with_partial_reallocation)


        self.lvst_decrease_exports = sc.Transformation(
            "LVST:DEC_EXPORTS", 
            self.transformation_lvst_decrease_exports,
            attr_strategy
        )
        all_transformations.append(self.lvst_decrease_exports)


        self.lvst_increase_productivity = sc.Transformation(
            "LVST:INC_PRODUCTIVITY", 
            self.transformation_lvst_increase_productivity,
            attr_strategy
        )
        all_transformations.append(self.lvst_increase_productivity)


        self.lvst_increase_productivity_with_partial_reallocation = sc.Transformation(
            "LVST:INC_PRODUCTIVITY_PLUR", 
            [
                self.transformation_lvst_increase_productivity,
                self.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.lvst_increase_productivity_with_partial_reallocation)
        

        self.lvst_reduce_enteric_fermentation = sc.Transformation(
            "LVST:DEC_ENTERIC_FERMENTATION", 
            self.transformation_lvst_reduce_enteric_fermentation,
            attr_strategy
        )
        all_transformations.append(self.lvst_reduce_enteric_fermentation)
        


        ##############################
        #    SOIL TRANSFORMATIONS    #
        ##############################
        
        self.soil_reduce_excess_fertilizer = sc.Transformation(
            "SOIL:DEC_N_APPLIED", 
            self.transformation_soil_reduce_excess_fertilizer,
            attr_strategy
        )
        all_transformations.append(self.soil_reduce_excess_fertilizer)


        self.soil_reduce_excess_liming = sc.Transformation(
            "SOIL:DEC_LIME_APPLIED", 
            self.transformation_soil_reduce_excess_lime,
            attr_strategy
        )
        all_transformations.append(self.soil_reduce_excess_liming)




        ## specify dictionary of transformations and get all transformations + baseline/non-baseline

        dict_transformations = dict(
            (x.id, x) 
            for x in all_transformations
            if x.id in attr_strategy.key_values
        )
        all_transformations = sorted(list(dict_transformations.keys()))
        all_transformations_non_baseline = [
            x for x in all_transformations 
            if not dict_transformations.get(x).baseline
        ]

        transformation_id_baseline = [
            x for x in all_transformations 
            if x not in all_transformations_non_baseline
        ]
        transformation_id_baseline = transformation_id_baseline[0] if (len(transformation_id_baseline) > 0) else None


        # SET ADDDITIONAL PROPERTIES

        self.all_transformations = all_transformations
        self.all_transformations_non_baseline = all_transformations_non_baseline
        self.dict_transformations = dict_transformations
        self.transformation_id_baseline = transformation_id_baseline

        return None





    ################################################
    ###                                          ###
    ###    OTHER NON-TRANSFORMATION FUNCTIONS    ###
    ###                                          ###
    ################################################

    def build_implementation_ramp_vector(self,
        year_0: Union[int, None] = None,
        n_years_ramp: Union[int, None] = None,
    ) -> np.ndarray:
        """
        Build the implementation ramp vector

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
		- year_0: last year without change from baseline
        - n_years_ramp: number of years to go from 0 to 1
        """
        year_0 = self.year_0_ramp if (year_0 is None) else year_0
        n_years_ramp = self.n_tp_ramp if (n_years_ramp is None) else n_years_ramp

        tp_0 = self.time_periods.year_to_tp(year_0) #10
        n_tp = len(self.time_periods.all_time_periods) #25

        vec_out = np.array([max(0, min((x - tp_0)/n_years_ramp, 1)) for x in range(n_tp)])

        return vec_out



    def build_strategies_long(self,
        df_input: Union[pd.DataFrame, None] = None,
        include_base_df: bool = True,
        strategies: Union[List[str], List[int], None] = None,
    ) -> pd.DataFrame:
        """
        Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
        - df_input: baseline (untransformed) data frame to use to build 
            strategies. Must contain self.key_region and 
            self.model_attributes.dim_time_period in columns. If None, defaults
            to self.baseline_inputs
        - include_base_df: include df_input in the output DataFrame? If False,
            only includes strategies associated with transformation 
        - strategies: strategies to build for. Can be a mixture of strategy_ids
            and names. If None, runs all available. 
        """

        # INITIALIZE STRATEGIES TO LOOP OVER

        strategies = (
            self.all_transformations_non_baseline
            if strategies is None
            else [x for x in self.all_transformations_non_baseline if x in strategies]
        )
        strategies = [self.get_strategy(x) for x in strategies]
        strategies = sorted([x.id for x in strategies if x is not None])
        n = len(strategies)


        # LOOP TO BUILD
        
        t0 = time.time()
        self._log(
            f"TransformationsAFOLU.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        # initialize baseline
        df_out = (
            self.transformation_af_baseline(df_input)
            if df_input is not None
            else (
                self.baseline_inputs
                if self.baseline_inputs is not None
                else None
            )
        )

        if df_out is None:
            return None
        
        # initialize to overwrite dataframes
        iter_shift = int(include_base_df)
        df_out = [df_out for x in range(len(strategies) + iter_shift)]

        for i, strat in enumerate(strategies):
            t0_cur = time.time()
            transformation = self.get_strategy(strat)

            tp = str(type(transformation))

            if transformation is not None:
                try:
                    df_out[i + iter_shift] = transformation(df_out[i + iter_shift])
                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built transformation {self.key_strategy} = {transformation.id} ('{transformation.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )

                except Exception as e: 
                    df_out[i + iter_shift] = None
                    self._log(
                        f"\tError trying to build transformation {self.key_strategy} = {transformation.id}: {e}",
                        type_log = "error"
                    )
            else:
                df_out[i + iter_shift] = None
                self._log(
                    f"\tTransformation {self.key_strategy} not found: check that a support_classes.Transformation object has been defined associated with the code.",
                    type_log = "warning"
                )

        # concatenate, log time elapsed and completion
        df_out = (
            pd.concat(df_out, axis = 0).reset_index(drop = True)
            if len([x for x in df_out if (x is not None)]) > 0
            else self.baseline_inputs[0:0]
        )

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"TransformationsAFOLU.build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out
    


    def check_implementation_ramp(self,
        vec_implementation_ramp: np.ndarray,
        df_input: Union[pd.DataFrame, None] = None,
    ) -> Union[np.ndarray, None]:
        """
        Check that vector `vec_implementation_ramp` ramp is the same length as 
            `df_input` and that it meets specifications for an implementation
            vector. If `df_input` is not specified, use `self.baseline_inputs`. 
        
        If anything fails, return `self.vec_implementation_ramp`.
        """

        df_input = (
            self.baseline_inputs 
            if not isinstance(df_input, pd.DataFrame)
            else df_input
        )

        out = tbg.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
            self,
        )

        return out
        
        

    def get_strategy(self,
        strat: Union[int, str, None],
        field_strategy_code: str = "strategy_code",
        field_strategy_name: str = "strategy",
    ) -> None:
        """
        Get strategy `strat` based on strategy code, id, or name
        
        If strat is None or an invalid valid of strat is entered, returns None; 
            otherwise, returns the sc.Transformation object. 
            
        Function Arguments
        ------------------
        - strat: strategy id, strategy name, or strategy code to use to retrieve 
            sc.Trasnformation object
            
        Keyword Arguments
        ------------------
        - field_strategy_code: field in strategy_id attribute table containing
            the strategy code
        - field_strategy_name: field in strategy_id attribute table containing
            the strategy name
        """

        if not (sf.isnumber(strat, integer = True) | isinstance(strat, str)):
            return None

        dict_code_to_strat = self.attribute_strategy.field_maps.get(
            f"{field_strategy_code}_to_{self.attribute_strategy.key}"
        )
        dict_name_to_strat = self.attribute_strategy.field_maps.get(
            f"{field_strategy_name}_to_{self.attribute_strategy.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(strat, str):
            strat = (
                dict_name_to_strat.get(strat)
                if strat in dict_name_to_strat.keys()
                else dict_code_to_strat.get(strat)
            )

        out = (
            None
            if strat not in self.attribute_strategy.key_values
            else self.dict_transformations.get(strat)
        )
        
        return out



    def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
        """
		Clean implementation of sf._optional_log in-line using default logger. See ?sf._optional_log for more information

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)

        return None





    ############################################
    ###                                      ###
    ###    BEGIN TRANSFORMATION FUNCTIONS    ###
    ###                                      ###
    ############################################

    ##########################################################
    #    BASELINE - TREATED AS TRANSFORMATION TO INPUT DF    #
    ##########################################################
    """
    NOTE: needed for certain modeling approaches; e.g., preventing new hydro 
        from being built. The baseline can be preserved as the input DataFrame 
        by the Transformation as a passthrough (e.g., return input DataFrame) 

    NOTE: modifications to input variables should ONLY affect IPPU variables
    """

    def transformation_af_baseline(self,
        df_input: pd.DataFrame,
        baseline_with_plur: Union[bool, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Baseline" from which other transformations deviate 
            (pass through)
        """

        # some initialization
        
        df_out = df_input.copy()
        
        baseline_with_plur = (
            self.baseline_with_plur
            if baseline_with_plur is None
            else baseline_with_plur
        )


        ##  SET LAND USE REALLOCATION FACTOR

        fields_lndu_reallocation = self.model_attributes.build_variable_fields(
            self.model_afolu.modvar_lndu_reallocation_factor,
        )
        df_out[fields_lndu_reallocation] = 0.0

        # determine if setting baseline with PLUR
        if baseline_with_plur:
            df_out = tbg.transformation_general(
                df_out,
                self.model_attributes,
                {
                    self.model_afolu.modvar_lndu_reallocation_factor: {
                        "bounds": (0.0, 1),
                        "magnitude": 0.5,
                        "magnitude_type": "final_value",
                        "vec_ramp": self.vec_implementation_ramp
                    }
                },
                field_region = self.key_region,
                strategy_id = strat,
            )
        

        if sf.isnumber(strat, integer = True):
            df_out = sf.add_data_frame_fields_from_dict(
                df_out,
                {
                   self.model_attributes.dim_strategy_id: int(strat)
                },
                overwrite_fields = True,
                prepend_q = True,
            )

        return df_out



    ##############################
    #    AGRC TRANSFORMATIONS    #
    ##############################

    def get_vec_implementation_ramp(
        df_input: pd.DataFrame,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
        vec_implementation_ramp_default: np.ndarray,
    ) -> np.ndarray:
        """
        Check an implmentation ramp against a default.
        """

        return None



    def transformation_agrc_decrease_exports(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude: float = 0.5,
        magnitude_type: str = "baseline_scalar",
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Decrease Exports" AGRC transformation on input 
            DataFrame df_input (reduce by 50%)

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of decrease in exports. If using the default 
            value of `magnitude_type == "scalar"`, this magnitude will scale the
            final time value downward by this factor. If magnitude_type changes,
            then the behavior of the trasnformation will change.
        - magnitude_type: type of magnitude, as specified in 
            transformers.lib.general.transformations_general. See 
            ?transformers.lib.general.transformations_general for more 
            information on the specification of magnitude_type for general
            transformation values. The 
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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



    def transformation_agrc_expand_conservation_agriculture(self,
        df_input: Union[pd.DataFrame, None] = None,
        dict_categories_to_magnitude: Union[Dict[str, float], None] = None,
        magnitude_burned: float = 0.0,
        magnitude_removed: float = 0.5,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Expand Conservation Agriculture" AGRC transformation on 
            input DataFrame df_input. 
            
        NOTE: Sets a new floor for F_MG (as described in in V4 Equation 2.25 
            (2019R)) to reduce losses of soil organic carbon through no-till 
            in cropland + reduces removals and burning of crop residues, 
            increasing residue covers on fields.
        

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_categories_to_magnitude: conservation agriculture is practically
            applied to only select crop types. Use the dictionary to map 
            SISEPUEDE crop categories to target implementation magnitudes.
            * If None, maps to the following dictionary:

                {
                    "cereals": 0.8,
                    "fibers": 0.8,
                    "other_annual": 0.8,
                    "pulses": 0.5,
                    "tubers": 0.5,
                    "vegetables_and_vines": 0.5,
                }
                
        - magnitude_burned: target fraction of residues that are burned
        - magnitude_removed: maximum fraction of residues that are removed
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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



    def transformation_agrc_improve_crop_residue_management(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_burned: float = 0.0,
        magnitude_removed: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Crop Management" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude_burned: target fraction of residues that are burned
        - magnitude_removed: maximum fraction of residues that are removed
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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



    def transformation_agrc_improve_rice_management(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Rice Management" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        
        df_out = tba.transformation_agrc_improve_rice_management(
            df_input,
            0.6, # CHANGEDFORINDIA 0.45
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out


    
    def transformation_agrc_increase_crop_productivity(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Crop Productivity" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

        
        df_out = tba.transformation_agrc_increase_crop_productivity(
            df_input,
            # CHANGEDFORINDIA - ORIG 0.2
            0.3, # can be specified as dictionary to affect different crops differently 
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def transformation_agrc_decrease_climate_productivity_climate_india(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Crop Productivity" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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


        # see excel for this
        dict_climate_impacts = {
            "cereals": -0.1113,
            "fruits": -0.097,
            "other_annual": -0.1677,
            "pulses": -0.04,
            "rice": -0.177,
            "tubers": 0.099,
            "vegetables_and_vines": -0.097,
        }
        impacts_mean = np.array(list(dict_climate_impacts.values())).mean()
        
        df_out = df_input.copy()

        # get ag cats
        cats_ag = self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_agrc)
        cats_ag = cats_ag.key_values

        for cat in cats_ag:
            # get mean value if otherwise undefined
            val = dict_climate_impacts.get(cat, impacts_mean)
        
            df_out = tba.transformation_agrc_increase_crop_productivity(
                df_out,
                val, # can be specified as dictionary to affect different crops differently 
                vec_implementation_ramp,
                self.model_attributes,
                categories = [cat],
                field_region = self.key_region,
                model_afolu = self.model_afolu,
                strategy_id = strat,
            )

        return df_out



    def transformation_agrc_reduce_supply_chain_losses(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Supply Chain Losses" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

        
        df_out = tba.transformation_agrc_reduce_supply_chain_losses(
            df_input,
            0.3,
            vec_implementation_ramp,
            self.model_attributes,
            model_afolu = self.model_afolu,
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out


    
    ##############################
    #    FRST TRANSFORMATIONS    #
    ##############################

    def transformation_lndu_increase_reforestation(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Reforestation" FRST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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


        df_out = tba.transformation_frst_increase_reforestation(
            df_input, 
            0.1, # double forests INDIA
            vec_implementation_ramp,
            self.model_attributes,
            cats_inflow_restriction = ["croplands", "other"], # SET FOR INDIA--NEED A BETTER WAY TO DETERMINE
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )

        return df_out



    def transformation_lndu_stop_deforestation(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Stop Deforestation" FRST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
            0.99999,
            vec_ramp,#self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    ##############################
    #    LNDU TRANSFORMATIONS    #
    ##############################

    def transformation_lndu_expand_silvopasture(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Increase the use of silvopasture by shifting pastures to secondary 
            forest. 
            
        NOTE: This transformation relies on modifying transition matrices, which 
            can compound some minor numerical errors in the crude implementation 
            taken here. Final area prevalences may not reflect 
            get_matrix_column_scalarget_matrix_column_scalarprecise shifts.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        

        df_out = tba.transformation_lndu_increase_silvopasture(
            df_input,
            0.1, # CHANGEDFORINDIA - ORIG 0.1
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    


    def transformation_lndu_expand_sustainable_grazing(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Expand Sustainable Grazing" LNDU transformation on input 
            DataFrame df_input. 
            
        NOTE: Sets a new floor for F_MG (as described in in V4 Equation 2.25 
            (2019R)) through improved grassland management (grasslands).
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
                self.model_afolu.modvar_lndu_frac_pastures_improved: {
                    "bounds": (0.0, 1.0),
                    "magnitude": 0.95,
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def transformation_lndu_integrated_transitions(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Increase the use of silvopasture by shifting pastures to secondary 
            forest AND reduce deforestation. Sets orderering of these 
            transformations for bundles.
            
        NOTE: This transformation relies on modifying transition matrices, which 
            can compound some minor numerical errors in the crude implementation 
            taken here. Final area prevalences may not reflect precise shifts.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        df_out = self.transformation_lndu_expand_silvopasture(
            df_input,
            strat = strat,
            vec_implementation_ramp = vec_implementation_ramp,
        )
        # then deforestation
        df_out = self.transformation_lndu_stop_deforestation(
            df_out,
            strat = strat,
            vec_implementation_ramp = vec_implementation_ramp,
        )
        
        return df_out


    
    def transformation_lndu_reallocate_land(self,HEREHERE
        df_input: Union[pd.DataFrame, None] = None,
        baseline_with_plur: Union[bool, None] = None,s
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Support land use reallocation in specification of multiple 
            transformations.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
        """

        # check if baseline contains PLUR
        baseline_with_plur = (
            self.baseline_with_plur
            if not isinstance(baseline_with_plur, bool)
            else baseline_with_plur
        )

        # check input dataframe
        df_out = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # if baseline includes PLUR, don't modify
        if not baseline_with_plur:
            df_out = tbg.transformation_general(
                df_out,
                self.model_attributes,
                {
                    self.model_afolu.modvar_lndu_reallocation_factor: {
                        "bounds": (0.0, 1),
                        "magnitude": 0.5,
                        "magnitude_type": "final_value",
                        "vec_ramp": vec_implementation_ramp
                    }
                },
                field_region = self.key_region,
                strategy_id = strat,
            )

        return df_out


        

    ##############################
    #    LSMM TRANSFORMATIONS    #
    ##############################

    def transformation_lsmm_improve_manure_management_cattle_pigs(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Livestock Manure Management for Cattle and Pigs" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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


        # some key transformation components
        frac_managed = 0.95
        dict_lsmm_pathways = {
            "anaerobic_digester": 0.625*frac_managed,
            "composting": 0.125*frac_managed,
            "daily_spread": 0.25*frac_managed,
            #"solid_storage": 0.125*frac_managed
        }
        vec_lvst_cats = [
            "cattle_dairy",
            "cattle_nondairy",
            "pigs",
        ]

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



    def transformation_lsmm_improve_manure_management_other(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Livestock Manure Management for Other Animals" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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


        # some key transformation components
        frac_managed = 0.95
        dict_lsmm_pathways = {
            "anaerobic_digester": 0.50*frac_managed,
            "composting": 0.25*frac_managed,
            "dry_lot": 0.125*frac_managed,
            "daily_spread": 0.125*frac_managed,
        }
        vec_lvst_cats = [
            "buffalo",
            "goats",
            "horses",
            "mules",
            "sheep",
        ]

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
    


    def transformation_lsmm_improve_manure_management_poultry(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Livestock Manure Management for Poultry" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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


        # some key transformation components
        frac_managed = 0.95
        dict_lsmm_pathways = {
            "anaerobic_digester": 0.50*frac_managed,
            "poultry_manure": 0.5*frac_managed
        }
        vec_lvst_cats = [
            "chickens"
        ]

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



    def transformation_lsmm_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Biogas Capture at Anaerobic Decomposition 
            Facilities" transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

            
        # update the biogas recovery factor
        df_out = tbg.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_afolu.modvar_lsmm_rf_biogas: {
                    "bounds": (0.0, 1),
                    "magnitude": 1.0, # CHANGEDFORINDIA 0.9
                    "magnitude_type": "final_value_floor",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    ##############################
    #    LVST TRANSFORMATIONS    #
    ##############################

    def transformation_lvst_decrease_exports(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Decrease Exports" LVST transformation on input 
            DataFrame df_input (reduce by 50%)
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
                self.model_afolu.modvar_lvst_equivalent_exports: {
                    "bounds": (0.0, np.inf),
                    "magnitude": 0.5,
                    "magnitude_type": "baseline_scalar",
                    "vec_ramp": vec_implementation_ramp
                },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )
        
        return df_out



    def transformation_lvst_increase_productivity(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Livestock Productivity" LVST transformation on 
            input DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

        
        df_out = tba.transformation_lvst_increase_productivity(
            df_input,
            0.3, # CHANGEDFORINDIA 0.2
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    def transformation_lvst_reduce_enteric_fermentation(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Enteric Fermentation" LVST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        
        
        df_out = tba.transformation_lvst_reduce_enteric_fermentation(
            df_input,
            {
                "buffalo": 0.4, # CHANGEDFORINDIA 0.4
                "cattle_dairy": 0.4, # CHANGEDFORINDIA 0.4
                "cattle_nondairy": 0.4, # CHANGEDFORINDIA 0.4
                "goats": 0.56,
                "sheep": 0.56
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out



    ##############################
    #    SOIL TRANSFORMATIONS    #
    ##############################

    def transformation_soil_reduce_excess_fertilizer(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Excess Fertilizer" SOIL transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

        
        df_out = tba.transformation_soil_reduce_excess_fertilizer(
            df_input,
            {
                "fertilizer_n": 0.2
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    


    def transformation_soil_reduce_excess_lime(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Excess Liming" SOIL transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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

        
        df_out = tba.transformation_soil_reduce_excess_fertilizer(
            df_input,
            {
                "lime": 0.2
            },
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_afolu = self.model_afolu,
            strategy_id = strat,
        )
        
        return df_out
    
