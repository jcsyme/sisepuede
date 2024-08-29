
import logging
import numpy as np
import os, os.path
import pandas as pd
import time
from typing import *


import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
#import sisepuede.legacy.transformations_base_circular_economy as tbc
import sisepuede.models.circular_economy as mc
import sisepuede.transformers.lib._baselib_circular_economy as tbc
import sisepuede.utilities.support_functions as sf




class TransformationsCircularEconomy:
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
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
		used in Julia NemoMod Electricity model
        * If None, defaults to a temporary path sql database
    - logger: optional logger object
    - model_circecon: optional CircularEconomy object to pass for property and 
        method access
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes
            argument are the same.
    """
    
    def __init__(self,
        model_attributes: ma.ModelAttributes,
        dict_config: Dict,
        field_region: Union[str, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
		logger: Union[logging.Logger, None] = None,
        model_circecon: Union[mc.CircularEconomy, None] = None,
    ):

        self.logger = logger

        self._initialize_attributes(field_region, model_attributes)
        self._initialize_config(dict_config = dict_config)
        self._initialize_models(model_circecon = model_circecon)
        self._initialize_parameters(dict_config = dict_config)
        self._initialize_ramp()
        self._initialize_baseline_inputs(df_input)
        self._initialize_transformations()




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
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.key_region
            * self.model_attributes
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in transformations_energy")

        # get strategy attribute, baseline strategy, and some fields
        attribute_strategy = model_attributes.get_dimensional_attribute_table(
            model_attributes.dim_strategy_id
        )

        baseline_strategy = int(
            attribute_strategy.table[
                attribute_strategy.table["baseline_strategy_id"] == 1
            ][attribute_strategy.key].iloc[0]
        )
        field_region = model_attributes.dim_region if (field_region is None) else field_region

        # set some useful classes
        time_periods = sc.TimePeriods(model_attributes)
        regions = sc.Regions(model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_strategy = attribute_strategy
        self.baseline_strategy = baseline_strategy
        self.key_region = field_region
        self.key_strategy = attribute_strategy.key
        self.model_attributes = model_attributes
        self.time_periods = time_periods
        self.regions = regions

        return None
    


    def _initialize_baseline_inputs(self,
        df_inputs: Union[pd.DataFrame, None],
    ) -> None:
        """
        Initialize the baseline inputs dataframe based on the initialization 
            value of df_inputs. It not initialied, sets as None. Sets the 
            following properties:

            * self.baseline_inputs
        """

        baseline_inputs = (
            self.transformation_ce_baseline(df_inputs, strat = self.baseline_strategy) 
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
        model_circecon: Union[mc.CircularEconomy, None] = None,
    ) -> None:
        """
        Define model objects for use in variable access and base estimates.

        Function Arguments
        ------------------
        - model_circecon: optional CircularEconomy object to pass for property 
            and method access
        """

        model_circecon = (
            mc.CircularEconomy(self.model_attributes)
            if model_circecon is None
            else model_circecon
        )

        self.model_circecon = model_circecon

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
            self.transformation_ce_baseline, 
            attr_strategy
        )
        all_transformations.append(self.baseline)



        #####################################################
        #    CircularEconomy SECTOR-WIDE TRANSFORMATIONS    #
        #####################################################
        
        self.ce_all = sc.Transformation(
            "CE:ALL", 
            [
                self.transformation_trww_increase_biogas_capture,
                self.transformation_trww_increase_septic_compliance, #CHANGEDFORINDIA, SHOULD BE COMMENTED OUT
                self.transformation_wali_improve_sanitation_industrial,
                self.transformation_wali_improve_sanitation_rural,
                self.transformation_wali_improve_sanitation_urban,
                self.transformation_waso_decrease_food_waste,
                self.transformation_waso_increase_anaerobic_treatment_and_composting,
                self.transformation_waso_increase_biogas_capture,
                self.transformation_waso_increase_energy_from_biogas,
                self.transformation_waso_increase_energy_from_incineration,
                self.transformation_waso_increase_landfilling,
                self.transformation_waso_increase_recycling,
            ],
            attr_strategy
        )
        all_transformations.append(self.ce_all)



        ##############################
        #    TRWW TRANSFORMATIONS    #
        ##############################

        self.trww_all = sc.Transformation(
            "TRWW:ALL", 
            [
                self.transformation_trww_increase_biogas_capture,
                #self.transformation_trww_increase_septic_compliance,
            ],
            attr_strategy
        )
        all_transformations.append(self.trww_all)


        self.trww_increase_biogas_capture = sc.Transformation(
            "TRWW:INC_CAPTURE_BIOGAS", 
            self.transformation_trww_increase_biogas_capture,
            attr_strategy
        )
        all_transformations.append(self.trww_increase_biogas_capture)


        self.trww_increase_septic_compliance = sc.Transformation(
            "TRWW:INC_COMPLIANCE_SEPTIC", 
            self.transformation_trww_increase_septic_compliance,
            attr_strategy
        )
        all_transformations.append(self.trww_increase_septic_compliance)


        ##############################
        #    WALI TRANSFORMATIONS    #
        ##############################

        self.wali_all = sc.Transformation(
            "WALI:ALL", 
            [
                self.transformation_wali_improve_sanitation_industrial,
                self.transformation_wali_improve_sanitation_rural,
                self.transformation_wali_improve_sanitation_urban
            ],
            attr_strategy
        )
        all_transformations.append(self.wali_all)


        self.wali_improve_sanitation_industrial = sc.Transformation(
            "WALI:INC_TREATMENT_INDUSTRIAL", 
            self.transformation_wali_improve_sanitation_industrial,
            attr_strategy
        )
        all_transformations.append(self.wali_improve_sanitation_industrial)


        self.wali_improve_sanitation_rural = sc.Transformation(
            "WALI:INC_TREATMENT_RURAL", 
            self.transformation_wali_improve_sanitation_rural,
            attr_strategy
        )
        all_transformations.append(self.wali_improve_sanitation_rural)


        self.wali_improve_sanitation_urban = sc.Transformation(
            "WALI:INC_TREATMENT_URBAN", 
            self.transformation_wali_improve_sanitation_urban,
            attr_strategy
        )
        all_transformations.append(self.wali_improve_sanitation_urban)



        ##############################
        #    WASO TRANSFORMATIONS    #
        ##############################

        self.waso_all = sc.Transformation(
            "WASO:ALL", 
            [
                self.transformation_waso_decrease_food_waste,
                self.transformation_waso_increase_anaerobic_treatment_and_composting,
                self.transformation_waso_increase_biogas_capture,
                self.transformation_waso_increase_energy_from_biogas,
                self.transformation_waso_increase_energy_from_incineration,
                self.transformation_waso_increase_landfilling,
                self.transformation_waso_increase_recycling,
            ],
            attr_strategy
        )
        all_transformations.append(self.waso_all)


        self.waso_descrease_consumer_food_waste = sc.Transformation(
            "WASO:DEC_CONSUMER_FOOD_WASTE",
            self.transformation_waso_decrease_food_waste, 
            attr_strategy
        )
        all_transformations.append(self.waso_descrease_consumer_food_waste)

        
        self.waso_increase_anaerobic_treatment_and_composting = sc.Transformation(
            "WASO:INC_ANAEROBIC_AND_COMPOST", 
            self.transformation_waso_increase_anaerobic_treatment_and_composting, 
            attr_strategy
        )
        all_transformations.append(self.waso_increase_anaerobic_treatment_and_composting)


        self.waso_increase_biogas_capture = sc.Transformation(
            "WASO:INC_CAPTURE_BIOGAS", 
            self.transformation_waso_increase_biogas_capture, 
            attr_strategy
        )
        all_transformations.append(self.waso_increase_biogas_capture)


        self.waso_energy_from_biogas = sc.Transformation(
            "WASO:INC_ENERGY_FROM_BIOGAS", 
            self.transformation_waso_increase_energy_from_biogas, 
            attr_strategy
        )
        all_transformations.append(self.waso_energy_from_biogas)


        self.waso_energy_from_incineration = sc.Transformation(
            "WASO:INC_ENERGY_FROM_INCINERATION", 
            self.transformation_waso_increase_energy_from_incineration, 
            attr_strategy
        )
        all_transformations.append(self.waso_energy_from_incineration)


        self.waso_increase_landfilling = sc.Transformation(
            "WASO:INC_LANDFILLING", 
            self.transformation_waso_increase_landfilling, 
            attr_strategy
        )
        all_transformations.append(self.waso_increase_landfilling)

        
        self.waso_increase_recycling = sc.Transformation(
            "WASO:INC_RECYCLING", 
            self.transformation_waso_increase_recycling, 
            attr_strategy
        )
        all_transformations.append(self.waso_increase_recycling)


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
            f"TransformationsCircularEconomy.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        # initialize baseline
        df_out = (
            self.transformation_ce_baseline(df_input)
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

            if transformation is not None:
                try:
                    df_out[i + iter_shift] = transformation(df_out[i + iter_shift])
                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built transformation {self.key_strategy} = {transformation.id} ('{transformation.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )

                except Exception as e: 
                    df_out[i + 1] = None
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
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"TransformationsCircularEconomy.build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out

        

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

    NOTE: modifications to input variables should ONLY affect CircularEconomy 
        variables
    """

    def transformation_ce_baseline(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Baseline" from which other transformations deviate 
            (pass through)
        """
        df_out = df_input.copy()

        if isinstance(strat, int):
            df_out = sf.add_data_frame_fields_from_dict(
                df_out,
                {
                   self.model_attributes.dim_strategy_id: strat
                },
                prepend_q = True,
                overwrite_fields = True
            )

        return df_out



    ##############################
    #    TRWW TRANSFORMATIONS    #
    ##############################

    
    def transformation_trww_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Biogas Capture at Wastewater Treatment Plants" 
            TRWW transformation on input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_trww_increase_gas_capture(
            df_input,
            # as float, applies to both landfill and biogas; specify as dict to break out; 
            # e.g., {"treated_advanced_anaerobic": 0.85, "treated_secondary_anaerobic": 0.5}
            1.0, # CHANGEDFORINDIA 0.85
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out


    
    def transformation_trww_increase_septic_compliance(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Compliance" TRWW transformation on input 
            DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_trww_increase_septic_compliance(
            df_input,
            0.9, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out



    ##############################
    #    WALI TRANSFORMATIONS    #
    ##############################

    def transformation_wali_improve_sanitation_industrial(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Industrial Sanitation" WALI transformation on 
            input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_industrial",
            {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_aerobic": 0.1,
                "treated_secondary_anaerobic": 0.1,
            },
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat,
        )


        return df_out



    def transformation_wali_improve_sanitation_rural(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Rural Sanitation" WALI transformation on 
            input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_domestic_rural",
            {
                "treated_septic": 1.0, 
            },
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat,
        )


        return df_out



    def transformation_wali_improve_sanitation_urban(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Improve Urban Sanitation" WALI transformation on 
            input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        df_out = tbc.transformation_wali_improve_sanitation(
            df_input,
            "ww_domestic_urban",
            {
                "treated_advanced_aerobic": 0.3,
                "treated_advanced_anaerobic": 0.3,
                "treated_secondary_aerobic": 0.2,
                "treated_secondary_anaerobic": 0.2,
            },
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat,
        )

        return df_out



    ##############################
    #    WASO TRANSFORMATIONS    #
    ##############################

    def transformation_waso_decrease_food_waste(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Decrease Municipal Solid Waste" WASO transformation on 
            input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # get categories and dictionary to specify parameters (move to config eventually)
        categories = self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_waso).key_values
        #dict_specify = dict((x, 0.25) for x in categories)
        #dict_specify.update({"food": 0.3})
        dict_specify = {"food": 0.3}

        df_out = tbc.transformation_waso_decrease_municipal_waste(
            df_input,
            dict_specify,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out



    def transformation_waso_increase_anaerobic_treatment_and_composting(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Anaerobic Treatment and Composting" WASO 
            transformation on input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_anaerobic_treatment_and_composting(
            df_input,
            0.475,
            0.475,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out


    
    def transformation_waso_increase_biogas_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Biogas Capture at Anaerobic Treatment Facilities
            and Landfills" WASO transformation on input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_gas_capture(
            df_input,
            1.0, # as float, applies to both landfill and biogas; specify as dict to break out # CHANGEDFORINDIA 0.85
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out


    
    def transformation_waso_increase_energy_from_biogas(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Energy from Captured Biogas" WASO transformation on input 
            DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_energy_from_biogas(
            df_input,
            0.85,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out
    


    def transformation_waso_increase_energy_from_incineration(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Energy from Solid Waste Incineration" WASO transformation 
            on input DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_energy_from_incineration(
            df_input,
            0.85,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out


    
    def transformation_waso_increase_landfilling(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Landfilling" WASO transformation on input 
            DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_landfilling(
            df_input,
            1.0,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out



    def transformation_waso_increase_recycling(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Recycling" WASO transformation on input 
            DataFrame df_input
        """
        # check input dataframe
        df_input = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        df_out = tbc.transformation_waso_increase_recycling(
            df_input,
            0.95,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_circecon = self.model_circecon,
            strategy_id = strat
        )

        return df_out

