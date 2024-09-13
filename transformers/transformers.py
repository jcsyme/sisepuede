
import datetime as dt
import logging
import numpy as np
import os, os.path
import pandas as pd
import shutil
import time
from typing import *


from sisepuede.core.attribute_table import AttributeTable
from sisepuede.models.afolu import is_sisepuede_model_afolu
from sisepuede.models.circular_economy import is_sisepuede_model_circular_economy
from sisepuede.models.energy_consumption import is_sisepuede_model_nfp_energy
from sisepuede.models.energy_production import is_sisepuede_model_fuel_production
from sisepuede.models.ippu import is_sisepuede_model_ippu
from sisepuede.models.socioeconomic import is_sisepuede_model_socioeconomic

import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.data_management.ingestion as ing
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.manager.sisepuede_models as sm
import sisepuede.transformers.afolu as dta
import sisepuede.transformers.circular_economy as dtc
import sisepuede.transformers.energy as dte
import sisepuede.transformers.ippu as dti
import sisepuede.transformers.lib._baselib_cross_sector as tbc
import sisepuede.transformers.lib._baselib_general as tbg
import sisepuede.transformers.lib._classes as trl
import sisepuede.utilities._toolbox as sf




#
#    SET SOME DEFAULT CONFIGURATION VALUES
#

def get_dict_config_default(
) -> dict:
    """
    Retrieve the dictionary of default configuration values for transformers.
    """
    dict_out = {
        # ENTC categories that are capped to 0 investment
        "categories_entc_max_investment_ramp": [
            "pp_hydropower"
        ],

        # ENTC categories considered renewable sources
        "categories_entc_renewable": [
            "pp_geothermal",
            "pp_hydropower",
            "pp_ocean",
            "pp_solar",
            "pp_wind"
        ],

        # INEN categories that have high heat
        "categories_inen_high_heat": [
            "cement", 
            "chemicals", 
            "glass", 
            "lime_and_carbonite", 
            "metals"
        ],

        # Target minimum share of production fractions for power plants in the renewable target tranformation
        "dict_entc_renewable_target_msp": {
            "pp_solar": 0.15,
            "pp_geothermal": 0.1,
            "pp_wind": 0.15
        },

        # fraction of high heat that can be electrified and hydrogenized
        "frac_inen_high_temp_elec_hydg": 0.5*0.45,

        # fraction of low temperature heat energy demand that can be electrified
        "frac_inen_low_temp_elec": 0.95*0.45,

        # number of time periods in the ramp
        "n_tp_ramp": None,

        # shape values for implementing caps on new technologies (description below)
        "vir_renewable_cap_delta_frac": 0.0075,
        "vir_renewable_cap_max_frac": 0.125,

        # first year to start transformations--default is to 2 years from present
        "year_0_ramp": dt.datetime.now().year + 2
    }

    return dict_out





class Transformers:
    """
    Build collection of Transformers that are used to define transformations.

    Includes some information on

    Initialization Arguments
    ------------------------
    - model_attributes: ModelAttributes object used to manage variables and
        coordination
    - dict_config: configuration dictionary used to pass parameters to 
        transformations. See ?TransformerEnergy._initialize_parameters() for
        more information on requirements.
    - dir_jl: location of Julia directory containing Julia environment and 
        support modules
    - fp_nemomod_reference_files: directory housing reference files called by
        NemoMod when running electricity model. Required to access data in 
        EnergyProduction. Needs the following CSVs:

        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
    - baseline_with_plur: set to True to let the baseline include partial land
        use reallocation in the baseline--passed to TransformersAFOLU as
        a keyword argument.
        * NOTE: If True, then transformation_lndu_reallocate_land() 
            has no effect.
        * NOTE: this is set in both `self.transformation_af_baseline()` and
            `self.transformation_lndu_reallocate_land()` separately
    - fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
        used in Julia NemoMod Electricity model
        * If None, defaults to a temporary path sql database
    - logger: optional logger object
    - model_afolu: optional AFOLU object to pass for property and method access.
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes
            argument are the same.
    - model_circecon: optional CircularEconomy object to pass for property and
        method access.
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes
            argument are the same.
    - model_enerprod: optional EnergyProduction object to pass for property and 
        method access.
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes
            argument are the same.
    - model_ippu: optional IPPU object to pass for property and method access.
        * NOTE: if passing, ensure that the ModelAttributes objects used to 
            instantiate the model + what is passed to the model_attributes
            argument are the same.
    - use_demo_template_on_missing: tries to instantiate a blank template if
        a template for a target region is missing. 
    - **kwargs 
    """
    
    def __init__(self,
        dict_config: Dict,
        baseline_with_plur: bool = False,
        df_input: Union[pd.DataFrame, None] = None,
        field_region: Union[str, None] = None,
        logger: Union[logging.Logger, None] = None,
        regex_template_prepend: str = "sisepuede_run",
        regions: Union[List[str], None] = None,
        use_demo_template_on_missing: bool = True,
        **kwargs
    ):

        self.logger = logger

        self._initialize_file_structure(
            regex_template_prepend = regex_template_prepend,
        )

        self._initialize_attributes(
            field_region,
        )

        self._initialize_config(dict_config = dict_config)
        self._initialize_parameters()
        self._initialize_ramp()

        # set transformations by sector, models (which come from sectoral transformations)
        self._initialize_sectoral_transformations(
            df_input = df_input,
            magnitude_lurf = magnitude_lurf,
            **kwargs,
        )
        self._initialize_models()
        self._initialize_transformations()
        
        return None




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def get_ramp_characteristics(self,
        n_tp_ramp: Union[int, None] = None,
        tp_0_ramp: Union[int, None] = None,
    ) -> List[str]:
        """
        Get parameters for the implementation of transformations. Returns a 
            tuple with the following elements:

            (
                n_tp_ramp,
                tp_0_ramp, 
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires those keys in dict_config to set. If not found, will set
            the following defaults:
                * year_0_ramp: 9th year (10th time period)
                * n_tp_ramp: n_tp - t0_ramp - 1 (ramps to 1 at final time 
                    period)

        Keyword Arguments
        -----------------
        - n_tp_ramp: number of time periods to increase to full implementation. 
            If None, defaults to final time period
        - tp_0_ramp: first time period of ramp (last == 0)
        """


        n_tp = len(self.time_periods.all_time_periods)
        
        # set y0 default if not specified correctly
        if not sf.isnumber(tp_0_ramp, integer = True):

            year_0_ramp = dt.datetime.now().year + 2
            tp_0_ramp = self.time_periods.year_to_tp(year_0_ramp)

            if tp_0_ramp is None:
                msg = f"Error setting default time period: year {year_0_ramp} undefined. Explicitly specify tp_0_ramp in Transformers"
                raise RuntimeError(msg)

        # ensure it's in the set of defined time periods
        tp_0_ramp = self.time_periods.get_closest_time_period(tp_0_ramp)

        # shift by 2--1 to account for baseline having no uncertainty, 1 for py reindexing
        default_tp_ramp = n_tp - tp_0_ramp - 1
        n_tp_ramp = (
            default_tp_ramp
            if not sf.isnumber(n_tp_ramp, integer = True)
            else min(default_tp_ramp, n_tp_ramp)
        )
       
        tup_out = (
            n_tp_ramp,
            year_0_ramp, 
        )

        return tup_out



    def _initialize_attributes(self,
        field_region: Union[str, None],
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_transformer_code
            * self.key_region
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (self.model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in TransformersIPPU")

        # get strategy attribute, baseline strategy, and some fields
        attribute_transformer_code = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_transformer_code
        )

        field_region = (
            self.model_attributes.dim_region 
            if (field_region is None) 
            else field_region
        )

        # set some useful classes
        time_periods = sc.TimePeriods(self.model_attributes)
        regions = sc.Regions(self.model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_transformer_code = attribute_transformer_code
        self.key_region = field_region
        self.key_transformer_code = attribute_transformer_code.key
        self.time_periods = time_periods
        self.regions_manager = regions

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
            self.transformer_baseline(
                df_inputs, 
                strat = None,
            ) 
            if isinstance(df_inputs, pd.DataFrame) 
            else None
        )

        
        ##  SET PROPERTIES
        
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
            * "categories_entc_pps_to_cap": list of power plant categories to
                prevent from new growth by capping MSP
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
            * "tp_0_ramp": last time period with no diversion from baseline 
                strategy (baseline for implementation ramp)
            * "vir_renewable_cap_delta_frac": change (applied downward from 
                "vir_renewable_cap_max_frac") in cap for for new technology
                capacities available to build in time period while transitioning
                to renewable capacties. Default is 0.01 (will decline by 1% each
                time period after "year_0_ramp")
            * "vir_renewable_cap_max_frac": cap for for new technology 
                capacities available to build in time period while transitioning
                to renewable capacties; entered as a fraction of estimated
                capacity in "year_0_ramp". Default is 0.05
            
        """

        dict_config = {} if not isinstance(dict_config, dict) else dict_config

        # set parameters
        self.config = dict_config

        self.key_config_cats_entc_max_investment_ramp = "categories_entc_max_investment_ramp"
        self.key_config_cats_entc_pps_to_cap = "categories_entc_pps_to_cap"
        self.key_config_cats_entc_renewable = "categories_entc_renewable"
        self.key_config_cats_inen_high_heat = "categories_inen_high_heat",
        self.key_config_dict_entc_renewable_target_msp = "dict_entc_renewable_target_msp"
        self.key_config_frac_inen_high_temp_elec_hydg = "frac_inen_low_temp_elec"
        self.key_config_frac_inen_low_temp_elec = "frac_inen_low_temp_elec"
        self.key_config_n_tp_ramp = "n_tp_ramp"
        self.key_config_tp_0_ramp = "tp_0_ramp" 
        self.key_config_vir_renewable_cap_delta_frac = "vir_renewable_cap_delta_frac"
        self.key_config_vir_renewable_cap_max_frac = "vir_renewable_cap_max_frac"

        return None
    


    def _initialize_file_structure(self,
        dir_ingestion: Union[str, None] = None,
        id_str: Union[str, None] = None,
        regex_template_prepend: str = "sisepuede_run"
    ) -> None:
        """
        Intialize the SISEPUEDEFileStructure object and model_attributes object.
            Initializes the following properties:

            * self.file_struct
            * self.model_attributes

        Optional Arguments
        ------------------
        - dir_ingestion: directory containing templates for ingestion. The
            ingestion directory should include subdirectories for each template
            class that may be run, including:
                * calibrated: input variables that are calibrated for each
                    region and sector
                * demo: demo parameters that are independent of region (default
                    in quick start)
                * uncalibrated: preliminary input variables defined for each
                    region that have not yet been calibrated
            The calibrated and uncalibrated subdirectories require separate
                subdrectories for each region, each of which contains an input
                template for each
        - id_str: Optional id_str used to create AnalysisID (see ?AnalysisID
            for more information on properties). Can be used to set outputs for
            a previous ID/restore a session.
            * If None, creates a unique ID for the session (used in output file
                names)
        """

        self.file_struct = None
        self.model_attributes = None

        try:
            self.file_struct = sfs.SISEPUEDEFileStructure(
                initialize_directories = False,
                logger = self.logger,
                regex_template_prepend = regex_template_prepend
            )
            self._log(f"Successfully initialized SISEPUEDEFileStructure.", type_log = "info")

        except Exception as e:
            self._log(f"Error trying to initialize SISEPUEDEFileStructure: {e}", type_log = "error")
            raise RuntimeError()

        self.model_attributes = self.file_struct.model_attributes

        return None



    def _initialize_models(self,
        **kwargs,
    ) -> None:
        """
        Define model objects for use in variable access and base estimates. Sets
            the following properties:

            * self.model_afolu
            * self.model_circular_economy
            * self.model_enerprod
            * self.model_enercons
            * self.model_ippu
            * self.model_socioeconomic

        Existing models can be passed to as keyword arguments
        """

        ##  INITIALIZE MODELS

        models = sm.SISEPUEDEModels(
            self.model_attributes,
            allow_electricity_run = False,
            fp_julia = self.file_struct.dir_jl,
            fp_nemomod_reference_files = self.file_struct.dir_ref_nemo,
            fp_nemomod_temp_sqlite_db = self.file_struct.fp_nemomod_temp_sqlite_db,
            initialize_julia = False,
            logger = self.logger,
        )

        # check AFOLU
        model_afolu = kwargs.get("model_afolu")
        if not is_sisepuede_model_afolu(model_afolu):
            model_afolu = models.afolu


        # check CircularEconomy
        model_circular_economy = kwargs.get("model_circular_economy")
        if not is_sisepuede_model_circular_economy(model_circular_economy):
            model_circular_economy = models.model_circecon
        
        # check Energy Consumption
        model_enercons = kwargs.get("model_enercons")
        if not is_sisepuede_model_nfp_energy(model_enercons):
            model_enercons = models.model_enercons
        
        # check Energy Production
        model_enerprod = kwargs.get("model_enerprod")
        if not is_sisepuede_model_fuel_production(model_enerprod):
            model_enerprod = models.model_enerprod

        # check IPPU
        model_ippu = kwargs.get("model_ippu")
        if not is_sisepuede_model_ippu(model_ippu):
            model_ippu = models.model_ippu
        
        # check Socioeconomic
        model_socioeconomic = kwargs.get("model_socioeconomic")
        if not is_sisepuede_model_socioeconomic(model_socioeconomic):
            model_socioeconomic = models.model_socioeconomic


        ##  SET PROPERTIES

        self.model_afolu = model_afolu
        self.model_circular_economy = model_circular_economy
        self.model_enerprod = model_enerprod
        self.model_enercons = model_enercons
        self.model_ippu = model_ippu
        self.model_socioeconomic = model_socioeconomic

        return None


    
    def _initialize_parameters(self,
        dict_config: Union[Dict[str, Any], None] = None,
    ) -> None:
        """
        Define key parameters for transformation. For keys needed to initialize
            and define these parameters, see ?self._initialize_config
    
        """

        dict_config = dict_config = (
            self.config 
            if dict_config is None
            else dict_config
        )

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
    


    def _initialize_sectoral_transformations(self,
        baseline_with_plur: bool = False,
        df_input: Union[pd.DataFrame, None] = None,
        dict_config: Union[Dict, None] = None,
        **kwargs,
    ) -> None:
        """
        Initialize other TransformerXXXX classes for use here.
            
        Sets the following properties:

            * self.transformations_afolu
            * self.transformations_circular_economy
            * self.transformations_energy
            * self.transformations_ippu

        Function Arguments
        ------------------
        - baseline_with_plur: set to True to let the baseline include partial 
            land use reallocation. 
        * NOTE: If True, then transformation_lndu_reallocate_land() 
            has no effect.
        * NOTE: this is set in both `self.transformation_af_baseline()` and
            `self.transformation_lndu_reallocate_land()` separately
        - dir_jl: location of Julia directory containing Julia environment and 
        support modules
        - fp_nemomod_reference_files: directory housing reference files called 
            by NemoMod when running electricity model. Required to access data 
            in EnergyProduction. Needs the following CSVs:

            * Required keys or CSVs (without extension):
                (1) CapacityFactor
                (2) SpecifiedDemandProfile

        Kewyord Arguments
        ------------------
        - dict_config: configuration dictionary passed to objects
        """

        dict_config = (
            self.config 
            if dict_config is None
            else dict_config
        )
        # initialize all transformations with df_input
        # then use those functions to set baseline_inputs

        ##  Finally -- initialize baseline using the data frame
        self._initialize_baseline_inputs(
            df_input,
        )

        return None

    


    def _initialize_transformations(self,
    ) -> None:
        """
        Initialize all trl.Transformer objects used to build transformations.

     
        Sets the following properties:

            * self.all_transformations
            * self.all_transformations_non_baseline
            * self.dict_transformations
            * self.transformation_id_baseline
            * self.transformation_***
        """

        attr_transformer_code = self.attribute_transformer_code
        all_transformations = []

        dict_transformations = {}




        ##################
        #    BASELINE    #
        ##################

        self.baseline = trl.Transformer(
            "BASE", 
            self.transformer_baseline, 
            attr_strategy
        )
        all_transformations.append(self.baseline)



        #################################
        #    CROSS-SECTOR PORTFOLIOS    #
        #################################


        ##  FOR ALL, CALL ALL FUNCTIONS FROM SUBSECTORS

        function_list = self.transformations_circular_economy.ce_all.function_list.copy()
        function_list += self.transformations_energy.en_all.function_list.copy()
        function_list += self.transformations_ippu.ip_all.function_list.copy()
        function_list += [
            self.transformation_pflo_industrial_ccs,
            self.transformation_pflo_healthier_diets
        ]

        # break out before adding AFOLU so that w & w/o reallocation can be sent to different transformations
        function_list_plur = function_list.copy()
        function_list_india_ccdr = function_list.copy()
        function_list_plur_no_deforestation_stoppage = function_list.copy()
        function_list += self.transformations_afolu.af_all.function_list.copy()

        self.pflo_all = trl.Transformer(
            "PFLO:ALL", 
            function_list, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all)


        ##  FOR PLUR, ENSURE PLUR IS ON
        
        function_list_plur += (
            self.transformations_afolu
            .af_all_with_partial_reallocation
            .function_list
            .copy()
        )

        self.pflo_all_with_partial_reallocation = trl.Transformer(
            "PFLO:ALL_PLUR", 
            function_list_plur, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_with_partial_reallocation)



        ##################################################
        #    TEMP: ADD INDIA SPECIFIC TRANSFORMATIONS    #
        ##################################################

        ##  START WITH INDIA PLUR WITH CC

    
        self.lndu_partial_reallocation_india_cc = trl.Transformer(
            "LNDU:PLUR_INDIA_CC", 
            [
                self.transformations_afolu.transformation_lndu_reallocate_land,
                self.transformations_afolu.transformation_agrc_decrease_climate_productivity_climate_india
            ],
            attr_strategy
        )
        all_transformations.append(self.lndu_partial_reallocation_india_cc)


        ##  BUILD INDIA CCDR

        function_list_india_ccdr += [
            #self.transformation_agrc_decrease_exports,
            self.transformations_afolu.transformation_agrc_expand_conservation_agriculture,
            self.transformations_afolu.transformation_agrc_improve_rice_management,
            self.transformations_afolu.transformation_agrc_increase_crop_productivity,
            self.transformations_afolu.transformation_agrc_reduce_supply_chain_losses,
            self.transformations_afolu.transformation_agrc_reduce_supply_chain_losses,
            # self.transformation_lndu_integrated_transitions replaces:
            #   self.transformation_lndu_expand_silvopasture,
            #   self.transformation_lndu_stop_deforestation
            #self.transformations_afolu.transformation_lndu_integrated_transitions,
            self.transformations_afolu.transformation_lndu_reallocate_land,
            self.transformations_afolu.transformation_lsmm_improve_manure_management_cattle_pigs,
            self.transformations_afolu.transformation_lsmm_improve_manure_management_other,
            self.transformations_afolu.transformation_lsmm_improve_manure_management_poultry,
            self.transformations_afolu.transformation_lsmm_increase_biogas_capture,
            #self.transformations_afolu.transformation_lvst_decrease_exports,
            #self.transformations_afolu.transformation_lvst_increase_productivity,
            self.transformations_afolu.transformation_lvst_reduce_enteric_fermentation,
            self.transformations_afolu.transformation_soil_reduce_excess_fertilizer,
            self.transformations_afolu.transformation_soil_reduce_excess_lime
        ]

        # drop healthier diets
        function_list_india_ccdr = [
            x for x in function_list_india_ccdr 
            if x != self.transformation_pflo_healthier_diets
        ]
        
        self.pflo_ccdr_india_with_partial_reallocation = trl.Transformer(
            "PFLO:INDIA_CCDR_PLUR", 
            function_list_india_ccdr, 
            attr_strategy
        )
        all_transformations.append(self.pflo_ccdr_india_with_partial_reallocation)


        ##  BUILD INDIA CCDR THAT INCLUDES CLIMATE (FLAG:INDIA)
        
        function_list_india_ccdr_cc = function_list_india_ccdr.copy()
        function_list_india_ccdr_cc.append(
            self.transformations_afolu.transformation_agrc_decrease_climate_productivity_climate_india
        )

        self.pflo_ccdr_india_with_partial_reallocation_india_cc = trl.Transformer(
            "PFLO:INDIA_CCDR_PLUR_INDIA_CC", 
            function_list_india_ccdr_cc, 
            attr_strategy
        )
        all_transformations.append(self.pflo_ccdr_india_with_partial_reallocation_india_cc)


        ##  BUILD AN INDIA PLUR THAT INCLUDES CLIMATE (FLAG:INDIA)
        
        function_list_plur_india = function_list_plur.copy()
        function_list_plur_india.append(
            self.transformations_afolu.transformation_agrc_decrease_climate_productivity_climate_india
        )

        self.pflo_all_with_partial_reallocation_india_cc = trl.Transformer(
            "PFLO:ALL_PLUR_INDIA_CC", 
            function_list_plur, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_with_partial_reallocation_india_cc)
        
        ###################
        #    END INDIA    #
        ###################




        ##  EXPLORE ALL W/O SILVOPASTURE (EXPLORATORY ONLY)

        function_list_plur_no_silvopasture = function_list_plur.copy()
        # drop the integrated transitions and add the stoppage of deforestation back in
        if self.transformations_afolu.transformation_lndu_integrated_transitions in function_list_plur_no_silvopasture:
            function_list_plur_no_silvopasture.remove(
                self.transformations_afolu.transformation_lndu_integrated_transitions
            )
            function_list_plur_no_silvopasture.append(
                self.transformations_afolu.transformation_lndu_stop_deforestation
            )
        

        self.pflo_all_with_partial_reallocation_no_silvopasture = trl.Transformer(
            "PFLO:ALL_PLUR_NO_SILVOPASTURE", 
            function_list_plur_no_silvopasture, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_with_partial_reallocation_no_silvopasture)


        ##  PROVIDE ALL W/O PREVENTING DEFORESTATION

        function_list_plur_no_deforestation_stoppage += (
            self.transformations_afolu
            .af_all_with_deforestation_and_partial_reallocation
            .function_list
            .copy()
        )

        self.pflo_all_with_deforestation_and_partial_reallocation = trl.Transformer(
            "PFLO:ALL_NO_STOPPING_DEFORESTATION_PLUR", 
            function_list_plur_no_deforestation_stoppage, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_with_deforestation_and_partial_reallocation)


        """
        ##  PROVIDE ONE W/O LVST EXPORTS

        # NOTE: would have to reinstantiate above if uncommenting
        function_list_plur_no_lvst_exp_reduction += (
            self.transformations_afolu
            .af_all_no_lvst_export_reduction_with_partial_reallocation
            .function_list
            .copy()
        )

        self.pflo_all_no_lvst_export_reduction_with_partial_reallocation = trl.Transformer(
            "PFLO:ALL_NO_LVST_EXPORT_REDUCTION_PLUR", 
            function_list_plur_no_lvst_exp_reduction, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_no_lvst_export_reduction_with_partial_reallocation)
        """;

        self.pflo_better_baseline = trl.Transformer(
            "PFLO:BETTER_BASE", 
            [
                self.transformations_afolu.transformation_agrc_improve_rice_management,
                self.transformations_afolu.transformation_agrc_increase_crop_productivity,
                self.transformations_afolu.transformation_agrc_reduce_supply_chain_losses,
                self.transformations_afolu.transformation_agrc_expand_conservation_agriculture,
                self.transformations_afolu.transformation_lndu_reallocate_land,
                self.transformations_afolu.transformation_lvst_increase_productivity,
                self.transformations_afolu.transformation_soil_reduce_excess_fertilizer,
                self.transformations_circular_economy.transformation_wali_improve_sanitation_industrial,
                self.transformations_circular_economy.transformation_wali_improve_sanitation_rural,
                self.transformations_circular_economy.transformation_wali_improve_sanitation_urban,
                self.transformations_circular_economy.transformation_waso_increase_landfilling,
                self.transformations_circular_economy.transformation_waso_increase_recycling,
                self.transformations_energy.transformation_entc_reduce_transmission_losses,
                self.transformations_energy.transformation_fgtv_maximize_flaring,
                self.transformations_energy.transformation_inen_maximize_efficiency_energy,
                self.transformations_energy.transformation_scoe_increase_applicance_efficiency,
                self.transformations_energy.transformation_scoe_reduce_heat_energy_demand,
                self.transformations_energy.transformation_trns_increase_efficiency_electric,
                self.transformations_energy.transformation_trns_increase_efficiency_non_electric,
                self.transformations_ippu.transformation_ippu_reduce_cement_clinker
            ], 
            attr_strategy
        )
        all_transformations.append(self.pflo_better_baseline)


        self.plfo_healthier_diets = trl.Transformer(
            "PFLO:BETTER_DIETS", 
            self.transformation_pflo_healthier_diets, 
            attr_strategy
        )
        all_transformations.append(self.plfo_healthier_diets)


        self.plfo_healthier_diets_with_partial_reallocation = trl.Transformer(
            "PFLO:BETTER_DIETS_PLUR", 
            [
                self.transformation_pflo_healthier_diets, 
                self.transformations_afolu.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.plfo_healthier_diets_with_partial_reallocation)


        self.pflo_industrial_ccs = trl.Transformer(
            "PFLO:IND_INC_CCS", 
            self.transformation_pflo_industrial_ccs, 
            attr_strategy
        )
        all_transformations.append(self.pflo_industrial_ccs)


        self.pflo_sociotechnical = trl.Transformer(
            "PFLO:CHANGE_CONSUMPTION",
            [
                self.transformation_pflo_healthier_diets,
                self.transformations_afolu.transformation_lndu_stop_deforestation,
                self.transformations_afolu.transformation_lndu_reallocate_land,
                self.transformations_afolu.transformation_lvst_decrease_exports,
                self.transformations_circular_economy.transformation_waso_decrease_food_waste,
                self.transformations_circular_economy.transformation_waso_increase_anaerobic_treatment_and_composting,
                self.transformations_energy.transformation_trns_electrify_road_light_duty,
                self.transformations_energy.transformation_trns_increase_occupancy_light_duty,
                self.transformations_energy.transformation_trns_mode_shift_public_private,
                self.transformations_energy.transformation_trns_mode_shift_regional
            ],
            attr_strategy
        )
        all_transformations.append(self.pflo_sociotechnical)


        self.pflo_supply_side_technology = trl.Transformer(
            "PFLO:SUPPLY_SIDE_TECH", 
            [
                self.transformation_pflo_industrial_ccs, 
                self.transformations_afolu.transformation_lndu_expand_silvopasture,
                self.transformations_afolu.transformation_lndu_reallocate_land,
                self.transformations_afolu.transformation_lsmm_improve_manure_management_cattle_pigs,
                self.transformations_afolu.transformation_lsmm_improve_manure_management_other,
                self.transformations_afolu.transformation_lsmm_improve_manure_management_poultry,
                self.transformations_afolu.transformation_lsmm_increase_biogas_capture,
                self.transformations_afolu.transformation_lvst_reduce_enteric_fermentation,
                self.transformations_circular_economy.transformation_trww_increase_biogas_capture,
                self.transformations_circular_economy.transformation_waso_increase_biogas_capture,
                self.transformations_circular_economy.transformation_waso_increase_energy_from_biogas,
                self.transformations_circular_economy.transformation_waso_increase_energy_from_incineration,
                self.transformations_energy.transformation_fgtv_minimize_leaks,
                self.transformations_energy.transformation_inen_fuel_switch_low_and_high_temp,
                self.transformations_energy.transformation_scoe_fuel_switch_electrify,
                self.transformations_energy.transformation_trns_electrify_rail,
                self.transformations_energy.transformation_trns_fuel_switch_maritime,
                self.transformations_energy.transformation_trns_fuel_switch_road_medium_duty,
                self.transformations_energy.transformation_trns_mode_shift_freight,
                self.transformations_energy.transformation_support_entc_clean_grid,
                self.transformations_ippu.transformation_ippu_reduce_demand,
                self.transformations_ippu.transformation_ippu_reduce_hfcs,
                self.transformations_ippu.transformation_ippu_reduce_n2o,
                self.transformations_ippu.transformation_ippu_reduce_other_fcs,
                self.transformations_ippu.transformation_ippu_reduce_pfcs
            ], 
            attr_strategy
        )
        all_transformations.append(self.pflo_supply_side_technology)

        

        ## specify dictionary of transformations and get all transformations + baseline/non-baseline

        dict_transformations.update(
            dict(
                (x.id, x) 
                for x in all_transformations
                if x.id in attr_strategy.key_values
            )
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
    


    def build_(self,
    ) -> None: 
        """
        Initialize the ramp vector for implementing transformations. Sets the 
            following properties:

            * self.dict_entc_renewable_target_cats_max_investment
            * self.vec_implementation_ramp
            * self.vec_implementation_ramp_renewable_cap
            * self.vec_msp_resolution_cap
        """
        
        vec_implementation_ramp = self.build_implementation_ramp_vector()
        vec_implementation_ramp_renewable_cap = self.get_vir_max_capacity(vec_implementation_ramp)
        vec_msp_resolution_cap = self.build_msp_cap_vector(vec_implementation_ramp)

        dict_entc_renewable_target_cats_max_investment = dict(
            (
                x, 
                {
                    "vec": vec_implementation_ramp_renewable_cap,
                    "type": "scalar"
                }
            ) for x in self.cats_entc_max_investment_ramp
        )
        

        ##  SET PROPERTIES
        self.dict_entc_renewable_target_cats_max_investment = dict_entc_renewable_target_cats_max_investment
        self.vec_implementation_ramp = vec_implementation_ramp
        self.vec_implementation_ramp_renewable_cap = vec_implementation_ramp_renewable_cap
        self.vec_msp_resolution_cap = vec_msp_resolution_cap

        return None
    


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

        

     def get(self,
        transformer: Union[int, str, None],
        field_transformer_id: str = "transformer_id",
        field_transformer_name: str = "transformer",
        return_code: bool = False,
    ) -> None:
        """
        Get `transformer` based on transformer code, id, or name
        
        If strat is None or an invalid valid of strat is entered, returns None; 
            otherwise, returns the trl.Transformer object. 

            
        Function Arguments
        ------------------
        - transformer: transformer_id, transformer name, or transformer code to 
            use to retrieve sc.Trasnformation object
            
        Keyword Arguments
        ------------------
        - field_transformer_code: field in transformer attribute table 
            containing the transformer code
        - field_transformer_name: field in transformer attribute table 
            containing the transformer name
        - return_code: set to True to return the transformer code only
        """

        # skip these types
        is_int = sf.isnumber(transformer, integer = True)
        return_none = not is_int
        return_none &= not isinstance(transformer, str)
        if return_none:
            return None

        # Transformer objects are tied to the attribute table, so these field maps work
        dict_id_to_code = self.attribute_transformer_code.field_maps.get(
            f"{field_transformer_id}_to_{self.attribute_transformer_code.key}"
        )
        dict_name_to_code = self.attribute_transformer_code.field_maps.get(
            f"{field_transformer_name}_to_{self.attribute_transformer_code.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(transformer, str):
            code = (
                transformer
                if transformer in attribute_transformer_code.key_values
                else dict_name_to_code.get(transformer)
            )
        
        elif is_int:
            code = dict_id_to_code.get(transformer)

        # check returns
        if code is None:
            return None

        if return_code:
            return code


        out = self.dict_transformations.get(code)
        
        return out
    


    def get_entc_cats_renewable(self,
        cats_renewable: Union[List[str], None] = None,
        key_renewable_default: str = "renewable_default",
    ) -> List[str]:

        # filter and return
        cats_renewable = self.model_attributes.get_valid_categories(
            cats_renewable,
            self.model_attributes.subsec_name_entc,
        )

        if not sf.islistlike(cats_renewable):
            cats_renewable = self.model_attributes.filter_keys_by_attribute(
                self.model_attributes.subsec_name_entc,
                {key_renewable_default: 1}
            )

        return cats_renewable
        
   

    def get_entc_dict_renewable_target_msp(self,
        cats_renewable: Union[List[str], None], 
        dict_entc_renewable_target_msp: dict,
    ) -> List[str]:
        """
        Set any targets for renewable energy categories. Relies on 
            cats_renewable to verify keys in renewable_target_entc
        
        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * dict_entc_renewable_target_msp: dictionary of renewable energy
                categories mapped to MSP targets under the renewable target
                transformation
        """
        dict_entc_renewable_target_msp = (
            {}
            if not isinstance(dict_entc_renewable_target_msp, dict)
            else dict(
                (k, v) for k, v in dict_entc_renewable_target_msp.items() 
                if (k in cats_renewable) and (sf.isnumber(v))
            )
        )

        return dict_entc_renewable_target_msp



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
        sf._optional_log(
            self.logger, 
            msg, 
            type_log = type_log, 
            **kwargs
        )

        return None





    ##################################################
    ###                                            ###
    ###    BEGIN DEFINING TRANSFORMER FUNCTIONS    ###
    ###                                            ###
    ##################################################

    ##########################################################
    #    BASELINE - TREATED AS TRANSFORMATION TO INPUT DF    #
    ##########################################################
    """
    NOTE: needed for certain modeling approaches; e.g., preventing new hydro 
        from being built. The baseline can be preserved as the input DataFrame 
        by the Transformer as a passthrough (e.g., return input DataFrame) 

    NOTE: modifications to input variables should ONLY affect IPPU variables
    """

    def transformer_baseline(self,
        df_input: pd.DataFrame,
        categories_entc_pps_to_cap: Union[List[str], None] = None,
        categories_entc_renewable: Union[List[str], None] = None,
        dict_entc_renewable_target_msp_baseline: dict = {},
        magnitude_lurf: Union[bool, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Baseline" from which other transformations deviate 
            (pass through)

        Function Arguments
        ------------------
        - df_input: input dataframe

        Keyword Arguments
        -----------------
        - categories_entc_pps_to_cap: ENTC categories to cap at current levels 
            when projecting minimum share of production (MSP) forward.
        - categories_entc_renewable: power plant energy technologies considered
            renewable. If nothing is specified, then defaults to ENTC attribute 
            table specification in field "renewable_default"
        - dict_entc_renewable_target_msp_baseline: optional dictionary to 
            specify minium share of production targets for renewables under the
            base case.
        - magnitude_lurf: magnitude of the land use reallocation factor under
            baseline
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
        """

        # clean production scalar so that they are always 1 in the first time period
        #df_out = tbg.prepare_demand_scalars(
        #    df_input,
        #    self.model_ippu.modvar_ippu_scalar_production,
        #    self.model_attributes,
        #    key_region = self.key_region,
        #)
        df_out = df_input.copy()

        ##  GET SOME PARAMETERS

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )
        
        # renewable categories
        categories_entc_renewable = self.get_entc_cats_renewable(
            categories_entc_renewable, 
        )

        # target magnitude of the land use reallocation factor
        magnitude_lurf = (  
            0.0 
            if not sf.isnumber(magnitude_lurf) 
            else max(min(magnitude_lurf, 1.0), 0.0)
        )
        
        # dictionary mapping to target minimum shares of production
        dict_entc_renewable_target_msp_baseline = self.get_entc_dict_renewable_target_msp(
            cats_renewable = categories_entc_renewable,
            dict_entc_renewable_target_msp = dict_entc_renewable_target_msp_baseline,
        )


        ##  AFOLU BASE

        # set land use reallocation factor
        df_out = tbg.transformation_general(
            df_out,
            self.model_attributes,
            {
                self.model_afolu.modvar_lndu_reallocation_factor: {
                    "bounds": (0.0, 1.0),
                    "magnitude": magnitude_lurf,
                    "magnitude_type": "final_value",
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat,
        )


        ##  CIRCULAR ECONOMY BASE

        # no actions


        ##  ENERGY BASE

        df_out = self.transformation_support_entc_change_msp_max(
            df_out,
            categories_entc_pps_to_cap,
            strat = None,
            vec_implementation_ramp = vec_implementation_ramp,
        )


        # NEW ADDITION (2023-09-27): ALLOW FOR BASELINE INCREASE IN RENEWABLE ADOPTION

        target_renewables_value_min = sum(dict_entc_renewable_target_msp_baseline.values())

        # apply transformation
        df_out = tbe.transformation_entc_renewable_target(
            df_out,
            target_renewables_value_min,
            vec_implementation_ramp,
            self.model_enerprod,
            dict_cats_entc_max_investment = self.dict_entc_renewable_target_cats_max_investment,
            field_region = self.key_region,
            include_target = False, # only want to adjust MSPs in line with this
            magnitude_as_floor = True,
            magnitude_renewables = dict_entc_renewable_target_msp_baseline,
            scale_non_renewables_to_match_surplus_msp = True,
            strategy_id = strat,
            **kwargs,
        )


        ##  IPPY BASE


        df_out = self.transformations_afolu.transformation_af_baseline(df_out)
        df_out = self.transformations_circular_economy.transformation_ce_baseline(df_out)
        df_out = self.transformations_energy.transformation_en_baseline(df_out)
        df_out = self.transformations_ippu.transformation_ip_baseline(df_out)

        # TEMP: for certain experiments, we want to treat PLUR as baseline. This implementation does that
        #df_out = self.transformations_afolu.transformation_lndu_reallocate_land(df_out)

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



    ########################################
    #    CROSS-SECTORAL TRANSFORMATIONS    #
    ########################################

    def transformation_pflo_healthier_diets(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Healthier Diets" transformation on input DataFrame
            df_input (affects IPPU and INEN).
        
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
                self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat: {
                    "bounds": (0, 1),
                    "magnitude": 0.5,
                    "magnitude_type": "final_value_ceiling",
                    "vec_ramp": vec_implementation_ramp
                },

                # TEMPORARY UNTIL A DEMAND SCALAR CAN BE ADDED IN
                self.model_afolu.modvar_agrc_elas_crop_demand_income: {
                    "bounds": (-2, 2),
                    "categories": ["sugar_cane"],
                    "magnitude": -0.2,
                    "magnitude_type": "final_value_ceiling",
                    "vec_ramp": vec_implementation_ramp
                },
            },
            field_region = self.key_region,
            strategy_id = strat,
        )

        return df_out



    def transformation_pflo_industrial_ccs(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Industrial Point of Capture" transformation on input 
            DataFrame df_input (affects IPPU and INEN).
        
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


        dict_magnitude_eff = None
        dict_magnitude_prev = {
            "cement": 0.8,
            "chemicals": 0.8,
            "metals": 0.8,
            "plastic": 0.8,
        }

        # increase prevalence of capture
        df_out = tbc.transformation_mlti_industrial_carbon_capture(
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


