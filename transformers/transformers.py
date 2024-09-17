
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
import sisepuede.transformers.lib._baselib_afolu as tba
import sisepuede.transformers.lib._baselib_circular_economy as tbc
import sisepuede.transformers.lib._baselib_cross_sector as tbs
import sisepuede.transformers.lib._baselib_energy as tbe
import sisepuede.transformers.lib._baselib_general as tbg
import sisepuede.transformers.lib._baselib_ippu as tbi
import sisepuede.transformers.lib._classes as trl
import sisepuede.utilities._toolbox as sf




#
#    SET SOME DEFAULT CONFIGURATION VALUES
#

def get_dict_config_default(
    key_baseline: str = "baseline",
    key_general: str = "general",
) -> dict:
    """
    Retrieve the dictionary of default configuration values for transformers, 
        including "general" and "baseline"
    """
    dict_out = {
        key_baseline: {
            "magnitude_lurf": 0.0, # default to not use Land Use Reallocation Factor
        },

        key_general: {
            #
            # ENTC categories that are capped to 0 investment--default to include 
            #"categories_entc_max_investment_ramp": [
            #    "pp_hydropower"
            #],
            #[
            #        "pp_geothermal",
            #        "pp_nuclear"
            #    ]

            # ENTC categories considered renewable sources--defaults to attribute table specs if not defined
            #"categories_entc_renewable": []

            # INEN categories that have high heat
            "categories_inen_high_heat": [
                "cement", 
                "chemicals", 
                "glass", 
                "lime_and_carbonite", 
                "metals"
            ],

            # Target minimum share of production fractions for power plants in the renewable target tranformation
            #"dict_entc_renewable_target_msp": {
            #    "pp_solar": 0.15,
            #    "pp_geothermal": 0.1,
            #    "pp_wind": 0.15
            #},

            # fraction of high heat that can be electrified and hydrogenized
            "frac_inen_high_temp_elec_hydg": 0.5*0.45,

            # fraction of low temperature heat energy demand that can be electrified
            "frac_inen_low_temp_elec": 0.95*0.45,

            # number of time periods in the ramp
            # "n_tp_ramp": None,

            # shape values for implementing caps on new technologies (description below)
            "vir_renewable_cap_delta_frac": 0.0075,
            "vir_renewable_cap_max_frac": 0.125,

            # first year to start transformations--default is to 2 years from present
            # "year_0_ramp": dt.datetime.now().year + 2
        }
    }

    out = sc.YAMLConfiguration(dict_out)

    return out





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
    - baseline_with_lurf: set to True to let the baseline include partial land
        use reallocation in the baseline--passed to TransformersAFOLU as
        a keyword argument.
        * NOTE: If True, then transformation_lndu_reallocate_land() 
            has no effect.
        * NOTE: this is set in both `self._trfunc_af_baseline()` and
            `self._trfunc_lndu_reallocate_land()` separately
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
    - **kwargs 
    """
    
    def __init__(self,
        dict_config: Dict,
        df_input: Union[pd.DataFrame, None] = None,
        field_region: Union[str, None] = None,
        logger: Union[logging.Logger, None] = None,
        regex_template_prepend: str = "sisepuede_run",
        **kwargs
    ):

        self.logger = logger

        self._initialize_file_structure(
            regex_template_prepend = regex_template_prepend, 
        )
        self._initialize_models()
        self._initialize_attributes(field_region, )

        self._initialize_config(dict_config = dict_config, )
        self._initialize_parameters()
        self._initialize_ramp()

        # set transformations by sector, models (which come from sectoral transformations)
        self._initialize_baseline_inputs(df_input, )
        self._initialize_transformers()
        
        return None




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

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
            raise RuntimeError(f"Error: invalid specification of model_attributes in Transformers")

        # get transformer attribute, technology attribute
        attribute_transformer_code = (
            self.model_attributes
            .get_other_attribute_table(
                self.model_attributes
                .dim_transformer_code
            )
        )

        # add technology attribute
        attribute_technology = (
            self.model_attributes
            .get_attribute_table(
                self.model_attributes
                .subsec_name_entc
            )
        )

        field_region = (
            self.model_attributes.dim_region 
            if (field_region is None) 
            else field_region
        )

        # set some useful classes
        time_periods = sc.TimePeriods(self.model_attributes)
        regions_manager = sc.Regions(self.model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_technology = attribute_technology
        self.attribute_transformer_code = attribute_transformer_code
        self.key_region = field_region
        self.key_transformer_code = attribute_transformer_code.key
        self.time_periods = time_periods
        self.regions_manager = regions_manager

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
            self._trfunc_baseline(
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
        key_baseline: str = "baseline",
        key_general: str = "general",
    ) -> None:
        """
        Define the configuration dictionary and paramter keys. Sets the 
            following properties:

            * self.config (configuration containing general and baseline)
            * self.key_* (keys)
            
        Function Arguments
        ------------------
        - dict_config: dictionary defining the configuration of (1) the baseline
            run and (2) general shared properties used across transforations. 

            * The "baseline" configuration dictionary can include the following 
                keys:
            
            * The "general" configuration dictionary can include the following 
                keys:

                * "categories_entc_max_investment_ramp": list of categories to 
                    apply self.vec_implementation_ramp_renewable_cap to with a 
                    maximum investment cap (implemented *after* turning on 
                    renewable target)
                * "categories_entc_pps_to_cap": list of power plant categories 
                    to prevent from new growth by capping MSP
                * "categories_entc_renewable": list of categories to tag as 
                    renewable for the Renewable Targets transformation (sets 
                    self.cats_renewable)
                * "n_tp_ramp": number of time periods to use to ramp up. If None 
                    or not specified, builds to full implementation by the final 
                    time period
                * "tp_0_ramp": last time period with no diversion from baseline 
                    strategy (baseline for implementation ramp)
                * "vir_renewable_cap_delta_frac": change (applied downward from 
                    "vir_renewable_cap_max_frac") in cap for for new technology
                    capacities available to build in time period while 
                    transitioning to renewable capacties. Default is 0.01 (will 
                    decline by 1% each time period after "tp_0_ramp")
                * "vir_renewable_cap_max_frac": cap for for new technology 
                    capacities available to build in time period while 
                    transitioning to renewable capacties; entered as a fraction 
                    of estimated capacity in "tp_0_ramp". Default is 0.05
            
        """
        # build config; start with default and overwrite as necessary
        config = get_dict_config_default(
            key_baseline = key_baseline,
            key_general = key_general,
        )

        if isinstance(dict_config, dict):
            config.dict_yaml.update(dict_config)
        

        ##  SET PARAMETERS

        self.config = config

        self.key_config_baseline = key_baseline
        self.key_config_cats_entc_max_investment_ramp = "categories_entc_max_investment_ramp"
        self.key_config_cats_entc_pps_to_cap = "categories_entc_pps_to_cap"
        self.key_config_cats_entc_renewable = "categories_entc_renewable"
        self.key_config_cats_inen_high_heat = "categories_inen_high_heat",
        self.key_config_frac_inen_high_temp_elec_hydg = "frac_inen_low_temp_elec"
        self.key_config_frac_inen_low_temp_elec = "frac_inen_low_temp_elec"
        self.key_config_general = key_general
        self.key_config_magnitude_lurf = "magnitude_lurf" # MUST be same as kwarg in _trfunc_baseline
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
            allow_electricity_run = True,
            fp_julia = self.file_struct.dir_jl,
            fp_nemomod_reference_files = self.file_struct.dir_ref_nemo,
            fp_nemomod_temp_sqlite_db = self.file_struct.fp_sqlite_tmp_nemomod_intermediate,
            initialize_julia = False,
            logger = self.logger,
        )

        # check AFOLU
        model_afolu = kwargs.get("model_afolu")
        if not is_sisepuede_model_afolu(model_afolu):
            model_afolu = models.model_afolu


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
    ) -> None:
        """
        Define key parameters for transformation. For keys needed to initialize
            and define these parameters, see ?self._initialize_config
    
        """

        # get parameters from configuration dictionary if specified
        (
            n_tp_ramp, 
            tp_0_ramp,  
            vir_renewable_cap_delta_frac,
            vir_renewable_cap_max_frac,
        ) = self.get_ramp_characteristics(
            n_tp_ramp = self.config.get(
                f"{self.key_config_general}.{self.key_config_n_tp_ramp}"
            ),
            tp_0_ramp = self.config.get(
                f"{self.key_config_general}.{self.key_config_tp_0_ramp}"
            ),
        )

        # check if baseline includes partial land use reallocation factor
        baseline_with_lurf = self.config.get(
            f"{self.key_config_baseline}.{self.key_config_magnitude_lurf}"
        ) > 0.0


        ##  SET PROPERTIES

        self.baseline_with_lurf = baseline_with_lurf
        self.n_tp_ramp = n_tp_ramp
        self.tp_0_ramp = tp_0_ramp
        self.vir_renewable_cap_delta_frac = vir_renewable_cap_delta_frac
        self.vir_renewable_cap_max_frac = vir_renewable_cap_max_frac

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



    def _initialize_transformers(self,
    ) -> None:
        """
        Initialize all trl.Transformer objects used to build transformations.

     
        Sets the following properties:

            * self.all_transformers
            * self.all_transformers_non_baseline
            * self.dict_transformers
            * self.transformer_id_baseline
            * self._trfunc_***
        """

        attr_transformer_code = self.attribute_transformer_code
        all_transformers = []

        dict_transformers = {}



        ##################
        #    BASELINE    #
        ##################

        self.baseline = trl.Transformer(
            "TFR:BASE", 
            self._trfunc_baseline, 
            attr_transformer_code
        )
        all_transformers.append(self.baseline)


        ###############
        #    AFOLU    #
        ###############

        ##  AGRC TRANSFORMERS

        self.agrc_improve_rice_management = trl.Transformer(
            "TFR:AGRC:DEC_CH4_RICE", 
            self._trfunc_agrc_improve_rice_management,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_improve_rice_management)


        self.agrc_decrease_exports = trl.Transformer(
            "TFR:AGRC:DEC_EXPORTS", 
            self._trfunc_agrc_decrease_exports,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_decrease_exports)


        self.agrc_expand_conservation_agriculture = trl.Transformer(
            "TFR:AGRC:INC_CONSERVATION_AGRICULTURE", 
            self._trfunc_agrc_expand_conservation_agriculture,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_expand_conservation_agriculture)


        self.agrc_increase_crop_productivity = trl.Transformer(
            "TFR:AGRC:INC_PRODUCTIVITY", 
            self._trfunc_agrc_increase_crop_productivity,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_increase_crop_productivity)


        self.agrc_reduce_supply_chain_losses = trl.Transformer(
            "TFR:AGRC:DEC_LOSSES_SUPPLY_CHAIN", 
            self._trfunc_agrc_reduce_supply_chain_losses,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_reduce_supply_chain_losses)


        ##  FRST TRANSFORMERS

        
        ##  LNDU TRANSFORMERS

        self.lndu_expand_silvopasture = trl.Transformer(
            "TFR:LNDU:INC_SILVOPASTURE", 
            self._trfunc_lndu_expand_silvopasture,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_expand_silvopasture)


        self.lndu_expand_sustainable_grazing = trl.Transformer(
            "TFR:LNDU:DEC_SOC_LOSS_PASTURES", 
            self._trfunc_lndu_expand_sustainable_grazing,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_expand_sustainable_grazing)


        self.lndu_increase_reforestation = trl.Transformer(
            "TFR:LNDU:INC_REFORESTATION", 
            self._trfunc_lndu_increase_reforestation,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_increase_reforestation)


        self.lndu_partial_reallocation = trl.Transformer(
            "TFR:LNDU:PLUR", 
            self._trfunc_lndu_reallocate_land,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_partial_reallocation)


        self.lndu_stop_deforestation = trl.Transformer(
            "TFR:LNDU:DEC_DEFORESTATION", 
            self._trfunc_lndu_stop_deforestation,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_stop_deforestation)


        ##  LSMM TRANSFORMATIONS

        self.lsmm_improve_manure_management_cattle_pigs = trl.Transformer(
            "TFR:LSMM:INC_MANAGEMENT_CATTLE_PIGS", 
            self._trfunc_lsmm_improve_manure_management_cattle_pigs,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_cattle_pigs)


        self.lsmm_improve_manure_management_other = trl.Transformer(
            "TFR:LSMM:INC_MANAGEMENT_OTHER", 
            self._trfunc_lsmm_improve_manure_management_other,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_other)
        

        self.lsmm_improve_manure_management_poultry = trl.Transformer(
            "TFR:LSMM:INC_MANAGEMENT_POULTRY", 
            self._trfunc_lsmm_improve_manure_management_poultry,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_poultry)


        self.lsmm_increase_biogas_capture = trl.Transformer(
            "TFR:LSMM:INC_CAPTURE_BIOGAS", 
            self._trfunc_lsmm_increase_biogas_capture,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_increase_biogas_capture)

        
        ##  LVST TRANSFORMERS
      
        self.lvst_decrease_exports = trl.Transformer(
            "TFR:LVST:DEC_EXPORTS", 
            self._trfunc_lvst_decrease_exports,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_decrease_exports)


        self.lvst_increase_productivity = trl.Transformer(
            "TFR:LVST:INC_PRODUCTIVITY", 
            self._trfunc_lvst_increase_productivity,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_increase_productivity)


        self.lvst_reduce_enteric_fermentation = trl.Transformer(
            "TFR:LVST:DEC_ENTERIC_FERMENTATION", 
            self._trfunc_lvst_reduce_enteric_fermentation,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_reduce_enteric_fermentation)
        

        ##  SOIL TRANSFORMERS
        
        self.soil_reduce_excess_fertilizer = trl.Transformer(
            "TFR:SOIL:DEC_N_APPLIED", 
            self._trfunc_soil_reduce_excess_fertilizer,
            attr_transformer_code
        )
        all_transformers.append(self.soil_reduce_excess_fertilizer)


        self.soil_reduce_excess_liming = trl.Transformer(
            "TFR:SOIL:DEC_LIME_APPLIED", 
            self._trfunc_soil_reduce_excess_lime,
            attr_transformer_code
        )
        all_transformers.append(self.soil_reduce_excess_liming)


        #######################################
        #    CIRCULAR ECONOMY TRANSFORMERS    #
        #######################################

        ##  TRWW TRANSFORMERS

        self.trww_increase_biogas_capture = trl.Transformer(
            "TFR:TRWW:INC_CAPTURE_BIOGAS", 
            self._trfunc_trww_increase_biogas_capture,
            attr_transformer_code
        )
        all_transformers.append(self.trww_increase_biogas_capture)


        self.trww_increase_septic_compliance = trl.Transformer(
            "TFR:TRWW:INC_COMPLIANCE_SEPTIC", 
            self._trfunc_trww_increase_septic_compliance,
            attr_transformer_code
        )
        all_transformers.append(self.trww_increase_septic_compliance)


        ##  WALI TRANSFORMERS
 
        self.wali_improve_sanitation_industrial = trl.Transformer(
            "TFR:WALI:INC_TREATMENT_INDUSTRIAL", 
            self._trfunc_wali_improve_sanitation_industrial,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_industrial)


        self.wali_improve_sanitation_rural = trl.Transformer(
            "TFR:WALI:INC_TREATMENT_RURAL", 
            self._trfunc_wali_improve_sanitation_rural,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_rural)


        self.wali_improve_sanitation_urban = trl.Transformer(
            "TFR:WALI:INC_TREATMENT_URBAN", 
            self._trfunc_wali_improve_sanitation_urban,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_urban)


        ##  WASO TRANSFORMERS

        self.waso_descrease_consumer_food_waste = trl.Transformer(
            "TFR:WASO:DEC_CONSUMER_FOOD_WASTE",
            self._trfunc_waso_decrease_food_waste, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_descrease_consumer_food_waste)

        
        self.waso_increase_anaerobic_treatment_and_composting = trl.Transformer(
            "TFR:WASO:INC_ANAEROBIC_AND_COMPOST", 
            self._trfunc_waso_increase_anaerobic_treatment_and_composting, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_anaerobic_treatment_and_composting)


        self.waso_increase_biogas_capture = trl.Transformer(
            "TFR:WASO:INC_CAPTURE_BIOGAS", 
            self._trfunc_waso_increase_biogas_capture, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_biogas_capture)


        self.waso_energy_from_biogas = trl.Transformer(
            "TFR:WASO:INC_ENERGY_FROM_BIOGAS", 
            self._trfunc_waso_increase_energy_from_biogas, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_energy_from_biogas)


        self.waso_energy_from_incineration = trl.Transformer(
            "TFR:WASO:INC_ENERGY_FROM_INCINERATION", 
            self._trfunc_waso_increase_energy_from_incineration, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_energy_from_incineration)


        self.waso_increase_landfilling = trl.Transformer(
            "TFR:WASO:INC_LANDFILLING", 
            self._trfunc_waso_increase_landfilling, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_landfilling)

        
        self.waso_increase_recycling = trl.Transformer(
            "TFR:WASO:INC_RECYCLING", 
            self._trfunc_waso_increase_recycling, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_recycling)


        #############################
        #    ENERGY TRANSFORMERS    #
        #############################

        ##  CCSQ

        self.ccsq_increase_air_capture = trl.Transformer(
            "TFR:CCSQ:INCREASE_CAPTURE", 
            self.transformation_ccsq_increase_air_capture, 
            attr_transformer_code
        )
        all_transformers.append(self.ccsq_increase_air_capture)


        ##  ENTC

        self.entc_clean_hydrogen = trl.Transformer(
            "TFR:ENTC:TARGET_CLEAN_HYDROGEN", 
            self._trfunc_entc_clean_hydrogen, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_clean_hydrogen)


        self.entc_least_cost = trl.Transformer(
            "TFR:ENTC:LEAST_COST", 
            self.transformation_entc_least_cost, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_least_cost)

        
        self.entc_reduce_transmission_losses = trl.Transformer(
            "TFR:ENTC:DEC_LOSSES", 
            self.transformation_entc_reduce_transmission_losses, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_reduce_transmission_losses)


        self.entc_renewable_electricity = trl.Transformer(
            "TFR:ENTC:TARGET_RENEWABLE_ELEC", 
            self.transformation_entc_renewables_target, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_renewable_electricity)


        ##  FGTV

        self.fgtv_maximize_flaring = trl.Transformer(
            "FGTV:INC_FLARE", 
            self.transformation_fgtv_maximize_flaring, 
            attr_transformer_code
        )
        all_transformers.append(self.fgtv_maximize_flaring)

        self.fgtv_minimize_leaks = trl.Transformer(
            "FGTV:DEC_LEAKS", 
            self.transformation_fgtv_minimize_leaks, 
            attr_transformer_code
        )
        all_transformers.append(self.fgtv_minimize_leaks)


        ##  INEN

        self.inen_fuel_switch_high_temp = trl.Transformer(
            "TFR:INEN:FUEL_SWITCH_HI_HEAT", 
            self.transformation_inen_fuel_switch_high_temp, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_fuel_switch_high_temp)


        self.inen_fuel_switch_low_temp_to_heat_pump = trl.Transformer(
            "TFR:INEN:FUEL_SWITCH_LO_HEAT", 
            self.transformation_inen_fuel_switch_low_temp_to_heat_pump, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_fuel_switch_low_temp_to_heat_pump)

        
        self.inen_maximize_energy_efficiency = trl.Transformer(
            "TFR:INEN:INC_EFFICIENCY_ENERGY", 
            self.transformation_inen_maximize_efficiency_energy, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_maximize_energy_efficiency)


        self.inen_maximize_production_efficiency = trl.Transformer(
            "TFR:INEN:INC_EFFICIENCY_PRODUCTION", 
            self.transformation_inen_maximize_efficiency_production, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_maximize_production_efficiency)


        ##  SCOE

        self.scoe_fuel_switch_electrify = trl.Transformer(
            "TFR:SCOE:FUEL_SWITCH_HEAT", 
            self.transformation_scoe_fuel_switch_electrify, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_fuel_switch_electrify)


        self.scoe_increase_applicance_efficiency = trl.Transformer(
            "TFR:SCOE:INC_EFFICIENCY_APPLIANCE", 
            self.transformation_scoe_increase_applicance_efficiency, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_increase_applicance_efficiency)


        self.scoe_reduce_heat_energy_demand = trl.Transformer(
            "TFR:SCOE:DEC_DEMAND_HEAT", 
            self.transformation_scoe_reduce_heat_energy_demand, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_reduce_heat_energy_demand)


        ###################
        #    TRNS/TRDE    #
        ###################

        self.trde_reduce_demand = trl.Transformer(
            "TFR:TRDE:DEC_DEMAND", 
            self.transformation_trde_reduce_demand, 
            attr_transformer_code
        )
        all_transformers.append(self.trde_reduce_demand)

        
        self.trns_electrify_light_duty_road = trl.Transformer(
            "TFR:TRNS:FUEL_SWITCH_LIGHT_DUTY", 
            self.transformation_trns_electrify_road_light_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_electrify_light_duty_road)

        
        self.trns_electrify_rail = trl.Transformer(
            "TFR:TRNS:FUEL_SWITCH_RAIL", 
            self.transformation_trns_electrify_rail, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_electrify_rail)

        
        self.trns_fuel_switch_maritime = trl.Transformer(
            "TFR:TRNS:FUEL_SWITCH_MARITIME", 
            self.transformation_trns_fuel_switch_maritime, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_fuel_switch_maritime)


        self.trns_fuel_switch_medium_duty_road = trl.Transformer(
            "TFR:TRNS:FUEL_SWITCH_MEDIUM_DUTY", 
            self.transformation_trns_fuel_switch_road_medium_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_fuel_switch_medium_duty_road)


        self.trns_increase_efficiency_electric = trl.Transformer(
            "TFR:TRNS:INC_EFFICIENCY_ELECTRIC", 
            self.transformation_trns_increase_efficiency_electric,
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_efficiency_electric)


        self.trns_increase_efficiency_non_electric = trl.Transformer(
            "TFR:TRNS:INC_EFFICIENCY_NON_ELECTRIC", 
            self.transformation_trns_increase_efficiency_non_electric,
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_efficiency_non_electric)


        self.trns_increase_occupancy_light_duty = trl.Transformer(
            "TFR:TRNS:INC_OCCUPANCY_LIGHT_DUTY", 
            self.transformation_trns_increase_occupancy_light_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_occupancy_light_duty)


        self.trns_mode_shift_freight = trl.Transformer(
            "TFR:TRNS:MODE_SHIFT_FREIGHT", 
            self.transformation_trns_mode_shift_freight, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_freight)


        self.trns_mode_shift_public_private = trl.Transformer(
            "TFR:TRNS:MODE_SHIFT_PASSENGER", 
            self.transformation_trns_mode_shift_public_private, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_public_private)


        self.trns_mode_shift_regional = trl.Transformer(
            "TFR:TRNS:MODE_SHIFT_REGIONAL", 
            self.transformation_trns_mode_shift_regional, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_regional)



        ######################################
        #    CROSS-SECTOR TRANSFORMATIONS    #
        ######################################


        self.plfo_healthier_diets = trl.Transformer(
            "TFR:PFLO:HEALTHIER_DIETS", 
            self._trfunc_pflo_healthier_diets, 
            attr_transformer_code
        )
        all_transformers.append(self.plfo_healthier_diets)



        self.pflo_industrial_ccs = trl.Transformer(
            "TFR:PFLO:INC_IND_CCS", 
            self._trfunc_pflo_industrial_ccs, 
            attr_transformer_code
        )
        all_transformers.append(self.pflo_industrial_ccs)


        ## specify dictionary of transformations and get all transformations + baseline/non-baseline

        dict_transformers.update(
            dict(
                (x.code, x) 
                for x in all_transformers
                if x.code in attr_transformer_code.key_values
            )
        )
        all_transformers = sorted(list(dict_transformers.keys()))
        all_transformers_non_baseline = [
            x for x in all_transformers 
            if not dict_transformers.get(x).baseline
        ]

        transformer_id_baseline = [
            x for x in all_transformers 
            if x not in all_transformers_non_baseline
        ]
        transformer_id_baseline = (
            transformer_id_baseline[0] 
            if (len(transformer_id_baseline) > 0) 
            else None
        )


        # SET ADDDITIONAL PROPERTIES

        self.all_transformers = all_transformers
        self.all_transformers_non_baseline = all_transformers_non_baseline
        self.dict_transformers = dict_transformers
        self.transformer_id_baseline = transformer_id_baseline

        return None
    


    ################################################
    ###                                          ###
    ###    OTHER NON-TRANSFORMATION FUNCTIONS    ###
    ###                                          ###
    ################################################

    def bounded_real_magnitude(self,
        magnitude: Union[float, int],
        default: Union[float, int],
        bounds: Tuple = (0.0, 1.0),
    ) -> float:
        """
        Shortcut function to clean up a common operation; bounds magnitude
            if specify as a float, otherwise reverts to default
        """
        out = (
            default
            if not isinstance(magnitude, float) 
            else max(min(magnitude, bounds[1]), bounds[0])
        )

        return out



    def build_implementation_ramp_vector(self,
        n_tp_ramp: Union[int, None] = None,
        tp_0_ramp: Union[int, None] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Build the implementation ramp vector

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - n_tp_ramp: number of years to go from 0 to 1
        - tp_0_ramp: last time period without change from baseline
        **kwargs: passed to sisepuede.utilities._toolbox.ramp_vector()
        """
        
        # some init
        #int_n_q = sf.isnumber(n_tp_ramp, integer = True)
        #int_t_q = sf.isnumber(tp_0_ramp, integer = True)
        n_tp = len(self.time_periods.all_time_periods)

        """
        ##  SET n_tp_ramp AND tp_0_ramp ON A CASE BASIS

        if not (int_n_q & int_t_q):
            # if both are not set, use defaults
            n_tp_ramp = self.n_tp_ramp 
            tp_0_ramp = self.tp_0_ramp 
        
        elif not int_n_q:
            # here, tp_0_ramp is specified, but we have to recalculate n_tp_ramp
            n_tp_ramp = n_tp - tp_0_ramp - 1
        
        elif not int_t_q:
            # here, the number of time periods is specifed; we check for tp_0_ramp
            n_tp_ramp = max(min(n_tp_ramp, n_tp - 1), 1)
            tp_0_ramp = n_tp - n_tp_ramp - 1
        """

        # verify the values
        n_tp_ramp, tp_0_ramp, _, _ = self.get_ramp_characteristics(
            n_tp_ramp = n_tp_ramp,
            tp_0_ramp = tp_0_ramp,
        )
        
        # get some shape parameters
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 2)
        c = kwargs.get("c", 1)
        d = kwargs.get("d") # default is None
 
        vec_out = sf.ramp_vector(
            n_tp, 
            a = a, 
            b = b, 
            c = c, 
            d = d,
            r_0 = tp_0_ramp,
            r_1 = tp_0_ramp + n_tp_ramp,
        )

        return vec_out
    


    def build_msp_cap_vector(self,
        vec_ramp: np.ndarray,
    ) -> np.ndarray:
        """
        Build the cap vector for MSP adjustments. Derived from 
            vec_implementation_ramp

        Function Arguments
		------------------
        - vec_ramp: implementation ramp vector to use. Will set cap at first 
            non-zero period

        Keyword Arguments
		-----------------
        """
        vec_out = np.array([
            (self.model_enerprod.drop_flag_tech_capacities if (x == 0) else 0) 
            for x in vec_ramp
        ])

        return vec_out
    


    def check_implementation_ramp(self,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None],
        df_input: Union[pd.DataFrame, None] = None,
    ) -> Union[np.ndarray, None]:
        """
        Check that vector `vec_implementation_ramp` ramp is the same length as 
            `df_input` and that it meets specifications for an implementation
            vector. If `df_input` is not specified, use `self.baseline_inputs`. 
        
        If anything fails, return `self.vec_implementation_ramp`.
        """

        # pull input dataframe
        df_input = (
            self.baseline_inputs 
            if not isinstance(df_input, pd.DataFrame)
            else df_input
        )

        # if dictionary, try to build as vector
        if isinstance(vec_implementation_ramp, dict):
            
            n_tp_ramp = vec_implementation_ramp.get("n_tp_ramp")
            tp_0_ramp = vec_implementation_ramp.get("tp_0_ramp")

            try:
                vec_implementation_ramp = self.build_implementation_ramp_vector(
                    **vec_implementation_ramp
                    #n_tp_ramp = n_tp_ramp,
                    #tp_0_ramp = tp_0_ramp,
                )

            except Exception as e:
                None
                

        out = tbg.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
            self,
        )

        return out
    


    def get_entc_cats_max_investment_ramp(self,
        cats_entc_max_investment_ramp: Union[List[str], None] = None,
    ) -> List[str]:
        """
        Set categories to which a cap on maximum investment is applied in the 
            renewables target shift.  If dict_config is None, uses self.config.
        
        Keyword Arguments
        -----------------
        - cats_entc_max_investment_ramp: list of categories to apply a maximum
            investment capacity to
        """

        cats_entc_max_investment_ramp = (
            self.model_attributes.get_valid_categories(
                cats_entc_max_investment_ramp,
                self.model_attributes.subsec_name_entc,
            )
            if sf.islistlike(cats_entc_max_investment_ramp)
            else self.config.get(
                f"{self.key_config_general}.{self.key_config_cats_entc_max_investment_ramp}",
                return_on_none = [],
            )
        )
        
        return cats_entc_max_investment_ramp
    


    def get_entc_cats_renewable(self,
        cats_renewable: Union[List[str], None] = None,
        key_renewable_default: str = "renewable_default",
    ) -> List[str]:

        # 1. filter if specified as a list
        if sf.islistlike(cats_renewable):
            cats_renewable = self.model_attributes.get_valid_categories(
                cats_renewable,
                self.model_attributes.subsec_name_entc,
            )
            cats_renewable = (
                None 
                if len(cats_renewable) == 0 
                else cats_renewable
            )

        # 2. if the input isn't listlike, try config
        if not sf.islistlike(cats_renewable):
            cats_renewable = self.config.get(
                f"{self.key_config_general}.{self.key_config_cats_entc_renewable}"
            )
            
            # if a list is returned, filter categories
            if cats_renewable is not None:
                cats_renewable = self.model_attributes.get_valid_categories(
                    cats_renewable,
                    self.model_attributes.subsec_name_entc,
                )

        # 3. finally, if not specified in config, shift to attribute default
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
                vir_renewable_cap_delta_frac,
                vir_renewable_cap_max_frac,
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires those keys in dict_config to set. If not found, will set
            the following defaults:
                * tp_0_ramp: current year (computational run time) + 2
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


        ##  GET PARAMETERS USED TO MODIFY MSPs IN CONJUNCTION WITH vec_implementation_ramp

        # get VIR (get_vir_max_capacity) delta_frac
        # default_vir_renewable_cap_delta_frac = 0.01
        vir_renewable_cap_delta_frac = self.config.get(
            f"{self.key_config_general}.{self.key_config_vir_renewable_cap_delta_frac}",
        )
        vir_renewable_cap_delta_frac = float(sf.vec_bounds(vir_renewable_cap_delta_frac, (0.0, 1.0)))

        # get VIR (get_vir_max_capacity) max_frac
        # default_vir_renewable_cap_max_frac = 0.05
        vir_renewable_cap_max_frac = self.config.get(
            f"{self.key_config_general}.{self.key_config_vir_renewable_cap_max_frac}",
        )
        vir_renewable_cap_max_frac = float(sf.vec_bounds(vir_renewable_cap_max_frac, (0.0, 1.0)))
       
        tup_out = (
            n_tp_ramp,
            tp_0_ramp,
            vir_renewable_cap_delta_frac,
            vir_renewable_cap_max_frac, 
        )

        return tup_out

        

    def get_transformer(self,
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
                if transformer in self.attribute_transformer_code.key_values
                else dict_name_to_code.get(transformer)
            )
        
        elif is_int:
            code = dict_id_to_code.get(transformer)

        # check returns
        if code is None:
            return None

        if return_code:
            return code


        out = self.dict_transformers.get(code)
        
        return out
    


    def get_vectors_for_ramp_and_cap(self,
        categories_entc_max_investment_ramp: Union[List[str], None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> Tuple: 
        """
        Get ramp vector and associated vectors for capping, including (in order)

            * dict_entc_renewable_target_cats_max_investment
            * vec_implementation_ramp
            * vec_implementation_ramp_renewable_cap
            * vec_msp_resolution_cap
        
        Keyword Arguments
        -----------------
        - categories_entc_max_investment_ramp: categories to cap investments in
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
        """

        # get the implementation ramp
        if vec_implementation_ramp is None:
            vec_implementation_ramp = self.build_implementation_ramp_vector(**kwargs)

        # build renwewable cap for MSP
        vec_implementation_ramp_renewable_cap = self.get_vir_max_capacity(vec_implementation_ramp)
        vec_msp_resolution_cap = self.build_msp_cap_vector(vec_implementation_ramp)

        # get max investment ramp categories 
        cats_entc_max_investment_ramp = self.get_entc_cats_max_investment_ramp(
            cats_entc_max_investment_ramp = categories_entc_max_investment_ramp,
        )

        dict_entc_renewable_target_cats_max_investment = dict(
            (
                x, 
                {
                    "vec": vec_implementation_ramp_renewable_cap,
                    "type": "scalar"
                }
            ) for x in cats_entc_max_investment_ramp
        )
        
        
        ##  RETURN AS TUPLE

        tuple_out = (
            dict_entc_renewable_target_cats_max_investment,
            vec_implementation_ramp,
            vec_implementation_ramp_renewable_cap,
            vec_msp_resolution_cap,
        )

        return tuple_out
    
    

    def get_vir_max_capacity(self,
        vec_implementation_ramp: np.ndarray,
        delta_frac: Union[float, None] = None,
        dict_values_to_inds: Union[Dict, None] = None,
        max_frac: Union[float, None] = None,
    ) -> np.ndarray:
        """
        Buil a new value for the max_capacity based on vec_implementation_ramp.
            Starts with max_frac of a technicology's maximum residual capacity
            in the first period when vec_implementation_ramp != 0, then declines
            by delta_frac the specified number of time periods. Ramp down a cap 
            based on the renewable energy target.

        Function Arguments
        ------------------
        - vec_implementation_ramp: vector of lever implementation ramp to use as
            reference

        Keyword Arguments
        -----------------
        - delta_frac: delta to apply at each time period after the first time
            non-0 vec_implementation_ramp time_period. Defaults to 
            self.vir_renewable_cap_delta_frac if unspecified
        - dict_values_to_inds: optional dictionary mapping a value to row 
            indicies to pass the value to. Can be used, for example, to provide 
            a cap on new investments in early time periods. 
         - max_frac: fraction of maximum residual capacity to use as cap in 
            first time period where vec_implementation_ramp > 0. Defaults to
            self.vir_renewable_cap_max_frac if unspecified
        """

        delta_frac = (
            self.vir_renewable_cap_delta_frac
            if not sf.isnumber(delta_frac)
            else float(sf.vec_bounds(delta_frac, (0.0, 1.0)))
        )
        max_frac = (
            self.vir_renewable_cap_max_frac
            if not sf.isnumber(max_frac)
            else float(sf.vec_bounds(max_frac, (0.0, 1.0)))
        )

        vec_implementation_ramp_max_capacity = np.ones(len(vec_implementation_ramp))
        i0 = None

        for i in range(len(vec_implementation_ramp)):
            if vec_implementation_ramp[i] == 0:
                vec_implementation_ramp_max_capacity[i] = -999
            else:
                i0 = i if (i0 is None) else i0
                vec_implementation_ramp_max_capacity[i] = max(max_frac - delta_frac*(i - i0), 0.0)


        if isinstance(dict_values_to_inds, dict):
            for k in dict_values_to_inds.keys():
                np.put(vec_implementation_ramp_max_capacity, dict_values_to_inds.get(k), k)

        return vec_implementation_ramp_max_capacity 



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
    


    def _trfunc_support_entc_change_msp_max(self,
        df_input: Union[pd.DataFrame, None],
        cats_to_cap: Union[List[str], None],
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Implement a transformation for the baseline to resolve constraint
            conflicts between TotalTechnologyAnnualActivityUpperLimit/
            TotalTechnologyAnnualActivityLowerLimit if MinShareProduction is 
            Specified. 

        This transformation will turn on the MSP Max method in EnergyProduction,
            which will cap electric production (for a given technology) at the 
            value estimated for the last non-engaged time period. 
            
        E.g., suppose a technology has the following estimated electricity 
            production (estimated endogenously and excluding demands for ENTC) 
            and associated value of msp_max (stored in the "Maximum Production 
            Increase Fraction to Satisfy MinShareProduction Electricity" 
            SISEPUEDE model variable):

            time_period     est. production     msp_max
                            implied by MSP     
            -----------     ---------------     -------
            0               10                  -999
            1               10.5                -999
            2               11                  -999
            3               11.5                -999
            4               12                  0
            .
            .
            .
            n - 2           23                  0
            n - 1           23.1                0

            Then the MSP for this technology would be adjusted to never exceed 
            the value of 11.5, which was found at time_period 3. msp_max = 0
            means that a 0% increase is allowable in the MSP passed to NemoMod,
            so the specified MSP trajectory (which is passed to NemoMod) is 
            adjusted to reflect this change.
        
        NOTE: Only the *first value* after that last non-specified time period
            affects this variable. Using the above table as an example, entering 
            0 in time_period 4 and 1 in time_period 5 means that 0 is used for 
            all time_periods on and after 4.
        

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - cats_to_cap: list of categories to cap using the transformation
            implementation vector self.vec_implementation_ramp. If None, 
            defaults to pp_hydropower
        - strat: strategy number to pass
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
        - **kwargs: passed to ade.transformations_general()
        """
       
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


        # CHECK CATEGORIES TO CAP

        cats_to_cap = [] if not sf.islistlike(cats_to_cap) else cats_to_cap
        cats_to_cap = [x for x in self.attribute_technology.key_values if x in cats_to_cap]
        if len(cats_to_cap) == 0:
            return df_input

        # build dictionary if valid
        dict_cat_to_vector = dict(
            (x, self.vec_msp_resolution_cap)
            for x in cats_to_cap
        )


        # USE CHANGE MSP MAX TRANSFORMATION FUNCTION

        df_out = tbe.transformation_entc_change_msp_max(
            df_input,
            dict_cat_to_vector,
            self.model_enerprod,
            drop_flag = self.model_enerprod.drop_flag_tech_capacities,
            field_region = self.key_region,
            vec_ramp = vec_implementation_ramp,
            strategy_id = strat,
            **kwargs
        )
 
        return df_out





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

    def _trfunc_baseline(self,
        df_input: pd.DataFrame,
        categories_entc_max_investment_ramp: Union[List[str], None] = None,
        categories_entc_pps_to_cap: Union[List[str], None] = None,
        categories_entc_renewable: Union[List[str], None] = None,
        dict_entc_renewable_target_msp_baseline: dict = {},
        magnitude_lurf: Union[bool, None] = None,
        n_tp_ramp: Union[int, None] = None,
        tp_0_ramp: Union[int, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Baseline" from which other transformations deviate 
            (pass through)

        Function Arguments
        ------------------
        - df_input: input dataframe

        Keyword Arguments
        -----------------
        - categories_entc_max_investment_ramp: categories to cap investments in
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
        - n_tp_ramp: number of time periods to increase to full implementation. 
            If None, defaults to final time period
        - tp_0_ramp: first time period of ramp (last == 0)
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

        # power plant categories to cap
        categories_entc_pps_to_cap = self.config.get(
            f"{self.key_config_general}.{self.key_config_cats_entc_pps_to_cap}",
        )
        
        # target magnitude of the land use reallocation factor
        magnitude_lurf = (  
            self.config.get(
                f"{self.key_config_baseline}.{self.key_config_magnitude_lurf}"
            )
            if not sf.isnumber(magnitude_lurf) 
            else max(min(magnitude_lurf, 1.0), 0.0)
        )

        
        # dictionary mapping to target minimum shares of production
        dict_entc_renewable_target_msp_baseline = self.get_entc_dict_renewable_target_msp(
            cats_renewable = categories_entc_renewable,
            dict_entc_renewable_target_msp = dict_entc_renewable_target_msp_baseline,
        )

        # characteristics for BASELINE MSP ramp 
        (
            dict_entc_renewable_target_cats_max_investment,
            vec_implementation_ramp,
            vec_implementation_ramp_renewable_cap,
            vec_msp_resolution_cap,
        ) = self.get_vectors_for_ramp_and_cap(
            categories_entc_max_investment_ramp = categories_entc_max_investment_ramp,
            vec_implementation_ramp = vec_implementation_ramp,
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


        ##  ENERGY BASE

        df_out = self._trfunc_support_entc_change_msp_max(
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
            dict_cats_entc_max_investment = dict_entc_renewable_target_cats_max_investment,
            field_region = self.key_region,
            include_target = False, # only want to adjust MSPs in line with this
            magnitude_as_floor = True,
            magnitude_renewables = dict_entc_renewable_target_msp_baseline,
            scale_non_renewables_to_match_surplus_msp = True,
            strategy_id = strat,
            #**kwargs,
        )


        ##  IPPU BASE


        ##  ADD STRAT IF APPLICABLE 

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




    #########################################
    ###                                   ###
    ###    AFOLU TRANSFORMER FUNCTIONS    ###
    ###                                   ###
    #########################################
    
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



    def _trfunc_agrc_improve_crop_residue_management(self,
        df_input: Union[pd.DataFrame, None] = None,
        magnitude_burned: float = 0.0,
        magnitude_removed: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, Dict[str, int], None] = None,
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
        """
        Implement the "Improve Rice Management" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: minimum target fraction of rice production under improved 
            management.
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
        """
        Implement the "Increase Crop Productivity" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of productivity increase; can be specified as
            * a float: apply a single scalar increase to productivity for all
                crops
            * a dictionary: specify crop productivity increases individually,
                with each key being a crop and the associated value being the
                productivity increase
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
        """
        Implement the "Reduce Supply Chain Losses" AGRC transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of reduction in supply chain losses. Specified
            as a fraction (e.g., a 30% reduction is specified as 0.3)
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
        """
        Implement the "Increase Reforestation" FRST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - cats_inflow_restriction: categories to allow to transition into 
            secondary forest; don't specify categories that cannot be reforested 
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional increase in secondary forest area to specify.
            E.g., a 10% increase in secondary forests is specified as 0.1
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
        """
        Implement the "Stop Deforestation" FRST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of final primary forest transition probability
            into itself; higher magnitudes indicate less deforestation.
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
        - magnitude: magnitude of increase in fraction of pastures subject to 
            silvopasture
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
        - magnitude: fraction of pastures subject to improved pasture 
            management. This value acts as a floor, so that if the existing
            value is greater than is specified by the transformation, the 
            existing value will be maintained. 
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
        - magnitude_deforestation: magnitude to apply to deforestation 
            (transition probability from primary forest into self). If None, 
            uses default.
        - magnitude_silvopasture: magnitude passed to silvopasture 
            transformation. If None, uses silvopasture magnitude default. 
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
        """
        Support land use reallocation in specification of multiple 
            transformations.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - force: If the baseline includes LURF > 0, then this transformation 
            will not work; set force = True to force the transformation to 
            further modify the LURF
        - magnitude: land use reallocation factor value with implementation
            ramp vector
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        """
        Implement the "Improve Livestock Manure Management for Cattle and Pigs" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_lsmm_pathways: dictionary allocating treatment to LSMM categories
            as fractional targets (must sum to <= 1). If None, defaults to

            dict_lsmm_pathways = {
                "anaerobic_digester": 0.59375, # 0.625*0.95,
                "composting": 0.11875, # 0.125*0.95,
                "daily_spread": 0.2375, # 0.25*0.95,
            }

        - strat: optional strategy value to specify for the transformation
        - vec_cats_lvst: LVST categories receiving treatment in this 
            transformation. Default (if None) is

            [
                "cattle_dairy",
                "cattle_nondairy",
                "pigs"
            ]
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
        """
        Implement the "Improve Livestock Manure Management for Other Animals" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_lsmm_pathways: dictionary allocating treatment to LSMM categories
            as fractional targets (must sum to <= 1). If None, defaults to

            dict_lsmm_pathways = {
                "anaerobic_digester": 0.475, # 0.5*0.95,
                "composting": 0.2375, # 0.25*0.95,
                "dry_lot": 0.11875, # 0.125*0.95,
                "daily_spread": 0.11875, # 0.125*0.95,
            }

        - strat: optional strategy value to specify for the transformation
        - vec_cats_lvst: LVST categories receiving treatment in this 
            transformation. Default (if None) is

            [
                "buffalo",
                "goats",
                "horses",
                "mules",
                "sheep",
            ]

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
        """
        Implement the "Improve Livestock Manure Management for Poultry" 
            transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_lsmm_pathways: dictionary allocating treatment to LSMM categories
            as fractional targets (must sum to <= 1). If None, defaults to

            dict_lsmm_pathways = {
                "anaerobic_digester": 0.475, # 0.5*0.95,
                "poultry_manure": 0.475, # 0.5*0.95,
            }

        - strat: optional strategy value to specify for the transformation
        - vec_cats_lvst: LVST categories receiving treatment in this 
            transformation. Default (if None) is

            [
                "chickens",
            ]

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
        """
        Implement the "Increase Biogas Capture at Anaerobic Decomposition 
            Facilities" transformation on the input DataFrame.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: target minimum fraction of biogas that is captured at
            anerobic decomposition facilities.
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
        """
        Implement the "Decrease Exports" LVST transformation on input 
            DataFrame df_input (reduce by 50%)
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in exports applied directly to time
            periods (reaches 100% implementation when ramp reaches 1)
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
        """
        Implement the "Increase Livestock Productivity" LVST transformation on 
            input DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional increase in productivity applied to carrying
            capcity for livestock
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
        """
        Implement the "Reduce Enteric Fermentation" LVST transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
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
        """
        Implement the "Reduce Excess Fertilizer" SOIL transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in fertilier N to achieve in
            accordane with vec_implementation_ramp
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
        """
        Implement the "Reduce Excess Liming" SOIL transformation on input 
            DataFrame df_input. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in lime application to achieve in
            accordane with vec_implementation_ramp
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
        """
        Implement the "Increase Biogas Capture at Wastewater Treatment Plants" 
            TRWW transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final magnitude of biogas capture at TRWW facilties.
            NOTE: If specified as a float, the same value applies to both 
                landfill and biogas. Specify as a dictionary to specifiy 
                different capture fractions by TRWW technology, e.g., 
                
                magnitude = {
                    "treated_advanced_anaerobic": 0.85, 
                    "treated_secondary_anaerobic": 0.5
                }
                
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
        """
        Implement the "Increase Compliance" TRWW transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final magnitude of compliance at septic tanks that are
            installed
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
        """
        Implement the "Improve Industrial Sanitation" WALI transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_magnitude: target allocation, across TRWW (Wastewater Treatment) 
            categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% 
            from scondary aerobic by the final time period, the following 
            dictionary would be specified:

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
        """
        Implement the "Improve Rural Sanitation" WALI transformation on 
            input DataFrame df_input

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_magnitude: target allocation, across TRWW (Wastewater Treatment) 
            categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% 
            from scondary aerobic by the final time period, the following 
            dictionary would be specified:

            dict_magnitude = {
                "treated_advanced_anaerobic": 0.8,
                "treated_secondary_anaerobic": 0.1
            }

            If None, defaults to:

            {
                "treated_septic": 1.0,
            }
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
        """
        Implement the "Improve Urban Sanitation" WALI transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - dict_magnitude: target allocation, across TRWW (Wastewater Treatment) 
            categories (categories are keys), of treatment as total fraction. 
            * E.g., to acheive 80% of treatment from advanced anaerobic and 10% 
            from scondary aerobic by the final time period, the following 
            dictionary would be specified:

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
        """
        Implement the "Decrease Municipal Solid Waste" WASO transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: reduction in food waste sent to munipal solid waste
            treatment stream
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
        """
        Implement the "Increase Anaerobic Treatment and Composting" WASO 
            transformation on input DataFrame df_input. 

        Note that 0 <= magnitude_biogas + magnitude_compost should be <= 1; 
            if they exceed 1, they will be scaled proportionally to sum to 1
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude_biogas: proportion of organic solid waste that is treated 
            using anaerobic treatment
        - magnitude_compost: proportion of organic solid waste that is treated 
            using compost
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
        """
        Implement the "Increase Biogas Capture at Anaerobic Treatment Facilities
            and Landfills" WASO transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final magnitude of biogas capture at landfill and anaerobic
            digestion facilties.
            NOTE: If specified as a float, the same value applies to both 
                landfill and biogas. Specify as a dictionary to specifiy 
                different capture fractions by WASO technology, e.g., 
                
                magnitude = {
                    "landfill": 0.85, 
                    "biogas": 0.5
                }

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
        """
        Implement the "Energy from Captured Biogas" WASO transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final magnitude of energy use from captured biogas. 
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
        """
        Implement the "Energy from Solid Waste Incineration" WASO transformation 
            on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final magnitude of waste that is incinerated that is
            recovered for energy use
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
        """
        Implement the "Increase Landfilling" WASO transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fraction of non-recycled solid waste (including composting 
            and anaerobic digestion) sent to landfills
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
        """
        Implement the "Increase Recycling" WASO transformation on input 
            DataFrame df_input

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of recylables that are recycled  
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
        """
        Implement the "Increase Direct Air Capture" CCSQ transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: final total, in MT, of direct air capture capacity 
            installed at 100% implementation
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
        """
        Implement "green hydrogen" transformation requirements by forcing at 
            least 95% of hydrogen production to come from electrolysis.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories_source: hydrogen-producing technology categories that are 
            reduced in response to increases in green hydrogen. 
            * If None, defaults to 
                [
                    "fp_hydrogen_gasification", 
                    "fp_hydrogen_reformation",
                    "fp_hydrogen_reformation_ccs"
                ]

        - categories_source: hydrogen-producing technology categories that are 
            considered green; they will produce `magnitude` of hydrogen by 100% 
            implementation. 
            * If None, defaults to 
                [
                    "fp_hydrogen_electrolysis"
                ]

        - df_input: data frame containing trajectories to modify
        - magnitude: target fraction of hydrogen from clean (categories_source)
            sources. In general, this is electrolysis
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

        magnitude = self.bounded_real_magnitude(magnitude, 0.95, )


        df_strat_cur = tbe.transformation_entc_clean_hydrogen(
            df_input,
            magnitude,
            vec_implementation_ramp,
            self.model_attributes,
            self.model_enerprod,
            cats_to_apply = cats_target,
            cats_response = cats_source,
            field_region = self.key_region,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_entc_least_cost(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Least Cost" ENTC transformation on input DataFrame
            df_input
        
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
        magnitude_type: str = "final_value_ceiling",
        min_loss: Union[float, None] = 0.02,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Transmission Losses" ENTC transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of transmission loss in final time; behavior 
            depends on `magnitude_type`
        - magnitude_type: 
            * scalar (if `magnitude_type == "basline_scalar"`)
            * final value (if `magnitude_type == "final_value"`)
            * final value ceiling (if `magnitude_type == "final_value_ceiling"`)
        - min_loss: minimum feasible transmission loss in the system
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
        dict_entc_renewable_target_msp: dict = {},
        magnitude: float = 0.95,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "renewables target" transformation (shared repeatability),
            which sets a Minimum Share of Production from renewable energy 
            as a target.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories_entc_max_investment_ramp: categories to cap investments in
        - categories_entc_renewable: power plant technologies to consider as
            renewable energy sources
        - df_input: data frame containing trajectories to modify
        - dict_entc_renewable_target_msp: optional dictionary mapping renewable 
            ENTC categories to MSP fractions. Can be used to ensure some minimum 
            contribution of certain renewables--e.g.,

                        {
                            "pp_hydropower": 0.1,
                            "pp_solar": 0.15
                        }

            will ensure that hydropower is at least 10% of the mix and solar is 
            at least 15%. 
        - magnitude: minimum target fraction of electricity produced from
            renewable sources by 100% implementation
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
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Flaring" FGTV transformation on input DataFrame
            df_input
        
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

        
        df_strat_cur = tbe.transformation_fgtv_maximize_flaring(
            df_input,
            0.8, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_fgtv_minimize_leaks(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Minimize Leaks" FGTV transformation on input DataFrame
            df_input
        
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

        
        df_strat_cur = tbe.transformation_fgtv_reduce_leaks(
            df_input,
            0.8, 
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

    def _trfunc_inen_fuel_switch_high_temp(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch medium and high-temp thermal processes to 
            hydrogen and electricity" INEN transformation on input DataFrame 
            df_input
        
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

        
        df_strat_cur = tbe.transformation_inen_shift_modvars(
            df_input,
            2*self.frac_inen_high_temp_elec_hydg,
            vec_implementation_ramp,
            self.model_attributes,
            categories = self.cats_inen_high_heat,
            dict_modvar_specs = {
                self.model_enercons.modvar_inen_frac_en_electricity: 0.5,
                self.model_enercons.modvar_inen_frac_en_hydrogen: 0.5,
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_inen_fuel_switch_low_and_high_temp(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch low-temp thermal processes to industrial heat 
            pumps" and "Fuel switch medium and high-temp thermal processes to 
            hydrogen and electricity" INEN transformations on input DataFrame 
            df_input (note: these must be combined in a new function instead of
            as a composition due to the electricity shift in high-heat 
            categories)
        
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

        
        # set up fractions 
        frac_shift_hh_elec = self.frac_inen_low_temp_elec + self.frac_inen_high_temp_elec_hydg
        frac_shift_hh_elec /= self.frac_inen_shift_denom

        frac_shift_hh_hydrogen = self.frac_inen_high_temp_elec_hydg
        frac_shift_hh_hydrogen /= self.frac_inen_shift_denom


        # HIGH HEAT CATS ONLY
        # Fuel switch high-temp thermal processes + Fuel switch low-temp thermal processes to industrial heat pumps
        df_out = tbe.transformation_inen_shift_modvars(
            df_input,
            self.frac_inen_shift_denom,
            vec_implementation_ramp, 
            self.model_attributes,
            categories = self.cats_inen_high_heat,
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
            self.frac_inen_shift_denom,
            vec_implementation_ramp, 
            self.model_attributes,
            categories = self.cats_inen_not_high_heat,
            dict_modvar_specs = {
                self.model_enercons.modvar_inen_frac_en_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_out



    def _trfunc_inen_fuel_switch_low_temp_to_heat_pump(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch low-temp thermal processes to industrial heat 
            pumps" INEN transformation on input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_inen_shift_modvars(
            df_input,
            self.frac_inen_low_temp_elec,
            vec_implementation_ramp,
            self.model_attributes,
            dict_modvar_specs = {
                self.model_enercons.modvar_inen_frac_en_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_inen_maximize_efficiency_energy(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Industrial Energy Efficiency" INEN 
            transformation on input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_inen_maximize_energy_efficiency(
            df_input,
            0.3, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_inen_maximize_efficiency_production(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Industrial Production Efficiency" INEN 
            transformation on input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_inen_maximize_production_efficiency(
            df_input,
            0.4, 
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
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Switch to electricity for heat using heat pumps, electric 
            stoves, etc." INEN transformation on input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_scoe_electrify_category_to_target(
            df_input,
            0.95,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_scoe_reduce_heat_energy_demand(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce end-use demand for heat energy by improving 
            building shell" SCOE transformation on input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_scoe_reduce_demand_for_heat_energy(
            df_input,
            0.5,
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )

        return df_strat_cur



    def _trfunc_scoe_increase_applicance_efficiency(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase appliance efficiency" SCOE transformation on 
            input DataFrame df_input
        
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

        
        df_strat_cur = tbe.transformation_scoe_reduce_demand_for_appliance_energy(
            df_input,
            0.5,
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
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Demand" TRDE transformation on input DataFrame
            df_input
        
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


        df_out = tbe.transformation_trde_reduce_demand(
            df_input,
            0.25, 
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
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Light-Duty" TRNS transformation on input 
            DataFrame df_input
        
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


        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_light"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def _trfunc_trns_electrify_rail(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Rail" TRNS transformation on input DataFrame
            df_input
        
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
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            0.25,
            vec_implementation_ramp,
            self.model_attributes,
            categories = ["rail_freight", "rail_passenger"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def _trfunc_trns_fuel_switch_maritime(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Swich Maritime" TRNS transformation on input 
            DataFrame df_input
        
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
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # transfer 70% of diesel + gasoline to hydrogen
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            vec_implementation_ramp,
            self.model_attributes,
            categories = ["water_borne"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_enercons.modvar_trns_fuel_fraction_diesel,
                self.model_enercons.modvar_trns_fuel_fraction_gasoline
            ],
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
            categories = ["water_borne"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_enercons.modvar_trns_fuel_fraction_diesel,
                self.model_enercons.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    def _trfunc_trns_fuel_switch_road_medium_duty(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Switch Medium Duty" TRNS transformation on input 
            DataFrame df_input
        
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
        
        model_enercons = self.model_enercons

        # check implementation ramp
        vec_implementation_ramp = self.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
        )


        # transfer 70% of diesel + gasoline to electricity
        df_out = tbe.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_heavy_freight", "road_heavy_regional", "public"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_enercons.modvar_trns_fuel_fraction_diesel,
                self.model_enercons.modvar_trns_fuel_fraction_gasoline
            ],
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
            categories = ["road_heavy_freight", "road_heavy_regional", "public"],
            dict_modvar_specs = {
                self.model_enercons.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_enercons.modvar_trns_fuel_fraction_diesel,
                self.model_enercons.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
    
        return df_out
    

    
    def _trfunc_trns_increase_efficiency_electric(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        
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

        
        df_out = tbe.transformation_trns_increase_energy_efficiency_electric(
            df_input,
            0.25, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_increase_efficiency_non_electric(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Non-Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        
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

        
        df_out = tbe.transformation_trns_increase_energy_efficiency_non_electric(
            df_input,
            0.25, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_increase_occupancy_light_duty(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Vehicle Occupancy" TRNS transformation on input 
            DataFrame df_input
        
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


        df_out = tbe.transformation_trns_increase_vehicle_occupancy(
            df_input,
            0.25, 
            vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_enercons = self.model_enercons,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_mode_shift_freight(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Freight" TRNS transformation on input 
            DataFrame df_input
        
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

        
        df_out = tbe.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_freight: {
                    "bounds": (0, 1),
                    "magnitude": 0.2,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["aviation", "road_heavy_freight"],
                    "categories_target": {
                        "rail_freight": 1.0
                    },
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
        )
        
        return df_out



    def _trfunc_trns_mode_shift_public_private(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Passenger Vehicles to Others" TRNS 
            transformation on input DataFrame df_input
        
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


        df_out = tbe.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_public_private: {
                    "bounds": (0, 1),
                    "magnitude": 0.3,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["road_light"],
                    "categories_target": {
                        "human_powered": (1/6),
                        "powered_bikes": (2/6),
                        "public": 0.5
                    },
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
        )
        
        return df_out
    
    

    def _trfunc_trns_mode_shift_regional(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Regional Travel" TRNS transformation on input 
            DataFrame df_input
        
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
        

        df_out = tbe.transformation_general(
            df_input,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_regional: {
                    "bounds": (0, 1),
                    "magnitude": 0.1,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["aviation"],
                    "categories_target": {
                        "road_heavy_regional": 1.0
                    },
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
        )

        df_out = tbe.transformation_general(
            df_out,
            self.model_attributes,
            {
                self.model_enercons.modvar_trns_modeshare_regional: {
                    "bounds": (0, 1),
                    "magnitude": 0.2,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["road_light"],
                    "categories_target": {
                        "road_heavy_regional": 1.0
                    },
                    "vec_ramp": vec_implementation_ramp
                }
            },
            field_region = self.key_region,
            strategy_id = strat
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
        """
        Implement the "Healthier Diets" transformation on input DataFrame
            df_input (affects IPPU and INEN).
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude_red_meat: final period maximum fraction of per capita red 
            meat consumption relative to baseline (e.g., 0.5 means that people
            eat 50% as much red meat as they would have without the 
            intervention)
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

