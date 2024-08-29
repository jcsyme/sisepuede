
import logging
import numpy as np
import os, os.path
import pandas as pd
import shutil
import time
from typing import *


from sisepuede.core.attribute_table import AttributeTable
import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.data_management.ingestion as ing
import sisepuede.transformers.afolu as dta
import sisepuede.transformers.circular_economy as dtc
import sisepuede.transformers.energy as dte
import sisepuede.transformers.ippu as dti
import sisepuede.transformers.lib._baselib_cross_sector as tbc
import sisepuede.transformers.lib._baselib_general as tbg
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.models.ippu as mi
import sisepuede.utilities.support_functions as sf




class TransformationsIntegrated:
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
        EnergyProduction. Needs the following CSVs:

        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
    - baseline_with_plur: set to True to let the baseline include partial land
        use reallocation in the baseline--passed to TransformationsAFOLU as
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
    - model_electricity: optional EnergyProduction object to pass for property and 
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
        self._initialize_base_input_database(
            regions = regions,
            use_demo_template_on_missing = use_demo_template_on_missing,
        )

        self._initialize_config(dict_config = dict_config)
        self._initialize_parameters()
        self._initialize_ramp()

        # set transformations by sector, models (which come from sectoral transformations)
        self._initialize_sectoral_transformations(
            baseline_with_plur = baseline_with_plur,
            df_input = df_input,
            **kwargs,
        )
        self._initialize_models()
        self._initialize_transformations()
        self._initialize_templates()
        
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
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.key_region
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (self.model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in TransformationsIPPU")

        # get strategy attribute, baseline strategy, and some fields
        attribute_strategy = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_strategy_id
        )

        baseline_strategy = int(
            attribute_strategy.table[
                attribute_strategy.table["baseline_strategy_id"] == 1
            ][attribute_strategy.key].iloc[0]
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
        
        self.attribute_strategy = attribute_strategy
        self.baseline_strategy = baseline_strategy
        self.key_region = field_region
        self.key_strategy = attribute_strategy.key
        self.time_periods = time_periods
        self.regions_manager = regions

        return None



    def _initialize_base_input_database(self,
        regions: Union[List[str], None] = None,
        use_demo_template_on_missing: bool = True,
    ) -> None:
        """
        Initialize the BaseInputDatabase class used to construct future
            trajectories. Initializes the following properties:

            * self.base_input_database
            * self.base_input_database_demo
            * self.baseline_strategy
            * self.regions


        Keyword Arguments
        ------------------
        - regions: list of regions to run experiment for
            * If None, will attempt to initialize all regions defined in
                ModelAttributes
        - use_demo_template_on_missing: tries to instantiate a blank template if
            a template for a target region is missing. 
        """

        self._log("Initializing BaseInputDatabase", type_log = "info")

        dir_templates = self.file_struct.dict_data_mode_to_template_directory.get("calibrated")
        dir_templates_demo = self.file_struct.dict_data_mode_to_template_directory.get("demo")
        
        # trying building for demo
        try:
            base_input_database_demo = ing.BaseInputDatabase(
                dir_templates_demo,
                self.model_attributes,
                regions[0],
                demo_q = True,
                logger = self.logger,
            )

        except Exception as e:
            msg = f"Error initializing BaseInputDatabase for demo -- {e}"
            self._log(msg, type_log = "error")
            raise RuntimeError(msg)


        # try building base input database for all
        # first, try to copy templates (if necessary and desired) fom demo to new region
        self._try_template_init_from_demo(
            base_input_database_demo,
            regions,
            fp_templates_target = dir_templates,
            try_instantiating_templates = use_demo_template_on_missing,
        )


        try:
            base_input_database = ing.BaseInputDatabase(
                dir_templates,
                self.model_attributes,
                regions,
                create_export_dir = True,
                demo_q = False,
                logger = self.logger
            )

        except Exception as e:
            
            # first, try building 

            msg = f"Error initializing BaseInputDatabase -- {e}"
            self._log(msg, type_log = "error")
            raise RuntimeError(msg)


        


        self.base_input_database = base_input_database
        self.base_input_database_demo = base_input_database_demo
        self.baseline_strategy = base_input_database.baseline_strategy
        self.regions = base_input_database.regions

        return None



    def _initialize_baseline_inputs(self,
        df_inputs: Union[pd.DataFrame, None],
    ) -> None:
        """
        Initialize the baseline inputs dataframe based on the initialization 
            value of df_inputs. It not initialied, sets as None. Sets the 
            following properties:

            * self.baseline_inputs

        NOTE: Additionally, to preserve consistency when calling transformations
            from the TransformationsIntegrated object, this overwrites baseline
            inputs for all component transformations, i.e., also resets

            * self.transformations_afolu.baseline_inputs
            * self.transformations_circular_economy.baseline_inputs
            * self.transformations_energy.baseline_inputs
            * self.transformations_ippu.baseline_inputs
        """

        baseline_inputs = (
            self.transformation_pflo_baseline(
                df_inputs, 
                strat = self.baseline_strategy,
            ) 
            if isinstance(df_inputs, pd.DataFrame) 
            else None
        )

        
        ##  SET PROPERTIES
        
        self.baseline_inputs = baseline_inputs
        self.transformations_afolu.baseline_inputs = baseline_inputs
        self.transformations_circular_economy.baseline_inputs = baseline_inputs
        self.transformations_energy.baseline_inputs = baseline_inputs
        self.transformations_ippu.baseline_inputs = baseline_inputs

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
    ) -> None:
        """
        Define model objects for use in variable access and base estimates. Sets
            the following properties:

            * self.model_afolu
            * self.model_circular_economy
            * self.model_electricity
            * self.model_energy
            * self.model_ippu
            * self.model_socioeconomic
        """

        self.model_afolu = self.transformations_afolu.model_afolu
        self.model_circular_economy = self.transformations_circular_economy.model_circecon
        self.model_electricity = self.transformations_energy.model_electricity
        self.model_energy = self.transformations_energy.model_energy
        self.model_ippu = self.transformations_ippu.model_ippu
        self.model_socioeconomic = self.model_ippu.model_socioeconomic

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
        Initialize other TransformationXXXX classes for use here.
            
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

        self.transformations_afolu = dta.TransformationsAFOLU(
            self.model_attributes,
            dict_config,
            baseline_with_plur = baseline_with_plur,
            df_input = df_input,
            field_region = self.key_region,
            logger = self.logger,
            model_afolu = kwargs.get("model_afolu"),
        )

        self.transformations_circular_economy = dtc.TransformationsCircularEconomy(
            self.model_attributes,
            dict_config,
            df_input = df_input,
            field_region = self.key_region,
            logger = self.logger,
            model_circecon = kwargs.get("model_circecon"),
        )

        self.transformations_energy = dte.TransformationsEnergy(
            self.model_attributes,
            dict_config,
            self.file_struct.dir_jl,
            self.file_struct.dir_ref_nemo,
            df_input = df_input,
            field_region = self.key_region,
            logger = self.logger,
            model_afolu = kwargs.get("model_afolu"),
            model_electricity = kwargs.get("model_electricity"),
        )

        self.transformations_ippu = dti.TransformationsIPPU(
            self.model_attributes,
            dict_config,
            df_input = df_input,
            field_region = self.key_region,
            logger = self.logger,
            model_ippu = kwargs.get("model_ippu"),
        )


        ##  Finally -- initialize baseline using the data frame
        self._initialize_baseline_inputs(
            df_input,
        )

        return None

    

    def _initialize_templates(self,
    ) -> None:
        """
        Initialize sectoral templates. Sets the following properties:

            * self.dict_sectoral_templates
            * self.input_template
        """

        # initialize some components
        input_template = ing.InputTemplate(
            None,
            self.model_attributes
        )
        attr_sector = self.model_attributes.get_sector_attribute_table()
        dict_sectoral_templates = {}

        for sector in self.model_attributes.all_sectors:
            
            # get baseline "demo" template, use for ranges
            fp_read = self.base_input_database_demo.get_template_path(
                self.regions[0], 
                sector
            )
            df_template = pd.read_excel(
                fp_read, 
                sheet_name = input_template.name_sheet_from_index(input_template.baseline_strategy)
            )

            # extract key fields (do not need time periods)
            fields_ext = [x for x in input_template.list_fields_required_base]
            fields_ext += [
                x for x in df_template.columns 
                if input_template.regex_template_max.match(str(x)) is not None
            ]
            fields_ext += [
                x for x in df_template.columns 
                if input_template.regex_template_min.match(str(x)) is not None
            ]
            df_template = df_template[fields_ext].drop_duplicates()

            dict_sectoral_templates.update({sector: df_template})

        self.dict_sectoral_templates = dict_sectoral_templates
        self.input_template = input_template

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
        dict_transformations.update(self.transformations_afolu.dict_transformations)
        dict_transformations.update(self.transformations_circular_economy.dict_transformations)
        dict_transformations.update(self.transformations_energy.dict_transformations)
        dict_transformations.update(self.transformations_ippu.dict_transformations)



        ##################
        #    BASELINE    #
        ##################

        self.baseline = sc.Transformation(
            "BASE", 
            self.transformation_pflo_baseline, 
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

        self.pflo_all = sc.Transformation(
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

        self.pflo_all_with_partial_reallocation = sc.Transformation(
            "PFLO:ALL_PLUR", 
            function_list_plur, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_with_partial_reallocation)



        ##################################################
        #    TEMP: ADD INDIA SPECIFIC TRANSFORMATIONS    #
        ##################################################

        ##  START WITH INDIA PLUR WITH CC

    
        self.lndu_partial_reallocation_india_cc = sc.Transformation(
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
        
        self.pflo_ccdr_india_with_partial_reallocation = sc.Transformation(
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

        self.pflo_ccdr_india_with_partial_reallocation_india_cc = sc.Transformation(
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

        self.pflo_all_with_partial_reallocation_india_cc = sc.Transformation(
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
        

        self.pflo_all_with_partial_reallocation_no_silvopasture = sc.Transformation(
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

        self.pflo_all_with_deforestation_and_partial_reallocation = sc.Transformation(
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

        self.pflo_all_no_lvst_export_reduction_with_partial_reallocation = sc.Transformation(
            "PFLO:ALL_NO_LVST_EXPORT_REDUCTION_PLUR", 
            function_list_plur_no_lvst_exp_reduction, 
            attr_strategy
        )
        all_transformations.append(self.pflo_all_no_lvst_export_reduction_with_partial_reallocation)
        """;

        self.pflo_better_baseline = sc.Transformation(
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


        self.plfo_healthier_diets = sc.Transformation(
            "PFLO:BETTER_DIETS", 
            self.transformation_pflo_healthier_diets, 
            attr_strategy
        )
        all_transformations.append(self.plfo_healthier_diets)


        self.plfo_healthier_diets_with_partial_reallocation = sc.Transformation(
            "PFLO:BETTER_DIETS_PLUR", 
            [
                self.transformation_pflo_healthier_diets, 
                self.transformations_afolu.transformation_lndu_reallocate_land
            ],
            attr_strategy
        )
        all_transformations.append(self.plfo_healthier_diets_with_partial_reallocation)


        self.pflo_industrial_ccs = sc.Transformation(
            "PFLO:IND_INC_CCS", 
            self.transformation_pflo_industrial_ccs, 
            attr_strategy
        )
        all_transformations.append(self.pflo_industrial_ccs)


        self.pflo_sociotechnical = sc.Transformation(
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


        self.pflo_supply_side_technology = sc.Transformation(
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
    


    def _try_template_init_from_demo(self,
        base_input_database_demo: Union[ing.BaseInputDatabase, None],
        regions: List[str],
        fp_templates_target: Union[str, None] = None,
        try_instantiating_templates: bool = True,
    ) -> None:
        """
        Try initializing templates for each sector using the demo templates. 

        Function Arguments
        ------------------
        - base_input_database_demo: Base input database for demo. If None, tries
            to call self.base_input_database_demo
        - regions: list of regions to init for. If None, tries self.regions


        Keyword Arguments
        -----------------
        - fp_templates_target: optional path to templates to try. Need to 
            specify for regions if base_input_database_demo is being used to 
            build paths for region templates
        - try_instantiating_templates: base functionality; if True, tries
            initializing new templates from demo
        """
        
        ##  SOME INITIALIZATION

        try:
            base_input_database_demo = (
                self.base_input_database_demo
                if base_input_database_demo is None
                else base_input_database_demo
            )

            regions = self.regions if (regions is None) else regions

        except Exception as e:
            return None

        if not try_instantiating_templates:
            return None


        ##  ITERATE OVER REGIONS TO BUILD TEMPLATES IF NEEDED

        for region in regions:
            for sector in self.model_attributes.all_sectors:

                # try to build regional path
                fp_region = base_input_database_demo.get_template_path(
                    region,
                    sector,
                    demo_q = False,
                    fp_templates = fp_templates_target, 
                )

                if os.path.exists(fp_region):
                    continue

                # otherwise, get demo path and copy over
                fp_demo = base_input_database_demo.get_template_path(
                    None,
                    sector,
                )
                
                shutil.copyfile(
                    fp_demo,
                    fp_region
                )
        
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
            f"TransformationsIPPU.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        # initialize baseline
        df_out = (
            self.transformation_ip_baseline(df_input)
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
            f"TransformationsIPPU.build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out
    


    def build_strategies_to_templates(self,
        df_base_trajectories: Union[pd.DataFrame, None] = None,
        df_exogenous_strategies: Union[pd.DataFrame, None] = None,
        regions: Union[List[str], None] = None,
        replace_template: bool = False,
        return_q: bool = False,
        sectors: Union[List[str], str] = None,
        strategies: Union[List[str], List[int], None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_base_trajectories: baseline (untransformed) data frame to use to 
            build strategies. Must contain self.key_region and 
            self.model_attributes.dim_time_period in columns. If None, defaults
            to self.baseline_inputs
        - df_exogenous_strategies: optional exogenous strategies to pass. Must 
            contain self.key_region and self.model_attributes.dim_time_period in 
            columns. If None, no action is taken. 
        - regions: optional list of regions to build strategies for. If None, 
            defaults to all defined.
        - replace_template: replace template if it exists? If False, tries to 
            overwrite existing sheets.
        - return_q: return an output dictionary? If True, will return all 
            templates in the form of a dictionary. Otherwise, writes to output 
            path implied by SISEPUEDEFileStructure
        - sectors: optional sectors to specify for export. If None, will export 
            all.
        - strategies: strategies to build for. Can be a mixture of strategy_ids
            and names. If None, runs all available. 
        - **kwargs: passed to self.input_template.template_from_inputs()
        """

        # INITIALIZE STRATEGIES TO LOOP OVER
        
        # initialize attributes and other basic variables
        attr_sector = self.model_attributes.get_sector_attribute_table()
        attr_strat = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_strategy_id
        )

        fields_var_all = self.model_attributes.build_variable_dataframe_by_sector(
            None, 
            include_time_periods = False
        )
        fields_var_all = list(fields_var_all["variable"])
        
        
        # initialize baseline dataframe
        df_out = (
            self.transformation_pflo_baseline(df_base_trajectories)
            if df_base_trajectories is not None
            else self.baseline_inputs
        )
        return_none = df_out is None
        
        
        # get input component and add baseline strategy marker
        fields_sort = (
            [
                attr_strat.key, 
                self.time_periods.field_time_period
            ] 
            if (attr_strat.key in df_out.columns) 
            else [field_time_period]
        )
        
        
        # check regions
        regions = (
            [x for x in self.regions if x in regions]
            if sf.islistlike(regions)
            else self.regions
        )
        n_r = len(regions)
        return_none |= (len(regions) == 0)
        
        
        # check sectors
        sectors = (
            [x for x in sectors if x in self.model_attributes.all_sectors]
            if sf.islistlike(sectors)
            else self.model_attributes.all_sectors
        )
        return_none |= (len(sectors) == 0)
        
        
        # check strategies
        strategies = (
            self.all_transformations_non_baseline
            if strategies is None
            else [x for x in self.all_transformations_non_baseline if x in strategies]
        )
        strategies = [self.get_strategy(x) for x in strategies]
        strategies = sorted([x.id for x in strategies if x is not None])
        n = len(strategies)
        
        if return_none:
            return None
        
        
        # LOOP TO BUILD

        t0 = time.time()
        self._log(
            f"Starting build of {n} strategies in {n_r} regions...",
            type_log = "info"
        )
        
        
        # initialize to overwrite dataframes
        dict_sectors = dict((s, {}) for s in attr_sector.key_values)
        dict_write = dict((r, dict_sectors) for r in regions)
        df_out_grouped = df_out.groupby([self.key_region])

        # try getting exogenous strategies grouped by region (as dict) -- if fails, will return None
        dict_exogenous_grouped = self.check_exogenous_strategies(df_exogenous_strategies)
    

        # iterate over regions
        for r, df in df_out_grouped:
            
            r = r[0] if isinstance(r, tuple) else r
            if r not in regions:
                continue
            
            self._log(f"Starting build for region {r}", type_log = "info")
            
            df_out_list = [df, df.copy()]
            dict_cur = (
                dict_write.get(r)
                if return_q
                else dict_sectors.copy()
            )
                

            ##  ITERATE OVER FUNCTIONAL TRANSFORMATIONS

            for i, strat in enumerate(strategies):
                t0_cur = time.time()
                transformation = self.get_strategy(strat)

                if transformation is None:
                    self._log(
                        f"\tTransformation {self.key_strategy} not found: check that a support_classes.Transformation object has been defined associated with the code.",
                        type_log = "warning"
                    )
                    continue


                # build strategies (baseline and alternative)
                try:
                    df_out_list[1] = transformation(df)

                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built transformation {self.key_strategy} = {transformation.id} ('{transformation.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )
                    skip_q = False
                    
                except Exception as e: 
                    df_out[i + 1] = None
                    self._log(
                        f"\tError trying to build transformation {self.key_strategy} = {transformation.id}: {e}",
                        type_log = "error"
                    )
                    continue
                
                # turn into input to template builder 
                df_cur = pd.concat(df_out_list, axis = 0).reset_index(drop = True)
                fields_drop = [x for x in df_cur.columns if x not in fields_var_all + fields_sort]
                df_cur = (
                    df_cur
                    .drop(fields_drop, axis = 1)
                    .sort_values(by = fields_sort)
                    .reset_index(drop = True)
                )

                global dc
                dc = df_cur.copy()
        
                # split the current transformation into 
                dict_cur = self.build_templates_dictionary_from_current_transformation(
                    df_cur,
                    attr_sector,
                    dict_cur,
                    **kwargs
                )
                

            ##  ADD ANY ADDITIONAL, EXOGENOUSLY DEFINED STRATEGIES?

            if isinstance(dict_exogenous_grouped, dict):
                
                self._log(
                    f"Starting integration of exogenous strategies in region '{r}",
                    type_log = "info"
                )

                df_cur_grouped = dict_exogenous_grouped.get(r)
                if df_cur_grouped is not None:
                    # ignore strategies that are defined functionally
                    df_cur_grouped = (
                        df_cur_grouped[
                            ~df_cur_grouped[self.key_strategy].isin(strategies)
                        ]
                        .groupby([self.key_strategy])
                    )

                    for strat, df_exog in df_cur_grouped:
                        
                        strat = strat[0] if isinstance(strat, tuple) else strat

                        df_cur = None
                        # try to concatenate current strategy
                        try:
                            df_out_list[1] = df_exog
                            df_cur = pd.concat(df_out_list, axis = 0).reset_index(drop = True)
                            self._log(
                                f"\tSuccessfully integrated exogenous strategy {self.key_strategy} = {strat}.",
                                type_log = "info"
                            )
                            
                        except Exception as e: 
                            df_out[i + 1] = None

                            self._log(
                                f"\tError trying to integrate exogenous strategy {self.key_strategy} = {strat}: {e}",
                                type_log = "error"
                            )

                            continue
                    
                        # drop unnecessary fields and prepare for templatization
                        fields_drop = [x for x in df_cur.columns if x not in fields_var_all + fields_sort]
                        df_cur = (
                            df_cur
                            .drop(fields_drop, axis = 1)
                            .sort_values(by = fields_sort)
                            .reset_index(drop = True)
                        )

                        # split the current exogenous transformation up
                        dict_cur = self.build_templates_dictionary_from_current_transformation(
                            df_cur,
                            attr_sector,
                            dict_cur,
                            **kwargs
                        )

            
            ##  EXPORT FILES?

            if not return_q:
                for sector_abv in attr_sector.key_values:
                    
                    sector = attr_sector.field_maps.get(f"{attr_sector.key}_to_sector").get(sector_abv)
                    if sector not in sectors:
                        continue
                        
                     # get path and write output   
                    fp_write = self.base_input_database.get_template_path(
                        r, 
                        sector,
                        create_export_dir = True
                    )
                    self._log(
                        f"Exporting '{sector_abv}' template in region {r} to {fp_write}", 
                        type_log = "info"
                    )

                    sf.dict_to_excel(
                        fp_write,
                        dict_cur.get(sector_abv),
                        replace_file = replace_template,
                    )
                    
            
            self._log(f"Templates build for region {r} complete.\n\n", type_log = "info")

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"Integrated transformations build complete in {t_elapse} seconds.",
            type_log = "info"
        )
        
        return_val = dict_write if return_q else 0

        return return_val
    


    def build_templates_dictionary_from_current_transformation(self,
        df_cur: pd.DataFrame,
        attr_sector: AttributeTable,
        dict_update: dict,
        **kwargs
    ) -> Union[Dict, None]:
        """
        Support function for build_strategies_to_templates(); 

        Function Arguments
        ------------------
        - df_cur: data frame that represents a transformation
        - attr_sector: sector attribute table
        - dict_update: dictionary to update

        Keyword Arguments
        -----------------
        **kwargs: passed to self.input_template.template_from_inputs()
        """

        for sector_abv in attr_sector.key_values:

            # get sector name
            sector = attr_sector.field_maps.get(f"{attr_sector.key}_to_sector").get(sector_abv)
            df_template = self.dict_sectoral_templates.get(sector)

            # build template
            dict_write_cur = self.input_template.template_from_inputs(
                df_cur,
                df_template,
                sector_abv,
                **kwargs
            )

            dict_update[sector_abv].update(dict_write_cur)

        return dict_update
    


    def check_exogenous_strategies(self,
        df_exogenous_strategies: pd.DataFrame,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """
        Check df_exogenous_strategies for build_strategies_to_templates(). If 
            df_exogenous_strategies is valid, will return a dictionary mapping
            a region (key) to a dataframe containing exogenous strategies.

            Otherwise, returns None. 
        """
        # check if exogenous strategies are being passed
        dict_exogenous_grouped = None
        return_none = False

        if isinstance(df_exogenous_strategies, pd.DataFrame):
            
            # check fields
            if self.key_region not in df_exogenous_strategies.columns:
                self._log(
                    f"Region key '{self.key_region}' missing from df_exogenous_strategies. Strategies will not be read from df_exogenous_strategies.",
                    type_log = "warning",
                )
                return_none = True
            
            if self.key_strategy not in df_exogenous_strategies.columns:
                self._log(
                    f"Strategy key '{self.key_strategy}' missing from df_exogenous_strategies. Strategies will not be read from df_exogenous_strategies.",
                    type_log = "warning",
                )
                return_none = True

            
            if return_none:
                return None

            # next, try to group the dataframe
            try:
                dict_exogenous_grouped = sf.group_df_as_dict(
                    df_exogenous_strategies,
                    [self.key_region],
                )

            except Exception as e:
                self._log(
                    f"Error trying to group df_exogenous_strategies in check_exogenous_strategies(): {e}.",
                    type_log = "error",
                )

        # return output
        return dict_exogenous_grouped
    


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
        sf._optional_log(
            self.logger, 
            msg, 
            type_log = type_log, 
            **kwargs
        )

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

    def transformation_pflo_baseline(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Baseline" from which other transformations deviate 
            (pass through)
        """

        # clean production scalar so that they are always 1 in the first time period
        #df_out = tbg.prepare_demand_scalars(
        #    df_input,
        #    self.model_ippu.modvar_ippu_scalar_production,
        #    self.model_attributes,
        #    key_region = self.key_region,
        #)
        df_out = df_input.copy()

        # call transformations from other sectors
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


