
import logging
import numpy as np
import pandas as pd
import pathlib
import time
from typing import *


from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import ModelAttributes
from sisepuede.manager.sisepuede_experimental_manager import *
from sisepuede.manager.sisepuede_file_structure import *
import sisepuede.core.support_classes as sc
import sisepuede.manager.sisepuede_models as sm
import sisepuede.manager.sisepuede_output_database as sod
import sisepuede.transformers as trf
import sisepuede.utilities._toolbox as sf



##########################
#    GLOBAL VARIABLES    #
##########################



# SET SOME OTHER DEFAULTS
_DEFAULT_PREFIX_SUMMARY_RUNS = "sisepuede_summary_results_run"
_DEFAULT_REGEX_TEMPLATE_PREPEND = "sisepuede_run"
_DEFAULT_TABLE_NAME_WIDE = "WIDE_INPUTS_OUTPUTS"

# KEYS IN THE CONFIG
_KEY_CONFIG_N_LHS = "num_lhc_samples"
_KEY_CONFIG_OUTPUT_METHOD = "output_method"
_KEY_CONFIG_RANDOM_SEED = "random_seed"
_KEY_CONFIG_TIME_PERIOD_U0 = "time_period_u0"


# INITIALIZE UUID
_MODULE_UUID = "DF134D51-DAD3-43CA-BCB3-7E92B568E357"




####################
#    MAIN CLASS    #
####################

class SISEPUEDE:
    """SISEPUEDE (SImulation of SEctoral Pathways and Uncertainty Exploration 
        for DEcarbonization) is an integrated modeling framework (IMF) used to
        assess decarbonization pathways under deep uncertainty. SISEPUEDE
        estimates GHG emissions primarily using the IPCC Guidelines for
        Greenhouse Gas Inventories (2006 and 2019R) and further includes costs
        and benefits of transformation-level strategies across 4 emission
        sectors and 16 emission subsectors.

        The SISEPUEDE IMF includes the following components:

        * Integrated GHG Inventory Model (SISEPUEDEModels)
        * Economic assessment of technical costs and co-benefits
        * Uncertainty tools (SISEPUEDEExperimentalManager)
        * Flexible database management (SISEPUEDEOutputDatabase)
        * Automated data population using open access data sources
        10-20 pre-defined transformations per sector + cross sectoral strategies

    More on SISPUEDE, including model documentation, a description of sectors,
        and a quick start guide, can be found at the SISEPUEDE documentation,
        located at

        https://self.readthedocs.io


    LICENSE
    -------

    Copyright (C) 2023 James Syme

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.



    Key Subclasses
    --------------
    SISEPUEDE organizes a number of subclasses under a shared umbrella, creating
        an integrated framework for decarbonization modeling and uncertainty
        exploration. The following classes are key to data organization and 
        management, trajectory generation and uncertainty quantification, and
        integrated decarboniation modeling. 

        - SISEPUEDEExperimentalManager (SISEPUEDE.experimental_manager)
            Includes the folliwing key data components:

            * FutureTrajectories (.dict_future_trajectories(region)): for each
                region, the FutureTrajectories is used to generate input
                databases based on strategy, future, and design.
            * LHSDesign (.dict_lhs_design(region)): for each region, the
                LHSDesign object stores LHC samples for lever effects 
                (arr_lhs_l) and exogenous uncertainties (arr_lhs_x)

        - SISEPUEDEFileStructure (SISEPUEDE.file_struct)
            Includes all default file names and the overarching organizational
                structure of files and paths for SISEPUEDE.
        
        - SISEPUEDEModels (SISEPUEDE.models)
            Includes all models and the implementation/organization of the
                SISEPUEDE directed acyclic graph (DAG). 

        - SISEPUEDEOutputDatabase (SISEPUEDE.database)
            Includes all IterativeDatabaseTables for SISEPUEDE, allowing runs to
                be saved to both CSV and SQLite databases.

        More information on these classes can be obtained by querying their
            docstrings directly (?class)


    Initialization Arguments
    ------------------------
    data_mode : str
        Template class to initialize from. Three options are allowed:
        * calibrated
        * demo
        * uncalibrated


    Optional Arguments
    ------------------
    Optional arguments are used to pass values to SISEPUEDE outside of the
        configuaration framework. This can be a desireable approach for
        iterative modeling or determining a suitable data pipeline.

    attribute_design : Union[AttributeTable, None]
        Optional AttributeTable object used to specify the design_id.
        * Note: If None, will attempt to find a table within the ModelAttributes
            object with key "dim_design_id". If none is found, will assume a
            single design where:

            (a) exogenous uncertainties vary at ranges specified in input
                templates; and
            (b) lever effects are fixed.
    attribute_time_period : Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None]
        Optional path to a csv, DataFrame, or AttributeTable object used to 
        specify the time_period.
        * Note: If None, will attempt to find a table within the ModelAttributes
            object with key "dim_time_period". If none is found, will throw an
            error.
    db_type : Union[str, None]
        Optional specification of an IterativeDataBase type. 
        * "csv": write to a CSV database (each table is a CSV)
        * "sqlite": write tables to sqlite database
        * None: defaults to configuration database
    dir_ingestion : Union[str, None]
        Directory storing SISEPUEDE templates for ingestion
        * Note: if running outside of demo mode, the directory should contain
            subdirectories dor each region, with each region including input
            templates for each of the 5 SISEPUEDE sectors. For example, the
            directory should have the following tree structure:

            * dir_ingestion
                |_ calibrated
                    |_ region_1
                        |_ model_input_variables_af_demo.xlsx
                        |_ model_input_variables_ce_demo.xlsx
                        |_ model_input_variables_en_demo.xlsx
                        |_ model_input_variables_ip_demo.xlsx
                        |_ model_input_variables_se_demo.xlsx
                    |_ region_2
                        |_ model_input_variables_af_demo.xlsx
                        .
                        .
                        .
                |_ demo
                    |_ model_input_variables_af_demo.xlsx
                    |_ model_input_variables_ce_demo.xlsx
                    |_ model_input_variables_en_demo.xlsx
                    |_ model_input_variables_ip_demo.xlsx
                    |_ model_input_variables_se_demo.xlsx
                |_ uncalibrated
                    |_ region_1
                        |_ model_input_variables_af_demo.xlsx
                        |_ model_input_variables_ce_demo.xlsx
                        |_ model_input_variables_en_demo.xlsx
                        |_ model_input_variables_ip_demo.xlsx
                        |_ model_input_variables_se_demo.xlsx
                    |_ region_2
                        |_ model_input_variables_af_demo.xlsx
                        .
                        .
                        .
    id_str : Union[str, None]
        Optional id_str used to create AnalysisID (see ?AnalysisID for more 
        information on properties). Can be used to set outputs for a previous 
        ID/restore a session.
        * If None, creates a unique ID for the session (used in output file
            names)
    initialize_as_dummy : bool
        Initialize as a dummy? If true, the following  outcomes occur:

        * Output directories are not created
        * Electricity model is not initialized

        NOTE: DO NOT SET TO TRUE UNDER NORMAL CIRCUMSTANCES. Dummy mode is used
            to access common SISEPUEDE components without leveraging model runs
            or database instantiation. 
    logger : Union[logging.Logger, None]
        Optional logging.Logger object to use for logging
    n_trials : Union[int, None]
        Number of LHS futures to generate data for
    random_seed : Union[int, None]
        Optional random seed to pass to SISEPUEDEExperimentalManager. If None, 
        defaults to Configuration seed. Can be used to coordinate experiments.
    regex_template_prepend : str
        String to prepend to output files tagged with the analysis id
    regions : Union[List[str], None]
        List of regions to include in the experiment
    replace_output_dbs_on_init : bool
        Default is set to false; if True, will destroy exisiting output tables 
        if an AnalysisID is specified.
    strategies : Union[Strategies, None]
        Optional Stratgies object used to exogenously specify strategies and 
        templates for use. 
        NOTE: If specified, then it overwrites `dir_ingestion` and forces 
            `data_mode` to calibrated 
    try_exogenous_xl_types_in_variable_specification : Union[bool, str, pathlib.Path, pd.DataFrame]
        Behavior by type:
        * Bool:                     Tries to read from 
                                        self.file_structure.fp_variable_specification_xl_types
                                        if True. If False, infers on a 
                                        region-by-region basis.
        * string or pathlib.Path:   Tries to read CSV at path containing 
                                        variable specifications and exogenous XL 
										types. Useful if coordinating across a 
										number of regions.
                                    NOTE: must have fields `field_variable` and 
									    `field_xl_type`)
        * pd.DataFrame:             Direct input of exogenous types as DataFrame
        * None or other:            Instantiates XL types by inference alone.
    """

    def __init__(self,
        data_mode: str,
        attribute_design: Union[AttributeTable, None] = None,
        attribute_time_period: Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None] = None,
        db_type: Union[str, None] = None,
        dir_ingestion: Union[str, None] = None,
        id_str: Union[str, None] = None,
        initialize_as_dummy: bool = False,
        logger: Union[logging.Logger, None] = None,
        n_trials: Union[int, None] = None,
        random_seed: Union[int, None] = None,
        regex_template_prepend: str = _DEFAULT_REGEX_TEMPLATE_PREPEND,
        regions: Union[List[str], None] = None,
        replace_output_dbs_on_init: bool = False,
        strategies: Union[trf.Strategies, None] = None,
        try_exogenous_xl_types_in_variable_specification: Union[bool, str, pathlib.Path, pd.DataFrame] = False,
        **kwargs,
    ) -> None:

        # initialize the file structure and generic properties - check strategies
        self._initialize_file_structure(
            attribute_time_period = attribute_time_period,
            dir_ingestion = dir_ingestion,
            id_str = id_str,
            logger = logger,
            regex_template_prepend = regex_template_prepend,
            strategies = strategies,
        )
        self._initialize_support_classes()
        self._initialize_attribute_design(attribute_design, )
        # self._initialize_attribute_time_period(attribute_time_period, )
        self._initialize_keys()

        # initialize the output database
        self._initialize_output_database(
            db_type = db_type,
            replace_output_dbs_on_init = replace_output_dbs_on_init,
        )
        self._initialize_data_mode(data_mode)
        self._initialize_experimental_manager(
            num_trials = n_trials,
            random_seed = random_seed,
            regions = regions,
            try_exogenous_xl_types_in_variable_specification = try_exogenous_xl_types_in_variable_specification,
        )

        # initialize models, aliases, and, finally, base tables
        self._initialize_models(
            initialize_as_dummy = initialize_as_dummy,
        )
        self._initialize_function_aliases()
        self._initialize_base_database_tables()

        # initialize key marker
        self._initialize_uuid()
    
        return None


    
    def __call__(self,
        *args,
        **kwargs,
    ):
        """
        Call self.project_scenarios()
        """
        out = self.project_scenarios(
            *args,
            **kwargs,
        )

        return out




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_attribute_design(self,
        attribute_design: Union[AttributeTable, None] = None,
    ) -> None:
        """Initialize and check the attribute design table. Sets the following
            properties:

            * self.attribute_design


        Keyword Arguments
        -----------------
        attribute_design: AttributeTable used to specify designs.
            * If None, tries to access "dim_design_id" from
                ModelAttributes.dict_attributes
        """

        key_model_attributes_design = self.model_attributes.dim_design_id

        # initialize the attribute design table -- checks on the table are run when experimental manager is initialized
        self.attribute_design = (
            self.model_attributes.get_dimensional_attribute_table(key_model_attributes_design)
            if not is_attribute_table(attribute_design) 
            else attribute_design
        )

        return None


    ##  HEREHERE DELETE
    def _initialize_attribute_time_period(self,
        attribute_time_period: Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None] = None,
    ) -> None:
        """Initialize and check the attribute design table. Sets the following
            properties:

            * self.attribute_time_period
 
        Keyword Arguments
        -----------------
        attribute_time_period : Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None]
            AttributeTable storing information on time periods and years
        """

        ##  GET THE ATTRIBUTE TABLE 

        attr_cur = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_time_period,
        )

        # finally, try to update
        self.model_attributes.update_dimensional_attribute_table(
            attribute_time_period, 
            key = attr_cur.key,
        )
    
        return None




    def get_config_parameter(self,
        parameter: str
    ) -> Union[int, List, str]:
        """
        Retrieve a configuration parameter from self.model_attributes. Must be 
            initialized after _initialize_file_structure()
        """

        return self.model_attributes.configuration.get(parameter)



    def _initialize_base_database_tables(self,
        try_write_lhs: bool = False,
    ) -> None:
        """
        Initialize database tables that characterize the analytical
            configuration. Initializes the following tables:

            * self.database.table_name_analysis_metadata
            * self.database.table_name_attribute_design
            * self.database.table_name_attribute_lhs_l
            * self.database.table_name_attribute_lhs_x
            * self.database.table_name_attribute_strategy
            * self.database.table_name_base_input

        Keyword Arguments
        -----------------
        - try_write_lhs: attempt to write LHS samples to table? Note that this
            may be difficult if there are a large number of parameters (SQLite
            has a default limit of 2000 columns).
        """

        if not self.from_existing_analysis_id:

            # get some tables
            df_analysis_metadata = self.model_attributes.configuration.to_data_frame()
            df_attribute_design = self.attribute_design.table
            df_lhs_l, df_lhs_x = self.build_lhs_tables()
            df_attribute_strategy = self.attribute_strategy.table
            df_base_input = self.experimental_manager.base_input_database.database


            ##  WRITE TABLES TO OUTPUT DATABASE

            self.database._write_to_table(
                self.database.table_name_analysis_metadata,
                df_analysis_metadata
            )

            self.database._write_to_table(
                self.database.table_name_attribute_design,
                df_attribute_design
            )

            (
                self.database._write_to_table(
                    self.database.table_name_attribute_lhs_l,
                    df_lhs_l
                ) 
                if (df_lhs_l is not None) & try_write_lhs
                else None
            )
            (
                self.database._write_to_table(
                    self.database.table_name_attribute_lhs_x,
                    df_lhs_x
                ) 
                if (df_lhs_x is not None) & try_write_lhs
                else None
            )

            self.database._write_to_table(
                self.database.table_name_attribute_strategy,
                df_attribute_strategy
            )

            self.database._write_to_table(
                self.database.table_name_base_input,
                df_base_input
            )
        
        else:
            self._log(
                "WARNING: re-initialization from analyis id requires addition coding to facilitate the re-generation of inputs properly. FIX THIS",
                type_log = "warning"
            )

        return None



    def _initialize_data_mode(self,
        data_mode: Union[str, None] = None,
        default_mode: str = "demo"
    ) -> None:
        """
        Initialize mode of operation. Sets the following properties:

            * self.data_mode
            * self.demo_mode
            * self.dir_templates
            * self.valid_data_modes
        """
        self.valid_data_modes = self.file_struct.valid_data_modes

        try:
            data_mode = default_mode if (data_mode is None) else data_mode
            self.data_mode = default_mode if (data_mode not in self.valid_data_modes) else data_mode
            self.demo_mode = (self.data_mode == "demo")
            self.dir_templates = (
                self.file_struct.dict_data_mode_to_template_directory.get(self.data_mode) 
                if (self.file_struct is not None) 
                else None
            )

            self._log(f"Running SISEPUEDE under template data mode '{self.data_mode}'.", type_log = "info")

        except Exception as e:
            self._log(f"Error in _initialize_data_mode(): {e}", type_log = "error")
            raise RuntimeError()

        return None



    def _initialize_experimental_manager(self,
        key_config_n_lhs: str = _KEY_CONFIG_N_LHS, 
        key_config_random_seed: str = _KEY_CONFIG_RANDOM_SEED,
        key_config_time_period_u0: str = _KEY_CONFIG_TIME_PERIOD_U0,
        num_trials: Union[int, None] = None,
        random_seed: Union[int, None] = None,
        regions: Union[List[str], None] = None,
        time_t0_uncertainty: Union[int, None] = None,
        try_exogenous_xl_types_in_variable_specification: Union[bool, str, pathlib.Path] = False,
    ) -> None:
        """Initialize the Experimental Manager for self. The
            SISEPUEDEExperimentalManager class reads in input templates to 
            generate input databases, controls deployment, generation of 
            multiple runs, writing output to applicable databases, and 
            post-processing of applicable metrics. Users should use 
            SISEPUEDEExperimentalManager to set the number of trials and the 
            start year of uncertainty. Sets the following properties:

            * self.baseline_future
            * self.baseline_strategy
            * self.experimental_manager
            * self.n_trials
            * self.odpt_primary
            * self.random_seed
            * self.regions
            * self.time_period_u0


        Keyword Arguments
        -----------------
        key_config_n_lhs : str
            Configuration key used to determine the number of LHC samples to
            generate
        key_config_random_seed : str
            Configuration key used to set the random seed
        key_config_time_period_u0 : str
            Configuration key used to determine the time period of initial 
            uncertainty in uncertainty assessment.
        num_trials : Union[int, None]
            Number if LHS trials to run.
            * If None, revert to configuration defaults from self.model_attributes
        random_seed : Union[int, None]
            Optional random seed used to generate LHS samples
            * If None, revert to configuration defaults from 
                self.model_attributes
            * To run w/o setting to configuration value, set random_seed = -1
        regions : Union[List[str], None]
            Regions to initialize.
            * If None, initialize using all regions
        time_t0_uncertainty : t
            Time where uncertainty starts
        try_exogenous_xl_types_in_variable_specification : bool 
            * If True, attempts to read exogenous XL type specifcations for 
                variable specifications (i.e., in SamplingUnit) from 
                self.file_structure.fp_variable_specification_xl_types. 
            * If False, infers on a region-by-region basis
        """

        # initialize experimental manager
        self.experimental_manager = None

        # get some key parameters, including: 
        #  - number of LHS trials
        #  - random seed 
        #  - initial time period for uncertainty
        #  - file path to exogenous XL types for variable specifications
        num_trials = (
            int(np.round(max(num_trials, 0)))
            if sf.isnumber(num_trials)
            else self.get_config_parameter(key_config_n_lhs)
        )
        
        random_seed = (
            self.get_config_parameter(key_config_random_seed)
            if not sf.isnumber(random_seed, integer = True)
            else random_seed
        )
        
        time_period_u0 = self.get_config_parameter(key_config_time_period_u0) #HEREHERE - edit to pull from default too
        
        # initialize and adjust by type
        xl_types_spec = None

        if isinstance(try_exogenous_xl_types_in_variable_specification, bool):
            if try_exogenous_xl_types_in_variable_specification:
                xl_types_spec = self.file_struct.fp_variable_specification_xl_types

        elif isinstance(try_exogenous_xl_types_in_variable_specification, (str, pathlib.Path, pd.DataFrame)):
            xl_types_spec = try_exogenous_xl_types_in_variable_specification


        try:
            self.experimental_manager = SISEPUEDEExperimentalManager(
                self.attribute_design,
                self.model_attributes,
                self.dir_templates,
                regions,
                time_period_u0,
                num_trials,
                demo_database_q = self.demo_mode,
                exogenous_xl_type_for_variable_specifcations = xl_types_spec,
                logger = self.logger,
                random_seed = random_seed,
            )

            self._log(
                f"Successfully initialized SISEPUEDEExperimentalManager.", 
                type_log = "info",
                warn_if_none = False,
            )

        except Exception as e:
            self._log(
                f"Error initializing the experimental manager in _initialize_experimental_manager(): {e}", 
                type_log = "error",
                warn_if_none = False,
            )
            raise RuntimeError(e)


        self.attribute_strategy = self.experimental_manager.attribute_strategy
        self.odpt_primary = self.experimental_manager.primary_key_database
        self.baseline_future = self.experimental_manager.baseline_future
        self.baseline_strategy = self.experimental_manager.baseline_strategy
        self.n_trials = self.experimental_manager.n_trials
        self.random_seed = self.experimental_manager.random_seed
        self.regions = self.experimental_manager.regions
        self.time_period_u0 = self.experimental_manager.time_period_u0

        return None



    def _initialize_file_structure(self,
        attribute_time_period: Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None] = None,
        dir_ingestion: Union[str, None] = None,
        id_str: Union[str, None] = None,
        logger: Union[logging.Logger, str, None] = None,
        regex_template_prepend: str = _DEFAULT_REGEX_TEMPLATE_PREPEND,
        strategies: Union[trf.Strategies, None] = None,
    ) -> None:
        """Intialize the SISEPUEDEFileStructure object and model_attributes 
            object. Initializes the following properties:

            * self.analysis_id
            * self.file_struct
            * self.fp_base_output_raw
            * self.fp_log 	(via self._initialize_logger())
            * self.id		(via self._initialize_logger())
            * self.id_fs_safe
            * self.logger
            * self.model_attributes

        Optional Arguments
        ------------------
        attribute_time_period : Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None]
            AttributeTable storing information on time periods and years
        dir_ingestion : Union[str, None]
            Directory containing templates for ingestion. The ingestion 
            directory should include subdirectories for each template class that 
            may be run, including:
                * calibrated: input variables that are calibrated for each
                    region and sector
                * demo: demo parameters that are independent of region (default
                    in quick start)
                * uncalibrated: preliminary input variables defined for each
                    region that have not yet been calibrated
            The calibrated and uncalibrated subdirectories require separate
                subdrectories for each region, each of which contains an input
                template for each
        id_str : Union[str, None]
            Optional id_str used to create AnalysisID (see ?AnalysisID for more 
            information on properties). Can be used to set outputs for a 
            previous ID/restore a session.
            * If None, creates a unique ID for the session (used in output file
                names)
        logger : Optional logging.Logger object OR file path to use for 
            logging. If None, create a log in the SISEPUEDE output directory 
            associated with the analysis ID
        strategies:  optional specification of Strategies object for use in
            defining experiments
        """

        ##  CHECK STRATEGIES

        attr_strategy = None

        # if a Strategies object is specified, set the ingestion directory and update the strategy attribute
        if trf.is_strategies(strategies):
            
            # overwrite dir ingestion
            dir_ingestion = os.path.dirname(strategies.dir_templates)
            attr_strategy = (
                strategies
                .model_attributes
                .get_dimensional_attribute_table(
                    strategies
                    .model_attributes
                    .dim_strategy_id
                )
            )


        ##  TRY INITIALIZING THE FILE STRUCTURE

        self.file_struct = None
        self.model_attributes = None

        try:
            self.file_struct = SISEPUEDEFileStructure(
                attribute_time_period = attribute_time_period,
                dir_ingestion = dir_ingestion,
                id_str = id_str,
                regex_template_prepend = regex_template_prepend,
            )
            print("yay")

        except Exception as e:
            print("yay")
            # occurs before logger is setup
            msg = f"Error trying to initialize SISEPUEDEFileStructure: {e}"
            raise RuntimeError(msg)
        
        # overwrite strategy attribute table?
        if attr_strategy is not None:
            key_dim = self.file_struct.model_attributes.attribute_group_key_dim
            key_strat = self.file_struct.model_attributes.dim_strategy_id

            (
                self
                .file_struct
                .model_attributes
                .dict_attributes
                .get(key_dim)
                .update({key_strat: attr_strategy, })
            )


        # setup logging
        self._initialize_logger(logger = logger)
        self._log(
            f"Successfully initialized SISEPUEDEFileStructure.", 
            type_log = "info",
        )

        # setup shortcut paths
        self.analysis_id = self.file_struct.analysis_id
        self.fp_base_output_raw = self.file_struct.fp_base_output_raw
        self.from_existing_analysis_id = self.file_struct.from_existing_analysis_id
        self.id = self.file_struct.id
        self.id_fs_safe = self.file_struct.id_fs_safe
        self.model_attributes = self.file_struct.model_attributes

        return None



    def _initialize_keys(self,
    ) -> None:
        """
        Initialize scenario dimension keys that are shared for initialization.
            Initializes the followin properties:

            * self.key_design
            * self.key_future
            * self.key_primary
            * self.key_region
            * self.key_strategy
            * self.key_time_period
            * self.keys_index

        NOTE: these keys are initialized separately within
            SISEPUEDEExperimentalManager, but they depend on the same shared
            sources (attribute_design and self.model_attributes).
        """

        # set keys
        self.key_design = self.attribute_design.key
        self.key_future = self.model_attributes.dim_future_id
        self.key_primary = self.model_attributes.dim_primary_id
        self.key_region = self.model_attributes.dim_region
        self.key_strategy = self.model_attributes.dim_strategy_id
        self.key_time_period = self.model_attributes.dim_time_period

        self.keys_index = [
            self.key_design,
            self.key_future,
            self.key_primary,
            self.key_region,
            self.key_strategy
        ]

        return None



    def _initialize_logger(self,
        format_str = "%(asctime)s - %(levelname)s - %(message)s",
        logger: Union[logging.Logger, str, None] = None,
        namespace: str = __name__,
    ) -> None:
        """Setup a logger object that leverages the file structure and current 
            id.
        
        NOTE: Must be called within self._initialize_file_structure() 

        Sets the following properties:

            * self.fp_log 
            * self.logger

        Keyword Arguments
        -----------------
        - format_str: string used to format output string
        - logger: optional logger object OR string giving path to output logger.
            If None, generates logger at self.file_struct.fp_log_default
        - namespace: namespace to use for the logger
        """

        fn_out = None

        if (not isinstance(logger, logging.Logger)):
            fn_out = (
                logger
                if isinstance(logger, str)
                else self.file_struct.fp_log_default
            )

            # setup logger
            logging.basicConfig(
                filename = fn_out,
                filemode = "w",
                format = format_str,
                level = logging.DEBUG
            )

            logger = logging.getLogger(namespace)

            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            # create formatter, add to console handler, and add the handler to logger
            formatter = logging.Formatter(format_str)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        self.fp_log = fn_out
        self.logger = logger

        return None



    def _initialize_models(self,
        dir_jl: Union[str, None] = None,
        dir_nemomod_reference_files: Union[str, None] = None,
        fp_sqlite_tmp_nemomod_intermediate: Union[str, None] = None,
        initialize_as_dummy: bool = False,
    ) -> None:
        """Initialize models for self. Sets the following properties:

            * self.dir_jl
            * self.dir_nemomod_reference_files
            * self.fp_sqlite_tmp_nemomod_intermediate
            * self.models

            
        For the following optional arguments, entering = None will return the 
            SISEPUEDE default

        Optional Arguments
        ------------------
        dir_jl : Union[str, None]
            File path to julia environment and supporting module directory
        dir_nemomod_reference_files : Union[str, None]
            Directory containing NemoMod reference files
        fp_nemomod_temp_sqlite_db : Union[str, None]
            File name for temporary database used to run NemoMod
        initialize_as_dummy : bool
            Initialize without electricity?
        """

        dir_jl = self.file_struct.dir_jl if (dir_jl is None) else dir_jl
        dir_nemomod_reference_files = (
            self.file_struct.dir_ref_nemo 
            if (dir_nemomod_reference_files is None) 
            else dir_nemomod_reference_files
        )
        fp_sqlite_tmp_nemomod_intermediate = (
            self.file_struct.fp_sqlite_tmp_nemomod_intermediate 
            if (fp_sqlite_tmp_nemomod_intermediate is None) 
            else fp_sqlite_tmp_nemomod_intermediate
        )

        try:
            
            init_elec = (not initialize_as_dummy) & self.file_struct.allow_electricity_run

            self.models = sm.SISEPUEDEModels(
                self.model_attributes,
                allow_electricity_run = init_elec,
                fp_julia = dir_jl,
                fp_nemomod_reference_files = dir_nemomod_reference_files,
                fp_nemomod_temp_sqlite_db = fp_sqlite_tmp_nemomod_intermediate,
                logger = self.logger
            )

            self._log(
                f"Successfully initialized SISEPUEDEModels.", 
                type_log = "info",
            )
            
            if not self.file_struct.allow_electricity_run:
                self._log(f"\tOne or more reference files are missing, and the electricity model cannot be run. This run will not include electricity results. Try locating the missing files and re-initializing SISEPUEDE to run the electricity model.", type_log = "warning")

        except Exception as e:
            self._log(f"Error trying to initialize models: {e}", type_log = "error")
            raise RuntimeError()

        self.dir_jl = dir_jl
        self.dir_nemomod_reference_files = dir_nemomod_reference_files
        self.fp_sqlite_tmp_nemomod_intermediate = fp_sqlite_tmp_nemomod_intermediate

        return None



    def _initialize_output_database(self,
        config_key_output_method: str = _KEY_CONFIG_OUTPUT_METHOD, 
        db_type: Union[str, None] = None,
        default_db_type: str = "sqlite",
        replace_output_dbs_on_init: bool = False,
    ) -> None:
        """
        Initialize the SISEPUEDEOutputDatabase structure. Allows for quick
            reading and writing of data files. Sets the following properties:

            * self.database


        Keyword Arguments
        -----------------
        - config_key_output_method: configuration key to use to determine the
            method for the output database.
        - db_type: optional specification of database type. If None, defaults to
            configuration.
        - default_db_type: default type of output database to use if invalid
            entry found from config.
        - replace_output_dbs_on_init: replace output database tables on
            initialization if they exist? Only applies if loading from an
            existing dataset.
        """
        # try getting the configuration parameter
        db_type = (
            self.get_config_parameter(config_key_output_method)
            if not isinstance(db_type, str)
            else db_type
        )
        db_type = (
            default_db_type 
            if not isinstance(db_type, str)
            else db_type
        )
        self.database = None

        try:
            self.database = sod.SISEPUEDEOutputDatabase(
                db_type,
                {
                    "design": self.key_design,
                    "future": self.key_future,
                    "primary": self.key_primary,
                    "region": self.key_region,
                    "strategy": self.key_strategy,
                    "time_series": None
                },
                analysis_id = self.analysis_id,
                fp_base_output = self.fp_base_output_raw,
                create_dir_output = True,
                logger = self.logger,
                replace_on_init = False,
            )

        except Exception as e:
            msg = f"Error initializing SISEPUEDEOutputDatabase: {e}"
            self._log(msg, type_log = "error")


        if self.database is None:
            return None

        # log if successful
        self._log(
            f"Successfully initialized database with:\n\ttype:\t{db_type}\n\tanalysis id:\t{self.id}\n\tfp_base_output:\t{self.fp_base_output_raw}",
            type_log = "info"
        )


        ##  COMPLETE SOME ADDITIONAL INITIALIZATIONS

        # remove the output database if specified
        if replace_output_dbs_on_init:
            tables_destroy = [
                self.database.table_name_output
            ]

            for table in tables_destroy:
                self._destroy_table(table)
        
        return None



    def _initialize_support_classes(self,
    ) -> None:
        """
        Initialize some simple shared objects for region and time period 
            management. Sets the following properties:

            * self.regions_definitions
            * self.time_period_definitions
        """

        region_definitions = sc.Regions(self.model_attributes)
        time_period_definitions = sc.TimePeriods(self.model_attributes)

        self.region_definitions = region_definitions
        self.time_period_definitions = time_period_definitions

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID
        """

        self.is_sisepuede = True
        self._uuid = _MODULE_UUID

        return None



    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See
            ?sf._optional_log for more information.

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)





    ############################
    #    SHORTCUT FUNCTIONS    #
    ############################

    def _destroy_table(self,
        table_name: Union[str, None]
    ) -> None:
        """
        Destroy a table (delete rows and reset columns) without removing from
            the database.
        """
        if table_name is None:
            return None

        self.database.db._destroy(table_name)



    def get_lhs_trajectories(
        key_specification: Union[dict, int],
        region: str,
    ) -> Tuple[pd.Series, pd.Series, bool]:
        """
        Get LHS trajectories for input to generator. Returns a tuple of the 
            following form:

            (lhs_l, lhs_x, base_future_q)

        Function Arguments
        ------------------
        - key_specification: dictionary to filter on containing SISEPUEDE keys 
            that map to values (e.g., key_strategy, key_future, key_design OR 
            key_primary) OR primary key. Should only map to a single primary key
        - region: region to generate LHS trajectories for
        """
        # retrieve objects
        future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
        lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
        region_out = self.get_output_region(region)

        # get primary keys info
        dict_primary_keys = self.get_primary_keys(key_specification)
        dict_primary_keys = dict_primary_keys[0] if sf.islistlike(dict_primary_keys) else dict_primary_keys
        if dict_primary_keys is None:
            return None

        dict_primary_keys = self.odpt_primary.get_dims_from_key(
            dict_primary_keys, 
            return_type = "dict"
        )


        design = dict_primary_keys.get(self.key_design)
        future = dict_primary_keys.get(self.key_future)
        strategy = dict_primary_keys.get(self.key_strategy)

        df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(design, return_type = pd.DataFrame)

        # reduce lhs tables - LEs
        df_lhs_l = (
            df_lhs_l[df_lhs_l[self.key_future].isin([future])] 
            if (df_lhs_l is not None) 
            else df_lhs_l
        )
        # Xs
        df_lhs_x = (
            df_lhs_x[df_lhs_x[self.key_future].isin([future])] 
            if (df_lhs_x is not None) 
            else df_lhs_x
        )


        ##  GET LHS SERIE BY FUTURE

        # determine if baseline future and fetch lhs rows
        base_future_q = (future == self.baseline_future)
        lhs_l = (
            df_lhs_l[df_lhs_l[self.key_future] == future].iloc[0] 
            if ((df_lhs_l is not None) and not base_future_q) 
            else None
        )
        lhs_x = (
            df_lhs_x[df_lhs_x[self.key_future] == future].iloc[0] 
            if ((df_lhs_x is not None) and not base_future_q) 
            else None
        )
        
        tup_out = lhs_l, lhs_x, base_future_q
        
        return tup_out



    def get_primary_keys(self,
        primary_keys: Union[List[int], Dict[str, int], None]
    ) -> List[int]:
        """
        Based on list of primary keys or subsetting dictioary, get a list of
            primary keys. Used to support filtering in a number of contexts.


        Function Arguments
        ------------------
        - primary_keys: list of primary keys to run OR dictionary of index keys
            (e.g., strategy_id, design_id) with scenarios associated as values
            (uses AND operation to filter scenarios). If None, returns all
            possible primary keys.
        """


        if isinstance(primary_keys, dict):

            df_odpt = (
                self.odpt_primary.get_indexing_dataframe_from_primary_key(
                    primary_keys.get(self.odpt_primary.key_primary),
                    keys_return = [self.odpt_primary.key_primary],
                )
                if self.odpt_primary.key_primary in primary_keys.keys()
                else self.odpt_primary.get_indexing_dataframe(
                    key_values = primary_keys,
                    keys_return = [self.odpt_primary.key_primary],
                )
            )

            primary_keys = sorted(list(df_odpt[self.odpt_primary.key_primary]))

        elif isinstance(primary_keys, list):
            primary_keys = sorted([x for x in primary_keys if x in self.odpt_primary.range_key_primary])
        
        elif primary_keys is None:
            primary_keys = self.odpt_primary.range_key_primary

        return primary_keys



    def _initialize_function_aliases(self,
    ) -> None:
        """
        Initialize function aliases.
        """

        self.get_output_region = self.experimental_manager.get_output_region

        return None



    def _read_table(self,
        primary_keys: Union[List[int], Dict[str, int], None],
        table_name: str,
        regions: Union[List[str], None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Underlying function to facilitate shortcuts self.read_input() and
            self.read_output()

        Function Arguments
        ------------------
        primary_keys : Union[List[int], Dict[str, int], None]
            Specification of primary keys, which is a ist of primary keys to run 
            OR dictionary of index keys (e.g., strategy_id, design_id etc) with 
            scenarios associated as values (uses AND operation to filter 
            scenarios). If None, returns all possible primary keys (only use if
            number of runs is known to be small enough)
        table_name : str
            Table name to read.

        Keyword Arguments
        -----------------
        drop_duplicates : bool 
            Drop duplicates in a CSV when reading? (only applies if the database 
            is initialized using CSVs)
            * Default is False to improve speeds
            * Set to True to ensure that only unique rows are read in
        query_logic : str
            Default is "and". Subsets table using:

            (
                WHERE field_a IN (val_a1, val_a2, ..., val_am) 
                ~ field_b IN (val_b1, val_b2, ..., val_bn)...
            )

            and the relation `~` is in `["and", "or"]`
        regions : Union[List[str], None]
            Optional list-like specification of regions to retrieve
        """

        # get primary keys and initialize subset
        primary_keys = self.get_primary_keys(primary_keys)
        dict_subset = (
            {
                self.key_primary: primary_keys
            } 
            if not isinstance(primary_keys, range) 
            else {}
        )
        
        # check region specification against (a) all regions and (b) available in system
        if regions is not None:
            regions = self.region_definitions.get_valid_regions(regions)
            regions = (
                [x for x in regions if regions in self.regions]
                if regions is not None
                else []
            )
            if len(regions) > 0:
                dict_subset.update({self.key_region: regions})

            else:
                return None
                

        # check for additional arguments passed and remove the subset dictionary if it is passed
        dict_subset_kwargs = kwargs.get("dict_subset")
        if isinstance(dict_subset_kwargs, dict):
            dict_subset.update(
                dict(
                    (k, v) for k, v in dict_subset_kwargs.items() 
                    if k not in dict_subset.keys()
                )
            )

        if dict_subset_kwargs is not None:
            del kwargs["dict_subset"]

        df_out = self.database.read_table(
            table_name,
            dict_subset = dict_subset,
            **kwargs
        )

        return df_out
    


    def read_input(self,
        primary_keys: Union[List[int], Dict[str, int], None],
        regions: Union[List[str], None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read input data generated after running .project_scenarios.

        Function Arguments
        ------------------
        primary_keys : Union[List[int], Dict[str, int], None]
            Specification of primary keys, which is a ist of primary keys to run 
            OR dictionary of index keys (e.g., strategy_id, design_id etc) with 
            scenarios associated as values (uses AND operation to filter 
            scenarios). If None, returns all possible primary keys (only use if
            number of runs is known to be small enough)

        Keyword Arguments
        -----------------
        dict_subset : Union[Dict[str, List], None]
            Dictionary with keys that are columns in the table and values, given 
            as a list, to subset the table. dict_subset is written as:

            dict_subset = {
                field_a = [val_a1, val_a2, ..., val_am],
                field_b = [val_b1, val_b2, ..., val_bn],
                .
                .
                .
            }

            NOTE: dict_subset should NOT contain self.key_primary (it will be
            removed if passed in dict_subset) since these are passed in the
            `primary_keys` argument
        drop_duplicates : bool 
            Drop duplicates in a CSV when reading? (only applies if the database 
            is initialized using CSVs)
            * Default is False to improve speeds
            * Set to True to ensure that only unique rows are read in
        fields_select : Union[List[str], None]
            Fields to read in. Reducing the number of fields to read can speed 
            up the ingestion process and reduce the data frame's memory 
            footprint.
        query_logic : str
            Default is "and". Subsets table using:

            (
                WHERE field_a IN (val_a1, val_a2, ..., val_am) 
                ~ field_b IN (val_b1, val_b2, ..., val_bn)...
            )

            and the relation `~` is in `["and", "or"]`
        regions : Union[List[str], None]
            Optional list-like specification of regions to retrieve
        """
        df_out = self._read_table(
            primary_keys,
            self.database.table_name_input,
            regions = regions,
            **kwargs
        )

        return df_out
    


    def read_output(self,
        primary_keys: Union[List[int], Dict[str, int], None],
        regions: Union[List[str], None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read output data generated after running .project_scenarios.

        Function Arguments
        ------------------
        primary_keys : Union[List[int], Dict[str, int], None]
            Specification of primary keys, which is a ist of primary keys to run 
            OR dictionary of index keys (e.g., strategy_id, design_id etc) with 
            scenarios associated as values (uses AND operation to filter 
            scenarios). If None, returns all possible primary keys (only use if
            number of runs is known to be small enough)

        Keyword Arguments
        -----------------
        dict_subset : Union[Dict[str, List], None]
            Dictionary with keys that are columns in the table and values, given 
            as a list, to subset the table. dict_subset is written as:

            dict_subset = {
                field_a = [val_a1, val_a2, ..., val_am],
                field_b = [val_b1, val_b2, ..., val_bn],
                .
                .
                .
            }

            NOTE: dict_subset should NOT contain self.key_primary (it will be
            removed if passed in dict_subset) since these are passed in the
            `primary_keys` argument
        drop_duplicates : bool 
            Drop duplicates in a CSV when reading? (only applies if the database 
            is initialized using CSVs)
            * Default is False to improve speeds
            * Set to True to ensure that only unique rows are read in
        fields_select : Union[List[str], None]
            Fields to read in. Reducing the number of fields to read can speed 
            up the ingestion process and reduce the data frame's memory 
            footprint.
        query_logic : str
            Default is "and". Subsets table using:

            (
                WHERE field_a IN (val_a1, val_a2, ..., val_am) 
                ~ field_b IN (val_b1, val_b2, ..., val_bn)...
            )

            and the relation `~` is in `["and", "or"]`
        regions : Union[List[str], None]
            Optional list-like specification of regions to retrieve
        """
        df_out = self._read_table(
            primary_keys,
            self.database.table_name_output,
            regions = regions,
            **kwargs
        )

        return df_out



    def _write_chunk_to_table(self,
        df_list: List[pd.DataFrame],
        check_duplicates: bool = False,
        reinitialize_on_verification_failure: bool = False,
        table_name: Union[str, None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Write a chunk of data frames to output database.

        Function Arguments
        ------------------
        - df_list: list of data frames to write

        Keyword Arguments
        -----------------
        = check_duplicates: check for duplicate rows?
        - reinitialize_on_verification_failure: reinitialize the table columns
            if verification fails? 
            * NOTE: Verification can occur due to unknown bug that causes 
                table.columns to accumulate dimensions
        - table_name: table name to write to. Default is
            self.database.table_name_output
        - **kwargs: passed to IterativeDatabaseTable._write_to_table
        """

        table_name = self.database.table_name_output if (table_name is None) else table_name

        df_out = pd.concat(df_list, axis = 0).reset_index(drop = True)
        df_out.drop_duplicates(inplace = True) if check_duplicates else None

        self.database._write_to_table(
            table_name,
            df_out,
            reinitialize_on_verification_failure = reinitialize_on_verification_failure,
            **kwargs
        )
        df_out = []

        return df_out




    #########################
    #    TABLE FUNCTIONS    #
    #########################

    def build_lhs_tables(self,
    ) -> pd.DataFrame:
        """
        Build LHS tables for export to database. Returns a tuple

            df_l, df_x

            where `df_l` is the database of lever effect LHC samples and `df_x`
            is the database of exogenous uncertainty LHC samples. Both are long
            by region and LHS key.
        """

        # initialize output
        df_l = []
        df_x = []

        for region in self.regions:

            lhsd = self.experimental_manager.dict_lhs_design.get(region)
            df_lhs_l, df_lhs_x = lhsd.retrieve_lhs_tables_by_design(None, return_type = pd.DataFrame)
            region_out = self.get_output_region(region)

            # lever effect LHS table
            if (df_lhs_l is not None):
                df_lhs_l = sf.add_data_frame_fields_from_dict(
                    df_lhs_l,
                    {
                        self.key_region: region_out
                    }
                )
                df_l.append(df_lhs_l)

            # exogenous uncertainty LHS table
            if (df_lhs_x is not None):
                df_lhs_x = sf.add_data_frame_fields_from_dict(
                    df_lhs_x,
                    {
                        self.key_region: region_out
                    }
                )
                df_x.append(df_lhs_x)

        df_l = pd.concat(df_l, axis = 0).reset_index(drop = True) if (len(df_l) > 0) else None
        if (df_l is not None):
            df_l.columns = [str(x) for x in df_l.columns]
            fields_ord_l = [self.key_region, lhsd.field_lhs_key]
            fields_ord_l += sf.sort_integer_strings([x for x in df_l.columns if x not in fields_ord_l])
            df_l = df_l[fields_ord_l]

        df_x = pd.concat(df_x, axis = 0).reset_index(drop = True) if (len(df_x) > 0) else None
        if df_x is not None:
            df_x.columns = [str(x) for x in df_x.columns]
            fields_ord_x = [self.key_region, lhsd.field_lhs_key]
            fields_ord_x += sf.sort_integer_strings([x for x in df_x.columns if x not in fields_ord_x])
            df_x = df_x[fields_ord_x]

        return df_l, df_x






    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def generate_scenario_database_from_primary_key(self,
        primary_key: Union[int, None],
        regions: Union[List[str], str, None] = None,
        **kwargs
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """Generate an input database for SISEPUEDE based on the primary key.

        Function Arguments
        ------------------
        primary_key : Union[int, None] 
            Primary key to generate input database for
            * returns None if primary key entered is invalid

        Keyword Arguments
        -----------------
        regions : Union[List[str], str, None]
            List of regions or string of a region to include.
            * If a list of regions or single region is entered, returns a
                dictionary of input databases of the form
                {region: df_input_region, ...}
            * Invalid regions return None
        **kwargs : 
            Passed to self.models.project(..., **kwargs)
        """

        # check primary keys to run
        if (primary_key not in self.odpt_primary.range_key_primary):
            self._log(f"Error in generate_scenario_database_from_primary_key: {self.key_primary} = {primary_key} not found.", type_log = "error")
            return None

        # check region
        regions = self.regions if (regions is None) else regions
        regions = [regions] if not isinstance(regions, list) else regions
        regions = [x for x in regions if x in self.regions]
        if len(regions) == 0:
            self._log(f"Error in generate_scenario_database_from_primary_key: no valid regions found in input.", type_log = "error")
            return None

        # get designs
        dict_primary_keys = self.odpt_primary.get_dims_from_key(
            primary_key,
            return_type = "dict"
        )

        # initialize output (TEMPORARY)
        dict_return = {}

        for region in regions:

            # retrieve region specific future trajectories and lhs design
            future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
            lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
            region_out = self.get_output_region(region)


            ##  GET DIMENSIONS

            design = dict_primary_keys.get(self.key_design) # int(df_primary_keys_cur_design[self.key_design].iloc[0])
            future = dict_primary_keys.get(self.key_future) # int(df_primary_keys_cur_design[self.key_future].iloc[0])
            strategy = dict_primary_keys.get(self.key_strategy) # int(df_primary_keys_cur_design[self.key_strategy].iloc[0])


            ##  GET LHS TABLES AND FILTER

            df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(design, return_type = pd.DataFrame)

            # reduce lhs tables - LEs
            df_lhs_l = (
                df_lhs_l[df_lhs_l[self.key_future].isin([future])] 
                if (df_lhs_l is not None) 
                else df_lhs_l
            )
            # Xs
            df_lhs_x = (
                df_lhs_x[df_lhs_x[self.key_future].isin([future])] 
                if (df_lhs_x is not None) 
                else df_lhs_x
            )


            ##  GENERATE INPUT BY FUTURE

            # determine if baseline future and fetch lhs rows
            base_future_q = (future == self.baseline_future)
            lhs_l = (
                df_lhs_l[df_lhs_l[self.key_future] == future].iloc[0] 
                if ((df_lhs_l is not None) and not base_future_q) 
                else None
            )
            lhs_x = (
                df_lhs_x[df_lhs_x[self.key_future] == future].iloc[0] 
                if ((df_lhs_x is not None) and not base_future_q) 
                else None
            )

            # generate the futures and get available strategies
            df_input = future_trajectories_cur.generate_future_from_lhs_vector(
                lhs_x,
                df_row_lhc_sample_l = lhs_l,
                future_id = future,
                baseline_future_q = base_future_q,
            )


            ##  FILTER BY STRATEGY

            df_input = (
                df_input[
                    df_input[self.key_strategy].isin([strategy])
                ]
                .sort_values(by = [self.model_attributes.dim_time_period])
                .drop(
                    [x for x in df_input.columns if x in self.keys_index], 
                    axis = 1
                )
                .reset_index(drop = True)
            )

            ##  ADD IDS AND RETURN

            df_input = sf.add_data_frame_fields_from_dict(
                df_input,
                {
                    self.key_region: region_out,
                    self.key_primary: primary_key,
                    self.key_time_period: None,
                },
                field_hierarchy = self.model_attributes.sort_ordered_dimensions_of_analysis,
                pass_none_to_shift_index = True,
                prepend_q = True,
                sort_input_fields = True,
            )

            dict_return.update({region_out: df_input})

        return dict_return
    


    def generate_summary_files(self,
        primary_keys: Union[List[int], Dict[str, int], None],
        build_inputs_on_none: bool = True, 
        export: bool = False, 
        **kwargs,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """Generate a summary file that merges inputs and outputs from 
            SISEPUEDE. Recommended to use only for smaller runs. 

        Function Arguments
        ------------------
        primary_keys : Union[List[int], Dict[str, int], None]
            Specification of primary keys, which is a ist of primary keys to run 
            OR dictionary of index keys (e.g., strategy_id, design_id etc) with 
            scenarios associated as values (uses AND operation to filter 
            scenarios). If None, returns all possible primary keys (only use if
            number of runs is known to be small enough)

        Keyword Arguments
        -----------------
        build_input_on_none : bool
            If the input table is not found (e.g., due to not saving or SQL 
            errors), build it on the fly? This will increase run time.
        export : bool
            Export output to a summary results package in the associated output 
            directory?
        **kwargs :
            Passed to read_output() and read_input()
        """
        
        # read input and output
        df_out = self.read_output(primary_keys, **kwargs, )
        df_in = self.read_input(primary_keys, **kwargs, )
        all_primaries = sorted(list(df_out[self.key_primary].unique()))
        
        # build inputs if unable to simply read the data frame
        if (df_in is None) and build_inputs_on_none:
            
            dict_df_in = {}

            for primary in all_primaries: 
                df_in_cur = self.generate_scenario_database_from_primary_key(primary)

                for k, v in df_in_cur.items():
                    (
                        dict_df_in[k].append(v)
                        if k in dict_df_in.keys()
                        else dict_df_in.update({k: [v]})
                    )
                        
            # sort by region and concatenate
            regions_sorted = sorted(list(dict_df_in.keys()))
            df_in = (
                pd.concat(
                    sum([dict_df_in.get(k) for k in regions_sorted], []), 
                    axis = 0,
                )
                .reset_index(drop = True, )
            )

        
        # build a wide dataframe and the primary ids
        df_wide = pd.merge(df_out, df_in, how = "left", )
        df_primary = self.odpt_primary.get_indexing_dataframe(all_primaries, )


        # LHS INFO
        # lhs_design = self.experimental_manager.dict_lhs_design.get("uganda")
        # lhs_design.retrieve_lhs_tables_by_design(3, return_type = pd.DataFrame, )
        #

        # build output dictionary
        dict_out = {
            _DEFAULT_TABLE_NAME_WIDE: df_wide,
            self.database.table_name_attribute_primary: df_primary,
        }

        for tab in [self.database.table_name_attribute_strategy]:
            df = self.database.db.read_table(tab)
            dict_out.update({tab: df, })


        ##  EXPORT?
        
        if export:
            # check output directory 
            dir_pkg = pathlib.Path(self.file_struct.dir_out).joinpath( 
                f"{_DEFAULT_PREFIX_SUMMARY_RUNS}_{self.id_fs_safe}"
            )
            dir_pkg.mkdir(exist_ok = True, ) if not dir_pkg.is_dir() else None

            # export each item in the dictionary
            for k, v in dict_out.items():
                v.to_csv(
                    dir_pkg.joinpath(f"{k}.csv"),
                    encoding = "UTF-8",
                    index = None,
                )


        return dict_out



    def project_scenarios(self,
        primary_keys: Union[List[int], Dict[str, int], None],
        check_results: bool = True,
        chunk_size: int = 10,
        force_overwrite_existing_primary_keys: bool = False,
        max_attempts: int = 2,
        regions: Union[List[str], str, None] = None,
        reinitialize_output_table_on_verification_failure: bool = False,
        save_inputs: Union[bool, None] = None,
        skip_nas_in_input: bool = False,
        **kwargs
    ) -> List[int]:
        """Project scenarios forward for a set of primary keys. Returns the set 
            of primary keys that ran successfully.

        Function Arguments
        ------------------
        primary_keys : Union[List[int], Dict[str, int], None]
            List of primary keys to run OR dictionary of index keys (e.g., 
            strategy_id, design_id, etc.) with scenarios associated as values 
            (uses AND operation to filter scenarios). If None, returns all 
            possible primary keys.

        Keyword Arguments
        -----------------
        check_results : bool
            Check output results when running? If True, verifies output results 
            do not exceed some threshold. See 
            SISEPUEDEModels.check_model_results() for more information (keyword
            arguments `epsilon` and `thresholds` may be passed in **kwargs)
        chunk_size : int
             size of chunk to use to write to IterativeDatabaseTable.
            If 1, updates table after every iteration; otherwise, stores chunks
            in memory, aggregates, then writes to IterativeDatabaseTable.
        force_overwrite_existing_primary_keys : bool
            If the primary key is already found in the output database table, 
            should it be overwritten? It is recommended that iterations on the 
            same scenarios be undertaken using different AnalysisID structures. 
            Otherwise, defaults to initialization resolutsion (write_skip)
        max_attempts : int
            Maximum number of attempts at successful model runs. On occasion, 
            solvers can encounter numerical instability and require a re-run; 
            setting this to greater than 1 gives the model the opportunity to 
            re-run. However, SISEPUEDE caps this number at 5.
        regions : Union[List[str], str, None]
            Optional list of regions (contained in self.regions) to project for
        reinitialize_output_table_on_verification_failure : bool
            Reinitialize the IterativeDatabaseTable output table columns if 
            there is a verification failure during iteration?
        save_inputs : bool
            Save inputs to input table in database? Defaults to configuration
            defaults if None
        skip_nas_in_input : bool
            skip futures with NAs on input? If true, will 
            skip any inputs that contain NAs
        **kwargs : 
            passed to self.models.project(..., **kwargs)
        """

        # maximum solve attempts
        max_attempts = (
            int(min(max(max_attempts, 1), 5))
            if sf.isnumber(max_attempts)
            else 2
        )
        save_inputs = (
            self.get_config_parameter("save_inputs")
            if save_inputs is None
            else save_inputs
        )

        # get all scenarios and designs associated with them
        primary_keys = self.get_primary_keys(primary_keys)
        df_primary_keys = self.odpt_primary.get_indexing_dataframe(
            key_values = primary_keys
        )
        all_designs = sorted(list(set(df_primary_keys[self.key_design])))

        # initializations
        df_out = []
        df_out_inputs = []
        df_out_primary = []
        dict_primary_keys_run = dict((x, [None for x in primary_keys]) for x in self.regions)
        iterate_outer = 0

        # available indices and resolution
        idt = self.database.db.dict_iterative_database_tables.get(
            self.database.table_name_output
        )
        index_conflict_resolution = None
        index_conflict_resolution = (
            "write_replace" 
            if (force_overwrite_existing_primary_keys or (idt.index_conflict_resolution == "write_replace")) 
            else None
        )
        set_available_ids = idt.available_indices

        # check regions specification
        regions = (
            [x for x in self.regions if x in regions]
            if sf.islistlike(regions)
            else None
        )
        regions = self.regions if (regions is None) else regions
        
        for region in regions:

            iterate_inner = 0

            # retrieve region specific future trajectories and lhs design
            future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
            lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
            region_out = self.get_output_region(region)

            self._log(f"\n***\tSTARTING REGION {region}\t***\n", type_log = "info")

            for design in all_designs:

                df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(
                    design,
                    return_type = pd.DataFrame
                )

                # get reduced set of primary keys
                df_primary_keys_cur_design = df_primary_keys[
                    df_primary_keys[self.key_design] == design
                ]
                keep_futures = sorted(list(set(df_primary_keys_cur_design[self.key_future])))

                # reduce lhs tables - LEs
                df_lhs_l = (
                    df_lhs_l[
                        df_lhs_l[self.key_future].isin(keep_futures)
                    ] 
                    if (df_lhs_l is not None) 
                    else df_lhs_l
                )
                # Xs
                df_lhs_x = (
                    df_lhs_x[
                        df_lhs_x[self.key_future].isin(keep_futures)
                    ] 
                    if (df_lhs_x is not None) 
                    else df_lhs_x
                )

                # next, loop over futures
                #  Note that self.generate_future_from_lhs_vector() will return a table for all strategies
                #  associated with the future, so we can prevent redundant calls by running all strategies
                #  that need to be run for a given future

                for future in keep_futures:

                    # determine if baseline future and fetch lhs rows
                    base_future_q = (future == self.baseline_future)
                    lhs_l = (
                        df_lhs_l[df_lhs_l[self.key_future] == future].iloc[0] 
                        if ((df_lhs_l is not None) and not base_future_q) 
                        else None
                    )
                    lhs_x = (
                        df_lhs_x[df_lhs_x[self.key_future] == future].iloc[0] 
                        if ((df_lhs_x is not None) and not base_future_q) 
                        else None
                    )

                    # generate the futures and get available strategies
                    df_input = future_trajectories_cur.generate_future_from_lhs_vector(
                        lhs_x,
                        df_row_lhc_sample_l = lhs_l,
                        future_id = future,
                        baseline_future_q = base_future_q
                    )
                    all_strategies = sorted(list(
                        set(df_input[self.key_strategy])
                    ))


                    for strategy in all_strategies:

                        # get primary id info
                        df_primary_keys_cur_design_fs = (
                            df_primary_keys_cur_design[
                                df_primary_keys_cur_design[self.key_future].isin([future]) &
                                df_primary_keys_cur_design[self.key_strategy].isin([strategy])
                            ]
                            .reset_index(drop = True)
                        )

                        id_primary = df_primary_keys_cur_design_fs[self.key_primary]
                        id_primary = int(id_primary.iloc[0]) if (len(id_primary) > 0) else None
                        write_q = ((region_out, id_primary) not in set_available_ids) 
                        write_q |= (index_conflict_resolution == "write_replace")

                        # skip iteration on these conditions
                        if not ((id_primary in primary_keys) & write_q):
                            continue
                

                        ##  FILTER THE DATA FRAME, AND, OPTIONALLY, SKIP IF NAs ARE PRESENT

                        df_input_cur = (
                            df_input[
                                df_input[self.key_strategy].isin([strategy])
                            ]
                            .copy()
                            .reset_index(drop = True)
                            .sort_values(by = [self.model_attributes.dim_time_period])
                            .drop([x for x in df_input.columns if x in self.keys_index], axis = 1)
                        )

                        # compare w/NAs and without; if skipping, log and move to next iteration
                        if skip_nas_in_input:
                            if len(df_input_cur.dropna()) != len(df_input_cur):
                                self._log(
                                    f"Skipping {self.key_primary} = {id_primary} in region {region}: NAs found in input future.", 
                                    type_log = "warning"
                                )

                                continue


                        success = False
                        self._log(
                            f"Trying run {self.key_primary} = {id_primary} in region {region}", 
                            type_log = "info"
                        )
                        


                        ##  TRY TO RUN THE MODEL AND REPORT ERRORS 

                        try:
                            t0 = time.time()

                            # initialize as None, then iterate (until max attempts) to control for numerical instabilities
                            df_output = None
                            i = 0

                            while (df_output is None) & (i < max_attempts):
                                
                                df_output = self.models.project(
                                    df_input_cur, 
                                    check_results = check_results,
                                    regions = region,
                                    **kwargs
                                )

                                i += 1
                            
                            # raise error, which will be caught and loggged
                            if df_output is None:
                                msg = f"Maximum number of attempts {max_attempts} reached without successful run. Skipping..."
                                raise RuntimeError(msg)

                            # add indices and append
                            df_output = sf.add_data_frame_fields_from_dict(
                                df_output,
                                {
                                    self.key_region: region_out,
                                    self.key_primary: id_primary,
                                    self.key_time_period: None,
                                },
                                field_hierarchy = self.model_attributes.sort_ordered_dimensions_of_analysis,
                                pass_none_to_shift_index = True,
                                prepend_q = True,
                                sort_input_fields = True,
                            )
                            df_out.append(df_output)
                            
                            t_elapse = sf.get_time_elapsed(t0)
                            success = True

                            self._log(
                                f"Model run for {self.key_primary} = {id_primary} successfully completed in {t_elapse} seconds (n_tries = {i}).", 
                                type_log = "info"
                            )

                        except Exception as e:

                            self._log(
                                f"Model run for {self.key_primary} = {id_primary} failed with the following error: {e}", 
                                type_log = "error"
                            )


                        # if the model run is successful and the chunk size is appropriate, update primary keys that ran successfully and write to output
                        if not success:
                            continue


                        ##  APPEND NON-RESULT TABLES (INPUTS, PRIMARY ATTRIBUTE, ETC.)
                
                        # append inputs if saving
                        if save_inputs:
                            df_input_cur = sf.add_data_frame_fields_from_dict(
                                df_input_cur,
                                {
                                    self.key_region: region_out,
                                    self.key_primary: id_primary,
                                    self.key_time_period: None,
                                },
                                field_hierarchy = self.model_attributes.sort_ordered_dimensions_of_analysis,
                                pass_none_to_shift_index = True,
                                prepend_q = True,
                                sort_input_fields = True,
                            )
                            df_out_inputs.append(df_input_cur)

                        # append primary
                        df_out_primary.append(df_primary_keys_cur_design_fs)


                        ##  WRITE CHUNKS

                        # write outputs
                        if (len(df_out)%chunk_size == 0) and (len(df_out) > 0):
                            df_out = self._write_chunk_to_table(
                                df_out,
                                table_name = self.database.table_name_output,
                                index_conflict_resolution = index_conflict_resolution,
                                reinitialize_on_verification_failure = reinitialize_output_table_on_verification_failure,
                            )

                        # write attribute_primary
                        if (len(df_out_primary)%chunk_size == 0) and (len(df_out_primary) > 0):
                            df_out_primary = self._write_chunk_to_table(
                                df_out_primary,
                                check_duplicates = True,
                                table_name = self.database.table_name_attribute_primary,
                                index_conflict_resolution = index_conflict_resolution
                            )

                        # write inputs (if desired)
                        if (len(df_out_inputs)%chunk_size == 0) & (len(df_out_inputs) > 0) & save_inputs:
                            df_out_inputs = self._write_chunk_to_table(
                                df_out_inputs,
                                table_name = self.database.table_name_input,
                                index_conflict_resolution = index_conflict_resolution,
                            )

                        # append to output
                        df_out_primary.append(df_primary_keys_cur_design_fs)
                        dict_primary_keys_run[region][iterate_inner] = id_primary

                        iterate_inner += 1 # number of iterations for this region
                        iterate_outer += 1 # number of total iterations

            # reduce length after running
            dict_primary_keys_run[region] = dict_primary_keys_run[region][0:iterate_inner]
            
            self._log(f"\n***\t REGION {region} COMPLETE\t***\n", type_log = "info")


        # write final output chunk
        if (len(df_out) > 0):
            self._write_chunk_to_table(
                df_out,
                table_name = self.database.table_name_output,
                index_conflict_resolution = index_conflict_resolution,
                reinitialize_on_verification_failure = reinitialize_output_table_on_verification_failure,
            )
        
        # write final primary attribute chunk
        if (len(df_out_primary) > 0):
            df_out_primary = self._write_chunk_to_table(
                df_out_primary,
                check_duplicates = True,
                table_name = self.database.table_name_attribute_primary,
                index_conflict_resolution = index_conflict_resolution
            )

        # write final inputs chunk (if saving)
        if (len(df_out_inputs) > 0) & save_inputs:
            df_out_inputs = self._write_chunk_to_table(
                df_out_inputs,
                table_name = self.database.table_name_input,
                index_conflict_resolution = index_conflict_resolution,
            )


        return dict_primary_keys_run





###################################
#    SIMPLE CHECKING FUNCTIONS    #
###################################

def is_sisepuede(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDE object
    """

    out = hasattr(obj, "is_sisepuede")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out