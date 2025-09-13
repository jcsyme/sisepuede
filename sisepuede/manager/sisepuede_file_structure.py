
import logging
import os, os.path
import pickle
import re
import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf
from typing import *


from sisepuede.core.analysis_id import AnalysisID
from sisepuede.models.energy_production import EnergyProduction




##########################
#    GLOBAL VARIABLES    #
##########################

# INITIALIZE UUID
_MODULE_UUID = "3BBF1FAF-D051-4F90-AD2F-08ECD92B853B"


# SET ANY OTHER IMPORTANT GLOBALS

_FILE_NAME_CONFIG = "sisepuede_config.yaml"
_REGEX_TEMPLATE_PREPEND = "sisepuede_run"





##  BASE CLASS

class SISEPUEDEFileStructure:
    """Create and verify the directory structure for SISEPUEDE.

    Optional Arguments
    ------------------
    attribute_time_period : Union[str, pathlib.Path, pd.DataFrame, AttributeTable, None]
        Optional time period attribute table to pass to model attributes for
        overwriting default. 
    dir_ingestion : Union[str, None]
        Directory containing templates for ingestion. The ingestion directory 
        should include subdirectories for each template class that may be run, 
        including:
            * calibrated: input variables that are calibrated for each region
                and sector
            * demo: demo parameters that are independent of region (default in
                quick start)
            * uncalibrated: preliminary input variables defined for each region
                that have not yet been calibrated

        The calibrated and uncalibrated subdirectories require separate
            subdrectories for each region, each of which contains an input
            template for each
    fn_config : str
        Name of configuration file in SISEPUEDE directory
    id_str : Union[str, None]
        Optional id_str used to create AnalysisID (see ?AnalysisID for
        more information on properties). Can be used to set outputs for a
        previous ID/restore a session.
        * If None, creates a unique ID for the session (used in output file
            names)
    initialize_directories : bool
        If False, will not create output directories or attempt to pickle
    logger : Union[logging.Logger, None]
        Optional logging.Logger object used for logging
    regex_template_prepend : str
        String to prepend to output files tagged with the analysis id.
    stop_on_missing_templates : bool
        If template directory is not found, stop? 
    """
    def __init__(self,
        attribute_time_period: Union[str, 'pathlib.Path', 'pd.DataFrame', 'AttributeTable', None] = None,
        dir_ingestion: Union[str, None] = None,
        fn_config: str = _FILE_NAME_CONFIG,
        id_str: Union[str, None] = None,
        initialize_directories: bool = True,
        logger: Union[logging.Logger, None] = None,
        regex_template_prepend: str = _REGEX_TEMPLATE_PREPEND,
        stop_on_missing_templates: bool = False,
    ) -> None:

        self.logger = logger
        self._set_basic_properties()
        self._initialize_analysis_id(
            id_str,
            regex_template_prepend = regex_template_prepend
        ) # initialize the model_attributes pickle key here since it depends on analysis id

        # run checks of directories
        self._check_config(fn_config)
        self._check_required_directories()
        self._check_ingestion(
            dir_ingestion,
            stop_on_missing = stop_on_missing_templates,
        )
        self._check_optional_directories()

        # initialize model attributes, set runtime id, then check/instantiate downstream file paths
        self._initialize_file_path_defaults(
            initialize_directories = initialize_directories,
        )
        self._initialize_model_attributes(
            attribute_time_period = attribute_time_period,
            initialize_directories = initialize_directories,
        )
        self._check_nemomod_reference_file_paths()
        
        # initialize the uuid
        self._initialize_uuid()

        return None





    ##############################
    #	SUPPORTING FUNCTIONS	#
    ##############################

    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """Clean implementation of sf._optional_log in-line using default 
            logger. See ?sf._optional_log for more information

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



    def _set_basic_properties(self,
    ) -> None:
        """Sets the following trivial properties:

        """

        return None





    ##########################
    #    DIRECTORY CHECKS    #
    ##########################

    def _check_config(self,
        fn_config: str,
    ) -> None:
        """Check the configuration file name. Sets the following properties:

            * self.fn_config
        """

        self.fn_config = _FILE_NAME_CONFIG
        if isinstance(fn_config, str):
            self.fn_config = fn_config if fn_config.endswith(".yaml") else self.fn_config
        
        return None



    def _check_required_directories(self,
    ) -> None:
        """Check directory structure for SISEPUEDE. Sets the following 
            properties:

            * self.dir_attribute_tables
            * self.dir_docs
            * self.dir_jl
            * self.dir_manager
            * self.dir_proj
            * self.dir_ref
            * self.dir_ref_nemo
            * self.fp_config
        """

        # initialize base paths
        self.dir_manager = os.path.dirname(os.path.realpath(__file__))
        self.dir_proj = os.path.dirname(self.dir_manager)

        # initialize error message
        count_errors = 0
        msg_error_dirs = ""


        # check configuration file
        self.fp_config = os.path.join(self.dir_proj, self.fn_config)
        if not os.path.exists(self.fp_config):
            count_errors += 1
            msg_error_dirs += f"\n\tConfiguration file '{self.fp_config}' not found"
            self.fp_config = None


        # check docs path
        self.dir_docs = (
            os.path.join(os.path.dirname(self.dir_proj), "docs", "source") 
            if (self.dir_proj is not None) 
            else ""
        )

        if not os.path.exists(self.dir_docs):
            self._log(
                f"\n\tDocs subdirectory '{self.dir_docs}' not found.",
                type_log = "warning",
                warn_if_none = False,
            )
            self.dir_docs = None


        # check attribute tables path (within docs path)
        self.dir_attribute_tables = os.path.join(self.dir_proj, "attributes") if (self.dir_proj is not None) else ""
        if not os.path.exists(self.dir_attribute_tables):
            count_errors += 1
            msg_error_dirs += f"\n\tAttribute tables subdirectory '{self.dir_attribute_tables}' not found"
            self.dir_attribute_tables = None


        # check command line utilities directory
        self.dir_command_line = os.path.join(self.dir_proj, "command_line")
        if not os.path.exists(self.dir_command_line):
            count_errors += 1
            msg_error_dirs += f"\n\tCommand line utility subdirectory '{self.dir_command_line}' not found"
            self.dir_command_line = None


        # check Julia directory
        self.dir_jl = os.path.join(self.dir_proj, "julia")
        if not os.path.exists(self.dir_jl):
            count_errors += 1
            msg_error_dirs += f"\n\tJulia subdirectory '{self.dir_jl}' not found"
            self.dir_jl = None


        # check reference directory
        self.dir_ref = os.path.join(self.dir_proj, "ref")
        if not os.path.exists(self.dir_ref):
            count_errors += 1
            msg_error_dirs += f"\n\tReference subdirectory '{self.dir_ref}' not found"
            self.dir_ref = None


        # check NemoMod reference directory (within reference directory)
        self.dir_ref_nemo = os.path.join(self.dir_ref, "nemo_mod") if (self.dir_ref is not None) else ""
        if not os.path.exists(self.dir_ref_nemo):
            count_errors += 1
            msg_error_dirs += f"\n\tNemoMod reference subdirectory '{self.dir_ref_nemo}' not found"
            self.dir_ref_nemo = None


        # error handling
        if count_errors > 0:
            self._log(f"There were {count_errors} errors initializing the SISEPUEDE directory structure:{msg_error_dirs}", type_log = "error")
            raise RuntimeError("SISEPUEDE unable to initialize file directories. Check the log for more information.")
        else:
            self._log(
                f"Verification of SISEPUEDE directory structure completed successfully with 0 errors.", 
                type_log = "info",
                warn_if_none = False,
            )

        return None



    def _check_ingestion(self,
        dir_ingestion: Union[str, None],
        stop_on_missing: bool = False,
    ) -> None:
        """Check path to templates. Sets the following properties:

            * self.dir_ingestion
            * self.dict_data_mode_to_template_directory
            * self.valid_data_modes

        Function Arguments
        ------------------
        dir_ingestion : Union[str, None]
            Ingestion directory storing input templates for SISEPUEDE
            * If None, defaults to ..PATH_SISEPUEDE/ref/ingestion

        Keyword Arguments
        -----------------
        stop_on_missing : bool
            If template directory is not found, stop? 
        """

        ##  Check template ingestion path (within reference directory)

        # initialize
        valid_data_modes = ["calibrated", "demo", "uncalibrated"]
        valid_data_modes_not_demo = [x for x in valid_data_modes if x != "demo"]
        dir_ingestion_default = os.path.join(self.dir_ref, "ingestion")
        dict_data_mode_to_template_directory = None

        # override if input path is specified
        set_default = not isinstance(dir_ingestion, str)
        set_default |= (not os.path.exists(dir_ingestion)) if not set_default else set_default
        set_default &= not stop_on_missing # we want it to fail if stopping on missing
        if set_default:
            dir_ingestion = dir_ingestion_default

        # check existence
        if not os.path.exists(dir_ingestion):
            msg = f"\tIngestion templates subdirectory '{self.dir_ingestion}' not found. Setting to default..."
            self._log(msg, type_log = "error", )
            raise RuntimeError(msg)


        # build the dictionary
        dict_data_mode_to_template_directory = {
            "demo": os.path.join(dir_ingestion_default, "demo")
        }

        dict_data_mode_to_template_directory.update(
            dict(
                zip(
                    valid_data_modes_not_demo,
                    [os.path.join(dir_ingestion, x) for x in valid_data_modes_not_demo]
                )
            )
        )
        
        
        ##  SET PROPERTIES

        self.dict_data_mode_to_template_directory = dict_data_mode_to_template_directory
        self.dir_ingestion = dir_ingestion
        self.dir_ingestion_default = dir_ingestion_default
        self.valid_data_modes = valid_data_modes

        return None



    def _check_optional_directories(self,
        initialize_directories: bool = True,
    ) -> None:
        """Check directories that are not critical to SISEPUEDE functioning, 
            including those that can be created if not found. Checks the 
            following properties:

            * self.dir_out
            * self.dir_ref_batch_data
            * self.dir_ref_examples
            * self.dir_ref_data_crosswalks
            * self.dir_ref_metadata
        """

        # output and temporary directories (can be created)
        self.dir_out = None
        self.dir_tmp = None

        if self.dir_proj is not None:
            self.dir_out = sf.check_path(
                os.path.join(self.dir_proj, "out"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )

            self.dir_tmp = sf.check_path(
                os.path.join(self.dir_proj, "tmp"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )


        # batch data directories (not required to run SISEPUEDE, but required for Data Generation notebooks and routines)
        self.dir_ref_batch_data = None
        self.dir_ref_examples = None
        self.dir_ref_data_crosswalks = None

        if self.dir_ref is not None:
            self.dir_ref_batch_data = sf.check_path(
                os.path.join(self.dir_ref, "batch_data_generation"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )

            self.dir_ref_data_crosswalks = sf.check_path(
                os.path.join(self.dir_ref, "data_crosswalks"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )

            self.dir_ref_examples = sf.check_path(
                os.path.join(self.dir_ref, "examples"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )

            self.dir_ref_metadata = sf.check_path(
                os.path.join(self.dir_ref, "metadata"), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )

        return None



    ###############################################
    #    INITIALIZE FILES AND MODEL ATTRIBUTES    #
    ###############################################

    def _check_nemomod_reference_file_paths(self,
    ) -> None:
        """Check and initiailize any NemoMod reference file file paths. Sets the 
            following properties:

            * self.allow_electricity_run
            * self.required_reference_tables_nemomod
        """

        # initialize
        self.allow_electricity_run = True
        self.required_reference_tables_nemomod = None

        # error handling
        count_errors = 0
        msg_error = ""

        if (self.dir_ref_nemo is not None) and (self.dir_jl is not None):

            # nemo mod input files - specify required, run checks
            model_enerprod = EnergyProduction(
                self.model_attributes,
                self.dir_jl,
                self.dir_ref_nemo,
                initialize_julia = False
            )
            self.required_reference_tables_nemomod = model_enerprod.required_reference_tables

            # initialize dictionary of file paths
            dict_nemomod_reference_tables_to_fp_csv = dict(zip(
                self.required_reference_tables_nemomod,
                [None for x in self.required_reference_tables_nemomod]
            ))

            # check all required tables
            for table in self.required_reference_tables_nemomod:
                fp_out = os.path.join(self.dir_ref_nemo, f"{table}.csv")
                if os.path.exists(fp_out):
                    dict_nemomod_reference_tables_to_fp_csv.update({table: fp_out})
                else:
                    count_errors += 1
                    msg_error += f"\n\tNemoMod reference table '{table}' not found in directory {self.dir_ref_nemo}."
                    self.allow_electricity_run = False
                    del dict_nemomod_reference_tables_to_fp_csv[table]
        else:
            count_errors += 1
            msg_error = "\n\tNo NemoMod model refererence files were found."
            self.allow_electricity_run = False

        if msg_error != "":
            self._log(
                f"There were {count_errors} while trying to initialize NemoMod:{msg_error}\nThe electricity model cannot be run. Disallowing electricity model runs.", 
                type_log = "error",
            )

        else:
            # log if logging--no warnings
            self._log(
                f"NemoMod reference file checks completed successfully.", 
                type_log = "info",
                warn_if_none = False,
            )



    def _initialize_analysis_id(self,
        id_str: Union[str, None],
        regex_template_prepend: str = "sisepuede_run",
    ) -> None:
        """Initialize the session id. Initializes the following properties:

            * self.analysis_id (AnalysisID object)
            * self.from_existing_analysis_id
            * self.id (shortcurt to self.analysis_id.id)
            * self.id_fs_safe (shortcurt to self.analysis_id.id_fs_safe)
            * self.model_attributes_pickle_archival_key
            * self.regex_template_analysis_id

        Function Arguments
        ------------------
        - id_str: input id_str. If None, initializes new AnalysisID. If passing
            a string, tries to read existing ID.

        Keyword Arguments
        -----------------
        - regex_template_prepend: string to prepend to output files tagged with
            the analysis id.
        """
        # set the template
        regex_template_prepend = (
            "sisepuede_run" 
            if not isinstance(regex_template_prepend, str) 
            else (
                "sisepuede_run" 
                if (len(regex_template_prepend) == 0) 
                else regex_template_prepend
            )
        )

        self.regex_template_analysis_id = re.compile(f"{regex_template_prepend}_(.+$)")
        self.analysis_id = AnalysisID(
            id_str = id_str,
            logger = self.logger,
            regex_template = self.regex_template_analysis_id
        )
        self.from_existing_analysis_id = (not self.analysis_id.new_init)
        self.id = self.analysis_id.id
        self.id_fs_safe = self.analysis_id.id_fs_safe
        self.model_attributes_pickle_archival_key = f"model_attributes_{self.id}"

        return None
        


    def _initialize_file_path_defaults(self,
        initialize_directories: bool = True,
    ) -> None:
        """Initialize any default file paths, including output and temporary 
            files. Sets the following properties:

                * self.dir_base_output_raw
                * self.fp_base_output_raw
                * self.fp_log_default
                * self.fp_pkl_model_attributes_archive
                * self.fp_sqlite_tmp_nemomod_intermediate
                * self.fp_variable_specification_xl_types
        """

        # initialize file base names
        fbn_log = f"{self.id_fs_safe}_log.log"
        fbn_output_db = f"{self.id_fs_safe}_output_database"
        fn_output_pkl = f"{self.id_fs_safe}_model_attributes.pkl"

        # initialize output paths
        dir_base_output_raw = None
        fp_base_output_raw = None
        fp_log_default = None
        fp_pkl_model_attributes_archive = None
        fp_sqlite_tmp_nemomod_intermediate = None
        fp_variable_specification_of_sampling_unit_types = None

        ##  BUILD SUBDIRECTORIES

        # create a subdirectory in which to store all files associated with a run -- include output db and model_attriutes pickle
        dir_base_output_raw = (
            sf.check_path(
                os.path.join(self.dir_out, self.id_fs_safe), 
                create_q = initialize_directories,
                throw_error_q = initialize_directories,
            )
            if self.dir_out is not None
            else None
        )


        ##  ANALYSIS-RUN ID DEPENDENT OUTPUT STRINGS

        if dir_base_output_raw is not None:
            # base output path for CSV or SQL--if CSVs, represents a directory. If SQLite, append .sqlite to get path
            fp_base_output_raw = os.path.join(dir_base_output_raw, fbn_output_db)
            # defaut logger path
            fp_log_default = os.path.join(dir_base_output_raw, fbn_log)
            # output path to store model_attributes pickle, including configuration parameters etc.
            fp_pkl_model_attributes_archive = os.path.join(dir_base_output_raw, fn_output_pkl)
        

        ##  OTHER FILES

        # SQLite Database location for intermediate NemoMod calculations
        fp_sqlite_tmp_nemomod_intermediate = os.path.join(self.dir_tmp, "nemomod_intermediate_database.sqlite")
        
        # file storing optional exogenous XL types for variable specifications
        fp_variable_specification_xl_types = os.path.join(self.dir_ref, "variable_specification_xl_types.csv")


        ##  ASSIGN PROPERTIES

        self.dir_base_output_raw = dir_base_output_raw
        self.fp_base_output_raw = fp_base_output_raw
        self.fp_log_default = fp_log_default
        self.fp_pkl_model_attributes_archive = fp_pkl_model_attributes_archive
        self.fp_sqlite_tmp_nemomod_intermediate = fp_sqlite_tmp_nemomod_intermediate
        self.fp_variable_specification_xl_types = fp_variable_specification_xl_types

        return None



    def _initialize_model_attributes(self,
        attribute_time_period: Union[str, 'pathlib.Path', 'pd.DataFrame', 'AttributeTable', None] = None,
        initialize_directories: bool = True,
    ) -> None:
        """
        Initialize SISEPUEDE model attributes from directory structure. Sets the following
            properties:

            * self.model_attributes
        """
        self.model_attributes = None
        from_existing = False

        # check if loading from existing
        if self.from_existing_analysis_id:
            model_attributes = self.try_restore_model_attributes_from_pickle()
            if model_attributes is not None:
                self.model_attributes = model_attributes
                from_existing = True

        # if not, instantiate using attribute tables and config
        create_from_id = self.dir_attribute_tables is not None
        create_from_id &= self.fp_config is not None
        create_from_id &= not from_existing

        if create_from_id:
            model_attributes = ma.ModelAttributes(self.dir_attribute_tables, self.fp_config, )
            (
                self._write_model_attributes_to_pickle(model_attributes)
                if initialize_directories
                else None
            )
            self.model_attributes = model_attributes
        
        
        ##  FINALLY, ATTEMPT TO UPDATE THE TIME PERIOD ATTRIBUTE

        attr_cur = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_time_period,
        )

        # finally, try to update
        self.model_attributes.update_dimensional_attribute_table(
            attribute_time_period, 
            key = attr_cur.key,
            stop_on_error = False,
        )

        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID
        """

        self.is_sisepuede_file_structure = True
        self._uuid = _MODULE_UUID

        return None



    def try_restore_model_attributes_from_pickle(self,
        fp_pkl: Union[str, None] = None,
        key_model_attributes: Union[str, None] = None
    ) -> Union[ma.ModelAttributes, None]:
        """
        Load a model attributes object from a SISEPUEDE archived Python pickle.
            Used to restore previous sessions. Returns a ma.ModelAttributes object
            if the model_attributes object is successfully found in the pickle.
            Called in self._initialize_model_attributes()

        Keyword Arguments
        -----------------
        - fp_pkl: file path of the pickle to use to load the ma.ModelAttributes
            object
        - key_model_attributes: dictionary key to use in pickle to find
            ma.ModelAttributes object
        """

        fp_pkl = (
            self.fp_pkl_model_attributes_archive 
            if (fp_pkl is None) 
            else fp_pkl
        )
        key_model_attributes = (
            self.model_attributes_pickle_archival_key 
            if not isinstance(key_model_attributes, str) 
            else key_model_attributes
        )
        out = None
        successfully_found_model_attributes = False

        # check path specification
        if (not os.path.exists(fp_pkl)) | (fp_pkl is None):
            self._log(f"Path to model_attributes pickle '{fp_pkl}' not found. The session cannot be loaded.", type_log = "error")
            return None


        try:
            # try to load from pickle
            with (open(fp_pkl, "rb")) as f:
                while not successfully_found_model_attributes:
                    try:
                        out = pickle.load(f)
                        if isinstance(out, dict):
                            out = out.get(key_model_attributes)
                            if isinstance(out, ma.ModelAttributes):
                                successfully_found_model_attributes = True
                    except EOFError:
                        break

        except Exception as e:
            self._log(f"Error trying to load model_attributes from pickle at '{fp_pkl}': {e}", type_log = "error")

        if successfully_found_model_attributes:
            msg = f"Successfully loaded model_attributes from pickle at '{fp_pkl}'."
            type_log = "info"
        else:
            msg = f"Error trying to load model_attributes from pickle at '{fp_pkl}': no model_attributes found with key {key_model_attributes}."
            type_log = "warning"
            out = None

        self._log(msg, type_log = type_log)

        return out



    def _write_model_attributes_to_pickle(self,
        model_attributes: ma.ModelAttributes,
        fp_pkl: Union[str, None] = None,
        key_model_attributes: Union[str, None] = None
    ) -> None:
        """
        Write a model attributes object to a SISEPUEDE archived Python pickle.
            Used to facilitate restoration of the session. Writes the
            self.model_attributes ma.ModelAttributes object to a pickle if that
            path does not already exist.

        Function Arguments
        ------------------
        - model_attributes: ma.ModelAttributes to pickle

        Keyword Arguments
        -----------------
        - fp_pkl: file path of the pickle to use to load the ma.ModelAttributes
            object
        - key_model_attributes: dictionary key to use in pickle to find
            ma.ModelAttributes object
        """

        fp_pkl = self.fp_pkl_model_attributes_archive if (fp_pkl is None) else fp_pkl
        key_model_attributes = self.model_attributes_pickle_archival_key if not isinstance(key_model_attributes, str) else key_model_attributes

        # check path specification
        if os.path.exists(fp_pkl):
            self._log(f"Path to model_attributes pickle '{fp_pkl}' already exists. The file will not be overwritten.", type_log = "error")
            return None

        # try to write to a new pickle
        try:
            with open(fp_pkl, "wb") as fp:
                pickle.dump(
                    {key_model_attributes: model_attributes},
                    fp,
                    protocol = pickle.HIGHEST_PROTOCOL
                )
            self._log(f"Successfully archived self.model_attributes to pickle at '{fp_pkl}'", type_log = "info")

        except Exception as e:
            self._log(f"Error trying to write self.model_attributes to pickle at '{fp_pkl}': {e}", type_log = "error")




###################################
#    SIMPLE CHECKING FUNCTIONS    #
###################################

def is_sisepuede_file_structure(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDEFileStructure object
    """

    out = hasattr(obj, "is_sisepuede_file_structure")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out