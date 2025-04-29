import logging
import numpy as np
import os, os.path
import pandas as pd
import sqlalchemy
import tempfile
from typing import *


from sisepuede.core.model_attributes import ModelAttributes
from sisepuede.models.afolu import AFOLU
from sisepuede.models.circular_economy import CircularEconomy
from sisepuede.models.energy_production import EnergyProduction
from sisepuede.models.energy_consumption import EnergyConsumption
from sisepuede.models.ippu import IPPU
from sisepuede.models.socioeconomic import Socioeconomic
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf



##########################
#    GLOBAL VARIABLES    #
##########################

# INITIALIZE UUID
_MODULE_UUID = "054201F6-8FBE-4DFF-A726-6D36CDFEADB7"



####################
#    MAIN CLASS    #
####################

class SISEPUEDEModels:
    """Instantiate models for SISEPUEDE.

    Initialization Arguments
    ------------------------
    model_attributes : ModelAttributes
        ModelAttributes object used to manage variables and coordination

    Optional Arguments
    ------------------
    allow_electricity_run : bool
        Allow the EnergyProduction model to run (high-runtime model)?
        * Should be left to True when running the model. Setting to False allows
            access to methods and properties without connecting to Julia and/or
            accessing the .project() method.
    fp_julia : Union[str, None]
        Path to Julia files in subdirectory to use. If None, cannot access Julia
        for EnergyProduction model. 
    fp_nemomod_reference_files : Union[str, None]
        Directory housing reference files called by NemoMod when running 
        electricity model
        * REQUIRED TO RUN ELECTRICITY MODEL
    fp_nemomod_temp_sqlite_db: Union[str, None]
        Optional file path to use for SQLite database used in Julia NemoMod 
        EnergyProduction model
        * If None, defaults to a temporary path sql database
    initialize_julia : bool
        Initialize julia? If False, only initializes non-julia EnergyProduction 
        methods and properties, which is often useful for accesing methods and 
        variables, but does not allow the model to run.
    logger : Union[logging.Logger, None]
        optional logging.Logger object used to log model events
    """
    def __init__(self,
        model_attributes: ModelAttributes,
        allow_electricity_run: bool = True,
        fp_julia: Union[str, None] = None,
        fp_nemomod_reference_files: Union[str, None] = None,
        fp_nemomod_temp_sqlite_db: Union[str, None] = None,
        initialize_julia: bool = True,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        # initialize input objects
        self._initialize_attributes(
            model_attributes,
            logger = logger,
        )

        # initialize sql path for electricity projection and path to electricity models
        self._initialize_path_nemomod_reference(
            allow_electricity_run, 
            fp_nemomod_reference_files,
        )
        self._initialize_path_nemomod_sql(fp_nemomod_temp_sqlite_db)
        # initialize last--depends on self.allow_electricity_run
        self._initialize_path_julia(fp_julia)

        # initialize models
        self._initialize_models(
            initialize_julia = initialize_julia,
        )

        # set the UUID
        self._initialize_uuid()

        return None




    ##############################################
    #	SUPPORT AND INITIALIZATION FUNCTIONS	#
    ##############################################

    def _initialize_attributes(self,
        model_attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        """Initialize key attributes for the model. Initializes the following 
            properties:

            * self.logger
            * self.model_attributes
            * self.time_periods
        """

        time_periods = sc.TimePeriods(model_attributes)

        self.logger = logger
        self.model_attributes = model_attributes
        self.time_periods = time_periods

        return None



    def _initialize_models(self,
        initialize_julia: bool = True,
    ) -> None:
        """Initialize the path to NemoMod reference files required for ingestion. 
            Initializes the following properties:

            * self.allow_electricity_run
            * self.fp_nemomod_reference_files
        """

        self.model_afolu = AFOLU(self.model_attributes)
        self.model_circecon = CircularEconomy(self.model_attributes)

        self.model_enerprod = None
        if self.allow_electricity_run:
            self.model_enerprod = EnergyProduction(
                self.model_attributes,
                self.fp_julia,
                self.fp_nemomod_reference_files,
                initialize_julia = initialize_julia,
                logger = self.logger,
            )

        self.model_enercons = EnergyConsumption(
            self.model_attributes,
            logger = self.logger
        )
        
        self.model_ippu = IPPU(self.model_attributes)
        self.model_socioeconomic = Socioeconomic(self.model_attributes)

        return None



    def _initialize_path_julia(self,
        fp_julia: Union[str, None]
    ) -> None:
        """Initialize the path to the NemoMod SQL database used to execute runs. 
            Initializes the following properties:

            * self.fp_julia

        NOTE: Will set `self.allow_electricity_run = False` if the path is 
            not found.
        """

        self.fp_julia = None
        if isinstance(fp_julia, str):
            if os.path.exists(fp_julia):
                self.fp_julia = fp_julia
                self._log(f"Set Julia directory for modules and environment to '{self.fp_julia}'.", type_log = "info")
            else:
                self.allow_electricity_run = False
                self._log(f"Invalid path '{fp_julia}' specified for Julia reference modules and environment: the path does not exist. Setting self.allow_electricity_run = False.", type_log = "error")

        return None



    def _initialize_path_nemomod_reference(self,
        allow_electricity_run: bool,
        fp_nemomod_reference_files: Union[str, None]
    ) -> None:
        """
        Initialize the path to NemoMod reference files required for ingestion. Initializes
            the following properties:

            * self.allow_electricity_run
            * self.fp_nemomod_reference_files

        Function Arguments
        ------------------
        - allow_electricity_run: exogenous specification of whether or not to allow the
            electricity model to run
        - fp_nemomod_reference_files: path to NemoMod reference files
        """

        self.allow_electricity_run = False
        self.fp_nemomod_reference_files = None

        try:
            self.fp_nemomod_reference_files = sf.check_path(fp_nemomod_reference_files, False)
            self.allow_electricity_run = allow_electricity_run

        except Exception as e:
            self._log(
                f"Path to NemoMod reference files '{fp_nemomod_reference_files}' not found. The Electricity model will be disallowed from running.", 
                type_log = "warning",
            )

        return None



    def _initialize_path_nemomod_sql(self,
        fp_nemomod_temp_sqlite_db: Union[str, None]
    ) -> None:
        """
        Initialize the path to the NemoMod SQL database used to execute runs. 
            Initializes the following properties:

            * self.fp_nemomod_temp_sqlite_db
        """

        valid_extensions = ["sqlite", "db"]

        # initialize as temporary
        fn_tmp = os.path.basename(tempfile.NamedTemporaryFile().name)
        fn_tmp = f"{fn_tmp}.sqlite"
        self.fp_nemomod_temp_sqlite_db = os.path.join(
            os.getcwd(),
            fn_tmp
        )

        if isinstance(fp_nemomod_temp_sqlite_db, str):
            try_endings = [fp_nemomod_temp_sqlite_db.endswith(x) for x in valid_extensions]

            if any(try_endings):
                self.fp_nemomod_temp_sqlite_db = fp_nemomod_temp_sqlite_db
                self._log(
                    f"Successfully initialized NemoMod temporary database path as {self.fp_nemomod_temp_sqlite_db}.", 
                    type_log = "info",
                    warn_if_none = False,
                )

            else:
                self._log(
                    f"Invalid path '{fp_nemomod_temp_sqlite_db}' specified as fp_nemomod_temp_sqlite_db. Using temporary path {self.fp_nemomod_temp_sqlite_db}.", 
                    type_log = "info",
                )


        # clear old temp database to prevent competing key information in sql schema
        os.remove(self.fp_nemomod_temp_sqlite_db) if os.path.exists(self.fp_nemomod_temp_sqlite_db) else None

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID
        """

        self.is_sisepuede_models = True
        self._uuid = _MODULE_UUID

        return None
        


    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """Clean implementation of sf._optional_log in-line using default 
            logger. See ?sf._optional_log for more information

        Function Arguments
        ------------------
        msg : str
            Message to log

        Keyword Arguments
        -----------------
        type_log : str
            Type of log to use
        **kwargs
            Passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def check_model_results(self,
        df_results: pd.DataFrame,
        verification_function: Callable,
        epsilon: float = 10**(-6),
        fields_check: Union[List[str], str, None] = None,
        fields_index: Union[List[str], None] = None,
        ignore_nas: bool = False,
        output_only: bool = True,
        thresholds: Tuple[float, float] = (10**(-5), 10**6),
    ) -> Union[bool, None]:
        """Verify numerical integrity of results by looking for fields that 
            include extreme outliers based on the skew function defined. 
        
        Returns:
            * True: if *no* columnar values of verification_function are outside
                acceptable bounds as defined by thresholds
            * False: if *any* columnar values of verification_function are 
                outside acceptable bounds as defined by thresholds
            * None: if any function input elements are misspecified
            
        Function Arguments
        ------------------
        df_results : pd.DataFrame
            DataFrame containing raw output results to verify
        verification_function : Callable
            Function that is applied along axis to verify values and compare 
            against thresholds
        
        Keyword Arguments
        ------------------
        epsilon : float
            Numerical value used to determine error in sf.vec_bounds comparison
        fields_check : Union[List[str], str, None]
            Optional specification of:
            * subset of fields to check (listlike)
            * "emissions_output" (to only check emissions output fields) 
                * NOT SUPPORTED AT MOMENT
            * "emissions_output_subsector_aggregate" (to only check subsector
                emission aggregate fields) 
            * None (to check all fields not associated with fields_ind)
            * NOTE: If any elements intersect with fields_ind, fields_ind takes 
                priority
        fields_index : Union[List[str], None]
            Fields to treat as index fields (exempt from checking). If None, 
            check every field in the data frame. If None, uses all indices 
        ignore_nas : bool
            Ignore any nas produced by verification function
        output_only : bool
            Check only output fields?
        thresholds : Tuple[float, float]
            Tuple specifying lower and upper limits of verification_function 
            value
        """
        
        # check fields and threshold specification
        return_none = not (thresholds[0] < thresholds[1])
        return_none |= not isinstance(df_results, pd.DataFrame)
        return_none |= not (
            (sf.islistlike(fields_index) or (fields_index is None))
            if not isinstance(fields_index, str)
            else (fields_index.lower() in ["emissions_output", "emissions_output_subsector_aggregate"])
        )
        return_none |= not isinstance(verification_function, Callable)
        if return_none:
            return None
        
        # set index and check fields; return True if no valid check fields are found
        fields_index = (
            [
                x for x in self.model_attributes.sort_ordered_dimensions_of_analysis
                if x in df_results.columns 
            ]
            if fields_index is None
            else fields_index
        )
        fields_index = [x for x in fields_index if (x in df_results.columns)]

        # set fields to check
        fields_check = (
            fields_check
            if not isinstance(fields_check, str)
            else (
                self.model_attributes.get_all_subsector_emission_total_fields()
                if fields_check == "emissions_output_subsector_aggregate"
                else None
            )
        )
        fields_check = (
            [x for x in df_results.columns if x not in fields_index]
            if not sf.islistlike(fields_check)
            else [x for x in fields_check if (x not in fields_index) and (x in df_results.columns)]
        )
        fields_check = (
            [x for x in fields_check if x in self.model_attributes.all_variable_fields_output]
            if output_only
            else fields_check
        )
        if len(fields_check) == 0:
            return True
        

        # apply verification function to columns, then determine if any values fall outside of specified thresholds
        arr_verify = np.array(df_results[fields_check])    
        out = np.abs(np.apply_along_axis(verification_function, 0, arr_verify))
        out = out[out != 0.0]
        
        # bound values
        out_compare = sf.vec_bounds(out, thresholds)
        
        vec_thresh_discrepancy = np.abs(out - out_compare)
        vec_thresh_discrepancy = (
            vec_thresh_discrepancy[~np.isnan(vec_thresh_discrepancy)]
            if ignore_nas
            else vec_thresh_discrepancy
        )
        
        check_val = (vec_thresh_discrepancy.max() <= epsilon)
        
        return check_val



    def project(self,
        df_input_data: pd.DataFrame,
        check_results: bool = True,
        fields_check: Union[List[str], str, None] = "emissions_output_subsector_aggregate",
        include_electricity_in_energy: bool = True,
        models_run: Union[List[str], None] = None,
        regions: Union[List[str], str, None] = None,
        run_integrated: bool = True,
        time_periods_run: Union[List[int], None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Execute the SISEPUEDE DAG.

        Function Arguments
        ------------------
        df_input_data: DataFrame containing SISEPUEDE inputs

        Optional Arguments
        ------------------
        - models_run: list of sector models to run as defined in
            SISEPUEDEModels.model_attributes. Can include the following values:

            * AFOLU (or af)
            * Circular Economy (or ce)
            * IPPU (or ip)
            * Energy (or en)
                * Note: set include_electricity_in_energy = False to avoid
                    running the electricity model with energy
            * Socioeconomic (or se)

        Keyword Arguments
        -----------------
        - check_results: verify output results using a verification function
            (see SISEPUEDEModels.check_model_results())
        - fields_check: passed to self.check_model_results() (only applicable if 
            check_results = True). Valid options are:
            * subset of fields to check (listlike)
            * "emissions_output" (to only check emissions output fields) 
            * "emissions_output_subsector_aggregate" (to only check subsector
                emission aggregate fields) 
            * None (to check all fields not associated with fields_ind)
            * NOTE: If any elements intersect with fields_ind, fields_ind takes 
                priority
        - include_electricity_in_energy: include the electricity model in runs
            of the energy model?
            * If False, runs without electricity (time intensive model)
        - regions: regions to run the model for (NEEDS ADDITIONAL WORK IN 
            NON-ELECTRICITY SECTORS)
        - run_integrated: run models as integrated collection?
            * If False, will run each model individually, without interactions
                (not recommended)
        - time_periods_run: optional specification of time periods to run
        - **kwargs: passed to SISEPUEDEModels.check_model_results()
        """

        df_return = []
        models_run = self.model_attributes.get_sector_list_from_projection_input(models_run)
        regions = self.model_attributes.get_region_list_filtered(regions)
        

        # check time periods
        time_periods_run = (
            [x for x in time_periods_run if x in self.time_periods.all_time_periods]
            if sf.islistlike(time_periods_run)
            else None
        )
        if time_periods_run is not None:
            time_periods_run = None if (len(time_periods_run) == 0) else time_periods_run

        df_input_data = (
            (
                df_input_data[
                    df_input_data[self.model_attributes.dim_time_period].isin(time_periods_run)
                ]
                .reset_index(drop = True)
            )
            if time_periods_run is not None
            else df_input_data
        )

        
        ##  1. Run AFOLU and collect output

        if "AFOLU" in models_run:
            self._log("Running AFOLU model", type_log = "info")
            try:
                df_return.append(self.model_afolu.project(df_input_data))
                self._log(f"AFOLU model run successfully completed", type_log = "info")

            except Exception as e:
                self._log(f"Error running AFOLU model: {e}", type_log = "error")


        ##  2. Run CircularEconomy and collect output - requires AFOLU to run integrated

        if "Circular Economy" in models_run:
            self._log("Running CircularEconomy model", type_log = "info")
            if run_integrated and set(["AFOLU"]).issubset(set(models_run)):
                df_input_data = self.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_return[0],
                    self.model_circecon.integration_variables
                )

            try:
                df_return.append(self.model_circecon.project(df_input_data))
                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )
                self._log(f"CircularEconomy model run successfully completed", type_log = "info")

            except Exception as e:
                self._log(f"Error running CircularEconomy model: {e}", type_log = "error")


        ##  3. Run IPPU and collect output

        if "IPPU" in models_run:
            self._log("Running IPPU model", type_log = "info")
            if run_integrated and set(["Circular Economy"]).issubset(set(models_run)):
                df_input_data = self.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_return[0],
                    self.model_ippu.integration_variables
                )

            try:
                df_return.append(self.model_ippu.project(df_input_data))
                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )
                self._log(f"IPPU model run successfully completed", type_log = "info")

            except Exception as e:
                self._log(f"Error running IPPU model: {e}", type_log = "error")


        ##  4. Run Non-Electric Energy (excluding Fugitive Emissions) and collect output

        if "Energy" in models_run:

            self._log(
                "Running Energy model (EnergyConsumption without Fugitive Emissions)", 
                type_log = "info",
            )

            if run_integrated and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
                df_input_data = self.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_return[0],
                    self.model_enercons.integration_variables_non_fgtv
                )

            try:
                df_return.append(self.model_enercons.project(df_input_data))
                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )
                self._log(
                    f"EnergyConsumption without Fugitive Emissions model run successfully completed", type_log = "info",
                )

            except Exception as e:
                self._log(
                    f"Error running EnergyConsumption without Fugitive Emissions: {e}",
                    type_log = "error",
                )
        

        ##  5. Run Electricity and collect output

        if ("Energy" in models_run) and include_electricity_in_energy and self.allow_electricity_run:

            self._log(
                "Running Energy model (Electricity and Fuel Production: trying to call Julia)", 
                type_log = "info",
            )

            if run_integrated and set(["Circular Economy", "AFOLU"]).issubset(set(models_run)):
                df_input_data = self.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_return[0],
                    self.model_enerprod.integration_variables
                )

            # create the engine and try to run Electricity
            engine = sqlalchemy.create_engine(f"sqlite:///{self.fp_nemomod_temp_sqlite_db}")
            try:
                df_elec = self.model_enerprod.project(
                    df_input_data, 
                    engine,
                    regions = regions
                )
                df_return.append(df_elec)
                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )

                self._log(
                    f"EnergyProduction model run successfully completed", 
                    type_log = "info",
                )

            except Exception as e:
                self._log(
                    f"Error running EnergyProduction model: {e}", 
                    type_log = "error",
                )

        
        ##  6. Add fugitive emissions from Non-Electric Energy and collect output

        if "Energy" in models_run:
            self._log(
                "Running Energy (Fugitive Emissions)", 
                type_log = "info",
            )

            if run_integrated and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
                df_input_data = self.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_return[0],
                    self.model_enercons.integration_variables_fgtv
                )

            try:
                df_return.append(
                    self.model_enercons.project(
                        df_input_data, 
                        subsectors_project = self.model_attributes.subsec_name_fgtv
                    )
                )

                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )

                self._log(
                    f"Fugitive Emissions from Energy model run successfully completed", 
                    type_log = "info",
                )

            except Exception as e:
                self._log(
                    f"Error running Fugitive Emissions from Energy model: {e}", 
                    type_log = "error",
                )

        
        ##  7. Add Socioeconomic output at the end to avoid double-initiation throughout models

        if len(df_return) > 0:
            self._log("Appending Socioeconomic outputs", type_log = "info")

            try:
                df_return.append(
                    self.model_socioeconomic.project(
                        df_input_data, 
                        project_for_internal = False
                    )
                )

                df_return = (
                    [sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
                    if run_integrated 
                    else df_return
                )

                self._log(
                    f"Socioeconomic outputs successfully appended.", 
                    type_log = "info",
                )

            except Exception as e:
                self._log(
                    f"Error appending Socioeconomic outputs: {e}", 
                    type_log = "error",
                )

        
        # build output data frame
        df_return = (
            sf.merge_output_df_list(
                df_return, 
                self.model_attributes, 
                merge_type = "concatenate"
            ) 
            if (len(df_return) > 0) 
            else pd.DataFrame()
        )

        if check_results:
            return_df = self.check_model_results(
                df_return,
                sf.mean_median_ratio,
                fields_check = fields_check,
                **kwargs,
            )
            df_return = (
                df_return
                if return_df
                else None
            )

        return df_return
    



###################################
#    SIMPLE CHECKING FUNCTIONS    #
###################################

def is_sisepuede_models(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDEModels object
    """

    out = hasattr(obj, "is_sisepuede_models")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out
