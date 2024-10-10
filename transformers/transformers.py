
import datetime as dt
import logging
import numpy as np
import os, os.path
import pandas as pd
import re
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
import sisepuede.utilities._toolbox as sf




_MODULE_CODE_SIGNATURE = "TFR"
_MODULE_UUID = "D3BC5456-5BB7-4F7A-8799-AFE0A44C3FFA" 


_DICT_KEYS = {
    "baseline": "baseline",
    "general": "general",
    "vec_implementation_ramp": "vec_implementation_ramp",
}



###############################################
#    SET SOME DEFAULT CONFIGURATION VALUES    #
###############################################

def get_dict_config_default(
    key_baseline: str = _DICT_KEYS.get("baseline"),
    key_general: str = _DICT_KEYS.get("general"),
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
            # - If None, no technologies investments are capped
            #
            "categories_entc_max_investment_ramp": None,

            # Power plant categories to cap in MSP ramp
            "categories_entc_pps_to_cap": None,

            # ENTC categories considered renewable sources--defaults to attribute table specs if not defined
            # - If None, defaults to attribute tables
            # "categories_entc_renewable": []
            "categories_entc_renewable": None,
            
            # INEN categories that have high heat and associated fractions (temporary until model can be revised)
            "categories_inen_high_heat_to_frac": {
                "cement": 0.88,
                "chemicals": 0.5, 
                "glass": 0.88, 
                "lime_and_carbonite": 0.88, 
                "metals": 0.92,
                "paper": 0.18, 
            },

            # Target minimum share of production fractions for power plants in the renewable target tranformation
            #"dict_entc_renewable_target_msp": {
            #    "pp_solar": 0.15,
            #    "pp_geothermal": 0.1,
            #    "pp_wind": 0.15
            #},

            # fraction of high heat that can be converted to electricity or hydrogen;
            # assume that 50% is electrified, 50% is hydrogenized
            "frac_inen_high_temp_elec_hydg": 0.45,

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





################################
#    START WITH TRANSFORMER    #
################################

class Transformer:
    """
    Create a Transformation class to support construction in sectoral 
        transformations. 

    Initialization Arguments
    ------------------------
    - code: transformer code used to map the transformer to the attribute table. 
        Must be defined in attr_transfomers.table[attr_transfomers.key]
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    - attr_transformers: AttributeTable usd to define transformers from 
        ModelAttributes

    Keyword Arguments
    -----------------
    - code_baseline: transformer code that stores the baseline code, which is 
        applied to raw data.
    - field_transformer_id: field in attr_transfomer.table containing the
        transformer index
    - field_transformer_name: field in attr_transfomer.table containing the
        transformer name
    - overwrite_docstr: overwrite the docstring if there's only one function?
    """
    
    def __init__(self,
        code: str,
        func: Callable,
        attr_transfomer: Union[AttributeTable, None],
        code_baseline: str = f"{_MODULE_CODE_SIGNATURE}:BASE",
        overwrite_docstr: bool = True,
        **kwargs,
    ) -> None:

        self._initialize_function(
            func, 
            overwrite_docstr,
        )

        self._initialize_fields(
            **kwargs,
        )

        self._initialize_code(
            code, 
            code_baseline,
            attr_transfomer, 
        )

        self._initialize_uuid()

        return None
        

    
    def __call__(self,
        *args,
        **kwargs
    ) -> Any:
        
        val = self.function(
            *args,
            # strat = self.id,
            **kwargs
        )

        return val



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_code(self,
        code: str,
        code_baseline: str,
        attr_transfomer: Union[AttributeTable, None],
    ) -> None:
        """
        Initialize transfomer identifiers, including the code (key), name, and
            ID. Sets the following properties:

            * self.baseline
            * self.code
            * self.id
            * self.name
        """
        
        # check code
        if code not in attr_transfomer.key_values:
            raise KeyError(f"Invalid Transformer code '{code}': code not found in attribute table.")

        # initialize and check code/id num
        id_num = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{self.field_transformer_id}")
            if attr_transfomer is not None
            else None
        )
        id_num = id_num.get(code) if (id_num is not None) else -1


        # initialize and check name/id num
        name = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{self.field_transformer_name}")
            if attr_transfomer is not None
            else None
        )
        name = name.get(code) if (name is not None) else ""

        # check baseline
        baseline = (code == code_baseline)


        ##  TRY GETTING OPTIONAL PROPERTIES

        # initialize and check citations
        citations = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{self.field_citations}")
            if attr_transfomer is not None
            else None
        )
        citations = citations.get(code) if citations is not None else None
        citations = None if sf.isnumber(citations, skip_nan = False) else citations


        # initialize and check description
        description = (
            attr_transfomer
            .field_maps
            .get(f"{attr_transfomer.key}_to_{self.field_description}")
            if attr_transfomer is not None
            else None
        )
        description = description.get(code) if description is not None else None
        description = None if sf.isnumber(description, skip_nan = False) else description


        ##  SET PROPERTIES

        self.baseline = bool(baseline)
        self.citations = citations
        self.code = str(code)
        self.code_baseline = code_baseline
        self.description = description
        self.id = int(id_num)
        self.name = str(name)
        
        return None

    
    
    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
        overwrite_docstr: bool = True,
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
            * self.function_list (list of callables, even if one callable is 
                passed. Allows for quick sharing across classes)
        """
        
        function = None

        if isinstance(func, list):

            func = [x for x in func if callable(x)]

            if len(func) > 0:  
                
                overwrite_docstr &= (len(func) == 1)

                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformer function for {self.name}
                    """
                    out = None
                    if len(args) > 0:
                        out = (
                            args[0].copy() 
                            if isinstance(args[0], pd.DataFrame) | isinstance(args[0], np.ndarray)
                            else args[0]
                        )

                    for f in func:
                        out = f(out, **kwargs)

                    return out

                function = function_out
                function_list = func
            
            else:
                overwrite_docstr = False

        elif callable(func):
            function = func
            function_list = [func]

        
        # overwrite doc?
        if overwrite_docstr:
            self.__doc__ = function_list[0].__doc__ 

        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function
        self.function_list = function_list
        
        return None
    


    def _initialize_fields(self,
        **kwargs,
    ) -> None:
        """
        Set the optional and required keys used to specify a transformation.
            Can use keyword arguments to set keys.
        """

        # set some shortcut codes 

        field_citations = kwargs.get("field_citations", "citations")
        field_description = kwargs.get("field_description", "description")
        field_transformer_id = kwargs.get("field_transformer_id", "transformer_id")
        field_transformer_name = kwargs.get("field_transformer_name", "transformer")
        

        ##  SET PARAMETERS

        self.field_citations = field_citations
        self.field_description = field_description
        self.field_transformer_id = field_transformer_id
        self.field_transformer_name = field_transformer_name

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Sets the following other properties:

            * self.is_transformer
            * self._uuid
        """

        self.is_transformer = True
        self._uuid = _MODULE_UUID

        return None






####################################
#    COLLECTION OF TRANSFORMERS    #
####################################

class Transformers:
    """
    Build collection of Transformers that are used to define transformations.

    Includes some information on

    Initialization Arguments
    ------------------------
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
    - regex_code_structure: regular expression defining the code structure of
        transformer codes
    - regex_template_prepend: passed to SISEPUEDEFileStructure()
    - **kwargs 
    """
    
    def __init__(self,
        dict_config: Dict,
        code_baseline: str = f"{_MODULE_CODE_SIGNATURE}:BASE",
        df_input: Union[pd.DataFrame, None] = None,
        field_region: Union[str, None] = None,
        logger: Union[logging.Logger, None] = None,
        regex_code_structure: re.Pattern = re.compile(f"{_MODULE_CODE_SIGNATURE}:(\D*):(.*$)"),
        regex_template_prepend: str = "sisepuede_run",
        **kwargs
    ):

        self.logger = logger

        self._initialize_file_structure(
            regex_template_prepend = regex_template_prepend, 
        )
        self._initialize_models(**kwargs)
        self._initialize_attributes(
            field_region,
            regex_code_structure,
        )

        self._initialize_config(
            dict_config, 
            code_baseline,
        )
        self._initialize_parameters()
        self._initialize_ramp()

        # set transformations by sector, models (which come from sectoral transformations)
        self._initialize_baseline_inputs(df_input, )
        self._initialize_transformers()
        self._initialize_uuid()

        return None




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_attributes(self,
        field_region: Union[str, None],
        regex_code_structure: re.Pattern,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_transformer_code
            * self.key_region
            * self.regex_code_structure
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

        # set the structure of the transformer codes
        regex_code_structure = (
            re.compile(f"{_MODULE_CODE_SIGNATURE}:(\D*):(.*$)")
            if not isinstance(regex_code_structure, re.Pattern)
            else regex_code_structure
        )

        # set some useful classes
        time_periods = sc.TimePeriods(self.model_attributes)
        regions_manager = sc.Regions(self.model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_technology = attribute_technology
        self.attribute_transformer_code = attribute_transformer_code
        self.key_region = field_region
        self.key_time_period = time_periods.field_time_period
        self.key_transformer_code = attribute_transformer_code.key
        self.regex_code_structure = regex_code_structure
        self.regions_manager = regions_manager
        self.time_periods = time_periods

        return None



    def _initialize_baseline_inputs(self,
        df_inputs: Union[pd.DataFrame, None],
    ) -> None:
        """
        Initialize the baseline inputs dataframe based on the initialization 
            value of df_inputs. It not initialied, sets as None. Sets the 
            following properties:

            * self.baseline_inputs
            * self.inputs_raw
            * self.regions

        """

        # initialize
        baseline_inputs = None
        inputs_raw = None
        regions = None

        if isinstance(df_inputs, pd.DataFrame):
            # verify that certain keys are included
            sf.check_fields(
                df_inputs,
                [
                    self.key_region,
                    self.key_time_period
                ],
                msg_prepend = "Fields required in input data frame used to initialize Transformers()"
            )

            
            # get regions
            regions = [
                x for x in self.regions_manager.all_regions
                if x in df_inputs[self.key_region].unique()
            ]
            
            if len(regions) == 0:
                raise RuntimeError(f"No valid regions found in input data frame `df_input`.")

            # build baseline inputs
            baseline_inputs = self._trfunc_baseline(
                df_inputs[
                    df_inputs[self.key_region]
                    .isin(regions)
                ]
                .reset_index(drop = True, ), 
                strat = None,
            ) 

        
        ##  SET PROPERTIES
        
        self.baseline_inputs = baseline_inputs
        self.inputs_raw = df_inputs
        self.regions = regions

        return None



    def _initialize_config(self,
        dict_config: Union[Dict[str, Any], None],
        code_baseline: str,
        key_baseline: str = _DICT_KEYS.get("baseline"),
        key_general: str = _DICT_KEYS.get("general"),
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

        ##  UPDATE CONFIG

        # start with default and overwrite as necessary
        config = get_dict_config_default(
            key_baseline = key_baseline,
            key_general = key_general,
        )

        if isinstance(dict_config, dict):
            config.dict_yaml.update(dict_config)
        

        ##  SET PARAMETERS

        self.code_baseline = code_baseline
        self.config = config
        self.key_config_baseline = key_baseline
        self.key_config_cats_entc_max_investment_ramp = "categories_entc_max_investment_ramp"
        self.key_config_cats_entc_pps_to_cap = "categories_entc_pps_to_cap"
        self.key_config_cats_entc_renewable = "categories_entc_renewable"
        self.key_config_dict_cats_inen_high_heat_to_frac = "categories_inen_high_heat_to_frac"
        self.key_config_frac_inen_high_temp_elec_hydg = "frac_inen_low_temp_elec"
        self.key_config_frac_inen_low_temp_elec = "frac_inen_low_temp_elec"
        self.key_config_general = key_general
        self.key_config_magnitude_lurf = "magnitude_lurf" # MUST be same as kwarg in _trfunc_baseline
        self.key_config_n_tp_ramp = "n_tp_ramp"
        self.key_config_tp_0_ramp = "tp_0_ramp" 
        self.key_config_vec_implementation_ramp = "vec_implementation_ramp"
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
            self._log(
                f"Successfully initialized SISEPUEDEFileStructure.", 
                type_log = "info",
                warn_if_none = False,
            )

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

        # check for specification of a default run db
        fp_nemomod_temp_sqlite_db = kwargs.get(
            "fp_nemomod_temp_sqlite_db",
            self.file_struct.fp_sqlite_tmp_nemomod_intermediate,
        )

        models = sm.SISEPUEDEModels(
            self.model_attributes,
            allow_electricity_run = True,
            fp_julia = self.file_struct.dir_jl,
            fp_nemomod_reference_files = self.file_struct.dir_ref_nemo,
            fp_nemomod_temp_sqlite_db = fp_nemomod_temp_sqlite_db,
            initialize_julia = False,
            logger = self.logger,
        )


        ##  CHECK IF ANY MODELS ARE SPECIFIED IN KEYWORD ARGUMENTS
        
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

        # init some params
        n_tp_ramp = None
        tp_0_ramp = None
        alpha_logistic = None
        d = None
        window_logistic = None


        ##  TRY GETTING DETAILS FROM CONFIG

        vec_implementation_ramp = self.config.get(
            f"{self.key_config_general}.{self.key_config_vec_implementation_ramp}"
        )

        if isinstance(vec_implementation_ramp, dict):

            n_tp_ramp = vec_implementation_ramp.get("n_tp_ramp")
            tp_0_ramp = vec_implementation_ramp.get("tp_0_ramp")
            alpha_logistic = vec_implementation_ramp.get("alpha_logistic")
            d = vec_implementation_ramp.get("d")
            window_logistic = vec_implementation_ramp.get("window_logistic")
            
            # drive to default if invalid
            if sf.islistlike(window_logistic):
                if len(window_logistic) >= 2:
                    window_logistic = tuple(window_logistic[0:2])
                
                else:
                    window_logistic = None
                    self._log(
                        "Invalid specification of window_logistic in configuration file. Setting to default...",
                        type_log = "warning",
                    )
        

        # build the vector
        vec_implementation_ramp = self.build_implementation_ramp_vector(
            n_tp_ramp = n_tp_ramp,
            tp_0_ramp = tp_0_ramp,
            alpha_logistic = alpha_logistic,
            d = d,
            window_logistic = window_logistic,
        )
        

        ##  SET PROPERTIES

        self.vec_implementation_ramp = vec_implementation_ramp

        return None



    def _initialize_transformers(self,
    ) -> None:
        """
        Initialize all Transformer objects used to build transformations.

     
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

        self.baseline = Transformer(
            self.code_baseline, 
            self._trfunc_baseline_return, 
            attr_transformer_code
        )
        all_transformers.append(self.baseline)


        ###############
        #    AFOLU    #
        ###############

        ##  AGRC TRANSFORMERS

        self.agrc_improve_rice_management = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:AGRC:DEC_CH4_RICE", 
            self._trfunc_agrc_improve_rice_management,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_improve_rice_management)


        self.agrc_decrease_exports = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:AGRC:DEC_EXPORTS", 
            self._trfunc_agrc_decrease_exports,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_decrease_exports)


        self.agrc_expand_conservation_agriculture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:AGRC:INC_CONSERVATION_AGRICULTURE", 
            self._trfunc_agrc_expand_conservation_agriculture,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_expand_conservation_agriculture)


        self.agrc_increase_crop_productivity = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:AGRC:INC_PRODUCTIVITY", 
            self._trfunc_agrc_increase_crop_productivity,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_increase_crop_productivity)


        self.agrc_reduce_supply_chain_losses = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:AGRC:DEC_LOSSES_SUPPLY_CHAIN", 
            self._trfunc_agrc_reduce_supply_chain_losses,
            attr_transformer_code
        )
        all_transformers.append(self.agrc_reduce_supply_chain_losses)


        ##  FRST TRANSFORMERS

        
        ##  LNDU TRANSFORMERS

        self.lndu_expand_silvopasture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LNDU:INC_SILVOPASTURE", 
            self._trfunc_lndu_expand_silvopasture,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_expand_silvopasture)


        self.lndu_expand_sustainable_grazing = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LNDU:DEC_SOC_LOSS_PASTURES", 
            self._trfunc_lndu_expand_sustainable_grazing,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_expand_sustainable_grazing)


        self.lndu_increase_reforestation = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LNDU:INC_REFORESTATION", 
            self._trfunc_lndu_increase_reforestation,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_increase_reforestation)


        self.lndu_partial_reallocation = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LNDU:PLUR", 
            self._trfunc_lndu_reallocate_land,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_partial_reallocation)


        self.lndu_stop_deforestation = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LNDU:DEC_DEFORESTATION", 
            self._trfunc_lndu_stop_deforestation,
            attr_transformer_code
        )
        all_transformers.append(self.lndu_stop_deforestation)


        ##  LSMM TRANSFORMATIONS

        self.lsmm_improve_manure_management_cattle_pigs = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LSMM:INC_MANAGEMENT_CATTLE_PIGS", 
            self._trfunc_lsmm_improve_manure_management_cattle_pigs,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_cattle_pigs)


        self.lsmm_improve_manure_management_other = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LSMM:INC_MANAGEMENT_OTHER", 
            self._trfunc_lsmm_improve_manure_management_other,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_other)
        

        self.lsmm_improve_manure_management_poultry = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LSMM:INC_MANAGEMENT_POULTRY", 
            self._trfunc_lsmm_improve_manure_management_poultry,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_improve_manure_management_poultry)


        self.lsmm_increase_biogas_capture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LSMM:INC_CAPTURE_BIOGAS", 
            self._trfunc_lsmm_increase_biogas_capture,
            attr_transformer_code
        )
        all_transformers.append(self.lsmm_increase_biogas_capture)

        
        ##  LVST TRANSFORMERS
      
        self.lvst_decrease_exports = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LVST:DEC_EXPORTS", 
            self._trfunc_lvst_decrease_exports,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_decrease_exports)


        self.lvst_increase_productivity = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LVST:INC_PRODUCTIVITY", 
            self._trfunc_lvst_increase_productivity,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_increase_productivity)


        self.lvst_reduce_enteric_fermentation = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:LVST:DEC_ENTERIC_FERMENTATION", 
            self._trfunc_lvst_reduce_enteric_fermentation,
            attr_transformer_code
        )
        all_transformers.append(self.lvst_reduce_enteric_fermentation)
        

        ##  SOIL TRANSFORMERS
        
        self.soil_reduce_excess_fertilizer = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:SOIL:DEC_N_APPLIED", 
            self._trfunc_soil_reduce_excess_fertilizer,
            attr_transformer_code
        )
        all_transformers.append(self.soil_reduce_excess_fertilizer)


        self.soil_reduce_excess_liming = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:SOIL:DEC_LIME_APPLIED", 
            self._trfunc_soil_reduce_excess_lime,
            attr_transformer_code
        )
        all_transformers.append(self.soil_reduce_excess_liming)


        #######################################
        #    CIRCULAR ECONOMY TRANSFORMERS    #
        #######################################

        ##  TRWW TRANSFORMERS

        self.trww_increase_biogas_capture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRWW:INC_CAPTURE_BIOGAS", 
            self._trfunc_trww_increase_biogas_capture,
            attr_transformer_code
        )
        all_transformers.append(self.trww_increase_biogas_capture)


        self.trww_increase_septic_compliance = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRWW:INC_COMPLIANCE_SEPTIC", 
            self._trfunc_trww_increase_septic_compliance,
            attr_transformer_code
        )
        all_transformers.append(self.trww_increase_septic_compliance)


        ##  WALI TRANSFORMERS
 
        self.wali_improve_sanitation_industrial = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WALI:INC_TREATMENT_INDUSTRIAL", 
            self._trfunc_wali_improve_sanitation_industrial,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_industrial)


        self.wali_improve_sanitation_rural = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WALI:INC_TREATMENT_RURAL", 
            self._trfunc_wali_improve_sanitation_rural,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_rural)


        self.wali_improve_sanitation_urban = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WALI:INC_TREATMENT_URBAN", 
            self._trfunc_wali_improve_sanitation_urban,
            attr_transformer_code
        )
        all_transformers.append(self.wali_improve_sanitation_urban)


        ##  WASO TRANSFORMERS

        self.waso_descrease_consumer_food_waste = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:DEC_CONSUMER_FOOD_WASTE",
            self._trfunc_waso_decrease_food_waste, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_descrease_consumer_food_waste)

        
        self.waso_increase_anaerobic_treatment_and_composting = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_ANAEROBIC_AND_COMPOST", 
            self._trfunc_waso_increase_anaerobic_treatment_and_composting, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_anaerobic_treatment_and_composting)


        self.waso_increase_biogas_capture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_CAPTURE_BIOGAS", 
            self._trfunc_waso_increase_biogas_capture, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_biogas_capture)


        self.waso_energy_from_biogas = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_ENERGY_FROM_BIOGAS", 
            self._trfunc_waso_increase_energy_from_biogas, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_energy_from_biogas)


        self.waso_energy_from_incineration = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_ENERGY_FROM_INCINERATION", 
            self._trfunc_waso_increase_energy_from_incineration, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_energy_from_incineration)


        self.waso_increase_landfilling = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_LANDFILLING", 
            self._trfunc_waso_increase_landfilling, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_landfilling)

        
        self.waso_increase_recycling = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:WASO:INC_RECYCLING", 
            self._trfunc_waso_increase_recycling, 
            attr_transformer_code
        )
        all_transformers.append(self.waso_increase_recycling)


        #############################
        #    ENERGY TRANSFORMERS    #
        #############################

        ##  CCSQ

        self.ccsq_increase_air_capture = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:CCSQ:INC_CAPTURE", 
            self._trfunc_ccsq_increase_air_capture, 
            attr_transformer_code
        )
        all_transformers.append(self.ccsq_increase_air_capture)


        ##  ENTC

        self.entc_clean_hydrogen = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:ENTC:TARGET_CLEAN_HYDROGEN", 
            self._trfunc_entc_clean_hydrogen, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_clean_hydrogen)


        self.entc_least_cost = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:ENTC:LEAST_COST_SOLUTION", 
            self._trfunc_entc_least_cost, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_least_cost)

        
        self.entc_reduce_transmission_losses = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:ENTC:DEC_LOSSES", 
            self._trfunc_entc_reduce_transmission_losses, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_reduce_transmission_losses)


        self.entc_renewable_electricity = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:ENTC:TARGET_RENEWABLE_ELEC", 
            self._trfunc_entc_renewables_target, 
            attr_transformer_code
        )
        all_transformers.append(self.entc_renewable_electricity)


        ##  FGTV

        self.fgtv_maximize_flaring = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:FGTV:INC_FLARE", 
            self._trfunc_fgtv_maximize_flaring, 
            attr_transformer_code
        )
        all_transformers.append(self.fgtv_maximize_flaring)

        self.fgtv_minimize_leaks = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:FGTV:DEC_LEAKS", 
            self._trfunc_fgtv_minimize_leaks, 
            attr_transformer_code
        )
        all_transformers.append(self.fgtv_minimize_leaks)


        ##  INEN

        self.inen_fuel_switch_heat = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:INEN:SHIFT_FUEL_HEAT", 
            self._trfunc_inen_fuel_switch_low_and_high_temp,
            attr_transformer_code
        )
        all_transformers.append(self.inen_fuel_switch_heat)

        
        self.inen_maximize_energy_efficiency = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:INEN:INC_EFFICIENCY_ENERGY", 
            self._trfunc_inen_maximize_efficiency_energy, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_maximize_energy_efficiency)


        self.inen_maximize_production_efficiency = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:INEN:INC_EFFICIENCY_PRODUCTION", 
            self._trfunc_inen_maximize_efficiency_production, 
            attr_transformer_code
        )
        all_transformers.append(self.inen_maximize_production_efficiency)


        ##  SCOE

        self.scoe_fuel_switch_electrify = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:SCOE:SHIFT_FUEL_HEAT", 
            self._trfunc_scoe_fuel_switch_electrify, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_fuel_switch_electrify)


        self.scoe_increase_applicance_efficiency = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:SCOE:INC_EFFICIENCY_APPLIANCE", 
            self._trfunc_scoe_increase_applicance_efficiency, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_increase_applicance_efficiency)


        self.scoe_reduce_heat_energy_demand = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:SCOE:DEC_DEMAND_HEAT", 
            self._trfunc_scoe_reduce_heat_energy_demand, 
            attr_transformer_code
        )
        all_transformers.append(self.scoe_reduce_heat_energy_demand)


        ###################
        #    TRNS/TRDE    #
        ###################

        self.trde_reduce_demand = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRDE:DEC_DEMAND", 
            self._trfunc_trde_reduce_demand, 
            attr_transformer_code
        )
        all_transformers.append(self.trde_reduce_demand)

        
        self.trns_electrify_light_duty_road = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_FUEL_LIGHT_DUTY", 
            self._trfunc_trns_electrify_road_light_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_electrify_light_duty_road)

        
        self.trns_electrify_rail = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_FUEL_RAIL", 
            self._trfunc_trns_electrify_rail, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_electrify_rail)

        
        self.trns_fuel_switch_maritime = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_FUEL_MARITIME", 
            self._trfunc_trns_fuel_switch_maritime, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_fuel_switch_maritime)


        self.trns_fuel_switch_medium_duty_road = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_FUEL_MEDIUM_DUTY", 
            self._trfunc_trns_fuel_switch_road_medium_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_fuel_switch_medium_duty_road)


        self.trns_increase_efficiency_electric = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:INC_EFFICIENCY_ELECTRIC", 
            self._trfunc_trns_increase_efficiency_electric,
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_efficiency_electric)


        self.trns_increase_efficiency_non_electric = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:INC_EFFICIENCY_NON_ELECTRIC", 
            self._trfunc_trns_increase_efficiency_non_electric,
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_efficiency_non_electric)


        self.trns_increase_occupancy_light_duty = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:INC_OCCUPANCY_LIGHT_DUTY", 
            self._trfunc_trns_increase_occupancy_light_duty, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_increase_occupancy_light_duty)


        self.trns_mode_shift_freight = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_MODE_FREIGHT", 
            self._trfunc_trns_mode_shift_freight, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_freight)


        self.trns_mode_shift_public_private = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_MODE_PASSENGER", 
            self._trfunc_trns_mode_shift_public_private, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_public_private)


        self.trns_mode_shift_regional = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:TRNS:SHIFT_MODE_REGIONAL", 
            self._trfunc_trns_mode_shift_regional, 
            attr_transformer_code
        )
        all_transformers.append(self.trns_mode_shift_regional)


        ###########################
        #    IPPU TRANSFORMERS    #
        ###########################

        self.ippu_demand_managment = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_DEMAND", 
            self._trfunc_ippu_reduce_demand,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_demand_managment)


        self.ippu_reduce_cement_clinker = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_CLINKER", 
            self._trfunc_ippu_reduce_cement_clinker,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_reduce_cement_clinker)


        self.ippu_reduce_hfcs = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_HFCS", 
            self._trfunc_ippu_reduce_hfcs,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_reduce_hfcs)


        self.ippu_reduce_other_fcs = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_OTHER_FCS", 
            self._trfunc_ippu_reduce_other_fcs,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_reduce_other_fcs)


        self.ippu_reduce_n2o = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_N2O", 
            self._trfunc_ippu_reduce_n2o,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_reduce_n2o)


        self.ippu_reduce_pfcs = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:IPPU:DEC_PFCS", 
            self._trfunc_ippu_reduce_pfcs,
            attr_transformer_code
        )
        all_transformers.append(self.ippu_reduce_pfcs)


        ######################################
        #    CROSS-SECTOR TRANSFORMATIONS    #
        ######################################

        self.plfo_healthier_diets = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:PFLO:INC_HEALTHIER_DIETS", 
            self._trfunc_pflo_healthier_diets, 
            attr_transformer_code
        )
        all_transformers.append(self.plfo_healthier_diets)



        self.pflo_industrial_ccs = Transformer(
            f"{_MODULE_CODE_SIGNATURE}:PFLO:INC_IND_CCS", 
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

        # get some other properties
        tr_tmp = dict_transformers.get(all_transformers[0])
        field_transformer_id = tr_tmp.field_transformer_id
        field_transformer_name = tr_tmp.field_transformer_name


        ##  SET ADDDITIONAL PROPERTIES

        self.all_transformers = all_transformers
        self.all_transformers_non_baseline = all_transformers_non_baseline
        self.dict_transformers = dict_transformers
        self.field_transformer_id = field_transformer_id
        self.field_transformer_name = field_transformer_name
        self.transformer_id_baseline = transformer_id_baseline

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the following properties:
        
            * self.is_transformers
            * self._uuid
        """

        self.is_transformers = True
        self._uuid = _MODULE_UUID
        
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
            if not sf.isnumber(magnitude) 
            else max(min(float(magnitude), bounds[1]), bounds[0])
        )

        return out



    def build_implementation_ramp_vector(self,
        alpha_logistic: Union[float, None] = 0.0,
        d: Union[float, int, None] = 0,
        n_tp_ramp: Union[int, None] = None,
        tp_0_ramp: Union[int, None] = None,
        window_logistic: Union[tuple, None] = (-8, 8),
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
        - alpha_logistic: fraction of the ramp that is logistic 
            (1 - alpha_logistic is linear)
        - d: centroid for the logit 
        - window_logistic: window for the standard logit function
        **kwargs: passed to sisepuede.utilities._toolbox.ramp_vector()
        """
        
        # some init
        n_tp = len(self.time_periods.all_time_periods)

        # verify the values
        n_tp_ramp, tp_0_ramp, _, _ = self.get_ramp_characteristics(
            n_tp_ramp = n_tp_ramp,
            tp_0_ramp = tp_0_ramp,
        )
        

        # verify types
        alpha_logistic = 0.0 if not sf.isnumber(alpha_logistic) else alpha_logistic
        d = 0 if not sf.isnumber(d) else d
        window_logistic = (-8, 8) if not isinstance(window_logistic, tuple) else window_logistic
 

        vec_out = sf.ramp_vector(
            n_tp, 
            alpha_logistic = alpha_logistic, 
            r_0 = tp_0_ramp,
            r_1 = tp_0_ramp + n_tp_ramp,
            window_logistic = window_logistic,
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
    


    def check_trns_fuel_switch_allocation_dict(self,
        dict_check: dict,
        dict_alternate: dict,
        input_keys_as_fuels: bool = True,
    ) -> bool:
        """
        Check to ensure that the fuel switching dictionary is specified 
            correctly
        """

        out = dict_alternate
        if isinstance(dict_check, dict):
            dict_check_out = self.model_attributes.get_valid_categories_dict(
                dict_check,
                self.model_attributes.subsec_name_enfu,
            )

            # convert to fuel fraction variables
            if input_keys_as_fuels: 
                dict_check_out = dict(
                    (
                        self.model_enercons
                        .dict_trns_fuel_categories_to_fuel_variables
                        .get(k)
                        .get("fuel_fraction"),
                        sf.scalar_bounds(v, (0, 1))
                    )
                    for (k, v) in dict_check_out.items()
                )
    

            if sum(dict_check_out.values()) == 1.0:
                out = dict_check_out
        
        return out
    


    def check_trns_tech_allocation_dict(self,
        dict_check: dict,
        dict_alternate: dict,
        sum_check: Union[str, None] = "eq",
    ) -> bool:
        """
        Check to ensure that the fuel switching dictionary is specified 
            correctly

        Keyword Arguments
        -----------------
        - sum_check: "eq" to force values to sum to to 1, "leq" to force leq 
            than 1. By default, always must be geq 0
        """

        out = dict_alternate
        if isinstance(dict_check, dict):
            dict_check_out = self.model_attributes.get_valid_categories_dict(
                dict_check,
                self.model_attributes.subsec_name_trns,
            )

            dict_check_out = dict(
                (k, sf.scalar_bounds(v, (0, 1)))
                for (k, v) in dict_check_out.items()
            )

            if isinstance(sum_check, str):

                return_dict = (sum_check == "eq") & (sum(dict_check_out.values()) == 1.0)
                return_dict = (sum_check == "leq") & (sum(dict_check_out.values()) <= 1.0)
                if return_dict:
                    out = dict_check_out
        
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

        if not sf.islistlike(cats_entc_max_investment_ramp):
            cats_entc_max_investment_ramp = []
        
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
    


    def get_inen_parameters(self,
        dict_cats_inen_high_heat_to_frac: Union[List[str], None] = None,
    ) -> List[str]:
        """
        Get INEN parameters for the implementation of transformations. Returns a 
            tuple with the following elements (dictionary keys, if present, are 
            shown within after comments; otherwise, calculated internally):

            (
                dict_cats_inen_high_heat_to_frac, # key "categories_inen_high_heat",
                cats_inen_not_high_heat,
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires keys in dict_config to set. If not found, will set the 
            following defaults:
                * dict_cats_inen_high_heat_to_frac: {
                    "cement": 0.88,
                    "chemicals": 0.5, 
                    "glass": 0.88, 
                    "lime_and_carbonite": 0.88, 
                    "metals": 0.92,
                    "paper": 0.18, 
                }
                * cats_inen_not_high_heat: derived from INEN Fuel Fraction 
                    variables and cats_inen_high_heat (complement)
                * frac_inen_high_temp_elec_hydg: (electrification and hydrogen
                    potential fractionation of industrial energy demand, 
                    targeted at high temperature demands: 50% of 1/2 of 90% of 
                    total INEN energy demand for each fuel)
                * frac_inen_low_temp_elec: 0.95*0.45 (electrification potential 
                    fractionation of industrial energy demand, targeted at low 
                    temperature demands: 95% of 1/2 of 90% of total INEN energy 
                    demand)
            
            The value of `frac_inen_shift_denom` is 
                frac_inen_low_temp_elec + 2*frac_inen_high_temp_elec_hydg


        Keyword Arguments
        -----------------
        - cats_inen_high_heat: optional specification of INEN categories that 
            include high heat 
        """


        attr_industry = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_ippu
        )


        ##  GET INEN HIGH HEAT CATEGORIES

        key =  f"{self.key_config_general}.{self.key_config_dict_cats_inen_high_heat_to_frac}"
        default_cats_inen_high_heat = get_dict_config_default()
        default_cats_inen_high_heat = default_cats_inen_high_heat.get(key)
        
        
        #
        dict_cats_inen_high_heat = (
            self.config.get(key)
            if not isinstance(dict_cats_inen_high_heat_to_frac, dict)
            else self.model_attributes.get_valid_categories_dict(
                dict_cats_inen_high_heat_to_frac,
                self.model_attributes.subsec_name_ippu
            )
        )

        if dict_cats_inen_high_heat is None:
            dict_cats_inen_high_heat = default_cats_inen_high_heat
       

        ##  GET INEN LOW AND MEDIUM HEAT CATEGORIES

        modvars_inen_fuel_switching = tbe.transformation_inen_shift_modvars(
            None,
            None,
            None,
            self.model_attributes,
            return_modvars_only = True
        )
        cats_inen_fuel_switching = set({})
        for modvar in modvars_inen_fuel_switching:
            cats_inen_fuel_switching = cats_inen_fuel_switching | set(self.model_attributes.get_variable_categories(modvar))

        cats_inen_not_high_heat = sorted(list(cats_inen_fuel_switching - set(dict_cats_inen_high_heat.keys()))) 

        """
        # fraction of energy that can be moved to electric/hydrogen, representing high heat transfer
        frac_inen_high_temp_elec_hydg = (
            self.config
            .get(
                f"{self.key_config_general}.{self.key_config_frac_inen_high_temp_elec_hydg}"
            )
        )
        if not sf.isnumber(frac_inen_high_temp_elec_hydg):
            tp = str(frac_inen_high_temp_elec_hydg)'
            msg = f"Invalid specification of '{self.key_config_frac_inen_high_temp_elec_hydg}' in configuration:
            type must of float, not {tp}. Check the configuration file used to initialize the
            Transformers object.
            "
            raise TypeError(tp)


        # fraction of energy that can be moved to electric, representing low heat transfer
        frac_inen_low_temp_elec = (
            self.config
            .get(
                f"{self.key_config_general}.{self.key_config_frac_inen_low_temp_elec}"
            )
        )
        if not sf.isnumber(frac_inen_low_temp_elec):
            tp = str(frac_inen_low_temp_elec)'
            msg = f"Invalid specification of '{self.frac_inen_low_temp_elec}' in configuration:
            type must of float, not {tp}. Check the configuration file used to initialize the
            Transformers object.
            "
            raise TypeError(tp)

        # shift denominator
        frac_inen_shift_denom = frac_inen_low_temp_elec + frac_inen_high_temp_elec_hydg
        """
        
        # setup return
        tup_out = (
            dict_cats_inen_high_heat,
            cats_inen_not_high_heat,
            #frac_inen_high_temp_elec_hydg,
            #frac_inen_low_temp_elec,
            #frac_inen_shift_denom,
        )
        
        return tup_out



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
        return_code: bool = False,
    ) -> None:
        """
        Get `transformer` based on transformer code, id, or name
        
        If strat is None or an invalid valid of strat is entered, returns None; 
            otherwise, returns the Transformer object. 

            
        Function Arguments
        ------------------
        - transformer: transformer_id, transformer name, or transformer code to 
            use to retrieve sc.Trasnformation object
            
        Keyword Arguments
        ------------------
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
            f"{self.field_transformer_id}_to_{self.attribute_transformer_code.key}"
        )
        dict_name_to_code = self.attribute_transformer_code.field_maps.get(
            f"{self.field_transformer_name}_to_{self.attribute_transformer_code.key}"
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
    


    def get_transformer_codes_by_sector(self,
        key_other: str = "other",
    ) -> dict:
        """
        Map transformers to the sector they are associated with (by code). If
            not associated with any sector, adds to `key_other` key
        """
        
        attribute_sectors = self.model_attributes.get_sector_attribute_table()
        dict_out = dict(
            (
                attribute_sectors.get_attribute(x, "sector"), 
                []
            ) 
            for x in attribute_sectors.key_values
        )
        
        # add other key
        dict_out.update({key_other: []})
        
        
        # check all transformer codes
        for code in self.all_transformers:
            
            # try to match the code; skip baseline
            code_match = self.regex_code_structure.match(code)
            if code_match is None:
                continue 
            
            # pull matching component and try to get the secod
            subsec = code_match.groups()[0]
            sec = self.model_attributes.get_subsector_attribute(subsec, "sector")
            sec = key_other if sec is None else sec

            dict_out.get(sec).append(code)
        
        
        return dict_out
    


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
        vec_msp_resolution_cap: Union[np.ndarray, Dict[str, int], None] = None,
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
        - vec_msp_resolution_cap: MSP cap for renewables
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
        dict_cat_to_vector = (
            dict(
                (x, vec_msp_resolution_cap)
                for x in cats_to_cap
            )
            if sf.islistlike(vec_msp_resolution_cap)
            else {}
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
        magnitude_lurf: Union[float, None] = 0.0,
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
        if not sf.isnumber(magnitude_lurf):
            magnitude_lurf = self.config.get(
                    f"{self.key_config_baseline}.{self.key_config_magnitude_lurf}",
                )
            
        magnitude_lurf = self.bounded_real_magnitude(magnitude_lurf, 0.0)

        
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
            vec_msp_resolution_cap = vec_msp_resolution_cap,
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
    


    def _trfunc_baseline_return(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Create a return function to support the baseline transformation

        Function Arguments
        ------------------
        

        Keyword Arguments
        -----------------
        - df_input: optional input dataframe
        - strat: optional strategy id to pass
        """

        # check input dataframe
        df_out = (
            self.baseline_inputs
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
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
        magnitude_type: str = "final_value", # behavior here is a ceiling
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
        magnitude: float = 0.8,
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
        - magnitude: fraction of vented methane that is flared
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
        """
        Implement the "Minimize Leaks" FGTV transformation on input DataFrame
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fraction of leaky sources (pipelines, storage, etc) that
            are fixed
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
        #dict_high_fuel_split: Union[dict, None] = None,
        df_input: Union[pd.DataFrame, None] = None,
        frac_high_given_high: Union[float, dict, None] = None,
        frac_switchable: float = 0.9, 
        strat: Union[int, None] = None,
        vec_implementation_ramp: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch low-temp thermal processes to industrial heat 
            pumps" or/and "Fuel switch medium and high-temp thermal processes to 
            hydrogen and electricity" INEN transformations on input DataFrame 
            df_input (note: these must be combined in a new function instead of
            as a composition due to the electricity shift in high-heat 
            categories)
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - dict_high_fuel_split: optional dictionary mapping high heat to target
            fuel splits. If None, defaults to
                {
                    "fuel_electricity": 0.5,
                    "fuel_hydrogen": 0.5
                }

        - df_input: data frame containing trajectories to modify
        - frac_high_given_high: in high heat categories, fraction of heat demand 
            that is high (NOTE: needs to be switched to per industry). 
            * If specified as a float, this is applied to all high heat 
                categories
            * If specified as None, uses the following dictionary:
                (TEMP: from https://www.sciencedirect.com/science/article/pii/S0360544222018175?via%3Dihub#bib34 [see sainz_et_al_2022])

                {
                    "cement": 0.88, # use non-metallic minerals
                    "chemicals": 0.5, 
                    "glass": 0.88, 
                    "lime_and_carbonite": 0.88, 
                    "metals": 0.92,
                    "paper": 0.18, 
                }

        - frac_switchable: fraction of demand that can be switched 
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
        """
        Implement the "Maximize Industrial Energy Efficiency" INEN 
            transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of energy efficiency increase (applied to
            industrial efficiency factor)
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
        """
        Implement the "Maximize Industrial Production Efficiency" INEN 
            transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of energy efficiency increase (applied to
            industrial production factor)
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
        """
        Implement the "Switch to electricity for heat using heat pumps, electric 
            stoves, etc." INEN transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: magntiude of fraction of heat energy that is electrified
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
        """
        Implement the "Reduce end-use demand for heat energy by improving 
            building shell" SCOE transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: reduction in heat energy demand, driven by retrofitting and
            changes in use
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
        """
        Implement the "Increase appliance efficiency" SCOE transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional increase in applieance energy efficiency
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
        """
        Implement the "Reduce Demand" TRDE transformation on input DataFrame
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in transportation demand
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
        """
        Implement the "Electrify Light-Duty" TRNS transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories: transportation categories to include; defaults to 
            "road_light"
        - dict_fuel_allocation: optional dictionary defining fractional 
            allocation of fuels in fuel switch. If undefined, defaults to
                {
                    "fuel_electricity": 1.0
                }
            
            NOTE: keys must be valid TRNS fuels and values in the dictionary 
            must sum to 1.

        - df_input: data frame containing trajectories to modify
        - magnitude: fraction of light duty vehicles electrified 
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
        """
        Implement the "Electrify Rail" TRNS transformation on input DataFrame
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories: transportation categories to include; defaults to 
            ["rail_freight", "rail_passenger"]
        - dict_fuel_allocation: optional dictionary defining fractional 
            allocation of fuels in fuel switch. If undefined, defaults to
                {
                    "fuel_electricity": 1.0
                }
            
            NOTE: keys must be valid TRNS fuels and values in the dictionary 
            must sum to 1.

        - df_input: data frame containing trajectories to modify
        - magnitude: fraction of light duty vehicles electrified 
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
        """
        Implement the "Fuel-Swich Maritime" TRNS transformation on input 
            DataFrame df_input. By default, transfers mangitude to hydrogen from 
            gasoline and diesel; e.g., with magnitude = 0.7, then 70% of diesel 
            and gas demand are transfered to fuels in fuels_target. The rest of
            the fuel demand is then transferred to electricity. 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - catgories: TRNS categories to apply to. Defaults to water_borne
        - dict_allocation_fuels_target: dictionary allocating target fuels. If
            None, defaults to
            {
                "fuel_hydrogen": 1.0,
            }

        - df_input: data frame containing trajectories to modify
        - fuels_source: fuels to transfer out; for F the percentage of TRNS
            demand met by fuels in fuels source, M*F (M = magtnitude) is
            transferred to fuels defined in dict_allocation_fuels_target
        - magnitude: fraction of water borne fuels_source (gas and diesel, 
            e.g.) that shifted to target fuels fuels_target (hydrogen is 
            default, can include ammonia). Note, remaining is shifted to 
            electricity
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        """
        Implement the "Fuel-Switch Medium Duty" TRNS transformation on input 
            DataFrame df_input. By default, transfers mangitude to electricity 
            from gasoline and diesel; e.g., with magnitude = 0.7, then 70% of 
            diesel and gas demand are transfered to fuels in fuels_target. The 
            rest of the fuel demand is then transferred to hydrogen. 
        
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - catgories: TRNS categories to apply to. Defaults to 
            [
                "road_heavy_freight", 
                "road_heavy_regional", 
                "public"
            ]
        - dict_allocation_fuels_target: dictionary allocating target fuels. If
            None, defaults to
            {
                "fuel_electricity": 1.0,
            }

        - df_input: data frame containing trajectories to modify
        - fuels_source: fuels to transfer out; for F the percentage of TRNS
            demand met by fuels in fuels source, M*F (M = magtnitude) is
            transferred to fuels defined in dict_allocation_fuels_target
        - magnitude: fraction of water borne fuels_source (gas and diesel, 
            e.g.) that shifted to target fuels fuels_target (hydrogen is 
            default, can include ammonia). Note, remaining is shifted to 
            electricity
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
        
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
        """
        Implement the "Increase Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: increase the efficiency of electric vehicales by this 
            proportion
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
        """
        Implement the "Increase Non-Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: increase the efficiency of non-electric vehicales by this 
            proportion
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
        """
        Implement the "Increase Vehicle Occupancy" TRNS transformation on input 
            DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: increase the occupancy rate of light duty vehicles by this
            proporiton
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
        """
        Implement the "Mode Shift Freight" TRNS transformation on input 
            DataFrame df_input. By Default, transfer 20% of aviation and road
            heavy freight to rail freight.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories_out: categories to shift out of 
        - dict_categories_target: dictionary mapping target categories to 
            proportional allocation of mode mass. If None, defaults to
            {
                "rail_freight": 1.0
            }

        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of mode mass to shift out of cats_out
        - strat: optional strategy value to specify for the transformation
        - vec_implementation_ramp: optional vector specifying the implementation
            scalar ramp for the transformation. If None, defaults to a uniform 
            ramp that starts at the time specified in the configuration.
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
        """
        Implement the "Mode Shift Passenger Vehicles to Others" TRNS 
            transformation on input DataFrame df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - categories_out: categories to shift out of 
        - dict_categories_target: dictionary mapping target categories to 
            proportional allocation of mode mass. If None, defaults to
            {
                "human_powered": (1/6),
                "powered_bikes": (2/6),
                "public": 0.5
            }

        - df_input: data frame containing trajectories to modify
        - magnitude: magnitude of mode mass to shift out of cats_out
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
        magnitude: float = 0.1,
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
        - dict_categories_out: dictionary mapping categories to shift out of to 
            the magnitude of the outward shift
        - dict_categories_target: dictionary mapping target categories to 
            proportional allocation of mode mass. If None, defaults to
            {
                "road_heavy_regional": 1.0
            }

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

        # check magnitude and categories
        magnitude = self.bounded_real_magnitude(magnitude, 0.1)

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
        """
        Implement the "Reduce cement clinker" IPPU transformation on input 
            DataFrame df_input. Implements a cap on the fraction of cement that
            is produced using clinker (magnitude)
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fraction of cement producd using clinker
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
        """
        Implement the "Demand Management" IPPU transformation on input DataFrame 
            df_input. Reduces industrial production.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in demand in accordance with 
            vec_implementation_ramp
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
        """
        Implement the "Reduces HFCs" IPPU transformation on input DataFrame 
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in HFC emissions in accordance with
            vec_implementation_ramp
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
        """
        Implement the "Reduces N2O" IPPU transformation on input DataFrame 
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in IPPU N2O emissions in accordance 
            with vec_implementation_ramp
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
        """
        Implement the "Reduces Other FCs" IPPU transformation on input DataFrame 
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in IPPU other FC emissions in 
            accordance with vec_implementation_ramp
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
        """
        Implement the "Reduces Other FCs" IPPU transformation on input DataFrame 
            df_input
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - df_input: data frame containing trajectories to modify
        - magnitude: fractional reduction in IPPU other FC emissions in 
            accordance with vec_implementation_ramp
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





########################
#    SOME FUNCTIONS    #
########################

def is_transformer(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Transformer
    """
    out = hasattr(obj, "is_transformer")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out



def is_transformers(
    obj: Any,
) -> bool:
    """
    Determine if the object is a Transformers
    """
    out = hasattr(obj, "is_transformers")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out

