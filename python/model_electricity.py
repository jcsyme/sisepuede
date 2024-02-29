from attribute_table import AttributeTable
import logging
import importlib
from model_attributes import *
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_energy import NonElectricEnergy
from model_socioeconomic import Socioeconomic
import numpy as np
import os, os.path
import pandas as pd
import support_functions as sf
import sqlalchemy
import sql_utilities as sqlutil
import time
from typing import *

##  import the julia api
from julia.api import Julia




###########################
###                     ###
###     ENERGY MODEL    ###
###                     ###
###########################

class ElectricEnergy:
    """
    Use ElectricEnergy to calculate emissions from electricity generation and 
        fuel production--including coal mining, hydrogen production, natural gas
        processing, and petroleum refinement--using NemoMod. Integrates with the 
        SISEPUEDE integrated modeling framework.

    For more information on the ElectricEnergy class, see the SISEPUEDE 
        readthedocs at

        https://sisepuede.readthedocs.io/en/latest/energy_electric.html


    Intialization Arguments
    -----------------------
    - model_attributes: ModelAttributes object used in SISEPUEDE
    - dir_jl: location of Julia directory containing Julia environment and 
        support modules
    - nemomod_reference_files: dictionary of input reference dataframes OR 
        directory containing required CSVs
        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
    - initialize_julia: initialize the Julia connection? Required to run the 
        model.
        * Set to False to access ElectricEnergy properties without initializing
            the connection to Julia.
    - logger: optional logger object to use for event logging
    - solver_time_limit: run-time limit for solver in seconds. If None, defaults
        to model_attributes configuration default.

    Requirements
    ------------
    - Julia 1.7+
    - Python PyJulia package
    - NemoMod (see https://sei-international.github.io/NemoMod.jl/stable/ for 
        the latest stable release)
    - At least one of the following solver packages (^ denotes open source):
        * Cbc^
        * Clp^
        * CPLEX
        * GAMS (to access GAMS solvers)
        * GPLK^
        * Gurobi
        * HiGHS^ (at least one)
    """

    def __init__(self,
        model_attributes: ModelAttributes,
        dir_jl: str,
        nemomod_reference_files: Union[str, dict],
        initialize_julia: bool = True,
        logger: Union[logging.Logger, None] = None,
        solver_time_limit: Union[int, None] = None,
    ):
        ##  INITIALIZE KEY PROPERTIES (ORDERED)

        # initalize the logger and model attributes
        self.logger = logger
        self.model_attributes = model_attributes

        # initialize names and shared fields
        self._initialize_subsector_names()
        self._initialize_nemomod_fields()
        self._initialize_input_output_components()
        self._initialize_other_properties(solver_time_limit = solver_time_limit)

        # initialize subsectoral model variables, categories, and indices
        self._initialize_subsector_vars_enfu()
        self._initialize_subsector_vars_entc()
        self._initialize_subsector_vars_enst()
        
        # initialize NemoMod properties
        self._initialize_dict_tables_required_to_required_fields()
        self._initialize_nemomod_output_tables()
        self._initialize_nemomod_reference_dict(nemomod_reference_files)

        # finally, initialize models, set integrated variables/field map dictionaries and initialize julia
        self._initialize_models()
        self._initialize_integrated_variables()
        self._initialize_julia(dir_jl, initialize_julia = initialize_julia)



    def __call__(self,
        *args,
        **kwargs
    ) -> pd.DataFrame:

        return self.project(*args, **kwargs)





    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. 
            See ?sf._optional_log for more information

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



    def check_df_fields(self,
        df_elec_trajectories: pd.DataFrame,
        subsector: str = "All",
        var_type: str = "input",
        msg_prepend: Union[str, None] = None
    ) -> None:
        if subsector == "All":
            check_fields = self.required_variables
            msg_prepend = "Electricity"
        else:
            self.model_attributes.check_subsector(subsector)
            if var_type == "input":
                check_fields, ignore_fields = self.model_attributes.get_input_output_fields([self.subsec_name_econ, self.subsec_name_gnrl, subsector])
            elif var_type == "output":
                ignore_fields, check_fields = self.model_attributes.get_input_output_fields([subsector])
            else:
                raise ValueError(f"Invalid var_type '{var_type}' in check_df_fields: valid types are 'input', 'output'")
            msg_prepend = msg_prepend if (msg_prepend is not None) else subsector
        sf.check_fields(df_elec_trajectories, check_fields, f"{msg_prepend} projection cannot proceed: fields ")

        return None


    
    def get_enfu_dict_subsectors_to_energy_variables(self,
    ) -> Dict:
        """
        Return a dictionary with emission-producing energy subsectorz as keys 
            based on the Energy Fuels attribute table:

            {
                subsec: {
                    "energy_demand": VARNAME_ENERGY, 
                    ...
                }
            }

            for each key, the dict includes variables associated with subsector
            ``subsec``

            - "energy_demand"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.model_attributes.subsec_name_enfu,
            "abbreviation_subsector",
            {
                "Energy Demand by Fuel": "energy_demand"
            },
        )

        return dict_out


    
    def get_entc_dict_subsectors_to_emission_variables(self,
    ) -> Dict:
        """
        Return a dictionary with emission-producing energy subsectorz as keys 
            based on the Energy Technology attribute table:

            {
                subsec: {
                    "emissions_ch4": VARNAME_EMISSIONS, 
                    ...
                }
            }

            for each key, the dict includes variables associated with subsector
            ``subsec``

            - "emissions_ch4"
            - "emissions_co2"
            - "emissions_n2o"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.model_attributes.subsec_name_entc,
            "abbreviation_subsector",
            {
                "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation": "emissions_ch4",
                "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation": "emissions_co2",
                "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation": "emissions_n2o"
            },
        )

        return dict_out
        


    def _initialize_dict_tables_required_to_required_fields(self,
    ) -> None:
        """
        Set a dictionary mapping required references tables to required fields. 
            Initializes the following properties:

            * self.dict_tables_required_to_required_fields
            * self.required_reference_tables
        """

        self.dict_tables_required_to_required_fields = {
            "CapacityFactor": [
                self.field_nemomod_region,
                self.field_nemomod_time_slice
            ],
            "SpecifiedDemandProfile": [
                self.field_nemomod_region,
                self.field_nemomod_time_slice,
                self.field_nemomod_value
            ]
        }

        self.required_reference_tables = sorted(list(
            self.dict_tables_required_to_required_fields.keys()
        ))

        return None



    def _initialize_input_output_components(self,
    ) -> None:
        """
        Set a range of input components, including required dimensions, 
            subsectors, input and output fields, and integration variables.
            Sets the following properties:

            * self.output_variables
            * self.required_dimensions
            * self.required_subsectors
            * self.required_base_subsectors
            * self.required_variables
            
        """

        ##  START WITH REQUIRED DIMENSIONS (TEMPORARY - derive from attributes later)

        required_doa = [self.model_attributes.dim_time_period]
        self.required_dimensions = required_doa


        ##  ADD REQUIRED SUBSECTORS (TEMPORARY - derive from attributes)

        subsectors = [self.subsec_name_enfu, self.subsec_name_enst, self.subsec_name_entc]#self.model_attributes.get_setor_subsectors("Energy")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]

        self.required_subsectors = subsectors
        self.required_base_subsectors = subsectors_base


        ##  SET ELECTRICITY INPUT/OUTPUT FIELDS

        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(subsectors)

        self.required_variables = required_vars + required_doa
        self.output_variables = output_vars

        return None



    def _initialize_integrated_variables(self
    ) -> None:
        """
        Sets the following integration variable properties:

            * self.list_vars_required_for_integration

            Updates the following properties:

            * self.required_variables
        """
        # set the integration variables
        self.integration_variables = [
            # AFOLU variables
            self.model_afolu.modvar_lsmm_recovered_biogas,
            # CircularEconomy variables
            self.model_circecon.modvar_trww_recovered_biogas,
            self.model_circecon.modvar_waso_emissions_ch4_incineration,
            self.model_circecon.modvar_waso_emissions_co2_incineration,
            self.model_circecon.modvar_waso_emissions_n2o_incineration,
            self.model_circecon.modvar_waso_recovered_biogas_anaerobic,
            self.model_circecon.modvar_waso_recovered_biogas_landfills,
            self.model_circecon.modvar_waso_waste_total_for_energy_isw,
            self.model_circecon.modvar_waso_waste_total_for_energy_msw,
            self.model_circecon.modvar_waso_waste_total_incineration,
            # Energy Fuel Demand outputs from other sectors
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns
        ]

        # in Electricity, update required variables
        for modvar in self.integration_variables:
            new_vars = self.model_attributes.build_variable_fields(modvar)
            self.required_variables += new_vars

        # set required variables and ensure no double counting
        self.required_variables = list(set(self.required_variables))
        self.required_variables.sort()

        return None

        

    def _initialize_julia(self,
        dir_jl: str,
        initialize_julia: bool = True,
        module_sisepuede_support_functions_jl: str = "SISEPUEDEPJSF",
        solver: Union[str, None] = None
    ) -> None:
        """
        Import packages and choose the solver from what is available
            (see SISEPUEDEPJSF for solver hierarchy). Sets the
            following properties:

            * self.dict_solver_to_julia_package
            * self.dir_jl
            * self.fp_pyjulia_support_functions
            * self.solver
            * self.solver_module
            * self.solver_package

        Function Arguments
        ------------------
        - dir_jl: SISEPUEDE directory containing Julia environment (with NemoMod) and
            associated modules
            * Must contain f"{module_sisepuede_support_functions_jl}.jl"

        Optional Arguments
        ------------------
        - solver: optional solver to specify. If None, defaults to configuration
        - module_sisepuede_support_functions_jl: name of the Julia module in dir_jl
            containing support functions to use in communication with Julia
            * NOTE: Should not include the jl extension
        """

        if not initialize_julia:
            return None


        ##  CHECK DIRECTORIES AND REQUIRED FILES

        self.dir_jl = dir_jl
        if not os.path.exists(dir_jl):
            self.dir_jl = None
            self._log(f"Path to Julia '{dir_jl}' not found. The electricity cannot be run.", type_log = "error")
            raise RuntimeError()

        self.fp_pyjulia_support_functions = os.path.join(self.dir_jl, f"{module_sisepuede_support_functions_jl}.jl")
        if not os.path.exists(self.fp_pyjulia_support_functions):
            self.fp_pyjulia_support_functions = None
            self._log(f"Path to support module '{self.fp_pyjulia_support_functions}' not found. The electricity cannot be run.", type_log = "error")
            raise RuntimeError()


        ##  LOAD SOME PACKAGES

        self._log(f"Calling Julia API...", type_log = "info")
        self.api = importlib.import_module("julia", package = "api")

        self.jl = self.api.Julia(compiled_modules = False)
        self._log(f"\tSuccessfully activated Julia with compiled_modules = False", type_log = "debug")

        self.julia_base = importlib.import_module("julia.Base")
        self._log(f"\tSuccessfully imported julia.Base", type_log = "debug")

        self.julia_main = importlib.import_module("julia.Main")
        self._log(f"\tSuccessfully imported julia.Main", type_log = "debug")

        self.julia_pkg = importlib.import_module("julia.Pkg")
        self._log(f"\tSuccessfully imported julia.Pkg", type_log = "debug")

        avail_packages = self.julia_main.collect(self.julia_main.keys(self.julia_pkg.project().dependencies))
        logstr = sf.format_print_list(avail_packages, delim = "\n\t\t")
        self._log(f"\tAvailable Packages:\n\t\t{logstr}", type_log = "debug")

        try:
            self.julia_pkg.activate(self.dir_jl)
            self._log(f"Successfully activated Julia environment at '{self.dir_jl}'", type_log = "info")

        except Exception as e:
            self._log(f"Error activating the Julia environment at '{self.dir_jl}': {e}", type_log = "error")

        try:
            self.julia_nemomod = importlib.import_module("julia.NemoMod")
            self.julia_jump = importlib.import_module("julia.JuMP")
        except Exception as e:
            self._log(f"Error activating NemoMod/JuMP packages: {e}", type_log = "error")


        ##  CHECK AND LOAD SOLVER

        # next, try to instantiate solvers - map solvers to
        self.dict_solver_to_julia_package = {
            "cbc": "Cbc",
            "clp": "Clp",
            "cplex": "CPLEX",
            "gams_cplex": "GAMS",
            "glpk": "GLPK",
            "gurobi": "Gurobi",
            "highs": "HiGHS",
            "ipopt": "Ipopt",
        }
        self.dict_julia_package_to_solver = sf.reverse_dict(self.dict_solver_to_julia_package)

        # check solver specification in configuration
        solver = self.model_attributes.configuration.get("nemomod_solver") if (solver is None) else solver
        sf.check_set_values([solver], self.model_attributes.configuration.valid_solver)
        if solver not in self.dict_solver_to_julia_package.keys():
            self._log(f"""
                Solver '{solver}' not found in _initialize_julia(); 
                check self.dict_solver_to_julia_package to ensure it lines up with 
                configuration.valid_solver. Resetting to best available...""", 
                type_log = "warning"
            )
            solver = "ANYSOLVER" # non-existant solver; SISEPUEDEPJSF.check_solvers will check for best available

        # check solver specification in julia, and reassign solver to dependent on Julia's specification
        self.julia_main.include(self.fp_pyjulia_support_functions)
        self.solver_package = self.julia_main.SISEPUEDEPJSF.check_solvers(
            self.dict_solver_to_julia_package.get(solver)
        )
        self.solver = self.dict_julia_package_to_solver.get(self.solver_package)

        # load
        try:
            self.solver_module = importlib.import_module(f"julia.{self.solver_package}")
            self._log(f"Successfully initialized JuMP optimizer from solver module {self.solver_package}.", type_log = "info")
        except Exception as e:
            self._log(f"An error occured while trying to initialize the JuMP optimizer from package: {e}", type_log = "error")

        return None



    def _initialize_models(self,
        model_attributes: Union[ModelAttributes, None] = None
    ) -> None:
        """
        Initialize SISEPUEDE model classes for fetching variables and 
            accessing methods. Initializes the following properties:

            * self.model_afolu
            * self.model_circecon
            * self.model_energy
            * self.model_socioeconomic

        Keyword Arguments
        -----------------
        - model_attributes: ModelAttributes object used to instantiate
            models. If None, defaults to self.model_attributes.
        """

        model_attributes = self.model_attributes if (model_attributes is None) else model_attributes
        
        self.model_afolu = AFOLU(self.model_attributes)
        self.model_circecon = CircularEconomy(self.model_attributes)
        self.model_energy = NonElectricEnergy(self.model_attributes)
        self.model_socioeconomic = Socioeconomic(self.model_attributes)

        return None



    def _initialize_nemomod_fields(self
    ) -> None:
        """
        Set common fields used in NemoMod. Sets the following properties:

            * self.dict_fields_nemomod_to_type
            * self.field_nemomod_####
            * self.fields_nemomod_sort_hierarchy
        """
    
        # add some key fields from nemo mod
        self.field_nemomod_description = "desc"
        self.field_nemomod_emission = "e"
        self.field_nemomod_fuel = "f"
        self.field_nemomod_id = "id"
        self.field_nemomod_lorder = "lorder"
        self.field_nemomod_mode = "m"
        self.field_nemomod_multiplier = "multiplier"
        self.field_nemomod_name = "name"
        self.field_nemomod_order = "order"
        self.field_nemomod_region = "r"
        self.field_nemomod_solvedtm = "solvedtm"
        self.field_nemomod_storage = "s"
        self.field_nemomod_table_name = "tablename"
        self.field_nemomod_technology = "t"
        self.field_nemomod_tg1 = "tg1"
        self.field_nemomod_tg2 = "tg2"
        self.field_nemomod_time_slice = "l"
        self.field_nemomod_value = "val"
        self.field_nemomod_year = "y"

        # dictionary to map fields to type
        self.dict_fields_nemomod_to_type = {
            self.field_nemomod_description: str,
            self.field_nemomod_emission: str,
            self.field_nemomod_fuel: str,
            self.field_nemomod_id: int,
            self.field_nemomod_lorder: int,
            self.field_nemomod_mode: str,
            self.field_nemomod_multiplier: float,
            self.field_nemomod_name: str,
            self.field_nemomod_order: int,
            self.field_nemomod_region: str,
            self.field_nemomod_storage: str,
            self.field_nemomod_table_name: str,
            self.field_nemomod_technology: str,
            self.field_nemomod_tg1: str,
            self.field_nemomod_tg2: str,
            self.field_nemomod_time_slice: str,
            self.field_nemomod_year: str
        }

        # sort hierarchy for tables
        self.fields_nemomod_sort_hierarchy = [
            self.field_nemomod_id,
            self.field_nemomod_region,
            self.field_nemomod_table_name,
            self.field_nemomod_technology,
            self.field_nemomod_storage,
            self.field_nemomod_fuel,
            self.field_nemomod_emission,
            self.field_nemomod_mode,
            self.field_nemomod_time_slice,
            self.field_nemomod_year,
            # value and description should always be at the end
            self.field_nemomod_value,
            self.field_nemomod_description
        ]

        return None
    


    def _initialize_nemomod_output_tables(self,
    ) -> None:
        """
        SET NemoMod tables that need to be extract to generate results in
            SISEPUEDE. Sets the following properties:

            * self.required_nemomod_output_tables
        """

        self.required_nemomod_output_tables = [
            self.model_attributes.table_nemomod_annual_demand_nn,
            self.model_attributes.table_nemomod_annual_emissions_by_technology,
            self.model_attributes.table_nemomod_capital_investment_discounted,
            self.model_attributes.table_nemomod_capital_investment_storage_discounted,
            self.model_attributes.table_nemomod_operating_cost_discounted,
            self.model_attributes.table_nemomod_production_by_technology,
            self.model_attributes.table_nemomod_total_annual_capacity,
            self.model_attributes.table_nemomod_use_by_technology
            #"vnewstoragecapacity",#TEMP
            #"vrateofstoragechargenn",
            #"vrateofstoragechargenodal",
            #"vrateofstoragedischargenn",
            #"vstoragelevelyearendnn",
            #"vusenn"
        ]

        return None



    def _initialize_nemomod_reference_dict(self,
        nemomod_reference_files: Union[str, dict],
        dict_tables_required_to_required_fields: Union[Dict[str, List[str]], None] = None,
        filter_regions_to_config: bool = False
    ) -> None:
        """
        Initialize the dictionary of reference files for NemoMod required to 
            populate the database. Sets the following properties:

            * self.dict_nemomod_reference_tables

        Function Arguments
        ------------------
        - nemomod_reference_files: file path of reference CSV files *OR* 
            dictionary. Required keys/files (without .csv extension):

            * CapacityFactor
            * SpecifiedDemandProfile

        Keyword Arguments
        -----------------
        - dict_tables_required_to_required_fields: dictionary mapping required 
            reference table names (str) to list of required fields.
            * If None, defaults to self.dict_tables_required_to_required_fields
        - filter_regions_to_config: filter regions to correspond with ModelAttributes 
            region attribute table
        """

        ##  INITIALIZATION

        # attribute tables
        attr_region = self.model_attributes.get_other_attribute_table(
            self.model_attributes.dim_region
        )
        attr_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc)
        attr_time_slice = self.model_attributes.get_other_attribute_table("time_slice")

        # attribute derivatives
        dict_tables_required_to_required_fields = (
            self.dict_tables_required_to_required_fields 
            if (dict_tables_required_to_required_fields is None) 
            else dict_tables_required_to_required_fields
        )
        dict_out = {}
        set_tables_required = set(self.dict_tables_required_to_required_fields.keys())


        ##  CHECK INPUT DIRCTORY AND TRY TO READ IN IF FILES

        if isinstance(nemomod_reference_files, str):

            # check the directory
            dir_nemomod_ref = sf.check_path(nemomod_reference_files, False)
            set_tables_available = set([x.replace(".csv", "") for x in os.listdir(dir_nemomod_ref) if x.endswith(".csv")])
            set_tables_available = set_tables_required & set_tables_available

            if not set_tables_required.issubset(set_tables_available):
                set_missing = sf.print_setdiff(set_tables_required, set_tables_available)
                raise RuntimeError(f"Initialization error in ElectricEnergy: required reference tables {set_missing} not found in {dir_nemomod_ref}.")
           
            # read in files
            for fbn in list(set_tables_required):
                fp_read = os.path.join(dir_nemomod_ref, f"{fbn}.csv")
                df_tmp = pd.read_csv(fp_read)
                dict_out.update({fbn: df_tmp})

                self._log(f"Successfully read NemoMod input table data from {fp_read}", type_log = "info")

        # if dictionary, simply copy into output dictionary
        elif isinstance(nemomod_reference_files, dict):
            sf.check_keys(nemomod_reference_files, set_tables_required)
            for k in list(set_tables_required):
                dict_out.update({k: nemomod_reference_files[k]})


        ##  VERIFY INPUT TABLES

        for k in dict_out.keys():

            # check that regions are correctly implemented
            if self.field_nemomod_region in dict_out[k]:

                df_filt = dict_out[k][dict_out[k][self.field_nemomod_region].isin(attr_region.key_values)]
                if len(df_filt) == 0:
                    raise RuntimeError(f"Error in ElectricEnergy._initialize_nemomod_reference_dict: no valid regions found in table {k}.")

                if filter_regions_to_config:
                    regions_config = self.model_attributes.configuration.get("region")
                    sf.check_set_values(regions_config, df_filt[self.field_nemomod_region])
                    df_filt = df_filt[df_filt[self.field_nemomod_region].isin(regions_config)]

                    # conditions needed for the regions
                    check_regions = (len(set(df_filt[self.field_nemomod_region])) == len(set(regions_config)))
                    if not check_regions:
                        missing_vals = sf.print_setdiff(set(regions_config), set(df_filt[self.field_nemomod_region]))
                        raise RuntimeError(f"Initialization error in ElectricEnergy: field {self.field_nemomod_region} in table {k} is missing regions {missing_vals}.")

                dict_out.update({k: df_filt})

            # check that time slices are correctly implemented
            if self.field_nemomod_time_slice in dict_out[k]:

                n = len(dict_out[k])
                df_filt = dict_out[k][dict_out[k][self.field_nemomod_time_slice].isin(attr_time_slice.key_values)]

                if len(set(df_filt[self.field_nemomod_time_slice])) != len(attr_time_slice.key_values):
                    missing_vals = sf.print_setdiff(set(attr_time_slice.key_values), set(df_filt[self.field_nemomod_time_slice]))
                    raise RuntimeError(f"Initialization error in ElectricEnergy: field {self.field_nemomod_time_slice} in table {k} is missing time_slices {missing_vals} .")

        # check fields in CapacityFactor
        sf.check_set_values(
            [x for x in dict_out[self.model_attributes.table_nemomod_capacity_factor].columns if (x not in [self.field_nemomod_region, self.field_nemomod_time_slice])],
            attr_technology.key_values
        )

        self.dict_nemomod_reference_tables = dict_out

        return None

        

    def _initialize_other_properties(self,
        include_supply_techs_for_all_fuels: bool = True,
        solver_time_limit: Union[int, None] = None,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.cat_enmo_gnrt
            * self.cat_enmo_stor
            * self.direction_exchange_year_time_period
            * self.drop_flag_tech_capacities
            * self.include_supply_techs_for_all_fuels
            * self.solver_time_limit
            * self.nemomod_time_period_as_year
            * self.units_energy_nemomod

        Keyword Arguments
        -----------------
        - include_supply_techs_for_all_fuels: set to True to include alternative
            supplies for each fuel. 
        - solver_time_limit: time limit to specify in seconds. If non-integer,
            sets based on configuration default 
            (nemomod_solver_time_limit_seconds)
        """

        # Energy (Electricity) Mode Fields

        # use unsafe functionality of filter_keys_by_attribute to filter an attribute table
        attr_mode = self.model_attributes.get_other_attribute_table(
            self.model_attributes.dim_mode,
        )
        self.cat_enmo_gnrt = self.model_attributes.filter_keys_by_attribute(
            attr_mode,
            {"generation_category": 1}
        )[0]
        self.cat_enmo_stor = self.model_attributes.filter_keys_by_attribute(
            attr_mode,
            {"storage_category": 1}
        )[0]

        # set the runtime limit
        solver_time_limit = (
            max(solver_time_limit, 15) 
            if isinstance(solver_time_limit, int) 
            else solver_time_limit
        )
        self.solver_time_limit = (
            self.model_attributes.configuration.get("nemomod_solver_time_limit_seconds")
            if not isinstance(solver_time_limit, int)
            else solver_time_limit
        )

        # other key variables
        self.drop_flag_tech_capacities = -999
        self.nemomod_time_period_as_year = True # use time periods as years in NemoMod?
        self.direction_exchange_year_time_period = (
            "time_period_as_year" 
            if self.nemomod_time_period_as_year 
            else "time_period_to_year"
        )
        self.include_supply_techs_for_all_fuels = include_supply_techs_for_all_fuels
        self.units_energy_nemomod = self.model_attributes.configuration.get("energy_units_nemomod")

        return None



    def _initialize_subsector_names(self,
    ) -> None:
        """
        Set subsector names (self.subsec_name_####)
        """
        # some subector reference variables
        self.subsec_name_ccsq = "Carbon Capture and Sequestration"
        self.subsec_name_econ = "Economy"
        self.subsec_name_enfu = "Energy Fuels"
        self.subsec_name_enst = "Energy Storage"
        self.subsec_name_entc = "Energy Technology"
        self.subsec_name_fgtv = "Fugitive Emissions"
        self.subsec_name_gnrl = "General"
        self.subsec_name_inen = "Industrial Energy"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_scoe = "Stationary Combustion and Other Energy"
        self.subsec_name_trns = "Transportation"
        self.subsec_name_trde = "Transportation Demand"

        return None



    def _initialize_subsector_vars_enfu(self,
    ) -> None:
        """
        Initialize model variables, categories, and indicies associated with
            ENFU (Energy Fuels). Sets the following properties:

            * self.cat_enfu_****
            * self.ind_enfu_****
            * self.modvar_enfu_****
        """
        # Energy Fuel model variables
        self.modvar_enfu_ef_combustion_co2 = ":math:\\text{CO}_2 Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_ch4 = ":math:\\text{CH}_4 Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_n2o = ":math:\\text{N}_2\\text{O} Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_ch4 = ":math:\\text{CH}_4 Stationary Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_n2o = ":math:\\text{N}_2\\text{O} Stationary Combustion Emission Factor"
        self.modvar_enfu_efficiency_factor_industrial_energy = "Average Industrial Energy Fuel Efficiency Factor"
        self.modvar_enfu_energy_demand_by_fuel_ccsq = "Energy Demand by Fuel in CCSQ"
        self.modvar_enfu_energy_demand_by_fuel_entc = "Energy Demand by Fuel in Energy Technology"
        self.modvar_enfu_energy_demand_by_fuel_inen = "Energy Demand by Fuel in Industrial Energy"
        self.modvar_enfu_energy_demand_by_fuel_scoe = "Energy Demand by Fuel in SCOE"
        self.modvar_enfu_energy_demand_by_fuel_total = "Total Energy Demand by Fuel"
        self.modvar_enfu_energy_demand_by_fuel_trns = "Energy Demand by Fuel in Transportation"
        self.modvar_enfu_energy_density_gravimetric = "Gravimetric Energy Density"
        self.modvar_enfu_energy_density_volumetric = "Volumetric Energy Density"
        self.modvar_enfu_exports_fuel = "Fuel Exports"
        self.modvar_enfu_exports_fuel_adjusted = "Adjusted Fuel Exports"
        self.modvar_enfu_frac_fuel_demand_imported = "Fraction of Fuel Demand Imported"
        self.modvar_enfu_imports_fuel = "Fuel Imports"
        self.modvar_enfu_minimum_frac_fuel_used_for_electricity = "Minimum Fraction of Fuel Used for Electricity Generation"
        self.modvar_enfu_nemomod_renewable_production_target = "NemoMod REMinProductionTarget"
        self.modvar_enfu_price_gravimetric = "Gravimetric Fuel Price"
        self.modvar_enfu_price_thermal = "Thermal Fuel Price"
        self.modvar_enfu_price_volumetric = "Volumetric Fuel Price"
        self.modvar_enfu_production_frac_petroleum_refinement = "Petroleum Refinery Production Fraction"
        self.modvar_enfu_production_frac_natural_gas_processing = "Natural Gas Processing Fraction"
        self.modvar_enfu_production_fuel = "Fuel Production"
        self.modvar_enfu_transmission_loss_electricity = "Electrical Transmission Loss"
        self.modvar_enfu_transmission_loss_frac_electricity = "Electrical Transmission Loss Fraction"
        self.modvar_enfu_unused_fuel_exported = "Unused Fuel Exported"
        self.modvar_enfu_value_of_fuel_ccsq = "Value of Fuel Consumed in CCSQ"
        self.modvar_enfu_value_of_fuel_entc = "Value of Fuel Consumed in Energy Technology"
        self.modvar_enfu_value_of_fuel_inen = "Value of Fuel Consumed in Industrial Energy"
        self.modvar_enfu_value_of_fuel_scoe = "Value of Fuel Consumed in SCOE"
        self.modvar_enfu_value_of_fuel_trns = "Value of Fuel Consumed in Transportation"
        
        # key categories
        self.cat_enfu_bgas = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_biogas_fuel_category: 1
            }
        )[0]
        self.cat_enfu_elec = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_electricity_demand_category: 1
            }
        )[0]
        self.cat_enfu_hgen = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_hydrogen_fuel_category: 1
            }
        )[0]
        self.cat_enfu_hpwr = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_hydropower_fuel_category : 1
            }
        )[0]
        self.cat_enfu_wste = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_waste_fuel_category: 1
            }
        )[0]

        # associated indices
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        self.ind_enfu_bgas = attr_enfu.get_key_value_index(self.cat_enfu_bgas)
        self.ind_enfu_elec = attr_enfu.get_key_value_index(self.cat_enfu_elec)
        self.ind_enfu_hgen = attr_enfu.get_key_value_index(self.cat_enfu_hgen)
        self.ind_enfu_wste = attr_enfu.get_key_value_index(self.cat_enfu_wste)

        # get pivot dictionary
        tuple_dicts = self.get_enfu_dict_subsectors_to_energy_variables()
        self.dict_enfu_subsectors_to_energy_variables = tuple_dicts[0]
        self.dict_enfu_subsectors_to_unassigned_enfu_variables = tuple_dicts[1]

        return None



    def _initialize_subsector_vars_entc(self,
    ) -> None:
        """
        Initialize model variables, categories, and indicies associated with
            ENTC (Energy Technology). Initializes the following properties:

            * self.cat_entc_****
            * dict_entc_fuel_categories_to_fuel_variables
            * dict_entc_fuel_categories_to_unassigned_fuel_variables
            * self.ind_entc_****
            * self.key_iar
            * self.key_oar
            * self.modvar_entc_****
        """

        # Energy (Electricity) Technology Variables
        self.modvar_entc_nemomod_capital_cost = "NemoMod CapitalCost"
        self.modvar_entc_ef_scalar_ch4 = ":math:\\text{CH}_4 NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_ef_scalar_co2 = ":math:\\text{CO}_2 NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_ef_scalar_n2o = ":math:\\text{N}_2\\text{O} NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_efficiency_factor_technology = "Technology Efficiency of Fuel Use"
        self.modvar_entc_fuelprod_emissions_activity_ratio_ch4 = ":math:\\text{CH}_4 Fuel Production NemoMod EmissionsActivityRatio"
        self.modvar_entc_fuelprod_emissions_activity_ratio_co2 = ":math:\\text{CO}_2 Fuel Production NemoMod EmissionsActivityRatio"
        self.modvar_entc_fuelprod_emissions_activity_ratio_n2o = ":math:\\text{N}_2\\text{O} Fuel Production NemoMod EmissionsActivityRatio"
        self.modvar_entc_fuelprod_input_activity_ratio_coal_deposits = "Fuel Production NemoMod InputActivityRatio Coal Deposits"
        self.modvar_entc_fuelprod_input_activity_ratio_crude = "Fuel Production NemoMod InputActivityRatio Crude"
        self.modvar_entc_fuelprod_input_activity_ratio_diesel = "Fuel Production NemoMod InputActivityRatio Diesel"
        self.modvar_entc_fuelprod_input_activity_ratio_electricity = "Fuel Production NemoMod InputActivityRatio Electricity"
        self.modvar_entc_fuelprod_input_activity_ratio_gasoline = "Fuel Production NemoMod InputActivityRatio Gasoline"
        self.modvar_entc_fuelprod_input_activity_ratio_natural_gas = "Fuel Production NemoMod InputActivityRatio Natural Gas"
        self.modvar_entc_fuelprod_input_activity_ratio_natural_gas_unprocessed = "Fuel Production NemoMod InputActivityRatio Natural Gas Unprocessed"
        self.modvar_entc_fuelprod_input_activity_ratio_oil = "Fuel Production NemoMod InputActivityRatio Oil"
        self.modvar_entc_fuelprod_output_activity_ratio_coal = "Fuel Production NemoMod OutputActivityRatio Coal"
        self.modvar_entc_fuelprod_output_activity_ratio_diesel = "Fuel Production NemoMod OutputActivityRatio Diesel"
        self.modvar_entc_fuelprod_output_activity_ratio_gasoline = "Fuel Production NemoMod OutputActivityRatio Gasoline"
        self.modvar_entc_fuelprod_output_activity_ratio_hgl = "Fuel Production NemoMod OutputActivityRatio Hydrocarbon Gas Liquids"
        self.modvar_entc_fuelprod_output_activity_ratio_hydrogen = "Fuel Production NemoMod OutputActivityRatio Hydrogen"
        self.modvar_entc_fuelprod_output_activity_ratio_kerosene = "Fuel Production NemoMod OutputActivityRatio Kerosene"
        self.modvar_entc_fuelprod_output_activity_ratio_natural_gas = "Fuel Production NemoMod OutputActivityRatio Natural Gas"
        self.modvar_entc_fuelprod_output_activity_ratio_oil = "Fuel Production NemoMod OutputActivityRatio Oil"
        self.modvar_entc_max_elec_prod_increase_for_msp = "Maximum Production Increase Fraction to Satisfy MinShareProduction Electricity"
        self.modvar_entc_nemomod_discounted_capital_investment = "NemoMod Discounted Capital Investment"
        self.modvar_entc_nemomod_discounted_operating_costs = "NemoMod Discounted Operating Costs"
        self.modvar_entc_nemomod_emissions_ch4_elec = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation"
        self.modvar_entc_nemomod_emissions_co2_elec = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation"
        self.modvar_entc_nemomod_emissions_n2o_elec = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation"
        self.modvar_entc_nemomod_emissions_ch4_fpr = "NemoMod :math:\\text{CH}_4 Emissions from Fuel Processing and Refinement"
        self.modvar_entc_nemomod_emissions_co2_fpr = "NemoMod :math:\\text{CO}_2 Emissions from Fuel Processing and Refinement"
        self.modvar_entc_nemomod_emissions_n2o_fpr = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Fuel Processing and Refinement"
        self.modvar_entc_nemomod_emissions_ch4_mne = "NemoMod :math:\\text{CH}_4 Emissions from Fuel Mining and Extraction"
        self.modvar_entc_nemomod_emissions_co2_mne = "NemoMod :math:\\text{CO}_2 Emissions from Fuel Mining and Extraction"
        self.modvar_entc_nemomod_emissions_n2o_mne = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Fuel Mining and Extraction"
        self.modvar_entc_nemomod_emissions_export_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for Export"
        self.modvar_entc_nemomod_emissions_export_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for Export"
        self.modvar_entc_nemomod_emissions_export_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for Export"
        self.modvar_entc_nemomod_emissions_subsector_ccsq_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for CCSQ"
        self.modvar_entc_nemomod_emissions_subsector_entc_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for Energy Technology"
        self.modvar_entc_nemomod_emissions_subsector_inen_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for Industrial Energy"
        self.modvar_entc_nemomod_emissions_subsector_scoe_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for SCOE"
        self.modvar_entc_nemomod_emissions_subsector_trns_co2 = "NemoMod :math:\\text{CO}_2 Emissions from Electricity Generation for Transportation"
        self.modvar_entc_nemomod_emissions_subsector_ccsq_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for CCSQ"
        self.modvar_entc_nemomod_emissions_subsector_entc_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for Energy Technology"
        self.modvar_entc_nemomod_emissions_subsector_inen_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for Industrial Energy"
        self.modvar_entc_nemomod_emissions_subsector_scoe_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for SCOE"
        self.modvar_entc_nemomod_emissions_subsector_trns_ch4 = "NemoMod :math:\\text{CH}_4 Emissions from Electricity Generation for Transportation"
        self.modvar_entc_nemomod_emissions_subsector_ccsq_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for CCSQ"
        self.modvar_entc_nemomod_emissions_subsector_entc_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for Energy Technology"
        self.modvar_entc_nemomod_emissions_subsector_inen_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for Industrial Energy"
        self.modvar_entc_nemomod_emissions_subsector_scoe_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for SCOE"
        self.modvar_entc_nemomod_emissions_subsector_trns_n2o = "NemoMod :math:\\text{N}_2\\text{O} Emissions from Electricity Generation for Transportation"
        self.modvar_entc_nemomod_fixed_cost = "NemoMod FixedCost"
        self.modvar_entc_nemomod_generation_capacity = "NemoMod Generation Capacity"
        self.modvar_entc_nemomod_min_share_production = "NemoMod MinShareProduction"
        self.modvar_entc_nemomod_production_by_technology = "NemoMod Production by Technology"
        self.modvar_entc_nemomod_renewable_tag_technology = "NemoMod RETagTechnology"
        self.modvar_entc_nemomod_reserve_margin = "NemoMod ReserveMargin"
        self.modvar_entc_nemomod_reserve_margin_tag_technology = "NemoMod ReserveMarginTagTechnology"
        self.modvar_entc_nemomod_residual_capacity = "NemoMod ResidualCapacity"
        self.modvar_entc_nemomod_total_annual_max_capacity = "NemoMod TotalAnnualMaxCapacity"
        self.modvar_entc_nemomod_total_annual_max_capacity_investment = "NemoMod TotalAnnualMaxCapacityInvestment"
        self.modvar_entc_nemomod_total_annual_min_capacity = "NemoMod TotalAnnualMinCapacity"
        self.modvar_entc_nemomod_total_annual_min_capacity_investment = "NemoMod TotalAnnualMinCapacityInvestment"
        self.modvar_entc_nemomod_variable_cost = "NemoMod VariableCost"

        # set dictionaries 
        self._set_dict_enfu_fuel_categories_to_entc_variables()

        # pivot dictionaries
        tuple_dicts = self.get_entc_dict_subsectors_to_emission_variables()
        self.dict_entc_subsectors_to_emission_variables = tuple_dicts[0]
        self.dict_entc_subsectors_to_unassigned_entc_variables = tuple_dicts[1]
        
        return None



    def _initialize_subsector_vars_enst(self,
    ) -> None:
        """
        Initialize model variables, categories, and indicies associated with
            ENST (Energy Storage). Sets the following properties:

            * self.cat_enst_****
            * self.ind_enst_****
            * self.modvar_enst_****
        """

        # Energy (Electricity) Storage Variables
        self.modvar_enst_nemomod_capital_cost_storage = "NemoMod CapitalCostStorage"
        self.modvar_enst_nemomod_discounted_capital_investment_storage = "NemoMod Discounted Capital Investment Storage"
        self.modvar_enst_nemomod_discounted_operating_costs_storage = "NemoMod Discounted Operating Costs Storage"
        self.modvar_enst_nemomod_residual_capacity = "NemoMod ResidualStorageCapacity"
        self.modvar_enst_nemomod_storage_start_level = "NemoMod StorageStartLevel"
        self.modvar_enst_nemomod_total_annual_max_capacity_storage = "NemoMod TotalAnnualMaxCapacityStorage"
        self.modvar_enst_nemomod_total_annual_max_capacity_investment_storage = "NemoMod TotalAnnualMaxCapacityInvestmentStorage"
        self.modvar_enst_nemomod_total_annual_min_capacity_storage = "NemoMod TotalAnnualMinCapacityStorage"
        self.modvar_enst_nemomod_total_annual_min_capacity_investment_storage = "NemoMod TotalAnnualMinCapacityInvestmentStorage"

        return None
        


    def _set_dict_enfu_fuel_categories_to_entc_variables(self,
        key_iar: str = "input_activity_ratio",
        key_mpi_frac_msp: str = "max_prod_increase_frac_msp",
        key_oar: str = "output_activity_ratio"
    ) -> None:
        """
        Set dictionaries mapping fuel categories to input variables in Energy 
            Technology. Sets the following properties:
        
            * self.dict_entc_fuel_categories_to_fuel_variables
            * self.dict_entc_fuel_categories_to_unassigned_fuel_variables
            * self.key_iar
            * self.key_oar

        Keyword Arguments
        -----------------
        - key_iar: key to use self.dict_entc_fuel_categories_to_fuel_variables
            for NemoMod InputActivityRatio variables associated with a fuel
            category. Set as self.key_iar
        - key_mpi_frac_msp: key to use 
            self.dict_entc_fuel_categories_to_fuel_variables for NemoMod 
            `Maximum Production Increase Fraction to Satisfy MinShareProduction`  
            variables associated with a fuel category. Set as 
            self.key_mpi_frac_msp
        - key_oar: key to use self.dict_entc_fuel_categories_to_fuel_variables
            for NemoMod OutputActivityRatio variables associated with a fuel
            category. Set as self.key_oar
        """

        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu,
             "pycategory_primary_element",
        ) 

        tuple_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.subsec_name_entc,
            pycat_enfu,
            {
                "Fuel Production NemoMod InputActivityRatio": key_iar,
                "Fuel Production NemoMod OutputActivityRatio": key_oar,
                "Maximum Production Increase Fraction to Satisfy MinShareProduction": key_mpi_frac_msp
            },
        )
        
        self.key_iar = key_iar
        self.key_oar = key_oar
        self.key_mpi_frac_msp = key_mpi_frac_msp
        self.dict_entc_fuel_categories_to_fuel_variables = tuple_out[0]
        self.dict_entc_fuel_categories_to_unassigned_fuel_variables = tuple_out[1]
        
        return None
    


    def transform_field_year_nemomod(self,
        vector: list,
        time_period_as_year: bool = None,
        direction: str = "to_nemomod",
        shift: int = 1000
    ) -> np.ndarray:
        """
        Transform a year field if necessary to ensure that the minimum is 
            greater than 0--NemoMod cannot handle a minimum of year == 0. Only 
            applied if time_period_as_year = True

        Function Arguments
        ------------------
        - vector: input vector of years

        Keyword Arguments
        -----------------
        - time_period_as_year: treat the time period as year in NemoMod?
        - direction: "to_nemomod" will transform the field to prepare it for 
            nemomod, while "from_nemomod" will prepapre it for SISEPUEDE
        - shift: integer shift to apply to field
            * NOTE: NemoMod apparently has a minimum year requirement. Default 
            of 1000 should protect against issues.
        """

        sf.check_set_values([direction], ["to_nemomod", "from_nemomod"])
        time_period_as_year = self.nemomod_time_period_as_year if (time_period_as_year is None) else time_period_as_year

        vector_out = np.array(vector)
        if time_period_as_year:
            vector_out = (vector_out + shift) if (direction == "to_nemomod") else (vector_out - shift)

        return vector_out






    ###############################################################
    #    GENERALIZED DATA FRAME CHECK FUNCTIONS FOR FORMATTING    #
    ###############################################################

    def add_index_field_from_key_values(self,
        df_input: pd.DataFrame,
        index_values: list,
        field_index: str,
        outer_prod: bool = True
    ) -> pd.DataFrame:
        """
        Add a field (if necessary) to input dataframe if it is missing based on 
            input index_values.

        Function Arguments
        ------------------
        - df_input: input data frame to modify
        - index_values: values to expand the data frame along
        - field_index: new field to add

        Keyword Arguments
        -----------------
        - outer_prod: assume data frame is repeated to all regions. If not, 
            assume that the index values are applied as a column only (must be 
            one element or of the same length as df_input)
        """

        field_dummy = "merge_key"

        # add the region field if it is not present
        if field_index not in df_input.columns:
            if len(df_input) == 0:
                df_input[field_index] = None
            elif outer_prod:
                # initialize the index values
                df_merge = pd.DataFrame({field_index: index_values})
                df_merge[field_dummy] = 0
                df_input[field_dummy] = 0
                # order columns and do outer product
                order_cols = list(df_input.columns)
                df_input = pd.merge(df_input, df_merge, on = field_dummy, how = "outer")
                df_input = df_input[[field_index] + [x for x in order_cols if (x != field_dummy)]]
            else:
                # check shape
                if (len(df_input) == len(index_values)) or (not (isinstance(index_values, list) or isinstance(index_values, np.ndarray))):
                    df_input[field_index] = index_values
                else:
                    raise ValueError(f"Error in add_index_field_from_key_values: invalid input shape in index_values. Set outer_prod = True to use outer product.")

        return df_input



    def add_index_field_fuel(self,
        df_input: pd.DataFrame,
        field_fuel: str = None,
        outer_prod: bool = True,
        restriction_fuels: list = None
    ) -> pd.DataFrame:
        """
        Add a fuel field (if necessary) to input dataframe if it is missing. 
            Defaults to all defined fuels, and assumes that the input data frame 
            is repeated across all fuels.

        Function Arguments
        ------------------
        - df_input: input data frame to add field to

        Keyword Arguments
        -----------------
        - field_fuel: the name of the field. Default is set to NemoMod naming 
            convention.
        - outer_prod: product against all fuels
        - restriction_fuels: subset of fuels to restrict addition to
        """

        field_fuel = self.field_nemomod_fuel if (field_fuel is None) else field_fuel

        # get regions
        fuels = self.model_attributes.get_attribute_table(self.subsec_name_enfu).key_values
        fuels = [x for x in fuels if x in restriction_fuels] if (restriction_fuels is not None) else fuels
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, fuels, field_fuel, outer_prod = outer_prod)

        return df_input



    def add_index_field_id(self,
        df_input: pd.DataFrame,
        field_id: str = None
    ) -> pd.DataFrame:
        """
            Add a the id field (if necessary) to input dataframe if it is missing.
        """
        field_id = self.field_nemomod_id if (field_id is None) else field_id

        # add the id field if it is not present
        if field_id not in df_input.columns:
            df_input[field_id] = range(1, len(df_input) + 1)

        # order columns and return
        order_cols = [field_id] + [x for x in list(df_input.columns) if (x != field_id)]
        df_input = df_input[order_cols]

        return df_input



    def add_index_field_region(self,
        df_input: pd.DataFrame,
        field_region: str = None,
        outer_prod: bool = True,
        restriction_regions: list = None
    ) -> pd.DataFrame:
        """
        Add a region field (if necessary) to input dataframe if it is missing. 
            Defaults to configuration regions, and assumes that the input data 
            frame is repeated across all regions.

        Function Arguments
        ------------------
        - df_input: input data frame to add field to

        Keyword Arguments
        -----------------
        - field_region: the name of the field. Default is set to NemoMod naming 
            convention.
        - outer_prod: product against all regions
        - restriction_regions: subset of regions to restrict addition to
        - restrict_to_config_region: only allow regions specified in the 
            configuration? Generally set to true, but can be set to false for 
            data construction
        """
        # get regions
        field_region = self.field_nemomod_region if (field_region is None) else field_region
        regions = self.model_attributes.get_region_list_filtered(restriction_regions)

        # add to output using outer product
        df_input = self.add_index_field_from_key_values(
            df_input, 
            regions, 
            field_region, 
            outer_prod = outer_prod
        )

        return df_input


 
    def add_index_field_technology(self,
        df_input: pd.DataFrame,
        field_technology: str = None,
        outer_prod: bool = True,
        restriction_technologies: list = None
    ) -> pd.DataFrame:
        """
        Add a technology field (if necessary) to input dataframe if it is 
            missing. Defaults to all defined technology, and assumes that the 
            input data frame is repeated across all technologies.

        Function Arguments
        ------------------
        - df_input: input data frame to add field to

        Keyword Arguments
        -----------------
        - field_technology: the name of the field. Default is set to NemoMod 
            naming convention.
        - outer_prod: product against all technologies
        - restriction_technologies: subset of technologies to restrict addition 
            to
        """

        field_technology = self.field_nemomod_technology if (field_technology is None) else field_technology

        # get regions
        techs = self.model_attributes.get_attribute_table(self.subsec_name_entc).key_values
        techs = [x for x in techs if x in restriction_technologies] if (restriction_technologies is not None) else techs
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, techs, field_technology, outer_prod = outer_prod)

        return df_input



    def add_index_field_year(self,
        df_input: pd.DataFrame,
        field_year: str = None,
        outer_prod: bool = True,
        restriction_years: list = None,
        time_period_as_year: bool = None,
        override_time_period_transformation: bool = False
    ) -> pd.DataFrame:
        """
        Add a year field (if necessary) to input dataframe if it is missing. 
            Defaults to all defined years (if defined in time periods), and 
            assumes that the input data frame is repeated across all years.

        Function Arguments
        ------------------
        - df_input: input data frame to add field to

        Keyword Arguments
        -----------------
        - field_year: the name of the field. Default is set to NemoMod naming 
            convention.
        - outer_prod: product against all years
        - restriction_years: subset of years to restrict addition to
        - time_period_as_year: If True, enter the time period as the year. If 
            None, default to ElectricEnergy.nemomod_time_period_as_year
        - override_time_period_transformation: In some cases, time periods 
            transformations can be applied multiple times from default 
            functions. Set override_time_period_transformation = True to remove 
            the application of ElectricEnergy.transform_field_year_nemomod()
        """

        time_period_as_year = (
            self.nemomod_time_period_as_year 
            if (time_period_as_year is None) 
            else time_period_as_year
        )

        field_year = (
            self.field_nemomod_year 
            if (field_year is None) 
            else field_year
        )

        # get time periods that are available
        years = (
            self.model_attributes.get_time_periods()[0] 
            if time_period_as_year 
            else self.model_attributes.get_time_period_years()
        )
        years = (
            [x for x in years if x in restriction_years] 
            if (restriction_years is not None) 
            else years
        )

        # add to output using outer product, then clean up the years for NemoMod to prevent any values of 0
        df_input = self.add_index_field_from_key_values(
            df_input, 
            years, 
            field_year, 
            outer_prod = outer_prod
        )
        if not override_time_period_transformation:
            df_input[field_year] = self.transform_field_year_nemomod(
                df_input[field_year], 
                time_period_as_year = time_period_as_year
            )

        return df_input



    def add_multifields_from_key_values(self,
        df_input_base: pd.DataFrame,
        fields_to_add: list,
        time_period_as_year: Union[bool, None] = None,
        override_time_period_transformation: Union[bool, None] = False,
        regions: Union[List[str], None] = None
    ) -> pd.DataFrame:
        """
        Add a multiple fields, assuming repitition of the data frame across 
            dimensions. Based on NemoMod defaults.

        Function Arguments
        ------------------
        - df_input_base: input data frame to add field to
        - fields_to_add: fields to add. Must be entered as NemoMod defaults.

        Keyword Arguments
        -----------------
        - override_time_period_transformation: override the time period 
            transformation step to prevent applying it twice
        - regions: regions to pass to add_index_field_region
        - time_period_as_year: enter values in field 
            ElectricEnergy.field_nemomod_year as time periods? If None, default 
            to ElectricEnergy.nemomod_time_period_as_year
        """

        time_period_as_year = (
            self.nemomod_time_period_as_year 
            if (time_period_as_year is None) 
            else time_period_as_year
        )
        override_time_period_transformation = (
            False 
            if (override_time_period_transformation is None) 
            else override_time_period_transformation
        )
        df_input = df_input_base.copy()

        # if id is in the table and we are adding other fields, rename it
        field_id_rnm = f"subtable_{self.field_nemomod_id}"
        if len([x for x in fields_to_add if (x != self.field_nemomod_id)]) > 0:
            df_input.rename(columns = {self.field_nemomod_id: field_id_rnm}, inplace = True) if (self.field_nemomod_id in df_input.columns) else None
        # ordered additions
        df_input = (
            self.add_index_field_technology(df_input) 
            if (self.field_nemomod_technology in fields_to_add) 
            else df_input
        )
        df_input = (
            self.add_index_field_fuel(df_input) 
            if (self.field_nemomod_fuel in fields_to_add) 
            else df_input
        )
        df_input = (
            self.add_index_field_region(
                df_input, 
                restriction_regions = regions
            )
            if (self.field_nemomod_region in fields_to_add) 
            else df_input
        )
        df_input = (
            self.add_index_field_year(
                df_input, 
                time_period_as_year = time_period_as_year, 
                override_time_period_transformation = override_time_period_transformation
            )
            if (self.field_nemomod_year in fields_to_add) 
            else df_input
        )

        # set sorting hierarchy, then drop original id field
        fields_sort_hierarchy = [x for x in self.fields_nemomod_sort_hierarchy if (x in fields_to_add) and (x != self.field_nemomod_id)]
        fields_sort_hierarchy = fields_sort_hierarchy + [field_id_rnm] if (field_id_rnm in df_input.columns) else fields_sort_hierarchy
        df_input = df_input.sort_values(by = fields_sort_hierarchy).reset_index(drop = True) if (len(fields_sort_hierarchy) > 0) else df_input
        df_input.drop([field_id_rnm], axis = 1, inplace = True) if (field_id_rnm in df_input.columns) else None

        # add the final id field if necessary
        df_input = self.add_index_field_id(df_input) if (self.field_nemomod_id in fields_to_add) else df_input
        df_input = df_input[[x for x in self.fields_nemomod_sort_hierarchy if x in df_input.columns]]

        return df_input



    def allocate_entc_emissions_by_energy_demand(self,
        df_elec_trajectories: pd.DataFrame,
        df_retrieval_trajectories: pd.DataFrame,
        cat_enfu_energy_source: Union[str, None] = None,
        dict_enfu_subsectors_to_energy_variables: Union[Dict, None] = None,
        dict_entc_subsectors_to_emission_variables: Union[Dict, None] = None,
        modvar_entc_nemomod_emissions_export_ch4: Union[str, None] = None,
        modvar_entc_nemomod_emissions_export_co2: Union[str, None] = None,
        modvar_entc_nemomod_emissions_export_n2o: Union[str, None] = None
    ) -> pd.DataFrame:
        """
        Allocate emissions from 

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of inputs to the electricity model
        - df_retrieval_trajectories: data frame of output trajectories from
            NemoMod containing energy and emissions data

        Keyword Arguments
        -----------------
        - cat_enfu_energy_source: optional category to use as source of fuel 
            demands. If None, defaults to self.cat_enfu_elec
        - dict_enfu_subsectors_to_energy_variables: dictionary mapping 
            subsectors to associated energy demand variables in ENFU (under key
            energy demand). See 
            ElectricEnergy.dict_enfu_subsectors_to_energy_variables (default if
            None) for structural example.
        - dict_entc_subsectors_to_emission_variables: dictionary mapping 
            subsectors to associated emission variables in ENTC (under key
            energy demand). See 
            ElectricEnergy.dict_entc_subsectors_to_emission_variables (default
            if None) for structural example.
        - modvar_entc_nemomod_emissions_export_ch4: model variable denoting 
            CH4 emissions attributable to exports.
        - modvar_entc_nemomod_emissions_export_co2: model variable denoting 
            CO2 emissions attributable to exports.
        - modvar_entc_nemomod_emissions_export_n2o: model variable denoting 
            N2O emissions attributable to exports.
        """

        ##  INITIALIZE SOME KEY VARS

        cat_enfu_energy_source = (
            self.cat_enfu_elec 
            if (cat_enfu_energy_source is None) 
            else cat_enfu_energy_source
        )
        dict_enfu_subsectors_to_energy_variables = (
            self.dict_enfu_subsectors_to_energy_variables
            if dict_enfu_subsectors_to_energy_variables is None
            else dict_enfu_subsectors_to_energy_variables
        )
        dict_entc_subsectors_to_emission_variables = (
            self.dict_entc_subsectors_to_emission_variables
            if dict_entc_subsectors_to_emission_variables is None
            else dict_entc_subsectors_to_emission_variables
        )

        # model variables
        modvar_entc_nemomod_emissions_export_ch4 = (
            self.modvar_entc_nemomod_emissions_export_ch4
            if modvar_entc_nemomod_emissions_export_ch4 is None
            else modvar_entc_nemomod_emissions_export_ch4
        )
        modvar_entc_nemomod_emissions_export_co2 = (
            self.modvar_entc_nemomod_emissions_export_co2
            if modvar_entc_nemomod_emissions_export_co2 is None
            else modvar_entc_nemomod_emissions_export_co2
        )
        modvar_entc_nemomod_emissions_export_n2o = (
            self.modvar_entc_nemomod_emissions_export_n2o
            if modvar_entc_nemomod_emissions_export_n2o is None
            else modvar_entc_nemomod_emissions_export_n2o
        )

        # get the fuel source index to use to allocate emissions
        attr_enfu = self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_enfu)
        ind_enfu_energy_source = attr_enfu.get_key_value_index(cat_enfu_energy_source)


        ##  BUILD PROPORTIONAL VECTOR OF ENERGY DEMANDS

        # get the exports of the fuel
        vec_enfu_exports = self.model_attributes.extract_model_variable(#
            df_retrieval_trajectories,
            self.modvar_enfu_exports_fuel_adjusted,
            expand_to_all_cats = True,
            return_type = "array_base",
        )[:, ind_enfu_energy_source]

        # get fuel imports of the selected source fuel
        vec_enfu_imports = self.model_attributes.extract_model_variable(#
            df_retrieval_trajectories,
            self.modvar_enfu_imports_fuel,
            expand_to_all_cats = True,
            return_type = "array_base",
        )[:, ind_enfu_energy_source]

        # initialize and calculate vector of electricity production
        vec_enfu_total_demand = -vec_enfu_imports

        # initialize dictionary of subsectors to vectors of proportional emission allocations
        dummy_cat_exports = "exports"
        dict_subsector_to_energy_demand_proportions = {}

        # subsectors to iterate over
        subsecs_iter = sorted(list(
            set(dict_enfu_subsectors_to_energy_variables.keys()) & 
            set(dict_entc_subsectors_to_emission_variables.keys())
        ))

        
        for subsec in subsecs_iter:

            # initialize the vector of fuel demand for the current subsector and check if energy variables are specified
            vec_demand_subsec_cur = None
            dict_energy_vars_cur = dict_enfu_subsectors_to_energy_variables.get(subsec)
     
            if dict_energy_vars_cur is not None:
                # try retrieving subsector demand from electric trajectories; if not there, look to retrieval (ENTC)
                # if neither works, save error and move on
                try: 
                    vec_demand_subsec_cur = self.model_attributes.extract_model_variable(#
                        df_elec_trajectories,
                        dict_energy_vars_cur.get("energy_demand"),
                        expand_to_all_cats = True,
                        return_type = "array_base",
                    )[:, ind_enfu_energy_source]
                        
                except:
                    try:
                        vec_demand_subsec_cur = self.model_attributes.extract_model_variable(#
                            df_retrieval_trajectories,
                            dict_energy_vars_cur.get("energy_demand"),
                            expand_to_all_cats = True,
                            return_type = "array_base",
                        )[:, ind_enfu_energy_source]

                    except:
                        self._log(
                            f"Error in `allocate_entc_emissions_by_energy_demand` retrieving energy demands for {cat_enfu_energy_source} subsector {subsec}. Emissions will not be allocated for this subsector. Skipping...", 
                            type_log = "error",
                        )

                if vec_demand_subsec_cur is not None:
                    vec_enfu_total_demand += vec_demand_subsec_cur
                    dict_subsector_to_energy_demand_proportions.update({subsec: vec_demand_subsec_cur})

        # convert to proportions - get total production
        vec_enfu_total_production = sf.vec_bounds(vec_enfu_total_demand, (0, np.inf))
        vec_enfu_total_production += vec_enfu_exports

        # emissions are allocated according to domestic production; exports - domestic demand (assume that imports are used homogenously and distributed proportional to subsector demands)
        vec_enfu_frac_from_exports = np.nan_to_num(vec_enfu_exports / vec_enfu_total_production, 0.0, posinf = 0.0)
        vec_enfu_frac_from_others = 1 - vec_enfu_frac_from_exports

        for subsec in dict_subsector_to_energy_demand_proportions.keys():
            vec = dict_subsector_to_energy_demand_proportions.get(subsec)
            vec *= vec_enfu_frac_from_others/(vec_enfu_total_demand + vec_enfu_imports)
            dict_subsector_to_energy_demand_proportions.update({subsec: vec})


        ##  BUILD OUTPUT

        df_out = []

        # loop over output emissions in electricity by gas - map emissions to export variable
        dict_modvars_emission_to_allocate = {
            self.modvar_entc_nemomod_emissions_ch4_elec: modvar_entc_nemomod_emissions_export_ch4,
            self.modvar_entc_nemomod_emissions_co2_elec: modvar_entc_nemomod_emissions_export_co2,
            self.modvar_entc_nemomod_emissions_n2o_elec: modvar_entc_nemomod_emissions_export_n2o
        }

        for modvar in dict_modvars_emission_to_allocate.keys():
            # get gas, pivot key, and export variable
            gas = self.model_attributes.get_variable_characteristic(
                modvar, 
                self.model_attributes.varchar_str_emission_gas
            )
            key_dict_emissions = f"emissions_{gas}"
            modvar_emissions_export = dict_modvars_emission_to_allocate.get(modvar)

            # get the total emissions in configuration units 
            vec_entc_emissions_total = self.model_attributes.extract_model_variable(#
                df_retrieval_trajectories,
                modvar,
                expand_to_all_cats = True,
                return_type = "array_base",
            ).sum(axis = 1)

            # allocate exports
            df_out.append(
                self.model_attributes.array_to_df(
                    vec_enfu_frac_from_exports*vec_entc_emissions_total, 
                    modvar_emissions_export
                )
            )

            # allocate ENTC emissions based on demands within each energy subsector
            for subsec in dict_entc_subsectors_to_emission_variables.keys():
                # get model variable and emissions 
                modvar_emission_cur_subsec = dict_entc_subsectors_to_emission_variables.get(subsec)
                modvar_emission_cur_subsec = modvar_emission_cur_subsec.get(key_dict_emissions) if (modvar_emission_cur_subsec is not None) else None
                vec_emissions = dict_subsector_to_energy_demand_proportions.get(subsec) if (modvar_emission_cur_subsec is not None) else None
        
                if vec_emissions is not None:
                    df_out.append(
                        self.model_attributes.array_to_df(
                            vec_emissions*vec_entc_emissions_total, 
                            modvar_emission_cur_subsec
                        )
                    )


        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        return df_out



    def build_dummy_tech_cost(self,
        price: Union[int, float],
        cost_type: str,
        attribute_technology: Union[AttributeTable, None] = None,
        override_time_period_transformation: Union[bool, None] = None
    ):
        """
        Build costs for dummy techs based on an input price.

        Function Arguments
        ------------------
        - price: variable cost to assign to dummy technologies. Should be large 
            relative to other technologies.
        - cost_type: one of
            * "capital": capital cost [t, y, val]
            * "fixed": fixed cost [t, y, val]
            * "variable": variable cost [t, y, mode, val]

        Keyword Arguments
        -----------------
        - attribute_technology: attribute table used to obtain dummy 
            technologies. If None, use ModelAttributes default.
        """
        # some attribute initializations
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        dict_tech_info = self.get_tech_info_dict(attribute_technology = attribute_technology)
        cost_type = cost_type if (cost_type in ["capital", "fixed", "variable"]) else "variable"

        df_out = {
            self.field_nemomod_technology: dict_tech_info.get("all_techs_dummy"),
            self.field_nemomod_value: price
        }
        df_out.update({self.field_nemomod_mode: self.cat_enmo_gnrt}) if (cost_type == "variable") else None
        df_out = pd.DataFrame(df_out)

        # order and add multifields
        fields_for_multifield = [
            self.field_nemomod_technology,
            self.field_nemomod_year,
            self.field_nemomod_value
        ]
        fields_for_multifield.append(self.field_nemomod_mode) if (cost_type == "variable") else None
        df_out = self.add_multifields_from_key_values(
            df_out, 
            fields_for_multifield,
            override_time_period_transformation = override_time_period_transformation
        )

        return df_out



    def conflict_resolution_func_vmmci(self,
        mm_tuple: tuple,
        approach: str = "swap",
        inequality_strength: str = "weak",
        max_min_distance_scalar = 1
    ) -> float:

        """
        Used in verify_min_max_constraint_inputs. Input a tuple of (min, max) 
            and resolve conflicting inputs if needed.

        Function Arguments
        ------------------
        - mm_tuple: tuple of (min, max) to resolve

        Keyword Arguments
        -----------------
        - appraoch: how to resolve the conflict
        - inequality_strength: "weak" or "strong". Weak comparison means that 
            min <= max is acceptable. Strong comparison means that 
            min < max must hold true.
        - max_min_distance_scalar: max ~ min*max_min_distance_scalar,
             where ~ = > or >=. Must be at least one.
        """

        max_spec = mm_tuple[1]
        min_spec = mm_tuple[0]
        resolve_q = (min_spec > max_spec) if (inequality_strength == "weak") else (min_spec >= max_spec)
        max_min_distance_scalar = max(max_min_distance_scalar, 1)
        # set the output min/max
        min_true = min(mm_tuple)
        max_true = max(max(mm_tuple), max_min_distance_scalar*min_true)
        out = mm_tuple

        if resolve_q:
            if approach == "swap":
                out = (min_true, max_true)
            elif approach == "max_sup":
                out = (max_true, max_true)
            elif approach == "min_sup":
                out = (min_true, min_true)
            elif approach == "mean":
                mean_true = (max_true + min_true)/2
                out = (mean_true, mean_true)
            elif approach == "keep_max_input":
                out = (max_spec, max_spec)
            elif approach == "keep_min_input":
                out = (min_spec, min_spec)

        return out
    


    def get_attribute_emission(self,
    ) -> AttributeTable:
        """
        Shortcut to get the emission attribute table
        """

        out = (
            self.model_attributes
            .get_other_attribute_table("emission_gas")
            .attribute_table
        )

        return out



    def get_attribute_enfu(self,
        **kwargs
    ) -> AttributeTable:
        """
        Shortcut to get the Energy Fuels attribute table
        """

        out = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_enfu,
            **kwargs
        )

        return out
    


    def get_attribute_entc(self,
        **kwargs
    ) -> AttributeTable:
        """
        Shortcut to get the Energy Technology attribute table
        """

        out = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_entc,
            **kwargs
        )

        return out



    def get_attribute_region(self,
    ) -> AttributeTable:
        """
        Shortcut to get the region table
        """

        out = self.model_attributes.get_other_attribute_table(
            self.model_attributes.dim_region,
        )

        return out



    def get_attribute_time_period(self,
    ) -> AttributeTable:
        """
        Shortcut to get the time period attribute table
        """

        out = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_time_period,
        )

        return out



    def get_biogas_components(self,
        df_elec_trajectories: pd.DataFrame
    ) -> tuple:

        """
        Retrieve total energy available from biogas collection and the minimum 
            use

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of input variables, which must 
            include livestock manure management and wastewater treatment sector 
            outputs used to calcualte emission factors
        """
        # initialize of some variables
        vec_enfu_total_energy_biogas = 0.0
        modvars_biogas = [
            self.model_afolu.modvar_lsmm_recovered_biogas,
            self.model_circecon.modvar_trww_recovered_biogas,
            self.model_circecon.modvar_waso_recovered_biogas_anaerobic,
            self.model_circecon.modvar_waso_recovered_biogas_landfills
        ]

        # get gravimetric density (aka specific energy)
        vec_enfu_energy_density_gravimetric = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.modvar_enfu_energy_density_gravimetric,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )
        vec_enfu_energy_density_gravimetric = vec_enfu_energy_density_gravimetric[:, self.ind_enfu_bgas]

        # get minimum fuel fraction to electricity
        vec_enfu_minimum_fuel_frac_to_elec = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.modvar_enfu_minimum_frac_fuel_used_for_electricity,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        vec_enfu_minimum_fuel_frac_to_elec = vec_enfu_minimum_fuel_frac_to_elec[:, self.ind_enfu_bgas]

        # iterate to add total biogas collected
        for modvar in modvars_biogas:
            # retrieve biogas totals
            tuple_biogas = self.model_attributes.get_optional_or_integrated_standard_variable(
                df_elec_trajectories,
                modvar,
                None,
                override_vector_for_single_mv_q = True,
                return_type = "array_base"
            )

            #
            if tuple_biogas is not None:
                # get mass of waste incinerated,
                modvar_biogas, array_mass_biogas = tuple_biogas
                vec_mass_biogas = np.sum(array_mass_biogas, axis = 1)

                # convert units -- first, in terms of mass incinerated, then in terms of energy density
                vec_enfu_energy_density_cur =  vec_enfu_energy_density_gravimetric/self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_energy_density_gravimetric,
                    modvar_biogas,
                    "mass"
                )
                vec_enfu_energy_density_cur *= self.get_nemomod_energy_scalar(self.modvar_enfu_energy_density_gravimetric)
                vec_enfu_total_energy_biogas += vec_enfu_energy_density_cur*vec_mass_biogas

        # get minimum fraction to electricity
        vec_enfu_minimum_fuel_energy_to_electricity_biogas = vec_enfu_total_energy_biogas*vec_enfu_minimum_fuel_frac_to_elec

        out = (vec_enfu_total_energy_biogas, vec_enfu_minimum_fuel_energy_to_electricity_biogas)

        return out



    def get_dummy_fuel_description(self,
        return_type: str = "fuel",
    ) -> str:
        """
        Provide a description of the dummy fuel. Set return_type = "tech" to
            generate for the supply tech.
        """
        return_val = "Dummy fuel used for accounting of production by dummy techs." if (
            return_type == "fuel"
        ) else "Required technology that provides dummy fuel used to account for dummy fuel techs."

        return return_val



    def get_dummy_fuel_name(self,
        return_type: str = "fuel"
    ) -> str:
        """
        To accurately track productionbytechnology from dummy (supply) techs,
            a dummy fuel is used, defined here. Set return_type = "tech" to
            generate the name for the supply tech.
        """
        return_val = "fuel_QUANTITY_PRODUCED" if (return_type == "fuel") else "supply_QUANTITY_PRODUCED"

        return return_val



    def get_dummy_fuel_tech_name(self, 
        fuel: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Map a fuel to a dummy technology that generates that fuel. Returns a 
            name for the dummy fuel tech.
        """
        return_val = f"supply_{fuel}" if isinstance(fuel, str) else [f"supply_{x}" for x in fuel]
        return return_val



    def get_dummy_fuel_techs(self,
        attribute_fuel: Union[AttributeTable, None] = None,
        drop_activity_ratio_fuels: Union[bool, None] = None,#True
        return_type: str = "dict"
    ) -> Dict:
        """
        Retrieve fuels that require dummy supply technologies. NOTE: swap
            default value of `drop_activity_ratio_fuels` to False to include
            dummy techs for *all* fuels. 
        
        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used to define universe of fuels. If 
            None, uses self.model_attributes default
        - drop_activity_ratio_fuels: filter out dummy techs that are associated
            with activity ratios? If True, drops dummy techs associated with 
            explicitly defined activity ratios. Otherwise, includes dummy techs
            for all fuels (used as emergencyies in the model to deal with
            otherwise infeasible problems). If None, defaults to 
            (not self.include_supply_techs_for_all_fuels)
        - return_type: "dict" or "pd.DataFrame". Default is dictionary.
        """
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        drop_activity_ratio_fuels = (
            (not self.include_supply_techs_for_all_fuels) 
            if (drop_activity_ratio_fuels is None) 
            else drop_activity_ratio_fuels
        )

        return_type = return_type if (return_type in ["dict", "pd.DataFrame"]) else "dict"

        fuels_from_dummy = [] if drop_activity_ratio_fuels else attribute_fuel.key_values

        if drop_activity_ratio_fuels:
            for fuel in attribute_fuel.key_values:
                ar_vars = self.dict_entc_fuel_categories_to_fuel_variables.get(fuel)
                use_dummy = (
                    (self.key_oar not in ar_vars.keys()) and (fuel != self.cat_enfu_elec) 
                    if (ar_vars is not None) 
                    else True
                )
                
                fuels_from_dummy.append(fuel) if use_dummy else None
        
        dict_fuels_dummy = dict((x, self.get_dummy_fuel_tech_name(x)) for x in fuels_from_dummy)
        dict_fuels_dummy = (
            pd.DataFrame(
                self.get_dummy_fuel_techs().items(), 
                columns = [self.field_nemomod_fuel, self.field_nemomod_technology]
            ) 
            if (return_type == "pd.DataFrame") 
            else dict_fuels_dummy
        )

        return dict_fuels_dummy



    def get_dummy_generation_fuel_techs(self,
        attribute_technology: AttributeTable = None
    ) -> dict:
        """
        Get a dictionary mapping fuels mapped to powerplant generation to 
            associated dummy techs

        Function Arguments
        ------------------
        - attribute_technology: AttributeTable for technology, used to identify 
            operational lives of generation and storage technologies. If None, 
            use ModelAttributes default.
        """
        # get some defaults
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        # map generation techs to dummy supplies
        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            f"electricity_generation_{pycat_enfu}",
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        fuels_keep = sorted([x for x in dict_techs_to_fuel.values() if (x != self.cat_enfu_elec)])
        techs_dummy = [self.get_dummy_fuel_tech_name(x) for x in fuels_keep]

        dict_return = dict(zip(fuels_keep, techs_dummy))

        return dict_return



    def get_enfu_cats_with_high_dummy_tech_costs(self,
        imports_only: bool = False,
        modvar_fuel_import_fraction: Union[str, None] = None,
        return_type: str = "fuels"
    ) -> float:
        """
        Energy Fuel categories to endogenize imports. In NemoMod, these are 
            modeled as dummy technologies that have very high costs coupled with 
            a MinimumShareProduction. Includes fields that are (1) associated 
            with imports or (2) are associated with a specified 
            OutputActivityRatio.
        
        Keyword Arguments
        -----------------
        - imports_only: only return high dummy techs associated with imports
        - modvar_fuel_import_fraction: model variable used to specify which 
            fuels are associtated with imports. If None, defaults to 
            self.modvar_enfu_frac_fuel_demand_imported
        - return_type: return "fuels" or "dummy_fuel_techs"
        """
        # check model variable to use
        modvar_fuel_import_fraction = (
            self.modvar_enfu_frac_fuel_demand_imported 
            if (modvar_fuel_import_fraction is None) 
            else modvar_fuel_import_fraction
        )
        return_type = (
            return_type 
            if (return_type in ["fuels", "dummy_fuel_techs"]) 
            else "fuels"
        )

        # get enfu categories from import fraction, then integrate those associated with output activity ratios
        cats_price_high = self.model_attributes.get_variable_categories(modvar_fuel_import_fraction)
        
        if not imports_only:
            dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables
            cats_price_high += [x for x in dict_fuel_cats.keys() if dict_fuel_cats.get(x).get(self.key_oar) is not None]
            cats_price_high = list(set(cats_price_high))
        
        # convert to fuel-techs if needed
        cats_price_high = (
            self.get_dummy_fuel_tech_name(cats_price_high) 
            if (return_type == "dummy_fuel_techs") 
            else cats_price_high
        )
        cats_price_high.sort()

        return cats_price_high


    
    def get_enfu_fuel_production_from_total_production(self,
        df_production_by_technology: pd.DataFrame,
        vector_reference_time_period: Union[list, np.ndarray],
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        dict_tech_info: Union[Dict, None] = None,
        dict_upstream: Union[Dict, None] = None,
        modvar_enfu_production: Union[str, None] = None
    ) -> pd.DataFrame:
        """
        Using the vproductionbytechnologyannual table, account for 
            upstream/downstream fuel swaps and imports to generate fuel 
            production.

        Function Arguments
        ------------------
        - df_production_by_technology: data frame of production by technology
            table pulled from NemoMod database
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used to define universe of fuels. If 
            None, uses self.model_attributes defaults
        - attribute_technology: attribute table used to obtain dummy 
            technologies. If None, use ModelAttributes default.
        - dict_tech_info: dictionary of technology info called from
            self.get_tech_info_dict()
        - dict_upstream: dictionary generated by 
            get_enfu_upstream_fuel_to_replace_downstream_fuel_consumption_map()
            with return_type = "dict_reverse"
        - modvar_enfu_production: model variable denoting regional fuel 
            production
        """
        
        # initialize some key components
        dict_upstream = self.get_enfu_upstream_fuel_to_replace_downstream_fuel_consumption_map(
            attribute_fuel = attribute_fuel,
            return_type = "dict_reverse"
        ) if (dict_upstream is None) else None
        dict_tech_info = self.get_tech_info_dict(
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology
        ) if (dict_tech_info is None) else dict_tech_info

        modvar_enfu_production = (
            self.modvar_enfu_production_fuel 
            if (modvar_enfu_production is None) 
            else modvar_enfu_production
        )

        # drop imports
        cats_entc_dummy_imports = self.get_enfu_cats_with_high_dummy_tech_costs(
            imports_only = True, 
            return_type = "dummy_fuel_techs"
        )
        cats_entc_dummy_drop = [self.get_dummy_fuel_name(return_type = "tech")]
        df_fuel_production = df_production_by_technology[
            ~df_production_by_technology[self.field_nemomod_technology].isin(cats_entc_dummy_imports + cats_entc_dummy_drop)
        ]

        # replace downstream fuels
        flag_drop = "DROPME"
        for cat_enfu_upstream in dict_upstream.keys():
            cat_enfu_downstream = dict_upstream.get(cat_enfu_upstream)
            df_fuel_production[self.field_nemomod_fuel].replace(
                {
                    cat_enfu_downstream: flag_drop, 
                    cat_enfu_upstream: cat_enfu_downstream
                }, 
                inplace = True
            )

        df_fuel_production = df_fuel_production[df_fuel_production[self.field_nemomod_fuel] != flag_drop].reset_index(drop = True)

        # convert to sisepuede format
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_production_fuel)
        df_fuel_production_out = self.retrieve_and_pivot_nemomod_table(
            df_fuel_production,
            modvar_enfu_production,
            None,
            vector_reference_time_period,
            dict_agg_info = {
                "fields_group": [
                    self.field_nemomod_fuel,
                    self.field_nemomod_region,
                    self.field_nemomod_year
                ],
                "dict_agg": {
                    self.field_nemomod_value: "sum"
                }
            },
            field_pivot = self.field_nemomod_fuel,
            techs_to_pivot = None
        )
        df_fuel_production_out /= scalar_div if (scalar_div is not None) else 1

        return df_fuel_production_out



    def get_enfu_non_entc_fuel_demands_from_annual_demand(self,
        df_demand_annual: pd.DataFrame,
        arr_enfu_exports: np.ndarray,
        vector_reference_time_period: Union[list, np.ndarray],
        modvar_enfu_demand: Union[str, None] = None,
        return_type: str = "array_base"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Using the vdemandannualnn table, account for 
            upstream/downstream fuel swaps and imports to generate fuel 
            production.

        Function Arguments
        ------------------
        - df_demand_annual: data frame of annual demand (non-nodal) extracted
            from NemoMod outputs
        - arr_enfu_exports: array of exports,
            * UNITS: NEMO MOD UNITS (may need to adjust)
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - modvar_enfu_demand: model variable used for Total Fuel Demand
        - return_type: 
            * "array_base": return an array (expanded to all categories) of 
                demand in NEMOMOD ENERGY UNITS
            * "data_frame": return a dataframe in energy units of 
                modvar_enfu_demand
        """

        modvar_enfu_demand = self.modvar_enfu_energy_demand_by_fuel_total if (modvar_enfu_demand is None) else modvar_enfu_demand
        return_type = "array_base" if (return_type not in ["array_base", "data_frame"]) else return_type

        # pivot table, capture full demand (demand + exports), then subtract out exports
        df_demands_with_exports = self.retrieve_and_pivot_nemomod_table(
            df_demand_annual,
            modvar_enfu_demand,
            None,
            vector_reference_time_period,
            field_pivot = self.field_nemomod_fuel,
            techs_to_pivot = None
        )

        arr_enfu_full_demand = self.model_attributes.extract_model_variable(#
            df_demands_with_exports,
            modvar_enfu_demand,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        arr_enfu_demand_non_entc = sf.vec_bounds(arr_enfu_full_demand - arr_enfu_exports, (0, np.inf))
        scalar_for_var = self.get_nemomod_energy_scalar(modvar_enfu_demand)

        arr_return = (
            self.model_attributes.array_to_df(
                arr_enfu_demand_non_entc,
                modvar_enfu_demand,
                reduce_from_all_cats_to_specified_cats = True,
            )/scalar_for_var
            if (return_type == "data_frame")
            else arr_enfu_demand_non_entc
        )

        return arr_return


    
    def get_enfu_upstream_fuel_to_replace_downstream_fuel_consumption_map(self,
        allow_oar_definition: bool = False, 
        attribute_fuel: Union[AttributeTable, None] = None,
        return_type: str = "dict"
    ) -> Union[Dict, List]:
        """
        Get a dictionary mapping an upstream fuel to the downstream fuel
            that it replaces in ENTC fuel demands. E.g., coal_deposits 
            replace coal as a fuel (since coal_deposits are used to represent
            the feedback between coal use in coal mining)
            
        Function Arguments
        ------------------
    

        Keyword Arguments
        -----------------
        - allow_oar_definition: allow the map to define a map if the upstream
            fuel is also defined in an OutputActivityRatio variable.
        - attribute_fuel: attribute table used for fuels. If None,
                defaults to self.model_attributes
        - return_type: return one of several options
            * "dict": dictionary mapping fuel to upstream fuel
            * "dict_reverse": dictionary mapping upstream fuel to downstream
            * "upstream_fuels": list of upstream fuels
        """
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        dict_upstream_fuel = attribute_fuel.field_maps.get(f"{attribute_fuel.key}_to_{self.model_attributes.field_enfu_upstream_to_fuel_category}")
        dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables
        
        dict_return = {}
        
        for cat_enfu in dict_upstream_fuel.keys():
            try_upstream_cat_enfu = clean_schema(dict_upstream_fuel.get(cat_enfu))

            if try_upstream_cat_enfu in attribute_fuel.key_values:
                # filter on input activity ratio (iar)
                # additionally, look for fuels that have no output activity ratio defined (they should be sourced from dummy techs)
                dict_iar_oar_var = dict_fuel_cats.get(cat_enfu)
                modvar_iar = dict_iar_oar_var.get(self.key_iar)
                modvar_oar = dict_iar_oar_var.get(self.key_oar)
                
                if (modvar_oar is None) or allow_oar_definition:
                    # if there is a tech defined with cat_enfu as the input fuel and try_upstream_cat_enfu as the output, assume conditions are met
                    dict_iar_oar_var = dict_fuel_cats.get(try_upstream_cat_enfu)
                    modvar_oar_try = dict_iar_oar_var.get(self.key_oar)
                    cats_shared = set(self.model_attributes.get_variable_categories(modvar_iar)) & set(self.model_attributes.get_variable_categories(modvar_oar_try))

                    dict_return.update({try_upstream_cat_enfu: cat_enfu}) if (len(cats_shared) > 0) else None
        
        dict_return = sf.reverse_dict(dict_return) if (return_type in ["dict_reverse", "upstream_fuels"]) else dict_return
        dict_return = list(dict_return.keys()) if (return_type in ["upstream_fuels"]) else dict_return
        
        return dict_return
        


    def get_entc_cat_by_type(self,
        type_tech: Union[str, list],
        attribute_technology:  Union[AttributeTable, None] = None
    ) -> Union[List[str], None]:
        """
        Retrieve an ordered (according to attribute_technology.key_values) list of 
            technologies by type. Types are:
        
            * `fp` or `fuel_processing`: non-electricity fuel processing 
                technologies
            * `me` or `mining_and_extraction`: technologies that represent 
                mining and extraction activities
            * `pp` or `power_plant`: electricity generation technologies
            * `st` or `storage`: technologies used to store electricity
            
        Function Arguments
        ------------------
        - type_tech: type of tech to generate list for (or list of tech types)
        
        Keyword Arguments
        -----------------
        - attribute_technology:  AttributeTable storing technology keys
        """
        
        type_tech = [type_tech] if isinstance(type_tech, str) else type_tech

        techs = []
        for tt in type_tech: 
            # get partition field, then grab categories
            tt_filt = self.get_entc_partition_field(tt)
            techs += (
                self.model_attributes.filter_keys_by_attribute(
                    self.subsec_name_entc,
                    {tt_filt: 1}
                ) 
                if (tt_filt is not None) 
                else []
            )

        techs = None if (len(techs) == 0) else techs

        return techs



    def get_entc_cat_for_integration(self,
        cat_name: str
    ) -> str:
        """
        Get the ENTC category used for waste incineration or biogas (integrated 
            with outputs from CircularEconomy)

        Function Arguments
        ------------------
        - cat_name: 
            * "biogas", or "bgas"
            * "hydropower" or "hpwr"
            * "waste", "wste"
        """
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.model_attributes.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        dict_cat_name_to_cat = {
            "bgas": self.cat_enfu_bgas,
            "biogas": self.cat_enfu_bgas,
            "hpwr": self.cat_enfu_hpwr,
            "hydropower": self.cat_enfu_hpwr,
            "waste": self.cat_enfu_wste,
            "wste": self.cat_enfu_wste,
        }

        cat_name = (
            cat_name 
            if (cat_name in dict_cat_name_to_cat.keys()) 
            else "waste"
        )
        cat_switch = dict_cat_name_to_cat.get(cat_name)

        cat_out = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_entc,
            {f"electricity_generation_{pycat_enfu}": unclean_category(cat_switch)} 
        )[0]

        return cat_out

    

    def get_entc_emissions_activity_ratio_comp_fp(self,
        df_elec_trajectories: pd.DataFrame,
        list_entc_modvars_fp_ear: Union[List[str], None] = None,
        regions: Union[List[str], None] = None
    ) -> Union[pd.DataFrame, None]:
        """
        Get emissions activity ratios for fuel production, which are explicitly 
            defined.
        
        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - dict_enfu_arrs_efs_scaled_to_nemomod: dictionary mapping combustion 
            factor variables to arrays adjusted EmissionsActivityRatio factors 
            (NemoMod emission mass/NemoMod unit energy)
        
        Keyword Arguments
        -----------------
        - attribute_fuel: attribute table used for fuels. If None,
                defaults to self.model_attributes
        - attribute_technology: attribute table used for technology. If None,
            defaults to self.model_attributes
        - attribute_time_period: attribute table used for time period. If None,
                defaults to self.model_attributes
        - list_entc_modvars_fp_ear: list of ENTC emissions activity ratio model 
            variables by gas.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """
        
        #
        list_entc_modvars_fp_ear = [
            self.modvar_entc_fuelprod_emissions_activity_ratio_ch4,
            self.modvar_entc_fuelprod_emissions_activity_ratio_co2,
            self.modvar_entc_fuelprod_emissions_activity_ratio_n2o
        ] if (list_entc_modvars_fp_ear is None) else list_entc_modvars_fp_ear

        
        ##  iterate over model variables
        
        df_out = []

        for modvar in list_entc_modvars_fp_ear:
            
            # get gas and scalar to correct units
            emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            scalar_correct = self.model_attributes.get_scalar(modvar, "mass")
            scalar_correct /= self.get_nemomod_energy_scalar(modvar)
            
            # get the model variable and pivot
            df_tmp = self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                modvar,
                "TMP",
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                regions = regions,
                scalar_to_nemomod_units = scalar_correct,
                var_bounds = (0, np.inf)
            ).get("TMP")
            
            # add emission variable and append
            df_tmp[self.field_nemomod_emission] = emission
            df_tmp[self.field_nemomod_mode] = self.cat_enmo_gnrt
            
            df_out.append(df_tmp)
            
            
        ##  prepare output dataframe
        df_out = self.add_multifields_from_key_values(
            pd.concat(df_out, axis = 0),
            [
                self.field_nemomod_id,
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_value,
                self.field_nemomod_year
            ],
            override_time_period_transformation = True,
            regions = regions
        )

        return df_out



    def get_entc_emissions_activity_ratio_comp_me(self,
        df_elec_trajectories: pd.DataFrame,
        dict_enfu_arrs_efs_scaled_to_nemomod: Dict[str, np.ndarray],
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        attribute_time_period: Union[AttributeTable, None] = None,
        regions: Union[List[str], None] = None
    ) -> Union[pd.DataFrame, None]:
        """
        Get emissions activity ratios for mining and extraction activities in 
            fuel produce. Assumes that input fuels are combusted. "Upstream 
            Fuel" categories are assumed to be used as passthroughs, and 
            InputActivityRatio values of greater than one are assumed to 
            represent energy inputs of (IAR - 1) for that fuel.
        
        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - dict_enfu_arrs_efs_scaled_to_nemomod: dictionary mapping combustion 
            factor variables to arrays adjusted EmissionsActivityRatio factors 
            (NemoMod emission mass/NemoMod unit energy)
        
        Keyword Arguments
        -----------------
        - attribute_fuel: attribute table used for fuels. If None,
                defaults to self.model_attributes
        - attribute_technology: attribute table used for technology. If None,
            defaults to self.model_attributes
        - attribute_time_period: attribute table used for time period. If None,
                defaults to self.model_attributes
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """
        
        # get some information
        attribute_fuel = (
            self.get_attribute_enfu()
            if not isinstance(attribute_fuel, AttributeTable) 
            else attribute_fuel
        )

        attribute_technology = (
            self.get_attribute_entc()
            if not isinstance(attribute_technology, AttributeTable) 
            else attribute_technology
        )

        attribute_time_period = (
            self.get_attribute_time_period()
            if not isinstance(attribute_time_period, AttributeTable) 
            else attribute_time_period
        )

        dict_tech_info = self.get_tech_info_dict()
        dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables


        ##  build dictionary of me techs to input fuels, which will be iterated over to gnerated emissions

        dict_enfu_arrs_efs = {}
        dict_enfu_arrs_iar = {}
        dict_entc_me_tech_to_input_fuels = {}

        for fuel in dict_fuel_cats.keys():
            # filter on input activity ratio (iar)
            dict_iar_oar_var = dict_fuel_cats.get(fuel)
            modvar_iar = dict_iar_oar_var.get(self.key_iar)

            if modvar_iar is not None:
                # check variable techs against me techs; if present, update dictionary
                cats_entc = self.model_attributes.get_variable_categories(modvar_iar)

                for cat in cats_entc:
                    if cat in dict_tech_info.get("all_techs_me"):
                        
                        (
                            dict_entc_me_tech_to_input_fuels[cat].append(fuel) 
                            if (cat in dict_entc_me_tech_to_input_fuels.keys()) 
                            else dict_entc_me_tech_to_input_fuels.update({cat: [fuel]})
                        )
                        
                        # retrieve array
                        if (cat not in dict_enfu_arrs_iar.keys()):
                            arr_entc_iar = self.model_attributes.extract_model_variable(#
                                df_elec_trajectories,
                                modvar_iar,
                                expand_to_all_cats = True,
                                return_type = "array_base", 
                            )
                            dict_enfu_arrs_iar.update({modvar_iar: arr_entc_iar})

                            
        ##  next, iterate over each tech to get IAR
        
        # get emission factor gasses
        dict_enfu_ef_to_gas = {}
        for modvar in dict_enfu_arrs_efs_scaled_to_nemomod.keys():
            emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            dict_enfu_ef_to_gas.update({modvar: emission})
            
        # find upstream fuel determinations and initialize output dataframe
        dict_upstream_fuel = attribute_fuel.field_maps.get(f"{attribute_fuel.key}_to_{self.model_attributes.field_enfu_upstream_to_fuel_category}")
        df_out = []
        
        # loop over map of mining and edtraction techs to input fuels
        for cat_entc in dict_entc_me_tech_to_input_fuels.keys():
            
            cats_enfu_iar = dict_entc_me_tech_to_input_fuels.get(cat_entc)
            ind_entc = attribute_technology.get_key_value_index(cat_entc)
            
            dict_enfu_vec_fuel_total = dict((dict_enfu_ef_to_gas.get(x), 0.0) for x in dict_enfu_arrs_efs_scaled_to_nemomod.keys())
            
            for cat_enfu in cats_enfu_iar:
                ind_enfu = attribute_fuel.get_key_value_index(cat_enfu)
                # get model variable and determine if the fuel is an "upstream fuel"
                # if upstream w/iar >= 1, assume that the me_tech is a pass through
                # e.g., coal_deposits
                modvar_iar = dict_fuel_cats.get(cat_enfu)
                modvar_iar = modvar_iar.get(self.key_iar)

                # try to see if the specified downstream fuel is produced by the current technology
                upstream_q = cat_enfu in self.get_enfu_upstream_fuel_to_replace_downstream_fuel_consumption_map(
                    attribute_fuel = attribute_fuel, 
                    return_type = "upstream_fuels"
                )
                
                arr_entc_iar = dict_enfu_arrs_iar.get(modvar_iar)
                vec_entc_iar = arr_entc_iar[:, ind_entc]
                vec_entc_iar = sf.vec_bounds(vec_entc_iar - 1, (0, np.inf)) if upstream_q else vec_entc_iar
                
                for modvar_ef in dict_enfu_arrs_efs_scaled_to_nemomod.keys():
                    vec_enfu_fuel_ef = dict_enfu_arrs_efs_scaled_to_nemomod.get(modvar_ef)
                    vec_enfu_fuel_ef = vec_enfu_fuel_ef[:, ind_enfu]
                    emission = dict_enfu_ef_to_gas.get(modvar_ef)
                    
                    dict_enfu_vec_fuel_total[emission] += vec_enfu_fuel_ef*vec_entc_iar
                

            # convert to data frame
            df_entc_tmp = pd.DataFrame(dict_enfu_vec_fuel_total)
            df_entc_tmp = self.model_attributes.exchange_year_time_period(
                df_entc_tmp,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period,
                direction = self.direction_exchange_year_time_period
            )
            
            # make long and add to output
            df_entc_tmp = pd.melt(
                df_entc_tmp,
                [self.field_nemomod_year],
                list(dict_enfu_vec_fuel_total.keys()),
                var_name = self.field_nemomod_emission,
                value_name = self.field_nemomod_value
            )
            df_entc_tmp[self.field_nemomod_mode] = self.cat_enmo_gnrt
            df_entc_tmp[self.field_nemomod_technology] = cat_entc
            
            df_out.append(df_entc_tmp)
            
        
        ##  prepare output dataframe
        df_out = self.add_multifields_from_key_values(
            pd.concat(df_out, axis = 0),
            [
                self.field_nemomod_id,
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_value,
                self.field_nemomod_year
            ],
            regions = regions
        )

        return df_out


    
    def get_entc_import_adjusted_msp(self,
        df_elec_trajectories: pd.DataFrame,
        arr_enfu_import_fractions_adj: np.ndarray,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        dict_tech_info: Union[Dict, None] = None,
        drop_flag: Union[float, int] = None,
        modvar_maxprod_msp_increase: Union[str, None] = None,
        modvar_msp: Union[str, None] = None,
        regions: Union[List[str], None] = None,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Minimum share of production is used to specify import fractions; this 
            function adjusts exogenous MinShareProduction fractions to account 
            for imports, returning a data frame that is ready to integrate into 
            the NemoMod input table.
        
        Called within the `.format_nemomod_table_min_share_production()` method.
        
        Returns a formatted dataframe.
           
        
        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - arr_enfu_import_fractions_adj: np.ndarray, wide by all fuel 
            categories, of adjusted import fractions (after accounting for the 
            integration of exports into SpecifiedAnnualDemands). This array is 
            used to adjust downward the exogenous specifications of 
            MinShareProduction.

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuel
        - attribute_technology: AttributeTable used to denote technologies with 
            MinShareProductions
        - dict_tech_info: optional tech_info dictionary specified by
            .get_tech_info_dict() method (can be passed to reduce computation)
        - drop_flag: optional specification of a drop flag used to indicate 
            rows/variables for which the MSP Max Production constraint is not 
            applicable (see 
            ElectricEnergy.get_entc_maxprod_increase_adjusted_msp for more 
            info). Defaults to self.drop_flag_tech_capacities if None.
        - modvar_maxprod_msp_increase: SISEPUEDE model variable storing the 
            maximum production increase (as a fraction of estimated last period
            with free production) allowable due to exogenous MinShareProduction
        - modvar_msp: model variable used to specify MinShareProduction (ENTC)
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        """
        # do some initialization from inputs
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        dict_tech_info = (
            self.get_tech_info_dict(attribute_technology = attribute_technology) 
            if (dict_tech_info is None) 
            else dict_tech_info
        )

        modvar_msp = self.modvar_entc_nemomod_min_share_production if (modvar_msp is None) else modvar_msp
        
        # initialize some shortcuts
        cats_entc_msp = self.model_attributes.get_variable_categories(modvar_msp)
        dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables
        subsec_name_enfu = self.model_attributes.subsec_name_enfu
        subsec_name_entc = self.model_attributes.subsec_name_entc
        
        # retrieve input MSP that has been adjusted to the presence of max production limits
        (
            arr_entc_msp, 
            arr_entc_activity_limits,
            vec_frac_msp_accounted_for_by_growth_limit
        ) = self.get_entc_maxprod_increase_adjusted_msp(
            df_elec_trajectories,
            adjust_free_msps_in_response = True,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            drop_flag = drop_flag,
            modvar_maxprod_msp_increase = modvar_maxprod_msp_increase,
            modvar_msp = modvar_msp,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
        )


        # initialize an output dictionary mapping each tech to fuel
        dict_output_tech_to_fuel = {}
        

        ##  GET ADJUSTMENTS FOR NON-ELECTRIC FUELS (SPECIFIED BY InputActivityRatio/OutputActivityRatio)
        # 20230522: use vec_frac_msp_accounted_for_by_growth_limit + arr_entc_msp.sum(axis = 1) to determine if 
        #  import renmorm needs to occur 
        for fuel in dict_fuel_cats.keys():

            # filter on output activity ratio (oar) -- skip if modvar not found
            dict_iar_oar_var = dict_fuel_cats.get(fuel)
            modvar_oar = dict_iar_oar_var.get(self.key_oar)
            if modvar_oar is None:
                continue

            # check categories -- skip if none are shared
            cats_entc = self.model_attributes.get_variable_categories(modvar_oar)
            cats_shared = set(cats_entc_msp) & set(cats_entc)  
            if len(cats_shared) == 0:
                continue
            
            cats_shared = sorted(list(cats_shared))
            dict_output_tech_to_fuel.update(
                dict((x, fuel) for x in cats_shared)
            )
            
            # get some indices
            ind_enfu = attribute_fuel.get_key_value_index(fuel)
            inds_entc = [attribute_technology.get_key_value_index(x) for x in cats_shared]

            # setup normalization of input fractions, but only apply if the total exceeds 1
            arr_entc_msp_fracs_specified = arr_entc_msp[:, inds_entc]
            max_entc_msp_fracs_norm = max(
                sf.vec_bounds(
                    arr_entc_msp_fracs_specified.sum(axis = 1),
                    (1, np.inf)
                )
            )
            arr_entc_msp_fracs_specified /= max_entc_msp_fracs_norm
            
            # get import fraction and re-normalize again
            arr_entc_msp[:, inds_entc] = sf.do_array_mult(
                arr_entc_msp_fracs_specified,
                1 - arr_enfu_import_fractions_adj[:, ind_enfu]
            )


        ##  NEXT, ADJUST SPECIFICATIONS FOR ELECTRICITY

        # get indices for pp techs
        inds_entc = [
            attribute_technology.get_key_value_index(x)
            for x in dict_tech_info.get("all_techs_pp")
        ]

        # setup normalization of input fractions, but only apply if the total exceeds 1
        arr_entc_msp_fracs_specified = arr_entc_msp[:, inds_entc]


        # CHANGED FROM MAX OF VECTOR TO WHOLE VECTOR 2023042027

        vec_entc_div = sf.vec_bounds(
            arr_entc_msp_fracs_specified.sum(axis = 1),
            (1, np.inf)
        )
        arr_entc_msp_fracs_specified = sf.do_array_mult(
            arr_entc_msp_fracs_specified, 
            1/vec_entc_div
        )
        
        # multiply by 1 - import fraction
        arr_entc_msp[:, inds_entc] = sf.do_array_mult(
            arr_entc_msp_fracs_specified,
            1 - arr_enfu_import_fractions_adj[:, self.ind_enfu_elec]
        )

        
        ##  FINAL REFORMATIONS FOR RETURN
        
        # finally, update the output dictionary for ease
        dict_output_tech_to_fuel.update(
            dict((x, self.cat_enfu_elec) for x in dict_tech_info.get("all_techs_pp"))
        )
        
        # convert to data frame and return
        df_entc_msp_formatted = self.model_attributes.array_to_df(
            arr_entc_msp,
            modvar_msp,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_entc_msp_formatted[self.model_attributes.dim_time_period] = list(df_elec_trajectories[self.model_attributes.dim_time_period])
         
        # get formatted data frame and drop all-0 groupings
        df_entc_msp_formatted = self.format_model_variable_as_nemomod_table( 
            df_entc_msp_formatted,
            modvar_msp,
            "TMP",
            [
                self.field_nemomod_id,
                self.field_nemomod_year,
                self.field_nemomod_region
            ],
            self.field_nemomod_technology,
            regions = regions,
            var_bounds = (0, 1)
        ).get("TMP")

        df_entc_msp_formatted[self.field_nemomod_fuel] = df_entc_msp_formatted[
            self.field_nemomod_technology
        ].replace(dict_output_tech_to_fuel)

        # drop techs that are all 0 for a region
        df_entc_msp_formatted = sf.filter_data_frame_by_group(
            df_entc_msp_formatted,  
            [
                self.field_nemomod_region,
                self.field_nemomod_technology
            ],
            self.field_nemomod_value
        )

        return df_entc_msp_formatted
    


    def get_entc_maxprod_increase_adjusted_msp(self,
        df_elec_trajectories: pd.DataFrame,
        adjust_free_msps_in_response: bool = True,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        build_for_activity_limit: bool = True,
        drop_flag: Union[float, int] = None,
        modvar_maxprod_msp_increase: Union[str, None] = None,
        modvar_msp: Union[str, None] = None,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Adjust the MinShareProduction input table to allow for the prevention of 
            increases in production to satisfy exogenously specified 
            MinShareProduction. Returns a tuple of three NumPy arrays:

            (
                arr_entc_msp, 
                arr_entc_activity_limits,
                vec_frac_msp_accounted_for_by_growth_limit
            )


            where `arr_entc_activity_limits` is None if there are no MSPs 
            entered; `arr_entc_activity_limits` gives an array of activity 
            limits that can be passed to TotalTechnologyAnnualActivityLowerLimit
            and TotalTechnologyAnnualActivityUpperLimit; and 
            `vec_frac_msp_accounted_for_by_growth_limit` gives the fraction of MSP 
            accountred for by the new growth limit.
            
        Example use case: if a baseline relies on the specification of 
            MinShareProduction, yet some technology will not be built after some
            point in time, this variable can be specified to avoid the conlict
            and preserve the relative balance of generation mixes.
            
        If no adjustments are found in modvar_maxprod_msp_increase, then the 
            exogenous specification is returned. 
            
        * Returns an np.ndarray wide by all ENTC categories and long by rows in
            df_elec_trajectories.
            

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - adjust_free_msps_in_response: MSP trajectories that are not subject to
            no-growth restrictions--or "Free MSPs"--can be adjusted to preserve
            the aggregate share of production that is accounted for by all MSP
            specifications (for a given fuel). If False, Free MSPs are not 
            adjusted.
        - attribute_fuel: AttributeTable for fuel
        - attribute_technology: AttributeTable used to denote technologies with 
            MinShareProductions
        - build_for_activity_limit: return output arrays designed for 
            transfering MaxProd MSP increase costraints to 
            TotalAnnualActivityLowerLimit and TotalAnnualActivityUpperLimit. If
            True, replaces MSPs (after initiation) with 0s for affected 
            technologies and provides estimates of limits. If False, preserves
            MSPs.
        - drop_flag: optional specification of a drop flag used to indicate 
            rows/variables for which the MSP Max Production constraint is not 
            applicable.Defaults to self.drop_flag_tech_capacities if None.
        - modvar_maxprod_msp_increase: SISEPUEDE model variable storing the 
            maximum production increase (as a fraction of estimated last period
            with free production) allowable due to exogenous MinShareProduction
        - modvar_msp: SISEPUEDE model variable storing the MinShareProduction
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        """ 
        
        ##  INITIALIZATION
        
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        drop_flag = (
            self.drop_flag_tech_capacities 
            if (drop_flag is None) 
            else drop_flag
        )
        modvar_maxprod_msp_increase = (
            self.modvar_entc_max_elec_prod_increase_for_msp
            if modvar_maxprod_msp_increase is None
            else modvar_maxprod_msp_increase
        )
        modvar_msp = (
            self.modvar_entc_nemomod_min_share_production
            if modvar_msp is None
            else modvar_msp
        )

        # 
        arr_entc_maxprod_msp_increase = self.model_attributes.extract_model_variable(#
            df_elec_trajectories, 
            modvar_maxprod_msp_increase,
            all_cats_missing_val = drop_flag,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        
        # get unadjusted MSP
        arr_entc_msp = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            modvar_msp,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        
        
        ##  CHECK arr_entc_maxprod_msp_increase FOR NON-DROPS
        ##    if there are no adjustments OR if there is no MSP to adjust
        ##    return arr_entc_msp
        
        w_not_drop = np.where(arr_entc_maxprod_msp_increase != drop_flag)
        if (len(w_not_drop[0]) == 0) | (np.max(arr_entc_msp.sum(axis = 1)) == 0):
            tup_out = arr_entc_msp, None, None
            return tup_out

        
        
        ##  PROCEED WITH ADJUSTMENTS IF NECESSARY - START BY GETTING
        
        # retrieve production (units do not matter since we'll work with adjusting fractions)
        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(
                df_elec_trajectories, 
                target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
            )
            if tuple_enfu_production_and_demands is None
            else tuple_enfu_production_and_demands
        )
        
        # get `vec_position_base_prod_est`, the vector storing the time period position of the last time period with free prodution estimate
        dict_entc_cat_to_position_base_prod_est = {}
        inds_modify = np.unique(w_not_drop[1])
        for j in inds_modify:
            # get i_last, the last row with a free production estimate
            w_rows = np.where(w_not_drop[1] == j)
            i_last = min(w_not_drop[0][w_rows[0]]) - 1
            
            cat = attribute_technology.key_values[j]
            dict_entc_cat_to_position_base_prod_est.update({cat: i_last}) if (i_last >= 0) else None
        

        # get fuels to consider in iteration
        fuels_adj = [
            k for k, v in self.dict_entc_fuel_categories_to_fuel_variables.items()
            if self.key_mpi_frac_msp in v.keys()
        ]
        
        # build activity limits to pass & initialize index storage for overwriting MSPs with 0 if build_for_activity_limit
        arr_entc_activity_limits = np.ones(arr_entc_msp.shape)*drop_flag
        w_set_to_no_growth = [[], []]
        
        

        ##  LOOP OVER FUELS ASSOCIATED WITH PRODUCTION TO MODIFY MSPS

        for fuel in fuels_adj:
            # 
            modvar = self.dict_entc_fuel_categories_to_fuel_variables.get(fuel).get(self.key_mpi_frac_msp)
            cats_modvar = self.model_attributes.get_variable_categories(modvar)
            cats_no_growth = [x for x in cats_modvar if x in dict_entc_cat_to_position_base_prod_est.keys()]
            cats_response = [x for x in cats_modvar if x not in cats_no_growth]

            # initialize fraction accounted for by MSP Max applicable categories
            vec_frac_msp_accounted_for_by_growth_limit = np.zeros(len(df_elec_trajectories))

            # get column indices of categories that won't grow + those that respond & check sum of MSPs that are subject to non-growth
            inds_no_growth = [attribute_technology.get_key_value_index(x) for x in cats_no_growth]
            inds_response = [attribute_technology.get_key_value_index(x) for x in cats_response]
            inds_all = inds_no_growth + inds_response
            
            if len(inds_no_growth) == 0:
                continue

            # get column totals of no-growth - only need to modify the array if there are no-growth ENTC cats associated with this fuel
            vec_entc_msp_no_growth = arr_entc_msp[:, inds_no_growth].sum(axis = 1).copy()
            vec_entc_msp_all = arr_entc_msp[:, inds_all].sum(axis = 1).copy()

            if np.max(vec_entc_msp_no_growth) == 0:
                continue

            # get projected demand for the fuel
            ind_enfu_fuel = attribute_fuel.get_key_value_index(fuel)
            vec_prod_est_cur_fuel = tuple_enfu_production_and_demands[4][:, ind_enfu_fuel]

            # ordered by cats_no_growth
            row_inds_no_growth = [dict_entc_cat_to_position_base_prod_est.get(x) for x in cats_no_growth]

            for i, j in enumerate(inds_no_growth):

                # get the estimated production in period tp(row)
                row = row_inds_no_growth[i]
                est_prod_floor = vec_prod_est_cur_fuel[row]*arr_entc_msp[row, j]

                # next, get estimated fraction associated with preserving this estimated production, but use *current* MSP as uppber bound
                fracs_new = est_prod_floor/vec_prod_est_cur_fuel[(row + 1):]
                fracs_new *= 1 + sf.vec_bounds(arr_entc_maxprod_msp_increase[(row + 1):, j], (0, np.inf))
                bounds = [(0.0, x) for x in arr_entc_msp[(row + 1):, j]]
                fracs_new = sf.vec_bounds(fracs_new, bounds)

                # overwrite MSP and add to activity limit output
                arr_entc_msp[(row + 1):, j] = fracs_new
                arr_entc_activity_limits[(row + 1):, j] = fracs_new*vec_prod_est_cur_fuel[(row + 1):]
                vec_frac_msp_accounted_for_by_growth_limit[(row + 1):] += fracs_new

                # update indices
                r = list(range(row + 1, len(arr_entc_msp)))
                w_set_to_no_growth[0] += r
                w_set_to_no_growth[1] += list([j for x in r])

            # next, check if response MSPs need to be re-scaled to preserve aggregate production shares
            if adjust_free_msps_in_response:

                vec_entc_msp_no_growth_post_adj = arr_entc_msp[:, inds_no_growth].sum(axis = 1)
                vec_entc_msp_all_post_adj = arr_entc_msp[:, inds_all].sum(axis = 1)
                
                vec_scale_response = vec_entc_msp_all - vec_entc_msp_no_growth_post_adj
                vec_scale_response /= vec_entc_msp_all - vec_entc_msp_no_growth
                vec_scale_response = np.nan_to_num(vec_scale_response, 1.0, posinf = 1.0)
                
                for j in inds_response:
                    arr_entc_msp[:, j] *= vec_scale_response

        if build_for_activity_limit:
            arr_entc_msp[w_set_to_no_growth[0], w_set_to_no_growth[1]] = 0.0

        # return MSP adjusted, activity limits, and the fraction of MSP accountred for by the new growth limit
        tup_out = arr_entc_msp, arr_entc_activity_limits, vec_frac_msp_accounted_for_by_growth_limit

        return tup_out



    def get_entc_partition_field(self,
        fld: str
    ) -> str:
        """
        Map a partition field or field abbreviation to a field
        """
        dict_abv = {
            "fp": "fuel_processing", 
            "me": "mining_and_extraction",
            "pp": "power_plant",
            "st": "storage"
        }
        
        return fld if (fld in dict_abv.values()) else dict_abv.get(fld)
    


    def get_gnrl_ccf_hydropower_factor_df(self,
        df_elec_trajectories: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Retrieve a data frame with time period, hydropower tech, and the climate
            change factor for hydropower production. Returns a tuple with two
            elements of the following form

            (
                df_gnrl_ccf_hydropower,
                field_gnrl_ccf_hydropower,
            )

            where `field_gnrl_ccf_hydropower` is the field containing the
            DataFrame
        """
        field_region = self.model_attributes.dim_region


        # get hydropower climate change factor
        df_gnrl_ccf_hydropower = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.model_socioeconomic.modvar_gnrl_climate_change_hydropower_availability,
            include_time_period = True,
            return_type = "data_frame",
        )

        if field_region in df_elec_trajectories.columns:
            df_gnrl_ccf_hydropower[field_region] = list(df_elec_trajectories[field_region])

        # add technology for hydro
        cat_entc_hpwr = self.get_entc_cat_for_integration("hpwr")
        df_gnrl_ccf_hydropower[self.field_nemomod_technology] = cat_entc_hpwr
        field_gnrl_ccf_hydropower = [
            x for x in df_gnrl_ccf_hydropower.columns 
            if x not in [self.model_attributes.dim_time_period, self.field_nemomod_technology]
        ][0]

        # convert  
        df_gnrl_ccf_hydropower[self.field_nemomod_year] = self.transform_field_year_nemomod(
        df_gnrl_ccf_hydropower[self.model_attributes.dim_time_period], 
            time_period_as_year = self.nemomod_time_period_as_year,
        )
        (
            df_gnrl_ccf_hydropower
            .drop(
                [self.model_attributes.dim_time_period], 
                axis = 1,
                inplace = True
            )
        )

        out = df_gnrl_ccf_hydropower, field_gnrl_ccf_hydropower

        return out



    def get_integrated_waste_emissions_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: Union[AttributeTable, None] = None,
        attribute_time_period: Union[AttributeTable, None] = None
    ) -> Union[pd.DataFrame, None]:
        """
        If waste composition is known from CircularEconomy, emission factors
            can be derived from the integrated model. Pulls emission factors
            from waste data in CircularEconomy.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: attribute table used for fuels. If None,
            defaults to self.model_attributes
        - attribute_technology: attribute table used for technology. If None,
            defaults to self.model_attributes
        - attribute_time_period: attribute table used for time period. If None,
            defaults to self.model_attributes
        """
        # get tech category for waste
        cat_entc_pp_waste = self.get_entc_cat_for_integration("wste") 

        # get total waste and emission factors from incineration as derived from solid waste - note: ef scalars are applied within get_waste_energy_components
        vec_enfu_total_energy_waste, vec_enfu_min_energy_to_elec_waste, dict_efs = self.get_waste_energy_components(
            df_elec_trajectories,
            attribute_technology = attribute_technology,
            return_emission_factors = True
        )

        # format to a new data frame
        df_enfu_efs_waste = None

        if (vec_enfu_total_energy_waste is not None) and len(dict_efs) > 0:

            # melt a data frame
            df_enfu_efs_waste = pd.DataFrame(dict_efs)
            df_enfu_efs_waste[self.field_nemomod_technology] = cat_entc_pp_waste
            df_enfu_efs_waste[self.field_nemomod_mode] = self.cat_enmo_gnrt
            df_enfu_efs_waste = self.model_attributes.exchange_year_time_period(
                df_enfu_efs_waste,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period,
                direction = self.direction_exchange_year_time_period
            )
            # melt into a long form table
            df_enfu_efs_waste = pd.melt(
                df_enfu_efs_waste,
                [self.field_nemomod_technology, self.field_nemomod_mode, self.field_nemomod_year],
                list(dict_efs.keys()),
                var_name = self.field_nemomod_emission,
                value_name = self.field_nemomod_value
            )

        return df_enfu_efs_waste



    def get_nemomod_energy_scalar(self, 
        modvar: Union[str, None],
        force_modvar_to_config_on_none: bool = False
    ) -> float:
        """
        return a scalar - use to reduce clutter in converting energy units to 
            NemoMod energy units. If modvar is None, uses configuration units.

        Function Arguments
        ------------------
        - modvar: model variable to convert to NemoMod energy units

        Keyword Arguments
        -----------------
        - force_modvar_to_config_on_none: force the model variable to 
            configuration units if not associated with energy

        """
        units_source = self.model_attributes.get_variable_characteristic(
            modvar, 
            self.model_attributes.varchar_str_unit_energy
        ) if (modvar is not None) else self.model_attributes.configuration.get("energy_units")
        units_source = (
            self.model_attributes.configuration.get("energy_units") 
            if (units_source is None) and force_modvar_to_config_on_none
            else units_source
        )

        scalar = self.model_attributes.get_energy_equivalent(units_source, self.units_energy_nemomod)
        
        return (scalar if (scalar is not None) else 1)
    


    def get_nemomod_optimizer(self,
        solver: Union[str, None] = None,
    ) -> "Pycall.jlwrap":
        """
        Retrieve the optimizer for NemoMod. Use to set any parameters 
            related to the solver (e.g., NumericFocus in Gurobi)

        Returns an optimizer object. 

        NOTE: Update to read from a config

        Keword Arguments
        ------------------
        - solver: string denoting solver to use. If None, defaults to 
            self.solver
        """
        
        # set optimizer
        solver = self.solver if (solver is None) else solver
        optimizer = self.julia_jump.Model(self.solver_module.Optimizer)

        # set some generic properties
        self.julia_jump.set_time_limit_sec(optimizer, self.solver_time_limit)
        self.julia_jump.set_silent(optimizer)


        ##  SOLVER SPECIFIC PROPERTIES (setup from config for later)

        # gams/cplex parameters
        if (solver == "gams_cplex"):
            self.julia_jump.set_optimizer_attribute(optimizer, "Solver", "cplex")

        # gurobi parameters            
        if (solver == "gurobi"):
            # see https://www.gurobi.com/documentation/9.5/refman/numericfocus.html#parameter:NumericFocus
            self.julia_jump.set_optimizer_attribute(optimizer, "NumericFocus", 2)
            #print(self.julia_base.propertynames(optimizer))

        # HiGHS parameters            
        if (solver == "highs"):
            # see https://www.gurobi.com/documentation/9.5/refman/numericfocus.html#parameter:NumericFocus
            self.julia_jump.set_optimizer_attribute(optimizer, "solver", "simplex")
            #TRUYING
        return optimizer



    def get_tech_info_dict(self,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None
    ) -> Dict:
        """
        Retrieve information relating technology to storage, including a map of 
            technologies to storage, storage to associated technology, and 
            classifications of generation techs vs. storage techs.

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable containing fuels
        - attribute_technology: AttributeTable containing technologies used to 
            divide techs. If None, uses ModelAttributes default.
        """
        # set some defaults
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        # get some categories associated with elements
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )
        pychat_entc = self.model_attributes.get_subsector_attribute(
            self.subsec_name_entc, 
            "pycategory_primary_element",
        )
        pycat_strg = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enst, 
            "pycategory_primary_element",
        )

        # tech -> fuel and fuel -> tech dictionaries
        dict_gnrt_tech_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            f"electricity_generation_{pycat_enfu}",
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        dict_fuel_to_tech = sf.reverse_dict(
            dict_gnrt_tech_to_fuel, 
            allow_multi_keys = True,
        )

        # tech -> storage and storage -> tech dictionaries
        dict_storage_techs_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_strg,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )
        dict_storage_to_storage_techs = sf.reverse_dict(dict_storage_techs_to_storage)

        # get dummy techs
        dict_fuels_to_dummy_techs = self.get_dummy_fuel_techs(attribute_fuel = attribute_fuel)
        dict_dummy_techs_to_fuels = sf.reverse_dict(dict_fuels_to_dummy_techs)
        all_techs_dummy = sorted(list(dict_dummy_techs_to_fuels.keys()))

        dict_return = {
            "all_techs_dummy": all_techs_dummy,
            "all_techs_fp": self.get_entc_cat_by_type("fp"),
            "all_techs_me": self.get_entc_cat_by_type("me"),
            "all_techs_pp": self.get_entc_cat_by_type("pp"),
            "all_techs_st": self.get_entc_cat_by_type("st"),
            "dict_fuel_to_pp_tech": dict_fuel_to_tech,
            "dict_pp_tech_to_fuel": dict_gnrt_tech_to_fuel,
            "dict_storage_techs_to_storage": dict_storage_techs_to_storage,
            "dict_storage_to_storage_techs": dict_storage_to_storage_techs,
            "dict_dummy_techs_to_fuels": dict_dummy_techs_to_fuels,
            "dict_fuels_to_dummy_techs": dict_fuels_to_dummy_techs
        }

        return dict_return



    def get_variable_cost_fuels_gravimetric_density(self,
        df_elec_trajectories: pd.DataFrame,
        override_time_period_transformation: bool = False,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        CURRENTLY DEPRICATED--CHECK

        Retrieve variable cost of fuels (entered as dummy technologies) with 
            prices based on gravimetric energy density in terms of Configuration 
            energy_units/monetary_units (used in NemoMod)

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame containing input variables as columns

        Keyword Arguments
        -----------------
        - override_time_period_transformation: if True, return raw time periods 
            instead of those transformed to fit NemoMod approach.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        ##  PREPARE SCALARS

        # get scalars to apply to prices - start with energy scalar (scale energy factor to configuration units--divide since energy is the denominator)
        scalar_energy = self.get_nemomod_energy_scalar(self.modvar_enfu_energy_density_gravimetric)
        
        # scaling to get masses (denominators)
        scalar_mass = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_energy_density_gravimetric,
            self.modvar_enfu_price_gravimetric,
            "mass"
        )
        # scale prices
        scalar_monetary = self.model_attributes.get_scalar(self.modvar_enfu_price_gravimetric, "monetary")
        scalar_price = scalar_mass * scalar_monetary


        ##  GET PRICES AND DENSITY

        # Fuel costs (enter as supply) - Gravimetric price in configuration Monetary/Mass (mass of modvar_enfu_energy_density_gravimetric)
        df_price = self.format_model_variable_as_nemomod_table(
            df_elec_trajectories,
            self.modvar_enfu_price_gravimetric,
            self.model_attributes.table_nemomod_variable_cost,
            [
                self.field_nemomod_year
            ],
            self.field_nemomod_technology,
            dict_fields_to_pass = {
                self.field_nemomod_mode: self.cat_enmo_gnrt
            },
            override_time_period_transformation = override_time_period_transformation,
            regions = regions,
            scalar_to_nemomod_units = scalar_price,
            var_bounds = (0, np.inf)
        )

        # get the energy density in terms of configuration Energy/Mass (mass of modvar_enfu_energy_density_gravimetric)
        df_density = self.format_model_variable_as_nemomod_table(
            df_elec_trajectories,
            self.modvar_enfu_energy_density_gravimetric,
            self.model_attributes.table_nemomod_variable_cost,
            [
                self.field_nemomod_year
            ],
            self.field_nemomod_technology,
            scalar_to_nemomod_units = scalar_energy,
            dict_fields_to_pass = {
                self.field_nemomod_mode: self.cat_enmo_gnrt
            },
            override_time_period_transformation = override_time_period_transformation,
            regions = regions,
            var_bounds = (0, np.inf)
        )


        ##  COMPARE AND GENERATE OUTPUT

        # rename price and density
        df_price = df_price[self.model_attributes.table_nemomod_variable_cost].rename(
            columns = {self.field_nemomod_value: "price"}
        )
        df_density = df_density[self.model_attributes.table_nemomod_variable_cost].rename(
            columns = {self.field_nemomod_value: "density"}
        )

        # merge, calcuate real price, then update tech names
        df_out = pd.merge(df_price, df_density)
        df_out[self.field_nemomod_value] = np.array(df_out["price"])*np.array(df_out["density"])
        df_out[self.field_nemomod_technology] = df_out[self.field_nemomod_technology].apply(self.get_dummy_fuel_tech_name)
        df_out.drop(["price", "density"], axis = 1, inplace = True)

        return df_out



    def get_variable_cost_fuels_volumetric_density(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ):
        """
        CURRENTLY DEPRICATED--CHECK
        
        Retrieve variable cost of fuels (entered as dummy technologies) with 
            prices based on volume in terms of Configuration 
            energy_units/monetary_units (used in NemoMod)

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame containing input variables as columns

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        ##  PREPARE SCALARS

        # get scalars to apply to prices - start with energy scalar (scale energy factor to configuration units--divide since energy is the denominator)
        scalar_energy = self.get_nemomod_energy_scalar(self.modvar_enfu_energy_density_volumetric)
        # scaling to get masses (denominators)
        scalar_volume = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_energy_density_volumetric,
            self.modvar_enfu_price_volumetric,
            "volume"
        )
        # scale prices
        scalar_monetary = self.model_attributes.get_scalar(self.modvar_enfu_price_volumetric, "monetary")
        scalar_price = scalar_volume * scalar_monetary


        ##  GET PRICES AND DENSITY

        # Fuel costs (enter as supply) - Volumetric price in configuration Monetary/Mass (mass of modvar_enfu_energy_density_gravimetric)
        df_price = self.format_model_variable_as_nemomod_table(
            df_elec_trajectories,
            self.modvar_enfu_price_volumetric,
            self.model_attributes.table_nemomod_variable_cost,
            [
                self.field_nemomod_year
            ],
            self.field_nemomod_technology,
            dict_fields_to_pass = {
                self.field_nemomod_mode: self.cat_enmo_gnrt
            },
            regions = regions,
            scalar_to_nemomod_units = scalar_price,
            var_bounds = (0, np.inf)
        )

        # get the energy density in terms of configuration Energy/Mass (mass of modvar_enfu_energy_density_gravimetric)
        df_density = self.format_model_variable_as_nemomod_table(
            df_elec_trajectories,
            self.modvar_enfu_energy_density_volumetric,
            self.model_attributes.table_nemomod_variable_cost,
            [
                self.field_nemomod_year
            ],
            self.field_nemomod_technology,
            dict_fields_to_pass = {
                self.field_nemomod_mode: self.cat_enmo_gnrt
            },
            regions = regions,
            scalar_to_nemomod_units = scalar_energy,
            var_bounds = (0, np.inf)
        )


        ##  COMPARE AND GENERATE OUTPUT

        # rename price and density
        df_price = df_price[self.model_attributes.table_nemomod_variable_cost].rename(
            columns = {self.field_nemomod_value: "price"}
        )
        df_density = df_density[self.model_attributes.table_nemomod_variable_cost].rename(
            columns = {self.field_nemomod_value: "density"}
        )

        # merge, calcuate real price, then update tech names
        df_out = pd.merge(df_price, df_density)
        df_out[self.field_nemomod_value] = np.array(df_out["price"])*np.array(df_out["density"])
        df_out[self.field_nemomod_technology] = df_out[self.field_nemomod_technology].apply(self.get_dummy_fuel_tech_name)
        df_out.drop(["price", "density"], axis = 1, inplace = True)

        # exchange year and time period
        df_out[self.field_nemomod_year] = self.transform_time_period()
        return df_out



    def get_waste_energy_components(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: AttributeTable = None,
        return_emission_factors: bool = True,
    ) -> tuple:

        """
        Retrieve total energy to be obtained from waste incineration (minimum 
            capacity) and implied annual emission factors derived from 
            incineration inputs in the waste sector (NemoMod emission/energy)

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of input variables, which must 
            include waste sector outputs used to calcualte emission factors
        
        Keyword Arguments
        -----------------
        - attribute_technology: technology attribute table, used to map fuel to 
            tech. If None, use ModelAttributes default.
        - return_emission_factors: bool--calculate emission factors?
        """
        # some attribute initializations
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        # output variable initialization
        dict_efs = {}
        vec_enfu_total_energy_waste = None

        # retrieve waste totals incinerated (in solid waste only)
        tuple_waso_incineration = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_elec_trajectories,
            self.model_circecon.modvar_waso_waste_total_incineration,
            None,
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )

        #
        if tuple_waso_incineration is not None:
            # get mass of waste incinerated,
            modvar_waso_mass_incinerated, array_waso_mass_incinerated = tuple_waso_incineration
            vec_waso_mass_incinerated = np.sum(array_waso_mass_incinerated, axis = 1)

            # convert to energy units using gravimetric density (aka specific energy)
            vec_enfu_energy_density_gravimetric = self.model_attributes.extract_model_variable(#
                df_elec_trajectories,
                self.modvar_enfu_energy_density_gravimetric,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
            )
            vec_enfu_energy_density_gravimetric = vec_enfu_energy_density_gravimetric[:, self.ind_enfu_wste]

            # also get minimum fuel fraction to electricity
            vec_enfu_minimum_fuel_frac_to_elec = self.model_attributes.extract_model_variable(#
                df_elec_trajectories,
                self.modvar_enfu_minimum_frac_fuel_used_for_electricity,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
                var_bounds = (0, 1),
            )
            vec_enfu_minimum_fuel_frac_to_elec = vec_enfu_minimum_fuel_frac_to_elec[:, self.ind_enfu_wste]

            # convert units -- first, in terms of mass incinerated, then in terms of energy density
            vec_enfu_energy_density_gravimetric /= self.model_attributes.get_variable_unit_conversion_factor(
                self.modvar_enfu_energy_density_gravimetric,
                modvar_waso_mass_incinerated,
                "mass",
            )
            vec_enfu_energy_density_gravimetric *= self.get_nemomod_energy_scalar(self.modvar_enfu_energy_density_gravimetric)
            vec_enfu_total_energy_waste = vec_enfu_energy_density_gravimetric*vec_waso_mass_incinerated

            # get minimum fraction to electricity
            vec_enfu_minimum_fuel_energy_to_electricity_waste = vec_enfu_total_energy_waste*vec_enfu_minimum_fuel_frac_to_elec


        # get emission factors?
        if (vec_enfu_total_energy_waste is not None) and return_emission_factors:

            # loop over waste emissions, divide by total energy
            list_modvars_enfu_to_tech = [
                (self.model_circecon.modvar_waso_emissions_ch4_incineration, self.modvar_entc_ef_scalar_ch4),
                (self.model_circecon.modvar_waso_emissions_co2_incineration, self.modvar_entc_ef_scalar_co2),
                (self.model_circecon.modvar_waso_emissions_n2o_incineration, self.modvar_entc_ef_scalar_n2o)
            ]

            for modvars in list_modvars_enfu_to_tech:

                modvar, modvar_scalar = modvars

                vec_waso_emissions_incineration = self.model_attributes.get_optional_or_integrated_standard_variable(
                    df_elec_trajectories,
                    modvar,
                    None,
                    override_vector_for_single_mv_q = False,
                    return_type = "array_base",
                )

                # skip if unavaiable
                if vec_waso_emissions_incineration is None:
                    continue
                
                # if the data are available, calculate the factor and add it to 
                # the dictionary (long by time periods in df_elec_trajectories)

                # get incineration emissions total and scale units
                emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
                modvar_waso_emissions_emissions, vec_waso_emissions_incineration = vec_waso_emissions_incineration
                vec_waso_emissions_incineration *= self.model_attributes.get_scalar(modvar, "mass")

                # get control scalar on reductions
                vec_entc_ear_scalar = self.model_attributes.extract_model_variable(#
                    df_elec_trajectories,
                    modvar_scalar,
                    expand_to_all_cats = True,
                    override_vector_for_single_mv_q = True,
                    return_type = "array_base",
                    var_bounds = (0, 1),
                )

                cat_tech = self.get_entc_cat_for_integration("wste")
                ind_tech = attribute_technology.get_key_value_index(cat_tech)
                vec_entc_ear_scalar = vec_entc_ear_scalar[:, ind_tech]

                dict_efs.update({
                    emission: vec_entc_ear_scalar*vec_waso_emissions_incineration/vec_enfu_total_energy_waste
                })


        tup_out = (
            vec_enfu_total_energy_waste, 
            vec_enfu_minimum_fuel_energy_to_electricity_waste, 
            dict_efs
        )

        return tup_out



    def format_model_variable_as_nemomod_table(self,
        df_elec_trajectories: pd.DataFrame,
        modvar: str,
        table_nemomod: str,
        fields_index_nemomod: list,
        field_melt_nemomod: str,
        df_append: Union[pd.DataFrame, None] = None,
        dict_fields_to_pass: dict = {},
        drop_flag: Union[float, int, None] = None,
        override_time_period_transformation: bool = False,
        regions: Union[List[str], None] = None,
        scalar_to_nemomod_units: Union[float, None] = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Format a SISEPUEDE variable as a nemo mod input table.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame containing input variables to be 
            reformatted
        - modvar: SISEPUEDE model variable to extract and reshape
        - table_nemomod: target NemoMod table
        - fields_index_nemomod: indexing fields to add/preserve in table
        - field_melt_nemomod: name of field to store columns under in long 
            format

        Keyword Arguments
        -----------------
        - df_append: pass a data frame from another source to append before 
            sorting and the addition of an id
        - dict_fields_to_pass: dictionary to pass fields to the output data 
            frame before sorting
            * Dictionary takes the form {field_1: new_col, ...}, where 
                new_col = [x_0, ..., x_{n - 1}] or new_col = obj
        - drop_flag: values that should be dropped from the table
        - override_time_period_transformation: override the time series 
            transformation? data frames will return raw years instead of 
            transformed years.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - scalar_to_nemomod_units: scalar applied to the values to convert to 
            proper units
        **kwargs: passed to ModelAttributes.extract_model_variable(#)
        """

        # set some defaults

        subsector = (
            modvar if (modvar in self.model_attributes.all_subsectors)
            else self.model_attributes.get_variable_subsector(modvar, throw_error_q = False)
        )
        if subsector is None:
            return None

        attr = self.model_attributes.get_attribute_table(subsector)
        scalar_to_nemomod_units = (
            1.0 
            if not sf.isnumber(scalar_to_nemomod_units) 
            else scalar_to_nemomod_units
        )

        # get the variable from the data frame
        df_out = (
            self.model_attributes.extract_model_variable(#
                df_elec_trajectories,
                modvar,
                expand_to_all_cats = False,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
                **kwargs
            )
            if modvar != subsector
            else np.array(df_elec_trajectories[[x for x in attr.key_values if x in df_elec_trajectories.columns]])
        )

        # do any conversions and initialize the output dataframe
        df_out *= scalar_to_nemomod_units
        cats_ordered_out = (
            self.model_attributes.get_variable_categories(modvar)
            if (modvar != subsector)
            else attr.key_values
        )
        df_out = pd.DataFrame(df_out, columns = cats_ordered_out)


        # add a year (would not be in data frame)
        exchange_year_tp = (self.field_nemomod_year in fields_index_nemomod)
        exchange_year_tp &= (self.model_attributes.dim_time_period in df_elec_trajectories.columns)
        if exchange_year_tp:
            df_out = self.model_attributes.exchange_year_time_period(
                df_out,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                direction = self.direction_exchange_year_time_period
            )

        # add any additional fields
        if len(dict_fields_to_pass) > 0:
            for fld in dict_fields_to_pass.keys():
                if (fld not in df_out.columns) :
                    df_out[fld] = dict_fields_to_pass[fld]
                    fields_index_nemomod += [fld] if (fld not in fields_index_nemomod) else []

        # next, melt, drop any values, and add keys/sort/reset index for NemoMod
        df_out = pd.melt(
            df_out,
            [x for x in df_out.columns if x in fields_index_nemomod],
            cats_ordered_out,
            var_name = field_melt_nemomod,
            value_name = self.field_nemomod_value
        )

        df_out = (
            df_out[~df_out[self.field_nemomod_value].isin([drop_flag])] 
            if (drop_flag is not None) 
            else df_out
        )

        if isinstance(df_append, pd.DataFrame):
            df_out = pd.concat([df_out, df_append[df_out.columns]], axis = 0).reset_index(drop = True)
        
        df_out = self.add_multifields_from_key_values(
            df_out,
            fields_index_nemomod,
            override_time_period_transformation = override_time_period_transformation,
            regions = regions
        )

        dict_out = {table_nemomod: df_out}

        return dict_out



    def format_dummy_tech_description_from_fuel(self, 
        fuel: str
    ) -> str:
        return f"Dummy supply technology for fuel {fuel} -- allows for solutions that would otherwise be infeasible."



    # defin a function to compare max/min for related constraints
    def verify_min_max_constraint_inputs(self,
        df_max: pd.DataFrame,
        df_min: pd.DataFrame,
        field_max: str,
        field_min: str,
        conflict_resolution_option: str = "swap",
        comparison: str = "weak",
        drop_invalid_comparisons_on_strong: bool = True,
        max_min_distance_scalar: Union[int, float] = 1,
        field_id: str = None,
        return_passthrough: bool = False
    ) -> Union[None, dict]:
        """
        Verify that a minimum trajectory is less than or equal (weak) or less 
            than (strong) a maximum trajectory. Data frames must have comparable 
            indices.

        Function Arguments
        ------------------
        - df_max: data frame containing the maximum trajectory
        - df_min: data frame containing the minimum trajectory
        - field_max: field in df_max to use to compare
        - field_min: field in df_min to use to compare

        Keyword Arguments
        -----------------
        - conflict_resolution_option: if the minimum trajectory is greater than 
            the maximum trajectory, this parameter is used to define the 
            resolution:
            * "error": stop and return an error
            * "keep_max_input": keep the values from the maximum input for both
            * "keep_min_input": keep the values from the minimum input for both
            * "max_sup": set the larger value as the minimum and the maximum
            * "mean": use the mean of the two as the minimum and the maximum
            * "min_sup": set the smaller value as the minimum and the maximum
            * "swap" (DEFAULT): swap instances where the minimum exceeds the 
                maximum
        - comparison: "weak" allows the minimum <= maximum, 
            while "strong" => minimum < maximum
            * If comparison == "strong", then cases where maximum == minimum 
                cannot be resolved will be dropped if 
                drop_invalid_comparisons_on_strong == True; otherwise, an error 
                will be returned (independent of conflict_resolution_option)
        - drop_invalid_comparisons_on_strong: drop cases where minimum == 
            maximum?
        - max_min_distance_scalar: max >= min*max_min_distance_scalar for 
            constraints. Default is 1.
        - field_id: id field contained in both that is used for re-merging
        - return_passthrough: if no changes are required, return original 
            dataframes?

        """

        suffix_max = "max"
        suffix_min = "min"
        # check for required field
        field_id = self.field_nemomod_id if (field_id is None) else field_id
        sf.check_fields(df_max, [field_id, field_max])
        sf.check_fields(df_min, [field_id, field_min])
        # temporary fields
        field_id_max = f"{field_id}_{suffix_max}"
        field_id_min = f"{field_id}_{suffix_min}"

        # merge to facilitate comparison
        fields_shared = list(set(df_max.columns) & set(df_max.columns))
        fields_shared = [x for x in fields_shared if x not in [field_min, field_max, field_id]]
        fields_max = fields_shared + [field_max]
        fields_min = fields_shared + [field_min]
        df_compare = pd.merge(
            df_max[fields_max],
            df_min[fields_min],
            on = fields_shared,
            suffixes = (f"_{suffix_max}", f"_{suffix_min}")
        )

        # set fields to use for comparison
        field_maxm = f"{field_max}_{suffix_max}" if (field_max == field_min) else field_max
        field_minm = f"{field_min}_{suffix_min}" if (field_max == field_min) else field_min
        #
        vec_comparison = np.array(df_compare[[field_minm, field_maxm]])
        w_resolve = np.where(vec_comparison[:, 1] < vec_comparison[:, 0]) if (comparison == "weak") else np.where(vec_comparison[:, 1] <= vec_comparison[:, 0])[0]

        if (len(w_resolve) > 0):
            if conflict_resolution_option != "error":
                df_new_vals = df_compare[[field_minm, field_maxm]].apply(
                    self.conflict_resolution_func_vmmci,
                    approach = conflict_resolution_option,
                    max_min_distance_scalar = max_min_distance_scalar,
                    axis = 1,
                    raw = True
                )
                # some replacements
                df_max_replace = pd.concat([df_compare[fields_shared], df_new_vals[[field_maxm]]], axis = 1).rename(
                    columns = {
                        field_id_max: field_id,
                        field_maxm: field_max
                    }
                )
                df_min_replace = pd.concat([df_compare[fields_shared], df_new_vals[[field_minm]]], axis = 1).rename(
                    columns = {
                        field_id_min: field_id,
                        field_minm: field_min
                    }
                )
                df_max_out = sf.replace_numerical_column_from_merge(df_max, df_max_replace, field_max)
                df_min_out = sf.replace_numerical_column_from_merge(df_min, df_min_replace, field_min)
            else:
                raise ValueError(f"Error in verify_min_max_constraint_inputs: minimum trajectory meets or exceeds maximum trajectory in at least one row.")

            return df_max_out, df_min_out
        else:
            return (df_max, df_min) if return_passthrough else None



    #######################################################################################
    #    ATTRIBUTE TABLE TRANSFORMATION FUNCTIONS TO FORMAT NEMOMOD DIMENSIONS FOR SQL    #
    #######################################################################################

    ##  format EMISSION for NemoMod
    def format_nemomod_attribute_table_emission(self,
        attribute_emission: AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
        Format the EMISSION dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - attribute_emission: Emission Gas AttributeTable. If None, use 
            ModelAttributes default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod
        """

        # set some defaults
        attribute_emission = (
            self.get_attribute_emission() 
            if (attribute_emission is None) 
            else attribute_emission
        )
        dict_rename = (
            {
                "emission_gas": self.field_nemomod_value, 
                "name": self.field_nemomod_description
            } 
            if (dict_rename is None) 
            else dict_rename
        )

        # set values out
        df_out = attribute_emission.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]

        df_out = (
            df_out[fields_ord]
            .sort_values(by = fields_ord)
            .reset_index(drop = True)
        )

        dict_out = {self.model_attributes.table_nemomod_emission: df_out}

        return dict_out


    def format_nemomod_attribute_table_fuel(self,
        attribute_fuel: AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
        Format the FUEL dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - attribute_fuel: Fuel AttributeTable. If None, use ModelAttributes 
            default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod
        """

        # set some defaults
        attribute_fuel = (
            self.get_attribute_enfu()
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        pycat_fuel = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element"
        )

        dict_rename = (
            {
                pycat_fuel: self.field_nemomod_value, 
                "description": self.field_nemomod_description
            } 
            if (dict_rename is None) 
            else dict_rename
        )

        # set values out
        df_out = attribute_fuel.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        
        #HEREFORDUMMYFUEL
        # add dummy fuel for tech production accounting from dummy techs
        df_append = pd.DataFrame({
            self.field_nemomod_value: [self.get_dummy_fuel_name()],
            self.field_nemomod_description: [self.get_dummy_fuel_description()]
        })
        fields_ext = list(dict_rename.values())
        df_out = pd.concat([df_out[fields_ext], df_append[fields_ext]], axis = 0).reset_index(drop = True)

        # order for output
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        dict_out = {self.model_attributes.table_nemomod_fuel: df_out}

        return dict_out



    def format_nemomod_attribute_table_mode_of_operation(self,
        attribute_mode: AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
        Format the MODE_OF_OPERATION dimension table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Keyword Arguments
        -----------------
        - attribute_mode: Mode of Operation AttributeTable. If None, use 
            ModelAttributes default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod
        """

        # get the region attribute - reduce only to applicable regions
        attribute_mode = (
            self.model_attributes.get_other_attribute_table(self.model_attributes.dim_mode) 
            if (attribute_mode is None) 
            else attribute_mode
        )

        dict_rename = (
            {
                self.model_attributes.dim_mode: self.field_nemomod_value, 
                "description": self.field_nemomod_description
            } 
            if (dict_rename is None) 
            else dict_rename
        )

        # set values out
        df_out = attribute_mode.table.copy().rename(columns = dict_rename)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        dict_out = {self.model_attributes.table_nemomod_mode_of_operation: df_out}

        return dict_out



    def format_nemomod_attribute_table_node(self,
        attribute_node: AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
        Format the NODE dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - attribute_node: Node AttributeTable. If None, use ModelAttributes 
            default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod

        CURRENTLY UNUSED
        """

        return None



    def format_nemomod_attribute_table_region(self,
        attribute_region: AttributeTable = None,
        dict_rename: dict = None,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the REGION dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - attribute_region: CAT-REGION AttributeTable. If None, use 
            ModelAttributes default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        # get the region attribute - reduce only to applicable regions
        attribute_region = (
            self.get_attribute_region()
            if (attribute_region is None) 
            else attribute_region
        )

        dict_rename = (
            {
                self.model_attributes.dim_region: self.field_nemomod_value, 
                "category_name": self.field_nemomod_description
            } 
            if (dict_rename is None) 
            else dict_rename
        )
        
        regions = self.model_attributes.get_region_list_filtered(regions, attribute_region = attribute_region)

        # set values out
        df_out = attribute_region.table.copy().rename(columns = dict_rename)
        df_out = df_out[df_out[self.field_nemomod_value].isin(regions)]
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return {self.model_attributes.table_nemomod_region: df_out}



    def format_nemomod_attribute_table_storage(self,
        attribute_storage: AttributeTable = None,
        dict_rename: dict = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the STORAGE dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
                and reference tables.

        Keyword Arguments
        -----------------
        - attribute_storage: CAT-STORAGE AttributeTable. If None, use 
            ModelAttributes default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for
             NemoMod
        """

        # set some defaults
        attribute_storage = (
            self.model_attributes.get_attribute_table(self.subsec_name_enst) 
            if (attribute_storage is None) 
            else attribute_storage
        )

        pycat_strg = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enst, 
            "pycategory_primary_element",
        )

        dict_rename = (
            {
                pycat_strg: self.field_nemomod_value, 
                "description": self.field_nemomod_description
            } 
            if (dict_rename is None) 
            else dict_rename
        )

        # set values out
        df_out = attribute_storage.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)

        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)] 
        fields_ord += [f"netzero{x}" for x in ["year", "tg1", "tg2"]]
        df_out = (
            df_out[fields_ord]
            .sort_values(by = fields_ord)
            .reset_index(drop = True)
        )

        dict_return = {self.model_attributes.table_nemomod_storage: df_out}

        return dict_return



    def format_nemomod_attribute_table_technology(self,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        dict_rename: dict = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the TECHNOLOGY dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
                and reference tables.

        Keyword Arguments
        -----------------
        - attribute_fuel: CAT-FUEL AttributeTable. If None, use ModelAttributes
            default.
        - attribute_technology: CAT-TECHNOLOGY AttributeTable. If None, use 
            ModelAttributes default.
        - dict_rename: dictionary to rename to "val" and "desc" fields for 
            NemoMod
        """

        # set some defaults
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        pycat_entc = self.model_attributes.get_subsector_attribute(
            self.subsec_name_entc, 
            "pycategory_primary_element",
        )

        dict_rename = (
            {
                pycat_entc: self.field_nemomod_value, 
                "description": self.field_nemomod_description,
            } 
            if (dict_rename is None) 
            else dict_rename
        )

        # add dummies
        dict_fuels_to_dummy_techs = self.get_dummy_fuel_techs(attribute_fuel = attribute_fuel)

        df_out_dummies = pd.DataFrame({self.field_nemomod_fuel: list(dict_fuels_to_dummy_techs.keys())})
        df_out_dummies[self.field_nemomod_value] = (
            df_out_dummies[self.field_nemomod_fuel]
            .replace(dict_fuels_to_dummy_techs)
        )
        df_out_dummies[self.field_nemomod_description] = (
            df_out_dummies[self.field_nemomod_fuel]
            .apply(self.format_dummy_tech_description_from_fuel)
        )

        df_out_dummies.drop(
            [self.field_nemomod_fuel], 
            axis = 1, 
            inplace = True,
        )

        # set values out
        df_out = attribute_technology.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)

        #HEREFORDUMMYFUEL
        # add dummy fuel for tech production accounting from dummy techs
        df_append = pd.DataFrame({
            self.field_nemomod_value: [self.get_dummy_fuel_name(return_type = "tech")],
            self.field_nemomod_description: [self.get_dummy_fuel_description(return_type = "tech")]
        })

        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = (
            pd.concat(
                [df_out[fields_ord], df_out_dummies[fields_ord], df_append[fields_ord]],
                 axis = 0
            )
            .sort_values(by = fields_ord)
            .reset_index(drop = True)
        )

        dict_return = {self.model_attributes.table_nemomod_technology: df_out}

        return dict_return



    def format_nemomod_attribute_table_year(self,
        time_period_as_year: bool = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the YEAR dimension table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - time_period_as_year: enter years as time periods? If None, default to 
            ElectricEnergy.nemomod_time_period_as_year
        * Based off of values defined in the attribute_time_period.csv attribute 
            table
        """

        time_period_as_year = self.nemomod_time_period_as_year if (time_period_as_year is None) else time_period_as_year
        years = self.model_attributes.get_time_periods()[0] if time_period_as_year else self.model_attributes.get_time_period_years()
        desc_name = "Time Period" if time_period_as_year else "Year"

        # clean the year if necessary
        years_clean = self.transform_field_year_nemomod(years, time_period_as_year = time_period_as_year)

        df_out = pd.DataFrame({
            self.field_nemomod_value: years_clean,
            self.field_nemomod_description: [f"SISEPUEDE {desc_name} {y}" for y in years]
        })

        dict_return = {self.model_attributes.table_nemomod_year: df_out}

        return dict_return




    ###########################################################
    #    FUNCTIONS TO FORMAT MODEL VARIABLE INPUTS FOR SQL    #
    ###########################################################

    def format_nemomod_table_annual_emission_limit(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_emission: AttributeTable = None,
        attribute_time_period: AttributeTable = None,
        dict_gas_to_emission_fields: dict = None,
        drop_flag: Union[int, None] = None,
        regions: Union[List[str], None] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the AnnualEmissionLimit input tables for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_emission: AttributeTable table with gasses. If None, use 
            ModelAttribute default.
        - attribute_time_period: AttributeTable table with time periods for year 
            identification. If None, use ModelAttribute default.
        - dict_gas_to_emission_fields: dictionary with gasses (in attribute_gas) 
            as keys that map to fields to use to calculate total exogenous 
            emissions
        - drop_flag: values to drop
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        # get some defaults and attribute tables
        dict_gas_to_emission_fields = (
            self.model_attributes.dict_gas_to_total_emission_fields 
            if (dict_gas_to_emission_fields is None) 
            else dict_gas_to_emission_fields
        )

        attribute_emission = (
            self.get_attribute_emission() 
            if (attribute_emission is None) 
            else attribute_emission
        )

        attribute_time_period = (
            self.get_attribute_time_period() 
            if (attribute_time_period is None) 
            else attribute_time_period
        )

        drop_flag = self.drop_flag_tech_capacities if (drop_flag is None) else drop_flag

        modvars_limit = [
            self.model_socioeconomic.modvar_gnrl_emission_limit_ch4,
            self.model_socioeconomic.modvar_gnrl_emission_limit_co2,
            self.model_socioeconomic.modvar_gnrl_emission_limit_n2o
        ]
        df_out = []

        for modvar in enumerate(modvars_limit):
            i, modvar = modvar

            # get emission and global warming potential (divide out for limit)
            emission = self.model_attributes.get_variable_characteristic(
                modvar, 
                self.model_attributes.varchar_str_emission_gas,
            )
            gwp = self.model_attributes.get_gwp(emission)

            # then, get total exogenous emissions
            fields = list(set(dict_gas_to_emission_fields[emission]) & set(df_elec_trajectories.columns))
            vec_exogenous_emissions = np.sum(np.array(df_elec_trajectories[fields]), axis = 1)
            
            # retrieve the limit, store the origina (for dropping), and convert units
            vec_emission_limit = self.model_attributes.extract_model_variable(#
                df_elec_trajectories,
                modvar,
                return_type = "array_base",
            )

            vec_drop_flag = vec_emission_limit.copy()
            vec_emission_limit *= self.model_attributes.get_scalar(modvar, "mass")

            # force limit to  to prevent infeasibilities
            vec_emission_limit_out = sf.vec_bounds(vec_emission_limit - vec_exogenous_emissions, (0, np.inf))
            
            df_lim = pd.DataFrame({
                "drop_flag": vec_drop_flag,
                self.field_nemomod_value: vec_emission_limit_out
            })
            df_lim[self.field_nemomod_emission] = emission

            df_lim = self.model_attributes.exchange_year_time_period(
                df_lim,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period,
                direction = self.direction_exchange_year_time_period,
            )

            if i == 0:
                df_out = [df_lim for x in modvars_limit]
            else:
                df_out[i] = df_lim

        # concatenate and order hierarchically
        df_out = pd.concat(df_out, axis = 0)
        df_out = df_out[~df_out["drop_flag"].isin([drop_flag])].drop(["drop_flag"], axis = 1)
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_emission,
                self.field_nemomod_year,
                self.field_nemomod_value,
                "drop_flag"
            ],
            regions = regions
        )
        
        dict_return = {self.model_attributes.table_nemomod_annual_emission_limit: df_out}

        return dict_return



    def format_nemomod_table_capacity_factor(self,
        df_elec_trajectories: pd.DataFrame,
        df_reference_capacity_factor: pd.DataFrame,
        attribute_technology: AttributeTable = None,
        attribute_region: AttributeTable = None,
        regions: Union[List[str], None] = None
    ) -> pd.DataFrame:
        """
        Format the CapacityFactor input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - df_reference_capacity_factor: data frame of regional capacity factors 
            for technologies that vary (others revert to default)

        Keyword Arguments
        -----------------
        - attribute_technology: AttributeTable for technology, used to separate 
            technologies from storage and identify primary fuels. If None, 
            defaults to ModelAttributes attribute table.
        - attribute_region: AttributeTable for regions. If None, defaults to 
            ModelAttributes attribute table.
        - regions: regions to keep in capacity factor table
        """
        ##  INITIALIZATION

        # check fields
        fields_req = [self.field_nemomod_region, self.field_nemomod_time_slice]
        sf.check_fields(df_reference_capacity_factor, fields_req)

        # attribute tables
        attribute_technology = (
            self.get_attribute_entc()
            if (attribute_technology is None) 
            else attribute_technology
        )

        attribute_region = (
            self.get_attribute_region()
            if (attribute_region is None) 
            else attribute_region
        )
        

        ##  GET CLIMATE CHANGE FACTORS

        # hydropower
        (
            df_gnrl_ccf_hydropower, 
            field_gnrl_ccf_hydropower
        ) = self.get_gnrl_ccf_hydropower_factor_df(
            df_elec_trajectories
        )
    

        # regions to keep
        regions = self.model_attributes.get_region_list_filtered(regions, attribute_region = attribute_region)
        regions_keep = set(attribute_region.key_values) & set(regions)
        regions_keep = (
            (regions_keep & set(df_reference_capacity_factor[self.field_nemomod_region])) 
            if (self.field_nemomod_region in df_reference_capacity_factor.columns) 
            else regions_keep
        )

        # reshape to long
        fields_melt = [x for x in df_reference_capacity_factor.columns if (x in attribute_technology.key_values)]
        df_out = pd.melt(
            df_reference_capacity_factor[
                df_reference_capacity_factor[self.field_nemomod_region].isin(regions_keep)
            ],
            [self.field_nemomod_region, self.field_nemomod_time_slice],
            fields_melt,
            self.field_nemomod_technology,
            self.field_nemomod_value
        )

        # add output fields
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_time_slice,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )


        ##  ADD IN CLIMATE CHANGE COMPONENTS
        
        df_out = pd.merge(
            df_out,
            df_gnrl_ccf_hydropower,
            how = "left"
        )
        
        df_out[field_gnrl_ccf_hydropower].fillna(1.0, inplace = True)
        df_out[self.field_nemomod_value] = (
            np.array(df_out[self.field_nemomod_value]) 
            * np.array(df_out[field_gnrl_ccf_hydropower])
        )

        df_out = (
            df_out
            .drop([field_gnrl_ccf_hydropower], axis = 1)
            .sort_index()
        )

        # ensure capacity factors are properly specified
        df_out[self.field_nemomod_value] = sf.vec_bounds(
            np.array(df_out[self.field_nemomod_value]), 
            (0, 1)
        )
        dict_return = {
            self.model_attributes.table_nemomod_capacity_factor: df_out
        }

        return dict_return



    def format_nemomod_table_capacity_to_activity_unit(self,
        regions: Union[List[str], None] = None,
        return_type: str = "table",
    ) -> pd.DataFrame:
        """
        Format the CapacityToActivityUnit input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: "table" or "value". If value, returns only the 
        CapacityToActivityUnit value for all techs (used in DefaultParams)
            * Based on configuration parameters
        """

        # first, get power units, swap to get energy unit equivalent, then get units for the default total energy variable
        units_power = self.model_attributes.get_unit("power").get_unit_key(
            self.model_attributes.configuration.get("power_units")
        )

        units_energy_power_equivalent = self.model_attributes.get_energy_power_swap(units_power)
        cau = self.model_attributes.get_energy_equivalent(
            units_energy_power_equivalent, 
            self.units_energy_nemomod,
        )

        if return_type == "table":
            df_out = pd.DataFrame({self.field_nemomod_value: [cau]})
            df_out = self.add_multifields_from_key_values(
                df_out, 
                [
                    self.field_nemomod_id,
                    self.field_nemomod_region, 
                    self.field_nemomod_technology
                ],
                regions = regions
            )

        elif return_type == "value":
            df_out = cau

        dict_return = {self.model_attributes.table_nemomod_capacity_to_activity_unit: df_out}

        return dict_return



    def format_nemomod_table_costs_technology(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        flag_dummy_price: Union[int, float] = -999,
        minimum_dummy_price: Union[int, float] = 100,
        regions: Union[List[str], None] = None,
        tables_with_dummy: List[str] = ["CapitalCost", "FixedCost", "VariableCost"]
    ) -> pd.DataFrame:
        """
        Format the CapitalCost, FixedCost, and VaribleCost input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: attribute table used for fuels. If None, defaults to 
            self.model_attributes default
        - flag_dummy_price: initial price to use, which is later replaced. 
            Should be a large magnitude negative number.
        - minimum_dummy_price: minimum price for dummy technologies
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - tables_with_dummy: list of tables to include dummy tech costs in. 
            Acceptable values are:

            * "CapitalCost"
            * "FixedCost"
            * "VariableCost"
        """

        # initialize some attribute components
        attr_enfu = (
            self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_enfu) 
            if not isinstance(attribute_fuel, AttributeTable) 
            else attribute_fuel
        )
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.model_attributes.subsec_name_enfu,
            "pycategory_primary_element"
        )

        dict_return = {}
        flag_dummy_price = min(flag_dummy_price, -1)

        # get some scalars (monetary and power)
        scalar_cost_capital = self.model_attributes.get_scalar(self.modvar_entc_nemomod_capital_cost, "monetary")
        scalar_cost_capital /= self.model_attributes.get_scalar(self.modvar_entc_nemomod_capital_cost, "power")
        scalar_cost_fixed = self.model_attributes.get_scalar(self.modvar_entc_nemomod_fixed_cost, "monetary")
        scalar_cost_fixed /= self.model_attributes.get_scalar(self.modvar_entc_nemomod_fixed_cost, "power")
        scalar_cost_variable = self.model_attributes.get_scalar(self.modvar_entc_nemomod_variable_cost, "monetary")
        scalar_cost_variable /= self.get_nemomod_energy_scalar(self.modvar_entc_nemomod_variable_cost)

        # CapitalCost
        df_append = (
            self.build_dummy_tech_cost(
                flag_dummy_price, 
                cost_type = "capital", 
                override_time_period_transformation = True
            ) 
            if ("CapitalCost" in tables_with_dummy) 
            else None
        )

        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_capital_cost,
                self.model_attributes.table_nemomod_capital_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                df_append = df_append,
                regions = regions,
                scalar_to_nemomod_units = scalar_cost_capital,
                var_bounds = (0, np.inf)
            )
        )
        
        # FixedCost
        df_append = (
            self.build_dummy_tech_cost(
                flag_dummy_price, 
                cost_type = "fixed", 
                override_time_period_transformation = True
            ) 
            if ("FixedCost" in tables_with_dummy) 
            else None
        )
        
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_fixed_cost,
                self.model_attributes.table_nemomod_fixed_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                df_append = df_append,
                regions = regions,
                scalar_to_nemomod_units = scalar_cost_fixed,
                var_bounds = (0, np.inf)
            )
        )

        
        ##  VariableCost -- Pull Variable O&M and Add Fuel Costs

        # get fuel costs -- specify energy & monetary units in terms of self.modvar_entc_nemomod_variable_cost
        units_enfu_costs_monetary = self.model_attributes.get_variable_characteristic(
            self.modvar_entc_nemomod_variable_cost,
            self.model_attributes.varchar_str_unit_monetary
        )
        arr_enfu_costs = self.model_energy.get_enfu_fuel_costs_per_energy(
            df_elec_trajectories,
            modvar_for_units_energy = self.modvar_entc_nemomod_variable_cost,
            units_monetary = units_enfu_costs_monetary
        )

        # get variable costs, add fuel costs, and create data frame to pass
        df_entc_variable_costs = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.modvar_entc_nemomod_variable_cost,
            return_type = "data_frame",
        )
        
        # add time period and get categories associated with each field (ordered the same as fields)
        df_entc_variable_costs[self.model_attributes.dim_time_period] = df_elec_trajectories[self.model_attributes.dim_time_period]
        cats_df_variable_costs = self.model_attributes.get_variable_categories(self.modvar_entc_nemomod_variable_cost)

        # loop over techs and find any associated fuels; if so, pull from arr_enfu_costs
        dict_cat_pp_to_cat_enfu = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            f"electricity_generation_{pycat_enfu}",
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        for cat_entc in enumerate(cats_df_variable_costs):
            
            j, cat_entc = cat_entc
            cat_enfu = dict_cat_pp_to_cat_enfu.get(cat_entc)
            if cat_enfu is None:
                continue
                
            ind_enfu = attr_enfu.get_key_value_index(cat_enfu)
            field_varcost = list(df_entc_variable_costs.columns)[j]
            
            df_entc_variable_costs[field_varcost] = np.array(df_entc_variable_costs[field_varcost]) + arr_enfu_costs[:, ind_enfu]
        
        #
        # dummy techs are high-cost technologies that help ensure there is no unmet demand in the system if other constraints create an issue
        # https://sei-international.github.io/NemoMod.jl/stable/model_concept/
        #
        
        df_append = (
            self.build_dummy_tech_cost(
                flag_dummy_price, 
                cost_type = "variable", 
                override_time_period_transformation = True
            ) 
            if ("VariableCost" in tables_with_dummy) 
            else None
        )

        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_entc_variable_costs,
                self.modvar_entc_nemomod_variable_cost,
                self.model_attributes.table_nemomod_variable_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                df_append = df_append,
                dict_fields_to_pass = {self.field_nemomod_mode: self.cat_enmo_gnrt},
                regions = regions,
                scalar_to_nemomod_units = scalar_cost_variable,
                var_bounds = (0, np.inf)
            )
        )
        

        # next, replace costs for dummy techs with costs that are at least 10x higher than max of other costs where applicable
        cats_entc_dummy_with_high_cost = self.get_enfu_cats_with_high_dummy_tech_costs(return_type = "dummy_fuel_techs")
        cats_entc_dummy = list(self.get_dummy_fuel_techs().values())
        cats_no_cost = set(cats_entc_dummy) - set(cats_entc_dummy_with_high_cost)
        
        for table_name in list(dict_return.keys()):
            df_tmp = dict_return.get(table_name)

            # set high price relative to other prices & determine where to keep specified costs (vec_high_cost_bool = 0 if it is contained in cats_no_cost)
            price_high = max(np.round(max(df_tmp[self.field_nemomod_value])*2)*10 + 10, minimum_dummy_price)
            vec_high_cost_bool = np.array([(x not in cats_no_cost) for x in list(df_tmp[self.field_nemomod_technology])]).astype(int)
            vals_new = np.array(df_tmp[self.field_nemomod_value].replace({flag_dummy_price: price_high})) * vec_high_cost_bool
            
            df_tmp[self.field_nemomod_value] = vals_new

            dict_return.update({table_name: df_tmp})
        
        return dict_return



    def format_nemomod_table_costs_storage(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the CapitalCostStorage input tables for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # get some scalars (monetary and power)
        scalar_cost_capital_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_capital_cost_storage, "monetary")
        scalar_cost_capital_storage /= self.get_nemomod_energy_scalar(self.modvar_enst_nemomod_capital_cost_storage)

        # CapitalCostStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_capital_cost_storage,
                self.model_attributes.table_nemomod_capital_cost_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                regions = regions,
                scalar_to_nemomod_units = scalar_cost_capital_storage,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format DefaultParameters for NemoMod
    def format_nemomod_table_default_parameters(self,
        attribute_nemomod_table: AttributeTable = None,
        field_default_values: str = "default_value"
    ) -> pd.DataFrame:
        """
        Format the DefaultParameters input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - attribute_nemomod_table: NemoMod tables AttributeTable that includes 
            default values stored in the field 'field_default_values'
        - field_default_values: string giving the name in the 
            attribute_nemomod_table with default values
        """

        attribute_nemomod_table = (
            self.model_attributes.get_other_attribute_table("nemomod_table") 
            if (attribute_nemomod_table is None) 
            else attribute_nemomod_table
        )

        # check fields (key is always contained in an attribute table if it is successfully initialized)
        sf.check_fields(attribute_nemomod_table.table, [field_default_values])

        # get dictionary and update parameters
        dict_repl = attribute_nemomod_table.field_maps[f"{attribute_nemomod_table.key}_to_{field_default_values}"].copy()
        dict_repl.update(
            self.format_nemomod_table_capacity_to_activity_unit(
                return_type = "value"
            )
        )
        dict_repl.update(
            self.format_nemomod_table_discount_rate(
                return_type = "value"
            )
        )

        # build output table
        df_out = attribute_nemomod_table.table[[attribute_nemomod_table.key]].copy().rename(columns = {attribute_nemomod_table.key: self.field_nemomod_table_name})
        df_out[self.field_nemomod_value] = df_out[self.field_nemomod_table_name].replace(dict_repl)
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id
            ]
        )

        dict_return = {self.model_attributes.table_nemomod_default_params: df_out}

        return dict_return



    def format_nemomod_table_discount_rate(self,
        regions: Union[List[str], None] = None,
        return_type: str = "table"
    ) -> pd.DataFrame:
        """
        Format the DiscountRate input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs,
            and reference tables.

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: "table" or "value". If value, returns only the 
            DiscountRate
            * Based on configuration specification of discount_rate
        """

        discount_rate = self.model_attributes.configuration.get("discount_rate")
        df_out = pd.DataFrame({self.field_nemomod_value: [discount_rate]})

        if return_type == "table":
            df_out = self.add_multifields_from_key_values(
                df_out, 
                [
                    self.field_nemomod_id, 
                    self.field_nemomod_region
                ],
                regions = regions
            )

        elif return_type == "value":
            df_out = discount_rate

        dict_out = {self.model_attributes.table_nemomod_discount_rate: df_out}

        return dict_out



    def format_nemomod_table_emissions_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        attribute_time_period: Union[AttributeTable, None] = None,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the EmissionsActivityRatio input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: attribute table used for fuels. If None,
            defaults to self.model_attributes
        - attribute_technology: attribute table used for technology. If None,
            defaults to self.model_attributes
        - attribute_time_period: attribute table used for time period. If None,
            defaults to self.model_attributes
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION

        attr_enfu = (
            self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_enfu) 
            if not isinstance(attribute_fuel, AttributeTable) 
            else attribute_fuel
        )
        attr_entc = (
            self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_entc) 
            if not isinstance(attribute_technology, AttributeTable) 
            else attribute_technology
        )
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        # get technology info and cat to fuel dictionary
        dict_tech_info = self.get_tech_info_dict(attribute_fuel = attribute_fuel)
        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            f"electricity_generation_{pycat_enfu}",
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        dict_fuel_to_techs = sf.reverse_dict(dict_techs_to_fuel, allow_multi_keys = True)
        dict_pp_tech_to_fuel = dict_tech_info.get("dict_pp_tech_to_fuel")

        # get some categories and ordered indexing to convert
        cat_entc_pp_waste = self.get_entc_cat_for_integration("wste") 
        cats_entc_ordered = [x for x in attr_entc.key_values if x in dict_pp_tech_to_fuel.keys()]
        inds_enfu_extract = [attr_enfu.get_key_value_index(dict_pp_tech_to_fuel.get(x)) for x in cats_entc_ordered]

        # set required variables for emission factors and initialize output dictionary
        list_modvars_enfu_to_tech = [
            (self.modvar_enfu_ef_combustion_stationary_ch4, self.modvar_entc_ef_scalar_ch4),
            (self.modvar_enfu_ef_combustion_co2, self.modvar_entc_ef_scalar_co2),
            (self.modvar_enfu_ef_combustion_stationary_n2o, self.modvar_entc_ef_scalar_n2o)
        ]

        df_out = []


        ########################################################################
        #    1. ESTIMATE EMISSIONS FACTORS FOR ELECTRICITY GENERATION TECHS    #
        ########################################################################

        # get a dictionary of adjusted fuel combustion factors in NemoMod emission mass/NemoMod units energy - used in mining and extraction
        dict_enfu_arrs_efs_scaled_to_nemomod = {}

        # loop over fuel emission factors to specify for each technology
        for modvars in enumerate(list_modvars_enfu_to_tech):
            ind, modvars = modvars
            modvar, modvar_scalar = modvars

            # get the fuel factors
            arr_enfu_tmp = self.model_attributes.extract_model_variable(#
                df_elec_trajectories, 
                modvar, 
                expand_to_all_cats = False,
                override_vector_for_single_mv_q = True, 
                return_type = "array_base", 
            )

            # convert emissions mass (configuration) and energy (self.modvar_enfu_energy_demand_by_fuel_total) to the units for NemoMod
            arr_enfu_tmp *= self.model_attributes.get_scalar(modvar, "mass")
            arr_enfu_tmp /= self.get_nemomod_energy_scalar(modvar)
            dict_enfu_arrs_efs_scaled_to_nemomod.update({modvar: arr_enfu_tmp})
            # get ordered indices for each fuel associated with a generation tech
            arr_enfu_tmp = arr_enfu_tmp[:, inds_enfu_extract]
            
            # expand to tech
            arr_entc_tmp = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                arr_enfu_tmp,
                None,
                missing_vals = 0.0,
                output_cats = cats_entc_ordered,
                output_subsec = self.model_attributes.subsec_name_entc,
            )

            # apply scalar
            arr_enfu_scalar = self.model_attributes.extract_model_variable(#
                df_elec_trajectories,
                modvar_scalar,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
                var_bounds = (0, 1),
            )

            arr_entc_tmp *= arr_enfu_scalar


            ##  FORMAT AS DATA FRAME

            emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            df_entc_tmp = pd.DataFrame(arr_entc_tmp, columns = attr_entc.key_values)
            df_entc_tmp = df_entc_tmp[cats_entc_ordered]

            # add some key fields (emission and year)
            df_entc_tmp[self.field_nemomod_emission] = emission
            df_entc_tmp[self.field_nemomod_mode] = self.cat_enmo_gnrt
            df_entc_tmp = self.model_attributes.exchange_year_time_period(
                df_entc_tmp,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period,
                direction = self.direction_exchange_year_time_period
            )

            # melt into a long form table
            df_entc_tmp = pd.melt(
                df_entc_tmp,
                [
                    self.field_nemomod_emission, 
                    self.field_nemomod_mode, 
                    self.field_nemomod_year
                ],
                cats_entc_ordered,
                var_name = self.field_nemomod_technology,
                value_name = self.field_nemomod_value
            )

            if len(df_out) == 0:
                df_out = [df_entc_tmp for x in range(len(list_modvars_enfu_to_tech))]
            else:
                df_out[ind] = df_entc_tmp[df_out[0].columns]


        ################################################
        #    2. ESTIMATE INTEGRATED WASTE EMISSIONS    #
        ################################################
        
        df_enfu_efs_waste = self.get_integrated_waste_emissions_activity_ratio(
            df_elec_trajectories,
            attribute_technology = attribute_technology,
            attribute_time_period = attribute_time_period
        )

        # concatenate and replace waste if applicable
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
        if df_enfu_efs_waste is not None:
            df_out = df_out[~df_out[self.field_nemomod_technology].isin([cat_entc_pp_waste])]
            df_out = pd.concat([df_out, df_enfu_efs_waste], axis = 0).reset_index(drop = True)

        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_value,
                self.field_nemomod_year
            ],
            regions = regions
        )


        #####################################################################################
        #    3. ESTIMATE EMISSIONS FACTORS FOR MINING AND EXTRACTION AND FUEL PRODUCTION    #
        #####################################################################################

        df_out_me = self.get_entc_emissions_activity_ratio_comp_me(
            df_elec_trajectories,
            dict_enfu_arrs_efs_scaled_to_nemomod,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            attribute_time_period = attribute_time_period,
            regions = regions
        )

        df_out_fp = self.get_entc_emissions_activity_ratio_comp_fp(
            df_elec_trajectories,
            regions = regions
        )
        
        # concatenate and filter out 0s
        df_out = sf.filter_data_frame_by_group(
            pd.concat([df_out, df_out_me, df_out_fp], axis = 0),  
            [
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology
            ],
            self.field_nemomod_value
        )

        # add keys and clean up
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_value,
                self.field_nemomod_year
            ],
            override_time_period_transformation = True,
            regions = regions
        )
    
        dict_return = {self.model_attributes.table_nemomod_emissions_activity_ratio: df_out}

        return dict_return



    ##  format FixedCost for NemoMod
    def format_nemomod_table_fixed_cost(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the FixedCost input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    ##  format InterestRateStorage for NemoMod
    def format_nemomod_table_interest_rate_storage(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the InterestRateStorage input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    ##  format InterestRateTechnology for NemoMod
    def format_nemomod_table_interest_rate_technology(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the InterestRateTechnology input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    def format_nemomod_table_input_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        max_ratio: float = 1000000.0,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the InputActivityRatio input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuel
        - attribute_technology: AttributeTable for technology, used to separate 
            technologies from storage and identify primary fuels.
        - max_ratio: replacement for any input_activity_ratio values derived 
            from efficiencies of 0
        - regions: regions to specify. If None, defaults to configuration 
            regions

        Model Notes
        -----------
        To model transmission losses in the absence of a network, the 
            InputActivityRatio for electricity-consuming technologies is 
            inflated by *= 1/(1 - loss). Upon extraction, demands are reduced by 
            *= (1 - loss) in the 
            retrieve_nemomod_tables_fuel_production_demand_and_trade()
            method.
        """

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )


        # cat to fuel dictionary + reverse
        dict_tech_info = self.get_tech_info_dict(
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology
        )
        dict_techs_to_fuel = dict_tech_info.get("dict_pp_tech_to_fuel")
        dict_tech_to_storage = dict_tech_info.get("dict_storage_techs_to_storage")     

        # revise some dictionaries for the output table
        all_techs_generate = [x for x in attribute_technology.key_values if x not in dict_tech_to_storage.keys()]
        dict_tech_to_mode = dict((x, self.cat_enmo_gnrt) for x in all_techs_generate)

        # update storage mode/fuel (will add keys to dict_techs_to_fuel)
        for k in dict_tech_to_storage.keys():
            dict_techs_to_fuel.update({k: self.cat_enfu_elec})
            dict_tech_to_mode.update({k: self.cat_enmo_stor})

        dict_return = {}


        #################################################################
        #    BUILD ELEC GENERATION AND STORAGE (NOT FUEL PRODUCTION)    #
        #################################################################

        # Initialize InputActivityRatio
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_efficiency_factor_technology,
                self.model_attributes.table_nemomod_input_activity_ratio,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                regions = regions,
                var_bounds = (0, np.inf)
            )
        )

        # make some modifications to generation
        df_iar = dict_return.get(self.model_attributes.table_nemomod_input_activity_ratio)
        df_iar[self.field_nemomod_fuel] = df_iar[self.field_nemomod_technology].replace(dict_techs_to_fuel)
        df_iar[self.field_nemomod_mode] = df_iar[self.field_nemomod_technology].replace(dict_tech_to_mode)

        # convert efficiency to input_activity_ratio_ratio
        df_iar[self.field_nemomod_value] = np.nan_to_num(1/np.array(df_iar[self.field_nemomod_value]), max_ratio, posinf = max_ratio)
        # re-sort using hierarchy
        df_iar = self.add_multifields_from_key_values(
            df_iar,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions,
            # the time period transformation was already applied in format_model_variable_as_nemomod_table, so we override additional transformation
            override_time_period_transformation = True
        )
        


        ###########################
        #  ADD FUEL PRODUCTION    #
        ###########################

        # dictionary mapping applicable fuel categories to input_activity_ratio/output_activity_ratio
        dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables 
        # initialize new dataframe with previous
        df_append = [df_iar]

        for fuel in dict_fuel_cats.keys():
            
            # filter in input activity ratio (iar)
            dict_iar_oar_var = dict_fuel_cats.get(fuel)
            modvar_iar = dict_iar_oar_var.get(self.key_iar)

            if modvar_iar is not None:

                df_tmp = self.format_model_variable_as_nemomod_table( 
                    df_elec_trajectories,
                    modvar_iar,
                    self.model_attributes.table_nemomod_input_activity_ratio,
                    [
                        self.field_nemomod_id,
                        self.field_nemomod_year,
                        self.field_nemomod_region
                    ],
                    self.field_nemomod_technology,
                    regions = regions,
                    var_bounds = (0, np.inf)
                ).get(self.model_attributes.table_nemomod_input_activity_ratio)
                
                # drop techs that are all 0
                ids_filt = []
                for i in df_tmp.groupby(self.field_nemomod_technology):
                    i, df = i
                    unique_vals = list(df[self.field_nemomod_value].unique())
                    ids_filt += list(df[self.field_nemomod_id]) if not ((len(unique_vals) == 1) and (0.0 in unique_vals)) else []
                df_tmp = df_tmp[df_tmp[self.field_nemomod_id].isin(ids_filt)]
                
                # add fuel and mode
                df_tmp[self.field_nemomod_fuel] = fuel
                df_tmp[self.field_nemomod_mode] = df_tmp[self.field_nemomod_technology].replace(dict_tech_to_mode)   
                df_append.append(df_tmp) if (len(df_tmp) > 0) else None
        


        ###########################################################
        #    ADD DUMMY TECHS (SET TO ONE TO TRACK IMPORTS/USE)    #
        ###########################################################
        
        # input comes from dummy fuel
        df_iar_dummies = self.get_dummy_fuel_techs(
            attribute_fuel = attribute_fuel, 
            return_type = "pd.DataFrame"
        )

        # finish with other variables
        df_iar_dummies[self.field_nemomod_fuel] = self.get_dummy_fuel_name()
        df_iar_dummies[self.field_nemomod_value] = 1.0
        df_iar_dummies[self.field_nemomod_mode] = self.cat_enmo_gnrt

        # add key values, like year
        df_iar_dummies = self.add_multifields_from_key_values(
            df_iar_dummies,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        df_append.append(df_iar_dummies)

        
        ##  ADD IN TRANSMISSION LOSS IN ELECTRICITY

        df_transmission_loss = self.format_model_variable_as_nemomod_table( 
            df_elec_trajectories,
            self.modvar_enfu_transmission_loss_frac_electricity,
            "TMP",
            [
                self.field_nemomod_year,
                self.field_nemomod_region
            ],
            self.field_nemomod_fuel,
            regions = regions,
            var_bounds = (0, 1)
        ).get("TMP")
        df_transmission_loss[self.field_nemomod_value] *= -1.0
        df_transmission_loss[self.field_nemomod_value] += 1.0

        # merge, multiply, and drop the scalar
        field_transmission_merge = "TRANSMISSION_TMP"
        df_transmission_loss.rename(columns = {self.field_nemomod_value: field_transmission_merge}, inplace = True)
        
        df_out = pd.merge(
            pd.concat(df_append, axis = 0),
            df_transmission_loss,
            how = "left"
        ).fillna(1.0)

        df_out[self.field_nemomod_value] = np.array(df_out[self.field_nemomod_value])/np.array(df_out[field_transmission_merge])
        df_out.drop([field_transmission_merge], axis = 1, inplace = True)


        ##  RE-SORT USING HIERARCHY AND PREP FOR NEMOMOD

        df_append = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            # the time period transformation was already applied in format_model_variable_as_nemomod_table, so we override additional transformation
            override_time_period_transformation = True,
            regions = regions
        )

        dict_return.update({self.model_attributes.table_nemomod_input_activity_ratio: df_append})

        return dict_return



    def format_nemomod_table_min_share_production(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        modvar_import_fraction: str = None,
        regions: Union[List[str], None] = None,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
    ) -> pd.DataFrame:
        """
        Format the MinShareProduction input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables. Used to implement electrification in 
            fuel-production inputs.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuel
        - attribute_technology: AttributeTable used to denote technologies with 
            MinShareProductions
        - modvar_import_fraction: SISEPUEDE model variable giving the import 
            fraction. If None, default to 
            NonElectricEnergy.modvar_enfu_frac_fuel_demand_imported
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        """
        # do some initialization
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        dict_tech_info = self.get_tech_info_dict(
            attribute_technology = attribute_technology
        )
        modvar_import_fraction = (
            self.modvar_enfu_frac_fuel_demand_imported 
            if (modvar_import_fraction is None) 
            else modvar_import_fraction
        )

        # get production, imports, exports, and demands to adjust import fractions
        if (tuple_enfu_production_and_demands is None) and (df_elec_trajectories is None):
            raise ValueError(f"Error in format_nemomod_table_min_share_production: tuple_enfu_production_and_demands and df_elec_trajectories cannot both be None.")

        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(
                df_elec_trajectories, 
                target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
            )
            if tuple_enfu_production_and_demands is None
            else tuple_enfu_production_and_demands
        )


        ##  ADJUST IMPORT FRACTIONS TO ACCOUNT FOR THE INCLUSION OF EXPORTS IN SpecifiedAnnualDemands
        
        global df_et
        df_et = df_elec_trajectories.copy()

        print(modvar_import_fraction)
        print(df_et.shape)

        arr_enfu_import_fractions = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            modvar_import_fraction,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # scale import fractions by ratio of demands to (demands + exports)
        arr_enfu_import_fractions_adj = np.nan_to_num(
            tuple_enfu_production_and_demands[0]/(tuple_enfu_production_and_demands[0] + tuple_enfu_production_and_demands[2]),
            1.0,
            posinf = 1.0
        )
        arr_enfu_import_fractions_adj *= arr_enfu_import_fractions

        # pass via dummy dataframe
        df_fracs_adj = [
            df_elec_trajectories[
                [x for x in df_elec_trajectories.columns if x in self.model_attributes.sort_ordered_dimensions_of_analysis]
            ].copy()
        ]
        df_fracs_adj += [
            self.model_attributes.array_to_df(
                arr_enfu_import_fractions_adj,
                modvar_import_fraction,
                reduce_from_all_cats_to_specified_cats = True
            )
        ]
        df_fracs_adj = pd.concat(df_fracs_adj, axis = 1)


        # import fractions are set as minimum shares of production and add technology
        df_out = self.format_model_variable_as_nemomod_table( 
            df_fracs_adj,
            modvar_import_fraction,
            self.model_attributes.table_nemomod_min_share_production,
            [
                self.field_nemomod_id,
                self.field_nemomod_year,
                self.field_nemomod_region
            ],
            self.field_nemomod_fuel,
            regions = regions,
            var_bounds = (0, 1)
        ).get(self.model_attributes.table_nemomod_min_share_production)

        df_out[self.field_nemomod_technology] = df_out[self.field_nemomod_fuel].replace(self.get_dummy_fuel_techs())

        # setup for NemoMod
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            override_time_period_transformation = True,
            regions = regions
        )


        ##  NEXT, GET EXOGENOUSLY SPECIFIED MinShareProduction VALUES (ADJUSTED FOR IMPORTS) 

        vec_entc_elec_demand_frac_from_tech_lower_limit = self.estimate_production_share_from_activity_limits(
            df_elec_trajectories,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
        )
        
        # copy and add fractions of demand represented by TechnologyTotalAnnualLowerLimit
        # --only used to adjust MSPs downward
        arr_enfu_import_fractions_adj_for_msp_adj = arr_enfu_import_fractions_adj.copy()
        arr_enfu_import_fractions_adj_for_msp_adj[:, self.ind_enfu_elec] += vec_entc_elec_demand_frac_from_tech_lower_limit
        
        df_entc_msp = self.get_entc_import_adjusted_msp(
            df_elec_trajectories,
            arr_enfu_import_fractions_adj_for_msp_adj,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            dict_tech_info = dict_tech_info,
            regions = regions,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
        )

        df_out = self.add_multifields_from_key_values(
            pd.concat([df_out, df_entc_msp[df_out.columns]], axis = 0),
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            override_time_period_transformation = True,
            regions = regions
        )

        dict_return = {self.model_attributes.table_nemomod_min_share_production: df_out}
        
        return dict_return



    def format_nemomod_table_min_storage_charge(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_storage: AttributeTable = None,
        field_attribute_min_charge: str = "minimum_charge_fraction",
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the MinStorageCharge input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_storage: AttributeTable used to identify minimum storage 
            charge by storage type. If None, defaults to ModelAttribute 
            cat_storage table
        - field_attribute_min_charge: field in attribute_storage containing the 
            minimum storage charge fraction by storage type
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        ##
        # NOTE: ADD A CHECK IN THE StorageStartLevel TABLE TO COMPARE TO MINIMUM STORAGE CHARGE AND SELECT MAX BETWEEN THE TWO

        # set some defaults
        attribute_storage = (
            self.model_attributes.get_attribute_table(self.subsec_name_enst) 
            if (attribute_storage is None) 
            else attribute_storage
        )
        
        # initialize storage info
        dict_strg_to_min_charge = attribute_storage.field_maps.get(
            f"{attribute_storage.key}_to_{field_attribute_min_charge}"
        )

        all_storage = list(dict_strg_to_min_charge.keys())
        df_out = pd.DataFrame({
            self.field_nemomod_storage: all_storage,
            self.field_nemomod_value: [dict_strg_to_min_charge.get(x) for x in all_storage]
        })

        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_storage,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        dict_return = {self.model_attributes.table_nemomod_min_storage_charge: df_out}

        return dict_return



    ##  format MinimumUtilization for NemoMod
    def format_nemomod_table_minimum_utilization(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the MinimumUtilization input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    ##  format ModelPeriodEmissionLimit for NemoMod
    def format_nemomod_table_model_period_emission_limit(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the ModelPeriodEmissionLimit input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    ##  format ModelPeriodExogenousEmission for NemoMod
    def format_nemomod_table_model_period_exogenous_emission(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the ModelPeriodExogenousEmission input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    def format_nemomod_table_operational_life(self,
        attribute_fuel: AttributeTable = None,
        attribute_storage: AttributeTable = None,
        attribute_technology: AttributeTable = None,
        operational_life_dummies: Union[float, int] = 250,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the OperationalLife and OperationalLifeStorage input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuel, used to set dummy fuel 
            supplies as a technology. If None, use ModelAttributes default.
        - attribute_storage: AttributeTable for storage, used to build 
            OperationalLifeStorage from Technology. If None, use 
            ModelAttributes default.
        - attribute_technology: AttributeTable for technology, used to identify 
            operational lives of generation and storage technologies. If None, 
            use ModelAttributes default.
        - operational_life_dummies: Operational life for dummy technologies that 
            are entered to account for fuel inputs.
        - regions: regions to specify. If None, defaults to configuration 
            regions

        Notes:
        - Validity checks for operational lives are performed on initialization 
            of the ModelAttributes class.
        """

        # set some defaults
        attribute_storage = (
            self.model_attributes.get_attribute_table(self.subsec_name_enst) 
            if (attribute_storage is None) 
            else attribute_storage
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        pycat_strg = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enst, 
            "pycategory_primary_element",
        )

        # cat storage dictionary
        dict_storage_techs_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_strg,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        # get the life time
        dict_techs_to_operational_life = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            "operational_life",
            return_type = dict,
            skip_none_q = True,
        )

        # get dummy techs
        dict_fuels_to_dummy_techs = self.get_dummy_generation_fuel_techs(
            attribute_technology = attribute_technology,
        )
        all_techs = sorted(list(dict_techs_to_operational_life.keys())) + sorted(list(dict_fuels_to_dummy_techs.values()))
        all_ols = [dict_techs_to_operational_life.get(x, operational_life_dummies) for x in all_techs]

        # initliaze data frame
        df_operational_life = pd.DataFrame({
            self.field_nemomod_technology: all_techs,
            self.field_nemomod_value: all_ols
        })
        # split off and perform some cleaning
        df_operational_life_storage = (
            df_operational_life[
                df_operational_life[self.field_nemomod_technology]
                .isin(dict_storage_techs_to_storage.keys())
            ]
            .copy()
        )

        df_operational_life_storage[self.field_nemomod_technology] = (
            df_operational_life_storage[self.field_nemomod_technology]
            .replace(dict_storage_techs_to_storage)
        )

        df_operational_life_storage.rename(
            columns = {self.field_nemomod_technology: self.field_nemomod_storage}, 
            inplace = True,
        )

        df_operational_life = (
            df_operational_life[
                ~df_operational_life[self.field_nemomod_technology]
                .isin(dict_storage_techs_to_storage.keys())
            ]
            .copy()
        )

        # add required fields
        fields_reg = [self.field_nemomod_id, self.field_nemomod_region]
        df_operational_life = self.add_multifields_from_key_values(
            df_operational_life, 
            fields_reg,
            regions = regions,
        )
        df_operational_life_storage = self.add_multifields_from_key_values(
            df_operational_life_storage, 
            fields_reg,
            regions = regions,
        )

        dict_return = {
            self.model_attributes.table_nemomod_operational_life: df_operational_life,
            self.model_attributes.table_nemomod_operational_life_storage: df_operational_life_storage
        }

        return dict_return



    def format_nemomod_table_output_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology:  Union[AttributeTable, None] = None,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the OutputActivityRatio input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuels, used to identify fuels that
            require dummy supply techs
        - attribute_technology:  AttributeTable for technology, used to separate 
            technologies from storage and identify primary fuels.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """
        
        ##  CATEGORY AND ATTRIBUTE INITIALIZATION

        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )


        ##  BUILD DUMMY SUPPLY TECHNOLOGIES FOR FUELS WITHOUT SPECIFIED OUTPUT ACTIVITY RATIOS

        df_out_dummies = self.get_dummy_fuel_techs(
            attribute_fuel = attribute_fuel, 
            return_type = "pd.DataFrame"
        )

        # Initialize OutputActivityRatio and add dummies
        techs_oar_elec = self.get_entc_cat_by_type(["pp", "st"])
        df_oar = pd.DataFrame({self.field_nemomod_technology: techs_oar_elec})
        df_oar[self.field_nemomod_fuel] = self.cat_enfu_elec
        df_oar = pd.concat([df_oar, df_out_dummies], axis = 0).reset_index(drop = True)

        # finish with other variables
        df_oar[self.field_nemomod_value] = 1.0
        df_oar[self.field_nemomod_mode] = self.cat_enmo_gnrt

        # add key values, like year
        df_oar = self.add_multifields_from_key_values(
            df_oar,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )


        ##  ADD IN EXPLICITLY SPECIFIED OUTPUT ACTIVITY RATIOS (USED IN Mining and Extraction AND Fuel Processing)

        dict_fuel_cats = self.dict_entc_fuel_categories_to_fuel_variables
        df_out = [df_oar]

        for fuel in dict_fuel_cats.keys():
            
            # filter on output activity ratio (oar)
            dict_iar_oar_var = dict_fuel_cats.get(fuel)
            modvar_oar = dict_iar_oar_var.get(self.key_oar)

            if modvar_oar is not None:

                df_tmp = self.format_model_variable_as_nemomod_table( 
                    df_elec_trajectories,
                    modvar_oar,
                    self.model_attributes.table_nemomod_output_activity_ratio,
                    [
                        self.field_nemomod_id,
                        self.field_nemomod_year,
                        self.field_nemomod_region
                    ],
                    self.field_nemomod_technology,
                    regions = regions,
                    var_bounds = (0, np.inf)
                ).get(self.model_attributes.table_nemomod_output_activity_ratio)
                
                # drop techs that are all 0
                ids_filt = []
                for i in df_tmp.groupby(self.field_nemomod_technology):
                    i, df = i
                    unique_vals = list(df[self.field_nemomod_value].unique())
                    ids_filt += list(df[self.field_nemomod_id]) if not ((len(unique_vals) == 1) and (0.0 in unique_vals)) else []
                df_tmp = df_tmp[df_tmp[self.field_nemomod_id].isin(ids_filt)]
                
                # add fuel and mode
                df_tmp[self.field_nemomod_fuel] = fuel
                df_tmp[self.field_nemomod_mode] = self.cat_enmo_gnrt   
                df_out.append(df_tmp) if (len(df_tmp) > 0) else None
        

        ## ADD DUMMY FUEL-TECH THAT FEEDS DUMMY FUEL (USED FOR ACCOUNTING)

        # add dummy fuel for tech production accounting from dummy techs
        df_append = pd.DataFrame({
            self.field_nemomod_fuel: [self.get_dummy_fuel_name()],
            self.field_nemomod_technology: [self.get_dummy_fuel_name(return_type = "tech")],
            self.field_nemomod_value: [1.0],
            self.field_nemomod_mode: [self.cat_enmo_gnrt]
        })
        # format with dimensions
        df_append = self.add_multifields_from_key_values(
            df_append,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )
        # append to output
        df_out.append(df_append)


        ##  SORT USING HIERARCHY AND PREPARE FOR NEMOMOD

        df_out = self.add_multifields_from_key_values(
            pd.concat(df_out, axis = 0),
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            # the time period transformation was already applied in format_model_variable_as_nemomod_table, so we override additional transformation
            override_time_period_transformation = True,
            regions = regions
        )

        # ensure changes are made to dict
        dict_return = {self.model_attributes.table_nemomod_output_activity_ratio: df_out}

        return dict_return



    def format_nemomod_table_re_min_production_target(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        modvar_import_fraction: Union[str, None] = None,
        modvar_renewable_target: Union[str, None] = None,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the REMinProductionTarget (renewable energy minimum production 
            target) input table for NemoMod based on SISEPUEDE configuration 
            parameters, input variables, integrated model outputs, and reference 
            tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used to specify fuels
        - modvar_import_fraction: model variable specifying fuel import 
            fractions. Defaults to self.modvar_enfu_frac_fuel_demand_imported
        - modvar_renewable_target: model variable used to specify renewable 
            energy target fractions. Defaults to 
            self.modvar_enfu_nemomod_renewable_production_target
        - regions: regions to specify. If None, defaults to configuration 
            regions

        Important Note - Conflicting Constratints
        -----------------------------------------
        SISEPUEDE specifies fuel import fractions using a dummy technology in 
            combination with (a) very high costs and (b) MinShareProduction. 
            For a given fuel, if the sum of the import fuel fraction 
            (specified as technology called "supply_FUELNAME") specified in
            MinShareProduction and the REMinProductionTarget exceed 1, then
            NemoMod will return an infeasibility. 

        To avoid this problem, SISEPUEDE specified REMinProductionTarget as a 
            fraction of non-imported energy. This function first checks for 
            the specification of import fracitons; if present, it interprets
            the specified minimum renewable production target RPT as 
            RPT' = RPT(1 - MSP).
        
        """
        # perform some initialization
        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
        modvar_import_fraction = self.modvar_enfu_frac_fuel_demand_imported if (modvar_import_fraction is None) else modvar_import_fraction
        modvar_renewable_target = self.modvar_enfu_nemomod_renewable_production_target if (modvar_renewable_target is None) else modvar_renewable_target

        # get imports and renewable energy minimum production targets 
        arr_enfu_imports = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            modvar_import_fraction,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_enfu_re_target = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            modvar_renewable_target,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        # adjust and convert to dataframe
        arr_enfu_re_target_adj = (1 - arr_enfu_imports)*arr_enfu_re_target
        df_return = self.model_attributes.array_to_df(
            arr_enfu_re_target_adj,
            modvar_renewable_target,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_return[self.model_attributes.dim_time_period] = list(df_elec_trajectories[self.model_attributes.dim_time_period])

        # check technologies that are optional from optional input
        df_return = self.format_model_variable_as_nemomod_table(
            df_return,
            self.modvar_enfu_nemomod_renewable_production_target,
            "TMP",
            [
                self.field_nemomod_id,
                self.field_nemomod_year,
                self.field_nemomod_region
            ],
            self.field_nemomod_fuel,
            regions = regions,
            var_bounds = (0, 1)
        ).get("TMP")

        # filter out groups that are all 0
        df_return = sf.filter_data_frame_by_group(
            df_return,  
            [
                self.field_nemomod_region,
                self.field_nemomod_fuel
            ],
            self.field_nemomod_value
        )

        return {self.model_attributes.table_nemomod_re_min_production_target: df_return}



    def format_nemomod_table_re_tag_technology(self,
        df_elec_trajectories: Union[pd.DataFrame, None],
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the RETagTechnology (renewable energy technology tag) input table 
            for NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories.
            * Note: If None, specifies renewable energy *only* according to 
            $CAT-TECHNOLOGY$ attribute table.

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions

        """

        # check technologies that are optional from optional input
        df_entc_re_tag = self.format_model_variable_as_nemomod_table(
            df_elec_trajectories,
            self.modvar_entc_nemomod_renewable_tag_technology,
            "TMP",
            [
                self.field_nemomod_id,
                self.field_nemomod_year,
                self.field_nemomod_region
            ],
            self.field_nemomod_technology,
            regions = regions,
            var_bounds = (0, 1)
        ).get("TMP")


        # filter out groups that are all 0
        df_entc_re_tag = sf.filter_data_frame_by_group(
            df_entc_re_tag,  
            [
                self.field_nemomod_region,
                self.field_nemomod_technology
            ],
            self.field_nemomod_value
        )

        df_entc_re_tag = self.add_multifields_from_key_values(
            df_entc_re_tag,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            override_time_period_transformation = True,
            regions = regions
        )

        dict_return = {self.model_attributes.table_nemomod_re_tag_technology: df_entc_re_tag}

        return dict_return



    def format_nemomod_table_reserve_margin(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the ReserveMargin input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # ReserveMargin
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_reserve_margin,
                self.model_attributes.table_nemomod_reserve_margin,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                regions = regions,
                var_bounds = (0, np.inf)
            )
        )
        dict_return[self.model_attributes.table_nemomod_reserve_margin].drop([self.field_nemomod_technology], axis = 1, inplace = True)

        return dict_return



    def format_nemomod_table_reserve_margin_tag_fuel(self,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the ReserveMargin input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """
        # build data frame
        df_out = pd.DataFrame({
            self.field_nemomod_fuel: [self.cat_enfu_elec],
            self.field_nemomod_value: [1]
        })

        # add dimensions
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_fuel,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        dict_return = {self.model_attributes.table_nemomod_reserve_margin_tag_fuel: df_out}

        return dict_return



    def format_nemomod_table_reserve_margin_tag_technology(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the ReserveMarginTagTechnology input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # ReserveMarginTagTechnology
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_reserve_margin_tag_technology,
                self.model_attributes.table_nemomod_reserve_margin_tag_technology,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                regions = regions,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    def format_nemomod_table_residual_capacity(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the ResidualCapacity input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # get some scalars
        scalar_residual_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_residual_capacity, "power")
        # ResidualCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_residual_capacity,
                self.model_attributes.table_nemomod_residual_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                regions = regions,
                scalar_to_nemomod_units = scalar_residual_capacity,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    def format_nemomod_table_residual_storage_capacity(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the ResidualStorageCapacity input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # get some scalars
        scalar_cost_capital_storage = self.get_nemomod_energy_scalar(self.modvar_enst_nemomod_residual_capacity)
        # ResidualCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_residual_capacity,
                self.model_attributes.table_nemomod_residual_storage_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                regions = regions,
                scalar_to_nemomod_units = scalar_cost_capital_storage,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    def estimate_production_share_from_activity_limits(self,
        df_elec_trajectories: pd.DataFrame,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None
    ) -> np.ndarray:
        """
        Estimate production share of techs specified with a 
            TotalTechnologyAnnualActivityLowerLimit. Use this function to avoid 
            conflicting constraints between 
            MinShareProduction/ReMinProductionTarget and 
            TotalTechnologyAnnualActivityLowerLimit/
            TotalTechnologyAnnualActivityUpperLimit by scaling 
            MinShareProduction. 

        NOTE: Only should be used with integration technologies, not those that 
            are the results of adjustments to MSP (see 
            ElectricEnergy.get_entc_maxprod_increase_adjusted_msp). 

        Returns np.ndarray (vector) of TotalTechnologyAnnualActivityLowerLimit
            as a fraction of non-fuel production energy demands. Note that this 
            fraction is >= than the true fraction, since demands for electricity 
            increase with fuel production; therefore, MinShareProductions may
            be decreased slighly more than is required and will decrease more
            as fuel production demands for electricity increase.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        
        Model Notes
        -----------
        To model transmission losses in the absence of a network, the 
            SpecifiedAnnualDemand for fuel_electricity is inflated by 
            *= 1/(1 - loss). Upon extraction, demands are reduced by 
            *= (1 - loss) in the 
            retrieve_nemomod_tables_fuel_production_demand_and_trade()
            method.
        """

        ##  GET PRODUCTION DEMAND FROM INTEGRATED MODEL

        # calculate total grid demand for electricity
        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(
                df_elec_trajectories, 
                target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
            ) 
            if (tuple_enfu_production_and_demands is None) 
            else tuple_enfu_production_and_demands
        )
        arr_enfu_demands, arr_enfu_demands_distribution, arr_enfu_export, arr_enfu_imports, arr_enfu_production = tuple_enfu_production_and_demands
        
        # updated 20230211 - NemoMod now solves for imports due to endogeneity of certain fuels. Demand is passed as production + imports, and import fractions are specified in MinShareProduction
        arr_enfu_production += arr_enfu_imports 

        # get transmission loss and calculate final demand
        #   NOTE: transmission loss in ENTC is modeled as an increase in input activity ratio *= (1/(1 - loss))
        arr_transmission_loss = self.model_attributes.extract_model_variable(#
            df_elec_trajectories, 
            self.modvar_enfu_transmission_loss_frac_electricity, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = False, 
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_enfu_production[:, self.ind_enfu_elec] = np.nan_to_num(
            arr_enfu_production[:, self.ind_enfu_elec]/(1 - arr_transmission_loss[:, self.ind_enfu_elec]), 
            0.0, 
            posinf = 0.0,
        )


        ##  RETRIEVE TotalTechnologyAnnualActivityLimitLower AND RETURN FRACTION OUT OF PRODUCTION

        #    NOTE: This fraction is >= than the true fraction, since demands for electricity increase with fuel production
        table_name = self.model_attributes.table_nemomod_total_technology_annual_activity_lower_limit
        df_tech_lower_limit = self.get_total_technology_activity_lower_limit_no_msp_adjustment(df_elec_trajectories).get(table_name)
        vector_reference_time_period = list(df_elec_trajectories[self.model_attributes.dim_time_period])

        vec_entc_prod_lower_limit = np.array(
            self.retrieve_and_pivot_nemomod_table(
                df_tech_lower_limit,
                self.modvar_entc_ef_scalar_co2, # arbitrary variable works
                table_name,
                vector_reference_time_period
            ).sum(axis = 1)
        )

        # get technology lower limit total as a fraction of estimated demand for electricity
        vec_fraction_tech = np.nan_to_num(vec_entc_prod_lower_limit/arr_enfu_production[:, self.ind_enfu_elec], 0.0, posinf = 0.0)
        
        return vec_fraction_tech
   


    def format_nemomod_table_specified_annual_demand(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_time_period: Union[AttributeTable, None] = None,
        regions: Union[List[str], None] = None,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None
    ) -> pd.DataFrame:
        """
        Format the SpecifiedAnnualDemand input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used for fuels
        - attribute_time_period: AttributeTable mapping 
            ModelAttributes.dim_time_period to year. If None, use 
            ModelAttributes default.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        
        Model Notes
        -----------
        To model transmission losses in the absence of a network, the 
            SpecifiedAnnualDemand for fuel_electricity is inflated by 
            *= 1/(1 - loss). Upon extraction, demands are reduced by 
            *= (1 - loss) in the 
            retrieve_nemomod_tables_fuel_production_demand_and_trade()
            method.
        """

        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
       

        ##  GET PRODUCTION DEMAND FROM INTEGRATED MODEL

        # calculate total grid demand for electricity
        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(
                df_elec_trajectories, 
                target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
            ) 
            if (tuple_enfu_production_and_demands is None) 
            else tuple_enfu_production_and_demands
        )
        arr_enfu_demands, arr_enfu_demands_distribution, arr_enfu_export, arr_enfu_imports, arr_enfu_production = tuple_enfu_production_and_demands
        
        # updated 20230211 - NemoMod now solves for imports due to endogeneity of certain fuels. Demand is passed as production + imports, and import fractions are specified in MinShareProduction
        arr_enfu_production += arr_enfu_imports 

        # get transmission loss and calculate final demand
        #   NOTE: transmission loss in ENTC is modeled as an increase in input activity ratio *= (1/(1 - loss))
        arr_transmission_loss = self.model_attributes.extract_model_variable(#
            df_elec_trajectories, 
            self.modvar_enfu_transmission_loss_frac_electricity, 
            expand_to_all_cats = True, 
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_enfu_production[:, self.ind_enfu_elec] = np.nan_to_num(
            arr_enfu_production[:, self.ind_enfu_elec]/(1 - arr_transmission_loss[:, self.ind_enfu_elec]), 
            0.0, 
            posinf = 0.0,
        )

        # drop fields that are associated with dummy techs + those with all zeros
        fields_drop = list(
            self.get_dummy_fuel_techs(
                attribute_fuel = attribute_fuel, 
                drop_activity_ratio_fuels = True
            ).keys()
        )
        df_enfu_production = pd.DataFrame(arr_enfu_production, columns = attribute_fuel.key_values)
        fields_all_zero = [x for x in df_enfu_production.columns if (len(set(df_enfu_production[x])) == 1) and max(list(df_enfu_production[x])) == 0]
        fields_drop = sorted(list(set(fields_drop + fields_all_zero)))
        df_enfu_production.drop(fields_drop, axis = 1, inplace = True)


        ##  FORMAT AS DATA FRAME

        # initialize and add year
        df_enfu_production = self.model_attributes.exchange_year_time_period(
            df_enfu_production,
            self.field_nemomod_year,
            df_elec_trajectories[self.model_attributes.dim_time_period],
            attribute_time_period = attribute_time_period,
            direction = self.direction_exchange_year_time_period
        )

        df_enfu_production = df_enfu_production.melt(
            id_vars = [self.field_nemomod_year]
        ).rename(
            columns = {"variable": self.field_nemomod_fuel, "value": self.field_nemomod_value}
        )

        # add additional required fields, then sort
        df_enfu_production = self.add_multifields_from_key_values(
            df_enfu_production,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        return {self.model_attributes.table_nemomod_specified_annual_demand: df_enfu_production}



    def format_nemomod_table_specified_demand_profile(self,
        df_reference_demand_profile: pd.DataFrame,
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_region: AttributeTable = None,
        attribute_time_slice: Union[AttributeTable, None] = None,
        fuels_to_specify: Union[List[str], None] = None,
        regions: Union[List[str], None] = None
    ) -> pd.DataFrame:
        """
        Format the SpecifiedDemandProfile input table for NemoMod based on 
            SISEPUEDE configuration parameters, input variables, integrated 
            model outputs, and reference tables.

        Function Arguments
        ------------------
        - df_reference_demand_profile: data frame of reference demand profile 
            for the region

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used for fuels
        - attribute_time_slice: AttributeTable used to define time slice 
            weights, which are used to allocate demand in the absence of other
            reference data.
        - attribute_region: AttributeTable for regions. If None, defaults to 
            ModelAttributes attribute table.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - fuels_to_specify: list of fuels to specify demand profiles for (can be
            passed from SpecifiedAnnualDemand input table). If None, defaults to
            all fuels.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        ##  INITIAlIZATION

        attribute_fuel = (
            self.get_attribute_enfu() 
            if (attribute_fuel is None) 
            else attribute_fuel
        )

        attribute_region = (
            self.get_attribute_region() 
            if (attribute_region is None) 
            else attribute_region
        )

        attribute_time_slice = (
            self.model_attributes.get_other_attribute_table("time_slice") 
            if (attribute_time_slice is None) 
            else attribute_time_slice
        )
        
        # attribute derivatives
        fuels_to_specify = (
            attribute_fuel.key_values 
            if (fuels_to_specify is None) 
            else [x for x in fuels_to_specify if x in attribute_fuel.key_values]
        )

        fuels_to_specify = (
            attribute_fuel.key_values 
            if (len(fuels_to_specify) == 0) 
            else fuels_to_specify
        )

        regions = self.model_attributes.get_region_list_filtered(regions, attribute_region = attribute_region)
        regions_keep = set(attribute_region.key_values) & set(regions)
        regions_keep = (
            (regions_keep & set(df_reference_demand_profile[self.field_nemomod_region])) 
            if (self.field_nemomod_region in df_reference_demand_profile.columns) 
            else regions_keep
        )

        if len(regions_keep) == 0:
            raise ValueError(f"No valid regions found for format_nemomod_table_specified_demand_profile() in df_reference_demand_profile")


        ##  BUILD COMPONENTS FROM REFERENCE

        # check for required fields
        fields_required = [self.field_nemomod_time_slice, self.field_nemomod_value]
        df_out = df_reference_demand_profile.copy()
        df_out = df_out[df_out[self.field_nemomod_region].isin(regions_keep)] if (self.field_nemomod_region in df_out.columns) else df_out
        
        sf.check_fields(
            df_out, 
            fields_required, 
            msg_prepend = f"Error in format_nemomod_table_specified_demand_profile: required fields "
        )
        #n = len(df_out[self.field_nemomod_region].unique())

        # specify fuels - assume electric if missing
        if self.field_nemomod_fuel not in df_out.columns:
            df_out[self.field_nemomod_fuel] = self.cat_enfu_elec

        # setup to the same structure as df_out to allow concatendation
        fields_required_0 = [
            self.field_nemomod_fuel,
            self.field_nemomod_region,
            self.field_nemomod_time_slice,
            self.field_nemomod_year
        ]
        df_out = self.add_multifields_from_key_values(
            df_out, 
            fields_required_0,
            regions = regions
        )


        ##  ADD DEFAULTS FOR OTHER FUELS

        # direct product of fuels to specify and time slices
        df_fuels_to_specify = pd.DataFrame({self.field_nemomod_fuel: fuels_to_specify})
        df_time_slices = attribute_time_slice.table[["time_slice", "weight"]].rename(columns = {
            attribute_time_slice.key: self.field_nemomod_time_slice,
            "weight": self.field_nemomod_value
        })
        df_append = sf.explode_merge(df_fuels_to_specify, df_time_slices)
        
        # setup to the same structure as df_out to allow concatendation, then concategnate
        df_append = self.add_multifields_from_key_values(
            df_append, 
            fields_required_0,
            regions = regions
        )
        df_append = df_append[
            ~(
                df_append[self.field_nemomod_fuel].isin(df_out[self.field_nemomod_fuel]) & 
                df_append[self.field_nemomod_region].isin(df_out[self.field_nemomod_region])
            )
        ]
        df_out = pd.concat(
            [df_out, df_append[df_out.columns]], 
            axis = 0
        ).reset_index(drop = True)


        # CLEAN, SORT AND ADD IDS

        df_out = self.add_multifields_from_key_values(
            df_out, 
            [
                self.field_nemomod_id, 
                self.field_nemomod_fuel,
                self.field_nemomod_region,
                self.field_nemomod_time_slice,
                self.field_nemomod_year
            ],
            override_time_period_transformation = True,
            regions = regions
        )
        dict_return = {self.model_attributes.table_nemomod_specified_demand_profile: df_out}

        return dict_return



    ##  format StorageMaxChargeRate, StorageMaxDishargeRate, and StorageStartLevel for NemoMod
    def format_nemomod_table_storage_attributes(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_storage: AttributeTable = None,
        field_attribute_min_charge: str = "minimum_charge_fraction",
        field_tmp: str = "TMPNEW",
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the StorageMaxChargeRate, StorageMaxDishargeRate, and 
            StorageStartLevel input tables for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_storage: AttributeTable used to ensure that start level 
            meets or exceeds the minimum allowable storage charge. If None, 
            use ModelAttributes default.
        - field_attribute_min_charge: field in attribute_storage.table used to 
            identify minimum required storage for each type of storage. If None, 
            use ModelAttributes default.
        - field_tmp: temporary field used in data frame
        - regions: regions to specify. If None, defaults to configuration 
            regions

        """

        # set some defaults
        attribute_storage = (
            self.model_attributes.get_attribute_table(self.subsec_name_enst) 
            if (attribute_storage is None) 
            else attribute_storage
        )

        dict_strg_to_min_charge = attribute_storage.field_maps.get(
            f"{attribute_storage.key}_to_{field_attribute_min_charge}"
        )

        dict_return = {}

        # StorageStartLevel
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_storage_start_level,
                self.model_attributes.table_nemomod_storage_level_start,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                regions = regions,
                var_bounds = (0, 1)
            )
        )

        # some cleaning of the data frame
        df_tmp = dict_return[self.model_attributes.table_nemomod_storage_level_start]
        df_tmp = (
            df_tmp[
                df_tmp[self.field_nemomod_year] == min(df_tmp[self.field_nemomod_year])
            ]
            .drop([self.field_nemomod_year], axis = 1,)
            .reset_index(drop = True)
        )

        # add bounds and drop the temporary field
        df_tmp[field_tmp] = df_tmp[self.field_nemomod_storage].replace(dict_strg_to_min_charge)
        bounds = list(zip(list(df_tmp[field_tmp]), list(np.ones(len(df_tmp)))))

        df_tmp[self.field_nemomod_value] = sf.vec_bounds(
            np.array(df_tmp[self.field_nemomod_value]),
            bounds
        )
        df_tmp.drop([field_tmp], axis = 1, inplace = True)

        # update
        dict_return.update({self.model_attributes.table_nemomod_storage_level_start: df_tmp})

        return dict_return



    ##  format StorageStartLevel for NemoMod
    def format_nemomod_table_storage_start_level(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format the StorageStartLevel input table for NemoMod based on SISEPUEDE 
            configuration parameters, input variables, integrated model outputs, 
            and reference tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None



    def format_nemomod_table_technology_from_and_to_storage(self,
        attribute_storage: AttributeTable = None,
        attribute_technology: AttributeTable = None,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the TechnologyFromStorage and TechnologyToStorage input table for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        Keyword Arguments
        -----------------
        - attribute_storage: AttributeTable for storage, used to identify 
            storage characteristics. If None, use ModelAttributes default.
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """


        # set some defaults
        attribute_storage = (
            self.model_attributes.get_attribute_table(self.subsec_name_enst) 
            if (attribute_storage is None) 
            else attribute_storage
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        pycat_strg = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enst, 
            "pycategory_primary_element",
        )


        # cat storage dictionary
        df_storage_techs_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_strg,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )

        df_storage_techs_to_storage = pd.DataFrame(
            df_storage_techs_to_storage.items(),
            columns = [self.field_nemomod_technology, self.field_nemomod_storage]
        )
        
        # build tech from storage
        df_tech_from_storage = df_storage_techs_to_storage.copy()
        df_tech_from_storage[self.field_nemomod_mode] = self.cat_enmo_gnrt
        df_tech_from_storage[self.field_nemomod_value] = 1.0
        df_tech_from_storage = self.add_multifields_from_key_values(
            df_tech_from_storage, 
            [
                self.field_nemomod_id, 
                self.field_nemomod_region
            ],
            regions = regions
        )

        # build tech to storage
        df_tech_to_storage = []
        df_tech_to_storage = df_storage_techs_to_storage.copy()
        df_tech_to_storage[self.field_nemomod_mode] = self.cat_enmo_stor
        df_tech_to_storage[self.field_nemomod_value] = 1.0
        df_tech_to_storage = self.add_multifields_from_key_values(
            df_tech_to_storage, 
            [
                self.field_nemomod_id,
                self.field_nemomod_region
            ],
            regions = regions
        )

        dict_return = {
            self.model_attributes.table_nemomod_technology_from_storage: df_tech_from_storage,
            self.model_attributes.table_nemomod_technology_to_storage: df_tech_to_storage
        }

        return dict_return



    ##  format LTsGroup, TSGROUP1, TSGROUP2, and YearSplit for NemoMod
    def format_nemomod_table_tsgroup_tables(self,
        attribute_time_slice: AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the LTsGroup, TIMESLICE, TSGROUP1, TSGROUP2, and YearSplit input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.

            Keyword Arguments
            -----------------
            - attribute_time_slice: AttributeTable for time slice, used to identify the maximum discharge rate. If None, use ModelAttributes default.
        """
        # retrieve the attribute and check fields
        fields_req = [
            "time_slice",
            "description",
            self.field_nemomod_tg1,
            self.field_nemomod_tg2,
            self.field_nemomod_lorder,
            "weight"
        ]

        attribute_time_slice = (
            self.model_attributes.get_other_attribute_table("time_slice")
            if (attribute_time_slice is None) 
            else attribute_time_slice
        )

        sf.check_fields(
            attribute_time_slice.table, 
            fields_req, 
            msg_prepend = "Missing fields in table 'LTsGroup': ",
        )


        ##  FORMAT THE TIMESLICE ATTRIBUTE TABLE

        df_time_slice = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_time_slice = df_time_slice[["time_slice", "description"]].rename(
            columns = {
                "time_slice": self.field_nemomod_value,
                "description": self.field_nemomod_description
            }
        )


        ##  FORMAT THE LTsGroup TABLE

        df_ltsgroup = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_ltsgroup[self.field_nemomod_id] = range(1, len(df_ltsgroup) + 1)
        df_ltsgroup.rename(columns = {"time_slice": self.field_nemomod_time_slice}, inplace = True)
        fields_ext = [
            self.field_nemomod_id,
            self.field_nemomod_time_slice,
            self.field_nemomod_tg1,
            self.field_nemomod_tg2,
            self.field_nemomod_lorder
        ]
        df_ltsgroup = df_ltsgroup[fields_ext]


        ##  FORMAT THE YearSplit TABLE

        df_year_split = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_year_split = df_year_split[["time_slice", "weight"]].rename(
            columns = {
                "time_slice": self.field_nemomod_time_slice,
                "weight": self.field_nemomod_value
            }
        )
        df_year_split = pd.merge(df_year_split, df_ltsgroup[[self.field_nemomod_time_slice, self.field_nemomod_id]], how = "left")
        df_year_split = self.add_multifields_from_key_values(
            df_year_split, 
            [
                self.field_nemomod_id, 
                self.field_nemomod_year
            ]
        )


        ##  FORMAT TSGROUP1 and TSGROUP2

        # get data used to identify order
        df_tgs = pd.merge(
            df_ltsgroup[[self.field_nemomod_id, self.field_nemomod_time_slice, self.field_nemomod_tg1, self.field_nemomod_tg2]],
            df_year_split[df_year_split[self.field_nemomod_year] == min(df_year_split[self.field_nemomod_year])][[self.field_nemomod_time_slice, self.field_nemomod_value]],
            how = "left"
        ).sort_values(by = [self.field_nemomod_id])
        # some dictionaries
        dict_field_to_attribute = {self.field_nemomod_tg1: "ts_group_1", self.field_nemomod_tg2: "ts_group_2"}
        dict_tg = {}

        # loop over fields
        for fld in [self.field_nemomod_tg1, self.field_nemomod_tg2]:

            # prepare from LTsGroup table
            dict_agg = {
                self.field_nemomod_id: "first",
                fld: "first",
            }

            df_tgs_out = (
                df_tgs[[
                    self.field_nemomod_id, fld
                ]]
                .groupby([fld])
                .agg(dict_agg)
                .sort_values(by = [self.field_nemomod_id])
                .reset_index(drop = True)
            )

            # get attribute for time slice group
            table_name = dict_field_to_attribute.get(fld)
            attr_cur = self.model_attributes.get_other_attribute_table(table_name).table.copy()

            attr_cur.rename(
                columns = {
                    table_name: self.field_nemomod_name,
                    "description": self.field_nemomod_description,
                    "multiplier": self.field_nemomod_multiplier
                }, inplace = True
            )

            df_tgs_out[self.field_nemomod_order] = range(1, len(df_tgs_out) + 1)
            df_tgs_out = df_tgs_out.drop([self.field_nemomod_id], axis = 1).rename(
                columns = {
                    fld: self.field_nemomod_name
                }
            )
            df_tgs_out = pd.merge(df_tgs_out, attr_cur).sort_values(by = [self.field_nemomod_order]).reset_index(drop = True)

            # order for output
            df_tgs_out = df_tgs_out[[
                self.field_nemomod_name,
                self.field_nemomod_description,
                self.field_nemomod_order,
                self.field_nemomod_multiplier,
            ]]

            dict_tg.update({fld: df_tgs_out})

        dict_return = {
            self.model_attributes.table_nemomod_time_slice_group_assignment: df_ltsgroup,
            self.model_attributes.table_nemomod_time_slice: df_time_slice,
            self.model_attributes.table_nemomod_ts_group_1: dict_tg[self.field_nemomod_tg1],
            self.model_attributes.table_nemomod_ts_group_2: dict_tg[self.field_nemomod_tg2],
            self.model_attributes.table_nemomod_year_split: df_year_split
        }

        return dict_return



    def format_nemomod_table_total_capacity_tables(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the 

            * TotalAnnualMaxCapacity
            * TotalAnnualMaxCapacityInvestment
            * TotalAnnualMinCapacity  
            * TotalAnnualMinCapacityInvestment 
            
            input tables for NemoMod based on SISEPUEDE configuration 
            parameters, input variables, integrated model outputs, and reference 
            tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}

        # get some scalars
        scalar_total_annual_max_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity, "power")
        scalar_total_annual_max_capacity_investment = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity_investment, "power")
        scalar_total_annual_min_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity, "power")
        scalar_total_annual_min_capacity_investment = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity_investment, "power")

        # TotalAnnualMaxCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_max_capacity,
                self.model_attributes.table_nemomod_total_annual_max_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity
            )
        )

        # TotalAnnualMaxCapacityInvestment
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_max_capacity_investment,
                self.model_attributes.table_nemomod_total_annual_max_capacity_investment,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_investment
            )
        )

        # TotalAnnualMinCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_min_capacity,
                self.model_attributes.table_nemomod_total_annual_min_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity
            )
        )
        
        # TotalAnnualMinCapacityInvestment
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_min_capacity_investment,
                self.model_attributes.table_nemomod_total_annual_min_capacity_investment,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_investment
            )
        )


        ##  CHECK MAX/MIN RELATIONSHIP--SWAP VALUES IF NEEDED

        # check tables - capacity
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return.get(self.model_attributes.table_nemomod_total_annual_max_capacity),
            dict_return.get(self.model_attributes.table_nemomod_total_annual_min_capacity),
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity: dfs_verify[1]
                }
            )

        # check tables - capacity investment
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return.get(self.model_attributes.table_nemomod_total_annual_max_capacity_investment),
            dict_return.get(self.model_attributes.table_nemomod_total_annual_min_capacity_investment),
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_investment: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_investment: dfs_verify[1]
                }
            )

        return dict_return



    ##  format TotalAnnualMaxCapacityStorage, TotalAnnualMaxCapacityInvestmentStorage, TotalAnnualMinCapacityStorage, TotalAnnualMinCapacityInvestmentStorage for NemoMod
    def format_nemomod_table_total_capacity_storage_tables(self,
        df_elec_trajectories: pd.DataFrame,
        regions: Union[List[str], None] = None,
    ) -> pd.DataFrame:
        """
        Format the 
            TotalAnnualMaxCapacityStorage, 
            TotalAnnualMaxCapacityInvestmentStorage, 
            TotalAnnualMinCapacityStorage, and 
            TotalAnnualMinCapacityInvestmentStorage 
            
            input tables for NemoMod based on SISEPUEDE configuration 
            parameters, input variables, integrated model outputs, and reference 
            tables.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - regions: regions to specify. If None, defaults to configuration 
            regions
        """

        dict_return = {}
        # get some scalars
        scalar_total_annual_max_capacity_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_total_annual_max_capacity_storage, "power")
        scalar_total_annual_max_capacity_investment_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_total_annual_max_capacity_investment_storage, "power")
        scalar_total_annual_min_capacity_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_total_annual_min_capacity_storage, "power")
        scalar_total_annual_min_capacity_investment_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_total_annual_min_capacity_investment_storage, "power")

        # TotalAnnualMaxCapacityStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_max_capacity_storage,
                self.model_attributes.table_nemomod_total_annual_max_capacity_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_storage
            )
        )
        # TotalAnnualMaxCapacityInvestmentStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_max_capacity_investment_storage,
                self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_investment_storage
            )
        )
        # TotalAnnualMinCapacityStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_min_capacity_storage,
                self.model_attributes.table_nemomod_total_annual_min_capacity_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_storage
            )
        )
        # TotalAnnualMinCapacityInvestmentStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_min_capacity_investment_storage,
                self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                drop_flag = self.drop_flag_tech_capacities,
                regions = regions,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_investment_storage
            )
        )


        ##  CHECK MAX/MIN RELATIONSHIP--SWAP VALUES IF NEEDED

        # check tables - capacity
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return.get(self.model_attributes.table_nemomod_total_annual_max_capacity_storage),
            dict_return.get(self.model_attributes.table_nemomod_total_annual_min_capacity_storage),
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_storage: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_storage: dfs_verify[1]
                }
            )

        # check tables - capacity investment
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return.get(self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage),
            dict_return.get(self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage),
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage: dfs_verify[1]
                }
            )

        return dict_return



    def format_nemomod_table_total_technology_activity_lower_limit(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: AttributeTable = None,
        attribute_technology: AttributeTable = None,
        drop_flag: Union[float, int] = None,
        regions: Union[List[str], None] = None, 
        return_type: str = "NemoMod",
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the TotalTechnologyAnnualActivityLowerLimit input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        In SISEPUEDE, this table is used in conjunction with 
            TotalTechnologyAnnualActivityUpperLimit to pass biogas and waste
            incineration production from collection in Circular Economy and 
            AFOLU.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuels
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - drop_flag: optional specification of a drop flag used to indicate 
            rows/variables for which the MSP Max Production constraint is not 
            applicable (see 
            ElectricEnergy.get_entc_maxprod_increase_adjusted_msp for more 
            info). Defaults to self.drop_flag_tech_capacities if None.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: type of return. Acceptable values are "NemoMod" and 
            "CapacityCheck". Invalid entries default to "NemoMod"
            * NemoMod (default): return the 
                TotalTechnologyAnnualActivityLowerLimit input table for the 
                NemoMod database
            * CapacityCheck: return a table of specified minimum capacities
                 associated with the technology.
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        - **kwargs: passed to self.get_entc_maxprod_increase_adjusted_msp
        """

        # some initialization
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        drop_flag = self.drop_flag_tech_capacities if (drop_flag is None) else drop_flag
        table_name = self.model_attributes.table_nemomod_total_technology_annual_activity_lower_limit
    

        ##  GET BASELIINE TotalTechnologyActivityLowerLimit + POTENTIAL ADD ON FROM MSP ADJUSTMENT

        # get baseline TTALL
        dict_out = self.get_total_technology_activity_lower_limit_no_msp_adjustment(
            df_elec_trajectories,
            attribute_technology = attribute_technology, 
            regions = regions,
            return_type = "NemoMod",
        )

        # update with MSP
        dict_out = self.update_ttal_dictionary_with_limit_from_msp(
            df_elec_trajectories,
            dict_out,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            drop_flag = drop_flag,
            key_ttal = table_name,
            regions = regions, 
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
            **kwargs
        )

        return dict_out
    


    def format_nemomod_table_total_technology_activity_upper_limit(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_fuel: AttributeTable = None,
        attribute_technology: AttributeTable = None,
        drop_flag: Union[float, int] = None,
        regions: Union[List[str], None] = None, 
        return_type: str = "NemoMod",
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the TotalTechnologyAnnualActivityUpperLimit input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        In SISEPUEDE, this table is used in conjunction with 
            TotalTechnologyAnnualActivityLowerLimit to pass biogas and waste
            incineration production from collection in Circular Economy and 
            AFOLU.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuels
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - drop_flag: optional specification of a drop flag used to indicate 
            rows/variables for which the MSP Max Production constraint is not 
            applicable (see 
            ElectricEnergy.get_entc_maxprod_increase_adjusted_msp for more 
            info). Defaults to self.drop_flag_tech_capacities if None.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: type of return. Acceptable values are "NemoMod" and 
            "CapacityCheck". Invalid entries default to "NemoMod"
            * NemoMod (default): return the 
                TotalTechnologyAnnualActivityLowerLimit input table for the 
                NemoMod database
            * CapacityCheck: return a table of specified minimum capacities
                 associated with the technology.
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        - **kwargs: passed to self.get_entc_maxprod_increase_adjusted_msp
        """

        # some initialization
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        drop_flag = self.drop_flag_tech_capacities if (drop_flag is None) else drop_flag
        table_name = self.model_attributes.table_nemomod_total_technology_annual_activity_upper_limit
    

        ##  GET BASELIINE TotalTechnologyActivityLowerLimit + POTENTIAL ADD ON FROM MSP ADJUSTMENT

        # get baseline TTALL
        dict_out = self.get_total_technology_activity_upper_limit_no_msp_adjustment(
            df_elec_trajectories,
            attribute_technology = attribute_technology, 
            regions = regions,
            return_type = "NemoMod",
        )

        # update with MSP
        dict_out = self.update_ttal_dictionary_with_limit_from_msp(
            df_elec_trajectories,
            dict_out,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            drop_flag = drop_flag,
            key_ttal = table_name,
            regions = regions, 
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
            **kwargs
        )

        return dict_out



    def get_total_technology_activity_lower_limit_no_msp_adjustment(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: AttributeTable = None,
        regions: Union[List[str], None] = None, 
        return_type: str = "NemoMod",
    ) -> Dict[str, pd.DataFrame]:
        """
        Construct the TotalTechnologyAnnualActivityLowerLimit input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables WITHOUT
            adjusting for the implementation of the Max Production Inrease from
            MinShareProduction. 

        NOTE: this table is called elsewhere in ElectricEnergy 
        
        In SISEPUEDE, this table is used in conjunction with 
            TotalTechnologyAnnualActivityUpperLimit to pass biogas and waste
            incineration production from collection in Circular Economy and 
            AFOLU.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: type of return. Acceptable values are "NemoMod" and 
            "CapacityCheck". Invalid entries default to "NemoMod"
            * NemoMod (default): return the 
                TotalTechnologyAnnualActivityLowerLimit input table for the 
                NemoMod database
            * CapacityCheck: return a table of specified minimum capacities
                 associated with the technology.
        """

        # check input of return_type
        try:
            sf.check_set_values([return_type], ["NemoMod", "CapacityCheck"])
        except:
            # LOG HERE
            return_type = "NemoMod"

        # some attribute initializations
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        # get some categories and keys
        cat_entc_pp_biogas = self.get_entc_cat_for_integration("bgas")
        cat_entc_pp_waste = self.get_entc_cat_for_integration("wste")
        ind_entc_pp_biogas = attribute_technology.get_key_value_index(cat_entc_pp_biogas)
        ind_entc_pp_waste = attribute_technology.get_key_value_index(cat_entc_pp_waste)

        # get some scalars to use if returning a capacity constraint dataframe
        if return_type == "CapacityCheck":
            units_energy_config = self.model_attributes.configuration.get("energy_units")
            units_power_config = self.model_attributes.configuration.get("power_units")
            units_energy_power_equivalent = self.model_attributes.get_energy_power_swap(units_power_config)
            scalar_energy_to_power_cur = self.model_attributes.get_energy_equivalent(
                units_energy_config, 
                units_energy_power_equivalent
            )


        ##  GET SUPPLY TO USE (MIN) AND TECH EFFICIENCIES

        # get efficiency factors--total production should match up to min supply utilization * efficiency
        arr_entc_efficiencies = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.modvar_entc_efficiency_factor_technology,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        # get biogas supply available
        vec_enfu_total_energy_supply_biogas, vec_enfu_min_energy_to_elec_biogas = self.get_biogas_components(
            df_elec_trajectories
        )
        vec_enfu_min_energy_to_elec_biogas *= arr_entc_efficiencies[:, ind_entc_pp_biogas]
        # get waste supply available
        vec_enfu_total_energy_supply_waste, vec_enfu_min_energy_to_elec_waste, dict_efs = self.get_waste_energy_components(
            df_elec_trajectories,
            return_emission_factors = True
        )
        vec_enfu_min_energy_to_elec_waste *= arr_entc_efficiencies[:, ind_entc_pp_waste]


        ##  BUILD OUTPUT DATAFRAME - ALLOW FOR

        # biogas component
        df_biogas = pd.DataFrame({
            self.field_nemomod_technology: cat_entc_pp_biogas,
            self.field_nemomod_value: vec_enfu_min_energy_to_elec_biogas,
            self.field_nemomod_year: list(df_elec_trajectories[self.model_attributes.dim_time_period])
        })
        # waste component
        df_waste = pd.DataFrame({
            self.field_nemomod_technology: cat_entc_pp_waste,
            self.field_nemomod_value: vec_enfu_min_energy_to_elec_waste,
            self.field_nemomod_year: list(df_elec_trajectories[self.model_attributes.dim_time_period])
        })
        # concatenate into output data frame
        df_out = pd.concat([df_biogas, df_waste], axis = 0)
        df_out = self.model_attributes.exchange_year_time_period(
            df_out,
            self.field_nemomod_year,
            df_out[self.field_nemomod_year],
            direction = self.direction_exchange_year_time_period
        )
        # add key values
        df_out = self.add_multifields_from_key_values(df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        # scale to power units if doing capacity check
        if return_type == "CapacityCheck":
            df_out[self.field_nemomod_value] = np.array(df_out[self.field_nemomod_value])*scalar_energy_to_power_cur

        # setup output dictionary and return
        dict_return = {
            self.model_attributes.table_nemomod_total_technology_annual_activity_lower_limit: df_out
        }

        return dict_return
    


    def get_total_technology_activity_upper_limit_no_msp_adjustment(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: AttributeTable = None,
        regions: Union[List[str], None] = None, 
        return_type: str = "NemoMod"
    ) -> pd.DataFrame:
        """
        Construct the TotalTechnologyAnnualActivityUpperLimit input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables WITHOUT
            adjusting for the implementation of the Max Production Inrease from
            MinShareProduction. 

        NOTE: this table is called elsewhere in ElectricEnergy 

        In SISEPUEDE, this table is used in conjunction with 
            TotalTechnologyAnnualActivityUpperLimit to pass biogas and waste
            incineration production from collection in Circular Economy and 
            AFOLU.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories

        Keyword Arguments
        -----------------
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: type of return. Acceptable values are "NemoMod" and 
            "CapacityCheck". Invalid entries default to "NemoMod"
            * NemoMod (default): return the 
                TotalTechnologyAnnualActivityUpperLimit input table for the 
                NemoMod database
            * CapacityCheck: return a table of specified minimum capacities
                 associated with the technology.
        """

        # check input of return_type
        try:
            sf.check_set_values([return_type], ["NemoMod", "CapacityCheck"])
        except:
            # LOG HERE
            return_type = "NemoMod"

        # some attribute initializations
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )

        # get some categories and keys
        cat_entc_pp_biogas = self.get_entc_cat_for_integration("bgas")
        cat_entc_pp_waste = self.get_entc_cat_for_integration("wste")
        ind_entc_pp_biogas = attribute_technology.get_key_value_index(cat_entc_pp_biogas)
        ind_entc_pp_waste = attribute_technology.get_key_value_index(cat_entc_pp_waste)

        # get some scalars to use if returning a capacity constraint dataframe
        if return_type == "CapacityCheck":
            units_energy_config = self.model_attributes.configuration.get("energy_units")
            units_power_config = self.model_attributes.configuration.get("power_units")
            units_energy_power_equivalent = self.model_attributes.get_energy_power_swap(units_power_config)
            scalar_energy_to_power_cur = self.model_attributes.get_energy_equivalent(units_energy_config, units_energy_power_equivalent)


        ##  GET SUPPLY TO USE (MIN) AND TECH EFFICIENCIES

        # get efficiency factors--total production should match up to min supply utilization * efficiency
        arr_entc_efficiencies = self.model_attributes.extract_model_variable(#
            df_elec_trajectories,
            self.modvar_entc_efficiency_factor_technology,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        # get biogas supply available
        vec_enfu_total_energy_supply_biogas, vec_enfu_min_energy_to_elec_biogas = self.get_biogas_components(
            df_elec_trajectories
        )
        vec_enfu_min_energy_to_elec_biogas *= arr_entc_efficiencies[:, ind_entc_pp_biogas]

        # get waste supply available
        (
            vec_enfu_total_energy_supply_waste, 
            vec_enfu_min_energy_to_elec_waste, 
            dict_efs
        ) = self.get_waste_energy_components(
            df_elec_trajectories,
            return_emission_factors = True,
        )

        vec_enfu_min_energy_to_elec_waste *= arr_entc_efficiencies[:, ind_entc_pp_waste]


        ##  BUILD OUTPUT DATAFRAME - ALLOW FOR

        # biogas component
        df_biogas = pd.DataFrame({
            self.field_nemomod_technology: cat_entc_pp_biogas,
            self.field_nemomod_value: vec_enfu_min_energy_to_elec_biogas,
            self.field_nemomod_year: list(df_elec_trajectories[self.model_attributes.dim_time_period])
        })

        # waste component
        df_waste = pd.DataFrame({
            self.field_nemomod_technology: cat_entc_pp_waste,
            self.field_nemomod_value: vec_enfu_min_energy_to_elec_waste,
            self.field_nemomod_year: list(df_elec_trajectories[self.model_attributes.dim_time_period])
        })

        # concatenate into output data frame
        df_out = pd.concat([df_biogas, df_waste], axis = 0)
        df_out = self.model_attributes.exchange_year_time_period(
            df_out,
            self.field_nemomod_year,
            df_out[self.field_nemomod_year],
            direction = self.direction_exchange_year_time_period
        )

        # add key values
        df_out = self.add_multifields_from_key_values(df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ],
            regions = regions
        )

        # scale to power units if doing capacity check
        if return_type == "CapacityCheck":
            df_out[self.field_nemomod_value] = np.array(df_out[self.field_nemomod_value])*scalar_energy_to_power_cur

        # setup output dictionary and return
        dict_return = {
            self.model_attributes.table_nemomod_total_technology_annual_activity_upper_limit: df_out
        }

        return dict_return
    


    def update_ttal_dictionary_with_limit_from_msp(self,
        df_elec_trajectories: pd.DataFrame,
        dict_ttal: Union[Dict[str, pd.DataFrame], None],
        attribute_fuel: AttributeTable = None,
        attribute_technology: AttributeTable = None,
        drop_flag: Union[float, int] = None,
        key_ttal: Union[str, None] = None,
        regions: Union[List[str], None] = None, 
        return_type: str = "NemoMod",
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Format the TotalTechnologyAnnualActivityLowerLimit input tables for 
            NemoMod based on SISEPUEDE configuration parameters, input 
            variables, integrated model outputs, and reference tables.

        In SISEPUEDE, this table is used in conjunction with 
            TotalTechnologyAnnualActivityUpperLimit to pass biogas and waste
            incineration production from collection in Circular Economy and 
            AFOLU.


        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of model variable input trajectories
        - dict_ttal: dictionary from TotalTechnologyActivityLowerLimit or
            TotalTechnologyActivityUpperLimit 

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable for fuels
        - attribute_technology: AttributeTable for technology, used to identify 
            whether or not a technology can charge a storage. If None, use 
            ModelAttributes default.
        - drop_flag: optional specification of a drop flag used to indicate 
            rows/variables for which the MSP Max Production constraint is not 
            applicable (see 
            ElectricEnergy.get_entc_maxprod_increase_adjusted_msp for more 
            info). Defaults to self.drop_flag_tech_capacities if None.
        - key_ttal: key in dict_ttal that includes maps to the table to modify.
            If None, calls first key available (will not present any issues if 
            only one key is passed)
        - regions: regions to specify. If None, defaults to configuration 
            regions
        - return_type: type of return. Acceptable values are "NemoMod" and 
            "CapacityCheck". Invalid entries default to "NemoMod"
            * NemoMod (default): return the 
                TotalTechnologyAnnualActivityLowerLimit input table for the 
                NemoMod database
            * CapacityCheck: return a table of specified minimum capacities
                 associated with the technology.
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        - **kwargs: passed to self.get_entc_maxprod_increase_adjusted_msp
        """

        # some initialization
        attribute_fuel = (
            self.model_attributes.get_attribute_table(self.subsec_name_enfu) 
            if (attribute_fuel is None) 
            else attribute_fuel
        )
        attribute_technology = (
            self.model_attributes.get_attribute_table(self.subsec_name_entc) 
            if (attribute_technology is None) 
            else attribute_technology
        )
        drop_flag = self.drop_flag_tech_capacities if (drop_flag is None) else drop_flag
        table_name = list(dict_ttal.keys())[0] if (key_ttal is None) else key_ttal


        # retrieve input MSP and activity limit
        (
            arr_entc_msp, 
            arr_entc_activity_limits, 
            vec_frac_msp_accounted_for_by_growth_limit
        ) = self.get_entc_maxprod_increase_adjusted_msp(
            df_elec_trajectories,
            adjust_free_msps_in_response = True,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
            **kwargs
        )

        # if the MSP is adjusted using TTALL, modify the outputs using the addition
        if arr_entc_activity_limits is not None:

            # set up activity limits to append
            df_entc_activity_limits_append = pd.DataFrame(
                arr_entc_activity_limits,
                columns = attribute_technology.key_values
            )
            df_entc_activity_limits_append[self.model_attributes.dim_time_period] = list(df_elec_trajectories[self.model_attributes.dim_time_period])

            # reformat for NemoMod input
            df_entc_activity_limits_append = self.format_model_variable_as_nemomod_table(
                df_entc_activity_limits_append,
                self.model_attributes.subsec_name_entc,
                "TMP",
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                drop_flag = drop_flag,
                regions = regions,
            ).get("TMP")

            # get the prepend (existing) and reduce the MSP appendage to eliminate potential conflicts
            df_prepend = dict_ttal.get(table_name)
            df_entc_activity_limits_append = df_entc_activity_limits_append[
                ~df_entc_activity_limits_append[self.field_nemomod_technology].isin(list(df_prepend[self.field_nemomod_technology]))
            ].reset_index(drop = True)

            # concatenate and reset IDs 
            df_out = self.add_multifields_from_key_values(
                pd.concat(
                    [df_prepend, df_entc_activity_limits_append], 
                    axis = 0
                ), 
                [
                    self.field_nemomod_id, 
                    self.field_nemomod_region,
                    self.field_nemomod_technology,
                    self.field_nemomod_year
                ],
                override_time_period_transformation = True,
                regions = regions
            )

            dict_ttal.update({table_name: df_out})

        return dict_ttal






    #######################################################################
    ###                                                                 ###
    ###    FUNCTIONS TO FORMAT NEMOMOD OUTPUT FROM SQL FOR SISEPUEDE    ###
    ###                                                                 ###
    #######################################################################

    def format_dataframe_from_retrieval(self,
        df_in: pd.DataFrame,
        field_pivot: str,
        field_index: str = None,
        field_values: str = None,
        attribute_time_period: AttributeTable = None,
        field_year: str = "year",
        time_period_as_year: bool = None,
    ) -> pd.DataFrame:
        """
        Initialize a data frame that has the correct dimensions for SISEPUEDE

        Function Arguments
        ------------------
        - df_in: data frame (table from SQL) to be formatted
        - field_pivot: field to pivot on, i.e., to use to convert from long to 
            wide

        Keyword Arguments
        ------------------
        - field_index: index field to preserve. If None, defaults to 
            ElectricEnergy.field_nemomod_year
        - field_values: field containing values. If None, defaults to 
            ElectricEnergy.field_nemomod_value
        - attribute_time_period: AttributeTable containing the time periods. If
             None, use ModelAttributes default.
        - field_year: field in attribute_time_period containing the year
        - time_period_as_year: are years time periods? If None, default to 
            ElectricEnergy.nemomod_time_period_as_year
        """

        field_index = self.field_nemomod_year if (field_index is None) else field_index
        field_values = self.field_nemomod_value if (field_values is None) else field_values
        time_period_as_year = self.nemomod_time_period_as_year if (time_period_as_year is None) else time_period_as_year

        df_out = df_in.copy()
        df_out[self.field_nemomod_year] = np.array(df_out[self.field_nemomod_year]).astype(int)

        # does the time period
        if self.nemomod_time_period_as_year:
            df_out[self.field_nemomod_year] = self.transform_field_year_nemomod(
                df_out[self.field_nemomod_year],
                time_period_as_year = time_period_as_year,
                direction = "from_nemomod"
            )

        else:
            attribute_time_period = (
                self.get_attribute_time_period()
                if (attribute_time_period is None) 
                else attribute_time_period
            )
            
            dict_map = attribute_time_period.field_maps.get(f"{field_year}_to_{attribute_time_period.key}")

        df_out = (
            pd.pivot(
                df_out,
                [field_index],
                field_pivot,
                field_values,
            )
            .reset_index(drop = False)
            .rename(
                columns = {
                    self.field_nemomod_year: self.model_attributes.dim_time_period,
                }
            )
            .rename_axis(None, axis = 1)
        )

        return df_out



    def retrieve_nemomod_table_discounted_capital_invesment(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod vdiscountedcapitalinvestment output table and 
            reformats for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the Discounted Capital Investment 
            table. If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = (
            self.model_attributes.table_nemomod_capital_investment_discounted 
            if (table_name is None) 
            else table_name
        )

        subsec = self.model_attributes.get_variable_subsector(self.modvar_entc_nemomod_discounted_capital_investment)


        df_out = self.retrieve_and_pivot_nemomod_table(
            engine,
            self.modvar_entc_nemomod_discounted_capital_investment,
            table_name,
            vector_reference_time_period
        )

        return df_out



    def retrieve_nemomod_table_discounted_capital_invesment_storage(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve NemoMod vdiscountedcapitalinvestment output table and reformat 
            for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the Discounted Capital Investment 
            table. If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = (
            self.model_attributes.table_nemomod_capital_investment_discounted 
            if (table_name is None) 
            else table_name
        )

        df_out = self.retrieve_and_pivot_nemomod_table(
            engine,
            self.modvar_enst_nemomod_discounted_capital_investment_storage,
            self.model_attributes.table_nemomod_capital_investment_storage_discounted,
            vector_reference_time_period,
            field_pivot = self.field_nemomod_storage,
            techs_to_pivot = ["all_techs_st"]
        )

        return df_out



    def retrieve_nemomod_table_discounted_operating_cost(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod generation technologies from vdiscountedoperatingcost 
            output table and reformats for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the Discounted Capital Investment 
            table. If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = self.model_attributes.table_nemomod_operating_cost_discounted if (table_name is None) else table_name

        df_out = self.retrieve_and_pivot_nemomod_table(
            engine,
            self.modvar_entc_nemomod_discounted_operating_costs,
            table_name,
            vector_reference_time_period,
            techs_to_pivot = ["all_techs_pp", "all_techs_fp"]
        )

        return df_out



    def retrieve_nemomod_table_discounted_operating_cost_storage(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod storage technologies from vdiscountedoperatingcost 
            output table and reformats for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the DiscountedOperatingCost table. 
            If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = self.model_attributes.table_nemomod_operating_cost_discounted if (table_name is None) else table_name

        df_out = self.retrieve_and_pivot_nemomod_table(
            engine,
            self.modvar_enst_nemomod_discounted_operating_costs_storage,
            table_name,
            vector_reference_time_period,
            techs_to_pivot = ["all_techs_st"]
        )

        return df_out



    def retrieve_nemomod_table_emissions_by_technology(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod vannualtechnologyemission output table and reformats 
            for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the DiscountedCapitalInvestment 
            table. If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = self.model_attributes.table_nemomod_annual_emissions_by_technology if (table_name is None) else table_name

        modvars_emit = [
            self.modvar_entc_nemomod_emissions_ch4_elec,
            self.modvar_entc_nemomod_emissions_co2_elec,
            self.modvar_entc_nemomod_emissions_n2o_elec,
            self.modvar_entc_nemomod_emissions_ch4_fpr,
            self.modvar_entc_nemomod_emissions_co2_fpr,
            self.modvar_entc_nemomod_emissions_n2o_fpr,
            self.modvar_entc_nemomod_emissions_ch4_mne,
            self.modvar_entc_nemomod_emissions_co2_mne,
            self.modvar_entc_nemomod_emissions_n2o_mne
        ]

        df_out = []
        for modvar in modvars_emit:

            # get the gas, global warming potential (to scale output by), and the query
            gas = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            gwp = self.model_attributes.get_gwp(gas)
            query_append = f"where {self.field_nemomod_emission} = '{gas}'"

            # retrieve and scale
            df_tmp = self.retrieve_and_pivot_nemomod_table(
                engine,
                modvar,
                table_name,
                vector_reference_time_period,
                query_append = query_append,
                techs_to_pivot = None
            )
            df_tmp *= gwp

            df_out.append(df_tmp)

        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        return df_out



    def retrieve_nemomod_fuel_sectoral_demands_and_imports(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        arr_transmission_loss_frac: Union[np.ndarray, None] = None,
        attribute_fuel: Union[AttributeTable, None] = None,
        table_name: Union[str, None] = None,
        table_name_demands: Union[str, None] = None,
        transform_time_period: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves sectoral demands for fuels used in NemoMod and total fuel 
            imports (global) for all fuels from vusebytechnologyannual output 
            table and reformats for SISEPUEDE (wide format data).

        Returns a Tuple of DataFrames

                (
                    df_out_enfu_demand_entc, 
                    df_out_enfu_imports 
                )

            associated with model variables

                (
                    ElectricEnergy.modvar_enfu_energy_demand_by_fuel_entc, 
                    ElectricEnergy.modvar_enfu_imports_fuel
                )


        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in merge--
            e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - arr_transmission_loss_frac: optional array specifying the transmission
            loss fraction by fuel (expanded to all categories). If passed,
            adjusts demands and imports downwards *= (1 - loss_frac)
        - attribute_fuel: AttributeTable used to define universe of fuels. If 
            None, uses self.model_attributes default
        - table_name: name in the database of the Discounted Capital Investment 
            table. If None, use ModelAttributes deault.
        - table_name_demands: table name storing
            DemandsAnnualNonNodal. Used to adjust demands of "upstream fuels"
            (e.g., coal_deposits over coal) that give the true fuel consumption
            of its downstream counterpart.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # some key initialization
        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
        table_name = self.model_attributes.table_nemomod_use_by_technology if (table_name is None) else table_name
        table_name_demands = self.model_attributes.table_nemomod_annual_demand_nn if (table_name_demands is None) else table_name_demands
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_energy_demand_by_fuel_entc)
        dict_tech_info = self.get_tech_info_dict()


        ##  GET FUEL DEMANDS IN ENTC HERE

        """
        upstream fuels are used to account for self-use during production here, 
            upstream production ADDS imports (#3 and #4) and SUBTRACTS specified 
            annual demand (see # 7) to get to sectoral demand for the upstream 
            fuel, which then replaces the downstream fuel.

            this adjustment is a major pAiN iN tHe AsS
        """
        # 1. get raw data frames, then perform adjustments for upstream fuel use
        dict_upstream = self.get_enfu_upstream_fuel_to_replace_downstream_fuel_consumption_map(
            attribute_fuel = attribute_fuel
        )
        all_techs_st = dict_tech_info.get("all_techs_st")
        df_use = sqlutil.sql_table_to_df(engine, table_name)
        df_use = df_use[~df_use[self.field_nemomod_technology].isin(all_techs_st)] if (all_techs_st is not None) else df_use

        # 2. format specified annual demands of downstream fuel, which will be deducted from upstream fuel
        df_demands = sqlutil.sql_table_to_df(engine, table_name_demands).drop(self.field_nemomod_solvedtm, axis = 1)
        field_val_demands = f"{self.field_nemomod_value}_DEMANDS"
        df_demands.rename(columns = {self.field_nemomod_value: field_val_demands}, inplace = True)
        df_demands = df_demands[
            df_demands[self.field_nemomod_fuel].isin(dict_upstream.keys())
        ]

        # 3. account for imports--overwrite import supply techs with fuel
        dict_fuels_to_dummy_techs = dict_tech_info.get("dict_fuels_to_dummy_techs")
        cats_enfu_import = self.get_enfu_cats_with_high_dummy_tech_costs(imports_only = True)
        dict_cats_entc_import = dict((dict_fuels_to_dummy_techs.get(x), dict_upstream.get(x, x)) for x in cats_enfu_import)
        inds = df_use[self.field_nemomod_technology].isin([dict_fuels_to_dummy_techs.get(x) for x in dict_upstream.keys() if dict_fuels_to_dummy_techs.get(x) in dict_cats_entc_import.keys()])

        # 4. break out imports and replace techs for 
        df_imports = df_use[df_use[self.field_nemomod_technology].isin(dict_cats_entc_import.keys())].copy().reset_index(drop = True)
        df_imports[self.field_nemomod_fuel] = df_imports[self.field_nemomod_technology].replace(dict_tech_info.get("dict_dummy_techs_to_fuels"))
        df_use.loc[inds, self.field_nemomod_fuel] = df_use.loc[inds, self.field_nemomod_technology].replace(dict_cats_entc_import)
        df_use = df_use[df_use[self.field_nemomod_fuel] != self.get_dummy_fuel_name()]

        # 5. drop downstream fields, then replace upstream
        flag_drop = "DROP"
        for k, v in dict_upstream.items():
            df_use[self.field_nemomod_fuel].replace({k: flag_drop, v:k}, inplace = True)
        df_use = df_use[df_use[self.field_nemomod_fuel] != flag_drop]

        # 6. aggregate use
        df_use0 = df_use.copy()
        df_use = sf.simple_df_agg(
            df_use,
            [
                self.field_nemomod_fuel,
                self.field_nemomod_region,
                self.field_nemomod_year
            ],
            {self.field_nemomod_value: "sum"}
        )

        # 7. merge in demands, adjust by removing demands
        df_use = pd.merge(df_use, df_demands, how = "left").fillna(0.0)
        df_use[self.field_nemomod_value] = sf.vec_bounds(
            np.array(df_use[self.field_nemomod_value]) - np.array(df_use[field_val_demands]), 
            (0, np.inf)
        )
        df_use.drop(field_val_demands, axis = 1, inplace = True)


        ##  FORMAT OUTPUTS FOR SISEPUEDE

        # sectoral demands for fuel
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_energy_demand_by_fuel_entc)
        df_out_enfu_demand_entc = self.retrieve_and_pivot_nemomod_table(
            df_use,
            self.modvar_enfu_energy_demand_by_fuel_entc,
            table_name,
            vector_reference_time_period,
            field_pivot = self.field_nemomod_fuel,
            techs_to_pivot = None
        )
        df_out_enfu_demand_entc /= scalar_div if (scalar_div is not None) else 1

        # total imports
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_imports_fuel)
        df_out_enfu_imports = self.retrieve_and_pivot_nemomod_table(
            df_imports,
            self.modvar_enfu_imports_fuel,
            table_name,
            vector_reference_time_period,
            field_pivot = self.field_nemomod_fuel,
            techs_to_pivot = None
        )
        df_out_enfu_imports /= scalar_div if (scalar_div is not None) else 1


        ##  PERFORM ANY ADJUSTMENTS FOR TRANSMISSION LOSS IF NECESSARY

        arr_transmission_loss_frac = (
            None
            if arr_transmission_loss_frac.shape != (len(df_out_enfu_demand_entc), attribute_fuel.n_key_values)
            else arr_transmission_loss_frac
        )

        if arr_transmission_loss_frac is not None:
            dict_dfs = {
                self.modvar_enfu_energy_demand_by_fuel_entc: df_out_enfu_demand_entc
                # self.modvar_enfu_imports_fuel: df_out_enfu_imports - don't adjust imports
            }

            for modvar in dict_dfs.keys():
                arr_cur = self.model_attributes.extract_model_variable(#
                    dict_dfs.get(modvar), 
                    modvar,
                    expand_to_all_cats = True,
                    return_type = "array_base",
                )

                dict_dfs.update(
                    {
                        modvar: self.model_attributes.array_to_df(
                            arr_cur*(1 - arr_transmission_loss_frac),
                            modvar,
                            reduce_from_all_cats_to_specified_cats = True,
                        )
                    }
                )

            df_out_enfu_demand_entc = dict_dfs.get(self.modvar_enfu_energy_demand_by_fuel_entc)
            # df_out_enfu_imports = dict_dfs.get(self.modvar_enfu_imports_fuel)

        
        return df_out_enfu_demand_entc, df_out_enfu_imports 



    def retrieve_nemomod_table_total_capacity(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        table_name: str = None,
        transform_time_period: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod vtotalcapacityannual output table and reformats for 
            SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
        - table_name: name in the database of the Discounted Capital Investment 
            table. If None, use ModelAttributes deault.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        table_name = self.model_attributes.table_nemomod_total_annual_capacity if (table_name is None) else table_name
        

        df_out = self.retrieve_and_pivot_nemomod_table(
            engine,
            self.modvar_entc_nemomod_generation_capacity,
            table_name,
            vector_reference_time_period,
            techs_to_pivot = ["all_techs_pp", "all_techs_st"]
        )

        return df_out



    def retrieve_nemomod_tables_fuel_production_demand_and_trade(self,
        engine: sqlalchemy.engine.Engine,
        vector_reference_time_period: Union[list, np.ndarray],
        df_elec_trajectories: Union[pd.DataFrame, None],
        attribute_fuel: Union[AttributeTable, None] = None,
        attribute_technology: Union[AttributeTable, None] = None,
        table_name_demand_annual: str = None,
        table_name_production_by_technology: str = None,
        transform_time_period: bool = True,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod vproductionbytechnologyannual output table and 
            reformats for SISEPUEDE (wide format data)

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table
        - vector_reference_time_period: reference time periods to use in 
            merge--e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]
        - df_elec_trajectories: data frame containing trajectories of input
            variables to SISEPUEDE for NonElectricEnergy. 
            * NOTE: required for extracting transmission losses total fuel use
                costs

        Keyword Arguments
        -----------------
        - attribute_fuel: AttributeTable used to define universe of fuels. If 
            None, uses self.model_attributes defaults
        - attribute_technology: attribute table used to obtain dummy 
            technologies. If None, use ModelAttributes default.
        - table_name_demand_annual: 
        - table_name_production_by_technology: name in the database of the 
            annual production by technology table. If None, use ModelAttributes 
            default.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )

            * NOTES: 
                * MUST BE IN NEMO MOD ENERGY UNITS
                * If None, extracts from df_elec_trajectories
        
        Model Notes
        -----------
        * To model transmission losses in the absence of a network, electricity 
            demands are modeled by inflating two key model elements by the 
            factor *= 1/(1 - loss):
                1. the InputActivityRatio for electricity-consuming
                    technologies (edogenizes transmission loss for fuel 
                    production) and
                2. demands from other energy sectors, which are passed in
                     SpecifiedAnnualDemand.
            This method() accounts for this increase by scaling production and 
            demands by *= (1 - loss) and assigning the total loss.

        * Demands that are returned EXCLUDE transmission losses, keeping with 
            the demands for fuels from other sectors. This is logically 
            consistent with the approach taken with fuel "consumption" by 
            energy subsectors (how much is actually used) versus fuel demands
            by other subsectors (point of use). Here, fuel is demanded to meet
            consumption requirements; what is produced in excess of demand--
            accounting for systemic inefficencies--does not count toward demand
            but consumption (or production in this case). 

        * The mass balance equations are as follows:

            (demands + exports)/(1 - transmission_loss_fractions)
            =
            (production + imports)

            Fuel Imports includes electricity eventually lost to transmission 
            because it represents how much electricity originates from other 
            regions (before loss reaches point of use demand). 
            
            Conversely, Fuel Exports does *not* include transmission loss 
            because it represents how much arrives in the other region, which 
            occurs *after* within-region transmission loss between production 
            and exporting.
        """
        # initialize some pieces
        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
        table_name_demand_annual = self.model_attributes.table_nemomod_annual_demand_nn if (table_name_demand_annual is None) else table_name_demand_annual
        table_name_production_by_technology = self.model_attributes.table_nemomod_production_by_technology if (table_name_production_by_technology is None) else table_name_production_by_technology
        dict_tech_info = self.get_tech_info_dict(
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology
        )
        df_out = [] # initialize output table

        # get fuel production and demands (pre-entc)
        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(df_elec_trajectories, target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")) 
            if (tuple_enfu_production_and_demands is None) 
            else tuple_enfu_production_and_demands
        )
        
        (
            arr_enfu_demands_no_entc, 
            arr_enfu_demands_distribution_no_entc, 
            arr_enfu_export_no_entc, 
            arr_enfu_imports_no_entc, 
            arr_enfu_production_no_entc
         ) = tuple_enfu_production_and_demands

        # get production by technology table from Nemomod
        df_demand_annual = sqlutil.sql_table_to_df(engine, table_name_demand_annual)
        df_production_by_technology = sqlutil.sql_table_to_df(engine, table_name_production_by_technology)       


        ##  GET FUEL DEMANDS WITHIN ENTC AND IMPORTS, FUEL PRODUCTION, AND FUEL DEMANDS

        # 1. retrieve transmission loss fraction (adjusts demands)
        arr_transmission_loss_frac = self.model_attributes.extract_model_variable(#
            df_elec_trajectories, 
            self.modvar_enfu_transmission_loss_frac_electricity, 
            expand_to_all_cats = True, 
            return_type = "array_base",
            var_bounds = (0, 1),
        )


        # 2. get demands in ENTC + imports
        #  These have units self.modvar_enfu_energy_demand_by_fuel_entc and self.modvar_enfu_imports_fuel
        df_enfu_demands_entc, df_enfu_imports = self.retrieve_nemomod_fuel_sectoral_demands_and_imports(
            engine,
            vector_reference_time_period,
            arr_transmission_loss_frac = arr_transmission_loss_frac,
            attribute_fuel = attribute_fuel,
            transform_time_period = transform_time_period,
        )

        df_out += [df_enfu_demands_entc, df_enfu_imports]
    
        arr_enfu_imports = self.model_attributes.extract_model_variable(#
            df_enfu_imports,
            self.modvar_enfu_imports_fuel,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        arr_enfu_imports *= self.get_nemomod_energy_scalar(self.modvar_enfu_imports_fuel)


        # 3. add fuel production
        df_fuel_production = self.get_enfu_fuel_production_from_total_production(
            df_production_by_technology,
            vector_reference_time_period,
            attribute_fuel = attribute_fuel,
            attribute_technology = attribute_technology,
            dict_tech_info = dict_tech_info,
            modvar_enfu_production = self.modvar_enfu_production_fuel
        )
        df_out += [df_fuel_production]


        # 4. get fuel production array for use in calculating adjusted exports
        arr_enfu_production = self.model_attributes.extract_model_variable(#
            df_fuel_production,
            self.modvar_enfu_production_fuel,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        arr_enfu_production *= self.get_nemomod_energy_scalar(self.modvar_enfu_production_fuel)


        # 5. get demands from other subsectors (pull from here) - option is self.get_enfu_non_entc_fuel_demands_from_annual_demand
        arr_enfu_demand_entc = self.model_attributes.extract_model_variable(#
            df_enfu_demands_entc,
            self.modvar_enfu_energy_demand_by_fuel_entc,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        scalar_enfu_energy_demand_entc_to_nemo_units = self.get_nemomod_energy_scalar(self.modvar_enfu_energy_demand_by_fuel_entc)

        # demands WITHOUT losses to transmission
        arr_enfu_demands = arr_enfu_demands_no_entc + arr_enfu_demand_entc*scalar_enfu_energy_demand_entc_to_nemo_units

        scalar_div = self.get_nemomod_energy_scalar(
            self.modvar_enfu_energy_demand_by_fuel_total, 
            force_modvar_to_config_on_none = True
        )
        df_enfu_demands = self.model_attributes.array_to_df(
            arr_enfu_demands/scalar_div,
            self.modvar_enfu_energy_demand_by_fuel_total,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_out += [df_enfu_demands]

        
        # 6. get total value of fuel CONSUMED in ENTC
        scalar_enfu_demand_entc_to_config_energy = self.model_attributes.get_scalar(
            self.modvar_enfu_energy_demand_by_fuel_entc, 
            "energy"
        )
        # arr_entc_total_fuel_value is in 
        #  - configuration monetary units (get_enfu_fuel_costs_per_energy default) and 
        #  - units of modvar_enfu_energy_demand_by_fuel_entc
        arr_entc_total_fuel_value = self.model_energy.get_enfu_fuel_costs_per_energy(
            df_elec_trajectories,
            modvar_for_units_energy = self.modvar_enfu_energy_demand_by_fuel_entc
        )
        # multply by   scalar_enfu_demand_entc_to_config_energy   to conver everything to configuration energy units
        df_enfu_costs = self.model_attributes.array_to_df(
            arr_entc_total_fuel_value*arr_enfu_demand_entc*scalar_enfu_demand_entc_to_config_energy,
            self.modvar_enfu_value_of_fuel_entc,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_out += [df_enfu_costs]


        # 7. add in transmission loss totals
        arr_enfu_transmission_losses = (arr_enfu_production + arr_enfu_imports)*arr_transmission_loss_frac
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_transmission_loss_electricity)
        df_enfu_transmission_losses = self.model_attributes.array_to_df(
            arr_enfu_transmission_losses/scalar_div,
            self.modvar_enfu_transmission_loss_electricity,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_out += [df_enfu_transmission_losses]


        # 8. get adjusted exports as production + imports - demands
        arr_enfu_exports_adj = arr_enfu_production + arr_enfu_imports - arr_enfu_transmission_losses - arr_enfu_demands
        arr_enfu_exports_adj = sf.vec_bounds(arr_enfu_exports_adj, (0, np.inf))
        scalar_div = self.get_nemomod_energy_scalar(self.modvar_enfu_exports_fuel_adjusted)
        df_enfu_exports_adj = self.model_attributes.array_to_df(
            arr_enfu_exports_adj/scalar_div,
            self.modvar_enfu_exports_fuel_adjusted,
            reduce_from_all_cats_to_specified_cats = True
        )
        df_out += [df_enfu_exports_adj]


        ## ADD ELECTRICITY GENERATION BY TECH

        # start by checking units
        scalar_arg = self.model_attributes.get_variable_characteristic(
            self.modvar_entc_nemomod_production_by_technology, 
            self.model_attributes.varchar_str_unit_energy
        )
        scalar_arg = None if (scalar_arg is None) else self.modvar_entc_nemomod_production_by_technology
        scalar_div = self.get_nemomod_energy_scalar(scalar_arg)

        df_out_production = self.retrieve_and_pivot_nemomod_table(
            df_production_by_technology,
            self.modvar_entc_nemomod_production_by_technology,
            None,
            vector_reference_time_period,
            techs_to_pivot = ["all_techs_pp", "all_techs_st"]
        )/scalar_div

        # add to output
        df_out += [df_out_production]
    
        return pd.concat(df_out, axis = 1).reset_index(drop = True)



    def retrieve_and_pivot_nemomod_table(self,
        engine: Union[pd.DataFrame, sqlalchemy.engine.Engine],
        modvar: str,
        table_name: str,
        vector_reference_time_period: Union[list, np.ndarray],
        dict_agg_info: Union[Dict, None] = None,
        dict_filter_override: Union[Dict, None] = None,
        dict_repl_values: Union[Dict[str, str], None] = None,
        field_pivot: Union[str, None] = None,
        query_append: Union[str, None] = None,
        techs_to_pivot: Union[List[str], None] = ["all_techs_pp"],
        transform_time_period: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves NemoMod output table and reformats for SISEPUEDE (wide format 
            data) when pivoting on technology

        Function Arguments
        ------------------
        - engine: SQLalchemy Engine used to retrieve this table OR data frame
            passed in place of raw output
        - modvar: output model variable
        - table_name: name in the database of the table to retrieve
        - vector_reference_time_period: reference time periods to use in merge--
            e.g., 
            df_elec_trajectories[ElectricEnergy.model_attributes.dim_time_period]

        Keyword Arguments
        -----------------
         - dict_agg_info: dictionary specificying optional fields to group on + 
            fields to aggregate. If specified, aggregation is applied to the 
            dataframe that comes from the NemoMod database. Dictionary should
            have the form:

            {
                "fields_group": [fld_1,..., fld_2],
                "agg_info: {
                    "fld_agg_1": func,
                    ...
                }
            }

            where `func` is an acceptable aggregation function passed to 
            pd.GroupedDataFrame.agg()
        - dict_filter_override: filtering dictionary to apply independently of 
            techs_to_pivot. Filters on top of techs_to_pivot if provided.
        - dict_repl_values: dictionary of dictionaries mapping a field to apply
            the replacement to (key) to a dictionary of replacement pairs 
            (value). Performed immediately *after* filtering.  
        - field_pivot: field to pivot on. Default is 
            ElecticEnergy.field_nemomod_technology, but 
            ElecticEnergy.field_nemomod_storage can be used to transform storage 
            outputs to technology.
        - query_append: appendage to query (e.g., "where X = 0")
        - techs_to_pivot: list of keys in ElecticEnergy.get_tech_info_dict() to 
            include in the pivot. Can include "all_techs_pp", 
            "all_techs_st", "all_techs_dummy" (only if output sector is fuel). 
            If None, keeps all values.
        - transform_time_period: Does the time period need to be transformed 
            back to SISEPUEDE terms?
        """

        # initialize some pieces
        subsec = self.model_attributes.get_variable_subsector(modvar)
        attr = self.model_attributes.get_attribute_table(subsec)
        attr_tech = attr if (subsec == self.subsec_name_entc) else self.model_attributes.get_attribute_table(self.subsec_name_entc)
        dict_tech_info = self.get_tech_info_dict(attribute_technology = attr_tech)
        field_pivot = self.field_nemomod_technology if (field_pivot is None) else field_pivot

        # techs to filter on
        cats_filter = sum([dict_tech_info.get(x) for x in techs_to_pivot], []) if (techs_to_pivot is not None) else None

        # generate any replacement dictionaries
        dict_repl = {}
        if (subsec == self.subsec_name_enfu) and (field_pivot != self.field_nemomod_fuel):
            dict_repl = dict_tech_info.get("dict_dummy_techs_to_fuels")
        elif (subsec == self.subsec_name_enst) and (field_pivot != self.field_nemomod_storage):
            dict_repl = dict_tech_info.get("dict_storage_techs_to_storage")

        # get data frame
        if isinstance(engine, sqlalchemy.engine.Engine):
            df_table_nemomod = sqlutil.sql_table_to_df(engine, table_name, query_append = query_append)
        elif isinstance(engine, pd.DataFrame):
            df_table_nemomod = engine
        else:
            tp = type(engine)
            raise ValueError(f"Error in retrieve_and_pivot_nemomod_table: invalid engine type '{tp}'")
        
        # apply field replacements if needed
        if dict_repl_values is not None:
            for field in dict_repl_values.keys():
                if field in df_table_nemomod.keys():
                    dict_repl_cur = dict_repl_values.get(field)
                    df_table_nemomod[field].replace(dict_repl_cur, inplace = True) if (dict_repl_cur is not None) else None

        # apply an optional aggregation to the dataframe after retrieving
        if dict_agg_info is not None:
            fields_group = dict_agg_info.get("fields_group")
            dict_agg = dict_agg_info.get("dict_agg")
            if (fields_group is not None) and (dict_agg is not None):
                df_table_nemomod = sf.simple_df_agg(
                    df_table_nemomod,
                    fields_group,
                    dict_agg
                )

        # reduce data frame to techs (should be trivial)
        df_source = df_table_nemomod[df_table_nemomod[field_pivot].isin(cats_filter)] if (cats_filter is not None) else df_table_nemomod
        df_source = sf.subset_df(df_source, dict_filter_override) if (dict_filter_override is not None) else df_source
        df_source[field_pivot] = df_source[field_pivot].replace(dict_repl)

        # build renaming dictionary
        cats_valid = self.model_attributes.get_variable_categories(modvar)
        dict_cats_to_varname = [x for x in attr.key_values if (x in list(df_source[field_pivot])) and (x in cats_valid)]
        varnames = self.model_attributes.build_variable_fields(
            modvar, 
            restrict_to_category_values = dict_cats_to_varname,
        )

        dict_cats_to_varname = dict(zip(dict_cats_to_varname, varnames))

        # reformat using pivot
        df_source = self.format_dataframe_from_retrieval(
            df_source[[
                field_pivot,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]],
            field_pivot
        ).rename(
            columns = dict_cats_to_varname
        )

        # initialize output
        df_out = self.model_attributes.instantiate_blank_modvar_df_by_categories(
            modvar, 
            len(vector_reference_time_period),
        )
        df_out[self.model_attributes.dim_time_period] = vector_reference_time_period

        # match the target
        df_out = sf.match_df_to_target_df(
            df_out,
            df_source,
            [self.model_attributes.dim_time_period]
        )

        # ensure time_period is properly ordered
        df_out = sf.orient_df_by_reference_vector(
            df_out,
            vector_reference_time_period,
            self.model_attributes.dim_time_period,
            drop_field_compare = True
        )

        return df_out



    ####################################################
    ###                                              ###
    ###    SQL FUNCTIONS FOR DATABASE INTERACTIONS   ###
    ###                                              ###
    ####################################################

    def generate_input_tables_for_sql(self,
        df_elec_trajectories: pd.DataFrame,
        df_reference_capacity_factor: pd.DataFrame,
        df_reference_specified_demand_profile: pd.DataFrame,
        dict_attributes: Dict[str, pd.DataFrame] = {},
        regions: Union[List[str], None] = None,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve tables from applicable inputs and format as dictionary. 
            Returns a dictionary of the form

            {NemoModTABLE: df_table, ...}

        where NemoModTABLE is an appropriate table.

        Function Arguments
        ------------------
        - df_elec_trajectories: input of required variabels passed from other 
            SISEPUEDE sectors.
        - df_reference_capacity_factor: reference data frame containing capacity 
            factors
        - df_reference_specified_demand_profile:r eference data frame containing 
            the specified demand profile by region

        Keyword Arguments
        -----------------
        - dict_attributes: dictionary of attribute tables that can be used to 
            pass attributes to downstream format_nemomod_table_ functions. If 
            passed, the following keys are used to represent attributes:
                * attribute_emission: EMISSION attribute table
                * attribute_fuel: FUEL attribute table
                * attribute_mode: MODE attribute table
                * attribute_storage: STORAGE attribute table
                * attribute_technology: TECHNOLOGY attribute table
                * attribute_time_slice: TIMESLICE attribute table
        - regions: regions to generate input tables for
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        """
        
        ##  INITIALIZE SHARED COMPONENTS

        # initilize attribute tables to pass--if they are not in the dictionary, they will return None, and defaults are used
        attribute_emission = dict_attributes.get("attribute_emission")
        attribute_fuel = dict_attributes.get("attribute_fuel")
        attribute_mode = dict_attributes.get("attribute_mode")
        attribute_nemomod_table = dict_attributes.get("nemomod_table")
        attribute_region =  dict_attributes.get("attribute_region")
        attribute_storage = dict_attributes.get("attribute_storage")
        attribute_technology = dict_attributes.get("attribute_technology")
        attribute_time_period = dict_attributes.get("attribute_time_period")
        attribute_time_slice = dict_attributes.get("attribute_time_slice")
        
        # check specification of regions and check energy production and demand inputs
        regions = self.model_attributes.get_region_list_filtered(regions, attribute_region = attribute_region)
        tuple_enfu_production_and_demands = (
            self.model_energy.project_enfu_production_and_demands(
                df_elec_trajectories, 
                target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
            )
            if tuple_enfu_production_and_demands is None
            else tuple_enfu_production_and_demands
        )


        ##  BUILD TABLES FOR NEMOMOD

        dict_out = {}


        ##  1. START WITH ATTRIBUTE TABLES
        
        # EMISSION
        dict_out.update(
            self.format_nemomod_attribute_table_emission(
                attribute_emission = attribute_emission
            )
        )
        # FUEL
        dict_out.update(
            self.format_nemomod_attribute_table_fuel(
                attribute_fuel = attribute_fuel
            )
        )
        # MODEOFOPERATION
        dict_out.update(
            self.format_nemomod_attribute_table_mode_of_operation(
                attribute_mode = attribute_mode
            )
        )
        # REGION
        dict_out.update(
            self.format_nemomod_attribute_table_region(
                attribute_region = attribute_region,
                regions = regions
            )
        )
        # STORAGE
        dict_out.update(
            self.format_nemomod_attribute_table_storage(
                attribute_storage = attribute_storage
            )
        )
        # TECHNOLOGY
        dict_out.update(
            self.format_nemomod_attribute_table_technology(
                attribute_technology = attribute_technology
            )
        )
        # YEAR
        dict_out.update(
            self.format_nemomod_attribute_table_year()
        )
        # TIMESLICE
        dict_out.update(
            self.format_nemomod_table_tsgroup_tables(
                attribute_time_slice = attribute_time_slice
            )
        )


        ##  2. ADD TABLES THAT ARE INDEPENDENT OF MODEL PASS THROUGH (df_elec_trajectories)

        # DefaultParams
        dict_out.update(
            self.format_nemomod_table_default_parameters(
                attribute_nemomod_table = attribute_nemomod_table
            )
        )
        # OperationalLife and OperationalLifeStorage
        dict_out.update(
            self.format_nemomod_table_operational_life(
                attribute_fuel = attribute_fuel,
                attribute_storage = attribute_storage,
                attribute_technology = attribute_technology,
                regions = regions
            )
        )
        # ReserveMarginTagFuel
        dict_out.update(
            self.format_nemomod_table_reserve_margin_tag_fuel(
                regions = regions
            )
        )
        # TechnologyFromStorage and TechnologyToStorage
        dict_out.update(
            self.format_nemomod_table_technology_from_and_to_storage(
                attribute_storage = attribute_storage,
                attribute_technology = attribute_technology,
                regions = regions
            )
        )


        ##  3. ADD TABLES DEPENDENT ON MODEL PASS THROUGH (df_elec_trajectories)

        if df_elec_trajectories is not None:
            # AnnualEmissionLimit
            dict_out.update(
                self.format_nemomod_table_annual_emission_limit(
                    df_elec_trajectories,
                    attribute_emission = attribute_emission,
                    attribute_time_period = attribute_time_period,
                    regions = regions
                )
            )
            # CapitalCostStorage
            dict_out.update(
                self.format_nemomod_table_costs_storage(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # CapitalCost, FixedCost, and VariableCost -- Costs (Technology)
            dict_out.update(
                self.format_nemomod_table_costs_technology(
                    df_elec_trajectories,
                    attribute_fuel = attribute_fuel,
                    regions = regions
                )
            )
            # EmissionsActivityRatio - Emission Factors
            dict_out.update(
                self.format_nemomod_table_emissions_activity_ratio(
                    df_elec_trajectories, 
                    attribute_fuel = attribute_fuel, 
                    attribute_technology = attribute_technology, 
                    attribute_time_period = attribute_time_period,
                    regions = regions
                )
            )
            # InputActivityRatio
            dict_out.update(
                self.format_nemomod_table_input_activity_ratio(
                    df_elec_trajectories, 
                    attribute_technology = attribute_technology,
                    regions = regions
                )
            )
            # MinShareProduction
            dict_out.update(
                self.format_nemomod_table_min_share_production(
                    df_elec_trajectories, 
                    attribute_fuel = attribute_fuel,
                    regions = regions,
                    tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
                )
            )
            # MinStorageCharge
            dict_out.update(
                self.format_nemomod_table_min_storage_charge(
                    df_elec_trajectories, 
                    attribute_storage = attribute_storage,
                    regions = regions
                )
            )
            # OutputActivityRatio
            dict_out.update(
                self.format_nemomod_table_output_activity_ratio(
                    df_elec_trajectories, 
                    attribute_fuel = attribute_fuel, 
                    attribute_technology = attribute_technology,
                    regions = regions
                )
            )
            # ReserveMargin
            dict_out.update(
                self.format_nemomod_table_reserve_margin(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # ReserveMarginTagTechnology
            dict_out.update(
                self.format_nemomod_table_reserve_margin_tag_technology(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # ResidualCapacity
            dict_out.update(
                self.format_nemomod_table_residual_capacity(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # ResidualStorageCapacity
            dict_out.update(
                self.format_nemomod_table_residual_storage_capacity(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # REMinProductionTarget
            dict_out.update(
                self.format_nemomod_table_re_min_production_target(
                    df_elec_trajectories,
                    attribute_fuel = attribute_fuel,
                    regions = regions
                )
            )
            # RETagTechnology
            dict_out.update(
                self.format_nemomod_table_re_tag_technology(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # SpecifiedAnnualDemand
            dict_out.update(
                self.format_nemomod_table_specified_annual_demand(
                    df_elec_trajectories, 
                    attribute_time_period = attribute_time_period, 
                    regions = regions,
                    tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
                )
            )
            # StorageMaxChargeRate (if included), StorageMaxDishargeRate (if included), and StorageStartLevel
            dict_out.update(
                self.format_nemomod_table_storage_attributes(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # TotalAnnualMax/MinCapacity +/-Investment
            dict_out.update(
                self.format_nemomod_table_total_capacity_tables(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # TotalAnnualMax/MinCapacity +/-Investment Storage
            dict_out.update(
                self.format_nemomod_table_total_capacity_storage_tables(
                    df_elec_trajectories,
                    regions = regions
                )
            )
            # TotalTechnologyAnnualActivityLowerLimit
            dict_out.update(
                self.format_nemomod_table_total_technology_activity_lower_limit(
                    df_elec_trajectories, 
                    attribute_technology = attribute_technology,
                    regions = regions,
                    tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
                )
            )
            # TotalTechnologyAnnualActivityUpperLimit
            dict_out.update(
                self.format_nemomod_table_total_technology_activity_upper_limit(
                    df_elec_trajectories, 
                    attribute_technology = attribute_technology,
                    regions = regions,
                    tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
                )
            )
            

        # CapacityFactor
        if df_reference_capacity_factor is not None:
            dict_out.update(
                self.format_nemomod_table_capacity_factor(
                    df_elec_trajectories,
                    df_reference_capacity_factor,
                    attribute_technology = attribute_technology,
                    attribute_region = attribute_region,
                    regions = regions,
                )
            )
        # SpecifiedDemandProfile
        if df_reference_specified_demand_profile is not None:
            dict_out.update(
                self.format_nemomod_table_specified_demand_profile(
                    df_reference_specified_demand_profile,
                    attribute_region = attribute_region,
                    regions = regions,
                )
            )


        ##  Prepare data for write
        for table in dict_out.keys():
            dict_dtype = {}
            for k in self.dict_fields_nemomod_to_type.keys():
                dict_dtype.update({k: self.dict_fields_nemomod_to_type[k]}) if (k in dict_out[table].columns) else None

            dict_out[table] = dict_out[table].astype(dict_dtype)

        return dict_out



    def retrieve_output_tables_from_sql(self,
        engine: sqlalchemy.engine.Engine,
        df_elec_trajectories: pd.DataFrame,
        tuple_enfu_production_and_demands: Union[Tuple[pd.DataFrame], None] = None
    ) -> dict:
        """
        Retrieve tables from applicable inputs and format as dictionary. Returns 
            a data frame ordered by time period that can be concatenated with 
            df_elec_trajectories.

        Function Arguments
        ------------------
        - engine: SQLalchemy engine used to connect to output database
        - df_elec_trajectories: input of required variabels passed from other 
            SISEPUEDE sectors.

        Keyword Arguments
        -----------------
        - tuple_enfu_production_and_demands: optional tuple of energy fuel 
            demands produced by 
            self.model_energy.project_enfu_production_and_demands():

            (
                arr_enfu_demands, 
                arr_enfu_demands_distribution, 
                arr_enfu_export, 
                arr_enfu_imports, 
                arr_enfu_production
            )
        """

        vec_time_period = list(df_elec_trajectories[self.model_attributes.dim_time_period])

        df_out = [
            df_elec_trajectories[[self.model_attributes.dim_time_period]],
            self.retrieve_nemomod_table_discounted_capital_invesment(
                engine, 
                vec_time_period
            ),
            self.retrieve_nemomod_table_discounted_capital_invesment_storage(
                engine, 
                vec_time_period
            ),
            self.retrieve_nemomod_table_discounted_operating_cost(
                engine, 
                vec_time_period
            ),
            self.retrieve_nemomod_table_discounted_operating_cost_storage(
                engine, 
                vec_time_period
            ),
            self.retrieve_nemomod_table_total_capacity(
                engine, 
                vec_time_period
            )
        ]

        ##  RETRIEVE EMISSIONS AND FUEL PRODUCTION DEMANDS SEPARATELY
        ##  - USE TO BUILD ALLOCATIONS OF ELEC GENERATION EMISSIONS TO OTHER SUBSECTORS 

        df_entc_emissions = self.retrieve_nemomod_table_emissions_by_technology(
            engine, 
            vec_time_period
        )
        df_entc_fuelprod = self.retrieve_nemomod_tables_fuel_production_demand_and_trade(
            engine, 
            vec_time_period, 
            df_elec_trajectories,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
        )

        # concatenate for use in allocation of emissions within ENTC
        df_entc_emissions_and_fuel_prod = pd.concat(
            [
                df_entc_emissions,
                df_entc_fuelprod
            ],
            axis = 1
        )

        df_out += [
            # add emissions and fuel production
            df_entc_emissions_and_fuel_prod,
            # add allocation of emissions by energy demand
            self.allocate_entc_emissions_by_energy_demand(
                df_elec_trajectories,
                df_entc_emissions_and_fuel_prod,
                cat_enfu_energy_source = self.cat_enfu_elec
            )
        ]

    
        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        return df_out






    ###############################
    ###                         ###
    ###    PROJECTION METHOD    ###
    ###                         ###
    ###############################

    def project(self,
        df_elec_trajectories: pd.DataFrame,
        engine: sqlalchemy.engine.Engine = None,
        fp_database: str = None,
        dict_ref_tables: dict = None,
        missing_vals_on_error: Union[int, float] = 0.0,
        regions: Union[List[str], None] = None,
        return_blank_df_on_error: bool = False,
        solver: str = None,
        vector_calc_time_periods: list = None
    ) -> pd.DataFrame:

        """
        Project electricity emissions and costs using NemoMod. Primary method of 
            ElectricEnergy.

        Function Arguments
        ------------------
        - df_elec_trajectories: data frame of input trajectories

        Keyword Arguments
        ------------------
        - engine: SQLalchemy database engine used to connect to the database. If 
            None, creates an engine using fp_database.
        - fp_database: file path to sqlite database to use for NemoMod. If None, 
            creates an SQLAlchemy engine (it is recommended that, if running in 
            batch, a single engine is created and called multiple times)
        - dict_ref_tables: dictionary of reference tables required to prepare 
            data for NemoMod. If None, use 
            ElectricEnergy.dict_nemomod_reference_tables (initialization data)
        - missing_vals_on_error: if a data frame is returned on an error, fill 
            with this value
        - regions: list of regions or str defining region to run. If None, 
            defaults to configuration specification
        - return_blank_df_on_error: on a NemoMod error (such as an 
            infeasibility), return a data frame filled with 
            missing_vals_on_error?
        - solver: string specifying the solver to use to run NemoMod. If None, 
            default to SISEPUEDE configuration value.
        - vector_calc_time_periods: list of time periods in NemoMod to run. If 
            None, use configuration defaults.

        Note
        ----
        * Either engine or fp_database must be specified to run project. If both 
            are specifed, engine takes precedence.
        """

        ##  CHECKS AND INITIALIZATION

        # make sure socioeconomic variables are added and
        df_elec_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_elec_trajectories)
        
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_elec_trajectories)
        (
            dict_dims, 
            df_elec_trajectories, 
            n_projection_time_periods, 
            projection_time_periods
        ) = self.model_attributes.check_projection_input_df(
            df_elec_trajectories, 
            True, 
            True, 
            True,
        )

        # check the dictionary of reference tables
        dict_ref_tables = self.dict_nemomod_reference_tables if (dict_ref_tables is None) else dict_ref_tables
        sf.check_keys(dict_ref_tables, [
            self.model_attributes.table_nemomod_capacity_factor,
            self.model_attributes.table_nemomod_specified_demand_profile
        ])

        # initialize output
        df_out = [df_elec_trajectories[self.required_dimensions].copy()]


        ####################################
        #    BEGIN NEMO MOD INTEGRATION    #
        ####################################

        ##  1. PREPARE AND POPULATE THE DATABASE

        # check engine/fp_database
        str_prepend_sqlite = "sqlite:///"
        if (engine is None) and (fp_database is None):
            raise RuntimeError(f"Error in ElectricEnergy.project(): either 'engine' or 'fp_database' must be specified.")
        elif (fp_database is None):
            fp_database = str(engine.url).replace(str_prepend_sqlite, "")

        # check path of NemoMod database and create if necessary
        recreate_engine_q = False
        if not os.path.exists(fp_database):
            self._log(f"\tPath to temporary NemoMod database '{fp_database}' not found. Creating...", type_log = "info")
            self.julia_nemomod.createnemodb(fp_database)
            recreate_engine_q = True

        # check the engine and respecify if the original database, for whatever reason, no longer exists
        if (engine is None) or recreate_engine_q:
            engine = sqlalchemy.create_engine(f"{str_prepend_sqlite}{fp_database}")

        # get shared energy variables that are required before and after NemoMod runs
        tuple_enfu_production_and_demands = self.model_energy.project_enfu_production_and_demands(
            df_elec_trajectories, 
            target_energy_units = self.model_attributes.configuration.get("energy_units_nemomod")
        )

        # get data for the database
        dict_to_sql = self.generate_input_tables_for_sql(
            df_elec_trajectories,
            dict_ref_tables.get(self.model_attributes.table_nemomod_capacity_factor),
            dict_ref_tables.get(self.model_attributes.table_nemomod_specified_demand_profile),
            regions = regions,
            tuple_enfu_production_and_demands = tuple_enfu_production_and_demands,
        )

        # try to write input tables to the NemoMod scenario database
        try:
            sqlutil._write_dataframes_to_db(
                dict_to_sql,
                engine
            )
        except Exception as e:
            self._log(f"Error writing data to {fp_database}: {e}", type_log = "error")
            return None


        ##  2. SET UP AND CALL NEMOMOD

        # get calculation time periods
        attr_time_period = self.model_attributes.get_attribute_time_period()
        vector_calc_time_periods = (
            self.model_attributes.configuration.get("nemomod_time_periods") 
            if (vector_calc_time_periods is None) 
            else [x for x in attr_time_period.key_values if x in vector_calc_time_periods]
        )
        vector_calc_time_periods = self.transform_field_year_nemomod(vector_calc_time_periods)
        
        self.julia_main.vector_calc_time_periods = vector_calc_time_periods
        self.julia_main.eval("vector_calc_time_periods = Int64.(collect(vector_calc_time_periods))")
        
        # get the optimizer (must reset each time) and vars to save
        optimizer = self.get_nemomod_optimizer(solver)
        vars_to_save = ", ".join(self.required_nemomod_output_tables)

        try:
            # call nemo mod
            result = self.julia_nemomod.calculatescenario(
                fp_database,
                jumpmodel = optimizer,
                numprocs = 1,
                calcyears = self.julia_main.vector_calc_time_periods,#vector_calc_time_periods
                reportzeros = False,
                varstosave = vars_to_save,
                quiet = True
            )

        except Exception as e:
            # LOG THE ERROR HERE
            self._log(f"Error in ElectricEnergy when trying to run NemoMod: {e}", type_log = "error")
            result = None


        ##  3. RETRIEVE OUTPUT TABLES

        # initialize as unsuccessful, then check if it worked--if so, retrieve output tables
        successful_run_q = False
        if result is not None:
            if "infeasible" not in str(result).lower():
                successful_run_q = True

                df_out += [
                    self.retrieve_output_tables_from_sql(
                        engine, 
                        df_elec_trajectories, 
                        tuple_enfu_production_and_demands = tuple_enfu_production_and_demands
                    )
                ]

        # if specified in output, create a uniformly-valued dataframe for runs that did not successfully complete
        if return_blank_df_on_error and not successful_run_q:
            modvars_instantiate = [
                self.modvar_entc_nemomod_discounted_capital_investment,
                self.modvar_enst_nemomod_discounted_capital_investment_storage,
                self.modvar_entc_nemomod_discounted_operating_costs,
                self.modvar_enst_nemomod_discounted_operating_costs_storage,
                self.modvar_entc_nemomod_emissions_ch4_elec,
                self.modvar_entc_nemomod_emissions_co2_elec,
                self.modvar_entc_nemomod_emissions_n2o_elec,
                self.modvar_enfu_energy_demand_by_fuel_entc,
                self.modvar_entc_nemomod_generation_capacity,
                self.modvar_entc_nemomod_production_by_technology
            ]

            df_out += [
                self.model_attributes.instantiate_blank_modvar_df_by_categories(
                    modvar, 
                    n = len(df_elec_trajectories), 
                    blank_val = missing_vals_on_error
                ) for modvar in modvars_instantiate
            ]

        msg = f"NemoMod ran successfully with the following status: {result}" if successful_run_q else f"NemoMod run failed with result {result}. Populating missing data with value {missing_vals_on_error}."
        self._log(msg, type_log = "info")


        ##  ADD IN UNUSED FUEL

        # try retrieving output from NemoMod
        try:
            arr_enfu_fuel_demand_elec = self.model_attributes.extract_model_variable(#
                df_out[- 1],
                self.modvar_enfu_energy_demand_by_fuel_entc,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
            )
            add_unused_fuel = True

        except:
            self._log(
                "Unable to retrieve energy demand by fuel in ENTC. Skipping adding unused fuel...", 
                type_log = "info",
            )
            add_unused_fuel = False

        # get biogas and waste supply available
        vec_enfu_total_energy_supply_biogas, vec_enfu_min_energy_to_elec_biogas = self.get_biogas_components(
            df_elec_trajectories
        )

        (
            vec_enfu_total_energy_supply_waste, 
            vec_enfu_min_energy_to_elec_waste, 
            dict_efs
        ) = self.get_waste_energy_components(
            df_elec_trajectories,
            return_emission_factors = False,
        )

        # adjust units from NemoMod Energy units to those of self.modvar_enfu_unused_fuel_exported
        scalar = self.get_nemomod_energy_scalar(self.modvar_enfu_unused_fuel_exported)
        vec_enfu_total_energy_supply_biogas /= scalar
        vec_enfu_total_energy_supply_waste /= scalar

        # adjust by fuel used?
        vec_used_bgas = 0.0
        vec_used_wste = 0.0
        if add_unused_fuel:
            # do units converison
            arr_enfu_fuel_demand_elec *= self.model_attributes.get_variable_unit_conversion_factor(
                self.modvar_enfu_energy_demand_by_fuel_entc,
                self.modvar_enfu_unused_fuel_exported,
                "energy"
            )
            vec_used_bgas = arr_enfu_fuel_demand_elec[:, self.ind_enfu_bgas]
            vec_used_wste = arr_enfu_fuel_demand_elec[:, self.ind_enfu_wste]

        # initialize output, add biogas/waste unused (presumed exported), and add to dataframe
        arr_enfu_total_unused_fuel_exported = np.zeros((len(df_elec_trajectories), self.model_attributes.get_attribute_table(self.subsec_name_enfu).n_key_values))
        arr_enfu_total_unused_fuel_exported[:, self.ind_enfu_bgas] = sf.vec_bounds(vec_enfu_total_energy_supply_biogas - vec_used_bgas, (0, np.inf))
        arr_enfu_total_unused_fuel_exported[:, self.ind_enfu_wste] = sf.vec_bounds(vec_enfu_total_energy_supply_waste - vec_used_wste, (0, np.inf))

        df_out += [
            self.model_attributes.array_to_df(
                arr_enfu_total_unused_fuel_exported, 
                self.modvar_enfu_unused_fuel_exported, 
                reduce_from_all_cats_to_specified_cats = True,
            )
        ]


        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, merge_type = "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_entc], False)


        return df_out
