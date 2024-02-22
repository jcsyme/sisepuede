from attribute_table import AttributeTable
import logging
from model_attributes import *
from model_socioeconomic import Socioeconomic
import numpy as np
import pandas as pd
import support_functions as sf
import time


#########################
###                   ###
###     IPPU MODEL    ###
###                   ###
#########################

class IPPU:
    """
    Use IPPU to calculate emissions from Industrial Processes and Product Use in
        SISEPUEDE. Includes emissions from the following subsectors:

        * Industrial Processes and Product Use (IPPU)

    Additionally, includes key function for estimating industrial production 
        with recycling adjustments (used in CircularEconomy) and connections to
        wood production/harvested wood products. See the 
        `IPPU.get_production_with_recycling_adjustment()` method for more 
        information on these key connections.

    For additional information, see the SISEPUEDE readthedocs at:

        https://sisepuede.readthedocs.io/en/latest/ippu.html


    Intialization Arguments
    -----------------------
    - model_attributes: ModelAttributes object used in SISEPUEDE

    Optional Arguments
    ------------------
    - logger: optional logger object to use for event logging
    """

    def __init__(self, 
        attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None
    ):
        self.logger = logger
        self.model_attributes = attributes
        
        self._initialize_subsector_names()
        self._initialize_subsector_vars_ippu()
        self._initialize_fc_emission_factor_modvars()

        self._initialize_input_output_components()
        self._initialize_models()
        self._initialize_other_properties()
        self._initialize_integrated_variables()


    
    def __call__(self,
        *args,
        **kwargs
    ) -> pd.DataFrame:

        return self.project(*args, **kwargs)





    ##############################################
    #    SUPPORT AND INITIALIZATION FUNCTIONS    #
    ##############################################

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



    def check_df_fields(self, 
        df_ippu_trajectories
    ) -> None:
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_ippu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_ippu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"IPPU projection cannot proceed: The fields {set_missing} are missing.")

        return None



    def _initialize_fc_emission_factor_modvars(self,
    ) -> None:
        """
        Initialize dictionary of FC emission factor model variables by type 
            (hfc, other_fc, and pfc). Sets the following properties:

            * self.all_fcs
            * self.dict_fc_ef_modvars_by_type
            * self.dict_fc_ef_modvars_to_gas
            * self.dict_gas_to_fc_ef_modvars
        """
        # get dictionaries mapping variables to gas
        subsec = self.model_attributes.subsec_name_ippu
        vec_modvars_ef_ippu = self.model_attributes.get_variables_from_attribute(
            subsec, 
            {"emission_factor": 1},
        )

        dict_fc_ef_modvars_to_gas = pd.DataFrame(
            [
                (x, self.model_attributes.get_variable_characteristic(x, self.model_attributes.varchar_str_emission_gas))
                for x in vec_modvars_ef_ippu
            ]
        )

        dict_fc_ef_modvars_to_gas = sf.build_dict(dict_fc_ef_modvars_to_gas)
        dict_gas_to_fc_ef_modvars = sf.reverse_dict(
            dict_fc_ef_modvars_to_gas, 
            allow_multi_keys = True, 
            force_list_values = True,
        )

        # all fluorinated compounds
        all_fcs = sorted(list(dict_gas_to_fc_ef_modvars.keys()))

        # get IPPU emission factor model variables by gas classification
        dict_fc_ef_modvars_by_type = {}
        
        for key, val in self.model_attributes.dict_fc_designation_to_gasses.items():
            modvars = sum(
                [
                    dict_gas_to_fc_ef_modvars.get(x) for x in val
                    if (dict_gas_to_fc_ef_modvars.get(x) is not None)
                ], []
            )
            
            dict_fc_ef_modvars_by_type.update({key: modvars})

        
        ##  SET PROPERTIES

        self.all_fcs = all_fcs
        self.dict_fc_ef_modvars_by_type = dict_fc_ef_modvars_by_type
        self.dict_fc_ef_modvars_to_gas = dict_fc_ef_modvars_to_gas
        self.dict_gas_to_fc_ef_modvars = dict_gas_to_fc_ef_modvars
        
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
        # initialzie dynamic variables
        
        # required dimensions of analysis
        required_doa = [self.model_attributes.dim_time_period]

        # required subsectors
        subsectors = self.model_attributes.get_sector_subsectors("IPPU")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]
        
        # input/output
        required_vars, output_vars = self.model_attributes.get_input_output_fields(subsectors)
        required_vars += required_doa

        # set output properties
        self.required_dimensions = required_doa
        self.required_subsectors = subsectors
        self.required_base_subsectors = subsectors_base
        self.required_variables = required_vars
        self.output_variables = output_vars

        return None


    
    def _initialize_integrated_variables(self,
    ) -> None:
        """
        Set the following properties:
        
            * self.integration_variables
        """
        list_vars_required_for_integration = [
            self.modvar_waso_waste_total_recycled
        ]

        self.integration_variables = list_vars_required_for_integration

        return None



    def _initialize_models(self,
        model_attributes: Union[ModelAttributes, None] = None
    ) -> None:
        """
        Initialize SISEPUEDE model classes for fetching variables and 
            accessing methods. Initializes the following properties:

            * self.model_socioeconomic

            as well as associated model variables from other sectors, such as 

            * self.modvar_waso_waste_total_recycled


        Keyword Arguments
        -----------------
        - model_attributes: ModelAttributes object used to instantiate
            models. If None, defaults to self.model_attributes.
        """

        model_attributes = self.model_attributes if (model_attributes is None) else model_attributes

        # add other model classes--required for integration variables
        self.model_socioeconomic = Socioeconomic(model_attributes)
        self.modvar_waso_waste_total_recycled = "Total Waste Recycled"

        return None



    def _initialize_other_properties(self,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.n_time_periods
            * self.time_periods
        """

        ##  TIME VARIABLES
        time_periods, n_time_periods = self.model_attributes.get_time_periods()
        self.time_periods = time_periods
        self.n_time_periods = n_time_periods

        return None



    def _initialize_subsector_names(self,
    ) -> None:
        """
        Initialize all subsector names (self.subsec_name_****)
        """
        # some subector reference variables
        self.subsec_name_agrc = "Agriculture"
        self.subsec_name_econ = "Economy"
        self.subsec_name_enfu = "Energy Fuels"
        self.subsec_name_frst = "Forest"
        self.subsec_name_gnrl = "General"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_lndu = "Land Use"
        self.subsec_name_lsmm = "Livestock Manure Management"
        self.subsec_name_lvst = "Livestock"
        self.subsec_name_scoe = "Stationary Combustion and Other Energy"
        self.subsec_name_soil = "Soil Management"
        self.subsec_name_waso = "Solid Waste"

        return None



    def _initialize_subsector_vars_ippu(self,
    ) -> None:
        """
        Initialize model variables, categories, and indicies associated with
            IPPU (Industrial P). Sets the following properties:

            * self.cat_ippu_****
            * self.ind_ippu_****
            * self.modvar_ippu_****
        """

        ##  NON-EMISSION VARIABLES
        
        self.modvar_ippu_average_construction_materials_required_per_household = "Average per Household Demand for Construction Materials"
        self.modvar_ippu_average_lifespan_housing = "Average Lifespan of Housing Construction"
        self.modvar_ippu_capture_efficacy_co2 = "Industrial :math:\\text{CO}_2 Capture Efficacy"
        self.modvar_ippu_capture_prevalence_co2 = "Industrial :math:\\text{CO}_2 Capture Prevalence"
        self.modvar_ippu_change_net_imports = "Change to Net Imports of Products"
        self.modvar_ippu_clinker_fraction_cement = "Clinker Fraction of Cement"
        self.modvar_ippu_demand_for_harvested_wood = "Demand for Harvested Wood"
        self.modvar_ippu_elast_ind_prod_to_gdp = "Elasticity of Industrial Production to GDP"
        self.modvar_ippu_elast_produserate_to_gdppc = "Elasticity of Product Use Rate to GDP per Capita"
        self.modvar_ippu_gas_captured_co2 = ":math:\\text{CO}_2 Captured in Industrial Processes and Product Use"
        self.modvar_ippu_max_recycled_material_ratio = "Maximum Recycled Material Ratio in Virgin Process"
        self.modvar_ippu_net_imports_clinker = "Net Imports of Cement Clinker"
        self.modvar_ippu_prod_qty_init = "Initial Industrial Production"
        self.modvar_ippu_qty_total_production = "Industrial Production"
        self.modvar_ippu_qty_recycled_used_in_production = "Recycled Material Used in Industrial Production"
        self.modvar_ippu_ratio_of_production_to_harvested_wood = "Ratio of Production to Harvested Wood Demand"
        self.modvar_ippu_scalar_production = "Industrial Production Scalar"
        self.modvar_ippu_useinit_nonenergy_fuel = "Initial Non-Energy Fuel Use"
        self.modvar_ippu_wwf_cod = "COD Wastewater Factor"
        self.modvar_ippu_wwf_vol = "Wastewater Production Factor"


        ##  EMISSION FACTOR VARIABLES (CO2, CH4, N2O, HFCs/PFCs/Other FCs)

        self.modvar_ippu_ef_ch4_per_prod_process = ":math:\\text{CH}_4 Production Process Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_process = ":math:\\text{CO}_2 Production Process Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_produse = ":math:\\text{CO}_2 Product Use Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_process_clinker = ":math:\\text{CO}_2 Clinker Production Process Emission Factor"
        self.modvar_ippu_ef_dodecafluoropentane_per_prod_process = "Dodecafluoropentane Production Process Emission Factor"
        self.modvar_ippu_ef_hcfc141b_per_gdp_produse = "HCFC-141b GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hcfc142b_per_gdp_produse = "HCFC-142b GDP Product Use Emission Factor"
        self.modvar_ippu_ef_n2o_per_gdp_process = ":math:\\text{N}_2\\text{O} GDP Production Process Emission Factor"
        self.modvar_ippu_ef_n2o_per_prod_process = ":math:\\text{N}_2\\text{O} Production Process Emission Factor"
        self.modvar_ippu_ef_nf3_per_prod_process = ":math:\\text{NF}_3 Production Process Emission Factor"
        self.modvar_ippu_ef_octafluoro_per_prod_process = "Octafluorooxolane Production Process Emission Factor"
        self.modvar_ippu_ef_sf6_per_gdp_process = ":math:\\text{SF}_6 GDP Production Process Emission Factor"
        self.modvar_ippu_ef_sf6_per_prod_process = ":math:\\text{SF}_6 Production Process Emission Factor"
        # HFCs
        self.modvar_ippu_ef_hfc23_per_gdp_produse = "HFC-23 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc23_per_prod_process = "HFC-23 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc32_per_gdp_produse = "HFC-32 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc32_per_prod_process = "HFC-32 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc41_per_prod_process = "HFC-41 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc125_per_gdp_produse = "HFC-125 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc125_per_prod_process = "HFC-125 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc134_per_gdp_produse = "HFC-134 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc134a_per_gdp_produse = "HFC-134a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc134a_per_prod_process = "HFC-134a Production Process Emission Factor"
        self.modvar_ippu_ef_hfc143_per_gdp_produse = "HFC-143 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc143a_per_gdp_produse = "HFC-143a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc143a_per_prod_process = "HFC-143a Production Process Emission Factor"
        self.modvar_ippu_ef_hfc152a_per_gdp_produse = "HFC-152a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc152a_per_prod_process = "HFC-152a Production Process Emission Factor"
        self.modvar_ippu_ef_hfc227ea_per_gdp_produse = "HFC-227ea GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc227ea_per_prod_process = "HFC-227ea Production Process Emission Factor"
        self.modvar_ippu_ef_hfc236fa_per_gdp_produse = "HFC-236fa GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc245fa_per_gdp_produse = "HFC-245fa GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc365mfc_per_gdp_produse = "HFC-365mfc GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc365mfc_per_prod_process = "HFC-365mfc Production Process Emission Factor"
        self.modvar_ippu_ef_hfc4310mee_per_gdp_produse = "HFC-43-10mee GDP Product Use Emission Factor"
        # PFCs
        self.modvar_ippu_ef_pfc14_per_gdp_produse = "PFC-14 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc14_per_prod_process = "PFC-14 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc116_per_gdp_produse = "PFC-116 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc116_per_prod_process = "PFC-116 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc218_per_prod_process = "PFC-218 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc1114_per_prod_process = "PFC-1114 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc3110_per_gdp_produse = "PFC-31-10 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc3110_per_prod_process = "PFC-31-10 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc5114_per_gdp_produse = "PFC-51-14 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc5114_per_prod_process = "PFC-51-14 Production Process Emission Factor"
        self.modvar_ippu_ef_pfcc318_per_prod_process = "PFC-C-318 Production Process Emission Factor"
        self.modvar_ippu_ef_pfcc1418_per_prod_process = "PFC-C-1418 Production Process Emission Factor"


        ##  EMISSION VARIABLES

        self.modvar_ippu_emissions_other_nonenergy_co2 = "Initial Other Non-Energy :math:\\text{CO}_2 Emissions"
        self.modvar_ippu_emissions_process_ch4 = ":math:\\text{CH}_4 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_co2 = ":math:\\text{CO}_2 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_co2 = ":math:\\text{CO}_2 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_dodecafluoropentane = "Dodecafluoropentane Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hcfc141b = "HCFC-141b Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hcfc142b = "HCFC-142b Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc = "HFC Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc = "HFC Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_n2o = ":math:\\text{N}_2\\text{O} Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_nf3 = ":math:\\text{NF}_3 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_other_fcs = "Other Fluorinated Compound Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_other_fcs = "Other Fluorinated Compound Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_pfc = "PFC Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc = "PFC Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_sf6 = ":math:\\text{SF}_6 Emissions from Industrial Production Processes"
        # INDIVIDUAL HFCs
        self.modvar_ippu_emissions_process_hfc23 = "HFC-23 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc23 = "HFC-23 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc32 = "HFC-32 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc32 = "HFC-32 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc41 = "HFC-41 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_hfc125 = "HFC-125 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc125 = "HFC-125 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hfc134 = "HFC-134 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc134a = "HFC-134a Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc134a = "HFC-134a Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hfc143 = "HFC-143 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc143a = "HFC-143a Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc143a = "HFC-143a Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc152a = "HFC-152a Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc152a = "HFC-152a Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc227ea = "HFC-227ea Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc227ea = "HFC-227ea Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hfc236fa = "HFC-236fa Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hfc245fa = "HFC-245fa Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc365mfc = "HFC-365mfc Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc365mfc = "HFC-365mfc Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_produse_hfc4310mee = "HFC-43-10mee Emissions from Industrial Product Use"
        # INDIVIDUAL PFCs
        self.modvar_ippu_emissions_process_pfc14 = "PFC-14 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc14 = "PFC-14 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_pfc116 = "PFC-116 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc116 = "PFC-116 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_pfc218 = "PFC-218 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_pfcc318 = "PFC-318 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_pfc1114 = "PFC-1114 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_pfcc1418 = "PFC-1418 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_pfc3110 = "PFC-31-10 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc3110 = "PFC-31-10 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_pfc5114 = "PFC-51-14 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc5114 = "PFC-51-14 Emissions from Industrial Product Use"

        return None





    ################################
    #    BASIC SHARED FUNCTIONS    #
    ################################

    #
    def calculate_emissions_by_gdp_and_production(self,
        df_ippu_trajectories: pd.DataFrame,
        array_production: np.ndarray,
        vec_gdp: np.ndarray,
        dict_base_emissions: Union[dict, None],
        dict_simple_efs: dict,
        include_carbon_capture: bool,
        modvar_carbon_capture_efficacy: Union[str, None] = None,
        modvar_carbon_capture_prevalence: Union[str, None] = None,
        modvar_carbon_capture_total: Union[str, None] = None,
        modvar_prod_mass: str = None,
    ) -> list:
        """
        Calculate emissions driven by GDP/Production and different factors. 
            Takes a production array (to drive production-based emissions), a 
            gdp vector, and dictionaries that contain (a) output variables as 
            keys and (b) lists of input gdp and/or production variables.

        NOTE: Returns aggregated carbon capture outputs as a single variable in 
            the output dataframe, so care should be taken if multiple calls to
            calculate_emissions_by_gdp_and_production() include carbon capture
            to aggregate across calls.

        Function Arguments
        ------------------
        - df_ippu_trajectories: pd.DataFrame with all required input variable 
            trajectories.
        - array_production: an array of production by industrial category
        - vec_gdp: vector of gdp
        - dict_base_emissions: dictionary of mapping an emission variable to an
            array to add 
        - dict_simple_efs: dict of the form 
            
                {
                    modvar_emission_out: (
                        [modvar_factor_gdp_1, ...], 
                        [modvar_factor_production_1, ... ]
                    )
                }
                
            Allows for multiple gasses to be summed over.
        - include_carbon_capture: bool denoting whether to include carbon 
            capture specification. For example, set to false when running with
            product use emission factors

        Keyword Arguments
        -----------------
        - modvar_carbon_capture_efficacy: model variable signifying carbon 
            capture efficacy (fraction of CO2 removed given CCS installed) for 
            industrial processes
        - modvar_carbon_capture_prevalence: model variable signifying fraction
            of industrial production subject to CCS
        - modvar_carbon_capture_total: model variable to use for total carbon
            captured as masss
        - modvar_prod_mass: variable with mass of production denoted in 
            array_production; used to match emission factors

        """

        # get the attribute table and model variables
        attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)
        modvar_carbon_capture_efficacy = (
            self.modvar_ippu_capture_efficacy_co2
            if (modvar_carbon_capture_efficacy is None) 
            else modvar_carbon_capture_efficacy
        )
        modvar_carbon_capture_prevalence = (
            self.modvar_ippu_capture_prevalence_co2
            if (modvar_carbon_capture_prevalence is None) 
            else modvar_carbon_capture_prevalence
        )
        modvar_carbon_capture_total = (
            self.modvar_ippu_gas_captured_co2
            if (modvar_carbon_capture_total is None) 
            else modvar_carbon_capture_total
        )
        modvar_prod_mass = (
            self.modvar_ippu_qty_total_production 
            if (modvar_prod_mass is None) 
            else modvar_prod_mass
        )

        # initialize cross-iteration variables
        array_emission_captured = 0.0
        df_out = []

        # process is identical across emission factors -- sum gdp-driven and production-driven factors
        for modvar in dict_simple_efs.keys():
            
            # check subsector and skip if invalid
            subsec = self.model_attributes.get_variable_subsector(
                modvar, 
                throw_error_q = False
            )
            if subsec is None:
                continue

            # get variables and initialize total emissions
            all_modvar_ef_gdp = dict_simple_efs[modvar][0]
            all_modvar_ef_prod = dict_simple_efs[modvar][1]
            array_emission = np.zeros((len(df_ippu_trajectories), attr_ippu.n_key_values))
            gas = self.model_attributes.get_variable_characteristic(
                modvar, 
                self.model_attributes.varchar_str_emission_gas
            )

            # check if there are gdp driven factors
            if isinstance(vec_gdp, np.ndarray):
                for modvar_ef_gdp in all_modvar_ef_gdp:
                    array_emission_cur = self.model_attributes.extract_model_variable(#
                        df_ippu_trajectories, 
                        modvar_ef_gdp, 
                        expand_to_all_cats = True,
                        return_type = "array_units_corrected",
                    )

                    array_emission += (
                        array_emission_cur*vec_gdp
                        if vec_gdp.shape == array_emission_cur.shape
                        else (array_emission_cur.transpose()*vec_gdp).transpose()
                    )


            # check if there is a production driven factor
            if isinstance(array_production, np.ndarray):
                for modvar_ef_prod in all_modvar_ef_prod:
                    scalar_ippu_mass = self.model_attributes.get_variable_unit_conversion_factor(
                        modvar_ef_prod,
                        modvar_prod_mass,
                        "mass",
                    )

                    array_emission_cur = self.model_attributes.extract_model_variable(#
                        df_ippu_trajectories, 
                        modvar_ef_prod, 
                        expand_to_all_cats = True,
                        return_type = "array_units_corrected",
                    )

                    array_emission += array_emission_cur*array_production/scalar_ippu_mass

            # add any baseline emissions from elsewhere
            array_emission += (
                dict_base_emissions.get(modvar, 0.0)
                if dict_base_emissions is not None
                else 0.0
            )
            

            # check if carbon capture should be incorporated
            if include_carbon_capture & (gas == "co2"):
                
                # fraction captured is prevalence * efficacy
                array_emission_frac_captured = self.model_attributes.extract_model_variable(#
                    df_ippu_trajectories, 
                    modvar_carbon_capture_prevalence, 
                    expand_to_all_cats = True,
                    return_type = "array_base",
                    var_bounds = (0, 1),
                )

                array_emission_frac_captured *= self.model_attributes.extract_model_variable(#
                    df_ippu_trajectories, 
                    modvar_carbon_capture_efficacy, 
                    expand_to_all_cats = True,
                    return_type = "array_base",
                    var_bounds = (0, 1),
                )

                array_emission_captured_cur = array_emission*array_emission_frac_captured
                array_emission_captured += array_emission_captured_cur
                array_emission -= array_emission_captured_cur


            # add to output dataframe if it's a valid model variable
            df_out += [
                self.model_attributes.array_to_df(
                    array_emission, 
                    modvar, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ]

        # add on carbon captured if spcified
        if include_carbon_capture & isinstance(array_emission_captured, np.ndarray):

            scalar_captured = self.model_attributes.get_scalar(modvar_carbon_capture_total, "mass")
            array_emission_captured /= scalar_captured

            df_out += [
                self.model_attributes.array_to_df(
                    array_emission_captured, 
                    modvar_carbon_capture_total, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ]
            
        return df_out



    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################

    def project_hh_construction(self,
        vec_hh: np.ndarray,
        vec_average_lifetime_hh: np.ndarray
    ) -> np.ndarray:
        """
        Project the number of households constructed based on the number of 
            households and the average lifetime of households.

        Function Arguments
        ------------------
        - vec_hh: vector of housholds by time period
        - vec_average_lifetime_hh: vector of average household lifetimes.
        """

        if len(vec_average_lifetime_hh) != len(vec_hh):
            self._log(
                f"Warning in `IPPU.project_hh_construction()`: average lifetime of housholds and number of households should have the same length vectors. Setting lifetime to repeat of final value.",
                type_log = "warning"
            )
            vec_average_lifetime_hh = np.conactenate([vec_average_lifetime_hh, np.array([vec_average_lifetime_hh[-1] for x in range(len(vec_hh) - len(vec_average_lifetime_hh))])])

        n_projection_time_periods = len(vec_hh)

        # get estimates for new housing stock -- last year, use trend
        vec_new_housing_stock_changes = sf.vec_bounds(vec_hh[1:] - vec_hh[0:-1], (0, np.inf))
        new_stock_final_period = np.nan_to_num(
            np.round(vec_new_housing_stock_changes[-1]**2/vec_new_housing_stock_changes[-2]), 
            0.0
        )

        vec_new_housing_stock_changes = np.insert(
            vec_new_housing_stock_changes, 
            len(vec_new_housing_stock_changes), 
            new_stock_final_period
        )

        # back-project to estimate replacement construction
        scalar_gr_hh = np.mean((vec_hh[1:]/vec_hh[0:-1])[0:3])
        vec_old_housing_stock_rev = np.round(vec_hh[0]*scalar_gr_hh**(-np.arange(1, 100 + 1)))
        vec_est_new_builds = np.zeros(n_projection_time_periods)

        for i in range(n_projection_time_periods):
            ind_lifetime_cur_stock = int(max(0, i - vec_average_lifetime_hh[0] + 1))
            ind_lifetime_old_stock = int(vec_average_lifetime_hh[0] - i - 1)
            if ind_lifetime_old_stock >= 0:
                old_stock = vec_old_housing_stock_rev[ind_lifetime_old_stock] if (ind_lifetime_old_stock < len(vec_old_housing_stock_rev)) else 0
                old_stock_refreshed = np.round(old_stock/vec_average_lifetime_hh[0])
            else:
                old_stock_refreshed = np.round(vec_hh[ind_lifetime_old_stock]/vec_average_lifetime_hh[ind_lifetime_cur_stock])

            vec_est_new_builds[i] = old_stock_refreshed + vec_new_housing_stock_changes[i]

        return vec_est_new_builds



    def project_industrial_production(self,
        df_ippu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None,
        modvar_average_lifespan_housing: str = None,
        modvar_elast_ind_prod_to_gdp: str = None,
        modvar_num_hh: str = None,
        modvar_prod_qty_init: str = None,
        modvar_scalar_prod: str = None
    ) -> np.ndarray:
        """
        Project industrial production. Called from other sectors to simplify 
            calculation of industrial production. Includes swap of demand for 
            cement product and wood products in new housing construction.

        Function Arguments
        ------------------
        - df_ippu_trajectories: pd.DataFrame of input variable trajectories.
        - vec_rates_gdp: vector of rates of change to gdp 
            (length = len(df_ippu_trajectories) - 1)

        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.
        - modvar_average_lifespan_housing: average lifespan of housing
        - modvar_elast_ind_prod_to_gdp: model variable giving elasticity of 
            production to gdp
        - modvar_num_hh: model variable giving the number of households
        - modvar_prod_qty_init: model variable giving initial production 
            quantity
        - modvar_scalar_prod: model variable with the production scalar

        Notes
        -----
        - If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            IPPU.project()), self.model_attributes.check_projection_input_df 
            will be run
        """
        # allows production to be run outside of the project method
        if any([(x is None) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            (
                dict_dims, 
                df_ippu_trajectories, 
                n_projection_time_periods,
                projection_time_periods
            ) = self.model_attributes.check_projection_input_df(
                df_ippu_trajectories, 
                True, 
                True, 
                True
            )

        # set defaults
        modvar_average_lifespan_housing = (
            self.modvar_ippu_average_lifespan_housing 
            if (modvar_average_lifespan_housing is None) 
            else modvar_average_lifespan_housing
        )
        modvar_elast_ind_prod_to_gdp = (
            self.modvar_ippu_elast_ind_prod_to_gdp 
            if (modvar_elast_ind_prod_to_gdp is None) 
            else modvar_elast_ind_prod_to_gdp
        )
        modvar_num_hh = (
            self.model_socioeconomic.modvar_grnl_num_hh 
            if (modvar_num_hh is None) 
            else modvar_num_hh
        )
        modvar_prod_qty_init = (
            self.modvar_ippu_prod_qty_init 
            if (modvar_prod_qty_init is None) 
            else modvar_prod_qty_init
        )
        modvar_scalar_prod = (
            self.modvar_ippu_scalar_production 
            if (modvar_scalar_prod is None) 
            else modvar_scalar_prod
        )

        # get initial production and apply elasticities to gdp to calculate growth in production
        array_ippu_prod_init_by_cat = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            modvar_prod_qty_init, 
            expand_to_all_cats = True, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        array_ippu_elasticity_prod_to_gdp = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            modvar_elast_ind_prod_to_gdp,
            expand_to_all_cats = True, 
            return_type = "array_base",
        )

        array_ippu_ind_growth = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp, 
            array_ippu_elasticity_prod_to_gdp
        )
        array_ippu_ind_prod = array_ippu_prod_init_by_cat[0]*array_ippu_ind_growth

        # set exogenous scaling of production
        array_prod_scalar = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            modvar_scalar_prod, 
            expand_to_all_cats = True, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )
        array_ippu_ind_prod *= array_prod_scalar

        # adjust housing construction
        vec_hh = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_grnl_num_hh, 
            return_type = "array_base",
        )

        vec_ippu_average_lifetime_hh = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_average_lifespan_housing, 
            return_type = "array_base",
        )

        vec_ippu_housing_construction = self.project_hh_construction(
            vec_hh, 
            vec_ippu_average_lifetime_hh,
        )

        # get average materials required, then project forward a "bau" approach (calculated using material reqs at t = 0)
        arr_ippu_materials_required = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_average_construction_materials_required_per_household, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        arr_ippu_materials_required *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_average_construction_materials_required_per_household,
            self.modvar_ippu_prod_qty_init,
            "mass",
        )

        arr_ippu_materials_required_baseline = np.outer(vec_ippu_housing_construction, arr_ippu_materials_required[0])
        arr_ippu_materials_required = (arr_ippu_materials_required.transpose()*vec_ippu_housing_construction).transpose()
        arr_ippu_materials_required_change = arr_ippu_materials_required - arr_ippu_materials_required_baseline
        
        # adjust production and net imports
        array_ippu_ind_balance = array_ippu_ind_prod + arr_ippu_materials_required_change
        array_ippu_ind_prod = sf.vec_bounds(array_ippu_ind_balance, (0, np.inf))
        array_ippu_change_to_net_imports_cur = array_ippu_ind_balance - array_ippu_ind_prod

        return array_ippu_ind_prod, array_ippu_change_to_net_imports_cur



    def get_production_with_recycling_adjustment(self,
        df_ippu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None,
        modvar_average_lifespan_housing: str = None,
        modvar_change_net_imports: str = None,
        modvar_demand_for_harvested_wood: str = None,
        modvar_elast_ind_prod_to_gdp: str = None,
        modvar_max_recycled_material_ratio: str = None,
        modvar_num_hh: str = None,
        modvar_prod_qty_init: str = None,
        modvar_qty_recycled_used_in_production: str = None,
        modvar_qty_total_production: str = None,
        modvar_ratio_of_production_to_harvested_wood: str = None,
        modvar_scalar_prod: str = None,
        modvar_waste_total_recycled: str = None
    ) -> tuple:
        """
        Retrieve production and perform the adjustment to virgin production due 
            to recycling changes from the CircularEconomy model.

        Function Arguments
        ------------------
        - df_ippu_trajectories: pd.DataFrame of input variable trajectories.
        - vec_rates_gdp: vector of rates of change of gdp. Entry at index t is 
            the change from time t-1 to t 
            (length = len(df_ippu_trajectories) - 1)
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Keyword Arguments
        -----------------
        - modvar_average_lifespan_housing: average lifetime of housing
        - modvar_change_net_imports: change to net imports
        - modvar_demand_for_harvested_wood: final demand for harvested wood
        - modvar_prod_qty_init: initial production quantity
        - modvar_elast_ind_prod_to_gdp: elasticity of production to gdp
        - modvar_max_recycled_material_ratio: maximum fraction of virgin 
            production that can be replaced by recylables (e.g., cullet in glass 
            production)
        - modvar_num_hh: number of households
        - modvar_qty_total_production: total industrial production
        - modvar_scalar_prod: scalar applied to future production--used to 
            change economic mix
        - modvar_ratio_of_production_to_harvested_wood: ratio of production 
            output to input wood
        - modvar_waste_total_recycled: total waste recycled (from 
            CircularEconomy)

        Notes
        -----
        - If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            IPPU.project()), self.model_attributes.check_projection_input_df 
            will be run
        """

        ##  GET DEFAULT VARIABLES

        modvar_average_lifespan_housing = (
            self.modvar_ippu_average_lifespan_housing 
            if (modvar_average_lifespan_housing is None) 
            else modvar_average_lifespan_housing
        )

        modvar_change_net_imports = (
            self.modvar_ippu_change_net_imports 
            if (modvar_change_net_imports is None) 
            else modvar_change_net_imports
        )

        modvar_demand_for_harvested_wood = (
            self.modvar_ippu_demand_for_harvested_wood 
            if (modvar_demand_for_harvested_wood is None) 
            else modvar_demand_for_harvested_wood
        )

        modvar_elast_ind_prod_to_gdp = (
            self.modvar_ippu_elast_ind_prod_to_gdp 
            if (modvar_elast_ind_prod_to_gdp is None) 
            else modvar_elast_ind_prod_to_gdp
        )

        modvar_max_recycled_material_ratio = (
            self.modvar_ippu_max_recycled_material_ratio 
            if (modvar_max_recycled_material_ratio is None) 
            else modvar_max_recycled_material_ratio
        )

        modvar_num_hh = (
            self.model_socioeconomic.modvar_grnl_num_hh 
            if (modvar_num_hh is None) 
            else modvar_num_hh
        )

        modvar_prod_qty_init = (
            self.modvar_ippu_prod_qty_init 
            if (modvar_prod_qty_init is None) 
            else modvar_prod_qty_init
        )

        modvar_qty_recycled_used_in_production = (
            self.modvar_ippu_qty_recycled_used_in_production 
            if (modvar_qty_recycled_used_in_production is None) 
            else modvar_qty_recycled_used_in_production
        )

        modvar_qty_total_production = (
            self.modvar_ippu_qty_total_production 
            if (modvar_qty_total_production is None) 
            else modvar_qty_total_production
        )

        modvar_ratio_of_production_to_harvested_wood = (
            self.modvar_ippu_ratio_of_production_to_harvested_wood 
            if (modvar_ratio_of_production_to_harvested_wood is None) 
            else modvar_ratio_of_production_to_harvested_wood
        )

        modvar_waste_total_recycled = (
            self.modvar_waso_waste_total_recycled 
            if (modvar_waste_total_recycled is None) 
            else modvar_waste_total_recycled
        )



        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        # get some attribute info
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute(self.subsec_name_waso, "pycategory_primary")
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]
        attr_waso = self.model_attributes.dict_attributes[pycat_waso]

        # get recycling
        array_ippu_recycled = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_ippu_trajectories,
            modvar_waste_total_recycled,
            None,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        # initialize production + initialize change to net imports as 0 (reduce categories later)
        array_ippu_production, array_ippu_change_net_imports = self.project_industrial_production(
            df_ippu_trajectories,
            vec_rates_gdp,
            dict_dims,
            n_projection_time_periods,
            projection_time_periods,
            modvar_average_lifespan_housing,
            modvar_elast_ind_prod_to_gdp,
            modvar_num_hh,
            modvar_prod_qty_init,
            modvar_scalar_prod
        )

        # perform adjustments to production if recycling is denoted
        if array_ippu_recycled is not None:

            # if recycling totals are passed from the waste model, convert to ippu categories
            cats_waso_recycle = self.model_attributes.get_variable_categories(modvar_waste_total_recycled)
            dict_repl = attr_waso.field_maps[f"{pycat_waso}_to_{pycat_ippu}"]
            cats_ippu_recycle = [clean_schema(dict_repl[x]) for x in cats_waso_recycle]

            array_ippu_recycled_waste = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                array_ippu_recycled[1],
                None,
                output_cats = cats_ippu_recycle,
                output_subsec = self.subsec_name_ippu,
            )

            # units correction to ensure consistency from waso -> ippu
            factor_ippu_waso_recycle_to_ippu_recycle = self.model_attributes.get_variable_unit_conversion_factor(
                modvar_waste_total_recycled,
                modvar_prod_qty_init,
                "mass"
            )

            array_ippu_recycled_waste *= factor_ippu_waso_recycle_to_ippu_recycle
            array_ippu_production += array_ippu_recycled_waste

            # next, check for industrial categories whose production is affected by recycling, then adjust downwards
            cats_ippu_to_recycle_ordered = self.model_attributes.get_ordered_category_attribute(
                self.subsec_name_ippu, 
                "target_cat_industry_to_adjust_with_recycling",
            )
            vec_ippu_cats_to_adjust_from_recycling = [clean_schema(x) for x in cats_ippu_to_recycle_ordered]

            # get indexes of of valid categories specified for recycling adjustments
            w = [
                i for i in range(len(vec_ippu_cats_to_adjust_from_recycling)) 
                if (
                    (vec_ippu_cats_to_adjust_from_recycling[i] != "none") and (vec_ippu_cats_to_adjust_from_recycling[i] in attr_ippu.key_values)
                )
            ]

            if len(w) > 0:
                # maximum proportion of virgin production (e.g., fraction of glass that is cullet) 
                # that can be replaced by recycled materials--if not specifed, default to 1
                array_ippu_maxiumum_recycling_ratio = self.model_attributes.extract_model_variable(#
                    df_ippu_trajectories,
                    self.modvar_ippu_max_recycled_material_ratio,
                    all_cats_missing_val = 1.0,
                    expand_to_all_cats = True,
                    return_type = "array_base",
                    var_bounds = (0, 1),
                )

                array_ippu_recycled_waste_adj = array_ippu_recycled_waste[:, w].copy()
                array_ippu_recycled_waste_adj = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                    array_ippu_recycled_waste_adj,
                    None,
                    output_cats = np.array(vec_ippu_cats_to_adjust_from_recycling)[w],
                    output_subsec = self.subsec_name_ippu
                )
                # inititialize production, then get change to net imports (anything negative) and reduce virgin production accordingly
                array_ippu_production_base = array_ippu_production*(1 - array_ippu_maxiumum_recycling_ratio)
                array_ippu_production = array_ippu_production*array_ippu_maxiumum_recycling_ratio - array_ippu_recycled_waste_adj
                # array of changes to net imports has to be mapped back to the original recycling categories
                array_ippu_change_net_imports = sf.vec_bounds(array_ippu_production, (-np.inf, 0))
                array_ippu_change_net_imports = self.model_attributes.swap_array_categories(
                    array_ippu_change_net_imports,
                    np.array(vec_ippu_cats_to_adjust_from_recycling)[w],
                    np.array(attr_ippu.key_values)[w],
                    self.subsec_name_ippu
                )
                array_ippu_production = sf.vec_bounds(array_ippu_production, (0, np.inf)) + array_ippu_production_base
                array_ippu_production += array_ippu_change_net_imports


        # ensure net imports are in the proper mass units
        array_ippu_change_net_imports *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_prod_qty_init,
            modvar_change_net_imports,
            "mass"
        )
        # get production in terms of output variable (should be 1, and add net imports and production to output dataframe)
        array_ippu_production *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_prod_qty_init,
            modvar_qty_total_production,
            "mass"
        )

        ##  finally, get wood harvested equivalent for AFOLU
        arr_ippu_ratio_of_production_to_wood_harvesting = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            modvar_ratio_of_production_to_harvested_wood, 
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        arr_ippu_harvested_wood = np.nan_to_num(array_ippu_production/arr_ippu_ratio_of_production_to_wood_harvesting, 0.0, posinf = 0.0)
        arr_ippu_harvested_wood *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_prod_qty_init,
            modvar_demand_for_harvested_wood,
            "mass",
        )


        ##  ADD TO OUTPUT DATA FRAME

        df_out = [
            # CHANGE TO NET IMPORTS
            self.model_attributes.array_to_df(
                array_ippu_change_net_imports, 
                modvar_change_net_imports, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL MASS OF HARVESTED WOOD PRODUCTS
            self.model_attributes.array_to_df(
                arr_ippu_harvested_wood, 
                modvar_demand_for_harvested_wood, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL PRODUCTION
            self.model_attributes.array_to_df(
                array_ippu_production, 
                modvar_qty_total_production, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # RECYCLED MATERIALS USED IN PRODUCTION
            self.model_attributes.array_to_df(
                array_ippu_production, 
                modvar_qty_recycled_used_in_production, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]

        return array_ippu_production, df_out



    def project(self, 
        df_ippu_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        project() takes a data frame of input variables (ordered by time series) 
            and returns a data frame of output variables (model projections for 
            industrial processes and product use--excludes industrial energy 
            (see Energy class)) the same order.

        Function Arguments
        ------------------
        - df_ippu_trajectories: pd.DataFrame with all required input variable 
            trajectories.

        Notes
        -----
        - df_ippu_trajectories should have all input fields required (see 
            IPPU.required_variables for a list of variables to be defined). The 
            model will not run if any required variables are missing, but errors 
            will detail which fields are missing.
        - the df_ippu_trajectories.project method will run on valid time periods 
            from 1 .. k, where k <= n (n is the number of time periods). By 
            default, it drops invalid time periods. If there are missing 
            time_periods between the first and maximum, data are interpolated.
        """

        # make sure socioeconomic variables are added and
        df_ippu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ippu_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ippu_trajectories)
        dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_gnrl = self.model_attributes.get_subsector_attribute(self.subsec_name_gnrl, "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute(self.subsec_name_waso, "pycategory_primary")
        # attribute tables
        attr_gnrl = self.model_attributes.dict_attributes[pycat_gnrl]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]
        attr_waso = self.model_attributes.dict_attributes[pycat_waso]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        array_pop = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_gnrl_subpop, 
            return_type = "array_base",
        )

        vec_gdp = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp, 
            return_type = "array_base",
        )

        vec_gdp_per_capita = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp_per_capita, 
            return_type = "array_base",
        )

        vec_hh = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_grnl_num_hh, 
            return_type = "array_base",
        )

        vec_pop = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.model_socioeconomic.modvar_gnrl_pop_total, 
            return_type = "array_base",
        )

        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())

        ##  OUTPUT INITIALIZATION
        df_out = [df_ippu_trajectories[self.required_dimensions].copy()]


        ######################################################
        #    INDUSTRIAL PRODUCTION + RECYCLING ADJUSTMENT    #
        ######################################################

        ##  PERFORM THE RECYCLING ADJUSTMENT (if recycling data are provided from the waste model)
        array_ippu_production = self.get_production_with_recycling_adjustment(df_ippu_trajectories, vec_rates_gdp)
        df_out += array_ippu_production[1]
        array_ippu_production = array_ippu_production[0]


        ############################
        #    PRODUCTION PROCESS    #
        ############################

        ##  GET CEMENT CLINKER EMISSIONS BEFORE GENERALIZED APPROACH

        scalar_ippu_mass_clinker = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_ef_co2_per_prod_process_clinker,
            self.modvar_ippu_qty_total_production,
            "mass",
        )

        array_ippu_emissions_clinker = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_ef_co2_per_prod_process_clinker, 
            expand_to_all_cats = True,
            return_type = "array_units_corrected",
        )/scalar_ippu_mass_clinker
        
        # get net imports and convert to units of production
        array_ippu_net_imports_clinker = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_net_imports_clinker,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        array_ippu_net_imports_clinker *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_net_imports_clinker,
            self.modvar_ippu_qty_total_production,
            "mass",
        )

        # get production of clinker, remove net imports (and cap reduction to 0), and calculate emissions
        array_ippu_production_clinker = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_clinker_fraction_cement,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        array_ippu_production_clinker = sf.vec_bounds(
            array_ippu_production_clinker*array_ippu_production - array_ippu_net_imports_clinker, 
            (0, np.inf)
        )
        array_ippu_emissions_clinker *= array_ippu_production_clinker


        ##  GENERAL EMISSIONS

        # dictionary that contains some baseline emissions from secondary sources (e.g., cement clinker) -- will add to those calculated from dict_ippu_simple_efs
        dict_ippu_proc_emissions_to_add = {
            self.modvar_ippu_emissions_process_co2: array_ippu_emissions_clinker
        }

        # dictionary variables mapping emission variable to component tuples (gdp, production). No factor is []
        dict_ippu_proc_simple_efs = {
            self.modvar_ippu_emissions_process_ch4: (
                [],
                [
                    self.modvar_ippu_ef_ch4_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_co2: (
                [],
                [
                    self.modvar_ippu_ef_co2_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_hfc: (
                [],
                [
                    self.modvar_ippu_ef_hfc23_per_prod_process,
                    self.modvar_ippu_ef_hfc32_per_prod_process,
                    self.modvar_ippu_ef_hfc125_per_prod_process,
                    self.modvar_ippu_ef_hfc134a_per_prod_process,
                    self.modvar_ippu_ef_hfc41_per_prod_process,
                    self.modvar_ippu_ef_hfc143a_per_prod_process,
                    self.modvar_ippu_ef_hfc152a_per_prod_process,
                    self.modvar_ippu_ef_hfc227ea_per_prod_process,
                    self.modvar_ippu_ef_hfc365mfc_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_n2o: (
                [
                    self.modvar_ippu_ef_n2o_per_gdp_process
                ],
                [
                    self.modvar_ippu_ef_n2o_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_other_fcs: (
                [],
                [
                    self.modvar_ippu_ef_dodecafluoropentane_per_prod_process,
                    self.modvar_ippu_ef_nf3_per_prod_process,
                    self.modvar_ippu_ef_octafluoro_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_pfc: (
                [],
                [
                    self.modvar_ippu_ef_pfc14_per_prod_process,
                    self.modvar_ippu_ef_pfc116_per_prod_process,
                    self.modvar_ippu_ef_pfc218_per_prod_process,
                    self.modvar_ippu_ef_pfc1114_per_prod_process,
                    self.modvar_ippu_ef_pfcc318_per_prod_process,
                    self.modvar_ippu_ef_pfcc1418_per_prod_process,
                    self.modvar_ippu_ef_pfc3110_per_prod_process,
                    self.modvar_ippu_ef_pfc5114_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_sf6: (
                [
                    self.modvar_ippu_ef_sf6_per_gdp_process
                ],
                [
                    self.modvar_ippu_ef_sf6_per_prod_process
                ]
            )
        }

        # use dictionary to calculate emissions from processes
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            array_ippu_production,
            vec_gdp,
            dict_ippu_proc_emissions_to_add,
            dict_ippu_proc_simple_efs,
            True,
        )



        #####################
        #    PRODUCT USE    #
        #####################

        ##  PRODUCT USE FROM PARAFFIN WAX AND LUBRICANTS
        array_ippu_useinit_nonenergy_fuel = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_useinit_nonenergy_fuel,
            expand_to_all_cats = True, 
            return_type = "array_base",
        )

        array_ippu_pwl_growth = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp, 
            np.ones(len(array_ippu_useinit_nonenergy_fuel)), 
            False, 
            "standard",
        )

        array_ippu_emissions_produse_nonenergy_fuel = np.outer(
            array_ippu_pwl_growth, 
            array_ippu_useinit_nonenergy_fuel[0]
        )

        array_ippu_production_scalar = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_scalar_production,
            expand_to_all_cats = True, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )
        
        # get the emission factor and project emissions (unitless emissions)
        array_ippu_ef_co2_produse = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_ef_co2_per_prod_produse,
            expand_to_all_cats = True, 
            return_type = "array_base",
        )

        array_ippu_emissions_produse_nonenergy_fuel *= array_ippu_ef_co2_produse*self.model_attributes.get_scalar(self.modvar_ippu_useinit_nonenergy_fuel, "mass")
        array_ippu_emissions_produse_nonenergy_fuel *= array_ippu_production_scalar

        array_ippu_elasticity_produse = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_elast_produserate_to_gdppc,
            expand_to_all_cats = True, 
            return_type = "array_base",
        )

        array_ippu_gdp_scalar_produse = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp_per_capita, 
            array_ippu_elasticity_produse, 
            False, 
            "standard",
        )
        
        # this scalar array accounts for elasticity changes in per/gdp product use rates due to increases in gdp/capita, increases in gdp, and exogenously-defined reductions to production
        array_ippu_gdp_scalar_produse = (array_ippu_gdp_scalar_produse.transpose()*np.concatenate([np.ones(1), np.cumprod(1 + vec_rates_gdp)])).transpose()
        array_ippu_gdp_scalar_produse = array_ippu_gdp_scalar_produse * vec_gdp[0]
        array_ippu_gdp_scalar_produse *= array_ippu_production_scalar


        ##  OTHER EMISSIONS (very small--NMVOC, e.g.)

        array_ippu_emissions_other_nonenergy_co2 = self.model_attributes.extract_model_variable(#
            df_ippu_trajectories, 
            self.modvar_ippu_emissions_other_nonenergy_co2, 
            expand_to_all_cats = True, 
            return_type = "array_units_corrected",
        )

        array_ippu_emissions_other_nonenergy_co2 = array_ippu_emissions_other_nonenergy_co2[0]*sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp, 
            np.ones(array_ippu_emissions_other_nonenergy_co2.shape), 
            False, 
            "standard",
        )
        
        array_ippu_emissions_other_nonenergy_co2 *= array_ippu_production_scalar


        ##  OTHER PRODUCT USE

        dict_ippu_produse_emissions_to_add = {
            self.modvar_ippu_emissions_produse_co2: (
                array_ippu_emissions_produse_nonenergy_fuel + array_ippu_emissions_other_nonenergy_co2
            )
        }

        dict_ippu_produse_simple_efs = {
            self.modvar_ippu_emissions_produse_co2: (
                [],
                []
            ),
            self.modvar_ippu_emissions_produse_hfc: (
                [
                    self.modvar_ippu_ef_hfc23_per_gdp_produse,
                    self.modvar_ippu_ef_hfc32_per_gdp_produse,
                    self.modvar_ippu_ef_hfc125_per_gdp_produse,
                    self.modvar_ippu_ef_hfc134_per_gdp_produse,
                    self.modvar_ippu_ef_hfc134a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc143_per_gdp_produse,
                    self.modvar_ippu_ef_hfc143a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc152a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc227ea_per_gdp_produse,
                    self.modvar_ippu_ef_hfc236fa_per_gdp_produse,
                    self.modvar_ippu_ef_hfc245fa_per_gdp_produse,
                    self.modvar_ippu_ef_hfc365mfc_per_gdp_produse,
                    self.modvar_ippu_ef_hfc4310mee_per_gdp_produse
                ],
                []
            ),
            self.modvar_ippu_emissions_produse_other_fcs: (
                [
                    self.modvar_ippu_ef_hcfc141b_per_gdp_produse,
                    self.modvar_ippu_ef_hcfc142b_per_gdp_produse
                ],
                []
            ),
            self.modvar_ippu_emissions_produse_pfc: (
                [
                    self.modvar_ippu_ef_pfc14_per_gdp_produse,
                    self.modvar_ippu_ef_pfc116_per_gdp_produse,
                    self.modvar_ippu_ef_pfc3110_per_gdp_produse,
                    self.modvar_ippu_ef_pfc5114_per_gdp_produse
                ],
                []
            )
        }


        # use dictionary to calculate emissions from product use
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            0,
            array_ippu_gdp_scalar_produse,
            dict_ippu_produse_emissions_to_add,
            dict_ippu_produse_simple_efs,
            False,
        )


        ############################################################
        #    ADD IN CALIBRATION TARGET INDIVIDUAL HFCs AND PFCs    #
        ############################################################

        # F-gasses that can be calibrated to from processes
        dict_ippu_proc_simple_efs_indiv_fgas = {
            self.modvar_ippu_emissions_process_dodecafluoropentane: ([], [self.modvar_ippu_ef_dodecafluoropentane_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc23: ([], [self.modvar_ippu_ef_hfc23_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc32: ([], [self.modvar_ippu_ef_hfc32_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc41: ([], [self.modvar_ippu_ef_hfc41_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc125: ([], [self.modvar_ippu_ef_hfc125_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc134a: ([], [self.modvar_ippu_ef_hfc134a_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc143a: ([], [self.modvar_ippu_ef_hfc143a_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc152a: ([], [self.modvar_ippu_ef_hfc152a_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc227ea: ([], [self.modvar_ippu_ef_hfc227ea_per_prod_process]),
            self.modvar_ippu_emissions_process_hfc365mfc: ([], [self.modvar_ippu_ef_hfc365mfc_per_prod_process]),
            self.modvar_ippu_emissions_process_nf3: ([], [self.modvar_ippu_ef_nf3_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc14: ([], [self.modvar_ippu_ef_pfc14_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc116: ([], [self.modvar_ippu_ef_pfc116_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc218: ([], [self.modvar_ippu_ef_pfc218_per_prod_process]),
            self.modvar_ippu_emissions_process_pfcc318: ([], [self.modvar_ippu_ef_pfcc318_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc1114: ([], [self.modvar_ippu_ef_pfc1114_per_prod_process]),
            self.modvar_ippu_emissions_process_pfcc1418: ([], [self.modvar_ippu_ef_pfcc1418_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc3110: ([], [self.modvar_ippu_ef_pfc3110_per_prod_process]),
            self.modvar_ippu_emissions_process_pfc5114: ([], [self.modvar_ippu_ef_pfc5114_per_prod_process])
        }
        # get HFC and PFC emissions from production processes
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            array_ippu_production,
            vec_gdp,
            None,
            dict_ippu_proc_simple_efs_indiv_fgas,
            False,
        )

        # F-gasses that can be calibrated to from product use
        dict_ippu_produse_simple_efs_indiv_fgas = {
            self.modvar_ippu_emissions_produse_hcfc141b: ([], [self.modvar_ippu_ef_hcfc141b_per_gdp_produse]),
            self.modvar_ippu_emissions_produse_hcfc142b: ([], [self.modvar_ippu_ef_hcfc142b_per_gdp_produse]),
            self.modvar_ippu_emissions_produse_hfc23: ([self.modvar_ippu_ef_hfc23_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc32: ([self.modvar_ippu_ef_hfc32_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc125: ([self.modvar_ippu_ef_hfc125_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc134: ([], [self.modvar_ippu_ef_hfc134_per_gdp_produse]),
            self.modvar_ippu_emissions_produse_hfc134a: ([self.modvar_ippu_ef_hfc134a_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc143: ([], [self.modvar_ippu_ef_hfc143_per_gdp_produse]),
            self.modvar_ippu_emissions_produse_hfc143a: ([self.modvar_ippu_ef_hfc143a_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc152a: ([self.modvar_ippu_ef_hfc152a_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc227ea: ([self.modvar_ippu_ef_hfc227ea_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc236fa: ([self.modvar_ippu_ef_hfc236fa_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc245fa: ([self.modvar_ippu_ef_hfc245fa_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc365mfc: ([self.modvar_ippu_ef_hfc365mfc_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_hfc4310mee: ([self.modvar_ippu_ef_hfc4310mee_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_pfc14: ([self.modvar_ippu_ef_pfc14_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_pfc116: ([self.modvar_ippu_ef_pfc116_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_pfc3110: ([self.modvar_ippu_ef_pfc3110_per_gdp_produse], []),
            self.modvar_ippu_emissions_produse_pfc5114: ([self.modvar_ippu_ef_pfc5114_per_gdp_produse], [])
        }
        # get emissions in data frame
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            0,
            array_ippu_gdp_scalar_produse,
            None,
            dict_ippu_produse_simple_efs_indiv_fgas,
            False,
        )

        """
        df_ippu_produse_indiv_fgas = pd.concat(df_ippu_produse_indiv_fgas, axis = 1)

        # combine output dataframes for individual F-gasses
        df_ippu_indiv_fgas = pd.concat(df_ippu_process_indiv_fgas, axis = 1)
        for k in df_ippu_produse_indiv_fgas.columns:
            vec = np.array(df_ippu_produse_indiv_fgas[k])
            df_ippu_indiv_fgas[k] = (
                np.array(df_ippu_indiv_fgas[k]) + vec 
                if (k in df_ippu_indiv_fgas.columns) 
                else vec
            )

        df_out += [df_ippu_indiv_fgas]
        """;

        # non-standard emission fields to include in emission total for IPPU
        vars_additional_sum = [
            self.modvar_ippu_emissions_process_hfc,
            self.modvar_ippu_emissions_produse_hfc,
            self.modvar_ippu_emissions_process_pfc,
            self.modvar_ippu_emissions_produse_pfc,
            self.modvar_ippu_emissions_process_other_fcs
        ]
        
        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, merge_type = "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)
        self.model_attributes._add_specified_total_fields_to_emission_total(df_out, vars_additional_sum)

        return df_out
