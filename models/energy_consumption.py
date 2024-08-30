
import logging
import numpy as np
import pandas as pd
from typing import *


from sisepuede.core.attribute_table import AttributeTable
from sisepuede.core.model_attributes import *
from sisepuede.models.ippu import IPPU
from sisepuede.models.socioeconomic import Socioeconomic
import sisepuede.utilities._toolbox as sf

###############################################
###                                         ###
###     NON-FUEL PRODUCTION ENERGY MODEL    ###
###                                         ###
###############################################

class EnergyConsumption:
    """
    Use EnergyConsumption to calculate emissions from energy production in
        SISEPUEDE. Includes emissions from the following subsectors:

        * Carbon Capture and Sequestration (CCSQ)
        * Industrial Energy (INEN)
        * Stationary Combustion and Other Energy (SCOE)
        * Transportation (TRNS)

    Additionally, includes the following non-emissions models:

        * Transportation Demand (TRDE)

    For additional information, see the SISEPUEDE readthedocs at:

        https://sisepuede.readthedocs.io/en/latest/energy_consumption.html

    

    Intialization Arguments
    -----------------------
    - model_attributes: ModelAttributes object used in SISEPUEDE

    Optional Arguments
    ------------------
    - logger: optional logger object to use for event logging
    """
    def __init__(self,
        attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None,
    ):

        self.logger = logger
        self.model_attributes = attributes

        self._initialize_subsector_names()
        self._initialize_input_output_components()

        # initialize model variables, categories, and fields
        self._initialize_sector_vars_afolu()
        self._initialize_subsector_vars_ccsq()
        self._initialize_subsector_vars_enfu()
        self._initialize_subsector_vars_fgtv()
        self._initialize_subsector_vars_inen()
        self._initialize_subsector_vars_scoe()
        self._initialize_subsector_vars_trde()
        self._initialize_subsector_vars_trns()

        # 
        self._initialize_models()
        self._initialize_other_properties()
        self._initialize_integrated_variables()

        return None
    


    def __call__(self,
        *args,
        **kwargs
    ) -> pd.DataFrame:

        return self.project(*args, **kwargs)

        



    ###########################
    #    SUPPORT FUNCTIONS    #
    ###########################

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






    ##################################################
    #    FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS    #
    ##################################################

    def check_df_fields(self,
        df_neenergy_trajectories: pd.DataFrame,
        subsector: str = "All",
        var_type: str = "input",
        msg_prepend: str = None,
    ) -> None:
        """
        Check for presence of subsector fields in df_neenergy_trajectories. If
            var_type == "input", checks for input variables to subsector model;
            if var_type == "output", checks for output variables from subsector
            model. If subsector == "All", checks for self.required_variables.
        """
        if subsector == "All":
            check_fields = self.required_variables
            msg_prepend = "Energy"
        else:
            self.model_attributes.check_subsector(subsector)

            if var_type == "input":
                check_fields, ignore_fields = self.model_attributes.get_input_output_fields(
                    [
                        self.subsec_name_econ, 
                        self.subsec_name_gnrl, 
                        subsector
                    ]
                )

            elif var_type == "output":
                ignore_fields, check_fields = self.model_attributes.get_input_output_fields([subsector])

            else:
                raise ValueError(f"Invalid var_type '{var_type}' in check_df_fields: valid types are 'input', 'output'")

            msg_prepend = msg_prepend if (msg_prepend is not None) else subsector

        sf.check_fields(
            df_neenergy_trajectories,
            check_fields,
            f"{msg_prepend} projection cannot proceed: fields "
        )

        return None



    def get_required_subsectors(self,
    ):
        ## TEMPORARY
        subsectors = [
            self.subsec_name_ccsq,
            self.subsec_name_enfu,
            self.subsec_name_fgtv,
            self.subsec_name_inen, 
            self.subsec_name_trns, 
            self.subsec_name_trde, 
            self.subsec_name_scoe
        ]#self.subsec_name_enfu,#self.model_attributes.get_setor_subsectors("Energy")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]

        return subsectors, subsectors_base



    def get_neenergy_input_output_fields(self,
    ):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)

        return required_vars + self.get_required_dimensions(), output_vars



    def get_projection_subsectors(self,
        subsectors_project: Union[list, str, None] = None,
        delim: str = "|",
        drop_fugitive_from_none_q: bool = True
    ) -> list:
        """
            Check and retrieve valid projection subsectors to run in EnergyConsumption.project()

            Keyword Arguments
            ------------------
            - subsectors_project: list or string to run. If None, all valid subsectors (exludes EnergyConsumption.subsec_name_fgtv if drop_fugitive_from_none_q = True)
            - delim: delimiter to use in input strings
            - drop_fugitive_from_none_q: drop EnergyConsumption.subsec_name_fgtv if subsectors_project == None?
        """
        # get subsector attribute
        attr_subsec = self.model_attributes.get_subsector_attribute_table()
        attr_subsec_table = attr_subsec.table[attr_subsec.table["subsector"].isin(self.valid_projection_subsecs)]
        valid_subsectors_project = list(attr_subsec_table[attr_subsec.key])
        dict_map = attr_subsec.field_maps.get(f"{attr_subsec.key}_to_subsector")

        # convert input to list
        if (subsectors_project is None):
            list_out = self.valid_projection_subsecs
            list_out = [x for x in list_out if (x != self.subsec_name_fgtv)] if drop_fugitive_from_none_q else list_out
        elif isinstance(subsectors_project, str):
            list_out = subsectors_project.split(delim)
        elif isinstance(subsectors_project, list) or isinstance(subsectors_project, np.ndarray):
            list_out = list(subsectors_project)
        # check values
        list_out = [dict_map.get(x) for x in valid_subsectors_project if (x in list_out) or (dict_map.get(x) in list_out)]

        return list_out



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

        subsectors = [
            self.subsec_name_ccsq,
            self.subsec_name_enfu,
            self.subsec_name_fgtv,
            self.subsec_name_inen, 
            self.subsec_name_trns, 
            self.subsec_name_trde, 
            self.subsec_name_scoe
        ]#self.subsec_name_enfu,#self.model_attributes.get_setor_subsectors("Energy")
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



    def _initialize_integrated_variables(self,
    ) -> None:
        """
        Sets integrated variables, including the following properties:

            * self.integration_variables_non_fgtv
            * self.integration_variables_fgtv
        """
        # set the integration variables
        list_vars_required_for_integration = [
            self.modvar_agrc_yield,
            self.model_ippu.modvar_ippu_qty_total_production,
            self.modvar_lvst_total_animal_mass
        ]

        # in Energy, update required variables
        new_vars = self.model_attributes.build_variable_fields(
            list_vars_required_for_integration
        )
        self.required_variables += new_vars

        # sot required variables and ensure no double counting
        self.required_variables = list(set(self.required_variables))
        self.required_variables.sort()

        # return variables required for secondary integrtion (i.e., for fugitive emissions only)
        list_vars_required_for_integration_fgtv = [
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_entc,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns,
            self.modvar_enfu_energy_demand_by_fuel_total,
            self.modvar_enfu_exports_fuel_adjusted,
            self.modvar_enfu_imports_fuel,
            self.modvar_enfu_production_fuel
        ]


        self.integration_variables_non_fgtv = list_vars_required_for_integration
        self.integration_variables_fgtv = list_vars_required_for_integration_fgtv

        return None


    
    def _initialize_models(self,
        model_attributes: Union[ModelAttributes, None] = None
    ) -> None:
        """
        Initialize SISEPUEDE model classes for fetching variables and 
            accessing methods. Initializes the following properties:

            * self.model_ippu
            * self.model_socioeconomic

        NOTE: CANNOT INITIALIZE AFOLU CLASS BECAUSE IT REQUIRES ACCESS TO 
            THE EnergyConsumption CLASS (circular logic)

        Keyword Arguments
        -----------------
        - model_attributes: ModelAttributes object used to instantiate
            models. If None, defaults to self.model_attributes.
        """

        model_attributes = self.model_attributes if (model_attributes is None) else model_attributes
        
        self.model_ippu = IPPU(model_attributes)
        self.model_socioeconomic = Socioeconomic(model_attributes)

        return None


    
    def _initialize_other_properties(self,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.is_sisepuede_model_nfp_energy
            * self.n_time_periods
            * self.time_periods
            * self.valid_projection_subsecs
        """
        # valid subsectors in .project()
        valid_projection_subsecs = [
            self.subsec_name_ccsq,
            self.subsec_name_fgtv,
            self.subsec_name_inen,
            self.subsec_name_scoe,
            self.subsec_name_trns
        ]

        # time variables
        time_periods, n_time_periods = self.model_attributes.get_time_periods()


        ##  SET PROPERTIES
        
        self.is_sisepuede_model_nfp_energy = True
        self.n_time_periods = n_time_periods
        self.time_periods = time_periods
        self.valid_projection_subsecs = valid_projection_subsecs

        return None
    


    def _initialize_sector_vars_afolu(self,
    ) -> None:
        """
        Initialize sector variables associated with AFOLU (non-exhaustive). Sets
            the following properties

            * self.modvar_agrc_yield
            * self.modvar_lvst_total_animal_mass
        """

        # variables from other sectors (NOTE: AFOLU INTEGRATION VARIABLES MUST BE SET HERE, CANNOT INITIALIZE AFOLU CLASS DUE TO DAG)
        self.modvar_agrc_yield = "Crop Yield"
        self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"

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
            self.subsec_name_fgtv = "Fugitive Emissions"
            self.subsec_name_gnrl = "General"
            self.subsec_name_inen = "Industrial Energy"
            self.subsec_name_ippu = "IPPU"
            self.subsec_name_scoe = "Stationary Combustion and Other Energy"
            self.subsec_name_trns = "Transportation"
            self.subsec_name_trde = "Transportation Demand"

            return None


            
    def _initialize_subsector_vars_ccsq(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            CCSQ (Carbon Capture and Sequestration). Sets the following 
            properties:

            * self.cat_ccsq_****
            * self.ind_ccsq_****
            * self.modvar_ccsq_****
            * self.modvar_dict_ccsq_****
            * self.modvar_dicts_ccsq_****
        """
        # Carbon Capture and Sequestration variables
        self.modvar_ccsq_demand_per_co2 = "CCSQ Energy Demand Per Mass of :math:\\text{CO}_2 Captured"
        self.modvar_ccsq_efficiency_fact_heat_en_geothermal = "CCSQ Efficiency Factor for Heat Energy from Geothermal"
        self.modvar_ccsq_efficiency_fact_heat_en_hydrogen = "CCSQ Efficiency Factor for Heat Energy from Hydrogen"
        self.modvar_ccsq_efficiency_fact_heat_en_natural_gas = "CCSQ Efficiency Factor for Heat Energy from Natural Gas"
        self.modvar_ccsq_energy_consumption_electricity = "Electrical Energy Consumption from CCSQ"
        self.modvar_ccsq_energy_consumption_electricity_agg = "Total Electrical Energy Consumption from CCSQ"
        self.modvar_ccsq_energy_consumption_total = "Energy Consumption from CCSQ"
        self.modvar_ccsq_energy_consumption_total_agg = "Total Energy Consumption from CCSQ"
        self.modvar_ccsq_emissions_ch4 = ":math:\\text{CH}_4 Emissions from CCSQ"
        self.modvar_ccsq_emissions_co2 = ":math:\\text{CO}_2 Emissions from CCSQ"
        self.modvar_ccsq_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from CCSQ"
        self.modvar_ccsq_frac_en_electricity = "CCSQ Fraction Energy Electricity"
        self.modvar_ccsq_frac_en_heat = "CCSQ Fraction Energy Heat"
        self.modvar_ccsq_frac_heat_en_geothermal = "CCSQ Fraction Heat Energy Demand Geothermal"
        self.modvar_ccsq_frac_heat_en_hydrogen = "CCSQ Fraction Heat Energy Demand Hydrogen"
        self.modvar_ccsq_frac_heat_en_natural_gas = "CCSQ Fraction Heat Energy Demand Natural Gas"
        self.modvar_ccsq_total_sequestration = "Annual Capture and Sequestration by Type"

        # get some dictionaries implied by the CCSQ attribute tables
        self.modvar_dicts_ccsq_fuel_vars = self.model_attributes.get_var_dicts_by_shared_category(
            self.subsec_name_ccsq,
            self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary_element"),
            ["energy_efficiency_variable_by_fuel", "fuel_fraction_variable_by_fuel"]
        )

        # reassign as variables
        self.modvar_dict_ccsq_fuel_fractions_to_efficiency_factors = self.modvar_dicts_ccsq_fuel_vars.get(
            "fuel_fraction_variable_by_fuel_to_energy_efficiency_variable_by_fuel"
        )

        return None



    def _initialize_subsector_vars_enfu(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            ENFU (Energy Fuels). Sets the following properties:

            * self.cat_enfu_****
            * self.ind_enfu_****
            * self.modvar_enfu_****
            * self.modvars_enfu_****
        """
        # Energy Fuel model variables
        self.modvar_enfu_energy_density_volumetric = "Volumetric Energy Density"
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

        # list of key variables - total energy demands by fuel
        self.modvars_enfu_energy_demands_total = [
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_entc,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns
        ]
        # total demand for fuels for estimating distribution
        self.modvars_enfu_energy_demands_distribution = [
            self.modvar_enfu_energy_demand_by_fuel_entc,
            self.modvar_enfu_energy_demand_by_fuel_scoe
        ]
        # key categories
        self.cat_enfu_electricity = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {
                self.model_attributes.field_enfu_electricity_demand_category: 1
            }
        )[0]

        return None



    def _initialize_subsector_vars_fgtv(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            FGTV (Fugitive Emissions). Sets the following properties:

            * self.cat_fgtv_****
            * self.ind_fgtv_****
            * self.modvar_fgtv_****
        """
        # Fugitive Emissions model variables
        self.modvar_fgtv_ef_ch4_distribution = ":math:\\text{CH}_4 FGTV Distribution Emission Factor"
        self.modvar_fgtv_ef_ch4_production_flaring = ":math:\\text{CH}_4 FGTV Production Flaring Emission Factor"
        self.modvar_fgtv_ef_ch4_production_fugitive = ":math:\\text{CH}_4 FGTV Production Fugitive Emission Factor"
        self.modvar_fgtv_ef_ch4_production_venting = ":math:\\text{CH}_4 FGTV Production Venting Emission Factor"
        self.modvar_fgtv_ef_ch4_transmission = ":math:\\text{CH}_4 FGTV Transmission Emission Factor"
        self.modvar_fgtv_ef_co2_distribution = ":math:\\text{CO}_2 FGTV Distribution Emission Factor"
        self.modvar_fgtv_ef_co2_production_flaring = ":math:\\text{CO}_2 FGTV Production Flaring Emission Factor"
        self.modvar_fgtv_ef_co2_production_fugitive = ":math:\\text{CO}_2 FGTV Production Fugitive Emission Factor"
        self.modvar_fgtv_ef_co2_production_venting = ":math:\\text{CO}_2 FGTV Production Venting Emission Factor"
        self.modvar_fgtv_ef_co2_transmission = ":math:\\text{CO}_2 FGTV Transmission Emission Factor"
        self.modvar_fgtv_ef_n2o_production_flaring = ":math:\\text{N}_2\\text{O} FGTV Production Flaring Emission Factor"
        self.modvar_fgtv_ef_n2o_production_fugitive = ":math:\\text{N}_2\\text{O} FGTV Production Fugitive Emission Factor"
        self.modvar_fgtv_ef_n2o_production_venting = ":math:\\text{N}_2\\text{O} FGTV Production Venting Emission Factor"
        self.modvar_fgtv_ef_n2o_transmission = ":math:\\text{N}_2\\text{O} FGTV Transmission Emission Factor"
        self.modvar_fgtv_ef_nmvoc_distribution = "NMVOC FGTV Distribution Emission Factor"
        self.modvar_fgtv_ef_nmvoc_production_flaring = "NMVOC FGTV Production Flaring Emission Factor"
        self.modvar_fgtv_ef_nmvoc_production_fugitive = "NMVOC FGTV Production Fugitive Emission Factor"
        self.modvar_fgtv_ef_nmvoc_production_venting = "NMVOC FGTV Production Venting Emission Factor"
        self.modvar_fgtv_ef_nmvoc_transmission = "NMVOC FGTV Transmission Emission Factor"
        self.modvar_fgtv_emissions_ch4 = ":math:\\text{CH}_4 Fugitive Emissions"
        self.modvar_fgtv_emissions_co2 = ":math:\\text{CO}_2 Fugitive Emissions"
        self.modvar_fgtv_emissions_n2o = ":math:\\text{N}_2\\text{O} Fugitive Emissions"
        self.modvar_fgtv_emissions_nmvoc = "NMVOC Fugitive Emissions"
        self.modvar_fgtv_frac_non_fugitive_flared = "Fraction Non-Fugitive :math:\\text{CH}_4 Flared"
        self.modvar_fgtv_frac_reduction_fugitive_leaks = "Reduction in Fugitive Leaks"

        return None



    def _initialize_subsector_vars_inen(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            INEN (Industrial Energy). Sets the following properties:

            * self.cat_inen_****
            * self.ind_inen_****
            * self.modvar_inen_****
            * self.modvar_dict_inen_****
        """
        # Industrial Energy model variables
        self.modvar_inen_demscalar = "Industrial Energy Demand Scalar"
        self.modvar_inen_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Industrial Energy"
        self.modvar_inen_emissions_co2 = ":math:\\text{CO}_2 Emissions from Industrial Energy"
        self.modvar_inen_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Industrial Energy"
        self.modvar_inen_energy_conumption_agrc_init = "Initial Energy Consumption in Agriculture and Livestock"
        self.modvar_inen_energy_consumption_electricity = "Electrical Energy Consumption from Industrial Energy"
        self.modvar_inen_energy_consumption_electricity_agg = "Total Electrical Energy Consumption from Industrial Energy"
        self.modvar_inen_energy_consumption_total = "Energy Consumption from Industrial Energy"
        self.modvar_inen_energy_consumption_total_agg = "Total Energy Consumption from Industrial Energy"
        self.modvar_inen_energy_demand_total = "Energy Demand in Industrial Energy"
        self.modvar_inen_en_gdp_intensity_factor = "Initial Energy Consumption Intensity of GDP"
        self.modvar_inen_en_prod_intensity_factor = "Initial Energy Consumption Intensity of Production"
        self.modvar_inen_frac_en_coal = "Industrial Energy Fuel Fraction Coal"
        self.modvar_inen_frac_en_coke = "Industrial Energy Fuel Fraction Coke"
        self.modvar_inen_frac_en_diesel = "Industrial Energy Fuel Fraction Diesel"
        self.modvar_inen_frac_en_electricity = "Industrial Energy Fuel Fraction Electricity"
        self.modvar_inen_frac_en_furnace_gas = "Industrial Energy Fuel Fraction Furnace Gas"
        self.modvar_inen_frac_en_gasoline = "Industrial Energy Fuel Fraction Gasoline"
        self.modvar_inen_frac_en_geothermal = "Industrial Energy Fuel Fraction Geothermal"
        self.modvar_inen_frac_en_hgl = "Industrial Energy Fuel Fraction Hydrocarbon Gas Liquids"
        self.modvar_inen_frac_en_hydrogen = "Industrial Energy Fuel Fraction Hydrogen"
        self.modvar_inen_frac_en_kerosene = "Industrial Energy Fuel Fraction Kerosene"
        self.modvar_inen_frac_en_natural_gas = "Industrial Energy Fuel Fraction Natural Gas"
        self.modvar_inen_frac_en_oil = "Industrial Energy Fuel Fraction Oil"
        self.modvar_inen_frac_en_solar = "Industrial Energy Fuel Fraction Solar"
        self.modvar_inen_frac_en_solid_biomass = "Industrial Energy Fuel Fraction Solid Biomass"
        self.modvar_inen_gas_captured_co2 = ":math:\\text{CO}_2 Captured in Industrial Energy"
        
        # get some dictionaries implied by the inen attribute tables
        self.dict_inen_fuel_categories_to_fuel_variables, self.dict_inen_fuel_categories_to_unassigned_fuel_variables = self.get_inen_dict_fuel_categories_to_fuel_variables()
        self.modvars_inen_list_fuel_fraction = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_inen_fuel_categories_to_fuel_variables,
            "fuel_fraction"
        )
        # key categories
        self.cat_inen_agricultural = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_inen, 
            {
                "agricultural_category": 1
            }
        )[0]

        return None



    def _initialize_subsector_vars_scoe(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            SCOE (Stationary Combustion and Other Energy). Sets the 
            following properties:

            * self.cat_scoe_****
            * self.ind_scoe_****
            * self.modvar_scoe_****
            * self.modvar_dict_scoe_****
            * self.modvar_dicts_scoe_****
        """
        # Stationary Combustion and Other Energy variables
        self.modvar_scoe_consumpinit_energy_per_hh_elec = "SCOE Initial Per Household Electric Appliances Energy Consumption"
        self.modvar_scoe_consumpinit_energy_per_hh_heat = "SCOE Initial Per Household Heat Energy Consumption"
        self.modvar_scoe_consumpinit_energy_per_mmmgdp_elec = "SCOE Initial Per GDP Electric Appliances Energy Consumption"
        self.modvar_scoe_consumpinit_energy_per_mmmgdp_heat = "SCOE Initial Per GDP Heat Energy Consumption"
        self.modvar_scoe_demscalar_elec_energy_demand = "SCOE Appliance Energy Demand Scalar"
        self.modvar_scoe_demscalar_heat_energy_demand = "SCOE Heat Energy Demand Scalar"
        self.modvar_scoe_efficiency_fact_heat_en_coal = "SCOE Efficiency Factor for Heat Energy from Coal"
        self.modvar_scoe_efficiency_fact_heat_en_diesel = "SCOE Efficiency Factor for Heat Energy from Diesel"
        self.modvar_scoe_efficiency_fact_heat_en_electricity = "SCOE Efficiency Factor for Heat Energy from Electricity"
        self.modvar_scoe_efficiency_fact_heat_en_gasoline = "SCOE Efficiency Factor for Heat Energy from Gasoline"
        self.modvar_scoe_efficiency_fact_heat_en_hgl = "SCOE Efficiency Factor for Heat Energy from Hydrocarbon Gas Liquids"
        self.modvar_scoe_efficiency_fact_heat_en_hydrogen = "SCOE Efficiency Factor for Heat Energy from Hydrogen"
        self.modvar_scoe_efficiency_fact_heat_en_kerosene = "SCOE Efficiency Factor for Heat Energy from Kerosene"
        self.modvar_scoe_efficiency_fact_heat_en_natural_gas = "SCOE Efficiency Factor for Heat Energy from Natural Gas"
        self.modvar_scoe_efficiency_fact_heat_en_solid_biomass = "SCOE Efficiency Factor for Heat Energy from Solid Biomass"
        self.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc = "SCOE Elasticity of Per Household Electrical Applicance Demand to GDP Per Capita"
        self.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc = "SCOE Elasticity of Per Household Heat Energy Demand to GDP Per Capita"
        self.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc = "SCOE Elasticity of Per GDP Electrical Applicance Demand to GDP Per Capita"
        self.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc = "SCOE Elasticity of Per GDP Heat Energy Demand to GDP Per Capita"
        self.modvar_scoe_emissions_ch4 = ":math:\\text{CH}_4 Emissions from SCOE"
        self.modvar_scoe_emissions_co2 = ":math:\\text{CO}_2 Emissions from SCOE"
        self.modvar_scoe_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from SCOE"
        self.modvar_scoe_energy_consumption_electricity = "Electrical Energy Consumption from SCOE"
        self.modvar_scoe_energy_consumption_electricity_agg = "Total Electrical Energy Consumption from SCOE"
        self.modvar_scoe_energy_consumption_total = "Energy Consumption from SCOE"
        self.modvar_scoe_energy_consumption_total_agg = "Total Energy Consumption from SCOE"
        self.modvar_scoe_energy_demand_heat_total = "Heat Energy Demand in SCOE"
        self.modvar_scoe_frac_heat_en_coal = "SCOE Fraction Heat Energy Demand Coal"
        self.modvar_scoe_frac_heat_en_diesel = "SCOE Fraction Heat Energy Demand Diesel"
        self.modvar_scoe_frac_heat_en_electricity = "SCOE Fraction Heat Energy Demand Electricity"
        self.modvar_scoe_frac_heat_en_gasoline = "SCOE Fraction Heat Energy Demand Gasoline"
        self.modvar_scoe_frac_heat_en_hgl = "SCOE Fraction Heat Energy Demand Hydrocarbon Gas Liquids"
        self.modvar_scoe_frac_heat_en_hydrogen = "SCOE Fraction Heat Energy Demand Hydrogen"
        self.modvar_scoe_frac_heat_en_kerosene = "SCOE Fraction Heat Energy Demand Kerosene"
        self.modvar_scoe_frac_heat_en_natural_gas = "SCOE Fraction Heat Energy Demand Natural Gas"
        self.modvar_scoe_frac_heat_en_solid_biomass = "SCOE Fraction Heat Energy Demand Solid Biomass"

        # get some dictionaries implied by the SCOE attribute tables
        
        self.modvar_dicts_scoe_fuel_vars = self.model_attributes.get_var_dicts_by_shared_category(
            self.subsec_name_scoe,
            self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary_element"),
            ["energy_efficiency_variable_by_fuel", "fuel_fraction_variable_by_fuel", "energy_demand_variable_by_fuel"]
        )
        
        # reassign as variables
        self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors = self.modvar_dicts_scoe_fuel_vars.get("fuel_fraction_variable_by_fuel_to_energy_efficiency_variable_by_fuel")

        return None



    def _initialize_subsector_vars_trde(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            trde (Transportation Demand). Sets the following properties:

            * self.cat_trde_****
            * self.ind_trde_****
            * self.modvar_trde_****
            * self.modvar_dict_trde_****
            * self.modvar_dicts_trde_****
        """
        # Transportation Demand variables
        self.modvar_trde_demand_scalar = "Transportation Demand Scalar"
        self.modvar_trde_elasticity_mtkm_to_gdp = "Elasticity of Megatonne-Kilometer Demand to GDP"
        self.modvar_trde_elasticity_pkm_to_gdp = "Elasticity of Passenger-Kilometer Demand per Capita to GDP per Capita"
        self.modvar_trde_demand_initial_mtkm = "Initial Megatonne-Kilometer Demand"
        self.modvar_trde_demand_initial_pkm_per_capita = "Initial per Capita Passenger-Kilometer Demand"
        self.modvar_trde_demand_mtkm = "Megatonne-Kilometer Demand"
        self.modvar_trde_demand_pkm = "Passenger-Kilometer Demand"

        self.cat_trde_frgt = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_trde, 
            {"freight_category": 1}
        )[0]


        return None



    def _initialize_subsector_vars_trns(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            trns (Transportation). Sets the following properties:

            * self.cat_trns_****
            * self.ind_trns_****
            * self.modvar_trns_****
            * self.modvar_dict_trns_****
            * self.modvar_dicts_trns_****
        """
        
        # Transportation variablesz
        self.modvar_trns_average_vehicle_load_freight = "Average Freight Vehicle Load"
        self.modvar_trns_average_passenger_occupancy = "Average Passenger Vehicle Occupancy Rate"
        self.modvar_trns_ef_combustion_mobile_biofuels_ch4 = ":math:\\text{CH}_4 Biofuels Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_diesel_ch4 = ":math:\\text{CH}_4 Diesel Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_gasoline_ch4 = ":math:\\text{CH}_4 Gasoline Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_hgl_ch4 = ":math:\\text{CH}_4 Hydrocarbon Gas Liquids Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_kerosene_ch4 = ":math:\\text{CH}_4 Kerosene Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_natural_gas_ch4 = ":math:\\text{CH}_4 Natural Gas Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_biofuels_n2o = ":math:\\text{N}_2\\text{O} Biofuels Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_diesel_n2o = ":math:\\text{N}_2\\text{O} Diesel Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_gasoline_n2o = ":math:\\text{N}_2\\text{O} Gasoline Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_hgl_n2o = ":math:\\text{N}_2\\text{O} Hydrocarbon Gas Liquids Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_kerosene_n2o = ":math:\\text{N}_2\\text{O} Kerosene Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_natural_gas_n2o = ":math:\\text{N}_2\\text{O} Natural Gas Mobile Combustion Emission Factor"
        self.modvar_trns_electrical_efficiency = "Electrical Vehicle Efficiency"
        self.modvar_trns_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Transportation"
        self.modvar_trns_emissions_co2 = ":math:\\text{CO}_2 Emissions from Transportation"
        self.modvar_trns_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Transportation"
        self.modvar_trns_energy_consumption_electricity = "Electrical Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_electricity_agg = "Total Electrical Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_total = "Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_total_agg = "Total Energy Consumption from Transportation"
        self.modvar_trns_fuel_fraction_biofuels = "Transportation Mode Fuel Fraction Biofuels"
        self.modvar_trns_fuel_fraction_diesel = "Transportation Mode Fuel Fraction Diesel"
        self.modvar_trns_fuel_fraction_electricity = "Transportation Mode Fuel Fraction Electricity"
        self.modvar_trns_fuel_fraction_gasoline = "Transportation Mode Fuel Fraction Gasoline"
        self.modvar_trns_fuel_fraction_hgl = "Transportation Mode Fuel Fraction Hydrocarbon Gas Liquids"
        self.modvar_trns_fuel_fraction_hydrogen = "Transportation Mode Fuel Fraction Hydrogen"
        self.modvar_trns_fuel_fraction_kerosene = "Transportation Mode Fuel Fraction Kerosene"
        self.modvar_trns_fuel_fraction_natural_gas = "Transportation Mode Fuel Fraction Natural Gas"
        self.modvar_trns_fuel_efficiency_biofuels = "Fuel Efficiency Biofuels"
        self.modvar_trns_fuel_efficiency_diesel = "Fuel Efficiency Diesel"
        self.modvar_trns_fuel_efficiency_gasoline = "Fuel Efficiency Gasoline"
        self.modvar_trns_fuel_efficiency_hgl = "Fuel Efficiency Hydrocarbon Gas Liquids"
        self.modvar_trns_fuel_efficiency_hydrogen = "Fuel Efficiency Hydrogen"
        self.modvar_trns_fuel_efficiency_kerosene = "Fuel Efficiency Kerosene"
        self.modvar_trns_fuel_efficiency_natural_gas = "Fuel Efficiency Natural Gas"
        self.modvar_trns_mass_distance_traveled = "Total Megatonne-Kilometer Demand by Vehicle"
        self.modvar_trns_modeshare_freight = "Freight Transportation Mode Share"
        self.modvar_trns_modeshare_public_private = "Private and Public Transportation Mode Share"
        self.modvar_trns_modeshare_regional = "Regional Transportation Mode Share"
        self.modvar_trns_passenger_distance_traveled = "Total Passenger Distance by Vehicle"
        self.modvar_trns_vehicle_distance_traveled = "Total Vehicle Distance Traveled"
        self.modvar_trns_vehicle_distance_traveled_biofuels = "Vehicle Distance Traveled from Biofuels"
        self.modvar_trns_vehicle_distance_traveled_diesel = "Vehicle Distance Traveled from Diesel"
        self.modvar_trns_vehicle_distance_traveled_electricity = "Vehicle Distance Traveled from Electricity"
        self.modvar_trns_vehicle_distance_traveled_gasoline = "Vehicle Distance Traveled from Gasoline"
        self.modvar_trns_vehicle_distance_traveled_hgl = "Vehicle Distance Traveled from Hydrocarbon Gas Liquids"
        self.modvar_trns_vehicle_distance_traveled_hydrogen = "Vehicle Distance Traveled from Hydrogen"
        self.modvar_trns_vehicle_distance_traveled_kerosene = "Vehicle Distance Traveled from Kerosene"
        self.modvar_trns_vehicle_distance_traveled_natural_gas = "Vehicle Distance Traveled from Natural Gas"

        # fuel variables dictionary for transportation
        tuple_dicts = self.get_trns_dict_fuel_categories_to_fuel_variables()
        self.dict_trns_fuel_categories_to_fuel_variables = tuple_dicts[0]
        self.dict_trns_fuel_categories_to_unassigned_fuel_variables = tuple_dicts[1]

        # some derivate lists of variables
        self.modvars_trns_list_fuel_fraction = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_fraction"
        )
        self.modvars_trns_list_fuel_efficiency = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_efficiency"
        )

        return None






    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################

    def get_agrc_lvst_prod_and_intensity(self,
        df_neenergy_trajectories: pd.DataFrame
    ) -> Tuple:
        """
        Retrieve agriculture and livstock production (total mass) and initial 
            energy consumption, then calculate energy intensity (in terms of 
            self.modvar_inen_en_prod_intensity_factor) and return production (in 
            terms of self.model_ippu.modvar_ippu_qty_total_production).

            Returns a tuple of the form:

            (
                index_inen_agrc, 
                vec_inen_energy_intensity_agrc_lvst, 
                vec_inen_prod_agrc_lvst,
            )

        Function Arguments
        ------------------
        - df_neenergy_trajectories: model input dataframe
        """

        attr_inen = self.model_attributes.get_attribute_table(self.subsec_name_inen)

        # add agricultural and livestock production to scale initial energy consumption
        modvar_inen_agrc_prod, arr_inen_agrc_prod = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_neenergy_trajectories,
            self.modvar_agrc_yield,
            None,
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )
        modvar_inen_lvst_prod, arr_inen_lvst_prod = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_neenergy_trajectories,
            self.modvar_lvst_total_animal_mass,
            None,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        # get initial energy consumption for agrc/lvst and then ensure unit are set
        arr_inen_init_energy_consumption_agrc = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_inen_energy_conumption_agrc_init, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_inen_init_energy_consumption_agrc *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_inen_energy_conumption_agrc_init,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        )

        # build production total mass, which is estimated as the driver of changes to initial demand
        vec_inen_prod_agrc_lvst = 0.0
        if modvar_inen_agrc_prod is not None:
            # if agricultural production is defined, convert to industrial production units and add to total output mass
            arr_inen_agrc_prod *= self.model_attributes.get_variable_unit_conversion_factor(
                self.modvar_agrc_yield,
                self.model_ippu.modvar_ippu_qty_total_production,
                "mass"
            )
            vec_inen_prod_agrc_lvst += arr_inen_agrc_prod
        if (modvar_inen_lvst_prod is not None):
            # if livestock production is defined, convert to industrial production units and add
            arr_inen_lvst_prod *= self.model_attributes.get_variable_unit_conversion_factor(
                self.modvar_lvst_total_animal_mass,
                self.model_ippu.modvar_ippu_qty_total_production,
                "mass"
            )
            vec_inen_prod_agrc_lvst += arr_inen_lvst_prod
        # collapse to vector
        vec_inen_prod_agrc_lvst = np.sum(vec_inen_prod_agrc_lvst, axis = 1)

        # get energy intensity
        index_inen_agrc = attr_inen.get_key_value_index(self.cat_inen_agricultural)
        vec_inen_energy_intensity_agrc_lvst = arr_inen_init_energy_consumption_agrc[:, index_inen_agrc].copy()
        vec_inen_energy_intensity_agrc_lvst = np.nan_to_num(vec_inen_energy_intensity_agrc_lvst/vec_inen_prod_agrc_lvst, 0.0, posinf = 0.0)

        # return index + vectors
        tup_out = index_inen_agrc, vec_inen_energy_intensity_agrc_lvst, vec_inen_prod_agrc_lvst

        return tup_out



    def get_enfu_fuel_costs_per_energy(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_for_units_energy: Union[str, None] = None,
        units_energy: Union[str, None] = None,
        units_monetary: Union[str, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieve the cost (in units_monetary) of fuels in terms per energy unit 
            units_energy for fuels. 

        * NOTE: Assumes that each fuel has *one* type of price specified: either
            gravimetric, thermal, or volumetric. USE WITH CAUTION.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: data frame containing input variables as 
            columns

        Keyword Arguments
        -----------------
        - modvar_for_units_energy: optional model variable for units_energy to
            match.
            * NOTE: Overridden by specification of units_energy. Only valid if
                units_energy is None.
        - units_energy: valid energy unit. If None (or if invalid), default to 
            configuration units.
        - units_monetary: valid monetary unit. If None (or if invalid), default 
            to configuration units.
        """

        if units_energy is None:
            units_energy = (
                self.model_attributes.get_variable_characteristic(
                    modvar_for_units_energy, 
                    self.model_attributes.varchar_str_unit_energy
                )
                if isinstance(modvar_for_units_energy, str)
                else None
            )
            
        arr_price = self.get_enfu_fuel_costs_per_energy_general(
            df_neenergy_trajectories,
            type_conversion = "gravimetric",
            units_energy = units_energy,
            units_monetary = units_monetary
        )

        arr_price += self.get_enfu_fuel_costs_per_energy_general(
            df_neenergy_trajectories,
            type_conversion = "volumetric",
            units_energy = units_energy,
            units_monetary = units_monetary
        )

        arr_price += self.get_enfu_fuel_costs_per_energy_thermal(
            df_neenergy_trajectories,
            units_energy = units_energy,
            units_monetary = units_monetary
        )

        return arr_price



    def get_enfu_fuel_costs_per_energy_general(self,
        df_neenergy_trajectories: pd.DataFrame,
        type_conversion: str,
        modvar_density: Union[str, None] = None,
        modvar_price: Union[str, None] = None,
        units_energy: Union[str, None] = None,
        units_monetary: Union[str, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieve the cost (in units_monetary) of fuels in terms per energy unit 
            units_energy for fuels with gravimetric or volumetric pricing. 

        Function Arguments
        ------------------
        - df_neenergy_trajectories: data frame containing input variables as 
            columns
        - type_conversion: either "gravimetric" or "volumetric". If invalid, 
            returns None

        Keyword Arguments
        -----------------
        - modvar_density: model variable used to specify density (e.g., 
            self.modvar_enfu_energy_density_gravimetric). Dependent on 
            type_conversion
        - modvar_price: model variable used to specify density (e.g., 
            self.modvar_enfu_energy_price_gravimetric). Dependent on 
            type_conversion
        - units_energy: valid energy unit. If None (or if invalid), default to 
            configuration units.
        - units_monetary: valid monetary unit. If None (or if invalid), default 
            to configuration units.
        """
        
        if (type_conversion not in ["gravimetric", "volumetric"]):
            return None

        modvar_density = self.modvar_enfu_energy_density_gravimetric if (type_conversion == "gravimetric") else self.modvar_enfu_energy_density_volumetric
        modvar_price = self.modvar_enfu_price_gravimetric if (type_conversion == "gravimetric") else self.modvar_enfu_price_volumetric

        ##  PREPARE SCALARS
        
        # check input units
        units_energy = (
            self.model_attributes.configuration.get("energy_units")
            if (self.model_attributes.get_energy_equivalent(units_energy) is None) 
            else units_energy
        )
        units_monetary = (
            self.model_attributes.configuration.get("monetary_units")
            if (self.model_attributes.get_monetary_equivalent(units_monetary) is None) 
            else units_monetary
        )
        
        # get some variable characteristics
        varchar_units_monetary_price = self.model_attributes.get_variable_characteristic(
            modvar_price, 
            self.model_attributes.varchar_str_unit_monetary
        )
        varchar_units_energy_density = self.model_attributes.get_variable_characteristic(
            modvar_density, 
            self.model_attributes.varchar_str_unit_energy
        )
        
        # scalars to apply for units conversion
        scalar_energy = self.model_attributes.get_energy_equivalent(
            varchar_units_energy_density,
            units_energy
        )
        factor_dim = "mass" if (type_conversion == "gravimetric") else "volume"
        scalar_pivot = self.model_attributes.get_variable_unit_conversion_factor(
            modvar_density,
            modvar_price,
            factor_dim
        )
        scalar_monetary = self.model_attributes.get_monetary_equivalent(
            varchar_units_monetary_price,
            units_monetary
        )
        

        ##  GET PRICES AND DENSITY
        
        # get price in terms of output monetary units
        arr_price = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_price,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        ) * scalar_monetary
        
        # get density in terms of price specified mass
        arr_density = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_density,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )/scalar_pivot
        
        # convert to price per unit of energy, then convert to energy units
        arr_price_per_energy = np.nan_to_num(arr_price/arr_density, 0.0, posinf = 0.0)
        arr_price_per_energy /= scalar_energy
        
        return arr_price_per_energy



    def get_enfu_fuel_costs_per_energy_thermal(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_price: Union[str, None] = None,
        units_energy: Union[str, None] = None,
        units_monetary: Union[str, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieve the cost (in units_monetary) of fuels in terms per energy unit 
            units_energy for fuels with thermal pricing (e.g., $/BTU) 

        Function Arguments
        ------------------
        - df_neenergy_trajectories: data frame containing input variables as 
            columns

        Keyword Arguments
        -----------------
        - modvar_price: model variable used to specify price (e.g., 
            self.modvar_enfu_price_thermal)
        - units_energy: valid energy unit. If None (or if invalid), default to 
            configuration units.
        - units_monetary: valid monetary unit. If None (or if invalid), default 
            to configuration units.
        """
        
        modvar_price = self.modvar_enfu_price_thermal if (modvar_price is None) else modvar_price
        
        ##  PREPARE SCALARS
        
        # check input units
        units_energy = (
            self.model_attributes.configuration.get("energy_units")
            if (self.model_attributes.get_energy_equivalent(units_energy) is None) 
            else units_energy
        )
        units_monetary = (
            self.model_attributes.configuration.get("monetary_units")
            if (self.model_attributes.get_monetary_equivalent(units_monetary) is None) 
            else units_monetary
        )
        
        # get some variable characteristics
        varchar_units_monetary_price = self.model_attributes.get_variable_characteristic(
            modvar_price, 
            self.model_attributes.varchar_str_unit_monetary
        )
        varchar_units_energy_density = self.model_attributes.get_variable_characteristic(
            modvar_price, 
            self.model_attributes.varchar_str_unit_energy
        )
        
        # scalars to apply for units conversion
        scalar_energy = self.model_attributes.get_energy_equivalent(
            varchar_units_energy_density,
            units_energy
        )
        
        scalar_monetary = self.model_attributes.get_monetary_equivalent(
            varchar_units_monetary_price,
            units_monetary
        )
        

        ##  GET PRICES AND DENSITY
        
        # get price in terms of output monetary units and convert to output energy units
        arr_price_per_energy = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_price,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        ) * scalar_monetary
        
        arr_price_per_energy = np.nan_to_num(
            arr_price_per_energy/scalar_energy,
            0.0,
            posinf = 0.0
        )
        
        return arr_price_per_energy



    def get_fgtv_array_for_fugitive_emissions(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_cur: str,
        array_energy_density: np.ndarray,
        modvar_energy_density: str = None
    ) -> np.ndarray:
        """
        Format an array for fugitive emissions calculations. Convert from 
            mass/volume (input/input) to mass/energy (config/config)

        Function Arguments
        ------------------
        - df_neenergy_trajectories: input data frame of trajectories
        - modvar_cur: mass/volume fuel emission factor model variable (used for 
            units conversion)
        - array_energy_density: array of volumetric energy density 
            (energy/volume)
        - modvar_energy_density: model variable giving volumetric density (used 
            for units conversion). If none, defaults to 
            EnergyConsumption.modvar_enfu_energy_density_volumetric
        """
        # check model variable input
        if modvar_cur is None:
            return None

        modvar_energy_density = (
            self.modvar_enfu_energy_density_volumetric 
            if (modvar_energy_density is None) 
            else modvar_energy_density
        )

        # get the variable and associated data (tonne/m3)
        arr_ef_mass_per_volume = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_cur,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_units_corrected_gas",
        )

        # get scalars to convert units (tonne/m3 to mj/litre => m3 -> litre = 1000)
        scalar_volume = self.model_attributes.get_variable_unit_conversion_factor(
            modvar_cur,
            modvar_energy_density,
            "volume"
        )

        scalar_energy = 1/self.model_attributes.get_scalar(modvar_energy_density, "energy") # 1/(mj -> pj) = mj/pj = 1000000000
        scalar_mass = self.model_attributes.get_scalar(modvar_cur, "mass") # tonne -> MT = 0.000001

        # mass (modvar_cur) per energy (arr_enfu_energy_density_volumetric)
        arr_ef_per_config_energy = arr_ef_mass_per_volume/(array_energy_density*scalar_volume) #tonne/m3 / (mj/litre*(litre/m3)) -> (tonne/m3)/(mj/m3) => tonne/mj
        arr_ef_per_config_energy *= scalar_energy # tonne/mj * (mj/pj) =? tonne/pj
        arr_ef_per_config_energy *= scalar_mass # tonne/pj * (mt/tonne) = mt/pj
        arr_ef_per_config_energy = np.nan_to_num(arr_ef_per_config_energy, 0, posinf = 0)

        return arr_ef_per_config_energy


    
    def get_fgtv_demands_and_trade(self,
        df_neenergy_trajectories: pd.DataFrame
    ) -> Tuple[pd.DataFrame]:
        """
        Fugitive Emissions can be run downstream of all of EnergyConsumption 
            models OR downstream of EnergyProduction. 
            
            * If run with EnergyProduction, aggregate demands are taken from 
                df_neenergy_trajectories. 
            * If run without EnergyProduction, aggregate demands (excluding ENTC), 
                imports, production, and adjusted exports (= Exports) are 
                calculated internally and returned in a data frame. 
            
        This function checks for the presence of the following variables to
            determine whether or not it was run downstream of EnergyProduction:

            - EnergyConsumption.modvar_enfu_energy_demand_by_fuel_total
            - EnergyConsumption.modvar_enfu_exports_fuel_adjusted
            - EnergyConsumption.modvar_enfu_imports_fuel
            - EnergyConsumption.modvar_enfu_production_fuel

        Returns a tuple of the form:

            tuple_out = (
                arr_fgtv_demands, 
                arr_demands_distribution, 
                arr_fgtv_export, 
                arr_fgtv_imports, 
                arr_fgtv_production,
                df_out
            ),
        
        where all arrays are in configuration energy units.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: input data

        Keyword Arguments
        -----------------
        
        """

        # initialize some outputs that are conditional
        arr_demands_distribution = None
        df_out = None

        ##  TRY TO GET FROM df_neenergy_trajectories OUTPUTS FROM EnergyProduction

        # demands
        arr_fgtv_demands = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_energy_demand_by_fuel_total,
            expand_to_all_cats = True,
            return_type = "array_base",
            throw_error_on_missing_fields = False,
        )

        if (arr_fgtv_demands is not None):
            arr_fgtv_demands *= self.model_attributes.get_scalar(
                self.modvar_enfu_energy_demand_by_fuel_total, 
                "energy"
            )

        # exports
        arr_fgtv_exports = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_exports_fuel_adjusted,
            expand_to_all_cats = True,
            return_type = "array_base",
            throw_error_on_missing_fields = False,
        )

        if (arr_fgtv_exports is not None):
            arr_fgtv_exports *= self.model_attributes.get_scalar(
                self.modvar_enfu_exports_fuel_adjusted, 
                "energy"
            )

        # imports
        arr_fgtv_imports = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_imports_fuel,
            expand_to_all_cats = True,
            return_type = "array_base",
            throw_error_on_missing_fields = False,
        )

        if (arr_fgtv_imports is not None):
            arr_fgtv_imports *= self.model_attributes.get_scalar(
                self.modvar_enfu_imports_fuel, 
                "energy"
            )

        # production 
        arr_fgtv_production = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_production_fuel,
            expand_to_all_cats = True,
            return_type = "array_base",
            throw_error_on_missing_fields = False,
        )

        if (arr_fgtv_production is not None):
            arr_fgtv_production *= self.model_attributes.get_scalar(
                self.modvar_enfu_production_fuel, 
                "energy"
            )
        
        
        """
        Get demands on distribution
        NOTE: If any of the previous variables are not found, then default to
            the assumption that EnergyProduction was *not* successfully run, so
            the variables have to be added
        """
        generate_demands_distribution = (arr_fgtv_demands is not None)
        generate_demands_distribution &= (arr_fgtv_exports is not None)
        generate_demands_distribution &= (arr_fgtv_imports is not None)
        generate_demands_distribution &= (arr_fgtv_production is not None)

        if generate_demands_distribution:
            # loop over outputs from other energy sectors
            arr_demands_distribution = 0.0
            for modvar in self.modvars_enfu_energy_demands_distribution:

                scalar = self.model_attributes.get_scalar(modvar, "energy")
                arr_tmp = 0.0

                try:
                    arr_tmp = self.model_attributes.extract_model_variable(#
                        df_neenergy_trajectories,
                        modvar,
                        expand_to_all_cats = True,
                        return_type = "array_base",
                    )
                except:
                    self._log(f"Warning in project_enfu_production_and_demands: Variable '{modvar}' not found in the data frame. Its fuel demands will not be included.", type_log = "warning")

                arr_tmp *= scalar
                arr_demands_distribution += arr_tmp

        else:
            
            (
                arr_fgtv_demands, 
                arr_demands_distribution, 
                arr_fgtv_export, 
                arr_fgtv_imports, 
                arr_fgtv_production
            ) = self.project_enfu_production_and_demands(
                df_neenergy_trajectories
            )

            df_out = []

            # add DEMANDS in terms of self.modvar_enfu_energy_demand_by_fuel_total variable units
            df_out += [
                self.model_attributes.array_to_df(
                    arr_fgtv_demands/self.model_attributes.get_scalar(self.modvar_enfu_energy_demand_by_fuel_total, "energy"), 
                    self.modvar_enfu_energy_demand_by_fuel_total, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ] if (arr_fgtv_demands is not None) else []

            # add ADJUSTED EXPORTS (in this case, equal to EXPORTS) in terms of self.modvar_enfu_exports_fuel_adjusted variable units
            df_out += [
                self.model_attributes.array_to_df(
                    arr_fgtv_exports/self.model_attributes.get_scalar(self.modvar_enfu_exports_fuel_adjusted, "energy"), 
                    self.modvar_enfu_exports_fuel_adjusted, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ] if (arr_fgtv_exports is not None) else []

            # add IMPORTS in terms of self.modvar_enfu_imports_fuel variable units
            df_out += [
                self.model_attributes.array_to_df(
                    arr_fgtv_imports/self.model_attributes.get_scalar(self.modvar_enfu_imports_fuel, "energy"), 
                    self.modvar_enfu_imports_fuel, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ] if (arr_fgtv_imports is not None) else []

            # add PRODUCTION in terms of self.modvar_enfu_production_fuel variable units
            df_out += [
                self.model_attributes.array_to_df(
                    arr_fgtv_production/self.model_attributes.get_scalar(self.modvar_enfu_production_fuel, "energy"), 
                    self.modvar_enfu_production_fuel, 
                    reduce_from_all_cats_to_specified_cats = True
                )
            ] if (arr_fgtv_production is not None) else []
            
            df_out = sf.merge_output_df_list(
                df_out, 
                self.model_attributes, 
                merge_type = "concatenate"
            )


        # return a tuple
        tuple_out = (
            arr_fgtv_demands, 
            arr_demands_distribution, 
            arr_fgtv_exports, 
            arr_fgtv_imports, 
            arr_fgtv_production,
            df_out
        )

        return tuple_out



    def get_inen_dict_fuel_categories_to_fuel_variables(self,
    ) -> Dict:
        """
        use get_inen_dict_fuel_categories_to_fuel_variables to return a 
            dictionary with fuel categories as keys based on the Industrial 
            Energy attribute table:

            {
                cat_fuel: {
                    "fuel_efficiency": VARNAME_FUELEFFICIENCY, 
                    ...
                }
            }

            for each key, the dict includes variables associated with the fuel 
            cat_fuel:

            - "fuel_fraction"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.subsec_name_inen,
            "cat_fuel",
            {
                "Fuel Fraction": "fuel_fraction"
            },
        )

        return dict_out



    def get_trns_dict_fuel_categories_to_fuel_variables(self
    ) -> dict:
        """
        Return a dictionary with fuel categories as keys based on the 
            Transportation attribute table:
    
            {
                cat_fuel: {
                    "fuel_efficiency": VARNAME_FUELEFFICIENCY, 
                    ...
                }
            }

            for each key, the dict includes variables associated with the fuel 
            cat_fuel:

            - "ef_ch4"
            - "ef_n2o"
            - "fuel_efficiency"
            - "fuel_fraction"
            - "modal_energy_consumption"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.subsec_name_trns,
            "cat_fuel",
            {
                "Fuel Efficiency": "fuel_efficiency",
                "Fuel Fraction": "fuel_fraction",
                "Transportation Modal Energy Consumption": "modal_energy_consumption",
                ":math:\\text{CH}_4": "ef_ch4",
                ":math:\\text{N}_2\\text{O}": "ef_n2o",
                "Vehicle Distance Traveled from": "vehicle_distance_traveled"
            },
        )

        return dict_out



    def project_energy_consumption_by_fuel_from_effvars(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_consumption: str,
        arr_activity: Union[np.ndarray, None],
        arr_elasticity: Union[np.ndarray, None],
        arr_elastic_driver: Union[np.ndarray, None],
        dict_fuel_fracs: dict,
        dict_fuel_frac_to_eff: dict = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        """
        Project energy consumption--in terms of configuration units for energy--
            for a consumption variable for each fuel specified as a key in 
            self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors.

        Returns a tuple of the form

            (
                arr_demand, # array of point of use demand
                dict_consumption_by_fuel_out # dictionary of consumption by fuel
            )


        Function Arguments
        ------------------
        - df_neenergy_trajectories: Dataframe of input variables
        - modvar_consumption: energy consumption variable, e.g.
            self.modvar_scoe_consumpinit_energy_per_hh_heat
        - arr_activity: per unit activity driving demands.
            * Specify as None if demands are not per-activity.
        - arr_elasticity: array of elasticities for each time step in
            df_neenergy_trajectories.
                * Setting to None will mean that specified future demands will
                be used (often constant).
        - arr_elastic_driver: the driver of elasticity in energy demands, e.g., 
            vector of change rates of gdp per capita.
            * Must be such that 
                df_neenergy_trajectories.shape[0] = arr_elastic_driver.shape[0] == arr_elasticity.shape[0] - 1.
            * Setting to None will mean that specified future demands will be 
                used (often constant).
        - dict_fuel_fracs: dictionary mapping each fuel fraction variable to its 
            fraction of energy.
            * Each key must be a key in dict_fuel_frac_to_eff.

        Keyword Arguments
        -----------------
        - dict_fuel_frac_to_eff: dictionary mapping fuel fraction variable to 
            its associated efficiency variable (SCOE and CCSQ)
        """

        ##  initialize consumption and the fraction -> efficiency dictionary

        # get consumption in terms of configuration output energy units
        arr_consumption = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_consumption,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )
        arr_consumption *= self.model_attributes.get_scalar(modvar_consumption, "energy")

        # get the dictionary/run checks
        if (dict_fuel_frac_to_eff is None):
            subsec_mv_consumption = self.model_attributes.get_variable_subsector(modvar_consumption)
            
            if subsec_mv_consumption is not None:
                if subsec_mv_consumption == self.subsec_name_scoe:
                    dict_fuel_frac_to_eff = self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors

                elif subsec_mv_consumption == self.subsec_name_ccsq:
                    dict_fuel_frac_to_eff = self.modvar_dict_ccsq_fuel_fractions_to_efficiency_factors

                else:
                    msg = f"""
                    Error in project_energy_consumption_by_fuel_from_effvars: 
                    unable to infer dictionary for dict_fuel_frac_to_eff based 
                    on model variable '{modvar_consumption}'.
                    """
                    raise ValueError(msg)
            else:
                msg = f"""
                Invalid model variable '{modvar_consumption}' found in project_energy_consumption_by_fuel_from_effvars: the variable is 
                undefined.
                """
                raise ValueError(msg)

        elif not isinstance(dict_fuel_frac_to_eff, dict):
            tp = str(type(dict_fuel_frac_to_eff))
            msg = f"""
            Error in project_energy_consumption_by_fuel_from_effvars: invalid 
            type '{tp}' specified for dict_fuel_frac_to_eff.
            """
            raise ValueError(msg)


        ##  estimate demand at point of use (account for heat delivery efficiency)

        # loop over the different fuels to generate the true demand
        arr_frac_norm = 0

        # use fractions of demand + efficiencies to calculate fraction of consumption
        for modvar_fuel_frac in dict_fuel_fracs.keys():
            # get efficiency, then fuel fractions
            modvar_fuel_eff = dict_fuel_frac_to_eff.get(modvar_fuel_frac)
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)

            arr_efficiency = self.model_attributes.extract_model_variable(#
                df_neenergy_trajectories,
                modvar_fuel_eff,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
            )

            arr_frac_norm += np.nan_to_num(arr_frac/arr_efficiency, 0.0)

        # project demand forward
        arr_demand = np.nan_to_num(arr_consumption/arr_frac_norm, 0.0)
        if (arr_elastic_driver is not None) and (arr_elasticity is not None):
            arr_growth_demand = sf.project_growth_scalar_from_elasticity(
                arr_elastic_driver, 
                arr_elasticity, 
                False, 
                "standard",
            )
            arr_demand = sf.do_array_mult(arr_demand[0]*arr_growth_demand, arr_activity)

        else:
            self._log(
                "Missing elasticity information found in 'project_energy_consumption_by_fuel_from_effvars': using specified future demands.", 
                type_log = "debug")

            arr_demand = (
                sf.do_array_mult(arr_demand, arr_activity) 
                if (arr_activity is not None) 
                else arr_demand
            )

        # calculate consumption
        dict_consumption_by_fuel_out = {}

        for modvar_fuel_frac in dict_fuel_fracs.keys():
            # get efficiency variable + variable arrays
            modvar_fuel_eff = dict_fuel_frac_to_eff.get(modvar_fuel_frac)
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)

            arr_efficiency = self.model_attributes.extract_model_variable(#
                df_neenergy_trajectories,
                modvar_fuel_eff,
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
            )

            # use consumption by fuel type and efficiency to get output demand for each fuel (in output energy units specified in config)
            arr_consumption_fuel = np.nan_to_num(arr_demand*arr_frac/arr_efficiency, 0.0)
            dict_consumption_by_fuel_out.update({modvar_fuel_frac: arr_consumption_fuel})

        return arr_demand, dict_consumption_by_fuel_out



    def project_energy_consumption_by_fuel_from_fuel_cats(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_consumption_intensity_initial: np.ndarray,
        arr_driver: np.ndarray,
        modvar_demscalar: str,
        modvar_fuel_efficiency: str,
        dict_fuel_fracs: dict,
        dict_fuel_frac_to_fuel_cat: dict,
    ) -> Union[np.ndarray, None]:
        """
        Project energy consumption--in terms of units of the input vector
            vec_consumption_initial--given changing demand fractions and
            efficiency factors.  

        Returns a tuple of the form

            (
                arr_demand, # array of point of use demand
                dict_consumption_by_fuel_out # dictionary of consumption by fuel
            )

            or None if model variables are incorrectly specified. 


        Function Arguments
        ------------------
        - df_neenergy_trajectories: Dataframe of input variables
        - vec_consumption_intensity_initial: array giving initial consumption
            (for initial time period only)
        - arr_driver: driver of demand--either shape of

            (n_projection_time_periods, len(vec_consumption_intensity_initial))

            or

            (n_projection_time_periods, )

        - modvar_demscalar: string model variable giving the demand scalar for
            INEN energy demand
        - modvar_fuel_efficiency: string model variable for enfu fuel efficiency
        - dict_fuel_fracs: dictionary mapping each fuel fraction variable to its
            fraction of energy.
            * Each key must be a key in dict_fuel_frac_to_eff.
        - dict_fuel_frac_to_fuel_cat: dictionary mapping fuel fraction variable
            to its associated fuel category
        """

        # initialize some model_attributes objects
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        return_none = self.model_attributes.get_variable(modvar_demscalar) is None
        return_none |= self.model_attributes.get_variable(modvar_fuel_efficiency) is None

        if return_none:
            self._log(
                f"Error in project_energy_consumption_by_fuel_from_fuel_cats(): invalid model variable specification.",
                type_log = "error",
            )

            self._log(
                f"Variable values:\n\tmodvar_demscalar = {modvar_demscalar}\n\tmodvar_fuel_efficiency = {modvar_fuel_efficiency}",
                type_log = "debug",
            )

            return None


        ##  START CALCULATIONS
        
        arr_frac_norm = 0

        # get some variables - start with energy efficiency
        arr_enfu_efficiency = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_fuel_efficiency,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        # scalar for point of use demand
        arr_inen_demscalar = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_demscalar,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        # use fractions of demand + efficiencies to calculate fraction of consumption
        for modvar_fuel_frac in dict_fuel_fracs.keys():

            cat_fuel = dict_fuel_frac_to_fuel_cat.get(modvar_fuel_frac)
            index_enfu_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # get efficiency, then fuel fractions
            vec_efficiency = arr_enfu_efficiency[:, index_enfu_fuel]
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)
            arr_frac_norm += np.nan_to_num(arr_frac.transpose()/vec_efficiency, nan = 0.0, posinf = 0.0)

        # transpose again and project demand forward
        arr_frac_norm = arr_frac_norm.transpose()
        arr_demand = np.nan_to_num(vec_consumption_intensity_initial/arr_frac_norm[0], nan = 0.0, posinf = 0.0)
        arr_demand = sf.do_array_mult(arr_driver, arr_demand)
        arr_demand *= arr_inen_demscalar

        # calculate consumption
        dict_consumption_by_fuel_out = {}
        for modvar_fuel_frac in dict_fuel_fracs.keys():
            cat_fuel = dict_fuel_frac_to_fuel_cat.get(modvar_fuel_frac)
            index_enfu_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # get efficiency, then fuel fractions
            vec_efficiency = arr_enfu_efficiency[:, index_enfu_fuel]
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)

            # use consumption by fuel type and efficiency to get output demand for each fuel (in output energy units specified in config)
            arr_consumption_fuel = np.nan_to_num((arr_demand*arr_frac).transpose()/vec_efficiency, nan = 0.0, posinf = 0.0).transpose()
            dict_consumption_by_fuel_out.update({modvar_fuel_frac: arr_consumption_fuel})

        return arr_demand, dict_consumption_by_fuel_out



    def project_enfu_production_and_demands(self,
        df_neenergy_trajectories: pd.DataFrame,
        attribute_fuel: AttributeTable = None,
        modvars_energy_demands: list = None,
        modvars_energy_distribution_demands: list = None,
        modvar_energy_exports: str = None,
        modvar_import_fraction: str = None,
        target_energy_units: str = None
    ) -> tuple:

        """
        Project imports, exports, and domestic production demands for fuels. 
            Returns a tuple of np.ndarrays with the following elements:

            (demands, distribution demands, exports, imports, production)

        Arrays are returned in order of attribute_fuel.key_values

        Output units are ModelAttributes.configuration energy units

        Function Arguments
        ------------------
        - df_neenergy_trajectories: Dataframe of input variables
        - attribute_fuel: AttributeTable with information on fuels. If None, use 
            ModelAttributes default.
        - modvars_energy_demands: list of SISEPUEDE model variables to extract 
            for use as energy demands. If None, defaults to 
            EnergyConsumption.modvars_enfu_energy_demands_total
       
        Keyword Arguments
        -----------------
        - modvars_energy_distribution_demands: list of SISEPUEDE model variables 
            to extract for use for distribution energy demands. If None, 
            defaults to 
            EnergyConsumption.modvars_enfu_energy_demands_distribution
        - modvar_energy_exports: SISEPUEDE model variable giving exports. If 
            None, default to EnergyConsumption.modvar_enfu_exports_fuel
        - modvar_import_fraction: SISEPUEDE model variable giving the import 
            fraction. If None, default to 
            EnergyConsumption.modvar_enfu_frac_fuel_demand_imported
        - target_energy_units: target energy units to convert output to. If 
            None, default to ModelAttributes.configuration energy_units.
        """

        # initialize some variables
        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
        modvars_energy_demands = self.modvars_enfu_energy_demands_total if (modvars_energy_demands is None) else modvars_energy_demands
        modvars_energy_distribution_demands = self.modvars_enfu_energy_demands_distribution if (modvars_energy_distribution_demands is None) else modvars_energy_distribution_demands
        modvar_energy_exports = self.modvar_enfu_exports_fuel if (modvar_energy_exports is None) else modvar_energy_exports
        modvar_import_fraction = self.modvar_enfu_frac_fuel_demand_imported if (modvar_import_fraction is None) else modvar_import_fraction
        
        # set energy units out
        output_energy_units = target_energy_units if (self.model_attributes.get_energy_equivalent(target_energy_units) is not None) else self.model_attributes.configuration.get("energy_units")


        ##  CALCULATE TOTAL DEMAND

        arr_demands = 0.0
        arr_demands_distribution = 0.0

        # loop over outputs from other energy sectors
        for modvar in modvars_energy_demands:

            energy_units = self.model_attributes.get_variable_characteristic(
                modvar,
                self.model_attributes.varchar_str_unit_energy
            )
            scalar = self.model_attributes.get_energy_equivalent(
                energy_units,
                output_energy_units
            )

            arr_tmp = 0.0

            # note: electricity may be missing
            try:
                arr_tmp = self.model_attributes.extract_model_variable(#
                    df_neenergy_trajectories,
                    modvar,
                    expand_to_all_cats = True,
                    return_type = "array_base",
                )

            except:
                self._log(
                    f"Warning in project_enfu_production_and_demands: Variable '{modvar}' not found in the data frame. Its fuel demands will not be included.", 
                    type_log = "warning"
                )

            arr_tmp *= scalar
            arr_demands += arr_tmp
            arr_demands_distribution += arr_tmp if (modvar in modvars_energy_distribution_demands) else 0.0


        ##  CALCULATE IMPORTS, EXPORTS, AND PRODUCTION
        
        # get import fractions and calculate imports
        arr_import_fracs = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_import_fraction,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        arr_imports = arr_import_fracs*arr_demands

        # get exports
        arr_exports = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            modvar_energy_exports,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        energy_units = self.model_attributes.get_variable_characteristic(
            modvar_energy_exports,
            self.model_attributes.varchar_str_unit_energy,
        )

        scalar = self.model_attributes.get_energy_equivalent(
            energy_units,
            output_energy_units,
        )
        arr_exports *= scalar

        # get production
        arr_production = arr_demands + arr_exports - arr_imports

        return arr_demands, arr_demands_distribution, arr_exports, arr_imports, arr_production




    ########################################
    ###                                  ###
    ###    PRIMARY PROJECTION METHODS    ###
    ###                                  ###
    ########################################

    def project_ccsq(self,
        df_neenergy_trajectories: pd.DataFrame,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
        SISEPUEDE model for Carbon Capture and Sequestration (CCSQ). Calculates  
            fuel demands required to acheieve specified sequestration targets
            and any associated combustion emissions. CCSQ does not include 
            point-of-capture CCSQ and is instead focused on scalable, industrial
            technologies like Direct Air Capture.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables

        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Notes
        -----
        If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            Energy.project()), self.model_attributes.check_projection_input_df 
            will be run

        """

        # allows production to be run outside of the project method
        if any([(x is None) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            (
                dict_dims, 
                df_neenergy_trajectories, 
                n_projection_time_periods, 
                projection_time_periods
            ) = self.model_attributes.check_projection_input_df(
                df_neenergy_trajectories, 
                True, 
                True, 
                True,
            )


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION

        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        # attribute tables
        attr_ccsq = self.model_attributes.get_attribute_table(self.subsec_name_ccsq)
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_ccsq_frac_energy = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            list(self.modvar_dict_ccsq_fuel_fractions_to_efficiency_factors.keys()),
            1,
            force_sum_equality = True,
            msg_append = "Carbon capture and sequestration heat energy fractions by category do not sum to 1. See definition of dict_arrs_ccsq_frac_energy."
        )


        ##  GET ENERGY DEMANDS

        # get sequestration totals and energy intensity, and
        arr_ccsq_demand_sequestration = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_ccsq_total_sequestration, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_ccsq_energy_intensity_sequestration = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_ccsq_demand_per_co2,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        # here, multiply by inverse (hence vars are reversed) to write intensity mass in terms of self.modvar_ccsq_total_sequestration; next, scale energy units to configuration units
        arr_ccsq_energy_intensity_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ccsq_total_sequestration,
            self.modvar_ccsq_demand_per_co2,
            "mass"
        )
        arr_ccsq_energy_intensity_sequestration *= self.model_attributes.get_scalar(self.modvar_ccsq_demand_per_co2, "energy")

        # get fraction of energy that is heat energy (from fuels) + fraction that is electric
        arr_ccsq_frac_energy_elec = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_ccsq_frac_en_electricity,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_ccsq_frac_energy_heat = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_ccsq_frac_en_heat,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        # next, use fuel mix + efficiencies to determine demands from final fuel consumption for heat energy_to_match (this will return the fractions of sequestration by consumption)
        arr_ccsq_demand_energy, dict_ccsq_demands_by_fuel_heat = self.project_energy_consumption_by_fuel_from_effvars(
            df_neenergy_trajectories,
            self.modvar_ccsq_total_sequestration,
            None, 
            None, 
            None,
            dict_arrs_ccsq_frac_energy,
        )

        fuels_loop = list(dict_ccsq_demands_by_fuel_heat.keys())
        for k in fuels_loop:
            dict_ccsq_demands_by_fuel_heat[k] = dict_ccsq_demands_by_fuel_heat[k]*arr_ccsq_energy_intensity_sequestration*arr_ccsq_frac_energy_heat

        # get electricity demand
        arr_ccsq_demand_electricity = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_ccsq_total_sequestration,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )
        arr_ccsq_demand_electricity *= arr_ccsq_frac_energy_elec*arr_ccsq_energy_intensity_sequestration



        ##  GET EMISSION FACTORS

        # methane - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_ch4 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_stationary_ch4, 
            return_type = "array_units_corrected",
        )
        arr_ccsq_ef_by_fuel_ch4 /= self.model_attributes.get_scalar(
            self.modvar_enfu_ef_combustion_stationary_ch4, 
            "energy",
        )

        # carbon dioxide - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_co2 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_co2,
            return_type = "array_units_corrected",
        )
        arr_ccsq_ef_by_fuel_co2 /= self.model_attributes.get_scalar(
            self.modvar_enfu_ef_combustion_co2, 
            "energy",
        )

        # nitrous oxide - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_n2o = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_stationary_n2o,
            return_type = "array_units_corrected",
        )
        arr_ccsq_ef_by_fuel_n2o /= self.model_attributes.get_scalar(
            self.modvar_enfu_ef_combustion_stationary_n2o, 
            "energy",
        )


        ##  CALCULATE EMISSIONS AND ELECTRICITY DEMAND

        # initialize electrical demand to pass and output emission arrays
        arr_ccsq_demand_by_fuel = np.zeros((n_projection_time_periods, len(attr_enfu.key_values)))
        arr_ccsq_demand_non_electric = 0.0
        arr_ccsq_demand_non_electric_total = 0.0
        arr_ccsq_emissions_ch4 = 0.0
        arr_ccsq_emissions_co2 = -1*arr_ccsq_demand_sequestration.transpose()
        arr_ccsq_emissions_n2o = 0.0

        # get the scalar to convert to correct output units
        scalar_ccsq_to_enfu_var_units = 1/self.model_attributes.get_scalar(self.modvar_enfu_energy_demand_by_fuel_ccsq, "energy")

        # loop over fuels to calculate demand totals
        for var_ener_frac in list(self.modvar_dict_ccsq_fuel_fractions_to_efficiency_factors.keys()):

            # retrive the fuel category and index
            cat_fuel = clean_schema(self.model_attributes.get_variable_attribute(var_ener_frac, pycat_enfu))
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # get the demand for the current fuel
            arr_ccsq_endem_cur_fuel = dict_ccsq_demands_by_fuel_heat[var_ener_frac]
            arr_ccsq_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_ccsq_endem_cur_fuel, axis = 1)*scalar_ccsq_to_enfu_var_units

            # get emissions
            arr_ccsq_emissions_ch4 += arr_ccsq_endem_cur_fuel.transpose()*arr_ccsq_ef_by_fuel_ch4[:, index_cat_fuel]
            arr_ccsq_emissions_co2 += arr_ccsq_endem_cur_fuel.transpose()*arr_ccsq_ef_by_fuel_co2[:, index_cat_fuel]
            arr_ccsq_emissions_n2o += arr_ccsq_endem_cur_fuel.transpose()*arr_ccsq_ef_by_fuel_n2o[:, index_cat_fuel]

            # add total energy demand
            if (cat_fuel != self.cat_enfu_electricity):
                arr_ccsq_demand_non_electric += arr_ccsq_endem_cur_fuel
            else:
                arr_ccsq_demand_electricity += arr_ccsq_endem_cur_fuel

        # add in total electricity demand from non-heat electric demand (overwrite since the iterator above includes it)
        index_cat_fuel = attr_enfu.get_key_value_index(self.cat_enfu_electricity)
        arr_ccsq_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_ccsq_demand_electricity, axis = 1)*scalar_ccsq_to_enfu_var_units

        # transpose outputs and prepare for output
        arr_ccsq_emissions_ch4 = arr_ccsq_emissions_ch4.transpose()
        arr_ccsq_emissions_co2 = arr_ccsq_emissions_co2.transpose()
        arr_ccsq_emissions_n2o = arr_ccsq_emissions_n2o.transpose()
        arr_ccsq_demand_non_electric_total = np.sum(arr_ccsq_demand_non_electric, axis = 1)
        arr_ccsq_demand_electricity_total = np.sum(arr_ccsq_demand_electricity, axis = 1)


        ##  ADD COSTS

        # fuel value--in terms of ccsq fuel demands
        arr_ccsq_total_fuel_value = self.get_enfu_fuel_costs_per_energy(
            df_neenergy_trajectories,
            modvar_for_units_energy = self.modvar_enfu_energy_demand_by_fuel_ccsq
        )
        


        ##  BUILD OUTPUT DATAFRAME

        df_out += [
            # CH4 EMISSIONS
            self.model_attributes.array_to_df(
                arr_ccsq_emissions_ch4, 
                self.modvar_ccsq_emissions_ch4
            ),
            # CO2 EMISSIONS
            self.model_attributes.array_to_df(
                arr_ccsq_emissions_co2, 
                self.modvar_ccsq_emissions_co2
            ),
            # N2O EMISSIONS
            self.model_attributes.array_to_df(
                arr_ccsq_emissions_n2o, 
                self.modvar_ccsq_emissions_n2o
            ),
            # TOTAL DEMAND BY FUEL IN CCSQ
            self.model_attributes.array_to_df(
                arr_ccsq_demand_by_fuel, 
                self.modvar_enfu_energy_demand_by_fuel_ccsq, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL FUEL VALUE OF FUEL CONSUMED
            self.model_attributes.array_to_df(
                arr_ccsq_total_fuel_value*arr_ccsq_demand_by_fuel, 
                self.modvar_enfu_value_of_fuel_ccsq, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # ELECTRICAL ENERGY CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_ccsq_demand_electricity, 
                self.modvar_ccsq_energy_consumption_electricity
            ),
            # AGGREGATE ELECTRICAL ENERGY CONSUMPTION
            self.model_attributes.array_to_df(
                arr_ccsq_demand_electricity_total, 
                self.modvar_ccsq_energy_consumption_electricity_agg
            ),
            # TOTAL ENERGY CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_ccsq_demand_non_electric + arr_ccsq_demand_electricity, 
                self.modvar_ccsq_energy_consumption_total
            ),
            # AGGREGATE ENERGY CONSUMPTION
            self.model_attributes.array_to_df(
                arr_ccsq_demand_non_electric_total + arr_ccsq_demand_electricity_total, 
                self.modvar_ccsq_energy_consumption_total_agg
            )
        ]


        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate"
        )
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_ccsq], False)

        return df_out



    def project_fugitive_emissions(self,
        df_neenergy_trajectories: pd.DataFrame,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
        SISEPUEDE model for Fugitive Emissions (FGTV). Calculate fugitive 
            emissions of gasses due to the production, transmission, and 
            distribution of coal, oil, and gas. Excludes process and combustion 
            emissions from mining, exploration, processing, and/or refinement of 
            these fuels, which are handled in Energy Technologies (ENTC).

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_gdp: np.ndarray vector of gdp (requires 
            len(vec_gdp) == len(df_neenergy_trajectories))

        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Notes
        -----
        This is the final model projected in the SISEPUEDE DAG as it depends on 
            all other energy models to determine mining production.

        If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            Energy.project()), self.model_attributes.check_projection_input_df 
            wil be run
        """

        # allows production to be run outside of the project method
        if any([(x is None) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            (
                dict_dims, 
                df_neenergy_trajectories, 
                n_projection_time_periods, 
                projection_time_periods
            ) = self.model_attributes.check_projection_input_df(
                df_neenergy_trajectories, 
                True, 
                True, 
                True,
            )


        ## ATTRIBUTE INITIALIZATION
        
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        attr_fgtv = self.model_attributes.get_attribute_table(self.subsec_name_fgtv)
        attr_inen = self.model_attributes.get_attribute_table(self.subsec_name_inen)
        attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)
        

        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################
         
        # get demands, exports, imports, and production, either from EnergyProduction or from the rest of the energy sectors
        (
            arr_fgtv_demands, 
            arr_demands_distribution, 
            arr_fgtv_export, 
            arr_fgtv_imports, 
            arr_fgtv_production,
            df_out
        ) = self.get_fgtv_demands_and_trade(df_neenergy_trajectories)

        # initialize the output - if demands are from EnergyProduction, df_out is None, so will disappear on pd.concat(); otherwise, sets those output variables
        df_out = [df_out]
        
        """
        # HERE--DEMANDS WILL HAVE TO COME FROM EnergyProduction
        # get all demands, imports, exports, and production in terms of configuration units
        arr_fgtv_demands, arr_demands_distribution, arr_fgtv_export, arr_fgtv_imports, arr_fgtv_production = self.project_enfu_production_and_demands(
            df_neenergy_trajectories
        )
        """;

        # define a dictionary to relate aggregate emissions to the components
        dict_emission_to_fugitive_components = {
            self.modvar_fgtv_emissions_ch4: {
                "distribution": self.modvar_fgtv_ef_ch4_distribution,
                "production_flaring": self.modvar_fgtv_ef_ch4_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_ch4_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_ch4_production_venting,
                "transmission": self.modvar_fgtv_ef_ch4_transmission
            },

            self.modvar_fgtv_emissions_co2: {
                "distribution": self.modvar_fgtv_ef_co2_distribution,
                "production_flaring":self.modvar_fgtv_ef_co2_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_co2_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_co2_production_venting,
                "transmission": self.modvar_fgtv_ef_co2_transmission
            },

            self.modvar_fgtv_emissions_n2o: {
                "distribution": None,
                "production_flaring": self.modvar_fgtv_ef_n2o_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_n2o_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_n2o_production_venting,
                "transmission": self.modvar_fgtv_ef_n2o_transmission
            },

            self.modvar_fgtv_emissions_nmvoc: {
                "distribution": self.modvar_fgtv_ef_nmvoc_distribution,
                "production_flaring": self.modvar_fgtv_ef_nmvoc_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_nmvoc_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_nmvoc_production_venting,
                "transmission": self.modvar_fgtv_ef_nmvoc_transmission
            }
        }

        # initialiize some shared data
        arr_enfu_energy_density_volumetric = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_energy_density_volumetric,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        arr_fgtv_frac_vent_to_flare = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_fgtv_frac_non_fugitive_flared,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        vec_fgtv_reduction_leaks = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_fgtv_frac_reduction_fugitive_leaks,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_enfu_zeros = np.zeros((len(df_neenergy_trajectories), attr_enfu.n_key_values))


        ##  LOOP OVER OUTPUT EMISSIONS TO GENERATE EMISSIONS

        for modvar_emission in dict_emission_to_fugitive_components.keys():

            # get the key emission factor arrays in terms of mass/energy
            arr_ef_distribution = self.get_fgtv_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["distribution"],
                arr_enfu_energy_density_volumetric
            )

            # production - flaring
            arr_ef_production_flaring = self.get_fgtv_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_flaring"],
                arr_enfu_energy_density_volumetric
            )

            # production - fugitive/leaks
            arr_ef_production_fugitive = self.get_fgtv_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_fugitive"],
                arr_enfu_energy_density_volumetric
            )

            # production - venting
            arr_ef_production_venting = self.get_fgtv_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_venting"],
                arr_enfu_energy_density_volumetric
            )

            # production - transmission
            arr_ef_transmission = self.get_fgtv_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["transmission"],
                arr_enfu_energy_density_volumetric
            )

            # weighted emission factor for tradeoff from flare to vent; 
            # note that categories for which arr_fgtv_frac_vent_to_flare is not 
            # defined have the arr_fgtv_frac_vent_to_flare = 1 (so that 
            # everything goes to flaring)
            arr_fgtv_ef_fv_flare = (
                arr_fgtv_frac_vent_to_flare*arr_ef_production_flaring 
                if (arr_ef_production_flaring is not None) 
                else 0.0
            )

            arr_fgtv_ef_fv_vent = (
                (1 - arr_fgtv_frac_vent_to_flare)*arr_ef_production_venting 
                if (arr_ef_production_flaring is not None) 
                else 0.0
            )

            arr_fgtv_ef_fv = arr_fgtv_ef_fv_flare + arr_fgtv_ef_fv_vent
            arr_fgtv_ef_fv += sf.do_array_mult(arr_ef_production_fugitive, 1 - vec_fgtv_reduction_leaks)
            
            # distribution, production, and transmission emissions
            arr_fgtv_emit_distribution = (
                sf.do_array_mult(arr_demands_distribution*arr_ef_distribution, 1 - vec_fgtv_reduction_leaks) 
                if (arr_ef_distribution is not None) 
                else arr_enfu_zeros
            )

            arr_fgtv_emit_production = arr_fgtv_production*arr_fgtv_ef_fv

            arr_fgtv_emit_transmission = (
                sf.do_array_mult(
                    arr_ef_transmission*(arr_fgtv_production + arr_fgtv_imports), 
                    1 - vec_fgtv_reduction_leaks
                ) 
                if (arr_ef_transmission is not None) 
                else arr_enfu_zeros
            )

            # get total and determine scalar
            arr_fgtv_emissions_cur = arr_fgtv_emit_distribution + arr_fgtv_emit_production + arr_fgtv_emit_transmission
            emission = self.model_attributes.get_variable_characteristic(
                modvar_emission,
                self.model_attributes.varchar_str_emission_gas,
            )

            arr_fgtv_emissions_cur *= (
                1/self.model_attributes.get_scalar(modvar_emission, "mass") 
                if (emission is None) 
                else 1
            )

            df_out.append(
                self.model_attributes.array_to_df(
                    arr_fgtv_emissions_cur,
                    modvar_emission,
                    include_scalars = False,
                    reduce_from_all_cats_to_specified_cats = True,
                )
            )

        """
        # set additional output
        arr_fgtv_imports /= self.model_attributes.get_scalar(self.modvar_enfu_imports_fuel, "energy")
        arr_fgtv_production /= self.model_attributes.get_scalar(self.modvar_enfu_production_fuel, "energy")

        df_out += [
            self.model_attributes.array_to_df(
                arr_fgtv_demands, self.modvar_enfu_energy_demand_by_fuel_total, reduce_from_all_cats_to_specified_cats = True
            ),
            self.model_attributes.array_to_df(
                arr_fgtv_imports, self.modvar_enfu_imports_fuel, reduce_from_all_cats_to_specified_cats = True
            ),
            self.model_attributes.array_to_df(
                arr_fgtv_production, self.modvar_enfu_production_fuel, reduce_from_all_cats_to_specified_cats = True
            )
        ]
        """;
    
        global df_out_exp
        df_out_exp = df_out

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate"
        )
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_fgtv], False)

        return df_out



    def project_industrial_energy(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None,
    ) -> pd.DataFrame:

        """
        SISEPUEDE model for Industrial Energy (INEN), which calculates emissions
            from fuel combustion and energy use arising from industrial 
            production and activities. Excludes energy industries, which are
            handled in Energy Technologies (ENTC).

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_gdp: np.ndarray vector of gdp (requires 
            len(vec_gdp) == len(df_neenergy_trajectories))

        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Notes
        -----
        If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            Energy.project()), self.model_attributes.check_projection_input_df 
            wil be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        # attribute tables
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        attr_inen = self.model_attributes.get_attribute_table(self.subsec_name_inen)
        attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)

        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_inen_frac_energy = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            self.modvars_inen_list_fuel_fraction,
            1,
            force_sum_equality = True,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_inen_frac_energy."
        )


        ##  GET ENERGY INTENSITIES

        # get production-based emissions - start with industrial production, energy demand
        arr_inen_prod = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_ippu.modvar_ippu_qty_total_production, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_inen_prod_energy_intensity = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_inen_en_prod_intensity_factor, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        # get agricultural and livestock production + intensities (in terms of self.model_ippu.modvar_ippu_qty_total_production (mass) and self.modvar_inen_en_prod_intensity_factor (energy), respectively)
        index_inen_agrc, vec_inen_energy_intensity_agrc_lvst, vec_inen_prod_agrc_lvst = self.get_agrc_lvst_prod_and_intensity(df_neenergy_trajectories)
        arr_inen_prod[:, index_inen_agrc] += vec_inen_prod_agrc_lvst

        # build dictionary for projection 
        dict_inen_fuel_frac_to_eff_cat = self.dict_inen_fuel_categories_to_fuel_variables.copy()
        for k in dict_inen_fuel_frac_to_eff_cat.keys():
            val = dict_inen_fuel_frac_to_eff_cat[k]["fuel_fraction"]
            dict_inen_fuel_frac_to_eff_cat.update({k: val})
        dict_inen_fuel_frac_to_eff_cat = sf.reverse_dict(dict_inen_fuel_frac_to_eff_cat)

        # energy consumption at time 0 due to production in terms of units modvar_ippu_qty_total_production
        arr_inen_energy_consumption_intensity_prod = arr_inen_prod_energy_intensity*self.model_attributes.get_variable_unit_conversion_factor(
            self.model_ippu.modvar_ippu_qty_total_production,
            self.modvar_inen_en_prod_intensity_factor,
            "mass"
        )

        # NOTE: add vec_inen_energy_intensity_agrc_lvst here because its mass is already in terms of self.model_ippu.modvar_ippu_qty_total_production
        arr_inen_energy_consumption_intensity_prod[:, index_inen_agrc] += vec_inen_energy_intensity_agrc_lvst

        # project future consumption
        arr_inen_demand_energy_prod, dict_inen_energy_consumption_prod = self.project_energy_consumption_by_fuel_from_fuel_cats(
            df_neenergy_trajectories,
            arr_inen_energy_consumption_intensity_prod[0],
            arr_inen_prod,
            self.modvar_inen_demscalar,
            self.modvar_enfu_efficiency_factor_industrial_energy,
            dict_arrs_inen_frac_energy,
            dict_inen_fuel_frac_to_eff_cat
        )

        # gdp-based emissions - get intensity, multiply by gdp, and scale to match energy units of production
        arr_inen_energy_consumption_intensity_gdp = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_inen_en_gdp_intensity_factor,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_inen_energy_consumption_intensity_gdp *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_inen_en_gdp_intensity_factor,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        ) 

        arr_inen_demand_energy_gdp, dict_inen_energy_consumption_gdp = self.project_energy_consumption_by_fuel_from_fuel_cats(
            df_neenergy_trajectories,
            arr_inen_energy_consumption_intensity_gdp[0],
            vec_gdp,
            self.modvar_inen_demscalar,
            self.modvar_enfu_efficiency_factor_industrial_energy,
            dict_arrs_inen_frac_energy,
            dict_inen_fuel_frac_to_eff_cat
        )

        # build aggregate consumption by fuel (in energy terms of modvar_inen_en_prod_intensity_factor)
        dict_inen_energy_consumption = {}
        for k in dict_inen_energy_consumption_prod.keys():
            dict_inen_energy_consumption[k] = dict_inen_energy_consumption_gdp[k] + dict_inen_energy_consumption_prod[k]


        ##  GET EMISSION FACTORS

        # methane - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_ch4 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_stationary_ch4, 
            return_type = "array_units_corrected",
        
        )
        arr_inen_ef_by_fuel_ch4 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_stationary_ch4,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        )

        # carbon dioxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_co2 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_co2, 
            return_type = "array_units_corrected",
        )

        arr_inen_ef_by_fuel_co2 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_co2,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        )

        # nitrous oxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_n2o = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_stationary_n2o, 
            return_type = "array_units_corrected",
        )

        arr_inen_ef_by_fuel_n2o *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_stationary_n2o,
            self.modvar_inen_en_prod_intensity_factor,
            "energy",
        )


        ##  CALCULATE EMISSIONS AND ELECTRICITY DEMAND

        # initialize electrical demand to pass and output emission arrays
        arr_inen_demand_by_fuel = np.zeros((n_projection_time_periods, len(attr_enfu.key_values)))
        arr_inen_demand_electricity = 0.0
        arr_inen_demand_electricity_total = 0.0
        arr_inen_demand_total = 0.0
        arr_inen_demand_total_total = 0.0
        arr_inen_emissions_ch4 = 0.0
        arr_inen_emissions_co2 = 0.0
        arr_inen_emissions_n2o = 0.0
        # set scalar to convert to enfu units
        scalar_inen_to_enfu_var_units = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_inen_en_prod_intensity_factor,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            "energy"
        )

        # loop over fuels to
        for var_ener_frac in self.modvars_inen_list_fuel_fraction:
            
            # retrive the fuel category and index
            cat_fuel = clean_schema(self.model_attributes.get_variable_attribute(var_ener_frac, pycat_enfu))
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # get the demand for the current fuel
            arr_inen_endem_cur_fuel = dict_inen_energy_consumption[var_ener_frac].copy()

            # get emissions from factors
            arr_inen_emissions_ch4 += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_ch4[:, index_cat_fuel]
            arr_inen_emissions_co2 += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_co2[:, index_cat_fuel]
            arr_inen_emissions_n2o += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_n2o[:, index_cat_fuel]

            # update the demand for fuel in the output array
            arr_inen_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_inen_endem_cur_fuel, axis = 1)

            # add electricity demand and total energy demand
            arr_inen_demand_electricity += arr_inen_endem_cur_fuel if (cat_fuel == self.cat_enfu_electricity) else 0.0
            arr_inen_demand_electricity_total += arr_inen_endem_cur_fuel.sum(axis = 1) if (cat_fuel == self.cat_enfu_electricity) else 0.0
            arr_inen_demand_total += arr_inen_endem_cur_fuel
            arr_inen_demand_total_total += arr_inen_endem_cur_fuel.sum(axis = 1)

        # transpose outputs
        arr_inen_emissions_ch4 = arr_inen_emissions_ch4.transpose()
        arr_inen_emissions_co2 = arr_inen_emissions_co2.transpose()
        arr_inen_emissions_n2o = arr_inen_emissions_n2o.transpose()

        # get scalar to transform units of self.modvar_inen_en_prod_intensity_factor -> configuration units
        scalar_energy = self.model_attributes.get_scalar(self.modvar_inen_en_prod_intensity_factor, "energy")

        
        ##  ADD IN POINT OF CAPTURE

        # fraction captured is prevalence * efficacy (both specified in IPPU)
        array_inen_emission_frac_captured = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_ippu.modvar_ippu_capture_prevalence_co2, 
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        array_inen_emission_frac_captured *= self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_ippu.modvar_ippu_capture_efficacy_co2, 
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        
        # capture and apply the scalar for output units
        array_inen_emissions_co2_captured = arr_inen_emissions_co2*array_inen_emission_frac_captured
        arr_inen_emissions_co2 -= array_inen_emissions_co2_captured
        scalar_captured = self.model_attributes.get_scalar(self.modvar_inen_gas_captured_co2, "mass")
        array_inen_emissions_co2_captured /= scalar_captured


        ##  ADD COSTS

        # total fuel value per unit of energy
        arr_inen_total_fuel_value = self.get_enfu_fuel_costs_per_energy(
            df_neenergy_trajectories,
            modvar_for_units_energy = self.modvar_enfu_energy_demand_by_fuel_inen
        )

        

        ##  BUILD OUTPUT DFs

        df_out += [
            # CH4 EMISSIONS
            self.model_attributes.array_to_df(
                arr_inen_emissions_ch4, 
                self.modvar_inen_emissions_ch4, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # CO2 EMISSIONS
            self.model_attributes.array_to_df(
                arr_inen_emissions_co2, 
                self.modvar_inen_emissions_co2, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # N2O EMISSIONS
            self.model_attributes.array_to_df(
                arr_inen_emissions_n2o, 
                self.modvar_inen_emissions_n2o, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # CO2 CAPTURED
            self.model_attributes.array_to_df(
                array_inen_emissions_co2_captured, 
                self.modvar_inen_gas_captured_co2, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # ENERGY DEMAND BY FUEL
            self.model_attributes.array_to_df(
                arr_inen_demand_by_fuel*scalar_inen_to_enfu_var_units, 
                self.modvar_enfu_energy_demand_by_fuel_inen, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # FUEL VALUE
            self.model_attributes.array_to_df(
                arr_inen_total_fuel_value*arr_inen_demand_by_fuel*scalar_inen_to_enfu_var_units, 
                self.modvar_enfu_value_of_fuel_inen, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # ELECTRICAL ENERGY CONSUMPTION
            self.model_attributes.array_to_df(
                arr_inen_demand_electricity*scalar_energy, 
                self.modvar_inen_energy_consumption_electricity, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL HEAT ENERGY CONSUMPTION BY CATEGORY (AGGREGATED)
            self.model_attributes.array_to_df(
                arr_inen_demand_electricity_total*scalar_energy, 
                self.modvar_inen_energy_consumption_electricity_agg
            ),
            # TOTAL CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_inen_demand_total*scalar_energy,
                self.modvar_inen_energy_consumption_total, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL CONSUMPTION BY CATEGORY, AGGREGATED
            self.model_attributes.array_to_df(
                arr_inen_demand_total_total*scalar_energy, 
                self.modvar_inen_energy_consumption_total_agg
            ),
            # POINT OF USE DEMAND BY CATEGORY
            self.model_attributes.array_to_df(
                (arr_inen_demand_energy_prod + arr_inen_demand_energy_gdp)*scalar_energy, 
                self.modvar_inen_energy_demand_total, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate"
        )
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_inen], False)

        return df_out



    def project_scoe(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_hh: np.ndarray,
        vec_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
        SISEPUEDE model for Stationary Combustion and Other Energy (SCOE),
            Stationary combustion primarily occurs in buildings. SCOE also
            allows for other energy exogenously specified energy emissions
            unaccounted for elsewhere.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_hh: np.ndarray vector of number of households (requires
            len(vec_hh) == len(df_neenergy_trajectories))
        - vec_gdp: np.ndarray vector of gdp (requires
            len(vec_gdp) == len(df_neenergy_trajectories))
        - vec_rates_gdp_per_capita: np.ndarray vector of growth rates in
            gdp/capita (requires
            en(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)
        - dict_dims: dict of dimensions (returned from
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from
            check_projection_input_df). Default is None.

        Notes
        -----
        If any of dict_dims, n_projection_time_periods, or
            projection_time_periods are unspecified (expected if ran outside of
            Energy.project()), self.model_attributes.check_projection_input_df
            will be run
        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        
        pycat_enfu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_enfu, 
            "pycategory_primary_element",
        )

        # attribute tables
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        attr_scoe = self.model_attributes.get_attribute_table(self.subsec_name_scoe)


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_scoe_frac_energy = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            list(self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors.keys()),
            1,
            force_sum_equality = True,
            msg_append = "SCOE heat energy fractions by category do not sum to 1. See definition of dict_arrs_scoe_frac_energy."
        )


        ##  GET ENERGY DEMANDS

        # get initial per-activity demands (can use to get true demands)
        arr_scoe_deminit_hh_elec = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_consumpinit_energy_per_hh_elec, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_scoe_deminit_hh_heat = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_consumpinit_energy_per_hh_heat, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_scoe_deminit_mmmgdp_elec = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_consumpinit_energy_per_mmmgdp_elec, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_scoe_deminit_mmmgdp_heat = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_consumpinit_energy_per_mmmgdp_heat, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )
        
        # get elasticities
        arr_scoe_enerdem_elasticity_hh_elec = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )
        
        arr_scoe_enerdem_elasticity_hh_heat = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_scoe_enerdem_elasticity_mmmgdp_elec = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_scoe_enerdem_elasticity_mmmgdp_heat = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )
        
        # get demand for electricity for households 
        arr_scoe_growth_demand_hh_elec = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp_per_capita, arr_scoe_enerdem_elasticity_hh_elec, 
            False, 
            "standard",
        )
        arr_scoe_demand_hh_elec = sf.do_array_mult(
            arr_scoe_deminit_hh_elec[0]*arr_scoe_growth_demand_hh_elec, 
            vec_hh
        )
        arr_scoe_demand_hh_elec *= self.model_attributes.get_scalar(
            self.modvar_scoe_consumpinit_energy_per_hh_elec, 
            "energy",
        )
        
        # get demand for electricity driven by GDP
        arr_scoe_growth_demand_mmmgdp_elec = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp_per_capita, 
            arr_scoe_enerdem_elasticity_hh_elec, 
            False, 
            "standard",
        )
        arr_scoe_demand_mmmgdp_elec = sf.do_array_mult(
            arr_scoe_deminit_mmmgdp_elec[0]*arr_scoe_growth_demand_mmmgdp_elec, 
            vec_gdp,
        )
        arr_scoe_demand_mmmgdp_elec *= self.model_attributes.get_scalar(
            self.modvar_scoe_consumpinit_energy_per_mmmgdp_elec, 
            "energy",
        )
        
        # get demand scalars
        arr_scoe_demscalar_elec_energy_demand = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_scoe_demscalar_elec_energy_demand,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        arr_scoe_demscalar_heat_energy_demand = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_scoe_demscalar_heat_energy_demand,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        # next, use fuel mix + efficiencies to determine demands from final fuel consumption for heat energy_to_match
        arr_scoe_demand_heat_energy_hh, dict_scoe_demands_by_fuel_heat_hh = self.project_energy_consumption_by_fuel_from_effvars(
            df_neenergy_trajectories,
            self.modvar_scoe_consumpinit_energy_per_hh_heat,
            vec_hh,
            arr_scoe_enerdem_elasticity_hh_heat,
            vec_rates_gdp_per_capita,
            dict_arrs_scoe_frac_energy
        )
        arr_scoe_demand_heat_energy_mmgdp, dict_scoe_demands_by_fuel_heat_mmmgdp = self.project_energy_consumption_by_fuel_from_effvars(
            df_neenergy_trajectories,
            self.modvar_scoe_consumpinit_energy_per_mmmgdp_heat,
            vec_gdp,
            arr_scoe_enerdem_elasticity_mmmgdp_heat,
            vec_rates_gdp_per_capita,
            dict_arrs_scoe_frac_energy
        )

        # get total demands by fuel
        dict_demands_by_fuel_heat = {}
        for k in list(set(dict_scoe_demands_by_fuel_heat_hh.keys()) & set(dict_scoe_demands_by_fuel_heat_mmmgdp.keys())):
            arr_tmp_demands_total = dict_scoe_demands_by_fuel_heat_hh[k] + dict_scoe_demands_by_fuel_heat_mmmgdp[k]
            arr_tmp_demands_total *= arr_scoe_demscalar_heat_energy_demand
            dict_demands_by_fuel_heat.update({k: arr_tmp_demands_total})
            

        ##  GET EMISSION FACTORS

        # methane - scale to ensure energy units are the same
        arr_scoe_ef_by_fuel_ch4 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_stationary_ch4,
            return_type = "array_units_corrected",
        )
        arr_scoe_ef_by_fuel_ch4 /= self.model_attributes.get_scalar(
            self.modvar_enfu_ef_combustion_stationary_ch4, 
            "energy"
        )

        # carbon dioxide - scale to ensure energy units are the same
        arr_scoe_ef_by_fuel_co2 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_co2,
            return_type = "array_units_corrected",
        )
        arr_scoe_ef_by_fuel_co2 /= self.model_attributes.get_scalar(
            self.modvar_enfu_ef_combustion_co2, 
            "energy"
        )

        # nitrous oxide - scale to ensure energy units are the same
        arr_scoe_ef_by_fuel_n2o = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_stationary_n2o,
            return_type = "array_units_corrected",
        )
        arr_scoe_ef_by_fuel_n2o /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_stationary_n2o, "energy")


        ##  CALCULATE EMISSIONS AND ELECTRICITY DEMAND

        # initialize electrical demand to pass and output emission arrays
        arr_scoe_demand_by_fuel = np.zeros((n_projection_time_periods, len(attr_enfu.key_values)))
        arr_scoe_demand_electricity = arr_scoe_demand_hh_elec + arr_scoe_demand_mmmgdp_elec
        arr_scoe_demand_electricity *= arr_scoe_demscalar_elec_energy_demand
        arr_scoe_demand_non_electric = 0.0
        arr_scoe_demand_non_electric_total = 0.0
        arr_scoe_emissions_ch4 = 0.0
        arr_scoe_emissions_co2 = 0.0
        arr_scoe_emissions_n2o = 0.0

        # initialize the scalar to convert energy units to ENFU output
        scalar_scoe_to_enfu_var_units = 1/self.model_attributes.get_scalar(self.modvar_enfu_energy_demand_by_fuel_scoe, "energy")

        # loop over fuels to calculate demand totals
        for var_ener_frac in list(self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors.keys()):
            
            # retrive the fuel category and index
            cat_fuel = clean_schema(self.model_attributes.get_variable_attribute(var_ener_frac, pycat_enfu))
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # get the demand for the current fuel
            arr_scoe_endem_cur_fuel = dict_demands_by_fuel_heat.get(var_ener_frac)
            arr_scoe_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_scoe_endem_cur_fuel, axis = 1)*scalar_scoe_to_enfu_var_units

            # apply emission factors
            arr_scoe_emissions_ch4 += arr_scoe_endem_cur_fuel.transpose()*arr_scoe_ef_by_fuel_ch4[:, index_cat_fuel]
            arr_scoe_emissions_co2 += arr_scoe_endem_cur_fuel.transpose()*arr_scoe_ef_by_fuel_co2[:, index_cat_fuel]
            arr_scoe_emissions_n2o += arr_scoe_endem_cur_fuel.transpose()*arr_scoe_ef_by_fuel_n2o[:, index_cat_fuel]

            # add electricity demand and total energy demand
            if (cat_fuel == self.cat_enfu_electricity):
                arr_scoe_demand_by_fuel[:, index_cat_fuel] += np.sum(arr_scoe_demand_electricity, axis = 1)*scalar_scoe_to_enfu_var_units
                arr_scoe_demand_electricity += arr_scoe_endem_cur_fuel
            else:
                arr_scoe_demand_non_electric += arr_scoe_endem_cur_fuel

        # transpose outputs
        arr_scoe_emissions_ch4 = arr_scoe_emissions_ch4.transpose()
        arr_scoe_emissions_co2 = arr_scoe_emissions_co2.transpose()
        arr_scoe_emissions_n2o = arr_scoe_emissions_n2o.transpose()
        # get some totals
        arr_scoe_demand_electricity_total = np.sum(arr_scoe_demand_electricity, axis = 1)
        arr_scoe_demand_non_electric_total = np.sum(arr_scoe_demand_non_electric, axis = 1)


        ##  GET COSTS

        # total fuel value per unit of energy
        arr_scoe_total_fuel_value = self.get_enfu_fuel_costs_per_energy(
            df_neenergy_trajectories,
            modvar_for_units_energy = self.modvar_enfu_energy_demand_by_fuel_scoe
        )


            
        ##  BUILD OUTPUT DFs

        df_out += [
            # CH4 EMISSIONS
            self.model_attributes.array_to_df(
                arr_scoe_emissions_ch4, 
                self.modvar_scoe_emissions_ch4
            ),
            # CO2 EMISSIONS
            self.model_attributes.array_to_df(
                arr_scoe_emissions_co2, 
                self.modvar_scoe_emissions_co2
            ),
            # N2O EMISSIONS
            self.model_attributes.array_to_df(
                arr_scoe_emissions_n2o, 
                self.modvar_scoe_emissions_n2o
            ),
            # DEMAND BY FUEL
            self.model_attributes.array_to_df(
                arr_scoe_demand_by_fuel, 
                self.modvar_enfu_energy_demand_by_fuel_scoe, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # FUEL VALUE
            self.model_attributes.array_to_df(
                arr_scoe_total_fuel_value*arr_scoe_demand_by_fuel, 
                self.modvar_enfu_value_of_fuel_scoe, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # ELECTRICITY CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_scoe_demand_electricity, 
                self.modvar_scoe_energy_consumption_electricity
            ),
            # ELECTRICITY CONSUMPTION (AGGREGATE)
            self.model_attributes.array_to_df(
                arr_scoe_demand_electricity_total, 
                self.modvar_scoe_energy_consumption_electricity_agg
            ),
            # TOTAL ENERGY CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_scoe_demand_non_electric + arr_scoe_demand_electricity, 
                self.modvar_scoe_energy_consumption_total
            ),
            # TOTAL ENERGY CONSUMPTION (AGGREGATE)
            self.model_attributes.array_to_df(
                arr_scoe_demand_non_electric_total + arr_scoe_demand_electricity_total, 
                self.modvar_scoe_energy_consumption_total_agg
            ),
            # POINT OF USE DEMAND BY CATEGORY
            self.model_attributes.array_to_df(
                arr_scoe_demand_heat_energy_hh + arr_scoe_demand_heat_energy_mmgdp, 
                self.modvar_scoe_energy_demand_heat_total
            )
        ]


        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate"
        )
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_scoe], False)

        return df_out



    def project_transportation(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_pop: np.ndarray,
        vec_rates_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
        Calculate emissions from fuel combustion in TRNS (Transportation). 
            Requires EnergyConsumption.project_transportation_demand() and all 
            output variables from TRDE (Transportation Demand) subsector.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_pop: np.ndarray vector of population (requires 
            len(vec_rates_gdp) == len(df_neenergy_trajectories))
        - vec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth 
            rate from t_i to t_{i + 1}) (requires 
            len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)
        - vec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth 
            rates (v_i = growth rate from t_i to t_{i + 1}) (requires 
            len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)
        
        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Notes
        -----
        If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            Energy.project()), self.model_attributes.check_projection_input_df 
            will be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)

        # check fields - transportation demand; if not present, add to the dataframe
        self.check_df_fields(df_neenergy_trajectories, self.subsec_name_trns)
        df_transport_demand = None
        append_trde_outputs = False

        try:
            self.check_df_fields(
                df_neenergy_trajectories,
                self.subsec_name_trde,
                "output",
                self.subsec_name_trns
            )
        except:
            df_transport_demand = self.project_transportation_demand(
                df_neenergy_trajectories,
                vec_pop,
                vec_rates_gdp,
                vec_rates_gdp_per_capita,
                dict_dims,
                n_projection_time_periods,
                projection_time_periods
            )

            df_neenergy_trajectories = sf.merge_output_df_list(
                [
                    df_neenergy_trajectories,
                    df_transport_demand
                ],
                self.model_attributes,
                merge_type = "concatenate"
            )

            append_trde_outputs = True


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_trde = self.model_attributes.get_subsector_attribute(
            self.subsec_name_trde, 
            "pycategory_primary_element",
        )

        # attribute tables
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        attr_trde = self.model_attributes.get_attribute_table(self.subsec_name_trde)
        attr_trns = self.model_attributes.get_attribute_table(self.subsec_name_trns)


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]

        # add transportation demand to outputs if necessary
        if append_trde_outputs:
            df_out += [
                self.model_attributes.extract_model_variable(#
                    df_transport_demand,
                    self.modvar_trde_demand_mtkm,
                    expand_to_all_cats = False,
                    return_type = "data_frame",
                ),

                self.model_attributes.extract_model_variable(#
                    df_transport_demand,
                    self.modvar_trde_demand_pkm,
                    return_type = "data_frame",
                )
            ]



        ############################
        #    MODEL CALCULATIONS    #
        ############################


        ##  START WITH DEMANDS

        # start with demands and map categories in attribute to associated variable
        dict_trde_cats_to_trns_vars = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_trns, 
            pycat_trde, 
            attr_type = "variable_definitions",
            clean_attribute_schema_q = True,
            return_type = dict, 
            skip_none_q = True,
        )

        dict_trde_cats_to_trns_vars = sf.reverse_dict(dict_trde_cats_to_trns_vars)
        array_trns_total_mass_distance_demand = 0.0
        array_trns_total_passenger_demand = 0.0
        array_trns_total_vehicle_demand = 0.0

        # get occupancy and freight occupancies
        array_trns_avg_load_freight = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_trns_average_vehicle_load_freight,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        array_trns_occ_rate_passenger = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories,
            self.modvar_trns_average_passenger_occupancy,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # convert average load to same units as demand
        array_trns_avg_load_freight *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trns_average_vehicle_load_freight,
            self.modvar_trde_demand_mtkm,
            "mass",
        )

        # convert freight vehicle demand to same length units as passenger
        scalar_tnrs_length_demfrieght_to_dempass = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trde_demand_mtkm,
            self.modvar_trde_demand_pkm,
            "length",
        )

        # loop over the demand categories to get transportation demand
        for category in dict_trde_cats_to_trns_vars.keys():

            # get key index, model variable, 
            index_key = (
                self.model_attributes
                .get_attribute_table(self.subsec_name_trde)
                .get_key_value_index(category)
            )
            
            modvar = self.model_attributes.get_variable_from_category(
                self.subsec_name_trde, 
                category, 
            )
            scalar_length = self.model_attributes.get_scalar(modvar, "length")

            # retrieve demand
            vec_trde_dem_cur = self.model_attributes.extract_model_variable(#
                df_neenergy_trajectories, 
                modvar, 
                expand_to_all_cats = True,
                return_type = "array_base",
            )[:, index_key]

            # retrieve the demand mix, convert to total activity-demand by category, then divide by freight/occ_rate
            array_trde_dem_cur_by_cat = self.model_attributes.extract_model_variable(#
                df_neenergy_trajectories,
                dict_trde_cats_to_trns_vars.get(category),
                expand_to_all_cats = True,
                force_boundary_restriction = True,
                return_type = "array_base",
                var_bounds = (0, 1),
            )
           
            # get current demand in terms of configuration units of length (across each model variable)
            array_trde_dem_cur_by_cat = sf.do_array_mult(
                array_trde_dem_cur_by_cat,
                vec_trde_dem_cur
            )*scalar_length

            # update vehicle demand by category and total mass distance
            if category == self.cat_trde_frgt:
                array_trde_vehicle_dem_cur_by_cat = np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_avg_load_freight, 0.0, neginf = 0.0, posinf = 0.0)*scalar_tnrs_length_demfrieght_to_dempass
                array_trns_total_mass_distance_demand += array_trde_dem_cur_by_cat
            else:
                array_trde_vehicle_dem_cur_by_cat = np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_occ_rate_passenger, 0.0, neginf = 0.0, posinf = 0.0)
                array_trns_total_passenger_demand += array_trde_dem_cur_by_cat
            
            # add in total vehicle-km demand
            array_trns_total_vehicle_demand += array_trde_vehicle_dem_cur_by_cat

        # add the vehicle and passenger distance to output using the units modvar_trde_demand_pkm
        scalar_trns_total_mass_distance_demand = 1/self.model_attributes.get_scalar(self.modvar_trns_mass_distance_traveled, "length")
        scalar_trns_total_mass_distance_demand *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trde_demand_mtkm,
            self.modvar_trns_mass_distance_traveled,
            "mass"
        )

        df_out += [
            # MASS-DISTANCE TRAVELED
            self.model_attributes.array_to_df(
                array_trns_total_mass_distance_demand*scalar_trns_total_mass_distance_demand,
                self.modvar_trns_mass_distance_traveled,
                reduce_from_all_cats_to_specified_cats = True
            ),
            # PASSENGER DISTANCE TRAVELED
            self.model_attributes.array_to_df(
                array_trns_total_passenger_demand,
                self.modvar_trns_passenger_distance_traveled,
                reduce_from_all_cats_to_specified_cats = True
            ),
            # VEHICLE DISTANCE TRAVELED
            self.model_attributes.array_to_df(
                array_trns_total_vehicle_demand,
                self.modvar_trns_vehicle_distance_traveled,
                reduce_from_all_cats_to_specified_cats = True
            )
        ]

        # convert length in to terms of self.modvar_trde_demand_pkm
        array_trns_total_vehicle_demand /= self.model_attributes.get_scalar(self.modvar_trde_demand_pkm, "length")


        ##  LOOP OVER FUELS

        # first, retrieve fuel-mix fractions and ensure they sum to 1
        dict_arrs_trns_frac_fuel = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            self.modvars_trns_list_fuel_fraction,
            1,
            force_sum_equality = False,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_trns_frac_fuel.",
        )

        # get carbon dioxide combustion factors (corrected to output units)
        arr_trns_ef_by_fuel_co2 = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_ef_combustion_co2,
            expand_to_all_cats = True,
            return_type = "array_units_corrected",
        )

        arr_trns_energy_density_fuel = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_enfu_energy_density_volumetric,
            expand_to_all_cats = True,
            return_type = "array_units_corrected",
        )

        # initialize electrical demand to pass and output emission arrays
        arr_trns_demand_by_category = 0.0
        arr_trns_demand_by_fuel = np.zeros((n_projection_time_periods, len(attr_enfu.key_values)))
        arr_trns_demand_electricity = 0.0
        arr_trns_demand_electricity_total = 0.0
        arr_trns_emissions_ch4 = 0.0
        arr_trns_emissions_co2 = 0.0
        arr_trns_emissions_n2o = 0.0

        # get conversion scalars
        scalar_trns_ved_to_enfu_var_units = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_energy_density_volumetric,
            self.modvar_enfu_energy_demand_by_fuel_trns,
            "energy"
        )
        scalar_trns_ved_to_output_scalar = self.model_attributes.get_scalar(
            self.modvar_enfu_energy_density_volumetric, 
            "energy"
        )

        # loop over fuels to calculate emissions and demand associated with each fuel
        fuels_loop = sorted(list(self.dict_trns_fuel_categories_to_fuel_variables.keys()))#HEREHERE
        for cat_fuel in fuels_loop:

            # get the index of the current category
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # set some model variables
            dict_tfc_to_fv_cur = self.dict_trns_fuel_categories_to_fuel_variables.get(cat_fuel)
            modvar_trns_ef_ch4_cur = dict_tfc_to_fv_cur.get("ef_ch4")
            modvar_trns_ef_n2o_cur = dict_tfc_to_fv_cur.get("ef_n2o")
            modvar_trns_fuel_efficiency_cur = dict_tfc_to_fv_cur.get("fuel_efficiency")
            modvar_trns_fuel_fraction_cur = dict_tfc_to_fv_cur.get("fuel_fraction")
            modvar_trns_modal_energy_demand_by_fuel = dict_tfc_to_fv_cur.get("modal_energy_consumption")

            # set some scalars for use in the calculations
            scalar_trns_fuel_efficiency_to_demand = self.model_attributes.get_variable_unit_conversion_factor(
                modvar_trns_fuel_efficiency_cur,
                self.modvar_trde_demand_pkm,
                "length"
            )

            # get the index and vector of co2 emission factors
            ind_enfu_cur = attr_enfu.get_key_value_index(cat_fuel)
            vec_trns_ef_by_fuel_co2_cur = arr_trns_ef_by_fuel_co2[:, ind_enfu_cur]
            vec_trns_volumetric_enerdensity_by_fuel = arr_trns_energy_density_fuel[:, ind_enfu_cur]
            
            # get arrays
            arr_trns_fuel_fraction_cur = dict_arrs_trns_frac_fuel.get(modvar_trns_fuel_fraction_cur)
            
            arr_trns_ef_ch4_cur = (
                self.model_attributes.extract_model_variable(#
                    df_neenergy_trajectories, 
                    modvar_trns_ef_ch4_cur, 
                    expand_to_all_cats = True,
                    return_type = "array_units_corrected",
                ) 
                if (modvar_trns_ef_ch4_cur is not None) 
                else 0
            )

            arr_trns_ef_n2o_cur = (
                self.model_attributes.extract_model_variable(#
                    df_neenergy_trajectories, 
                    modvar_trns_ef_n2o_cur,
                    expand_to_all_cats = True,
                    return_type = "array_units_corrected",
                ) 
                if (modvar_trns_ef_n2o_cur is not None) 
                else 0
            )

            arr_trns_fuel_efficiency_cur = self.model_attributes.extract_model_variable(#
                df_neenergy_trajectories, 
                modvar_trns_fuel_efficiency_cur,
                expand_to_all_cats = True,
                return_type = "array_base",
            )

            # current demand associate with the fuel (in distance terms of configuration units)
            arr_trns_vehdem_cur_fuel = array_trns_total_vehicle_demand*arr_trns_fuel_fraction_cur
            
            # add vmt by fuel to output and get scalar to convert self.modvar_trde_demand_pkm to configuration units
            modvar_trns_vmt_by_fuel_cut = self.dict_trns_fuel_categories_to_fuel_variables.get(cat_fuel)
            modvar_trns_vmt_by_fuel_cut = (
                modvar_trns_vmt_by_fuel_cut.get("vehicle_distance_traveled") 
                if (modvar_trns_vmt_by_fuel_cut is not None) 
                else None
            )
            scalar_trns_vkm_to_config = self.model_attributes.get_scalar(self.modvar_trde_demand_pkm, "length")

            df_out += [
                (
                    self.model_attributes.array_to_df(
                        arr_trns_vehdem_cur_fuel*scalar_trns_vkm_to_config,
                        modvar_trns_vmt_by_fuel_cut,
                        reduce_from_all_cats_to_specified_cats = True
                    )
                    if modvar_trns_vmt_by_fuel_cut is not None
                    else None
                )
            ]
            

            if (arr_trns_fuel_efficiency_cur is not None):

                # get demand for fuel in terms of modvar_trns_fuel_efficiency_cur, then get scalars to conert to emission factor fuel volume units
                arr_trns_fueldem_cur_fuel = np.nan_to_num(arr_trns_vehdem_cur_fuel/arr_trns_fuel_efficiency_cur, neginf = 0.0, posinf = 0.0)
                arr_trns_energydem_cur_fuel = (arr_trns_fueldem_cur_fuel.transpose()*vec_trns_volumetric_enerdensity_by_fuel).transpose()
                arr_trns_energydem_cur_fuel *= self.model_attributes.get_variable_unit_conversion_factor(
                    modvar_trns_fuel_efficiency_cur,
                    self.modvar_enfu_energy_density_volumetric,
                    "volume"
                )


                ##  CH4 EMISSIONS

                # get scalar to prepare fuel energies for the emission factor
                scalar_fuel_energy_to_ef_ch4 = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_energy_density_volumetric,
                    modvar_trns_ef_ch4_cur,
                    "energy"
                ) if (modvar_trns_ef_ch4_cur is not None) else 0
                arr_trns_fuel_energydem_cur_fuel_ch4 = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_ch4
                arr_emissions_ch4_cur_fuel = arr_trns_ef_ch4_cur*arr_trns_fuel_energydem_cur_fuel_ch4
                arr_trns_emissions_ch4 += arr_emissions_ch4_cur_fuel


                ##  CO2 EMISSIONS

                # get scalar to prepare fuel energies for the emission factor
                scalar_fuel_energy_to_ef_co2 = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_energy_density_volumetric,
                    self.modvar_enfu_ef_combustion_co2,
                    "energy"
                )
                arr_trns_fuel_energydem_cur_fuel_co2 = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_co2
                arr_emissions_co2_cur_fuel = (arr_trns_fuel_energydem_cur_fuel_co2.transpose()*vec_trns_ef_by_fuel_co2_cur).transpose()
                arr_trns_emissions_co2 += arr_emissions_co2_cur_fuel


                ##  N2O EMISSIONS

                # n2o scalar
                scalar_fuel_energy_to_ef_n2o = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_energy_density_volumetric,
                    modvar_trns_ef_n2o_cur,
                    "energy"
                ) if (modvar_trns_ef_n2o_cur is not None) else 0
                arr_trns_fuel_energydem_cur_fuel_n2o = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_n2o
                arr_emissions_n2o_cur_fuel = arr_trns_ef_n2o_cur*arr_trns_fuel_energydem_cur_fuel_n2o
                arr_trns_emissions_n2o += arr_emissions_n2o_cur_fuel


                ##  ENERGY DEMANDS

                arr_trns_modal_energy_demand_by_fuel = arr_trns_energydem_cur_fuel*scalar_trns_ved_to_output_scalar
                arr_trns_demand_by_category += arr_trns_modal_energy_demand_by_fuel
                arr_trns_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_trns_energydem_cur_fuel, axis = 1)*scalar_trns_ved_to_enfu_var_units

                # add modal demand to output
                df_out += [
                    self.model_attributes.array_to_df(
                        arr_trns_modal_energy_demand_by_fuel,
                        modvar_trns_modal_energy_demand_by_fuel,
                        reduce_from_all_cats_to_specified_cats = True
                    )
                ]


            elif cat_fuel == self.cat_enfu_electricity:

                # get scalar for energy
                scalar_electric_eff_to_distance_equiv = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_trns_electrical_efficiency,
                    self.modvar_trde_demand_pkm,
                    "length"
                )
                
                # get demand for fuel in terms of modvar_trns_fuel_efficiency_cur, then get scalars to conert to emission factor fuel volume units
                arr_trns_elect_efficiency_cur = self.model_attributes.extract_model_variable(#
                    df_neenergy_trajectories,
                    self.modvar_trns_electrical_efficiency,
                    expand_to_all_cats = True,
                    return_type = "array_base",
                )
                arr_trns_elect_efficiency_cur *= scalar_electric_eff_to_distance_equiv
                
                # calculate energy demand and write in terms of output units
                arr_trns_energydem_elec = np.nan_to_num(
                    arr_trns_vehdem_cur_fuel/arr_trns_elect_efficiency_cur, 
                    0.0,
                    posinf = 0.0, 
                    neginf = 0.0
                )

                arr_trns_energydem_elec *= self.model_attributes.get_scalar(
                    self.modvar_trns_electrical_efficiency, 
                    "energy"
                )
                vec_trns_energydem_elec_total = np.sum(arr_trns_energydem_elec, axis = 1)

                # update energy demand by category and fuel
                arr_trns_demand_by_category += arr_trns_energydem_elec
                arr_trns_demand_by_fuel[:, index_cat_fuel] = vec_trns_energydem_elec_total/self.model_attributes.get_scalar(self.modvar_enfu_energy_demand_by_fuel_trns, "energy")

                # add modal demand to output
                df_out += [
                    self.model_attributes.array_to_df(
                        arr_trns_energydem_elec,
                        modvar_trns_modal_energy_demand_by_fuel,
                        reduce_from_all_cats_to_specified_cats = True,
                    )
                ]

        vec_trns_demand_by_category_total = np.sum(arr_trns_demand_by_category, axis = 1)


        ##  BUILD COSTS

        # total fuel value per unit of energy
        arr_trns_total_fuel_value = self.get_enfu_fuel_costs_per_energy(
            df_neenergy_trajectories,
            modvar_for_units_energy = self.modvar_enfu_energy_demand_by_fuel_trns
        )

            
        
        ##  BUILD OUTPUT DF

        df_out += [
            # CH4 EMISSIONS
            self.model_attributes.array_to_df(
                arr_trns_emissions_ch4, 
                self.modvar_trns_emissions_ch4
            ),
            # CO2 EMISSIONS
            self.model_attributes.array_to_df(
                arr_trns_emissions_co2, 
                self.modvar_trns_emissions_co2
            ),
            # N2O EMISSIONS
            self.model_attributes.array_to_df(
                arr_trns_emissions_n2o, 
                self.modvar_trns_emissions_n2o
            ),
            # TOTAL DEMAND FOR EACH FUEL
            self.model_attributes.array_to_df(
                arr_trns_demand_by_fuel, 
                self.modvar_enfu_energy_demand_by_fuel_trns, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL VALUE OF FUEL DEMANDED
            self.model_attributes.array_to_df(
                arr_trns_total_fuel_value*arr_trns_demand_by_fuel, 
                self.modvar_enfu_value_of_fuel_trns, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL ENERGY CONSUMPTION BY CATEGORY
            self.model_attributes.array_to_df(
                arr_trns_demand_by_category, 
                self.modvar_trns_energy_consumption_total, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL ENERGY CONSUMPTION (AGGREGATE)
            self.model_attributes.array_to_df(
                vec_trns_demand_by_category_total, 
                self.modvar_trns_energy_consumption_total_agg
            ),
            # TOTAL ELECTRICITY CONSUMPTION (BY CATEGORY)
            self.model_attributes.array_to_df(
                arr_trns_energydem_elec, 
                self.modvar_trns_energy_consumption_electricity, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            # TOTAL ELECTRICITY CONSUMPTION (AGGREGATE)
            self.model_attributes.array_to_df(
                vec_trns_energydem_elec_total, 
                self.modvar_trns_energy_consumption_electricity_agg
            )
        ]

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate"
        )
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_trns], False)

        return df_out



    def project_transportation_demand(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_pop: np.ndarray,
        vec_rates_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:
        """
        Calculate transportation demands and associated metrics (TRDE)

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_pop: np.ndarray vector of population (requires 
            len(vec_rates_gdp) == len(df_neenergy_trajectories))
        - vec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth 
            rate from t_i to t_{i + 1}) (requires 
            len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)
        - vec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth 
            rates (v_i = growth rate from t_i to t_{i + 1}) (requires 
            len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)
       
        Keyword Arguments
        -----------------
        - dict_dims: dict of dimensions (returned from 
            check_projection_input_df). Default is None.
        - n_projection_time_periods: int giving number of time periods (returned 
            from check_projection_input_df). Default is None.
        - projection_time_periods: list of time periods (returned from 
            check_projection_input_df). Default is None.

        Notes
        -----
        If any of dict_dims, n_projection_time_periods, or 
            projection_time_periods are unspecified (expected if ran outside of 
            Energy.project()), self.model_attributes.check_projection_input_df 
            will be run
        """

        # allows production to be run outside of the project method
        check_none = any(
            [(x is None) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]
        )
        if check_none:
            (
                dict_dims, 
                df_neenergy_trajectories, 
                n_projection_time_periods, 
                projection_time_periods
            ) = self.model_attributes.check_projection_input_df(
                df_neenergy_trajectories, 
                True, 
                True, 
                True,
            )


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
         
        # attribute tables
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        attr_trde = self.model_attributes.get_attribute_table(self.subsec_name_trde)
        attr_trns = self.model_attributes.get_attribute_table(self.subsec_name_trns)


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # get the demand scalar
        array_trde_demscalar = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_trde_demand_scalar,
            expand_to_all_cats = True, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        # start with freight/megaton km demands
        array_trde_dem_init_freight = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_trde_demand_initial_mtkm, 
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        array_trde_elast_freight_demand_to_gdp = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_trde_elasticity_mtkm_to_gdp,
            expand_to_all_cats = True, 
            return_type = "array_base",
        )

        array_trde_growth_freight_dem_by_cat = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp, 
            array_trde_elast_freight_demand_to_gdp,
            elasticity_type = "standard", 
            rates_are_factors = False, 
        )

        # multiply and add to the output
        array_trde_freight_dem_by_cat = array_trde_dem_init_freight[0]*array_trde_growth_freight_dem_by_cat
        array_trde_freight_dem_by_cat *= array_trde_demscalar
        df_out.append(
            self.model_attributes.array_to_df(
                array_trde_freight_dem_by_cat, 
                self.modvar_trde_demand_mtkm, 
                reduce_from_all_cats_to_specified_cats = True,
            )
        )

        # deal with person-km
        array_trde_dem_init_passenger = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_trde_demand_initial_pkm_per_capita, 
            expand_to_all_cats = True,
            return_type = "array_base", 
        )

        array_trde_elast_passenger_demand_to_gdppc = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.modvar_trde_elasticity_pkm_to_gdp, 
            expand_to_all_cats = True,
            return_type = "array_base", 
        )

        array_trde_growth_passenger_dem_by_cat = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp_per_capita, 
            array_trde_elast_passenger_demand_to_gdppc, 
            elasticity_type = "standard",
            rates_are_factors = False, 
        )

        # project the growth in per capita, multiply by population, then add it to the output
        array_trde_passenger_dem_by_cat = array_trde_dem_init_passenger[0]*array_trde_growth_passenger_dem_by_cat
        array_trde_passenger_dem_by_cat = (array_trde_passenger_dem_by_cat.transpose()*vec_pop).transpose()
        array_trde_passenger_dem_by_cat *= array_trde_demscalar

        # add to output dataframe
        df_out.append(
            self.model_attributes.array_to_df(
                array_trde_passenger_dem_by_cat, 
                self.modvar_trde_demand_pkm, 
                reduce_from_all_cats_to_specified_cats = True,
            )
        )

        # build output dataframe
        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate",
        )

        return df_out



    def project(self,
        df_neenergy_trajectories: pd.DataFrame,
        subsectors_project: Union[list, str, None] = None
    ) -> pd.DataFrame:
        """
        Run the EnergyConsumption model. Take a data frame of input variables 
            (ordered by time series) and return a data frame of output variables 
            (model projections for energy--including carbon capture and 
            sequestration (CCSQ), fugitive emission (FGTV), industrial energy 
            (INEN), stationary combustion (SCOE), and transportation (TRNS)) the 
            same order.

        NOTE: Fugitive Emissions requires output from EnergyProduction to complete 
            a full accounting for fuel production and use. In SISEPUEDE, 
            integrated runs should be run in the order of:

            * EnergyConsumption.project(*args)
            * EnergyProduction.project(*args)
            * EnergyConsumption.project(*args, 
                subsectors_project = "Fugitive Emissions")

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame with all required input fields 
            as columns. The model will not run if any required variables are 
            missing, but errors will detail which fields are missing.
        - subsectors_project: list of subsectors or pipe-delimited string of 
            subsectors. If None, run all subsectors EXCEPT for Fugitive 
            Emissions. Valid list entries/subsectors are:

            * "Carbon Capture and Sequestration" or "ccsq"
            * "Fugitive Emissions" or "fgtv"
            * "Industrial Energy" or "inen"
            * "Stationary Combustion and Other Energy" or "scoe"
            * "Transportation" or "trns"

        Notes
        -----
        - The .project() method is designed to be parallelized or called from 
            command line via __main__ in run_sector_models.py.
        - df_neenergy_trajectories should have all input fields required (see 
            Energy.required_variables for a list of variables to be defined)
        - the df_neenergy_trajectories.project() method will run on valid time 
            periods from 1 .. k, where k <= n (n is the number of time periods). 
            By default, it drops invalid time periods. If there are missing 
            time_periods between the first and maximum, data are interpolated.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_neenergy_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_neenergy_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_neenergy_trajectories)
        dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)
        subsectors_project = self.get_projection_subsectors(subsectors_project = subsectors_project)


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors from the se model
        array_pop = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_socioeconomic.modvar_gnrl_subpop, 
            return_type = "array_base",
        )

        vec_gdp = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp, 
            return_type = "array_base",
        )

        vec_gdp_per_capita = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp_per_capita, 
            return_type = "array_base",
        )

        vec_hh = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_socioeconomic.modvar_grnl_num_hh, 
            return_type = "array_base",
        )

        vec_pop = self.model_attributes.extract_model_variable(#
            df_neenergy_trajectories, 
            self.model_socioeconomic.modvar_gnrl_pop_total, 
            return_type = "array_base",
        )

        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]



        #########################################
        #    MODEL CALCULATIONS BY SUBSECTOR    #
        #########################################

        # add industrial energy, transportation, and SCOE
        if self.subsec_name_ccsq in subsectors_project:
            df_out.append(
                self.project_ccsq(
                    df_neenergy_trajectories, 
                    dict_dims, 
                    n_projection_time_periods, 
                    projection_time_periods
                )
            )

        if self.subsec_name_inen in subsectors_project:
            df_out.append(
                self.project_industrial_energy(
                    df_neenergy_trajectories, 
                    vec_gdp, 
                    dict_dims, 
                    n_projection_time_periods, 
                    projection_time_periods
                )
            )

        if self.subsec_name_scoe in subsectors_project:
            df_out.append(
                self.project_scoe(
                    df_neenergy_trajectories, 
                    vec_hh, 
                    vec_gdp, 
                    vec_rates_gdp_per_capita, 
                    dict_dims, 
                    n_projection_time_periods, 
                    projection_time_periods
                )
            )

        if self.subsec_name_trns in subsectors_project:
            df_out.append(
                self.project_transportation(
                    df_neenergy_trajectories, 
                    vec_pop, 
                    vec_rates_gdp, 
                    vec_rates_gdp_per_capita, 
                    dict_dims, 
                    n_projection_time_periods, 
                    projection_time_periods
                )
            )

        # run fugitive emissions?
        if self.subsec_name_fgtv in subsectors_project:
            df_trajectories = sf.merge_output_df_list(
                [df_neenergy_trajectories] + df_out,
                self.model_attributes,
                merge_type = "concatenate",
            )

            df_out.append(
                self.project_fugitive_emissions(
                    df_trajectories, 
                    dict_dims, 
                    n_projection_time_periods, 
                    projection_time_periods
                )
            )

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(
            df_out, 
            self.model_attributes, 
            merge_type = "concatenate",
        )

        
        return df_out





###################################
###                             ###
###    SOME SIMPLE FUNCTIONS    ###
###                             ###
###################################


def is_sisepuede_model_nfp_energy(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDE Non-Fuel Production Energy model
    """

    out = hasattr(obj, "is_sisepuede_model_nfp_energy")
    out &= obj.is_sisepuede_model_nfp_energy if out else False

    return out