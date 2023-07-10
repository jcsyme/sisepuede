from attribute_table import AttributeTable
import logging
from model_attributes import *
from model_ippu import IPPU
from model_socioeconomic import Socioeconomic
import numpy as np
import pandas as pd
import support_functions as sf
import time
from typing import *


###########################
###                     ###
###     ENERGY MODEL    ###
###                     ###
###########################

class NonElectricEnergy:
    """
    NonElectricEnergy DOCSTRING to go here

    """
    def __init__(self,
        attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None
    ):

        self.logger = logger

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
        # valid subsectors in .project()
        self.valid_projection_subsecs = [
            self.subsec_name_ccsq,
            self.subsec_name_fgtv,
            self.subsec_name_inen,
            self.subsec_name_scoe,
            self.subsec_name_trns
        ]

        # initialize dynamic variables
        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_neenergy_input_output_fields()


        ##  SET MODEL FIELDS

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
            self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary"),
            ["energy_efficiency_variable_by_fuel", "fuel_fraction_variable_by_fuel"]
        )
        # reassign as variables
        self.modvar_dict_ccsq_fuel_fractions_to_efficiency_factors = self.modvar_dicts_ccsq_fuel_vars["fuel_fraction_variable_by_fuel_to_energy_efficiency_variable_by_fuel"]

        # Energy Fuel model variables
        self.modvar_enfu_energy_density_volumetric = "Volumetric Energy Density"
        self.modvar_enfu_ef_combustion_co2 = ":math:\\text{CO}_2 Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_ch4 = ":math:\\text{CH}_4 Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_n2o = ":math:\\text{N}_2\\text{O} Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_ch4 = ":math:\\text{CH}_4 Stationary Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_n2o = ":math:\\text{N}_2\\text{O} Stationary Combustion Emission Factor"
        self.modvar_enfu_efficiency_factor_industrial_energy = "Average Industrial Energy Fuel Efficiency Factor"
        self.modvar_enfu_energy_demand_by_fuel_ccsq = "Energy Demand by Fuel in CCSQ"
        self.modvar_enfu_energy_demand_by_fuel_elec = "Energy Demand by Fuel in Electricity"
        self.modvar_enfu_energy_demand_by_fuel_inen = "Energy Demand by Fuel in Industrial Energy"
        self.modvar_enfu_energy_demand_by_fuel_scoe = "Energy Demand by Fuel in SCOE"
        self.modvar_enfu_energy_demand_by_fuel_total = "Total Energy Demand by Fuel"
        self.modvar_enfu_energy_demand_by_fuel_trns = "Energy Demand by Fuel in Transportation"
        self.modvar_enfu_energy_density_gravimetric = "Gravimetric Energy Density"
        self.modvar_enfu_energy_density_volumetric = "Volumetric Energy Density"
        self.modvar_enfu_exports_fuel = "Fuel Exports"
        self.modvar_enfu_frac_fuel_demand_imported = "Fraction of Fuel Demand Imported"
        self.modvar_enfu_imports_electricity = "Electricity Imports"
        self.modvar_enfu_imports_fuel = "Fuel Imports"
        self.modvar_enfu_minimum_frac_fuel_used_for_electricity = "Minimum Fraction of Fuel Used for Electricity Generation"
        self.modvar_enfu_price_gravimetric = "Gravimetric Fuel Price"
        self.modvar_enfu_price_thermal = "Thermal Fuel Price"
        self.modvar_enfu_price_volumetric = "Volumetric Fuel Price"
        self.modvar_enfu_production_frac_petroleum_refinement = "Petroleum Refinery Production Fraction"
        self.modvar_enfu_production_frac_natural_gas_processing = "Natural Gas Processing Fraction"
        self.modvar_enfu_production_fuel = "Fuel Production"
        self.modvar_enfu_transmission_loss_electricity = "Electrical Transmission Loss"
        self.modvar_enfu_unused_fuel_exported = "Unused Fuel Exported"
        # list of key variables - total energy demands by fuel
        self.modvars_enfu_energy_demands_total = [
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_elec,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns
        ]
        # total demand for fuels for estimating distribution
        self.modvars_enfu_energy_demands_distribution = [
            self.modvar_enfu_energy_demand_by_fuel_elec,
            self.modvar_enfu_energy_demand_by_fuel_scoe
        ]
        # key categories
        self.cat_enfu_electricity = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {self.model_attributes.field_enfu_electricity_demand_category: 1})[0]

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
        self.modvar_inen_en_gdp_intensity_factor = "Initial Energy Consumption Intensity of GDP"
        self.modvar_inen_en_prod_intensity_factor = "Initial Energy Consumption Intensity of Production"
        self.modvar_inen_frac_en_coal = "Industrial Energy Fuel Fraction Coal"
        self.modvar_inen_frac_en_coke = "Industrial Energy Fuel Fraction Coke"
        self.modvar_inen_frac_en_diesel = "Industrial Energy Fuel Fraction Diesel"
        self.modvar_inen_frac_en_electricity = "Industrial Energy Fuel Fraction Electricity"
        self.modvar_inen_frac_en_furnace_gas = "Industrial Energy Fuel Fraction Furnace Gas"
        self.modvar_inen_frac_en_gasoline = "Industrial Energy Fuel Fraction Gasoline"
        self.modvar_inen_frac_en_hydrogen = "Industrial Energy Fuel Fraction Hydrogen"
        self.modvar_inen_frac_en_kerosene = "Industrial Energy Fuel Fraction Kerosene"
        self.modvar_inen_frac_en_natural_gas = "Industrial Energy Fuel Fraction Natural Gas"
        self.modvar_inen_frac_en_oil = "Industrial Energy Fuel Fraction Oil"
        self.modvar_inen_frac_en_pliqgas = "Industrial Energy Fuel Fraction Petroleum Liquid Gas"
        self.modvar_inen_frac_en_solar = "Industrial Energy Fuel Fraction Solar"
        self.modvar_inen_frac_en_solid_biomass = "Industrial Energy Fuel Fraction Solid Biomass"
        # get some dictionaries implied by the inen attribute tables
        self.dict_inen_fuel_categories_to_fuel_variables, self.dict_inen_fuel_categories_to_unassigned_fuel_variables = self.get_dict_inen_fuel_categories_to_fuel_variables()
        self.modvars_inen_list_fuel_fraction = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_inen_fuel_categories_to_fuel_variables,
            "fuel_fraction"
        )
        # key categories
        self.cat_inen_agricultural = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_inen, {"agricultural_category": 1})[0]

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
        self.modvar_scoe_efficiency_fact_heat_en_hydrogen = "SCOE Efficiency Factor for Heat Energy from Hydrogen"
        self.modvar_scoe_efficiency_fact_heat_en_kerosene = "SCOE Efficiency Factor for Heat Energy from Kerosene"
        self.modvar_scoe_efficiency_fact_heat_en_natural_gas = "SCOE Efficiency Factor for Heat Energy from Natural Gas"
        self.modvar_scoe_efficiency_fact_heat_en_pliqgas = "SCOE Efficiency Factor for Heat Energy from Petroleum Liquid Gas"
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
        self.modvar_scoe_frac_heat_en_coal = "SCOE Fraction Heat Energy Demand Coal"
        self.modvar_scoe_frac_heat_en_diesel = "SCOE Fraction Heat Energy Demand Diesel"
        self.modvar_scoe_frac_heat_en_electricity = "SCOE Fraction Heat Energy Demand Electricity"
        self.modvar_scoe_frac_heat_en_gasoline = "SCOE Fraction Heat Energy Demand Gasoline"
        self.modvar_scoe_frac_heat_en_hydrogen = "SCOE Fraction Heat Energy Demand Hydrogen"
        self.modvar_scoe_frac_heat_en_kerosene = "SCOE Fraction Heat Energy Demand Kerosene"
        self.modvar_scoe_frac_heat_en_natural_gas = "SCOE Fraction Heat Energy Demand Natural Gas"
        self.modvar_scoe_frac_heat_en_pliqgas = "SCOE Fraction Heat Energy Demand Petroleum Liquid Gas"
        self.modvar_scoe_frac_heat_en_solid_biomass = "SCOE Fraction Heat Energy Demand Solid Biomass"
        # get some dictionaries implied by the SCOE attribute tables
        self.modvar_dicts_scoe_fuel_vars = self.model_attributes.get_var_dicts_by_shared_category(
            self.subsec_name_scoe,
            self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary"),
            ["energy_efficiency_variable_by_fuel", "fuel_fraction_variable_by_fuel", "energy_demand_variable_by_fuel"]
        )
        # reassign as variables
        self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors = self.modvar_dicts_scoe_fuel_vars["fuel_fraction_variable_by_fuel_to_energy_efficiency_variable_by_fuel"]


        # Transportation variablesz
        self.modvar_trns_average_vehicle_load_freight = "Average Freight Vehicle Load"
        self.modvar_trns_average_passenger_occupancy = "Average Passenger Vehicle Occupancy Rate"
        self.modvar_trns_electrical_efficiency = "Electrical Vehicle Efficiency"
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
        self.modvar_trns_energy_consumption_electricity = "Electrical Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_electricity_agg = "Total Electrical Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_total = "Energy Consumption from Transportation"
        self.modvar_trns_energy_consumption_total_agg = "Total Energy Consumption from Transportation"
        self.modvar_trns_fuel_efficiency_biofuels = "Fuel Efficiency Biofuels"
        self.modvar_trns_fuel_efficiency_diesel = "Fuel Efficiency Diesel"
        self.modvar_trns_fuel_efficiency_gasoline = "Fuel Efficiency Gasoline"
        self.modvar_trns_fuel_efficiency_hgl = "Fuel Efficiency Hydrocarbon Gas Liquids"
        self.modvar_trns_fuel_efficiency_hydrogen = "Fuel Efficiency Hydrogen"
        self.modvar_trns_fuel_efficiency_kerosene = "Fuel Efficiency Kerosene"
        self.modvar_trns_fuel_efficiency_natural_gas = "Fuel Efficiency Natural Gas"
        self.modvar_trns_modeshare_freight = "Freight Transportation Mode Share"
        self.modvar_trns_modeshare_public_private = "Private and Public Transportation Mode Share"
        self.modvar_trns_modeshare_regional = "Regional Transportation Mode Share"
        self.modvar_trns_fuel_fraction_biofuels = "Transportation Mode Fuel Fraction Biofuels"
        self.modvar_trns_fuel_fraction_diesel = "Transportation Mode Fuel Fraction Diesel"
        self.modvar_trns_fuel_fraction_electricity = "Transportation Mode Fuel Fraction Electricity"
        self.modvar_trns_fuel_fraction_gasoline = "Transportation Mode Fuel Fraction Gasoline"
        self.modvar_trns_fuel_fraction_hgl = "Transportation Mode Fuel Fraction Hydrocarbon Gas Liquids"
        self.modvar_trns_fuel_fraction_hydrogen = "Transportation Mode Fuel Fraction Hydrogen"
        self.modvar_trns_fuel_fraction_kerosene = "Transportation Mode Fuel Fraction Kerosene"
        self.modvar_trns_fuel_fraction_natural_gas = "Transportation Mode Fuel Fraction Natural Gas"
        self.modvar_trns_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Transportation"
        self.modvar_trns_emissions_co2 = ":math:\\text{CO}_2 Emissions from Transportation"
        self.modvar_trns_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Transportation"
        self.modvar_trns_passenger_distance_traveled = "Total Passenger Distance by Vehicle"
        self.modvar_trns_vehicle_distance_traveled = "Total Vehicle Distance Traveled"
        # fuel variables dictionary for transportation
        self.dict_trns_fuel_categories_to_fuel_variables, self.dict_trns_fuel_categories_to_unassigned_fuel_variables = self.get_dict_trns_fuel_categories_to_fuel_variables()
        # some derivate lists of variables
        self.modvars_trns_list_fuel_fraction = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_fraction"
        )
        self.modvars_trns_list_fuel_efficiency = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_efficiency"
        )

        # Transportation Demand variables
        self.modvar_trde_demand_scalar = "Transportation Demand Scalar"
        self.modvar_trde_elasticity_mtkm_to_gdp = "Elasticity of Megatonne-Kilometer Demand to GDP"
        self.modvar_trde_elasticity_pkm_to_gdp = "Elasticity of Passenger-Kilometer Demand per Capita to GDP per Capita"
        self.modvar_trde_demand_initial_mtkm = "Initial Megatonne-Kilometer Demand"
        self.modvar_trde_demand_initial_pkm_per_capita = "Initial per Capita Passenger-Kilometer Demand"
        self.modvar_trde_demand_mtkm = "Megatonne-Kilometer Demand"
        self.modvar_trde_demand_pkm = "Passenger-Kilometer Demand"

        # variables from other sectors (NOTE: AFOLU INTEGRATION VARIABLES MUST BE SET HERE, CANNOT INITIALIZE AFOLU CLASS)
        self.modvar_agrc_yield = "Crop Yield"
        self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"

        # add other model classes (NOTE: CANNOT INITIALIZE AFOLU CLASS BECAUSE IT REQUIRES ACCESS TO THE ENERGY CLASS)
        self.model_socioeconomic = Socioeconomic(self.model_attributes)
        self.model_ippu = IPPU(self.model_attributes)

        # optional integration variables (uses calls to other model classes)
        self._set_integrated_variables()


        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()






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



    def _set_integrated_variables(self,
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
        for modvar in list_vars_required_for_integration:
            subsec = self.model_attributes.get_variable_subsector(modvar)
            new_vars = self.model_attributes.build_varlist(subsec, modvar)
            self.required_variables += new_vars

        # sot required variables and ensure no double counting
        self.required_variables = list(set(self.required_variables))
        self.required_variables.sort()

        # return variables required for secondary integrtion (i.e., for fugitive emissions only)
        list_vars_required_for_integration_fgtv = [
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_elec,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns
        ]


        self.integration_variables_non_fgtv = list_vars_required_for_integration
        self.integration_variables_fgtv = list_vars_required_for_integration_fgtv






    ##################################################
    #    FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS    #
    ##################################################

    def check_df_fields(self,
        df_neenergy_trajectories: pd.DataFrame,
        subsector: str = "All",
        var_type: str = "input",
        msg_prepend: str = None
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
                check_fields, ignore_fields = self.model_attributes.get_input_output_fields([self.subsec_name_econ, self.subsec_name_gnrl, subsector])
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



    def get_required_subsectors(self):
        ## TEMPORARY
        subsectors = [self.subsec_name_enfu, self.subsec_name_inen, self.subsec_name_trns, self.subsec_name_trde, self.subsec_name_scoe]#self.subsec_name_enfu,#self.model_attributes.get_setor_subsectors("Energy")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]
        return subsectors, subsectors_base



    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa



    def get_neenergy_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)

        return required_vars + self.get_required_dimensions(), output_vars



    # get subsector specification
    def get_projection_subsectors(self,
        subsectors_project: Union[list, str, None] = None,
        delim: str = "|",
        drop_fugitive_from_none_q: bool = True
    ) -> list:
        """
            Check and retrieve valid projection subsectors to run in NonElectricEnergy.project()

            Keyword Arguments
            ------------------
            - subsectors_project: list or string to run. If None, all valid subsectors (exludes NonElectricEnergy.subsec_name_fgtv if drop_fugitive_from_none_q = True)
            - delim: delimiter to use in input strings
            - drop_fugitive_from_none_q: drop NonElectricEnergy.subsec_name_fgtv if subsectors_project == None?
        """
        # get subsector attribute
        attr_subsec = self.model_attributes.dict_attributes.get("abbreviation_subsector")
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






    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################

    ##
    def get_agrc_lvst_prod_and_intensity(self,
        df_neenergy_trajectories: pd.DataFrame
    ):
        """
            Retrieve agriculture and livstock production (total mass) and initial energy consumption, then calculate energy intensity (in terms of self.modvar_inen_en_prod_intensity_factor) and return production (in terms of self.model_ippu.modvar_ippu_qty_total_production)

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
            return_type = "array_base"
        )
        # get initial energy consumption for agrc/lvst and then ensure unit are set
        arr_inen_init_energy_consumption_agrc = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_inen_energy_conumption_agrc_init, True, "array_base", expand_to_all_cats = True)
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
        return index_inen_agrc, vec_inen_energy_intensity_agrc_lvst, vec_inen_prod_agrc_lvst



    ##  support for project_fugitive_emissions
    def get_array_for_fugitive_emissions(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_cur: str,
        array_energy_density: np.ndarray,
        modvar_energy_density: str = None
    ) -> np.ndarray:
        """
            Format an array for fugitive emissions calculations. Convert from mass/volume (input/input) to mass/energy (config/config)

            Function Arguments
            ------------------
            - df_neenergy_trajectories: input data frame of trajectories
            - modvar_cur: mass/volume fuel emission factor model variable (used for units conversion)
            - array_energy_density: array of volumetric energy density (energy/volume)
            - modvar_energy_density: model variable giving volumetric density (used for units conversion). If none, defaults to NonElectricEnergy.modvar_enfu_energy_density_volumetric
        """
        # check model variable input
        if modvar_cur is None:
            return None

        modvar_energy_density = self.modvar_enfu_energy_density_volumetric if (modvar_energy_density is None) else modvar_energy_density

        # get the variable and associated data (tonne/m3)
        arr_ef_mass_per_volume = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            modvar_cur,
            return_type = "array_units_corrected_gas",
            expand_to_all_cats = True
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



    ##  industrial energy variables from fuel categories as specified by a matchstring
    def get_dict_inen_fuel_categories_to_fuel_variables(self):
        """
            use get_dict_inen_fuel_categories_to_fuel_variables to return a dictionary with fuel categories as keys based on the Transportation attribute table;
            {cat_fuel: {"fuel_efficiency": VARNAME_FUELEFFICIENCY, ...}}

            for each key, the dict includes variables associated with the fuel cat_fuel:

            - "fuel_fraction"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.subsec_name_inen,
            "cat_fuel",
            {
                "Fuel Fraction": "fuel_fraction"
            },
            "varreqs_partial",
            True
        )

        return dict_out



    ##  transportation variables from fuel categories as specified by a matchstring
    def get_dict_trns_fuel_categories_to_fuel_variables(self):
        """
            use get_dict_trns_fuel_categories_to_fuel_variables to return a dictionary with fuel categories as keys based on the Transportation attribute table;
            {cat_fuel: {"fuel_efficiency": VARNAME_FUELEFFICIENCY, ...}}

            for each key, the dict includes variables associated with the fuel cat_fuel:

            - "fuel_efficiency"
            - "fuel_fraction"
            - "ef_ch4"
            - "ef_n2o"

        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.subsec_name_trns,
            "cat_fuel",
            {
                "Fuel Efficiency": "fuel_efficiency",
                "Fuel Fraction": "fuel_fraction",
                ":math:\\text{CH}_4": "ef_ch4",
                ":math:\\text{N}_2\\text{O}": "ef_n2o"
            },
            "varreqs_partial",
            True
        )

        return dict_out



    ##  project energy consumption for scoe/ccsq
    def project_energy_consumption_by_fuel_from_effvars(self,
        df_neenergy_trajectories: pd.DataFrame,
        modvar_consumption: str,
        arr_activity: Union[np.ndarray, None],
        arr_elasticity: Union[np.ndarray, None],
        arr_elastic_driver: Union[np.ndarray, None],
        dict_fuel_fracs: dict,
        dict_fuel_frac_to_eff: dict = None
    ) -> np.ndarray:

        """
            Project energy consumption--in terms of configuration units for
                energy--for a consumption variable for each fuel specified as a
                key in self.modvar_dict_scoe_fuel_fractions_to_efficiency_factors

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
            - arr_elastic_driver: the driver of elasticity in energy demands,
                e.g., vector of change rates of gdp per capita.
                * Must be such that df_neenergy_trajectories.shape[0] = arr_elastic_driver.shape[0] == arr_elasticity.shape[0] - 1.
                * Setting to None will mean that specified future demands will
                    be used (often constant).
            - dict_fuel_fracs: dictionary mapping each fuel fraction variable to
                its fraction of energy.
                * Each key must be a key in dict_fuel_frac_to_eff.
            - dict_fuel_frac_to_eff: dictionary mapping fuel fraction variable
                to its associated efficiency variable (SCOE and CCSQ)
        """

        ##  initialize consumption and the fraction -> efficiency dictionary

        # get consumption in terms of configuration output energy units
        arr_consumption = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            modvar_consumption,
            True,
            "array_base",
            expand_to_all_cats = True
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
                    raise ValueError(f"Error in project_energy_consumption_by_fuel_from_effvars: unable to infer dictionary for dict_fuel_frac_to_eff based on model variable '{modvar_consumption}'.")
            else:
                raise ValueError(f"Invalid model variable '{modvar_consumption}' found in project_energy_consumption_by_fuel_from_effvars: the variable is undefined.")
        elif not isinstance(dict_fuel_frac_to_eff, dict):
            raise ValueError(f"Error in project_energy_consumption_by_fuel_from_effvars: invalid type '{type(dict_fuel_frac_to_eff)}' specified for dict_fuel_frac_to_eff.")


        ##  estimate demand at point of use (account for heat delivery efficiency)

        # loop over the different fuels to generate the true demand
        arr_frac_norm = 0

        # use fractions of demand + efficiencies to calculate fraction of consumption
        for modvar_fuel_frac in dict_fuel_fracs.keys():
            # get efficiency, then fuel fractions
            modvar_fuel_eff = dict_fuel_frac_to_eff.get(modvar_fuel_frac)
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)
            arr_efficiency = self.model_attributes.get_standard_variables(
                df_neenergy_trajectories,
                modvar_fuel_eff,
                True,
                "array_base",
                expand_to_all_cats = True
            )
            arr_frac_norm += np.nan_to_num(arr_frac/arr_efficiency, 0.0)

        # project demand forward
        arr_demand = np.nan_to_num(arr_consumption/arr_frac_norm, 0.0)
        if (arr_elastic_driver is not None) and (arr_elasticity is not None):
            arr_growth_demand = sf.project_growth_scalar_from_elasticity(arr_elastic_driver, arr_elasticity, False, "standard")
            arr_demand = sf.do_array_mult(arr_demand[0]*arr_growth_demand, arr_activity)
        else:
            self._log("Missing elasticity information found in 'project_energy_consumption_by_fuel_from_effvars': using specified future demands.", type_log = "debug")
            arr_demand = sf.do_array_mult(arr_demand, arr_activity) if (arr_activity is not None) else arr_demand

        # calculate consumption
        dict_consumption_by_fuel_out = {}
        for modvar_fuel_frac in dict_fuel_fracs.keys():
            # get efficiency variable + variable arrays
            modvar_fuel_eff = dict_fuel_frac_to_eff.get(modvar_fuel_frac)
            arr_frac = dict_fuel_fracs.get(modvar_fuel_frac)
            arr_efficiency = self.model_attributes.get_standard_variables(
                df_neenergy_trajectories,
                modvar_fuel_eff,
                True,
                "array_base",
                expand_to_all_cats = True
            )
            # use consumption by fuel type and efficiency to get output demand for each fuel (in output energy units specified in config)
            arr_consumption_fuel = np.nan_to_num(arr_demand*arr_frac/arr_efficiency, 0.0)
            dict_consumption_by_fuel_out.update({modvar_fuel_frac: arr_consumption_fuel})

        return dict_consumption_by_fuel_out



    # get energy consumption using fuel categories (average over input subsector categories) instead of individualized efficiency variables -- use for Industrial Energy
    def project_energy_consumption_by_fuel_from_fuel_cats(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_consumption_intensity_initial: np.ndarray,
        arr_driver: np.ndarray,
        modvar_fuel_efficiency: str,
        dict_fuel_fracs: dict,
        dict_fuel_frac_to_fuel_cat: dict
    ) -> np.ndarray:

        """
        Project energy consumption--in terms of units of the input vector
            vec_consumption_initial--given changing demand fractions and
            efficiency factors

        Function Arguments
        ------------------
        - df_neenergy_trajectories: Dataframe of input variables
        - vec_consumption_intensity_initial: array giving initial consumption
            (for initial time period only)
        - arr_driver: driver of demand--either shape of

            (n_projection_time_periods, len(vec_consumption_intensity_initial))

            or

            (n_projection_time_periods, )

        - modvar_fuel_efficiency: string model variable for enfu fuel efficiency
        - dict_fuel_fracs: dictionary mapping each fuel fraction variable to its
            fraction of energy.
            * Each key must be a key in dict_fuel_frac_to_eff.
        - dict_fuel_frac_to_fuel_cat: dictionary mapping fuel fraction variable
            to its associated fuel category
        """

        ##  estimate demand at point of use (account for heat delivery efficiency)

        # initializations
        arr_frac_norm = 0
        arr_enfu_efficiency = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            modvar_fuel_efficiency,
            True,
            "array_base",
            expand_to_all_cats = True
        )
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)

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

        return dict_consumption_by_fuel_out



    ##  project imports, exports, and local production of fuels
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
            Project imports, exports, and domestic production demands for fuels. Returns a tuple of np.ndarrays with the following elements:

            demands, distribution demands, exports, imports, production

            Arrays are returned in order of attribute_fuel.key_values

            Function Arguments
            ------------------
            - df_neenergy_trajectories: Dataframe of input variables
            - attribute_fuel: AttributeTable with information on fuels. If None, use ModelAttributes default.
            - modvars_energy_demands: list of SISEPUEDE model variables to extract for use as energy demands. If None, defaults to NonElectricEnergy.modvars_enfu_energy_demands_total
            - modvars_energy_distribution_demands: list of SISEPUEDE model variables to extract for use for distribution energy demands. If None, defaults to NonElectricEnergy.modvars_enfu_energy_demands_distribution
            - modvar_energy_exports: SISEPUEDE model variable giving exports. If None, default to NonElectricEnergy.modvar_enfu_exports_fuel
            - modvar_import_fraction: SISEPUEDE model variable giving the import fraction. If None, default to NonElectricEnergy.modvar_enfu_frac_fuel_demand_imported
            - target_energy_units: target energy units to convert output to. If None, default to ModelAttributes.configuration energy_units.
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
                arr_tmp = self.model_attributes.get_standard_variables(
                    df_neenergy_trajectories,
                    modvar,
                    return_type = "array_base",
                    expand_to_all_cats = True
                )
            except:
                self._log(f"Warning in project_enfu_production_and_demands: Variable '{modvar}' not found in the data frame. Its fuel demands will not be included.", type_log = "warning")

            arr_tmp *= scalar
            arr_demands += arr_tmp
            arr_demands_distribution += arr_tmp if (modvar in modvars_energy_distribution_demands) else 0.0


        ##  CALCULATE IMPORTS, EXPORTS, AND PRODUCTION

        # get import fractions and calculate imports
        arr_import_fracs = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            modvar_import_fraction,
            return_type = "array_base",
            expand_to_all_cats = True,
            var_bounds = (0, 1)
        )
        arr_imports = arr_import_fracs*arr_demands

        # get exports
        arr_exports = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            modvar_energy_exports,
            return_type = "array_base",
            expand_to_all_cats = True,
            var_bounds = (0, 1)
        )
        energy_units = self.model_attributes.get_variable_characteristic(
            modvar_energy_exports,
            self.model_attributes.varchar_str_unit_energy
        )
        scalar = self.model_attributes.get_energy_equivalent(
            energy_units,
            output_energy_units
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

    ##  carbon capture and sequestration model
    def project_ccsq(self,
        df_neenergy_trajectories: pd.DataFrame,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            project_ccsq can be called from other sectors to simplify calculation of emissions from carbon capture and sequestration.

            Function Arguments
            ------------------
            - df_neenergy_trajectories: pd.DataFrame of input variables
            - dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.
            - n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.
            - projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.

            Notes
            -----
            If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of Energy.project()), self.model_attributes.check_projection_input_df wil be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_ccsq = self.model_attributes.get_subsector_attribute(self.subsec_name_ccsq, "pycategory_primary")
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        # attribute tables
        attr_ccsq = self.model_attributes.dict_attributes[pycat_ccsq]
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]


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
        arr_ccsq_demand_sequestration = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ccsq_total_sequestration, True, "array_base", expand_to_all_cats = True)
        arr_ccsq_energy_intensity_sequestration = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ccsq_demand_per_co2, True, "array_base", expand_to_all_cats = True)
        # here, multiply by inverse (hence vars are reversed) to write intensity mass in terms of self.modvar_ccsq_total_sequestration; next, scale energy units to configuration units
        arr_ccsq_energy_intensity_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ccsq_total_sequestration,
            self.modvar_ccsq_demand_per_co2,
            "mass"
        )
        arr_ccsq_energy_intensity_sequestration *= self.model_attributes.get_scalar(self.modvar_ccsq_demand_per_co2, "energy")
        # get fraction of energy that is heat energy (from fuels) + fraction that is electric
        arr_ccsq_frac_energy_elec = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ccsq_frac_en_electricity, True, "array_base", expand_to_all_cats = True)
        arr_ccsq_frac_energy_heat = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ccsq_frac_en_heat, True, "array_base", expand_to_all_cats = True)
        # next, use fuel mix + efficiencies to determine demands from final fuel consumption for heat energy_to_match (this will return the fractions of sequestration by consumption)
        dict_ccsq_demands_by_fuel_heat = self.project_energy_consumption_by_fuel_from_effvars(
            df_neenergy_trajectories,
            self.modvar_ccsq_total_sequestration,
            None, None, None,
            dict_arrs_ccsq_frac_energy
        )
        fuels_loop = list(dict_ccsq_demands_by_fuel_heat.keys())
        for k in fuels_loop:
            dict_ccsq_demands_by_fuel_heat[k] = dict_ccsq_demands_by_fuel_heat[k]*arr_ccsq_energy_intensity_sequestration*arr_ccsq_frac_energy_heat
        # get electricity demand
        arr_ccsq_demand_electricity = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ccsq_total_sequestration, True, "array_base", expand_to_all_cats = True)
        arr_ccsq_demand_electricity *= arr_ccsq_frac_energy_elec*arr_ccsq_energy_intensity_sequestration



        ##  GET EMISSION FACTORS

        # methane - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_ch4 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_ch4, return_type = "array_units_corrected")
        arr_ccsq_ef_by_fuel_ch4 /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_stationary_ch4, "energy")
        # carbon dioxide - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected")
        arr_ccsq_ef_by_fuel_co2 /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_co2, "energy")
        # nitrous oxide - scale to ensure energy units are the same
        arr_ccsq_ef_by_fuel_n2o = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_n2o, return_type = "array_units_corrected")
        arr_ccsq_ef_by_fuel_n2o /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_stationary_n2o, "energy")


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

        # build output dataframe components
        df_out += [
            self.model_attributes.array_to_df(arr_ccsq_emissions_ch4, self.modvar_ccsq_emissions_ch4),
            self.model_attributes.array_to_df(arr_ccsq_emissions_co2, self.modvar_ccsq_emissions_co2),
            self.model_attributes.array_to_df(arr_ccsq_emissions_n2o, self.modvar_ccsq_emissions_n2o),
            self.model_attributes.array_to_df(arr_ccsq_demand_by_fuel, self.modvar_enfu_energy_demand_by_fuel_ccsq),
            self.model_attributes.array_to_df(arr_ccsq_demand_electricity, self.modvar_ccsq_energy_consumption_electricity),
            self.model_attributes.array_to_df(arr_ccsq_demand_electricity_total, self.modvar_ccsq_energy_consumption_electricity_agg),
            self.model_attributes.array_to_df(arr_ccsq_demand_non_electric + arr_ccsq_demand_electricity, self.modvar_ccsq_energy_consumption_total),
            self.model_attributes.array_to_df(arr_ccsq_demand_non_electric_total + arr_ccsq_demand_electricity_total, self.modvar_ccsq_energy_consumption_total_agg)
        ]

        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_ccsq], False)

        return df_out
    


    def project_fuel_production(self,
        df_neenergy_trajectories: pd.DataFrame,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:
        """
        Calculate direct emissions from the production of fuels. Includes 
            emissions from the manufacture of energy-generating infrastructure
            (e.g., solar panels, wind turbines, reservoirs, lithium, etc.) and 
            the direct production and/or refinement of energy-producing fuels
            such as oil, coal, and natural gas. Relies on integration with 
            ElectricEnergy to generate fuel demands.

        This is the second to last model projected in the SISEPUEDE DAG as it 
            depends on all other energy models to determine mining production.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_gdp: np.ndarray vector of gdp (requires 
            len(vec_gdp) == len(df_neenergy_trajectories))
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
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")
        pycat_inen = self.model_attributes.get_subsector_attribute(self.subsec_name_inen, "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes.get(pycat_enfu)
        attr_entc = self.model_attributes.dict_attributes.get(pycat_entc)
        attr_inen = self.model_attributes.dict_attributes.get(pycat_inen)
        attr_ippu = self.model_attributes.dict_attributes.get(pycat_ippu)


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # get HEREHERE




    def project_fugitive_emissions(
        self,
        df_neenergy_trajectories: pd.DataFrame,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
        Calculate fugitive emissions of gasses from coal, oil, and gas 
            production, transmission, and distribution.

        This is the final model projected in the SISEPUEDE DAG as it depends on 
            all other energy models to determine mining production.

        Function Arguments
        ------------------
        - df_neenergy_trajectories: pd.DataFrame of input variables
        - vec_gdp: np.ndarray vector of gdp (requires 
            len(vec_gdp) == len(df_neenergy_trajectories))
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
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_fgtv = self.model_attributes.get_subsector_attribute(self.subsec_name_fgtv, "pycategory_primary")
        pycat_inen = self.model_attributes.get_subsector_attribute(self.subsec_name_inen, "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_fgtv = self.model_attributes.dict_attributes[pycat_fgtv]
        attr_inen = self.model_attributes.dict_attributes[pycat_inen]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # get all demands, imports, exports, and production in terms of configuration units
        arr_fgtv_demands, arr_demands_distribution, arr_fgtv_export, arr_fgtv_imports, arr_fgtv_production = self.project_enfu_production_and_demands(
            df_neenergy_trajectories
        )

        # define a dictionary to relate aggregate emissions to the components
        dict_emission_to_fugitive_components = {
            self.modvar_fgtv_emissions_ch4: {
                "distribution": self.modvar_fgtv_ef_ch4_distribution ,
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
        arr_enfu_energy_density_volumetric = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_enfu_energy_density_volumetric,
            return_type = "array_base",
            expand_to_all_cats = True
        )
        arr_fgtv_frac_vent_to_flare = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_fgtv_frac_non_fugitive_flared,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        vec_fgtv_reduction_flared_leaks = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_fgtv_frac_reduction_fugitive_leaks,
            return_type = "array_base",
            var_bounds = (0, 1)
        )


        ##  LOOP OVER OUTPUT EMISSIONS TO GENERATE RELEVANT OUTPUT

        df_out = []
        for modvar_emission in dict_emission_to_fugitive_components.keys():

            # get the key emission factor arrays in terms of mass/energy
            arr_ef_distribution = self.get_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["distribution"],
                arr_enfu_energy_density_volumetric
            )
            # production - flaring
            arr_ef_production_flaring = self.get_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_flaring"],
                arr_enfu_energy_density_volumetric
            )
            # production - fugitive/leaks
            arr_ef_production_fugitive = self.get_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_fugitive"],
                arr_enfu_energy_density_volumetric
            )
            # production - venting
            arr_ef_production_venting = self.get_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["production_venting"],
                arr_enfu_energy_density_volumetric
            )
            # production - transmission
            arr_ef_transmission = self.get_array_for_fugitive_emissions(
                df_neenergy_trajectories,
                dict_emission_to_fugitive_components[modvar_emission]["transmission"],
                arr_enfu_energy_density_volumetric
            )
            # weighted emission factor for tradeoff from flare to vent; note that categories for which arr_fgtv_frac_vent_to_flare is not defined have the arr_fgtv_frac_vent_to_flare = 1 (so that everything goes to flaring)
            arr_fgtv_ef_fv_flare = arr_fgtv_frac_vent_to_flare*arr_ef_production_flaring if (arr_ef_production_flaring is not None) else 0.0
            arr_fgtv_ef_fv_vent = (1 - arr_fgtv_frac_vent_to_flare)*arr_ef_production_venting if (arr_ef_production_flaring is not None) else 0.0
            arr_fgtv_ef_fv = arr_fgtv_ef_fv_flare + arr_fgtv_ef_fv_vent
            # distribution, production, and transmission emissions
            arr_fgtv_emit_distribution = arr_demands_distribution*arr_ef_distribution if (arr_ef_distribution is not None) else 0.0
            arr_fgtv_emit_production = arr_fgtv_production*arr_fgtv_ef_fv
            arr_fgtv_emit_transmission = arr_ef_transmission*(arr_fgtv_production + arr_fgtv_imports) if (arr_ef_transmission is not None) else 0.0
            # get total and determine scalar
            arr_fgtv_emissions_cur = arr_fgtv_emit_distribution + arr_fgtv_emit_production + arr_fgtv_emit_transmission
            emission = self.model_attributes.get_variable_characteristic(
                modvar_emission,
                self.model_attributes.varchar_str_emission_gas
            )
            arr_fgtv_emissions_cur *= 1/self.model_attributes.get_scalar(modvar_emission, "mass") if (emission is None) else 1

            df_out.append(
                self.model_attributes.array_to_df(
                    arr_fgtv_emissions_cur,
                    modvar_emission,
                    include_scalars = False,
                    reduce_from_all_cats_to_specified_cats = True
                )
            )

        # set additional output
        arr_fgtv_imports /= self.model_attributes.get_scalar(self.modvar_enfu_imports_fuel, "energy")
        arr_fgtv_production /= self.model_attributes.get_scalar(self.modvar_enfu_imports_fuel, "energy")

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

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_fgtv], False)

        return df_out



    ##  industrial energy model
    def project_industrial_energy(
        self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            Calculate emissions from fuel combustion in industrial energy.

            Function Arguments
            ------------------
            - df_neenergy_trajectories: pd.DataFrame of input variables
            = vec_gdp: np.ndarray vector of gdp (requires len(vec_gdp) == len(df_neenergy_trajectories))
            - dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.
            - n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.
            - projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.

            Notes
            -----
            If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of Energy.project()), self.model_attributes.check_projection_input_df wil be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_inen = self.model_attributes.get_subsector_attribute(self.subsec_name_inen, "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_inen = self.model_attributes.dict_attributes[pycat_inen]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]


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
        arr_inen_prod = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_ippu.modvar_ippu_qty_total_production, True, "array_base", expand_to_all_cats = True)
        arr_inen_prod_energy_intensity = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_inen_en_prod_intensity_factor, True, "array_base", expand_to_all_cats = True)

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
            self.modvar_inen_en_prod_intensity_factor,
            self.model_ippu.modvar_ippu_qty_total_production,
            "mass"
        )

        # NOTE: add vec_inen_energy_intensity_agrc_lvst here because its mass is already in terms of self.model_ippu.modvar_ippu_qty_total_production
        arr_inen_energy_consumption_intensity_prod[:, index_inen_agrc] += vec_inen_energy_intensity_agrc_lvst

        # project future consumption
        dict_inen_energy_consumption_prod = self.project_energy_consumption_by_fuel_from_fuel_cats(
            df_neenergy_trajectories,
            arr_inen_energy_consumption_intensity_prod[0],
            arr_inen_prod,
            self.modvar_enfu_efficiency_factor_industrial_energy,
            dict_arrs_inen_frac_energy,
            dict_inen_fuel_frac_to_eff_cat
        )

        # gdp-based emissions - get intensity, multiply by gdp, and scale to match energy units of production
        arr_inen_energy_consumption_intensity_gdp = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_inen_en_gdp_intensity_factor, True, "array_base", expand_to_all_cats = True)
        arr_inen_energy_consumption_intensity_gdp *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_inen_en_gdp_intensity_factor,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        ) 
        dict_inen_energy_consumption_gdp = self.project_energy_consumption_by_fuel_from_fuel_cats(
            df_neenergy_trajectories,
            arr_inen_energy_consumption_intensity_gdp[0],
            vec_gdp,
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
        arr_inen_ef_by_fuel_ch4 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_ch4, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_ch4 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_stationary_ch4,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        )
        # carbon dioxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_co2 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_co2,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
        )
        # nitrous oxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_n2o = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_n2o, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_n2o *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_enfu_ef_combustion_stationary_n2o,
            self.modvar_inen_en_prod_intensity_factor,
            "energy"
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
        # set energy data frames
        scalar_energy = self.model_attributes.get_scalar(self.modvar_inen_en_prod_intensity_factor, "energy")


        ##  BUILD OUTPUT DFs

        df_out += [
            self.model_attributes.array_to_df(arr_inen_emissions_ch4, self.modvar_inen_emissions_ch4, False, True),
            self.model_attributes.array_to_df(arr_inen_emissions_co2, self.modvar_inen_emissions_co2, False, True),
            self.model_attributes.array_to_df(arr_inen_emissions_n2o, self.modvar_inen_emissions_n2o, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_by_fuel*scalar_inen_to_enfu_var_units, self.modvar_enfu_energy_demand_by_fuel_inen),
            self.model_attributes.array_to_df(arr_inen_demand_electricity*scalar_energy, self.modvar_inen_energy_consumption_electricity, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_electricity_total*scalar_energy, self.modvar_inen_energy_consumption_electricity_agg, False),
            self.model_attributes.array_to_df(arr_inen_demand_total*scalar_energy, self.modvar_inen_energy_consumption_total, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_total_total*scalar_energy, self.modvar_inen_energy_consumption_total_agg, False)
        ]

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_inen], False)

        return df_out



    ##  stationary combustion and other energy
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
        Calculation other energy, including stationary combustion (including
            buildings) and other energy exogenously specified emissions
            unaccounted for elsewhere

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
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_scoe = self.model_attributes.get_subsector_attribute(self.subsec_name_scoe, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_scoe = self.model_attributes.dict_attributes[pycat_scoe]


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
        arr_scoe_deminit_hh_elec = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_consumpinit_energy_per_hh_elec, True, "array_base", expand_to_all_cats = True)
        arr_scoe_deminit_hh_heat = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_consumpinit_energy_per_hh_heat, True, "array_base", expand_to_all_cats = True)
        arr_scoe_deminit_mmmgdp_elec = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_consumpinit_energy_per_mmmgdp_elec, True, "array_base", expand_to_all_cats = True)
        arr_scoe_deminit_mmmgdp_heat = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_consumpinit_energy_per_mmmgdp_heat, True, "array_base", expand_to_all_cats = True)
        # get elasticities
        arr_scoe_enerdem_elasticity_hh_elec = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc, True, "array_base", expand_to_all_cats = True)
        arr_scoe_enerdem_elasticity_hh_heat = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc, True, "array_base", expand_to_all_cats = True)
        arr_scoe_enerdem_elasticity_mmmgdp_elec = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc, True, "array_base", expand_to_all_cats = True)
        arr_scoe_enerdem_elasticity_mmmgdp_heat = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc, True, "array_base", expand_to_all_cats = True)
        # get demand for electricity for households and gdp driven demands
        arr_scoe_growth_demand_hh_elec = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, arr_scoe_enerdem_elasticity_hh_elec, False, "standard")
        arr_scoe_demand_hh_elec = sf.do_array_mult(arr_scoe_deminit_hh_elec[0]*arr_scoe_growth_demand_hh_elec, vec_hh)
        arr_scoe_demand_hh_elec *= self.model_attributes.get_scalar(self.modvar_scoe_consumpinit_energy_per_hh_elec, "energy")
        arr_scoe_growth_demand_mmmgdp_elec = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, arr_scoe_enerdem_elasticity_hh_elec, False, "standard")
        arr_scoe_demand_mmmgdp_elec = sf.do_array_mult(arr_scoe_deminit_mmmgdp_elec[0]*arr_scoe_growth_demand_mmmgdp_elec, vec_gdp)
        arr_scoe_demand_mmmgdp_elec *= self.model_attributes.get_scalar(self.modvar_scoe_consumpinit_energy_per_mmmgdp_elec, "energy")
        # get demand scalars
        arr_scoe_demscalar_elec_energy_demand = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_scoe_demscalar_elec_energy_demand,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            expand_to_all_cats = True,
            all_cats_missing_val = 1.0
        )
        arr_scoe_demscalar_heat_energy_demand = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_scoe_demscalar_heat_energy_demand,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            expand_to_all_cats = True,
            all_cats_missing_val = 1.0
        )

        # next, use fuel mix + efficiencies to determine demands from final fuel consumption for heat energy_to_match
        dict_scoe_demands_by_fuel_heat_hh = self.project_energy_consumption_by_fuel_from_effvars(
            df_neenergy_trajectories,
            self.modvar_scoe_consumpinit_energy_per_hh_heat,
            vec_hh,
            arr_scoe_enerdem_elasticity_hh_heat,
            vec_rates_gdp_per_capita,
            dict_arrs_scoe_frac_energy
        )
        dict_scoe_demands_by_fuel_heat_mmmgdp = self.project_energy_consumption_by_fuel_from_effvars(
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
        arr_scoe_ef_by_fuel_ch4 = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_stationary_ch4,
            return_type = "array_units_corrected"
        )
        arr_scoe_ef_by_fuel_ch4 /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_stationary_ch4, "energy")

        # carbon dioxide - scale to ensure energy units are the same
        arr_scoe_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_co2,
            return_type = "array_units_corrected"
        )
        arr_scoe_ef_by_fuel_co2 /= self.model_attributes.get_scalar(self.modvar_enfu_ef_combustion_co2, "energy")

        # nitrous oxide - scale to ensure energy units are the same
        arr_scoe_ef_by_fuel_n2o = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_enfu_ef_combustion_stationary_n2o,
            return_type = "array_units_corrected"
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

        ##  BUILD OUTPUT DFs
        df_out += [
            self.model_attributes.array_to_df(arr_scoe_emissions_ch4, self.modvar_scoe_emissions_ch4),
            self.model_attributes.array_to_df(arr_scoe_emissions_co2, self.modvar_scoe_emissions_co2),
            self.model_attributes.array_to_df(arr_scoe_emissions_n2o, self.modvar_scoe_emissions_n2o),
            self.model_attributes.array_to_df(arr_scoe_demand_by_fuel, self.modvar_enfu_energy_demand_by_fuel_scoe),
            self.model_attributes.array_to_df(arr_scoe_demand_electricity, self.modvar_scoe_energy_consumption_electricity),
            self.model_attributes.array_to_df(arr_scoe_demand_electricity_total, self.modvar_scoe_energy_consumption_electricity_agg),
            self.model_attributes.array_to_df(arr_scoe_demand_non_electric + arr_scoe_demand_electricity, self.modvar_scoe_energy_consumption_total),
            self.model_attributes.array_to_df(arr_scoe_demand_non_electric_total + arr_scoe_demand_electricity_total, self.modvar_scoe_energy_consumption_total_agg)
        ]

        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_scoe], False)

        return df_out



    ##  transportation emissions
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
            Calculate emissions from fuel combustion in transportation. Requires NonElectricEnergy.project_transportation_demand() and all variables from the transportation demand sector.

            Function Arguments
            ------------------
            - df_neenergy_trajectories: pd.DataFrame of input variables
            - dvec_pop: np.ndarray vector of population (requires len(vec_rates_gdp) == len(df_neenergy_trajectories))
            - dvec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)
            - dvec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)
            - ddict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.
            - dn_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.
            - dprojection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.

            Notes
            -----
            If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of Energy.project()), self.model_attributes.check_projection_input_df wil be run

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
                "concatenate"
            )

            append_trde_outputs = True


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_trde = self.model_attributes.get_subsector_attribute(self.subsec_name_trde, "pycategory_primary")
        pycat_trns = self.model_attributes.get_subsector_attribute(self.subsec_name_trns, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_trde = self.model_attributes.dict_attributes[pycat_trde]
        attr_trns = self.model_attributes.dict_attributes[pycat_trns]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]
        # add transportation demand to outputs if necessary
        if append_trde_outputs:
            df_out += [
                self.model_attributes.get_standard_variables(
                    df_transport_demand,
                    self.modvar_trde_demand_mtkm,
                    return_type = "data_frame",
                    expand_to_all_cats = False
                ),

                self.model_attributes.get_standard_variables(
                    df_transport_demand,
                    self.modvar_trde_demand_pkm,
                    return_type = "data_frame",
                    expand_to_all_cats = False
                )
            ]



        ############################
        #    MODEL CALCULATIONS    #
        ############################


        ##  START WITH DEMANDS

        # start with demands and map categories in attribute to associated variable
        dict_trns_vars_to_trde_cats = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_trns, 
            "cat_transportation_demand", 
            attr_type = "key_varreqs_partial", 
            skip_none_q = True, 
            return_type = dict, 
            clean_attribute_schema_q = True
        )
        dict_trns_vars_to_trde_cats = sf.reverse_dict(dict_trns_vars_to_trde_cats)
        array_trns_total_passenger_demand = 0.0
        array_trns_total_vehicle_demand = 0.0
        # get occupancy and freight occupancies
        array_trns_avg_load_freight = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_trns_average_vehicle_load_freight,
            return_type = "array_base",
            expand_to_all_cats = True
        )
        array_trns_occ_rate_passenger = self.model_attributes.get_standard_variables(
            df_neenergy_trajectories,
            self.modvar_trns_average_passenger_occupancy,
            return_type = "array_base",
            expand_to_all_cats = True
        )
        # convert average load to same units as demand
        array_trns_avg_load_freight *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trns_average_vehicle_load_freight,
            self.modvar_trde_demand_mtkm,
            "mass"
        )
        # convert freight vehicle demand to same length units as passenger
        scalar_tnrs_length_demfrieght_to_dempass = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trde_demand_mtkm,
            self.modvar_trde_demand_pkm,
            "length"
        )

        # loop over the demand categories to get transportation demand
        for category in dict_trns_vars_to_trde_cats.keys():
            # get key index, model variable, and the current demand
            index_key = self.model_attributes.get_attribute_table(self.subsec_name_trde).get_key_value_index(category)
            modvar = self.model_attributes.get_variable_from_category(self.subsec_name_trde, category, "partial")
            vec_trde_dem_cur = self.model_attributes.get_standard_variables(
                df_neenergy_trajectories, 
                modvar, 
                return_type = "array_base", 
                expand_to_all_cats = True
            )[:, index_key]
            # retrieve the demand mix, convert to total activity-demand by category, then divide by freight/occ_rate
            array_trde_dem_cur_by_cat = self.model_attributes.get_standard_variables(
                df_neenergy_trajectories,
                dict_trns_vars_to_trde_cats[category],
                return_type = "array_base",
                expand_to_all_cats = True,
                var_bounds = (0, 1),
                force_boundary_restriction = True
            )

            array_trde_dem_cur_by_cat = (array_trde_dem_cur_by_cat.transpose()*vec_trde_dem_cur).transpose()
            """
            freight and passenger should be mutually exclusive categories
            - e.g., if the iterating variable category == "freight", then 
                array_trde_dem_cur_by_cat*array_trns_occ_rate_passenger should 
                be 0
            - if category != "freight", then 
                array_trde_dem_cur_by_cat*array_trns_avg_load_freight should 
                be 0
            - demand length units should be in terms of 
                'modvar_trns_average_passenger_occupancy' (see scalar multiplication)
            """
            array_trde_vehicle_dem_cur_by_cat = np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_avg_load_freight, 0.0, neginf = 0.0, posinf = 0.0)*scalar_tnrs_length_demfrieght_to_dempass
            array_trde_vehicle_dem_cur_by_cat += np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_occ_rate_passenger, 0.0, neginf = 0.0, posinf = 0.0)
            # update total passenger distance and vehicle-km demand; note that passenger distance will be reduced to exclude freight categories on output
            array_trns_total_passenger_demand += array_trde_dem_cur_by_cat
            array_trns_total_vehicle_demand += array_trde_vehicle_dem_cur_by_cat

        # add the vehicle and passenger distance to output using the units modvar_trde_demand_pkm
        scalar_trns_total_vehicle_demand = self.model_attributes.get_scalar(self.modvar_trde_demand_pkm, "length")
        df_out += [
            self.model_attributes.array_to_df(
                array_trns_total_passenger_demand*scalar_trns_total_vehicle_demand,
                self.modvar_trns_passenger_distance_traveled,
                include_scalars = False,
                reduce_from_all_cats_to_specified_cats = True
            ),
            self.model_attributes.array_to_df(
                array_trns_total_vehicle_demand*scalar_trns_total_vehicle_demand,
                self.modvar_trns_vehicle_distance_traveled,
                include_scalars = False,
                reduce_from_all_cats_to_specified_cats = True
            )
        ]


        ##  LOOP OVER FUELS

        # first, retrieve fuel-mix fractions and ensure they sum to 1
        dict_arrs_trns_frac_fuel = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            self.modvars_trns_list_fuel_fraction,
            1,
            force_sum_equality = False,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_trns_frac_fuel."
        )
        # get carbon dioxide combustion factors (corrected to output units)
        arr_trns_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected", expand_to_all_cats = True)
        arr_trns_energy_density_fuel = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_energy_density_volumetric, return_type = "array_units_corrected", expand_to_all_cats = True)

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
        scalar_trns_ved_to_output_scalar = self.model_attributes.get_scalar(self.modvar_enfu_energy_density_volumetric,"energy")

        # loop over fuels to calculate emissions and demand associated with each fuel
        fuels_loop = sorted(list(self.dict_trns_fuel_categories_to_fuel_variables.keys()))
        for cat_fuel in fuels_loop:

            # get the index of the current category
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)

            # set some model variables
            dict_tfc_to_fv_cur = self.dict_trns_fuel_categories_to_fuel_variables.get(cat_fuel)
            modvar_trns_ef_ch4_cur = dict_tfc_to_fv_cur.get("ef_ch4")
            modvar_trns_ef_n2o_cur = dict_tfc_to_fv_cur.get("ef_n2o")
            modvar_trns_fuel_efficiency_cur = dict_tfc_to_fv_cur.get("fuel_efficiency")
            modvar_trns_fuel_fraction_cur = dict_tfc_to_fv_cur.get("fuel_fraction")

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
            arr_trns_ef_ch4_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_ef_ch4_cur, return_type = "array_units_corrected", expand_to_all_cats = True) if (modvar_trns_ef_ch4_cur is not None) else 0
            arr_trns_ef_n2o_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_ef_n2o_cur, return_type = "array_units_corrected", expand_to_all_cats = True) if (modvar_trns_ef_n2o_cur is not None) else 0
            arr_trns_fuel_efficiency_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_fuel_efficiency_cur, return_type = "array_base", expand_to_all_cats = True)

            # current demand associate with the fuel (in terms of modvar_trde_demand_pkm)
            arr_trns_vehdem_cur_fuel = array_trns_total_vehicle_demand*arr_trns_fuel_fraction_cur

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

                arr_trns_demand_by_category += arr_trns_energydem_cur_fuel*scalar_trns_ved_to_output_scalar
                arr_trns_demand_by_fuel[:, index_cat_fuel] = np.sum(arr_trns_energydem_cur_fuel, axis = 1)*scalar_trns_ved_to_enfu_var_units


            elif cat_fuel == self.cat_enfu_electricity:

                # get scalar for energy
                scalar_electric_eff_to_distance_equiv = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_trns_electrical_efficiency,
                    self.modvar_trde_demand_pkm,
                    "length"
                )
                # get demand for fuel in terms of modvar_trns_fuel_efficiency_cur, then get scalars to conert to emission factor fuel volume units
                arr_trns_elect_efficiency_cur = self.model_attributes.get_standard_variables(
                    df_neenergy_trajectories,
                    self.modvar_trns_electrical_efficiency,
                    return_type = "array_base",
                    expand_to_all_cats = True
                )
                arr_trns_elect_efficiency_cur *= scalar_electric_eff_to_distance_equiv
                arr_trns_energydem_elec = arr_trns_vehdem_cur_fuel/arr_trns_elect_efficiency_cur

                # write in terms of output units
                arr_trns_energydem_elec *= self.model_attributes.get_scalar(self.modvar_trns_electrical_efficiency, "energy")
                arr_trns_energydem_elec = np.nan_to_num(arr_trns_energydem_elec, posinf = 0, neginf = 0)
                vec_trns_energydem_elec_total = np.sum(arr_trns_energydem_elec, axis = 1)

                # update energy demand by category and fuel
                arr_trns_demand_by_category += arr_trns_energydem_elec
                arr_trns_demand_by_fuel[:, index_cat_fuel] = vec_trns_energydem_elec_total/self.model_attributes.get_scalar(self.modvar_enfu_energy_demand_by_fuel_trns, "energy")

        vec_trns_demand_by_category_total = np.sum(arr_trns_demand_by_category, axis = 1)

        # add all aggregate emissions to output
        df_out += [
            self.model_attributes.array_to_df(arr_trns_emissions_ch4, self.modvar_trns_emissions_ch4),
            self.model_attributes.array_to_df(arr_trns_emissions_co2, self.modvar_trns_emissions_co2),
            self.model_attributes.array_to_df(arr_trns_emissions_n2o, self.modvar_trns_emissions_n2o),
            self.model_attributes.array_to_df(arr_trns_demand_by_fuel, self.modvar_enfu_energy_demand_by_fuel_trns),
            self.model_attributes.array_to_df(arr_trns_demand_by_category, self.modvar_trns_energy_consumption_total, reduce_from_all_cats_to_specified_cats = True),
            self.model_attributes.array_to_df(vec_trns_demand_by_category_total, self.modvar_trns_energy_consumption_total_agg),
            self.model_attributes.array_to_df(arr_trns_energydem_elec, self.modvar_trns_energy_consumption_electricity, reduce_from_all_cats_to_specified_cats = True),
            self.model_attributes.array_to_df(vec_trns_energydem_elec_total, self.modvar_trns_energy_consumption_electricity_agg)
        ]


        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, [self.subsec_name_trns], False)

        return df_out



    ##  transportation demands
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
            Calculate transportation demands and associated metrics.

            Function Arguments
            ------------------
            - df_neenergy_trajectories: pd.DataFrame of input variables
            - vec_pop: np.ndarray vector of population (requires len(vec_rates_gdp) == len(df_neenergy_trajectories))
            - vec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)
            - vec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)
            - dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.
            - n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.
            - projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.

            Notes
            -----
            If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of Energy.project()), self.model_attributes.check_projection_input_df wil be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_trde = self.model_attributes.get_subsector_attribute(self.subsec_name_trde, "pycategory_primary")
        pycat_trns = self.model_attributes.get_subsector_attribute(self.subsec_name_trns, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_trde = self.model_attributes.dict_attributes[pycat_trde]
        attr_trns = self.model_attributes.dict_attributes[pycat_trns]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # get the demand scalar
        array_trde_demscalar = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_scalar, return_type = "array_base", expand_to_all_cats = True, var_bounds = (0, np.inf))
        # start with freight/megaton km demands
        array_trde_dem_init_freight = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_initial_mtkm, return_type = "array_base", expand_to_all_cats = True)
        array_trde_elast_freight_demand_to_gdp = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_elasticity_mtkm_to_gdp, return_type = "array_base", expand_to_all_cats = True)
        array_trde_growth_freight_dem_by_cat = sf.project_growth_scalar_from_elasticity(vec_rates_gdp, array_trde_elast_freight_demand_to_gdp, False, "standard")
        # multiply and add to the output
        array_trde_freight_dem_by_cat = array_trde_dem_init_freight[0]*array_trde_growth_freight_dem_by_cat
        array_trde_freight_dem_by_cat *= array_trde_demscalar
        df_out.append(
            self.model_attributes.array_to_df(array_trde_freight_dem_by_cat, self.modvar_trde_demand_mtkm, False, True)
        )

        # deal with person-km
        array_trde_dem_init_passenger = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_initial_pkm_per_capita, return_type = "array_base", expand_to_all_cats = True)
        array_trde_elast_passenger_demand_to_gdppc = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_elasticity_pkm_to_gdp, return_type = "array_base", expand_to_all_cats = True)
        array_trde_growth_passenger_dem_by_cat = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, array_trde_elast_passenger_demand_to_gdppc, False, "standard")
        # project the growth in per capita, multiply by population, then add it to the output
        array_trde_passenger_dem_by_cat = array_trde_dem_init_passenger[0]*array_trde_growth_passenger_dem_by_cat
        array_trde_passenger_dem_by_cat = (array_trde_passenger_dem_by_cat.transpose()*vec_pop).transpose()
        array_trde_passenger_dem_by_cat *= array_trde_demscalar
        df_out.append(
            self.model_attributes.array_to_df(array_trde_passenger_dem_by_cat, self.modvar_trde_demand_pkm, False, True)
        )

        # build output dataframe
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")

        return df_out



    ##  primary method
    def project(self,
        df_neenergy_trajectories: pd.DataFrame,
        subsectors_project: Union[list, str, None] = None
    ) -> pd.DataFrame:

        """
            Take a data frame of input variables (ordered by time series) and return a data frame of output variables (model projections for energy--including carbon capture and sequestration, fugitive emissions, industrial energy, stationary combustion, and transportation) the same order.

            NOTE: Fugitive Emissions requires output from ElectricEnergy to complete a full accounting for fuel production and use. In SISEPUEDE, integrated runs should be run in the order of:

            NonElectricEnergy.project(*args)
            ElectricEnergy.project(*args)
            NonElectricEnergy.project(*args, subsectors_project = "Fugitive Emissions")

            Function Arguments
            ------------------
            - df_neenergy_trajectories: pd.DataFrame with all required input fields as columns. The model will not run if any required variables are missing, but errors will detail which fields are missing.
            - subsectors_project: list of subsectors or pipe-delimited string of subsectors. If None, run all subsectors EXCEPT for Fugitive Emissions. Valid list entries/subsectors are:
                * "Carbon Capture and Sequestration" or "ccsq"
                * "Fugitive Emissions" or "fgtv"
                * "Industrial Energy" or "inen"
                * "Stationary Combustion and Other Energy" or "scoe"
                * "Transportation" or "trns"

            Notes
            -----
            - The .project() method is designed to be parallelized or called from command line via __main__ in run_sector_models.py.
            - df_neenergy_trajectories should have all input fields required (see Energy.required_variables for a list of variables to be defined)
            - the df_neenergy_trajectories.project() method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.
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
        vec_hh = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_grnl_num_hh, return_type = "array_base")
        vec_gdp = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_econ_gdp, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, return_type = "array_base")
        array_pop = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_gnrl_subpop, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]



        #########################################
        #    MODEL CALCULATIONS BY SUBSECTOR    #
        #########################################

        # add industrial energy, transportation, and SCOE
        if self.subsec_name_ccsq in subsectors_project:
            df_out.append(self.project_ccsq(df_neenergy_trajectories, dict_dims, n_projection_time_periods, projection_time_periods))
        if self.subsec_name_inen in subsectors_project:
            df_out.append(self.project_industrial_energy(df_neenergy_trajectories, vec_gdp, dict_dims, n_projection_time_periods, projection_time_periods))
        if self.subsec_name_scoe in subsectors_project:
            df_out.append(self.project_scoe(df_neenergy_trajectories, vec_hh, vec_gdp, vec_rates_gdp_per_capita, dict_dims, n_projection_time_periods, projection_time_periods))
        if self.subsec_name_trns in subsectors_project:
            df_out.append(self.project_transportation(df_neenergy_trajectories, vec_pop, vec_rates_gdp, vec_rates_gdp_per_capita, dict_dims, n_projection_time_periods, projection_time_periods))
        # run fugitive emissions?
        if self.subsec_name_fgtv in subsectors_project:
            df_trajectories = sf.merge_output_df_list(
                [df_neenergy_trajectories] + df_out,
                self.model_attributes,
                "concatenate"
            )
            df_out.append(self.project_fugitive_emissions(df_trajectories, dict_dims, n_projection_time_periods, projection_time_periods))

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        return df_out
