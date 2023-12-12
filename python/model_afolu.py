from attribute_table import AttributeTable
import itertools
import logging
from model_attributes import *
from model_socioeconomic import Socioeconomic
from model_energy import NonElectricEnergy
from model_ippu import IPPU
import numpy as np
import pandas as pd
import support_functions as sf
import time
from typing import *

##########################
###                    ###
###     AFOLU MODEL    ###
###                    ###
##########################

class AFOLU:
    """
    Use AFOLU to calculate emissions from Agriculture, Forestry, and Land Use in
        SISEPUEDE. Includes emissions from the following subsectors:

        * Agriculture (AGRC)
        * Forestry (FRST)
        * Land Use (LNDU)
        * Livestock (LVST)
        * Livestock Manure Management (LSMM)
        * Soil Management (SOIL)

    For additional information, see the SISEPUEDE readthedocs at:

        https://sisepuede.readthedocs.io/en/latest/afolu.html


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
        self._initialize_input_output_components()


        ##  SET MODEL FIELDS

        self._initialize_subsector_vars_agrc()
        self._initialize_subsector_vars_frst()
        self._initialize_subsector_vars_lndu()
        self._initialize_subsector_vars_lsmm()
        self._initialize_subsector_vars_lvst()
        self._initialize_subsector_vars_soil()

        self._initialize_models()
        self._initialize_integrated_variables()
        self._initialize_other_properties()
    


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


    # assign some land use categories
    def _assign_cats_lndu(self
    ) -> None:

        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        pycat_frst = self.model_attributes.get_subsector_attribute(self.subsec_name_frst, "pycategory_primary")
        
        ##  set categories

        # crop category
        self.cat_lndu_crop = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"crop_category": 1}
        )[0]
        
        # mangrove forest category
        self.cat_lndu_fstm = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu,
            {pycat_frst: f"``{self.cat_frst_mang}``"}
        )[0]
        
        # primary forest category
        self.cat_lndu_fstp = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu,
            {pycat_frst: f"``{self.cat_frst_prim}``"}
        )[0]

        # secondary forest category
        self.cat_lndu_fsts = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu,
            {pycat_frst: f"``{self.cat_frst_scnd}``"}
        )[0]

        # other land use
        self.cat_lndu_othr = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"other_category": 1}
        )[0]

        # grassland, used for pasture
        self.cat_lndu_grass = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"pasture_category": 1}
        )[0]

        # settlements
        self.cat_lndu_stlm = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"settlements_category": 1}
        )[0]
        self.cat_lndu_wetl = self.model_attributes.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"wetlands_category": 1}
        )[0]
        
        # list of categories to use to "max out" transition probabilities when scaling land use prevelance
        self.cats_lndu_max_out_transition_probs = self.model_attributes.get_categories_from_attribute_characteristic(
            self.model_attributes.subsec_name_lndu,
            {
                "reallocation_transition_probability_exhaustion_category": 1
            }
        )

        # assign indices
        self.ind_lndu_crop = attr_lndu.get_key_value_index(self.cat_lndu_crop)
        self.ind_lndu_fstm = attr_lndu.get_key_value_index(self.cat_lndu_fstm)
        self.ind_lndu_fstp = attr_lndu.get_key_value_index(self.cat_lndu_fstp)
        self.ind_lndu_fsts = attr_lndu.get_key_value_index(self.cat_lndu_fsts)
        self.ind_lndu_othr = attr_lndu.get_key_value_index(self.cat_lndu_othr)
        self.ind_lndu_grass = attr_lndu.get_key_value_index(self.cat_lndu_grass)
        self.ind_lndu_stlm = attr_lndu.get_key_value_index(self.cat_lndu_stlm)
        self.ind_lndu_wetl = attr_lndu.get_key_value_index(self.cat_lndu_wetl)

        return None



    def _assign_lndu_frst_dicts(self,
    ) -> None:
        """
        Generate dictionaries mapping land use categories to forest categories.
            Sets the following properties:

            * self.dict_cats_lndu_to_cats_frst
            * self.dict_cats_frst_to_cats_lndu
        """
        pycat_frst = self.model_attributes.get_subsector_attribute(
            self.model_attributes.subsec_name_frst,
            "pycategory_primary"
        )

        dict_lndu_to_frst = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_lndu,
            pycat_frst,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )
        dict_frst_to_lndu = sf.reverse_dict(dict_lndu_to_frst)
        
        self.dict_cats_frst_to_cats_lndu = dict_frst_to_lndu
        self.dict_cats_lndu_to_cats_frst = dict_lndu_to_frst

        return None



    def check_df_fields(self, 
        df_afolu_trajectories: pd.DataFrame, 
        check_fields: Union[List[str], None] = None
    ) -> None:
        check_fields = self.required_variables if (check_fields is None) else check_fields
        # check for required variables
        if not set(check_fields).issubset(df_afolu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_afolu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"AFOLU projection cannot proceed: The fields {set_missing} are missing.")

        return None
    


    def get_lvst_dict_lsmm_categories_to_lvst_fraction_variables(self,
    ) -> Dict:
        """
        Return a dictionary with Livestock Manure Management categories as keys 
            based on the Livestock attribute table:

            {
                cat_lsmm: {
                    "mm_fraction": VARNAME_MANURE_MANAGEMENT, 
                    ...
                }
            }

            for each key, the dict includes variables associated with the 
                livestock manure management cat_lsmm: 

            - "mm_fraction"
        """
        subsec_lvst = self.model_attributes.subsec_name_lvst
        subsec_lsmm = self.model_attributes.subsec_name_lsmm
        pycat_lsmm = self.model_attributes.get_subsector_attribute(
            subsec_lsmm, 
            "pycategory_primary"
        )

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            subsec_lvst,
            pycat_lsmm,
            {
                "Livestock Manure Management Fraction": "mm_fraction"
            },
            "varreqs_all",
            True
        )

        return dict_out



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
        subsectors = self.model_attributes.get_sector_subsectors("AFOLU")
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



    def _initialize_integrated_variables(self
    ) -> None:
        """
        Initialize variables required for integration. Sets the following 
            properties:

            * self.dict_integration_variables_by_subsector
            * self.integration_variables
        """
        dict_vars_required_for_integration = {
            # ippu variables required for estimating HWP
            self.subsec_name_ippu: [
                self.model_ippu.modvar_ippu_average_lifespan_housing,
                self.model_ippu.modvar_ippu_change_net_imports,
                self.model_ippu.modvar_ippu_demand_for_harvested_wood,
                self.model_ippu.modvar_ippu_elast_ind_prod_to_gdp,
                self.model_ippu.modvar_ippu_max_recycled_material_ratio,
                self.model_ippu.model_socioeconomic.modvar_grnl_num_hh,
                self.model_ippu.modvar_ippu_prod_qty_init,
                self.model_ippu.modvar_ippu_qty_recycled_used_in_production,
                self.model_ippu.modvar_ippu_qty_total_production,
                self.model_ippu.modvar_ippu_ratio_of_production_to_harvested_wood,
                self.model_ippu.modvar_waso_waste_total_recycled
            ],
            # SCOE variables required for projecting changes to wood energy demand
            self.subsec_name_scoe: [
                self.model_energy.modvar_scoe_consumpinit_energy_per_hh_elec,
                self.model_energy.modvar_scoe_consumpinit_energy_per_hh_heat,
                self.model_energy.modvar_scoe_consumpinit_energy_per_mmmgdp_elec,
                self.model_energy.modvar_scoe_consumpinit_energy_per_mmmgdp_heat,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_coal,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_diesel,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_electricity,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_gasoline,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_hgl,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_hydrogen,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_kerosene,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_natural_gas,
                self.model_energy.modvar_scoe_efficiency_fact_heat_en_solid_biomass,
                self.model_energy.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc,
                self.model_energy.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc,
                self.model_energy.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc,
                self.model_energy.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc,
                self.model_energy.modvar_scoe_frac_heat_en_coal,
                self.model_energy.modvar_scoe_frac_heat_en_diesel,
                self.model_energy.modvar_scoe_frac_heat_en_electricity,
                self.model_energy.modvar_scoe_frac_heat_en_gasoline,
                self.model_energy.modvar_scoe_frac_heat_en_hgl,
                self.model_energy.modvar_scoe_frac_heat_en_hydrogen,
                self.model_energy.modvar_scoe_frac_heat_en_kerosene,
                self.model_energy.modvar_scoe_frac_heat_en_natural_gas,
                self.model_energy.modvar_scoe_frac_heat_en_solid_biomass
            ]
        }

        # set complete output list of integration variables
        list_vars_required_for_integration = []
        for k in dict_vars_required_for_integration.keys():
            list_vars_required_for_integration += dict_vars_required_for_integration[k]

        self.dict_integration_variables_by_subsector = dict_vars_required_for_integration
        self.integration_variables = list_vars_required_for_integration

        return None



    def _initialize_models(self,
        model_attributes: Union[ModelAttributes, None] = None
    ) -> None:
        """
        Initialize SISEPUEDE model classes for fetching variables and 
            accessing methods. Initializes the following properties:

            * self.model_energy
            * self.model_ippu
            * self.model_socioeconomic

            as well as additional associated categories, such as 

            * self.cat_enfu_biomass
            * self.cat_ippu_paper
            * self.cat_ippu_wood


        Keyword Arguments
        -----------------
        - model_attributes: ModelAttributes object used to instantiate
            models. If None, defaults to self.model_attributes.
        """

        model_attributes = self.model_attributes if (model_attributes is None) else model_attributes

        # add other model classes--required for integration variables
        self.model_energy = NonElectricEnergy(model_attributes)
        self.model_ippu = IPPU(model_attributes)
        self.model_socioeconomic = Socioeconomic(model_attributes)

        # key categories
        self.cat_enfu_biomass = model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {"biomass_demand_category": 1})[0]
        self.cat_ippu_paper = model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_ippu, {"virgin_paper_category": 1})[0]
        self.cat_ippu_wood = model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_ippu, {"virgin_wood_category": 1})[0]
        
        return None



    def _initialize_other_properties(self,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.factor_c_to_co2
            * self.factor_n2on_to_n2o
            * self.n_time_periods
            * self.time_dependence_stock_change
            * self.time_periods
        """
        
        # from IPCC docs
        self.factor_c_to_co2 = float(11/3)
        self.factor_n2on_to_n2o = float(11/7)

        # see IPCC 2006/2019R GNGHGI V4 CH2 FOR D = 20
        self.time_dependence_stock_change = 20

        # time variables
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

        return None


    def _initialize_subsector_vars_agrc(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            AGRC (Agriculture). Sets the following properties:

            * self.cat_agrc_****
            * self.ind_agrc_****
            * self.modvar_agrc_****
            * self.modvar_dict_agrc_****
            * self.modvar_list_agrc_****
        """
        # agricultural model variables
        self.modvar_agrc_adjusted_equivalent_exports = "Adjusted Agriculture Equivalent Exports"
        self.modvar_agrc_adjusted_equivalent_imports = "Adjusted Agriculture Equivalent Imports"
        self.modvar_agrc_area_prop_calc = "Cropland Area Proportion"
        self.modvar_agrc_area_prop_init = "Initial Cropland Area Proportion"
        self.modvar_agrc_area_crop = "Crop Area"
        self.modvar_agrc_changes_to_net_imports_lost = "Changes to Agriculture Net Imports Lost"
        self.modvar_agrc_combustion_factor = "AGRC Combustion Factor"
        self.modvar_agrc_demand_crops = "Crop Demand"
        self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Anaerobic Decomposition Emission Factor"
        self.modvar_agrc_ef_ch4_burning = ":math:\\text{CH}_4 Crop Biomass Burning Emission Factor"
        self.modvar_agrc_ef_co2_biomass = ":math:\\text{CO}_2 Crop Biomass Emission Factor"
        self.modvar_agrc_ef_n2o_burning = ":math:\\text{N}_2\\text{O} Crop Biomass Burning Emission Factor"
        self.modvar_agrc_ef_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Crop Fertilizer and Lime Emission Factor"
        self.modvar_agrc_elas_crop_demand_income = "Crop Demand Income Elasticity"
        self.modvar_agrc_emissions_ch4_biomass_burning = ":math:\\text{CH}_4 Emissions from Biomass Burning"
        self.modvar_agrc_emissions_ch4_rice = ":math:\\text{CH}_4 Emissions from Rice"
        self.modvar_agrc_emissions_co2_biomass = ":math:\\text{CO}_2 Emissions from Biomass Carbon Stock Changes"
        self.modvar_agrc_emissions_co2_soil_carbon_organic = ":math:\\text{CO}_2 Emissions from Soil Carbon in Organic Soils"
        self.modvar_agrc_emissions_n2o_biomass_burning = ":math:\\text{N}_2\\text{O} Emissions from Biomass Burning"
        self.modvar_agrc_emissions_n2o_crop_residues = ":math:\\text{N}_2\\text{O} Emissions from Crop Residues"
        self.modvar_agrc_equivalent_exports = "Agriculture Equivalent Exports"
        self.modvar_agrc_frac_animal_feed = "Crop Fraction Animal Feed"
        self.modvar_agrc_frac_demand_imported = "Fraction of Agriculture Demand Imported"
        self.modvar_agrc_frac_dry = "Agriculture Fraction Dry"
        self.modvar_agrc_frac_dry_matter_in_crop = "Dry Matter Fraction of Harvested Crop"
        self.modvar_agrc_frac_no_till = "No Till Crop Fraction"
        self.modvar_agrc_frac_production_lost = "Fraction of Food Produced Lost Before Consumption"
        self.modvar_agrc_frac_production_loss_to_msw = "Fraction of Food Loss Sent to Municipal Solid Waste"
        self.modvar_agrc_frac_residues_removed = "Fraction of Residues Removed"
        self.modvar_agrc_frac_residues_burned = "Fraction of Residues Burned"
        self.modvar_agrc_frac_temperate = "Agriculture Fraction Temperate"
        self.modvar_agrc_frac_tropical = "Agriculture Fraction Tropical"
        self.modvar_agrc_frac_wet = "Agriculture Fraction Wet"
        self.modvar_agrc_n_content_of_above_ground_residues = "N Content of Above Ground Residues"
        self.modvar_agrc_n_content_of_below_ground_residues = "N Content of Below Ground Residues"
        self.modvar_agrc_ratio_above_ground_residue_to_harvested_yield = "Ratio of Above Ground Residue to Harvested Yield"
        self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass = "Ratio of Below Ground Biomass to Above Ground Biomass"
        self.modvar_agrc_regression_m_above_ground_residue = "Above Ground Residue Dry Matter Slope"
        self.modvar_agrc_regression_b_above_ground_residue = "Above Ground Residue Dry Matter Intercept"
        self.modvar_agrc_total_food_lost_in_ag = "Total Food Produced Lost Before Consumption"
        self.modvar_agrc_total_food_lost_in_ag_to_msw = "Total Food Loss Sent to Municipal Solid Waste"
        self.modvar_agrc_yf = "Crop Yield Factor"
        self.modvar_agrc_yield = "Crop Yield"
        
        # additional lists
        self.modvar_list_agrc_frac_drywet = [
            self.modvar_agrc_frac_dry,
            self.modvar_agrc_frac_wet
        ]
        self.modvar_list_agrc_frac_temptrop = [
            self.modvar_agrc_frac_temperate,
            self.modvar_agrc_frac_tropical
        ]
        self.modvar_list_agrc_frac_residues_removed_burned = [
            self.modvar_agrc_frac_residues_burned,
            self.modvar_agrc_frac_residues_removed
        ]
        # some key categories
        self.cat_agrc_rice = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_agrc, {"rice_category": 1})[0]

        return None



    def _initialize_subsector_vars_frst(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            FRST (Forest). Sets the following properties:

            * self.cat_frst_****
            * self.ind_frst_****
            * self.modvar_frst_****
            * self.modvar_dict_frst_****
            * self.modvar_list_frst_****
        """

        # forest model variables
        self.modvar_frst_average_fraction_burned_annually = "Average Fraction of Forest Burned Annually"
        self.modvar_frst_biomass_consumed_fire_temperate = "Fire Biomass Consumption for Temperate Forests"
        self.modvar_frst_biomass_consumed_fire_tropical = "Fire Biomass Consumption for Tropical Forests"
        self.modvar_frst_ef_c_per_hwp = "C Carbon Harvested Wood Products Emission Factor"
        self.modvar_frst_ef_co2_fires = ":math:\\text{CO}_2 Forest Fire Emission Factor"
        self.modvar_frst_ef_ch4 = ":math:\\text{CH}_4 Forest Methane Emissions"
        self.modvar_frst_emissions_co2_fires = ":math:\\text{CO}_2 Emissions from Forest Fires"
        self.modvar_frst_emissions_co2_hwp = ":math:\\text{CO}_2 Emissions from Harvested Wood Products"
        self.modvar_frst_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Forests"
        self.modvar_frst_emissions_co2_sequestration = ":math:\\text{CO}_2 Emissions from Forest Sequestration"
        self.modvar_frst_frac_temperate_nutrient_poor = "Forest Fraction Temperate Nutrient Poor"
        self.modvar_frst_frac_temperate_nutrient_rich = "Forest Fraction Temperate Nutrient Rich"
        self.modvar_frst_frac_tropical = "Forest Fraction Tropical"
        self.modvar_frst_hwp_half_life_paper = "HWP Half Life Paper"
        self.modvar_frst_hwp_half_life_wood = "HWP Half Life Wood"
        self.modvar_frst_sq_co2 = "Forest Sequestration Emission Factor"
        self.modvar_frst_init_per_hh_wood_demand = "Initial Per Household Wood Demand"
        #additional lists
        self.modvar_list_frst_frac_temptrop = [
            self.modvar_frst_frac_temperate_nutrient_poor,
            self.modvar_frst_frac_temperate_nutrient_rich,
            self.modvar_frst_frac_tropical
        ]

        # assign some key categories
        self.cat_frst_mang = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_frst, {"mangroves_forest_category": 1})[0]
        self.cat_frst_prim = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_frst, {"primary_forest_category": 1})[0]
        self.cat_frst_scnd = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_frst, {"secondary_forest_category": 1})[0]


        return None



    def _initialize_subsector_vars_lndu(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LNDU (Land Use). Sets the following properties:

            * self.cat_lndu_****
            * self.ind_lndu_****
            * self.modvar_lndu_****
            * self.modvar_dict_lndu_****
            * self.modvar_list_lndu_****
        """
        # land use model variables
        self.modvar_lndu_area_by_cat = "Land Use Area"
        self.modvar_lndu_area_improved = "Area of Improved Land"
        self.modvar_lndu_area_converted = "Area of Land Use Converted"
        self.modvar_lndu_area_converted_from_type = "Area of Land Use Converted Away from Type"
        self.modvar_lndu_area_converted_to_type = "Area of Land Use Converted to Type"
        self.modvar_lndu_ef_co2_conv = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
        self.modvar_lndu_emissions_conv = ":math:\\text{CO}_2 Emissions from Land Use Conversion"
        self.modvar_lndu_emissions_conv_away = ":math:\\text{CO}_2 Emissions from Conversion Away from Land Use Type"
        self.modvar_lndu_emissions_ch4_from_wetlands = ":math:\\text{CH}_4 Emissions from Wetlands"
        self.modvar_lndu_factor_soil_carbon = "Soil Carbon Land Use Factor"
        self.modvar_lndu_factor_soil_management_infinum = "Unimproved Soil Carbon Land Management Factor"
        self.modvar_lndu_factor_soil_management_supremum = "Maximum Soil Carbon Land Management Factor"
        self.modvar_lndu_factor_soil_inputs_supremum_no_manure = "Maximum Soil Carbon Land Input Factor Without Manure"
        self.modvar_lndu_frac_dry = "Land Use Fraction Dry"
        self.modvar_lndu_frac_fertilized = "Land Use Fraction Fertilized"
        self.modvar_lndu_frac_grassland_that_is_pasture = "Pasture Fraction of Grassland"
        self.modvar_lndu_frac_increasing_net_exports_met = "Fraction of Increasing Net Exports Met"
        self.modvar_lndu_frac_increasing_net_imports_met = "Fraction of Increasing Net Imports Met"
        self.modvar_lndu_frac_mineral_soils = "Fraction of Soils Mineral"
        self.modvar_lndu_frac_pastures_improved = "Fraction of Pastures Improved"
        self.modvar_lndu_frac_temperate = "Land Use Fraction Temperate"
        self.modvar_lndu_frac_tropical = "Land Use Fraction Tropical"
        self.modvar_lndu_frac_wet = "Land Use Fraction Wet"
        self.modvar_lndu_initial_frac = "Initial Land Use Area Proportion"
        self.modvar_lndu_ef_ch4_boc = "Land Use BOC :math:\\text{CH}_4 Emission Factor"
        self.modvar_lndu_prob_transition = "Unadjusted Land Use Transition Probability"
        self.modvar_lndu_reallocation_factor = "Land Use Yield Reallocation Factor"
        self.modvar_lndu_vdes = "Vegetarian Diet Exchange Scalar"

        # additional lists
        self.modvar_list_lndu_frac_drywet = [
            self.modvar_lndu_frac_dry,
            self.modvar_lndu_frac_wet
        ]
        self.modvar_list_lndu_frac_temptrop = [
            self.modvar_lndu_frac_temperate,
            self.modvar_lndu_frac_tropical
        ]
        
        # some key categories
        self._assign_cats_lndu()
        self._assign_lndu_frst_dicts()

        return None



    def _initialize_subsector_vars_lsmm(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LSMM (Livestock Manure Management). Sets the following 
            properties:

            * self.cat_lsmm_****
            * self.ind_lsmm_****
            * self.modvar_lsmm_****
            * self.modvar_dict_lsmm_****
        """
        
        # manure management variables
        self.modvar_lsmm_dung_incinerated = "Dung Incinerated"
        self.modvar_lsmm_ef_direct_n2o = ":math:\\text{N}_2\\text{O} Manure Management Emission Factor"
        self.modvar_lsmm_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Manure Management"
        self.modvar_lsmm_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Manure Management"
        self.modvar_lsmm_frac_loss_leaching = "Fraction of Nitrogen Lost to Leaching"
        self.modvar_lsmm_frac_loss_volatilisation = "Fraction of Nitrogen Lost to Volatilisation"
        self.modvar_lsmm_frac_n_available_used = "Fraction of Nitrogen Used in Fertilizer"
        self.modvar_lsmm_mcf_by_pathway = "Manure Management Methane Correction Factor"
        self.modvar_lsmm_n_from_bedding = "Nitrogen from Bedding per Animal"
        self.modvar_lsmm_n_from_codigestates = "Nitrogen from Co-Digestates Factor"
        self.modvar_lsmm_n_to_pastures = "Total Nitrogen to Pastures"
        self.modvar_lsmm_n_to_fertilizer = "Nitrogen Available for Fertilizer"
        self.modvar_lsmm_n_to_fertilizer_agg_dung = "Total Nitrogen Available for Fertilizer from Dung"
        self.modvar_lsmm_n_to_fertilizer_agg_urine = "Total Nitrogen Available for Fertilizer from Urine"
        self.modvar_lsmm_n_to_other_use = "Total Nitrogen Available for Construction/Feed/Other"
        self.modvar_lsmm_ratio_n2_to_n2o = "Ratio of :math:\\text{N}_2 to :math:\\text{N}_2\\text{O}"
        self.modvar_lsmm_recovered_biogas = "LSMM Biogas Recovered from Anaerobic Digesters"
        self.modvar_lsmm_rf_biogas = "Biogas Recovery Factor at LSMM Anaerobic Digesters"

        # some categories
        self.cat_lsmm_incineration = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"incineration_category": 1})[0]
        self.cat_lsmm_pasture = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"pasture_category": 1})[0]

        return None



    def _initialize_subsector_vars_lvst(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LVST (Livestock). Sets the following properties:

            * self.cat_lvst_****
            * self.ind_lvst_****
            * self.modvar_lvst_****
            * self.modvar_dict_lvst_****
        """
        # livestock model variables
        self.modvar_lvst_adjusted_equivalent_exports = "Adjusted Livestock Equivalent Exports"
        self.modvar_lvst_adjusted_equivalent_imports = "Adjusted Livestock Equivalent Imports"
        self.modvar_lvst_animal_weight = "Animal Weight"
        self.modvar_lvst_carrying_capacity_scalar = "Carrying Capacity Scalar"
        self.modvar_lvst_changes_to_net_imports_lost = "Changes to Livestock Net Imports Lost"
        self.modvar_lvst_demand_livestock = "Livestock Demand"
        self.modvar_lvst_dry_matter_consumption = "Daily Dry Matter Consumption"
        self.modvar_lvst_ef_ch4_ef = ":math:\\text{CH}_4 Enteric Fermentation Emission Factor"
        self.modvar_lvst_elas_lvst_demand = "Elasticity of Livestock Demand to GDP per Capita"
        self.modvar_lvst_emissions_ch4_ef = ":math:\\text{CH}_4 Emissions from Livestock Enteric Fermentation"
        self.modvar_lvst_equivalent_exports = "Livestock Equivalent Exports"
        self.modvar_lvst_frac_demand_imported = "Fraction of Livestock Demand Imported"
        self.modvar_lvst_frac_exc_n_in_dung = "Fraction Nitrogen Excretion in Dung"
        self.modvar_lvst_frac_mm_anaerobic_digester = "Livestock Manure Management Fraction Anaerobic Digester"
        self.modvar_lvst_frac_mm_anaerobic_lagoon = "Livestock Manure Management Fraction Anaerobic Lagoon"
        self.modvar_lvst_frac_mm_composting = "Livestock Manure Management Fraction Composting"
        self.modvar_lvst_frac_mm_daily_spread = "Livestock Manure Management Fraction Daily Spread"
        self.modvar_lvst_frac_mm_deep_bedding = "Livestock Manure Management Fraction Deep Bedding"
        self.modvar_lvst_frac_mm_dry_lot = "Livestock Manure Management Fraction Dry Lot"
        self.modvar_lvst_frac_mm_incineration = "Livestock Manure Management Fraction Incineration"
        self.modvar_lvst_frac_mm_liquid_slurry = "Livestock Manure Management Fraction Liquid Slurry"
        self.modvar_lvst_frac_mm_poultry_manure = "Livestock Manure Management Fraction Poultry Manure"
        self.modvar_lvst_frac_mm_ppr = "Livestock Manure Management Fraction Paddock Pasture Range"
        self.modvar_lvst_frac_mm_solid_storage = "Livestock Manure Management Fraction Solid Storage"
        self.modvar_lvst_genfactor_nitrogen = "Daily Nitrogen Generation Factor"
        self.modvar_lvst_genfactor_volatile_solids = "Daily Volatile Solid Generation Factor"
        self.modvar_lvst_b0_manure_ch4 = "Maximum Manure :math:\\text{CH}_4 Generation Capacity"
        self.modvar_lvst_pop = "Livestock Head Count"
        self.modvar_lvst_pop_init = "Initial Livestock Head Count"
        self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"

        # dictionaries and list variables
        tup = self.get_lvst_dict_lsmm_categories_to_lvst_fraction_variables()
        modvar_list_lvst_mm_fractions = [v.get("mm_fraction") for v in tup[0].values()]
        modvar_list_lvst_mm_fractions = [x for x in modvar_list_lvst_mm_fractions if (x is not None)]

        self.dict_lsmm_categories_to_lvst_fraction_variables = tup[0]
        self.dict_lsmm_categories_to_unassigned_variables = tup[1]
        self.modvar_list_lvst_mm_fractions = modvar_list_lvst_mm_fractions
        """
        self.modvar_list_lvst_mm_fractions = [
            self.modvar_lvst_frac_mm_anaerobic_digester,
            self.modvar_lvst_frac_mm_anaerobic_lagoon,
            self.modvar_lvst_frac_mm_composting,
            self.modvar_lvst_frac_mm_daily_spread,
            self.modvar_lvst_frac_mm_deep_bedding,
            self.modvar_lvst_frac_mm_dry_lot,
            self.modvar_lvst_frac_mm_incineration,
            self.modvar_lvst_frac_mm_liquid_slurry,
            self.modvar_lvst_frac_mm_poultry_manure,
            self.modvar_lvst_frac_mm_ppr,
            self.modvar_lvst_frac_mm_solid_storage
        ]
        """;
        return None



    def _initialize_subsector_vars_soil(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            SOIL (Soil Management). Sets the following properties:

            * self.cat_soil_****
            * self.ind_soil_****
            * self.modvar_soil_****
            * self.modvar_dict_soil_****
        """
        # soil management variables
        self.modvar_soil_demscalar_fertilizer = "Fertilizer N Demand Scalar"
        self.modvar_soil_demscalar_liming = "Liming Demand Scalar"
        self.modvar_soil_ef1_n_managed_soils_rice = ":math:\\text{EF}_{1FR} - \\text{N}_2\\text{O} Rice Fields"
        self.modvar_soil_ef1_n_managed_soils_org_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Organic Amendments and Fertilizer"
        self.modvar_soil_ef1_n_managed_soils_syn_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Synthetic Fertilizer"
        self.modvar_soil_ef2_n_organic_soils = ":math:\\text{EF}_2 - \\text{N}_2\\text{O} Emissions from Drained and Managed Organic Soils"
        self.modvar_soil_ef3_n_prp = "EF3 N Pasture Range and Paddock"
        self.modvar_soil_ef4_n_volatilisation = "EF4 N Volatilisation and Re-Deposition Emission Factor"
        self.modvar_soil_ef5_n_leaching = "EF5 N Leaching and Runoff Emission Factor"
        self.modvar_soil_ef_c_liming_dolomite = "C Liming Emission Factor Dolomite"
        self.modvar_soil_ef_c_liming_limestone = "C Liming Emission Factor Limestone"
        self.modvar_soil_ef_c_organic_soils = "C Annual Cultivated Organic Soils Emission Factor"
        self.modvar_soil_ef_c_urea = "C Urea Emission Factor"
        self.modvar_soil_emissions_co2_lime = ":math:\\text{CO}_2 Emissions from Lime"
        self.modvar_soil_emissions_co2_soil_carbon_mineral = ":math:\\text{CO}_2 Emissions from Soil Carbon in Mineral Soils"
        self.modvar_soil_emissions_co2_urea = ":math:\\text{CO}_2 Emissions from Urea"
        self.modvar_soil_emissions_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Emissions from Fertilizer Use"
        self.modvar_soil_emissions_n2o_mineral_soils = ":math:\\text{N}_2\\text{O} Emissions from Mineral Soils"
        self.modvar_soil_emissions_n2o_organic_soils = ":math:\\text{N}_2\\text{O} Emissions from Organic Soils"
        self.modvar_soil_emissions_n2o_ppr = ":math:\\text{N}_2\\text{O} Emissions from Paddock Pasture and Range"
        self.modvar_soil_frac_n_lost_leaching = "Leaching Fraction of N Lost"
        self.modvar_soil_frac_n_lost_volatilisation_on = "Volatilisation Fraction from Organic Amendments and Fertilizers"
        self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea = "Volatilisation Fraction from Non-Urea Synthetic Fertilizers"
        self.modvar_soil_frac_n_lost_volatilisation_sn_urea = "Volatilisation Fraction from Urea Synthetic Fertilizers"
        self.modvar_soil_frac_synethic_fertilizer_urea = "Fraction Synthetic Fertilizer Use Urea"
        self.modvar_soil_fertuse_final_organic = "Organic Fertilizer N Use"
        self.modvar_soil_fertuse_final_synthetic = "Synthetic Fertilizer N Use"
        self.modvar_soil_fertuse_final_total = "Total Fertilizer N Use"
        self.modvar_soil_fertuse_init_synthetic = "Initial Synthetic Fertilizer Use"
        self.modvar_soil_limeuse_total = "Total Lime Use"
        self.modvar_soil_organic_c_stocks = "Soil Organic C Stocks"
        self.modvar_soil_qtyinit_liming_dolomite = "Initial Liming Dolomite Applied to Soils"
        self.modvar_soil_qtyinit_liming_limestone = "Initial Liming Limestone Applied to Soils"
        self.modvar_soil_ratio_c_to_n_soil_organic_matter = "C to N Ratio of Soil Organic Matter"
        self.modvar_soil_ureause_total = "Total Urea Use"

        return None





    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################

    ###   AGRICULTURE

    def check_cropland_fractions(self,
        df_in: pd.DataFrame,
        frac_type: str = "initial",
        thresh_for_correction: float = 0.01,
    ) -> np.ndarray:
        """
        Check cropland fractions and extract from data frame. Returns np.ndarray
            ordered by AGRC attribute keys
        """
        if frac_type not in ["initial", "calculated"]:
            raise ValueError(f"Error in frac_type '{frac_type}': valid values are 'initial' and 'calculated'.")

        varname = (
            self.modvar_agrc_area_prop_init 
            if frac_type == "initial"
            else self.modvar_agrc_area_prop_calc
        )

        arr = self.model_attributes.get_standard_variables(
            df_in, 
            varname, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base"
        )

        totals = sum(arr.transpose())
        m = max(np.abs(totals - 1))
        if m > thresh_for_correction:
            msg = f"""
                Invalid crop areas found in check_cropland_fractions. The
                maximum fraction total was {m}; the maximum allowed deviation 
                from 1 is {thresh_for_correction}.
            """
            raise ValueError(msg)

        arr = (arr.transpose()/totals).transpose()

        return arr



    ###   LAND USE

    def adjust_transition_matrix(self,
        mat: np.ndarray,
        dict_tuples_scale: dict,
        ignore_diag_on_col_scale: bool = False,
        mat_bounds: tuple = (0, 1),
        response_columns = None
    ) -> np.ndarray:
        """
        Rescale elements of a row-stochastic transition matrix Q (n x n) to 
            account for scalars applied to columns or entries defined in 
            dict_tuples_scale. The columns that are adjusted in response to 
            these exogenous scalars are said to be to subject to automated 
            rescaling.

        dict_tuples_scale is the mechanism for passing scalars to apply to a 
            transition matrix It accepts two types of tuples for keys

            - to scale an entire column, enter a single tuple (j, )
            - to scale a point, use (j, k)
            
        For example,

            dict_tuples_scale = {(i, ): scalar_1, (j, k): scalar_2}

        will scale:
            * all transition probabilities in column i using scalar_1
            * the transition probabilty at (j, k) using scalar_2
            * all other probabilities in a given row uniformly to ensure 
                summation to 1

        Function Arguments
        ------------------
        - mat: row-stochastic transition matrix to apply scalars to
        - dict_tuples_scale: dictionary of tuples defining columnar or 
            point-based probabilities to scale

        Keyword Arguments
        -----------------
        - ignore_diag_on_col_scale: if True, diagonals on the transition matrix 
            are not scaled in response to other changing probabilties.
        - mat_bounds: bounds for elements in the matrix (weak inequalities).
        - response_columns: the columns in the matrix that are subject to 
            automated rescaling in response to to exogenous scalars. If None, 
            then, for each row, columns that are not affected by exogenous 
            scalars are subject to automated rescaling.



        Example and Notes
        -----------------
        * The final transition matrix may not reflect the scalars that are 
            passed. For example, considr the matrix

            array([
                [0.5, 0, 0.5],
                [0.2, 0.7, 0.1],
                [0.0, 0.1, 0.9]
            ])

        if dict_tuples_scale = {(0, ): 2, (0, 2): 1.4}, then, before rescaling 
            probabilities that are not specified in dict_tuples_scale, the 
            matrix becomes

            array([
                [1.0, 0, 0.7],
                [0.4, 0.7, 0.1],
                [0.0, 0.1, 0.9]
            ])

        Since all of the non-zero elements in the first row are subject to 
            scaling, the normalization to sum to 1 reduces the effect of the 
            specified scalar. The final matrix becomes (approximately)

            array([
                [0.5883, 0, 0.4117],
                [0.4, 0.525, 0.075],
                [0.0, 0.1, 0.9]
            ])

        * NOTE: Applying large scalars will lead to dominance, and eventually 
            singular values.

        """

        # assume that the matrix is square - get the scalar, then get the mask to use adjust transition probabilities not specified as a scalar
        mat_scale = np.ones(mat.shape)
        mat_pos_scale = np.zeros(mat.shape)
        mat_mask = np.ones(mat.shape)
        
        # assign columns that will be adjusted in response to changes - default to all that aren't scaled
        if (response_columns == None):
            mat_mask_response_nodes = np.ones(mat.shape)
        else:
            mat_mask_response_nodes = np.zeros(mat.shape)
            for col in [x for x in response_columns if x < mat.shape[0]]:
                mat_mask_response_nodes[:, col] = 1

        m = mat_scale.shape[0]

        ##  PERFORM SCALING

        # adjust columns first
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 1]:
            # overwrite the column
            mat_scale[:, ind[0]] = np.ones(m)*dict_tuples_scale[ind]
            mat_pos_scale[:, ind[0]] = np.ones(m)
            mat_mask[:, ind[0]] = np.zeros(m)

        # it may be of interest to ignore the diagonals when scaling columns
        if ignore_diag_on_col_scale:
            mat_diag = np.diag(tuple(np.ones(m)))
            # reset ones on the diagonal
            mat_scale = (np.ones(mat.shape) - mat_diag)*mat_scale + mat_diag
            mat_pos_scale = sf.vec_bounds(mat_pos_scale - mat_diag, (0, 1))
            mat_mask =  sf.vec_bounds(mat_mask + mat_diag, (0, 1))
        
        # next, adjust points - operate on the transpose of the matrix
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 2]:
            i, j = ind
            mat_scale[i, j] = dict_tuples_scale.get(ind)
            mat_pos_scale[i, j] = 1
            mat_mask[i, j] = 0


        """
        Get the total that needs to be removed from masked elements (those that 
            are not scaled)

        NOTE: bound scalars at the low end by 0 (if 
            mask_shift_total_i > sums_row_mask_i, then the scalar is negative.
            This occurs if the row total of the adjusted values exceeds 1.Set 
            `mask_scalar` using a minimum value of 0 and implement row 
            normalization—if there's no way to rebalance response columns, 
            everything gets renormalized. We correct for this below by 
            implementing row normalization to mat_out.
        """
        # get new mat and restrict values to 0, 1
        mat_new_scaled = sf.vec_bounds(mat*mat_scale, mat_bounds)
        sums_row = sum(mat_new_scaled.transpose())
        sums_row_mask = sum((mat_mask_response_nodes*mat_mask*mat).transpose())
        
        # get shift and positive scalar to apply to valid masked elements
        mask_shift_total = sums_row - 1
        mask_scalar = np.nan_to_num(
            sf.vec_bounds(
                (sums_row_mask - mask_shift_total)/sums_row_mask, 
                (0, np.inf)
            ), 
            1.0, 
            posinf = 1.0
        )

        # get the masked nodes, multiply by the response scalar (at applicable columns, denoted by mat_mask_response_nodes), then add to
        mat_out = ((mat_mask_response_nodes*mat_mask*mat).transpose() * mask_scalar).transpose()
        mat_out += sf.vec_bounds(mat_mask*(1 - mat_mask_response_nodes), (0, 1))*mat
        mat_out += mat_pos_scale*mat_new_scaled
        mat_out = (mat_out.transpose()/sum(mat_out.transpose())).transpose()

        mat_out = sf.vec_bounds(mat_out, mat_bounds)

        return mat_out



    def calculate_ipcc_soc_deltas(self,
        vec_soc: np.ndarray,
        approach: int = 1
    ):
        """
        Calculate the annual change in soil carbon using Approach 1 
            (even though we have a transition matrix). 

        Function Arguments
        ------------------
        - vec_soc: vector of soil carbon

        Keyword Arguments
        -----------------
        - approach: either 1 or 2
            * 1: use IPCC approach 1
            * 2: use change in soil carbon year over year implied by vec_soc

        """
        if approach not in [1, 2]:
            self._log(
                f"Warning in 'calculate_ipcc_soc_deltas': approach '{approach}' not found--please enter 1 or 2. Default will be set to 1.",
                type_log = "warning"
            )
            approach = 1

        if approach == 1:
            vec_soc_delta = np.concatenate([np.ones(self.time_dependence_stock_change)*vec_soc[0], vec_soc])
            vec_soc_delta = (vec_soc_delta[self.time_dependence_stock_change:] - vec_soc_delta[0:(len(vec_soc_delta) - self.time_dependence_stock_change)])/self.time_dependence_stock_change
        elif approach == 2:
            vec_soc_delta = vec_soc[1:] - vec_soc[0:-1]
            vec_soc_delta = np.insert(vec_soc_delta, 0, vec_soc_delta[0])

        return vec_soc_delta



    def calculate_soc_stock_change_with_time_dependence(self,
        arrs_lndu_land_conv: np.ndarray,
        arrs_lndu_soc_conversion_factors: np.ndarray,
        time_dependence_stock_change: int,
        shape_param: Union[float, int] = None,
    ) -> np.ndarray:
        """
        Calculate the SOC stock change with time dependence (includes some 
            qualitative non-linearities)

        Function Arguments
        ------------------
        - arrs_lndu_land_conv: arrays with land use conversion totals
        - arrs_lndu_soc_conversion_factors: arrays with SOC conversion factors 
            between types
        - time_dependence_stock_change: time-dependent stock change factor to 
            use

        Keyword Arguments
        -----------------
        - shape_param: parameter that expands the sigmoid. If None, defaults to 
            time_dependence_stock_change/10.

        Notes
        -----

        See Volume 4, Chapter 2 of the IPCC 2006 Guidance for National 
            Greenhouse Gas Inventories, page 2.38 for the following description 
            of changes to soil carbon:

            "Changes in C stocks normally occur in a non-linear fashion, and it
            is possible to further develop the time dependence of stock change
            factors to reflect this pattern. For changes in land use or
            management that cause a decrease in soil C content, the rate of
            change is highest during the first few years, and progressively
            declines with time. In contrast, when soil C is increasing due to
            land-use or management change, the rate of accumulation tends to
            follow a sigmoidal curve, with rates of change being slow at the
            beginning, then increasing and finally decreasing with time. If
            historical changes in land-use or management practices are
            explicitly tracked by re- surveying the same locations (i.e.,
            Approach 2 or 3 activity data, see Chapter 3), it may be possible
            to implement a Tier 2 method that incorporates the non-linearity
            of changes in soil C stock."

        We use this guidance to assign different shapes to SOC for releases and
            sequestration. 

        """

        # get length
        n = min(len(arrs_lndu_land_conv), len(arrs_lndu_soc_conversion_factors))
        D = int(np.round(time_dependence_stock_change))
        shape_param = D/10 if (shape_param is None) else shape_param

        # get functions that characterize SOC stock change from conversion - sequestration (sigmoid) and emissions (complementary sigmoid with double phase)
        def emission_curve(x: float) -> float:
            d = D/2
            x_pr = x/(2*shape_param) + d*(shape_param - 1)/shape_param
            sig = 1/(1 + np.e**(d - x_pr))

            return 2*(1 - sig) - 1


        def sequestration_curve(x: float) -> float:
            d = D/2
            #f_fan(x/shape_param + d*(shape_param - 1)/shape_param, D, 1, 0, np.e, d)
            x_pr = x/shape_param + d*(shape_param - 1)/shape_param
            sig = 1/(1 + np.e**(d - x_pr))

            return sig

        # get emission/sequestration proportions
        vec_proportional_emission = np.array([emission_curve(x) for x in range(D)])
        vec_proportional_emission /= np.sum(vec_proportional_emission)
        vec_proportional_sequestration = np.array([sequestration_curve(x) for x in range(D)])
        vec_proportional_sequestration /= np.sum(vec_proportional_sequestration)
        
        #vec_proportional_emission = np.ones(D)/D#np.array(range(D, 0, -1))/(D*(D + 1)/2)
        #vec_proportional_sequestration = np.ones(D)/D#np.array(range(1, D + 1))/(D*(D + 1)/2)
        # get arrays in terms of sequestration and emission
        arrs = arrs_lndu_land_conv*arrs_lndu_soc_conversion_factors
        arrs_sequestration = sf.vec_bounds(arrs, (0, np.inf))
        arrs_emission = sf.vec_bounds(arrs, (-np.inf, 0))

        v_out = np.zeros(n)
        
        for i in range(n):
            v_cur_emission = vec_proportional_emission*(arrs_emission[i].sum())
            v_cur_sequestration = vec_proportional_sequestration*(arrs_sequestration[i].sum())
            v_cur = v_cur_emission + v_cur_sequestration
            v_out[i:min(n, i + D)] += v_cur[0:min(D, n - i)]

        return v_out



    def check_markov_shapes(self, 
        arrs: np.ndarray, 
        function_var_name:str
    ) -> None:
        """
        Check the shape of transition/emission factor matrices sent to 
            project_land_use
        """
        # get land use info
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

        if len(arrs.shape) < 3:
            raise ValueError(f"Invalid shape for array {function_var_name}; the array must be a list of square matrices.")
        elif arrs.shape[1:3] != (attr_lndu.n_key_values, attr_lndu.n_key_values):
            raise ValueError(f"Invalid shape of matrices in {function_var_name}. They must have shape ({attr_lndu.n_key_values}, {attr_lndu.n_key_values}).")

        return None



    def get_lndu_scalar_max_out_states(self,
        scalar_in: Union[int, float],
        attribute_land_use: AttributeTable = None,
        cats_max_out: Union[List[str], None] = None,
        lmo_approach: Union[str, None] = None,
    ):
        """
        Retrieve the vector of scalars to apply to land use based on an input 
            preliminary scalar and configuration parameters for "maxing out" 
            transition probabilities.

        Function Arguments
        ------------------
        - scalar_in: input scalar

        Keyword Arguments
        -----------------
        - attribute_land_use: AttributeTable for $CAT-LANDUSE$. If None, use 
            self.model_attributes default.
        - cats_max_out: categories that are available to be maxed out. If None, 
            defaults to AFOLU.cats_lndu_max_out_transition_probs
        - lmo_approach: approach to take for land use max out approach. If None,
            defaults to configuration. Values are:

            * "decrease_only": apply "max_out" approach only if scalar < 1
            * "increase_only": apply "max_out" approach only if scalar > 1
            * "decrease_and_increase": apply "max_out" to categories in either 
                case
        """
        attribute_land_use = (
            self.model_attributes.get_attribute_table(self.subsec_name_lndu) 
            if (attribute_land_use is None) 
            else attribute_land_use
        )
        cats_max_out = (
            self.cats_lndu_max_out_transition_probs
            if not isinstance(cats_max_out, list)
            else [x for x in attribute_land_use.key_values if x in cats_max_out]
        )

        # check specification of land use max out approach
        valid_lmo = [
            "decrease_only",
            "decrease_and_increase",
            "increase_only"
        ]
        lmo_approach = (
            self.model_attributes.configuration.get("land_use_reallocation_max_out_directionality")
            if lmo_approach not in valid_lmo
            else lmo_approach
        )
        lmo_approach = (
            "decrease_only" 
            if (lmo_approach is None) 
            else lmo_approach
        )

        # proceed only if any of the following 3 conditions are true
        proceed_q = False
        proceed_q |= ((lmo_approach == "decrease_only") & (scalar_in < 1))
        proceed_q |= ((lmo_approach == "increase_only") & (scalar_in > 1))
        proceed_q |= (lmo_approach == "decrease_and_increase")

        out = (
            np.array([int(x in cats_max_out) for x in attribute_land_use.key_values]).astype(int)
            if proceed_q
            else np.zeros(attribute_land_use.n_key_values).astype(int)
        )

        return out
    



    def get_lndu_soil_soc_factors(self,
        df_afolu_trajectories: pd.DataFrame,
        arr_lndu_prevalence: np.ndarray,
        arr_agrc_crop_fractions: np.ndarray,
        dict_fracs_residues_removed_burned: Dict,
        vec_lndu_frac_pasture: np.ndarray,
        default_fmg_grassland_not_pasture: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate soild soc factors and areas of land-use types under 
            improved management (in terms of configuration units).
        
        NOTE: Treats F_I and F_MG as independent; right now, F_I is scaled
            by the fraction of residues that are not removed or burned, and
            it is assumed that inputs can be used without no till.

        Returns a tuple of np.ndarrays

            arr_lndu_factor_soil_management, arr_lndu_area_improved

        Function Arguments
        ------------------
        - df_afolu_trajectories: data frame containing input AFOLU trajectories
        - arr_lndu_prevalence: array giving land use prevalence in terms of 
        - arr_agrc_crop_fractions: array giving crop areas as fraction of 
            cropland
        - dict_fracs_residues_removed_burned: dictionary with keys 
            self.modvar_agrc_frac_residues_burned and 
            self.modvar_agrc_frac_residues_removed
        - vec_lndu_frac_pasture: vector giving the fraction of grassland that is 
            pasture (over time)

        Keyword Arguments
        -----------------
        - default_fmg_grassland_not_pasture: F_MG for grasslands not pasture. 
            Default is 1 (can be read in later)
        """
        # lower bound for F_MG
        arr_lndu_factor_soil_management_inf = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_management_infinum,
            return_type = "array_base",
            expand_to_all_cats = True,
            all_cats_missing_val = 1.0
        )

        # upper bound for F_MG
        arr_lndu_factor_soil_management_sup = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_management_supremum,
            return_type = "array_base",
            expand_to_all_cats = True,
            all_cats_missing_val = 1.0
        )

        # upper bound for F_I, or input component w/o manure
        arr_lndu_factor_soil_inputs_no_manure_sup = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_inputs_supremum_no_manure,
            return_type = "array_base",
            expand_to_all_cats = True,
            all_cats_missing_val = 1.0
        )

        # fraction of croplands using no-till
        arr_agrc_frac_improved_by_crop = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_frac_no_till,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # fraction of pastures that are improved
        vec_lndu_frac_pastures_improved = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_frac_pastures_improved,
            expand_to_all_cats = False,
            return_type = "array_base",
        )


        ##  1. START WITH CROPLAND FACTORS

        arr_agrc_crop_area = sf.do_array_mult(
            arr_agrc_crop_fractions,
            arr_lndu_prevalence[:, self.ind_lndu_crop]
        )

        # start by calculating average management factor, based on the fraction using no-till (sup) and unimproved (inf) 
        arr_agrc_crop_area_improved = arr_agrc_crop_area*arr_agrc_frac_improved_by_crop
        vec_lndu_frac_cropland_improved = (arr_agrc_crop_area_improved).sum(axis = 1)
        vec_lndu_frac_cropland_improved /= arr_lndu_prevalence[:, self.ind_lndu_crop] # should be a denominator of 1, but just to be safe

        # get the cropland management factor
        vec_lndu_fmg_cropland = vec_lndu_frac_cropland_improved*arr_lndu_factor_soil_management_sup[:, self.ind_lndu_crop]
        vec_lndu_fmg_cropland += (1 - vec_lndu_frac_cropland_improved)*arr_lndu_factor_soil_management_inf[:, self.ind_lndu_crop]

        # get fraction removed/burned
        vec_agrc_frac_residue_burned = dict_fracs_residues_removed_burned.get(self.modvar_agrc_frac_residues_burned).flatten()
        vec_agrc_frac_residue_removed = dict_fracs_residues_removed_burned.get(self.modvar_agrc_frac_residues_removed).flatten()
        vec_agrc_frac_residue_inputs = 1 - vec_agrc_frac_residue_burned - vec_agrc_frac_residue_removed
        
        # calculate the input componen
        vec_agrc_inputs_component = sf.vec_bounds(arr_lndu_factor_soil_inputs_no_manure_sup[:, self.ind_lndu_crop] - 1, (0, 1))
        vec_agrc_inputs_component *= vec_agrc_frac_residue_inputs
        vec_agrc_inputs_component += 1.0

        # assume that inputs can be provided independently of 
        vec_lndu_fmg_cropland *= vec_agrc_inputs_component


        ##  2. GET GRASSLAND FACTOR

        # 
        vec_lndu_fmg_grassland = (1 - vec_lndu_frac_pastures_improved)*arr_lndu_factor_soil_management_inf[:, self.ind_lndu_grass]
        vec_lndu_fmg_grassland += vec_lndu_frac_pastures_improved*arr_lndu_factor_soil_management_sup[:, self.ind_lndu_grass]
        vec_lndu_fmg_grassland *= vec_lndu_frac_pasture
        vec_lndu_fmg_grassland += default_fmg_grassland_not_pasture*(1 - vec_lndu_frac_pasture) # "1" is default 
        
        vec_lndu_area_grassland_improved = arr_lndu_prevalence[:, self.ind_lndu_grass]
        vec_lndu_area_grassland_improved *= vec_lndu_frac_pasture*vec_lndu_frac_pastures_improved

        # update output array
        arr_lndu_factor_soil_management = np.ones(arr_lndu_factor_soil_management_sup.shape)
        arr_lndu_factor_soil_management[:, self.ind_lndu_crop] = vec_lndu_fmg_cropland
        arr_lndu_factor_soil_management[:, self.ind_lndu_grass] = vec_lndu_fmg_grassland

        # build output areas
        arr_lndu_area_improved = np.ones(arr_lndu_factor_soil_management_sup.shape)
        arr_lndu_area_improved[:, self.ind_lndu_crop] = arr_agrc_crop_area_improved.sum(axis = 1)
        arr_lndu_area_improved[:, self.ind_lndu_grass] = vec_lndu_area_grassland_improved


        return arr_lndu_factor_soil_management, arr_lndu_area_improved



    def get_markov_matrices(self,
        df_ordered_trajectories: pd.DataFrame,
        correct_emission_units: bool = True,
        get_emission_factors: bool = True,
        n_tp: Union[int, None] = None,
        target_units_area_modvar: Union[str, None] = None,
        thresh_correct: float = 0.0001,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the transition and emission factors matrices from the data frame 
            df_ordered_trajectories. Assumes that the input data frame is 
            ordered by time_period

        Function Arguments
        ------------------
        - df_ordered_trajectories: input data frame containing columnar 
            variables for conversion transition probabilities and emission 
            factors.

        Keyword Arguments
        -----------------
        - correct_emission_units: correct emissions to configuration units?
        - get_emission_factors: return emission factors as well?
            * If False, returns ONLY the array of transition probabilities
        - target_units_area_modvar: modvar that will drive emissions, used to 
            identify target area units to which conversion emission factors are 
            applied. 
            * If None, no adjustment is made
        - n_tp: the number of time periods. Default value is None, which implies 
            all time periods
        - thresh_correct is used to decide whether or not to correct the 
            transition matrix (assumed to be row stochastic) to sum to 1; if the 
            absolute value of the sum is outside this range, an error will be 
            thrown

        Notes
        -----
        - fields_pij and fields_efc will be properly ordered by categories for 
            this transformation
        """
        
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        n_cats = attr_lndu.n_key_values
        n_tp = n_tp if sf.isnumber(n_tp, integer = True) else self.n_time_periods

        fields_pij = self.model_attributes.dict_model_variables_to_variables.get(self.modvar_lndu_prob_transition)
        fields_efc = self.model_attributes.dict_model_variables_to_variables.get(self.modvar_lndu_ef_co2_conv)
        if (fields_pij is None) | ((fields_efc is None) & get_emission_factors):
            return None

        fields_check = (
            fields_pij + fields_efc
            if get_emission_factors
            else fields_pij
        )
        sf.check_fields(df_ordered_trajectories, fields_check)

        # fetch arrays of transition probabilities and co2 emission factors
        arr_pr = np.array(df_ordered_trajectories[fields_pij])
        arr_pr = arr_pr.reshape((n_tp, n_cats, n_cats))

        # RETURN if not retrieving emission factors
        if not get_emission_factors:
            return arr_pr


        arr_ef = np.array(df_ordered_trajectories[fields_efc])
        arr_ef = arr_ef.reshape((n_tp, n_cats, n_cats))

        # convert units for conversion emission factors
        arr_ef *= (
            self.model_attributes.get_scalar(
                self.modvar_lndu_ef_co2_conv,
                "mass"
            )
            if correct_emission_units
            else 1.0
        )

        # ensure area is accounted for
        arr_ef *= (
            self.model_attributes.get_variable_unit_conversion_factor(
                target_units_area_modvar,
                self.modvar_lndu_ef_co2_conv,
                "area"
            )
            if self.model_attributes.check_modvar(target_units_area_modvar) is not None
            else 1.0
        )

        return arr_pr, arr_ef



    def get_matrix_column_scalar(self,
        mat_column: np.ndarray,
        target_scalar: np.ndarray,
        vec_x: np.ndarray,
        eps: float = 0.000001,
        mask_max_out_states: np.ndarray = None,
        max_iter: int = 100,
    ) -> float:

        """
        Return a vector of scalars that need to be applied to acheieve a 
            specified scaled change in state, i.e. finds the true matrix column 
            scalar required to achieve x(1)_i -> x(0)_i*target_scalar. If any 
            transition probabilities are 1 w/scalar > 1 (or 0 with scalar < 0), 
            the scalar becomes inadequate; iterative function will search for 
            adequate column scalar.

        Function Arguments
        ------------------
        - mat_column: column in transition matrix (assuming row-stochastic) 
            representing probabilities of entry into a state
        - target_scalar: the target change in output area to achieve
        - vec_x: current state of areas. Target area is sum(vec_x)*target_scalar

        Keyword Arguments
        -----------------
        - eps: area convergence tolerance.
        - mask_max_out_states: np.array of same length as mat_column that 
            contains a mask for states that should be "maxed out" (sent to 0 
            or 1) first. The default is for all states to be scalable.
            * For example, to ensure that masses of area are first shifted 
                to/from state 1 out of states 0, 1, 2, 3, 4, 5, the vector 
                should be

                mask_max_out_states = np.array([0, 1, 0, 0, 0])

            * Default is None. If None, all states are scalable states.

        - max_iter: maximum number of iterations. Default is 100.


        Notes and Expanded Description
        ------------------------------

        For x(0), x(1) \in \mathbb{R}^(1 x n) and a row-stochastic transition 
            matrix Q in \mathbb{R}^{n x n}, the value

            x(1) = x(0)Q

        transforms the state vector x(0) -> x(1). In some cases, it is desirable 
            to ensure that x(1)_i = \alpha*x(0)_i, i.e., that a transition 
            matrix enforces an observed growth from period 0 -> 1. This can 
            generally be acheived in AFOLU using the 
            AFOLU.adjust_transition_matrix() method.

        However, some transition matrices may have 0s or 1s in column entries, 
            meaning that specified input scalar may be unable to acheieve a 
            desired scalar change in input area, so other columnar entries may 
            have to be scaled more to acheive the desired outcome of 
            x(1)_i = \alpha*x(0)_i.

        """
        # check input
        if not sf.islistlike(mask_max_out_states):
            # LOG HERE - INVALID STATE
            mask_max_out_states = None
        elif len(mask_max_out_states) != len(mask_max_out_states):
            mask_max_out_states = None
        else:
            mask_max_out_states = sf.vec_bounds(np.round(mask_max_out_states).astype(int), (0, 1))

        if mask_max_out_states is None:
            mask_max_out_states = np.zeros(len(mat_column))


        ##  START BY GETTING AREA THAT CAN BE MAXED OUT FROM INPUT CLASSES

        # true area, target area, and area incoming from
        area = np.dot(vec_x, mat_column)
        area_target = target_scalar*area
        area_incoming_from_max_out_states_at_full = np.sum(mask_max_out_states*vec_x)
        area_incoming_from_max_out_states_at_current = np.sum(mask_max_out_states*vec_x*mat_column)
        
        delta_area  = area_target - area    
        scalar_adj = (1 - mask_max_out_states)
    
        if np.sum(mask_max_out_states) > 0:
            # if the max-out states can't absorb the required change, then max them out and allow the rest to continue
            max_scale = (
                area_incoming_from_max_out_states_at_full/area_incoming_from_max_out_states_at_current
                if area_incoming_from_max_out_states_at_current != 0
                else 1.0
            )
            
            scale_max_out_states = np.nan_to_num(delta_area/area_incoming_from_max_out_states_at_current, 0.0, posinf = 0.0)
            scale_max_out_states = min(max(1 + scale_max_out_states, 0), max_scale)
            scalar_adj += mask_max_out_states*scale_max_out_states


        q_j = sf.vec_bounds(mat_column*scalar_adj, (0, 1))
        area_target = target_scalar*np.dot(vec_x, mat_column)

        # index of probabilities that are scalable
        w_scalable = np.where((q_j < 1) & (q_j > 0))[0]
        w_unscalable = np.where((q_j == 1) | (q_j == 0))[0]
        i = 0
        n = len(vec_x)
        n_iter = max_iter#*2 if (sum(mask_max_out_states) < len(mat_column)) else max_iter

        # prevent iterating if every entry is a 1/0, double number of iterations to allow to turn off mask halfway through (after max_iter) if still non-convergent
        while (np.abs(area/area_target - 1) > eps) and (len(w_unscalable) < n) and (i < n_iter):
            # scalar is capped at one, so we are effectively applying this to only scalable indices
            area = np.dot(vec_x, q_j)
            area_unscalable = np.dot(vec_x[w_unscalable], q_j[w_unscalable])
            scalar_cur = np.nan_to_num((area_target - area_unscalable)/(area - area_unscalable), 1.0, posinf = 1.0)
            if (scalar_cur < 0):
                break
            scalar_adj *= scalar_cur

            # recalculate vector + implied area
            q_j = sf.vec_bounds(mat_column*scalar_adj, (0, 1))

            w_scalable = np.where((q_j < 1) & (q_j > 0))[0]
            w_unscalable = np.where((q_j == 1) | (q_j == 0))[0]
            i += 1

        return scalar_adj



    def get_mineral_soc_change_matrices(self,
        arr_agrc_crop_area: np.ndarray,
        arr_lndu_area: np.ndarray,
        arr_lndu_factor_soil_carbon: np.ndarray,
        arr_lndu_factor_soil_management: np.ndarray,
        arr_lndu_frac_mineral_soils: np.ndarray,
        arr_soil_ef1_organic: np.ndarray,
        arr_soil_soc_stock: np.ndarray,
        vec_lndu_frac_grassland_pasture: np.ndarray,
        vec_soil_area_crop_pasture: np.ndarray,
        dict_soil_fracs_to_use_agrc: dict,
        dict_soil_fracs_to_use_frst: dict,
        dict_soil_fracs_to_use_lndu: dict
    ) -> tuple:
        """
        Retrieve matrices of delta SOC (- means loss of SOC, + indicates 
            sequestration) for conversion from class i to class j in each time 
            period, as well as a land-area weighted average EF1 for each time 
            period. Note that all areas should be in the same units.

        Returns: tuple of form (arrs_delta_soc, vec_avg_ef1)

        Function Arguments
        ------------------
        - arr_agrc_crop_area: array giving crop areas
        - arr_lndu_area: array of land use areas (ordered by 
            attr_lndu.key_values)
        - arr_lndu_factor_soil_carbon: array of F_LU SOC factors by land use 
            type
        - arr_lndu_factor_soil_management: array of F_MG SOC adjustment factors 
            by land use type
        - arr_lndu_frac_mineral_soils: array giving the fraction of soils that 
            are mineral by each land use category
        - arr_soil_ef1_organic: N2O Organic EF1 by soil type
        - arr_soil_soc_stock: array giving soil carbon content of soil types 
            (30cm) for each time period (n_tp x attr_soil.n_key_values)
        - vec_lndu_frac_grassland_pasture: fraction of grassland that is pasture
        - vec_soil_area_crop_pasture: area of crop and pasture by time period
        - dict_soil_fracs_to_use_agrc: dictionary mapping soil fraction model 
            variable string to array (wide by crop categories)
        - dict_soil_fracs_to_use_frst: dictionary mapping soil fraction model 
            variable string to fraction array (wide by forest categories)
        - dict_soil_fracs_to_use_lndu: dictionary mapping soil fraction model 
            variable string to fraction array (wide by land use categories)
        """

        # get attribute tables
        attr_agrc = self.model_attributes.get_attribute_table(self.subsec_name_agrc)
        attr_frst = self.model_attributes.get_attribute_table(self.subsec_name_frst)
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        attr_soil = self.model_attributes.get_attribute_table(self.subsec_name_soil)
        # initialize SOC transition arrays
        n_tp = len(arr_agrc_crop_area)
        arrs_delta_soc_source = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_delta_soc_target = arrs_delta_soc_source.copy()
        # initialize dictionary for building matrix lists out and vector of average ef1 soild
        dict_lndu_avg_soc_vecs = {}
        vec_soil_ef1_soc_est = 0.0


        ##  GET AVERAGE SOC CONTENT IN CROPLANDS

        vec_agrc_area = np.sum(arr_agrc_crop_area, axis = 1)
        vec_agrc_avg_soc = 0.0
        for modvar in dict_soil_fracs_to_use_agrc.keys():
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            #
            vec_agrc_avg_soc_cur = np.sum(arr_agrc_crop_area*dict_soil_fracs_to_use_agrc[modvar], axis = 1)
            vec_soil_ef1_soc_est += vec_agrc_avg_soc_cur*arr_soil_ef1_organic[:, ind_soil]/vec_soil_area_crop_pasture
            vec_agrc_avg_soc_cur = np.nan_to_num(vec_agrc_avg_soc_cur/vec_agrc_area, 0.0, posinf = 0.0)
            vec_agrc_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, self.ind_lndu_crop]*arr_lndu_factor_soil_management[:, self.ind_lndu_crop]
            vec_agrc_avg_soc += vec_agrc_avg_soc_cur*arr_lndu_frac_mineral_soils[:, self.ind_lndu_crop]

        dict_lndu_avg_soc_vecs.update({self.ind_lndu_crop: vec_agrc_avg_soc})


        ##  GET AVERAGE SOC CONTENT IN FORESTS

        arr_frst_avg_soc = 0.0
        dict_lndu_forest_to_frst_forest = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_lndu,
            attr_frst.key,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True
        )
        inds_frst = np.array([(attr_lndu.get_key_value_index(x), attr_frst.get_key_value_index(dict_lndu_forest_to_frst_forest[x])) for x in dict_lndu_forest_to_frst_forest.keys()])
        inds_lndu = inds_frst[:, 0]
        inds_frst = inds_frst[:, 1]

        for modvar in dict_soil_fracs_to_use_frst.keys():
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            #

            arr_frst_avg_soc_cur = arr_lndu_area[:, inds_lndu]*dict_soil_fracs_to_use_frst[modvar][:, inds_frst]
            arr_frst_avg_soc_cur = np.nan_to_num(arr_frst_avg_soc_cur/arr_lndu_area[:, inds_lndu], 0.0, posinf = 0.0).transpose()
            arr_frst_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, inds_lndu].transpose()
            arr_frst_avg_soc += arr_frst_avg_soc_cur.transpose()*arr_lndu_frac_mineral_soils[:, inds_lndu]


        for i in enumerate(inds_lndu):
            i, ind = i
            dict_lndu_avg_soc_vecs.update({ind: arr_frst_avg_soc[:, i]})


        ##  GET AVERAGE SOC CONTENT IN ADDITIONAL LAND USE CATEGORIES

        arr_lndu_avg_soc = 0.0
        cats_lndu = list(set(sum([self.model_attributes.get_variable_categories(x) for x in dict_soil_fracs_to_use_lndu.keys()], [])))
        inds_lndu = sorted([attr_lndu.get_key_value_index(x) for x in cats_lndu])
        ind_grass = inds_lndu.index(attr_lndu.get_key_value_index(self.cat_lndu_grass))

        for modvar in dict_soil_fracs_to_use_lndu.keys():
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            # get soil management factor for grasslands
            arr_lndu_soil_management_factor_cur = (1 - vec_lndu_frac_grassland_pasture) + vec_lndu_frac_grassland_pasture*arr_lndu_factor_soil_management[:, inds_lndu].transpose()
            #
            arr_lndu_avg_soc_cur = arr_lndu_area[:, inds_lndu]*dict_soil_fracs_to_use_lndu[modvar][:, inds_lndu]
            vec_soil_ef1_soc_est += arr_lndu_avg_soc_cur[:, ind_grass]*arr_soil_ef1_organic[:, ind_soil]*vec_lndu_frac_grassland_pasture/vec_soil_area_crop_pasture
            arr_lndu_avg_soc_cur = np.nan_to_num(arr_lndu_avg_soc_cur/arr_lndu_area[:, inds_lndu], 0.0, posinf = 0.0).transpose()
            arr_lndu_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, inds_lndu].transpose()*arr_lndu_soil_management_factor_cur
            arr_lndu_avg_soc += arr_lndu_avg_soc_cur.transpose()*arr_lndu_frac_mineral_soils[:, inds_lndu]

        for i in enumerate(inds_lndu):
            i, ind = i
            dict_lndu_avg_soc_vecs.update({ind: arr_lndu_avg_soc[:, i]})


        ##  UPDATE SOURCE/TARGET ARRAYS USING AVERAGE SOC

        for k in dict_lndu_avg_soc_vecs.keys():
            vec = dict_lndu_avg_soc_vecs[k]
            w = np.where(vec != 0)[0]
            vec = np.interp(range(len(vec)), w, vec[w]) if (len(w) > 0) else vec
            for i in range(n_tp):
                arrs_delta_soc_source[i, k, :] = vec[i]
                arrs_delta_soc_target[i, :, k] = vec[i]

        arrs_delta = arrs_delta_soc_target - arrs_delta_soc_source

        return arrs_delta, vec_soil_ef1_soc_est
    


    def get_transition_matrix_from_long_df(self,
        df_transition: pd.DataFrame,
        field_i: str,
        field_j: str,
        field_pij: str,
    ) -> np.ndarray:
        """
        Convert a long data frame to a transition matrix
        
        Function Arguments
        ------------------
        - df_transition: data frame storing transition probabilities in long form
        - field_i: field storing source class
        - field_j: field sotring target class
        - field_pij: field storing probabilities
        """
        
        # get attribute table
        attr_lndu = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_lndu
        )
        
        return_df = not isinstance(df_transition, pd.DataFrame)
        return_df |= (
            not set([field_i, field_j, field_pij]).issubset(set(df_transition.columns))
            if not return_df
            else return_df
        )
        
        if return_df:
            return df_transition
        
        
        # filter out invalid rows
        df_transition_out = (
            df_transition[
                df_transition[field_i].isin(attr_lndu.key_values)
                & df_transition[field_j].isin(attr_lndu.key_values)
            ]
            .copy()
            .reset_index(drop = True)
        )
        
        # initialize output matrix and fill in
        n = attr_lndu.n_key_values
        mat_out = np.zeros(n**2)
        
        # indices
        vec_i = np.array(
            df_transition_out[field_i]
            .apply(attr_lndu.get_key_value_index)
        )
        
        vec_j = np.array(
            df_transition_out[field_j]
            .apply(attr_lndu.get_key_value_index)
        )
        
        vec_inds = vec_i*n + vec_j
        vec_probs = df_transition_out[field_pij].to_numpy()
        
        # overwrite values in the matrix and reshape
        np.put(mat_out, vec_inds, vec_probs)
        mat_out = mat_out.reshape((n, n))
        
        return mat_out



    def format_transition_matrix_as_input_dataframe(self,
        mat: np.ndarray,
        exclude_time_period: bool = False,
        field_key: str = "key",
        key_vals: Union[List[str], None] = None,
        modvar: Union[str, None] = None,
        vec_time_periods: Union[List, np.ndarray, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Convert an input transition matrix mat to a wide dataframe using 
            AFOLU.modvar_lndu_prob_transition as variable template

        NOTE: Returns None on error.


        Function Arguments
        ------------------
        - mat: row-stochastic transition matrix to format as wide data frame OR
            array of row-stochastic transition matrices to format. If specifying
            as array matrices, format_transition_matrix_as_input_dataframe()
            will interpret as being associated with sequential time periods. 
                * NOTE: Time periods can be specified using vec_time_periods. 
                    Time periods specified in this way must have the same length
                    as the array of matrices.

        Keyword Arguments
        -----------------
        - exclude_time_period: drop time period from data frame?
        - field_key: temporary field to use for merging
        - key_vals: ordered land use categories representing states in mat. If
            None, default to attr_lndu.key_values. Should be of length n for
             mat an nxn transition matrix.
        - modvar: optional specification of I/J model variable to use. If None,
            defaults to transition probability 
            (self.modvar_lndu_prob_transition)
        - vec_time_periods: optional vector of time periods to specify for 
            matrices. Does not affect implementation with single matrix. If
            None and mat is an array of arrays, takes value of: 
            
            * ModelAttributes.time_period key_values (if mat is of same length)
            * range(len(mat)) otherwise
        """
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        field_time_period = self.model_attributes.dim_time_period
        attr_time_period = self.model_attributes.dict_attributes.get(f"dim_{field_time_period}")
        key_vals = (
            attr_lndu.key_values 
            if (key_vals is None)
            else [x for x in key_vals if x in attr_lndu.key_values]
        )
        
        modvar = (
            modvar
            if self.model_attributes.check_modvar(modvar) is not None
            else self.modvar_lndu_prob_transition
        )
        """
        # PRESERVING IN CASE ISSUES ARISE WITH REVISION

        # setup as matrix
        df_mat = pd.DataFrame(mat, columns = key_vals)
        df_mat[field_key] = key_vals
        df_mat_melt = pd.melt(df_mat, id_vars = [field_key], value_vars = key_vals).sort_values(by = [field_key, "variable"])

        # get field names
        fields_names = self.model_attributes.build_varlist(
            self.subsec_name_lndu,
            self.modvar_lndu_prob_transition,
            restrict_to_category_values = key_vals
        )

        df_mat_melt["field"] = fields_names
        df_mat_melt.set_index("field", inplace = True)
        df_mat_melt = df_mat_melt[["value"]].transpose().reset_index(drop = True)
        
        # doing this only to drop index
        df_out = pd.DataFrame()
        df_out[df_mat_melt.columns] = df_mat_melt[df_mat_melt.columns]
        """;

        ##  INITIALIZATION AND CHECKS

        # shared variables in melt/pivot
        field_field = "field"
        field_value = "value"
        field_variable = "variable"

        # check if is a single matrix; if so, treat as list and reduce later
        is_single_matrix = (len(mat.shape) == 2)
        mat = np.array([mat]) if is_single_matrix else mat
        mat_shape = mat.shape
        vec_time_periods = np.zeros(1) if is_single_matrix else vec_time_periods

        # check specification of time periods
        if vec_time_periods is None:
            # default to model attributes time period key values if same length
            vec_time_periods = (
                np.arange(mat_shape[0]).astype(int)
                if mat_shape[0] != len(attr_time_period.key_values)
                else attr_time_period.key_values
            )

        elif not sf.islistlike(vec_time_periods):
            tp = str(type(vec_time_periods))
            self._log(
                f"Error in format_transition_matrix_as_input_dataframe: invalid type specifiation '{tp}' of vec_time_periods.",
                type_log = "error"
            )
            return None

        elif len(vec_time_periods) != len(mat):
            self._log(
                f"Error in format_transition_matrix_as_input_dataframe: invalid specifiation of vec_time_periods. The length of vec_time_periods ({len(vec_time_periods)}) and mat ({mat.shape[0]}) must be the same.",
                type_log = "error"
            )
            return None


        ##  BUILD DATA FRAME

        # setup new fields - key values and time periods
        field_key_vals = [x[1] for x in itertools.product(np.ones(mat_shape[0]), key_vals)]
        field_time_periods = (
            np.concatenate(
                np.outer(
                    vec_time_periods, 
                    np.ones(mat.shape[1])
                )
            )
            .astype(int)
        )

        # generate data frame before melting
        df_mat = pd.DataFrame(
            np.concatenate(mat), 
            columns = key_vals
        )
        df_mat[field_key] = field_key_vals
        df_mat[field_time_period] = field_time_periods


        ##  MELT, APPLY VARIABLES, THEN PIVOT

        # note: sorting is important here to ensure that variable names generated below are in correct order
        df_out = (
            pd.melt(
                df_mat, 
                id_vars = [field_time_period, field_key], 
                value_vars = key_vals,
                var_name = field_variable,
                value_name = field_value,
            )
            .sort_values(by = [field_time_period, field_key, field_variable])
            .reset_index(drop = True)
        )

        # get vars and expand 
        field_variables = self.model_attributes.build_varlist(
            self.subsec_name_lndu,
            modvar,
            restrict_to_category_values = key_vals
        )
        field_variables = [x[1] for x in itertools.product(np.ones(mat_shape[0]), field_variables)]
        df_out[field_field] = field_variables

        # pivot and drop time period if returning as single matrix
        df_out = sf.pivot_df_clean(
            df_out[[field_time_period, field_value, field_field]],
            [field_field],
            [field_value]
        )
        (
            df_out.drop([field_time_period], axis = 1, inplace = True)
            if is_single_matrix | exclude_time_period
            else None
        )

        return df_out



    def project_agrc_lvst_integrated_demands(self,
        df_afolu_trajectories: pd.DataFrame,
        area_agrc_cropland_init: float,
        vec_pop: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray
    ) -> tuple:
        """
        Calculate agricultural and livestock demands, imports, exports, and 
            production. Agriculture and Livestock demands are related through 
            feed demands, requiring integration.

        Function Arguments
        ------------------
        - df_afolu_trajectories: data frame containing input variables
        - area_agrc_cropland_init: initial area of cropland
        - vec_pop: vector of population
        - vec_rates_gdp_per_capita: vector of gdp/capita growth rates

        Notes
        -----

        Let:
            D = Domestic production to satisfy domestic demand
            E = Exports
            F = Fraction of domestic demand met by imports
            I = Imports
            M = Domestic demand
            P = Domestic production

        Then:
            P = D + E, MF = I, M - I = D => M - MF = D =>

            P = M - MF + E : Domestic Production
            M = (P - E)/(1 - F) : Domestic Demand

        """

        ###########################
        #    LIVESTOCK DEMANDS    #
        ###########################

        # get variables requried to estimate demand - start with exports
        arr_lvst_exports_unadj = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lvst_equivalent_exports,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )
        arr_lvst_frac_imported = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lvst_frac_demand_imported,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        fields_lvst_elas = self.model_attributes.switch_variable_category(
            self.subsec_name_lvst,
            self.modvar_lvst_elas_lvst_demand,
            "demand_elasticity_category"
        )
        vec_gnrl_frac_eating_red_meat = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        vec_lvst_production_init = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lvst_pop_init,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )[0]
        arr_lvst_elas_demand = np.array(df_afolu_trajectories[fields_lvst_elas])


        # 1. calculate domestic demand
        #
        # adjust exports to have cap at production; modify series downward
        vec_lvst_exports_modifier = sf.vec_bounds(vec_lvst_production_init - arr_lvst_exports_unadj[0], (-np.inf, 0))
        arr_lvst_exports_unadj = sf.vec_bounds(arr_lvst_exports_unadj + vec_lvst_exports_modifier, (0, np.inf))
        # get the demand
        vec_lvst_domestic_demand_init = sf.vec_bounds(vec_lvst_production_init - arr_lvst_exports_unadj[0], (0, np.inf))
        vec_lvst_domestic_demand_init /= (1 - arr_lvst_frac_imported[0])
        vec_lvst_domestic_demand_init = sf.vec_bounds(np.nan_to_num(vec_lvst_domestic_demand_init, 0.0, posinf = 0.0), (0, np.inf))
        # project aggregate domestic demand forward
        vec_gnrl_frac_eating_red_meat_scalar = sf.vec_bounds(vec_gnrl_frac_eating_red_meat/vec_gnrl_frac_eating_red_meat[0], (0, 1))
        arr_lvst_domestic_demand_unadj = self.project_per_capita_demand(
            vec_lvst_domestic_demand_init,
            vec_pop,
            vec_rates_gdp_per_capita,
            arr_lvst_elas_demand,
            vec_gnrl_frac_eating_red_meat_scalar,
            int
        )

        # get imports, domestic production for domestic demand (unadj), and domestic production
        arr_lvst_imports_unadj = arr_lvst_domestic_demand_unadj*arr_lvst_frac_imported
        arr_lvst_domestic_production_for_domestic_demand_unadj = arr_lvst_domestic_demand_unadj - arr_lvst_imports_unadj
        arr_lvst_domestic_production_unadj = arr_lvst_domestic_production_for_domestic_demand_unadj + arr_lvst_exports_unadj


        #  2. get weights for allocating grazing area and feed requirement to animals - based on first year only
        #
        vec_lvst_base_graze_weights = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lvst_dry_matter_consumption,
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )[0]
        vec_lvst_feed_allocation_weights = (vec_lvst_production_init*vec_lvst_base_graze_weights)/np.dot(vec_lvst_production_init, vec_lvst_base_graze_weights)
        
        # get carrying capacity scalar relative to baseline
        vec_lvst_carry_capacity_scale = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lvst_carrying_capacity_scalar,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )
        vec_lvst_carry_capacity_scale = np.nan_to_num(
            vec_lvst_carry_capacity_scale/vec_lvst_carry_capacity_scale[0],
            nan = 1.0,
            posinf = 1.0
        )


        #############################
        #    AGRICULTURE DEMANDS    #
        #############################

        # variables required for demand
        arr_agrc_elas_crop_demand = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_elas_crop_demand_income,
            return_type = "array_base"
        )
        arr_agrc_exports_unadj = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_equivalent_exports,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )
        arr_agrc_frac_feed = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_frac_animal_feed,
            return_type = "array_base"
        )
        arr_agrc_frac_imported = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_frac_demand_imported,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        # do some unit conversion
        arr_agrc_exports_unadj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_equivalent_exports,
            self.modvar_agrc_yf,
            "mass"
        )


        ##   1. get initial cropland areas and yields
        
        vec_agrc_frac_cropland_area = self.check_cropland_fractions(df_afolu_trajectories, "initial")[0]
        vec_agrc_cropland_area = area_agrc_cropland_init*vec_agrc_frac_cropland_area
        # get yield factors and calculate yield
        arr_agrc_yf = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_yf,
            return_type = "array_base"
        )
        arr_agrc_yf *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        vec_agrc_yield_init = arr_agrc_yf[0]*vec_agrc_cropland_area


        ##  2. get dietary demand scalar for crop demand (increases based on reduction in red meat demand) - depends on how many people eat red meat (vec_gnrl_frac_eating_red_meat)
        
        vec_agrc_diet_exchange_scalar = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_vdes,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )
        vec_agrc_demscale = vec_gnrl_frac_eating_red_meat + vec_agrc_diet_exchange_scalar - vec_gnrl_frac_eating_red_meat*vec_agrc_diet_exchange_scalar
        vec_agrc_demscale = np.nan_to_num(vec_agrc_demscale/vec_agrc_demscale[0], 1.0, posinf = 1.0)
        # get categories that need to be scaled
        vec_agrc_scale_demands_for_veg = np.array(self.model_attributes.get_ordered_category_attribute(self.subsec_name_agrc, "apply_vegetarian_exchange_scalar"))
        arr_agrc_demscale = np.outer(vec_agrc_demscale, vec_agrc_scale_demands_for_veg)
        arr_agrc_demscale = arr_agrc_demscale + np.outer(np.ones(len(vec_agrc_demscale)), 1 - vec_agrc_scale_demands_for_veg)


        ##  3. Calculate crop demands split into yield for livestock feed (responsive to changes in domestic livestock population) and yield for consumption and export (nonlvstfeed)
        
        vec_agrc_domestic_demand_init = sf.vec_bounds(vec_agrc_yield_init - arr_agrc_exports_unadj[0], (0, np.inf))
        vec_agrc_domestic_demand_init /= (1 - arr_agrc_frac_imported[0])
        vec_agrc_domestic_demand_init = sf.vec_bounds(np.nan_to_num(vec_agrc_domestic_demand_init, 0.0, posinf = 0.0), (0, np.inf))
        
        # split out livestock demands and human consumption
        vec_agrc_domestic_demand_init_lvstfeed = vec_agrc_domestic_demand_init*arr_agrc_frac_feed[0]
        vec_agrc_domestic_demand_init_nonlvstfeed = vec_agrc_domestic_demand_init - vec_agrc_domestic_demand_init_lvstfeed
        vec_agrc_production_init_lvstfeed = vec_agrc_domestic_demand_init_lvstfeed*(1 - arr_agrc_frac_imported[0])
        
        # project domestic demand
        arr_agrc_domestic_demand_nonfeed_unadj = self.project_per_capita_demand(
            vec_agrc_domestic_demand_init_nonlvstfeed,
            vec_pop,
            vec_rates_gdp_per_capita,
            arr_agrc_elas_crop_demand,
            arr_agrc_demscale,
            float
        )
        arr_agrc_imports_unadj = arr_agrc_domestic_demand_nonfeed_unadj*arr_agrc_frac_imported
        arr_agrc_production_nonfeed_unadj = arr_agrc_domestic_demand_nonfeed_unadj - arr_agrc_imports_unadj + arr_agrc_exports_unadj
 

        ##  4. Apply production wasted scalar to projected supply requirements, then re-calculate demand (reducing domestic production loss will drop future demands)
        
        vec_agrc_frac_production_wasted = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_agrc_frac_production_lost,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        vec_agrc_frac_production_wasted_scalar = np.nan_to_num((1 - vec_agrc_frac_production_wasted[0])/(1 - vec_agrc_frac_production_wasted), posinf = 0.0)
       
        arr_agrc_production_nonfeed_unadj = (arr_agrc_production_nonfeed_unadj.transpose()*vec_agrc_frac_production_wasted_scalar).transpose()
        arr_agrc_domestic_demand_nonfeed_unadj = arr_agrc_production_nonfeed_unadj + arr_agrc_imports_unadj
        
        # array gives the total yield of crop type i allocated to livestock type j at time 0
        arr_lndu_yield_i_reqd_lvst_j_init = np.outer(vec_agrc_production_init_lvstfeed, vec_lvst_feed_allocation_weights)

        return (
            # agrc vars
            arr_agrc_domestic_demand_nonfeed_unadj,
            arr_agrc_exports_unadj,
            arr_agrc_imports_unadj,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            vec_agrc_frac_cropland_area,
            vec_agrc_frac_production_wasted,
            # lvst vars
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_domestic_demand_unadj,
            arr_lvst_exports_unadj,
            arr_lvst_imports_unadj,
            arr_lvst_domestic_production_unadj,
            vec_lvst_carry_capacity_scale,
            vec_lvst_feed_allocation_weights
        )



    def project_per_capita_demand(self,
        dem_0: np.ndarray, # initial demand (e.g., total yield/livestock produced per acre) ()
        pop: np.ndarray, # population (vec_pop)
        gdp_per_capita_rates: np.ndarray, # driver of demand growth: gdp/capita (vec_rates_gdp_per_capita)
        elast: np.ndarray, # elasticity of demand per capita to growth in gdp/capita (e.g., arr_lvst_elas_demand)
        dem_pc_scalar_exog = None, # exogenous demand per capita scalar representing other changes in the exogenous per-capita demand (can be used to represent population changes)
        return_type: type = float # return type of array
    ) -> np.ndarray:

        """
        Project per capita demand for agriculture and/or livestock

        Function Arguments
        ------------------
        - dem_0: initial demand (e.g., total yield/livestock produced per acre)
        - pop: population (vec_pop)
        - gdp_per_capita_rates driver of demand growth: gdp/capita 
            (vec_rates_gdp_per_capita)
        - elast elasticity of demand per capita to growth in gdp/capita (e.g., 
            arr_lvst_elas_demand)
        - dem_pc_scalar_exog: exogenous demand per capita scalar representing 
            other changes in the exogenous per-capita demand (can be used to 
            represent population changes)
        - return_type: return type of array
        """

        # get the demand scalar to apply to per-capita demands
        dem_scale_proj_pc = (gdp_per_capita_rates.transpose()*elast[0:-1].transpose()).transpose()
        dem_scale_proj_pc = np.cumprod(1 + dem_scale_proj_pc, axis = 0)
        dem_scale_proj_pc = np.concatenate([np.ones((1,len(dem_scale_proj_pc[0]))), dem_scale_proj_pc])
        
        # estimate demand for livestock (used in CBA) - start with livestock population per capita
        dem_pc_scalar_exog = np.ones(pop.shape) if (dem_pc_scalar_exog is None) else dem_pc_scalar_exog

        if dem_pc_scalar_exog.shape == pop.shape:
            arr_dem_base = np.outer(pop*dem_pc_scalar_exog, dem_0/pop[0])
        elif dem_pc_scalar_exog.shape == dem_scale_proj_pc.shape:
            arr_pc = (dem_0/pop[0])*dem_pc_scalar_exog
            arr_dem_base = (pop*arr_pc.transpose()).transpose()
        else:
            raise ValueError(f"Invalid shape of dem_pc_scalar_exog: valid shapes are '{pop.shape}' and '{dem_scale_proj_pc.shape}'.")

        # get the total demand
        arr_dem_base = np.array(dem_scale_proj_pc*arr_dem_base).astype(return_type)

        return arr_dem_base



    def project_integrated_land_use(self,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_efs: np.ndarray,
        arr_agrc_production_nonfeed_unadj: np.ndarray,
        arr_agrc_yield_factors: np.ndarray,
        arr_lndu_frac_increasing_net_exports_met: np.ndarray,
        arr_lndu_frac_increasing_net_imports_met: np.ndarray,
        arr_lndu_yield_by_lvst: np.ndarray,
        arr_lvst_dem: np.ndarray,
        vec_agrc_frac_cropland_area: np.ndarray,
        vec_lndu_frac_grassland_pasture: np.ndarray,
        vec_lndu_yrf: np.ndarray,
        vec_lvst_pop_init: np.ndarray,
        vec_lvst_pstr_weights: np.ndarray,
        vec_lvst_scale_cc: np.ndarray,
        n_tp: Union[int, None] = None,
    ) -> Tuple:
        """
        Integrated land use model, which performs required land use transition 
            adjustments

        Function Arguments
        ------------------
        - vec_initial_area: initial state vector of area
        - arrs_transitions: array of transition matrices, ordered by time period
        - arrs_efs: array of emission factor matrices, ordered by time period
        - arr_agrc_production_nonfeed_unadj: array of agricultural non-feed 
            demand yield (human consumption)
        - arr_agrc_yield_factors: array of agricultural yield factors
        - arr_lndu_frac_increasing_net_exports_met: fraction--by land use type--
            of increases to net exports that are met. Adjusts production demands 
            downward if less than 1.
        - arr_lndu_frac_increasing_net_exports_met: fraction--by land use type--
            of increases to net imports that are met. Adjusts production demands 
            upward if less than 1.
        - arr_lndu_yield_by_lvst: array of lvst yield by land use category (used 
            to project future livestock supply). Array gives the total yield of 
            crop type i allocated to livestock type j at time 0 
            (attr_agriculture.n_key_values x attr_livestock.n_key_values)
        - arr_lvst_dem: array of livestock demand
        - vec_agrc_frac_cropland_area: vector of fractions of agricultural area 
            fractions by classes
        - vec_lndu_frac_grassland_pasture: vector giving the fraction of 
            grassland that is pasture
        - vec_lndu_yrf: vector of land use reallocation factor
        - vec_lvst_pop_init: vector, by livestock class, of initial livestock 
            populations
        - vec_lvst_pstr_weights: vector of weighting of animal classes to 
            determine which changes in animal population affect demand for land
        - vec_lvst_scale_cc: vector of livestock carrying capacity scalar to 
            apply

        Keyword Arguments
        ------------------
        n_tp: number of time periods to run. If None, runs AFOLU.n_time_periods

        Returns
        -------
        Tuple with 14 elements:
        - arr_agrc_change_to_net_imports_lost,
        - arr_agrc_frac_cropland,
        - arr_agrc_net_import_increase,
        - arr_agrc_yield,
        - arr_emissions_conv,
        - arr_emissions_conv_matrices,
        - arr_land_use,
        - arr_lvst_change_to_net_imports_lost,
        - arr_lvst_net_import_increase,
        - arr_lvst_pop_ardj,
        - arrs_land_conv,
        - arrs_transitions_adj,
        - arrs_yields_per_livestock

        """

        t0 = time.time()

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get attributes
        pycat_agrc = self.model_attributes.get_subsector_attribute(self.subsec_name_agrc, "pycategory_primary")
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        pycat_lvst = self.model_attributes.get_subsector_attribute(self.subsec_name_lvst, "pycategory_primary")
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]

        # set some commonly called attributes and indices in arrays
        m = attr_lndu.n_key_values

        # initialize variables for livestock
        vec_lvst_dem_gr_iterator = np.ones(len(arr_lvst_dem[0]))
        vec_lvst_cc_denom = vec_initial_area[self.ind_lndu_grass]*vec_lvst_pstr_weights*vec_lndu_frac_grassland_pasture[0]
        vec_lvst_cc_init = np.nan_to_num(
            vec_lvst_pop_init/vec_lvst_cc_denom, 
            nan = 0.0, 
            posinf = 0.0
        )

        # intilize output arrays, including land use, land converted, emissions, and adjusted transitions
        arr_agrc_frac_cropland = np.array([vec_agrc_frac_cropland_area for k in range(n_tp)])
        arr_agrc_net_import_increase = np.zeros((n_tp, attr_agrc.n_key_values))
        arr_agrc_change_to_net_imports_lost = np.zeros((n_tp, attr_agrc.n_key_values))
        arr_agrc_yield = np.array([(vec_initial_area[self.ind_lndu_crop]*vec_agrc_frac_cropland_area*arr_agrc_yield_factors[0]) for k in range(n_tp)])
        arr_emissions_conv = np.zeros((n_tp, attr_lndu.n_key_values))
        arr_emissions_conv_matrices = np.zeros(arrs_transitions.shape)
        arr_land_use = np.array([vec_initial_area for k in range(n_tp)])
        arr_lvst_dem_adj = arr_lvst_dem.copy().astype(int)
        arr_lvst_pop_adj = arr_lvst_dem.copy().astype(int)
        arr_lvst_net_import_increase = np.zeros((n_tp, attr_lvst.n_key_values))
        arr_lvst_change_to_net_imports_lost = np.zeros((n_tp, attr_lvst.n_key_values))
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_transitions_adj = np.zeros(arrs_transitions.shape)
        arrs_yields_per_livestock = np.array([arr_lndu_yield_by_lvst for k in range(n_tp)])

        """
        Rough note on the transition adjustment process:

        a. Start with land use prevalance at time ind_first_nz 
            i. estimated as prevalence at 
                x_{ind_first_nz - 1}Q_{ind_first_nz - 1}
        b. Using land use carrying capacity and pasture fraction & crop demands,
             get requirements for livestock and use to adjust pasture and
             cropland requirement (see scalar_lndu_pstr and scalar_lndu_crop,
             respectively)
            i. Use `model_afolu.get_lndu_scalar_max_out_states` to get true 
                positional scalars. This accounts for states that might 
                "max out" (as determined by 
                model_afolu.mask_lndu_max_out_states), or reach 1 or 0 
                probability during the scaling process.
        c. Then, with the scalars obtained, adjust the matrix using 
            model_afolu.adjust_transition_matrix
        """
        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp - 1:
            
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            if i_ef != i:
                self._log(
                    f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "warning"
                )

            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            if i_tr != i:
                self._log(
                    f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "warning"
                )

            # calculate the unadjusted land use areas (projected to time step i + 1)
            area_crop_cur = x[self.ind_lndu_crop]
            area_crop_proj = np.dot(x, arrs_transitions[i_tr][:, self.ind_lndu_crop])
            area_pstr_cur = x[self.ind_lndu_grass]*vec_lndu_frac_grassland_pasture[i]
            area_pstr_proj = np.dot(x, arrs_transitions[i_tr][:, self.ind_lndu_grass])*vec_lndu_frac_grassland_pasture[i + 1]
            vec_agrc_cropland_area_proj = area_crop_proj*arr_agrc_frac_cropland[i]


            ##  LIVESTOCK - calculate carrying capacities, demand used for pasture reallocation, and net surplus
            
            vec_lvst_cc_proj = vec_lvst_scale_cc[i + 1]*vec_lvst_cc_init
            inds_lvst_where_pop_noncc = np.where(vec_lvst_cc_proj == 0)[0]
            vec_lvst_prod_proj = vec_lvst_cc_proj*area_pstr_proj*vec_lvst_pstr_weights
            vec_lvst_unmet_demand = np.nan_to_num(arr_lvst_dem[i + 1] - vec_lvst_prod_proj)
            
            # calculate net surplus met
            vec_lvst_unmet_demand_to_impexp = (
                sf.vec_bounds(vec_lvst_unmet_demand, (0, np.inf))
                *arr_lndu_frac_increasing_net_imports_met[i + 1, self.ind_lndu_grass]
            )
            vec_lvst_unmet_demand_to_impexp += (
                sf.vec_bounds(vec_lvst_unmet_demand, (-np.inf, 0))
                *arr_lndu_frac_increasing_net_exports_met[i + 1, self.ind_lndu_grass]
            )
            vec_lvst_unmet_demand_lost = vec_lvst_unmet_demand - vec_lvst_unmet_demand_to_impexp
            
            # update production in livestock to account for lost unmet demand (unmet demand is + if imports increase, - if exports increase)
            vec_lvst_pop_adj = vec_lvst_prod_proj + vec_lvst_unmet_demand_lost
            if len(inds_lvst_where_pop_noncc) > 0:
                np.put(vec_lvst_unmet_demand_to_impexp, inds_lvst_where_pop_noncc, 0)
                np.put(vec_lvst_unmet_demand_lost, inds_lvst_where_pop_noncc, 0)
                np.put(vec_lvst_pop_adj, inds_lvst_where_pop_noncc, arr_lvst_dem[i + 1, inds_lvst_where_pop_noncc])
            
            # get unmet demand, reallocation, and final increase to net imports, then update the population
            vec_lvst_unmet_demand = vec_lvst_unmet_demand_to_impexp
            vec_lvst_reallocation = vec_lvst_unmet_demand*vec_lndu_yrf[i + 1] # demand for livestock met by reallocating land
            vec_lvst_net_import_increase = vec_lvst_unmet_demand - vec_lvst_reallocation # demand for livestock met by increasing net imports (neg => net exports)
            vec_lvst_pop_adj += vec_lvst_reallocation

            # update output arrays
            arr_lvst_pop_adj[i + 1] = np.round(vec_lvst_pop_adj).astype(int)
            
            # set growth in livestock relative to baseline and carrying capacity
            vec_lvst_dem_gr_iterator = np.nan_to_num(vec_lvst_pop_adj/arr_lvst_dem[0], 1.0, posinf = 1.0)
            vec_lvst_cc_proj_adj = np.nan_to_num(vec_lvst_pop_adj/(area_pstr_proj*vec_lvst_pstr_weights), 0.0, posinf = 0.0)

            # calculate required increase in transition probabilities
            area_lndu_pstr_increase = sum(np.nan_to_num(vec_lvst_reallocation/vec_lvst_cc_proj_adj, 0, posinf = 0.0))
            area_grss_proj = np.nan_to_num(area_pstr_proj/vec_lndu_frac_grassland_pasture[i + 1], 0.0, posinf = 0.0)
            scalar_lndu_pstr = (area_grss_proj + area_lndu_pstr_increase)/np.dot(x, arrs_transitions[i_tr][:, self.ind_lndu_grass])
            
            mask_lndu_max_out_states_pstr = self.get_lndu_scalar_max_out_states(scalar_lndu_pstr)
            scalar_lndu_pstr = self.get_matrix_column_scalar(
                arrs_transitions[i_tr][:, self.ind_lndu_grass],
                scalar_lndu_pstr,
                x,
                mask_max_out_states = mask_lndu_max_out_states_pstr
            )


            ##  AGRICULTURE - calculate demand increase in crops, which is a function of gdp/capita (exogenous) and livestock demand (used for feed)
            
            vec_agrc_feed_dem_yield = sum((arr_lndu_yield_by_lvst*vec_lvst_dem_gr_iterator).transpose())
            vec_agrc_total_dem_yield = (arr_agrc_production_nonfeed_unadj[i + 1] + vec_agrc_feed_dem_yield)
            
            # calculate net surplus for yields
            vec_agrc_proj_yields = vec_agrc_cropland_area_proj*arr_agrc_yield_factors[i + 1]
            vec_agrc_unmet_demand_yields = vec_agrc_total_dem_yield - vec_agrc_proj_yields
            vec_agrc_unmet_demand_yields_to_impexp = sf.vec_bounds(vec_agrc_unmet_demand_yields, (0, np.inf))*arr_lndu_frac_increasing_net_imports_met[i + 1, self.ind_lndu_crop]
            vec_agrc_unmet_demand_yields_to_impexp += sf.vec_bounds(vec_agrc_unmet_demand_yields, (-np.inf, 0))*arr_lndu_frac_increasing_net_exports_met[i + 1, self.ind_lndu_crop]
            vec_agrc_unmet_demand_yields_lost = vec_agrc_unmet_demand_yields - vec_agrc_unmet_demand_yields_to_impexp
            
            # adjust yields for import/export scalar
            vec_agrc_proj_yields_adj = vec_agrc_proj_yields + vec_agrc_unmet_demand_yields_lost
            vec_agrc_yield_factors_adj = np.nan_to_num(vec_agrc_proj_yields_adj/vec_agrc_cropland_area_proj, 0.0, posinf = 0.0) # replaces arr_agrc_yield_factors[i + 1] below
            vec_agrc_total_dem_yield = vec_agrc_proj_yields_adj + vec_agrc_unmet_demand_yields_to_impexp
            
            # now, generate modified crop areas and net surplus of crop areas
            vec_agrc_dem_cropareas = np.nan_to_num(vec_agrc_total_dem_yield/vec_agrc_yield_factors_adj, posinf = 0.0)
            vec_agrc_unmet_demand = vec_agrc_dem_cropareas - vec_agrc_cropland_area_proj
            vec_agrc_reallocation = vec_agrc_unmet_demand*vec_lndu_yrf[i + 1]
            
            # get surplus yield (increase to net imports)
            vec_agrc_net_imports_increase = (vec_agrc_unmet_demand - vec_agrc_reallocation)*vec_agrc_yield_factors_adj
            vec_agrc_cropareas_adj = vec_agrc_cropland_area_proj + vec_agrc_reallocation
            vec_agrc_yield_adj = vec_agrc_total_dem_yield - vec_agrc_net_imports_increase
            
            # get the true scalar
            scalar_lndu_crop = sum(vec_agrc_cropareas_adj)/np.dot(x, arrs_transitions[i_tr][:, self.ind_lndu_crop])
            mask_lndu_max_out_states_crop = self.get_lndu_scalar_max_out_states(scalar_lndu_crop)
            scalar_lndu_crop = self.get_matrix_column_scalar(
                arrs_transitions[i_tr][:, self.ind_lndu_crop],
                scalar_lndu_crop,
                x,
                mask_max_out_states = mask_lndu_max_out_states_crop
            )


            ##  SET TRANSITION ADJUSTMENTS

            # force all changes in the forest_primary row to be applied to forest primary--e.g., reductions in fp -> pstr and fp -> crop should => increase in fp -> fp only
            dict_adj = {}
            dict_adj.update(
                dict(zip([(r, self.ind_lndu_grass) for r in range(len(scalar_lndu_pstr))], scalar_lndu_pstr))
            )
            dict_adj.update(
                dict(zip([(r, self.ind_lndu_crop) for r in range(len(scalar_lndu_crop))], scalar_lndu_crop))
            )

            # force settlements and mangroves to stay stable
            dict_adj.update({
                (self.ind_lndu_fstm, ): 1,
                (self.ind_lndu_stlm, ): 1
            })

            # force primary forest to other categories to stay stable (if applicable)
            cats_ignore = [self.cat_lndu_crop, self.cat_lndu_fstp, self.cat_lndu_grass]
            ind_lndu_fstp = attr_lndu.get_key_value_index(self.cat_lndu_fstp)
            for cat in attr_lndu.key_values:
                ind = attr_lndu.get_key_value_index(cat)
                if cat not in cats_ignore:
                    dict_adj.update({(ind_lndu_fstp, ind): 1})


            # adjust the transition matrix
            trans_adj = self.adjust_transition_matrix(
                arrs_transitions[i_tr], 
                dict_adj
            )
            # calculate final land conversion and emissions
            arr_land_conv = (trans_adj.transpose()*x.transpose()).transpose()
            arr_emissions_conv_matrix = (trans_adj*arrs_efs[i_ef]).transpose()*x.transpose() # sums across columns
            arr_emissions_conv_matrix = arr_emissions_conv_matrix.transpose()
            vec_emissions_conv = arr_emissions_conv_matrix.sum(axis = 1)

            if i + 1 < n_tp:
                # update agriculture arrays
                rng_agrc = list(range((i + 1)*attr_agrc.n_key_values, (i + 2)*attr_agrc.n_key_values))
                np.put(arr_agrc_change_to_net_imports_lost, rng_agrc, vec_agrc_unmet_demand_yields_lost)
                np.put(arr_agrc_frac_cropland, rng_agrc, vec_agrc_cropareas_adj/sum(vec_agrc_cropareas_adj))
                np.put(arr_agrc_net_import_increase, rng_agrc, np.round(vec_agrc_net_imports_increase), 2)
                np.put(arr_agrc_yield, rng_agrc, vec_agrc_yield_adj)

                # update livestock arrays
                arr_lvst_change_to_net_imports_lost[i + 1] = vec_lvst_unmet_demand_lost
                arr_lvst_net_import_increase[i + 1] = np.round(vec_lvst_net_import_increase).astype(int)

            # non-ag arrays
            rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)

            arr_emissions_conv_matrices[i] = arr_emissions_conv_matrix
            arrs_land_conv[i] = arr_land_conv
            arrs_transitions_adj[i] = trans_adj

            # update land use vector and iterate
            x = np.matmul(x, trans_adj)
            i += 1

        # add on final time step by repeating the transition matrix
        trans_adj = arrs_transitions_adj[i - 1]
        # calculate final land conversion and emissions
        arr_land_conv = (trans_adj.transpose()*x.transpose()).transpose()
        arr_emissions_conv_matrix = (trans_adj*arrs_efs[len(arrs_efs) - 1]).transpose()*x.transpose() # sums across columns
        arr_emissions_conv_matrix = arr_emissions_conv_matrix.transpose()
        vec_emissions_conv = arr_emissions_conv_matrix.sum(axis = 1)
        
        # add to tables
        rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
        np.put(arr_land_use, rng_put, x)
        np.put(arr_emissions_conv, rng_put, vec_emissions_conv)

        # update matrix lists
        arr_emissions_conv_matrices[i] = arr_emissions_conv_matrix
        arrs_land_conv[i] = arr_land_conv
        arrs_transitions_adj[i] = trans_adj

        return (
            arr_agrc_change_to_net_imports_lost,
            arr_agrc_frac_cropland,
            arr_agrc_net_import_increase,
            arr_agrc_yield,
            arr_emissions_conv,
            arr_emissions_conv_matrices,
            arr_land_use,
            arr_lvst_change_to_net_imports_lost,
            arr_lvst_net_import_increase,
            arr_lvst_pop_adj,
            arrs_land_conv,
            arrs_transitions_adj,
            arrs_yields_per_livestock
        )



    def project_land_use(self,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_efs: np.ndarray,
        n_tp: Union[int, None] = None,
    ) -> Tuple:

        """
        NEED DOCSTRING

        Function Arguments
        ------------------
        """

        t0 = time.time()

        np.seterr(divide = "ignore", invalid = "ignore")

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get land use info
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

        # intilize the land use and conversion emissions array
        shp_init = (n_tp, attr_lndu.n_key_values)
        arr_land_use = np.zeros(shp_init)
        arr_emissions_conv = np.zeros(shp_init)
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp:
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            (
                self._log(
                    f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "info"
                )
                if i_ef != i
                else None
            )

            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            (
                self._log(
                    f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "info"
                )
                if i_tr != i
                else None
            )

            # calculate land use, conversions, and emissions
            vec_emissions_conv = sum((arrs_transitions[i_tr] * arrs_efs[i_ef]).transpose()*x.transpose())
            arr_land_conv = (arrs_transitions[i_tr].transpose()*x.transpose()).transpose()
            
            # update matrices
            rng_put = np.arange(i*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)
            np.put(arrs_land_conv, np.arange(i*attr_lndu.n_key_values**2, (i + 1)*attr_lndu.n_key_values**2), arr_land_conv)
            
            # update land use vector
            x = np.matmul(x, arrs_transitions[i_tr])

            i += 1

        t1 = time.time()
        t_elapse = round(t1 - t0, 2)
        self._log(f"Land use projection complete in {t_elapse} seconds.", type_log = "info")

        return arr_emissions_conv, arr_land_use, arrs_land_conv



    def project_harvested_wood_products(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_hh: np.ndarray,
        vec_gdp: np.ndarray,
        vec_rates_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict,
        n_projection_time_periods: int,
        projection_time_periods: np.ndarray,
        dict_check_integrated_variables: dict
    ) -> List[pd.DataFrame]:
        """
        Project sequestration from harvested wood products. Returns

        Function Arguments
        ------------------
        - df_afolu_trajectories: DataFrame containing 
        - vec_hh: np vector giving the number of households in same order as 
            df_afolu_trajectories
        - vec_gdp: np vector giving the GDP in same order as 
            df_afolu_trajectories
        - vec_rates_gdp: np vector giving rates of GDP growth in same order as 
            df_afolu_trajectories [length = n_rows(df_afolu_trajectories) - 1]
        - vec_rates_gdp_per_capita: np vector giving rates of GDP/Capita growth 
            in same order as df_afolu_trajectories 
            [length = n_rows(df_afolu_trajectories) - 1]
        - dict_dims: 
        - n_projection_time_periods:
        - projection_time_periods:
        - dict_check_integrated_variables:
        """


        # IPPU components
        if dict_check_integrated_variables[self.subsec_name_ippu]:

            # get projections of industrial wood and paper product demand
            attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)
            ind_paper = attr_ippu.get_key_value_index(self.cat_ippu_paper)
            ind_wood = attr_ippu.get_key_value_index(self.cat_ippu_wood)

            # production data
            arr_production, dfs_ippu_harvested_wood = self.model_ippu.get_production_with_recycling_adjustment(df_afolu_trajectories, vec_rates_gdp)
            list_ippu_vars = self.model_attributes.build_varlist(self.subsec_name_ippu, self.model_ippu.modvar_ippu_demand_for_harvested_wood)
            arr_frst_harvested_wood_industrial = 0.0
            vec_frst_harvested_wood_industrial_paper = 0.0
            vec_frst_harvested_wood_industrial_wood = 0.0

            # find the data frame with output
            keep_going = True
            i = 0
            while (i < len(dfs_ippu_harvested_wood)) and keep_going:
                df = dfs_ippu_harvested_wood[i]
                if set(list_ippu_vars).issubset(df.columns):
                    arr_frst_harvested_wood_industrial = self.model_attributes.get_standard_variables(df, self.model_ippu.modvar_ippu_demand_for_harvested_wood, False, "array_base", expand_to_all_cats = True)
                    vec_frst_harvested_wood_industrial_paper = arr_frst_harvested_wood_industrial[:, ind_paper]
                    vec_frst_harvested_wood_industrial_wood = arr_frst_harvested_wood_industrial[:, ind_wood]
                    keep_going = False

                i += 1

            # remove some unneeded vars
            array_ippu_production = 0
            dfs_ippu_harvested_wood = 0

        else:
            arr_frst_harvested_wood_industrial = 0.0
            vec_frst_harvested_wood_industrial_paper = 0.0
            vec_frst_harvested_wood_industrial_wood = 0.0


        # get initial domestic demand for wood products
        vec_frst_harvested_wood_domestic = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_frst_init_per_hh_wood_demand, False, "array_base")
        vec_frst_harvested_wood_domestic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_frst_init_per_hh_wood_demand,
            self.model_ippu.modvar_ippu_demand_for_harvested_wood,
            "mass"
        )

        # If energy components are available, scale hh demand from SCOE
        if dict_check_integrated_variables[self.subsec_name_scoe]:

            # get changes in biomass energy demand for stationary emissions (largely driven by wood)
            df_scoe = self.model_energy.project_scoe(
                df_afolu_trajectories,
                vec_hh,
                vec_gdp,
                vec_rates_gdp_per_capita,
                dict_dims,
                n_projection_time_periods,
                projection_time_periods
            )
            # get the enfu attrbute table and use it to retrieve the biomass fuel demand
            attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
            ind_biomass = attr_enfu.get_key_value_index(self.cat_enfu_biomass)
            vec_scoe_biomass_fuel_demand = self.model_attributes.get_standard_variables(
                df_scoe,
                self.model_energy.modvar_enfu_energy_demand_by_fuel_scoe,
                False,
                "array_base"
            )
            vec_scoe_biomass_fuel_demand = vec_scoe_biomass_fuel_demand[:, ind_biomass]
            # calculate the change and growth rate
            vec_scoe_biomass_fuel_demand_change = np.nan_to_num(vec_scoe_biomass_fuel_demand[1:]/vec_scoe_biomass_fuel_demand[0:-1], 1.0, posinf = 1.0)
            vec_scoe_biomass_fuel_demand_growth_rate = np.cumprod(np.insert(vec_scoe_biomass_fuel_demand_change, 0, 1.0))
            df_scoe = 0
            
            vec_frst_harvested_wood_domestic *= vec_hh[0]
            vec_frst_harvested_wood_domestic = vec_frst_harvested_wood_domestic[0]*vec_scoe_biomass_fuel_demand_growth_rate
        else:
            # assume growth proportional to HHs
            vec_frst_harvested_wood_domestic *= vec_hh

        # get half-life factors for FOD model
        vec_frst_k_hwp_paper = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_frst_hwp_half_life_paper,
            False,
            "array_base"
        )
        vec_frst_k_hwp_paper = np.log(2)/vec_frst_k_hwp_paper
        vec_frst_k_hwp_wood = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_frst_hwp_half_life_wood,
            False,
            "array_base"
        )
        vec_frst_k_hwp_wood = np.log(2)/vec_frst_k_hwp_wood

        # totals
        vec_frst_ef_c = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_frst_ef_c_per_hwp,
            False,
            "array_base",
            var_bounds = (0, np.inf)
        )
        vec_frst_c_paper = vec_frst_harvested_wood_industrial_paper*vec_frst_ef_c
        vec_frst_c_wood = (vec_frst_harvested_wood_industrial_wood + vec_frst_harvested_wood_domestic)*vec_frst_ef_c
        self.vec_frst_c_wood = vec_frst_c_wood

        # set a lookback based on some number of years (max half-life to estimate some amount of carbon stock)
        if self.model_attributes.configuration.get("historical_harvested_wood_products_method") == "back_project":
            n_years_lookback = int(self.model_attributes.configuration.get("historical_back_proj_n_periods"))#int(np.round(np.log(2)*max(max(1/vec_frst_k_hwp_paper), max(1/vec_frst_k_hwp_wood))))
            
            if n_years_lookback > 0:
                # need to subtract 1 because we apply the mean to rates
                n_years_mean = min(5, len(vec_frst_c_paper) - 1)

                # back-project previous paper products
                r_paper = np.mean(vec_frst_c_paper[1:(1 + n_years_mean)]/vec_frst_c_paper[0:n_years_mean])
                vec_frst_c_paper = np.concatenate([
                    np.array(
                        [
                            vec_frst_c_paper[0]*(r_paper**(x - n_years_lookback)) 
                            for x in range(n_years_lookback)
                        ]
                    ),
                    vec_frst_c_paper
                ])

                # back-project previous wood products
                r_wood = np.mean(vec_frst_c_wood[1:(1 + n_years_mean)]/vec_frst_c_wood[0:n_years_mean])
                vec_frst_c_wood = np.concatenate([
                    np.array(
                        [
                            vec_frst_c_wood[0]*(r_wood**(x - n_years_lookback)) 
                            for x in range(n_years_lookback)
                        ]
                    ),
                    vec_frst_c_wood
                ])
                vec_frst_k_hwp_paper = np.concatenate([np.array([vec_frst_k_hwp_paper[0] for x in range(n_years_lookback)]), vec_frst_k_hwp_paper])
                vec_frst_k_hwp_wood = np.concatenate([np.array([vec_frst_k_hwp_wood[0] for x in range(n_years_lookback)]), vec_frst_k_hwp_wood])
        else:
            # set up n_years_lookback to be based on historical
            n_years_lookback = 0
            raise ValueError(f"Error in project_harvested_wood_products: historical_harvested_wood_products_method 'historical' not supported at the moment.")

        # initialize and run using assumptions of steady-state (see Equation 12.4)
        vec_frst_c_from_hwp_paper = np.zeros(len(vec_frst_k_hwp_paper))
        vec_frst_c_from_hwp_paper[0] = np.mean(vec_frst_c_paper[0:min(5, len(vec_frst_c_paper))])/vec_frst_k_hwp_paper[0]
        vec_frst_c_from_hwp_wood = np.zeros(len(vec_frst_k_hwp_wood))
        vec_frst_c_from_hwp_wood[0] = np.mean(vec_frst_c_wood[0:min(5, len(vec_frst_c_wood))])/vec_frst_k_hwp_wood[0]

        # execute the FOD model
        for i in range(len(vec_frst_c_from_hwp_paper) - 1):
            # paper
            current_stock_paper = vec_frst_c_from_hwp_paper[0] if (i == 0) else vec_frst_c_from_hwp_paper[i]
            exp_k_paper = np.exp(-vec_frst_k_hwp_paper[i])
            vec_frst_c_from_hwp_paper[i + 1] = current_stock_paper*exp_k_paper + ((1 - exp_k_paper)/vec_frst_k_hwp_paper[i])*vec_frst_c_paper[i]
            # wood
            current_stock_wood = vec_frst_c_from_hwp_wood[0] if (i == 0) else vec_frst_c_from_hwp_wood[i]
            exp_k_wood = np.exp(-vec_frst_k_hwp_wood[i])
            vec_frst_c_from_hwp_wood[i + 1] = current_stock_wood*exp_k_wood + ((1 - exp_k_wood)/vec_frst_k_hwp_wood[i])*vec_frst_c_wood[i]

        # reduce from look back
        if n_years_lookback > 0:
            vec_frst_c_from_hwp_paper = vec_frst_c_from_hwp_paper[(n_years_lookback - 1):]
            vec_frst_c_from_hwp_wood = vec_frst_c_from_hwp_wood[(n_years_lookback - 1):]
        vec_frst_c_from_hwp_paper_delta = vec_frst_c_from_hwp_paper[1:] - vec_frst_c_from_hwp_paper[0:-1]
        vec_frst_c_from_hwp_wood_delta = vec_frst_c_from_hwp_wood[1:] - vec_frst_c_from_hwp_wood[0:-1]

        v_print = self.factor_c_to_co2*vec_frst_c_wood*self.model_attributes.get_scalar(self.model_ippu.modvar_ippu_demand_for_harvested_wood, "mass")
        v_mult = (1 - np.exp(-vec_frst_k_hwp_wood))/vec_frst_k_hwp_wood

        # get emissions from co2
        vec_frst_emissions_co2_hwp = vec_frst_c_from_hwp_paper_delta + vec_frst_c_from_hwp_wood_delta
        vec_frst_emissions_co2_hwp *= self.factor_c_to_co2
        vec_frst_emissions_co2_hwp *= -1*self.model_attributes.get_scalar(self.model_ippu.modvar_ippu_demand_for_harvested_wood, "mass")

        list_dfs_out = [
            self.model_attributes.array_to_df(vec_frst_emissions_co2_hwp, self.modvar_frst_emissions_co2_hwp)
        ]

        return list_dfs_out



    ##  LIVESTOCK

    def reassign_pops_from_proj_to_carry(self,
        arr_lu_derived: np.ndarray,
        arr_dem_based: np.ndarray,
    ) -> np.ndarray:
        """
        Before assigning net imports, there are many non-grazing animals to 
            consider (note that these animals are generally not 
            emission-intensive animals).
        
        Due to 0 graze area, their estimated population is infinite, or stored 
            as a negative.

        We assign their population as the demand-estimated population.

        Function Arguments
        ------------------
        - arr_lu_derived: array of animal populations based on land use
        - arr_dem_based: array of animal populations based on demand
        """
        if arr_lu_derived.shape != arr_dem_based.shape:
            raise ValueError(f"Error in reassign_pops_from_proj_to_carry: array dimensions do not match: arr_lu_derived = {arr_lu_derived.shape}, arr_dem_based = {arr_dem_based.shape}.")

        cols = np.where(arr_lu_derived[0] < 0)[0]
        n_row, n_col = arr_lu_derived.shape

        for w in cols:
            rng = np.arange(w*n_row, (w + 1)*n_row)
            np.put(arr_lu_derived.transpose(), rng, arr_dem_based[:, w])

        return arr_lu_derived



    ####################################
    ###                              ###
    ###    PRIMARY MODEL FUNCTION    ###
    ###                              ###
    ####################################

    def project(self,
        df_afolu_trajectories: pd.DataFrame,
        passthrough_tmp: str = None
    ) -> pd.DataFrame:

        """
        The project() method takes a data frame of input variables (ordered by 
            time series) and returns a data frame of output variables (model
            projections for agriculture and livestock, forestry, and land use) 
            the same order.

        Function Arguments
        ------------------
        - df_afolu_trajectories: pd.DataFrame with all required input fields as 
            columns. The model will not run if any required variables are 
            missing, but errors will detail which fields are missing.

        Notes
        -----
        - The .project() method is designed to be parallelized or called from 
            command line via __main__ in run_sector_models.py.
        - df_afolu_trajectories should have all input fields required (see 
            AFOLU.required_variables for a list of variables to be defined)
        - the df_afolu_trajectories.project method will run on valid time 
            periods from 1 .. k, where k <= n (n is the number of time periods). 
            By default, it drops invalid time periods. If there are missing 
            time_periods between the first and maximum, data are interpolated.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_afolu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_afolu_trajectories)

        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_afolu_trajectories)
        (
            dict_dims, 
            df_afolu_trajectories, 
            n_projection_time_periods, 
            projection_time_periods
        ) = self.model_attributes.check_projection_input_df(
            df_afolu_trajectories, 
            True, 
            True, 
            True
        )

        # check integrated variables for HWP
        dict_check_integrated_variables = self.model_attributes.check_integrated_df_vars(
            df_afolu_trajectories, 
            self.dict_integration_variables_by_subsector, 
            "all"
        )

        ##  CATEGORY INITIALIZATION
        pycat_agrc = self.model_attributes.get_subsector_attribute(self.subsec_name_agrc, "pycategory_primary")
        pycat_frst = self.model_attributes.get_subsector_attribute(self.subsec_name_frst, "pycategory_primary")
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        pycat_lsmm = self.model_attributes.get_subsector_attribute(self.subsec_name_lsmm, "pycategory_primary")
        pycat_lvst = self.model_attributes.get_subsector_attribute(self.subsec_name_lvst, "pycategory_primary")
        pycat_soil = self.model_attributes.get_subsector_attribute(self.subsec_name_soil, "pycategory_primary")
        
        # attribute tables
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        attr_frst = self.model_attributes.dict_attributes[pycat_frst]
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        attr_lsmm = self.model_attributes.dict_attributes[pycat_lsmm]
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]
        attr_soil = self.model_attributes.dict_attributes[pycat_soil]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base"
        )
        vec_gdp_per_capita = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp_per_capita, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base"
        )
        vec_hh = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_grnl_num_hh, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base"
        )
        vec_pop = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_gnrl_pop_total, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base"
        )

        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_afolu_trajectories[self.required_dimensions].copy()]


        ########################################
        #    LAND USE - UNADJUSTED VARIABLES   #
        ########################################

        # area of the country + the applicable scalar used to convert outputs
        area = float(
            self.model_attributes.get_standard_variables(
                df_afolu_trajectories,
                self.model_socioeconomic.modvar_gnrl_area,
                return_type = "array_base"
            )[0]
        )
        scalar_lndu_input_area_to_output_area = self.model_attributes.get_scalar(
            self.model_socioeconomic.modvar_gnrl_area, 
            "area"
        )

        # get the initial distribution of land
        vec_modvar_lndu_initial_frac = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lndu_initial_frac, 
            return_type = "array_base"
        )[0]

        vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_frac*area
        area_agrc_cropland_init = area*vec_modvar_lndu_initial_frac[self.ind_lndu_crop]

        arrs_lndu_q_unadj, arrs_lndu_ef_conv = self.get_markov_matrices(
            df_afolu_trajectories, 
            correct_emission_units = True,
            n_tp = n_projection_time_periods,
            target_units_area_modvar = self.model_socioeconomic.modvar_gnrl_area,
        )

        # factor for reallocating land in adjustment
        vec_lndu_reallocation_factor = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_reallocation_factor,
            return_type = "array_base"
        )

        # get fractions of increasing net exports/exports met
        arr_lndu_frac_increasing_net_exports_met = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_frac_increasing_net_exports_met,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        arr_lndu_frac_increasing_net_imports_met = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_frac_increasing_net_imports_met,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )

        # get fraction of grassland that is pasture
        vec_lndu_frac_grassland_pasture = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_frac_grassland_that_is_pasture,
            return_type = "array_base",
            var_bounds = (0, 1)
        )


        ##  GET INTEGRATED AGRICULTURE AND LIVESTOCK DEMANDS

        (
            # agrc vars
            arr_agrc_domestic_demand_nonfeed_unadj,
            arr_agrc_exports_unadj,
            arr_agrc_imports_unadj,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            vec_agrc_frac_cropland_area,
            vec_agrc_frac_production_wasted,
            # lvst vars
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_domestic_demand_unadj,
            arr_lvst_exports_unadj,
            arr_lvst_imports_unadj,
            arr_lvst_domestic_production_unadj,
            vec_lvst_carry_capacity_scale,
            vec_lvst_feed_allocation_weights
        ) = self.project_agrc_lvst_integrated_demands(
            df_afolu_trajectories,
            area_agrc_cropland_init,
            vec_pop,
            vec_rates_gdp_per_capita
        )



        ################################################
        #    CALCULATE LAND USE + AGRC/LVST DRIVERS    #
        ################################################

        # get land use projections (np arrays) - note, arrs_land_conv returns a list of matrices for troubleshooting
        (
            arr_agrc_change_to_net_imports_lost,
            arr_agrc_frac_cropland,
            arr_agrc_net_import_increase,
            arr_agrc_yield,
            arr_lndu_emissions_conv,
            arrs_lndu_emissions_conv_matrices,
            arr_land_use,
            arr_lvst_change_to_net_imports_lost,
            arr_lvst_net_import_increase,
            arr_lvst_pop,
            arrs_lndu_land_conv,
            arrs_lndu_q_adj,
            self.yields_per_livestock
        ) = self.project_integrated_land_use(
            vec_modvar_lndu_initial_area,
            arrs_lndu_q_unadj,
            arrs_lndu_ef_conv,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            arr_lndu_frac_increasing_net_exports_met,
            arr_lndu_frac_increasing_net_imports_met,
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_domestic_production_unadj,
            vec_agrc_frac_cropland_area,
            vec_lndu_frac_grassland_pasture,
            vec_lndu_reallocation_factor,
            arr_lvst_domestic_production_unadj[0],
            vec_lvst_feed_allocation_weights,
            vec_lvst_carry_capacity_scale,
            n_tp = n_projection_time_periods,
        )
        self.arrs_lndu_land_conv = arrs_lndu_land_conv

        # update imports/exports for agriculture
        arr_agrc_exports_adj = sf.vec_bounds(
            arr_agrc_exports_unadj - sf.vec_bounds(arr_agrc_net_import_increase, (-np.inf, 0)),
            (np.zeros(arr_agrc_yield.shape), arr_agrc_yield)
        )
        arr_agrc_imports_adj = sf.vec_bounds(
            arr_agrc_imports_unadj + sf.vec_bounds(arr_agrc_net_import_increase, (0, np.inf)),
            (0, np.inf)
        )

        # update demand from population
        arr_lvst_exports_adj = sf.vec_bounds(
            arr_lvst_exports_unadj - sf.vec_bounds(arr_lvst_net_import_increase, (-np.inf, 0)),
            (np.zeros(arr_lvst_pop.shape), arr_lvst_pop)
        )
        arr_lvst_imports_adj = sf.vec_bounds(
            arr_lvst_imports_unadj + sf.vec_bounds(arr_lvst_net_import_increase, (0, np.inf)),
            (0, np.inf)
        )
        arr_lvst_demand = arr_lvst_pop + arr_lvst_imports_adj - arr_lvst_exports_adj

        # assign some dfs that are used below in other subsectors
        df_agrc_frac_cropland = self.model_attributes.array_to_df(
            arr_agrc_frac_cropland, 
            self.modvar_agrc_area_prop_calc
        )

        df_land_use = self.model_attributes.array_to_df(
            arr_land_use, 
            self.modvar_lndu_area_by_cat
        )


        # calculate land use conversions
        arrs_lndu_conv_to = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 0) for x in arrs_lndu_land_conv])
        arrs_lndu_conv_from = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 1) for x in arrs_lndu_land_conv])

        # get total production wasted FLAG!!HEREHERE - check if arr_agrc_production_nonfeed_unadj is correct

        vec_agrc_food_produced_wasted_before_consumption = np.sum(
            arr_agrc_production_nonfeed_unadj.transpose()*vec_agrc_frac_production_wasted, 
            axis = 0
        )
        vec_agrc_food_produced_wasted_before_consumption *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_total_food_lost_in_ag,
            "mass"
        )
        
        # get total production that is wasted or lost that ends up in landfills
        vec_agrc_frac_production_loss_to_landfills = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_frac_production_loss_to_msw, 
            override_vector_for_single_mv_q = False,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        vec_agrc_food_wasted_to_landfills = vec_agrc_food_produced_wasted_before_consumption*vec_agrc_frac_production_loss_to_landfills
        vec_agrc_food_wasted_to_landfills *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_total_food_lost_in_ag,
            self.modvar_agrc_total_food_lost_in_ag_to_msw,
            "mass"
        )


        ##  UNIT CONVERSIONS

        # convert yield out units
        arr_agrc_yield_out = arr_agrc_yield*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_yield,
            "mass"
        )
        # convert exports/imports
        arr_agrc_exports_adj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_adjusted_equivalent_exports,
            "mass"
        )
        arr_agrc_imports_adj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_adjusted_equivalent_imports,
            "mass"
        )

        # convert change to net imports to yield units
        arr_agrc_change_to_net_imports_lost *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_changes_to_net_imports_lost,
            "mass"
        )
        """
        # JSYME REMOVED 2023-07-13: removing modvar_agrc_net_imports variable
        # self.modvar_agrc_net_imports = "Change to Net Imports of Crops"
        # convert change to net imports loss
        arr_agrc_net_import_increase *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_net_imports,
            "mass"
        )
        """;
        # get total domestic crop demand
        arr_agrc_demand_out = arr_agrc_yield_out*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yield,
            self.modvar_agrc_demand_crops,
            "mass"
        )
        arr_agrc_demand_out += arr_agrc_imports_adj*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_adjusted_equivalent_imports,
            self.modvar_agrc_demand_crops,
            "mass"
        )
        arr_agrc_demand_out -= arr_agrc_exports_adj*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_adjusted_equivalent_exports,
            self.modvar_agrc_demand_crops,
            "mass"
        )

        # convert land use conversion emission totals to config
        df_lndu_emissions_conv_matrices = self.format_transition_matrix_as_input_dataframe(
            arrs_lndu_emissions_conv_matrices,
            exclude_time_period = True,
            modvar = self.modvar_lndu_emissions_conv,
        )

        # convert land use conversion areas totals to config area
        df_lndu_area_conv_matrices = self.format_transition_matrix_as_input_dataframe(
            arrs_lndu_land_conv*scalar_lndu_input_area_to_output_area,
            exclude_time_period = True,
            modvar = self.modvar_lndu_area_converted,
        )
        
        # add to output data frame
        df_out += [
            df_agrc_frac_cropland,
            self.model_attributes.array_to_df(
                arr_agrc_change_to_net_imports_lost, 
                self.modvar_agrc_changes_to_net_imports_lost
            ),
            self.model_attributes.array_to_df(
                arr_agrc_demand_out, 
                self.modvar_agrc_demand_crops
            ),
            self.model_attributes.array_to_df(
                arr_agrc_exports_adj, 
                self.modvar_agrc_adjusted_equivalent_exports
            ),
            self.model_attributes.array_to_df(
                arr_agrc_imports_adj, 
                self.modvar_agrc_adjusted_equivalent_imports
            ),
            self.model_attributes.array_to_df(
                arr_agrc_yield_out, 
                self.modvar_agrc_yield
            ),
            self.model_attributes.array_to_df(
                vec_agrc_food_produced_wasted_before_consumption, 
                self.modvar_agrc_total_food_lost_in_ag
            ),
            self.model_attributes.array_to_df(
                vec_agrc_food_wasted_to_landfills, 
                self.modvar_agrc_total_food_lost_in_ag_to_msw
            ),
            self.model_attributes.array_to_df(
                arr_land_use*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_by_cat
            ),
            self.model_attributes.array_to_df(
                arrs_lndu_conv_from*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_converted_from_type
            ),
            self.model_attributes.array_to_df(
                arrs_lndu_conv_to*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_converted_to_type
            ),
            # matrix of conversion areas
            df_lndu_area_conv_matrices,
            
            self.model_attributes.array_to_df(
                arr_lndu_emissions_conv, 
                self.modvar_lndu_emissions_conv_away, 
                include_scalars = True
            ),
            # matrix of conversion emissions
            df_lndu_emissions_conv_matrices,

            self.model_attributes.array_to_df(
                arr_lvst_change_to_net_imports_lost, 
                self.modvar_lvst_changes_to_net_imports_lost
            ),
            self.model_attributes.array_to_df(
                arr_lvst_demand, 
                self.modvar_lvst_demand_livestock
            ),
            self.model_attributes.array_to_df(
                arr_lvst_exports_adj, 
                self.modvar_lvst_adjusted_equivalent_exports
            ),
            self.model_attributes.array_to_df(
                arr_lvst_imports_adj, 
                self.modvar_lvst_adjusted_equivalent_imports
            )
        ]


        ##  EXISTENCE EMISSIONS FOR OTHER LANDS, INCLUDING AG ACTIVITY ON PASTURES

        # get CH4 emissions from wetlands
        arr_lndu_ef_ch4_boc = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_ef_ch4_boc, True, "array_units_corrected")
        arr_lndu_ef_ch4_boc *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_lndu_ef_ch4_boc,
            "area"
        )

        fields_lndu_for_ch4_boc = self.model_attributes.build_target_varlist_from_source_varcats(
            self.modvar_lndu_ef_ch4_boc, 
            self.modvar_lndu_area_by_cat
        )
        arr_lndu_area_ch4_boc = np.array(df_land_use[fields_lndu_for_ch4_boc])

        df_out += [
            self.model_attributes.array_to_df(arr_lndu_area_ch4_boc*arr_lndu_ef_ch4_boc, self.modvar_lndu_emissions_ch4_from_wetlands)
        ]



        ##########################################
        #    BUILD SOME SHARED FACTORS (EF_i)    #
        ##########################################

        # agriculture fractions in dry/wet climate
        dict_arrs_agrc_frac_drywet = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_drywet,
            1,
            force_sum_equality = True,
            msg_append = "Agriculture dry/wet fractions by category do not sum to 1. See definition of dict_arrs_agrc_frac_drywet."
        )
        # agriculture fractions in temperate/tropical climate
        dict_arrs_agrc_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Agriculture temperate/tropical fractions by category do not sum to 1. See definition of dict_arrs_agrc_frac_temptrop."
        )
        # forest fractions in temperate/tropical climate
        dict_arrs_frst_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_frst_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Forest temperate NP/temperate NR/tropical fractions by category do not sum to 1. See definition of dict_arrs_frst_frac_temptrop."
        )
        # land use fractions in dry/wet climate
        dict_arrs_lndu_frac_drywet = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lndu_frac_drywet,
            1,
            force_sum_equality = True,
            msg_append = "Land use dry/wet fractions by category do not sum to 1. See definition of dict_arrs_lndu_frac_drywet."
        )
        # land use fractions in temperate/tropical climate
        dict_arrs_lndu_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lndu_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Land use temperate/tropical fractions by category do not sum to 1. See definition of dict_arrs_lndu_frac_temptrop."
        )

        ##  BUILD SOME FACTORS

        # get original EF4
        arr_soil_ef4_n_volatilisation = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_soil_ef4_n_volatilisation,
            expand_to_all_cats = True,
            return_type = "array_base"
        )
        # get EF4 for land use categories based on dry/wet (only applies to grassland)
        arr_lndu_ef4_n_volatilisation = 0.0
        for modvar_lndu_frac_drywet in dict_arrs_lndu_frac_drywet.keys():
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar_lndu_frac_drywet, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            arr_lndu_ef4_n_volatilisation += (dict_arrs_lndu_frac_drywet[modvar_lndu_frac_drywet].transpose()*arr_soil_ef4_n_volatilisation[:, ind_soil]).transpose()




        ##################
        #    FORESTRY    #
        ##################

        # get ordered fields from land use
        fields_lndu_forest_ordered = [self.dict_cats_frst_to_cats_lndu.get(x) for x in attr_frst.key_values]
        fields_ext = self.model_attributes.build_varlist(
            None, 
            restrict_to_category_values = fields_lndu_forest_ordered,
            variable_subsec = self.modvar_lndu_area_by_cat,
        )
        arr_area_frst = np.array(df_land_use[fields_ext])
        
        # get different variables
        arr_frst_ef_sequestration = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_sq_co2, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        arr_frst_ef_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_sq_co2,
            "area"
        )
        arr_frst_ef_methane = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_ef_ch4, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        arr_frst_ef_methane *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_ef_ch4,
            "area"
        )
        # build output variables
        df_out += [
            self.model_attributes.array_to_df(-1*arr_area_frst*arr_frst_ef_sequestration, self.modvar_frst_emissions_co2_sequestration),
            self.model_attributes.array_to_df(arr_area_frst*arr_frst_ef_methane, self.modvar_frst_emissions_ch4)
        ]


        ##  FOREST FIRES

        # initialize some variables that are called below
        arr_frst_frac_burned = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_average_fraction_burned_annually, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            expand_to_all_cats = True, 
            var_bounds = (0, 1)
        )
        arr_frst_ef_co2_fires = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_ef_co2_fires, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            expand_to_all_cats = True
        )
        # temperate biomass burned
        arr_frst_biomass_consumed_temperate = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_biomass_consumed_fire_temperate, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        arr_frst_biomass_consumed_temperate *= self.model_attributes.get_scalar(self.modvar_frst_biomass_consumed_fire_temperate, "mass")
        arr_frst_biomass_consumed_temperate /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_frst_biomass_consumed_fire_temperate,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        # tropical biomass burned
        arr_frst_biomass_consumed_tropical = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_frst_biomass_consumed_fire_tropical, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        arr_frst_biomass_consumed_tropical *= self.model_attributes.get_scalar(self.modvar_frst_biomass_consumed_fire_tropical, "mass")
        arr_frst_biomass_consumed_tropical /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_frst_biomass_consumed_fire_tropical,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        # setup biomass arrays as a dictionary
        dict_frst_modvar_to_array_forest_fires = {
            self.modvar_frst_frac_temperate_nutrient_poor: arr_frst_biomass_consumed_temperate,
            self.modvar_frst_frac_temperate_nutrient_rich: arr_frst_biomass_consumed_temperate,
            self.modvar_frst_frac_tropical: arr_frst_biomass_consumed_tropical
        }
        # loop over tropical/temperate NP/temperate NR
        arr_frst_emissions_co2_fires = 0.0
        for modvar in self.modvar_list_frst_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            self.model_attributes.get_standard_variables(
                df_afolu_trajectories, 
                self.modvar_frst_ef_ch4, 
                override_vector_for_single_mv_q = True, 
                return_type = "array_units_corrected"
            )
            # get forest area
            arr_frst_area_temptrop_burned_cur = arr_area_frst*dict_arrs_frst_frac_temptrop[modvar]*arr_frst_frac_burned
            arr_frst_total_dry_mass_burned_cur = arr_frst_area_temptrop_burned_cur*dict_frst_modvar_to_array_forest_fires[modvar]
            arr_frst_emissions_co2_fires += arr_frst_total_dry_mass_burned_cur*arr_frst_ef_co2_fires

        # add to output
        df_out += [
            self.model_attributes.array_to_df(np.sum(arr_frst_emissions_co2_fires, axis = 1), self.modvar_frst_emissions_co2_fires)
        ]


        ##  HARVESTED WOOD PRODUCTS

        # add to output
        df_out += self.project_harvested_wood_products(
            df_afolu_trajectories,
            vec_hh,
            vec_gdp,
            vec_rates_gdp,
            vec_rates_gdp_per_capita,
            dict_dims,
            n_projection_time_periods,
            projection_time_periods,
            self.dict_integration_variables_by_subsector
        )



        #####################
        #    AGRICULTURE    #
        #####################

        # get area of cropland
        field_crop_array = self.model_attributes.build_varlist(self.subsec_name_lndu, variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_crop])[0]
        vec_cropland_area = np.array(df_land_use[field_crop_array])

        # fraction of cropland represented by each crop
        arr_agrc_frac_cropland_area = self.check_cropland_fractions(df_agrc_frac_cropland, "calculated")
        arr_agrc_crop_area = (arr_agrc_frac_cropland_area.transpose()*vec_cropland_area.transpose()).transpose()
       
        # unit-corrected emission factors - ch4
        arr_agrc_ef_ch4 = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_ef_ch4,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        arr_agrc_ef_ch4 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_ch4,
            "area"
        )

        # biomass
        arr_agrc_ef_co2_biomass = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_ef_co2_biomass, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        arr_agrc_ef_co2_biomass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_co2_biomass,
            "area"
        )
        # biomass burning n2o is dealt with below in "soil management", where crop residues are calculated

        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(
                arr_agrc_crop_area*scalar_lndu_input_area_to_output_area, 
                self.modvar_agrc_area_crop
            ),
            self.model_attributes.array_to_df(
                arr_agrc_ef_ch4*arr_agrc_crop_area, 
                self.modvar_agrc_emissions_ch4_rice, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            self.model_attributes.array_to_df(
                arr_agrc_ef_co2_biomass*arr_agrc_crop_area, 
                self.modvar_agrc_emissions_co2_biomass, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]



        ###################
        #    LIVESTOCK    #
        ###################

        # get area of grassland/pastures
        field_lvst_graze_array = self.model_attributes.build_varlist(self.subsec_name_lndu, variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_grass])[0]
        vec_lvst_graze_area = np.array(df_land_use[field_lvst_graze_array])
        # estimate the total number of livestock that are raised - arr_lvst_pop is a direct output of project_integrated_land_use
        arr_lvst_total_weight = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_animal_weight,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )
        arr_lvst_total_animal_mass = arr_lvst_pop*arr_lvst_total_weight
        arr_lvst_aggregate_animal_mass = np.sum(arr_lvst_total_animal_mass, axis = 1)
        arr_lvst_aggregate_animal_mass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lvst_total_animal_mass,
            "mass"
        )
        # get the enteric fermentation emission factor
        arr_lvst_emissions_ch4_ef = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_ef_ch4_ef, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(arr_lvst_emissions_ch4_ef*arr_lvst_pop, self.modvar_lvst_emissions_ch4_ef),
            self.model_attributes.array_to_df(arr_lvst_pop, self.modvar_lvst_pop),
            self.model_attributes.array_to_df(arr_lvst_aggregate_animal_mass, self.modvar_lvst_total_animal_mass)
        ]


        ##  MANURE MANAGEMENT DATA

        # nitrogen and volative solids generated (passed to manure management--unitless, so they take the mass of modvar_lvst_animal_weight)
        arr_lvst_nitrogen = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_genfactor_nitrogen,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base"
        )
        arr_lvst_nitrogen *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")
        arr_lvst_volatile_solids = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_genfactor_volatile_solids,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base"
        )
        arr_lvst_volatile_solids *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")
        arr_lvst_b0 = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_b0_manure_ch4,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected_gas"
        )
        # get ratio of n to volatile solids
        arr_lvst_ratio_vs_to_n = arr_lvst_volatile_solids/arr_lvst_nitrogen


        #####################################
        #    LIVESTOCK MANURE MANAGEMENT    #
        #####################################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_lsmm_frac_manure = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lvst_mm_fractions,
            1,
            force_sum_equality = True,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_inen_frac_energy."
        )

        # get variables that can be indexed below 
        arr_lsmm_ef_direct_n2o = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_ef_direct_n2o,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )
        arr_lsmm_frac_lost_leaching = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_loss_leaching,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        arr_lsmm_frac_lost_volatilisation = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_loss_volatilisation,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        arr_lsmm_frac_used_for_fertilizer = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_n_available_used,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        arr_lsmm_mcf_by_pathway = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_mcf_by_pathway,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        arr_lsmm_n_from_bedding = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_n_from_bedding,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )
        arr_lsmm_n_from_codigestates = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_n_from_codigestates,
            expand_to_all_cats = True,  
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf)
        )
        arr_lsmm_rf_biogas = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_rf_biogas,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        vec_lsmm_frac_n_in_dung = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lvst_frac_exc_n_in_dung, 
            override_vector_for_single_mv_q = False,
            return_type = "array_base",
            var_bounds = (0, 1)
        )
        vec_lsmm_ratio_n2_to_n2o = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_lsmm_ratio_n2_to_n2o, 
            override_vector_for_single_mv_q = False,
            return_type = "array_base"
        )

        # soil EF4/EF5 from Table 11.3 - use average fractions from grasslands
        vec_soil_ef_ef4 = attr_lndu.get_key_value_index(self.cat_lndu_grass)
        vec_soil_ef_ef4 = arr_lndu_ef4_n_volatilisation[:, vec_soil_ef_ef4]
        vec_soil_ef_ef5 = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_ef5_n_leaching, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base"
        )

        # convert bedding/co-digestates to animal weight masses
        arr_lsmm_n_from_bedding *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_from_bedding,
            self.modvar_lvst_animal_weight,
            "mass"
        )

        # initialize output arrays
        arr_lsmm_biogas_recovered = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_ch4 = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_n2o = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_nitrogen_available = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        # initialize some aggregations
        vec_lsmm_nitrogen_to_other = 0.0
        vec_lsmm_nitrogen_to_fertilizer_dung = 0.0
        vec_lsmm_nitrogen_to_fertilizer_urine = 0.0
        vec_lsmm_nitrogen_to_pasture = 0.0
        # categories that allow for manure retrieval and use in fertilizer
        cats_lsmm_manure_retrieval = self.model_attributes.get_variable_categories(self.modvar_lsmm_frac_n_available_used)

        # loop over manure pathways to
        for var_lvst_mm_frac in self.modvar_list_lvst_mm_fractions:
            # get the current variable
            arr_lsmm_fracs_by_lvst = dict_arrs_lsmm_frac_manure[var_lvst_mm_frac]
            arr_lsmm_total_nitrogen_cur = arr_lvst_nitrogen*arr_lsmm_fracs_by_lvst
            # retrive the livestock management category
            cat_lsmm = clean_schema(self.model_attributes.get_variable_attribute(var_lvst_mm_frac, pycat_lsmm))
            index_cat_lsmm = attr_lsmm.get_key_value_index(cat_lsmm)


            ##  METHANE EMISSIONS

            # get MCF, b0, and total volatile solids - USE EQ. 10.23
            vec_lsmm_mcf_cur = arr_lsmm_mcf_by_pathway[:, index_cat_lsmm]
            arr_lsmm_emissions_ch4_cur = arr_lvst_b0*arr_lvst_volatile_solids*arr_lsmm_fracs_by_lvst
            arr_lsmm_emissions_ch4_cur = (arr_lsmm_emissions_ch4_cur.transpose()*vec_lsmm_mcf_cur).transpose()
            
            # get biogas recovery
            arr_lsmm_biogas_recovered_cur = (arr_lsmm_emissions_ch4_cur.transpose()*arr_lsmm_rf_biogas[:, index_cat_lsmm]).transpose()
            arr_lsmm_emissions_ch4_cur -= arr_lsmm_biogas_recovered_cur
            arr_lsmm_biogas_recovered[:, index_cat_lsmm] = np.sum(arr_lsmm_biogas_recovered_cur, axis = 1)
            
            # adjust
            arr_lsmm_emissions_ch4_cur *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")
            arr_lsmm_emission_ch4[:, index_cat_lsmm] = np.sum(arr_lsmm_emissions_ch4_cur, axis = 1)


            ##  NITROGEN EMISSIONS AND FERTILIZER AVAILABILITY

            # get total nitrogen deposited
            vec_lsmm_nitrogen_treated_cur = np.sum(arr_lsmm_total_nitrogen_cur, axis = 1)
            vec_lsmm_n_from_bedding = arr_lsmm_n_from_bedding[:, index_cat_lsmm]
            vec_lsmm_n_from_codigestates = arr_lsmm_n_from_codigestates[:, index_cat_lsmm]
            # get nitrogen from bedding per animal
            vec_lsmm_n_from_bedding *= np.sum(arr_lvst_pop*arr_lsmm_fracs_by_lvst, axis = 1)

            # get totals lost to different pathways
            vec_lsmm_frac_lost_direct = sf.vec_bounds((1 + vec_lsmm_ratio_n2_to_n2o)*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm], (0, 1))
            vec_lsmm_frac_lost_leaching = arr_lsmm_frac_lost_leaching[:, index_cat_lsmm]
            vec_lsmm_frac_lost_volatilisation = arr_lsmm_frac_lost_volatilisation[:, index_cat_lsmm]
            # apply the limiter, which prevents their total from exceeding 1
            vec_lsmm_frac_lost_direct, vec_lsmm_frac_lost_leaching, vec_lsmm_frac_lost_volatilisation = sf.vector_limiter(
                [
                    vec_lsmm_frac_lost_direct,
                    vec_lsmm_frac_lost_leaching,
                    vec_lsmm_frac_lost_volatilisation
                ],
                (0, 1)
            )
            vec_lsmm_frac_loss_ms = vec_lsmm_frac_lost_leaching + vec_lsmm_frac_lost_volatilisation + vec_lsmm_frac_lost_direct
            vec_lsmm_n_lost = vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates)*self.factor_n2on_to_n2o

            # 10.25 FOR DIRECT EMISSIONS
            arr_lsmm_emission_n2o[:, index_cat_lsmm] = vec_lsmm_n_lost*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm]
            # 10.28 FOR LOSSES DUE TO VOLATILISATION
            arr_lsmm_emission_n2o[:, index_cat_lsmm] += vec_lsmm_n_lost*vec_soil_ef_ef4*vec_lsmm_frac_lost_volatilisation
            # 10.29 FOR LOSSES DUE TO LEACHING
            arr_lsmm_emission_n2o[:, index_cat_lsmm] += vec_lsmm_n_lost*vec_soil_ef_ef5*vec_lsmm_frac_lost_leaching
            # BASED ON EQ. 10.34A in IPCC GNGHGI 2019R FOR NITROGEN AVAILABILITY - note: co-digestates are entered as an inflation factor
            vec_lsmm_nitrogen_available = (vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates))*(1 - vec_lsmm_frac_loss_ms) + vec_lsmm_n_from_bedding

            # check categories
            if cat_lsmm in cats_lsmm_manure_retrieval:
                if cat_lsmm == self.cat_lsmm_incineration:

                    ##  MANURE (VOLATILE SOLIDS) FOR INCINERATION:

                    vec_lsmm_volatile_solids_incinerated = np.sum(arr_lvst_volatile_solids*arr_lsmm_fracs_by_lvst, axis = 1)
                    vec_lsmm_volatile_solids_incinerated *= self.model_attributes.get_variable_unit_conversion_factor(
                        self.modvar_lvst_animal_weight,
                        self.modvar_lsmm_dung_incinerated,
                        "mass"
                    )

                    ##  N2O WORK

                    # if incinerating, send urine nitrogen to pasture
                    vec_lsmm_nitrogen_to_pasture += vec_lsmm_nitrogen_available*(1 - vec_lsmm_frac_n_in_dung)
                    # get incinerated N in dung - not used yet
                    vec_lsmm_nitrogen_to_incinerator = vec_lsmm_nitrogen_available*vec_lsmm_frac_n_in_dung
                    vec_lsmm_nitrogen_to_incinerator *= self.model_attributes.get_variable_unit_conversion_factor(
                        self.modvar_lvst_animal_weight,
                        self.modvar_lsmm_dung_incinerated,
                        "mass"
                    )

                    # add to output
                    df_out += [
                        self.model_attributes.array_to_df(vec_lsmm_volatile_solids_incinerated, self.modvar_lsmm_dung_incinerated)
                    ]

                else:
                    # account for fraction used for fertilizer
                    vec_lsmm_nitrogen_cur = vec_lsmm_nitrogen_available*arr_lsmm_frac_used_for_fertilizer[:, index_cat_lsmm]
                    vec_lsmm_nitrogen_to_other += vec_lsmm_nitrogen_available - vec_lsmm_nitrogen_cur
                    # add to total by animal and splits by dung/urea (used in Soil Management subsector)
                    arr_lsmm_nitrogen_available[:, index_cat_lsmm] += vec_lsmm_nitrogen_cur
                    vec_lsmm_nitrogen_to_fertilizer_dung += vec_lsmm_nitrogen_cur*vec_lsmm_frac_n_in_dung
                    vec_lsmm_nitrogen_to_fertilizer_urine += vec_lsmm_nitrogen_cur*(1 - vec_lsmm_frac_n_in_dung)

            elif cat_lsmm == self.cat_lsmm_pasture:
                vec_lsmm_nitrogen_to_pasture += vec_lsmm_nitrogen_available


        ##  UNITS CONVERSTION

        # biogas recovery
        arr_lsmm_biogas_recovered *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_recovered_biogas,
            "mass"
        )
        # total nitrogen available for fertilizer by pathway
        arr_lsmm_nitrogen_available *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer,
            "mass"
        )
        # total nitrogen available for other uses by pathway
        vec_lsmm_nitrogen_to_other *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_other_use,
            "mass"
        )
        # total nitrogen sent to pasture
        vec_lsmm_nitrogen_to_pasture *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_pastures,
            "mass"
        )
        # nitrogen available from dung/urea
        vec_lsmm_nitrogen_to_fertilizer_dung *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_lsmm_nitrogen_to_fertilizer_urine *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_urine,
            "mass"
        )
        # n2o emissions
        arr_lsmm_emission_n2o *= self.model_attributes.get_scalar(self.modvar_lsmm_emissions_n2o, "gas")
        arr_lsmm_emission_n2o *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")

        df_out += [
            self.model_attributes.array_to_df(
                arr_lsmm_emission_ch4, 
                self.modvar_lsmm_emissions_ch4
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_emission_n2o, 
                self.modvar_lsmm_emissions_n2o
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_pasture, 
                self.modvar_lsmm_n_to_pastures
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_nitrogen_available, 
                self.modvar_lsmm_n_to_fertilizer
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_other, 
                self.modvar_lsmm_n_to_other_use
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_fertilizer_dung, 
                self.modvar_lsmm_n_to_fertilizer_agg_dung
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_fertilizer_urine, 
                self.modvar_lsmm_n_to_fertilizer_agg_urine
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_biogas_recovered, 
                self.modvar_lsmm_recovered_biogas, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]




        #############################
        ###                       ###
        ###    SOIL MANAGEMENT    ###
        ###                       ###
        #############################

        # get inital demand for fertilizer N - start with area of land receiving fertilizer (grasslands and croplands)
        # put units in terms of modvar_lsmm_n_to_fertilizer_agg_dung
        #
        vec_soil_init_n_fertilizer_synthetic = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_fertuse_init_synthetic, False, "array_base")
        vec_soil_init_n_fertilizer_synthetic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_init_synthetic,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_soil_n_fertilizer_use_organic = vec_lsmm_nitrogen_to_fertilizer_urine
        vec_soil_n_fertilizer_use_organic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_urine,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_soil_n_fertilizer_use_organic += vec_lsmm_nitrogen_to_fertilizer_dung
        vec_soil_init_n_fertilizer_total = vec_soil_init_n_fertilizer_synthetic + vec_soil_n_fertilizer_use_organic
        
        # get land that's fertilized and use to project fertilizer demand - add in fraction of grassland that is pasture
        arr_lndu_frac_fertilized = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_frac_fertilized, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_land_use_with_pastures_as_grassland = arr_land_use.copy()
        arr_land_use_with_pastures_as_grassland[:, self.ind_lndu_grass] *= vec_lndu_frac_grassland_pasture
        vec_soil_area_fertilized = np.sum(arr_lndu_frac_fertilized*arr_land_use_with_pastures_as_grassland, axis = 1)
        vec_soil_demscalar_fertilizer = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_demscalar_fertilizer, False, "array_base", var_bounds = (0, np.inf))
        
        # estimate fertilizer demand
        vec_soil_n_fertilizer_use_total = np.concatenate([np.ones(1), np.cumprod(vec_soil_area_fertilized[1:]/vec_soil_area_fertilized[0:-1])])
        vec_soil_n_fertilizer_use_total *= vec_soil_demscalar_fertilizer
        vec_soil_n_fertilizer_use_total *= vec_soil_init_n_fertilizer_total[0]
        
        # estimate synthetic fertilizer demand - send extra manure back to pasture/paddock/range treatment flow
        vec_soil_n_fertilizer_use_synthetic = vec_soil_n_fertilizer_use_total - vec_soil_n_fertilizer_use_organic
        vec_soil_n_fertilizer_use_organic_to_pasture = sf.vec_bounds(vec_soil_n_fertilizer_use_synthetic, (0, np.inf))
        vec_soil_n_fertilizer_use_organic_to_pasture -= vec_soil_n_fertilizer_use_synthetic
        vec_soil_n_fertilizer_use_synthetic = sf.vec_bounds(vec_soil_n_fertilizer_use_synthetic, (0, np.inf))
        
        # split synthetic fertilizer use up
        vec_soil_frac_synthetic_fertilizer_urea = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_frac_synethic_fertilizer_urea,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        vec_soil_n_fertilizer_use_synthetic_urea = vec_soil_n_fertilizer_use_synthetic*vec_soil_frac_synthetic_fertilizer_urea
        vec_soil_n_fertilizer_use_synthetic_nonurea = vec_soil_n_fertilizer_use_synthetic - vec_soil_n_fertilizer_use_synthetic_urea
        
        # next, initialize the component n_inputs (N20_DIRECT-N from equation 11.1)
        arr_soil_n_inputs = 0.0
        arr_soil_area_by_drywet = 0.0
        arr_soil_ef1_organic = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef1_n_managed_soils_org_fert, True, "array_base", expand_to_all_cats = True)
        arr_soil_ef1_synthetic = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef1_n_managed_soils_syn_fert, True, "array_base", expand_to_all_cats = True)
        vec_soil_ef1_rice = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef1_n_managed_soils_rice, False, "array_base")
        
        # finally, get the emission factor for C in organic cultivated soils as part of soil carbon
        arr_soil_ef_c_organic_cultivated_soils = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef_c_organic_soils, True, "array_base", expand_to_all_cats = True)
        arr_soil_ef_c_organic_cultivated_soils *= self.model_attributes.get_scalar(
            self.modvar_soil_ef_c_organic_soils,
            "mass"
        )
        arr_soil_ef_c_organic_cultivated_soils /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_ef_c_organic_soils,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )


        ##############################################################
        #    N2O DIRECT - INPUT EMISSIONS (PT. 1 OF EQUATION 11.1)   #
        ##############################################################

        ##  SOME SHARED VARIABLES

        # get crop components of synthetic and organic fertilizers for ef1 (will overwrite rice)
        ind_rice = attr_agrc.get_key_value_index(self.cat_agrc_rice)
        # some variables
        arr_lndu_frac_mineral_soils = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_frac_mineral_soils, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
    
        arr_lndu_frac_organic_soils = 1 - arr_lndu_frac_mineral_soils
        vec_soil_area_crop_pasture = arr_land_use[:, [self.ind_lndu_crop, self.ind_lndu_grass]]
        vec_soil_area_crop_pasture[:, 1] *= vec_lndu_frac_grassland_pasture
        vec_soil_area_crop_pasture = np.sum(vec_soil_area_crop_pasture, axis = 1)
        

        ##  F_ON AND F_SN - SYNTHETIC FERTILIZERS AND ORGANIC AMENDMENTS

        # initialize some components
        dict_soil_fertilizer_application_by_climate_organic = {}
        dict_soil_fertilizer_application_by_climate_synthetic = {}
        vec_soil_n2odirectn_fon = 0.0
        vec_soil_n2odirectn_fsn = 0.0
        vec_soil_n2odirectn_fon_rice = 0.0
        vec_soil_n2odirectn_fsn_rice = 0.0

        # crop component
        for modvar in self.modvar_list_agrc_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            
            # get current factors
            arr_agrc_cur_wetdry_fertilized_crop = dict_arrs_agrc_frac_drywet[modvar]*arr_agrc_crop_area
            arr_agrc_cur_wetdry_fertilized_crop = arr_agrc_cur_wetdry_fertilized_crop.transpose()*arr_lndu_frac_fertilized[:, self.ind_lndu_crop]
            
            # get fraction of fertilized land represented by current area of cropland
            arr_soil_frac_cur_drywet_crop = (arr_agrc_cur_wetdry_fertilized_crop/vec_soil_area_fertilized)
            arr_soil_frac_cur_drywet_crop_organic = arr_soil_frac_cur_drywet_crop*vec_soil_n_fertilizer_use_organic
            arr_soil_frac_cur_drywet_crop_synthetic = arr_soil_frac_cur_drywet_crop*vec_soil_n_fertilizer_use_synthetic
            
            # update the dictionary for use later
            dict_soil_fertilizer_application_by_climate_organic.update({cat_soil: np.sum(arr_soil_frac_cur_drywet_crop_organic, axis = 0)})
            dict_soil_fertilizer_application_by_climate_synthetic.update({cat_soil: np.sum(arr_soil_frac_cur_drywet_crop_synthetic, axis = 0)})
            
            # get rice components
            vec_soil_n2odirectn_fon_rice += arr_soil_frac_cur_drywet_crop_organic[ind_rice, :]*vec_soil_ef1_rice
            vec_soil_n2odirectn_fsn_rice += arr_soil_frac_cur_drywet_crop_synthetic[ind_rice, :]*vec_soil_ef1_rice
            
            # remove rice and carry on
            arr_soil_frac_cur_drywet_crop_organic[ind_rice, :] = 0.0
            arr_soil_frac_cur_drywet_crop_synthetic[ind_rice, :] = 0.0
            vec_soil_n2odirectn_fon += np.sum(arr_soil_frac_cur_drywet_crop_organic, axis = 0)*arr_soil_ef1_organic[:, ind_soil]
            vec_soil_n2odirectn_fsn += np.sum(arr_soil_frac_cur_drywet_crop_synthetic, axis = 0)*arr_soil_ef1_synthetic[:, ind_soil]


        # grassland component
        for modvar in self.modvar_list_lndu_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            
            # get current factors
            vec_soil_cur_wetdry_fertilized_pstr = dict_arrs_lndu_frac_drywet[modvar]*arr_land_use_with_pastures_as_grassland
            vec_soil_cur_wetdry_fertilized_pstr = (vec_soil_cur_wetdry_fertilized_pstr*arr_lndu_frac_fertilized)[:, self.ind_lndu_grass]
            
            # get fraction of fertilized land represented by current area of cropland
            vec_soil_frac_cur_drywet_pstr = (vec_soil_cur_wetdry_fertilized_pstr/vec_soil_area_fertilized)
            vec_soil_n2odirectn_fon += (vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_organic)*arr_soil_ef1_organic[:, ind_soil]
            vec_soil_n2odirectn_fsn += (vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_synthetic)*arr_soil_ef1_synthetic[:, ind_soil]
            
            # update the dictionary for use later
            v_cur = dict_soil_fertilizer_application_by_climate_organic[cat_soil].copy()
            dict_soil_fertilizer_application_by_climate_organic.update({cat_soil: v_cur + vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_organic})
            v_cur = dict_soil_fertilizer_application_by_climate_synthetic[cat_soil].copy()
            dict_soil_fertilizer_application_by_climate_synthetic.update({cat_soil: v_cur + vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_synthetic})


        ##  F_CR - CROP RESIDUES

        arr_soil_yield = arr_agrc_yield*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yield,
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )
        arr_soil_crop_area = arr_agrc_crop_area*self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_regression_m_above_ground_residue,
            "area"
        )

        # get the regression information
        arr_agrc_regression_m = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_regression_m_above_ground_residue, True, "array_base", expand_to_all_cats = True)
        arr_agrc_regression_b = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_regression_b_above_ground_residue, True, "array_base", expand_to_all_cats = True)
        arr_agrc_regression_b *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_b_above_ground_residue,
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )
        arr_agrc_regression_b /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_b_above_ground_residue,
            self.modvar_agrc_regression_m_above_ground_residue,
            "area"
        )

        # get crop dry matter
        arr_agrc_crop_frac_dry_matter = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_frac_dry_matter_in_crop, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_agrc_crop_drymatter_per_unit = np.nan_to_num(arr_soil_yield/arr_soil_crop_area, 0.0, posinf = 0.0)
        arr_agrc_crop_drymatter_per_unit = arr_agrc_regression_m*arr_agrc_crop_drymatter_per_unit + arr_agrc_regression_b
        arr_agrc_crop_drymatter_above_ground = arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area
        
        # get fraction removed/burned
        dict_agrc_frac_residues_removed_burned = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_residues_removed_burned,
            1,
            force_sum_equality = False,
            msg_append = "Agriculture crop residue fractions by exceed 1. See definition of dict_agrc_frac_residues_removed_burned."
        )
        vec_agrc_frac_residue_burned = dict_agrc_frac_residues_removed_burned.get(self.modvar_agrc_frac_residues_burned).flatten()
        vec_agrc_frac_residue_removed = dict_agrc_frac_residues_removed_burned.get(self.modvar_agrc_frac_residues_removed).flatten()
        arr_agrc_combustion_factor = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_combustion_factor, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
       
        # get n available in above ground/below ground residues
        arr_agrc_n_content_ag_residues = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_n_content_of_above_ground_residues, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_agrc_n_content_bg_residues = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_n_content_of_below_ground_residues, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        
        # get total n HERE IS TOTAL N BURNED FROM CROP RESIDUE (in terms of modvar_agrc_regression_m_above_ground_residue)
        vec_agrc_total_n_residue_burned = np.sum(arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues, axis = 1)*vec_agrc_frac_residue_burned
        arr_agrc_total_n_residue_removed = (arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues).transpose()*vec_agrc_frac_residue_removed
        arr_agrc_total_n_above_ground_residues_burncomponent = (arr_agrc_crop_drymatter_above_ground*arr_agrc_combustion_factor*arr_agrc_n_content_ag_residues).transpose()*vec_agrc_frac_residue_burned
        arr_agrc_total_n_above_ground_residues = (arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues).transpose() - arr_agrc_total_n_residue_removed - arr_agrc_total_n_above_ground_residues_burncomponent
        
        # get dry/wet and rice residuces
        vec_agrc_total_n_above_ground_residues_rice = arr_agrc_total_n_above_ground_residues[ind_rice, :].copy()
        arr_agrc_total_n_above_ground_residues[ind_rice, :] = 0
        vec_agrc_total_n_above_ground_residues_dry = np.sum(arr_agrc_total_n_above_ground_residues.transpose()*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_dry], axis = 1)
        vec_agrc_total_n_above_ground_residues_wet = np.sum(arr_agrc_total_n_above_ground_residues.transpose()*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_wet], axis = 1)
        
        # move to below ground and get total biomass (used for biomass burning)
        arr_agrc_ratio_bg_biomass_to_ag_biomass = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass, True, "array_base", expand_to_all_cats = True)
        arr_agrc_bg_biomass = (arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area + arr_soil_yield)*arr_agrc_ratio_bg_biomass_to_ag_biomass
        vec_agrc_crop_residue_biomass = np.sum(arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area + arr_agrc_bg_biomass, axis = 1)
       
        # get n from below ground residues
        arr_agrc_total_n_below_ground_residues = arr_agrc_bg_biomass*arr_agrc_n_content_bg_residues
        vec_agrc_total_n_below_ground_residues_rice = arr_agrc_total_n_below_ground_residues[:, ind_rice].copy()
        arr_agrc_total_n_below_ground_residues[:, ind_rice] = 0
        vec_agrc_total_n_below_ground_residues_dry = np.sum(arr_agrc_total_n_below_ground_residues*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_dry], axis = 1)
        vec_agrc_total_n_below_ground_residues_wet = np.sum(arr_agrc_total_n_below_ground_residues*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_wet], axis = 1)
        
        # get total crop residue and conver to units of F_ON and F_SN
        scalar_soil_residue_to_fertilizer_equivalent = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_m_above_ground_residue,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_agrc_total_n_residue_dry = (vec_agrc_total_n_above_ground_residues_dry + vec_agrc_total_n_below_ground_residues_dry)*scalar_soil_residue_to_fertilizer_equivalent
        vec_agrc_total_n_residue_wet = (vec_agrc_total_n_above_ground_residues_wet + vec_agrc_total_n_below_ground_residues_wet)*scalar_soil_residue_to_fertilizer_equivalent
        vec_agrc_total_n_residue_rice = (vec_agrc_total_n_above_ground_residues_rice + vec_agrc_total_n_below_ground_residues_rice)*scalar_soil_residue_to_fertilizer_equivalent
        
        # finally, get ef1 component
        dict_agrc_modvar_to_n_residue = {
            self.modvar_agrc_frac_dry: vec_agrc_total_n_residue_dry,
            self.modvar_agrc_frac_wet: vec_agrc_total_n_residue_wet
        }
        vec_soil_n2odirectn_fcr = 0.0
        vec_soil_n2odirectn_fcr_rice = vec_agrc_total_n_residue_rice*vec_soil_ef1_rice

        # loop over dry/wet
        for modvar in self.modvar_list_agrc_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_n2odirectn_fcr += dict_agrc_modvar_to_n_residue[modvar]*arr_soil_ef1_organic[:, ind_soil]


        ##  ADD IN BIOMASS BURNING (FROM EQ. 2.27 & V4 SECTION 5.3.4.1)

        # dimensionless, buin terms of modvar_agrc_regression_m_above_ground_residue
        vec_agrc_ef_ch4_biomass_burning = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_ef_ch4_burning, 
            return_type = "array_units_corrected"
        )
        vec_agrc_ef_n2o_biomass_burning = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_ef_n2o_burning, 
            return_type = "array_units_corrected"
        )
        vec_agrc_crop_residue_burned = vec_agrc_crop_residue_biomass*vec_agrc_frac_residue_burned

        # get average combustion factor, or fraction of crops burned by fire
        vec_agrc_avg_combustion_factor = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_agrc_combustion_factor, 
            expand_to_all_cats = False,
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        cats_agrc_avg_combustion_factor = self.model_attributes.get_variable_categories(self.modvar_agrc_combustion_factor)
        inds_agrc_avg_combustion_factor = [attr_agrc.get_key_value_index(x) for x in cats_agrc_avg_combustion_factor]
        vec_agrc_avg_combustion_factor = np.sum(vec_agrc_avg_combustion_factor*arr_agrc_crop_area[:, inds_agrc_avg_combustion_factor], axis = 1)
        vec_agrc_avg_combustion_factor /= np.sum(arr_agrc_crop_area[:, inds_agrc_avg_combustion_factor], axis = 1)
        # multiply by combustion factor to get final mass of crops burned
        vec_agrc_crop_residue_burned *= vec_agrc_avg_combustion_factor

        # get estimate of emissions of ch4
        vec_agrc_emissions_ch4_biomass_burning = vec_agrc_crop_residue_burned*vec_agrc_ef_ch4_biomass_burning
        vec_agrc_emissions_ch4_biomass_burning *= self.model_attributes.get_scalar(
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )

        # get estimate of emissions of n2o
        vec_agrc_emissions_n2o_biomass_burning = vec_agrc_crop_residue_burned*vec_agrc_ef_n2o_biomass_burning
        vec_agrc_emissions_n2o_biomass_burning *= self.model_attributes.get_scalar(
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )
        
        # add to output
        df_out += [
            # CH4 FROM BIOMASS BURNING
            self.model_attributes.array_to_df(
                vec_agrc_emissions_ch4_biomass_burning, 
                self.modvar_agrc_emissions_ch4_biomass_burning
            ),
            # N2O EMISSIONS FROM BIOMASS BURNING
            self.model_attributes.array_to_df(
                vec_agrc_emissions_n2o_biomass_burning, 
                self.modvar_agrc_emissions_n2o_biomass_burning
            )
        ]


        ##  F_SOM AND AGRICULTURAL SOIL CARBON

        # get soil management factors
        arr_lndu_factor_soil_management, arr_lndu_area_improved = self.get_lndu_soil_soc_factors(
            df_afolu_trajectories,
            arr_land_use,
            arr_agrc_frac_cropland,
            dict_agrc_frac_residues_removed_burned,
            vec_lndu_frac_grassland_pasture,
        )
        self.arr_lndu_factor_soil_management = arr_lndu_factor_soil_management
        df_out += [
            self.model_attributes.array_to_df(
                arr_lndu_area_improved*scalar_lndu_input_area_to_output_area,
                self.modvar_lndu_area_improved,
                reduce_from_all_cats_to_specified_cats = True,
            )
        ]

        # get carbon stocks and ratio of c to n - conver to self.modvar_lsmm_n_to_fertilizer_agg_dung units for N2O/EF1
        arr_lndu_factor_soil_carbon = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_carbon,
            return_type = "array_base",
            expand_to_all_cats = True
        )
        
        arr_soil_organic_c_stocks = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_soil_organic_c_stocks,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            expand_to_all_cats = True
        )
        arr_soil_organic_c_stocks *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_organic_c_stocks,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        arr_soil_organic_c_stocks /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_organic_c_stocks,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )

        # get some other factors
        vec_soil_ratio_c_to_n_soil_organic_matter = self.model_attributes.get_standard_variables(
            df_afolu_trajectories,
            self.modvar_soil_ratio_c_to_n_soil_organic_matter,
            return_type = "array_base"
        )


        # get arrays of SOC conversion per area by land use
        arrs_lndu_soc_conversion_factors, vec_soil_ef1_soc_est = self.get_mineral_soc_change_matrices(
            arr_agrc_crop_area,
            arr_land_use,
            arr_lndu_factor_soil_carbon,
            arr_lndu_factor_soil_management,
            arr_lndu_frac_mineral_soils,
            arr_soil_ef1_organic,
            arr_soil_organic_c_stocks,
            vec_lndu_frac_grassland_pasture,
            vec_soil_area_crop_pasture,
            dict_arrs_agrc_frac_drywet,
            dict_arrs_frst_frac_temptrop,
            dict_arrs_lndu_frac_drywet
        )
        self.arrs_lndu_soc_conversion_factors = arrs_lndu_soc_conversion_factors
        vec_soil_delta_soc_mineral = self.calculate_soc_stock_change_with_time_dependence(
            arrs_lndu_land_conv,
            arrs_lndu_soc_conversion_factors,
            20# get from config HEREHERE
        )
        
        # initialize organic SOC, then loop over tropical/temperate cropland to get soil carbon for organic drained soils
        vec_soil_emission_co2_soil_carbon_organic = 0.0
        for modvar in self.modvar_list_agrc_frac_temptrop:
            # get appropriate soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            # get SOC totals and integrate land-use specific mineral fractions
            arr_soil_soc_crop_temptrop_cur = (arr_agrc_crop_area*dict_arrs_agrc_frac_temptrop[modvar]).transpose()
            arr_soil_soc_crop_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]
            vec_soil_emission_co2_soil_carbon_organic += np.sum(arr_soil_soc_crop_temptrop_cur*arr_soil_ef_c_organic_cultivated_soils[:, ind_soil], axis = 0)

        """
        Alternate approach to calculating SOC stock changes
        # calculate the change in soil carbon year over year for all and for mineral
        vec_soil_delta_soc = self.calculate_ipcc_soc_deltas(vec_soil_soc_total, 2)
        vec_soil_delta_soc_mineral = self.calculate_ipcc_soc_deltas(vec_soil_soc_total_mineral, 2)
        vec_soil_ratio_c_to_n_soil_organic_matter = vec_soil_ratio_c_to_n_soil_organic_matter
        """;

        # calculate FSOM from fraction mineral
        vec_soil_n2odirectn_fsom = -(sf.vec_bounds(vec_soil_delta_soc_mineral, (-np.inf, 0))/vec_soil_ratio_c_to_n_soil_organic_matter)*vec_soil_ef1_soc_est
        vec_soil_emission_co2_soil_carbon_mineral = -self.factor_c_to_co2*vec_soil_delta_soc_mineral
        vec_soil_emission_co2_soil_carbon_mineral *= self.model_attributes.get_scalar(self.modvar_lsmm_n_to_fertilizer_agg_dung, "mass")
        vec_soil_emission_co2_soil_carbon_mineral *= self.model_attributes.get_gwp("co2")
        # get soil carbon from organic drained soils
        vec_soil_emission_co2_soil_carbon_organic *= self.factor_c_to_co2*self.model_attributes.get_gwp("co2")


        ##  FINAL EF1 COMPONENTS

        # different tablulations (totals will run across EF1, EF2, EF3, EF4, and EF5)
        vec_soil_n2on_direct_input = vec_soil_n2odirectn_fon + vec_soil_n2odirectn_fon_rice + vec_soil_n2odirectn_fsn + vec_soil_n2odirectn_fsn_rice + vec_soil_n2odirectn_fcr + vec_soil_n2odirectn_fcr_rice + vec_soil_n2odirectn_fsom
        vec_soil_emission_n2o_crop_residue = vec_soil_n2odirectn_fcr + vec_soil_n2odirectn_fcr_rice
        vec_soil_emission_n2o_fertilizer = vec_soil_n2odirectn_fon + vec_soil_n2odirectn_fon_rice + vec_soil_n2odirectn_fsn + vec_soil_n2odirectn_fsn_rice
        vec_soil_emission_n2o_mineral_soils = vec_soil_n2odirectn_fsom



        #####################################################################
        #    N2O DIRECT - ORGANIC SOIL EMISSIONS (PT. 2 OF EQUATION 11.1)   #
        #####################################################################

        # get the emission factor variable
        arr_soil_ef2 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef2_n_organic_soils, True, "array_base", expand_to_all_cats = True)
        arr_soil_ef2 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_ef2_n_organic_soils,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        arr_soil_ef2 /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_ef2_n_organic_soils,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        vec_soil_n2on_direct_organic = 0.0
        
        # loop over dry/wet to estimate carbon stocks in crops
        for modvar in self.modvar_list_agrc_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_crop_temptrop_cur = np.sum(arr_agrc_crop_area*dict_arrs_agrc_frac_temptrop[modvar], axis = 1)
            vec_soil_crop_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]*arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += vec_soil_crop_temptrop_cur
        
        # loop over dry/wet to estimate carbon stocks in grassland
        for modvar in self.modvar_list_lndu_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_pstr_temptrop_cur = (arr_land_use*dict_arrs_lndu_frac_temptrop[modvar])[:, self.ind_lndu_grass]
            vec_soil_pstr_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_grass]*arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += vec_soil_pstr_temptrop_cur
        
        # loop over tropical/temperate NP/temperate NR
        for modvar in self.modvar_list_frst_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            # get land use category for soil carbon facto
            cats_lndu = [clean_schema(x) for x in self.model_attributes.get_ordered_category_attribute(self.subsec_name_frst, pycat_lndu)]
            inds_lndu = [attr_lndu.get_key_value_index(x) for x in cats_lndu]
            arr_soil_frst_temptrop_cur = np.sum(arr_area_frst*dict_arrs_frst_frac_temptrop[modvar]*arr_lndu_frac_organic_soils[:, inds_lndu], axis = 1)
            arr_soil_frst_temptrop_cur *= arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += arr_soil_frst_temptrop_cur

        # initialize output emission vector
        vec_soil_emission_n2o_organic_soils = vec_soil_n2on_direct_organic



        ####################################################################
        #    N2O DIRECT - PASTURE/RANGE/PADDOCK (PT. 3 OF EQUATION 11.1)   #
        ####################################################################

        #
        arr_soil_ef3 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef3_n_prp, True, "array_base", expand_to_all_cats = True)
        vec_lsmm_nitrogen_to_pasture *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_pastures,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        # loop over dry/wet for EF3, pasture, range, and paddock
        vec_soil_n2on_direct_prp = 0.0
        dict_soil_ppr_n_by_climate = {}
        for modvar in self.modvar_list_lndu_frac_drywet:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_frac_pstr_drywet_cur = (arr_land_use*dict_arrs_lndu_frac_drywet[modvar])[:, self.ind_lndu_grass]/arr_land_use[:, self.ind_lndu_grass]
            # add component to EF1 estimate for F_SOM
            vec_soil_prp_cur = (vec_lsmm_nitrogen_to_pasture + vec_soil_n_fertilizer_use_organic_to_pasture)*vec_soil_frac_pstr_drywet_cur
            vec_soil_n2on_direct_prp += vec_soil_prp_cur*arr_soil_ef3[:, ind_soil]
            dict_soil_ppr_n_by_climate.update({cat_soil: vec_soil_prp_cur})

        # initialize output emissions
        vec_soil_emission_n2o_ppr = vec_soil_n2on_direct_prp



        ###########################################################
        #    N2O INDIRECT - VOLATISED EMISSIONS (EQUATION 11.9)   #
        ###########################################################

        # get volatilisation vars
        vec_soil_frac_gasf_non_urea = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        vec_soil_frac_gasf_urea = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_sn_urea,
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        vec_soil_frac_gasm = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_on, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        arr_soil_ef4 = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_ef4_n_volatilisation, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        # loop over dry/wet
        vec_soil_n2on_indirect_volatilisation = 0.0
        vec_soil_n2on_indirect_volatilisation_gasf = 0.0
        vec_soil_n2on_indirect_volatilisation_gasm_on = 0.0
        vec_soil_n2on_indirect_volatilisation_gasm_ppr = 0.0

        for modvar in self.modvar_list_lndu_frac_drywet:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            # GASF component--synthetic by urea/non-urea
            vec_soil_fert_sn_cur_non_urea = dict_soil_fertilizer_application_by_climate_synthetic[cat_soil].copy()
            vec_soil_fert_sn_cur_urea = vec_soil_fert_sn_cur_non_urea*vec_soil_frac_synthetic_fertilizer_urea
            vec_soil_fert_sn_cur_non_urea -= vec_soil_fert_sn_cur_urea
            vec_soil_component_gasf_cur = vec_soil_fert_sn_cur_non_urea*vec_soil_frac_gasf_non_urea + vec_soil_fert_sn_cur_urea*vec_soil_frac_gasf_urea
            vec_soil_component_gasf_cur *= arr_soil_ef4[:, ind_soil]

            # GASM component--organic
            vec_soil_component_gasm_on_cur = dict_soil_fertilizer_application_by_climate_organic[cat_soil]*vec_soil_frac_gasm*arr_soil_ef4[:, ind_soil]
            vec_soil_component_gasm_ppr_cur = dict_soil_ppr_n_by_climate[cat_soil]*vec_soil_frac_gasm*arr_soil_ef4[:, ind_soil]

            # aggregates
            vec_soil_n2on_indirect_volatilisation_gasf += vec_soil_component_gasf_cur
            vec_soil_n2on_indirect_volatilisation_gasm_on += vec_soil_component_gasm_on_cur
            vec_soil_n2on_indirect_volatilisation_gasm_ppr += vec_soil_component_gasm_ppr_cur
            vec_soil_n2on_indirect_volatilisation += vec_soil_component_gasf_cur + vec_soil_component_gasm_on_cur + vec_soil_component_gasm_ppr_cur

        # update emissions
        vec_soil_emission_n2o_fertilizer += vec_soil_n2on_indirect_volatilisation_gasf + vec_soil_n2on_indirect_volatilisation_gasm_on
        vec_soil_emission_n2o_ppr += vec_soil_n2on_indirect_volatilisation_gasm_ppr



        ###########################################################
        #    N2O INDIRECT - LEACHING EMISSIONS (EQUATION 11.10)   #
        ###########################################################

        # get some components
        vec_soil_ef5 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef5_n_leaching, False, "array_base")
        vec_soil_frac_leaching = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_frac_n_lost_leaching, False, "array_base", var_bounds = (0, 1))
        
        # add up sources of N
        vec_soil_n2on_indirect_leaching_fert = vec_soil_n_fertilizer_use_organic + vec_soil_n_fertilizer_use_synthetic
        vec_soil_n2on_indirect_leaching_fert *= vec_soil_frac_leaching*vec_soil_ef5
        vec_soil_n2on_indirect_leaching_ppr = vec_lsmm_nitrogen_to_pasture + vec_soil_n_fertilizer_use_organic_to_pasture
        vec_soil_n2on_indirect_leaching_ppr *= vec_soil_frac_leaching*vec_soil_ef5
        vec_soil_n2on_indirect_leaching_cr = vec_agrc_total_n_residue_dry + vec_agrc_total_n_residue_rice + vec_agrc_total_n_residue_wet
        vec_soil_n2on_indirect_leaching_cr *= vec_soil_frac_leaching*vec_soil_ef5
        vec_soil_n2on_indirect_leaching_mineral_soils = vec_soil_delta_soc_mineral/vec_soil_ratio_c_to_n_soil_organic_matter
        vec_soil_n2on_indirect_leaching_mineral_soils *= vec_soil_frac_leaching*vec_soil_ef5
        
        # build aggregate emissions
        vec_soil_n2on_indirect_leaching = (vec_soil_n2on_indirect_leaching_fert + vec_soil_n2on_indirect_leaching_ppr + vec_soil_n2on_indirect_leaching_cr + vec_soil_n2on_indirect_leaching_mineral_soils)
        vec_soil_emission_n2o_crop_residue += vec_soil_n2on_indirect_leaching_cr
        vec_soil_emission_n2o_fertilizer += vec_soil_n2on_indirect_leaching_fert
        vec_soil_emission_n2o_mineral_soils += vec_soil_n2on_indirect_leaching_mineral_soils
        vec_soil_emission_n2o_ppr += vec_soil_n2on_indirect_leaching_ppr

        # get fertilizer use totals
        vec_soil_n_fertilizer_use_organic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_fertuse_final_organic,
            "mass"
        )
        vec_soil_n_fertilizer_use_synthetic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_fertuse_final_synthetic,
            "mass"
        )
        # total
        vec_soil_n_fertilizer_use_total = vec_soil_n_fertilizer_use_organic*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_final_organic,
            self.modvar_soil_fertuse_final_total,
            "mass"
        )
        vec_soil_n_fertilizer_use_total += vec_soil_n_fertilizer_use_synthetic*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_final_synthetic,
            self.modvar_soil_fertuse_final_total,
            "mass"
        )

        #####################################################
        #    SUMMARIZE N2O EMISSIONS AS DIRECT + INDIRECT   #
        #####################################################

        scalar_n2on_to_emission_out = self.factor_n2on_to_n2o*self.model_attributes.get_scalar(self.modvar_lsmm_n_to_fertilizer_agg_dung, "mass")
        scalar_n2on_to_emission_out *= self.model_attributes.get_gwp("n2o")
        
        # build emissions outputs
        df_out += [
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_crop_residue*scalar_n2on_to_emission_out, 
                self.modvar_agrc_emissions_n2o_crop_residues
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_soil_carbon_organic, 
                self.modvar_agrc_emissions_co2_soil_carbon_organic
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_soil_carbon_mineral, 
                self.modvar_soil_emissions_co2_soil_carbon_mineral
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_fertilizer*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_fertilizer
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_mineral_soils*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_mineral_soils
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_organic_soils*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_organic_soils
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_ppr*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_ppr
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_organic, 
                self.modvar_soil_fertuse_final_organic
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_synthetic, 
                self.modvar_soil_fertuse_final_synthetic
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_total, 
                self.modvar_soil_fertuse_final_total
            )
        ]



        #####################################################
        #    CO2 EMISSIONS FROM LIMING + UREA APPLICATION   #
        #####################################################

        ##  LIMING

        # use land that's fertilized to project lime demand
        vec_soil_demscalar_liming = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_demscalar_liming, 
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )
        vec_soil_lime_init_dolomite = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_qtyinit_liming_dolomite,
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )
        vec_soil_lime_init_limestone = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_qtyinit_liming_limestone,
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )

        # get emission factors
        vec_soil_ef_liming_dolomite = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_liming_dolomite,
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )
        vec_soil_ef_liming_limestone = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_liming_limestone,
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )

        # write in terms of dolomite
        vec_soil_lime_init_limestone *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_qtyinit_liming_limestone,
            self.modvar_soil_qtyinit_liming_dolomite,
            "mass"
        )

        # estimate liming demand using the area of land that's fertilized
        vec_soil_lime_use_growth_rate = np.concatenate([np.ones(1), np.cumprod(vec_soil_area_fertilized[1:]/vec_soil_area_fertilized[0:-1])])
        vec_soil_lime_use_growth_rate *= vec_soil_demscalar_liming
        vec_soil_lime_use_dolomite = vec_soil_lime_init_dolomite[0]*vec_soil_lime_use_growth_rate
        vec_soil_lime_use_limestone = vec_soil_lime_init_limestone[0]*vec_soil_lime_use_growth_rate
       
        # get output emissions
        vec_soil_emission_co2_lime_use = vec_soil_lime_use_dolomite*vec_soil_ef_liming_dolomite + vec_soil_lime_use_limestone*vec_soil_ef_liming_limestone
        vec_soil_emission_co2_lime_use *= self.model_attributes.get_scalar(
            self.modvar_soil_qtyinit_liming_dolomite,
            "mass"
        )
        vec_soil_emission_co2_lime_use *= self.factor_c_to_co2
        vec_soil_emission_co2_lime_use *= self.model_attributes.get_gwp("co2")
        
        # total lime applied
        vec_soil_lime_use_total = vec_soil_lime_use_limestone + vec_soil_lime_use_dolomite
        vec_soil_lime_use_total *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_qtyinit_liming_dolomite,
            self.modvar_soil_limeuse_total,
            "mass"
        )
         

        ##  UREA

        vec_soil_ef_urea = self.model_attributes.get_standard_variables(
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_urea, 
            return_type = "array_base", 
            var_bounds = (0, np.inf)
        )
        vec_soil_emission_co2_urea_use = vec_soil_ef_urea*vec_soil_n_fertilizer_use_synthetic_urea
        vec_soil_emission_co2_urea_use *= self.model_attributes.get_scalar(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_soil_emission_co2_urea_use *= self.factor_c_to_co2
        vec_soil_emission_co2_urea_use *= self.model_attributes.get_gwp("co2")
        
        # get total urea applied (based on synthetic fertilizer, which was in terms of modvar_lsmm_n_to_fertilizer_agg_dung)
        vec_soil_n_fertilizer_use_synthetic_urea *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_ureause_total,
            "mass"
        )
        

        # add to output
        df_out += [
            # CO2 EMISSIONS FROM LIMING
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_lime_use, 
                self.modvar_soil_emissions_co2_lime
            ),
            # CO2 EMISSIONS FROM UREA
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_urea_use, 
                self.modvar_soil_emissions_co2_urea
            ),
            # TOTAL LIME USE
            self.model_attributes.array_to_df(
                vec_soil_lime_use_total, 
                self.modvar_soil_limeuse_total
            ),
            # TOTAL UREA USE
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_synthetic_urea, 
                self.modvar_soil_ureause_total
            )
        ]






        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)

        if passthrough_tmp is None:
            return df_out
        else:
            return df_out, passthrough_tmp
