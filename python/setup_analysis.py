from attribute_table import AttributeTable
from model_attributes import *
import numpy as np
import os, os.path
import pandas as pd
import support_functions as sf


##  SETUP DIRECTORIES AND KEY FILES

# setup some
dir_py = os.path.dirname(os.path.realpath(__file__))
dir_proj = os.path.dirname(dir_py)
fp_config = os.path.join(dir_proj, "sisepuede.config")
# key subdirectories for the project
dir_jl = sf.check_path(os.path.join(dir_proj, "julia"), False)
dir_out = sf.check_path(os.path.join(dir_proj, "out"), True)
dir_ref = sf.check_path(os.path.join(dir_proj, "ref"), False)
dir_ref_batch_data = sf.check_path(os.path.join(dir_ref, "batch_data_generation"), False)
dir_ref_data_crosswalks = sf.check_path(os.path.join(dir_ref, "data_crosswalks"), False)
dir_ref_nemo = sf.check_path(os.path.join(dir_ref, "nemo_mod"), False)
dir_tmp = sf.check_path(os.path.join(dir_proj, "tmp"), True)
# attribute tables and readthedocs NOTE HEREHERE replace docs_new with docs when back
dir_docs = sf.check_path(os.path.join(os.path.dirname(dir_py), "docs", "source"), False)
dir_attribute_tables = sf.check_path(os.path.join(dir_docs, "csvs"), False)
# get model attributes
model_attributes = ModelAttributes(dir_attribute_tables, fp_config)


##  INGESTION DATA STRUCTURE (DEPENDS ON ATTRIBUTES)

dir_ingestion = sf.check_path(os.path.join(dir_ref, "ingestion"), True)
# storage for parameter sheets of calibrated parameters
dir_parameters_calibrated = sf.check_path(os.path.join(dir_ingestion, "calibrated"), True)
# sheets used to demonstrate the structure of parameter input tables
dir_parameters_demo = sf.check_path(os.path.join(dir_ingestion, "demo"), True)
# sheets with raw, best-guess uncalibrated parameters by country
dir_parameters_uncalibrated = sf.check_path(os.path.join(dir_ingestion, "uncalibrated"), True)


##  DEVELOP SOME FILE PATHS

# key outputs for analysis run
fp_csv_default_single_run_out = os.path.join(dir_out, "single_run_output.csv")
fp_csv_tmp_inputs = os.path.join(dir_tmp, "temporary_full_inputs.csv")
# nemo mod input files - specify required, run checks
required_tables_nemomod = [
    model_attributes.table_nemomod_capacity_factor,
    model_attributes.table_nemomod_specified_demand_profile
]
dict_fp_csv_nemomod = {}
for table in required_tables_nemomod:
    fp_out = sf.check_path(os.path.join(dir_ref_nemo, f"{table}.csv"), False)
    dict_fp_csv_nemomod.update({table: fp_out})
# SQLite Database location
fp_sqlite_nemomod_db_tmp = os.path.join(dir_tmp, "nemomod_intermediate_database.sqlite")


##  BATCH DATA GENERATION DIRECTORIES AND FILES

##  DIRECTORIES
dir_rbd_afolu_exports_imports = sf.check_path(os.path.join(dir_ref_batch_data, "afolu_imports_exports"), True)
dir_rbd_afolu_land_use_efs = sf.check_path(os.path.join(dir_ref_batch_data, "afolu_land_use_emission_factors"), True)
dir_rbd_agrc_initial_cropland_fracs = sf.check_path(os.path.join(dir_ref_batch_data, "agrc_initial_cropland_fracs"), True)
dir_rbd_baseline_transition_probs = sf.check_path(os.path.join(dir_ref_batch_data, "baseline_transition_probability_estimates"), True)
dir_rbd_energy = sf.check_path(os.path.join(dir_ref_batch_data, "non_electric_energy_inputs"), True)
dir_rbd_generic = sf.check_path(os.path.join(dir_ref_batch_data, "generic"), True)
dir_rbd_ind_cement_clinker_fraction = sf.check_path(os.path.join(dir_ref_batch_data, "ippu_cement_clinker_fraction"), True)
dir_rbd_ind_efs_fcs_ippu = sf.check_path(os.path.join(dir_ref_batch_data, "ippu_emission_factors_fcs"), True)
dir_rbd_ind_prod_components = sf.check_path(os.path.join(dir_ref_batch_data, "industrial_production_components"), True)
dir_rbd_kcc = sf.check_path(os.path.join(dir_ref_batch_data, "koppen_climate_classifications"), True)
dir_rbd_soc = sf.check_path(os.path.join(dir_ref_batch_data, "soil_grids_soil_organic_carbon"), True)
dir_rbd_tillage = sf.check_path(os.path.join(dir_ref_batch_data, "afolu_tillage"), True)
dir_rbd_nemomod_energy_inputs = sf.check_path(os.path.join(dir_ref_batch_data, "nemomod_energy_inputs"), True)


##  GENERIC

# batch data field index
fp_csv_batch_data_field_index = os.path.join(dir_ref_batch_data, "batch_data_field_index.csv")

# data crosswalks
fp_csv_cw_fao_crops = os.path.join(dir_ref_data_crosswalks, "fao_crop_categories.csv")
fp_csv_cw_fao_product_demand_categories_for_ie = os.path.join(dir_ref_data_crosswalks, "fao_product_demand_categories_for_import_export.csv")

# other generic data (used across sectors or for regions/attributes)
fp_csv_countries_by_iso = os.path.join(dir_rbd_generic, "countries_by_iso.csv")
fp_csv_population_centroids_by_iso = os.path.join(dir_rbd_generic, "population_centroids_by_iso.csv")


##  AFOLU

# files for initial cropland fractions
fp_csv_agrc_initial_cropland_fracs = os.path.join(dir_rbd_agrc_initial_cropland_fracs, "initial_cropland_fractions.csv")

# files for afolu imports/exports
fp_csv_afolu_import_exports = os.path.join(dir_rbd_afolu_exports_imports, "afolu_import_(ofdem)_export_(ofprod)_fractions.csv")
fp_csv_afolu_import_export_estimates = os.path.join(dir_rbd_afolu_exports_imports, "afolu_import_fracs_and_export_totals.csv")


# files for Koppen Climate Classification
fp_csv_kcc_cells_merged_to_country = os.path.join(dir_rbd_kcc, "kcc_cells_merged_to_country.csv")
fp_csv_kcc_cell_counts_by_country_kcc = os.path.join(dir_rbd_kcc, "kcc_cell_counts_by_country.csv")
fp_csv_climate_fields_by_country_simple = os.path.join(dir_rbd_kcc, "climate_fields_by_country.csv")

# files containing land use emission factors
fp_csv_lndu_ef_forest_sequestration_co2 = os.path.join(dir_rbd_afolu_land_use_efs, "lndu_ef_forest_sequestration_co2.csv")
fp_csv_lndu_ef_conversion_co2 = os.path.join(dir_rbd_afolu_land_use_efs, "lndu_ef_conversion_co2.csv")

# files for Soil Organic Carbon by country from SoilGrids
fp_csv_soc_cells_merged_to_country = os.path.join(dir_rbd_soc, "scc_cells_merged_to_country.csv")
fp_csv_soc_average_soc_by_country = os.path.join(dir_rbd_soc, "soc_average_soc_by_country.csv")
fp_csv_soc_fields_by_country_simple = os.path.join(dir_rbd_soc, "soc_fields_by_country.csv")

# files for no till (conservation ag) fractions from Kassam et al. 2018
fp_csv_frac_no_till = os.path.join(dir_rbd_tillage, "agrc_frac_no_till.csv")

# files for afolu transition probabilities
fp_csv_transition_probability_estimation_annual = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_and_year.csv")
fp_csv_transition_probability_estimation_annual_copernicus = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_and_year_copernicus.csv")
fp_csv_transition_probability_estimation_annual_copernicus_adjusted = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_and_year_copernicus_adjusted.csv")
fp_csv_transition_probability_estimation_mean = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean.csv")
fp_csv_transition_probability_estimation_mean_recent = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_recent_only.csv")
fpt_csv_transition_probability_estimation_mean_with_growth = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_with_target_growth-%s.csv")
fpt_pkl_transition_probability_estimation_mean_with_growth_assumptions = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_with_target_growth-%s_assumptions.pkl")


##  ENERGY

# files for energy/nemomod inputs
fp_csv_nemomod_fuel_costs = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_modvar_enfu_fuel_costs.csv")
fp_csv_nemomod_hydropower_max_tech_capacity = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_modvar_entc_nemomod_max_technological_capacity.csv")
fp_csv_nemomod_minimum_share_of_production_baselines = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_minimum_share_of_production_baseline.csv")
fp_csv_nemomod_residual_capacity_inputs = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_modvar_entc_nemomod_residual_capacity.csv")
fp_csv_nemomod_transmission_losses = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_modvar_enfu_transmission_loss_frac_electricity.csv")

# files for energy and energy demand
fp_csv_scoe_elasticity_of_energy_consumption = os.path.join(dir_rbd_energy, "scoe_elasticity_of_energy_consumption.csv")
fp_csv_scoe_initial_energy_consumption = os.path.join(dir_rbd_energy, "scoe_initial_energy_consumption.csv")
fp_csv_scoe_consumption_scalar = os.path.join(dir_rbd_energy, "scoe_consumption_scalar.csv")


##  INDUSTRIAL PRODUCTION

# files for industry
fp_csv_elasticity_of_industrial_production = os.path.join(dir_rbd_ind_prod_components, "elasticity_of_industrial_production_to_gdp.csv")
fp_csv_industrial_production_scalar = os.path.join(dir_rbd_ind_prod_components, "industrial_production_scalar.csv")
fp_csv_initial_industrial_production = os.path.join(dir_rbd_ind_prod_components, "initial_industrial_production.csv")
fp_csv_ippu_fc_efs = os.path.join(dir_rbd_ind_efs_fcs_ippu, "emission_factors_ippu_fcs.csv")
fp_csv_ippu_frac_cement_clinker = os.path.join(dir_rbd_ind_cement_clinker_fraction, "clinker_fraction_cement_ippu.csv")
fp_csv_ippu_net_imports_cement_clinker = os.path.join(dir_rbd_ind_cement_clinker_fraction, "net_imports_cement_clinker.csv")



