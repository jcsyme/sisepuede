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
# attribute tables and readthedocs
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

dir_rbd_afolu_exports_imports = sf.check_path(os.path.join(dir_ref_batch_data, "afolu_imports_exports"), True)
dir_rbd_baseline_transition_probs = sf.check_path(os.path.join(dir_ref_batch_data, "baseline_transition_probability_estimates"), True)
dir_rbd_kcc = sf.check_path(os.path.join(dir_ref_batch_data, "koppen_climate_classifications"), True)
dir_rbd_soc = sf.check_path(os.path.join(dir_ref_batch_data, "soil_grids_soil_organic_carbon"), True)
dir_rbd_nemomod_energy_inputs = sf.check_path(os.path.join(dir_ref_batch_data, "nemomod_energy_inputs"), True)
# data crosswalks
fp_csv_cw_fao_crops = os.path.join(dir_ref_data_crosswalks, "fao_crop_categories.csv")
fp_csv_cw_fao_product_demand_categories_for_ie = os.path.join(dir_ref_data_crosswalks, "fao_product_demand_categories_for_import_export.csv")
# files for afolu imports/exports
fp_csv_afolu_import_exports = os.path.join(dir_rbd_afolu_exports_imports, "afolu_import_(ofdem)_export_(ofprod)_fractions.csv")
# files for energy/nemomod inputs
fp_csv_nemomod_residual_capacity_inputs = os.path.join(dir_rbd_nemomod_energy_inputs, "inputs_by_country_modvar_entc_nemomod_residual_capacity.csv")
# files for afolu transition probabilities
fp_csv_transition_probability_estimation_annual = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_and_year.csv")
fp_csv_transition_probability_estimation_mean = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean.csv")
fp_csv_transition_probability_estimation_mean_recent = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_recent_only.csv")
fpt_csv_transition_probability_estimation_mean_with_growth = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_with_target_growth-%s.csv")
fpt_pkl_transition_probability_estimation_mean_with_growth_assumptions = os.path.join(dir_rbd_baseline_transition_probs, "transition_probs_by_region_mean_with_target_growth-%s_assumptions.pkl")
# files for Koppen Climate Classification
fp_csv_kcc_cells_merged_to_country = os.path.join(dir_rbd_kcc, "kcc_cells_merged_to_country.csv")
fp_csv_kcc_cell_counts_by_country_kcc = os.path.join(dir_rbd_kcc, "kcc_cell_counts_by_country.csv")
fp_csv_climate_fields_by_country_simple = os.path.join(dir_rbd_kcc, "climate_fields_by_country.csv")
# fiels for Soil Organic Carbon by country from SoilGrids
fp_csv_soc_cells_merged_to_country = os.path.join(dir_rbd_soc, "scc_cells_merged_to_country.csv")
fp_csv_soc_average_soc_by_country = os.path.join(dir_rbd_soc, "soc_average_soc_by_country.csv")
fp_csv_soc_fields_by_country_simple = os.path.join(dir_rbd_soc, "soc_fields_by_country.csv")


##  FILE-PATH DEPENDENT FUNCTIONS

def excel_template_path(sector: str, region: str, type_db: str, create_export_dir: bool = True) -> str:
    """
        sector: the emissions sector (e.g., AFOLU, Circular Economy, etc.)
        region: three-character region code
        type_db: one of "calibrated", "demo", "uncalibrated"
    """

    # check type specification
    dict_valid_types = {
        "calibrated": dir_parameters_calibrated,
        "demo": dir_parameters_demo,
        "uncalibrated": dir_parameters_uncalibrated
    }

    if type_db not in dict_valid_types.keys():
        valid_types = sf.format_print_list(list(dict_valid_types.keys()))
        raise ValueError(f"Invalid parameter db type '{type_db}' specified: valid types are {valid_types}.")

    # check sector
    if sector in model_attributes.all_sectors:
        abv_sector = model_attributes.get_sector_attribute(sector, "abbreviation_sector")
    else:
        valid_sectors = sf.format_print_list(model_attributes.all_sectors)
        raise ValueError(f"Invalid sector '{sector}' specified: valid sectors are {valid_sectors}.")

    if type_db != "demo":
        # check region and create export directory if necessary
        if region.lower() in model_attributes.dict_attributes["region"].key_values:
            abv_region = region.lower()
            if (type_db != "demo"):
                dir_exp = sf.check_path(os.path.join(dict_valid_types[type_db], abv_region), create_export_dir)
                print(dir_exp)
                dict_valid_types.update({type_db: dir_exp})
        else:
            valid_regions = sf.format_print_list(model_attributes.dict_attributes["region"].key_values)
            raise ValueError(f"Invalid region '{region}' specified: valid regions are {valid_regions}.")

        fn_out = f"model_input_variables_{abv_region}_{abv_sector}_{type_db}.xlsx"

    else:
        fn_out = f"model_input_variables_{abv_sector}_{type_db}.xlsx"

    return os.path.join(dict_valid_types[type_db], fn_out)
