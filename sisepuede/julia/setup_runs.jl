##################################################################
#    IMPORTANT: ANALYTICAL QUERIES THAT ARE CALLED THROUGHOUT    #
##################################################################

#delimiter to separate ACS fields that need to be aggregated
delim_agg_fields = ";;"
#overwrite data if they exist?
overwrite_data_q = true
#overwrite fp_csv_ca_pl_unused if it exists?
overwrite_unused_fields_q = true
#use existing dataset in "build restore_existing_database_q" notebook?
restore_existing_database_q = false



#########################
#    SET DIRECTORIES    #
#########################

#set the acs year
acs_yr = 2010
census_yr = 2010

#high level
dir_cur = dirname(pwd())
dir_data = joinpath(dirname(dir_cur), "data")
dir_bin = joinpath(dir_cur, "bin")
dir_acs = joinpath(dir_data, "ACS_$(acs_yr)")
dir_ref = joinpath(dir_cur, "ref")
dir_ref_acs = joinpath(dir_ref, "ACS_$(acs_yr)")
dir_out = joinpath(dir_cur, "out")
dir_out_img = joinpath(dir_out, "img")

#acs subdirectories
dir_acs_templates = joinpath(dir_acs, "$(acs_yr)_5yr_Summary_FileTemplates")
dir_acs_data = joinpath(dir_acs, "California_Tracts_Block_Groups_Only")
dir_data_estimates = joinpath(dir_data, "preliminary_downscaled_estimates", "ACS-$(acs_yr)_TO_CENSUS-$(census_yr)")
#california
dir_cvap = joinpath(dir_data, "CA_block_level")
#set directory for state blocks
dir_sblks = joinpath(dir_data, "state_blocks_$(census_yr)_shp")
#pl 94 for california
dir_capl = joinpath(dir_data, "ca$(census_yr).pl")
dir_casf = joinpath(dir_data, "ca$(census_yr).sf1")

# check that directories exist/make them if need be
for d_check in [dir_out, dir_out_img, dir_data_estimates]
    if !ispath(d_check)
        mkdir(d_check)
    end
end


#######################
#    SET FILEPATHS    #
#######################

##  CSVs
fp_csv_acs_field_sequence_index = joinpath(dir_acs, "acs_field_sequence_index_$(acs_yr).csv")
fp_csv_acs_geodat = joinpath(dir_acs_data, "g$(acs_yr)5ca.csv")
fp_csv_adjacecy_index = joinpath(dir_data, "adjacency_index.csv")


fp_csv_aers_counts_stage_1 = joinpath(dir_out, "aers_counts_acs-$(acs_yr)_to_census-$(census_yr)_stage-1.csv")
fp_csv_aers_counts_stage_2 = joinpath(dir_out, "aers_counts_acs-$(acs_yr)_to_census-$(census_yr)_stage-2.csv")
fp_csv_aers_termination_status_stage_1 = joinpath(dir_out, "aers_termination_status_stage-1.csv")
fp_csv_aers_termination_status_stage_2 = joinpath(dir_out, "aers_termination_status_stage-2.csv")

fp_csv_attribute_census_sf1_table = joinpath(dir_ref, "attribute_census_sf1_table_id.csv")
fp_csv_attribute_demographic = joinpath(dir_ref, "attribute_demographic_id.csv")
fp_csv_attribute_household = joinpath(dir_ref_acs, "attribute_household_id.csv")
fp_csv_attribute_geo_unit = joinpath(dir_ref, "attribute_geo_unit.csv")
fp_csv_attribute_variable = joinpath(dir_ref_acs, "attribute_variable_id.csv")

fp_csv_ca_pl94_template = joinpath(dir_capl, "ca0000##SEGMENT##$(census_yr).pl")
fp_csv_ca_pl_unused = joinpath(dir_data_estimates, "california_pl_94_data_codes-unused.csv")
fp_csv_composite_data_block_group = joinpath(dir_data_estimates, "composite_data-block_group.csv")
fp_csv_composite_data_tract = joinpath(dir_data_estimates, "composite_data-tract.csv")
fp_csv_codemap_sf1_aggregation = joinpath(dir_ref, "codemap_sf1_aggregation.csv")
fp_csv_codemap_sf1_to_acs = joinpath(dir_ref_acs, "codemap_sf1_to_acs.csv")
fp_csv_extraction_fields_acs = joinpath(dir_ref_acs, "extraction_fields-acs_$(acs_yr).csv")
fp_csv_extraction_fields_file_reduce = joinpath(dir_ref, "extraction_fields_2020_pl94_file_reduce.csv")
fp_csv_geo_ca = joinpath(dir_acs, "geo_ca.csv")
fp_csv_out_constraint_matrix = joinpath(dir_data_estimates, "base_dasymmetric_constraints.csv")
fp_csv_preliminary_block_household_variable_estimates = joinpath(dir_data_estimates, "california_preliminary_block_household_variable_estimates.csv")
fp_csv_preliminary_block_population_variable_estimates = joinpath(dir_data_estimates, "california_preliminary_block_population_variable_estimates.csv")
fp_csv_ref_constraint_matrix_manual = joinpath(dir_ref_acs, "manual_dasymmetric_constraints.csv")
fp_csv_summing_matrix = joinpath(dir_ref, "summing_matrix-demographic_id.csv")


##  SHAPEFILEs
fp_shp_state_blocks = joinpath(dir_sblks, "state_blks$(census_yr - 2000).shp")

##  OTHER TEXT FILES
fp_fwtxt_ca_pl94_geoheader = joinpath(dir_capl, "cageo$(census_yr).pl")
fp_fwtxt_ca_sf1_geoheader = joinpath(dir_casf, "cageo$(census_yr).sf1")

##  EXCEL FILES
fp_xlsx_acs_geodat_header = joinpath(dir_acs_templates, "$(acs_yr)_SFGeoFileTemplate.xlsx")

