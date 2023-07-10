import argparse
from util.config import ConfigurationHandler
import numpy as np
import os, os.path
import pandas as pd




def sm_agriculture(df_in, all_ag, area):
  	
	#initialize dict of output
	dict_out = {}
	#initialize total
	vec_total_emit = 0.
	#gdp based
	for ag in all_ag:
		#idenfity some fields
		field_area = "frac_lu_" + ag
		field_ef = ag + "_kg_co2e_ha"
		#emission total fields
		field_emit = "emissions_agriculture_" + ag + "_MT_co2e"
		#update total emissions (gg co2e)
		vec_emit = area*(np.array(df_in[field_area])*np.array(df_in[field_ef])*(10**(-6))).astype(float)
		#conver to MT
		vec_emit = vec_emit/1000
		#add to dictionary
		dict_out.update({field_emit: vec_emit})
		#update
		vec_total_emit = vec_total_emit + vec_emit
	#add to dictionary
	dict_out.update({"emissions_agriculture_crops_total_MT_co2e": vec_total_emit})
    #return
	return dict_out




###################
#    LIVESTOCK    #
###################

def sm_livestock(df_in, all_ls):

	#initialize output dictionary
	dict_out = {}
	#initialize total emissions by type
	vec_total_emit_man = 0.0
	vec_total_emit_fer = 0.0
	vec_total_emit = 0.0
	#gdp based
	for lsc in all_ls:
		#idenfity some fields
		field_count = lsc
		#get emissions factors fields
		field_ef_f = lsc + "_fermentation_ef_c1_gg_co2e_head"
		field_ef_m = lsc + "_manure_ef_c1_gg_co2e_head"
		#emission total fields
		field_emit_f = "emissions_livestock_" + lsc + "_fermentation_MT_co2e"
		field_emit_m = "emissions_livestock_" + lsc + "_manure_MT_co2e"
		#update emissions by type for this livestock class
		vec_emit_man = np.array(df_in[field_count])*np.array(df_in[field_ef_m]).astype(float)/1000
		vec_emit_fer = np.array(df_in[field_count])*np.array(df_in[field_ef_f]).astype(float)/1000
		#update totals by type
		vec_total_emit_man = vec_total_emit_man + vec_emit_man
		vec_total_emit_fer = vec_total_emit_fer + vec_emit_fer
		#add to dictionary
		dict_out.update({
			field_emit_f: vec_emit_fer,
			field_emit_m: vec_emit_man
		})
	
	#update total emissions for livestock
	vec_total_emit = vec_total_emit_man + vec_total_emit_fer
	#add to dictionary
	dict_out.update({
		"emissions_livestock_fermentation_MT_co2e": vec_total_emit_fer,
		"emissions_livestock_manure_MT_co2e": vec_total_emit_man,
		"emissions_livestock_total_MT_co2e": vec_total_emit
	})

	#return
	return dict_out





##################
#    LAND USE    #
##################

##  TEMPORARY MODEL (to be replaced by Markov Chain)
def sm_land_use(df_in, all_lu, all_forest, all_lu_conv, tuple_my_dim, area, use_lu_diff_for_conv_q):

	dict_out = {}
	#
	#NOTE: assumes that df_in is sorted by master id, then year
	#tuple_my_dim = (n_master, n_year)
	#
	n_master = tuple_my_dim[0]
	n_year = tuple_my_dim[1]
	#initialize totals
	vec_total_emit = 0.
	vec_forest_emit = 0.
	#gdp based
	for lu in (all_lu + all_forest):
		#idenfity some fields
		field_area = "frac_lu_" + lu
		field_ef = lu + "_ef_c1_gg_co2e_ha"
		#emission total fields
		field_emit = "emissions_land_use_existence_" + lu + "_MT_co2e"
		#update total emissions
		vec_emit = area*np.array(df_in[field_area])*np.array(df_in[field_ef])/1000
		#update
		dict_out.update({field_emit: vec_emit})
		#add
		vec_total_emit = vec_total_emit + vec_emit
		#update forest emissions
		if lu in all_forest:
			vec_forest_emit = vec_forest_emit + vec_emit
			
		#check for conversion
		if lu in all_lu_conv:
			#initialize
			field_emit_conv = "emissions_land_use_conversion_" + lu + "_MT_co2e"
			#use difference in area, or is there
			if use_lu_diff_for_conv_q:
				#vector of area
				vec_area = area*np.array(df_in[field_area]).astype(float)
				#get vec of diffs
				vec_diff = vec_area[1:len(vec_area)] - vec_area[0:(len(vec_area) - 1)]
				vec_diff = np.array([vec_diff[0]] + list(vec_diff)).astype(float)
				#vector of differences in years
				vec_diff_years = np.array(df_in["year"])
				vec_diff_years = vec_diff_years[1:len(vec_diff_years)] - vec_diff_years[0:(len(vec_diff_years) - 1)]
				vec_diff_years = np.array([vec_diff_years[0]] + list(vec_diff_years))

				#update differences so that base year is properly accounted for
				for i in range(0, n_master):
					#get min range
					r_min = i*n_year
					r_max = (i + 1)*n_year
					#update
					vec_diff[r_min] = vec_diff[r_min + 1]
					vec_diff_years[r_min] = vec_diff_years[r_min + 1]

				#get estimated emissions
				est_ce = vec_diff*vec_ef/vec_diff_years
				#convert to zero
				est_ce[np.where(est_ce < 0)] = 0
			else:
				#initialize new estimate of conversion emissions
				est_ce = 0.

				#get conversion fields
				fields_ef_conv = [x + "_to_" + lu + "_ef_conversion_c1_gg_co2e_ha" for x in all_forest]
				fields_area_conv = [x + "_conv_to_" + lu + "_area_ha" for x in all_forest]
				#array of emissions factors/areas
				array_ef = np.array(df_in[fields_ef_conv]).astype(float)
				array_area = np.array(df_in[fields_area_conv]).astype(float)
				#add conversion emissions estimate
				est_ce = sum((array_ef*array_area).transpose())
				
			#convert to MT
			est_ce = est_ce/1000
			#add to dictionary
			dict_out.update({field_emit_conv: est_ce})
			#add to total
			vec_total_emit = vec_total_emit + est_ce
	#add to dictionary
	dict_out.update({
		"emissions_land_use_forested_MT_co2e": vec_forest_emit,
		"emissions_land_use_net_MT_co2e": vec_total_emit
	})
	
	return dict_out