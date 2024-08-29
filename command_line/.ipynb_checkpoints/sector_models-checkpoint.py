import multiprocessing as mp
import csv
import xlrd
import os, os.path
from copy import deepcopy
import itertools
import numpy
import errno
import math
import time
import pandas as pd
import numpy as np

################################
#    AGRICULTURAL EMISSIONS    #
################################

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
#    BUILDINGS    #
###################

def sm_buildings(df_in):

	
	#dictionary of fields to apply scalar to
	all_builds = ["agriculture", "commercial", "industry", "residential"]
	#dictionary to map sectors to shorthand
	dict_shorthand = dict([[x, x[0:3]] for x in all_builds])
	
	
	#output dictionary
	dict_out = {}
	#gdp based
	for build in list(set(all_builds) - {"residential"}):
		
		if build == "commercial":
			field_emit = "emissions_buildings_stationary_MT_co2e_com"
		else:
			field_emit = "emissions_" + build + "_energy_input_MT_co2e"
		#add in livestock
		if build == "agriculture":
			additional_gdp = np.array(df_in["va_livestock"])
		else:
			additional_gdp = float(0)
			
		#update field for energy consumption
		field_energy_demand_elec = "energy_consumption_electricity_PJ_" + dict_shorthand[build]
		field_energy_demand_nonelec = "energy_consumption_non_electricity_PJ_" + dict_shorthand[build]
		#some components
		field_dem_fac = build + "_df_pj_per_million_gdp"
		field_factor = build + "_ef_kt_co2e_per_pj"
		field_elec_frac = build + "_frac_electric"
		field_va = "va_" + build
		#energy demand
		dem_ener = (np.array(df_in[field_va]) + additional_gdp)*np.array(df_in[field_dem_fac])
		frac_elec = np.array(df_in[field_elec_frac])
		vec_out = (1 - frac_elec)*dem_ener*np.array(df_in[field_factor])
		#convert to megatons
		vec_out = vec_out/1000
		#
		dict_out.update({
			field_emit: vec_out,
			field_energy_demand_elec: frac_elec*dem_ener,
			field_energy_demand_nonelec: (1 - frac_elec)*dem_ener
		})
		
	# RESIDENTIAL
	build = "residential"
	field_emit = "emissions_buildings_stationary_MT_co2e_res"
	#some components
	field_dem_fac = build + "_df_pj_per_hh"
	field_factor = build + "_ef_kt_co2e_per_pj"
	field_elec_frac = build + "_frac_electric"
	field_or = "occ_rate"
	#energy demand
	field_energy_demand_elec = "energy_consumption_electricity_PJ_" + dict_shorthand[build]
	field_energy_demand_nonelec = "energy_consumption_non_electricity_PJ_" + dict_shorthand[build]
	#number of households
	vec_hh = np.array(df_in["total_population"])/np.array(df_in[field_or])
	#energy demand
	dem_ener = vec_hh*np.array(df_in[field_dem_fac])
	frac_elec = np.array(df_in[field_elec_frac])
	#output emissions
	vec_out = (1 - frac_elec)*dem_ener*np.array(df_in[field_factor])
	vec_out = vec_out/1000
	
	dict_out.update({
		field_emit: vec_out,
		field_energy_demand_elec: frac_elec*dem_ener,
		field_energy_demand_nonelec: (1 - frac_elec)*dem_ener
	})

	
	#fields to sum for buildings total over
	fields_total = ["emissions_buildings_stationary_MT_co2e_" + x for x in ["res", "com"]]
	#set total vector
	total_em = sum(np.array([dict_out[x] for x in fields_total]))
	#update
	dict_out.update({"emissions_buildings_total_MT_co2e": total_em})

	#return
	return dict_out






################
#    ENERGY    #
################

def sm_energy(vec_pop, vec_gdp, vec_dem_pgdp_grid_com, vec_dem_pgdp_ind, vec_dem_pc_grid_res, frac_ind_energy_elec, vec_ef_per_energy_ind, dict_pp_props, dict_pp_ef, dict_co2e):
		
	# vec_pop is the population in millions
	# vec_gdp is total gdp in billion $USD
	# vec_dem_pgdp_grid_com is commercial demand factor in PJ/billion $USD
	# vec_dem_pgdp_grid_ind is industrial demand factor in PJ/billion $USD
	# vec_dem_pc_grid_res is residential demand factor in PJ/million people
	# frac_ind_energy_elec is the fraction of industrial energy that comes from electricity
	# vec_ef_per_energy_ind is the emissions factor for non-electric industrial energy (KTCO2e/PJ)
	# dict_pp_props is a dictionary (power plant, subtype) of proportion of power provided by each type
	# dict_pp_ef is a dictionary (gas, power plant, pp subtype) of emissions factors by gas, pp, and pps
	# dict_co2e gives the co2e transformations for different gasses

	#names from each dictionary

	#set all classes
	all_pp = [x for x in dict_pp_props.keys()]
	all_pps = [x for x in dict_pp_props[all_pp[0]]]
	all_gasses = [x for x in dict_co2e if (x in dict_pp_ef.keys())]

	#generate total grid demand by sector
	dem_grid_com = vec_gdp * vec_dem_pgdp_grid_com
	dem_grid_ind = vec_gdp * vec_dem_pgdp_ind * frac_ind_energy_elec
	dem_grid_res = vec_pop * vec_dem_pc_grid_res
	
	#get non-electric industrial energy emissions
	dem_nongrid_ind = vec_gdp * vec_dem_pgdp_ind * (1 - frac_ind_energy_elec)
	em_nonelectric_energy_ind = dem_nongrid_ind * vec_ef_per_energy_ind
	
	
	#initialize each grid emissions type
	grid_em_com = [0 for x in range(len(vec_gdp))]
	grid_em_ind = [0 for x in range(len(vec_gdp))]
	grid_em_res = [0 for x in range(len(vec_pop))]

	#loop over gas types to generate emissions totals
	for gas in all_gasses:
		#multiply by co2e
		equiv = dict_co2e[gas]
		#loop over each power plant type
		for ppt in all_pp:
			for ppst in all_pps:
				#get proportion of emissions from the type
				prop = dict_pp_props[ppt][ppst]
				#get emissions factors per PJ
				ef = dict_pp_ef[gas][ppt][ppst]
				#get total commercial and residential grid demand satisfied
				grid_em_com = grid_em_com + ef * prop * dem_grid_com
				grid_em_ind = grid_em_ind + ef * prop * dem_grid_ind
				grid_em_res = grid_em_res + ef * prop * dem_grid_res
				
	#set total vector
	total_em_grid = [(grid_em_com[x] + grid_em_ind[x] + grid_em_res[x])/1000 for x in range(len(grid_em_com))]
	#set total vector
	total_em = [(total_em_grid[x] + (em_nonelectric_energy_ind[x]/1000)) for x in range(len(grid_em_com))]
				
	#set output
	dict_output = {
		"energy_consumption_electricity_PJ_com": dem_grid_com,
		"energy_consumption_electricity_PJ_ind": dem_grid_ind,
		"energy_consumption_electricity_PJ_res": dem_grid_res,
		"energy_consumption_non_electricity_PJ_ind": dem_nongrid_ind,
		"emissions_electricity_MT_co2e_com": grid_em_com/1000,
		"emissions_electricity_MT_co2e_ind": grid_em_ind/1000,
		"emissions_electricity_MT_co2e_res": grid_em_res/1000,
		"emissions_non_electricity_energy_MT_co2e_ind": em_nonelectric_energy_ind/1000,
		"emissions_electricity_total_MT_co2e": total_em_grid,
		"emissions_energy_sector_total_MT_co2e": total_em
	}
	#return
	return dict_output



#####################################
#    INDUSTRY - PROCESS EMISSIONS   #
#####################################
    
def sm_industrial(df_in, dict_gdp_field):
    #initialize output dictionary
	dict_out = {}
	vec_total_emit = 0.
	#gdp based
	for prod in ["carburo", "cal", "vidrio", "industry_at_large"]:
		if prod != "industry_at_large":
			field_emit = "emissions_industry_indproc_" + prod + "_MT_co2e"
		else:
			field_emit = "emissions_industry_indproc_general_use_MT_co2e"
		field_gdp = dict_gdp_field[prod]
		field_factor = prod + "_kt_co2e_per_million_usd"
		#update emissions
		vec_emit = np.array(df_in[field_gdp])*np.array(df_in[field_factor])/1000
		#add to total
		vec_total_emit = vec_total_emit + vec_emit
		#update dictionary
		dict_out.update({field_emit: vec_emit})
		
	#cement
	prod = "cemento"
	field_emit = "emissions_industry_indproc_" + prod + "_MT_co2e"
	field_prod = "production_industry_" + prod + "_KT"
	field_gdp = dict_gdp_field[prod]
	field_factor = prod + "_kt_co2e_per_kt_prod"
	field_prod_factor = prod + "_kt_prod_per_million_usd"
	#estimate production
	vec_prod = np.array(df_in[field_prod_factor])*np.array(df_in[field_gdp])
	#calculate emissions
	vec_emit = vec_prod*np.array(df_in[field_factor])/1000
	#add to total
	vec_total_emit = vec_total_emit + vec_emit
	#update emissions
	dict_out.update({
		field_prod: vec_prod,
		field_emit: vec_emit,
		"emissions_industry_indproc_total_MT_co2e": vec_total_emit
	})

	#return
	return dict_out



##################
#    LAND USE    #
##################

##TEMPORARY MODEL
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



###############
#    WASTE    #
###############

def sm_waste(
	waste_df,
	waste_df_base,
	gasses,
	all_keys_sdrd,
	all_ar,
	dict_proportion_group_substrs,
	param_m,
	param_docf,
	param_f,
	dict_gas_co2e,
	dict_fields = {"population": "total_population", "gdp_ind": "va_industry"},
	compost_reciclaje_keys = set({"alimiento", "jardin"})
):

	###   SOME FIELDS
	
	field_pop = dict_fields["population"]
	field_gdp = dict_fields["gdp_ind"]
	
	#proportion of non-recycled waste heading to landfill
	field_frac_sdrd = "frac_rso_sdrd"
	field_frac_quemado = "frac_rso_quemado"
	if "year" in waste_df.columns:
		field_year = "year"
	elif "Year" in waste_df.columns:
		field_year = "Year"
	elif "anho" in waste_df.columns:
		field_year = "anho"

	
	###   BUILD SDRD DATA
	
	fields_waste_sdrd = [x for x in waste_df_base.columns if (x in waste_df.columns)]
	waste_df_sdrd = pd.concat([waste_df_base[fields_waste_sdrd], waste_df[fields_waste_sdrd]])
	#length of waste_df
	n_wdf = len(waste_df)
	
	###   PARAMTERS OF INTEREST

	years = waste_df[field_year]
	methane_proportion_captured = np.array(waste_df_sdrd["frac_recap"])
	factor_ggri_per_bilpib = waste_df_sdrd["ggri_per_bilpib"]
	
	#conversion factors for CH4 and N20 to CO2e -- these are unique and taken from estimates
	#dict_gas_co2e = {}
	#set n; TEMP: USE ONLY BASELINE VALUE
	#n = 0#len(waste_df) - 1
	#for gas in gasses:
	#	field_wdf = gas + "_to_co2e"
	#	dict_gas_co2e.update({gas: waste_df[field_wdf][n]})

	###   WASTE TOTALS

	##  SDRD
	
	#get population in millions
	pop_millions = np.array(waste_df[field_pop])*(10**(-6))
	pop_millions_sdrd = np.array(waste_df_sdrd[field_pop])*(10**(-6))
	#build total waste
	rso_produced_sdrd = pop_millions_sdrd * waste_df_sdrd["residuos_per_capita_per_anho_kg"]
	#industrial waste totals (gdp will be in millions of dollars, not billions)
	sdrd_waste_ind = np.array(waste_df_sdrd[field_gdp])*factor_ggri_per_bilpib/1000

	##  OTHER SOLID WASTE METHODS
	rso_produced = np.array(pop_millions * waste_df["residuos_per_capita_per_anho_kg"])
	#get proportion of waste to landfill
	rso_frac_sdrd = np.array(waste_df_sdrd["frac_rso_sdrd"])
	
	##  NET EMISSIONS FROM RECYCLING/COMPOST
	em_recycling = 0.
	em_compost = 0.
	
	#fraction
	if True:
		#get substring id to get proportion of waste by type
		substr_identifier_pwt = str(dict_proportion_group_substrs["typo_de_residuo_en_sdrd"])
		#initialize dictionary of waste output by type
		dict_rso_by_type = {}
		#initialize vectors of quantities
		vec_recycled = 0.
		vec_sdrd = 0.
		vec_compost = 0.
		vec_otro = 0.
		vec_quemado = 0.
		#recycle/compost keys
		keys_rc = list(set(all_keys_sdrd) - set({"industrial"}))
		keys_rc.sort()
		#loop
		for wc in keys_rc:
			#initialize the dictionary that splits out quantities of waste by type (rec/sdrd)
			dict_rso_by_type.update({wc: {}})
			#check for field
			field_rec = "frac_" + wc + "_rec"
			#field giving proportion of waste heading to landfill that is of type wc
			field_frac_rso = "sdrd_frac_" + wc
			#field giving the emissions factors by type
			field_ef_wc = "ef_rec_" + wc
			#total rso representing waste of type wc (estimated)
			rso_produced_wc = np.array(rso_produced_sdrd)*np.array(waste_df_sdrd[field_frac_rso])
			#get quantity of produced waste of type wc that is recycled
			rso_recycled = rso_produced_wc*np.array(waste_df_sdrd[field_rec])
			#get quantity that heads to landfill
			rso_sdrd = (rso_produced_wc - rso_recycled)*np.array(waste_df_sdrd[field_frac_sdrd])
			#get fraction burned
			rso_quemado = (rso_produced_wc - (rso_recycled + rso_sdrd))*np.array(waste_df_sdrd[field_frac_quemado])
			#waste that is unaccounted for
			rso_otro = rso_produced_wc - (rso_recycled + rso_sdrd + rso_quemado)
			#add to recycling
			if wc not in compost_reciclaje_keys:
				vec_recycled = vec_recycled + rso_recycled
				#get emissions
				em_recycling = em_recycling + rso_recycled*np.array(waste_df_sdrd[field_ef_wc])
			else:
				vec_compost = vec_compost + rso_recycled
				#get emissions
				em_compost = em_recycling + rso_recycled*np.array(waste_df_sdrd[field_ef_wc])

			#update other values (index only to waste_df)
			vec_otro = vec_otro + rso_otro[-n_wdf:]
			vec_quemado = vec_quemado + rso_quemado[-n_wdf:]
			vec_sdrd = vec_sdrd + rso_sdrd[-n_wdf:]
			#add to dictionary
			dict_rso_by_type[wc].update({"recycled": rso_recycled, "sdrd": rso_sdrd, "burned": rso_quemado, "other": rso_otro})

	#generate estimate of rso that is burned (quemado)
	rso_quemado = vec_quemado
	#other rso
	rso_otro = vec_otro
	#reduce recycling and compost to waste_df length
	vec_recycled = vec_recycled[-n_wdf:]
	em_recycling = em_recycling[-n_wdf:]
	
	vec_compost = vec_compost[-n_wdf:]
	em_compost = em_compost[-n_wdf:]

	##  DOMESTIC/INDUSTRIAL SEWAGE, GG
	em_industrial_inc_dbo = 0.25
	#kT waste (pop is in millions)
	dbo_waste_dom = pop_millions * waste_df["kg_dbo5_per_capita_per_anho"]
	dbo_waste_ind = dbo_waste_dom * em_industrial_inc_dbo#waste_df["em_industrial_inc_dbo"]
	

	######################
	#    BURNED WASTE    #
	######################

	#estimate emissions from burned waste
	em_quemado = np.array([0 for x in range(len(rso_quemado))])
	#set all emissions types
	for emt in gasses:
		#get
		key = "tonne_rq_gg_em_" + emt
		#add?
		if key in waste_df.columns:
			#get emissions factor
			factor_em = dict_gas_co2e[emt]
			#return emissions; multiply by 1000 since rso_quemado is in terms of KT (GG), needs to be in tonnes
			em_quemado = em_quemado + 1000 * rso_quemado * waste_df[key] * factor_em


	####################
	#    SDRD WASTE    #
	####################

	#get monthly exponent component
	exp_comp_month = (13 - param_m)/12
	#set molecular weight ratio of ch4 to c
	weight_ratio = 4/3

	#set output
	dict_em_rso = {}
	dict_em_rso_all_years = {}
	dict_arrays = {}
	dict_earrays = {}
	dict_ddoc_waste = {}
	#get substring id to get proportion of waste by type
	substr_identifier = str(dict_proportion_group_substrs["typo_de_residuo_en_sdrd"])

	#loop over names
	for wt in all_keys_sdrd:
		#get appropriate fields
		field_k = "rso_k_" + wt
		field_doc = "rso_doc_" + wt
		field_mcf = "mean_mcf_sdrd"
		#get emissions decay factor (k)
		k = list(waste_df_sdrd[field_k])
		#TEMPORARY AS OF 20191217 - INDEX TO FIRST YEAR TO ELIMINATE UNCERTAINTY SPREAD; APPLIES TO k AND doc
		k = k[0]#k[len(k) - 1]
		#get doc
		doc = list(waste_df_sdrd[field_doc])
		doc = doc[0]#doc[len(doc) - 1]
		#get mcf vector
		mcf_vector = np.array(waste_df_sdrd[field_mcf])
		#check
		
		#get proportion of all sdrd waste that is of type wt
		field_wt = substr_identifier + wt

		if field_wt in waste_df.columns:
			#set total waste
			residuos_vol = np.array(dict_rso_by_type[wt]["sdrd"])
		else:
			residuos_vol = sdrd_waste_ind

		#get the estimated mass of decompostable degradable organic carbon (DDOC) deposited each year
		ddoc_waste = list(residuos_vol * doc * param_docf * mcf_vector)

		#gives proportion of mass that has decayed during time period i (index 1, the second element, represents waste deposited in year 0 decaying in year 1)
		vec_decay_prop = [0] + [(math.e**(-k * (x - 1)))*(1 - math.e**(-k)) for x in range(1, len(ddoc_waste))]
		#build matrix where each row i represents the proportion of waste deposited at time i that emits in time j
		array_decay_prop = [([0 for y in range(0, int(x))] + vec_decay_prop[0:(len(vec_decay_prop) - (int(x)))]) for x in range(0, len(vec_decay_prop))]
		array_decay_prop = np.array(array_decay_prop)
		#build array where each row i, column j gives total amount of CH4 emissions at time j due to waste deposited at time i
		emissions_array = np.array([ddoc_waste[int(x)] * param_f * weight_ratio * array_decay_prop[int(x)] for x in range(0, len(array_decay_prop))])
		#get
		y = list(emissions_array.sum(axis = 0))
		#incorporate capture proportion
		y = list(np.array(y) * (1 - np.array(methane_proportion_captured)) * dict_gas_co2e["ch4"])
		#reduce y for only applicable model years
		y_red = [y[x] for x in range(len(waste_df_sdrd)) if (list(waste_df_sdrd[field_year])[x] >= min(waste_df[field_year]))]

		dict_ddoc_waste.update({wt: ddoc_waste})
		dict_arrays.update({wt: array_decay_prop})
		dict_earrays.update({wt: emissions_array})
		dict_em_rso_all_years.update({wt: y})
		dict_em_rso.update({wt: y_red})

	#get total
	em_rso_total = [0 for x in range(0, len(dict_em_rso["jardin"]))]
	em_rso_total_all_years = [0 for x in range(0, len(dict_em_rso_all_years["jardin"]))]

	for key in list(dict_em_rso.keys()):
		em_rso_total = [em_rso_total[int(x)] + dict_em_rso[key][int(x)] for x in range(0, len(dict_em_rso[key]))]
		em_rso_total_all_years = [em_rso_total_all_years[int(x)] + dict_em_rso_all_years[key][int(x)] for x in range(0, len(dict_em_rso_all_years[key]))]
	em_rso_total = em_rso_total + em_recycling + em_compost
	#add to dictionary
	dict_em_rso.update({"total": em_rso_total})
	dict_em_rso_all_years.update({"total": em_rso_total_all_years})


	################################
	#    AGUAS RESIDUALES WASTE    #
	################################

	#initialize mcf for ar
	ar_mcf = np.array([0 for x in range(len(waste_df))])
	#loop to build mean mcf
	for art in all_ar:
		#set fields
		field_ar_prop = dict_proportion_group_substrs["typo_de_ar"] + art
		field_ar_mcf = "ar_typo_mcf_" + art
		ar_mcf = ar_mcf + waste_df[field_ar_prop] * waste_df[field_ar_mcf]
	em_dbo_dom = ar_mcf * dbo_waste_dom * waste_df["max_ch4_per_dbo"] * dict_gas_co2e["ch4"]
	em_dbo_ind = ar_mcf * dbo_waste_ind * waste_df["max_ch4_per_dbo"] * dict_gas_co2e["ch4"]


	#########################
	#    PROTEINAS WASTE    #
	#########################

	#define conversion factor from IPCC model kg N2O-N into kg N2O (44/28)
	factor_n2on_n2o = 11/7
	#calculate total volume of protein (n2o, not ch4, emissions)
	vec_proteina = np.array(pop_millions * waste_df["kg_por_persona_proteina_per_anho"] * waste_df["factor_nonconsum_ind_proteina"] * waste_df["factor_nonconsum_dom_proteina"])
	#note: calculations in 0 SCS *exclude* the 1.25 industrial factor. These are included here
	em_n20_proteina = vec_proteina * factor_n2on_n2o * np.array(waste_df["kg_n2_per_kg_proteina"] * waste_df["kg_n2o_per_kg_n"])
	#convert
	em_proteina = em_n20_proteina * dict_gas_co2e["n2o"]

	###   GET THE TOTAL WASTE
	total_waste = np.array(dict_em_rso["total"]) + np.array(em_quemado) + np.array(em_dbo_dom) + np.array(em_dbo_ind) + np.array(em_proteina)

	###   OUTPUT DICTIONARY - divide by 1000 to convert to megatons
	dict_out = {
		"year": list(waste_df[field_year]),
		"emissions_waste_total_MT_co2e": total_waste/1000,
		"emissions_waste_dbo_industrial_MT_co2e": np.array(em_dbo_ind)/1000,
		"emissions_waste_dbo_domestico_MT_co2e": np.array(em_dbo_dom)/1000,
		"emissions_waste_rso_quemado_MT_co2e": np.array(em_quemado)/1000,
		"emissions_waste_proteina_MT_co2e": np.array(em_proteina)/1000,
		"emissions_waste_compost_MT_co2e": np.array(em_compost)/1000,
		"emissions_waste_recycling_MT_co2e": np.array(em_recycling)/1000,
		"waste_generated_recycled_KT": vec_recycled,
		"waste_generated_compost_KT": vec_compost,
		"waste_generated_landfill_KT": vec_sdrd,
		"waste_generated_burned_KT": vec_quemado,
		"waste_generated_other_KT": vec_otro,
		"wastewater_generated_protein_n2o_KT": vec_proteina,
		"wastewater_generated_bdo_domestic_ch4_KT": dbo_waste_dom,
		"wastewater_generated_bdo_industrial_ch4_KT": dbo_waste_ind,
		"wastewater_generated_total_KT": (vec_proteina + dbo_waste_dom + dbo_waste_ind)
	}

	for wt in all_keys_sdrd:
		#new field name
		field_new = "emissions_waste_sdrd_" + wt
		#add to dictionary
		dict_out.update({field_new: np.array(dict_em_rso[wt])/1000})

	return pd.DataFrame(dict_out)


