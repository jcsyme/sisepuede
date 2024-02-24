from attribute_table import AttributeTable
import model_afolu as mafl
import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import setup_analysis as sa
import support_functions as sf
from typing import *
import warnings




class MixedLNDUTransitionFromBounds:
	"""
	Preliminary implenetation (not in proper data pipeline form) to mix transition matrices
		between bounds.

	Keyword Arguments
	-----------------
	eps: acceptable error in row sums for matrices
	filter_quality_ranking: filter the design based on "design_quality_rank" (describes)
		the quality of the LP used to infer the transition probability matrix. Lower
		numbers are better.
		* If filter_quality_ranking is None, keep all ranks
		* If filter_quality_ranking == -1, keep only the best
		* Otherwise, specify a quality rank threshold (may eliminate some regions)
	"""
	def __init__(self,
		model_attributes: ma.ModelAttributes = sa.model_attributes,
		fp_all_transitions: Union[str, None] = None,
		fp_mean_transitions: Union[str, None] = None,
		bound_0: str = "min_diagonal",
		bound_1: str = "max_diagonal",
		eps: float = 0.000001,
		field_design_quality_rank: str = "design_quality_rank",
		field_optimization_bound: str = "optimization_bound",
		field_region: str = "country",
		field_year: str = "year",
		field_prepend_dfs: str = "pij_lndu_",
		filter_quality_ranking: Union[int, None] = None,
	):

		# initialize basic properties
		self.bound_0 = bound_0
		self.bound_1 = bound_1
		self.field_design_quality_rank = field_design_quality_rank
		self.field_optimization_bound = field_optimization_bound
		self.field_region = field_region
		self.field_year = field_year
		self.field_prepend_dfs = field_prepend_dfs
		self.model_attributes = model_attributes


		# initialize other properties
		self._initialize_afolu_info()
		self._initialize_tables(
			fp_all_transitions,
			fp_mean_transitions
		)
		self._initialize_fields()
		self._initialize_dicts(
			eps = eps,
			filter_quality_ranking = filter_quality_ranking
		)



	#############################
	#	SOME INITIALIZATION	#
	#############################

	def _initialize_afolu_info(self,
	) -> None:
		"""
		Set some properties associated with AFOLU. Sets the
			following properties:

		* self.attr_lndu
		* self.dict_fields_sum_check
		* self.model_afolu
		"""
		attr_lndu = self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_lndu)
		model_afolu = mafl.AFOLU(self.model_attributes)

		# fields to use to check sums
		dict_fields_sum_check = {}

		


		##  BUILD LAND USE DICTIONARY MAPPING 

		all_vars = self.model_attributes.build_variable_fields(
			model_afolu.modvar_lndu_prob_transition,
		)

		pycat_elem = self.model_attributes.get_subsector_attribute(
			self.model_attributes.subsec_name_lndu, 
			"pycategory_primary_element"
		)

		for key in self.attr_lndu.key_values:
			
			fields_row = self.model_attributes.build_variable_fields(
				model_afolu.modvar_lndu_prob_transition,
				restrict_to_category_values = {f"{pycat_elem}_dim1": [key]}
			)
			
			fields_row = [x.replace(self.field_prepend_dfs, "") for x in fields_row]

			#varlist = [x.replace(self.field_prepend_dfs, "") for x in all_vars if x.startswith(f"{self.field_prepend_dfs}{v}_to")]
			self.dict_fields_sum_check.update({v: fields_row})


		##  SET PROPERTIES

		self.attr_lndu = attr_lndu
		self.dict_fields_sum_check = dict_fields_sum_check
		self.model_afolu = model_afolu

		return None




	def _initialize_dicts(self,
		eps: float,
		filter_quality_ranking: Union[int, None]
	) -> None:
		"""
		Set some dictionaries and other properties:

			* self.all_regions
			* self.df_all_years
			* self.dict_all_transitions: dictionary used to generate
				mixes using all transitions (for all available years).

				Takes form:

				{
					country: {
						{
							self.bound_0: df_bound_0,
							self.bound_1: df_bound_1
						}
					},

					country_2: {
						{
							self.bound_0: df_bound_0,
							self.bound_1: df_bound_1
						}
					},
					.
					.
					.
				}

			* self.dict_mean_transitions: dictionary that can be used to
				generate mixes in future years using mean transition. Takes
				same form as self.dict_all_transitions.
			* self.filter_quality_ranking

		Function Arguments
		------------------
		eps: acceptable error in row sums for matrices
		filter_quality_ranking: filter the
		"""

		self.all_regions = sorted(list(
			set(self.df_all_transitions[self.field_region]) &
			set(self.df_mean_transitions[self.field_region])
		))

		# initialize dicts
		dict_all_transitions = {}
		dict_mean_transitions = {}

		# initialize all years
		self.df_all_years = self.df_all_transitions[[self.field_year]].drop_duplicates(
		).sort_values(
			by = [self.field_year]
		).reset_index(
			drop = True
		)

		# check the quality rank filter
		self.filter_quality_ranking = filter_quality_ranking if (isinstance(filter_quality_ranking, int) or (filter_quality_ranking is None)) else None

		for region in self.all_regions:

			dict_all_transitions.update({region: {}})
			dict_mean_transitions.update({region: {}})

			for bound in [self.bound_0, self.bound_1]:

				# start with all
				df_all_cur = sf.subset_df(
					self.df_all_transitions,
					{
						self.field_region: [region],
						self.field_optimization_bound: [bound]
					}
				).replace(
					{
						np.nan: pd.NA
					}
				)

				# filter on rank?
				if self.filter_quality_ranking is not None:
					if self.filter_quality_ranking < 0:
						df_all_cur = df_all_cur[
							df_all_cur[self.field_design_quality_rank] == min(df_all_cur[self.field_design_quality_rank])
						]
					else:
						df_all_cur = df_all_cur[
							df_all_cur[self.field_design_quality_rank] <= self.filter_quality_ranking
						]

				# check sums and drop rows where sum is not equal to 1
				df_all_cur = self.check_row_sums(df_all_cur, eps) if (len(df_all_cur) > 0) else None

				if df_all_cur is not None:
					# merge in
					df_all_cur = pd.merge(
						self.df_all_years, df_all_cur, how = "left", on = [self.field_year]
					).drop(
						[self.field_region, self.field_optimization_bound],
						axis = 1
					).sort_values(
						by = [self.field_year]
					).interpolate(
					).reset_index(
						drop = True
					)

				dict_all_transitions[region].update({bound: df_all_cur})


				# add in means
				df_mean_cur = sf.subset_df(
					self.df_mean_transitions,
					{
						self.field_region: [region],
						self.field_optimization_bound: [bound]
					}
				).drop(
					[self.field_region, self.field_optimization_bound],
					axis = 1
				).reset_index(
					drop = True
				)
				df_mean_cur = self.check_row_sums(df_mean_cur, eps)

				dict_all_transitions[region].update({bound: df_all_cur})
				dict_mean_transitions[region].update({bound: df_mean_cur})


		self.dict_all_transitions = dict_all_transitions
		self.dict_mean_transitions = dict_mean_transitions



	def _initialize_fields(self,
	) -> None:
		"""
		Initialize fields to mix. Includes renaming dictionary. Sets the following
			properties:

			* self.fields_mix
			* self.dict_fields_mix_to_fields_out
		"""

		self.fields_mix = [x for x in self.df_all_transitions.columns if (x in self.df_mean_transitions.columns) and (x not in [self.field_region, self.field_year, self.field_optimization_bound])]
		self.dict_fields_mix_to_fields_out = dict([(x, f"{self.field_prepend_dfs}{x}") for x in self.fields_mix])



	def _initialize_tables(self,
		fp_all_transitions: Union[str, None] = None,
		fp_mean_transitions: Union[str, None] = None
	) -> None:

		"""
		Initialize paths and input tables used in setting matrices. Sets:

			* self.df_all_transitions
			* self.df_best_design_quality_rankings (best ranking design result
				across all years for each region/optimization bound)
			* self.df_mean_transitions
			* self.fp_all_transitions
			* self.fp_mean_transitions

		"""

		self.fp_all_transitions = sa.fp_csv_transition_probability_estimation_annual if (fp_all_transitions is None) else fp_all_transitions
		self.fp_mean_transitions = sa.fp_csv_transition_probability_estimation_mean if (fp_mean_transitions is None) else fp_mean_transitions

		try:
			self.df_all_transitions = pd.read_csv(self.fp_all_transitions)
			self.df_mean_transitions = pd.read_csv(self.fp_mean_transitions)
		except Exception as e:
			warnings.warn(f"Error initializing table in _initialize_tabless: {e}")

		# set a design quality ranking df for reference
		fields_grp = [self.field_region, self.field_optimization_bound]
		fields_min = [self.field_design_quality_rank]
		dict_agg = dict([(x, "first") for x in fields_grp])
		dict_agg.update(dict([(x, "min") for x in fields_min]))
		self.df_best_design_quality_rankings = self.df_all_transitions.groupby(fields_grp).agg(dict_agg).reset_index(drop = True)



	#############################
	#	SOME INITIALIZATION	#
	#############################

	def check_row_sums(self,
		df_in: pd.DataFrame,
		eps: float,
		field_tmp: str = "SUM_TMP"
	) -> Union[pd.DataFrame, None]:

		df_ret = df_in.copy()
		w = set(list(range(len(df_in))))

		for cat in self.dict_fields_sum_check:
			fields_cur = self.dict_fields_sum_check.get(cat)

			# filter rows to keep at end
			arr_cur = np.array(df_in[fields_cur])
			vec_total = np.sum(arr_cur, axis = 1)
			w = w & set(np.where(np.abs(vec_total - 1) < eps)[0])

			# normalize
			arr_cur = (arr_cur.transpose()/vec_total).transpose()
			for fld in enumerate(fields_cur):
				i, fld = fld
				df_ret[fld] = arr_cur[:, i]

		df_ret = df_ret.iloc[sorted(list(w))] if (len(w) > 0) else None

		return df_ret





	def mix_transitions(self,
		frac_mix: float,
		region: str,
		transition_type: str = "annual"
	) -> pd.DataFrame:
		"""
		Return a data frame that mixes between bound_0 and bound_1. Calculates as:

		frac_mix*bound_1 + (1 - frac_mix)*bound_0.

		For transition_type, specify:

			* transition_type = "annual": returns data frame mix for all historical
				years for which annual transitions are available. Backfills data for
				years where no data are available.
			* transition_type = "mean": returns data frame mix for mean transition
				matrix (which is based on recent historical averages)
		"""

		##  RUN CHECKS

		frac_mix = min(max(frac_mix, 0.0), 1.0)

		# check transition type
		if transition_type not in ["annual", "mean"]:
			warnings.warn(f"Warning in mix_transitions: transition_type = {transition_type} not found. Please specify 'annual' or 'mean'. Returning 'annual'... ")
			transition_type = "annual"

		# get regional dictionary
		dict_by_region = self.dict_all_transitions.get(region) if (transition_type == "annual") else self.dict_mean_transitions.get(region)
		if dict_by_region is None:
			warnings.warn(f"Error: region '{region}' not found. Returning None.")
			return None

		# get bound 0
		df_0 = dict_by_region.get(self.bound_0)
		if df_0 is None:
			warnings.warn(f"Error: {self.field_optimization_bound} = '{self.bound_0}' not found for region '{region}'. Returning None.")
			return None

		# get bound 1
		df_1 = dict_by_region.get(self.bound_1)
		if df_1 is None:
			warnings.warn(f"Error: {self.field_optimization_bound} = '{self.bound_1}' not found for region '{region}'. Returning None.")
			return None


		##  CALCULATE MIX

		m_0 = np.array(df_0[self.fields_mix])
		m_1 = np.array(df_1[self.fields_mix])
		m_out = frac_mix*m_1 + (1 - frac_mix)*m_0

		df_out = pd.DataFrame(m_out, columns = self.fields_mix)
		df_out.rename(columns = self.dict_fields_mix_to_fields_out, inplace = True)

		df_out = pd.concat([self.df_all_years, df_out], axis = 1).reset_index(drop = True) if (transition_type == "annual") else df_out

		return df_out
