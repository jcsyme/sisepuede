from attribute_table import AttributeTable
import itertools
import logging
import math
import numpy as np
import pandas as pd
import os, os.path
import re
import support_functions as sf
import time
from typing import *









##  class for sampling and generating experimental design
class SamplingUnitLegacy:
	"""
	Generate future trajectories based on an input database.

	Initialization Arguments
	------------------------
	- df_variable_definition: DataFrame used to define variable specifications
	- dict_baseline_ids: dictionary mapping a string of a baseline id field to a
		baseline id value (integer)
	- time_period_u0: first time period with uncertainty

	Keyword Arguments
	-----------------
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_time_period: field used to denote the time period
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group
		(integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group
		type (max, min, mix, or lhs)
	- key_strategy: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed
			differently than uncetainty in other variables
	- missing_flag_trajgroup: flag used to identify null trajgroups (default is
		-999)
	- regex_id: regular expression used to identify id fields in the input
		template
	- regex_max: re.Pattern (compiled regular expression) used to match the
		field storing the maximum scalar values at the final time period
	- regex_min: re.Pattern used to match the field storing the minimum scalar
		values at the final time period
	- regex_tp: re.Pattern used to match the field storing data values for each
		time period
	"""
	def __init__(self,
		df_variable_definition: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,
		fan_function_specification: str = "linear",
		field_time_period: str = "time_period",
		field_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		key_strategy: str = "strategy_id",
		missing_trajgroup_flag: int = -999,
		regex_id: re.Pattern = re.compile("(\D*)_id$"),
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)")
	):

		##  set some attributes

		# from function args
		self.field_time_period = field_time_period
		self.field_trajgroup_no_vary_q = field_trajgroup_no_vary_q
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_variable = field_variable
		self.key_strategy = key_strategy
		self.missing_trajgroup_flag = missing_trajgroup_flag
		self.time_period_end_certainty = self.check_time_start_uncertainty(time_period_u0)
		# others
		self._set_parameters()
		self._set_attributes_from_table(
			df_variable_definition,
			regex_id,
			regex_max,
			regex_min,
			regex_tp
		)

		# perform initializations
		self._initialize_uncertainty_functional_form(fan_function_specification)
		self._initialize_scenario_variables(dict_baseline_ids)
		self._initialize_trajectory_arrays()
		self._initialize_xl_type()





	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	# get attributes from the table
	def _set_attributes_from_table(self,
		df_variable_definition: pd.DataFrame,
		regex_id: re.Pattern,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		regex_tp: re.Pattern
	) -> None:
		"""
		Set a range of attributes derived the input df_variable_definition.
			Sets the following properties:

			* self.df_variable_definitions
			* self.fields_id
			* self.required_fields

		Function Arguments
		------------------
		- df_variable_definition: data frame used to set variable specifications
		- regex_id: regular expression used to identify id fields in the input
			template
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum scalar
			values at the final time period
		- regex_tp: re.Pattern used to match the field storing data values for
			each time period

		"""
		self.df_variable_definitions, self.required_fields = self.check_input_data_frame(df_variable_definition)
		self.fields_id = self.get_id_fields(regex_id)
		self.field_min_scalar, self.field_max_scalar, self.time_period_scalar = self.get_scalar_time_period(
			regex_max,
			regex_min
		)
		self.fields_time_periods, self.time_periods = self.get_time_periods(regex_tp)
		self.variable_trajectory_group = self.get_trajgroup()
		self.variable_trajectory_group_vary_q = self.get_trajgroup_vary_q()



	def _set_parameters(self,
	) -> None:
		"""
		Set some key parameters.

		* self.dict_required_tg_spec_fields
		* self.key_mix_trajectory
		* self.key_inf_traj_boundary
		* self.key_sup_traj_boundary
		* self.primary_key_id_coordinates
		* self.required_tg_specs

		"""

		self.key_mix_trajectory = "mixing_trajectory"
		self.key_inf_traj_boundary = "trajectory_boundary_0"
		self.key_sup_traj_boundary = "trajectory_boundary_1"
		self.primary_key_id_coordinates = "primary_key_id_coord"

		# maps internal name (key) to classification in the input data frame (value)
		self.dict_required_tg_spec_fields = {
			self.key_mix_trajectory: "mix",
			self.key_inf_traj_boundary: "trajectory_boundary_0",
			self.key_sup_traj_boundary: "trajectory_boundary_1"
		}
		self.required_tg_specs = list(self.dict_required_tg_spec_fields.values())



	def check_input_data_frame(self,
		df_in: pd.DataFrame
	):
		"""
		Check df_in for required fields. Sets the following attributes:

			* self.df_variable_definitions
			* self.required_fields
		"""
		# some standardized fields to require
		fields_req = [
			self.key_strategy,
			self.field_trajgroup_no_vary_q,
			self.field_variable_trajgroup,
			self.field_variable_trajgroup_type,
			self.field_uniform_scaling_q,
			self.field_variable
		]

		if len(set(fields_req) & set(df_in.columns)) < len(set(fields_req)):
			fields_missing = list(set(fields_req) - (set(fields_req) & set(df_in.columns)))
			fields_missing.sort()
			str_missing = ", ".join([f"'{x}'" for x in fields_missing])
			raise ValueError(f"Error: one or more columns are missing from the data frame. Columns {str_missing} not found")

		elif (self.key_strategy in df_in.columns) and ("_id" not in self.key_strategy):
			raise ValueError(f"Error: the strategy field '{self.key_strategy}' must contain the substring '_id'. Check to ensure this substring is specified.")

		return df_in.drop_duplicates(), fields_req



	def _initialize_scenario_variables(self,
		dict_baseline_ids: Dict[str, int],
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
		field_merge_key: Union[str, None] = None
	) -> None:
		"""
		Check inputs of the input data frame and id fields. Sets the following
			properties:

			* self.data_table
			* self.df_id_coordinates
			* self.dict_baseline_ids
			* self.dict_id_values
			* self.dict_variable_info
			* self.id_coordinates
			* self.num_scenarios
			* self.variable_specifications


		Function Arguments
		------------------
		- dict_baseline_ids: dictionary mapping each dimensional key to nominal
			baseline

		Keyword Arguments
		-----------------
		- df_in: input data frame used to specify variables
		- fields_id: id fields included in df_in
		- field_merge_key: scenario key
		"""
		df_in = self.df_variable_definitions if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		field_merge_key = self.primary_key_id_coordinates if (field_merge_key is None) else field_merge_key
		tups_id = set([tuple(x) for x in np.array(df_in[fields_id])])


		for tg_type in self.required_tg_specs:
			df_check = df_in[df_in[self.field_variable_trajgroup_type] == tg_type]
			for vs in list(df_check[self.field_variable].unique()):
				tups_id = tups_id & set([tuple(x) for x in np.array(df_check[df_check[self.field_variable] == vs][fields_id])])
		#
		tups_id = sorted(list(tups_id))
		df_scen = pd.DataFrame(tups_id, columns = fields_id)
		df_in = pd.merge(df_in, df_scen, how = "inner", on = fields_id)
		df_scen[field_merge_key] = range(len(df_scen))
		tups_id = sorted(list(tups_id))

		# id values and baseline ids
		dict_id_values, dict_baseline_ids = self.get_scenario_values(
			dict_baseline_ids,
			df_in = df_in,
			fields_id = fields_id
		)
		var_specs = self.get_all_vs(df_in)

		self.data_table = df_in
		self.df_id_coordinates = df_scen
		self.dict_baseline_ids = dict_baseline_ids
		self.dict_id_values = dict_id_values
		self.dict_variable_info = self.get_variable_dictionary(df_in, var_specs)
		self.id_coordinates = tups_id
		self.num_scenarios = len(tups_id)
		self.variable_specifications = var_specs




	def check_time_start_uncertainty(self,
		t0: int
	) -> int:
		return max(t0, 1)



	def generate_indexing_data_frame(self,
		df_id_coords: Union[pd.DataFrame, None] = None,
		dict_additional_fields: Dict[str, Union[float, int, str]] = None,
		field_primary_key_id_coords: Union[str, None] = None,
		field_time_period: Union[str, None] = None
	) -> pd.DataFrame:

		"""
		Generate an data frame long by time period and all id coordinates included in the sample unit.

		Keyword Arguments
		-----------------
		- df_id_coords: data frame containing id coordinates + primary key (in field_primary_key_id_coords)
			* If None, default to self.df_id_coordinates
		- dict_additional_fields: dictionary mapping additional fields to values to add
			* If None, no additional fields are added
		- field_primary_key_id_coords: field in df_id_coords denoting the primary key
			* If None, default to self.primary_key_id_coordinates
		- field_time_period: field to use for data frame
			* If None, default to self.field_time_period
		"""

		df_id_coords = self.df_id_coordinates if (df_id_coords is None) else df_id_coords
		field_primary_key_id_coords = self.primary_key_id_coordinates if (field_primary_key_id_coords is None) else field_primary_key_id_coords
		field_time_period = self.field_time_period if (field_time_period is None) else field_time_period

		# build array of coordinates x time periods
		df_coords_by_future = np.array([
			np.repeat(
				df_id_coords[field_primary_key_id_coords],
				len(self.time_periods)
			),
			np.concatenate(
				np.repeat(
					[self.time_periods],
					len(self.df_id_coordinates),
					axis = 0
				)
			)
		]).transpose()

		# convert to data frame
		df_coords_by_future = pd.DataFrame(
			df_coords_by_future,
			columns = [field_primary_key_id_coords, field_time_period]
		)

		df_coords_by_future = pd.merge(
			df_coords_by_future,
			df_id_coords,
			how = "left"
		).sort_values(
			by = [field_primary_key_id_coords, field_time_period]
		).reset_index(
			drop = True
		).drop(
			field_primary_key_id_coords, axis = 1
		)

		if dict_additional_fields is not None:
			df_coords_by_future = sf.add_data_frame_fields_from_dict(
				df_coords_by_future,
				dict_additional_fields
			)

		return df_coords_by_future



	def get_all_vs(self,
		df_in: pd.DataFrame
	) -> List:
		"""
		Get all variable schema associated with input template df_in
		"""
		if not self.field_variable in df_in.columns:
			raise ValueError(f"Field '{self.field_variable}' not found in data frame.")
		all_vs = sorted(list(df_in[self.field_variable].unique()))

		return all_vs



	def get_id_fields(self,
		regex_id: re.Pattern,
		df_in: Union[pd.DataFrame, None] = None
	) -> List:
		"""
		Get all id fields associated with input template df_in.

		Function Arguments
		------------------
		- regex_id: regular expression used to identify id fields

		Keyword Arguments
		-----------------
		- df_in: data frame to use to find id fields. If None, use
			self.df_variable_definitions

		"""

		if not isinstance(regex_id, re.Pattern):
			fields_out = []
		else:
			df_in = self.df_variable_definitions if (df_in is None) else df_in
			fields_out = sorted(
				[x for x in df_in.columns if (regex_id.match(x) is not None)]
			)

		if len(fields_out) == 0:
			raise ValueError(f"No id fields found in data frame.")

		return fields_out



	def _initialize_trajectory_arrays(self,
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
		fields_time_periods: Union[list, None] = None,
		variable_specifications: Union[list, None] = None
	) -> Dict[str, np.ndarray]:
		"""
		Order trajectory arrays by id fields; used for quicker lhs application
			across id dimensions. Sets the following properties:

			* self.ordered_trajectory_arrays
			* self.scalar_diff_arrays

		Keyword Arguments
		-----------------
		- df_in: variable specification data frame
		- fields_id: id fields included in the specification of variables
		- fields_time_periods: fields denoting time periods
		- variable_specifications: list of variable specifications included in
			df_in
		"""

		# initialize defaults
		df_in = self.data_table if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		fields_time_periods = self.fields_time_periods if not isinstance(fields_time_periods, list) else fields_time_periods
		variable_specifications = self.variable_specifications if not isinstance(variable_specifications, list) else variable_specifications

		# set some other variables
		tp_end = self.fields_time_periods[-1]
		dict_scalar_diff_arrays = {}
		dict_ordered_traj_arrays = {}

		for vs in variable_specifications:

			df_cur_vs = df_in[df_in[self.field_variable].isin([vs])]
			dfs_cur = [(None, df_cur_vs)] if (self.variable_trajectory_group is None) else df_cur_vs.groupby([self.field_variable_trajgroup_type])

			for df_cur in dfs_cur:
				tgs, df_cur = df_cur
				df_cur.sort_values(by = fields_id, inplace = True)

				# ORDERED TRAJECTORY ARRAYS
				array_data = np.array(df_cur[fields_time_periods])
				coords_id = df_cur[fields_id]
				dict_ordered_traj_arrays.update(
					{
						(vs, tgs): {
							"data": np.array(df_cur[fields_time_periods]),
							"id_coordinates": df_cur[fields_id]
						}
					}
				)

				# SCALAR DIFFERENCE ARRAYS - order the max/min scalars
				var_info = self.dict_variable_info.get((vs, tgs))
				vec_scale_max = np.array([var_info.get("max_scalar")[tuple(x)] for x in np.array(coords_id)])
				vec_scale_min = np.array([var_info.get("min_scalar")[tuple(x)] for x in np.array(coords_id)])

				# difference, in final time period, between scaled value and baseline value-dimension is # of scenarios
				dict_tp_end_delta = {
					"max_tp_end_delta": array_data[:,-1]*(vec_scale_max - 1),
					"min_tp_end_delta": array_data[:,-1]*(vec_scale_min - 1)
				}

				dict_scalar_diff_arrays.update({(vs, tgs): dict_tp_end_delta})

		self.ordered_trajectory_arrays = dict_ordered_traj_arrays
		self.scalar_diff_arrays = dict_scalar_diff_arrays



	def get_scalar_time_period(self,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		df_in:Union[pd.DataFrame, None] = None
	) -> Tuple[str, str, int]:
		"""
		Determine final time period (tp_final) as well as the fields associated with the minimum
			and maximum scalars (field_min/field_max) using input template df_in. Returns a tuple
			with the following elements:

			* field_min
			* field_max
			* tp_final

		Function Arguments
		------------------
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum scalar
			values at the final time period

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.df_variable_definitions
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		field_min = [x for x in df_in.columns if (regex_min.match(x) is not None)]
		if len(field_min) == 0:
			raise ValueError("No field associated with a minimum scalar value found in data frame.")
		else:
			field_min = field_min[0]

		# determine max field/time period
		field_max = [x for x in df_in.columns if (regex_max.match(x) is not None)]
		if len(field_max) == 0:
			raise ValueError("No field associated with a maximum scalar value found in data frame.")
		else:
			field_max = field_max[0]

		tp_min = int(field_min.split("_")[1])
		tp_max = int(field_max.split("_")[1])
		if (tp_min != tp_max) | (tp_min == None):
			raise ValueError(f"Fields '{tp_min}' and '{tp_max}' imply asymmetric final time periods.")
		else:
			tp_out = tp_min

		return (field_min, field_max, tp_out)



	def get_scenario_values(self,
		dict_baseline_ids: Dict[str, int],
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
	) -> Tuple[Dict, Dict]:
		"""
		Get scenario index values by scenario dimension and verifies baseline
			values. Returns a tuple:

			dict_id_values, dict_baseline_ids

		where `dict_id_values` maps each dimensional key (str) to a list of
			values and `dict_baseline_ids` maps each dimensional key (str) to a
			baseline scenario index

		Function Arguments
		------------------
		- df_in: data frame containing the input template
		- fields_id: list of id fields
		- dict_baseline_ids: dictionary mapping each dimensional key to nominal
			baseline

		Function Arguments
		------------------
		"""

		df_in = self.data_table if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		#
		dict_id_values = {}
		dict_id_baselines = dict_baseline_ids.copy()

		for fld in fields_id:
			dict_id_values.update({fld: list(df_in[fld].unique())})
			dict_id_values[fld].sort()

			# check if baseline for field is determined
			if fld in dict_id_baselines.keys():
				bv = int(dict_id_baselines[fld])

				if bv not in dict_id_values[fld]:
					if fld == self.key_strategy:
						raise ValueError(f"Error: baseline {self.key_strategy} scenario index '{bv}' not found in the variable trajectory input sheet. Please ensure the basline strategy is specified correctly.")
					else:
						msg_warning = f"The baseline id for dimension {fld} not found. The experimental design will not include futures along this dimension of analysis."
						warnings.warn(msg_warning)
			else:
				# assume minimum >= 0
				bv = min([x for x in dict_id_values[fld] if x >= 0])
				dict_id_baselines.update({fld: bv})
				msg_warning = f"No baseline scenario index found for {fld}. It will be assigned to '{bv}', the lowest non-negative integer."
				warnings.warn(msg_warning)

		return dict_id_values, dict_id_baselines



	def get_time_periods(self,
		regex_tp: re.Pattern,
		df_in:Union[pd.DataFrame, None] = None
	) -> Tuple[List, List]:
		"""
		Get fields associated with time periods in the template as well as time
			periods defined in input template df_in. Returns the following
			elements:

			fields_time_periods, time_periods

			where

			* fields_time_periods: nominal fields in df_in containing time
				periods
			* time_periods: ordered list of integer time periods

		Function Arguments
		-----------------
		- regex_tp: re.Pattern used to match the field storing data values for
			each time period

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.

		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		#fields_time_periods = [x for x in df_in.columns if x.isnumeric()]
		#fields_time_periods = [x for x in fields_time_periods if int(x) == float(x)]
		fields_time_periods = [str(x) for x in df_in.columns if (regex_tp.match(str(x)) is not None)]
		if len(fields_time_periods) == 0:
			raise ValueError("No time periods found in data frame.")

		time_periods = sorted([int(x) for x in fields_time_periods])
		fields_time_periods = [str(x) for x in time_periods]

		return fields_time_periods, time_periods



	def get_trajgroup(self,
		df_in:Union[pd.DataFrame, None] = None
	) -> Union[int, None]:
		"""
		Get the trajectory group for the sampling unit from df_in.

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		if not self.field_variable_trajgroup in df_in.columns:
			raise ValueError(f"Field '{self.field_variable_trajgroup}' not found in data frame.")
		# determine if this is associated with a trajectory group
		if len(df_in[df_in[self.field_variable_trajgroup] > self.missing_trajgroup_flag]) > 0:
			return int(list(df_in[self.field_variable_trajgroup].unique())[0])
		else:
			return None



	def get_trajgroup_vary_q(self,
		df_in:Union[pd.DataFrame, None] = None
	) -> Union[int, None]:
		"""
		Get the trajectory group for the sampling unit from df_in.

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		if not self.field_trajgroup_no_vary_q in df_in.columns:
			raise ValueError(f"Field '{self.field_trajgroup_no_vary_q}' not found in data frame.")
		# determine if this is associated with a trajectory group
		out = (len(df_in[df_in[self.field_trajgroup_no_vary_q] == 1]) == 0)

		return out



	def get_variable_dictionary(self,
		df_in: pd.DataFrame,
		variable_specifications: list,
		fields_id: list = None,
		field_max: str = None,
		field_min: str = None,
		fields_time_periods: list = None
	) -> None:
		"""
		Retrieve a dictionary mapping a vs, tg pair to a list of information.
			The variable dictionary includes information on sampling ranges for
			time period scalars, wether the variables should be scaled
			uniformly, and the trajectories themselves.

		"""
		fields_id = self.fields_id if (fields_id is None) else fields_id
		field_max = self.field_max_scalar if (field_max is None) else field_max
		field_min = self.field_min_scalar if (field_min is None) else field_min
		fields_time_periods = self.fields_time_periods if (fields_time_periods is None) else fields_time_periods

		dict_var_info = {}
		tgs_loops = self.required_tg_specs if (self.variable_trajectory_group is not None) else [None]

		for vs in variable_specifications:

			df_vs = sf.subset_df(df_in, {self.field_variable: vs})

			for tgs in tgs_loops:

				df_cur = df_vs if (tgs is None) else sf.subset_df(df_vs, {self.field_variable_trajgroup_type: tgs})
				dict_vs = {
					"max_scalar": sf.build_dict(df_cur[fields_id + [field_max]], force_tuple = True),
					"min_scalar": sf.build_dict(df_cur[fields_id + [field_min]], force_tuple = True),
					"uniform_scaling_q": sf.build_dict(df_cur[fields_id + [self.field_uniform_scaling_q]], force_tuple = True),
					"trajectories": sf.build_dict(df_cur[fields_id + fields_time_periods], (len(self.fields_id), len(self.fields_time_periods)), force_tuple = True)
				}

				dict_var_info.update({(vs, tgs): dict_vs})

		return dict_var_info



	# determine if the sampling unit represents a strategy (L) or an uncertainty (X)
	def _initialize_xl_type(self,
		thresh: float = (10**(-12))
	) -> Tuple[str, Dict[str, Any]]:
		"""
		Infer the sampling unit type--strategy (L) or an uncertainty (X)--by
			comparing variable specification trajectories across strategies.
			Sets the following properties:

			* self.fields_order_strat_diffs
			* self.xl_type, self.dict_strategy_info

		Keyword Arguments
		-----------------
		- thresh: threshold used to identify significant difference between
			variable specification trajectories across strategies. If a
			variable specification trajectory shows a difference of diff between
			any strategy of diff > thresh, it is defined to be a strategy.
 		"""

		# set some field variables
		fields_id_no_strat = [x for x in self.fields_id if (x != self.key_strategy)]
		fields_order_strat_diffs = fields_id_no_strat + [self.field_variable, self.field_variable_trajgroup_type]
		fields_merge = [self.field_variable, self.field_variable_trajgroup_type] + fields_id_no_strat
		fields_ext = fields_merge + self.fields_time_periods

		# some strategy distinctions -- baseline vs. non- baseline
		strat_base = self.dict_baseline_ids.get(self.key_strategy)
		strats_not_base = [x for x in self.dict_id_values.get(self.key_strategy) if (x != strat_base)]

		# pivot by strategy--sort by fields_order_strat_diffs for use in dict_strategy_info
		df_pivot = pd.pivot(
			self.data_table,
			fields_merge,
			[self.key_strategy],
			self.fields_time_periods
		).sort_values(by = fields_order_strat_diffs)
		fields_base = [(x, strat_base) for x in self.fields_time_periods]

		# get the baseline strategy specification + set a renaming dictionary for merges - pivot inclues columns names as indices, they resturn when calling to_flat_index()
		df_base = df_pivot[fields_base].reset_index()
		df_base.columns = [x[0] for x in df_base.columns.to_flat_index()]
		arr_base = np.array(df_pivot[fields_base])

		dict_out = {
			"baseline_strategy_data_table": df_base,
			"baseline_strategy_array": arr_base,
			"difference_arrays_by_strategy": {}
		}
		dict_diffs = {}
		strategy_q = False

		for strat in strats_not_base:

			arr_cur = np.array(df_pivot[[(x, strat) for x in self.fields_time_periods]])
			arr_diff = arr_cur - arr_base
			dict_diffs.update({strat: arr_diff})

			strategy_q = (max(np.abs(arr_diff.flatten())) > thresh) | strategy_q


		dict_out.update({"difference_arrays_by_strategy": dict_diffs}) if strategy_q else None
		type_out = "L" if strategy_q else "X"

		# set properties
		self.dict_strategy_info = dict_out
		self.fields_order_strat_diffs = fields_order_strat_diffs
		self.xl_type = type_out



	############################
	#    CORE FUNCTIONALITY    #
	############################

	def get_scalar_diff_arrays(self,
	) ->  None:
		"""
		Get the scalar difference arrays, which are arrays that are scaled
			and added to the baseline to represent changes in future
			trajectries. Sets the following properties:

			* self.
		"""

		tgs_loops = self.required_tg_specs if (self.variable_trajectory_group is not None) else [None]
		tp_end = self.fields_time_periods[-1]
		dict_out = {}

		for vs in self.variable_specifications:

			for tgs in tgs_loops:

				# get the vector (dim is by scenario, sorted by self.fields_id )
				vec_tp_end = self.ordered_trajectory_arrays.get((vs, tgs)).get("data")[:,-1]
				tups_id_coords = [tuple(x) for x in np.array(self.ordered_trajectory_arrays.get((vs, tgs))["id_coordinates"])]

				# order the max/min scalars
				vec_scale_max = np.array([self.dict_variable_info[(vs, tgs)]["max_scalar"][x] for x in tups_id_coords])
				vec_scale_min = np.array([self.dict_variable_info[(vs, tgs)]["min_scalar"][x] for x in tups_id_coords])

				# difference, in final time period, between scaled value and baseline value-dimension is # of scenarios
				dict_tp_end_delta = {
					"max_tp_end_delta": vec_tp_end*(vec_scale_max - 1),
					"min_tp_end_delta": vec_tp_end*(vec_scale_min - 1)
				}

				dict_out.update({(vs, tgs): dict_tp_end_delta})

		return dict_out



	def mix_tensors(self,
		vec_b0: np.ndarray,
		vec_b1: np.ndarray,
		vec_mix: np.ndarray,
		constraints_mix: tuple = (0, 1)
	) -> np.ndarray:

		v_0 = np.array(vec_b0)
		v_1 = np.array(vec_b1)
		v_m = np.array(vec_mix)

		if constraints_mix != None:
			if constraints_mix[0] >= constraints_mix[1]:
				raise ValueError("Constraints to the mixing vector should be passed as (min, max)")
			v_alpha = v_m.clip(*constraints_mix)
		else:
			v_alpha = np.array(vec_mix)

		if len(v_alpha.shape) == 0:
			v_alpha = float(v_alpha)
			check_val = len(set([v_0.shape, v_1.shape]))
		else:
			check_val = len(set([v_0.shape, v_1.shape, v_alpha.shape]))

		if check_val > 1:
			raise ValueError("Incongruent shapes in mix_tensors")

		return v_0*(1 - v_alpha) + v_1*v_alpha



	def ordered_by_ota_from_fid_dict(self, dict_in: dict, key_tuple: tuple):
		return np.array([dict_in[tuple(x)] for x in np.array(self.ordered_trajectory_arrays[key_tuple]["id_coordinates"])])


	## UNCERTAINY FAN FUNCTIONS

	 # construct the "ramp" vector for uncertainties
	def build_ramp_vector(self,
		tuple_param: Union[tuple, None] = None
	) -> np.ndarray:
		"""
		Convert tuple_param to a vector for characterizing uncertainty.

		Keyword Arguments
		-----------------
		- tuple_param: tuple of parameters to pass to f_fan

		"""
		tuple_param = self.get_f_fan_function_parameter_defaults(self.uncertainty_fan_function_type) if (tuple_param is None) else tuple_param

		if len(tuple_param) == 4:
			tp_0 = self.time_period_end_certainty
			n = len(self.time_periods) - tp_0 - 1

			return np.array([int(i > tp_0)*self.f_fan(i - tp_0 , n, *tuple_param) for i in range(len(self.time_periods))])
		else:
			raise ValueError(f"Error: tuple_param {tuple_param} in build_ramp_vector has invalid length. It should have 4 parameters.")


	# basic function that determines the shape; based on a generalization of the sigmoid (includes linear option)
	def f_fan(self, x, n, a, b, c, d):
		"""
		 *defaults*

		 for linear:
			set a = 0, b = 2, c = 1, d = n/2
		 for sigmoid:
			set a = 1, b = 0, c = math.e, d = n/2


		"""
		return (a*n + b*x)/(n*(1 + c**(d - x)))


	# parameter defaults for the fan, based on the number of periods n
	def get_f_fan_function_parameter_defaults(self,
		n: int,
		fan_type: str,
		return_type: str = "params"
	) -> list:

		dict_ret = {
			"linear": (0, 2, 1, n/2),
			"sigmoid": (1, 0, math.e, n/2)
		}

		if return_type == "params":
			return dict_ret.get(fan_type)
		elif return_type == "keys":
			return list(dict_ret.keys())
		else:
			str_avail_keys = ", ".join(list(dict_ret.keys()))
			raise ValueError(f"Error: invalid return_type '{return_type}'. Ensure it is one of the following: {str_avail_keys}.")



	# verify fan function parameters
	def _initialize_uncertainty_functional_form(self,
		fan_type: Union[str, Tuple[Union[float, int]]],
		default_fan_type: str = "linear"
	) -> None:
		"""
		Set function parameters surrounding fan function. Sets the following
			properties:

			* self.uncertainty_fan_function_parameters
			* self.uncertainty_fan_function_type
			* self.uncertainty_ramp_vector
			* self.valid_fan_type_strs

		Behavioral Notes
		----------------
		- Invalid types for fan_type will result in the default_fan_type.
		- The dead default (if default_fan_type is invalid as a keyword) is
			linear

		Function Arguments
		------------------
		- fan_type: string specifying fan type OR tuple (4 values) specifying
			arguments to

		Keyword Arguments
		-----------------
		- default_fan_type: default fan function to use to describe uncertainty
		"""

		self.uncertainty_fan_function_parameters = None
		self.uncertainty_fan_function_type = None
		self.uncertainty_ramp_vector = None
		self.valid_fan_type_strs = self.get_f_fan_function_parameter_defaults(0, "", return_type = "keys")

		default_fan_type = "linear" if (default_fan_type not in self.valid_fan_type_strs) else default_fan_type
		n_uncertain = len(self.time_periods) - self.time_period_end_certainty
		params = None

		# set to default if an invalid type is entered
		if not (isinstance(fan_type, str) or isinstance(fan_type, typle)):
			fan_type = default_fan_type

		if isinstance(fan_type, tuple):
			if len(fan_type) != 4:
				#raise ValueError(f"Error: fan parameter specification {fan_type} invalid. 4 Parameters are required.")
				fan_type = default_fan_type
			elif not all(set([isinstance(x, int) or isinstance(x, float) for x in fan_type])):
				#raise ValueError(f"Error: fan parameter specification {fan_type} contains invalid parameters. Ensure they are numeric (int or float)")
				fan_type = default_fan_type
			else:
				fan_type = "custom"
				params = fan_type

		# only implemented if not otherwise set
		if params is None:
			fan_type = default_fan_type if (fan_type not in self.valid_fan_type_strs) else fan_type
			params = self.get_f_fan_function_parameter_defaults(n_uncertain, fan_type, return_type = "params")

		# build ramp vector
		tp_0 = self.time_period_end_certainty
		n = n_uncertain - 1
		vector_ramp = np.array([int(i > tp_0)*self.f_fan(i - tp_0 , n, *params) for i in range(len(self.time_periods))])

		#
		self.uncertainty_fan_function_parameters = params
		self.uncertainty_fan_function_type = fan_type
		self.uncertainty_ramp_vector = vector_ramp



	def generate_future(self,
		lhs_trial_x: float,
		lhs_trial_l: float = 1.0,
		baseline_future_q: bool = False,
		constraints_mix_tg: tuple = (0, 1),
		flatten_output_array: bool = False,
		vary_q: Union[bool, None] = None
	) -> Dict[str, np.ndarray]:
		"""
		Generate a dictionary mapping each variable specification to futures ordered by self.ordered_trajectory_arrays((vs, tg))["id_coordinates"]

		Function Arguments
		------------------
		- lhs_trial_x: LHS trial used to generate uncertainty fan for base future

		Keyword Arguments
		------------------
		- lhs_trial_l: LHS trial used to modify strategy effect
		- baseline_future_q: generate a baseline future? If so, lhs trials do not apply
		- constraints_mix_tg: constraints on the mixing fraction for trajectory groups
		- flatten_output_array: return a flattened output array (apply np.flatten())
		- vary_q: does the future vary? if not, returns baseline
		"""

		vary_q = self.variable_trajectory_group_vary_q if not isinstance(vary_q, bool) else vary_q

		# clean up some cases for None entries
		baseline_future_q = True if (lhs_trial_x is None) else baseline_future_q
		lhs_trial_x = 1.0 if (lhs_trial_x is None) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l is None) else lhs_trial_l

		# some additional checks for potential negative numbers
		baseline_future_q = True if (lhs_trial_x < 0) else baseline_future_q
		lhs_trial_x = 1.0 if (lhs_trial_x < 0) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l < 0) else lhs_trial_l

		# set to baseline if not varying
		baseline_future_q = baseline_future_q | (not vary_q)

		# initialization
		all_strats = self.dict_id_values.get(self.key_strategy)
		n_strat = len(all_strats)
		strat_base = self.dict_baseline_ids.get(self.key_strategy)

		# index by variable_specification at keys
		dict_out = {}

		if self.variable_trajectory_group is not None:
			#list(set([x[0] for x in self.ordered_trajectory_arrays.keys()]))
			cat_mix = self.dict_required_tg_spec_fields.get("mixing_trajectory")
			cat_b0 = self.dict_required_tg_spec_fields.get("trajectory_boundary_0")
			cat_b1 = self.dict_required_tg_spec_fields.get("trajectory_boundary_1")

			# use mix between 0/1 (0 = 100% trajectory_boundary_0, 1 = 100% trajectory_boundary_1)
			for vs in self.variable_specifications:

				dict_ordered_traj_arrays = self.ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))

				dict_arrs = {
					cat_b0: self.ordered_trajectory_arrays[(vs, cat_b0)].get("data"),
					cat_b1: self.ordered_trajectory_arrays[(vs, cat_b1)].get("data"),
					cat_mix: self.ordered_trajectory_arrays[(vs, cat_mix)].get("data")
				}

				# for trajectory groups, the baseline is the specified mixing vector
				mixer = dict_arrs[cat_mix] if baseline_future_q else lhs_trial_x
				arr_out = self.mix_tensors(dict_arrs[cat_b0], dict_arrs[cat_b1], mixer, constraints_mix_tg)

				if self.xl_type == "L":
					#
					# if the XL is an L, then we use the modified future as a base (reduce to include only baseline strategy), then add the uncertainty around the strategy effect
					#
					# get id coordinates( any of cat_mix, cat_b0, or cat_b1 would work -- use cat_mix)
					df_ids_ota = pd.concat([
						self.ordered_trajectory_arrays.get((vs, cat_mix))["id_coordinates"].copy().reset_index(drop = True),
						pd.DataFrame(arr_out, columns = self.fields_time_periods)],
						axis = 1
					)
					w = np.where(df_ids_ota[self.key_strategy] == strat_base)
					df_ids_ota = df_ids_ota.iloc[w[0].repeat(n_strat)].reset_index(drop = True)

					arr_out = np.array(df_ids_ota[self.fields_time_periods])
					arrs_strategy_diffs = self.dict_strategy_info.get("difference_arrays_by_strategy")
					df_baseline_strategy = self.dict_strategy_info.get("baseline_strategy_data_table")
					inds0 = set(np.where(df_baseline_strategy[self.field_variable] == vs)[0])
					l_modified_cats = []

					for cat_cur in [cat_b0, cat_b1, cat_mix]:

						# get the index for the current vs/cat_cur
						inds = np.sort(np.array(list(inds0 & set(np.where(df_baseline_strategy[self.field_variable_trajgroup_type] == cat_cur)[0]))))
						n_inds = len(inds)
						df_ids0 = df_baseline_strategy[[x for x in self.fields_id if (x != self.key_strategy)]].loc[inds.repeat(n_strat)].reset_index(drop = True)
						new_strats = list(np.zeros(len(df_ids0)).astype(int))

						# initialize as list - we only do this to guarantee the sort is correct
						df_future_strat = np.zeros((n_inds*n_strat, len(self.fields_time_periods)))
						ind_repl = 0

						# iterate over strategies
						for strat in all_strats:
							# replace strategy ids
							new_strats[ind_repl*n_inds:((ind_repl + 1)*n_inds)] = [strat for x in inds]
							# get the strategy difference that is adjusted by lhs_trial_x_delta; if baseline strategy, use 0s
							df_repl = np.zeros((n_inds, len(self.fields_time_periods))) if (strat == strat_base) else arrs_strategy_diffs[strat][inds, :]*lhs_trial_l
							np.put(
								df_future_strat,
								range(
									n_inds*len(self.fields_time_periods)*ind_repl,
									n_inds*len(self.fields_time_periods)*(ind_repl + 1)
								),
								df_repl
							)
							ind_repl += 1

						df_ids0[self.key_strategy] = new_strats
						df_future_strat = pd.concat([df_ids0, pd.DataFrame(df_future_strat, columns = self.fields_time_periods)], axis = 1).sort_values(by = self.fields_id).reset_index(drop = True)
						l_modified_cats.append(dict_arrs[cat_cur] + np.array(df_future_strat[self.fields_time_periods]))

					arr_out = self.mix_tensors(*l_modified_cats, constraints_mix_tg)

				# to compare the difference between the "L" design uncertainty and the baseline and add this to the uncertain future (final array)
				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		else:

			rv = self.uncertainty_ramp_vector

			for vs in self.variable_specifications:

				dict_ordered_traj_arrays = self.ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))

				# order the uniform scaling by the ordered trajectory arrays
				vec_unif_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["uniform_scaling_q"], (vs, None))
				# gives 1s where we keep standard fanning (using the ramp vector) and 0s where we use uniform scaling
				vec_base = 1 - vec_unif_scalar
				#
				if max(vec_unif_scalar) > 0:
					vec_max_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["max_scalar"], (vs, None))
					vec_min_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["min_scalar"], (vs, None))
					vec_unif_scalar = vec_unif_scalar*(vec_min_scalar + lhs_trial_x*(vec_max_scalar - vec_min_scalar)) if not baseline_future_q else np.ones(vec_unif_scalar.shape)

				vec_unif_scalar = np.array([vec_unif_scalar]).transpose()
				vec_base = np.array([vec_base]).transpose()

				delta_max = dict_scalar_diff_arrays.get("max_tp_end_delta")
				delta_min = dict_scalar_diff_arrays.get("min_tp_end_delta")
				delta_diff = delta_max - delta_min
				delta_val = delta_min + lhs_trial_x*delta_diff

				# delta and uniform scalar don't apply if operating under baseline future
				delta_vec = 0.0 if baseline_future_q else (rv * np.array([delta_val]).transpose())

				arr_out = dict_ordered_traj_arrays.get("data") + delta_vec
				arr_out = arr_out*vec_base + vec_unif_scalar*dict_ordered_traj_arrays.get("data")

				if self.xl_type == "L":
					# get series of strategies
					series_strats = list(dict_ordered_traj_arrays.get("id_coordinates")[self.key_strategy])
					w = np.where(np.array(series_strats) == strat_base)[0]
					# get strategy adjustments
					lhs_mult_deltas = 1.0 if baseline_future_q else lhs_trial_l
					vec_alt = np.zeros((1, len(self.time_periods)))
					array_strat_deltas = np.concatenate(
						[
							self.dict_strategy_info["difference_arrays_by_strategy"].get(x, vec_alt)
							for x in series_strats
						]
					)
					#array_strat_deltas = np.concatenate(
					#	series_strats.apply(
					#		self.dict_strategy_info["difference_arrays_by_strategy"].get,
					#		args = (np.zeros((1, len(self.time_periods))), )
					#	)
					#)*lhs_mult_deltas

					array_strat_deltas *= lhs_mult_deltas
					arr_out = (array_strat_deltas + arr_out[w, :]) if (len(w) > 0) else arr_out

				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		return dict_out












