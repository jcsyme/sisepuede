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
class SamplingUnit:
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
	- check_duplicates_in_variable_definition: check for duplicate rows in the
		input `df_variable_definition`? If False, skips checking.
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
	- group: optional group specification for tracking collections of 
		SamplingUnits
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
		check_duplicates_in_variable_definition: bool = True,
		fan_function_specification: str = "linear",
		field_time_period: str = "time_period",
		field_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		group: Union[int, None] = None,
		key_strategy: str = "strategy_id",
		missing_trajgroup_flag: int = -999,
		regex_id: re.Pattern = re.compile("(\D*)_id$"),
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)"),
	):

		# perform initializations
		self._initialize_parameters(missing_trajgroup_flag)
		self._initialize_base_fields_and_keys(
			field_time_period,
			field_trajgroup_no_vary_q,
			field_uniform_scaling_q,
			field_variable_trajgroup,
			field_variable_trajgroup_type,
			field_variable,
			key_strategy,
		)
		self._initialize_time_start_uncertainty(time_period_u0)
		self._initialize_attributes_from_table(
			df_variable_definition,
			regex_id,
			regex_max,
			regex_min,
			regex_tp,
			check_duplicates = check_duplicates_in_variable_definition,
		)
		self._initialize_properties(
			group = group,
		)
		self._initialize_uncertainty_functional_form(fan_function_specification)
		self._initialize_scenario_variables(dict_baseline_ids)
		self._initialize_variable_dictionaries()






	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	def check_input_data_frame(self,
		df_in: pd.DataFrame,
		drop_duplicates: bool = True,
		fields_req: Union[List[str], None] = None,
		field_req_variable_trajectory_group_trajectory_type: Union[str, None] = None
	) -> pd.DataFrame:
		"""
		Check df_in for required fields. Returns a data frame with variable
			definitions.

		Function Arguments
		------------------
		- df_in: data frame to check

		Keyword Arguments
		-----------------
		- drop_duplicates: drop duplicate rows? Set to False if input DataFrame
			is assured to contain unique rows (can be faster in batch
			implentation)
		- field_req_variable_trajectory_group_trajectory_type: field used to
			denote the variable trajectory group trajectory type
		- fields_req: fields that the dataframe is required to contain
		"""
		# some standardized fields to require
		field_req_variable_trajectory_group_trajectory_type = (
			self.field_variable_trajgroup_type 
			if (field_req_variable_trajectory_group_trajectory_type is None) 
			else field_req_variable_trajectory_group_trajectory_type
		)
		fields_req = self.required_fields if fields_req is None else fields_req

		# raise an error if any required fields are missing
		if not set(fields_req).issubset(set(df_in.columns)):
			fields_missing = list(set(fields_req) - (set(fields_req) & set(df_in.columns)))
			fields_missing.sort()
			str_missing = ", ".join([f"'{x}'" for x in fields_missing])
			raise ValueError(f"Error: one or more columns are missing from the data frame. Columns {str_missing} not found")

		# some cleaning
		df_out = df_in.drop_duplicates() if drop_duplicates else df_in
		df_out[field_req_variable_trajectory_group_trajectory_type].replace({np.nan: None}, inplace = True)

		return df_out



	def _initialize_time_start_uncertainty(self,
		t0: int
	) -> None:
		"""
		Initialize the following properties:

			* self.time_period_end_certainty

		Function Arguments
		------------------
		- t0: input integer secifying start time for uncertainty
		"""
		self.time_period_end_certainty = max(t0, 1)

		return None



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
			* If None, default to self.df_coordinates_id
		- dict_additional_fields: dictionary mapping additional fields to values to add
			* If None, no additional fields are added
		- field_primary_key_id_coords: field in df_id_coords denoting the primary key
			* If None, default to self.primary_key_id_coordinates
		- field_time_period: field to use for data frame
			* If None, default to self.field_time_period
		"""

		df_id_coords = self.df_coordinates_id if (df_id_coords is None) else df_id_coords
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
					len(self.df_coordinates_id),
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



	def get_coords(self,
		field_ans_key: Union[str, None] = None,
		field_id_key: Union[str, None] = None,
		field_vvt_key: Union[str, None] = None,
		fields_ans: Union[list, None] = None,
		fields_id: Union[list, None] = None,
		fields_vvt: Union[list, None] = None,
		tups_id: Union[List[Tuple], None] = None,
		tups_vvt: Union[List[Tuple], None] = None,
	) -> List[Tuple]:
		"""
		Get the coordinates for sorting by strategy (used in strategy info).
			Returns a tuple with the following elements:

			df_ans_key, df_id_key, df_vvt_key, tups_ans, tups_id, tups_vvt

		Keyword Arguments
		-----------------
		- fields_ans: fields used to define coordinates_ans
		- fields_id: id fields included in df_in
		- fields_vvt: variable and variable trajgroup type fields, used to index
			arrays for comparison against base strategy.
		- field_ans_key: field used as a key for direct product of all
			field dimensions without strategy
		- field_id_key: field used as a key for direct product of ID values
		- field_vvt_key: field used as a key for direct product of variable and
			variable trajgroup type fields combinations
		- tups_id: ordered list of ID coordinates as tuples (ordered by
			self.fields_id)
		- tups_vvt: ordered list of VVT coordinates as tuples (ordered by
			self.fields_vvt_coodinates)
		"""

		field_ans_key = self.primary_key_ans_coordinates if (field_ans_key is None) else field_ans_key
		field_id_key = self.primary_key_id_coordinates if (field_id_key is None) else field_id_key
		field_vvt_key = self.primary_key_vvt_coordinates if (field_vvt_key is None) else field_vvt_key
		fields_ans = self.fields_ans_coordinates if not isinstance(fields_ans, list) else fields_ans
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		fields_vvt = self.fields_vvt_coordinates if not isinstance(fields_vvt, list) else fields_vvt

		##  BUILD ANS (ALL NO-STRAT) COORDS

		# ensure sorting of input ID/VVT coords
		tups_id = sorted(list(tups_id))
		tups_vvt = sorted(list(tups_vvt))
		n_vvt = len(tups_vvt)

		# initialize components for
		ind_filt = self.fields_id.index(self.key_strategy) # tuple index to filter in tups_id
		tups_ans = []
		tups_drop = []

		# initialize iterator vars
		ind_id = 0

		# iterate over tups_id: drop strategy index, then expand against tups_vvt. If already done, skip to next row
		for tup in enumerate(tups_id):
			i, tup = tup
			tup_keep = sf.filter_tuple(tup, ind_filt)

			if tup_keep not in tups_drop:
				tups_ans += [tup_keep + x for x in tups_vvt]
				tups_drop.append(tup_keep)


		##  GET ID COORDINATES DATAFRAMES

		df_id_key = pd.DataFrame(tups_id, columns = fields_id)
		df_id_key[field_id_key] = range(len(df_id_key))
		dict_id_key = dict(zip(tups_id, range(len(df_id_key))))
		def fun_id_key(
			tup_in: Union[Tuple, np.ndarray, List]
		) -> Union[int, None]:
			return dict_id_key.get(tuple(tup_in))

		df_vvt_key = pd.DataFrame(tups_vvt, columns = fields_vvt)
		df_vvt_key[field_vvt_key] = range(len(df_vvt_key))
		dict_vvt_key = dict(zip(tups_vvt, range(len(df_vvt_key))))
		def fun_vvt_key(
			tup_in: Union[Tuple, np.ndarray, List]
		) -> Union[int, None]:
			return dict_vvt_key.get(tuple(tup_in))

		df_ans_key = pd.DataFrame(tups_ans, columns = fields_ans)
		df_ans_key[field_ans_key] = range(len(df_ans_key))
		dict_ans_key = dict(zip(tups_ans, range(len(df_ans_key))))
		def fun_ans_key(
			tup_in: Union[Tuple, np.ndarray, List]
		) -> Union[int, None]:
			return dict_ans_key.get(tuple(tup_in))

		# organize output tuple
		out = (
			df_ans_key, 
			df_id_key, 
			df_vvt_key, 
			fun_ans_key, 
			fun_id_key, 
			fun_vvt_key, 
			tups_ans, 
			tups_id, 
			tups_vvt
		)

		return out



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



	def get_scalar_time_period(self,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		df_in:Union[pd.DataFrame, None] = None
	) -> Tuple[str, str, int]:
		"""
		Determine final time period (tp_final) as well as the fields associated 
			with the minimum and maximum scalars (field_min/field_max) using 
			input template df_in. Returns a tuple with the following elements:

			* field_min
			* field_max
			* tp_final

		Function Arguments
		------------------
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum 
			scalar values at the final time period

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.df_variable_definitions
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		field_min = [x for x in df_in.columns if (regex_min.match(x) is not None)]
		if len(field_min) == 0:
			raise ValueError("No field associated with a minimum scalar value found in data frame.")

		field_min = field_min[0]

		# determine max field/time period
		field_max = [x for x in df_in.columns if (regex_max.match(x) is not None)]
		if len(field_max) == 0:
			raise ValueError("No field associated with a maximum scalar value found in data frame.")
			
		field_max = field_max[0]

		tp_min = int(field_min.split("_")[1])
		tp_max = int(field_max.split("_")[1])
		if (tp_min != tp_max) | (tp_min is None):
			raise ValueError(f"Fields '{tp_min}' and '{tp_max}' imply asymmetric final time periods.")

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

		df_in = self.df_variable_definitions if not isinstance(df_in, pd.DataFrame) else df_in
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



	def get_strategy_to_coordinate_rows_dict(self,
		df_coordinates_id: Union[pd.DataFrame, None] = None,
		dict_id_values: Union[Dict, None] = None,
		key_strategy: Union[str, None] = None
	) -> Dict[str, List[int]]:
		"""
		Build a dictionary mapping strategy key values to row indices in
			self.df_coordinates_id.

		Keyword Arguments
		-----------------
		- df_coordinates_id: data frame of id coordinates to use as reference
		- key_strategy: strategy key
		"""

		df_coordinates_id = self.df_coordinates_id if (df_coordinates_id is None) else df_coordinates_id
		dict_id_values = self.dict_id_values if (dict_id_values is None) else dict_id_values
		key_strategy = self.key_strategy if (key_strategy is None) else field_key_strategy

		dict_strategy_id_to_coordinate_rows = dict((x, []) for x in dict_id_values.get(key_strategy))
		vec_strats_in_df_coords = list(df_coordinates_id[key_strategy])
		for i in range(len(df_coordinates_id)):
			strat = vec_strats_in_df_coords[i]
			dict_strategy_id_to_coordinate_rows[strat].append(i)

		return dict_strategy_id_to_coordinate_rows



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
		df_in:Union[pd.DataFrame, None] = None,
	) -> Union[int, None]:
		"""
		Get the trajectory group for the sampling unit from df_in.

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications.
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in
		if not self.field_variable_trajgroup in df_in.columns:
			raise ValueError(f"Field '{self.field_variable_trajgroup}' not found in data frame.")

		# determine if this is associated with a trajectory group		
		out = (
			(
				int(list(df_in[self.field_variable_trajgroup].unique())[0])
				if len(self.get_all_vs(df_in)) > 1
				else None
			)
			if len(df_in[df_in[self.field_variable_trajgroup] > self.missing_trajgroup_flag]) > 0
			else None
		)

		return out



	def get_trajgroup_vary_q(self,
		df_in:Union[pd.DataFrame, None] = None,
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



	def _initialize_attributes_from_table(self,
		df_variable_definition: pd.DataFrame,
		regex_id: re.Pattern,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		regex_tp: re.Pattern,
		check_duplicates: bool = True,
	) -> None:
		"""
		Set a range of attributes derived the input df_variable_definition.
			Sets the following properties:

			* self.df_variable_definitions
			* self.fields_ans_coordinates
			* self.fields_id
			* self.fields_id_no_strat
			* self.required_fields

		Function Arguments
		------------------
		- df_variable_definition: data frame used to set variable specifications
		- regex_id: regular expression used to identify id fields in the input
			template
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum
			scalar values at the final time period
		- regex_tp: re.Pattern used to match the field storing data values for
			each time period

		Keyword Arguments
		-----------------
		- check_duplicates: check for duplicate rows in
			`df_variable_definition`?
		"""
		self.required_fields = [
			self.key_strategy,
			self.field_trajgroup_no_vary_q,
			self.field_variable_trajgroup,
			self.field_variable_trajgroup_type,
			self.field_uniform_scaling_q,
			self.field_variable
		]

		self.df_variable_definitions = self.check_input_data_frame(
			df_variable_definition,
			drop_duplicates = check_duplicates,
		)

		self.fields_id = self.get_id_fields(regex_id)
		self.fields_id_no_strat = [x for x in self.fields_id if (x != self.key_strategy)]
		self.fields_ans_coordinates = self.fields_id_no_strat + self.fields_vvt_coordinates

		self.field_min_scalar, self.field_max_scalar, self.time_period_scalar = self.get_scalar_time_period(
			regex_max,
			regex_min
		)
		self.fields_time_periods, self.time_periods = self.get_time_periods(regex_tp)
		self.variable_trajectory_group = self.get_trajgroup()
		self.variable_trajectory_group_vary_q = self.get_trajgroup_vary_q()

		return None
 


	def _initialize_base_fields_and_keys(self,
		field_time_period: str,
		field_trajgroup_no_vary_q: str,
		field_uniform_scaling_q: str,
		field_variable_trajgroup: str,
		field_variable_trajgroup_type: str,
		field_variable: str,
		key_strategy: str,
	) -> None:
		"""
		Initialize some fields and keys. Sets the following properties:

			* self.field_time_period
			* self.field_trajgroup_no_vary_q
			* self.field_uniform_scaling_q
			* self.field_variable_trajgroup
			* self.field_variable_trajgroup_type
			* self.field_variable
			* self.key_strategy
			* self.fields_vvt_coordinates
		"""

		self.field_time_period = field_time_period
		self.field_trajgroup_no_vary_q = field_trajgroup_no_vary_q
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_variable = field_variable
		self.key_strategy = key_strategy

		# set fields for sort ordering
		self.fields_vvt_coordinates = [
			self.field_variable,
			self.field_variable_trajgroup_type
		]

		return None



	def initialize_dict_sda(self,
		array_data: np.ndarray,
		vec_scale_max: np.ndarray,
		vec_scale_min: np.ndarray
	) -> Dict[str, Dict]:
		"""
		Initialize a dictionary for scalar diff arrays
		"""
		dict_out = {
			"max_tp_end_delta": array_data[:,-1]*(vec_scale_max - 1),
			"min_tp_end_delta": array_data[:,-1]*(vec_scale_min - 1),
		}

		return dict_out



	def initialize_dict_vi(self,
	) -> Dict[str, Dict]:
		"""
		Initialize the dictionary of variable information
		"""
		dict_out = {
			"max_scalar": {},
			"min_scalar": {},
			"trajectories": {},
			"uniform_scaling_q": {}
		}

		return dict_out




	def _initialize_parameters(self,
		missing_trajgroup_flag: int,
	) -> None:
		"""
		Set some key parameters.

			* self.dict_required_tg_spec_fields
			* self.key_mix_trajectory
			* self.key_inf_traj_boundary
			* self.key_sup_traj_boundary
			* self.missing_trajgroup_flag
			* self.primary_key_ans_coordinates
			* self.primary_key_id_coordinates
			* self.primary_key_vvt_coordinates
			* self.required_tg_specs

		Function Arguments
		------------------
		- missing_trajgroup_flag: flag to use for missing trajectoy group in
			self.df_variable_definitions
		"""

		self.key_mix_trajectory = "mixing_trajectory"
		self.key_inf_traj_boundary = "trajectory_boundary_0"
		self.key_sup_traj_boundary = "trajectory_boundary_1"
		self.missing_trajgroup_flag = missing_trajgroup_flag
		self.primary_key_ans_coordinates = "primary_key_ans_coord"
		self.primary_key_id_coordinates = "primary_key_id_coord"
		self.primary_key_vvt_coordinates = "primary_key_vvt_coord"

		# maps internal name (key) to classification in the input data frame (value)
		self.dict_required_tg_spec_fields = {
			self.key_mix_trajectory: "mix",
			self.key_inf_traj_boundary: "trajectory_boundary_0",
			self.key_sup_traj_boundary: "trajectory_boundary_1"
		}
		self.required_tg_specs = list(self.dict_required_tg_spec_fields.values())
		
		return None



	def _initialize_scenario_variables(self,
		dict_baseline_ids: Dict[str, int],
		df_in: Union[pd.DataFrame, None] = None,
		fields_ans: Union[list, None] = None,
		fields_id: Union[list, None] = None,
		fields_vvt: Union[list, None] = None,
		field_ans_key: Union[str, None] = None,
		field_id_key: Union[str, None] = None,
		field_vvt_key: Union[str, None] = None
	) -> None:
		"""
		Check inputs of the input data frame and id fields. Sets the following
			properties:

			* self.coordinates_ans
			* self.coordinates_id
			* self.coordinates_vvt
			* self.df_coordinates_ans
			* self.df_coordinates_id
			* self.df_coordinates_vvt
			* self.dict_baseline_ids
			* self.dict_id_values
			* self.dict_variable_info
			* self.num_scenarios
			* self.variable_specifications

		Updates the following properties after previous initialization:

			* self.df_variable_definitions

		Function Arguments
		------------------
		- dict_baseline_ids: dictionary mapping each dimensional key to nominal
			baseline

		Keyword Arguments
		-----------------
		- df_in: input data frame used to specify variables
		- fields_ans: fields used to define coordinates_ans
		- fields_id: id fields included in df_in
		- fields_vvt: variable and variable trajgroup type fields, used to index
			arrays for comparison against base strategy.
		- field_ans_key: field used as a key for direct product of all
			field dimensions without strategy
		- field_id_key: field used as a key for direct product of ID values
		- field_vvt_key: field used as a key for direct product of variable and
			variable trajgroup type fields combinations
		"""
		t0 = time.time()
		df_in = self.df_variable_definitions if not isinstance(df_in, pd.DataFrame) else df_in
		field_ans_key = self.primary_key_ans_coordinates if (field_ans_key is None) else field_ans_key
		field_id_key = self.primary_key_id_coordinates if (field_id_key is None) else field_id_key
		field_vvt_key = self.primary_key_vvt_coordinates if (field_vvt_key is None) else field_vvt_key
		fields_ans = self.fields_ans_coordinates if not isinstance(fields_ans, list) else fields_ans
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		fields_vvt = self.fields_vvt_coordinates if not isinstance(fields_vvt, list) else fields_vvt
		#
		#arr_filt = np.array(df_in[fields_id + fields_vvt])
		#tups_id = set([tuple(x) for x in arr_filt[:,0:len(fields_id)]])
		tups_id = set(sf.df_to_tuples(df_in[fields_id], nan_to_none = True))
		tups_vvt = set(sf.df_to_tuples(df_in[fields_vvt], nan_to_none = True))

		#tups_vvt = set([tuple(x) for x in arr_filt[:,len(fields_id):arr_filt.shape[1]]])
		check_tups = len(set(self.required_tg_specs) & set(df_in[self.field_variable_trajgroup_type])) > 0

		# filter id tuples to only include thse that are defined for all groups of coordinates
		if check_tups:

			# check id tuples
			dfs_ids_intersect = df_in.groupby(fields_vvt)
			tups_add = []
			for df in dfs_ids_intersect:
				index, df = df
				tups_add += sf.df_to_tuples(df[fields_id], nan_to_none = True)#[tuple(x) for x in np.array(df[fields_id])]
			tups_id = tups_id & set(tups_add)

			# check vvt tuples
			dfs_vvt_intersect = df_in.groupby(fields_id)
			tups_add = []
			for df in dfs_vvt_intersect:
				index, df = df
				tups_add += sf.df_to_tuples(df[fields_vvt], nan_to_none = True)
			tups_vvt = tups_vvt & set(tups_add)

		t_elapse = sf.get_time_elapsed(t0, n_digits = 4)
		#print(f"mark isv_1: {t_elapse}")

		(
			df_ans_key,
			df_id_key,
			df_vvt_key,
			fun_ans_key,
			fun_id_key,
			fun_vvt_key,
			tups_ans,
			tups_id,
			tups_vvt
		) = self.get_coords(
			field_ans_key = field_ans_key,
			field_id_key = field_id_key,
			field_vvt_key = field_vvt_key,
			fields_ans = fields_ans,
			fields_id = fields_id,
			fields_vvt = fields_vvt,
			tups_id = tups_id,
			tups_vvt = tups_vvt
		)

		t_elapse = sf.get_time_elapsed(t0, n_digits = 4)
		#print(f"mark isv_2: {t_elapse}")
		df_in[field_ans_key] = df_in[fields_ans].apply(fun_ans_key, raw = True, axis = 1)
		df_in[field_id_key] = df_in[fields_id].apply(fun_id_key, raw = True, axis = 1)
		df_in[field_vvt_key] = df_in[fields_vvt].apply(fun_vvt_key, raw = True, axis = 1)
		df_in.sort_values(
			by = [field_id_key, field_vvt_key], inplace = True # CHECK THIS SORT,
		)
		df_in.reset_index(
			drop = True, inplace = True
		)

		t_elapse = sf.get_time_elapsed(t0, n_digits = 4)
		#print(f"mark isv_5: {t_elapse}")
		# id values and baseline ids
		dict_id_values, dict_baseline_ids = self.get_scenario_values(
			dict_baseline_ids,
			df_in = df_in,
			fields_id = fields_id
		)



		var_specs = self.get_all_vs(df_in)
		t_elapse = sf.get_time_elapsed(t0, n_digits = 4)

		self.coordinates_ans = tups_ans
		self.coordinates_id = tups_id
		self.coordinates_vvt = tups_vvt
		self.df_variable_definitions = df_in
		self.df_coordinates_ans = df_ans_key
		self.df_coordinates_id = df_id_key
		self.df_coordinates_vvt = df_vvt_key
		self.dict_baseline_ids = dict_baseline_ids
		self.dict_id_values = dict_id_values
		self.fun_ans_key = fun_ans_key
		self.fun_id_key = fun_id_key
		self.fun_vvt_key = fun_vvt_key
		self.num_scenarios = len(tups_id)
		self.variable_specifications = var_specs

		return None



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

		return None



	def _initialize_variable_dictionaries(self,
		df_in: Union[pd.DataFrame, None] = None,
		field_ans_key: Union[str, None] = None,
		field_max: Union[str, None] = None,
		field_min: Union[str, None] = None,
		field_uniform_scaling: Union[str, None] = None,
		fields_ans: Union[List, None] = None,
		fields_id: Union[List, None] = None,
		fields_time_periods: Union[List, None] = None,
		fields_vvt: Union[List, None] = None,
		key_strategy: Union[str, None] = None,
		thresh: float = (10**(-12)),
		variable_specifications: Union[List, None] = None
	) -> None:

		"""
		Retrieve a dictionary mapping a vs, tg pair to a list of information.
			The variable dictionary includes information on sampling ranges for
			time period scalars, wether the variables should be scaled
			uniformly, and the trajectories themselves. Sets the following
			proprties:

			* self.dict_ordered_trajectory_arrays
			* self.dict_scalar_diff_arrays
			* self.dict_strategy_info
			* self.dict_variable_info
			* self.xl_type

		Function Arguments
		------------------
		- df_in: input data frame containing data used to set key scenario v
			ariables

		Keyword Arguments
		-----------------
		- field_ans_key: field used as a key for direct product of all
			field dimensions without strategy
		- field_id_key: field used as a key for direct product of ID values
		- field_max: field giving the max scalar for uncertainty in the final time
			period
		- field_min: field giving the min scalar for uncertainty in the final time
		period
		- field_uniform_scaling: field giving the uniform scalar in df_in
		- field_vvt_key: field used as a key for direct product of variable and
			variable trajgroup type fields combinations
		- fields_ans: fields used to define coordinates_ans
		- fields_id: id fields included in df_in
		- fields_time_periods: fields denoting time periods
		- fields_vvt: variable and variable trajgroup type fields, used to index
			arrays for comparison against base strategy.
		- thresh: threshold used to identify significant difference between
			variable specification trajectories across strategies. If a
			variable specification trajectory shows a difference of diff between
			any strategy of diff > thresh, it is defined to be a strategy.

		"""

		# check and initialize input variables
		df_in = self.df_variable_definitions if (df_in is None) else df_in
		field_ans_key = self.primary_key_ans_coordinates if (field_ans_key is None) else field_ans_key
		field_max = self.field_max_scalar if (field_max is None) else field_max
		field_min = self.field_min_scalar if (field_min is None) else field_min
		field_uniform_scaling = self.field_uniform_scaling_q if (field_uniform_scaling is None) else field_uniform_scaling
		fields_ans = self.fields_ans_coordinates if (fields_ans is None) else fields_ans
		fields_id = self.fields_id if (fields_id is None) else fields_id
		fields_vvt = self.fields_vvt_coordinates if (fields_vvt is None) else fields_vvt
		fields_time_periods = self.fields_time_periods if (fields_time_periods is None) else fields_time_periods
		key_strategy = self.key_strategy if (key_strategy is None) else key_strategy
		variable_specifications = self.variable_specifications if (variable_specifications is None) else variable_specifications

		# initialize data frame grouping for iteration
		fields_group = fields_vvt if (self.variable_trajectory_group is not None) else [self.field_variable]

		# initialize arrays of indices
		arr_tups_ans = sf.df_to_tuples(df_in[fields_ans])
		arr_tups_id = sf.df_to_tuples(df_in[fields_id])
		arr_tups_vvt = [tuple(x) for x in np.array(df_in[fields_group])] if (len(fields_group) > 1) else [tuple(np.append(x, None)) for x in np.array(df_in[fields_group])]

		# initialize arrays of data
		arr_time_periods = np.array(df_in[fields_time_periods])
		vec_max_scalar = np.array(df_in[field_max])
		vec_min_scalar = np.array(df_in[field_min])
		vec_uniform_scaling = np.array(df_in[field_uniform_scaling])

		# initialize variable output templates
		array_base_ans = np.zeros((len(self.df_coordinates_ans), len(fields_time_periods)))
		array_base_id = np.zeros((len(self.df_coordinates_id), len(fields_time_periods)))
		array_base_vvt = np.zeros((len(self.df_coordinates_vvt), len(fields_time_periods)))
		vec_base_id = np.zeros(len(self.df_coordinates_id))

		# initialize some strategy info
		strat_base = self.dict_baseline_ids.get(self.key_strategy)
		strats_not_base = [x for x in self.dict_id_values.get(self.key_strategy) if (x != strat_base)]
		ind_strat = fields_id.index(key_strategy)

		if ind_strat is None:
			raise RuntimeError(f"Error in _initialize_variable_dictionaries: Strategy key '{key_strategy}' not found in fields_id.")

		# initialize dictionaries
		dict_ordered_traj_arrays = {}
		dict_ordered_traj_arrays_by_ans = {}
		dict_ordered_vec_max_scalars = {}
		dict_ordered_vec_min_scalars = {}
		dict_scalar_diff_arrays = {}
		dict_strategy_info = {}
		dict_var_info = {}

		# indexing dictionaries (temp)
		dict_iter_ind_id = {}
		dict_iter_ind_vvt = {}

		# initialize strategy base array
		df_base = (
			df_in[
				df_in[key_strategy] == strat_base
			][
				fields_vvt + fields_time_periods + [field_ans_key]
			]
			.sort_values(by = [field_ans_key])
			.reset_index(drop = True)
			.drop([field_ans_key], axis = 1)
		)
		dict_strategy_info.update({"baseline_strategy_data_table": df_base})

		global dict_vi2
		dict_vi2 = None
		# iterate over rows to assign outputs to dictionaries
		for i in range(len(df_in)):

			#
			tup_ans = arr_tups_ans[i]
			tup_id = arr_tups_id[i]
			tup_vvt = arr_tups_vvt[i]
			vs, tgs = tup_vvt
			id_strat = tup_id[ind_strat]

			# get row indices in arrays (vvt arrays are those that are GROUPED by ID [and represent a given combination of IDs, etc])
			ind_ans = self.fun_ans_key(tup_ans)
			ind_id = self.fun_id_key(tup_id) # for fixed V/VTT
			ind_vvt = self.fun_vvt_key(tup_vvt) # for fixed ID

			
			# initialize dictionary components
			(
				dict_var_info.update({tup_vvt: self.initialize_dict_vi()}) 
				if (dict_var_info.get(tup_vvt) is None) 
				else None
			)
			(
				dict_ordered_traj_arrays.update({tup_vvt: array_base_id.copy()}) 
				if (dict_ordered_traj_arrays.get(tup_vvt) is None) 
				else None
			)
			(
				dict_ordered_traj_arrays_by_ans.update({id_strat: array_base_ans.copy()}) 
				if (dict_ordered_traj_arrays_by_ans.get(id_strat) is None) 
				else None
			)
			(
				dict_ordered_vec_max_scalars.update({tup_vvt: vec_base_id.copy()}) 
				if (dict_ordered_vec_max_scalars.get(tup_vvt) is None) 
				else None
			)
			(
				dict_ordered_vec_min_scalars.update({tup_vvt: vec_base_id.copy()}) 
				if (dict_ordered_vec_min_scalars.get(tup_vvt) is None) 
				else None
			)

			# update variable info components
			dict_var_info[tup_vvt]["max_scalar"].update({tup_id: vec_max_scalar[i]})
			dict_var_info[tup_vvt]["min_scalar"].update({tup_id: vec_min_scalar[i]})
			dict_var_info[tup_vvt]["trajectories"].update({tup_id: arr_time_periods[i, :]})
			dict_var_info[tup_vvt]["uniform_scaling_q"].update({tup_id: vec_uniform_scaling[i]})

			#
			dict_ordered_traj_arrays[tup_vvt][ind_id, :] = arr_time_periods[i, :]
			dict_ordered_traj_arrays_by_ans[id_strat][ind_ans, :] = arr_time_periods[i, :]
			dict_ordered_vec_max_scalars[tup_vvt][ind_id] = vec_max_scalar[i]
			dict_ordered_vec_min_scalars[tup_vvt][ind_id] = vec_min_scalar[i]

			if "electric" in tup_vvt[0]:
				dict_vi2 = "elec!"
				# future of this wrt research? dict_var_info

		global dict_ordered_traj_arrays_by_ans2
		dict_ordered_traj_arrays_by_ans2 = dict_ordered_traj_arrays_by_ans
		

		##  BUILD ORDERED TRAJECTORY ARRAYS

		# get by VVT group
		for k in dict_ordered_traj_arrays.keys():
			dict_scalar_diff_arrays.update({
				k: self.initialize_dict_sda(
					dict_ordered_traj_arrays.get(k),
					dict_ordered_vec_max_scalars.get(k),
					dict_ordered_vec_min_scalars.get(k)
				)
			})

		# get by strat grouping and determine if strategy
		arr_base_strat = dict_ordered_traj_arrays_by_ans.get(strat_base)
		dict_diff_arrays_by_strat = {}
		strategy_q = False

		for k in dict_ordered_traj_arrays_by_ans.keys():
			arr_cur = dict_ordered_traj_arrays_by_ans.get(k)
			arr_diff = arr_cur - arr_base_strat
			dict_diff_arrays_by_strat.update({k: arr_diff})

			strategy_q = (max(np.abs(arr_diff.flatten())) > thresh) | strategy_q

		dict_strategy_info.update({"difference_arrays_by_strategy": dict_diff_arrays_by_strat}) if strategy_q else None
		type_out = "L" if strategy_q else "X"

		# assign output properties
		self.dict_strategy_info = dict_strategy_info
		self.dict_variable_info = dict_var_info
		self.dict_ordered_trajectory_arrays = dict_ordered_traj_arrays
		self.dict_scalar_diff_arrays = dict_scalar_diff_arrays
		self.xl_type = type_out

		return None
	


	def _initialize_properties(self,
		group: Union[int, None] = None,
	) -> None:
		"""
		Initialize properties definining whether or not the trajectory can vary
			with LHS trials or uncertainty assessment. Sets the following
			properties:

			* self.group
			* self.x_varies

		Keyword Arguments
		-----------------
		- group: optional group id to use for tracking collections of sampling 
			units 
		"""

		# check whether or not there will be variation
		df = self.df_variable_definitions
		field_max = self.field_max_scalar
		field_min = self.field_min_scalar

		s_max = set(df[field_max].astype(float))
		s_min = set(df[field_min].astype(float))
		x_varies = not (
			len(df[[field_max, field_min]].drop_duplicates()) == 1
			& (s_max == s_min)
			& (s_max == set({1.0}))
		)

		# check the group specification
		group = int(group) if sf.isnumber(group) else None


		##  SET PROPERTIES
		
		self.group = group
		self.x_varies = x_varies

		return None





	############################
	#    CORE FUNCTIONALITY    #
	############################

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



	def ordered_by_ota_from_fid_dict(self,
		dict_in: dict,
		key_tuple: tuple
	) -> np.ndarray:
		"""
		Transform keys from input dictionary into np.array that's ordered by
			self.coordinates_id
		"""

		arr_out = np.array([dict_in.get(x) for x in self.coordinates_id])

		return arr_out



	## UNCERTAINY FAN FUNCTIONS

	def build_ramp_vector(self,
		tuple_param: Union[tuple, None] = None
	) -> np.ndarray:
		"""
		Convert tuple_param to a vector for characterizing uncertainty.

		Keyword Arguments
		-----------------
		- tuple_param: tuple of parameters to pass to f_fan

		"""
		tuple_param = (
			self.get_f_fan_function_parameter_defaults(self.uncertainty_fan_function_type) 
			if (tuple_param is None) 
			else tuple_param
		)

		if len(tuple_param) != 4:
			raise ValueError(f"Error: tuple_param {tuple_param} in build_ramp_vector has invalid length. It should have 4 parameters.")
		
		tp_0 = self.time_period_end_certainty
		n = len(self.time_periods) - tp_0 - 1

		arr_out = np.array(
			[
				int(i > tp_0)*self.f_fan(i - tp_0 , n, *tuple_param) 
				for i in range(len(self.time_periods))
			]
		)

		return arr_out
			


	def f_fan(self, x, n, a, b, c, d):
		"""
		Basic function that determines the shape of the uncertainty fan; based 
			on a generalization of the sigmoid (includes linear option).

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



	def generate_future(self,
		lhs_trial_x: float,
		lhs_trial_l: float = 1.0,
		baseline_future_q: bool = False,
		constraints_mix_tg: tuple = (0, 1),
		flatten_output_array: bool = False,
		vary_q: Union[bool, None] = None,
	) -> Dict[str, np.ndarray]:
		"""
		Generate a dictionary mapping each variable specification to futures 
			ordered by self.coordinates_id

		Function Arguments
		------------------
		- lhs_trial_x: LHS trial used to generate uncertainty fan for base 
			future

		Keyword Arguments
		------------------
		- lhs_trial_l: LHS trial used to modify strategy effect
		- baseline_future_q: generate a baseline future? If so, lhs trials do 
			not apply
		- constraints_mix_tg: constraints on the mixing fraction for trajectory 
			groups
		- flatten_output_array: return a flattened output array (apply 
			np.flatten())
		- vary_q: does the future vary? if not, returns baseline
		"""

		vary_q = (
			self.variable_trajectory_group_vary_q 
			if not isinstance(vary_q, bool) 
			else vary_q
		)

		# clean up some cases for None entries
		baseline_future_q |= (lhs_trial_x is None)
		baseline_future_q |= (not vary_q)

		lhs_trial_x = 1.0 if (lhs_trial_x is None) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l is None) else lhs_trial_l

		# some additional checks for potential negative numbers
		no_vary_x = (baseline_future_q | (lhs_trial_x < 0))
		no_vary_l = (baseline_future_q | (lhs_trial_l < 0))
		lhs_trial_x = 1.0 if (lhs_trial_x < 0) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l < 0) else lhs_trial_l

		# initialization
		all_strats = self.dict_id_values.get(self.key_strategy)
		n_strat = len(all_strats)
		strat_base = self.dict_baseline_ids.get(self.key_strategy)

		# index by variable_specification at keys
		dict_out = {}
	
		if self.variable_trajectory_group is not None:

			cat_mix = self.dict_required_tg_spec_fields.get("mixing_trajectory")
			cat_b0 = self.dict_required_tg_spec_fields.get("trajectory_boundary_0")
			cat_b1 = self.dict_required_tg_spec_fields.get("trajectory_boundary_1")

			# use mix between 0/1 (0 = 100% trajectory_boundary_0, 1 = 100% trajectory_boundary_1)
			for vs in self.variable_specifications:

				#ordered_traj_array = self.dict_ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.dict_scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))
				dict_arrs = {
					cat_b0: self.dict_ordered_trajectory_arrays.get((vs, cat_b0)),
					cat_b1: self.dict_ordered_trajectory_arrays.get((vs, cat_b1)),
					cat_mix: self.dict_ordered_trajectory_arrays.get((vs, cat_mix))
				}

				# for trajectory groups, the baseline is the specified mixing vector
				mixer = dict_arrs.get(cat_mix) if no_vary_x else lhs_trial_x
				arr_out = self.mix_tensors(
					dict_arrs[cat_b0], 
					dict_arrs[cat_b1], 
					mixer, 
					constraints_mix_tg
				)

				if self.xl_type == "L":
					#
					# if the XL is an L, then we use the modified future as a base (reduce to include only baseline strategy), then add the uncertainty around the strategy effect
					#
					# get id coordinates( any of cat_mix, cat_b0, or cat_b1 would work -- use cat_mix)
					df_ids_ota = pd.concat([
						self.df_coordinates_id,
						#self.dict_ordered_trajectory_arrays.get((vs, cat_mix))["id_coordinatesREPL"].copy().reset_index(drop = True),
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
						inds = np.sort(
							np.array(
								list(inds0 & set(np.where(df_baseline_strategy[self.field_variable_trajgroup_type] == cat_cur)[0]))
							)
						)
						n_inds = len(inds)
						df_ids0 = (
							df_baseline_strategy[
								[x for x in self.fields_id if (x != self.key_strategy)]
							]
							.loc[inds.repeat(n_strat)]
							.reset_index(drop = True)
						)
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
						df_future_strat = (
							pd.concat(
								[
									df_ids0, 
									pd.DataFrame(df_future_strat, columns = self.fields_time_periods)
								], 
								axis = 1
							)
							.sort_values(by = self.fields_id)
							.reset_index(drop = True)
						)
						l_modified_cats.append(
							dict_arrs[cat_cur] + np.array(df_future_strat[self.fields_time_periods])
						)

					arr_out = self.mix_tensors(*l_modified_cats, constraints_mix_tg)

				# to compare the difference between the "L" design uncertainty and the baseline and add this to the uncertain future (final array)
				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		else:

			rv = self.uncertainty_ramp_vector

			for vs in self.variable_specifications:

				ordered_traj_array = self.dict_ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.dict_scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))

				# order the uniform scaling by the ordered trajectory arrays
				vec_unif_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["uniform_scaling_q"], (vs, None))
				
				# gives 1s where we keep standard fanning (using the ramp vector) and 0s where we use uniform scaling
				vec_base = 1 - vec_unif_scalar
				#
				if max(vec_unif_scalar) > 0:
					vec_max_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["max_scalar"], (vs, None))
					vec_min_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["min_scalar"], (vs, None))
					vec_unif_scalar = (
						vec_unif_scalar*(vec_min_scalar + lhs_trial_x*(vec_max_scalar - vec_min_scalar)) 
						if not no_vary_x 
						else np.ones(vec_unif_scalar.shape)
					)

				vec_unif_scalar = np.array([vec_unif_scalar]).transpose()
				vec_base = np.array([vec_base]).transpose()

				delta_max = dict_scalar_diff_arrays.get("max_tp_end_delta")
				delta_min = dict_scalar_diff_arrays.get("min_tp_end_delta")
				delta_diff = delta_max - delta_min
				delta_val = delta_min + lhs_trial_x*delta_diff

				# delta and uniform scalar don't apply if operating under baseline future (which forces no_vary_x to be true)
				delta_vec = 0.0 if no_vary_x else (rv * np.array([delta_val]).transpose())
				arr_out = ordered_traj_array + delta_vec
				arr_out = arr_out*vec_base + vec_unif_scalar*ordered_traj_array

				if self.xl_type == "L":
					# get series of strategies
					series_strats = self.df_coordinates_id[self.key_strategy]
					w = np.where(np.array(series_strats) == strat_base)[0]
					
					# get strategy adjustments
					lhs_mult_deltas = 1.0 if no_vary_l else lhs_trial_l
					array_strat_deltas = np.concatenate(
						series_strats.apply(
							self.dict_strategy_info["difference_arrays_by_strategy"].get,
							args = (np.zeros((1, len(self.time_periods))), )
						)
					)
					array_strat_deltas *= lhs_mult_deltas

					arr_out = (array_strat_deltas + arr_out[w, :]) if (len(w) > 0) else arr_out

				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		return dict_out






class FutureTrajectories:

	"""
	Create a collection of SampleUnit objects to use to generate futures.

	Initialization Arguments
	------------------------
	- df_input_database: DataFrame to use as database of baseline inputs (across strategies)
	- dict_baseline_ids: dictionary mapping a string of a baseline id field to a baseline id value (integer)
	- time_period_u0: first time period with uncertainty

	Keyword Arguments
	-----------------
	- check_duplicates_in_variable_definition: check for duplicate rows in the
		input `df_variable_definition`? If False, skips checking.
	- dict_all_dims: optional dictionary defining all values associated with
		keys in dict_baseline_ids to pass to each SamplingUnit. If None
		(default), infers from df_input_database. Takes the form
		{
			index_0: [id_val_00, id_val_01,... ],
			index_1: [id_val_10, id_val_11,... ],
			.
			.
			.
		}
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_sample_unit_group: field used to identify sample unit groups. Sample unit groups are composed of:
		* individual variable specifications
		* trajectory groups
	- field_time_period: field used to specify the time period
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group (integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group type (max, min, mix, or lhs)
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- key_future: field used to identify the future
	- key_strategy: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed differently than uncetainty in other variables
	- logger: optional logging.Logger object used to track generation of futures
	- regex_id: regular expression used to identify id fields in the input template
	- regex_trajgroup: Regular expression used to identify trajectory group variables in `field_variable` of `df_input_database`
	- regex_trajmax: Regular expression used to identify trajectory maxima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmin: Regular expression used to identify trajectory minima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmix: Regular expression used to identify trajectory baseline mix (fraction maxima) in variables and trajgroups specified in `field_variable` of `df_input_database`

	"""
	def __init__(self,
		df_input_database: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,
		check_duplicates_in_variable_definition: bool = False,
		dict_all_dims: Union[Dict[str, List[int]], None] = None,
		fan_function_specification: str = "linear",
		field_sample_unit_group: str = "sample_unit_group",
		field_time_period: str = "time_period",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_variable: str = "variable",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		key_future: str = "future_id",
		key_strategy: str = "strategy_id",
		# optional logger
		logger: Union[logging.Logger, None] = None,
		# regular expressions used to define trajectory group components in input database and
		regex_id: re.Pattern = re.compile("(\D*)_id$"),
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)"),
		regex_trajgroup: re.Pattern = re.compile("trajgroup_(\d*)-(\D*$)"),
		regex_trajmax: re.Pattern = re.compile("trajmax_(\D*$)"),
		regex_trajmin: re.Pattern = re.compile("trajmin_(\D*$)"),
		regex_trajmix: re.Pattern = re.compile("trajmix_(\D*$)"),
		# some internal vars
		specification_tgt_lhs: str = "lhs",
		specification_tgt_max: str = "trajectory_boundary_1",
		specification_tgt_min: str = "trajectory_boundary_0"
	):

		##  INITIALIZE PARAMETERS

		# dictionary of baseline ids and fan function
		self.dict_baseline_ids = dict_baseline_ids
		self.fan_function_specification = fan_function_specification

		# set default fields
		self.key_future = key_future
		self.field_sample_unit_group = field_sample_unit_group
		self.key_strategy = key_strategy
		self.field_time_period = field_time_period
		self.field_trajgroup_no_vary_q = field_trajgroup_no_vary_q
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable = field_variable
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type

		# logging.Logger
		self.logger = logger

		# missing values flag
		self.missing_flag_int = -999

		# default regular expressions
		self.regex_id = regex_id
		self.regex_max = regex_max
		self.regex_min = regex_min
		self.regex_tp = regex_tp
		self.regex_trajgroup = regex_trajgroup
		self.regex_trajmax = regex_trajmax
		self.regex_trajmin = regex_trajmin
		self.regex_trajmix = regex_trajmix

		# some default internal specifications used in templates
		self.specification_tgt_lhs = specification_tgt_lhs
		self.specification_tgt_max = specification_tgt_max
		self.specification_tgt_min = specification_tgt_min

		# first period with uncertainty
		self.time_period_u0 = time_period_u0


		##  KEY INITIALIZATIONS

		self._initialize_input_database(df_input_database)
		self._initialize_dict_all_dims(dict_all_dims)
		self._initialize_sampling_units(check_duplicates_in_variable_definition)
		self._set_xl_sampling_units()



	###########################################################
	#	SOME BASIC INITIALIZATIONS AND INTERNAL FUNCTIONS	#
	###########################################################

	def _initialize_dict_all_dims(self,
		dict_all_dims: Union[Dict, None]
	) -> None:
		"""
		Initialize the dictionary of all dimensional values to accomodate
			for each sampling unit--ensures that each SamplingUnit has the
			same dimensional values (either strategy or discrete baselines).
			Sets the following properties:

			* self.dict_all_dimensional_values

		Function Arguments
		------------------
		- dict_all_dims: dictionary of all dimensional values to preserve.
			Takes the form

			{
				index_0: [id_val_00, id_val_01,... ],
				index_1: [id_val_10, id_val_11,... ],
				.
				.
				.
			}
		"""

		dict_all_dims_out = {}
		self.dict_all_dimensional_values = None

		# infer from input database if undefined
		if dict_all_dims is None:
			for k in self.dict_baseline_ids:
				dict_all_dims_out.update({
					k: sorted(list(self.input_database[k].unique()))
				})

		else:
			# check that each dimension is defined in the baseline ids
			for k in dict_all_dims.keys():
				if k in self.dict_baseline_ids.keys():
					dict_all_dims_out.update({k: dict_all_dims.get(k)})

		self.dict_all_dimensional_values = dict_all_dims_out

		return None
	


	def _initialize_input_database(self,
		df_in: pd.DataFrame,
		field_sample_unit_group: Union[str, None] = None,
		field_variable: Union[str, None] = None,
		field_variable_trajgroup: Union[str, None] = None,
		field_variable_trajgroup_type: Union[str, None] = None,
		missing_trajgroup_flag: Union[int, None] = None,
		regex_trajgroup: Union[re.Pattern, None] = None,
		regex_trajmax: Union[re.Pattern, None] = None,
		regex_trajmin: Union[re.Pattern, None] = None,
		regex_trajmix: Union[re.Pattern, None] = None,
	) -> None:
		"""
		Prepare the input database for sampling by adding sample unit group,
			cleaning up trajectory groups, etc. Sets the following properties:

			* self.input_database

		Function Arguments
		------------------
		- df_in: input database to use to generate SampleUnit objects

		Keyword Arguments
		-----------------
		- field_sample_unit_group: field used to identify groupings of sample 
			units
		- field_variable: field in df_in used to denote the database
		- field_variable_trajgroup: field denoting the variable trajectory group
		- field_variable_trajgroup_type: field denoting the type of the variable 
			within a variable trajectory group
		- missing_trajgroup_flag: missing flag for trajectory group values
		- regex_trajgroup: regular expression used to match trajectory group 
			variable specifications
		- regex_trajmax: regular expression used to match the maximum trajectory 
			component of a trajectory group variable element
		- regex_trajmin: regular expression used to match the minimum trajectory 
			component of a trajectory group variable element
		- regex_trajmix: regular expression used to match the mixing component 
			of a trajectory group variable element
		"""

		# key fields
		field_sample_unit_group = (
			self.field_sample_unit_group 
			if (field_sample_unit_group is None) 
			else field_sample_unit_group
		)
		field_variable = (
			self.field_variable 
			if (field_variable is None) 
			else field_variable
		)
		field_variable_trajgroup = (
			self.field_variable_trajgroup 
			if (field_variable_trajgroup is None) 
			else field_variable_trajgroup
		)
		field_variable_trajgroup_type = (
			self.field_variable_trajgroup_type 
			if (field_variable_trajgroup_type is None) 
			else field_variable_trajgroup_type
		)

		# set the missing flag
		missing_flag = self.missing_flag_int if (missing_trajgroup_flag is None) else int(missing_trajgroup_flag)
		
		# regular expressions
		regex_trajgroup = self.regex_trajgroup if (regex_trajgroup is None) else regex_trajgroup
		regex_trajmax = self.regex_trajmax if (regex_trajmax is None) else regex_trajmax
		regex_trajmin = self.regex_trajmin if (regex_trajmin is None) else regex_trajmin
		regex_trajmix = self.regex_trajmix if (regex_trajmix is None) else regex_trajmix

		##  split traj groups
		new_col_tg = []
		new_col_spec_type = []

		# split out traj group and variable specificationa
		df_add = (
			df_in[[field_variable, field_variable_trajgroup]]
			.apply(
				self.get_trajgroup_and_variable_specification,
				#kwargs = (
				regex_trajgroup = regex_trajgroup,
				regex_trajmax = regex_trajmax,
				regex_trajmin = regex_trajmin,
				regex_trajmix = regex_trajmix,
				#),
				axis = 1,
				raw = True,
			)
		)

		# add the variable trajectory group
		df_add = (
			[np.array(x) for x in df_add]
			if not isinstance(df_add, pd.DataFrame)
			else np.array(df_add)
		)
		df_add = pd.DataFrame(
			df_add, 
			columns = [field_variable_trajgroup, field_variable_trajgroup_type]
		)
		
		df_add[field_variable_trajgroup] = (
			df_add[field_variable_trajgroup]
			.replace({None: missing_flag})
			.astype(int)
		)

		df_in = pd.concat([
				df_in.drop([field_variable_trajgroup, field_variable_trajgroup_type], axis = 1),
				df_add
			],
			axis = 1
		)

		# update trajgroups to add dummies
		new_tg = df_in[df_in[field_variable_trajgroup] >= 0][field_variable_trajgroup]
		new_tg = 1 if (len(new_tg) == 0) else max(np.array(new_tg)) + 1
		tgs = list(df_in[field_variable_trajgroup].copy())
		tgspecs = list(df_in[field_variable_trajgroup_type].copy())

		# initialization outside of the iteration
		var_list = list(df_in[field_variable].copy())
		dict_parameter_to_tg = {}
		dict_repl_tgt = {
			"max": self.specification_tgt_max, 
			"min": self.specification_tgt_min,
		}

		for i in range(len(df_in)):

			# get trajgroup, trajgroup type, and variable specification for current row
			tg = int(df_in[field_variable_trajgroup].iloc[i])
			tgspec = str(df_in[field_variable_trajgroup_type].iloc[i])
			vs = str(df_in[field_variable].iloc[i])

			# skip if the trajgroup type is unspecified
			if (tgspec == "<NA>") | (tgspec == "None"):
				continue
			
			new_tg_q = True

			if tg > 0:
				# drop the group/remove the trajmax/min/mix
				vs = regex_trajgroup.match(vs).groups()[0]
				new_tg_q = False

			# check for current trajgroup type
			for regex in [regex_trajmax, regex_trajmin, regex_trajmix]:
				matchstr = regex.match(vs)
				vs = matchstr.groups()[0] if (matchstr is not None) else vs

			# update the variable list
			var_list[i] = vs

			# update indexing of trajectory groups
			if not new_tg_q:
				continue

			if vs in dict_parameter_to_tg.keys():
				tgs[i] = int(dict_parameter_to_tg.get(vs))
			else:
				dict_parameter_to_tg.update({vs: new_tg})
				tgs[i] = new_tg
				new_tg += 1

		# update outputs
		df_in[field_variable] = var_list
		df_in[field_variable_trajgroup] = tgs
		df_in = (
			df_in[
				~df_in[field_variable].isin([self.specification_tgt_lhs])
			]
			.reset_index(drop = True)
		)
		df_in[field_variable_trajgroup_type].replace(dict_repl_tgt, inplace = True)

		# add sample_unit_group field
		dict_var_to_su = sf.build_dict(
			df_in[
				df_in[field_variable_trajgroup] > 0
			][
				[field_variable, field_variable_trajgroup]
			]
			.drop_duplicates()
		)
		vec_vars_to_assign = sorted(list(set(df_in[df_in[field_variable_trajgroup] <= 0][field_variable])))
		min_val = (max(dict_var_to_su.values()) + 1) if (len(dict_var_to_su) > 0) else 1
		dict_var_to_su.update(
			dict(zip(
				vec_vars_to_assign,
				list(range(min_val, min_val + len(vec_vars_to_assign)))
			))
		)
		df_in[field_sample_unit_group] = df_in[field_variable].replace(dict_var_to_su)

		self.input_database = df_in

		return None



	def _initialize_sampling_units(self,
		check_duplicates_in_variable_definition: bool,
		df_in: Union[pd.DataFrame, None] = None,
		dict_all_dims: Union[Dict[str, List[int]], None] = None,
		fan_function: Union[str, None] = None,
		**kwargs
	) -> None:
		"""
		Instantiate all defined SamplingUnits from input database. Sets the
			following properties:

			* self.n_su
			* self.all_sampling_units
			* self.dict_sampling_units

		Behavioral Notes
		----------------
		- _initialize_sampling_units() will try to identify the availablity of
			dimensions specified in dict_all_dims within df_in. If a dimension
			specified in dict_all_dims is not found within df_in, the funciton
			will replace the value with the baseline strategy.
		- If a product specified within self.dict_baseline_ids is missing in
			the dataframe, then an error will occur.

		Keword Arguments
		----------------
		- df_in: input database used to identify sampling units. Must include
			self.field_sample_unit_group
		- dict_all_dims: optional dictionary to pass that contains dimensions.
			This dictionary is used to determine which dimensions *should* be
			present. If any of the dimension sets specified in dict_all_dims
			are not found in df_in, their values are replaced with associated
			baselines.
		- fan_function: function specification to use for uncertainty fans
		- **kwargs: passed to SamplingUnit initialization
		"""

		# get some defaults
		df_in = self.input_database if (df_in is None) else df_in
		fan_function = self.fan_function_specification if (fan_function is None) else fan_function
		dict_all_dims = self.dict_all_dimensional_values if not isinstance(dict_all_dims, dict) else dict_all_dims

		# get some key fields for defining sampling units
		field_time_period = kwargs.get("field_time_period", self.field_time_period)
		field_uniform_scaling_q = kwargs.get("field_uniform_scaling_q", self.field_uniform_scaling_q)
		field_trajgroup_no_vary_q = kwargs.get("field_trajgroup_no_vary_q", self.field_trajgroup_no_vary_q)
		field_variable_trajgroup = kwargs.get("field_variable_trajgroup", self.field_variable_trajgroup)
		field_variable_trajgroup_type = kwargs.get("field_variable_trajgroup_type", self.field_variable_trajgroup_type)
		field_variable = kwargs.get("field_variable", self.field_variable)
		key_strategy = kwargs.get("key_strategy", self.key_strategy)

		# retrieve regular expressions used in the input template
		regex_id = kwargs.get("regex_id", self.regex_id)
		regex_max = kwargs.get("regex_max", self.regex_max)
		regex_min = kwargs.get("regex_min", self.regex_min)
		regex_tp = kwargs.get("regex_tp", self.regex_tp)

		dict_sampling_units = {}

		dfgroup_sg = df_in.drop_duplicates().groupby(self.field_sample_unit_group)
		all_sample_groups = sorted(list(set(df_in[self.field_sample_unit_group])))

		n_sg = len(dfgroup_sg)
		if isinstance(dict_all_dims, dict):
			dict_all_dims = dict((k, v) for k, v in dict_all_dims.items() if k in self.dict_baseline_ids.keys())


		##  GENERATE SamplingUnit FROM DATABASE

		self._log(f"Instantiating {n_sg} sampling units.", type_log = "info")
		t0 = time.time()

		for iterate in enumerate(dfgroup_sg):

			i, df_sg = iterate
			df_sg = df_sg[1]
			sg = int(df_sg[self.field_sample_unit_group].iloc[0])

			# fill in missing values from baseline if dims are missing in input database
			df_sg = (
				self.clean_sampling_unit_input_df(
					df_sg, 
					dict_all_dims = dict_all_dims
				) 
				if (dict_all_dims is not None) 
				else df_sg
			)

			samp = SamplingUnit(
				df_sg.drop(self.field_sample_unit_group, axis = 1),
				self.dict_baseline_ids,
				self.time_period_u0,
				check_duplicates_in_variable_definition = check_duplicates_in_variable_definition,
				fan_function_specification = fan_function,
				field_time_period = field_time_period,
				field_uniform_scaling_q = field_uniform_scaling_q,
				field_trajgroup_no_vary_q = field_trajgroup_no_vary_q,
				field_variable_trajgroup = field_variable_trajgroup,
				field_variable_trajgroup_type = field_variable_trajgroup_type,
				field_variable = field_variable,
				group = sg,
				key_strategy = key_strategy,
				missing_trajgroup_flag = self.missing_flag_int,
				regex_id = regex_id,
				regex_max = regex_max,
				regex_min = regex_min,
				regex_tp = regex_tp
			)

			
			# update sampling unit dictionary
			dict_sampling_units = (
				dict(zip(all_sample_groups, [samp for x in range(n_sg)])) 
				if (i == 0) 
				else dict_sampling_units
			)
			dict_sampling_units.update({sg: samp})

			self._log(f"Iteration {i} complete.", type_log = "info") if (i%250 == 0) else None

		t_elapse = sf.get_time_elapsed(t0)
		self._log(f"\t{n_sg} sampling units complete in {t_elapse} seconds.", type_log = "info")


		##  SET PROPERTIES

		self.n_su = n_sg
		self.all_sampling_units = all_sample_groups
		self.dict_sampling_units = dict_sampling_units

		return None



	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	):
		"""
		Clean implementation of sf._optional_log in-line using default logger. 
			See ?sf._optional_log for more information

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
		sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



	def _set_xl_sampling_units(self,
	) -> None:
		"""
		Determine X/L sampling units--sets three properties:

			* self.all_sampling_units_l
			* self.all_sampling_units_x
			* self.dict_sampling_unit_to_xl_type

		NOTE: any sampling units that *do not vary* are ignored and do not end
			up being assigned to all_sampling_units_x or all_sampling_units_l
		"""
		all_sampling_units_l = []
		all_sampling_units_x = []
		dict_sampling_unit_to_xl_type = {}

		for k, samp in self.dict_sampling_units.items():
			xl_type = samp.xl_type
			x_varies = samp.x_varies

			dict_sampling_unit_to_xl_type.update({k: xl_type})
			
			(
				all_sampling_units_x.append(k) 
				if (xl_type == "X") & x_varies
				else None
			)

			(	
			 all_sampling_units_l.append(k)
			 if (xl_type == "L")
			 else None
			)


		##  SET PROPERTIES

		self.all_sampling_units_l = all_sampling_units_l
		self.all_sampling_units_x = all_sampling_units_x
		self.dict_sampling_unit_to_xl_type = dict_sampling_unit_to_xl_type

		return None





	####################################
	#    PREPARE THE INPUT DATABASE    #
	####################################

	def clean_sampling_unit_input_df(self,
		df_su_input: pd.DataFrame,
		dict_all_dims: Union[Dict[str, List], None] = None,
		dict_baseline_ids: Union[Dict[str, int], None] = None,
		dict_expand_vars: Union[Dict[str, List[str]], None] = None,
		drop_duplicates: bool = False,
		sample_unit_id: Any = None,
	) -> pd.DataFrame:
		"""
		Prepare an input data frame for initializing SamplingUnit within
			FutureTrajectories. Ensures that all dimensions that are specified
			in the global database are defined in the Sampling Unit. Replaces
			missing dimensions with core baseline (e.g., (0, 0, 0)).

		Function Arguments
		------------------
		- df_su_input: input data frame to SamplingUnit
		- dict_all_dims: optional dictionary to pass that contains dimensions.
			This dictionary is used to determine which dimensions *should* be
			present. If any of the dimension sets specified in dict_all_dims
			are not found in df_in, their values are replaced with associated
			baselines.

		Keyword Arguments
		-----------------
		- dict_baseline_ids: dictionary mapping index fields to baseline
			values
		- dict_expand_vars: dictionary of variable specification components to
			expand. If none, expands along all uniquely defined values for
			self.field_variable_trajgroup_type and self.variable
		- drop_duplicates: drop duplicates in the input dataframe?
		- sample_unit_id: optional id to pass for error troubleshooting
		"""
		dict_all_dims = self.dict_all_dimensional_values if (dict_all_dims is None) else dict_all_dims
		if not isinstance(dict_all_dims, dict):
			return df_su_input


		##  CHECK BASELINES DEFINED

		dict_baseline_ids = self.dict_baseline_ids if not isinstance(dict_baseline_ids, dict) else dict_baseline_ids
		df_su_base = sf.subset_df(
			df_su_input,
			dict_baseline_ids
		)
		df_su_base.drop_duplicates(inplace = True) if drop_duplicates else None

		# add expansion variables
		dict_expand_vars = (
			{
				self.field_variable_trajgroup_type: list(df_su_input[self.field_variable_trajgroup_type].unique()),
				self.field_variable: list(df_su_input[self.field_variable].unique())
			}
			if (dict_expand_vars is None) 
			else dict_expand_vars
		)
		dict_all_dims.update(dict_expand_vars)

		# 
		n_req_baseline = np.prod([len(v) for v in dict_expand_vars.values()])
		if len(df_su_base) != n_req_baseline:
			sg = "" if (sample_unit_id is not None) else f" {sample_unit_id}"
			msg = f"Unable to initialize sample group{sg}: one or more variables and/or variable trajectory group types are missing. Check the input data frame."
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)


		##  BUILD EXPANDED DATAFRAME, MERGE IN AVAILABLE, THEN FILL IN ROWS

		dims_expand = [x for x in df_su_input.columns if x in dict_all_dims.keys()]
		vals_expand = [dict_all_dims.get(x) for x in dims_expand]
		df_dims_req = pd.DataFrame(
			list(itertools.product(*vals_expand)),
			columns = dims_expand
		)
		df_dims_req = pd.merge(df_dims_req, df_su_input, how = "left")

		# get merge fields (expansion fields that exclude baseline ids) and subset fields (data fields to replace)
		fields_merge = list(dict_expand_vars.keys())
		fields_subset = [x for x in df_dims_req.columns if x not in dims_expand]
		df_out = sf.fill_df_rows_from_df(
			df_dims_req,
			df_su_base,
			fields_merge,
			fields_subset
		)

		return df_out[df_su_input.columns]



	def generate_future_from_lhs_vector(self,
		df_row_lhc_sample_x: Union[pd.Series, pd.DataFrame, None],
		df_row_lhc_sample_l: Union[pd.Series, pd.DataFrame, None] = None,
		future_id: Union[int, None] = None,
		baseline_future_q: bool = False,
		dict_optional_dimensions: Dict[str, int] = {}
	) -> pd.DataFrame:
		"""
		Build a data frame of a single future for all sample units

		Function Arguments
		------------------
		- df_row_lhc_sample_x: data frame row with column names as sample groups 
			for all sample groups to vary with uncertainties OR None (acceptable
			if running baseline future)

		Keyword Arguments
		-----------------
		- df_row_lhc_sample_l: data frame row with column names as sample groups 
			for all sample groups to vary with uncertainties
			* If None, lhs_trial_l = 1 in all samples (constant strategy effect 
				across all futures)
		- future_id: optional future id to add to the dataframe using 
			self.future_id
		- baseline_future_q: generate the dataframe for the baseline future?
		- dict_optional_dimensions: dictionary of optional dimensions to pass to 
			the output data frame (form: {key_dimension: id_value})
		"""

		# check the specification of
		if not (isinstance(df_row_lhc_sample_x, pd.DataFrame) or isinstance(df_row_lhc_sample_x, pd.Series) or (df_row_lhc_sample_x is None)):
			tp = str(type(df_row_lhc_sample_x))
			self._log(f"Invalid input type {tp} specified for df_row_lhc_sample_x in generate_future_from_lhs_vector: pandas Series or DataFrames (first row) are acceptable inputs. Returning baseline future.", type_log = "warning")
			df_row_lhc_sample_x = None

		# initialize outputs and iterate
		dict_df = {}
		df_out = []
		for k in enumerate(self.all_sampling_units):
			k, su = k
			samp = self.dict_sampling_units.get(su)

			# some skipping conditions
			if samp is None:
				continue

			if (not samp.x_varies) & (samp.xl_type == "X"):
				# if the sampling unit doesn't vary as X, return baseline future
				dict_fut = samp.generate_future(
					None,
					None,
					baseline_future_q = True,
				)

			else:
				# get LHC samples for X and L
				lhs_x = self.get_df_row_element(df_row_lhc_sample_x, su)
				lhs_l = self.get_df_row_element(df_row_lhc_sample_l, su, 1.0)

				# note: if lhs_x is None, returns baseline future no matter what,
				dict_fut = samp.generate_future(
					lhs_x,
					lhs_l,
					baseline_future_q = baseline_future_q
				)

			dict_df.update(
				dict((key, value.flatten()) for key, value in dict_fut.items())
			)

			# initialize indexing if necessary
			if len(df_out) == 0:
				dict_fields = None if (future_id is None) else {self.key_future: future_id}
				df_out.append(
					samp.generate_indexing_data_frame(
						dict_additional_fields = dict_fields
					)
				)

		df_out.append(pd.DataFrame(dict_df))
		df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
		df_out = sf.add_data_frame_fields_from_dict(df_out, dict_optional_dimensions) if isinstance(dict_optional_dimensions, dict) else df_out

		return df_out
	


	def get_df_row_element(self,
		row: Union[pd.Series, pd.DataFrame, None],
		index: Union[int, float, str],
		return_def: Union[int, float, str, None] = None,
	) -> float:
		"""
		Support for self.generate_future_from_lhs_vector. Read an element from a 
			named series or DataFrame.

		Function Arguments
		------------------
		- row: Series or DataFrame. If DataFrame, only reads the first row
		- index: column index in input DataFrame to read (field)

		Keyword Arguments
		-----------------
		- return_def: default return value if row is None
		"""
		out = return_def
		if isinstance(row, pd.DataFrame):
			out = float(row[index].iloc[0]) if (index in row.columns) else out
		elif isinstance(row, pd.Series):
			out = float(row[index]) if (index in row.index) else out

		return out



	def get_trajgroup_and_variable_specification(self,
		input_var_spec: np.ndarray,#str,
		#trajgroup_pass: Union[int, None],
		regex_trajgroup: Union[re.Pattern, None] = None,
		regex_trajmax: Union[re.Pattern, None] = None,
		regex_trajmin: Union[re.Pattern, None] = None,
		regex_trajmix: Union[re.Pattern, None] = None,
		#trajgroup_pass: Union[int, None] = None,
	) -> Tuple[str, str]:
		"""
		Derive a trajectory group and variable specification from variables in 
			an input variable specification

		Function Arguments
		------------------
		- input_var_spec: variable specification string

		Keyword Arguments
		-----------------
		- regex_trajgroup: Regular expression used to identify trajectory group 
			variables in `field_variable` of `df_input_database`
		- regex_trajmax: Regular expression used to identify trajectory maxima 
			in variables and trajgroups specified in `field_variable` of 
			`df_input_database`
		- regex_trajmin: Regular expression used to identify trajectory minima 
			in variables and trajgroups specified in `field_variable` of 
			`df_input_database`
		- regex_trajmix: Regular expression used to identify trajectory baseline 
			mix (fraction maxima) in variables and trajgroups specified in `
			field_variable` of `df_input_database`
		"""
		input_var_spec, trajgroup_pass = input_var_spec

		input_var_spec = str(input_var_spec)
		regex_trajgroup = self.regex_trajgroup if (regex_trajgroup is None) else regex_trajgroup
		regex_trajmax = self.regex_trajmax if (regex_trajmax is None) else regex_trajmax
		regex_trajmin = self.regex_trajmin if (regex_trajmin is None) else regex_trajmin
		regex_trajmix = self.regex_trajmix if (regex_trajmix is None) else regex_trajmix

		trajgroup_pass_test = sf.isnumber(trajgroup_pass)
		tg = None if not trajgroup_pass_test else int(trajgroup_pass)
		var_spec = None
		

		# check trajgroup match
		check_spec_string = input_var_spec
		trajgroup_match = regex_trajgroup.match(input_var_spec)
		if trajgroup_match is not None:
			tg = int(trajgroup_match.groups()[0])
			check_spec_string = str(trajgroup_match.groups()[1])

		# check trajectory max/min/mix
		if regex_trajmax.match(check_spec_string) is not None:
			var_spec = "max"
		elif regex_trajmin.match(check_spec_string) is not None:
			var_spec = "min"
		elif regex_trajmix.match(check_spec_string) is not None:
			var_spec = "mix"
		elif (check_spec_string == "lhs") and (tg is not None):
			var_spec = "lhs"

		# set output
		tup_out = (tg, var_spec)

		return tup_out
	


	def get_variable_specification_index(self,
		varname: str,
	) -> Union[int, None]:
		"""
		Return the key in self.dict_sampling_units of the sampling unit that 
			controls variable `varname`. Returns None if not found.
		"""

		ind = None

		for k, v in self.dict_sampling_units.items():
			ind = k if (varname in v.variable_specifications) else ind
		
		return ind
		


