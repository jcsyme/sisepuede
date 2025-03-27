
import logging
import numpy as np
import pandas as pd
import pyDOE2 as pyd
from typing import *

from sisepuede.core.attribute_table import AttributeTable
import sisepuede.utilities._toolbox as sf



class LHSDesign:
	"""LHSDesign stores LHC sample tables for Xs and Ls, managing different 
	    designs based on a design_id attribute table

	Initialization Arguments
	------------------------
	attribute_design_id : AttributeTable
	    AttributeTable containing information on the uncertainty design, 
		including transformation parameters for assesseing uncertainties in 
		lever (strategy) effects
	field_lhs_key : Union[str, None]
	    Field to use to as a key for indexing unique LHS trials

	Keyword Arguments
	------------------
	The following arguments can be set at initialization and/or updated 
		dynamically with LHSDesign._set_lhs_parameters():
	n_factors_l : Union[int, None]
	    Optional number of factors associated with lever (strategy) 
		uncertainties to set at initialization
	n_factors_x : Union[int, None]
	    Optional number of factors associated with exogenous uncertainties to 
		set at initialization
	n_trials : Union[int, None]
	    Optional number of trials to set at initialization
	random_seed : Union[int, None]
	    Optional random seed to specify in generation of tables (sequentially 
		increases by one for ach additional LHS table)

	Additional Keyword Arguments
	----------------------------
	default_return_type : type
	    Governs output type for LHS arrays
		* pd.DataFrame or np.ndarray
	field_transform_b : str
	    Field in AttributeTable giving the value of `b` for each design_id
	field_transform_m : str
	    Field in AttributeTable giving the value of `m` for each design_id
	field_transform_inf : str
	    Field in AttributeTable giving the value of `inf` for each design_id
	field_transform_sup : str
	    Field in AttributeTable giving the value of `sup` for each design_id
	field_vary_l : str
	    Field in AttributeTable giving the binary value of whether or not to 
		vary lever effects
	field_vary_x : str
	    Field in AttributeTable giving the binary value of whether or not to 
		vary exogenous uncertainties
	fields_factors_l : Union[List[str], List[int], None]
	    Fields used to name factors associated with lever effects in LHS tables 
		retrieved using LHSDesign.retrieve_lhs_tables_by_design()
		* If None, defaults to numnerical ordering 
			(i.e., 0, 1, 2, ... , n_factors_l - 1)
	fields_factors_x : Union[List[str], List[int], None]
	    Fields used to name factors associated with exogenous uncertainties in 
		LHS tables retrieved using self.retrieve_lhs_tables_by_design()
		* If None, defaults to numnerical ordering 
			(i.e., 0, 1, 2, ... , n_factors_x - 1)

		* NOTE for fields_factors_l and fields_factors_x: if n_factors_x is 
			reset using LHSDesign._set_lhs_parameters(), then the associated 
			fields_factors_# should also be updated. If not, the fields are 
			reset to numerical indexing.
	ignore_trial_flag : float
	    Flag in output LHS tables to use as a flag. Should be a negative float
	"""

	def __init__(self,
		attribute_design_id: AttributeTable,
		field_lhs_key: Union[str, None],
		n_factors_l: Union[int, None] = None,
		n_factors_x: Union[int, None] = None,
		n_trials: Union[int, None] = None,
		random_seed: Union[int, None] = None,
		default_return_type: type = pd.DataFrame,
		field_transform_b: str = "linear_transform_l_b",
		field_transform_m: str = "linear_transform_l_m",
		field_transform_inf: str = "linear_transform_l_inf",
		field_transform_sup: str = "linear_transform_l_sup",
		field_vary_l: str = "vary_l",
		field_vary_x: str = "vary_x",
		fields_factors_l: Union[List[str], List[int], None] = None,
		fields_factors_x: Union[List[str], List[int], None] = None,
		ignore_trial_flag: float = -1.0,
		logger: Union[logging.Logger, None] = None,
	):
		self.attribute_design_id = attribute_design_id

		# set some general fields
		self.field_lhs_key = field_lhs_key
		self.field_transform_b = field_transform_b
		self.field_transform_m = field_transform_m
		self.field_transform_inf = field_transform_inf
		self.field_transform_sup = field_transform_sup
		self.field_vary_l = field_vary_l
		self.field_vary_x = field_vary_x

		# initialize other properties
		self.default_return_type = pd.DataFrame
		self.logger = logger
		self._set_ignore_trial_flag(ignore_trial_flag)

		# check parameters and set lhs tables to initialize

		self.fields_factors_l = None
		self.fields_factors_x = None
		self.n_factors_l = None
		self.n_factors_x = None
		self.n_trials = None
		self.random_seed = None

		self._set_lhs_parameters(
			fields_factors_l = fields_factors_l,
			fields_factors_x = fields_factors_x,
			n_factors_x = n_factors_x,
			n_factors_l = n_factors_l,
			n_trials = n_trials,
			random_seed = random_seed
		)



	################################################
	#	SET KEY VALUES IN THE LHSDesign OBJECT	#
	################################################

	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
		"""
		Clean implementation of sf._optional_log in-line using default logger. See ?sf._optional_log for more information

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
		sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



	def _set_ignore_trial_flag(self,
		ignore_trial_flag: float
	) -> None:
		self.ignore_trial_flag = float(min(ignore_trial_flag, -1.0)) if (isinstance(ignore_trial_flag, float) or isinstance(ignore_trial_flag, int)) else -1.0



	def _set_lhs_parameters(self,
		fields_factors_l: Union[List[str], List[int], None] = None,
		fields_factors_x: Union[List[str], List[int], None] = None,
		n_factors_x: Union[int, None] = None,
		n_factors_l: Union[int, None] = None,
		n_trials: Union[int, None] = None,
		random_seed: Union[int, None] = None
	) -> None:
		"""
		 Set parameters if missing.

		Keyword Arguments
		------------------
		Some arguments can be set at initialization and/or updated dynamically
			with ._set_lhs_tables(...)
			- fields_factors_l: fields used to label lever effect output
				DataFrames retrieved using self.retrieve_lhs_tables_by_design()
				by factor
			- fields_factors_x: fields used to label exogenous uncertainty
				output DataFrames retrieved using
				self.retrieve_lhs_tables_by_design() by factor
			- n_factors_x: optional number of factors associated with exogenous
				uncertainties to set at initialization
			- n_factors_l: optional number of factors associated with lever
				(strategy) uncertainties to set at initialization
			- n_trials: optional number of trials to set at initialization
			- random_seed: optional random seed to specify in generation of
				tables (sequentially increases by one for ach additional LHS
				table)
		"""

		# get current values
		cur_fields_factors_l = self.fields_factors_l
		cur_fields_factors_x = self.fields_factors_x
		cur_n_factors_l = self.n_factors_l
		cur_n_factors_x = self.n_factors_x
		cur_n_trials = self.n_trials
		cur_random_seed = self.random_seed

		# refresh the lhs table?
		refresh_lhs = False

		# update number of factors for l
		if (self.n_factors_l is None) and (n_factors_l is not None):
			self.n_factors_l = n_factors_l if (n_factors_l > 0) else None
		elif n_factors_l is not None:
			self.n_factors_l = n_factors_l if (n_factors_l > 0) else None
		#
		refresh_lhs = refresh_lhs or (self.n_factors_l != cur_n_factors_l)


		# update fields of factors for l
		if (self.n_factors_l != cur_n_factors_l) or isinstance(fields_factors_l, list):
			if (self.n_factors_l is not None):
				self.fields_factors_l = list(range(self.n_factors_l))
				if isinstance(fields_factors_l, list):
					if len(fields_factors_l) == self.n_factors_l:
						self.fields_factors_l = fields_factors_l
						self._log(f"LHSDesign.fields_factors_l reset successful.", type_log = "info")
					else:
						self._log(
							f"""
							Warning in _set_lhs_parameters(): the length of 
							fields_factors_l did not match self.n_factors_l. 
							Setting output fields_factors_l to default integer 
							indexing.
							""", 
							type_log = "warning"
						)
			else:
				self.fields_factors_l = None


		# update number of factors for x
		if (self.n_factors_x is None) and (n_factors_x is not None):
			self.n_factors_x = n_factors_x if (n_factors_x > 0) else None
		elif n_factors_x is not None:
			self.n_factors_x = n_factors_x if (n_factors_x > 0) else None
		#
		refresh_lhs = refresh_lhs or (self.n_factors_x != cur_n_factors_x)


		if (self.n_factors_x != cur_n_factors_x) or isinstance(fields_factors_x, list):
			if (self.n_factors_x is not None):
				self.fields_factors_x = list(range(self.n_factors_x))
				if isinstance(fields_factors_x, list):
					if len(fields_factors_x) == self.n_factors_x:
						self.fields_factors_x = fields_factors_x
						self._log(f"LHSDesign.fields_factors_x reset successful.", type_log = "info")
					else:
						self._log(
							f"""
							Warning in _set_lhs_parameters(): the length of 
							fields_factors_x did not match self.n_factors_x. 
							Setting output fields_factors_x to default integer 
							indexing.
							""", 
							type_log = "warning"
						)
			else:
				self.fields_factors_x = None


		# update number of trials and vector of lhs key values
		if self.n_trials is None:
			self.n_trials = n_trials
		elif n_trials is not None:
			self.n_trials = n_trials if (n_trials > 0) else None
		#
		self.vector_lhs_key_values = list(range(1, self.n_trials + 1)) if (self.n_trials is not None) else None
		#
		refresh_lhs = refresh_lhs or (self.n_trials != cur_n_trials)


		# update random seed
		if self.random_seed is None:
			self.random_seed = random_seed
		elif random_seed is not None:
			self.random_seed = random_seed if (random_seed > 0) else None
		#
		refresh_lhs = refresh_lhs or (self.random_seed != random_seed)


		# refresh the LHS tables?
		self._set_lhs_tables() if refresh_lhs else None

		return None



	def _set_lhs_tables(self,
	) -> None:
		"""
		Create LHS tables for X (exogenous uncertainties) and LEs (lever effects). Can be refreshed.

		Assigns properties:

		- self.arr_lhs_l
		- self.arr_lhs_x
		"""

		# run some checks
		return_none = False
		if (self.n_factors_l is None) and (self.n_factors_x is None):
			return_none = True
		elif self.n_trials is None:
			return_none = True

		self.arr_lhs_l, self.arr_lhs_x = (None, None) if return_none else self.generate_lhs()





	############################
	#	CORE FUNCTIONALITY	#
	############################

	def generate_lhs(self,
		n_factors_l: Union[int, None] = None,
		n_factors_x: Union[int, None] = None,
		n_trials: Union[int, None] = None,
		random_seed: Union[int, None] = None
	):
		"""
		Generate LHC Sample tables for Xs and Ls to use in generating a database of output trajectories

		Function Arguments
		------------------


		Keyword Arguments
		-----------------
		- field_lhs_key: field used to as key for each lhs trial. Defaults to "future_id"
		- n_trials: number of LHS trials to generate
		- n_factors_x: number of factors associated with uncertainties
		- n_factors_l: number of factors associated with levers
		- random_seed: optional random seed to specify for generating LHC trials

		"""

		# initialize components
		n_factors_l = self.n_factors_l if (n_factors_l is None) else n_factors_l
		n_factors_x = self.n_factors_x if (n_factors_x is None) else n_factors_x
		n_trials = self.n_trials if (n_trials is None) else n_trials
		random_seed = self.random_seed if (random_seed is None) else random_seed

		# check specifications
		check_xl_specification = (n_factors_l is not None) or (n_factors_x is not None)

		retun_none = (n_trials is None) or not check_xl_specification
		if retun_none:
			self._log(
				f"Warning in generate_lhs: one or more elements are missing. If not initialized with n_factors_l, n_factors_x, or n_trials, update with self._set_lhs", 
				type_log = "warning"
			)
			return None

		# generate trials
		rs_l = random_seed
		rs_x = random_seed + 1 if (random_seed is not None) else None

		df_lhs_l = pyd.lhs(n_factors_l, n_trials, random_state = rs_l) if (n_factors_l is not None) else None
		df_lhs_x = pyd.lhs(n_factors_x, n_trials, random_state = rs_x) if (n_factors_x is not None) else None

		return df_lhs_l, df_lhs_x



	def retrieve_lhs_tables_by_design(self,
		design_id: Union[int, None],
		arr_lhs_l: Union[np.ndarray, None] = None,
		arr_lhs_x: Union[np.ndarray, None] = None,
		attr_design_id: Union[AttributeTable, None] = None,
		ignore_trial_flag: Union[float, None] = None,
		field_lhs_key: Union[str, None] = None,
		field_vary_l: Union[str, None] = None,
		field_vary_x: Union[str, None] = None,
		return_type: Union[type, None] = None
	) -> Union[
		Tuple[Union[None, pd.DataFrame], Union[None, pd.DataFrame]],
		Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]
	]:
		"""
		Retrieve LHS tables for a particular design (applies any necessary
			modifications to base LHS table)

		Function Arguments
		------------------
		- design_id: design_id to retrieve table for. If None, returns raw LHC
			samples.

		Keyword Arguments
		-----------------
		- arr_lhs_l: np.ndarray of LHS samples used to explore around lever
			effects
			* If None, defaults to self.arr_lhs_l
		- arr_lhs_x: np.ndarray of LHS samples used to explore around exogenous
			uncertainties
			* If None, defaults to self.arr_lhs_x
		- attr_design_id: AttributeTable used to determine design indexing
			* If None, defaults to self.attribute_design_id
		- ignore_trial_flag: flag to use for invalid trials
		- field_lhs_key = self.field_lhs_key if (field_lhs_key is None) else
			field_lhs_key
		- field_vary_l: field in attr_design_id.table denoting whether or not
			LEs vary under the design
		- field_vary_x: field in attr_design_id.table denoting whether or not
			Xs vary under the design
		- return_type: type of array to return. Valid types are pd.DataFrame or
			np.ndarray. If a data frame, adds index fields for design and
			field_lhs_key

		Notes
		-----
		- LHS Key values are *always* 1-indexed; i.e., they start at 1 instead of 0 (to avoid interfering with potential "baseline" trials).
		"""

		# some basic fields and variables
		field_lhs_key = self.field_lhs_key if (field_lhs_key is None) else field_lhs_key
		field_vary_l = self.field_vary_l if (field_vary_l is None) else field_vary_l
		field_vary_x = self.field_vary_x if (field_vary_x is None) else field_vary_x
		ignore_trial_flag = self.ignore_trial_flag if (ignore_trial_flag is None) else ignore_trial_flag
		return_type = self.default_return_type if ((return_type is None) or (return_type not in [pd.DataFrame, np.ndarray])) else return_type

		# get arrays and return none if nothing is specified
		arr_lhs_l = self.arr_lhs_l if (arr_lhs_l is None) else arr_lhs_l
		arr_lhs_x = self.arr_lhs_x if (arr_lhs_x is None) else arr_lhs_x
		if (arr_lhs_l is None) and (arr_lhs_x is None):
			return None

		# check return type
		return_type = pd.DataFrame if (return_type not in [pd.DataFrame, np.ndarray]) else return_type

		if design_id is None:
			arr_lhs_out_l = arr_lhs_l
			arr_lhs_out_x = arr_lhs_x

		else:
			# get attribute table and check
			attr_design_id = self.attribute_design_id if (attr_design_id is None) else attr_design_id
			if design_id not in attr_design_id.key_values:
				design_base_assumed = min(attr_design_id.key_values)
				self._log(f"Error in retrieve_lhs_tables_by_design: invalid design_id '{design_id}'. Defaulting to design_id '{design_base_assumed}'.", type_log = "warning")
				design_id = design_base_assumed

			# initialize some variables for determining x/l
			key_vary_l = f"{attr_design_id.key}_to_{field_vary_l}"
			key_vary_x = f"{attr_design_id.key}_to_{field_vary_x}"
			vary_l_q = bool(attr_design_id.field_maps.get(key_vary_l).get(design_id))
			vary_x_q = bool(attr_design_id.field_maps.get(key_vary_x).get(design_id))

			# apply vectorization to get array for LEs if necessary
			np_trans_strat = np.vectorize(self.transform_strategy_lhs_trial_from_design)

			# get LE uncertainty array
			arr_lhs_out_l = None
			if arr_lhs_l is not None:
				arr_lhs_out_l = np_trans_strat(arr_lhs_l, design_id, attr_design_id) if vary_l_q else np.ones(arr_lhs_l.shape)

			# get X uncertainty array
			arr_lhs_out_x = None
			if arr_lhs_x is not None:
				arr_lhs_out_x = arr_lhs_x if vary_x_q else ignore_trial_flag*np.ones(arr_lhs_x.shape)


		if return_type == pd.DataFrame:

			dict_keys = {field_lhs_key: self.vector_lhs_key_values}
			dict_keys.update({attr_design_id.key: design_id}) if (design_id is not None) else None

			arr_lhs_out_l = sf.add_data_frame_fields_from_dict(
				pd.DataFrame(arr_lhs_out_l, columns = self.fields_factors_l),
				dict_keys
			) if (arr_lhs_out_l is not None) else arr_lhs_out_l

			arr_lhs_out_x = sf.add_data_frame_fields_from_dict(
				pd.DataFrame(arr_lhs_out_x, columns = self.fields_factors_x),
				dict_keys
			) if (arr_lhs_out_x is not None) else arr_lhs_out_x

		return arr_lhs_out_l, arr_lhs_out_x



	def transform_strategy_lhs_trial_from_design(self,
		x: Union[float, int],
		design_id: int,
		attr_design: Union[AttributeTable, None] = None,
		field_transform_b: Union[str, None] = None,
		field_transform_m: Union[str, None] = None,
		field_transform_inf: Union[str, None] = None,
		field_transform_sup: Union[str, None] = None,
		include: Union[bool, int] = 1
	) -> float:
		"""
		Transformation function that applies to raw LHS samples to create designs around strategy uncertainties. Based on the following fields
	 and the equation

		 y = max(min(mx + b, sup), inf)

		 where

			* field_transform_b := b
			* field_transform_m := m
			* field_transform_inf := inf
			* field_transform_sup := sup

		Function Arguments
		------------------
		- x: the trial to transform
		- design_id: index in design AttributeTable to use to govern

		Keyword Arguments
		-----------------
		- attr_design: AttributeTable used to pull m, b, inf, and sup
		- field_transform_b: field in AttributeTable giving the value of `b` for each design_id
		- field_transform_m: field in AttributeTable giving the value of `m` for each design_id
		- field_transform_inf: field in AttributeTable giving the value of `inf` for each design_id
		- field_transform_sup: field in  sAttributeTable giving the value of `sup` for each design_id

		"""

		# set fields
		attr_design = self.attribute_design_id if (attr_design is None) else attr_design
		field_transform_b = self.field_transform_b if (field_transform_b is None) else field_transform_b
		field_transform_m = self.field_transform_m if (field_transform_m is None) else field_transform_m
		field_transform_inf = self.field_transform_inf if (field_transform_inf is None) else field_transform_inf
		field_transform_sup = self.field_transform_sup if (field_transform_sup is None) else field_transform_sup

		# initialize output and get parameters from design table
		out = 1.0
		key_b = f"{attr_design.key}_to_{field_transform_b}"
		key_m = f"{attr_design.key}_to_{field_transform_m}"
		key_sup = f"{attr_design.key}_to_{field_transform_sup}"
		key_inf = f"{attr_design.key}_to_{field_transform_inf}"

		b = attr_design.field_maps.get(key_b).get(design_id)
		m = attr_design.field_maps.get(key_m).get(design_id)
		sup = attr_design.field_maps.get(key_sup).get(design_id)
		inf = attr_design.field_maps.get(key_inf).get(design_id)

		if all([(y is not None) for y in [b, m, sup, inf]]):
			out = max(min(m*x + b, sup), inf)

		return out
