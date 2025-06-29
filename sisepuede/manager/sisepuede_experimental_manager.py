import logging
import os, os.path
import pandas as pd
from typing import *

from sisepuede.core.attribute_table import *
from sisepuede.core.model_attributes import ModelAttributes
from sisepuede.data_management.ingestion import *
from sisepuede.data_management.lhs_design import LHSDesign
from sisepuede.data_management.ordered_direct_product_table import *
from sisepuede.data_management.sampling_unit import FutureTrajectories
import sisepuede.utilities._toolbox as sf





class SISEPUEDEExperimentalManager:
	"""Launch and manage experiments based on LHS sampling over trajectories. 
	    The SISEPUEDEExperimentalManager class reads in input templates to 
		generate input databases, controls deployment, generation of multiple 
		runs, writing output to applicable databases, and post-processing of
		applicable metrics. Users should use SISEPUEDEExperimentalManager to
		set the number of trials and the start year of uncertainty.


	Initialization Arguments
	------------------------
	- attribute_design: AttributeTable required to define experimental designs
		to run. Lever effects are evaluated using scalars `y`, which are derived
		from LHC samples that are subject to a linear transformation of the form

			`y = max(min(mx + b, sup), inf)`.

		For each design `_d`, the table passes information on the values of
			`m_d`, `b_d`, `sup_d` and `inf_d`. Each row of the table represents
			a different design.

	The table should include the following fields:
		* `linear_transform_l_b`: field containing `b`
		* `linear_transform_l_m`: field containing `m`
		* `linear_transform_l_inf`: field containing the infinum of lever effect
			scalar
		* `linear_transform_l_sup`: field containing the supremeum of lever
			effect scalar
		* `vary_l`: whether or not lever effects vary in the design (binary)
		* `vary_x`: whether or not exogenous uncertainties vary in the design
			(binary)

	- fp_templates: file path to directory containing input Excel templates
	- model_attributes: ModelAttributes class used to build baseline databases
	- regions: regions (degined in ModelAttributes) to run and build futures
		for.


	Optional Initialization Arguments
	---------------------------------
	- attribute_strategy: AttributeTable defining strategies. If not defined,
		strategies are inferred from templates.
	- demo_database_q: whether or not the input database is used as a demo
		* If run as demo, then `fp_templates` does not need to include
			subdirectories for each region specified
	- fp_exogenous_xl_type_for_variable_specifcations: 
		* If string, tries to read CSV at path containing variable 
			specifications and exogenous XL types. Useful if coordinating across 
			a number of regions.
			(must have fields `field_variable` and `field_xl_type`)
		* If None, instantiates XL types by inference alone.
	- sectors: sectors to include
		* If None, then try to initialize all input sectors
	- random_seed: optional random seed to specify

	Notes
	-----

	"""

	def __init__(self,
		attribute_design: AttributeTable,
		model_attributes: ModelAttributes,
		fp_templates: str,
		regions: Union[list, None],
		# lhs characteristics
		time_period_u0: int,
		n_trials: int,
		# optional/keyword arguments
		attribute_strategy: Union[AttributeTable, None] = None,
		demo_database_q: bool = True,
		sectors: Union[list, None] = None,
		base_future: Union[int, None] = None,
		fan_function_specification: str = "linear",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		field_xl_stype: str = "xl_type",
		fp_exogenous_xl_type_for_variable_specifcations: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
		random_seed: Union[int, None] = None,
	):

		self.model_attributes = model_attributes

		self._initialize_fields(
			field_uniform_scaling_q,
			field_variable,
			field_variable_trajgroup,
			field_variable_trajgroup_type,
			field_xl_stype,
		)

		self._initialize_other_properties(
			n_trials,
			time_period_u0,
			demo_database_q = demo_database_q,
			fan_function_specification = fan_function_specification,
			logger = logger,
			random_seed = random_seed,
		)

		# initialize some SQL information for restoration and/or archival
		self._initialize_archival_settings()

		# initialize key elements of trajectories to sample around
		self._initialize_attribute_design(attribute_design)
		self._initialize_base_future(base_future)
		self._initialize_baseline_database(
			fp_templates,
			regions,
			demo_database_q
		)

		# generate future trajectories and design
		self._initialize_future_trajectories(
			fan_function_specification = self.fan_function_specification,
			field_time_period = self.field_time_period,
			field_uniform_scaling_q = self.field_uniform_scaling_q,
			field_variable = self.field_variable,
			field_variable_trajgroup = self.field_variable_trajgroup,
			field_variable_trajgroup_type = self.field_variable_trajgroup_type,
			fp_exogenous_xl_type_for_variable_specifcations = fp_exogenous_xl_type_for_variable_specifcations,
			key_future = self.key_future,
			key_strategy = self.key_strategy,
			logger = self.logger,
		)
		self._initialize_lhs_design()

		# generate some elements
		self._initialize_primary_keys_index()





	##################################
	#    INITIALIZATION FUNCTIONS    #
	##################################

	def _initialize_archival_settings(self,
	) -> None:
		"""
		Initialize key archival settings used to store necessary experimental
			parameters, Latin Hypercube Samples, ModelAttribute tables, and
			more. Sets the following properties:

			* self.

		"""

		self.archive_table_name_experimental_configuration = "EXPERIMENTAL_CONFIGURATION"
		self.archive_table_name_lhc_samples_l = "LHC_SAMPLES_LEVER_EFFECTS"
		self.archive_table_name_lhc_samples_x = "LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES"

		return None
		


	def _initialize_attribute_design(self,
		attribute_design: AttributeTable,
		field_transform_b: str = "linear_transform_l_b",
		field_transform_m: str = "linear_transform_l_m",
		field_transform_inf: str = "linear_transform_l_inf",
		field_transform_sup: str = "linear_transform_l_sup",
		field_vary_l: str = "vary_l",
		field_vary_x: str = "vary_x",
		logger: Union[logging.Logger, None] = None
	) -> None:
		"""
		Verify AttributeTable attribute_design specified for the design and set
			properties if valid. Initializes the following properties if
			successful:

		* self.attribute_design
		* self.field_transform_b
		* self.field_transform_m
		* self.field_transform_inf
		* self.field_transform_sup
		* self.field_vary_l
		* self.field_vary_x
		* self.key_design


		Function Arguments
		------------------
		- attribute_design: AttributeTable used to define different designs

		Keyword Arguments
		-----------------
		- field_transform_b: field in attribute_design.table giving the value of
			`b` for each attribute_design.key_value
		- field_transform_m: field in attribute_design.table giving the value of
			`m` for each attribute_design.key_value
		- field_transform_inf: field in attribute_design.table giving the value
			of `inf` for each attribute_design.key_value
		- field_transform_sup: field in  attribute_design.table giving the value
			of `sup` for each attribute_design.key_value
		- field_vary_l: required field in attribute_design.table denoting
			whether or not LEs vary under the design
		- field_vary_x: required field in attribute_design.table denoting
			whether or not Xs vary under the design
		- logger: optional logging.Logger() object to log to
			* if None, warnings are sent to standard out

		"""

		# verify input type
		if not isinstance(attribute_design, AttributeTable):
			tp = str(type(attribute_design))
			self._log(f"Invalid type '{tp}' in specification of attribute_design: attribute_design should be an AttributeTable.", type_log = "error")

		# check required fields (throw error if not present)
		required_fields = [
			field_transform_b,
			field_transform_m,
			field_transform_inf,
			field_transform_sup,
			field_vary_l,
			field_vary_x
		]
		sf.check_fields(attribute_design.table, required_fields)

		# if successful, set properties
		self.attribute_design = attribute_design
		self.field_transform_b = field_transform_b
		self.field_transform_m = field_transform_m
		self.field_transform_inf = field_transform_inf
		self.field_transform_sup = field_transform_sup
		self.field_vary_l = field_vary_l
		self.field_vary_x = field_vary_x
		self.key_design = attribute_design.key

		return None



	def _initialize_base_future(self,
		future: Union[int, None]
	) -> None:
		"""
		Set the baseline future. If None, defaults to 0. Initializes the following
			properties:

			* self.baseline_future
		"""

		self.baseline_future = int(min(future, 0)) if (future is not None) else 0

		return None



	def _initialize_baseline_database(self,
		fp_templates: str,
		regions: Union[List[str], None],
		demo_q: bool
	) -> None:
		"""
		Initialize the BaseInputDatabase class used to construct future
			trajectories. Initializes the following properties:

			* self.attribute_strategy
			* self.base_input_database
			* self.baseline_strategy
			* self.regions


		Function Arguments
		------------------
		- fp_templates: path to templates (see ?BaseInputDatabase for more
			information)
		- regions: list of regions to run experiment for
			* If None, will attempt to initialize all regions defined in
				ModelAttributes
		- demo_q: import templates run as a demo (region-independent)?
		"""

		self._log("Initializing BaseInputDatabase", type_log = "info")

		try:
			self.base_input_database = BaseInputDatabase(
				fp_templates,
				self.model_attributes,
				regions,
				demo_q = demo_q,
				logger = self.logger
			)

			self.attribute_strategy = self.base_input_database.attribute_strategy
			self.baseline_strategy = self.base_input_database.baseline_strategy
			self.regions = self.base_input_database.regions

		except Exception as e:
			msg = f"Error initializing BaseInputDatabase -- {e}"
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)

		return None
	


	def _initialize_fields(self,
		field_uniform_scaling_q: str,
		field_variable: str,
		field_variable_trajgroup: str,
		field_variable_trajgroup_type: str,
		field_xl_type: str,
		field_year: str = "year",
	) -> None:
		"""
		Initialize fields and keys used in experiments. Sets the following
			properties:

			* self.field_region
			* self.field_time_period
			* self.field_time_series_id
			* self.field_uniform_scaling_q
			* self.field_variable
			* self.field_variable_trajgroup
			* self.field_variable_trajgroup_type
			* self.field_xl_type
			* self.field_year
			* self.key_future
			* self.key_primary
			* self.key_strategy
		"""
		# initialize some key fields
		self.field_region = self.model_attributes.dim_region
		self.field_time_period = self.model_attributes.dim_time_period
		self.field_time_series_id = self.model_attributes.dim_time_series_id
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable = field_variable
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_xl_type = field_xl_type
		self.field_year = field_year

		# initialize keys--note: key_design is assigned in self._initialize_attribute_design
		self.key_future = self.model_attributes.dim_future_id
		self.key_primary = self.model_attributes.dim_primary_id
		self.key_strategy = self.model_attributes.dim_strategy_id

		return None



	def _initialize_future_trajectories(self,
		fp_exogenous_xl_type_for_variable_specifcations: Union[str, None] = None,
		**kwargs
	) -> None:
		"""
		Initialize the FutureTrajectories object for executing experiments.
			Initializes the following properties:

			* self.dict_future_trajectories
			* self.dict_n_factors
			* self.dict_n_factors_l
			* self.dict_n_factors_x

			Additionally, can update

			* self.regions

			if any regions fail.

		Keyword Arguements
		------------------
		- fp_exogenous_xl_type_for_variable_specifcations: 
			* If string, tries to read CSV at path containing variable 
				specifications and exogenous XL types. Useful if coordinating 
				across a number of regions.
				(must have fields `field_variable` and `field_xl_type`)
			* If None, instantiates XL types by inference alone.
		- **kwargs: passed to FutureTrajectories
		"""

		self._log("Initializing FutureTrajectories", type_log = "info")
		
		self.dict_future_trajectories = {}
		self.dict_n_factors = {}
		self.dict_n_factors_varying = {}
		self.dict_n_factors_l = {}
		self.dict_n_factors_x = {}
		self.dict_sampling_units_varying = {}

		drop_regions = []

		# get strategies to instantiate sampling units for
		attr_strat = self.model_attributes.get_dimensional_attribute_table(
			self.model_attributes.dim_strategy_id
		)
		dict_all_dims = {
			#self.key_time_series #HEREHERE
			self.key_strategy: attr_strat.key_values
		}

		# filter base input database for each region to instantiate a new FutureTrajectories object
		dfg = (
			self.base_input_database
			.database
			.groupby([self.field_region])
		)

		# try retrieving any exogenous variable types - will be None if fails
		dict_exogenous_xl_types = self.get_exogenous_xl_types(
			fp_exogenous_xl_type_for_variable_specifcations,
		)


		for region, df in dfg:

			region = region[0] if isinstance(region, tuple) else region
			region_print = self.get_output_region(region)

			try:
				future_trajectories_cur = FutureTrajectories(
					df.reset_index(drop = True),
					{
						self.key_strategy: self.base_input_database.baseline_strategy,
					},
					self.time_period_u0,
					dict_all_dims = dict_all_dims,
					dict_variable_specification_to_xl_types = dict_exogenous_xl_types,
					**kwargs
				)

				self.dict_future_trajectories.update({
					region: future_trajectories_cur
				})

				
				##  GET SOME NUMBERS OF FACTORS ETC.

				# set the number of factors to sample
				n_factors = len(future_trajectories_cur.all_sampling_units_l)
				n_factors += len(future_trajectories_cur.all_sampling_units_x)

				# all baseline varying (allows for levers as well)
				sampling_units_varying_x = [
					k for k, v in future_trajectories_cur.dict_sampling_units.items()
					if v.x_varies
				]
				n_factors_varying = len(sampling_units_varying_x)


				# assign sampling units
				self.dict_sampling_units_varying.update({
					region: sampling_units_varying_x
				})
				# total number of factors
				self.dict_n_factors.update({
					region: n_factors
				})
				# number of factors that vary as exogenous uncertainties (can include baselines for strategies) 
				self.dict_n_factors_varying.update({
					region: n_factors_varying
				})
				# number of L(ever) factors
				self.dict_n_factors_l.update({
					region: len(future_trajectories_cur.all_sampling_units_l)
				})
				# number of (e)X(ogenous uncertainties)
				self.dict_n_factors_x.update({
					region: len(future_trajectories_cur.all_sampling_units_x)
				})

				self._log(f"\tFutureTrajectories for '{region_print}' complete.", type_log = "info")

			except Exception as e:
				self._log(
					f"Error initializing FutureTrajectories for region {region_print} -- {e}.", 
					type_log = "error"
				)
				self._log(
					f"Dropping region '{region_print}' due to error in FutureTrajectories initialization.", 
					type_log = "warning"
				)
				drop_regions.append(region)

		# update regions if necessary
		self.regions = [x for x in self.regions if (x not in drop_regions)]
		if len(self.regions) == 0:
			raise RuntimeError(f"Error initializing SISEPUEDE: no regions left to instantiate.")

		return None



	def _initialize_lhs_design(self,
	) -> None:
		"""
		Initializes LHS design and associated tables used in the Experiment.
			Creates the following properties:

			* self.dict_lhs_design
			* self.vector_lhs_key_values

			Additionally, can update

			* self.regions

			if any regions fail.
		"""

		self._log("Initializing LHSDesign", type_log = "info")

		self.dict_lhs_design = {}
		self.vector_lhs_key_values = None

		drop_regions = []

		for region in self.regions:

			region_print = self.get_output_region(region)

			try:
				
				fields_factors_x = self.dict_sampling_units_varying.get(region)
				future_trajectories_cur = self.dict_future_trajectories.get(region)
				n_factors_varying = self.dict_n_factors_varying.get(region)
				n_factors_l = self.dict_n_factors_l.get(region)

				lhs_design_cur = LHSDesign(
					self.attribute_design,
					self.key_future,
					n_factors_l = n_factors_l,
					n_factors_x = n_factors_varying,
					n_trials = self.n_trials,
					random_seed = self.random_seed,
					fields_factors_l = future_trajectories_cur.all_sampling_units_l,
					fields_factors_x = fields_factors_x,#future_trajectories_cur.all_sampling_units,
					logger = self.logger
				)

				self.dict_lhs_design.update({
					region: lhs_design_cur
				})

				self.vector_lhs_key_values = (
					lhs_design_cur.vector_lhs_key_values 
					if (self.vector_lhs_key_values is None) 
					else self.vector_lhs_key_values
				)

				self._log(f"\tLHSDesign for region '{region_print}' complete.", type_log = "info")

			except Exception as e:
				self._log(f"Error initializing LHSDesign for region '{region_print}' -- {e}.", type_log = "error")
				self._log(f"Dropping region '{region_print}' due to error in LHSDesign initialization.", type_log = "warning")
				drop_regions.append(region)

		# update regions if necessary
		self.regions = [x for x in self.regions if (x not in drop_regions)]
		if len(self.regions) == 0:
			raise RuntimeError(f"Error initializing SISEPUEDE: no regions left to instantiate.")



	def _initialize_other_properties(self,
		n_trials: int,
		time_period_u0: int,
		demo_database_q: bool = True,
		fan_function_specification: str = "linear",
		logger: Union[logging.Logger, None] = None,
		random_seed: Union[int, None] = None,
	) -> None:
		"""
		Set some key parameters used in managing the experiment (dependent on 
			self.model_attributes). Sets the following properties:

			* self.demo_mode
			* self.fan_function_specification
			* self.logger
			* self.n_trials
			* self.sort_ordered_dimensions_of_analysis
			* self.time_period_u0
			* self.random_seed
		"""
		# ordered by sort hierarchy
		self.sort_ordered_dimensions_of_analysis = self.model_attributes.sort_ordered_dimensions_of_analysis

		# initialize additional components
		self.demo_mode = demo_database_q
		self.fan_function_specification = fan_function_specification
		self.logger = logger
		self.n_trials = n_trials
		self.time_period_u0 = time_period_u0
		self.random_seed = random_seed

		return None
	


	def _initialize_primary_keys_index(self,
	) -> None:
		"""
		Generate a data frame of primary scenario keys. Assigns the following
			properties:

			* self.primary_key_database
		"""

		self._log(f"Generating primary keys (values of {self.key_primary})...", type_log = "info")

		# get all designs, strategies, and futures
		all_designs = self.attribute_design.key_values
		all_strategies = self.base_input_database.attribute_strategy.key_values
		all_futures = [self.baseline_future]
		all_futures += self.vector_lhs_key_values if (self.vector_lhs_key_values is not None) else []

		odtp_database = OrderedDirectProductTable(
		    {
				self.key_design: all_designs,
				self.key_future: all_futures,
				self.key_strategy: all_strategies
			},
		    [self.key_design, self.key_strategy, self.key_future],
		    key_primary = self.key_primary,
		)

		self.primary_key_database = odtp_database

		return None



	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
		"""
		Clean implementation of sf._optional_log in-line using default logger.
			See ?sf._optional_log for more information.

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
		sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



	def _restore_from_database(self,
		table_name_experimental_configuration: Union[str, None] = None,
		table_name_lhs_l: Union[str, None] = None,
		table_name_lhs_x: Union[str, None] = None
	) -> None:
		"""
		Restore a SISEPUEDE Experimental Session from an SQL database containing
			the following tables:

			* NEEDS TO BE FILLED OUT

		"""


		return None




	##############################
	#    SUPPORTING FUNCTIONS    #
	##############################

	def generate_database(self,
		list_primary_keys: Union[list, None] = None
	) -> pd.DataFrame:
		"""
		Generate an data of inputs for primary keys specified in list_primary_keys.

		Optional Arguments
		------------------
		- list_primary_keys: list of primary keys to include in input database.
			* If None, uses
		"""
		return None
	


	def get_exogenous_xl_types(self,
		fp_exogenous_xl_type_for_variable_specifcations: Union[str, None],
	) -> Union[Dict[str, str], None]:
		"""
		Try reading in exogenous XL types from external file. If successful,
			returns a dictionary mapping variable specifications to XL types
			(does not have to be exhaustive).

		Function Arguements
		-------------------
		- fp_exogenous_xl_type_for_variable_specifcations: 
			* If string, tries to read CSV at path containing variable 
				specifications and exogenous XL types. Useful if coordinating 
				across a number of regions.
				(must have fields `field_variable` and `field_xl_type`)
			* If None, instantiates XL types by inference alone.
		"""

		# some basic checks on inputs; if fed None, will return None
		return_none = not isinstance(fp_exogenous_xl_type_for_variable_specifcations, str)
		return_none |= (
			not os.path.exists(fp_exogenous_xl_type_for_variable_specifcations)
			if not return_none
			else return_none
		)
		
		if return_none:

			return None
		
		# try reading the inputs
		df_inputs = None

		try:
			df_inputs = pd.read_csv(fp_exogenous_xl_type_for_variable_specifcations)
		except Exception as e:
			self._log(
				"Error in try_retrieving_exogenous_xl_types: {e}. Exogenous XL types for variable specifications will be inferred.",
				type_log = "error",
			)

			return None
		
		# check fields
		if not set([self.field_variable, self.field_xl_type]).issubset(set(df_inputs.columns)):
			self._log(
				f"Error in try_retrieving_exogenous_xl_types: one or more of '{self.field_variable}', '{self.field_xl_type}' not found in the data frame. Exogenous XL types for variable specifications will be inferred.",
				type_log = "error",
			)
			return None

		
		# otherwise, build dictionary and return
		dict_out = sf.build_dict(df_inputs[[self.field_variable, self.field_xl_type]])

		return dict_out
	


	def get_output_region(self,
		region: str,
		str_demo_region: str = "DEMO"
	) -> str:
		"""
		Retrieve a region for output tables

		Function Arguments
		------------------
		- region: input region to convert

		Keyword Arguments
		-----------------
		- str_demo_region: string specifying a region for a demo run
		"""

		out = str_demo_region if self.demo_mode else region

		return out
