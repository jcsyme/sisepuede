import logging
import pandas as pd
import sqlalchemy
from typing import *


from sisepuede.core.analysis_id import AnalysisID
from sisepuede.core.model_attributes import ModelAttributes
from sisepuede.data_management.iterative_database import IterativeDatabase
import sisepuede.utilities._toolbox as sf




class SISEPUEDEOutputDatabase:
	"""
	Manage output from SISEPUEDE in a cohesive and flexible database structure,
		including output to SQLite, CSVs, or remote SQL databases. The output
		database includes a number of tables and allows for the specification
		of optional post-processing functions and additional tables.

	The following is a list of table name keyword arguments for all default
		tables generated ny SISEPUEDE, along with the table's default name and
		a description of the table.

		* table_name_analysis_metadata -> "ANALYSIS_METADATA"
			* The analysis metadata table stores information associated with
				each unique SISEPUEDE run, including configuration options
				(both analytical and experimental parameters), time of run,
				analysis id, and more.
			* No indexing
			* The analysis metadata table does not suppoort the specification
				of derivative tables

		* table_name_attribute_design -> "ATTRIBUTE_DESIGN"
			* The design attribute table stores the attribute_design table
				associated with the run.
			* Indexed by design key
			* Derivative tables that summarize designs can be passed using
				`dict_derivative_table_functions`. Derivative tables of
				"ATTRIBUTE_DESIGN" are indexed by the design key.

		* table_name_attribute_lhs_l -> "ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS"
			* The lever effect latin hypercube sample attribute table stores the
				latin hypercube samples associated with plausible future lever
				effects.
			* Indexed by future key
			* Derivative tables that summarize LHS samples for lever effect
				uncertainties can be passed using
				`dict_derivative_table_functions`. Derivative tables of
				"ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS" are indexed by the
				primary key.

		* table_name_attribute_lhs_x -> "ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES"
			* The exogenous uncertainties latin hypercube sample attribute table
				stores the latin hypercube samples associated with plausible
				future uncertainties.
			* Indexed by future key
			* Derivative tables that summarize LHS samples for exogenous
				uncertainties can be passed using
				`dict_derivative_table_functions`. Derivative tables of
				"ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES" are indexed by
				the primary key.

		* table_name_attribute_primary -> "ATTRIBUTE_PRIMARY"
			* The primary key attribute table stores the attribute_primary table
				associated with the run
			* Indexed by primary key.
			* Derivative tables that summarize primary keys can be passed using
				the `dict_derivative_table_functions`. Derivative tables of
				"ATTRIBUTE_PRIMARY" are indexed by the primary key.

		* table_name_attribute_strategy -> "ATTRIBUTE_STRATEGY"
			* The strategy attribute table stores the attribute_strategy table
				associated with the run, which governs information about
				strategies (across all sectors).
			* Indexed by strategy key
			* Derivative tables that summarize strategies can be passed using
				the `dict_derivative_table_functions`. Derivative tables of
				"ATTRIBUTE_STRATEGY" are indexed by the strategy key.

		* table_name_base_input -> "MODEL_BASE_INPUT_DATABASE"
			* The base input database is derived from input templates and is used
				as the basis for generating all future trajectories. It is stored
				in the SISEPUEDE class as `SISEPUEDE.base_input_datase.database`.
			* No indexing
			* Derivative tables that summarize base inputs can be passed using
				the `dict_derivative_table_functions`. Derivative tables of
				"MODEL_BASE_INPUT_DATABASE" do not have indexing.

		* table_name_input -> "MODEL_INPUT"
			* The model input database is the entire range of futures. To avoid
				unecessary storage, this database, by default, is *not* written
				to a table. Note that individual futures (and the table itself)
				can be reproduced quickly using SISEPUEDE's internal functions
				in combination with LHS tables, which are saved by default.
			* Indexed by primary key
			* Derivative tables that summarize inputs can be passed using
				the `dict_derivative_table_functions`. Derivative tables of
				"MODEL_INPUT" will also be indexed by the primary key.

		* table_name_output -> "MODEL_OUTPUT"
			* The model output database includes all outputs across the entire
				range of futures. While large, this database is included in
				default outputs to save compute power/time.
			* Indexed by primary key
			* Derivative tables that summarize outputs can be passed using
				the `dict_derivative_table_functions`. Derivative tables of
				"MODEL_OUTPUT" will also be indexed by the primary key.


	Initialization Arguments
	------------------------
	- engine: string specifying output method, or, optionally, sqlalchemy
		engine connected to a database/schema used to output data. Options for
		export are:
		* string:
			* "csv": exports CSVs to subdirectory (associated with analysis run id)
				located in output directory
			* "sqlite": exports all tables to a SQL lite database located in the
				output directory
		* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
			and schema to write output tables to. This engine can be used to write
			to a remote database service.
		* None:
			If `None`, defaults to SQLite in SISEPUEDE output subdirectory
	- dict_dimensional_keys: dictionary providing keys for design, primary, and
		strategy. The dictionary *MUST* take the following form:

			dict_dimensional_keys = {
				"design": key_design,
				"future": key_future,
				"primary": key_primary,
				"region": key_region,
				"strategy": key_strategy,
				"time_series": key_time_series
			}

		where the values are strings giving the key value.

	- fp_base_output: output file path to write output to *excluding the file
		extension*.
		* If engine is an instance of sqlalchemy.engine.Engine, then
			fp_base_output is unused
		* If engine == "csv", then the tables are saved under the directory
			fp_base_output; e.g.,

			fp_base_output
			|_ table_1.csv
			|_ table_2.csv
			.
			.
			.
			|_ table_n.csv

		* if engine == "sqlite", then the tables are saved to an sqlite
			database at f"{fp_base_output}.sqlite"
		* if None, defaults to export in the present working directory.
		* NOTE: fp_base_output should NOT include the file extension


	Optional Arguments
	------------------
	- analysis_id: optional specification of a SISEPUEDE analysis run id. Can be
		enetered in any of the following forms:
		* SISPUEDEAnalysisID: pass a SISPUEDEAnalysisID object to use
		* str: pass a string of a SISPUEDEAnalysisID; this will initialize a
			new SISPUEDEAnalysisID object within the database structure, but
			allows for connections with databases associated with the specified
			SISEPUEDEAnalysisID
		* None: initialize a new SISPUEDEAnalysisID for the database

	- dict_derivative_table_functions: optional dictionary used to specify
		additional tables (not a standard output). The dictionary maps new table
		names to a tuple, where the first element of the tuple, source_table,
		represents a source table used to calculate the derivative table and
		FUNCTION_APPLY_i gives a function that is applied to the source table
		(source_table) to develop the derivative table TABLE_NAME_i (dictionary
		key). The dictionary should then have the following form:

		dict_derivative_table_functions = {
			"TABLE_NAME_1": (source_table, FUNCTION_APPLY_1),
			.
			.
			.
			"TABLE_NAME_N": (source_table, FUNCTION_APPLY_N)
		}

		The function to apply, FUNCTION_APPLY_i, requires the following 
			positional arguments:

			(1) model_attributes:ModelAttributes
			(2) df_source:pd.DataFrame

		Each function should return a data frame. In docstring form, the 
			function would be defined using the following form:

			def FUNCTION_APPLY_i(
				model_attributes:ModelAttributes,
				df_source:pd.DataFrame
			) -> pd.DataFrame:

		If dict_derivative_table_functions is None, or if the dictionary is 
			empty, then no derivative tables are generated or written.

	- logger: optional log object to pass


	Keyword Arguments
	-----------------
	- create_dir_output: Create output directory implied by fp_base_output if it
		does not exist
	- tables_write_exclude: list of tables to exclude from writing to output. 
		Default is ["inputs"] (unless extensive storage is available, writing 
		raw inputs is not recommended.)

		The following are table names for output tables, which can be changed 
			using keyword arguments. See the descriptions above of each for 
			default names of tables. Note that the following lists include the 
			"names" to use in `tables_write_exclude` to exclude that table from 
			output.

			- table_name_attribute_design: table name to use for storing the 
				attribute table for the design key
				* To exclude from the output database, include 
					"attribute_design" in `tables_write_exclude`

			- table_name_attribute_lhs_l: table name to use for storing the 
				attribute table for lever effect Latin Hypercube samples
				* To exclude from the output database, include "attribute_lhs_l" 
					in `tables_write_exclude`

			- table_name_attribute_lhs_x: table name to use for storing the 
				attribute table for exogenous uncertainty Latin Hypercube 
				samples
				* To exclude from the output database, include "attribute_lhs_x" 
					in `tables_write_exclude`

			- table_name_analysis_metadata: table name to use for storing 
				analysis metadata
				* To exclude from the output database, include 
					"analysis_metadata" in `tables_write_exclude`

			- table_name_attribute_primary: table name to use for storing the 
				attribute table for the primary key
				* To exclude from the output database, include 
					"attribute_primary" in `tables_write_exclude`

			- table_name_attribute_strategy: table name to use for storing the 
				attribute table for the strategy key
				* To exclude from the output database, include 
					"attribute_strategy" in `tables_write_exclude`

			- table_name_base_input: table name to use for storing the base 
				input database used to input variables
				* To exclude from the output database, include "base_input" in
					`tables_write_exclude`

			- table_name_input: table name to use for storing the complete 
				database of SISEPUEDE model inputs
				* To exclude from the output database, include "input" in
					`tables_write_exclude`

			- table_name_output: table name to use for storing the complete 
				database of SISEPUEDE model outputs
				* To exclude from the output database, include "output" in
					`tables_write_exclude`


	The following arguments are passed to IterativeDatabaseTable

	- fields_index: optional fields used to index the table. This can be used to
		prevent writing information to the table twice.
		* If using an SQL engine, specifying index fields provides the option to
			drop rows containing the index.
		* If using CSVs, will disallow writing to the file if an index is 
			already specified.
			* CAUTION: if the CSV is manipulated while a table session is
				is running, the index will not be automatically renewed. If 
				writing to the table from another program, you must re-run
				`IterativeDatabaseTable._initialize_table_indices()` to ensure 
				the index is up to date.
		* Avoid specifying many fields when working with large tables to 
			preserve memory requirements.
		* If None, no indexing protection is provided.
	- index_conflict_resolution: string or None specifying approach to deal
		with attempts to write new rows to a table that already contains
		index values. Can take the following options:
		* None: do nothing, write as is (not recommended)
		* skip: write nothing to the table
		* stop: stop the process if an index conflict is identified.
		* write_replace (SQL only--same as `write_skip` for CSV): write all
			rows associated with new index values and replace any rows
			associated with those index values that already exist in the
			table.
		* write_skip: write all rows associated with new index values and
			skip any rows associated with index value that are already
			contained in the data frame.
	- keep_stash: if a write attempt fails, the "stash" allows failed rows to be
		stored in a dictionary (IterativeDatabaseTable.stash)
	- replace_on_init: replace a table existing at the connection on 
		initialization?
		* Note: should be used with caution. If True, will eliminate any table
			at the specified directory
	"""
	def __init__(self,
		engine: Union[sqlalchemy.engine.Engine, str, None],
		dict_dimensional_keys: Dict[str, str],
		analysis_id: Union[AnalysisID, str, None] = None,
		fields_index: Union[List[str], None] = None,
		fp_base_output: Union[str, None] = None,
		create_dir_output: bool = True,
		logger: Union[logging.Logger, None] = None,
		dict_derivative_table_functions: Union[Dict[str, Tuple[str, Callable[[ModelAttributes, pd.DataFrame], pd.DataFrame]]], None] = None,
		table_name_analysis_metadata: str = "ANALYSIS_METADATA",
		table_name_attribute_design: str = "ATTRIBUTE_DESIGN",
		table_name_attribute_lhs_l: str = "ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS",
		table_name_attribute_lhs_x: str = "ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES",
		table_name_attribute_primary: str = "ATTRIBUTE_PRIMARY",
		table_name_attribute_strategy: str = "ATTRIBUTE_STRATEGY",
		table_name_base_input: str = "MODEL_BASE_INPUT_DATABASE",
		table_name_input: str = "MODEL_INPUT",
		table_name_output: str = "MODEL_OUTPUT",
		tables_write_exclude: Union[List[str], None] = ["input"],
		# IterativeDatabaseTable Keywords
		keep_stash: bool = False,
		index_conflict_resolution: Union[str, None] = "write_skip",
		replace_on_init = False,
	):
		# initiqlize some simple properties
		self.keep_stash = keep_stash
		self.logger = logger
		self.replace_on_init = replace_on_init

		self._check_dict_dimensional_keys(dict_dimensional_keys)
		self._initialize_table_dicts(
			dict_derivative_table_functions,
			table_name_analysis_metadata,
			table_name_attribute_design,
			table_name_attribute_lhs_l,
			table_name_attribute_lhs_x,
			table_name_attribute_primary,
			table_name_attribute_strategy,
			table_name_base_input,
			table_name_input,
			table_name_output
		)
		self._initialize_output_database(
			engine,
			analysis_id = analysis_id,
			fp_base_output = fp_base_output,
			create_dir_output = create_dir_output,
			keep_stash = self.keep_stash,
			logger = self.logger,
			replace_on_init = self.replace_on_init,
		)

		return None




	#############################
	#    INITIALIZE DATABASE    #
	#############################

	def _check_dict_dimensional_keys(self,
		dict_dimensional_keys: Dict[str, str]
	) -> None:
		"""
		Check the dictionary of dimensional keys. The dictionary *MUST* take the
			following form:

				dict_dimensional_keys = {
					"design": key_design,
					"future": key_future,
					"primary": key_primary,
					"region": key_region,
					"strategy": key_strategy,
					"time_series": key_time_series
				}

			where the values are strings giving the key value.

			Sets the following properties:

			* self.dict_dimensional_keys

			If keys are missing, sets to None.
		"""
		self.required_dimensional_keys = [
			"design",
			"future",
			"primary",
			"region",
			"strategy",
			"time_series"
		]

		self.dict_dimensional_keys = {}

		for key in self.required_dimensional_keys:
			val = None
			if key in dict_dimensional_keys.keys():
				val = dict_dimensional_keys.get(key)
				val = [val] if isinstance(val, str) else None
				self.dict_dimensional_keys.update({key: val})

			# throw an error if missing
			if val is None:
				self._log(f"Missing key dict_dimensional_keys: key {key} not found. Tables that rely on the {key} will not have index checking.", type_log = "warning")

		# set keys
		self.key_design = self.dict_dimensional_keys.get("design")
		self.key_future = self.dict_dimensional_keys.get("future")
		self.key_primary = self.dict_dimensional_keys.get("primary")
		self.key_region = self.dict_dimensional_keys.get("region")
		self.key_strategy = self.dict_dimensional_keys.get("strategy")
		self.key_time_series = self.dict_dimensional_keys.get("time_series")

		return None



	def _initialize_output_database(self,
		engine: Union[sqlalchemy.engine.Engine, str, None],
		**kwargs
	) -> None:
		"""
		Initialize the IterativeDatabase for SISEPUEDE. Sets the following
			properties and aliases:

			* self.analysis_id
			* self.db
			* self.engine
			* self.id
			* self.id_fs_safe
			* self.fp_base_output
			* self.read_table
			* self._write_to_table


		Function Arguments
		------------------
		- engine: string specifying output method, or, optionally, sqlalchemy
			engine connected to a database/schema used to output data. Options for
			export are:
			* string:
				* "csv": exports CSVs to subdirectory (associated with analysis run id)
					located in output directory
				* "sqlite": exports all tables to a SQL lite database located in the
					output directory
			* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
				and schema to write output tables to. This engine can be used to write
				to a remote database service.
			* None:
				If `None`, defaults to SQLite in SISEPUEDE output subdirectory

		Keyword Arguments
		-----------------
		- **kwargs: keyword arguments passed to IterativeDatabase, including:
			analysis_id: Union[AnalysisID, str, None] = None,
			fields_index_default: Union[List[str], None] = None,
			fp_base_output: Union[str, None] = None,
			create_dir_output: bool = True,
			logger: Union[logging.Logger, None] = None,
			# IterativeDatabaseTable Keywords
			keep_stash: bool = False,
			index_conflict_resolution: Union[str, None] = "write_skip",
			replace_on_init = False
		"""

		# initialize
		self.analysis_id = None
		self.db = None
		self.engine = None
		self.fp_base_output = None
		self.id = None
		self.id_fs_safe = None
		self.read_table = None
		self._write_to_table = None

		try:
			self.db = IterativeDatabase(
				engine,
				self.dict_all_tables,
				**kwargs
			)

			self._log(f"SISEPUEDEOutputDatabase successfully initialized IterativeDatabase.", type_log = "info")

		except Exception as e:
			self._log(f"Error in SISEPUEDEOutputDatabase initializing IterativeDatabase: {e}", type_log = "error")

		# set some aliases
		if self.db is not None:
			self.analysis_id = self.db.analysis_id
			self.engine = self.db.engine
			self.fp_base_output = self.db.fp_base_output
			self.id = self.db.id
			self.id_fs_safe = self.db.id_fs_safe
			self.read_table = self.db.read_table
			self._write_to_table = self.db._write_to_table

		return None



	def _initialize_table_dicts(self,
		dict_derivative_table_functions: Union[Dict[str, Tuple[str, Callable[[ModelAttributes, pd.DataFrame], pd.DataFrame]]], None],
		table_name_analysis_metadata: str,
		table_name_attribute_design: str,
		table_name_attribute_lhs_l: str,
		table_name_attribute_lhs_x: str,
		table_name_attribute_primary: str,
		table_name_attribute_strategy: str,
		table_name_base_input: str,
		table_name_input: str,
		table_name_output: str,
		derivate_table_conflict_appendage: str = "DERIV",
		str_fields_index_param: str = "fields_index"
	) -> None:
		"""
		Initialize input table dictionary for IterativeDatabase. Sets the
			following properties:

			* self.dict_all_tables
			* self.dict_derivative_table_functions
			* self.tables_indexed_by_design
			* self.tables_indexed_by_future
			* self.tables_indexed_by_primary
			* self.tables_indexed_by_strategy
			* self.tables_indexed_by_time_series
			* self.table_name_analysis_metadata
			* self.table_name_attribute_design
			* self.table_name_attribute_lhs_l
			* self.table_name_attribute_lhs_x
			* self.table_name_attribute_primary
			* self.table_name_attribute_strategy
			* self.table_name_base_input
			* self.table_name_input
			* self.table_name_output

		Function Arguments
		------------------

		Keyword Arguments
		-----------------
		- derivate_table_conflict_appendage: string appendage used to resolve
			conflicts in table names passed `dict_derivative_table_functions`.
			If any of the keys in derivate_table_conflict_appendage exist as
			SISEPUEDE table names, this string is appended to the table name.
			An error is thrown if that table is also contained in the SISEPUEDE
			input tables, as users should reconsider naming the table to avoid
			conflicts.
		- str_fields_index_param: parameter name for index fields to pass to
			IterativeDatabaseTable for each table.
		"""

		##  INITIALIZATION

		dict_derivative_table_functions = {} if not isinstance(dict_derivative_table_functions, dict) else dict_derivative_table_functions

		self.table_name_analysis_metadata = table_name_analysis_metadata
		self.table_name_attribute_design = table_name_attribute_design
		self.table_name_attribute_lhs_l = table_name_attribute_lhs_l
		self.table_name_attribute_lhs_x = table_name_attribute_lhs_x
		self.table_name_attribute_primary = table_name_attribute_primary
		self.table_name_attribute_strategy = table_name_attribute_strategy
		self.table_name_base_input = table_name_base_input
		self.table_name_input = table_name_input
		self.table_name_output = table_name_output


		##  SET TABLE CLASSES AND GROUP BY INDEXING

		# tables index by each dimension
		self.tables_indexed_by_design = [
			table_name_attribute_design
		]
		self.tables_indexed_by_future = [
			table_name_attribute_lhs_l,
			table_name_attribute_lhs_x
		]
		self.tables_indexed_by_primary = [
			table_name_attribute_primary,
			table_name_input,
			table_name_output
		]
		self.tables_indexed_by_strategy = [
			table_name_attribute_strategy
		]
		self.tables_indexed_by_time_series = [
		]

		# inialize the dictionary of inputs to IterativeDatabase
		dict_all_tables = {
			table_name_analysis_metadata: None,
			table_name_attribute_design: {
				str_fields_index_param: self.key_design
			},
			table_name_attribute_lhs_l: {
				str_fields_index_param: (self.key_region + self.key_future)
			},
			table_name_attribute_lhs_x: {
				str_fields_index_param: (self.key_region + self.key_future)
			},
			table_name_attribute_primary: {
				str_fields_index_param: self.key_primary
			},
			table_name_attribute_strategy: {
				str_fields_index_param: self.key_strategy
			},
			table_name_base_input: {
				str_fields_index_param: self.key_region
			},
			table_name_input: {
				str_fields_index_param: (self.key_region + self.key_primary)
			},
			table_name_output: {
				str_fields_index_param: (self.key_region + self.key_primary)
			}
		}


		##  CHECK DERIVATIVE TABLE FUNCTIONS AND SET PROPERTIES

		dict_return_functions, dict_return_idf_specs = self.validate_derivative_tables(
			dict_all_tables,
			dict_derivative_table_functions
		)

		dict_all_tables.update(dict_return_idf_specs) if (dict_return_idf_specs is not None) else None
		self.all_tables = sorted(list(dict_all_tables.keys()))
		self.dict_all_tables = dict_all_tables
		self.dict_derivative_table_functions = dict_return_functions if (dict_return_functions is not None) else None

		return None



	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
		"""
		Clean implementation of sf._optional_log in-line using default logger. See
			?sf._optional_log for more information.

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
		sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



	def validate_derivative_table_function(self,
		src_table: str,
		function: Callable,
		dict_source_tables: Union[Dict, None] = None
	) -> Tuple[str, Dict[str, Callable]]:
		"""
		Validate specified derivative table function specifications.

		Returns

			(src_table, function)

		if valid and

			(None, None)

		if invalid.


		Function Arguments
		------------------
		- src_table: source table to apply derivative function to
		- function: callable (function or method) to apply to source table to
			generate derivative table

		Keyword Arguments
		-----------------
		- dict_source_tables: dictionary of source tables. If None, uses
			self.dict_all_tables
		"""

		# initialize output and dict_source_tables
		out = (None, None)
		dict_source_tables = self.dict_all_tables if not isinstance(dict_source_tables, dict) else dict_source_tables

		# simple approach to start with--check type and presence in tables
		if isinstance(src_table, str) and isinstance(function, Callable):
			if src_table in dict_source_tables.keys():
				out = (src_table, function)

		return out



	def validate_derivative_tables(self,
		dict_all_tables: Union[Dict[str, Dict[str, Any]], None],
		dict_derivative_table_functions: Union[Dict[str, Tuple[str, Callable[[ModelAttributes, pd.DataFrame], pd.DataFrame]]], None],
		append_integer_to_field: bool = False,
		dict_source_tables: Union[Dict, None] = None,
		max_iter: int = 10000,
		str_fields_index_param: str = "fields_index"
	) -> Tuple[Dict[str, Callable], Dict[str, Dict[str, Any]]]:
		"""
		Validate the specification of dertivative table functions by
			checking names against names contained in the SISEPUEDE
			database.

		Returns a tuple of the form

			dict_return_functions, dict_return_idf_specs

			where:

			* `dict_return_functions` is the validated, cleaned version of the
				dictionary `dict_derivative_table_functions`, mapping table
				names to derivative table functions (see
				?SISEPUEDEOutputDatabase for more information on this function)
			* dict_return_idf_specs is a dictionary that is merged to
				self.dict_all_tables to specify input tables for
				IterativeDatabase


		Function Arguments
		------------------
		- dict_all_tables: dictionary of table specifications to pass to
			IterativeDatabase containing SISPUEDE default tables.

		Keyword Arguments
		-----------------
		- append_integer_to_field: append an integer (counting upwards) to
			fields that are duplicates of existing fields? Prevents an error
			but may lead to confusing table names.
		- dict_source_tables: dictionary of source tables. If None, uses
			self.dict_all_tables
		- max_iter: maximum number of iterations to use in an attempt to
			append integers to conflicting fields.
		- str_fields_index_param: parameter name for index fields to pass to
			IterativeDatabaseTable for each table.
		"""

		dict_return_functions = {}
		dict_return_idf_specs = {}

		for x in dict_derivative_table_functions.keys():

			# first, check the specification
			tup_func = dict_derivative_table_functions.get(x)
			table_src, func = self.validate_derivative_table_function(*tup_func)

			if table_src is not None:
				key_try = x
				if x in dict_all_tables.keys():
					self._log(f"Derivative table name '{x}' already found in list of SISEPUEDE tables. Trying to rename...", type_log = "warning")
					key_try = f"{x}_{derivate_table_conflict_appendage}"

					if key_try in dict_all_tables.keys():
						error_q = True
						# try to append an integer to the field to prevent conflicts
						if append_integer_to_field:
							error_q = False
							i = 0
							key_try = f"{key_try}_{i}"
							while (key_try in dict_all_tables.keys()) and (i < max_iter):
								i += 1
								key_try = f"{key_try}_{i}"

							error_q = True if (i == max_iter) else error_q

						if error_q:
							msg = f"Error trying to set derivative table '{x}' as name {key_try}. Try a different specification for the table. "
							self._log(msg, type_log = "error")
							raise RuntimeError(msg)

					else:
						self._log(f"Dertivative table '{x}' successfully renamed to '{key_try}'", type_log = "info")

				fields_index = self.dict_all_tables.get(table_src)
				fields_index = fields_index.get(str_fields_index_param) if isinstance(fields_index, dict) else None
				dict_return_functions.update({key_try: func})
				dict_return_idf_specs.update({key_try: {str_fields_index_param: fields_index}})

		return dict_return_functions, dict_return_idf_specs
