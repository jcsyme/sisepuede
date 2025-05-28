from sisepuede.core.analysis_id import AnalysisID
import datetime
import logging
import numpy as np
import os, os.path
import pandas as pd
from typing import *
import sisepuede.utilities._toolbox as sf
import sqlalchemy
import sisepuede.utilities._sql as sqlutil



class IterativeDatabaseTable:
	"""
	Create a data table to use for storing output results. Includes 
		functionality for reading, writing, verifying keys, and more.

		* Includes a potential "stash" (IterativeDatabaseTable.stash), which 
			stores rows that failed to successfully write to the table.


	Initialization Arguments
	------------------------
	- table_name: name of the table, which is used across media; e.g., same in
		SQL as in CSVs
	- engine: string specifying (1) table parent directory or (2) sqlalchemy
		engine connected to a database/schema used to output data.
		* table parent directory: exports CSVs with name `table_name` to
			directory `engine`.
			* Note: Ensure `create_directory = True` to force the creation of
				the output directory.
		* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
			and schema to write output tables to. This engine can be used to 
			write to a remote database service.

	Optional Arguments
	------------------
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

	Keyword Arguments
	-----------------
	- create_directory: create directories for target CSV table if needed?
	- logger: optional logging.Logger object for event logging


	"""

	def __init__(self,
		table_name: str,
		engine: Union[sqlalchemy.engine.Engine, str, None],
		fields_index: Union[List[str], None] = None,
		create_directory: bool = True,
		index_conflict_resolution: str = "write_replace",
		keep_stash: bool = True,
		replace_on_init: bool = False,
		logger: Union[logging.Logger, None] = None
	):
		# initialize some simple properties - note, columns is initialized on the first read or write
		self.keep_stash = keep_stash
		self._logger = logger
		self.table_name = table_name

		self._clean_stash()
		self._initialize_engine(engine, create_directory = create_directory)
		self._initialize_table(replace_on_init = replace_on_init)
		self._initialize_table_indices(fields_index)
		self._initialize_conflict_resolution(index_conflict_resolution)


	################################################
	#    INITIALIZATION AND WORKHORSE ARGUMENTS    #
	################################################

	def check_table_indices(self,
		df_addition: pd.DataFrame
	) -> Tuple[Set[Tuple], Union[Set[Tuple], None]]:
		"""
		Update the available indices when writing a data frame. Returns

				None

			if the fields_index is not initialized and (a) the table does not
			exist or (b) columns are not initialized. Otherwise, returns a
			tuple of the following form

				set_new_indices, set_in_index = Union[Set[Tuple], None], Union[Set[Tuple], None]

			where

				* `set_new_indices` is the set of indices to add
				* `set_in_index` is the set of indices that are already in the index

			* NOTE: If the index is initialized but `df_addition` does not contain it,

				`set_new_indices` = `set_new_indices` = None



		Function Arguments
		------------------
		- df_addition: data frame with index columns present (self.fields_index)
		"""

		# case where table is not yet initiated (no tuple)
		if (self.fields_index is None) and ((not self.exists) or (self.columns is None)):
			return None

		# case where index fields are defined
		if (self.fields_index is not None):
			if not set(self.fields_index).issubset(df_addition.columns):
				missing_cols = sf.print_setdiff(set(self.fields_index), set(df_addition.columns))
				msg = f"Error in check_table_indices: index columns {missing_cols} not found."
				self._log(msg, type_log = "error")
				return None, None

		# case where columns are defined
		if (self.columns is not None):
			if not set(self.columns).issubset(df_addition.columns):
				missing_cols = sf.print_setdiff(set(self.columns), set(df_addition.columns))
				msg = f"Error in check_table_indices: columns {missing_cols} not found."
				self._log(msg, type_log = "error")
				return None, None

		# case where table is initiated but does not have an index
		if self.fields_index is None:
			return None, None

		# get new index values that may be updated and the intersection
		new_indices = set(
			[tuple(x) for x in np.array(df_addition[self.fields_index].drop_duplicates())]
		)
		shared_indices = self.available_indices.intersection(new_indices)

		return new_indices, shared_indices



	def _clean_stash(self,
	) -> None:
		"""
		Initialize and clean self.stash, the dictionary that
			stores failed table writes.

		Keyword Arguments
		-----------------
		- clear: delete all elements of the stash if it exists
		"""
		self.stash = None
		if self.keep_stash:
			self.stash = {}
			self._log(f"Cleaned stash", type_log = "info")



	def filter_data_frame_from_index_tuples(self,
		df_write: pd.DataFrame,
		elements: Union[Set, None]
	) -> Union[pd.DataFrame, None]:
		"""
		Filter a data frame based on available index tuples

		Function Arguments
		------------------
		- df_write: data frame to write, which contains index fields to filter
			on
		- elements: index tuples used to filter
		"""

		if (elements is None) or (self.fields_index is None):
			return df_write

		df_merge = pd.DataFrame(list(elements), columns = self.fields_index)

		df_merge = pd.merge(
			df_merge,
			df_write,
			on = self.fields_index,
			how = "inner"
		)

		return df_merge



	def _initialize_conflict_resolution(self,
		index_conflict_resolution: Union[str, None],
		default_resolution: str = "write_replace"
	) -> None:
		"""
		Check specification of `index_conflict_resolution` and initialize the
			following properties:

			* self.index_conflict_resolution
			* self.valid_resolutions

		Keyword Arguments
		-----------------
		- default_resolution: default resolution approach to use if an invalid
			approach is entered. Must be a member of self.valid_resolutions.
		"""

		self.valid_resolutions = [
			"skip",
			"stop",
			"write_replace",
			"write_skip"
		]
		self.index_conflict_resolution = default_resolution if (index_conflict_resolution not in self.valid_resolutions) else index_conflict_resolution



	def _initialize_engine(self,
		engine: Union[sqlalchemy.engine.Engine, str, None],
		create_directory: bool = True
	) -> None:
		"""
		Check the engine and set the following properties:

			* self.engine
			* self.fp_table
			* self.interaction_type

		Function Arguments
		------------------
		- engine: sqlalchemy.engine.Engine connecting to SQL database or
			string giving directory to export CSVs to.
			* If not a string or sqlalchemy.engine.Engine, will set to export
				CSVs to the current working directory.

		Keyword Arguments
		-----------------
		- create_directory: if using an output directory as database for tables,
			create the directory if it does not exist?
		"""

		self.engine = None
		self.fp_table = None
		self.interaction_type = None

		engine = engine if (
			isinstance(engine, str) or isinstance(engine, sqlalchemy.engine.Engine)
		) else os.getcwd()

		if isinstance(engine, str):
			dir_table = sf.check_path(
				engine,
				create_q = create_directory,
				throw_error_q = True
			)
			self.fp_table = os.path.join(dir_table, f"{self.table_name}.csv")
			self.interaction_type = "csv"

		elif isinstance(engine, sqlalchemy.engine.Engine):
			self.engine = engine
			self.interaction_type = "sql"



	def _initialize_table(self,
		engine: Union[sqlalchemy.engine.Engine, str, None] = None,
		fp_table: Union[str, None] = None,
		replace_on_init: bool = False,
	) -> None:
		"""
		Initialize some table-specific properties: determine whether or not the
			table exists already, and, depdending on `replace_on_init`, start a
			new table. If the table exists and is not to be replaced, initializes
			columns.

			* self.columns
			* self.exists

		"""

		engine = self.engine if (not isinstance(engine, sqlalchemy.engine.Engine)) else engine
		fp_table = self.fp_table if (not isinstance(fp_table, str)) else fp_table

		self.columns = None
		self.exists = False

		if self.interaction_type == "sql":

			query_delete = None

			table_names = sqlutil.get_table_names(engine, error_return = [])
			self.exists = (self.table_name in table_names)

			if self.exists:
				# if not replacing, initialize columns; otherwise, delete the table and wait to initialize columns until the first call to "write"
				if not replace_on_init:

					# try retrieving columns
					query = f"select * from {self.table_name} limit 0;"
					df_columns = None
					try:
						self._log(f"Trying to get columns for {self.table_name} with query: {query}", type_log = "debug")
						with engine.connect() as con:
							df_columns = pd.read_sql_query(query, con)

						self._log(f"Table {self.table_name} found in sql connection; it will not be replaced.", type_log = "info")

					except Exception as e:
						self._log(f"Query {query_delete} failed with error: {e}", type_log = "error")

					# update columns if successful
					if (df_columns is not None):
						self.columns = list(df_columns.columns)
						self._log(f"\tColumns successfully retrieved.", type_log = "info")


				else:
					# try to remove the table
					query_delete = f"drop table {self.table_name};"
					try:
						self._log(
							f"Trying to remove {self.table_name} with query: {query_delete}", 
							type_log = "debug",
						)

						with engine.connect() as con:
							con.execute(sqlalchemy.text(query_delete))
							con.commit()

						self._log(
							f"Table {self.table_name} was found in sql connection and was successsfully removed. Columns will be initialized on the first write.", 
							type_log = "info",
						)

					except Exception as e:
						self._log(
							f"Query {query_delete} failed with error: {e}", 
							type_log = "error",
						)
						query_delete = None

					# update existence
					self.exists = False if (query_delete is not None) else self.exists


		elif self.interaction_type == "csv":

			self.exists = os.path.exists(fp_table) if (fp_table is not None) else False

			if self.exists:
				# if not replacing, initialize columns; otherwise, delete the table and wait to initialize columns until the first call to "write"
				if not replace_on_init:

					# try retrieving columns
					df_columns = None
					try:
						self._log(f"Trying to get columns for {self.table_name} from table at {fp_table}.", type_log = "debug")
						df_columns = pd.read_csv(fp_table, nrows = 0)

						self._log(f"Table {self.table_name} found at {fp_table}; it will not be replaced.", type_log = "info")

					except Exception as e:
						self._log(f"Attempt to access CSV at {fp_table} failed with error: {e}", type_log = "error")

					# update columns if successful
					if (df_columns is not None):
						self.columns = list(df_columns.columns)
						self._log(f"\tColumns successfully retrieved.", type_log = "info")

				else:
					os.remove(fp_table) if os.path.exists(fp_table) else None
					self._log(f"Table {self.table_name} was found at {fp_table} and was successsfully removed. Columns will be initialized on the first write.", type_log = "info")
					self.exists = False
		
		return None



	def _initialize_table_indices(self,
		fields_index: Union[List[str], None] = None,
		index_chunk_size: int = 100000
	) -> None:
		"""
		Initialize available index field tuples in the table. Sets the following
			properties:

			* self.available_indices
			* self.fields_index

		Function Arguments
		------------------
		- fields_index: fields in table to use as indices.
			* Warning: keep short with CSVs, and avoid using indices that will

		Optional Arguments
		-----------------
		- index_chunk_size: chunk size to use in calling sf.get_csv_subset

		"""

		# initialize properties and check
		self.available_indices = None
		self.fields_index = None

		if (fields_index is None) or (fields_index == []):
			self._log(f"No index fields defined. Index field values will not be checked when writing to tables.", type_log = "warning")
			return None

		if (not self.exists) or (self.columns is None):
			self._log(f"No index fields found in {self.table_name}. Initializing index fields.", type_log = "warning")
			self.fields_index = fields_index
			self.available_indices = set()
			return None

		# re-initialize after determining existence
		self.fields_index = [x for x in fields_index if (x in self.columns)]
		self.available_indices = set() if (len(self.fields_index) > 0) else self.available_indices

		# initialize defaults
		index_chunk_size = int(max(index_chunk_size, 1))
		str_log_msg = ""

		if (self.interaction_type == "sql"):
			try:
				query = ", ".join(self.fields_index)
				query = f"select distinct {query} from {self.table_name};"

				with self.engine.connect() as con:
					df_index = pd.read_sql_query(query, con)

				str_log_msg = f"SQL."

			except Exception as e:
				self._log(f"Error trying to initialize available indices from SQL: {e}", type_log = "error")
				return None


		elif (self.interaction_type == "csv"):
			try:
				df_index = sf.get_csv_subset(
					self.fp_table,
					None,
					fields_extract = self.fields_index,
					chunk_size = index_chunk_size,
					drop_duplicates = True
				)

				str_log_msg = f"CSV at {self.fp_table}"

			except Exception as e:
				self._log(f"Error trying to initialize available indices from CSV at {self.fp_table}: {e}", type_log = "error")
				return None

		self.available_indices = set([tuple(x) for x in np.array(df_index)])
		self._log(f"Successfully initiatlized available indices from {str_log_msg}", type_log = "info")

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
		sf._optional_log(self._logger, msg, type_log = type_log, **kwargs)

		return None



	def _stash(self,
		df_stash: Union[pd.DataFrame, None]
	) -> None:
		"""
		Add `df_stash` to the stash. Uses isoformat of datetime.now() to
		"""
		key = datetime.datetime.now().isoformat()

		if isinstance(df_stash, pd.DataFrame) and self.keep_stash and isinstance(self.stash, dict):
			self.stash.update({key: df_stash})
			shp = str(df_stash.shape)
			self._log(f"Successfully stashed data frame with shape {shp}", type_log = "info")
		else:
			self._log(f"Stash failed", type_log = "warning")

		return None



	def _verify_existence(self,
		check_columns: bool = False,
	) -> None:
		"""
		Verify that the table exists. Sets the following properts:

		* self.exists

		Keyword Arguments
		-----------------
		- check_columns: if True, check that columns in target line with
			self.columns
		"""

		cols_cur = self.columns
		error_msg = f"Column asymmetry discovered in table {self.table_name} -- check the table."

		if self.interaction_type == "sql":

			table_names = sqlutil.get_table_names(self.engine, error_return = [])
			self.exists = (self.table_name in table_names)

			if check_columns and self.exists:
				query = f"select * from {self.table_name} limit 0;"
				df_columns = None
				try:
					with self.engine.connect() as con:
						df_columns = pd.read_sql_query(query, con)
					cols_cur = list(df_columns.columns)

				except Exception as e:
					self._log(f"Query {query} failed with error: {e}", type_log = "error")
					cols_cur = []

		elif self.interaction_type == "csv":
			self.exists = os.path.exists(self.fp_table)
			if check_columns and self.exists:
				try:
					df_columns = pd.read_csv(self.fp_table, nrows = 0)
					cols_cur = list(df_columns.columns)

				except Exception as e:
					self._log(f"CSV read from {self.fp_table} failed with error: {e}", type_log = "error")
					cols_cur = []

		if check_columns:
			if (cols_cur != self.columns) and (self.columns is not None):
				self._log(error_msg, type_log = "error")
				raise RuntimeError(error_msg)
				
			elif (self.columns is None) and self.exists:
				self.columns = cols_cur

		self.columns = None if not self.exists else self.columns

		return None





	############################
	#    CORE FUNCTIONALITY    #
	############################

	def _check_columns_and_index_on_write(self,
		df_write: pd.DataFrame
	) -> None:
		"""
		Check columns in df_write against self.columns, and initialize
			self.columns if it does not exist.

			* Additionally, checks columns against self.fields_index if they
			are specified and prevents initialization if the index fields are
			not found in the data frame.

			* Sets the following properties if unset:

				* self.columns
		"""

		columns = list(df_write.columns) if not self.exists else self.columns
		set_ind_check = set(self.fields_index) if (self.fields_index is not None) else set({})

		if not set_ind_check.issubset(set(columns)):
			missing_fields = sf.print_setdiff(set_ind_check, set(columns))
			raise KeyError(f"Index fields {missing_fields} not found in df_write.")

		self.columns = columns

		return None



	def _destroy(self,
	)->None:
		"""
		Delete the target table and reset columns.
		"""

		# no action if the table does not exist
		if not self.exists:
			return None

		delete_q = False
		if (self.interaction_type == "sql"):
			# try to remove the table
			query_delete = f"drop table {self.table_name};"
			try:
				self._log(
					f"Trying to remove {self.table_name} with query: {query_delete}", 
					type_log = "debug",
				)

				with self.engine.connect() as con:
					con.execute(sqlalchemy.text(query_delete))
					con.commit()
                

				self._log(
					f"Table {self.table_name} was found in sql connection and was successsfully removed. Columns will be initialized on the first write.", 
					type_log = "info",
				)

				delete_q = True


			except Exception as e:
				self._log(f"Query {query_delete} failed with error: {e}", type_log = "error")
				return None

		elif (self.interaction_type == "csv"):

			try:
				os.remove(self.fp_table) if (self.fp_table is not None) else None
				delete_q = True

			except Exception as e:
				self._log(
					f"Attempt to remove table {self.fp_table} failed with error: {e}", 
					type_log = "error",
				)
				return None

		# update existence - only occurs if successful
		if delete_q:
			self._initialize_table_indices(self.fields_index)
			self.columns = None
			self.exists = False
		
		return None
	


	def _drop_indices(self,
		indices_drop: List[Tuple[Any]],
	) -> Union[List[Tuple], None]:
		"""
		Drop rows associated with indices frame a table. Returns a tuple of 
			indices that were successfully dropped. If no valid tuples are
			found, or an error occurs while executing the query, returns None.

		NOTE: Only available for SQL at moment.

		Modifies the following properties:

			* self.available_indices


		Function Arguments
		------------------
		- indices_Drop: List of ordered tuples mapping an index field to drop 
			values, e.g., 

			[
				(val_00, val_01), (val_01, val_11), ...
			]

		"""

		return_none = not sf.islistlike(indices_drop)
	
		if not return_none:
			indices_drop = [x for x in indices_drop if x in self.available_indices]
			return_none |= (len(indices_drop) == 0)

		if return_none:
			return None

		# build query
		key = tuple(self.fields_index)
		query = sqlutil.format_tuples_to_query_filter_string(self.fields_index, indices_drop)
		query = f"delete from {self.table_name} where {query};"

		try:
			with self.engine.connect() as con:
				con.execute(sqlalchemy.text(query))
				con.commit()
			
			success = True

		except Exception as e:
			msg = f"""
				An error occured in IterativeDatabaseTable._drop_indices() with 
				table name {self.table_name}: {e}
			"""

			msg_debug = f"""
				Error {e} in IterativeDatabaseTable._drop_indices() found while 
				attempting the following query:\n\n{query}
			"""
			self._log(msg, type_log = "error")
			self._log(msg_debug, type_log = "debug")
			success = False

		# return dropped indices if successful and drop from available indicies
		out = None
		if success:
			out = indices_drop
			self.available_indices -= set(indices_drop)
		
		return out



	def read_table(self,
		dict_subset: Union[Dict[str, List], None] = None,
		fields_select: Union[List[str], None] = None,
		drop_duplicates: bool = False,
		query_logic: str = "and",
		**kwargs,
	) -> pd.DataFrame:
		"""Read a subset of rows from a table.

		Optional Arguments
		------------------
		dict_subset: Union[Dict[str, List], None]
		    Dictionary with keys that are columns in the table and values, given 
			as a list, to subset the table. dict_subset is written as:

			dict_subset = {
				field_a = [val_a1, val_a2, ..., val_am],
				field_b = [val_b1, val_b2, ..., val_bn],
				.
				.
				.
			}
		fields_select : Union[List[str], None]
		    Fields to read in. Reducing the number of fields to read
			can speed up the ingestion process and reduce the data frame's memory
			footprint.

		Keyword Arguments
		-----------------
		drop_duplicates : bool
		    Drop duplicates in the CSV when reading?
			* Default is False to improve speeds
			* Set to True to ensure that only unique rows are read in
		query_logic : str
		    Default is "and". Subsets table to as

			where field_a in (val_a1, val_a2, ..., val_am) ~ field_b in (val_b1, val_b2, ..., val_bn)...

			where `~ in ["and", "or"]`
		"""

		df_return = None
		if not self.exists:
			return None

		if self.interaction_type == "sql":
			#
			query_append = sqlutil.dict_subset_to_query_append(
				dict_subset,
				query_logic = query_logic
			) if (dict_subset is not None) else None

			try:
				df_return = sqlutil.sql_table_to_df(
					self.engine,
					self.table_name,
					fields_select = fields_select,
					query_append = query_append
				)

			except Exception as e:
				self._log(f"Error trying to read table {self.table_name} via {self.interaction_type}: {e}", type_log = "error")
				return None

		elif self.interaction_type == "csv":

			df_return = sf.get_csv_subset(
				self.fp_table,
				dict_subset,
				fields_extract = fields_select,
				drop_duplicates = drop_duplicates
			)

		return df_return



	def resolve_index_conlicts(self,
		df_write: Union[pd.DataFrame, None],
		index_conflict_resolution: Union[str, None] = None
	) -> Tuple[pd.DataFrame, set]:
		"""
		Resolve conflicts betweeen index rows in df_write and those in
			the table. Returns a tuple of the following objects:

			df_write, set_indices_to_write, set_indices_to_stash

			where

			* `df_write` is the data frame to write that has been filtered
				to remove any rows associated with conflicting indices (if
				applicable, based on the value of
				`index_conflict_resolution`)
			* `set_indices_to_write` is the set of index tuples that will be
				written to the table
			* `set_indices_to_stash` is the set of index tuples that will not
				be written but that can, instead, be "stashed" in the
				self.stash list of dataframes.


		Function Arguments
		------------------
		- df_write: data frame containing index columns to check
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

			If None, defaults to self.index_conflict_resolution
		"""

		##  RUN CHECKS AND INITIALIZE - start by returning special cases

		# case where no index is defined
		check_indices_tuple = self.check_table_indices(df_write)
		if check_indices_tuple is None:
			return df_write, None, None

		set_new_indices, set_shared_indices = check_indices_tuple
		df_out = None

		# case where the data frame is missing required columns - return df_write as None
		if (set_new_indices is None) and (set_shared_indices is None):
			return None, None, None

		# case where there are no conflicts
		if len(set_shared_indices) == 0:
			return df_write, set_new_indices, set_shared_indices

		# set valid resolution conflicts and initialize output sets
		index_conflict_resolution = self.index_conflict_resolution if (index_conflict_resolution not in self.valid_resolutions) else index_conflict_resolution
		set_indices_to_write = set({})
		set_indices_to_stash = set({})

		# build a log message
		msg_str = sf.str_replace(
			f"Warning: index tuples {set_shared_indices} found in table {self.table_name}.",
			{"{": "", "}": ""}
		)
		self._log(msg_str, type_log = "warning")
		self._log(f"Running conflict resolution '{index_conflict_resolution}' for table {self.table_name}.", type_log = "warning")


		##  RESOLVE CONFLICTS BASED ON index_conflict_resolution

		# STOP on a conflict
		if index_conflict_resolution == "stop":
			raise RuntimeError(msg_str)


		# SKIP on conflict, and stash the entire dataframe
		elif index_conflict_resolution == "skip":

			set_indices_to_stash = set_new_indices
			self._stash(df_write)
			out = (df_out, set_indices_to_write, set_indices_to_stash) # df_out is None

			return out


		# ATTEMPT TO REPLACE rows with existing indices - only applicable if interaction type is SQL
		#   defaults to write_stash if the query fails
		elif (index_conflict_resolution == "write_replace") and (self.interaction_type == "sql"):

			# get query to delete existing rows
			query = sqlutil.format_listlike_elements_for_filter_query(
				list(set_shared_indices),
				self.fields_index
			)
			query = f"delete from {self.table_name} where {query};"
			self._log(f"Trying query: {query}", type_log = "info")

			try:
				with self.engine.connect() as con:
					con.execute(sqlalchemy.text(query))
					con.commit()

				# if successful, output dataframe is entire data frame, and set_indices_to_write is overwritten to all
				df_out = df_write
				set_indices_to_write = set_new_indices

				self._log(
					f"Successfully removed index rows for overwrite in {self.table_name} in SQL.", 
					type_log = "info",
				)


			except Exception as e:
				self._log(
					f"Deletion of rows failed: {e}. Setting index_conflict_resolution = 'write_skip'", 
					type_log = "warning",
				)
				index_conflict_resolution == "write_skip"



		##  SKIP AND STASH - break out to allow write_replace to shift to write_skip on a fail
		#  - note: depends on other conditionals, so leave as an 'if' statement

		skip_and_stash = (index_conflict_resolution == "write_skip")
		skip_and_stash |= ((self.interaction_type == "csv") and (index_conflict_resolution == "write_replace"))

		if skip_and_stash:
			set_indices_to_write = set_new_indices.difference(set_shared_indices)
			set_indices_to_stash = set_shared_indices
			df_out = self.filter_data_frame_from_index_tuples(df_write, set_indices_to_write)

			df_stash = (
				self.filter_data_frame_from_index_tuples(
					df_write, 
					set_shared_indices
				) 
				if self.keep_stash 
				else None
			)
			self._stash(df_stash)


		return df_out, set_indices_to_write, set_indices_to_stash



	def _write_to_table(self,
		df_write: pd.DataFrame,
		append_q: bool = True,
		index_conflict_resolution: Union[str, None] = None,
		reinitialize_on_verification_failure: bool = False,
		verify: bool = True,
	) -> None:
		"""
		Write a data frame to the table. Can include table initialization
			(if not self.exists) and appending to an existing table.

		Function Arguments
		------------------
		- df_write: DataFrame to write to the table

		Keyword Arguments
		-----------------
		- append_q: append to an existing table if found? Default is True.
			* If False, REPLACES existing table
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


			Note that "write_replace" and "write_skip" have the same behavior for
			CSV tables.
		- reinitialize_on_verification_failure: reinitialize columns following
			a failure in verification? Only should be True if known bugs lead to
			column names being modified
		- verify: verify table columns before each write

		"""

		##  CHECK TABLE AND RESOLVE CONFLICTS

		# verify existence; reinitialize if forced to given errors
		try:
			self._verify_existence(check_columns = True)
		except Exception as e:
			if not reinitialize_on_verification_failure:
				raise RuntimeError(e)
			self._initialize_table()

		index_conflict_resolution = self.index_conflict_resolution if (index_conflict_resolution not in self.valid_resolutions) else index_conflict_resolution
		df_write, set_indices_to_write, set_indices_to_stash = self.resolve_index_conlicts(
			df_write,
			index_conflict_resolution
		)

		if df_write is None:
			return None

		# other init
		write_str = "appended" if (self.exists & append_q) else "written"


		##  WRITE UNDER DIFFERENT CONDITIONS

		# use _check_columns_and_index_on_write() to set the columns and check against index fields
		if self.interaction_type == "sql":
			try:
				self._check_columns_and_index_on_write(df_write)
				sqlutil._write_dataframes_to_db(
					{
						self.table_name: df_write[self.columns]
					},
					self.engine,
					append_q = self.exists & append_q
				)

				self.exists = True
				self.available_indices.update(set_indices_to_write) if (set_indices_to_write is not None) else None
				self._log(f"Table {self.table_name} successfully {write_str} to database.", type_log = "info")

			except Exception as e:
				self._log(f"Error in _write_to_table trying to write {self.table_name}: {e}", type_log = "error")


		elif self.interaction_type == "csv":
			try:
				self._check_columns_and_index_on_write(df_write)
				mode = "a" if (self.exists & append_q) else "w"

				df_write[self.columns].to_csv(
					self.fp_table,
					index = None,
					encoding = "UTF-8",
					mode = mode,
					header = (mode == "w")
				)

				self.exists = True
				self.available_indices.update(set_indices_to_write) if (set_indices_to_write is not None) else None
				self._log(f"Table {self.table_name} successfully {write_str} to {self.fp_table}.", type_log = "info")

			except Exception as e:
				self._log(f"Error in _write_to_table trying to write {self.table_name}: {e}", type_log = "error")






class IterativeDatabase:
	"""Manage tables from iterative models in a cohesive and flexible database
		structure, including output to SQLite, CSVs, or remote SQL databases.
		The output database includes a number of tables and allows for the
		specification of optional post-processing functions and additional
		tables.


	Initialization Arguments
	------------------------
	engine : Union[sqlalchemy.engine.Engine, str, None]
	    String specifying output method, or, optionally, sqlalchemy engine 
		connected to a database/schema used to output data. Options for export 
		are:
		* string:
			* "csv": exports CSVs to subdirectory (associated with analysis run
				id) located in output directory
			* "sqlite": exports all tables to a SQL lite database located in the
				output directory
		* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
			and schema to write output tables to. This engine can be used to
			write to a remote database service.
		* None:
			If `None`, defaults to SQLite in SISEPUEDE output subdirectory
	dict_all_tables : Union[Dict[str, Dict[str, Any]], List[str], None]
	    Dictionary mapping a list of tables to initialize and store OR list of 
		table names (converted to dictionary of keys to None).
		
		The dictionary maps keys (table names) to dictionaries that define
		properties of the table to pass to the IterativeDatabase. If there are
		no properties to pass, the value of the entry should be associated with
		`None`. An example of the input dictionary is:

			dict_all_tables = {
				TABLE_1: {
					"fields_index": [field_1, field_2],
					"keep_stash": True
				},
				TABLE_2: None,
				TABLE_3: {
					"keep_stash": False
				}
			}

		* If tables are not initialized with a property, they revert to using
			the IterativeDatabaseTable default associated with that property.
		* Once tables are initialized, the can be modified using the
			IterativeDatabaseTable `read_table` and `_write_to_table` methods.
	fp_base_output : Union[str, None]
	    Output file path to write output to *excluding the file extension*.
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

		* if engine == "sqlite", then the tables are saved to an sqlite database
			at f"{fp_base_output}.sqlite"
		* if None, defaults to export in the present working directory.
		* NOTE: fp_base_output should NOT include the file extension


	Optional Arguments
	------------------
	analysis_id : Union[AnalysisID, str, None]
	    Optional specification of a SISEPUEDE analysis run id. Can be entered in 
		any of the following forms:
		* AnalysisID: pass an AnalysisID object to use
		* str: pass a string of an AnalysisID; this will initialize a
			new AnalysisID object within the database structure, but
			allows for connections with databases associated with the specified
			AnalysisID
		* None: initialize a new AnalysisID for the database
	fields_index_default : Union[List[str], None]
	    Optional fields used to index tables by default. This can be used to 
		prevent writing information to the table twice.
		* If using an SQL engine, specifying index fields provides the option to
			drop rows containing the index.
		* If using CSVs, will disallow writing to the file if an index is
			already specified.
			* CAUTION: if the CSV is manipulated while a table session is
				running, the index will not be automatically renewed. If writing
				to the table from another program, you must re-run
				`IterativeDatabaseTable._initialize_table_indices()` to ensure
				the index is up to date.
		* Avoid specifying many fields when working with large tables to
			preserve memory requirements.
		* If None, no indexing protection is provided.
		* If no value for "fields_index" is passed for a table in
			`dict_all_tables` initializes without a index fields (i.e., as
			None)
	logger : Union[logging.Logger, None]
	    Optional log object to pass



	Keyword Arguments
	-----------------
	create_dir_output : bool
	    Create output directory implied by fp_base_output if it does not exist?

	The following arguments are passed to IterativeDatabaseTable

	index_conflict_resolution : Union[str, None]
	    String or None specifying approach to deal with attempts to write new 
		rows to a table that already contains index values. Can take the 
		following options:
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
	keep_stash : bool
	    If a write attempt fails, the "stash" allows failed rows to be stored in 
		a dictionary (IterativeDatabaseTable.stash). Applies to all tables 
		unless specified otherwise in `dict_all_tables`
	replace_on_init : False
	    Replace a table existing at the connection on initialization?
		* Note: should be used with caution. If True, will eliminate any table
			at the specified directory

	"""

	def __init__(self,
		engine: Union[sqlalchemy.engine.Engine, str, None],
		dict_all_tables: Union[Dict[str, Dict[str, Any]], List[str], None],
		analysis_id: Union[AnalysisID, str, None] = None,
		fields_index_default: Union[List[str], None] = None,
		fp_base_output: Union[str, None] = None,
		create_dir_output: bool = True,
		logger: Union[logging.Logger, None] = None,
		# IterativeDatabaseTable Keywords
		keep_stash: bool = False,
		index_conflict_resolution: Union[str, None] = "write_skip",
		replace_on_init: bool = False,
	) -> None:
		# initialize some simple properties
		self.keep_stash = keep_stash
		self.logger = logger
		self.replace_on_init = replace_on_init

		self._initialize_analysis_id(analysis_id, )
		self._initialize_engine(
			engine,
			fp_base_output,
			create_dir_output = create_dir_output,
		)
		self._initialize_tables(
			dict_all_tables,
			fields_index_default,
			index_conflict_resolution,
		)

		return None



	##############################################
	#	INITIALIZATION AND SUPPORT FUNCTIONS	#
	##############################################

	def get_idt_engine(self,
	) -> Union[sqlalchemy.engine.Engine, str]:
		"""
		Get the input engine for IterativeDatabaseTable
		"""
		engine = self.engine if (self.interaction_type in ["sql", "sqlite"]) else self.fp_base_output
		return engine



	def _initialize_analysis_id(self,
		analysis_id: Union[AnalysisID, str, None]
	) -> None:
		"""
		Initialize the session id. Initializes the following properties:

			* self.analysis_id (AnalysisID object)
			* self.id (shortcurt to self.analysis_id.id)
			* self.id_fs_safe
		"""

		# initialize
		aid = None
		id_shortcut = None
		id_fs_safe = None

		if isinstance(analysis_id, AnalysisID):
			aid = analysis_id

		elif isinstance(analysis_id, str):

			try:
				aid = AnalysisID(
					id_str = analysis_id,
					logger = self.logger
				)

			except Exception as e:

				self._log(
					f"Invalid analysis_id string {analysis_id}: {e}", 
					type_log = "warning"
				)

				raise RuntimeError(e)

		id_shortcut = aid.id if (aid is not None) else None
		id_fs_safe = aid.id_fs_safe if (aid is not None) else None


		##  SET PROPERTIES

		self.analysis_id = aid
		self.id = id_shortcut
		self.id_fs_safe = id_fs_safe

		return None



	def _initialize_engine(self,
		engine: Union[sqlalchemy.engine.Engine, str],
		fp_base_output: Union[str, None],
		create_dir_output: bool = True,
		default_engine_str: str = "sqlite"
	) -> None:
		"""
		Initialize the output directory. Sets the following properties:

			* self.engine
			* self.fp_base_output
			* self.interaction_type
			* self.valid_engine_strs

		Function Arguments
		------------------
		- engine: string specifying output method, or, optionally, sqlalchemy
		engine connected to a database/schema used to output data. Options for
		export are:
			* string:
				* "csv": exports CSVs to subdirectory (associated with analysis 
					run id) located in output directory
				* "sqlite": exports all tables to a SQL lite database located in 
					the output directory
			* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a 
				database
				and schema to write output tables to. This engine can be used to write
				to a remote database service.
			* None:
				If `None`, defaults to output in present working directory
		- fp_base_output: base file path to use for output (appends .sqlite if writing
			to SQLite database, or treated as output directory for CSVs).
			* Unused if engine is a sqlalchemy.engine.Engine

		keyword Arguments
		-----------------
		- create_dir_output: create the output directory if it does not exist?
		- default_engine_str: default output approach if invalid entries are
			specified

		"""
		# set some valid sets
		self.valid_engine_strs = ["csv", "sqlite"]
		self.fp_base_output = None
		default_engine_str = "sqlite" if (default_engine_str not in self.valid_engine_strs) else default_engine_str

		# check the export engine type
		if not (isinstance(engine, str) or isinstance(engine, sqlalchemy.engine.Engine)):
			tp = str(type(engine))
			engine = default_engine_str
			self._log(f"Invalid export engine type {tp} specified: setting to str '{default_engine_str}'.", type_log = "warning")

		#
		if isinstance(engine, str):

			# check default engine spcification and engine
			#self.engine = default_engine_str if (engine not in self.valid_engine_strs) else engine
			self.interaction_type = default_engine_str if (engine not in self.valid_engine_strs) else engine
			self._log(f"\tSetting export engine to '{self.interaction_type}'.", type_log = "info")

			# initialize a default
			fbn_outputs = f"{self.id_fs_safe}_" if (self.id_fs_safe is not None) else ""
			fbn_outputs = f"{fbn_outputs}outputs_raw"

			self.fp_base_output = os.path.join(
				os.getcwd(),
				fbn_outputs
			)


			##  CLEAN fp_base_output AND CHECK DIRECTORY IF APPLICABLE

			if isinstance(fp_base_output, str):

				fp_base_output = sf.str_replace(
					fp_base_output,
					dict([(f".{x}", "") for x in (self.valid_engine_strs)])
				)
				dir_output = fp_base_output if (self.interaction_type == "csv") else os.path.dirname(fp_base_output)
				
				if create_dir_output:
					dir_output = sf.check_path(dir_output, True)

				elif not os.path.exists(dir_output):
					msg = f"FATAL ERROR: specified output directory '{dir_output}' does not exist. To avoid this error, do not set create_dir_output = False"
					self._log(msg, type_log = "error")
					raise RuntimeError(msg)

				self.fp_base_output = fp_base_output

			# update for sqlite specification
			self.fp_base_output = f"{self.fp_base_output}.sqlite" if (engine == "sqlite") else self.fp_base_output


			##  CREATE SQLITE ENGINE IF NEEDED

			str_prepend_sqlite = "sqlite:///"
			self.engine = (
				sqlalchemy.create_engine(f"{str_prepend_sqlite}{self.fp_base_output}") 
				if (self.interaction_type == "sqlite") 
				else self.interaction_type
			)

		elif isinstance(engine, sqlalchemy.engine.Engine):

			self.engine = engine
			self.interaction_type = "sql"
			self._log(f"\tSetting export engine to SQLAlchemy engine.", type_log = "info")


		return None



	def _initialize_tables(self,
		dict_all_tables: Union[Dict[str, Dict[str, Any]], List[str], None],
		fields_index_default: Union[List[str], None],
		index_conflict_resolution: Union[str, None],
		dict_idt: Union[Dict[str, IterativeDatabaseTable], None] = None,
		overwrite_existing_if_in_dict_idt: bool = False
	) -> None:
		"""
		Initialize the output database based on the engine. Sets the following
			properties:

			* self.all_tables
			* self.dict_all_tables
			* self.dict_iterative_database_tables
			* self.fields_index
			* self.index_conflict_resolution

		Function Arguments
		------------------
		- dict_all_tables: dictionary mapping a list of tables to initialize and
			store OR list of table names (converted to dictionary of keys to 
			None). The dictionary maps keys (table names) to dictionaries that 
			define properties of the table to pass to the IterativeDatabase. If 
			there are no properties to pass, the value of the entry should be 
			associated with `None`. An example of the input dictionary is:

				dict_all_tables = {
					TABLE_1: {
						"fields_index": [field_1, field_2],
						"keep_stash": True
					},
					TABLE_2: None,
					TABLE_3: {
						"keep_stash": False
					}
				}

			* If tables are not initialized with a property, they revert to 
				using the IterativeDatabaseTable default associated with that 
				property.
			* Once tables are initialized, the can be modified using the
				IterativeDatabaseTable `read_table` and `_write_to_table` 
				methods.
		- fields_index_default: default fields_index_default to use if none are
			specified in dict_all_tables.
		- index_conflict_resolution: conflict resolution approach to take (see
			IterativeDatabaseTable)

		Keyword Arguments
		-----------------
		- dict_idt: existing dictionary of IterativeDatabaseTables to pass. Not
			used in initialization.
		- overwrite_existing_if_in_dict_idt: if passing an
			IterativeDatabaseTable in dict_idt with a name that already exists,
			should it overwrite the existing table? USE WITH CAUTION TO AVOID
			ERASING TABLES.
		"""

		# set table information
		dict_all_tables = (
			dict([(str(x), None) for x in dict_all_tables]) 
			if isinstance(dict_all_tables, list) 
			else dict_all_tables
		)

		self.dict_all_tables = {} if not isinstance(dict_all_tables, dict) else dict_all_tables
		self.all_tables = sorted(list(self.dict_all_tables.keys()))

		# set engine and some defaults
		engine = self.get_idt_engine()
		self.fields_index_default = fields_index_default if isinstance(fields_index_default, list) else None
		self.index_conflict_resolution = None

		# initialize existing tables to pass (if specified) -- must be first defined in self.all_tables to properly update
		self.dict_iterative_database_tables = {}


		for table_name in enumerate(self.all_tables):
			i, table_name = table_name
			table_idb = None

			# try retrieving parameters for the table
			dict_params = self.dict_all_tables.get(table_name)
			fields_index = self.fields_index_default
			keep_stash = self.keep_stash

			# this setup allows for None to be passed in the dictionary
			if dict_params is not None:

				fields_index = (
					dict_params.get("fields_index") 
					if ("fields_index" in dict_params.keys()) 
					else self.fields_index_default
				)
				keep_stash = dict_params.get("keep_stash")
				keep_stash = keep_stash if isinstance(keep_stash, bool) else False

			try:
				table_idb = IterativeDatabaseTable(
					table_name,
					engine,
					fields_index,
					keep_stash = keep_stash,
					logger = self.logger,
					replace_on_init = self.replace_on_init,
					index_conflict_resolution = index_conflict_resolution
				)

				self.index_conflict_resolution = (
					table_idb.index_conflict_resolution 
					if (self.index_conflict_resolution is None) else 
					self.index_conflict_resolution
				)

				self._log(
					f"Successfully instantiated table {table_name}", 
					type_log = "info",
				)

			except Exception as e:
				self._log(
					f"Error in _initialize_tables() trying to instantiate table {table_name}: {e}", 
					type_log = "error"
				)

			if (table_idb is not None):
				self.dict_iterative_database_tables.update({table_name: table_idb}) 

		
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





	##############################
	#    SOME ALIAS FUNCTIONS    #
	##############################

	def _destroy(self,
		table_name: str
	) -> None:

		"""
		Alias for IterativeDatabaseTable.read_table for table `table_name`.

			* Preserves the the IterativeDatabaseTable object for `table_name`
				within IterativeDatabase. Use `_drop()` to remove the table from
				the database and destroy the stored data.

		Function Arguments
		------------------
		- table_name: table_name to read. Must be defined in
			self.all_tables

		"""
		# check table specification
		table = self.dict_iterative_database_tables.get(table_name)
		if not isinstance(table, IterativeDatabaseTable):
			return None

		table._destroy()

		return None



	def read_table(self,
		table_name: str,
		**kwargs
	) -> pd.DataFrame:
		"""
		Alias for IterativeDatabaseTable.read_table for table `table_name`

		Function Arguments
		------------------
		- table_name: table_name to read. Must be defined in
			self.all_tables

		Keyword Arguments
		-----------------

		Keyword and optional arguments specified in **kwargs are passed to
			IterativeDatabaseTable.read_table. Those arguments are listend
			below.

		- dict_subset: dictionary with keys that are columns in the table and values,
			given as a list, to subset the table. dict_subset is written as:

			dict_subset = {
				field_a = [val_a1, val_a2, ..., val_am],
				field_b = [val_b1, val_b2, ..., val_bn],
				.
				.
				.
			}
		- drop_duplicates: drop duplicates in the CSV when reading?
			* Default is False to improve speeds
			* Set to True to ensure that only unique rows are read in
		- fields_select: fields to read in. Reducing the number of fields to read
			can speed up the ingestion process and reduce the data frame's memory
			footprint.
		- query_logic: default is "and". Subsets table to as

			where field_a in (val_a1, val_a2, ..., val_am) ~ field_b in (val_b1, val_b2, ..., val_bn)...

			where `~ in ["and", "or"]`

		"""
		# check table specification
		table = self.dict_iterative_database_tables.get(table_name)
		if not isinstance(table, IterativeDatabaseTable):
			return None

		return table.read_table(**kwargs)



	def _write_to_table(self,
		table_name: str,
		df_write: pd.DataFrame,
		append_q: bool = True,
		verify: bool = True,
		index_conflict_resolution: Union[str, None] = None,
		reinitialize_on_verification_failure: bool = False,
	) -> None:
		"""
		Write a data frame to the table. Alias for
			IterativeDatabaseTable._write_to_table for table `table_name`

		Function Arguments
		------------------
		- table_name: table_name to read. Must be defined in
			self.all_tables
		- df_write: DataFrame to write to the table

		Keyword Arguments
		-----------------
		- append_q: append to an existing table if found? Default is True.
			* If False, REPLACES existing tables
		- verify: verify table columns before each write
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
		- reinitialize_on_verification_failure: reinitialize table names after 
			writing if there is a verification failure?

		"""

		# check table specification
		table = self.dict_iterative_database_tables.get(table_name)
		if not isinstance(table, IterativeDatabaseTable):
			return None

		index_conflict_resolution = None if (index_conflict_resolution not in table.valid_resolutions) else index_conflict_resolution

		table._write_to_table(
			df_write,
			append_q = append_q,
			verify = verify,
			index_conflict_resolution = index_conflict_resolution,
			reinitialize_on_verification_failure = reinitialize_on_verification_failure,
		)

		return None



	###########################################
	#    ADDITIONAL DATABASE FUNCTIONALITY    #
	###########################################

	def _add_tables(self,
		dict_new_tables: Union[Dict[str, IterativeDatabaseTable], None],
		overwrite_existing_if_in_dict_idt: bool = False,
	) -> None:
		"""
		Initialize the output database based on the engine. Modifies the
			following properties:

			* self.all_tables
			* self.dict_all_tables
			* self.dict_iterative_database_tables

		Function Arguments
		------------------
		- dict_new_tables: dictionary mapping a list of tables to initialize and
			store OR list of table names (converted to dictionary of keys to
			None). Takes same form as `dict_all_tables`, used to initialize
			IterativeDatabase. The dictionary maps keys (table names) to
			dictionaries that define properties of the table to pass to the
			IterativeDatabaseTable. See ?IterativeDatabase for more information
			on the form of `dict_all_tables`.

		- overwrite_existing_if_in_dict_idt: if passing an
			IterativeDatabaseTable in dict_idt with a name that already exists,
			should it overwrite the existing table? USE WITH CAUTION TO AVOID
			ERASING TABLES.
		"""

		# check new table names against exisitng; if overwriting, delete the existing table
		dict_new_tables = (
			dict([(str(x), None) for x in dict_new_tables]) 
			if isinstance(dict_new_tables, list) 
			else dict_new_tables
		)
		dict_new_tables = {} if not isinstance(dict_new_tables, dict) else dict_new_tables
		dict_tables_merge = {}

		# initialize tables to keep/drop; won't update individual tables until IterativeDatabaseTable runs successfully
		tables_new_eval = list(dict_new_tables.keys())
		tables_all_drop = []
		tables_new_drop = []

		for k in tables_new_eval:
			if k not in self.dict_all_tables.keys():
				continue

			(
				tables_all_drop.append(k) 
				if overwrite_existing_if_in_dict_idt 
				else tables_new_drop.append(k)
			)

		tables_new_eval = [x for x in tables_new_eval if (x not in tables_new_drop)]

		# get the engine
		engine = self.get_idt_engine()

		# loops over tables that are to be merged (accounts for overwriting)
		for table_name in tables_new_eval:

			table_name = table_name
			table_idb = None

			# try retrieving parameters for the table
			dict_params = dict_new_tables.get(table_name)
			fields_index = self.fields_index_default
			keep_stash = self.keep_stash

			# this setup allows for None to be passed in the dictionary
			if dict_params is not None:
				fields_index = dict_params.get("fields_index", self.fields_index_default) 
				keep_stash = dict_params.get("keep_stash")
				keep_stash = keep_stash if isinstance(keep_stash, bool) else False

			try:
				table_idb = IterativeDatabaseTable(
					table_name,
					engine,
					fields_index,
					keep_stash = keep_stash,
					logger = self.logger,
					replace_on_init = self.replace_on_init,
					index_conflict_resolution = self.index_conflict_resolution,
				)

				self._log(
					f"Successfully added table {table_name}", 
					type_log = "info",
				)

			except Exception as e:
				self._log(
					f"Error in _add_table() trying to add table {table_name}: {e}", 
					type_log = "error",
				)

			if (table_idb is not None):
				dict_tables_merge.update({table_name: table_idb})
				

		# update properties
		self.all_tables = sorted(list(set(self.all_tables) | set(tables_new_eval)))
		self.dict_all_tables.update(
			dict((k, v) for k, v in dict_new_tables.items() if (k in dict_tables_merge.keys()))
		)
		self.dict_iterative_database_tables.update(dict_tables_merge)

		return None



	def _drop(self,
		table_names: Union[List, str],
		preserve_storage: bool = False
	) -> None:
		"""
		Remove table(s) from the IterativeDatabase. Set `preserve_storage` to
			True to remove from the IterativeDatabase object but leave the table
			in storage. Generally preserve_storage = False.

			Modifies the following properties:

			* self.all_tables
			* self.dict_all_tables
			* self.dict_iterative_database_tables

		Function Arguments
		------------------
		- table_names: name of table, or list of table names, to drop.

		Keyword Arguments
		-----------------
		- preserve_storage: set to True to avoid calling _destroy()
		"""

		table_names = [table_names] if isinstance(table_names, str) else table_names
		if not isinstance(table_names, list):
			return None

		tables_drop = []
		for table_name in table_names:
			del_q = False
			if table_name in self.all_tables:
				try:
					self._destroy(table_name) if not preserve_storage else None
					del_q = True

				except Exception as e:
					self._log(f"Error trying to drop table {table_name}: {e}", type_log = "error")

				if del_q:
					tables_drop.append(table_name)
					del self.dict_all_tables[table_name]
					del self.dict_iterative_database_tables[table_name]

		self.all_tables = [x for x in self.all_tables if x not in tables_drop]

		return None