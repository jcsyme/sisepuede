from analysis_id import AnalysisID
import datetime
import itertools
import os, os.path
import numpy as np
import pandas as pd
from typing import *
import support_functions as sf
import sqlalchemy
import sql_utilities as sqlutil
import logging
from model_attributes import ModelAttributes
from attribute_table import *
from iterative_database import *


class SISEPUEDEModelAttributesArchive:
	"""


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
		replace_on_init = False
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
			replace_on_init = self.replace_on_init
		)




	#############################
	#	INITIALIZE DATABASE	#
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
					"strategy": key_strategy,
					"time_series": key_time_series
				}

			where the values are strings giving the key value.

			Sets the following properties:

			* self.dict_dimensional_keys

			If keys are missing, sets to None.
		"""
