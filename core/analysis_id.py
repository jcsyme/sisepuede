import datetime
import logging
import os, os.path
import re
import sisepuede.utilities._toolbox as sf
import time
from typing import *



class AnalysisID:
	"""
	Create a unique ID for each session/set of runs. Can be instantiated using a
		string (from a previous run) or empty, which creates a new ID.

	Initialization Arguments
	------------------------
	- id_str: optional entry of a previous string containing an ID.
		* If None, creates a new ID based on time in isoformat
	- logger: optional log object to pass
	- regex_template: optional regular expression used to parse id
		* Should take form
			re.compile("TEMPLATE_STRING_HERE_(.+$)")
		where whatever is contained in (.+$) is assumed to be an isoformat time.
		* If None is entered, defaults to
			re.compile("analysis_run_(.+$)")

	"""
	def __init__(self,
		id_str: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
		regex_template: Union[str, None] = None
	):
		self.logger = logger
		self._check_id(
			id_str = id_str,
			regex_template = regex_template
		)
		self._set_file_string()



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



	def _check_id(self,
		id_str: Union[str, None] = None,
		regex_template: Union[re.Pattern, None] = None
	) -> None:
		"""
		Set the runtime ID to distinguish between different analytical
		 	runs. Sets the following properties:

			* self.day
			* self.default_regex_template
			* self.hour
			* self.id
			* self.isoformat
			* self.microsecond
			* self.minute
			* self.month
			* self.new_init
			* self.regex_template
			* self.second
			* self.year
		"""

		self.default_regex_template = re.compile("analysis_run_(.+$)")
		self.isoformat = None
		self.new_init = True
		self.regex_template = self.default_regex_template if not isinstance(regex_template, re.Pattern) else regex_template
		# get regex substitution
		date_info = None
		str_regex_sub = [x for x in self.regex_template.split(self.regex_template.pattern) if (x != "")]
		str_regex_sub = str_regex_sub[0] if (len(str_regex_sub) > 0) else None


		# try to initialize from string if specified
		if isinstance(id_str, str):
			match = self.regex_template.match(id_str)
			if match is not None:
				try:
					date_info = datetime.datetime.fromisoformat(match.groups()[0])
					self.id = id_str
					self.isoformat = match.groups()[0]
					self.new_init = False

				except Exception as e:
					self._log(f"Error in AnalysisID trying to initialize ID '{id_str}': {e}.\n\tDefaulting new ID.", type_log = "warning")
					id_str = None
			else:
				id_str = None

		# otherwise, create a new one
		if id_str is None:
			date_info = datetime.datetime.now()
			self.isoformat = date_info.isoformat()
			self.id = self.regex_template.pattern.replace(str_regex_sub, self.isoformat) if (str_regex_sub is not None) else f"{self.regex_template.pattern}_{self.isoformat}"

		# set properties
		(
			self.year,
			self.month,
			self.day,
			self.hour,
			self.minute,
			self.second,
			self.microsecond
		) = (
			date_info.year,
			date_info.month,
			date_info.day,
			date_info.hour,
			date_info.minute,
			date_info.second,
			date_info.microsecond
		)

		# note the success, but only if logging
		self._log(
			f"Successfully initialized Analysis ID '{self.id}'", 
			type_log = "info",
			warn_if_none = False,
		)



	def _set_file_string(self,
	) -> None:
		"""
		Set the file-system safe string. Sets the following properties:

		* self.id_fs_safe
		* self.dict_id_from_fs_safe_replacements
		* self.dict_id_to_fs_safe_replacements

		"""

		self.dict_id_to_fs_safe_replacements = {":": ";"}
		self.dict_id_from_fs_safe_replacements = sf.reverse_dict(self.dict_id_to_fs_safe_replacements)
		self.id_fs_safe = self.id_to_file_safe_id()


	########################################################################
	#    SOME FUNCTIONS FOR CONVERTING TO/FROM FILE SYSTEM-SAFE STRINGS    #
	########################################################################

	def id_from_file_safe_id(self,
		id: str,
		dict_replacements: Union[Dict, None] = None
	) -> str:
		"""
		Convert a file-system safe string to an ID string (invert invalid characters
			to support POSIX strings).

		Function Arguments
		------------------
		- id: file-system safe string to initialize as id

		Keyword Arguments
		-----------------
		- dict_replacements: dictionary to use to replace file-system safe substrings
			with ID-valid strings
		"""

		dict_replacements = self.dict_id_from_fs_safe_replacements if (dict_replacements is None) else dict_replacements

		return sf.str_replace(id, dict_replacements)



	def id_to_file_safe_id(self,
		id: Union[str, None] = None,
		dict_replacements: Union[Dict, None] = None
	) -> str:
		"""
		Convert an id to a file-system safe string (replace invalid characters).

		Keyword Arguments
		-----------------
		- id: POSIX-time based AnalysisID.id string to replace
		- dict_replacements: dictionary to use to replace substrings
		"""
		id = self.id if (id is None) else id
		dict_replacements = self.dict_id_to_fs_safe_replacements if (dict_replacements is None) else dict_replacements

		return sf.str_replace(id, dict_replacements)
