from attribute_table import *
import logging
import model_variable as mv
import numpy as np
import os, os.path
import pandas as pd
import re
import support_functions as sf
from typing import *
import warnings



class Units:
    """
    Create a class for converting between units of a given dimension

    Initialization Arguments
    ------------------------
    - attributes: file path to attribute table OR attribute table to 
        initialize from

    Optional Arguments
    ------------------
    - key_prependage: optional prepandage that is in the attribute key. If
        specified, then the unit key will drop this prependage
    """
    def __init__(self,
        attributes: Union[AttributeTable, str],
        key_prependage: Union[str, None] = None,
    ) -> None:

        self._initialize_properties()
        self._initialize_attribute_table(
            attributes, 
            key_prependage = key_prependage,
        )

        return None


    

    ############################
    ###                      ###   
    ###    INITIALIZATION    ###
    ###                      ### 
    ############################

    def _initialize_attribute_table(self,
        attributes: Union[AttributeTable, str],
        key_prependage: Union[str, None] = None,
    ) -> None:
        """
        Load all attribute tables and set the following parameters:

            * self.attribute_table
            * self.key

        Function Arguments
        ------------------
        - attributes: file path to attribute table OR attribute table to 
            initialize from

        Keyword Arguments
        -----------------
        - key_prependage: optional prepandage that is in the attribute key. If
            specified, then the unit key will drop this prependage
        """

        # try getting attributes if a string
        if isinstance(attributes, str):
            try:
                obj = self.read_attributes(attributes)
            except Exception as e:
                raise RuntimeError(f"Error reading units attributes: {e}")

        if not isinstance(attributes, AttributeTable):
            tp = str(type(attributes))
            raise RuntimeError(f"Invalid type '{tp}' for attributes found in Units initialization.")
        
        key = attributes.key
        key = (
            key.replace(key_prependage, "")
            if key.startswith(key_prependage)
            else key
        )

        # get ordered search fields
        attributes_search_ordered = self.get_attribute_fields(
            attributes,
            key
        )

        ##  SET PROPERTIES

        self.attributes_search_ordered = attributes_search_ordered
        self.attribute_table = attributes
        self.key = key

        return None



    def _initialize_properties(self,
    ) -> None:
        """
        Set properties required throughout. Sets the following properties:

            * self.

        Function Arguments
        ------------------
        - 
        
        Keyword Arguments

        """
    

        return None
    


    def read_attributes(self,
        dir_attributes: str,
        stop_on_error: bool = True,
    ) -> None:
        """
        Read unit attribute tables from a directory

        Function Arguments
        ------------------
        - dir_attributes: directory containing attribute tables
        
        Keyword Arguments
        -----------------
        - stop_on_error: if False, returns None instad of raising an error
        """ 

        if not isinstance(dir_attributes, str):
            return None

        # check directory if string is passed
        try:
            sf.check_path(dir_attributes, False)
        except Exception as e:
            if stop_on_error:
                raise RuntimeError(e)
            else:
                return None


        # try to read tables
        dict_read = dict(
            (x, self.regex_attribute_match(x))
            for x in os.listdir(dir_attributes)
            if self.regex_attribute_match(x) is not None
        )
        if len(dict_read) == 0:
            return None
        
        # iterate over tables to load
        dict_tables = {}

        for k, v in dict_read.items():
            
            fp = os.path.join(dir_attributes, k)
            key = v.groups()[0]

            try:
                attr = AttributeTable(fp, key, clean_table_fields = True, )

            except Exception as e:
                self._log(
                    f"Error trying to initialize attribute {key}: {e}.\nSkipping...", 
                    type_log = "error"
                )

                continue

            dict_tables.update({key: attr})

        
        return dict_tables



    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def build_conversion_target_field(self,
        unit_target: str,
    ) -> Union[str, None]:
        """
        For a conversion target unit, build the field needed
        """
        out = f"{self.key}_equivalent_{unit_target}"
        return out



    def convert(self,
        units_in: str,
        units_out: str,
        missing_return_val: Union[float, int, None] = 1,
    ) -> Union[float, int, None]:
        """
        Get a conversion factor x to write units_in in terms of units_out; i.e.,

            units_in * x = units_out

        Returns `missing_return_val` by default if no conversion is found
        """
        # verify input units
        units_in = self.get_unit_key(units_in)
        units_out = self.get_unit_key(units_out)
        if (units_in is None) | (units_out is None):
            return missing_return_val

        # get the field and extract from the table
        field_units_out = self.build_conversion_target_field(units_out)
        factor = self.get_attribute(units_in, field_units_out)

        factor = float(factor) if sf.isnumber(factor) else missing_return_val

        return factor



    def get_attribute(self,
        unit_specification: str,
        attribute: str,
        clean: bool = False,
        none_flag: Any = None,
    ) -> Union[Any, None]:
        """
        Retrieve `attribute` associated with unit specification 
            `unit_specification`. 

        Function Arguments
        ------------------
        - attribute_table: attribute table to search over
        - unit_key: unit key value. Used to verify if same as attribute table
            key
        
        Keyword Arguments
        -----------------
        - clean: Set clean to True to apply model_variable.clean_element() to 
            the output
        - none_flag: If not None, return None if this flag is specified.
            NOTE: This is applied *after* cleaning the variable if 
                `clean == True`
        """
        unit = self.get_unit_key(unit_specification)
        if unit is None:
            return None
        
        out = self.attribute_table.get_attribute(unit, attribute)
        out = mv.clean_element(out) if clean else out
        if none_flag is not None:
            out = None if (out == none_flag) else out

        return out
        


    def get_attribute_fields(self,
        attribute_table: AttributeTable,
        unit_key: str,
        field_name: str = "name",
    ) -> Union[List[str], None]:
        """
        Retrieve a list of attribute fields that can be used acceptably 

        Function Arguments
        ------------------
        - attribute_table: attribute table to search over
        - unit_key: unit key value. Used to verify if same as attribute table
            key
        
        Keyword Arguments
        -----------------
        - field_name: optional name field to check for
        """

        fields_avail = [
            x for x in attribute_table.table.columns
            if attribute_table.table[x].dtype not in ["float64", "int64"]
        ]

        # order the output fields
        fields_ord = [attribute_table.key] 
        fields_ord.append(unit_key) if (unit_key != attribute_table.key) else None
        fields_ord.append(field_name) if (field_name in fields_avail) else None
        fields_ord += [x for x in fields_avail if x not in fields_ord]
        

        return fields_ord



    def get_unit_key(self,
        unit_specification: str,
        flags_missing: Union[Any, List[Any]] = "none",
    ) -> Union[str, None]:
        """
        Based on an input unit value, try to get the unit key from the attribute
            table. If not found, returns None

        Function Arguments
        ------------------
        - unit_specification: input unit specification to attempt to retrieve 
            key for
        - flags_missing: optional flag or list of flags to signify as missing
        """
        # initialize and check for missing flags
        attr = self.attribute_table
        flags_missing = (
            [flags_missing] 
            if not sf.islistlike(flags_missing) 
            else list(flags_missing)
        )
        if unit_specification in flags_missing:
            return None

        i = -1
        out = None


        while (out is None) and (i < len(self.attributes_search_ordered) - 1):
            
            i += 1
            prop = self.attributes_search_ordered[i]

            # check if in key values
            if prop == attr.key:
                out = (
                    unit_specification 
                    if unit_specification in attr.key_values
                    else None
                )

            # otherwise, try the field maps
            field_map = f"{prop}_to_{attr.key}"
            dict_map = attr.field_maps.get(field_map)
            if dict_map is None:
                continue
            
            out = dict_map.get(unit_specification)
            out = None if (out in flags_missing) else out

        return out





"""
Using attributes, setup units conversion mechanisms that can be used to ensure
    variables are converted properly. 

NOTE: INCOMPLETE

Initialization Arguments
------------------------
- attributes: either a directory containing attribute tables (CSVs) or a
    list of attribute tables with keys as units

Optional Arguments
------------------
- logger: optional context-dependent logger to pass
"""
class UnitsManager:

    def __init__(self,
        attributes: Union[AttributeTable, str, List[AttributeTable]],
        logger: Union[logging.Logger, None] = None,
    ) -> None:

        self.logger = logger
        self._initialize_logger(logger)
        self._initialize_properties()
        self._initialize_attribute_tables(attributes,)

        return None


    

    ############################
    ###                      ###   
    ###    INITIALIZATION    ###
    ###                      ### 
    ############################

    def _initialize_attribute_tables(self,
        attributes = Union[AttributeTable, str, List[AttributeTable]],
    ) -> None:
        """
        Load all attribute tables and set the following parameters:

            * self.

        Function Arguments
        ------------------
        - dir_att: directory containing attribute tables

        Keyword Arguments
        -----------------
        - table_name_attr_sector: table name used to assign sector table
        - table_name_attr_subsector: table name used to assign subsector table

        """

        # try getting attributes if a string
        if isinstance(attributes, str):

            try:
                obj = self.read_attributes(attributes)
            except Exception as e:
                raise RuntimeError(f"Error reading units attributes: {e}")


        attribute_directory = (
            sf.check_path(attributes, False)
            if isinstance(attributes, str)
            else None
        )



        return None
    


    def _initialize_logger(self,
    logger: Union[logging.Logger, None] = None,
    ) -> None:
        """
        Initialize a logger object?

        Function Arguments
        ------------------
        - logger: optional context-dependent logger to pass

        Keyword Arguments
        -----------------
        """

        logger = None if not isinstance(logger, logging.Logger) else logger

        self.logger = logger

        return None



    def _initialize_properties(self,
        regex_attribute_match: Union[re.Pattern, None] = None,
    ) -> None:
        """
        Set properties required throughout. Sets the following properties:

            * self.regex_attribute_match:
                Regular expression used to parse expressions (e.g., mutable 
                element dictionaries) from initialization strings
        


        Function Arguments
        ------------------
        - 
        
        Keyword Arguments
        -----------------
        - regex_match: optional regular expression to use to match file names to
            identify attribute tables
        """
        
        regex_attribute_match = (
            re.compile("attribute_unit_(.*).csv")
            if not isinstance(regex_attribute_match, re.Pattern)
            else regex_attribute_match
        )


        ##  SET PROPERTIES

        self.regex_attribute_match = regex_attribute_match

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

        return None



    def read_attributes(self,
        dir_attributes: str,
        stop_on_error: bool = True,
    ) -> None:
        """
        Read unit attribute tables from a directory

        Function Arguments
        ------------------
        - dir_attributes: directory containing attribute tables
        
        Keyword Arguments
        -----------------
        - stop_on_error: if False, returns None instad of raising an error
        """ 

        if not isinstance(dir_attributes, str):
            return None

        # check directory if string is passed
        try:
            sf.check_path(dir_attributes, False)
        except Exception as e:
            if stop_on_error:
                raise RuntimeError(e)
            else:
                return None


        # try to read tables
        dict_read = dict(
            (x, self.regex_attribute_match(x))
            for x in os.listdir(dir_attributes)
            if self.regex_attribute_match(x) is not None
        )
        if len(dict_read) == 0:
            return None
        
        # iterate over tables to load
        dict_tables = {}

        for k, v in dict_read.items():
            
            fp = os.path.join(dir_attributes, k)
            key = v.groups()[0]

            try:
                attr = AttributeTable(fp, key, clean_table_fields = True, )

            except Exception as e:
                self._log(
                    f"Error trying to initialize attribute {key}: {e}.\nSkipping...", 
                    type_log = "error"
                )

                continue

            dict_tables.update({key: attr})

        
        return dict_tables

            

        

        