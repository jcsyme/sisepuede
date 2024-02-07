from attribute_table import *
import model_variable as mv
import numpy as np
import os, os.path
import pandas as pd
import support_functions as sf
from typing import *
import warnings


"""
Using attributes, setup units conversion mechanisms that can be used to ensure
    variables are converted properly. 

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

            self.all_attributes
            self.all_dims
            self.all_pycategories
            self.attribute_analytical_parameters
            self.attribute_directory
            self.attribute_experimental_parameters
            self.dict_attributes
            self.dict_varreqs
            self.table_name_attr_sector
            self.table_name_attr_subsector

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


        ##  batch load attributes/variable requirements and turn them into AttributeTable objects
        dict_attributes = {}
        dict_varreqs = {}
        attribute_analytical_parameters = None
        attribute_experimental_parameters = None

        for att in all_types:
            fp = os.path.join(dir_att, att)
            if self.substr_dimensions in att:
                nm = att.replace(self.substr_dimensions, "").replace(self.attribute_file_extension, "")
                k = f"dim_{nm}"
                att_table = AttributeTable(fp, nm)
                dict_attributes.update({k: att_table})
                all_dims.append(nm)

            elif self.substr_categories in att:
                df_cols = pd.read_csv(fp, nrows = 0).columns 
                
                # try to set key
                nm = sf.clean_field_names([x for x in df_cols if ("$" in x) and (" " not in x.strip())])
                nm = nm[0] if (len(nm) > 0) else None
                if nm is None:
                    nm = sf.clean_field_names([att.replace("attribute_", "").replace(".csv", "")])[0]
                    nm = nm if (nm in df_cols) else None

                # skip if it is impossible
                if nm is None:
                    continue

                att_table = AttributeTable(fp, nm)
                dict_attributes.update({nm: att_table})
                all_pycategories.append(nm)

            elif (self.substr_varreqs_allcats in att) or (self.substr_varreqs_partialcats in att):
                nm = att.replace(self.substr_varreqs, "").replace(self.attribute_file_extension, "")
                att_table = AttributeTable(fp, "variable")
                dict_varreqs.update({nm: att_table})

            elif (att == f"{self.substr_analytical_parameters}{self.attribute_file_extension}"):
                attribute_analytical_parameters = AttributeTable(fp, "analytical_parameter")

            elif (att == f"{self.substr_experimental_parameters}{self.attribute_file_extension}"):
                attribute_experimental_parameters = AttributeTable(fp, "experimental_parameter")

            else:
                raise ValueError(f"Invalid attribute '{att}': ensure '{self.substr_categories}', '{self.substr_varreqs_allcats}', or '{self.substr_varreqs_partialcats}' is contained in the attribute file.")

        # add some subsector/python specific information into the subsector table
        field_category = "primary_category"
        field_category_py = field_category + "_py"

        # check sector and subsector specifications
        if not set({table_name_attr_sector, table_name_attr_subsector}).issubset(set(dict_attributes.keys())):
            missing_vals = sf.print_setdiff(
                set({table_name_attr_sector, table_name_attr_subsector}),
                set(dict_attributes.keys())
            )
            raise RuntimeError(f"Error initializing attribute tables: table names {missing_vals} not found.")


        ##  UPDATE THE SUBSECTOR ATTRIBUTE TABLE

        # add a new field
        df_tmp = dict_attributes[table_name_attr_subsector].table
        df_tmp[field_category_py] = sf.clean_field_names(df_tmp[field_category])
        df_tmp = df_tmp[df_tmp[field_category_py] != "none"].reset_index(drop = True)

        # set a key and prepare new fields
        key = field_category_py
        fields_to_dict = [x for x in df_tmp.columns if x != key]

        # next, create dict maps to add to the table
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"
            field_maps.update({field_fwd: sf.build_dict(df_tmp[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(df_tmp[fld])
            if (len(vals_unique) == len(df_tmp)):
                field_maps.update({field_rev: sf.build_dict(df_tmp[[fld, key]])})

        dict_attributes[table_name_attr_subsector].field_maps.update(field_maps)


        ##  SET PROPERTIES

        self.attribute_directory = attribute_directory
        self.all_pycategories = all_pycategories
        self.all_dims = all_dims
        self.all_attributes = all_types
        self.attribute_analytical_parameters = attribute_analytical_parameters
        self.attribute_experimental_parameters = attribute_experimental_parameters
        self.dict_attributes = dict_attributes
        self.dict_varreqs = dict_varreqs
        self.table_name_attr_sector = table_name_attr_sector
        self.table_name_attr_subsector = table_name_attr_subsector

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

            

        

        