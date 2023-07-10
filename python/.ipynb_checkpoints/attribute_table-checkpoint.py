import numpy as np
import os, os.path
import pandas as pd
import support_functions as sf
from typing import Union


##  the AttributeTable class checks existence, keys, key values, and generates field maps
class AttributeTable:
    """
    AttributeTable class checks existence, keys, key values, and generates field maps

    Function Arguments
    ------------------
    - fp_table:
    - key:
    - fields_to_dict:

    Keyword Arguments
    -----------------
    - clean_table_fields: clean field names from input CSV or DataFrame to ensure lower case/no spaces
    """
    def __init__(self,
        fp_table: Union[str, pd.DataFrame],
        key: str,
        fields_to_dict: list,
        clean_table_fields: bool = True
    ):

        # verify table exists and check keys
        if isinstance(fp_table, str):
            table = pd.read_csv(sf.check_path(fp_table, False), skipinitialspace = True)
        elif isinstance(fp_table, pd.DataFrame):
            table = fp_table.copy()
            fp_table = None

        fields_to_dict = [x for x in fields_to_dict if x != key]

        # clean the fields in the attribute table?
        dict_fields_clean_to_fields_orig = {}
        if clean_table_fields:
            fields_orig = list(table.columns)
            dict_fields_clean_to_fields_orig = dict(zip(sf.clean_field_names(fields_orig), fields_orig))
            table = sf.clean_field_names(table)
            fields_to_dict = sf.clean_field_names(fields_to_dict)
            key = sf.clean_field_names([key])[0]


        # add a key if not specified
        if not key in table.columns:
            print(f"Key {key} not found in table '{fp_table}''. Adding integer key.")
            table[key] = range(len(table))
        # check all fields
        sf.check_fields(table, [key] + fields_to_dict)
        # check key
        if len(set(table[key])) < len(table):
            raise ValueError(f"Invalid key {key} found in '{fp_table}': the key is not unique. Check the table and specify a unique key.")


        # if no fields for the dictionary are specified, default to all
        if len(fields_to_dict) == 0:
            fields_to_dict = [x for x in table.columns if (x != key)]

        # clear RST formatting in the table if applicable
        if table[key].dtype in [object, str]:
            table[key] = np.array([sf.str_replace(str(x), {"`": "", "\$": ""}) for x in list(table[key])]).astype(str)
        # set all keys
        key_values = list(table[key])
        key_values.sort()

        # next, create dict maps
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"

            field_maps.update({field_fwd: sf.build_dict(table[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(table[fld])
            if (len(vals_unique) == len(table)):
                field_maps.update({field_rev: sf.build_dict(table[[fld, key]])})

        self.dict_fields_clean_to_fields_orig = dict_fields_clean_to_fields_orig
        self.field_maps = field_maps
        self.fp_table = fp_table
        self.key = key
        self.key_values = key_values
        self.n_key_values = len(key_values)
        self.table = table

    # function for the getting the index of a key value
    def get_key_value_index(self, key_value):
        if key_value not in self.key_values:
            raise KeyError(f"Error: invalid AttributeTable key value {key_value}.")
        return self.key_values.index(key_value)
