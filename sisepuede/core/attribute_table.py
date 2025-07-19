import numpy as np
import os, os.path
import pandas as pd
import pathlib
import sisepuede.utilities._toolbox as sf
import warnings
from typing import *


##  GLOBAL VARIABLES

_MODULE_UUID = "9DA52BFC-A67C-4EA7-AFBE-3906DC8F8510"





class AttributeTable:
    """AttributeTable class checks existence, keys, key values, and generates 
        field maps. Useful for quickly accessing table keys and information 
        associated with keys.

    Function Arguments
    ------------------
    fp_table : Union[str, pathlib.Path, pd.DataFrame]
        String or pathlib.Path giving file path to CSV OR DataFrame to use as 
        AttributeTable
    key : str
        Key in fp_table to use

    Keyword Arguments
    -----------------
    fields_to_dict : Union[list, None]
        Optional fields to include in fields maps. If None, will include map of 
        key to all fields + inverse for bijective maps
    clean_table_fields : bool
        Clean field names from input CSV or DataFrame to ensure lower case/no 
        spaces?
    """
    def __init__(self,
        fp_table: Union[str, pd.DataFrame],
        key: str,
        fields_to_dict: Union[list, None] = None,
        clean_table_fields: bool = True,
    ) -> None:

        self._initialize_table(
            fp_table,
            key,
            clean_table_fields = clean_table_fields,
            fields_to_dict = fields_to_dict,
        )

        self._initialize_uuid()

        return None
    


    def __repr__(self,
    ) -> str:
        """
        Call data frame's method
        """
        return self.table.__repr__()


    
    def _repr_html_(self
    ) -> Union[str, None]:
        """
        Return a html representation for a particular DataFrame, mainly for 
            IPython notebook. NOTE: this method is shortcut to same method in 
            pd.DataFrame()
        """

        return self.table._repr_html_()



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def get_field_maps(self,
        fields_to_dict: List[str],
        table: pd.DataFrame,
        key: str,
    ) -> None:
        """
        Get field maps from a list of fields to ditionary and base table table
        """

        if not sf.islistlike(fields_to_dict):
            return None

        # next, create dict maps
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"

            field_maps.update({field_fwd: sf.build_dict(table[[key, fld]])})

            # check for 1:1 correspondence before adding reverse
            vals_unique = set(table[fld])
            (
                field_maps.update({field_rev: sf.build_dict(table[[fld, key]])})
                if (len(vals_unique) == len(table))
                else None
            )

        return field_maps



    def _initialize_table(self,
        fp_table: Union[str, pathlib.Path, pd.DataFrame],
        key: str,
        clean_table_fields: bool = True,
        fields_to_dict: Union[list, None] = None,
    ) -> None:
        """
        Initialize the input table and file path. Sets the following properties:
            
            * self.dict_fields_clean_to_fields_orig
            * self.field_maps
            * self.fp_table
            * self.key
            * self.key_values
            * self.n_key_values
            * self.table
        """

        table = get_attribute_table_df(
            fp_table, 
            allow_attribute_arg = False, 
            stop_on_error = True,
        )


        ##  CHECK FIELDS

        fields_to_dict = (
            [x for x in fields_to_dict if x != key]
            if sf.islistlike(fields_to_dict)
            else [x for x in table.columns if (x != key)]
        )

        # clean the fields in the attribute table?
        dict_fields_clean_to_fields_orig = {}
        if clean_table_fields:
            fields_orig = list(table.columns)
            dict_fields_clean_to_fields_orig = dict(zip(sf.clean_field_names(fields_orig), fields_orig))
            table = sf.clean_field_names(table)
            fields_to_dict = sf.clean_field_names(fields_to_dict)
            key = sf.clean_field_names([key])[0]


        # add a key if not specified and check all fields
        if not key in table.columns:
            warnings.warn(f"Key {key} not found in table '{fp_table}''. Adding integer key.")
            table[key] = range(len(table))
        sf.check_fields(table, [key] + fields_to_dict)

        # check key
        if len(set(table[key])) < len(table):
            msg = f"Invalid key {key} found in '{fp_table}': the key is not unique. Check the table and specify a unique key."
            raise RuntimeError(msg)


        # if no fields for the dictionary are specified, default to all
        if len(fields_to_dict) == 0:
            fields_to_dict = [x for x in table.columns if (x != key)]

        # clear RST formatting in the table if applicable
        if table[key].dtype in [object, str]:
            table[key] = np.array([sf.str_replace(str(x), {"`": "", "\$": ""}) for x in list(table[key])]).astype(str)
        
        # set all keys
        key_values = list(table[key])
        key_values.sort()

        # get field maps
        field_maps = self.get_field_maps(
            fields_to_dict, 
            table,
            key,
        )


        self.dict_fields_clean_to_fields_orig = dict_fields_clean_to_fields_orig
        self.field_maps = field_maps
        self.fp_table = fp_table
        self.key = key
        self.key_values = key_values
        self.n_key_values = len(key_values)
        self.table = table

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """Initialize the UUID
        """

        self.is_attribute_table = True
        self._uuid = _MODULE_UUID

        return None


    
    ###############################
    #    OPERATIONAL FUNCTIONS    #
    ###############################

    def get_attribute(self,
        key_value: Any,
        attribute: str,
    ) -> Union[Any, None]:
        """
        Get value of `attribute` associated with key value `key_value`
        """

        if attribute == self.key:
            return key_value

        key_try = f"{self.key}_to_{attribute}"
        dict_attribute = self.field_maps.get(key_try)
        output = (
            dict_attribute.get(key_value)
            if dict_attribute is not None
            else None
        )

        return output


        
    def get_key_value_index(self, 
        key_value: Any,
        throw_error: bool = True,
    ) -> Union[int, None]:
        """
        Get the ordered index of key value key_value
        """
        if key_value not in self.key_values:
            if throw_error:
                raise KeyError(f"Error: invalid AttributeTable key value {key_value}.")

            return None

        out = self.key_values.index(key_value)

        return out

    

    def to_csv(self,
        *args,
        **kwargs,
    ) -> None:
        """
        Write the attribute table to csv
        """

        self.table.to_csv(*args, **kwargs)

        return None




###################################
#    SOME SUPPORTING FUNCTIONS    #
###################################

def concatenate_attribute_tables(
    key_shared: str,
    *args,
    fields_to_dict: Union[List, None] = None,
    resolve_key_conflicts: Union[str, bool] = False,
    **kwargs
) -> AttributeTable:
    """
    Merge attribute tables to a shared key.

    Function Arguments
    ------------------
    - key_shared: new key to use across attribute tables
    * args: AttributeTables to concatenate

    Keyword Arguments
    -----------------
    - fields_to_dict: fields to include in field maps.
        * If None, attempts to create field maps for all fields
    - resolve_key_conflicts: passed to pd.DataFrae.drop_duplicates()
        to reconcile duplicate key entries. Options are detailed
        below (from ?pd.DataFrame.drop_duplicates):

        "
        Determines which duplicates (if any) to keep.
        - ``first``: Drop duplicates except for the first occurrence.
        - ``last``: Drop duplicates except for the last occurrence.
        - False: Drop all duplicates.
        "
    - **kwargs: passed to AttributeTable to initialize output table
    """
    att_out = []
    header = None

    for att in args:
        if not is_attribute_table(att, ):
            continue 
        
        tab_cur = att.table.copy().rename(columns = {att.key: key_shared})
        header = list(tab_cur.columns) if (header is None) else header

        if set(header).issubset(set(tab_cur.columns)):
            att_out.append(tab_cur)

    if len(att_out) == 0:
        return None


    # concatenate the table and drop any duplicate rows
    att_out = (
        pd.concat(
            att_out, 
            axis = 0,
        )
        .drop_duplicates()
        .reset_index(
            drop = True
        )
    )

    # check key and drop
    att_out.drop_duplicates(
        subset = [key_shared],
        keep = resolve_key_conflicts,
        inplace = True
    )

    # create attribute table
    att_out = AttributeTable(
        att_out,
        key_shared,
        att_out,
        **kwargs
    )

    return att_out





###################################
#    SIMPLE CHECKING FUNCTIONS    #
###################################

def get_attribute_table_df(
    fp_table: Union[str, pathlib.Path, pd.DataFrame],
    allow_attribute_arg: bool = True,
    stop_on_error: bool = True,
) -> Union[pd.DataFrame, AttributeTable, None]:
    """Retrieve an AttributeTable from a path (str or pathlib.Path), DataFrame.
    
    Function Arguments
    ------------------
    fp_table : Union[str, pathlib.Path, pd.DataFrame]
        String or pathlib.Path giving file path to CSV OR DataFrame to use as 
        AttributeTable

    Keyword Arguments
    -----------------
    allow_attribute_arg : bool
        Set to True to allow fp_table to be an AttributeTable. If False, will
        throw an error if stop_on_error, otherwise return None
    stop_on_error : bool
        Stop on errors (True), or return None (False)
    """
    
    # check if the object is an AttributeTable (doesn't use type checking)
    if is_attribute_table(fp_table):
        if allow_attribute_arg:
           return fp_table

        if stop_on_error:
            raise RuntimeError(f"Cannot pass an AttributeTable as argument.")
        
        return None

    # verify table exists and check keys
    if isinstance(fp_table, (str, pathlib.Path, )):
        table = pd.read_csv(
            sf.check_path(fp_table, False), 
            skipinitialspace = True,
        )
    
    elif isinstance(fp_table, pd.DataFrame):
        table = fp_table.copy()
        fp_table = None
    
    else:
        if stop_on_error:
            tp = str(type(fp_table))
            msg = f"""Error initializing AttributeTable: invalid type '{tp}' of 
            fp_table specified. Must be str, pathlib.Path, or DataFrame."""
            raise RuntimeError(msg)

        return None
    
    return table



def is_attribute_table(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDE object
    """

    out = hasattr(obj, "is_attribute_table")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out