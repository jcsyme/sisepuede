import os, os.path
import numpy as np
import pandas as pd


##  build a dictionary from a dataframe
def build_dict(df_in, dims = None):
    
    if len(df_in.columns) == 2:
        dict_out = dict([x for x in zip(df_in.iloc[:, 0], df_in.iloc[:, 1])])
    else:
        if dims == None:
            dims = (len(df_in.columns) - 1, 1)
        n_key = dims[0]
        n_val = dims[1]
        if n_key + n_val != len(df_in.columns):
            raise ValueError(f"Invalid dictionary dimensions {dims}: the sum of dims should be equal to the number of columns in the input dataframe ({len(df_in.columns)}). They sum to {n_key + n_val}.")
        
        # keys to zip
        if n_key == 1:
            keys = df_in.iloc[:, 0]
        else:
            keys = [tuple(x) for x in np.array(df_in[list(df_in.columns)[0:n_key]])]
        # values to zip
        if n_val == 1:
            vals = df_in.iloc[:, len(df_in.columns) - 1]
        else:
            vals = [np.array(x) for x in np.array(df_in[list(df_in.columns)[n_key:(n_key + n_val)]])]

        dict_out = dict([x for x in zip(keys, vals)])

    return dict_out

##  check path 
def check_path(fp, create_q = False):
    if os.path.exists(fp):
        return fp
    elif create_q:
        os.mkdirs(fp, exist_ok = True)
    else:
        raise ValueError(f"Path '{fp}' not found. It will not be created.")
        
# simple but often used function
def format_print_list(list_in, delim = ","):
    return ((f"{delim} ").join(["'%s'" for x in range(len(list_in))]))%tuple(list_in)

# check a data frame
def check_fields(df, fields):
    s_fields_df = set(df.columns)
    s_fields_check = set(fields)
    
    if s_fields_check.issubset(s_fields_df):
        return True
    else:
        fields_missing = format_print_list(s_fields_check - s_fields_df)
        raise ValueError(f"Required fields {fields_missing} not fields_df in the data frame.")

# check a ddictionary
def check_keys(dict_in, keys):
    s_keys_dict = set(dict_in.keys())
    s_keys_check = set(keys)
    
    if s_keys_check.issubset(s_keys_dict):
        return True
    else:
        fields_missing = format_print_list(s_keys_check - s_keys_dict)
        raise ValueError(f"Required keys {fields_missing} not found in the dictionary.")

# 
def clean_field_names(nms, dict_repl: dict = {"  ": " ", " ": "_", "$": "", "\\": "", "\$": "", "`": "", "-": "_", ".": "_", "\ufeff": "", ":math:text": "", "{": "", "}": ""}):
    
    # check return type
    return_df_q =  False
    if type(nms) in [pd.core.frame.DataFrame]:
        df = nms
        nms = list(df.columns)
        return_df_q = True
    
    # get namses to clean, then loop
    nms = [str_replace(nm.lower(), dict_repl) for nm in nms]
    
    for i in range(len(nms)):
        nm = nms[i]
        # drop characters in front 
        while (nm[0] in ["_", "-", "."]) and (len(nm) > 1):
            nm = nm[1:]
        # drop trailing characters 
        while (nm[-1] in ["_", "-", "."]) and (len(nm) > 1):
            nm = nm[0:-1]
        nms[i] = nm
        
    if return_df_q:
        nms = df.rename(columns = dict(zip(list(df.columns), nms)))
        
    return nms
    
    
def df_get_missing_fields_from_source_df(df_target, df_source, side = "right", column_vector = None):
    
    if df_target.shape[0] != df_source.shape[0]:
        raise RuntimeError(f"Incompatible shape found in data frames; the target number of rows ({df_target.shape[0]}) should be the same as the source ({df_source.shape[0]}).")
    # concatenate
    flds_add = [x for x in df_source.columns if x not in df_target]
    
    if side.lower() == "right":
        lcat = [df_target.reset_index(drop = True), df_source[flds_add].reset_index(drop = True)]
    elif side.lower() == "left":
        lcat = [df_source[flds_add].reset_index(drop = True), df_target.reset_index(drop = True)]
    else:
        raise ValueError(f"Invalid side specification {side}. Specify a value of 'right' or 'left'.")
        
    df_out = pd.concat(lcat,  axis = 1)
    
    if type(column_vector) == list:
        flds_1 = [x for x in column_vector if (x in df_out.columns)]
        flds_2 = [x for x in df_out.columns if (x not in flds_1)]
        df_out = df_out[flds_1 + flds_2]
        
    return df_out
    
    
def str_replace(str_in, dict_replace):
    for k in dict_replace.keys():
        str_in = str_in.replace(k, dict_replace[k])
        
    return str_in


def subset_df(df, dict_in):
    for k in dict_in.keys():
        if k in df.columns:
            if type(dict_in[k]) != list:
                val = [dict_in[k]]
            else:
                val = dict_in[k]
            df = df[df[k].isin(val)]
    return df