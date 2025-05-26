import numpy as np
import os, os.path
import pandas as pd
import sqlalchemy
from typing import *



def dict_subset_to_query_append(
    dict_subset: Union[Dict[str, List], None],
    query_logic: str = "and"
) -> str:
    """
    Convert a subsetting dictionary to a "where" clause in an SQL query.

    Function Arguments
    ------------------
    - dict_subset: dictionary with keys that are columns in the table and 
        values, given as a list, to subset the table. dict_subset is written as:

        dict_subset = {
            field_a = [val_a1, val_a2, ..., val_am],
            field_b = [val_b1, val_b2, ..., val_bn],
            .
            .
            .
        }

    Keyword Arguments
    -----------------
    - query_logic: default is "and". Subsets table to as

        where field_a in (val_a1, val_a2, ..., val_am) ~ field_b in 
        (val_b1, val_b2, ..., val_bn)...

        where `~ in ["and", "or"]`

    """

    if dict_subset is None:
        return None

    # initialize query string and list used to build query string
    val_list = []

    for k in dict_subset.keys():
        vals = dict_subset.get(k)
        if vals is not None:
            val_str = join_list_for_query(vals) if isinstance(vals, list) else join_list_for_query([vals])
            val_str = f"{k} in ({val_str})" if isinstance(vals, list) else f"{k} = {val_str}"
            val_list.append(val_str)

    # only return a string if there are values to filter on
    query_str = ""
    if (len(val_list) > 0):
        query_str = f" {query_logic} ".join(val_list)
        query_str = f" where {query_str}"

    return query_str



def fetch_query_as_df(
    query: str,
    engine: sqlalchemy.engine.Engine,
) -> Union[pd.DataFrame, None]:
    """
    Execute a query and return as a DataFrame using Pandas.

    Function Arguments
    ------------------
    - query: string query to pass
    - engine: SQLAlchemy engine to use for query execution
    """
    df_out = None
    if not (isinstance(engine, sqlalchemy.engine.Engine) & isinstance(query, str)):
        return df_out

    # try the connection
    with engine.connect() as con:
        try:
            df_out = pd.read_sql_query(query, con)
        except Exception as e:
            # LOGHERE
            raise RuntimeError(f"Error in get_query: the service returned error '{e}'.\n\nQuery:\n\t'{query}'.")

    return df_out



def fetch_query_result(
    engine: sqlalchemy.engine.Engine,
    query: str,
) -> Any:
    """Execute a query and retieve the output. Returns None if the
        engine is invalid.
    """
    return_none = not isinstance(engine, sqlalchemy.engine.Engine)
    return_none |= not isinstance(query, str, )
    if return_none:
        return None

    # initialize result and send query
    result = None
    if query is not None:
        with engine.connect() as con:
            result = con.execute(sqlalchemy.text(query))
            con.commit() # what a fuckin pita to figure this out

    return result



def format_listlike_elements_for_filter_query(
    elements: Union[List, Tuple, np.ndarray, None],
    fields: Union[List, Tuple, np.ndarray, None],
    query_logic: str = "and"
) -> str:
    """
    Format a list-like set of elements for a filtering query
        using filtering fields and associated elements. Creates
        a query of the form

        (fields_0 = elements_0 ~ fields_1 = elements_1 ~ ...),

        where ~ is "and" or "or"

        * NOTE: if len(fields) != len(elements), reduces both
            to the minimum length available between the two.

    Function Arguments
    ------------------
    - elements: ordered elements to filter on, i.e.,

        [elements_0, elements_1, ..., elements_n]

    - fields: fields to use for filtering, i.e.,
        [fields_0, fields_1, ..., fields_n]

    Optional Arguments
    ------------------
    - query_logic: "and" or "or", used to define the query
    """
    # some checks
    if (elements is None) or (fields is None):
        return ""

    query_logic = "and" if not (query_logic in ["and", "or"]) else query_logic
    query = []

    for elem in elements:
        n = min(len(elem), len(fields))

        elem_cur = elem[0:n]
        fields_cur = fields[0:n]

        query_component = f" {query_logic} ".join([f"{x} = {format_type_for_sql_query(y)}" for x, y in zip(fields_cur, elem_cur)])
        query_component = f"({query_component})"
        query.append(query_component)

    query = " or ".join(query) if (len(query) > 0) else ""

    return query



def format_tuple_as_query_tuple(
    index_tuple: Tuple[Any],
    include_single_quotes: bool = True,
) -> Union[str, None]:
    """
    Map an index tuple to a list of strings for SQL query. Set 
        include_single_quotes = True to wrap elements in a single quote
    """
    query_tuple = ", ".join([(f"'{x}'" if include_single_quotes else f"{x}") for x in index_tuple])
    query_tuple = f"({query_tuple})"
    
    return query_tuple



def format_tuples_to_query_filter_string(
    fields_index: List[str],
    indices: List[Tuple[Any]],
) -> Union[str, None]:
    """
    Convert a list of tuples to a filter query string (e.g., 
    
        "(primary_id = 0 and region = 'mexico') or (...)..."
        
    NOTE: if both logic_inner and logic_outer are "and", returns None 
        (impossible)
        
    Function Arguments
    ------------------
    - indices: list of index tuples to filter on

    """
    
    query_fields = format_tuple_as_query_tuple(
        fields_index, 
        include_single_quotes = False
    )
    
    query_vals = [format_tuple_as_query_tuple(x) for x in indices]
    query_vals = format_tuple_as_query_tuple(
        query_vals, 
        include_single_quotes = False
    )
    
    query_component = f"{query_fields} in {query_vals}"
    
    return query_component



def format_type_for_sql_query(
    val: Union[float, int, str]
) -> str:
    """
    Format values based on input type. If val is a string, adds quotes
    """

    val = f"'{val}'" if isinstance(val, str) else str(val)

    return val



def generate_schema_from_df(
    df_in: pd.DataFrame,
    dict_fields_to_dtype_sql: Union[Dict[str, Union[str, type]], None] = None,
    field_type_object_default: str = "string",
    float_as_double: bool = False,
    sep: str = ",\n",
    type_str_float: str = "FLOAT",
    type_str_integer: str = "INTEGER",
    type_str_string: str = "STRING",
) -> str:
    """
    Generate an SQL schema from a data frame. For use in automating table
        setup in remote SQL or SQL-like databases.
    
    Returns a two-ple of the following form:
    
        (schema_out, fields_out)
 
 
    Function Arguments
    ------------------
    - df_in: input data frame
    
    Keyword Arguments
    -----------------
    - dict_fields_to_dtype_sql: dictionary mapping fields to specific SQL data 
        types; keys are fields and dtypes are SQL data types.
        * If None, infers from DataFrame dtype
    - field_type_object_default: default data type for fields specified as 
        object (any valid SQL data type)
    - sep: string separator to use in schema (", " or "\n, ", should always have
        a comma)
    - type_str_float: string used to denote FLOAT type (can vary based on 
        platform)
    - type_str_integer: string used to denote INTEGER type (can vary based on 
        platform)
    - type_str_strings: tring used to denote STRING type (can vary based on 
        platform)
    """
    # initialize fields, schema out, and fields that were successfully pushed to the schema
    fields = list(df_in.columns)
    schema_out = []
    fields_out = []
    
    # some data type dictionaries that are used
    dict_dtypes = df_in.dtypes.to_dict()
    dict_dtypes_to_sql_types = {
        "string": type_str_string,
        "o": type_str_string,
        "float64": type_str_float,
        "int64": type_str_integer,
    }
    dict_fields_to_dtype_sql = (
        dict_fields_to_dtype_sql
        if isinstance(dict_fields_to_dtype_sql, dict)
        else {}
    )
    
    for i, field in enumerate(fields):
        
        # try getting sql data type from external dictionary; otherwise, pull from pandas datatype
        dtype_sql = dict_fields_to_dtype_sql.get(field)
        if dtype_sql is None:
            dtype_sql = dict_dtypes.get(field)
            dtype_sql = (
                dict_dtypes_to_sql_types.get(dtype_sql.name, type_str_string)
                if dtype_sql is not None
                else None
            )
        
        # skip if failed
        if dtype_sql is None:
            continue
            
        # otherwise, build schema
        schema_out.append(f"{field} {dtype_sql}")
        fields_out.append(field)
    
    schema_out = str(sep).join(schema_out)
    
    return schema_out, fields_out



def get_table_names(
    engine:sqlalchemy.engine,
    error_return: Union[Any, None] = None,
) -> Union[List[str], None]:
    """
    Return a list of table name contained in the SQL Alchemy engine `engine`. 
        On an error, returns `error_return`
    """

    try:
        out = sqlalchemy.inspect(engine).get_table_names()
    except:
        out = error_return
    
    return out



def join_list_for_query(
    list_in: list,
    delim: str = ", ",
    quote: str = "'",
) -> str:
    """
    Join the elements of a list to format for a query.

    Function Arguments
    ------------------
    - list_in: list of elements to format for query
        * If elements are strings, then adds ''
        * Otherwise, enter elements

    Keyword Arguments
    -----------------
    - delim: delimiter
    - quote: quote character
    """

    list_join = [
        f"{quote}{x}{quote}" if isinstance(x, str) else str(x)
        for x in list_in
    ]
    list_join = delim.join(list_join)

    return list_join



def order_df_as_table_schema(
    df_in: pd.DataFrame,
    schema_in: str
) -> pd.DataFrame:
    """
    Order a dataframe to match sql schema and check datatypes in output

    Function Arguments
    ------------------
    - df_in: input dataframe
    - schema_in: comma-separated string of fields and types
    """

    # set types
    dict_type_map = dict(
        (x, float) for x in ["FLOAT", "DOUBLE"]
    )
    dict_type_map.update(
        dict(
            (x, int) for x in ["INT", "INTEGER", "BIGINT", "MEDIUMINT", "SMALLINT"]
        )
    )
    dict_type_map.update(
        dict(
            (x, str) for x in ["TEXT", "CHAR", "VARCHAR"]
        )
    )

    schema_in = schema_in.replace("\n", "")
    schema_list = schema_in.split(",")
    schema_list = [x.strip().split(" ") for x in schema_list]

    fields, types = tuple(zip(*schema_list))
    fields_ext = list(fields)
    types_ext = list(types)
    dict_dtypes = {}

    # check fields and ordering
    for i, field in enumerate(fields):

        field_ext = [x for x in df_in.columns if (x.upper().replace("-", "_") == field.upper())]
        if len(field_ext) == 0:
            raise KeyError(f"Error in order_df_as_table_schema: no field matching {field} found in input data frame.")

        fields_ext[i] = field_ext[0]
        type_clean = types[i].split("(")[0]
        dtype = dict_type_map.get(type_clean)
        dtype = str if (dtype is None) else dtype

        df_in[field_ext[0]] = df_in[field_ext[0]].astype(dtype)

    out = df_in[fields_ext]

    return out



def sql_table_to_df(
    engine: sqlalchemy.engine.Engine,
    table_name: str,
    fields_select: Union[list, str] = None,
    query_append: str = None,
) -> pd.DataFrame:
    """
    Query a database, retrieve a table, and convert it to a dataframe.

    Function Arguments
    ------------------
    - engine: SQLalchemy Engine used to create a connection and query the 
        database
    - table_name: the table in the database to query

    Keyword Arguments
    -----------------
    - fields_select: a list of fields to select or a string of fields to 
        select (comma-delimited). If None, return all fields.
    - query_append: any restrictions to place on the query (e.g., where). If 
        None, return all records.
    """

    # check table names
    table_names = get_table_names(engine, error_return = [])
    if table_name not in table_names:
        # LOGHERE
        return None

    # build the query
    if fields_select is not None:
        fields_select_str = (
            ", ".join(fields_select) 
            if isinstance(fields_select, list) 
            else fields_select
        )

    else:
        fields_select_str = "*"

    query_append = "" if (query_append is None) else f" {query_append}"
    query = f"select {fields_select_str} from {table_name}{query_append};"


    # try the connection and return output
    with engine.connect() as con:
        try:
            df_out = pd.read_sql_query(query, con)
        except Exception as e:
            # LOGHERE
            raise RuntimeError(f"Error in sql_table_to_df: the service returned error '{e}'.\n\nQuery:\n\t'{query}'.")


    return df_out



def _write_dataframes_to_db(
    dict_tables: dict,
    db_engine: Union[sqlalchemy.engine.Engine, str],
    preserve_table_schema: bool = True,
    append_q: bool = False,
) -> None:
    """
    Write a dictionary of tables to an SQL database.

    Function Arguments
    ------------------
    - dict_tables: dictionary of form {TABLENAME: pd.DataFrame, ...} used to 
        write the table to the database
    - db_engine: an existing SQLAlchemy database engine or a file path to an 
        SQLite database used to establish a connection
        * If a file path is specified, the connection will be opened and closed 
            within the function

    Keyword Arguments
    -----------------
    - preserve_table_schema: preserve existing schema? If so, before writing new 
        tables, rows in existing tables will be deleted and the table will be 
        appended.
    - append_q: set to True top append tables to existing tables if they exist 
        in the database
    """

    # check input specification
    if isinstance(db_engine, str):
        if os.path.exists(db_engine) and db_engine.endswith(".sqlite"):
            try:
                db_engine = sqlalchemy.create_engine(f"sqlite:///{db_engine}")

            except Exception as e:
                raise ValueError(f"Error establishing a connection to sqlite database at {db_engine}: {e} ")
    
    elif not isinstance(db_engine, sqlalchemy.engine.Engine):
        t = type(db_engine)
        raise ValueError(f"Invalid db_con type {t}: only types str, sqlalchemy.engine.Engine are valid.")


    # get available tables
    tables_avail = get_table_names(db_engine, error_return = [])

    #with db_engine.connect() as con:

    for table in dict_tables.keys():
        
        df_write = dict_tables.get(table)

        # simply write the table if it's not present
        if table not in tables_avail:
            df_write.to_sql(
                table, 
                db_engine, 
                if_exists = "replace", 
                index = None,
            )

            continue
    

        # try retrieving columns
        df_columns = pd.read_sql_query(
            f"select * from {table} limit 0;", 
            db_engine
        )

        # initialize writing based on appendage/preserving schema
        cols_write = list(df_columns.columns)
        on_exists = "append"
        query = None
        write_q = set(df_columns.columns).issubset(set(df_write.columns))

        if not append_q:

            cols_write = cols_write if preserve_table_schema else list(df_write.columns)
            on_exists = on_exists if preserve_table_schema else "replace"

            query = (
                f"delete from {table};" 
                if preserve_table_schema 
                else f"drop table {table};"
            )

            write_q = write_q if preserve_table_schema else True

        if query is not None:
            with db_engine.connect() as con:
                con.execute(sqlalchemy.text(query))
                con.commit() # what a fucking pain in the ass it was to find THIS was necessary 

        if write_q:
            df_write[cols_write].to_sql(
                table, 
                db_engine, 
                if_exists = on_exists, 
                index = None,
            ) 


    return None
