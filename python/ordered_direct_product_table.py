import numpy as np
import pandas as pd
import support_functions as sf
from typing import *


class OrderedDirectProductTable:
    """
    Define an index table to map the direct product of multiple dimensions to a
        key. There are four key methods (among others) used to identify scenario
        dimension information:

        * OrderedDirectProductTable.get_dims_from_key()
            Get dimensional values associated with a key (inverse of
                get_key_value)
        * OrderedDirectProductTable.get_key_value()
            Get a key value associated with dimensional values (inverse of
                get_dims_from_key)
        * OrderedDirectProductTable.get_indexing_dataframe()
            Get a data frame associated with select dimensional values or with
                key values 
        * OrderedDirectProductTable.get_indexing_dataframe_from_primary_key()
            Get a data frame associated with the primary key only. 

            NOTE: get_indexing_dataframe_from_primary_key() is a separate method
            from get_indexing_dataframe() to avoid potential discrepancies in 
            input dictionaries and reduce ambiguity.
            

    Function Arguments
    ------------------
    - dict_dims: dictionary mapping dimensions to all available values
    - list_dims_ordered: list of available

    Keyword Arguments
    -----------------
    - key_primary: key field to use for product of dims
    """

    def __init__(self,
        dict_dims: Dict[str, List[Any]],
        list_dims_ordered: List[str],
        key_primary: str = "primary_id"
    ):

        self._initialize_dims(
            dict_dims,
            list_dims_ordered,
            key_primary
        )
        self._initialize_cumulative_dim_products()
        self._initialize_moving_cardinality()



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_cumulative_dim_products(self,
    ) -> None:
        """
        Initialize the dimensional products to use for indexing. Sets the
            following products:

            * self.cumulative_dimensional_products
            * self.cumulative_dimensional_products_reversed
        """
        prods = [1 for x in self.cardinality_ordered_reversed]
        n = len(prods) # 3
        card_total = 1

        for card in enumerate(self.cardinality_ordered_reversed[0:-1]):
            i, card = card
            card_total *= card
            prods[n - i - 2] = card_total

        self.cumulative_dimensional_products = prods
        self.cumulative_dimensional_products_reversed = list(reversed(prods))

        return None



    def _initialize_dims(self,
        dict_dims: Dict[str, List[Any]],
        list_dims_ordered: List[str],
        key_primary: str
    ) -> None:
        """
        Set the following properties:

            * self.cardinality_ordered
            * self.cardinality_ordered_reversed
            * self.dim_cardinality
            * self.dims_ordered
            * self.dims_ordered_reversed
            * self.indices_to_values_by_dim
            * self.key_primary
            * self.values_by_dim
            * self.values_to_indices_by_dim

        Function Arguments
        ------------------
        - dict_dims: dictionary mapping dimensions to all available values
        - list_dims_ordered: list of available
        - key_primary: primary key
        """

        # initialize and run checks
        self.cardinality_ordered = None
        self.cardinality_ordered_reversed = None
        self.dim_cardinality = None
        self.dims_ordered = None
        self.dims_ordered_reversed = None
        self.indices_to_values_by_dim = None
        self.key_primary = key_primary if isinstance(key_primary, str) else "primary_id"
        self.range_key_primary = None
        self.values_by_dim = None
        self.values_to_indices_by_dim = None


        if not isinstance(dict_dims, dict):
            tp = str(type(dict_dims))
            raise RuntimeError(f"Invalid type '{tp}' for dict_dims: dict_dims must be a dict.")

        if not isinstance(list_dims_ordered, list):
            tp = str(type(list_dims_ordered))
            raise RuntimeError(f"Invalid type '{tp}' for list_dims_ordered: dict_dims must be a list.")

        # check ordered dims
        dims_ordered = [x for x in list_dims_ordered if x in dict_dims.keys()]
        if len(dims_ordered) == 0:
            raise RuntimeError(f"Invalid specification of dimensions: no dimensions were found in list_dims_ordered.")


        ##  SET OUTPUT PROPERTIES

        dict_values_by_dim = dict((k, sorted(v)) for k, v in dict_dims.items())
        dict_dim_cardinalities = dict((k, len(v)) for k, v in dict_values_by_dim.items())
        dims_ordered_reversed = list(reversed(dims_ordered))
        cardinality_ordered = [dict_dim_cardinalities.get(x) for x in dims_ordered]
        cardinality_ordered_reversed = [dict_dim_cardinalities.get(x) for x in dims_ordered_reversed]

        dict_values_to_index_by_dim = dict((k, dict(zip(v, range(len(v))))) for k, v in dict_dims.items())
        dict_index_to_values_by_dim = dict((k, dict(zip(range(len(v)), v))) for k, v in dict_dims.items())

        # set output properties
        self.cardinality_ordered = cardinality_ordered
        self.cardinality_ordered_reversed = cardinality_ordered_reversed
        self.dim_cardinality = dict_dim_cardinalities
        self.dims_ordered = dims_ordered
        self.dims_ordered_reversed = dims_ordered_reversed
        self.indices_to_values_by_dim = dict_index_to_values_by_dim
        self.range_key_primary = range(int(np.prod(cardinality_ordered))) # store as range
        self.values_by_dim = dict_values_by_dim
        self.values_to_indices_by_dim = dict_values_to_index_by_dim

        return None



    def _initialize_moving_cardinality(self,
        cardinality_ordered: Union[List[int], None] = None
    ) -> None:
        """
        Initialize the "moving cardinality", or windows of repeat lengths
            (outer/inner) for each dimension. Sets the following propertes:

            * self.moving_cardinality_ordered
                List of tuples [(outer_i, inner_i)...] giving the outer
                    repetition (outer_i) and inner reptition (inner_i) for each
                    dimension used in the indexing table.
            * self.moving_cardinality_ordered_reversed
                Reversed list of self.moving_cardinality_ordered

        Keyword Arguments
        -----------------
        - cardinality_ordered: list gving ordered cardinality of dimensions. If
            None, uses self.cardinality_ordered
        """

        cardinality_ordered = (
            self.cardinality_ordered 
            if (cardinality_ordered is None) 
            else cardinality_ordered
        )
        n = len(cardinality_ordered)

        moving_cardinality = [(0, 0) for x in range(n)]

        for i in range(n):

            outer = int(np.prod(cardinality_ordered[0:i]))
            inner = int(np.prod(cardinality_ordered[(i + 1):n]))

            moving_cardinality[i] = (outer, inner)

        self.moving_cardinality_ordered = moving_cardinality
        self.moving_cardinality_ordered_reversed = list(reversed(moving_cardinality))

        return None





    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def get_dims_from_key(self,
        key_value: int,
        return_type: str = "tuple"
    ) -> int:
        """
        Get the dimensional values--ordered--from an input primary key.
            Acts as inverse to self.get_key_value(), i.e.

            key = self.get_key_value(**self.get_dims_from_key(key, return_type = "dict"))

            and

            dict_dims = self.get_dims_from_key(self.get_key_value(**dict_dims), return_type = "dict")

        Keyword Arguments
        -----------------
        - return_type: "tuple" or "dict"
            * if "tuple" returns a tuple ordered by self.dims_ordered
            * if "dict", returns a dictionary mapping each dimension to
                the associated value
        """

        key_iterator = key_value
        out = [None for x in self.dims_ordered]

        for dim in enumerate(self.dims_ordered):
            i, dim = dim

            cumulative_prod = self.cumulative_dimensional_products[i]
            card = self.cardinality_ordered[i]

            rem = key_iterator%cumulative_prod
            ind = int((key_iterator - rem)/cumulative_prod)
            val = self.indices_to_values_by_dim.get(dim).get(ind)

            out[i] = val

            key_iterator = rem

        out = (
            tuple(out) 
            if (return_type == "tuple") 
            else dict(zip(self.dims_ordered, out))
        )

        return out



    def get_indexing_dataframe(self,
        key_values: Union[Dict[str, List[int]], List[int], None] = None,
        keys_return: Union[List[str], None] = None,
        key_dict_logic: str = "and",
    ) -> pd.DataFrame:
        """
        Generate an indexing data frame that includes the primary key as well as
            component dimensions associated with those keys.

        Keyword Arguments
        -----------------
        - key_values: set of keys to return specified as a list of primary keys
            OR a dictionary of a dimensional key to values within that
            dimension. If None, returns all keys.
            * NOTE: caution should be exercised in returning all keys. The
                OrderedDirectProductTable class is designed to reduced the
                memory footprint of index tables, and returning the entire data
                frame can create a large dataframe.
        - keys_return: fields to return. If None, will return all defined keys.
        - key_dict_logic: "and" or "or".
            * If "and", when dimensional elements are specified in key_values as
                a dictionary, the data frame will only return rows for which
                *each* dimensional restriction is satisfied.
            * If set to "or", then rows are returned where *any* dimensional
                restriction is satisfied.
        """

        ##  CHECKS AND INITIALIZATION

        # check output keys
        fields_all_out = [self.key_primary] + self.dims_ordered

        keys_return = (
            [x for x in fields_all_out if x in keys_return] 
            if (keys_return is not None) 
            else fields_all_out
        )

        if len(keys_return) == 0:
            return None

        # check values and keep index
        n = int(np.prod(self.cardinality_ordered))
        key_values = (
            list(range(n)) 
            if (key_values is None) 
            else (
                [x for x in sorted(key_values) if x < n] 
                if not isinstance(key_values, dict) 
                else dict((k, v) for k, v in key_values.items() if k in self.dims_ordered)
            )
        )
        if (key_values is not None) and (len(key_values) == 0):
            return None

        keep_index = key_values if isinstance(key_values, list) else None

        # use intersection in filtering?
        use_intersection = True if (key_dict_logic not in ["and", "or"]) else (key_dict_logic == "and")


        ##  DO ITERATION - START WITH INDICES IN DICT (IF APPLICABLE), THEN FILTER ON VECTORS

         # iterate over dimensions in dict to get indices to keep if applicable
        if keep_index is None:
            for key in enumerate(self.dims_ordered):
                i, key = key
                if key in key_values.keys():
                    outer, inner = self.moving_cardinality_ordered[i]

                    # get values and appropriate indices
                    vals = self.values_by_dim.get(key)
                    vals_keep = [x for x in key_values.get(key) if x in vals]
                    inds = [self.values_to_indices_by_dim.get(key).get(x) for x in vals_keep]

                    w = sf.get_repeating_vec_element_inds(
                        inds,
                        len(vals),
                        inner,
                        outer
                    )

                    keep_index = set(w) if (keep_index is None) else (
                        keep_index.intersection(set(w)) if use_intersection else keep_index.union(set(w))
                    )

            keep_index = sorted(list(keep_index)) if (keep_index is not None) else None

        # initialize output array and other vars
        key_vals = np.arange(n) if (keep_index is None) else np.array(keep_index)
        arr_out = np.zeros((len(key_vals), len(keys_return))).astype(int)
        arr_out[:, 0] = key_vals

        # iterate over dimensions to get applicable subset
        j = 1
        for key in enumerate(self.dims_ordered):
            i, key = key

            if key in keys_return:
                outer, inner = self.moving_cardinality_ordered[i]

                vec_add = sf.build_repeating_vec(
                    self.values_by_dim.get(key),
                    inner,
                    outer,
                    keep_index = keep_index
                )

                arr_out[:, j] = vec_add

                j += 1

        # convert to dataframe
        arr_out = pd.DataFrame(arr_out, columns = keys_return)

        return arr_out


        
    def get_indexing_dataframe_from_primary_key(self,
        key_values: Union[List[int], None],
        keys_return: Union[List[str], None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Generate an indexing data frame that includes the primary key as well as
            component dimensions associated with those keys.

        Keyword Arguments
        -----------------
        - key_values: set of keys to return specified as a list of primary keys
            OR a dictionary of a dimensional key to values within that
            dimension. If None, returns all keys.
            * NOTE: caution should be exercised in returning all keys. The
                OrderedDirectProductTable class is designed to reduced the
                memory footprint of index tables, and returning the entire data
                frame can create a large dataframe.
        - keys_return: fields to return. If None, will return all defined keys.
        """

        # filter keys specified
        key_values_iter = [x for x in key_values if x in self.range_key_primary]
        df_out = [self.cardinality_ordered for x in key_values_iter]

        if len(key_values_iter) == 0:
            return None


        # iterate over primary keys to overwrite in list
        for i, key in enumerate(key_values_iter):
            df_out[i] = self.get_dims_from_key(
                key, 
                return_type = "tuple"
            )

        # build data frame and dropp any fields 
        df_out = pd.DataFrame(df_out, columns = self.dims_ordered)
        df_out[self.key_primary] = key_values_iter
        df_out = df_out[[self.key_primary] + self.dims_ordered]

        if sf.islistlike(keys_return):
            keys_drop = [x for x in df_out.columns if x not in keys_return]
            (
                df_out.drop(keys_drop, axis = 1, inplace = True)
                if len(keys_drop) > 0
                else None
            )

            if df_out.shape[1] == 0:
                return None

        return df_out



    def get_key_value(self,
        **kwargs
    ) -> int:
        """
        Get the key value associated with an input set of dimensional values.
        """
        val_out = 0

        for dim in enumerate(self.dims_ordered_reversed):
            i, dim = dim

            # retrieve the value and the associated dimensional index
            val_default = min(self.values_by_dim.get(dim))
            val = kwargs.get(dim, val_default)
            ind = self.values_to_indices_by_dim.get(dim).get(val)

            val_out += self.cumulative_dimensional_products_reversed[i]*ind

        return val_out
