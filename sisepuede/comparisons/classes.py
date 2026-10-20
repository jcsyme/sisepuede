"""Module containing functions to compare variables 
"""
import numpy as np
import pandas as pd
import sisepuede.utilities._toolbox as sf
from sisepuede.core.model_attributes import is_model_attributes 
from typing import *



##########################
#    GLOBAL VARIABLES    #
##########################

# UUID
_MODULE_UUID = "8F8448E2-28A6-4E21-B75C-66ACDB7A229C"

# 
_PREFIX_BASE = "BASE_VAL_"
_PREFIX_DIFF = "DIFF_VAL_"
_PREFIX_PERTURB = "PERTURB_VAL_"


##  ERROR CLASSES

class IntersectionError(Exception):
    pass




##################
#    FUNCTION    #
##################

class DFComparer:
    """Create a baseline comparison class for calculating differences and
        changes between DataFrames (model input and output) in SISEPUEDE.
    
    Initialization Arguments
    ------------------------
    df_baseline : pd.DataFrame
        DataFrame against which other values will be compared. In SISEPUEDE, 
        this is often
    model_attributes : ModelAttributes
    """

    def __init__(self,
        df_baseline: pd.DataFrame,
        model_attributes: 'ModelAttributes',
    ) -> None: 


        self._initialize_model_attributes(model_attributes, )
        self._initialize_df_baseline(df_baseline, )
        self._initialize_other_parameters()
        
        return None



    def _check_df_baseline(self,
        df_baseline: pd.DataFrame,
    ) -> Tuple[List[str], pd.DataFrame]:
        """Check the baseline DataFrame
        """
        if not isinstance(df_baseline, pd.DataFrame):
            tp = str(type(df_baseline))
            raise TypeError(f"Invalid type '{tp}' of df_baseline: must be pandas DataFrame.")

        # verify indexing
        fields_ind_baseline = [
            x for x in self.model_attributes.sort_ordered_dimensions_of_analysis
            if x in df_baseline.columns
        ]

        out = (
            fields_ind_baseline,
            df_baseline,
        )

        return out



    def _initialize_df_baseline(self,
        df_baseline: pd.DataFrame,
    ) -> None:
        """Initialize the baseline DataFrame to use for differences.
        """

        # check and remove 
        _, df_baseline = self._check_df_baseline(df_baseline, )
        df_baseline = (
            df_baseline
            .copy()
            .reset_index(drop = True, )
        )

        
        ##  SET PROPERTIES

        self.df_baseline = df_baseline

        return None



    def _initialize_model_attributes(self,
        model_attributes: 'ModelAttributes',
    ) -> None:
        """Initialize the ModelAttributes core and any associated derivative
            shortcuts.
        """

        if not is_model_attributes(model_attributes, ):
            tp = str(type(model_attributes))

            raise TypeError(f"Invalid type '{tp}' of model_attributes: must be ModelAttributes object.")


        ##  SET PROPERTIES

        self.model_attributes = model_attributes

        return None



    def _initialize_other_parameters(self,
    ) -> None:
        """Initialize other basic parameters. Sets the following properties:

            * self.prefix_diff

        """

        # get prefixes
        prefix_uuid = _MODULE_UUID.split("-")[-1]
        prefix_base = f"{_PREFIX_BASE}{prefix_uuid}"
        prefix_diff = f"{_PREFIX_DIFF}{prefix_uuid}"
        prefix_perturb = f"{_PREFIX_PERTURB}{prefix_uuid}"


        ##  SET PROPERTIES

        self.prefix_base = prefix_base
        self.prefix_diff = prefix_diff
        self.prefix_perturb = prefix_perturb

        return None





    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def get_diff(self,
        df_perturbation: pd.DataFrame,
        fields: Union[List[str], np.ndarray],
        base_index_in_model_attributes: bool = True,
        df_baseline: Union[pd.DataFrame, None] = None,
        fields_index: Union[List[str], np.ndarray, None] = None,
        merge_type: str = "inner",
        rename_output: bool = True,
        sign: int = 1,
        sort_on_fields_index: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate the difference in fields

                df_perturbation[fields] - df_baseline[fields] 

            On a merged index. Returns a tuple of the form

            (
                df_out,
                fields_data_out,
            )


        Function Arguments
        ------------------
        df_perturbation : pd.DataFrame
            DataFrame storing data against the baseline that is 
        fields: Union[List[str], np.ndarray]
            Fields to calculate difference for
        fields: Union[List[str], np.ndarray]
        
        Keyword Arguments
        -----------------
        base_index_in_model_attributes : bool
            Set to True to ensure that index fields are the intersection of all
            shared fields with the those in 
            model_attributes.sort_ordered_dimensions_of_analysis.
        df_baseline : Union[pd.DataFrame, None]
            Optional specification of a different baseline dataframe
        fields_index : Union[List[str], np.ndarray, None]
            Optional index fields to specify. Check for interactions with 
            base_index_in_model_attributes. If None, defaults to shared columns.
        merge_type : str
            Optional specification of merge type. The order is always

                pd.merge(
                    df_baseline,
                    df_perturbation,
                    how = merge_type,
                    ...
                )

            Allows for left/right merges.
        rename_output : bool
            Rename the output to the original fields? If False, leaves the
            fields with the diff prefix
        sign : int
            Sign to use for subtraction. 
            * 1:            df_perturbation[fields] - df_baseline[fields] 
                            remains
            * -1:           df_baseline[fields] - df_perturbation[fields]
            * other number: uses sign of that number
            * not number:   1
        sort_on_fields_index : bool
            Optional specification to sort, ordered by fields_index, on final
            output DataFrame. 
        """

        ##  INITIALIZATION

        # get baseline df and check 
        df_baseline = self.get_df_baseline(df_baseline, )
        sf.check_fields(df_baseline, fields)
        sf.check_fields(df_perturbation, fields)

        # get index fields
        fields_ind = self.get_fields_index(
            df_perturbation,
            base_in_model_attributes = base_index_in_model_attributes,
            df_baseline = df_baseline,
            fields_index = fields_index,
        )

        # check data fields--hack this function
        fields_data = self.get_fields_index(
            df_perturbation,
            base_in_model_attributes = False,
            df_baseline = df_baseline,
            fields_index = fields,
        )
        if len(fields_data) != len(fields):
            fields_missing = sf.print_setdiff(fields, fields_data)
            raise IntersectionError(f"Fields {fields_missing} not found in either df_baseline or df_perturbation.")


        ##  MERGE
        
        # get renaming dictionaries and fields
        dict_rename_pert, fields_data_pert = self.get_dict_rename_for_diff(
            fields_data, 
            "perturbation", 
        )

        # those for differences
        dict_rename_diff, fields_data_diff = self.get_dict_rename_for_diff(
            fields_data, 
            "diff", 
        )


        # start merge
        df_merge = pd.merge(
            df_baseline
            .get(fields_ind + fields_data),

            df_perturbation
            .rename(columns = dict_rename_pert, )
            .get(fields_ind + fields_data_pert),

            how = merge_type,
            on = fields_ind,
        )


        ##  GET DIFF

        # set sign
        sign = (
            1
            if not sf.isnumber(sign)
            else np.sign(sign)
        )

        # get difference and optional sort
        df_out = df_merge[fields_ind].copy()
        df_out[fields_data_diff] = sign*(
            df_merge[fields_data_pert].to_numpy() - df_merge[fields_data].to_numpy()
        )


        ##  PREPARE OUTPUT
        
        if sort_on_fields_index:
            df_out = (
                df_out
                .sort_values(by = fields_ind)
                .reset_index(drop = True)
            )

        # rename back?
        fields_data_out = fields_data_diff
        if rename_output:
            dict_rnm = sf.reverse_dict(dict_rename_diff, )
            df_out = df_out.rename(columns = dict_rnm, )
            fields_data_out = [dict_rnm.get(x) for x in fields_data_diff]


        ##  SET OUTPUT

        out = (
            df_out,
            fields_data_out,
        )

        return out



    def get_dict_rename_for_diff(self,
        fields_rename: List[str],
        rename_type: str,
    ) -> Tuple[Dict[str, str], List[str]]:
        """Get a renaming dictionary and new field names. Returns a tuple of
            the form

            (
                dict_out,
                fields_out,
            )

        Function Arguments
        ------------------
        fields_rename : List[str]
            Base fields to rename
        rename_type : str
            * "base":           Rename to use base prefix
            * "diff":           Rename to use difference prefix
            * "perturbation":   Rename to use perturbation prefix
        """
        # get the rename
        rename_type = str(rename_type).lower()
        if rename_type == "base":
            prefix = self.prefix_base
        if rename_type == "diff":
            prefix = self.prefix_diff
        if rename_type == "perturbation":
            prefix = self.prefix_perturb

        # set output dictionary
        dict_out = dict((x, f"{prefix}_{x}") for x in fields_rename)
        fields_out = [dict_out.get(x) for x in fields_rename]

        out = (
            dict_out,
            fields_out,
        )

        return out

        

        

    def get_df_baseline(self,
        df_baseline: Union[pd.DataFrame, None] = None,
    ) -> pd.DataFrame:
        """Get a baseline DataFrame. If None, defaults to self.df_baseline
        """

        if not isinstance(df_baseline, pd.DataFrame):
            return self.df_baseline

        return df_baseline



    def get_fields_index(self,
        df_perturbation: pd.DataFrame,
        base_in_model_attributes: bool = True,
        df_baseline: Union[pd.DataFrame, None] = None,
        fields_index: Union[List[str], np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """Get index fields for merging. 
        
        * NOTE: If valid fields_index is specificied, iteration order for 
                intersection of fields is in order of fields_index.


        Function Arguments
        ------------------
        df_perturbation : pd.DataFrame
            Perturbed DataFrame (non-baseline) to compare

        Keyword Arguments
        -----------------
        base_in_model_attributes : Bool
            If base_in_model_attributes is True, will only seek index fields in
            model_attributes.sort_ordered_dimensions_of_analysis that are in the
            intersection. Otherwise, checks against fields only in the
            intersection.
        df_baseline: Union[pd.DataFrame, None]
            Optional baseline dataframe to specify
        """

        # get the baseline df
        df_baseline = self.get_df_baseline(df_baseline, )  

        # check fields
        intersect = list(set(df_baseline.columns) & set(df_perturbation.columns))
        fields_index = (
            intersect
            if not sf.islistlike(fields_index)
            else [x for x in fields_index if x in intersect]
        )

        fields_index = (
            fields_index
            if not base_in_model_attributes
            else [
                x for x in self.model_attributes.sort_ordered_dimensions_of_analysis 
                if x in fields_index
            ]
        )

        # check to make sure some are defined
        if len(fields_index) == 0:
            raise IntersectionError(f"No shared index columns found in df_perturbation and df_baseline.")

        
        return fields_index

