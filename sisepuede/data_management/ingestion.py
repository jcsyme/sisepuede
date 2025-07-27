
import logging
import os, os.path
import pandas as pd
import re
from typing import *

from sisepuede.core.attribute_table import AttributeTable, is_attribute_table
import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf



###############################
#    SOME GLOBAL VARIABLES    #
###############################

class MissingScalarsError(Exception):
    pass

class SheetReadError(Exception):
    pass





#
class InputTemplate:
    """The InputTemplate class is used to ingest an input data template and 
        format it for the SISEPUEDE DAG.

    See https://sisepuede.readthedocs.io for more information on the input 
        template.

    Initialization Arguments
    ------------------------
    template : Union[str, dict, None]
        The InputTemplate can be initialized using a file path to an 
        Excel file or a dictionary of
        * path : 
        if initializing using a path, the template should point to an 
            Excel workbook containing the input data template. A description of 
            the workbook's format is found below under "Template Formatting".
        * dict : 
        if initializing using a dictionary, the dictionary should have 
            the following structure:
            {
                "strategy_id-X0" : 
        pd.DataFrame(),
                "strategy_id-X1" : 
        pd.DataFrame()...
            }

            I.e., keys should follow the

    model_attributes : Union[str, dict, None]
        a ModelAttributes data structure used to coordinate
        variables and inputs

    Optional Arguments
    ------------------
    attribute_strategy : Union[AttributeTable, None]
        AttributeTable used to define input strategies and
        filter undefined.
        * If None (default), try to read from model_attributes.dict_attributes
        * If AttributeTable, checks key against ModelAttributes.dim_strategy_id
            * If either check is unsuccessful, will turn off filtering of
                undefined strategies and set 
                InputTemplate.attribute_strategy = None

    Keyword Arguments
    -----------------
    default_max_scalar : float
        Default max uncertainty scalar to apply to variables that might be 
        missing in a variable specification (XL exogenous) file. 
    default_min_scalar : float
        Default min uncertainty scalar to apply to variables that might be 
        missing in a variable specification (XL exogenous) file. 
    field_prepend_req_attr_baseline_scenario : str
        prepandage applied to AttributeTable key to generate field required in 
        attribute tables to specify a baseline scenario. E.g.,
        field_prepend_req_attr_baseline_scenario = "baseline_" means that the 
        baseline strategy_id is stored in field "baseline_strategy" in the 
        attribute_strategyattribute table.
        * Only applies for attributes not passed through ModelAttributes
    field_req_normalize_group : str
        Required field used to specify whether or not to normalize a group 
        (ensures always sums to 1)
    field_req_subsector : str
        Required field used to define the subsector associated with a variable
    field_req_trajgroup_no_vary_q : str
        Required field used to determine whether or not a trajectory group may 
        vary
        * Note:     all values in the same trajectory group must be the same
    field_req_uniform_scaling_q : str
        Required field used to determine whether or not a variable trjaectory 
        should be scaled uniformly over all time periods
        * E.g., many biophysical parameters may be uncertain but not change
            over time
    field_req_variable : str
        Required field used name the variable
        * Trajectory groups require special naming convention used to define all 
            parts:
            (INFO HERE)
    field_req_variable_trajectory_group : str
        Field used to explicitly add trajectory group (added after import)
    field_req_variable_trajectory_group_trajectory_type : str
        Field used to explicitly add trajectory group type for variables in a 
        trajectory group (added after import)
    filter_invalid_strategies : bool
        filter strategies that aren't defined in an attribute table
    logger : Union[logging.Logger, None] = None
        optional logging object to pass
    regex_max : re.Pattern
        re.Pattern (compiled regular expression) used to match the field storing 
        the maximum scalar values at the final time period
    regex_min : re.Pattern
        re.Pattern used to match the field storing the minimum scalar values at 
        the final time period
    regex_tp : re.Pattern
        re.Pattern used to match the field storing data values for each time 
        period


    Template Formatting
    -------------------

    (info here)


    """
    def __init__(self,
        template: Union[str, dict, None],
        model_attributes: ma.ModelAttributes,
        attribute_strategy: Union[AttributeTable, str, None] = None,
        default_max_scalar: float = 1.0,
        default_min_scalar: float = 1.0,
        field_prepend_req_attr_baseline_scenario: str = "baseline_",
        field_req_normalize_group: str = "normalize_group",
        field_req_subsector: str = "subsector",
        field_req_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
        field_req_uniform_scaling_q: str = "uniform_scaling_q",
        field_req_variable: str = "variable",
        field_req_variable_trajectory_group: str = "variable_trajectory_group",
        field_req_variable_trajectory_group_trajectory_type: str = "variable_trajectory_group_trajectory_type",
        filter_invalid_strategies: bool = True,
        logger: Union[logging.Logger, None] = None,
        regex_max: re.Pattern = re.compile("max_(\d*$)"),
        regex_min: re.Pattern = re.compile("min_(\d*$)"),
        regex_tp: re.Pattern = re.compile("(\d*$)"),
    ) -> None:
        
        self.logger = logger

        # initialize some basic properties
        self._initialize_basic_properties(
            default_max_scalar,
            default_min_scalar,
            filter_invalid_strategies,
        )

        #
        self._initialize_model_attributes(model_attributes, )
        self._initialize_basic_required_fields(
            field_prepend_req_attr_baseline_scenario,
            field_req_normalize_group,
            field_req_subsector,
            field_req_trajgroup_no_vary_q,
            field_req_uniform_scaling_q,
            field_req_variable,
            field_req_variable_trajectory_group,
            field_req_variable_trajectory_group_trajectory_type,
        )

        # initialize additional key elements
        self._initialize_regex_patterns(
            regex_max,
            regex_min,
            regex_tp
        )
        self._initialize_attribute_strategy(attribute_strategy, )
        self._set_regex_sheet_name()
        self._initialize_template(template, )

        return None





    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_attribute_strategy(self,
        attribute_strategy: Union[AttributeTable, str, None]
    ) -> None:
        """
        Initialize the strategy attribute table based on one of two potential 
            inputs. Sets two InputTemplate parameters:

        self.attribute_strategy
        self.baseline_strategy

        Function Arguments
        ------------------
        - attribute_strategy: One of the following:
            * None: no strategy table is passed, and templates are read without 
                consideration of defined strategies
            * AttributeTable: any attribute table with a key that matches
            self.model_attributes.dict_attributes(
                f"dim_{self.model_attributes.dim_strategy_id}"
            ). 

            The AttributeTable should also include the following fields:
                * baseline_scenario: binary field denoting the baseline 
                    scenario. Only one should exist per

        """
        # default to model attributes designation
        self.baseline_strategy = self.model_attributes.get_baseline_scenario_id(self.model_attributes.dim_strategy_id)

        if is_attribute_table(attribute_strategy, ):
            out = attribute_strategy
            field_check = f"{self.field_prepend_req_attr_baseline_scenario}{out.key}"

            # verify key match - if matched, check for baseline strategy id
            if attribute_strategy.key != self.model_attributes.dim_strategy_id:

                msg = f"""Invalid attribute_strategy passed to InputTemplate: 
                    strategy key '{attribute_strategy.key}' does not match the 
                    ModelAttributes strategy key 
                    '{self.model_attributes.dim_strategy_id}'. Check the file at 
                    '{attribute_strategy.fp_table}'. Setting 
                    self.attribute_strategy = None.
                    """

                self._log(msg, type_log = "warning")
                out = None

            elif (field_check in out.table.columns) and (len(out.table) > 0):
                tab_filt = out.table[out.table[field_check] == 1][out.key]
                if len(tab_filt) > 0:
                    msg = f"""Multiple specifications of baseline {out.key} 
                        found in field '{field_check}' in attribute_strategy. 
                        Check the file at '{out.fp_table}'. Inferring baseline 
                        {out.key} = {self.baseline_strategy}.
                        """

                    self.baseline_strategy = min(tab_filt)
                    self._log(msg, type_log = "info") if (len(tab_filt) > 1) else None

                else:
                    msg = f"""No specifications of baseline {out.key} found in 
                        field '{field_check}' in attribute_strategy. Check the 
                        file at '{out.fp_table}'. Inferring baseline 
                        {out.key} = {self.baseline_strategy}.
                        """

                    self.baseline_strategy = min(out.table[out.key])
                    self._log(msg, type_log = "info")

        elif (attribute_strategy is None) and (self.filter_invalid_strategies):

            key_try = self.model_attributes.dim_strategy_id
            out = self.model_attributes.get_dimensional_attribute_table(key_try)
            if out is None:
                msg = f"""No strategy attribute found in ModelAttributes using 
                    dict_attributes key '{key_try}'. Setting 
                    self.attribute_strategy = None and self.baseline_strategy = 
                    {self.baseline_strategy}.
                    """

                self._log(msg, type_log = "warning")

        else:
            tp = str(type(attribute_strategy))
            msg = f"""Invalid type '{tp}' of attribute_strategy passed to 
                InputTemplate. Setting self.attribute_strategy = None and 
                self.baseline_strategy = {self.baseline_strategy}.
                """
            self._log(msg, type_log)
            out = None

        self.attribute_strategy = out

        return None



    def _initialize_basic_properties(self,
        default_max_scalar: float = 1.0,
        default_min_scalar: float = 1.0,
        filter_invalid_strategies: bool = True,
    ) -> None:
        """Initialize some basic properties. Sets the following properties:

            * self.default_max_scalar
            * self.default_min_scalar
            * self.filter_invalid_strategies

        Function Arguments
        ------------------
        default_max_scalar : float
            Default max uncertainty scalar to apply to variables that might be 
            missing in a variable specification (XL exogenous) file. 
        default_min_scalar : float
            Default min uncertainty scalar to apply to variables that might be 
            missing in a variable specification (XL exogenous) file. 
        filter_invalid_strategies : bool
            Filter strategies that aren't defined in an attribute table

        """

        default_max_scalar = 1.0 if not sf.isnumber(default_max_scalar) else default_max_scalar
        default_min_scalar = 1.0 if not sf.isnumber(default_min_scalar) else default_min_scalar
        
        m0 = min(default_max_scalar, default_min_scalar)
        m1 = max(default_max_scalar, default_min_scalar)


        ##  SET PROPERTIES

        self.default_max_scalar = m1
        self.default_min_scalar = m0
        self.filter_invalid_strategies = filter_invalid_strategies

        return None



    def _initialize_basic_required_fields(self,
        field_prepend_req_attr_baseline_scenario: str,
        field_req_normalize_group: str,
        field_req_subsector: str,
        field_req_trajgroup_no_vary_q: str,
        field_req_uniform_scaling_q: str,
        field_req_variable: str,
        field_req_variable_trajectory_group: str,
        field_req_variable_trajectory_group_trajectory_type: str
    ) -> None:
        """
        Initialize required fields (explicitly assigned within the
            function). Sets the following properties:

            * self.field_prepend_req_attr_baseline_scenario
            * self.field_req_normalize_group
            * self.field_req_subsector
            * self.field_req_trajgroup_no_vary_q
            * self.field_req_uniform_scaling_q
            * self.field_req_variable
            * self.field_req_variable_trajectory_group
            * self.field_req_variable_trajectory_group_trajectory_type
            * self.list_fields_required_base
            * self.list_fields_required_binary

        Function Arguments
        ------------------
        - field_prepend_req_attr_baseline_scenario: prepandage applied to 
            AttributeTable key to generate field required in attribute tables to 
            specify a baseline scenario. E.g., 
            field_prepend_req_attr_baseline_scenario = "baseline_" means that 
            the baseline strategy_id is stored in field "baseline_strategy" in 
            the attribute_strategy attribute table.
            * Only applies for attributes not passed through ModelAttributes
        - field_req_normalize_group: Required field used to specify whether or 
            not to normalize a group (ensures always sums to 1)
        - field_req_subsector: Required field used to define the ubsector 
            associated with a variable
        - field_req_trajgroup_no_vary_q: Required field used to determine 
            whether or not a trajectory group may vary
            * Note: all values in the same trajectory group must be the same
        - field_req_uniform_scaling_q: Required field used to determine whether 
            or not a variable trjaectory should be scaled uniformly over all 
            time periods
            * E.g., many biophysical parameters may be uncertain but not change 
                over time
        - field_req_variable: Required field used name the variable
            * Trajectory groups require special naming convention used to define 
                all parts: (INFO HERE)
        - field_req_variable_trajectory_group: Field used to explicitly add 
            trajectory group (added after import)
        - field_req_variable_trajectory_group_trajectory_type: Field used to 
            explicitly add trajectory group type for variables in a trajectory 
            group (added after import)

        """
        # set characteristics of the template (can be modified if needed)
        self.field_prepend_req_attr_baseline_scenario = field_prepend_req_attr_baseline_scenario
        self.field_req_normalize_group = field_req_normalize_group
        self.field_req_subsector = field_req_subsector
        self.field_req_trajgroup_no_vary_q = field_req_trajgroup_no_vary_q
        self.field_req_uniform_scaling_q = field_req_uniform_scaling_q
        self.field_req_variable = field_req_variable
        self.field_req_variable_trajectory_group = field_req_variable_trajectory_group
        self.field_req_variable_trajectory_group_trajectory_type = field_req_variable_trajectory_group_trajectory_type
        self.list_fields_required_base = [
            self.field_req_normalize_group,
            self.field_req_subsector,
            self.field_req_trajgroup_no_vary_q,
            self.field_req_uniform_scaling_q,
            self.field_req_variable,
            self.field_req_variable_trajectory_group,
            self.field_req_variable_trajectory_group_trajectory_type
        ]
        self.list_fields_required_binary = [
            self.field_req_normalize_group,
            self.field_req_trajgroup_no_vary_q,
            self.field_req_uniform_scaling_q
        ]

        return None



    def _initialize_model_attributes(self,
        model_attributes: ma.ModelAttributes
    ) -> None:
        """
        Initialize model attributes. Sets the following properties:

            * self.model_attributes

        Function Arguments
        ------------------
        - model_attributes: ModelAttributes object used to organize variables
            and model structure.
        """
        # check type
        if not ma.is_model_attributes(model_attributes):
            tp = str(type(model_attributes))
            msg = f"Invalid type '{tp}' specified for model_attributes in Ingestion. Must be a ModelAttributes object."
            raise TypeError(msg)

        self.model_attributes = model_attributes

        return None



    def _initialize_regex_patterns(self,
        regex_max: re.Pattern = re.compile("max_(\d*$)"),
        regex_min: re.Pattern = re.compile("min_(\d*$)"),
        regex_tp: re.Pattern = re.compile("(\d*$)")
    ) -> None:
        """
        Initialize some regular expressions used in the templates. Sets the
            following properties:

            * self.regex_template_max
            * self.regex_template_min
            * self.regex_template_time_period
        """
        self.regex_template_max = regex_max if isinstance(regex_max, re.Pattern) else re.compile("max_(\d*$)")
        self.regex_template_min = regex_min if isinstance(regex_min, re.Pattern) else re.compile("min_(\d*$)")
        self.regex_template_time_period = regex_tp if isinstance(regex_tp, re.Pattern) else re.compile("(\d*$)")

        return None



    def _initialize_template(self,
        template: Union[str, dict, None]
    ) -> None:
        """
        Initialize template components. Sets the following properties:

            * self.dict_strategy_id_to_sheet
            * self.dict_strategy_id_to_strategy_sheet
            * self.field_max
            * self.field_min
            * self.fields_tp
        """

        # initialize as None
        self.dict_strategy_id_to_sheet = None
        self.dict_strategy_id_to_strategy_sheet = None
        self.field_max = None
        self.field_min = None
        self.fields_tp = None

        # try reading the template tuple
        template_tuple = self.read_template(template, )
        if template_tuple is not None:
            (
                self.dict_strategy_id_to_sheet,
                self.dict_strategy_id_to_strategy_sheet,
                self.field_max,
                self.field_min,
                self.fields_tp
            ) = template_tuple
        
        return None



    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """Clean implementation of sf._optional_log in-line using default 
            logger. See ?sf._optional_log for more information

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



    def _set_regex_sheet_name(self,
    ) -> None:
        """Set the regular expression for input sheets to match. Sets the 
            following properties:

            * self.regex_sheet_name
        """
        self.regex_sheet_name = re.compile(f"{self.model_attributes.dim_strategy_id}-(\d*$)")

        return None



    #########################
    #	TEMPLATE FUNCTIONS	#
    #########################

    def build_inputs_by_strategy(self,
        dict_strategies_to_sheet: Union[dict, None] = None,
        strategies_include: Union[list, None] = None,
    ) -> pd.DataFrame:
        """Built a sectoral input variable database for SISEPUEDE based on the 
            input template. This database can be combined across multiple 
            templates and used to create a SampleUnit object to explore 
            uncertainty.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        dict_strategies_to_sheet : Union[dict, None]
            Dictionary of type {int -> pd.DataFrame} that maps a strategy id to 
            its associated template sheet
            * If None (default), uses InputTemplate.dict_strategy_id_to_sheet
        strategies_include : Union[list, None]
            List or list-like (np.array) of strategy ids to include in the 
            database (integer). If None, include all.

        """
        dict_strategies_to_sheet = (
            self.dict_strategy_id_to_sheet 
            if (dict_strategies_to_sheet is None) 
            else dict_strategies_to_sheet
        )
        strat_base = self.baseline_strategy
        strats_all = sorted(list(dict_strategies_to_sheet.keys()))
        strategies_include = (
            strats_all 
            if (strategies_include is None)
            else [x for x in strategies_include if x in strats_all]
        )

        if (strat_base not in strategies_include) & (strat_base not in dict_strategies_to_sheet.keys()):
            raise KeyError(f"Error in build_inputs_by_strategy: key '{strat_base}' (baseline strategy) not found")
                
        strategies_include = [strat_base] + [x for x in strategies_include if (x != strat_base)]

        #
        df_out = []

        for strat in strategies_include:
            df_sheet = dict_strategies_to_sheet.get(strat)
            df_sheet = self.model_attributes.add_index_fields(
                df_sheet,
                strategy_id = strat
            )

            if strat != strat_base:
                #
                # strat_base is always the first
                #
                vars_cur = list(df_sheet[self.field_req_variable])
                df_sheet_base = df_out[0][~df_out[0][self.field_req_variable].isin(vars_cur)].copy()
                df_sheet_base[self.model_attributes.dim_strategy_id] = (
                    df_sheet_base[self.model_attributes.dim_strategy_id]
                    .replace({strat_base: strat, })
                )

                df_sheet = (
                    pd.concat(
                        [
                            df_sheet, 
                            df_sheet_base[df_sheet.columns]
                        ], 
                        axis = 0
                    )
                    .reset_index(drop = True)
                )

            (
                df_out.append(df_sheet)
                if len(df_out) == 0
                else df_out.append(df_sheet[df_out[0].columns])
            )
                

        # do some conversions
        df_out = pd.concat(df_out, axis = 0)
        df_out = self.clean_inputs(df_out)

        return df_out
    


    def clean_inputs(self,
        df_in: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply some cleaning to the inputs 
        """

        # clean up the normalization group field
        df_in[self.field_req_normalize_group] = (
            df_in[self.field_req_normalize_group]
            .replace({" ": 0})
            .fillna(0)
            .astype(int)
        )

        # clean up other fields as type integer
        flds_clean = [
            self.field_req_trajgroup_no_vary_q,
            self.field_req_uniform_scaling_q,
        ]
        
        #for fld in flds_clean:
        df_in[flds_clean] = df_in[flds_clean].astype(int)

        return df_in
    


    def get_sheet_strategy(self,
        sheet_name: str,
        return_type: type = int
    ) -> Union[int, dict]:
        """Get the strategy associated with a sheet.

        Function Arguments
        ------------------
        sheet_name : str
            The name of the sheet to import

        Keyword Arguments
        -----------------
        return_type : type
            * return_type = int will return the strategy number associated with 
                a sheet
            * return_type = dict will return a dictionary mapping the sheet name 
                to the strategy number
        """
        out = self.regex_sheet_name.match(sheet_name)
        # update if there is a match
        if out is not None:
            id = int(out.groups()[0])
            out = {sheet_name: id} if (return_type == dict) else id

        return out



    def name_field_maxmin_from_index(self,
        index: int,
        regex: Union[re.Pattern, str, None] = None
    ) -> str:
        """Substitute index into the match pattern contained in regex.

        Function Arguments
        ------------------
        index : int
            Integer index to use for field max

        Keyword Arguments
        -----------------
        regex : Union[re.Pattern, str, None]
            Regular expression (re.Pattern) used to define field for scalars at 
            time t_max or t_min. Optional values include:
            * "max":        use default regular expression for field_max
            * "min":        use default regular expression for field_min
            * re.Pattern:   use a different pattern
            * None:         return str(index)
        """
        out = None
        if not isinstance(index, int):
            return out

        if regex in ["max", "min"]:
            regex = self.regex_template_max if (regex == "max") else self.regex_template_min

        if isinstance(regex, re.Pattern):
            str_prepend = regex.pattern.split("(")[0]
            out = f"{str_prepend}{index}"

        return out



    def name_sheet_from_index(self,
        index: int
    ) -> str:
        """Using the regular expression template self.regex_sheet_name, build a
            sheet name in a template.

        Function Arguments
        ------------------
        index : int
            Integer index to use in sheet naming
        """
        prepend = self.regex_sheet_name.pattern.split("(")[0]

        return f"{prepend}{index}"



    def read_template(self,
        template_input: Union[str, dict],
    ) -> tuple:
        """Import the InputTemplate, check strategies, and set characteristics

        Returns
        -------
        Returns a 5-tuple of the following order:
        
            dict_outputs:               dictionary {int: pd.DataFrame} mapping a 
                                            strategy id to the associated
                                            DataFrame
            dict_sheet_to_strategy:     dictionary mapping a sheet to a strategy 
                                            name
            field_max:                  field specifying the maximum scalar in 
                                            the final time period
            field_min:                  field specifying the minimum scalar in
                                            the final time period
            fields_tp:                  fields specifying the time periods

        Function Arguments
        ------------------
        template_input : Union[str, dict]
            File path (str) to Excel Template or input dictionary (dict) with 
            keys matching InputTemplate.regex_sheet_name
        """

        if isinstance(template_input, str):
            fp_read = sf.check_path(template_input, False)
            dict_inputs = pd.read_excel(template_input, sheet_name = None)

        elif not isinstance(template_input, dict):
            return None
        else:
            dict_inputs = template_input

        # iteration initializations - objects used to check
        all_time_period_fields = None
        all_time_period_fields_max = None
        any_baseline = False
        strat_base = self.baseline_strategy

        # outputs and iterators
        dict_outputs = {}
        dict_strategy_to_sheet = {}
        sheets_iterate = list(dict_inputs.keys())

        for k in sheets_iterate:
            
            # if the sheet doesn't match, skip
            if self.regex_sheet_name.match(k) is None:
                continue

            # get the strategy, test if it is baseline
            dict_strat_cur = self.get_sheet_strategy(k, return_type = dict)
            strat_cur = dict_strat_cur.get(k)

            # check strategy filter
            proceed_q = False if self.filter_invalid_strategies else True
            if (not proceed_q) & (self.attribute_strategy is not None):
                proceed_q = (
                    True
                    if strat_cur in self.attribute_strategy.key_values
                    else proceed_q
                )

            # if the strategy is invalid, skip
            if not proceed_q:
                self._log(
                    f"InputTemplate initialization warning: Strategy '{strat_cur}' not found--it will not be included in the input database.", 
                    type_log = "warning"
                )
                continue
            
            
            # get the data frame and check if baseline strategy
            baseline_q = (strat_cur == strat_base)
            any_baseline |= baseline_q
            df_template_sheet = dict_inputs.get(k)

            # verify and skip on fail. Note: error passing and descriptions are handled in verify_input_template_sheet
            tup = self.verify_input_template_sheet(df_template_sheet, base_strategy_q = baseline_q)
            if tup is None:
                continue

            dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp = tup

            # check time period fields
            all_time_period_fields = set(fields_tp) if (all_time_period_fields is None) else (all_time_period_fields & set(fields_tp))
            if all_time_period_fields != set(fields_tp):
                raise SheetReadError(f"Encountered inconsistent definition of time periods in sheet {k}.")

            # check max time period fields
            all_time_period_fields_max = (
                set([field_max]) 
                if (all_time_period_fields_max is None) 
                else (all_time_period_fields_max | set([field_max]))
            )
            if len(all_time_period_fields_max) > 1:
                raise SheetReadError(f"Encountered inconsistent definition of fields specifying maximum and minimum scalars in sheet {k}.")

            # check binary fields HEREHERE
            for fld in self.list_fields_required_binary:
                df_template_sheet = sf.check_binary_fields(df_template_sheet, fld)

            # update outputs
            dict_strategy_to_sheet.update({strat_cur: k})
            df_template_sheet = df_template_sheet[self.list_fields_required_base + [field_min, field_max] + fields_tp]
            dict_outputs.update({strat_cur: df_template_sheet})


        if not any_baseline:
            self._log(
                f"Note: no sheets associated with the baseline strategy {strat_base} were found in the input template. Check the template before proceeding to build the input database.", 
                type_log = "warning"
            )


        tup_out = (
            dict_outputs, 
            dict_strategy_to_sheet, 
            field_max, 
            field_min, 
            fields_tp
        )

        return tup_out
    


    def template_from_inputs(self,
        df_input: pd.DataFrame,
        df_variable_information: pd.DataFrame,
        sectors: Union[str, List[str], None],
        df_trajgroup: Union[pd.DataFrame, None] = None,
        field_key_strategy: Union[str, None] = None,
        field_req_normalize_group: Union[str, None] = None,
        field_req_subsector: Union[str, None] = None,
        field_req_trajgroup_no_vary_q: Union[str, None] = None,
        field_req_uniform_scaling_q: Union[str, None] = None,
        field_req_variable: Union[str, None] = None,
        field_req_variable_trajectory_group: Union[str, None] = None,
        field_req_variable_trajectory_group_trajectory_type: Union[str, None] = None,
        fill_missing_scalars_with_defaults: bool = True, 
        include_simplex_group_as_trajgroup: bool = True,
        regex_max: Union[re.Pattern, None] = None,
        regex_min: Union[re.Pattern, None] = None,
        regex_tp: Union[re.Pattern, None] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Convert an input DataFrame to a template dictionary that can be 
            written to Excel as a template.

        Function Arguments
        ------------------
        df_input : pd.DataFrame
            DataFrame containing inputs to templatize.
        variable_ranges : pd.DataFrame
            Input ranges for sampling to use in template.
        sectors : Union[str, List[str], None]
            List of sectors to include (also allows for pipe-delimitted string 
            or None; if is None, applies for all valid sectors)

        Keyword Arguments
        -----------------
        df_trajgroup : Union[pd.DataFrame, None]  
            Optional dataframe mapping each field variable to trajectory groups. 
            * Must contain field_subsector, field_variable, and 
                field_variable_trajectory_group as fields
            * Overrides include_simplex_group_as_trajgroup if specified and 
                conflicts occur
        field_key_strategy : Union[str, None]
            Strategy key included in df_input--used to pivot templates (template 
            sheets are indexed by strategy key when defined)
        field_prepend_req_attr_baseline_scenario : Union[str, None]
            Prepandage applied to AttributeTable key to generate field required 
            in attribute tables to specify a baseline scenario. E.g.,
            field_prepend_req_attr_baseline_scenario = "baseline_" means that
            the baseline strategy_id is stored in field "baseline_strategy" in
            the attribute_strategyattribute table.
            * Only applies for attributes not passed through ModelAttributes
        field_req_normalize_group : Union[str, None]  
            Required field used to specify whether or not to normalize a group 
            (ensures always sums to 1)
        field_req_subsector : Union[str, None]
            Required field used to define the ubsector associated with a 
            variable
        field_req_trajgroup_no_vary_q : Union[str, None]
            Required field used to determine whether or not a trajectory group 
            may vary
            * NOTE: all values in the same trajectory group must be the same
        field_req_uniform_scaling_q : Union[str, None] 
            Required field used to determine whether or not a variable 
            trjaectory should be scaled uniformly over all time periods
            * E.g., many biophysical parameters may be uncertain but not change
                over time
        field_req_variable : Union[str, None]
            Required field used name the variable
        field_req_variable_trajectory_group : Union[str, None]
            Field used to explicitly add trajectory group (added after import)
        field_req_variable_trajectory_group_trajectory_type : Union[str, None]
            Field used to explicitly add trajectory group type for variables in 
            a trajectory group (added after import)
        fill_missing_scalars_with_defaults : bool
            If max/min scalars are missing from a variable specification, will
            fill with default values if set to true. If False, an error will be
            thrown.
        include_simplex_group_as_trajgroup : bool
            Default to include simplex group from attributes as trajectory g
            roup?
        regex_max : Union[re.Pattern, None]
            re.Pattern (compiled regular expression) used to match the field 
            storing the maximum scalar values at the final time period
        regex_min : Union[re.Pattern, None]
            re.Pattern used to match the field storing the minimum scalar values
            at the final time period
        regex_tp : Union[re.Pattern, None]
            re.Pattern used to match the field storing data values for each time 
            period
        """
        # attributes
        attr_tp = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_time_period
        )
        
        # fields
        field_key_strategy = (
            self.attribute_strategy.key 
            if (field_key_strategy is None) 
            else field_key_strategy
        )
        field_req_normalize_group = (
            self.field_req_normalize_group 
            if (field_req_normalize_group is None) 
            else field_req_normalize_group
        )
        field_req_subsector = (
            self.field_req_subsector 
            if (field_req_subsector is None) 
            else field_req_subsector
        )
        field_req_trajgroup_no_vary_q = (
            self.field_req_trajgroup_no_vary_q 
            if (field_req_trajgroup_no_vary_q is None) 
            else field_req_trajgroup_no_vary_q
        )
        field_req_uniform_scaling_q = (
            self.field_req_uniform_scaling_q 
            if (field_req_uniform_scaling_q is None) 
            else field_req_uniform_scaling_q
        )
        field_req_variable = (
            self.field_req_variable 
            if (field_req_variable is None) 
            else field_req_variable
        )
        field_req_variable_trajectory_group = (
            self.field_req_variable_trajectory_group 
            if (field_req_variable_trajectory_group is None) 
            else field_req_variable_trajectory_group
        )
        field_req_variable_trajectory_group_trajectory_type = (
            self.field_req_variable_trajectory_group_trajectory_type 
            if (field_req_variable_trajectory_group_trajectory_type is None) 
            else field_req_variable_trajectory_group_trajectory_type
        )
        field_time_period = self.model_attributes.dim_time_period
        
        # regular expressions
        regex_max = self.regex_template_max if not isinstance(regex_max, re.Pattern) else regex_max
        regex_min = self.regex_template_min if not isinstance(regex_min, re.Pattern) else regex_min
        regex_tp = self.regex_template_time_period if not isinstance(regex_tp, re.Pattern) else regex_tp


        ##  BUILD BASE DATA FRAME

        # initialize base variables
        field_melted_value = "value"
        df_base = self.model_attributes.build_variable_dataframe_by_sector(
            sectors,
            df_trajgroup = df_trajgroup,
            field_subsector = field_req_subsector,
            field_variable_field = field_req_variable,
            field_variable_trajectory_group = field_req_variable_trajectory_group,
            include_model_variable = False,
            include_simplex_group_as_trajgroup = include_simplex_group_as_trajgroup,
            include_time_periods = True,
        )
        trajgroup_passed_to_base_q = (field_req_variable_trajectory_group in df_base.columns)

        # melt input dataframe to long and filter
        fields_id = [x for x in df_input.columns if x in [field_key_strategy, field_time_period]]
        df_input_merge = pd.melt(
            df_input,
            id_vars = fields_id,
            var_name = field_req_variable
        )
        df_input_merge = (
            df_input_merge[
                df_input_merge[field_req_variable]
                .isin(list(df_base[field_req_variable]))
            ]
            .reset_index(drop = True)
        )

        # add baseline strategy
        if (field_key_strategy not in df_input_merge.columns):
            df_input_merge[field_key_strategy] = self.baseline_strategy

        all_strategies = sorted(list(set(df_input_merge[field_key_strategy])))


        # split up by strategy
        dict_inputs_by_strat = {}
        df_input_merge = df_input_merge.groupby([field_key_strategy])

        global dfi
        global dfb
        #dfb = df_variable_information.copy()
        #dfi = df_input_merge
        #dfb = df_base.copy()
            
        for strat, df in df_input_merge:

            strat = strat[0] if isinstance(strat, tuple) else strat

            # keep all rows if baseline--otherwise, only keep those that are defined
            merge_type = "left" if (strat == self.baseline_strategy) else "inner"
            df_merge = (
                df.drop([field_req_variable_trajectory_group], axis = 1)
                if trajgroup_passed_to_base_q & (field_req_variable_trajectory_group in df.columns)
                else df
            )

            df = pd.merge(df_base, df_merge, how = merge_type)

            # clean some integer fields
            try:
                for field in [field_key_strategy, field_time_period]:
                    df[field] = df[field].astype(int)

            except Exception as e:
                vars_missing = sorted(list(set(df[df["value"].isna()][field_req_variable])))
                vars_missing = sf.format_print_list(vars_missing)
                msg = f"Error in `InputTemplate.template_from_inputs()`: variables {vars_missing} not found."
                self._log(
                    msg,
                    type_log = "error"
                )
                raise RuntimeError(msg)

            if len(df) > 0:
                df.drop([field_key_strategy], axis = 1, inplace = True)
                dict_inputs_by_strat.update({strat: df})

        # global dict_is
        # dict_is = dict_inputs_by_strat


        ##  CHECK THE VARIABLE INFORMATION DATA FRAME AND ADD COLUMNS

        fields_var_info = [field_req_variable, field_time_period]
        fields_skip_merge = (
            [field_req_variable_trajectory_group]
            if trajgroup_passed_to_base_q
            else []
        )
        fields_var_info.extend(fields_skip_merge)
        
        df_var_info = df_base[fields_var_info].drop_duplicates()

        # set max/min fields (based on input time periods)
        max_time_period = max(attr_tp.key_values)
        field_max = self.name_field_maxmin_from_index(max_time_period, "max")
        field_min = self.name_field_maxmin_from_index(max_time_period, "min")

        # check validity of df_variable_information
        if isinstance(df_variable_information, pd.DataFrame):

            # get existing ield_max/field_min contained in df_variable_information (if exist)
            field_max_nms = [
                x for x in df_variable_information.columns 
                if (regex_max.match(str(x)) is not None)
            ]
            field_max_nms = field_max_nms[0] if (len(field_max_nms) > 0) else None
            
            field_min_nms = [
                x for x in df_variable_information.columns 
                if (regex_min.match(str(x)) is not None)
            ]
            field_min_nms = field_min_nms[0] if (len(field_min_nms) > 0) else None

            # update extraction fields to pull from df_variable_information
            nms = [
                x for x in df_variable_information.columns 
                if (x in self.list_fields_required_base)
                and (x not in fields_skip_merge)
            ]
            nms += [field_max_nms] if (field_max_nms is not None) else []
            nms += [field_min_nms] if (field_min_nms is not None) else []

            # merge in available fields from df_variable_information
            df_var_info = (
                pd.merge(
                    df_var_info,
                    df_variable_information[nms],
                    how = "left"
                ) 
                if (len(nms) > 0) 
                else df_var_info
            )

            # rename to ensure max/min defined properly
            dict_rnm = {}
            if (field_max_nms in df_var_info.columns): dict_rnm.update({field_max_nms: field_max})
            if (field_min_nms in df_var_info.columns): dict_rnm.update({field_min_nms: field_min})
            df_var_info.rename(columns = dict_rnm, inplace = True)

        # add empty columns if not defined in df_var_info
        for x in self.list_fields_required_base + [field_max, field_min]:
            if x not in df_var_info.columns:
                df_var_info[x] = None if (x in self.list_fields_required_base) else 1

        
        ##  CLEAN SHEETS AND DEFINE ANEW USING SHEET NAME

        # clean data frames to filter out repeat rows in non-baseline strategies
        sheet_names = []
        fields_index = [field_req_subsector, field_req_variable, field_time_period]
        keys_iterate_0 = list(dict_inputs_by_strat.keys())
        keys_iterate = [self.baseline_strategy] if self.baseline_strategy in keys_iterate_0 else []
        keys_iterate += sorted([x for x in keys_iterate_0 if (x != self.baseline_strategy)])

        for strat in keys_iterate:
            #
            sheet_name = self.name_sheet_from_index(strat)
            sheet_names.append(sheet_name)
            df_cur = dict_inputs_by_strat.get(strat)
            df_cur_out = df_cur
            df_var_info_out = df_var_info

            # check if baseline?
            is_not_baseline = (strat != self.baseline_strategy)
            if is_not_baseline:
                df_cur = sf.filter_df_on_reference_df_rows(
                    df_cur,
                    dict_inputs_by_strat.get(self.baseline_strategy),
                    fields_index,
                    [field_melted_value],
                    fields_groupby = [field_req_variable]
                )

            if df_cur is None: continue

            # add in variable info and pivot to wide by time period
            df_cur = pd.merge(
                df_cur, 
                df_var_info, 
                how = "left", 
            )

            df_cur = sf.pivot_df_clean(
                df_cur,
                [field_time_period],
                [field_melted_value],
            )
            
            dfb = df_cur.copy()
            

            ##  CHECK SCALARS

            # verify the input template to get the fields we need to fill
            tup = self.verify_input_template_sheet(df_cur, base_strategy_q = (not is_not_baseline), )
            if tup is None:
                raise RuntimeError("Unable to verify shape of input template")

            # dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp = tup
            _, _, field_min, field_max, _ = tup

            # check scalars
            shp = df_cur.shape
            if df_cur.dropna(subset = [field_min, field_max]).shape != shp:
                if not fill_missing_scalars_with_defaults:
                    raise MissingScalarsError(f"One or more scalars in {field_min}, {field_max} are missing: check the exogenous specification.")

                # otherwise, fill
                df_cur[field_max] = df_cur[field_max].fillna(self.default_max_scalar)
                df_cur[field_min] = df_cur[field_min].fillna(self.default_min_scalar)


            dict_inputs_by_strat.update({sheet_name: df_cur})


        dict_inputs_by_strat = dict(
            (k, v) for k, v in dict_inputs_by_strat.items() if k in sheet_names
        )

        return dict_inputs_by_strat



    def verify_and_return_sheet_time_periods(self,
        df_in: pd.DataFrame,
        regex_max: Union[re.Pattern, None] = None,
        regex_min: Union[re.Pattern, None] = None,
        regex_tp: Union[re.Pattern, None] = None
    ) -> tuple:
        """Get time periods in a sheet in addition to min/max specification 
            fields.

        Returns
        -------
        Returns a 5-tuple in the following order:

            (
                dict_field_tp_to_tp, 
                df_in, 
                field_min, 
                field_max, 
                fields_tp,
            )

            where

            dict_field_tp_to_tp:    Dictionary mapping time period field to 
                                        associated time period
            df_in:                  cleaned DataFrame that excludes invalid time 
                                        periods
            field_min:              field that stores the minimum scalar for the 
                                        final time period in the template
            field_max:              field that stores the maximum scalar for the 
                                        final time period in the template
            fields_tp:              fields denoting time periods


        Function Arguments
        ------------------
        df_in : pd.DataFrame
            Input data frame storing template values


        Keyword Arguments
        -----------------
        regex_max : Union[re.Pattern, None]
            re.Pattern (compiled regular expression) used to match the field 
            storing the maximum scalar values at the final time period
        regex_min : Union[re.Pattern, None]
            re.Pattern used to match the field storing the minimum scalar values 
            at the final time period
        regex_tp : Union[re.Pattern, None]
            re.Pattern used to match the field storing data values for each time 
            period
        """
        # initialize some variables
        regex_max = self.regex_template_max if not isinstance(regex_max, re.Pattern) else regex_max
        regex_min = self.regex_template_min if not isinstance(regex_min, re.Pattern) else regex_min
        regex_tp = self.regex_template_time_period if not isinstance(regex_tp, re.Pattern) else regex_tp


        ##  GET MIN/MAX AT FINAL TIME PERIOD

        # determine max field/time period
        field_max = [regex_max.match(str(x)) for x in df_in.columns if (regex_max.match(str(x)) is not None)]

        if len(field_max) == 0:
            raise KeyError("No field associated with a maximum scalar value found in data frame.")
        elif len(field_max) > 1:
            fpl = sf.format_print_list(field_max)
            raise KeyError(f"Multiple maximum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
        else:
            field_max = field_max[0]
            tp_max = int(field_max.groups()[0])
            field_max = field_max.string

        # determine min field/time period
        field_min = [regex_min.match(str(x)) for x in df_in.columns if (regex_min.match(str(x)) is not None)]
        if len(field_min) == 0:
            raise KeyError("No field associated with a minimum scalar value found in data frame.")
        elif len(field_min) > 1:
            fpl = sf.format_print_list(field_min)
            raise KeyError(f"Multiple minimum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
        else:
            field_min = field_min[0]
            tp_min = int(field_min.groups()[0])
            field_min = field_min.string

        # check that min/max specify final time period
        if (tp_min != tp_max):
            raise ValueError(f"Fields '{field_min}' and '{field_max}' imply asymmetric final time periods.")


        ##  GET TIME PERIODS

        # get initial information on time periods and rename the dataframe to ensure fields are strings
        fields_tp = [regex_tp.match(str(x)) for x in df_in.columns if (regex_tp.match(str(x)) is not None)]
        dict_rnm = dict([(x, str(x)) for x in df_in.columns if not isinstance(x, str)])
        df_in.rename(columns = dict_rnm, inplace = True, )

        dict_field_tp_to_tp = dict([(x.string, int(x.groups()[0])) for x in fields_tp])

        # check fields for definition in attribute_time_period
        attr_tp = self.model_attributes.get_dimensional_attribute_table(
            self.model_attributes.dim_time_period
        )

        # fields to keep/drop
        fields_valid = [x.string for x in fields_tp if (dict_field_tp_to_tp.get(x.string) in attr_tp.key_values)]
        fields_invalid = [x.string for x in fields_tp if (x.string not in fields_valid)]
        defined_tp = [dict_field_tp_to_tp.get(x) for x in fields_valid]

        if (tp_max not in defined_tp):
            msg = f"""Error trying to define template: the final time period {tp_max} 
            defined in the input template does not exist in the {self.model_attributes.dim_time_period} 
            attribute table at '{attr_tp.fp_table}'
            """
            raise ValueError(msg)

        if len(fields_invalid) > 0:
            flds_drop = sf.format_print_list([x for x in fields_invalid])
            self._log(
                f"Dropping fields {flds_drop} from input template: the time periods are not defined in the {self.model_attributes.dim_time_period} attribute table at '{attr_tp.fp_table}'", 
                type_log = "warning"
            )
            df_in.drop(fields_invalid, axis = 1, inplace = True)

        out = (dict_field_tp_to_tp, df_in, field_min, field_max, fields_valid, )

        return out
    


    def verify_input_template_sheet(self,
        df_template_sheet: pd.DataFrame,
        base_strategy_q: bool = False,
        sheet_name: str = None
    ) -> Union[tuple, None]:
        """Verify the formatting of an input template sheet and retrieve 
            information to verify all strategies

        Returns
        -------
        Returns a 5-tuple in the following order:

            (
                dict_field_tp_to_tp, 
                df_in, 
                field_min, 
                field_max, 
                fields_tp,
            )

            where

            dict_field_tp_to_tp:    Dictionary mapping time period field to 
                                        associated time period
            df_in:                  cleaned DataFrame that excludes invalid time 
                                        periods
            field_min:              field that stores the minimum scalar for the 
                                        final time period in the template
            field_max:              field that stores the maximum scalar for the 
                                        final time period in the template
            fields_tp:              fields denoting time periods

        *NOTE*: returns None if errors occur trying to load, so return values 
            should not be assigned as tuple elements


        Function Arguments
        ------------------
        df_template_sheet : pd.DataFrame
            A data frame representing the input template sheet by strategy

        Keyword Arguments
        -----------------
        base_strategy_q : bool
            Running the base strategy? If so, requirements for input variables 
            are different.
        sheet_name : str
            Name of the sheet passed for error handling and troubleshooting
        """

        # check fields and retrieve information about time periods
        try:

            sf.check_fields(df_template_sheet, self.list_fields_required_base)
            (
                dict_field_tp_to_tp,
                df_template_sheet,
                field_min,
                field_max,
                fields_tp
            ) = self.verify_and_return_sheet_time_periods(df_template_sheet)

        except Exception as e:
            sheet_str = f" '{sheet_name}'" if (sheet_name is not None) else ""
            self._log(
                f"Trying to verify sheet{sheet_str} produced the following error in verify_input_template_sheet:\n\t{e}\nReturning None", 
                type_log = "warning"
            )

            return None

        out = (
            dict_field_tp_to_tp, 
            df_template_sheet, 
            field_min, 
            field_max, 
            fields_tp
        )

        return out









class BaseInputDatabase:
    """The BaseInputDatabase class is used to combine InputTemplates from 
        multiple sectors into a single input for

    Initialization Arguments
    ------------------------
    - fp_templates: file path to directory containing input Excel templates
    - model_attributes: ModelAttributes object used to define sectors and check 
        templates
    - regions: regions to include
        * If None, then try to initialize all input regions

    Optional Arguments
    --------=---------
    - demo_q: whether or not the database is run as a demo
        * If run as demo, then `fp_templates` does not need to include 
            subdirectories for each region specified
    - sectors: sectors to include
        * If None, then try to initialize all input sectors
    - attribute_strategy: strategy attribute used to filter out invalid or
        undefined strategies

    Keyword Arguments
    -----------------
    The following keyword arguments are passed to the InputTemplate classes used 
        to Instantiate a BaseInputDatabase
    - field_req_normalize_group: Required field used to specify whether or not 
        to normalize a group (ensures always sums to 1)
    - field_req_subsector: Required field used to define the subsector 
        associated with a variable
    - field_req_trajgroup_no_vary_q: Required field used to determine whether or 
        not a trajectory group may vary
        * Note: all values in the same trajectory group must be the same
    - field_req_uniform_scaling_q: Required field used to determine whether or 
        not a variable trjaectory should be scaled uniformly over all
        time periods
        * E.g., many biophysical parameters may be uncertain but not change
            over time
    - field_req_variable: Required field used name the variable
        * Trajectory groups require special naming convention used to define
            all parts:
            (INFO HERE)
    - field_req_variable_trajectory_group: Field used to explicitly add
        trajectory group (added after import)
    - field_req_variable_trajectory_group_trajectory_type: Field used to
        explicitly add trajectory group type
        for variables in a trajectory group (added after import)
    - filter_invalid_strategies: filter strategies that are not defined in
        the attribute_strategy input table
    - logger: optional logging object to pass
    - **kwargs: passed to self.get_template_path()
    """
    def __init__(self,
        fp_templates: str,
        model_attributes: ma.ModelAttributes,
        regions: Union[list, None],
        attribute_strategy: Union[AttributeTable, str, None] = None,
        demo_q: bool = True,
        filter_invalid_strategies: bool = True,
        logger: Union[logging.Logger, None] = None,
        sectors: Union[list, None] = None,
        **kwargs,
    ):
        self.demo_q = demo_q
        self.fp_templates = fp_templates
        self.model_attributes = model_attributes

        self._initialize_fields(**kwargs, )
        
        # additional initialization
        self.attribute_strategy = attribute_strategy
        self.filter_invalid_strategies = filter_invalid_strategies
        self.logger = logger

        self.regions = self.get_regions(regions)
        self.sectors = self.get_sectors(sectors)
        self.database = self.generate_database(
            **kwargs,
        )

        return None




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
    def _initialize_fields(self,
        **kwargs,
    ) -> None:
        """
        Initialize fields used in the BaseInputDatabase
        """

        field_req_normalize_group = kwargs.get(
            "field_req_normalize_group", 
            "normalize_group", 
        )
        field_req_subsector = kwargs.get(
            "field_req_subsector",
            "subsector",
        )
        field_req_trajgroup_no_vary_q = kwargs.get(
            "field_req_trajgroup_no_vary_q",
            "trajgroup_no_vary_q",
        )
        field_req_uniform_scaling_q = kwargs.get(
            "field_req_uniform_scaling_q",
            "uniform_scaling_q",
        )
        field_req_variable = kwargs.get(
            "field_req_variable",
            "variable",
        )
        field_req_variable_trajectory_group = kwargs.get(
            "field_req_variable_trajectory_group",
            "variable_trajectory_group",
        )
        field_req_variable_trajectory_group_trajectory_type = kwargs.get(
            "field_req_variable_trajectory_group_trajectory_type",
            "variable_trajectory_group_trajectory_type",
        )


        ##  SET PROPERTIES

        self.field_req_normalize_group = field_req_normalize_group
        self.field_req_subsector = field_req_subsector
        self.field_req_trajgroup_no_vary_q = field_req_trajgroup_no_vary_q
        self.field_req_uniform_scaling_q = field_req_uniform_scaling_q
        self.field_req_variable = field_req_variable
        self.field_req_variable_trajectory_group = field_req_variable_trajectory_group
        self.field_req_variable_trajectory_group_trajectory_type = field_req_variable_trajectory_group_trajectory_type

        return None



    def get_regions(self,
        regions: Union[str,  None],
    ) -> list:
        """
        Import regions for the BaseInputDatabase class from BaseInputDatabase
        """
        attr_region = (
            self
            .model_attributes
            .get_other_attribute_table(
                self.model_attributes.dim_region
            )
        )

        if regions is None:
            regions_out = attr_region.key_values

        else:
            regions_out = [self.model_attributes.clean_region(region) for region in regions]
            regions_out = [region for region in regions_out if region in attr_region.key_values]

        if self.demo_q and len(regions_out) > 0:
            regions_out = [regions_out[0]]

        return regions_out



    def get_sectors(self, 
        sectors: Union[str, None],
    ) -> list:
        """
        Import regions for the BaseInputDatabase class from BaseInputDatabase
        """
        attr_sector = self.model_attributes.get_sector_attribute_table()
        key_dict = f"{attr_sector.key}_to_sector"
        dict_conv = attr_sector.field_maps.get(key_dict)
        all_sectors = [dict_conv.get(x) for x in attr_sector.key_values]

        sectors_out = (
            all_sectors
            if sectors is None
            else [sector for sector in sectors if sector in all_sectors]
        )

        return sectors_out



    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. 
            See ?sf._optional_log for more information

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



    ############################
    #	CORE FUNCTIONALITY	#
    ############################

    def generate_database(self,
        regions: Union[list, None] = None,
        sectors: Union[list, None] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, None]:
        """
        Load templates and generate a base input database.
            * Returns None if no valid templates are found.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        regions: List of regions to load. If None, load 
            BaseInputDatabase.regions.
        sectors: List of sectors to load. If None, load 
            BaseInputDatabase.sectors
        **kwargs: passed to BaseInputDatabase.get_template_path()
        """
        # initialize output
        all_fields = None
        df_out = []
        regions = self.regions if (regions is None) else self.get_regions(regions)
        sectors = self.sectors if (sectors is None) else self.get_regions(sectors)

        for i, region in enumerate(regions):

            df_out_region = []

            for j, sector in enumerate(sectors):

                # read the input database for the sector
                try:
                    
                    path = self.get_template_path(
                        region,
                        sector,
                        **kwargs
                    )
                    
                    template_cur = InputTemplate(
                        path,
                        self.model_attributes,
                        attribute_strategy = self.attribute_strategy,
                        field_req_normalize_group = self.field_req_normalize_group,
                        field_req_subsector = self.field_req_subsector,
                        field_req_trajgroup_no_vary_q = self.field_req_trajgroup_no_vary_q,
                        field_req_uniform_scaling_q = self.field_req_uniform_scaling_q,
                        field_req_variable = self.field_req_variable,
                        field_req_variable_trajectory_group = self.field_req_variable_trajectory_group,
                        field_req_variable_trajectory_group_trajectory_type = self.field_req_variable_trajectory_group_trajectory_type,
                        filter_invalid_strategies = self.filter_invalid_strategies,
                        logger = self.logger
                    )

                    # update strategy attribute (will be same for all InputTemplates)
                    self.attribute_strategy = (
                        template_cur.attribute_strategy 
                        if (self.attribute_strategy is None) 
                        else self.attribute_strategy
                    )
                    self.baseline_strategy = template_cur.baseline_strategy

                    df_template_db = template_cur.build_inputs_by_strategy()

                except Exception as e:

                    msg = f"Warning in generate_database--template read for sector '{sector}' in region '{region}' failed. The following error was returned: {e}"

                    self._log(
                        msg, 
                        type_log = "warning"
                    )
                    df_template_db = None

                if df_template_db is not None:

                    # check time period fields
                    set_template_cols = set(df_template_db.columns)
                    if all_fields is not None:
                        if not set(df_template_db.columns).issubset(all_fields):
                            self._log(
                                f"Error in sector '{sector}', region '{region}': encountered inconsistent definition of template fields. Dropping...", 
                                type_log = "warning"
                            )
                            df_template_db = None

                        else:
                            fields_drop = list(set_template_cols - all_fields)
                            df_template_db.drop(fields_drop, axis = 1, inplace = True) if (len(fields_drop) > 0) else None
                    else:
                        all_fields = set_template_cols

                # update dataframe list
                if (len(df_out_region) == 0) and (df_template_db is not None):
                    df_out_region = [df_template_db for x in range(len(self.sectors))]

                elif len(df_out_region) > 0:
                    df_out_region[j] = df_template_db

            # add region
            df_out_region = (
                (
                    pd.concat(
                        df_out_region,
                        axis = 0
                    )
                    .reset_index(drop = True) 
                )
                if (len(df_out_region) > 0) 
                else None
            )
                

            df_out_region = self.model_attributes.add_index_fields(
                df_out_region,
                region = region
            )

            # add to outer df
            if (len(df_out) == 0) and (df_out_region is not None):
                df_out = [df_out_region for x in range(len(self.regions))]
            elif len(df_out) > 0:
                df_out[i] = df_out_region

        df_out = (
            (
                pd.concat(df_out, axis = 0)
                .reset_index(drop = True) 
            )
            if (len(df_out) > 0) 
            else None
        )

        return df_out



    def get_template_path(self,
        region: Union[str, None],
        sector: str,
        append_base_directory: bool = True,
        create_export_dir: bool = False,
        demo_q: Union[bool, None] = None,
        fp_templates: Union[str, None] = None,
        template_base_str: str = "model_input_variables",
        **kwargs,
    ) -> str:
        """
        Generate a path for an input template based on a sector, region, a 
            database regime type, and a dictionary mapping different database 
            regime types to input directories storing the input Excel templates.

        Function Arguments
        ------------------
        - region: three-character region code
        - sector: the emissions sector (e.g., AFOLU, Circular Economy, etc.

        Keyword Arguments
        -----------------
        - append_base_directory: append the base directory name 
            (basename = os.path.basename(self.fp_templates)) to the template? 
            Default is true to reduce ambiguity.
            * if True, templates take form 
                `model_input_variables_{region}_{abv_sector}_{basename}.xlsx`
            * if False, templates take form 
                `model_input_variables_{region}_{abv_sector}.xlsx`
        - create_export_dir: boolean indicating whether or not to create a 
            directory specified in dict_valid_types if it does not exist.
        - demo_q: initialize as demo? If None, defaults to self.demo_q
        - fp_templates: optional specificaiton of a path to templates. If None,
            defaults to self.fp_templates
        - template_base_str: baseline string for naming templates
        """
        
        ##  INITIALIZATION AND CHECKS

        attr_region = self.model_attributes.get_other_attribute_table(self.model_attributes.dim_region)

        # check sector
        if sector not in self.model_attributes.all_sectors:
            valid_sectors = sf.format_print_list(self.model_attributes.all_sectors)
            raise ValueError(f"Invalid sector '{sector}' specified: valid sectors are {valid_sectors}.")

        abv_sector = self.model_attributes.get_sector_attribute(sector, "abbreviation_sector")

        # initialize some parameters
        demo_q = (
            self.demo_q 
            if not isinstance(demo_q, bool) 
            else demo_q
        )

        fp_templates = (
            self.fp_templates 
            if not isinstance(fp_templates, str) 
            else fp_templates
        )


        # check region
        if not demo_q:

            # check
            if region is None:
                msg = f"Invalid specification of region: a region must be specified unless the database is initialized in demo mode."
                raise ValueError(msg)

            region_lower = self.model_attributes.clean_region(region)
            region_str = f"_{region_lower}"

            # check region and create export directory if necessary
            if region_lower not in attr_region.key_values:
                valid_regions = sf.format_print_list(attr_region.key_values)
                msg = f"Invalid region '{region}' specified: valid regions are {valid_regions}."
                raise ValueError(msg)

            # 
            dir_exp = sf.check_path(
                os.path.join(
                    fp_templates, 
                    region_lower
                ), 
                create_q = create_export_dir,
            )
            
            
        else:
            region_str = ""
            dir_exp = self.fp_templates

        # check appendage
        if append_base_directory:
            append_str = os.path.basename(fp_templates)
            append_str = f"_{append_str}"

        else:
            append_str = ""


        fn_out = f"{template_base_str}{region_str}_{abv_sector}{append_str}.xlsx"

        return os.path.join(dir_exp, fn_out)
