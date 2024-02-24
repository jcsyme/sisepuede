from attribute_table import *
from configuration import *
import itertools
import model_variable as mv
import numpy as np
import os, os.path
import pandas as pd
import pathlib
import re
import support_functions as sf
from typing import *
import units_manager as um
import warnings





class ModelAttributesNew:
    """
    Create a centralized object for managing inter-sectoral objects, dimensions,
        attributes, and variables.

    
    """
    def __init__(self,
        dir_attributes: Union[pathlib.Path, str],
        fp_config: str = None,
    ) -> None:

        ############################################
        #    INITIALIZE SHARED CLASS PROPERTIES    #
        ############################################

        # initialize "basic" properties--properties that are explicitly set in each initialization function
        self._initialize_basic_dimensions_of_analysis()
        self._initialize_basic_other_properties()
        self._initialize_basic_subsector_names()
        self._initialize_basic_table_names_nemomod()
        self._initialize_basic_template_substrings()
        self._initialize_basic_varchar_components()

        # initialize some properties and elements (ordered)
        self._initialize_attribute_tables(dir_attributes)
        #self._initialize_dims()
        self._initialize_other_attributes()
        self._initialize_units()
        self._initialize_variables()

        self._initialize_config(fp_config)
        self._initialize_sector_sets()
        self._initialize_variables_by_subsector()
        #self._initialize_all_primary_category_flags()
        #self._initialize_emission_modvars_by_gas()
        #self._initialize_gas_attributes()
        #self._initialize_other_dictionaries()

        #self._check_attribute_tables()

        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _check_attribute_tables(self,
    ) -> None:
        """
        Set some required attribute fields and check the attribute tables.
            Sets the following properties:
        """

        # run checks and raise errors if invalid data are found in the attribute tables
        self._check_attribute_tables_abv_subsector()
        self._check_attribute_tables_agrc()
        self._check_attribute_tables_enfu()
        self._check_attribute_tables_enst()
        self._check_attribute_tables_entc()
        self._check_attribute_tables_inen()
        self._check_attribute_tables_ippu()
        self._check_attribute_tables_lndu()
        self._check_attribute_tables_lsmm()
        self._check_attribute_tables_trde()
        self._check_attribute_tables_trns()
        self._check_attribute_tables_wali()
        self._check_attribute_tables_waso()

        return None
    


    # HEREHERE: MOVE DOWN LATER
    def iat_support_get_attribute_group(self,
        fn_attribute: str,
        attribute_groups_ordered: List[str],
        dict_attribute_group_to_regex: Dict[str, re.Pattern],
    ) -> Union[str, None]:
        """
        Get the attribute group associated with fn_attribute

        Function Arguments
        ------------------
        - fn_attribute: attribute file name
        - attribute_groups_ordered: ordered search list of attribtue 
            groups
        - dict_attribute_group_to_regex: dictionary mapping attribute
            groups to regular expressions

        Keyword Arguments
        -----------------
        """
        # get group 
        keep_going = True
        i = 0

        while keep_going & (i < len(attribute_groups_ordered)):

            group = attribute_groups_ordered[i]
            regex = dict_attribute_group_to_regex.get(group)

            if regex.match(fn_attribute) is not None:
                keep_going = False
            
            i += 1

        out = group if not keep_going else None

        return out
    


    def _iat_support_clean_subsector_table(self,
        dict_to_update: Dict[str, Dict[str, AttributeTable]],
        table_name_attr_subsector: str,
        field_category: str,
        field_category_py: str,
        key_cat: str = "cat",
        key_other: str = "other",
    ) -> None:
        """
        Update the subsector attribute table to include clean categories

        Function Arguments
        ------------------
        - dict_to_update: dictionary containing attribute groups that map to 
            individual dictionaries
        - table_name_attr_subsector: attribute name for the subsector table
        - field_category_py: new field to add to the attribute table

        Keyword Arguments
        -----------------
        - key_cat: attribute group key for categories
        - key_other: attribute group key containing subsector attribute
        """

        dict_attr = dict_to_update.get(key_other)
        if dict_attr is None:
            return None

        # add a new field
        df_tmp = dict_attr[table_name_attr_subsector].table

        vec = sf.clean_field_names(df_tmp[field_category])
        df_tmp[field_category_py] = [x.replace(f"{key_cat}_", "") for x in vec]

        df_tmp = (
            df_tmp[
                df_tmp[field_category_py] != "none"
            ]
            .reset_index(drop = True)
        )

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

        dict_attr[table_name_attr_subsector].field_maps.update(field_maps)

        return None
    


    def iat_support_get_all_keys(self,
        dict_attributes: Dict[str, Dict[str, AttributeTable]],
    ) -> None:
        """
        Get all values associated with each key

        Function Arguments
        ------------------
        - dict_attributes: dictionary of items to return all keys for
    
        Keyword Arguments
        -----------------
        """
        dict_out = {}

        for k, v in dict_attributes.items():
            if not isinstance(v, dict):
                continue

            all_items = sorted(list(v.keys()))
            dict_out.update({k: all_items})

        return dict_out
    


    def _iat_support_update_grouped_attributes(self,
        fp_attribute: str,
        attribute_groups_ordered: List[str],
        dict_attribute_group_to_regex: Dict[str, re.Pattern],
        dict_to_update: Dict[str, Dict[str, AttributeTable]],
        key_cat: str,
        key_dim: str,
        key_other: str,
        key_unit: str,
    ) -> None:
        """
        Format the attribute tables for assignment in dictionaries

        Function Arguments
        ------------------
        - fp_attribute: attribute file path
        - attribute_groups_ordered: ordered search list of attribtue 
            groups
        - dict_attribute_group_to_regex: dictionary mapping attribute
            groups to regular expressions
        - dict_to_update: dictionary to update. Keys are groups and 
            values are dictionaries mapping keys to attribute tables
        - key_cat: attribute group key for categories
        - key_dim: attribute group key for dimensions
        - key_other: attribute group key for others (not assigned elsewhere)
        - key_unit: attribute group key for units
    
        Keyword Arguments
        -----------------
        """
        ##  INITIALIZATION

        # get group and regular expression; return None if not found

        fn = os.path.basename(fp_attribute)

        group = self.iat_support_get_attribute_group(
            fn,
            attribute_groups_ordered,
            dict_attribute_group_to_regex,
        )

        regex = (
            dict_attribute_group_to_regex.get(group)
            if (group is not None)
            else None
        )

        if regex is None:
            return None


        ##  CHECK GROUP MEMBERSHIP AND UPDATE DICTIONARY IF NECESSARY

        if group not in dict_to_update.keys():
            dict_to_update.update({group: {}})

        if group == key_dim:

            nm = regex.match(fn).groups()[0]
            nm = sf.clean_field_names(nm)
            att_table = AttributeTable(fp_attribute, nm)
            dict_to_update[group].update({nm: att_table})


        elif group in [key_cat, key_other, key_unit]:
            #
            df_cols = pd.read_csv(fp_attribute, nrows = 0).columns 
            
            # try to set key
            nm = sf.clean_field_names([x for x in df_cols if ("$" in x) and (" " not in x.strip())])
            nm = nm[0] if (len(nm) > 0) else None
            if nm is None:
                nm = regex.match(fn).groups()[0]
                nm = nm if (nm in df_cols) else None

            # skip if impossible
            if nm is None:
                return None

            att_table = AttributeTable(fp_attribute, nm)
            key = nm.replace(f"{group}_", "")
            dict_to_update[group].update({key: att_table})


        else:
            msg = f"""
            Invalid attribute '{att}': No attribute group or expression 
            was found to support its addition.
            """
            raise ValueError(msg)
        
        
        return None



    def _initialize_all_primary_category_flags(self,
    ) -> None:
        """
        Sets all primary category flags, e.g., $CAT-CCSQ$ or $CAT-AGRICULTURE$.
            Sets the following properties:

            * all_primary_category_flags
        """
        attr_subsec = self.dict_attributes.get(self.table_name_attr_subsector)
        all_pcflags = None

        if attr_subsec is not None:
            all_pcflags = sorted(list(set(attr_subsec.table["primary_category"])))
            all_pcflags = [x.replace("`", "") for x in all_pcflags if sf.clean_field_names([x])[0] in self.all_pycategories]

        self.all_primary_category_flags = all_pcflags

        return None


            
    def _initialize_attribute_tables(self,
        dir_att: str,
        attribute_group_protected_other: str = "other",
        attribute_groups: Union[List[str], None] = None,
        stop_on_error: bool = True,
        table_name_attr_sector: str = "abbreviation_sector",
        table_name_attr_subsector: str = "abbreviation_subsector",
        variable_definition_key: str = "variable_definitions",
    ) -> None:
        """
        Load all attribute tables and set the following parameters:

            * self.all_attributes
            * self.all_dims
            * self.all_pycategories
            * self.attribute_analytical_parameters
            * self.attribute_directory
            * self.attribute_experimental_parameters
            * self.attribute_group_key_cat
            * self.attribute_group_key_dim
            * self.attribute_group_key_other
            * self.attribute_group_key_unit
            * self.dict_attributes
            * self.dict_variable_definitions
            * self.regex_attribute_*:
                regular expressions used to read in attribute tables 
            * self.subsector_field_category_py:
                field in subsector attribute table used to identify the 
                python name for a category
            * self.table_name_attr_sector
            * self.table_name_attr_subsector

        Function Arguments
        ------------------
        - dir_att: directory containing attribute tables

        Keyword Arguments
        -----------------
        - attribute_group_protected_other: protected group used to identify 
            attribute tables that don't fit elsewhere.
        - attribute_groups: optional list of attribute table types to specify;
            types will be identified as "attribute_YYY_MATCH" for YYY in 
            attribute_groups; all others that do not match will be defined as 
            other. If None, defaults to ["cat", "dim", "unit"]
            NOTE: cannot contain `attribute_group_protected_other`
        - stop_on_error: stop if there's an error?
        - table_name_attr_sector: table name used to assign sector table
        - table_name_attr_subsector: table name used to assign subsector table
        - variable_definition_key: prependage used to identify variable
            definition tables
        """

        self.attribute_directory = sf.check_path(dir_att, False)


        ##  INITIALIZATION

        # initialize attribute groups, which are used to group attributes; 
        #   Use `attribute_groups_ordered` to ensure `other` tables are identified last
        key_cat = "cat"
        key_dim = "dim"
        key_unit = "unit"

        attribute_groups = (
            [key_cat, key_dim, key_unit] 
            if not sf.islistlike(attribute_groups) 
            else list(attribute_groups)
        )
        attribute_groups_ordered = sorted(attribute_groups) + [attribute_group_protected_other]

        # build dictionary mappinug groups to regular expression
        dict_attribute_group_to_regex = dict(
            (x, re.compile(f"attribute_{x}_(.*).csv"))
            for x in attribute_groups if (x != "other")
        )
        dict_attribute_group_to_regex.update(
            {attribute_group_protected_other: re.compile(f"attribute_(.*).csv")}
        )
        
        regex_vardef = re.compile(f"{variable_definition_key}_(.*).csv")


        # get available types
        all_types = [
            x for x in os.listdir(dir_att) 
            if (self.attribute_file_extension in x) and (
                any([(v.match(x) is not None) for v in dict_attribute_group_to_regex.values()])
                or (regex_vardef.match(x) is not None)
                or (self.substr_analytical_parameters in x)
                or (self.substr_experimental_parameters in x)
            )
        ]


        ##  LOAD AttributeTables IN BATCH

        dict_attributes = dict((x, {}) for x in dict_attribute_group_to_regex.keys())
        dict_variable_definitions = {}
        attribute_analytical_parameters = None
        attribute_experimental_parameters = None

        for att in all_types:
            fp = os.path.join(dir_att, att)

            # check variable specification first
            if regex_vardef.match(att) is not None:
                nm = regex_vardef.match(att).groups()[0]
                dict_variable_definitions.update({nm: AttributeTable(fp, "variable")})
                continue

            elif (self.substr_analytical_parameters in att):
                attribute_analytical_parameters = AttributeTable(fp, "analytical_parameter")
                continue

            elif (self.substr_experimental_parameters in att):
                attribute_experimental_parameters = AttributeTable(fp, "experimental_parameter")
                continue


            ##  CASE WHERE ATTRIBUTE TABLES BELONG TO ONE OF THE GROUPS

            try:
               self._iat_support_update_grouped_attributes(
                    fp,
                    attribute_groups_ordered,
                    dict_attribute_group_to_regex,
                    dict_attributes,
                    key_cat,
                    key_dim,
                    attribute_group_protected_other,
                    key_unit,
                )

            except Exception as e:
                msg = f"Error trying to initialize attribute {att}: {e}"
                # self._log(msg, type_log = "error") # ADD LOGGER!
                if stop_on_error:
                    raise RuntimeError(msg)
            

        # add some subsector/python specific information into the subsector table
        field_category = "primary_category"
        field_category_py = field_category + "_py"

        # check sector and subsector specifications
        set_sector_tables = set({table_name_attr_sector, table_name_attr_subsector})
        set_avail_others = set(dict_attributes.get("other").keys())
        if not set_sector_tables.issubset(set_avail_others):
            missing_vals = sf.print_setdiff(set_sector_tables, set_avail_others)
            raise RuntimeError(f"Error initializing attribute tables: table names {missing_vals} not found.")


        ##  UPDATE THE SUBSECTOR ATTRIBUTE TABLE

        self._iat_support_clean_subsector_table(
            dict_attributes,
            table_name_attr_subsector,
            field_category,
            field_category_py,
            key_cat = key_cat,
            key_other = attribute_group_protected_other,
        )

        # get sets of all available values
        dict_all_attribute_values = self.iat_support_get_all_keys(dict_attributes)
        all_dims = dict_all_attribute_values.get(key_dim)
        all_pycategories = dict_all_attribute_values.get(key_cat)
        all_units = dict_all_attribute_values.get(key_unit)
 
   
        ##  SET PROPERTIES

        self.all_dims = all_dims
        self.all_pycategories = all_pycategories
        self.all_units = all_units
        self.all_attributes = all_types
        self.attribute_analytical_parameters = attribute_analytical_parameters
        self.attribute_experimental_parameters = attribute_experimental_parameters
        self.attribute_group_key_cat = key_cat
        self.attribute_group_key_dim = key_dim
        self.attribute_group_key_other = attribute_group_protected_other
        self.attribute_group_key_unit = key_unit
        self.dict_attributes = dict_attributes
        self.dict_variable_definitions = dict_variable_definitions
        self.subsector_field_category_py = field_category_py
        self.table_name_attr_sector = table_name_attr_sector
        self.table_name_attr_subsector = table_name_attr_subsector

        return None



    def _initialize_basic_dimensions_of_analysis(self,
    ) -> None:
        """
        Initialize dimensions of anlaysis. Sets the following properties:

            * self.dim_design_id
            * self.dim_future_id
            * self.dim_mode
            * self.dim_region
            * self.dim_strategy_id
            * self.dim_time_period
            * self.dim_time_series_id
            * self.dim_primary_id
            * self.field_dim_year
            * self.sort_ordered_dimensions_of_analysis

        """
        # initialize dimensions of analysis - later, check for presence
        self.dim_design_id = "design_id"
        self.dim_future_id = "future_id"
        self.dim_mode = "mode"
        self.dim_region = "region"
        self.dim_strategy_id = "strategy_id"
        self.dim_time_period = "time_period"
        self.dim_time_series_id = "time_series_id"
        self.dim_primary_id = "primary_id"

        # setup dtypes
        self.dict_dtypes_doas = {
            self.dim_design_id: "int64",
            self.dim_future_id: "int64",
            self.dim_mode: "string",
            self.dim_region: "string",
            self.dim_strategy_id: "int64",
            self.dim_time_period: "int64",
            self.dim_time_series_id: "int64",
            self.dim_primary_id: "int64",
        }

        # ordered by sort hierarchy
        self.sort_ordered_dimensions_of_analysis = [
            self.dim_primary_id,
            self.dim_design_id,
            self.dim_region,
            self.dim_time_series_id,
            self.dim_strategy_id,
            self.dim_future_id,
            self.dim_time_period
        ]

        # some common shared fields
        self.field_dim_year = "year"

        return None
        


    def _initialize_basic_other_properties(self,
    ) -> None:
        """
        Set some additional properties that are not set in other basic
            initialization functions. Sets the following properties:

            * self.attribute_file_extension
            * self.delim_multicats
            * self.field_emissions_total_flag
            * self.matchstring_landuse_to_forests
        """
        self.attribute_file_extension = ".csv"
        self.delim_multicats = "|"
        self.field_emissions_total_flag = "emissions_total_by_gas_component"
        self.matchstring_landuse_to_forests = "forests_"

        return None



    def _initialize_basic_table_names_nemomod(self,
    ) -> None:
        """
        Initialize table names used in NemoMod. Sets the following properties:

            * self.table_nemomod_annual_emission_limit
            * self.table_nemomod_annual_emissions_by_technology
            * self.table_nemomod_capacity_factor
            * self.table_nemomod_capacity_to_activity_unit
            * self.table_nemomod_capital_cost
            * self.table_nemomod_capital_cost_storage
            * self.table_nemomod_capital_investment
            * self.table_nemomod_capital_investment_discounted
            * self.table_nemomod_capital_investment_storage
            * self.table_nemomod_capital_investment_storage_discounted
            * self.table_nemomod_default_params
            * self.table_nemomod_discount_rate
            * self.table_nemomod_emission
            * self.table_nemomod_emissions_activity_ratio
            * self.table_nemomod_fixed_cost
            * self.table_nemomod_fuel
            * self.table_nemomod_input_activity_ratio
            * self.table_nemomod_min_storage_charge
            * self.table_nemomod_mode_of_operation
            * self.table_nemomod_model_period_emission_limit
            * self.table_nemomod_model_period_exogenous_emission
            * self.table_nemomod_new_capacity
            * self.table_nemomod_node
            * self.table_nemomod_operating_cost
            * self.table_nemomod_operating_cost_discounted
            * self.table_nemomod_operational_life
            * self.table_nemomod_operational_life_storage
            * self.table_nemomod_output_activity_ratio
            * self.table_nemomod_production_by_technology
            * self.table_nemomod_re_tag_technology
            * self.table_nemomod_region
            * self.table_nemomod_reserve_margin
            * self.table_nemomod_reserve_margin_tag_fuel
            * self.table_nemomod_reserve_margin_tag_technology
            * self.table_nemomod_residual_capacity
            * self.table_nemomod_residual_storage_capacity
            * self.table_nemomod_specified_annual_demand
            * self.table_nemomod_specified_annual_demand
            * self.table_nemomod_specified_demand_profile
            * self.table_nemomod_storage
            * self.table_nemomod_storage_level_start
            * self.table_nemomod_technology
            * self.table_nemomod_technology_from_storage
            * self.table_nemomod_technology_to_storage
            * self.table_nemomod_time_slice
            * self.table_nemomod_time_slice_group_assignment
            * self.table_nemomod_total_annual_capacity
            * self.table_nemomod_total_annual_max_capacity
            * self.table_nemomod_total_annual_max_capacity_investment
            * self.table_nemomod_total_annual_max_capacity_investment_storage
            * self.table_nemomod_total_annual_max_capacity_storage
            * self.table_nemomod_total_annual_min_capacity
            * self.table_nemomod_total_annual_min_capacity_investment
            * self.table_nemomod_total_annual_min_capacity_investment_storage
            * self.table_nemomod_total_annual_min_capacity_storage
            * self.table_nemomod_total_technology_annual_activity_lower_limit
            * self.table_nemomod_total_technology_annual_activity_upper_limit
            * self.table_nemomod_ts_group_1
            * self.table_nemomod_ts_group_2
            * self.table_nemomod_use_by_technology
            * self.table_nemomod_variable_cost
            * self.table_nemomod_year
            * self.table_nemomod_year_split
        """
        # nemomod shared tables - dimensions
        self.table_nemomod_emission = "EMISSION"
        self.table_nemomod_fuel = "FUEL"
        self.table_nemomod_mode_of_operation = "MODE_OF_OPERATION"
        self.table_nemomod_node = "NODE"
        self.table_nemomod_region = "REGION"
        self.table_nemomod_storage = "STORAGE"
        self.table_nemomod_technology = "TECHNOLOGY"
        self.table_nemomod_time_slice = "TIMESLICE"
        self.table_nemomod_ts_group_1 = "TSGROUP1"
        self.table_nemomod_ts_group_2 = "TSGROUP2"
        self.table_nemomod_year = "YEAR"
        # nemomod shared tables - parameters
        self.table_nemomod_annual_emission_limit = "AnnualEmissionLimit"
        self.table_nemomod_capacity_factor = "CapacityFactor"
        self.table_nemomod_capacity_to_activity_unit = "CapacityToActivityUnit"
        self.table_nemomod_capital_cost = "CapitalCost"
        self.table_nemomod_capital_cost_storage = "CapitalCostStorage"
        self.table_nemomod_default_params = "DefaultParams"
        self.table_nemomod_discount_rate = "DiscountRate"
        self.table_nemomod_emissions_activity_ratio = "EmissionActivityRatio"
        self.table_nemomod_fixed_cost = "FixedCost"
        self.table_nemomod_input_activity_ratio = "InputActivityRatio"
        self.table_nemomod_min_share_production = "MinShareProduction"
        self.table_nemomod_min_storage_charge = "MinStorageCharge"
        self.table_nemomod_model_period_emission_limit = "ModelPeriodEmissionLimit"
        self.table_nemomod_model_period_exogenous_emission = "ModelPeriodExogenousEmission"
        self.table_nemomod_operational_life = "OperationalLife"
        self.table_nemomod_operational_life_storage = "OperationalLifeStorage"
        self.table_nemomod_output_activity_ratio = "OutputActivityRatio"
        self.table_nemomod_residual_capacity = "ResidualCapacity"
        self.table_nemomod_residual_storage_capacity = "ResidualStorageCapacity"
        self.table_nemomod_re_min_production_target = "REMinProductionTarget"
        self.table_nemomod_re_tag_technology = "RETagTechnology"
        self.table_nemomod_reserve_margin = "ReserveMargin"
        self.table_nemomod_reserve_margin_tag_fuel = "ReserveMarginTagFuel"
        self.table_nemomod_reserve_margin_tag_technology = "ReserveMarginTagTechnology"
        self.table_nemomod_specified_demand_profile = "SpecifiedDemandProfile"
        self.table_nemomod_specified_annual_demand = "SpecifiedAnnualDemand"
        self.table_nemomod_storage_level_start = "StorageLevelStart"
        self.table_nemomod_technology_from_storage = "TechnologyFromStorage"
        self.table_nemomod_technology_to_storage = "TechnologyToStorage"
        self.table_nemomod_time_slice_group_assignment = "LTsGroup"
        self.table_nemomod_total_annual_max_capacity = "TotalAnnualMaxCapacity"
        self.table_nemomod_total_annual_max_capacity_investment = "TotalAnnualMaxCapacityInvestment"
        self.table_nemomod_total_annual_max_capacity_storage = "TotalAnnualMaxCapacityStorage"
        self.table_nemomod_total_annual_max_capacity_investment_storage = "TotalAnnualMaxCapacityInvestmentStorage"
        self.table_nemomod_total_annual_min_capacity = "TotalAnnualMinCapacity"
        self.table_nemomod_total_annual_min_capacity_investment = "TotalAnnualMinCapacityInvestment"
        self.table_nemomod_total_annual_min_capacity_storage = "TotalAnnualMinCapacityStorage"
        self.table_nemomod_total_annual_min_capacity_investment_storage = "TotalAnnualMinCapacityInvestmentStorage"
        self.table_nemomod_total_technology_annual_activity_lower_limit = "TotalTechnologyAnnualActivityLowerLimit"
        self.table_nemomod_total_technology_annual_activity_upper_limit = "TotalTechnologyAnnualActivityUpperLimit"
        self.table_nemomod_specified_annual_demand = "SpecifiedAnnualDemand"
        self.table_nemomod_variable_cost = "VariableCost"
        self.table_nemomod_year_split = "YearSplit"

        # nemomod shared tables - output variables
        self.table_nemomod_annual_demand_nn = "vdemandannualnn"
        self.table_nemomod_annual_emissions_by_technology = "vannualtechnologyemission"
        self.table_nemomod_capital_investment = "vcapitalinvestment"
        self.table_nemomod_capital_investment_discounted = "vdiscountedcapitalinvestment"
        self.table_nemomod_capital_investment_storage = "vcapitalinvestmentstorage"
        self.table_nemomod_capital_investment_storage_discounted = "vdiscountedcapitalinvestmentstorage"
        self.table_nemomod_new_capacity = "vnewcapacity"
        self.table_nemomod_operating_cost = "voperatingcost"
        self.table_nemomod_operating_cost_discounted = "vdiscountedoperatingcost"
        self.table_nemomod_production_by_technology = "vproductionbytechnologyannual"
        self.table_nemomod_total_annual_capacity = "vtotalcapacityannual"
        self.table_nemomod_use_by_technology = "vusebytechnologyannual"

        return None



    def _initialize_basic_subsector_names(self,
    ) -> None:
        """
        Set properties associated with subsector names. Sets the following
            properties:

            * self.subsec_name_agrc
            * self.subsec_name_ccsq
            * self.subsec_name_econ
            * self.subsec_name_enfu
            * self.subsec_name_enst
            * self.subsec_name_entc
            * self.subsec_name_fgtv
            * self.subsec_name_frst
            * self.subsec_name_gnrl
            * self.subsec_name_inen
            * self.subsec_name_ippu
            * self.subsec_name_lndu
            * self.subsec_name_lsmm
            * self.subsec_name_lvst
            * self.subsec_name_scoe
            * self.subsec_name_soil
            * self.subsec_name_trde
            * self.subsec_name_trns
            * self.subsec_name_trww
            * self.subsec_name_wali
            * self.subsec_name_waso
        """
        # set some subsector names
        self.subsec_name_agrc = "Agriculture"
        self.subsec_name_frst = "Forest"
        self.subsec_name_lndu = "Land Use"
        self.subsec_name_lsmm = "Livestock Manure Management"
        self.subsec_name_lvst = "Livestock"
        self.subsec_name_soil = "Soil Management"
        self.subsec_name_wali = "Liquid Waste"
        self.subsec_name_waso = "Solid Waste"
        self.subsec_name_trww = "Wastewater Treatment"
        self.subsec_name_ccsq = "Carbon Capture and Sequestration"
        self.subsec_name_enfu = "Energy Fuels"
        self.subsec_name_enst = "Energy Storage"
        self.subsec_name_entc = "Energy Technology"
        self.subsec_name_fgtv = "Fugitive Emissions"
        self.subsec_name_inen = "Industrial Energy"
        self.subsec_name_scoe = "Stationary Combustion and Other Energy"
        self.subsec_name_trns = "Transportation"
        self.subsec_name_trde = "Transportation Demand"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_econ = "Economy"
        self.subsec_name_gnrl = "General"

        return None



    def _initialize_basic_template_substrings(self,
    ) -> None:
        """
        Set properties related to substrings used to identify input template
            fields for SISEPUEDE. Sets the following properties:

            * self.substr_analytical_parameters
            * self.substr_categories
            * self.substr_dimensions
            * self.substr_experimental_parameters
            * self.substr_varreqs
            * self.substr_varreqs_allcats
            * self.substr_varreqs_partialcats
        """
        self.substr_analytical_parameters = "analytical_parameters"
        self.substr_experimental_parameters = "experimental_parameters"
        self.substr_dimensions = "attribute_dim_"
        self.substr_categories = "attribute_"
        self.substr_varreqs = "table_varreqs_by_"
        self.substr_varreqs_allcats = f"{self.substr_varreqs}category_"
        self.substr_varreqs_partialcats = f"{self.substr_varreqs}partial_category_"

        return None



    def _initialize_basic_varchar_components(self,
    ) -> None:
        """
        Set variable character substrings used in variable schema--e.g.,
            $EMISSION-GAS$--used to substitute in known components. Sets the
            following properties:

            * self.varchar_str_emission_gas
            * self.varchar_str_unit_area
            * self.varchar_str_unit_energy
            * self.varchar_str_unit_length
            * self.varchar_str_unit_mass
            * self.varchar_str_unit_monetary
            * self.varchar_str_unit_power
            * self.varchar_str_unit_volume
        """
        # temporary - but read from table at some point
        self.varchar_str_emission_gas = "$EMISSION-GAS$"
        self.varchar_str_unit_area = "$UNIT-AREA$"
        self.varchar_str_unit_energy = "$UNIT-ENERGY$"
        self.varchar_str_unit_length = "$UNIT-LENGTH$"
        self.varchar_str_unit_mass = "$UNIT-MASS$"
        self.varchar_str_unit_monetary = "$UNIT-MONETARY$"
        self.varchar_str_unit_power = "$UNIT-POWER$"
        self.varchar_str_unit_volume = "$UNIT-VOLUME$"

        return None



    def _initialize_config(self,
        fp_config: str
    ) -> None:
        """
        Initialize the config object. Sets the following parameters:

            * self.attribute_configuration_parameters
            * self.configuration

        Function Arguments
        ------------------
        - fp_config: path to configuration file to read from
        """

        # finally, create the full analytical parameter attribute table - concatenate_attribute_tables from attribute_table
        self.attribute_configuration_parameters = concatenate_attribute_tables(
            "configuration_parameter",
            self.attribute_analytical_parameters,
            self.attribute_experimental_parameters
        )

        # get configuration
        self.configuration = Configuration(
            fp_config,
            self.get_unit_attribute("area"),
            self.get_unit_attribute("energy"),
            self.get_other_attribute("emission_gas").attribute_table, # the emission_gas table is stored as a um.Unit
            self.get_unit_attribute("length"),
            self.get_unit_attribute("mass"),
            self.get_unit_attribute("monetary"),
            self.get_unit_attribute("power"),
            self.get_other_attribute("region"),
            self.get_dimensional_attribute_table(self.dim_time_period),
            self.get_unit_attribute("volume"),
            attr_required_parameters = self.attribute_configuration_parameters,
        )

        return None
    


    def _initialize_dims(self,           
    ) -> None:
        """
        Initialize some dimensional information 
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """
        
            
        return None



    def _initialize_emission_modvars_by_gas(self,
        key_other_totals: str = "multigas",
    ) -> None:
        """
        Get dictionaries that gives all total emission component variables
            by gas. Sets the following properties:

            * self.dict_gas_to_total_emission_fields
            * self.dict_gas_to_total_emission_modvars

        Keyword Arguments
        -----------------
        - key_other_totals: key to use for gasses that are associated with 
            multiple gasses (if applicable)
        """
        # get tables and initialize dictionary out
        all_tabs = self.dict_variable_definitions.keys()
        dict_fields_by_gas = {}
        dict_modvar_by_gas = {}
        for tab in all_tabs:
            tab = self.dict_variable_definitions.get(tab).table
            modvars = list(
                tab[
                    tab[self.field_emissions_total_flag] == 1
                ]["variable"]
            )

            for modvar in modvars:
                # build the variable list
                varlist = self.build_variable_fields(modvar)

                # get emission and add to dictionary
                emission = self.get_variable_characteristic(
                    modvar, 
                    self.varchar_str_emission_gas
                )

                key = emission if (emission is not None) else key_other_totals

                # add to fields by gas
                (
                    dict_fields_by_gas[key].extend(varlist)
                    if key in dict_fields_by_gas.keys()
                    else dict_fields_by_gas.update({key: varlist})
                )
                # add to modvars by gas
                (
                    dict_modvar_by_gas[key].append(modvar)
                    if key in dict_modvar_by_gas.keys()
                    else dict_modvar_by_gas.update({key: [modvar]})
                )

        self.dict_gas_to_total_emission_fields = dict_fields_by_gas
        self.dict_gas_to_total_emission_modvars = dict_modvar_by_gas

        return None
    


    def _initialize_gas_attributes(self,
    ) -> None:
        """
        Initialize some shared gas attribute objects. Sets the following 
            properties:

            * self.dict_fc_designation_to_gas
            * self.dict_gas_to_fc_designation
        """

        dict_fc_designation_to_gas = self.get_fluorinated_compound_dictionaries()
        dict_gas_to_fc_designation = dict(
            sum(
                [
                    [(x, k) for x in v]
                    for k, v in dict_fc_designation_to_gas.items()
                ], []
            )
        )

        self.dict_fc_designation_to_gasses = dict_fc_designation_to_gas
        self.dict_gas_to_fc_designation = dict_gas_to_fc_designation

        return None
    


    def _initialize_other_attributes(self,           
    ) -> None:
        """
        Initialize other attributes defined in the input attribute unit tables. 
            Modifies entries in 

                self.dict_attributes.get(self.attribute_group_key_other)

            Some of these are converted to units (emission_gas)
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """
        
        dict_other = self.dict_attributes.get(self.attribute_group_key_other)
        if dict_other is None:
            return None
        
        # update the entries that should be units
        tables_as_units = ["emission_gas"]

        for k in tables_as_units:

            v = dict_other.get(k)
            if v is None:
                continue

            try:
                unit = um.Units(v)

            except Exception as e:
                msg = f"Error trying to set other attribute {k} as uit: {e}"
                #LOG self._log(msg, type_log = "error")
                continue
            
            dict_other.update({k: unit})
            
        return None
    


    def _initialize_other_dictionaries(self,
    ) -> None:
        """
        Initialize some dictionaries that are dependent on global variable 
            properties (must be initialized AFTER 
            self._initialize_variables_by_subsector()). Sets the following
            properties:

            * self.dict_variable_to_simplex_group
        """

        # get simplex groups
        dict_variable_to_simplex_group = self.get_variable_to_simplex_group_dictionary()
        

        ##  SET PROPERTIES
        
        self.dict_variable_to_simplex_group = dict_variable_to_simplex_group

        return None



    def _initialize_sector_sets(self,
    ) -> None:
        """
        Initialize properties around subsectors. Sets the following properties:

            * self.all_sectors
            * self.all_sectors_abvs
            * self.all_subsectors
            * self.all_subsector_abvs
            * self.all_subsectors_with_primary_category
            * self.all_subsectors_without_primary_category
            * self.emission_subsectors
        """
        attr_sec = self.get_sector_attribute_table()
        attr_subsec = self.get_subsector_attribute_table()

        # all sectors and subsectors + emission subsectors
        all_sectors = sorted(list(attr_sec.table["sector"].unique()))
        all_subsectors = sorted(list(attr_subsec.table["subsector"].unique()))
        emission_subsectors = self.get_emission_subsectors()

        # some subsector splits based on w+w/o primary categories
        l_with = attr_subsec.field_maps.get(
            f"subsector_to_{self.subsector_field_category_py}"
        )
        l_with = sorted(list(l_with.keys())) if isinstance(l_with, dict) else []
        l_without = sorted(list(set(all_subsectors) - set(l_with)))


        ##  SET PROPERTIES

        self.all_sectors = all_sectors
        self.all_sectors_abvs = attr_sec.key_values
        self.all_subsectors = all_subsectors
        self.all_subsector_abvs = attr_subsec.key_values
        self.all_subsectors_with_primary_category = l_with
        self.all_subsectors_without_primary_category = l_without
        self.emission_subsectors = emission_subsectors

        return None
    
    

    def _initialize_units(self,           
    ) -> None:
        """
        Initialize the units defined in the input attribute unit tables. 
            Modifies entries in 

                self.dict_attributes.get(self.attribute_group_key_unit,

            converting these to um.Units objects that invlude conversion 
            mechanisms.

            NOTE: Units can be accessed using the `get_unit()` method of the
            ModelAttributes class.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """

        ##  UPDATE dict_attributes UNIT SUBDICT TO HAVE UNITS AS ELEMENTS

        dict_units = self.dict_attributes.get(self.attribute_group_key_unit)
        if dict_units is None:
            return None
        
        # update the entries in dict_attributes self.attribute_group_key_unit subdict
        key_prependage = f"{self.attribute_group_key_unit}_"

        for k, v in dict_units.items():
            
            try:
                unit = um.Units(v, key_prependage = key_prependage, )

            except Exception as e:
                msg = f"Error trying to set unit {k}: {e}"
                #LOG self._log(msg, type_log = "error")
                continue
            
            dict_units.update({k: unit})
        

        ##  SET SOME OTHER PROPERTIES

        # initialize as available units
        valid_rts_unit_conversion = list(dict_units.keys())
        valid_rts_unit_conversion.append("total")

        # emission gas available?
        keys_other = self.dict_attributes.get(self.attribute_group_key_other).keys()
        if "emission_gas" in keys_other:
            valid_rts_unit_conversion.append("gas")
        valid_rts_unit_conversion.sort()


        ##  SET PROPERTIES

        self.valid_return_types_unit_conversion = valid_rts_unit_conversion

        return None
    


    def _initialize_variables(self,           
    ) -> None:
        """
        Initialize variables as model_variable.ModelVariable objects. Sets the 
            following properties:

                * self.dict_variables

        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """

        dict_variables = self.get_variable_dict()

        # get some derivative values
        all_model_variables = sorted(list(dict_variables.keys()))


        ##  SET PROPERTIES

        self.all_variables = all_model_variables
        self.dict_variables = dict_variables

        return None
            


    def _initialize_variables_by_subsector(self,
    ) -> None:
        """
        Initialize some dictionaries describing variables by subsector.
            Initializes the following properties:

            * self.all_variables
            * self.all_variable_fields
            * self.all_variable_fields_input
            * self.all_variable_fields_output
            * self.dict_model_variables_by_subsector
            * self.dict_model_variable_to_subsector
            * self.dict_model_variable_to_category_restriction
            * self.dict_variable_to_simplex_group
            * self.dict_model_variables_to_variables
            * self.dict_variables_to_model_variables

        """
        # initialize lists and dicts
        all_variable_fields = []
        all_variable_fields_input = []
        all_variable_fields_output = []
        dict_fields_to_vars = {}
        dict_vars_by_subsector = {} #
        dict_vars_to_subsector = {} #
        dict_vars_to_fields = {} #

        modvars_all = []

        for subsector in self.all_subsectors_with_primary_category:
            # get model variable lists
            subsector_modvars = self.get_subsector_variables(subsector)

            for modvar_name in subsector_modvars:

                modvar = self.dict_variables.get(modvar_name)
                if modvar is None:
                    continue
                
                # update output lists
                all_variable_fields.extend(modvar.fields)
                var_type = modvar.get_property("variable_type")
                if isinstance(var_type, str):
                    var_type = var_type.lower()
                    if var_type == "input":
                        all_variable_fields_input.extend(modvar.fields)

                    elif var_type == "output":
                        all_variable_fields_output.extend(modvar.fields)

                # update relavent dictionaries
                dict_vars_to_fields.update({modvar_name: modvar.fields})
                dict_fields_to_vars.update(dict((x, modvar_name) for x in modvar.fields))

            # extend other dictionaries
            dict_vars_by_subsector.update({subsector: subsector_modvars})
            dict_vars_to_subsector.update(dict((x, subsector) for x in subsector_modvars))


        # get all variables as a list
        all_variable_fields.sort()
        all_variable_fields_input.sort()
        all_variable_fields_output += self.get_all_subsector_emission_total_fields()
        all_variable_fields_output.sort()


        ##  SET PROPERTIES

        self.all_variable_fields = all_variable_fields
        self.all_variable_fields_input = all_variable_fields_input
        self.all_variable_fields_output = all_variable_fields_output
        self.dict_model_variables_by_subsector = dict_vars_by_subsector
        self.dict_model_variable_to_subsector = dict_vars_to_subsector
        self.dict_model_variables_to_variables = dict_vars_to_fields
        self.dict_variables_to_model_variables = dict_fields_to_vars

        return None





    ######################################################################################
    ###                                                                                ###
    ###    CHECKS FOR ATTRIBUTE TABLES AND CROSS-SECTORAL INTEGRATION WITHIN TABLES    ###
    ###                                                                                ###
    ######################################################################################

    def _check_binary_category_partition(self,
        attr: AttributeTable,
        fields: List[str],
        allow_subset: bool = False,
    ) -> None:
        """
        Check a set of binary fields specificied in an attribute table to 
            determine if they specify a partition across the categories. Assumes
            fields are binary.

        Function Arguments
        ------------------
        - attr: AttributeTable to check
        - fields: fields to check

        Keyword Arguments
        -----------------
        - allow_subset: if True, then allows values associated with the fields
            to go unassigned
        """

        fields = [x for x in fields if x in attr.table.columns]
        vec_check = np.array(attr.table[fields]).sum(axis = 1).astype(int)
        set_ref = set({0, 1}) if allow_subset else set({1})

        if set(vec_check) != set_ref:
            fields_print = ", ".join([f"'{x}'" for x in fields])
            raise RuntimeError(f"Invalid specifcation of attribute table with key {attr.key}--the fields {fields_print} do not partition the categories.")

        return None



    def _check_binary_fields(self,
        attr: AttributeTable,
        subsec: str,
        fields: str,
        force_sum_to_one: bool = False,
    ) -> None:
        """
        Check fields `fields` in attr to ensure they are all binary (1 or 0). 
            Set `force_sum_to_one` = True to ensure that exactly one record 
            associated with each field is 1.
        """
        for fld in fields:
            valid_sum = (sum(attr.table[fld]) == 1) if force_sum_to_one else True

            if fld not in attr.table.columns:
                raise ValueError(
                    f"Error in subsector {subsec}: required field '{fld}' not found in the table at '{attr.fp_table}'."
                )

            elif not all([((x == 1) | (x == 0)) for x in list(attr.table[fld])]):
                raise ValueError(
                    f"Error in subsector {subsec}:  invalid values found in field '{fld}' in the table at '{attr.fp_table}'. Only 0 or 1 should be specified."
                )

            elif not valid_sum:
                raise ValueError(
                    f"Invalid specification of field '{fld}' found in {subsec} attribute table: exactly 1 category or variable should be specfied in the field '{fld}'.\n\nUse 1 to flag the category; all other values should be 0."
                )

        return None



    def _check_numeric_fields(self,
        attr: AttributeTable,
        subsec: str,
        fields: str,
        integer_q: bool = False,
        nonnegative_q: bool = True,
        check_bounds: tuple = None,
    ) -> None:
        """
        Verify numeric fields in attr
        """
        # loop over fields to do checks
        for fld in fields:
            if fld not in attr.table.columns:
                msg = f"""
                Error in subsector {subsec}: required field '{fld}' not found in 
                the table at '{attr.fp_table}'.
                """
                raise ValueError(msg)
            else:
                try:
                    vals = list(attr.table[fld].astype(float))
                except:
                    msg = f"""
                    Error in subsector {subsec}: Non-numeric values found in 
                    field '{fld}'. Check the table at '{attr.fp_table}'.
                    """
                    raise ValueError(msg)

                # check additional restrictions
                if check_bounds is not None:
                    if (min(vals) < check_bounds[0]) or (max(vals) > check_bounds[1]):
                        msg = f"""
                        Error in subsector {subsec}: values in field '{fld}' 
                        outside of bounds ({check_bounds[0]}, {check_bounds[1]}) 
                        specified. Check the attribute table at 
                        '{attr.fp_table}'.
                        """
                        raise ValueError(msg)

                elif nonnegative_q and (min(vals) < 0):
                        msg = f"""
                        Error in subsector {subsec}: Negative values found in 
                        field '{fld}'. The field should only have non-negative 
                        numbers. Check the table at '{attr.fp_table}'.
                        """
                        raise ValueError(msg)

                if integer_q:
                    vals_check = [int(x) == x for x in vals]
                    if not all(vals_check):
                        msg = f"""
                        Error in subsector {subsec}: Non-integer equivalent 
                        values found in the field {fld}. Entries in '{fld}' 
                        should be integers or float equivalents. Check the table 
                        at '{attr.fp_table}'.
                        """
                        raise ValueError(msg)

        return None



    def _check_subsector_attribute_table_crosswalk(self, #VISIT
        dict_subsector_primary: dict,
        subsector_target: str,
        type_primary: str = "categories",
        type_target: str = "categories",
        injection_q: bool = True,
        allow_multiple_cats_q: bool = False,
    ):
        """
        Check the validity of categories specified as an attribute 
            (subsector_target) of a primary subsector category 
            (subsector_primary)

        Function Arguments
        ------------------
        - dict_subsector_primary: dictionary of form {subsector_primary: 
            field_attribute_target}. The key gives the primary subsector, 
            and 'field_attribute_target' is the field in the attribute table 
            associated with the categories to check.
            * NOTE: dict_subsector_primary can also be specified only as a 
                string (subsector_primary) -- if dict_subsector_primary is a 
                string, then field_attribute_target is assumed to be the primary 
                python category of subsector_target (e.g., $CAT-TARGET$)
        - subsector_target: target subsector to check values against

        Keyword Arguments
        -----------------
        - allow_multiple_cats_q: allow the target field to specify multiple 
            categories using the default delimiter (|)?
        - injection_q: default = True. If injection_q, then target categories 
            should be associated with a unique primary category (exclding those 
            are specified as 'none').
        - type_primary: default = "categories". Represents the type of attribute 
            table for the primary table; valid values are 'categories', 
            'varreqs_all', and 'varreqs_partial'
        - type_target: default = "categories". Type of the target table. Valid 
            values are the same as those for type_primary.
        """

        ##  RUN CHECKS ON INPUT SPECIFICATIONS

        # check type specifications
        dict_valid_types_to_attribute_keys = {
            "categories": "pycategory_primary",
            "varreqs_all": "key_varreqs_all",
            "varreqs_partial": "key_varreqs_partial"
        }
        valid_types = list(dict_valid_types_to_attribute_keys.keys())
        str_valid_types = sf.format_print_list(valid_types)

        if type_primary not in valid_types:
            raise ValueError(f"Invalid type_primary '{type_primary}' specified. Valid values are '{str_valid_types}'.")

        if type_target not in valid_types:
            raise ValueError(f"Invalid type_target '{type_target}' specified. Valid values are '{str_valid_types}'.")

        # get the primary subsector + field, then run checks
        if isinstance(dict_subsector_primary, dict):
            if len(dict_subsector_primary) != 1:
                raise KeyError(f"Error in dictionary dict_subsector_primary: only one key (subsector_primary) should be specified.")
            subsector_primary = list(dict_subsector_primary.keys())[0]

        elif isinstance(dict_subsector_primary, str):
            subsector_primary = dict_subsector_primary

        else:
            t_str = str(type(dict_subsector_primary))
            raise ValueError(f"Invalid type '{t_str}' of dict_subsector_primary: 'dict' and 'str' are acceptable values.")


        # check that the subsectors are valid
        self.check_subsector(subsector_primary)
        self.check_subsector(subsector_target)

        # check primary table type and fetch attribute
        dict_tables_primary = (
            self.dict_attributes 
            if (type_primary == "categories") 
            else self.dict_variable_definitions
        )

        key_primary = self.get_subsector_attribute(
            subsector_primary, 
            dict_valid_types_to_attribute_keys[type_primary]
        )

        if not key_primary:
            msg = f"""
            Invalid type_primary '{type_primary}' specified for primary 
            subsector '{subsector_primary}': type not found.
            """
            raise ValueError(msg)

        attr_prim = dict_tables_primary[key_primary]


        # check target table type and fetch attribute
        dict_tables_primary = (
            self.dict_attributes 
            if (type_target == "categories") 
            else self.dict_variable_definitions
        )

        key_target = self.get_subsector_attribute(
            subsector_target, 
            dict_valid_types_to_attribute_keys[type_target]
        )

        key_target_pycat = self.get_subsector_attribute(
            subsector_target, 
            "pycategory_primary_element"
        )

        if not key_primary:
            msg = f"""
            Invalid type_primary '{type_target}' specified for primary subsector 
            '{subsector_target}': type not found.
            """
            raise ValueError(msg)

        attr_targ = dict_tables_primary[key_target]


        # check that the field is properly specified in the primary table
        field_subsector_primary = (
            str(dict_subsector_primary[subsector_primary]) 
            if (type(dict_subsector_primary) == dict) 
            else key_target
        )

        if field_subsector_primary not in attr_prim.table.columns:
            msg = f"""
            Error in _check_subsector_attribute_table_crosswalk: field 
            '{field_subsector_primary}' not found in the '{subsector_primary}' 
            attribute table. Check the file at '{attr_prim.fp_table}'.
            """
            raise ValueError(msg)


        ##  CHECK ATTRIBUTE TABLE CROSSWALKS

        # get categories specified in the
        primary_cats_defined = list(attr_prim.table[field_subsector_primary])
        if allow_multiple_cats_q:
            primary_cats_defined = sum([[clean_schema(y) for y in x.split(self.delim_multicats)] for x in primary_cats_defined if (x != "none")], []) if (key_target == key_target_pycat) else [x for x in primary_cats_defined if (x != "none")]

        else:
            primary_cats_defined = (
                [clean_schema(x) for x in primary_cats_defined if (x != "none")] 
                if (key_target == key_target_pycat) 
                else [x for x in primary_cats_defined if (x != "none")]
            )

        # ensure that all population categories properly specified
        if not set(primary_cats_defined).issubset(set(attr_targ.key_values)):
            valid_vals = sf.format_print_list(set(attr_targ.key_values))
            invalid_vals = sf.format_print_list(list(set(primary_cats_defined) - set(attr_targ.key_values)))

            msg = f"""
            Invalid categories {invalid_vals} specified in field 
            '{field_subsector_primary}' of the {subsector_primary} attribute 
            table at '{attr_prim.fp_table}'.
            
            Valid categories from {subsector_target} are: {valid_vals}
            """

            raise ValueError(msg)


        if injection_q:
            # check that categories are mapped 1:1 to a category
            if len(set(primary_cats_defined)) != len(primary_cats_defined):
                duplicate_vals = sf.format_print_list(
                    set([
                        x for x in primary_cats_defined 
                        if primary_cats_defined.count(x) > 1
                    ])
                )

                msg = f"""
                Error in {subsector_primary} attribute table at 
                '{attr_prim.fp_table}': duplicate specifications of target 
                categories {duplicate_vals}. There map of {subsector_primary} 
                categories to {subsector_target} categories should be an 
                injection map.
                """
                raise ValueError(msg)

        return None



    ####################################################
    #    SECTOR-SPECIFIC AND CROSS SECTORIAL CHECKS    #
    ####################################################

    def _check_attribute_tables_abv_subsector(self,
    ) -> None:
        """
        Check subsector attribute table
        """
        # check for proper specifications in attribute table
        attr = self.dict_attributes.get("abbreviation_subsector")

        self._check_binary_fields(
            attr, 
            "subsector",
            ["emission_subsector"]
        )
       
        return None



    def _check_attribute_tables_agrc(self,
    ) -> None:
        """
        Check agricultural attribute tables (category and variables)
        """
        # check for proper specifications in attribute table
        attr = self.get_attribute_table(self.subsec_name_agrc)
        fields_req = ["apply_vegetarian_exchange_scalar", "rice_category"]
        self._check_binary_fields(attr, self.subsec_name_agrc, ["apply_vegetarian_exchange_scalar"])
        self._check_binary_fields(attr, self.subsec_name_agrc, ["rice_category"], force_sum_to_one = True)

        # next, check the crosswalk for correct specification of soil management categories
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_agrc,
            self.subsec_name_soil,
            type_primary = "varreqs_all",
            type_target = "categories",
            injection_q = True
        )

        return None



    def _check_attribute_tables_enfu(self,
    ) -> None:
        """
        Check the Energy Fuels table to ensure that an electricity category is 
            specified

            * self.field_enfu_biofuels_demand_category
            * self.field_enfu_biogas_fuel_category
            * self.field_enfu_electricity_demand_category
            * self.field_enfu_hydrogen_fuel_category
            * self.field_enfu_hydropower_fuel_category
            * self.field_enfu_upstream_to_fuel_category
            * self.field_enfu_waste_fuel_category
        """
        # share values
        subsec = self.subsec_name_enfu
        attr = self.get_attribute_table(subsec)

        # miscellaneous parameters that need to be checked before running
        self.field_enfu_biofuels_demand_category = "biomass_demand_category"
        self.field_enfu_biogas_fuel_category = "biogas_fuel_category"
        self.field_enfu_electricity_demand_category = "electricity_demand_category"
        self.field_enfu_hydrogen_fuel_category = "hydrogen_fuel_category"
        self.field_enfu_hydropower_fuel_category = "hydropower_fuel_category"
        self.field_enfu_upstream_to_fuel_category = "upstream_to_fuel_category"
        self.field_enfu_waste_fuel_category = "waste_fuel_category"

        # check binary variables
        fields_req_bin = [
            self.field_enfu_biofuels_demand_category,
            self.field_enfu_biogas_fuel_category,
            self.field_enfu_electricity_demand_category,
            self.field_enfu_hydrogen_fuel_category,
            self.field_enfu_hydropower_fuel_category,
            self.field_enfu_waste_fuel_category
        ]
        self._check_binary_fields(
            attr, 
            self.subsec_name_enfu, 
            fields_req_bin, 
            force_sum_to_one = True
        )

        # check specification of upstream fuel
        self._check_subsector_attribute_table_crosswalk(
            {self.subsec_name_enfu: self.field_enfu_upstream_to_fuel_category},
            self.subsec_name_enfu,
            type_primary = "categories",
            injection_q = True
        )

        return None



    def _check_attribute_tables_enst(self,
    ) -> None:
        """
        Check the Energy Storage table to check the technology crosswalk and 
            ensure the proper specification of the following fields:

            * "minimum_charge_fraction" (numeric, bounded 0, 1)
            * "netzeroyear" (binary)
            * "netzerotg1" (binary)
            * "netzerotg2" (binary)
        """
        # some shared values
        subsec = self.subsec_name_enst
        attr = self.get_attribute_table(subsec)

        # check required binary fields
        fields_req_bin = ["netzeroyear", "netzerotg1", "netzerotg2"]
        self._check_binary_fields(attr, subsec, fields_req_bin)

        # check numeric field
        self._check_numeric_fields(attr, subsec, ["minimum_charge_fraction"], check_bounds = (0, 1))

        # check storage/technology crosswalk
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_enst, 
            self.subsec_name_entc, 
            type_primary = "categories", 
            injection_q = False
        )

        return None



    def _check_attribute_tables_entc(self,
    ) -> None:
        """
        Check the Energy Technology table to check the following components:

            * Check crosswalks with ENST and ENFU to ensure fields are properly
                specified:
                * electricity_generation_{pycat_enfu} (ENFU)
                * generates_fuel_{pycat_enfu} (ENFU)
                * {pycat_enst} (ENST)
                * technology_from_storage (ENST)
                * technology_to_storage (ENST)

            * Ensure the proper specification of the following fields:
                * "fuel_processing" (binary)
                * "mining_and_extraction" (binary)
                * "operational_life" (numeric, >= 0)
                * "power_plant" (binary)
                * "storage" (binary)

            * Checks that each technology is specified in exactly one of the
                following fields:
                * "fuel_processing"
                * "mining_and_extraction"
                * "power_plant"
                * "storage"

        """
        # some shared values
        pycat_enfu = self.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary_element")
        subsec = self.subsec_name_entc
        attr = self.get_attribute_table(subsec)


        # check required fields - binary - and the partition of types
        fields_req_bin = ["fuel_processing", "mining_and_extraction", "power_plant", "storage"]
        self._check_binary_fields(attr, subsec, fields_req_bin)

        fields_partition_bin = ["fuel_processing", "mining_and_extraction", "power_plant", "storage"]
        self._check_binary_category_partition(attr, fields_partition_bin)

        # check required fields - numeric
        fields_req_num = ["operational_life"]
        self._check_numeric_fields(attr, subsec, fields_req_num, integer_q = False, nonnegative_q = True)

        # check technology/fuel crosswalks
        self._check_subsector_attribute_table_crosswalk(
            {self.subsec_name_entc: f"electricity_generation_{pycat_enfu}"},
            self.subsec_name_enfu,
            type_primary = "categories",
            injection_q = False
        )
        # check specifications of fuel in fuel generation
        self._check_subsector_attribute_table_crosswalk(
            {self.subsec_name_entc: f"generates_fuel_{pycat_enfu}"},
            self.subsec_name_enfu,
            type_primary = "categories",
            injection_q = False
        )
        # check technology/storage crosswalks
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_entc,
            self.subsec_name_enst,
            type_primary = "categories",
            injection_q = False
        )
        # check specifications of storage in technology from storage
        self._check_subsector_attribute_table_crosswalk(
            {self.subsec_name_entc: "technology_from_storage"},
            self.subsec_name_enst,
            injection_q = False,
            allow_multiple_cats_q = True
        )
        # check specifications of storage in technology to storage
        self._check_subsector_attribute_table_crosswalk(
            {self.subsec_name_entc: "technology_to_storage"},
            self.subsec_name_enst,
            injection_q = False,
            allow_multiple_cats_q = True
        )

        return None



    def _check_attribute_tables_inen(self,
    ) -> None:
        """
        Check specification of the Industrial Energy attribute table, including 
            the industrial energy/fuels cw in industrial energy.
        """
        # some shared values
        subsec = self.subsec_name_inen
        attr = self.get_attribute_table(subsec, "variable_definitions")

        # check required fields - binary
        fields_req_bin = ["fuel_fraction_variable_by_fuel"]
        self._check_binary_fields(attr, subsec, fields_req_bin)

        # function to check the industrial energy/fuels cw in industrial energy
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_inen, 
            self.subsec_name_enfu, 
            type_primary = "varreqs_partial", 
            injection_q = False
        )

        return None



    def _check_attribute_tables_ippu(self,
    ) -> None:
        """
        Check specification of the Industrial Processes and Product Use 
            attribute table, including specification of HFC/PFC/Other FC 
            emission factors as a variable attribute.
        """
        # some shared values
        subsec = self.subsec_name_ippu
        attr = self.get_attribute_table(subsec, "variable_definitions")

        # check required fields - binary
        fields_req_bin = [
            "emission_factor",
        ]
        self._check_binary_fields(attr, subsec, fields_req_bin)

        return None



    def _check_attribute_tables_lndu(self,
    ) -> None:
        """
        Check that the land use attribute tables are specified correctly.
        """
        # specify some generic variables
        attribute_forest = self.get_attribute_table(self.subsec_name_frst)
        attribute_landuse = self.get_attribute_table(self.subsec_name_lndu)

        cats_forest = attribute_forest.key_values
        cats_landuse = attribute_landuse.key_values
        matchstr_forest = self.matchstring_landuse_to_forests

        # function to check the land use/forestry crosswalk
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_lndu,
            self.subsec_name_frst,
            injection_q = True
        )

        ##  check that all forest categories are in land use and that all categories specified as forest are in the land use table
        set_cats_forest_in_land_use = set([matchstr_forest + x for x in cats_forest])
        set_land_use_forest_cats = set([x.replace(matchstr_forest, "") for x in cats_landuse if (matchstr_forest in x)])

        if not set_cats_forest_in_land_use.issubset(set(cats_landuse)):

            missing_vals = set_cats_forest_in_land_use - set(cats_landuse)
            missing_str = sf.format_print_list(missing_vals)

            msg = f"""
            Missing key values in land use attribute file 
            '{attribute_landuse.fp_table}': did not find land use categories 
            {missing_str}.
            """

            raise KeyError(msg)

        elif not set_land_use_forest_cats.issubset(cats_forest):

            extra_vals = set_land_use_forest_cats - set(cats_forest)
            extra_vals = sf.format_print_list(extra_vals)

            msg = f"""
            Undefined forest categories specified in land use attribute file 
            '{attribute_landuse.fp_table}': did not find forest categories 
            {extra_vals}.
            """

            raise KeyError(msg)


        # check specification of crop category & pasture category
        fields_req_bin = [
            "crop_category", 
            "other_category", 
            "pasture_category", 
            "settlements_category", 
            "wetlands_category"
        ]

        self._check_binary_fields(
            attribute_landuse, 
            self.subsec_name_lndu, 
            fields_req_bin, 
            force_sum_to_one = 1,
        )

        # check
        fields_req_bin = ["reallocation_transition_probability_exhaustion_category"]
        self._check_binary_fields(
            attribute_landuse, 
            self.subsec_name_lndu, 
            fields_req_bin, 
            force_sum_to_one = 0,
        )


        # check to ensure that source categories for mineralization in soil management are specified properly
        field_mnrl = "mineralization_in_land_use_conversion_to_managed"
        cats_crop = self.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {"crop_category": 1}
        )
        cats_mnrl = self.get_categories_from_attribute_characteristic(
            self.subsec_name_lndu, 
            {field_mnrl: 1}
        )
        
        if len(set(cats_crop) & set(cats_mnrl)) > 0:
            msg = f"""
            f"Invalid specification of field '{field_mnrl}' in 
            {self.subsec_name_lndu} attribute located at 
            {attribute_landuse.fp_table}. Category '{cats_crop[0]}' cannot be 
            specified as a target category."
            """
            raise ValueError(msg)

        # check that land use/soil and forest/soil crosswalks are properly specified
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_frst,
            self.subsec_name_soil,
            type_primary = "varreqs_all",
            type_target = "categories",
            injection_q = True
        )

        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_lndu,
            self.subsec_name_soil,
            type_primary = "varreqs_partial",
            type_target = "categories",
            injection_q = True
        )

        # check that forest/land use crosswalk is set properly
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_frst,
            self.subsec_name_lndu,
            type_primary = "categories",
            type_target = "categories",
            injection_q = True
        )

        # check required fields - binary
        fields_req_bin = [
            "mangroves_forest_category", 
            "primary_forest_category", 
            "secondary_forest_category"
        ]

        self._check_binary_fields(
            attribute_forest, 
            self.subsec_name_frst, 
            fields_req_bin, 
            force_sum_to_one = 1,
        )

        return None



    def _check_attribute_tables_lsmm(self,
    ) -> None:
        """
        Check the livestock manure management attribute table
        """

        subsec = "Livestock Manure Management"
        attr = self.get_attribute_table(subsec)
        fields_check_sum = ["incineration_category", "pasture_category"]

        # check that the integration fields are properly specified
        for field in fields_check_sum:
            vals = set(attr.table[field])
            if (not vals.issubset(set({0, 1}))) or (sum(attr.table[field]) > 1):
                msg = f"""
                Invalid specification of field '{field}' in {subsec} attribute 
                located at {attr.fp_table}. Check to ensure that at most 1 is 
                specified; all other entries should be 0.
                """
                raise ValueError(msg)

        # next, check that the fields are not assigning categories to multiple types
        fields_check_sum = [x for x in fields_check_sum if x in attr.table]
        vec_max = np.array(attr.table[fields_check_sum].sum(axis = 1))
        if max(vec_max) > 1:
            fields = sf.format_print_list(fields_check_sum)
            msg = f"""
            Invalid specification of fields {fields} in {subsec} attribute 
            located at {attr.fp_table}: Non-injective mapping specified--
            categories can map to at most 1 of these fields.
            """
            raise ValueError(msg)

        return None



    def _check_attribute_tables_trde(self,
    ) -> None:
        """
        Check specification of Transportation Demand attribute tables.
        """
        # some shared values
        subsec = self.subsec_name_trde
        attr = self.get_attribute_table(subsec)

        # check required fields - binary
        fields_req_bin = ["freight_category"]
        self._check_binary_fields(
            attr, 
            subsec, 
            fields_req_bin,
            force_sum_to_one = True
        )

        # function to check the TRDE crosswalk of a variable name to 
        self._check_subsector_attribute_table_crosswalk(
            {subsec: "partial_category_en_trde"},
            subsec,
            injection_q = False,
            type_primary = "categories",
            type_target = "varreqs_partial",
        )

        return None



    def _check_attribute_tables_trns(self,
    ) -> None:
        """
        Check specification of Transportation attribute tables, including the
            transportation/transportation demand crosswalk in both the attribute 
            table and the varreqs table.
        """

        #attr = self.get_attribute_table(self.subsec_name_trns)
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_trns, 
            self.subsec_name_trde, 
            type_primary = "varreqs_partial", 
            injection_q = True
        )
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_trns, 
            self.subsec_name_trde, 
            injection_q = False
        )

        return None



    def _check_attribute_tables_wali(self,
    ) -> None:
        """
        Check specification of Liquid Waste attribute tables, including the
            check the liquid waste/population crosswalk in liquid waste and the
            liquid waste/wastewater crosswalk.
        """

        # check the liquid waste/population crosswalk in liquid waste
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_wali, 
            self.subsec_name_gnrl, 
            injection_q = True
        )

        # liquid waste/wastewater crosswalk
        self._check_subsector_attribute_table_crosswalk(
            self.subsec_name_wali, 
            self.subsec_name_trww, 
            type_primary = "varreqs_all"
        )

        return None



    def _check_attribute_tables_waso(self,
    ) -> None:
        """
        Check if the solid waste attribute table is properly defined.
        """
        # check that exactly one category is associated with sludge/food
        attr = self.get_attribute_table(self.subsec_name_waso)
        fields_req_bin = ["food_category", "sewage_sludge_category"]
        
        self._check_binary_fields(
            attr, 
            self.subsec_name_waso, 
            fields_req_bin,
            force_sum_to_one = True
        ) 
        
        return None






    ############################################################
    #   FUNCTIONS FOR ATTRIBUTE TABLES, DIMENSIONS, SECTORS    #
    ############################################################

    def add_index_fields(self,
        df_input: pd.DataFrame,
        design_id: Union[None, int] = None,
        future_id: Union[None, int] = None,
        primary_id: Union[None, int] = None,
        region: Union[None, int] = None,
        strategy_id: Union[None, int] = None,
        time_period: Union[None, int] = None,
        time_series_id: Union[None, int] = None,
        overwrite_fields: bool = False
    ) -> pd.DataFrame:
        """
        Add scenario and dimensional index fields to a data frame using 
            consistent field hierachy

        Function Arguments
        ------------------
        - df_input: Input DataFrame to add indexes to

        Keyword Arguments
        -----------------
        - design_id: value for index ModelAttributes.dim_design_id; if None, the 
            index is not added
        - future_id: value for index ModelAttributes.dim_future_id; if None, the 
            index is not added
        - primary_id: value for index ModelAttributes.dim_primary_id; if None, 
            the index is not added
        - region: value for index ModelAttributes.dim_region; if None, the index 
            is not added
        - strategy_id: value for index ModelAttributes.dim_strategy_id; if None, 
            the index is not added
        - time_period: value for index ModelAttributes.dim_time_period; if None, 
            the index is not added
        - time_series_id: value for index ModelAttributes.dim_time_series_id; if 
            None, the index is not added
        - overwrite_fields:
            * If True, if the index field already iexists in `df_input`, it will 
                be overwritten with the value passed to add_index_fields
            * Otherwise, the existing field will be left.
            * NOTE: if a value is passed with overwrite_q = False, then the data 
                frame will still order fields hierarchically
        """

        if df_input is None:
            return None

        dict_indices = {}

        # update values
        dict_indices.update({self.dim_design_id: design_id}) if (design_id is not None) else None
        dict_indices.update({self.dim_future_id: future_id}) if (future_id is not None) else None
        dict_indices.update({self.dim_primary_id: primary_id}) if (primary_id is not None) else None
        dict_indices.update({self.dim_region: region}) if (region is not None) else None
        dict_indices.update({self.dim_strategy_id: strategy_id}) if (strategy_id is not None) else None
        dict_indices.update({self.dim_time_period: time_period}) if (time_period is not None) else None
        dict_indices.update({self.dim_time_series_id: time_series_id}) if (time_series_id is not None) else None

        df_input = sf.add_data_frame_fields_from_dict(
            df_input,
            dict_indices,
            field_hierarchy = self.sort_ordered_dimensions_of_analysis,
            prepend_q = True,
            overwrite_fields = overwrite_fields
        )

        return df_input



    def check_dimensions_of_analysis(self,
    ) -> None:
        """
        Ensure dimensions of analysis are properly specified
        """
        if not set(self.sort_ordered_dimensions_of_analysis).issubset(set(self.all_dims)):

            missing_vals = sf.print_setdiff(set(self.sort_ordered_dimensions_of_analysis), set(self.all_dims))

            msg = f"""
            f"Missing specification of required dimensions of analysis: no 
            attribute tables for dimensions {missing_vals} found in directory 
            '{self.attribute_directory}'."
            """
            raise ValueError(msg)
        
        return None



    def check_integrated_df_vars(self,
        df_in: pd.DataFrame,
        dict_integrated_vars: dict,
        subsec: str = "all",
    ) -> dict:
        """
        Check data frames specified for integrated variables
        """
        # initialize list of subsectors to provide checks for
        subsecs = list(dict_integrated_vars.keys()) if (subsec == "all") else [subsec]
        dict_out = {}
        #
        for subsec0 in subsecs:

            subsec = self.check_subsector(subsec0, throw_error_q = False)

            if (subsec is not None):

                fields_req = []
                for modvar in dict_integrated_vars.get(subsec):
                    fields_req += self.build_variable_fields(modvar)

                # check for required variables
                subsec_val = True

                if not set(fields_req).issubset(df_in.columns):
                    set_missing = list(set(fields_req) - set(df_in.columns))
                    set_missing = sf.format_print_list(set_missing)
                    warnings.warn(f"Integration in subsector '{subsec}' cannot proceed: The fields {set_missing} are missing.")
                    subsec_val = False

            else:
                warnings.warn(f"Invalid subsector '{subsec}' found in check_integrated_df_vars: The subsector does not exist.")
                subsec_val = False

            dict_out.update({subsec: subsec_val})

        return dict_out[subsec] if (subsec != "all") else dict_out


    
    def check_modvar(self,
        modvar: Union[Any, None],
    ) -> Union[str, None]:
        """
        Check if modvar is valid. If so, will return modvar. If not, returns
            None.
        """

        if not isinstance(modvar, str):
            return None

        out = (
            modvar
            if modvar in self.all_variables
            else None
        )

        return out
        


    def check_region(self,
        region: str,
        allow_unclean: bool = False
    ) -> None:
        """
        Ensure a region is properly specified
        """
        region = self.clean_region(region) if allow_unclean else region
        attr_region = self.dict_attributes.get(self.dim_region)

        # check sectors
        if region not in attr_region.key_values:
            valid_regions = sf.format_print_list(attr_region.key_values)
            raise ValueError(f"Invalid region specification '{region}': valid sectors are {valid_region}")

        return None



    def check_sector(self, 
        sector: str,
        throw_error: bool = True,
    ) -> Union[str, None]:
        """
        Ensure a sector is properly specified
        """
        # check sectors
        
        out = None if throw_error else sector

        if sector not in self.all_sectors:
            if throw_error:
                valid_sectors = sf.format_print_list(self.all_sectors)
                raise ValueError(f"Invalid sector specification '{sector}': valid sectors are {valid_sectors}")
            else:
                # return None if not throwing an error
                out = None

        return out



    def check_subsector(self,
        subsector: str,
        throw_error_q = True
    ) -> Union[str, None]:
        """
        Ensure a subsector is properly specified
        """
        # check sectors
        if subsector not in self.all_subsectors:
            valid_subsectors = sf.format_print_list(self.all_subsectors)
            if throw_error_q:
                raise ValueError(f"Invalid subsector specification '{subsector}': valid sectors are {valid_subsectors}")
            else:
                return False
        else:
            return (None if throw_error_q else subsector)



    def check_restricted_value_argument(self, 
        arg: Any, 
        valid_values: list, 
        func_arg: str = "", 
        func_name: str = ""
    ) -> None:
        """
        Commonly used--restrict variable values. Throws an error if the argument
            is not in valid values. Reports out valid values in error message.
        """
        if arg not in valid_values:
            vrts = sf.format_print_list(valid_values)
            raise ValueError(f"Invalid {func_arg} in {func_name}: valid values are {vrts}.")

        return None



    def clean_region(self,
        region: str
    ) -> str:
        """
        inline function to clean regions (commonly called)
        """
        out = region.strip().lower().replace(" ", "_")

        return out
    


    def extract_model_variable(self,
        df_in: pd.DataFrame,
        modvar: str,
        all_cats_missing_val: float = 0.0,
        expand_to_all_cats: bool = False,
        extraction_logic: str = "all",
        force_boundary_restriction: bool = True,
        include_time_period: bool = False,
        override_vector_for_single_mv_q: bool = False,
        return_num_type: type = np.float64,
        return_type: str = "data_frame",
        throw_error_on_missing_fields: bool = True,
        var_bounds = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Extract an array or data frame of input variables. If 
            return_type == "array_units_corrected", then ModelAttributes will 
            re-scale emissions factors to reflect the desired output emissions 
            mass (as defined in the configuration).

        Function Arguments
        ------------------
        - df_in: data frame containing input variables
        - modvar_name: name of variable to retrieve OR model variable object
            (if latter, acts as a wrapper)
        
        Keyword Arguments
        -----------------
        - all_cats_missing_val: default is 0. If expand_to_all_cats == True, 
            categories not associated with modvar with be filled with this 
            value.
        - expand_to_all_cats: default is False. If True, return the variable in 
            the shape of all categories.
        - extraction_logic: set logic used on extraction
            * "all": throws an error if any field in self.fields is missing
            * "any": extracts any field in self.fields available in `obj`
                and fills any missing values with fill_value (or default value)
        - force_boundary_restriction: default is True. Set to True to enforce 
            the boundaries on the variable. If False, a variable that is out of 
            bounds will raise an error.
        - include_time_period: include the time period? Only applies if 
            return_type == "data_frame"
        - override_vector_for_single_mv_q: default is False. Set to True to 
            return an array if the dimension of the variable is 1; otherwise, a 
            vector will be returned (if not a dataframe).
        - return_num_type: return type for numeric values
        - return_type: valid values are: 
            * "data_frame"
            * "array_base" (np.ndarray not corrected for configuration 
                emissions)
            * "array_units_corrected" (emissions corrected to reflect 
                configuration output emission units)
        - throw_error_on_missing_fields: set to True to throw an error if the
            fields associated with modvar are not found in df_in.
            * If False, returns None if fields implied by modvar are not found 
                in df_in
        - var_bounds: Default is None (no bounds). Otherwise, gives boundaries 
            to enforce variables that are retrieved. For example, some variables 
            may be restricted to the range (0, 1). Use a list-like structure to 
            pass a minimum and maximum bound (np.inf can be used to as no 
            bound).
        """
        ##  INITIALIZATION AND CHECKS

        # check model variable specificaiton and data frame
        modvar = self.get_variable(modvar)
        if modvar is None:
            raise ValueError(f"Invalid variable specified in extract_model_variable: variable '{modvar}' not found.")            

        if not isinstance(df_in, pd.DataFrame):
            return None
        
        # check some arguments
        self.check_restricted_value_argument(
            return_type,
            ["data_frame", "array_base", "array_units_corrected", "array_units_corrected_gas"],
            "return_type", 
            "extract_model_variable"
        )
        self.check_restricted_value_argument(
            return_num_type,
            [float, int, np.float64, np.int64],
            "return_num_type", 
            "extract_model_variable"
        )


        ##  START EXTRACTION

        # will default to model variable default value
        fill_value = (
            all_cats_missing_val
            if sf.isnumber(all_cats_missing_val)
            else None
        )

        try:
            out = modvar.get_from_dataframe(
                df_in,
                expand_to_all_categories = expand_to_all_cats,
                extraction_logic = extraction_logic,
                fill_value = fill_value,
            )
        except Exception as e:
            msg = f"""
            Unable to extract variable {modvar.name} in extract_model_variable: {e}
            """
            raise ValueError(msg)
        
        # convert to array
        out = out.to_numpy().astype(return_num_type)
        out = (
            out[:, 0] 
            if ((len(modvar.fields) == 1) and not override_vector_for_single_mv_q)
            else out
        )

        # initialize output, apply various common transformations based on type
        if return_type == "array_units_corrected":
            out *= self.get_scalar(modvar, "total")

        elif return_type == "array_units_corrected_gas":
            out *= self.get_scalar(modvar, "gas")

        if type(var_bounds) in [tuple, list, np.ndarray]:
            # get numeric values and check
            var_bounds = [x for x in var_bounds if type(x) in [int, float]]
            if len(var_bounds) <= 1:
                raise ValueError(f"Invalid specification of variable bounds '{var_bounds}': there must be a maximum and a minimum numeric value specified.")

            # ensure array
            out = np.array(out)
            b_0, b_1 = np.min(var_bounds), np.max(var_bounds)
            m_0, m_1 = np.min(out), np.max(out)

            # check bounds
            if m_1 > b_1:
                str_warn = f"Invalid maximum value of '{modvar}': specifed value of {m_1} exceeds bound {b_1}."
                if not force_boundary_restriction:
                    raise ValueError(str_warn)
                
                warnings.warn(str_warn + "\nForcing maximum value in trajectory.")

            # check min
            if m_0 < b_0:
                str_warn = f"Invalid minimum value of '{modvar}': specifed value of {m_0} below bound {b_0}."
                if not force_boundary_restriction:
                    raise ValueError(str_warn)
                
                warnings.warn(str_warn + "\nForcing minimum value in trajectory.")

            # force boundary if required
            out = sf.vec_bounds(out, var_bounds) if force_boundary_restriction else out


        # merge output to all categories?
        if expand_to_all_cats:
            out = np.array([out]).transpose() if (len(out.shape) == 1) else out
            out = self.merge_array_var_partial_cat_to_array_all_cats(
                np.array(out), 
                modvar, 
                missing_vals = all_cats_missing_val,
            )
            if return_type == "data_frame":
                sec = self.get_variable_subsector(modvar)
                flds = self.get_attribute_table(sec).key_values


        # convert back to data frame if necessary
        if (return_type == "data_frame"):
            flds = [flds] if (not type(flds) in [list, np.ndarray]) else flds
            out = pd.DataFrame(out, columns = flds)

            # add the time period?
            if include_time_period & (self.dim_time_period in df_in.columns):
                out[self.dim_time_period] = list(df_in[self.dim_time_period])
                out = out[[self.dim_time_period] + flds]


        return out



    def get_all_subsector_emission_total_fields(self,
        filter_on_emitting_only: bool = True,
    ) -> List[str]:
        """
        Generate a list of all subsector emission total fields added to
            model outputs. Set `filter_on_emitting_only = False` to include 
            nominal fields for non-emitting subsectors.
        """
        # get emission subsectors
        attr = self.get_subsector_attribute_table()
        subsectors_emission = (
            list(
                attr.table[
                    attr.table["emission_subsector"] == 1
                ]["subsector"]
            )
            if filter_on_emitting_only
            else self.all_subsectors
        )

        out = [self.get_subsector_emission_total_field(x) for x in subsectors_emission]

        return out
    


    def get_attribute_table(self, #FIXED
        subsector: str,
        table_type = "primary_category",
    ) -> Union[AttributeTable, None]:
        """
        Simplify retrieval of attribute tables within functions. 

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - table_type: one of the following values
            * "primary_category": primary category table associated with the 
                subsector
            * "variable_definitions"
        """

        # check input type
        valid_types = {
            "primary_category": "pycategory_primary", 
            "variable_definitions": "key_variable_definitions",
        }

        if table_type not in valid_types.keys():
            tps = sf.format_print_list(sorted(list(valid_types.keys())))
            msg = f"Invalid table_type '{table_type}': valid options are {tps}."
            raise ValueError(msg)

        # get different table by different type
        subsec_attr = valid_types.get(table_type)
        key_dict = self.get_subsector_attribute(subsector, subsec_attr)

        if table_type == "primary_category":
            dict_retrieve = self.dict_attributes.get(self.attribute_group_key_cat)
            out = dict_retrieve.get(key_dict)

        elif table_type == "variable_definitions":
            out = self.dict_variable_definitions.get(key_dict)

        return out



    def get_baseline_scenario_id(self, #FIXED
        dim: str,
        infer_baseline_as_minimum: bool = True,
    ) -> int:

        """
        Return the scenario id associated with a baseline scenario (as specified 
            in the attribute table)

        Function Arguments
        ------------------
        - dim: a scenario dimension specified in an attribute table 
            (attribute_dim_####.csv) within the ModelAttributes class
        - infer_baseline_as_minimum: If True, infers the baseline scenario as 
            the minimum specified.
        """
        if dim not in self.all_dims:
            fpl = sf.format_print_list(self.all_dims)
            raise ValueError(f"Invalid dimension '{dim}': valid dimensions are {fpl}.")

        attr = self.get_dimensional_attribute_table(dim)
        min_val = min(attr.key_values)

        # get field to check
        field_check = f"baseline_{dim}"
        if field_check not in attr.table.columns:
            str_append = (
                f" Inferring minimum key value {min_val} as baseline." 
                if infer_baseline_as_minimum 
                else " Returning None."
            )
            warnings.warn(f"No baseline specified for dimension '{dim}'.{str_append}")
            ret = min_val if infer_baseline_as_minimum else None

        else:
            tab_red = sorted(list(attr.table[attr.table[field_check] == 1][dim]))

            if len(tab_red) > 1:
                ret = tab_red[0]
                msg = f"""
                Multiple baselines specified for dimension {dim}. Ensure that 
                only one baseline is set in the attribute table at 
                '{tab.fp_table}'. Defaulting to minimum value of {ret}.
                """
                warnings.warn(msg)

            elif len(tab_red) == 0:
                str_append = (
                    f" Inferring minimum key value {min_val} as baseline." 
                    if infer_baseline_as_minimum 
                    else " Returning None."
                )
                warnings.warn(f"No baseline specified for dimension '{dim}'.{str_append}")
                ret = min_val if infer_baseline_as_minimum else None

            else:
                ret = tab_red[0]

        return ret



    def get_categories_from_attribute_characteristic(self, #FIXED
        subsector: str,
        dict_subset: dict,
        attribute_type: str = "primary_category",
        subsector_extract_key: Union[str, None] = None,
    ) -> Union[list, None]:
        """
        Return categories from an attribute table that match some 
            characteristics (defined in dict_subset)

        Keyword Arguments
        -----------------
        - attribute_type: "primary_category" or "variable_definitions"
        - subsector_extract_key: optional key to specity to retrieve. If None,
            retrieves subsector attribute key
        """
        #
        attr = self.get_attribute_table(subsector, attribute_type)
        if attr is None:
            return None

        subsector_extract_key = (
            subsector_extract_key
            if isinstance(subsector_extract_key, str) & (subsector_extract_key in attr.table.columns)
            else attr.key
        )

        # 
        return_val = list(sf.subset_df(attr.table, dict_subset)[subsector_extract_key])
        
        # ensure proper sorting
        return_val = (
            [x for x in attr.key_values if x in return_val] 
            if subsector_extract_key == attr.key
            else return_val
        )

        return return_val
    


    def get_df_dimensions_of_analysis(self, #FIXED
        df_in: pd.DataFrame, 
        df_in_shared: pd.DataFrame = None,
    ) -> list:
        """
        Get all dimensions of analysis in a data frame - can be used on two 
            data frames for merges
        """
        domain = set(df_in.columns)
        if isinstance(df_in_shared, pd.DataFrame):
            domain &= set(df_in_shared.columns)
        
        cols = [x for x in self.sort_ordered_dimensions_of_analysis if x in domain]

        return cols
    


    def get_dimensional_attribute_table(self, #FIXED
        dimension: str,
        stop_on_error: bool = False,
    ) -> Union[AttributeTable, None]:
        """
        Retrieve a dimension of analysis attribute table.
        """

        if dimension not in self.all_dims:
            if stop_on_error:
                valid_dims = sf.format_print_list(self.all_dims)
                raise ValueError(f"Invalid dimension '{dimension}'. Valid dimensions are {valid_dims}.")

            return None

        # add attributes here
        out = self.dict_attributes.get(self.attribute_group_key_dim)
        if out is None:
            return None

        out = out.get(dimension)
        
        return out
    


    def get_emission_subsectors(self, #FIXED
    ) -> List[str]:
        """
        Get subsectors that generate emissions
        """
        attr = self.get_subsector_attribute_table()
        subsectors_emission = list(
            attr.table[
                attr.table["emission_subsector"] == 1
            ]["subsector"]
        )

        return subsectors_emission
    


    def get_fields_from_variables(self, #FIXED
        modvars: Union[str, List[str]],
        sort: bool = False,
    ) -> Union[List[str], None]:
        """
        Using a list of model variables (or individual), return list of fields.
            Option to sort using sort = True
        """
        
        modvars = [modvars] if isinstance(modvars, str) else list(modvars)
        out = []
        
        for modvar_name in modvars:
            modvar = self.dict_variables.get(modvar_name)
            if modvar is None:
                continue
            
            out += modvar.fields
        
        out.sort() if sort else None
        
        return out
    


    def get_fluorinated_compound_dictionaries(self, #FIXED
        field_fc_designation: str = "flourinated_compound_designation",
    ) -> Dict[str, List[str]]:
        """
        Build a dictionary mapping FC designation to a list of gasses. Generates 
            a dictionary with the following keys (from gas attribute table):
            
            * hfc
            * none
            * other_fc
            * pfc
        
        Keyword Arguments
        -----------------
        - field_fc_designation: field in emission_gas attribute table containing
            the fluorinated compound designation of the gas 
        """
        
        attr_gas = self.get_other_attribute("emission_gas")
        dict_out = {}
        df_by_designation = attr_gas.table.groupby([field_fc_designation])
        
        for desig, df in df_by_designation:
            desig = str(desig).lower().replace(" ", "_")
            dict_out.update({desig: list(df[attr_gas.key])})
            
        return dict_out



    def get_ordered_category_attribute(self, #FIXED
        subsector: str,
        attribute: str,
        attr_type: str = "primary_category",
        clean_attribute_schema_q: bool = False,
        flag_none: str = "none",
        skip_none_q: bool = False,
        return_type: type = list,
    ) -> list:
        """
        Get attribute column from an attribute table ordered the same as key 
            values.

        Function Arguments
        ------------------
        - subsector: subsector to get categories in
        - attribute: attribute to retrieve ordered outcomes for

        Keyword Arguments
        -----------------
        - attr_type: either "primary_category" or "variable_definitions". Passed
            to get_attribute_table(subsect, table_type = attr_type, )
        - clean_attribute_schema_q: clean the target attribute using 
            mv.clean_element
        - flag_none: string identifying values as none
        - skip_none_q: if True, will skip values identifyed == flag_none
        - return_type: output return type. Acceptable types include 
            * dict
            * list
            * np.ndarray
        """

        ##  INITIALIZATION AND CHECKS

        valid_return_types = [list, np.ndarray, dict]
        if return_type not in valid_return_types:
            str_valid_types = sf.format_print_list(valid_return_types)
            raise ValueError(f"Invalid return_type '{return_type}': valid types are {str_valid_types}.")
    
        # get and check attribute table
        attr_cur = self.get_attribute_table(subsector, table_type = attr_type)
        if attr_cur is None:
            msg = f"""
            Invalid attribute type '{attr_type}': select 'primary_category' or 
            'variable_definitions'.
            """
            raise ValueError(msg)

        # verify the attribute is available
        if attribute not in attr_cur.table.columns:
            msg = f"""
            f"Missing attribute column '{attribute}': attribute not found in 
            '{subsector}' attribute table.
            """
            raise ValueError(msg)


        ##  FILTER THE TABLE AND GET ORDERING

        tab = (
            attr_cur.table[attr_cur.table[attribute] != flag_none] 
            if skip_none_q 
            else attr_cur.table
        )

        dict_map = sf.build_dict(tab[[attr_cur.key, attribute]]) 
        if clean_attribute_schema_q:
            dict_map = dict((k, mv.clean_element(v)) for k, v in dict_map.items())

        # return the dictionary?
        if return_type == dict:
            return dict_map

        # otherwise, order the output
        kv = [x for x in attr_cur.key_values if x in list(tab[attr_cur.key])]
        out = [dict_map.get(x) for x in kv]
        out = np.array(out) if (return_type == np.ndarray) else out

        return out



    def get_ordered_vars_by_nonprimary_category(self, #FIXED
        subsector_var: str,
        subsector_targ: str,
        flag_none: str = "none",
        return_type: str = "vars"
    ) -> Union[List[int], List[str]]:
        """
        Return a list of variables from one subsector that are ordered according 
            to a primary category (which the variables are mapped to) from 
            another subsector
        """
        # get var requirements for the variable subsector + the attribute for the target categories
        attr_vr_var = self.get_attribute_table(subsector_var, "variable_definitions")
        attr_targ = self.get_attribute_table(subsector_targ, "primary_category")
        pycat_targ = attr_targ.key

        # use the attribute table to map the category to the original variable
        tab_for_cw = attr_vr_var.table[attr_vr_var.table[pycat_targ] != flag_none]
        vec_var_targs = [mv.clean_element(x) for x in list(tab_for_cw[pycat_targ])]
        inds_varcats_to_cats = [vec_var_targs.index(x) for x in attr_targ.key_values]

        # check reutnr type
        if return_type not in ["inds", "vars"]:
            raise ValueError(f"Invalid return_type '{return_type}' in order_vars_by_category: valid types are 'inds', 'vars'.")
        
        vars_ordered = list(tab_for_cw["variable"])
        return_val = (
            inds_varcats_to_cats 
            if (return_type == "inds") 
            else [vars_ordered[x] for x in inds_varcats_to_cats]
        )
        
        return return_val
    


    def get_other_attribute(self, #FIXED
        attribute: str,
    ) -> AttributeTable:
        """
        Simplify retrieval of an `other` attribute object.
        """

        # get different table by different type
        dict_retrieve = self.dict_attributes.get(self.attribute_group_key_other)
        out = dict_retrieve.get(attribute)

        return out
    


    def get_region_list_filtered(self,
        regions: Union[List[str], str, None], 
        attribute_region: Union[AttributeTable, None] = None,
    ) -> List[str]:
        """
        Return a list of regions validly defined within Model Attributes.

        Function Arguments
        ------------------
        - regions: List of regions or string of region to run. If None, defaults 
            to configuration specification.

        Keyword Arguments
        -----------------
        - attribute_region: optional regional attribute to specify
        """

        attribute_region = (
            self.get_other_attribute("region")
            if (attribute_region is None) 
            else attribute_region
        )

        # format regions
        regions = [regions] if isinstance(regions, str) else regions
        regions = [x for x in regions if x in attribute_region.key_values] if isinstance(regions, List) else None
        if isinstance(regions, List):
            regions = None if (len(regions) == 0) else regions

        regions = self.configuration.get("region") if (regions is None) else regions

        return regions



    def get_sector_attribute(self, #FIXED
        sector: str,
        return_type: str,
    ) -> Union[float, int, str, None]:
        """
        Retrieve different attributes associated with a sector
        """

        # check sector specification
        self.check_sector(sector)
        attr_sec = self.get_sector_attribute_table()

        # initialize some key vars
        match_str_to = (
            "sector_to_" if 
            (return_type == self.table_name_attr_sector) 
            else "abbreviation_sector_to_"
        )
        
        maps = [x for x in attr_sec.field_maps.keys() if (match_str_to in x)]
        map_retrieve = f"{match_str_to}{return_type}"

        if not map_retrieve in maps:
            # warn user, but still allow a return
            valid_rts = sf.format_print_list([x.replace(match_str_to, "") for x in maps])
            warnings.warn(f"Invalid sector attribute '{return_type}'. Valid return type values are:{valid_rts}")
            
            return None


        dict_map = attr_sec.field_maps.get("sector_to_abbreviation_sector")

        # set the key
        key = (
            sector 
            if (return_type == self.table_name_attr_sector) 
            else dict_map.get(sector)
        )
        
        sf.check_keys(attr_sec.field_maps[map_retrieve], [key])
        out = attr_sec.field_maps.get(map_retrieve)
        out = out.get(key) if out is not None else out

        return out
    


    def get_sector_attribute_table(self, #FIXED
    ) -> Union[AttributeTable, None]:
        """
        Retrieve the sector attribute table.
        """
        # retrieve some dictionaries
        dict_other = self.dict_attributes.get(self.attribute_group_key_other)
        if dict_other is None:
            return None

        attr_sector = dict_other.get(self.table_name_attr_sector)
        
        return attr_sector



    def get_sector_subsectors(self, #FIXED
        sector: str,
        return_type: str = "name",
    ) -> List[str]:
        """
        Return a list of subsectors by sector. 

        Set return_type = "name" to return the name or "abv"/"abbreviation" to 
            return 4-character subsector codes.
        """

        self.check_sector(sector)
        attr = self.get_subsector_attribute_table()

        field_ext = (
            "subsector"
            if return_type in ["name"]
            else attr.key
        )

        subsectors = list(
            sf.subset_df(
                attr.table,
                {
                    "sector": [sector]
                }
            )[field_ext]
        )

        return subsectors



    def get_sector_variables(self, #FIXED
        sector: str,
        **kwargs,
    ) -> Union[List[str], None]:
        """
        Return a list of all model variables associated with a sector
        
        **kwargs include "var_type", which can be used to obtain input or output
            variables
        """
        sector = self.check_sector(sector, throw_error = False)
        if sector is None:
            return None

        subsecs = self.get_sector_subsectors(sector)
        modvars = []
        for subsec in subsecs:
            modvars += self.get_subsector_variables(subsec, **kwargs)

        return modvars



    def get_subsector_attribute(self, #FIXED
        subsector: str,
        return_type: str
    ) -> Union[float, int, str, None]:
        """
        Retrieve different attributes associated with a subsector. Valid values 
            of return_type are:

            * abv_subsector
            * key_variable_definitions
            * pycategory_primary:
                return the primary category name associated with the subsector
                EXCLUDING the self.attribute_group_key_cat prependage (not
                variable schema element focused)
            * pycategory_primary_element:
                return the primary category name associated with the subsector
                INCLUDING the self.attribute_group_key_cat prependage
            * sector
            * subsector
        """

        # retrieve attribute tables
        attr_sector = self.get_sector_attribute_table()
        attr_subsector = self.get_subsector_attribute_table()
        if (attr_subsector is None) | (attr_sector is None):
            return None

        # if the primary category, simply get it and return it
        if return_type in ["pycategory_primary", "pycategory_primary_element"]:

            dict_map = attr_subsector.field_maps.get(f"subsector_to_{self.subsector_field_category_py}")
            out = dict_map.get(subsector) if isinstance(dict_map, dict) else None

            # modify if seeking the element
            if isinstance(out, str) & (return_type == "pycategory_primary_element"):
                out = f"{self.attribute_group_key_cat}_{out}"

            return out

    
        # otherwise, get abbreviation
        dict_subsec_to_abv = attr_subsector.field_maps.get(f"subsector_to_abbreviation_subsector")
        if not isinstance(dict_subsec_to_abv, dict):
            return None

        # return the subsector abbreviation?
        abv_subsector = dict_subsec_to_abv.get(subsector)
        if return_type == "abv_subsector":
            return abv_subsector
        
        # return the sector?
        dict_map_abv_to_sector = attr_subsector.field_maps.get(f"{attr_subsector.key}_to_sector") 
        sector = dict_map_abv_to_sector.get(dict_subsec_to_abv.get(subsector))
        if return_type == "sector":
            return sector

        # return the sector abbreviation?
        dict_sector_to_abv_sector = attr_sector.field_maps.get(f"sector_to_{attr_sector.key}")
        abv_sector = dict_sector_to_abv_sector.get(sector)
        if return_type == "abv_sector":
            return abv_sector

        # return the variable definition key?
        key_variable_definition = f"{abv_sector}_{abv_subsector}"
        if return_type == "key_variable_definitions":
            return key_variable_definition

        # temporary to support debugging
        if return_type in ["key_varreqs_all", "key_varreqs_partial"]:
            msg = f"""
            ERROR in get_subector_attribute: invalid key {return_type}--this key 
            type is not supported. Check source code and modify."
            """
            raise RuntimeError(msg)
        
        # warn user, but still allow a return
        valid_rts = [
            "abv_sector",
            "abv_subsector",
            "key_variable_definitions",
            "pycategory_primary",
            "sector"
        ]

        valid_rts = sf.format_print_list(valid_rts)
        msg = f"""
        Invalid subsector attribute '{return_type}'. Valid return type values 
        are:{valid_rts}
        """
        warnings.warn(msg)

        return None
    


    def get_subsector_attribute_table(self, #FIXED
    ) -> Union[AttributeTable, None]:
        """
        Retrieve the subsector attribute table.
        """
        # retrieve some dictionaries
        dict_other = self.dict_attributes.get(self.attribute_group_key_other)
        if dict_other is None:
            return None

        attr_subsector = dict_other.get(self.table_name_attr_subsector)
        
        return attr_subsector
    


    def get_subsector_emission_total_field(self, #FIXED
        subsector: str,
        emission_total_schema_prepend: str = "emission_co2e_subsector_total",
    ) -> str:
        """
        Specify the aggregate emission field added to each subsector output 
            data frame
        """
        # add subsector abbreviation
        fld_nam = self.get_subsector_attribute(subsector, "abv_subsector")
        fld_nam = f"{emission_total_schema_prepend}_{fld_nam}"
        
        return fld_nam



    def get_subsector_variables(self, #FIXED
        subsector: str,
        var_type = None,
    ) -> list:
        """
        Get all variables associated with a subsector (will not function if
            there is no primary category)
        """

        # initialize output list, dictionary of variable to categorization (all or partial), and loop
        vars_by_subsector = []

        for k, v in self.dict_variables.items():
            
            if v.get_property("subsector") != subsector:
                continue
            
            # get type
            v_type = v.get_property("variable_type")
            v_type = v_type.lower() if isinstance(v_type, str) else ""

            append_q = (
                (v_type == var_type.lower()) 
                if isinstance(var_type, str) 
                else True
            )

            vars_by_subsector.append(k) if append_q else None

        return vars_by_subsector



    def get_time_periods(self, #FIXED
    ) -> tuple:
        """
        Get all time periods defined in SISEPUEDE. Returns a tuple of the form 
            (time_periods, n), where:

            * time_periods is a list of all time periods
            * n is the number of defined time periods
        """
        attr_tp = self.get_dimensional_attribute_table(self.dim_time_period)
        time_periods = attr_tp.key_values
        n_time_periods = attr_tp.n_key_values

        return time_periods, n_time_periods



    def get_time_period_years(self, #FIXED
        field_year: str = "year",
    ) -> list:
        """
        Get a list of all years (as integers) associated with time periods in 
            SISEPUEDE. Returns None if no years are defined.
        """
        attr_tp = self.get_dimensional_attribute_table(self.dim_time_period)

        # initialize output years
        all_years = None
        if field_year in attr_tp.table.columns:
            all_years = sorted(list(set(attr_tp.table[field_year])))
            all_years = [int(x) for x in all_years]

        return all_years
    


    def get_unit(self, #FIXED
        unit: str,
        return_type: str = "unit",
    ) -> AttributeTable:
        """
        Simplify retrieval of a unit object. Set return_type to "unit" or
            "attribute_table"
        """

        # check input type
        valid_types = ["unit", "attribute_table"]
        return_type = "unit" if return_type not in valid_types else return_type

        # get different table by different type
        dict_retrieve = self.dict_attributes.get(self.attribute_group_key_unit)
        out = dict_retrieve.get(unit)
        
        out = (
            out.attribute_table 
            if ((return_type == "attribute_table") & (out is not None))
            else out
        )

        return out
    


    def get_unit_attribute(self, #FIXED
        unit: str,
    ) -> AttributeTable:
        """
        Simplify retrieval of a unit attribute table. 
        """
        out = self.get_unit(unit, return_type = "attribute_table")
        return out


    
    def get_valid_categories(self, 
        categories: Union[List[str], str],
        subsector: str,
    ) -> Union[List[str], None]:
        """
        Check categories specified in list `categories`. Returns all valid 
            categories specified within subsector `subsector`. 
        
            * If none are found, returns None

        Function Arguments
        ------------------
        - categories: list of categories to check. If None, returns all valid 
            categories in subsector
        - subsector: SISEPUEDE subsector to check categories against. If not
            a valid subsector, returns None. 
        """

        subsec_check = self.check_subsector(subsector, throw_error_q = False, )
        if subsector != subsec_check:
            return None

        attr = self.get_attribute_table(subsector)

        # filter categories 
        categories = (
            [x for x in categories if x in attr.key_values]
            if sf.islistlike(categories)
            else (
                attr.key_values if (categories is None) else []
            )
        )
        categories = None if (len(categories) == 0) else categories

        return categories
    


    def get_variable_dict(self, #FIXED 
    ) -> None:
        """
        Initialize the units defined in the input attribute unit tables. Sets
            the following properties:

            * self.dict_variables

            NOTE: Variables can be accessed using the `get_variable()` 

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """
        
        ##  INITIALIZATION
        
        dict_vardefs = self.dict_variable_definitions
        dict_variables = {}
        
        # get subsector
        attr_subsector = (
            self.dict_attributes
            .get(self.attribute_group_key_other)
            .get(self.table_name_attr_subsector)
        )
        
        # get dicts
        dict_subsector_abv_to_subsector = attr_subsector.field_maps.get(
            f"{attr_subsector.key}_to_subsector"
        )
        dict_subsector_abv_to_sector = attr_subsector.field_maps.get(
            f"{attr_subsector.key}_to_sector"
        )
        dict_subsector_abv_to_pycat = attr_subsector.field_maps.get(
            f"{attr_subsector.key}_to_{self.subsector_field_category_py}"
        )

        
        for k, v in dict_vardefs.items():

            # initialize table
            tab = v.table
            
            # get sector/subsector specs
            sector_abv, subsector_abv = k.split("_")
            sector = dict_subsector_abv_to_sector.get(subsector_abv)
            subsector = dict_subsector_abv_to_subsector.get(subsector_abv)
            
            attr_cats = self.get_attribute_table(subsector)

            # dictionary to pass to each variable
            dict_sector_info = {
                "sector": sector,
                "sector_abv": sector_abv,
                "subsector": subsector,
                "subsector_abv": subsector_abv,
            }

            # iterate to build variable objects
            for i, row in tab.iterrows():

                dict_row = row.to_dict()
                dict_row.update(dict_sector_info)
                
                modvar = mv.ModelVariable(
                    dict_row,
                    attr_cats,
                )
                
                dict_variables.update({modvar.name: modvar})
        
        return dict_variables



    def get_var_dicts_by_shared_category(self, #FIXED
        subsector: str,
        category_pivot: str,
        fields_to_filter_on: list,
    ) -> dict:
        """
        Retrieve a dictionary that maps variables to each other based on shared 
            categories within a subsector
        """
        dict_out = {}

        # check attribute table
        attr_table = self.get_attribute_table(subsector, "variable_definitions")
        if attr_table is None:
            return None

        # get columns available in the data
        cols = list(set(attr_table.table.columns) & set(fields_to_filter_on))
        if (len(cols) > 0 & (category_pivot in attr_table.table.columns)):
            for field in cols:
                df_tmp = (
                    attr_table.table[
                        attr_table.table[field] == 1
                    ][
                        [category_pivot, "variable"]
                    ]
                    .copy()
                )

                # clean and build dictionary
                df_tmp[category_pivot] = df_tmp[category_pivot].apply(clean_schema)
                dict_map = sf.build_dict(df_tmp[[category_pivot, "variable"]])

                dict_out.update({field: dict_map})


        # next, loop over available combinations to build cross dictionaries
        dict_mapping = {}
        keys_to_pair = list(dict_out.keys())

        for pair in list(itertools.combinations(keys_to_pair, 2)):

            # get keys from dict and set keys for dict_mapping
            key_1 = pair[0]
            key_2 = pair[1]
            key_new = f"{key_1}_to_{key_2}"
            key_new_rev = f"{key_2}_to_{key_1}"

            # categories available in both dictionaries are used to update the dict_mapping
            shared_cats = list(set(dict_out[key_1]) & set(dict_out[key_2]))
            dict_mapping.update({
                key_new: dict([(dict_out[key_1][x], dict_out[key_2][x]) for x in shared_cats]),
                key_new_rev: dict([(dict_out[key_2][x], dict_out[key_1][x]) for x in shared_cats])
            })

        return dict_mapping



    def merge_array_var_partial_cat_to_array_all_cats(self,#VISIT
        array_vals: np.ndarray,
        modvar: str,
        missing_vals: float = 0.0,
        output_cats: Union[list, None] = None,
        output_subsec: Union[str, None] = None
    ) -> np.ndarray:
        """
        Reformat a partial category array (with partical categories along 
            columns) to place columns appropriately for a full category array. 
            Useful for simplifying matrix operations between variables.

        Function Arguments
        ------------------
        - array_vals: input array of data with column categories
        - modvar: the variable associated with the *input* array. This is used 
            to identify which categories are represented in the array's columns. 
            If None, then output_cats and output_subsec must be specified.
        
        Keyword Arguments
        -----------------
        - missing_vals: values to set for categories not in array_vals. Default 
            is 0.0.
        - output_cats: vector of categories associated with the output variable. 
            Only used if modvar == None. The combination of 
            output_cats + output_subsec provide a manual override to the modvar 
            option.
        - output_subsec: output subsector. Default is None. Only used if 
            modvar is None. The combination of output_cats + output_subsec 
            provide a manual override to the modvar option.
        """

        # check inputs
        if (modvar is None) and any([(x is None) for x in [output_cats, output_subsec]]):
            msg = f"""
            Error in input specification. If modvar == None, then output_cats 
            and output_subsec cannot be None.
            """
            raise ValueError(msg)
        
        if not sf.isnumber(missing_vals):
            msg = f"""
            Error in input specification of missing_vals: missing_vals should be 
            a floating point number of integer.
            """
            raise ValueError(msg)

        # get subsector/categories information
        if modvar is not None:
            # check variable first
            if modvar not in self.all_variables:
                raise ValueError(f"Invalid model variable '{modvar}' found in get_variable_characteristic.")

            subsector = self.get_variable_subsector(modvar)
            attr_subsec = self.get_attribute_table(subsector)
            cat_restriction_type = self.dict_model_variable_to_category_restriction.get(modvar)

        else:
            subsector = output_subsec
            attr_subsec = self.get_attribute_table(subsector)
            cat_restriction_type = None

            # check that all categories are defined
            if not set(output_cats).issubset(set(attr_subsec.key_values)):
                invalid_values = sf.format_print_list(list(set(output_cats) - set(attr_subsec.key_values)))
                msg = f"""
                Error in merge_array_var_partial_cat_to_array_all_cats: Invalid 
                categories {invalid_values} specified for subsector {subsector} 
                in output_cats.
                """
                raise ValueError(msg)

            # check that all categories are unique
            if len(set(output_cats)) != len(output_cats):
                msg = f"""
                Error in merge_array_var_partial_cat_to_array_all_cats: 
                Categories specified in output_cats are not unique. Check that 
                categories are unique.
                """
                raise ValueError(msg)

        # return the array if all categories are specified
        if cat_restriction_type == "all":
            return array_vals

        else:
            array_default = np.ones((len(array_vals), attr_subsec.n_key_values))*missing_vals
            cats = self.get_variable_categories(modvar) if (type(modvar) != type(None)) else output_cats
            inds_cats = [attr_subsec.get_key_value_index(x) for x in cats]
            inds = np.repeat([inds_cats], len(array_default), axis = 0)
            np.put_along_axis(array_default, inds, array_vals, axis = 1)

            return array_default



    def reduce_all_cats_array_to_partial_cat_array(self, #VISIT
        array_vals: np.ndarray, 
        modvar: str
    ) -> np.ndarray:
        """
        Reduce an all category array (with all categories along columns) to 
            columns associated with the variable modvar. Inverse of 
            merge_array_var_partial_cat_to_array_all_cats.

        Function Arguments
        ------------------
        - array_vals: input array of data with column categories
        - modvar: the variable associated with the desired *output* array. This 
            is used to identify which categories should be selected.
        """

        # check variable first
        if modvar not in self.all_variables:
            raise ValueError(f"Invalid model variable '{modvar}' found in get_variable_characteristic.")

        subsector = self.get_variable_subsector(modvar)
        attr_subsec = self.get_attribute_table(subsector)
        cat_restriction_type = self.dict_model_variable_to_category_restriction[modvar]

        if cat_restriction_type == "all":
            return array_vals
        else:
            cats = self.get_variable_categories(modvar)
            inds_cats = [attr_subsec.get_key_value_index(x) for x in cats]
            return array_vals[:, inds_cats]




    #########################################################################
    #    QUICK RETRIEVAL OF FUNDAMENTAL TRANSFORMATIONS (GWP, MASS, ETC)    #
    #########################################################################

    def get_unit_equivalent(self, #FIXED
        unit_type: str,
        unit: str,
        unit_to_match: Union[str, None],
        config_str: str,
        stop_on_error: bool = True,
    ) -> Union[float, None]:
        """
        For a given unit, get the scalar to convert to units unit_to_match. 
            Used for area, energy, length, mass, monetary, power, volume, 
            and other conversions.

        Function Arguments
        ------------------
        - unit_type: the unit name, passed to self.get_unit() (e.g., "area")
        - unit: a unit from a specified unit dimension (e.g., "km")
        - unit_to_match: A unit value to match unit to. The scalar `a` that is 
            returned is multiplied by unit, i.e., 
                unit*a = unit_to_match
            If None (default), uses the configuration value
        - config_str: the configuration parameter associated with the defualt 
            unit
        
        Keyword Arguments
        -----------------
        - stop_on_error: throw an error on bad unit? If False and a unit is 
            invalid, returns None
        """

        unit_to_match = (
            self.configuration.get(config_str) 
            if (unit_to_match is None) 
            else unit_to_match
        )
        if unit_to_match is None:
            if stop_on_error:
                raise KeyError(f"Invalid configuration string '{config_str}' specified in get_unit_equivalent.")
            return None


        # get the units object
        units_obj = self.get_unit(unit_type)
        if units_obj is None:
            if stop_on_error:
                valid_dims = sf.format_print_list(self.all_units)
                msg = f"""
                Invalid unit_type '{unit_type}' specified in 
                get_unit_equivalent. Valid dimensions are: {valid_dims}"
                """
                raise KeyError(msg)

            return None

        # convert
        out = units_obj.convert(unit, unit_to_match)

        return out



    def get_area_equivalent(self, #FIXED
        area: str, 
        area_to_match: str = None,
    ) -> float:
        """
        For a given area unit *area*, get the scalar to convert to units 
            *area_to_match*

        Function Arguments
        ------------------
        - area: a unit of area defined in the unit_area attribute table

        Keyword Arguments
        -----------------
        - area_to_match: Default is None. A unit of area to match. The scalar 
            `a` that is returned is multiplied by area, i.e., 
            area*a = area_to_match. If None (default), return the configuration 
            default.
        """
        out = self.get_unit_equivalent(
            "area",
            area,
            area_to_match,
            "area_units",
        )

        return out



    def get_energy_equivalent(self, #FIXED
        energy: str, 
        energy_to_match: str = None,
    ) -> float:

        """
        For a given energy unit *energy*, get the scalar to convert to units 
            *energy_to_match*

        Function Arguments
        ------------------
        - energy: a unit of energy defined in the unit_energy attribute table

        Keyword Arguments
        -----------------
        - energy_to_match: Default is None. A unit of energy to match. The 
            scalar a that is returned is multiplied by energy, i.e., 
            energy*a = energy_to_match. If None (default), return the 
            configuration default.
        """

        out = self.get_unit_equivalent(
            "energy",
            energy,
            energy_to_match,
            "energy_units",
        )

        return out



    def get_energy_power_swap(self, #FIXED
        input_unit: str,
        time_period: str = "annualized",
    ) -> str:
        """
        Enter an energy unit E to retrieve the equivalent unit of power P so 
            that P*T = E OR enter a power unit P to retrieve the equivalent 
            energy unit E so that E/T = P, where

            T = one year if time_period == "annualized"
            T = one hour if time_period == "hourly"

        Function Arguments
        ------------------
        - input_unit: input unit to enter. Must be a valid power or energy unit
        - time_period: "annual" or "hourly"
        """
        # check time period and initialize units
        time_period = "annualized" if (time_period not in ["annualized", "hourly"]) else time_period
        unit_energy = self.get_unit("energy")
        unit_power = self.get_unit("power")

        # check energy, then power and specify which it is
        units = unit_energy.get_unit_key(input_unit)
        input_type = "energy" if (units is not None) else None

        if input_type is None:
            units = unit_power.get_unit_key(input_unit) 
            input_type = "power" if (units is not None) else None

        if units is None:
            return None


        ##  CONTINUE WITH SWAP

        input_unit = units

        if input_type == "energy":
            field_equivalent = f"{time_period}_unit_power_equivalent"
            out = unit_energy.get_attribute(input_unit, field_equivalent)
            out = unit_power.get_unit_key(mv.clean_element(out)) # makes sure output is specified in power

        elif input_type == "power":
            field_equivalent = f"{time_period}_unit_energy_equivalent"
            out = unit_power.get_attribute(input_unit, field_equivalent)
            out = unit_energy.get_unit_key(mv.clean_element(out)) # makes sure output is specified in energy

        return out



    def get_gwp(self, #FIXED
        gas: str, 
        gwp: Union[int, None] = None,
    ) -> float:
        """
        For a given gas, get the scalar to convert to CO2e using the specified 
            global warming potential *gwp*

        Function Arguments
        ------------------
        - gas: a gas defined in the emission_gas attribute table
        
        Keyword Arguments
        -----------------
        - gwp: Default is None. Global warming potential of "gas" over "gwp" 
            time period (gwp is a number of years, e.g., 20, 100, 500).
        """
        # none checks
        unit_gas = self.get_other_attribute("emission_gas")
        gas = unit_gas.get_unit_key(gas)
        if gas is None:
            return None
        
        gwp = (
            int(self.configuration.get("global_warming_potential"))
            if not sf.isnumber(gwp)
            else gwp
        )

        # get attribute
        attr_gas = unit_gas.attribute_table
        key_dict = f"emission_gas_to_global_warming_potential_{gwp}"

        # check that the target energy unit is defined
        dict_map = attr_gas.field_maps.get(key_dict)
        if dict_map is None:
            valid_gwps = sf.format_print_list(self.configuration.valid_gwp)
            msg = f"Invalid GWP '{gwp}': defined global warming potentials are {valid_gwps}."
            raise KeyError(msg)

        # check gas and return if valid
        out = dict_map.get(gas)
        if out is None:
            valid_gasses = sf.format_print_list(self.dict_attributes["emission_gas"].key_values)
            raise KeyError(f"Invalid gas '{gas}': defined gasses are {valid_gasses}.")

        return out



    def get_length_equivalent(self, #FIXED
        length: str, 
        length_to_match: Union[str, None] = None,
    ) -> float:
        """
        for a given length unit *length*, get the scalar to convert to units 
            *length_to_match*

        Function Arguments
        ------------------
        - length: a unit of length defined in the unit_length attribute table
        
        Keyword Arguments
        -----------------
        - length_to_match: Default is None. A unit of length to match. The 
            scalar a that is returned is multiplied by length, i.e., 
            length*a = length_to_match. If None (default), return the 
            configuration default.
        """
        out = self.get_unit_equivalent(
            "length",
            length,
            length_to_match,
            "length_units",
        )

        return out



    def get_mass_equivalent(self, #FIXED
        mass: str, 
        mass_to_match: Union[str, None] = None,
    ) -> float:
        """
        For a given mass unit *mass*, get the scalar to convert to units 
            *mass_to_match*

        Function Arguments
        ------------------
        - mass: a unit of mass defined in the unit_mass attribute table

        Keyword Arguments
        -----------------
        - mass_to_match: Default is None. A unit of mass to match. The scalar a 
            that is returned is multiplied by mass, i.e., 
            mass*a = mass_to_match. If None (default), return the configuration 
            default.
        """
        out = self.get_unit_equivalent(
            "mass",
            mass,
            mass_to_match,
            "emissions_mass",
        )

        return out



    def get_monetary_equivalent(self, #FIXED
        monetary: str, 
        monetary_to_match: str = None
    ) -> float:
        """
        For a given monetary unit *monetary*, get the scalar to convert to units 
            *monetary_to_match*

        Function Arguments
        ------------------
        - monetary: a unit of monetary defined in the unit_monetary attribute 
            table

        Keyword Arguments
        -----------------
        - monetary_to_match: Default is None. A unit of monetary to match. The 
            scalar a that is returned is multiplied by monetary, i.e., 
            monetary*a = monetary_to_match. If None (default), return the 
            configuration default.
        """
        out = self.get_unit_equivalent(
            "monetary",
            monetary,
            monetary_to_match,
            "monetary_units",
        )

        return out



    def get_power_equivalent(self, #FIXED
        power: str, 
        power_to_match: str = None
    ) -> float:
        """
        For a given power unit *power*, get the scalar to convert to units 
            *power_to_match*

        Function Arguments
        ------------------
        - power: a unit of power defined in the unit_power attribute table

        Keyword Arguments
        -----------------
        - power_to_match: Default is None. A unit of power to match. The scalar 
            a that is returned is multiplied by power, i.e., 
            power*a = power_to_match. If None (default), return the 
            configuration default.
        """
        out = self.get_unit_equivalent(
            "power",
            power,
            power_to_match,
            "power_units",
        )

        return out
    


    def get_variable(self,
        modvar: Union[str, mv.ModelVariable],
    ) -> Union[mv.ModelVariable, None]:
        """
        Get a model variable
        """
        out = (
            modvar
            if mv.is_model_variable(modvar)
            else self.dict_variables.get(modvar)
        )

        return out



    def get_volume_equivalent(self, #FIXED
        volume: str, 
        volume_to_match: str = None
    ) -> float:
        """
        For a given volume unit *volume*, get the scalar to convert to units 
            *volume_to_match*

        Function Arguments
        ------------------
        - volume: a unit of volume defined in the unit_volume attribute table

        Keyword Arguments
        -----------------
        - volume_to_match: Default is None. A unit of volume to match. The 
            scalar a that is returned is multiplied by volume, i.e., 
            volume*a = volume_to_match. If None (default), return the 
            configuration default.
        """
        out = self.get_unit_equivalent(
            "volume",
            volume,
            volume_to_match,
            "volume_units",
        )

        return out



    def get_scalar(self, #FIXED
        modvar: Union[str, mv.ModelVariable],
        return_type: str = "total",
    ) -> float:
        """
        Get the scalar a to convert units from modvar to configuration units,
            i.e.

            modvar_units * a = configuration_units
        """

        # check return type
        valid_rts = self.valid_return_types_unit_conversion
        if return_type not in valid_rts:
            tps = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return type '{return_type}' in get_scalar: valid types are {tps}.")


        ##  INITIALIZE OUTPUT SCALAR AND MULTIPLY AS NEEDED

        scalar_out = 1

        if return_type == "area":
            area = self.get_variable_characteristic(modvar, self.varchar_str_unit_area)
            scalar_out *= (
                self.get_area_equivalent(area.lower()) 
                if isinstance(area, str) 
                else 1
            )
        
        if return_type == "energy":
            energy = self.get_variable_characteristic(modvar, self.varchar_str_unit_energy)
            scalar_out *= (
                self.get_energy_equivalent(energy.lower()) 
                if isinstance(energy, str) 
                else 1
            )

        if return_type in ["gas", "total"]: # total is used for scaling gas & mass to co2e in proper units
            gas = self.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
            scalar_out *= (
                self.get_gwp(gas.lower()) 
                if isinstance(gas, str) 
                else 1
            )

        if return_type == "length":
            length = self.get_variable_characteristic(modvar, self.varchar_str_unit_length)
            scalar_out *= (
                self.get_length_equivalent(length.lower()) 
                if isinstance(length, str) 
                else 1
            )

        if return_type in ["mass", "total"]: # total is used for scaling gas & mass to co2e in proper units
            mass = self.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
            scalar_out *= (
                self.get_mass_equivalent(mass.lower()) 
                if isinstance(mass, str) 
                else 1
            )

        if return_type == "monetary":
            monetary = self.get_variable_characteristic(modvar, self.varchar_str_unit_monetary)
            scalar_out *= (
                self.get_monetary_equivalent(monetary.lower()) 
                if isinstance(monetary, str) 
                else 1
            )

        if return_type == "power":
            power = self.get_variable_characteristic(modvar, self.varchar_str_unit_power)
            scalar_out *= (
                self.get_power_equivalent(power.lower()) 
                if isinstance(power, str) 
                else 1
            )

        if return_type == "volume":
            volume = self.get_variable_characteristic(modvar, self.varchar_str_unit_volume)
            scalar_out *= (
                self.get_volume_equivalent(volume.lower()) 
                if isinstance(volume, str) 
                else 1
            )


        return scalar_out



    def check_projection_input_df(self,
        df_project: pd.DataFrame,
        interpolate_missing_q: bool = True,
        strip_dims: bool = True,
        drop_invalid_time_periods: bool = True,
        override_time_periods: bool = False,
    ) -> tuple:
        """
        Check the projection input dataframe and (1) return time periods 
            available, (2) a dictionary of scenario dimenions, and (3) an 
            interpolated data frame if there are missing values.
        """
        # check for required fields
        sf.check_fields(df_project, [self.dim_time_period])

        # field initialization
        fields_dat = [x for x in df_project.columns if (x not in self.sort_ordered_dimensions_of_analysis)]
        fields_dims_notime = [x for x in self.sort_ordered_dimensions_of_analysis if (x != self.dim_time_period) and (x in df_project.columns)]

        # check that there's only one primary key included (or one dimensional vector)
        if len(fields_dims_notime) > 0:
            df_fields_dims_notime = df_project[fields_dims_notime].drop_duplicates()
            if len(df_fields_dims_notime) > 1:
                raise ValueError(f"Error in project: the input data frame contains multiple dimensions of analysis. The project method is restricted to a single dimension of analysis. The following dimensions were found:\n{df_fields_dims_notime}")
            else:
                dict_dims = dict(zip(fields_dims_notime, list(df_fields_dims_notime.iloc[0])))
        else:
            dict_dims = {}

        # next, check time periods
        df_time = self.dict_attributes["dim_time_period"].table[[self.dim_time_period]]
        set_times_project = set(df_project[self.dim_time_period])
        set_times_defined = set(df_time[self.dim_time_period])
        set_times_keep = (
            set_times_project & set_times_defined
            if not override_time_periods
            else set_times_project 
        )

        # raise errors if issues occur
        if (not set_times_project.issubset(set_times_defined)) and (not drop_invalid_time_periods):
            sf.check_set_values(set_times_project, set_times_defined, " in projection dataframe. Set 'drop_invalid_time_periods = True' to drop these time periods and proceed.")

        # intiialize interpolation_q and check for consecutive time steps to determine if a merge + interpolation is needed
        interpolate_q = False
        
        if (set_times_keep != set(range(min(set_times_keep), max(set_times_keep) + 1))):
            if not interpolate_missing_q:
                msg = f"""
                Error in specified times: some time periods are missing and 
                interpolate_missing_q = False. Modeling will not proceed. Set 
                interpolate_missing_q = True to interpolate missing values.
                """
                raise ValueError()

            else:
                set_times_keep = set(range(min(set_times_keep), max(set_times_keep) + 1))
                df_project = pd.merge(
                    df_time[df_time[self.dim_time_period].isin(set_times_keep)],
                    df_project,
                    how = "left",
                    on = [self.dim_time_period]
                )
                interpolate_q = True

        elif len(df_project[fields_dat].dropna()) != len(df_project):
            interpolate_q = True

        # set some information on time series
        projection_time_periods = list(set_times_keep)
        projection_time_periods.sort()
        n_projection_time_periods = len(projection_time_periods)

        # format data frame
        df_project = df_project.interpolate() if interpolate_q else df_project
        df_project = df_project[df_project[self.dim_time_period].isin(set_times_keep)]
        df_project.sort_values(by = [self.dim_time_period], inplace = True)
        
        df_project = (
            df_project[[self.dim_time_period] + fields_dat] 
            if strip_dims 
            else df_project[fields_dims_notime + [self.dim_time_period] + fields_dat]
        )

        out = (dict_dims, df_project, n_projection_time_periods, projection_time_periods)

        return out


    def transfer_df_variables(self,
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        variables_transfer: list,
        fields_index: Union[list, None] = None,
        join_type: str = "concatenate",
        overwrite_targets: bool = False,
        stop_on_error: bool = True
    ) -> pd.DataFrame:
        """
        Transfar SISEPUEDE model variables from source data frame to target data 
            frame.

        Function Arguments
        ------------------
        - df_target: data frame to receive variables
        - df_source: data frame to send variables from
        - variables_transfer: list of SISEPUEDE model variables to transfer from 
            source to target

        Keyword Arguments
        -----------------
        - fields_index: index fields shared by each data frame
        - join_type: valid values are "concatenate" and "merge". If index 
            field(s) ordering is the same, concatenation is recommended.
        - overwrite_targets: overwrite existing model variable fields in 
            df_target if they exist? Default is false.
        - stop_on_error: stop the transfer on an error. If False, variables that 
            are not available in df_source will be ignored.

        Notes
        -----
        * Assumes that variable schema are unique for each model variable
        """
        dfs_extract = [df_target]
        fields_index = [] if (fields_index is None) else fields_index
        variables_transfer = list(set(variables_transfer))

        for var_int in variables_transfer:
            df_ext = None

            try:
                df_ext = self.get_optional_or_integrated_standard_variable(
                    df_source, 
                    var_int, 
                    None,
                )

            except Exception as e:
                if stop_on_error:
                    msg = f"""
                    Error in transfer_df_variables: 
                    get_optional_or_integrated_standard_variable returned '{e}'.
                    """
                    raise RuntimeError(msg)
            
            if df_ext is None:
                continue

            # drop variables that are already in the target df
            varlist = self.build_variable_fields(var_int)
            vars_to_drop = list(
                set(df_ext[1].columns) & set(dfs_extract[0].columns) & set(varlist)
            )

            if len(vars_to_drop) > 0:
                if overwrite_targets:
                    dfs_extract[0].drop(
                        vars_to_drop, 
                        axis = 1, 
                        inplace = True,
                    )

                else:
                    df_ext[1].drop(
                        vars_to_drop, 
                        axis = 1, 
                        inplace = True,
                    )

            dfs_extract.append(df_ext[1])

        out = sf.merge_output_df_list(
            dfs_extract, 
            self, 
            merge_type = join_type,
        )

        return out




    #########################################################
    #    VARIABLE REQUIREMENT AND MANIPULATION FUNCTIONS    #
    #########################################################

    def _add_specified_total_fields_to_emission_total(self,
        df_in: pd.DataFrame,
        varlist: list,
    ) -> None:
        """
        Add a total of emission fields that are specified. Inline function 
            (does not return).

        Function Arguments
        ------------------
        - df_in: Data frame with emission outputs to be aggregated
        - varlist: variables to include in the sum
        """
        #initialize dictionary
        dict_totals = {}
        dict_fields = {}
        # loop over variables to
        for var in varlist:
            subsec = self.get_variable_subsector(var, throw_error_q = False)
            if subsec is not None:
                array_cur = self.extract_model_variable(#
                    df_in, 
                    var, 
                    expand_to_all_cats = True, 
                    return_type = "array_base",
                )

                if subsec not in dict_totals.keys():
                    field_total = self.get_subsector_emission_total_field(subsec)
                    if (field_total in df_in.columns):
                        dict_totals.update({subsec: 0.0})
                        dict_fields.update({subsec: field_total})
                dict_totals[subsec] += array_cur
            else:
                warning(f"In _add_specified_total_fields_to_emission_total, subsector '{subsec}' not found. Skipping...")

        # next, update dataframe
        for subsec in dict_totals.keys():
            array_totals = np.sum(dict_totals[subsec], axis = 1)
            field_total = dict_fields[subsec]
            cur_emissions = np.array(df_in[field_total]) if (field_total in df_in.columns) else 0
            df_in[field_total] = cur_emissions + array_totals

        return None



    def add_subsector_emissions_aggregates(self,
        df_in: pd.DataFrame,
        list_subsectors: list,
        stop_on_missing_fields_q: bool = False,
        skip_non_emission_subsectors: bool = True,
    ) -> str:
        """
        Add a total of all emission fields (across those output variables 
            specified with $EMISSION-GAS$). Inline operation on DataFrame that
            returns the subsector total.

        Function Arguments
        ------------------
        - df_in: Data frame with emission outputs to be aggregated
        - list_subsectors: subsectors to apply totals to

        Keyword Arguments
        -----------------
        - stop_on_missing_fields_q: default = False. If True, will stop if any 
            component emission variables are missing.
        - skip_non_emission_subsectors: skip subsectors that don't generate
            emissions? Otherwise, adds a field without 0
        """

        list_subsectors = (
            [x for x in list_subsectors if x in self.emission_subsectors]
            if skip_non_emission_subsectors
            else list_subsectors
        )

        # loop over base subsectors
        for subsector in list_subsectors:

            vars_subsec = self.dict_model_variables_by_subsector.get(subsector)

            # add subsector abbreviation
            fld_nam = self.get_subsector_emission_total_field(subsector)
            flds_add = []
            for var in vars_subsec:
                var_type = self.get_variable_attribute(var, "variable_type").lower()
                gas = self.get_variable_characteristic(var, self.varchar_str_emission_gas)

                if (var_type == "output") and gas:
                    total_emission_modvars_by_gas = self.dict_gas_to_total_emission_modvars.get(gas)
                    if total_emission_modvars_by_gas is not None:
                        flds_add += (
                            self.dict_model_variables_to_variables.get(var) 
                            if var in self.dict_gas_to_total_emission_modvars.get(gas)
                            else []
                        )

            # check for missing fields; notify
            missing_fields = [x for x in flds_add if x not in df_in.columns]
            if len(missing_fields) > 0:
                str_mf = sf.print_setdiff(set(df_in.columns), set(flds_add))
                str_mf = f"Missing fields {str_mf}.%s"
                if stop_on_missing_fields_q:
                    raise ValueError(str_mf%(" Subsector emission totals will not be added."))

                warnings.warn(str_mf%(" Subsector emission totals will exclude these fields."))
            
            # update output fields
            keep_fields = [x for x in flds_add if x in df_in.columns]
            df_in[fld_nam] = df_in[keep_fields].sum(axis = 1)

        return fld_nam



    def exchange_year_time_period(self, #REVIEW: CALLS SHOULD BE SWAPPED TO USE sc.TimePeriods
        df_in: pd.DataFrame,
        field_year_new: str,
        series_time_domain: pd.core.series.Series,
        attribute_time_period: Union[AttributeTable, None] = None,
        field_year_in_attribute: str = "year",
        direction: str = "time_period_to_year",
    ) -> pd.DataFrame:
        """
        Add year field to a data frame if missing

        Function Arguments
        ------------------
        - df_in: input dataframe to add column to
        - field_year_new: field name to store year
        - series_time_domain: pandas series of time periods

        Keyword Arguments
        -----------------
        - attribute_time_period: AttributeTable mapping 
            ModelAttributes.dim_time_period to year field
        - field_year_in_attribute: field in attribute_time_period containing the 
            year
        - direction: which direction to map; acceptable values include:
            * time_period_to_year: convert a time period in the series to year 
                under field field_year_new (default)
            * time_period_as_year: enter the time period in the year field 
                field_year_new (used for NemoMod)
            * year_to_time_period: convert a year back to time period if there 
                is an injection
        """

        # check direction specification
        sf.check_set_values(
            [direction], 
            ["time_period_as_year", "time_period_to_year", "year_to_time_period"], 
            " in exchange_year_time_period."
        )

        # get time period attribute and initialize the output data frame
        attribute_time_period = self.get_dimensional_attribute_table(self.dim_time_period)
        df_out = df_in.copy()

        if (direction in ["time_period_as_year"]):
            df_out[field_year_new] = np.array(series_time_domain.copy())

        elif (direction in ["time_period_to_year", "year_to_time_period"]):
            key_fm = (
                f"{attribute_time_period.key}_to_{field_year_in_attribute}" 
                if (direction == "time_period_to_year") 
                else f"{field_year_in_attribute}_to_{attribute_time_period.key}"
            )

            dict_repl = attribute_time_period.field_maps.get(key_fm)
            if dict_repl is not None:
                df_out[field_year_new] = series_time_domain.replace(dict_repl)

        else:
            msg = f"""
            Invalid direction '{direction}' in exchange_year_time_period: 
            specify 'time_period_to_year' or 'year_to_time_period'.
            """
            raise ValueError(msg)

        return df_out



    def array_to_df(self,
        arr_in: np.ndarray,
        modvar: Union[str, mv.ModelVariable],
        include_scalars: bool = False,
        reduce_from_all_cats_to_specified_cats: bool = False,
    ) -> pd.DataFrame:
        """
        Convert an input np.ndarray into a data frame that has the proper 
            variable labels (ordered by category for the appropriate subsector)

        Function Arguments
        ------------------
        - arr_in: np.ndarray to convert to data frame. If entered as a vector, 
            it will be converted to a (n x 1) array, where n = len(arr_in)
        - modvar: the name of the model variable to use to name the dataframe OR
            the ModelVariable object

        Keyword Arguments
        -----------------
        - include_scalars: If True, will rescale to reflect emissions mass 
            correction.
        - reduce_from_all_cats_to_specified_cats: If True, the input data frame 
            is given across all categories and needs to be reduced to the set of 
            categories associated with the model variable (selects subset of 
            columns).
        """

        # get subsector and fields to name based on variable HEREHERE
        subsector = self.get_variable_subsector(modvar)
        fields = self.build_variable_fields(modvar)

        # transpose if needed
        arr_in = (
            np.array([arr_in]).transpose() 
            if (len(arr_in.shape) == 1) 
            else arr_in
        )

        # is the array that's being passed column-wise associated with all categories?
        if reduce_from_all_cats_to_specified_cats:
            attr = self.get_attribute_table(subsector)
            cats = self.get_variable_categories(modvar)
            indices = [attr.get_key_value_index(x) for x in cats]
            arr_in = arr_in[:, indices]

        scalar_em = 1
        scalar_me = 1

        if include_scalars:
            # get scalars
            gas = self.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
            mass = self.get_variable_characteristic(modvar, self.varchar_str_unit_mass)

            # will conver ch4 to co2e e.g. + kg to MT
            scalar_em = 1 if not gas else self.get_gwp(gas.lower())
            scalar_me = 1 if not mass else self.get_mass_equivalent(mass.lower())

        # raise error if there's a shape mismatch
        if len(fields) != arr_in.shape[1]:
            flds_print = sf.format_print_list(fields)

            msg = f"""
            f"Array shape mismatch for fields {flds_print}: the array only has 
            {arr_in.shape[1]} columns."
            """

            raise ValueError(msg)

        return pd.DataFrame(arr_in*scalar_em*scalar_me, columns = fields)



    def assign_keys_from_attribute_fields(self,#VISIT
        subsector: str,
        field_attribute: str,
        dict_assignment: dict,
        type_table: str = "categories",
        clean_field_vals: bool = True,
        clean_attr_key: bool = False,
    ) -> tuple:
        """
        Assign key_values that are associated with a secondary category. Use 
            matchstrings defined in dict_assignment to create an output 
            dictionary.

            Returns a tuple of following structure:
            * tuple: (dict_out, vars_unassigned)
            * dict_out: takes form
                {
                    key_value: {
                        assigned_dictionary_key: variable_name, 
                        ...
                    },
                    ...
                }

        Function Arguments
        ------------------
        - dict_assignment: dict. {match_str: assigned_dictionary_key} map a 
            variable match string to an assignment
        - field_attribute: field in the attribute table to use to split elements
        - subsector: the subsector to pull the attribute table from
        
        Keyword Arguments
        -----------------
        - clean_attr_key: default is False. Apply clean_schema() to the keys 
            that are assigned to the output dictionary (e.g., 
            clean_schema(variable_name))
        - clean_field_vals: default = True. Apply clean_schema() to the values 
            found in attr_subsector[field_attribute]?
        - type_table: default = "categories". Represents the type of attribute
            table; valid values are 'categories', 'varreqs_all', and 
            'varreqs_partial'
        """

        # check the subsector and type specifications
        self.check_subsector(subsector)
        dict_valid_types_to_attribute_keys = {
            "categories": "primary_category", #"primary_category"
            "variables": "variable_definitions", #"variable_definitions"
        }
        
        attr_type_retrieve = dict_valid_types_to_attribute_keys.get(type_table)
        if attr_type_retrieve is None:
            str_valid_types = sf.format_print_list(
                list(dict_valid_types_to_attribute_keys.keys())
            )

            msg = f"""
            Invalid type_primary '{type_primary}' specified. Valid values are 
            '{str_valid_types}'.
            """
            raise ValueError(msg)


        # retrieve the attribute table and check the field specification
        
        attr_subsector = self.get_attribute_table(subsector, attr_type_retrieve)
        sf.check_fields(attr_subsector.table, [field_attribute])

        # get the unique field values
        all_field_values = list(set(
            self.get_ordered_category_attribute(
                subsector,
                field_attribute,
                skip_none_q = True,
                attr_type = attr_type_retrieve,
            )
        ))
        all_field_values.sort()

        # loop to build the output dictionaries
        dict_out = {}
        dict_vals_unassigned = {}

        for val in all_field_values:
            
            dict_out_key = clean_schema(val) if clean_field_vals else val
            subsec_keys = attr_subsector.table[
                attr_subsector.table[field_attribute] == val
            ][attr_subsector.key]

            # loop over the keys to assign
            dict_assigned = {}
            for subsec_key in subsec_keys:
                for k in dict_assignment.keys():
                    if k not in subsec_key:
                        continue

                    val_assigned = clean_schema(subsec_key) if clean_attr_key else subsec_key
                    dict_assigned.update({dict_assignment[k]: val_assigned})
            
            vals_unassigned = list(set(dict_assignment.values()) - set(dict_assigned.keys()))
            dict_out.update({dict_out_key: dict_assigned})
            dict_vals_unassigned.update({dict_out_key: vals_unassigned})

        return dict_out, dict_vals_unassigned



    def get_vars_by_assigned_class_from_akaf(self,
        dict_in: dict,
        var_class: str
    ) -> list:
        """
        Support function for assign_keys_from_attribute_fields (akaf)
        """
        out = [
            x.get(var_class) for x in dict_in.values() 
            if (x.get(var_class) is not None)
        ]

        return out



    def build_default_sampling_range_df(self,
    ) -> pd.DataFrame:
        """
        Build a sampling range dataframe from defaults contained in
            AttributeTables.
        """
        df_out = []
        # set field names
        pd_max = max(self.get_time_periods()[0])
        field_max = f"max_{pd_max}"
        field_min = f"min_{pd_max}"

        for sector in self.all_sectors:
            subsectors_cur = list(
                sf.subset_df(
                    self.dict_attributes[self.table_name_attr_subsector].table, 
                    {"sector": [sector]}
                )["subsector"]
            )

            for subsector in subsectors_cur:
                for variable in self.dict_model_variables_by_subsector[subsector]:
                    
                    # skip non-input variables
                    variable_type = self.get_variable_attribute(variable, "variable_type")
                    if (variable_type.lower() != "input"):
                        continue

                    max_ftp_scalar = self.get_variable_attribute(
                        variable, 
                        "default_lhs_scalar_maximum_at_final_time_period"
                    )
                    min_ftp_scalar = self.get_variable_attribute(
                        variable, 
                        "default_lhs_scalar_minimum_at_final_time_period"
                    )
                    mvs = self.dict_model_variables_to_variables[variable]

                    df_out.append(
                        pd.DataFrame(
                            {
                                "variable": mvs, 
                                field_max: [max_ftp_scalar for x in mvs], 
                                field_min: [min_ftp_scalar for x in mvs]
                            }
                        )
                    )

        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

        return df_out



    def build_variable_dataframe_by_sector(self,
        sectors_build: Union[List[str], str, None],
        df_trajgroup: Union[pd.DataFrame, None] = None,
        field_subsector: str = "subsector",
        field_variable: str = "variable",
        field_variable_trajectory_group: str = "variable_trajectory_group",
        include_simplex_group_as_trajgroup: bool = False,
        include_time_periods: bool = True,
        vartype: str = "input",
    ) -> pd.DataFrame:
        """
        Build a data frame of all variables long by subsector and variable.
            Optional includion of time_periods.

        Function Arguments
        ------------------
        - sectors_build: sectors to include subsectors for

        Keyword Arguments
        -----------------
        - df_trajgroup: optional dataframe mapping each field variable to 
            trajectory groups. 
            * Must contain field_subsector, field_variable, and 
                field_variable_trajectory_group as fields
            * Overrides include_simplex_group_as_trajgroup if specified and 
                conflicts occur
        - field_subsector: subsector field for output data frame
        - field_variable: variable field for output data frame
        - field_variable_trajectory_group: field giving the output variable
            trajectory group (only included if
            include_simplex_group_as_trajgroup == True)
        - include_simplex_group_as_trajgroup: include variable trajectory group 
            defined by Simplex Group in attribute tables?
        - include_time_periods: include time periods? If True, makes data frame
            long by time period
        - vartype: "input" or "output"
        """
        df_out = []
        sectors_build = self.get_sector_list_from_projection_input(sectors_build)

        # loop over sectors/subsectors to construct subsector and all variables
        for sector in sectors_build:
            subsectors = self.get_sector_subsectors(sector)

            for subsector in subsectors:
                modvars_cur = self.get_subsector_variables(
                    subsector,
                    var_type = vartype
                )

                vars_cur = sum([self.dict_model_variables_to_variables.get(x) for x in modvars_cur], [])
                df_out += [(subsector, x) for x in vars_cur]

        # convert to data frame and return
        fields_sort = [field_subsector, field_variable]
        df_out = pd.DataFrame(
            df_out,
            columns = fields_sort
        )

        # include simplex group as a trajectory group?
        if include_simplex_group_as_trajgroup:
            col_new = list(df_out[field_variable].apply(self.get_simplex_group))
            df_out[field_variable_trajectory_group] = col_new
            df_out[field_variable_trajectory_group] = df_out[field_variable_trajectory_group].astype("float")
        
        # use an exogenous specification of variable trajectory groups?
        if isinstance(df_trajgroup, pd.DataFrame):
            
            fields_sort_with_tg = fields_sort + [field_variable_trajectory_group]

            if (
                set([field_variable, field_variable_trajectory_group])
                .issubset(set(df_trajgroup.columns))
            ):
                df_trajgroup.dropna(
                    subset = [field_variable, field_variable_trajectory_group],
                    how = "any",
                    inplace = True
                )
                # if the trajgroup is already defined, split into 
                # - variables that are assigned by not in df_trajgroup
                # - variables that are assigned and in df_trajgroup
                if (field_variable_trajectory_group in df_out.columns):
                    
                    vars_to_assign = sorted(list(df_trajgroup[field_variable].unique()))
                    tgs_to_assign = sorted(list(df_trajgroup[field_variable_trajectory_group].unique()))
                    # split into values to keep (but re-index) and those to overwrite
                    df_out_keep = df_out[
                        ~df_out[field_variable]
                        .isin(vars_to_assign)
                    ]
                    df_out_overwrite = (
                        df_out[
                            df_out[field_variable]
                            .isin(vars_to_assign)
                        ]
                        .drop(
                            [field_variable_trajectory_group],
                            axis = 1
                        )
                    )

                    # get values to reindex and apply
                    dict_to_reindex = sorted(list(set(df_out_keep[field_variable_trajectory_group])))
                    dict_to_reindex = dict(
                        (x, i + max(tgs_to_assign) + 1)
                        for i, x in enumerate(dict_to_reindex)
                        if not np.isnan(x)
                    )
                    
                    (
                        df_out_keep[field_variable_trajectory_group]
                        .replace(
                            dict_to_reindex,
                            inplace = True,
                        )
                    )

                    # merge in required fields
                    df_out_overwrite = pd.merge(
                        df_out_overwrite,
                        df_trajgroup[
                            [x for x in fields_sort_with_tg if x in df_trajgroup.columns]
                        ]
                        .dropna(),
                        how = "left",
                    )

                    df_out = pd.concat(
                        [df_out_keep, df_out_overwrite],
                        axis = 0,
                    )
                
                else:
                    # merge in required fields
                    df_out = pd.merge(
                        df_out,
                        (
                            df_trajgroup[
                                [x for x in fields_sort_with_tg if x in df_trajgroup.columns]
                            ]
                            .dropna()
                        ),
                        how = "left",
                    )


        if include_time_periods:
            time_period = self.dim_time_period
            df_out = sf.explode_merge(
                df_out,
                self.dict_attributes.get("dim_time_period").table[[time_period]]
            )

            fields_sort += [time_period]

        df_out = (
            df_out
            .sort_values(by = fields_sort)
            .reset_index(drop = True)
        )

        return df_out



    def build_vars_basic(self,
        dict_vr_varschema: dict,
        dict_vars_to_cats: dict,
        category_to_replace: str
    ) -> List[str]:
        """
        Build a basic variable list from varible schema
        """
        # dict_vars_to_loop has keys that are variables to loop over that map to category values
        vars_out = []
        vars_loop = list(set(dict_vr_varschema.keys()) & set(dict_vars_to_cats.keys()))
        # loop over required variables (exclude transition probability)
        for var in vars_loop:
            error_str = f"Invalid value associated with variable key '{var}'  build_vars_basic/dict_vars_to_cats: the value in the dictionary should be the string 'none' or a list of category values."
            var_schema = clean_schema(dict_vr_varschema[var])
            if type(dict_vars_to_cats[var]) == list:
                for catval in dict_vars_to_cats[var]:
                    vars_out.append(var_schema.replace(category_to_replace, catval))
            elif type(dict_vars_to_cats[var]) == str:
                if dict_vars_to_cats[var].lower() == "none":
                    vars_out.append(var_schema)
                else:
                    raise ValueError(error_str)
            else:
                raise ValueError(error_str)

        return vars_out



    def build_target_varlist_from_source_varcats(self, 
        modvar_source: str, 
        modvar_target: str
    ):
        """
        Build a variable using an ordered set of categories associated with 
            another variable

        Function Arguments
        ------------------
        - modvar_source: source model variable (includes source categories)
        - modvar_target: target model variable (replaced with source categories)

        """
        # get source categories
        cats_source = self.get_variable_categories(modvar_source)

        # build the target variable list using the source categories
        subsector_target = self.dict_model_variable_to_subsector[modvar_target]
        vars_target = self.build_varlist(
            subsector_target, 
            variable_subsec = modvar_target, 
            restrict_to_category_values = cats_source
        )

        return vars_target



    def build_variable_fields(self,
        variable_specification: Union[mv.ModelVariable, str, List[mv.ModelVariable], List[str], None],
        category_restrictions_as_full_spec: bool = False,
        restrict_to_category_values: Union[Dict[str, List[str]], List[str], str, None] = None,
        sort: bool = True,
        variable_type: Union[str, None] = None,
    ) -> List[str]:
        """
        Build a list of fields (complete variable schema from a data frame) 
            based on the subsector and variable name.

        Function Arguments
        ------------------
        - variable_specification: specification for variables to build. Accepts
            the following options:

            * sector: sector name or abbreviation
            * subsector: subsector name or abbreviation
            * variable name: name of a variable to retrieve
            * model variable: model variable object used to define fields

        Keyword Arguments
        -----------------
        - category_restrictions_as_full_spec: Passed to each 
            ModelVariable.build_fields(). Set to True to treat 
            `restrict_to_category_values` as the full specification dictionary
        - restrict_to_category_values: default is None. If None, applies to all 
            categories specified in attribute tables. 

            * dict: should map a mutable element to a list of categories
                    associated with that element. 
                    * RETURNS: list of fields
            * list: only available if the number of mutable elements in the
                schema is 1; assumes that categories are associated with 
                that element.
                * RETURNS: list of fields

                NOTE: If the mutable elements are all associated with a 
                single root element (e.g., cat_landuse_dim1 and 
                cat_landuse_dim2 both share the parent cat_landuse), then 
                a list is assumed to specify the space for the root element;
                all dimensions will take this restriction.
                
            * str: only available if the number of mutable elements in the
                schema is 1; behavior is the same as a single-element list.
                * RETURNS: field (string) or None if the category is not
                    associated with the variable subsector

            * NOTE: careful when using if variable_specification includes model
                variables for multiple sectors; if categories are specified as
                a list, the function operates under the assumption that *all*
                variables are restricted to the same categories.

            * NOTE: with multi-dimensional variables, if the category 
                restriction is specified as a list, all dimensions associated
                with a single category (for a given variable) will be subject
                to the category restrictions. Use a dictionary that specifies
                dimensions individually to allow different category restrictions
                along different dimensions (e.g., $CAT-LANDUSE-DIM1$ and 
                $CAT-LANDUSE-DIM2$)

            * NOTE: if `category_restrictions_as_full_spec == True`, then 
                category_restrictions is treated as the initialization 
                dictionary

        - sort: sort the output 
        - variable_type: input or output. If None, defaults to input.
        """

        ##  INITIALIZATION 

        modvars_to_build = self.decompose_variable_specification(
            variable_specification,
            return_type = "variable",
        )
        if modvars_to_build is None:
            return None

        # check variable type
        variable_type = (
            variable_type.lower() 
            if isinstance(variable_type, str) 
            else variable_type
        )
        variable_type = (
            None 
            if variable_type not in ["input", "output"] 
            else variable_type
        )


        ##  ITERATE TO BUILD
        
        fields_out = []

        for modvar in modvars_to_build:
            
            # skip if variable type is specified
            if variable_type is not None:
                vt = modvar.get_property("variable_type").lower()
                if isinstance(vt, str):
                    if vt.lower() != variable_type:
                        continue
            
            fields = modvar.build_fields(
                category_restrictions = restrict_to_category_values,
                category_restrictions_as_full_spec = category_restrictions_as_full_spec,
            )

            if fields is None:
                continue
            
            (
                fields_out.extend(fields)
                if isinstance(fields, list)
                else fields_out.append(fields)
            )  
        
        # return to string if entered as a string
        if isinstance(restrict_to_category_values, str) & (len(fields_out) == 1):
            fields_out = fields_out[0]

        if isinstance(fields_out, list) & sort:
            fields_out.sort()

        return fields_out



    def check_category_restrictions(self, 
        categories_to_restrict_to: Union[List, None], 
        attribute_table: AttributeTable, 
        stop_process_on_error: bool = True
    ) -> Union[List, None]:
        """
        Check category subsets that are specified.

        Function Arguments
        ------------------
        - categories_to_restrict_to: categories to check against attribute_table
        - attribute_table: AttributeTable to use to check categories

        Keyword Arguments
        -----------------
        - stop_process_on_error: throw an error?
        """
        if categories_to_restrict_to is not None:
            
            # check type
            if not isinstance(categories_to_restrict_to, list):
                if stop_process_on_error:
                    raise TypeError(f"Invalid type of categories_to_restrict_to: valid types are 'None' and 'list'.")
                else:
                    return None

            # get valid/invalid categories
            valid_cats = [x for x in attribute_table.key_values if x in categories_to_restrict_to]
            invalid_cats = sorted([x for x in categories_to_restrict_to if (x not in attribute_table.key_values)])

            if len(invalid_cats) > 0:
                missing_cats = sf.format_print_list(invalid_cats)
                msg_err = f"Invalid categories {invalid_cats} found."
                if stop_process_on_error:
                    raise ValueError(msg_err)
                else:
                    warnings.warn(msg_err + " They will be dropped.")

            return valid_cats
        else:
            return attribute_table.key_values



    def clean_partial_category_dictionary(self,
        dict_in: dict,
        all_category_values: list,
        delim: str = None
    ) -> dict:
        """
        Clean a partial category dictionary to return either none (no 
            categorization) or a list of applicable categories
        """

        delim = self.delim_multicats if (delim is None) else delim

        for k in dict_in.keys():
            if "none" == dict_in[k].lower().replace(" ", ""):
                dict_in.update({k: "none"})
            else:
                cats = dict_in[k].replace("`", "").split(delim)
                dict_in.update(
                    {
                        k: [x for x in all_category_values if x in cats]
                    }
                )
                missing_vals = [x for x in cats if x not in dict_in[k]]
                if len(missing_vals) > 0:
                    missing_vals = sf.format_print_list(missing_vals)
                    warnings.warn(f"clean_partial_category_dictionary: Invalid categories values {missing_vals} dropped when cleaning the dictionary. Category values not found.")

        return dict_in
    


    def decompose_variable_specification(self,
        variable_specification: Union[mv.ModelVariable, str, List[mv.ModelVariable], List[str], None],
        return_type: str = "variable_name",
    ) -> Union[List[mv.ModelVariable], List[str], None]:
        """
        Decompose variable_specification into a list of model variables. Allows
            for a range of specifications of variables, including sector, 
            subsector, variable name, and ModelVariable objects.

        Function Arguments
        ------------------
        - variable_specification: specification for variables to build. Accepts
            the following options:

            * sector: sector name or abbreviation
            * subsector: subsector name or abbreviation
            * variable name: name of a variable to retrieve
            * model variable: model variable object used to define fields
        
        Keyword Arguments
        -----------------
        - return_type: one of the following types:
            * "variable": list of mv.ModelVariable objects
            * "variable_name": list of variable names as strings
        """
        
        ##  DIVIDE INTO STRING AND MODEL VARIABLE ELEMENTS

        var_spec_str = []
        var_spec_mv = []
        
        if isinstance(variable_specification, str):
            var_spec_str = [variable_specification]
        
        elif mv.is_model_variable(variable_specification):
            var_spec_mv = [variable_specification]

        elif sf.islistlike(variable_specification):
            var_spec_str = [x for x in variable_specification if isinstance(x, str)]
            var_spec_mv = [x for x in variable_specification if mv.is_model_variable(x)]


        ## START BY DECOMPOSING ANY SECTORS

        # check for sectors/subsectors
        sectors = [x for x in var_spec_str if x in self.all_sectors]
        subsectors = [x for x in var_spec_str if x in self.all_subsectors]
        for sector in sectors:
            sector_subsecs = self.get_sector_subsectors(sector) 
            subsectors.extend([x for x in sector_subsecs if x not in subsectors])

        # identify model variables that are specified as a string
        modvars_str = [x for x in var_spec_str if x in self.all_variables]
        
        # next, iterate over over subsectors to update model variables
        for subsec in subsectors:
            modvars_cur = self.dict_model_variables_by_subsector.get(subsec)
            if modvars_cur is None:
                continue
            modvars_str += modvars_cur 

        # finally, combine based on output type
        if return_type == "variable_name":
            modvars_out = set(modvars_str + [x.name for x in var_spec_mv])
            modvars_out = sorted(list(modvars_out))

        elif return_type == "variable":
            modvars_out = set(var_spec_mv + [self.get_variable(x) for x in modvars_str])

        return modvars_out
    


    def get_field_subsector(self, 
        field: str, 
        throw_error_q: bool = True
    ) -> Union[str, None]:
        """
        Easy function for getting a field's (variable input) subsector
        """
        dict_check = self.dict_variables_to_model_variables
        val_out = dict_check.get(field)
        if (val_out is None):
            return None

        val_out = self.get_variable_subsector(val_out)

        return val_out



    def get_input_output_fields(self, #VISIT
        subsectors_io: list, 
        build_df_q: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Get input/output fields for a list of subsectors
        """

        # initialize output lists
        vars_out = []
        vars_in = []
        subsectors_out = []
        subsectors_in = []

        for subsector in subsectors_io:

            vars_subsector_in = self.build_variable_fields(subsector, variable_type = "input")
            vars_subsector_out = self.build_variable_fields(subsector, variable_type = "output")

            vars_in += vars_subsector_in
            vars_out += vars_subsector_out

            if build_df_q:
                subsectors_out += [subsector for x in vars_subsector_out]
                subsectors_in += [subsector for x in vars_subsector_in]

        if build_df_q:
            vars_in = (
                pd.DataFrame({
                    "subsector": subsectors_in, 
                    "variable": vars_in
                })
                .sort_values(by = ["subsector", "variable"])
                .reset_index(drop = True)
            )
            vars_out = (
                pd.DataFrame({
                    "subsector": subsectors_out, 
                    "variable": vars_out
                })
                .sort_values(by = ["subsector", "variable"])
                .reset_index(drop = True)
            )

        return vars_in, vars_out
    


    def get_modvar_from_varspec(self,
        variable_specification: str,
    ) -> Union[str, None]:
        """
        Return a model variable from a model variable OR field. 

        Function Arguments
        ------------------
        - variable_specification: model variable OR field. Hierarchically,
            checks to see if variable_specification is a model variable; if not,
            looks to fields.
        """

        check_val = self.check_modvar(variable_specification)
        check_val = (
            self.dict_model_variables_to_variables.get(variable_specification)
            if check_val is None
            else check_val
        )

        return check_val



    def get_multivariables_with_bounded_sum_by_category(self,
        df_in: pd.DataFrame,
        modvars: list,
        sum_restriction: float,
        correction_threshold: float = 0.000001,
        force_sum_equality: bool = False,
        msg_append: str = ""
    ) -> dict:

        """
        use get_multivariables_with_bounded_sum_by_category() to retrive
            multiple variables that, across categories, must sum to some value.
            Gives a correction threshold to allow for small errors.

        Function Arguments
        ------------------
        - df_in: data frame containing input variables
        - modvars: variables to sum over and restrict
        - sum_restriction: maximium sum that array may equal

        Keyword Arguments
        -----------------
        - correction_threshold: tolerance for correcting categories that exceed
            the sum restriction
        - force_sum_equality: default is False. If True, will force the sum to
            equal one (overrides correction_threshold)
        - msg_append: use to passage an additional error message to support
            troubleshooting

        """
        # retrieve arrays
        arr = 0
        init_q = True
        dict_arrs = {}
        for modvar in modvars:

            if modvar not in self.dict_model_variables_to_variables.keys():
                raise ValueError(f"Invalid variable specified in extract_model_variable: variable '{modvar}' not found.")

            else:

                subsector_cur = self.get_variable_subsector(modvar)
                cats = self.get_variable_categories(modvar)

                if init_q:
                    subsector = subsector_cur
                    init_q = False

                elif subsector_cur != subsector:
                    raise ValueError(f"Error in get_multivariables_with_bounded_sum_by_category: variables must be from the same subsector.")
                
                # get current variable, merge to all categories, update dictionary, and check totals
                arr_cur = self.extract_model_variable(#
                    df_in, 
                    modvar, 
                    override_vector_for_single_mv_q = True, 
                    return_typ = "array_base",
                )

                arr_cur = (
                    self.merge_array_var_partial_cat_to_array_all_cats(arr_cur, modvar) 
                    if (cats is not None) 
                    else arr_cur
                )

                dict_arrs.update({modvar: arr_cur})
                arr += arr_cur


        if force_sum_equality:
            for modvar in modvars:
                arr_cur = dict_arrs[modvar]
                arr_cur = np.nan_to_num(arr_cur/arr, 0.0)
                dict_arrs.update({modvar: arr_cur})
        else:
            # correction sums if within correction threshold
            w = np.where(arr > sum_restriction + correction_threshold)[0]
            if len(w) > 0:
                raise ValueError(f"Invalid summations found: some categories exceed the sum threshold.{msg_append}")

            # find locations where the array is in the "in-between" and correct
            w = np.where((arr <= sum_restriction + correction_threshold) & (arr > sum_restriction))

            if len(w[0]) > 0:
                inds = w[0]*len(arr[0]) + w[1]
                for modvar in modvars:
                    arr_cur = dict_arrs.get(modvar)
                    new_vals = sum_restriction*arr_cur[w[0], w[1]].flatten()/arr[w[0], w[1]].flatten()
                    np.put(arr_cur, inds, new_vals)

                    dict_arrs.update({modvar: arr_cur})

        return dict_arrs



    def get_optional_or_integrated_standard_variable(self,
        df_in: pd.DataFrame,
        var_integrated: str,
        var_optional: Union[str, None],
        **kwargs
    ) -> Union[tuple, None]:
        """
        Function to return an optional variable if another (integrated) variable 
            is not passed
        """
        # get fields needed
        subsector_integrated = self.get_variable_subsector(var_integrated)
        fields_check = self.build_variable_fields(var_integrated)
        out = None

        # check and return the output variable + which variable was selected
        if set(fields_check).issubset(set(df_in.columns)):
            out = self.extract_model_variable(#
                df_in, 
                var_integrated, 
                **kwargs
            )

            out = (var_integrated, out)

        elif var_optional is not None:
            out = self.extract_model_variable(#
                df_in, 
                var_optional,
                 **kwargs
            )

            out = (var_optional, out)

        return out



    def get_partial_category_dictionaries(self,
        subsector: str,
        category_outer_tuple: tuple,
        key_type: str = "key_varreqs_partial",
        delim: str = "|",
        variable_in = None,
        restrict_to_category_values = None,
        var_type = None
    ) -> tuple:
        """
        Build a dictionary of categories applicable to a give variable; split by 
            unidim/outer
        """
        key_attribute = self.get_subsector_attribute(subsector, key_type)
        valid_cats = self.check_category_restrictions(
            restrict_to_category_values, 
            self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")]
        )

        # initialize
        dict_vr_vvs_cats_ud = {}
        dict_vr_vvs_cats_outer = {}

        if key_attribute is not None:
            dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer = self.separate_varreq_dict_for_outer(
                subsector, 
                key_type, 
                category_outer_tuple, 
                target_field = "categories", 
                variable = variable_in,
                variable_type = var_type
            )
            dict_vr_vvs_cats_ud = self.clean_partial_category_dictionary(
                dict_vr_vvs_cats_ud, 
                valid_cats, 
                delim
            )
            dict_vr_vvs_cats_outer = self.clean_partial_category_dictionary(
                dict_vr_vvs_cats_outer, 
                valid_cats, 
                delim
            )

        tup_out = dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer

        return tup_out



    def get_sector_list_from_projection_input(self,
        sectors_project: Union[list, str, None] = None,
        delim: str = "|"
    ) -> list:
        """
        Check and retrieve valid projection subsectors to from input
            `sectors_project`

        Keyword Arguments
        ------------------
        - sectors_project: list or string of sectors to run. If None, will run
            all valid sectors defined in model attributes.
            * NOTE: sectors or sector abbreviations are accepted as valid inputs
        - delim: delimiter to use in input strings
        """
        # get subsector attribute
        attr_sec = self.dict_attributes.get("abbreviation_sector")
        dict_map = attr_sec.field_maps.get(f"{attr_sec.key}_to_sector")
        valid_sectors_project = [dict_map.get(x) for x in attr_sec.key_values]

        # convert input to list
        if (sectors_project is None):
            list_out = valid_sectors_project
        elif isinstance(sectors_project, str):
            list_out = sectors_project.split(delim)
        elif isinstance(sectors_project, list) or isinstance(sectors_project, np.ndarray):
            list_out = list(sectors_project)

        # check values
        list_out = [dict_map.get(x, x) for x in list_out if dict_map.get(x, x) in valid_sectors_project]

        return list_out
    


    def get_simplex_group(self,
        variable: str,
    ) -> Union[int, None]:
        """
        Return a simplex group from a variable.

        Function Arguments
        ------------------
        - variable: field variable. Cannot be done from modvar since one modvar
            may have components associated with different simplex groups
        """

        out = self.dict_variable_to_simplex_group.get(variable)
        
        return out
    


    def get_simplex_group_specification(self,
        modvars_to_check: Union[str, List[str]],
    ) -> Union[List[str], None]:
        """
        Check the specification of simplex groups. Ensures that variables
            are in the same sector and either are all either (1) associated 
            with categories or (b) unassociated with categories.

        Function Arguments
        ------------------
        - modvars_to_check: a model variable or list of model variables to 
            validate as

        Keyword Arguments
        -----------------
        """

        # check input specification - convert string to list and verify that there is at least one valid variable
        modvars_to_check = (
            [modvars_to_check] 
            if isinstance(modvars_to_check, str) 
            else modvars_to_check
        )
        modvars_to_check = (
            [x for x in modvars_to_check if self.check_modvar(x) is not None]
            if sf.islistlike(modvars_to_check) 
            else None
        )
        modvars_to_check = (
            (None if (len(modvars_to_check) == 0) else modvars_to_check)
            if modvars_to_check is not None
            else modvars_to_check
        )

        if modvars_to_check is None:
            return None


        # iterate over inputs to 
        subsecs = set([self.get_variable_subsector(x) for x in modvars_to_check])
        category_specs = set(
            [
                (self.get_variable_categories(x) is None)
                for x in modvars_to_check
            ]
        )

        # valid if only one subsec and either all are categorized or none are categorized
        valid_q = len(subsecs) == 1 
        valid_q = (len(category_specs) == 1)

        # don't allow one variable if it has no categories
        valid_q &= not ((True in category_specs) & (len(modvars_to_check) == 1)) 
        if not valid_q:
            return None


        ##  NEXT, SHIFT TO MAP VARIABLES TO GROUPINGS

        ind_base = 1
        dict_out = {}

        if len(modvars_to_check) == 1:
            dict_out.update(
                dict(
                    (x, ind_base) for x in self.build_variable_fields(modvars_to_check[0])
                )
            )

        else:
            # get attribute table and check categories
            attr_subsec = self.get_attribute_table(list(subsecs)[0])

            cats = [self.get_variable_categories(x) for x in modvars_to_check]
            cats = set(sum(cats, [])) if (None not in cats) else None
            cats = (
                [x for x in attr_subsec.key_values if x in cats]
                if cats is not None
                else [None]
            )

            # iterate across categories to build fields that must stay in one group
            for cat in cats:

                fields_group = []

                for modvar in modvars_to_check:

                    field = None
                    try:
                        field = self.build_variable_fields(
                            modvar,
                            restrict_to_category_values = cat,
                        )

                    except Exception as e:
                        # skip on an error
                        continue

                    (
                        fields_group.append(field) 
                        if isinstance(field, str)
                        else (
                            fields_group.extend(field) 
                            if isinstance(field, list) 
                            else None
                        )
                    )

                (
                    dict_out.update(
                        dict((x, ind_base) for x in fields_group)
                    )
                    if len(fields_group) > 0
                    else None
                )

                ind_base += 1

        return dict_out
                
    

    



    def get_variable_attribute(self, #FIXED
        variable: Union[str, mv.ModelVariable], 
        attribute: str,
        stop_on_error: bool = True,
    ) -> Union[str, None]:
        """
        use get_variable_attribute to retrieve a variable attribute--any cleaned 
            field available in the variable requirements table--associated with 
            a variable.
        """
        # check variable first
        variable = self.get_variable(variable)
        if variable is None:
            if stop_on_error:
                raise ValueError(f"Invalid model variable '{variable}' specified in get_variable_attribute.")
            
            return None

        # get the subsector
        subsector = self.get_variable_subsector(variable)
        attr_subsector = self.get_attribute_table(
            subsector, 
            table_type = "variable_definitions",
        )

        # get the attribute
        var_attr = attr_subsector.get_attribute(
            variable.name,
            attribute,
        )

        if (var_attr is None) & stop_on_error:
            msg = f"""
            Error searching for {variable.name} attribute in 
            get_variable_attribute: Attribute {attribute} not found in 
            {subsector} attribute table.
            """
            raise RuntimeError(msg)

        return var_attr



    def get_variable_categories(self, #FIXED
        variable: Union[str, mv.ModelVariable],
        force_dict_return: bool = False,
        stop_on_error: bool = False,
    ) -> Union[List[str], Dict[str, List[str]], None]:
        """
        Retrieve an (ordered) list of categories for a variable. Returns None if
            the variable is not associated with any categories.

        Function Arguments
        ------------------
        - variable: variable name to get categories for OR ModelVariable object

        Keyword Arguments
        -----------------
        - force_dict_return: the ModelVariable object stores categories in a 
            dictionary, where schema elements are keys. Set 
            `force_dict_return = True` to force the return of this dictionary. 
            Otherwise, if only one set of categories is defined (across 
            potentially multiple elements), this function will return the list
            of categories
        - stop_on_error: stop on an error? Otherwise, returns None
        """
        # get the model variable
        modvar = self.get_variable(variable)
        if modvar is None:
            if stop_on_error:
                raise ValueError(f"Invalid variable '{variable}': variable not found.")
            return None

        # get variable categories, defined in a dictionary
        dict_cats = modvar.dict_category_keys
        return_none = (dict_cats is None)
        return_none |= (len(dict_cats) == 0) if not return_none else return_none
        if return_none:
            return None

        # if all elements have the same categories, return those as a list     
        set_all_cats = set(sum(modvar.dict_category_keys.values(), []))
        one_val = True
        for val in dict_cats.values():
            one_val &= set_all_cats.issubset(set(val))

        out = val if (one_val and not force_dict_return) else dict_cats

        return out



    def get_variable_characteristic(self, #FIXED
        modvar: Union[str, mv.ModelVariable], 
        characteristic: str,
    ) -> Union[str, None]:
        """
        use get_variable_characteristic to retrieve a characterisetic--e.g., 
            characteristic = "$UNIT-MASS$" or 
            characteristic = "$EMISSION-GAS$"--associated with a variable.

            NOTE: also accepts clean versions, e.g., "unit_mass" or 
            "emission_gas"
        """

        modvar = self.get_variable(modvar)
        if modvar is None:
            return None
        
        characteristic = mv.clean_element(characteristic)
        out = modvar.attribute(characteristic)

        return out



    def get_variable_from_category(self, #FIXED
        subsector: str, 
        category: str, 
    ) -> str:
        """
        Retrieve a variable that is associated with a category in a file (see 
            Transportation Demand for an example)
        """

        # run some checks
        self.check_subsector(subsector)

        # get the value from the dictionary
        attr_subsec = self.get_attribute_table(subsector)
        subsec_abv = self.get_subsector_attribute(subsector, "abv_subsector")
        attr_retrieve = f"{subsec_abv}_variable"

        # retrieve output
        out = attr_subsec.get_attribute(
            category,
            attr_retrieve,
        )

        return out



    def get_variable_subsector(self, #FIXED
        modvar: Union[str, mv.ModelVariable], 
        key_subsector: str = "subsector",
        throw_error_q: bool = True,
    ) -> Union[str, None]:
        """
        Easy function for getting a variable subsector
        """
        
        out = (
            modvar.get_property(key_subsector)
            if mv.is_model_variable(modvar)
            else self.dict_model_variable_to_subsector.get(modvar)
        )

        if (out is None) and throw_error_q:
            raise KeyError(f"Invalid model variable '{modvar}': model variable not found.")

        return out



    def get_variable_unit_conversion_factor(self, #FIXED
        var_to_convert: Union[str, mv.ModelVariable],
        var_to_match: Union[str, mv.ModelVariable],
        units: str
    ) -> Union[float, int, None]:

        """
        Conversion factor to scale 'var_to_convert' to the same unit type 
            'units' as 'var_to_match'. Returns None if var_to_convert is None

        Function Arguments
        ------------------
        - var_to_convert: string of a model variable to scale units
        - var_to_match: string of a model variable to match units
        - units: valid values are defined in self.all_units
        """
        # return None if no variable passed
        if var_to_convert is None:
            return None
        
        # retrieve units object
        units = self.get_unit(units)
        if units is None:

            str_valid_units = sf.format_print_list(self.all_units)
            msg = f"""
            Invalid units '{units}' specified in 
            get_variable_conversion_factor: valid values are {str_valid_units}
            """

            raise ValueError(msg)
        

        # get characteristic and convert
        unit_key = f"{self.attribute_group_key_unit}_{units.key}"
        args = (
            self.get_variable_characteristic(var_to_convert, unit_key),
            self.get_variable_characteristic(var_to_match, unit_key)
        )

        out = units.convert(*args)

        return out



    def get_variables_from_attribute(self, #VISIT
        subsec: str,
        dict_attributes: str,
    ) -> Union[List[str], None]:
        """
        Retrieve a list of model variables from an attribute. Returns an 
            empty list if no variables match the specification.
            
        NOTE: Returns None if subsec is invalid, attribute is not found. 
        
        Function Arguments
        ------------------
        - subsec: subsector to get variables from
        - dict_attributes: dictionary mapping field in variable attribute tables
            to values to use for filtering
        
        Keyword Arguments
        -----------------
        """
        
        subsec = self.check_subsector(subsec, throw_error_q = False)
        if (subsec == False) or not isinstance(dict_attributes, dict):
            return None
        
        vars_out = None
        
        # check each attribute table
        for key in ["key_varreqs_all", "key_varreqs_partial"]:
            
            attr = self.get_attribute_table(subsec, table_type = key)
            
            # check if we should attempt to filter the dataframe
            continue_q = True
            if attr is not None:
                continue_q = (
                    not any([x in attr.table.columns for x in dict_attributes.keys()])
                    if len(attr.table) > 0
                    else True
                )
            if continue_q:
                continue
            
            # otherwise, filter out
            vars_match = list(
                sf.subset_df(
                    attr.table,
                    dict_attributes
                )[attr.key]
            )
            
            vars_out = (
                vars_match
                if vars_out is None
                else (vars_out + vars_match)
            )
            
        return vars_out
    


    def get_variable_to_simplex_group_dictionary(self,
        field_model_variable: str = "variable",
        field_simplex_group: str = "simplex_group",
        str_split_varreqs_key: str = "category_",
        trajgroup_0: int = 1,
    ) -> Dict[str, int]:
        """
        Map each model variable to a standard simplex group (meaning they sum
            to 1). 

        Used mainly for building templates; for example, if 
            include_simplex_group_as_trajgroup = True in 
            self.build_variable_dataframe_by_sector(), then simplex groups are
            used as the default variable trajectory group in build templates.

        NOTE: LATER INSTANTATION OF THIS CAN MAKE USE OF Variable CLASS

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - field_model_variable: field containing the model variable
        - field_simplex_group: field identifying the simplex group
        - str_split_varreqs_key: string to split keys in 
            model_attributes.dict_variable_definitions on; the second element is 
            used for sorting tables to assign simplex groups for
        - trajgroup_0: base trajectory group to start from when making 
            assignments
        """

        # initialize output dict and first-stage group list
        dict_variable_to_simplex_group = {}
        simplex_groups = []

        for key, attr in self.dict_variable_definitions.items():

            # get sortable class for this table
            #sort_class = key.split(str_split_varreqs_key)
            #if len(sort_class) < 2:
            #    continue
            #sort_class = sort_class[1]
            sort_class = key

            # skip if unidentified
            if field_simplex_group not in attr.table.columns:
                continue


            ##  iterate over groups that are identified

            dfg = attr.table.groupby([field_simplex_group])

            for i, df in dfg:

                # skip if misspecified
                if not sf.isnumber(i, skip_nan = True):
                    continue

                i = int(i)

                # check groupings and skip if invalid
                simplex_modvars = list(df[field_model_variable])
                dict_vars_to_sg = self.get_simplex_group_specification(simplex_modvars)
                if dict_vars_to_sg is None:
                    continue

                ##
                ## TEMPORARY SINCE THIS IS NOT MISSION CRITICAL:
                ##

                var_transitions = "Unadjusted Land Use Transition Probability"
                if var_transitions in simplex_modvars:
                    print(
                        f"MISSIONSEARCHNOTE: As of 2023-10-06, there is a temporary solution implemeted in ModelAttributes.get_variable_to_simplex_group_dictionary() to ensure that transition probability rows are enforced on a simplex.\n\nFIX THIS ASAP TO DERIVE PROPERLY."
                    )

                    fields_to_split = self.build_variable_fields(var_transitions)
                    attr_lndu = self.get_attribute_table(self.subsec_name_lndu)
                    dict_assigned = {}

                    ind = 1
                    for cat in attr_lndu.key_values:
                        dict_assigned.update(
                            dict((x, ind) for x in fields_to_split if f"{cat}_to" in x)
                        )
                        ind += 1
                    
                    dict_vars_to_sg = dict_assigned

                ##
                ##
                ##

                # otherwise, add to output 
                simplex_groups.append((sort_class, dict_vars_to_sg))
        
        


        ##  NOW, SORT BY SECTOR/SUBSECTOR AND ASSIGN

        # get sort classes and initialize simplex group
        all_sort_class = sorted(list(set([x[0] for x in simplex_groups])))
        simplex_group = trajgroup_0

        for sort_class in all_sort_class:
            
            # get some groupings that are used to id
            # `groups` is a list of dictionary mapping fields to internal groupings
            # `dict_field_to_group_ind` maps each field to its index in groups
            # `dict_group_ind_to_sg`, assigned below, is used to track assignments for (group_ind, inner_simplex_group) to outer simplex_groups
            
            groups = [x[1] for x in simplex_groups if x[0] == sort_class]
            dict_field_to_group_ind = dict(
                (field, i) for i, g in enumerate(groups)
                for field in g.keys()
            )
            all_fields = sorted(list(dict_field_to_group_ind.keys()))
            
            # initialize the group/inner simplex group assignment dictionary, then iterate over ordered fields
            dict_group_ind_to_sg = {}
            
            for field in all_fields:
                
                ind = dict_field_to_group_ind.get(field)
                sg_cur = groups[ind].get(field)
                tup_index = (ind, sg_cur)
                
                # check to see if grp/inner is already assigned; if not, assign it, then move to next iteration
                simplex_group_assign = dict_group_ind_to_sg.get(tup_index)

                if simplex_group_assign is None:
                    simplex_group_assign = simplex_group
                    dict_group_ind_to_sg.update({tup_index: simplex_group})

                    simplex_group += 1
                
                # update field to outer simplex group dictionary
                dict_variable_to_simplex_group.update({field: simplex_group_assign})


        return dict_variable_to_simplex_group



    def instantiate_blank_modvar_df_by_categories(self,
        modvar: str,
        n: int,
        blank_val: Union[int, float] = 0.0,
    ) -> pd.DataFrame:
        """
        Create a blank data frame, filled with blank_val, with properly ordered 
            variable names.

        Function Arguments
        ------------------
        - modvar: the model variable to build the dataframe for
        - n: the length of the data frame

        Keyword Arguments
        -----------------
        - blank_val: the value to use to fill the frame
        """
        cols = self.build_variable_fields(modvar)
        df_out = pd.DataFrame(np.ones((n, len(cols)))*blank_val, columns = cols)

        return df_out



    def separate_varreq_dict_for_outer(self, #REMOVE
        subsector: str,
        key_type: str,
        category_outer_tuple: tuple,
        target_field: str = "variable_schema",
        field_to_split_on: str = "variable_schema",
        variable = None,
        variable_type = None
    ) -> tuple:
        """
        Separate a variable requirement dictionary into those associated with
            simple vars and those with outer

        Function Arguments
        ------------------
        - key_type: key_varreqs_all, key_varreqs_partial

        Keyword Arguments
        -----------------
        - field_to_split_on: gives the field from the attribute table to use to
            split between outer and unidim
        - target_field: the field to return in the dictionary

        """
        key_attribute = self.get_subsector_attribute(subsector, key_type)
        if key_attribute is not None:
            dict_vr_vvs = self.dict_variable_definitions[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{field_to_split_on}"].copy()
            dict_vr_vtf = self.dict_variable_definitions[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{target_field}"].copy()

            # filter on variable type if specified
            if variable_type is not None:
                if variable is not None:
                    warnings.warn(f"variable and variable_type both specified in separate_varreq_dict_for_outer: the variable assignment is higher priority, and variable_type will be ignored.")
                else:
                    dict_var_types = self.dict_variable_definitions[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_variable_type"]
                    drop_vars = [x for x in dict_var_types.keys() if dict_var_types[x].lower() != variable_type.lower()]
                    [dict_vr_vvs.pop(x) for x in drop_vars]
                    [dict_vr_vtf.pop(x) for x in drop_vars]

            dict_vr_vtf_outer = dict_vr_vtf.copy()

            # raise a traceable error
            try:
                vars_outer = [x for x in dict_vr_vtf.keys() if (category_outer_tuple[0] in dict_vr_vvs[x]) and (category_outer_tuple[1] in dict_vr_vvs[x])]
                vars_unidim = [x for x in dict_vr_vtf.keys() if (x not in vars_outer)]
                [dict_vr_vtf_outer.pop(x) for x in vars_unidim]
                [dict_vr_vtf.pop(x) for x in vars_outer]
            except:
                raise ValueError(f"Invalid attribute table designations found in subsector '{subsector}': check the field '{target_field}'.")

            if variable != None:
                vars_outer = list(dict_vr_vtf_outer.keys())
                vars_unidim = list(dict_vr_vtf.keys())
                [dict_vr_vtf_outer.pop(x) for x in vars_outer if (x != variable)]
                [dict_vr_vtf.pop(x) for x in vars_unidim if (x != variable)]
        else:
            dict_vr_vtf = {}
            dict_vr_vtf_outer = {}

        return dict_vr_vtf, dict_vr_vtf_outer



    def swap_array_categories(self, #VISIT
        array_in: np.ndarray,
        vec_ordered_cats_source: np.ndarray,
        vec_ordered_cats_target: np.ndarray,
        subsector: str,
    ) -> np.ndarray:
        """
        Swap category columns in an array

        Function Arguments
        ------------------
        - array_in: array with data. Must be merged to all categories for the 
            subsector.
        - vec_ordered_cats_source: array of source categories to swap with 
            targets (source_i -> target_i). Must be well defined categories
        - vec_ordered_cats_target: array of target categories to swap with the 
            source. Must be well defined categories
        - subsector: subsector in which the swap occurs

        Notes
        -----
        - Source categories cannot be defined in the target categories vector, 
            and vis-versa
        - Categories that aren't well-defined will be dropped
        """

        # get and check subsector
        subsector = self.check_subsector(subsector, throw_error_q = False)
        if subsector is None:
            return array_in

        # check subsector attribute
        attr = self.get_attribute_table(subsector)
        if len(set(vec_ordered_cats_source) & set(vec_ordered_cats_target)) > 0:
            warnings.warn("Invalid swap specification in 'swap_array_categories': categories can only exist in source or target")
            return array_in


        ##  BUILD SOURCE/TARGET SWAPS

        vec_source = []
        vec_target = []

        # iterate to get well-defined swaps
        for i in range(min(len(vec_ordered_cats_source), len(vec_ordered_cats_target))):
            cat_source = clean_schema(vec_ordered_cats_source[i])
            cat_target = clean_schema(vec_ordered_cats_target[i])
            if (cat_source in attr.key_values) and (cat_target in attr.key_values):
                vec_source.append(cat_source)
                vec_target.append(cat_target)

        # some warnings - source
        set_drops_source = set(vec_ordered_cats_source) - set(vec_source)
        if len(set_drops_source) > 0:
            vals_dropped_source = sf.format_print_list(list(set_drops_source))
            warnings.warn(f"Source values {vals_dropped_source} dropped in swap_array_categories (either not well-defined categories or there was no associated target category).")

        # some warnings - target
        set_drops_target = set(vec_ordered_cats_target) - set(vec_target)
        if len(set_drops_target) > 0:
            vals_dropped_target = sf.format_print_list(list(set_drops_target))
            warnings.warn(f"target values {vals_dropped_target} dropped in swap_array_categories (either not well-defined categories or there was no associated target category).")

        # build dictionary and set up the new categories
        dict_swap = dict(zip(vec_source, vec_target))
        dict_swap.update(sf.reverse_dict(dict_swap))
        cats_new = [dict_swap.get(x, x) for x in attr.key_values]

        array_new = self.merge_array_var_partial_cat_to_array_all_cats(
            array_in,
            None,
            output_cats = cats_new,
            output_subsec = subsector
        )

        return array_new







def clean_schema(
    var_schema: str, 
    return_default_dict_q: bool = False
) -> str:
    """
    Clean a variable schema input `var_schema`
    """
    var_schema = var_schema.split("(")
    var_schema[0] = var_schema[0].replace("`", "").replace(" ", "")

    dict_repls = {}
    if len(var_schema) > 1:
        repls =  var_schema[1].replace("`", "").split(",")
        for dr in repls:
            dr0 = dr.replace(" ", "").replace(")", "").split("=")
            var_schema[0] = var_schema[0].replace(dr0[0], dr0[1])
            dict_repls.update({dr0[0]: dr0[1]})

    if return_default_dict_q:
        return dict_repls
    else:
        return var_schema[0]



def unclean_category(
    cat: str
) -> str:
    """
    Convert a category to "unclean" by adding tick marks
    """
    return f"``{cat}``"
