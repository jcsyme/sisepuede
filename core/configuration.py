import os, os.path
import pandas as pd
from typing import *

from sisepuede.core.attribute_table import *
import sisepuede.utilities.support_functions as sf


"""
Setup a configuration for SISEPUEDE
"""
class Configuration:

    def __init__(self,
        fp_config: str,
        attr_area: AttributeTable,
        attr_energy: AttributeTable,
        attr_gas: AttributeTable,
        attr_length: AttributeTable,
        attr_mass: AttributeTable,
        attr_monetary: AttributeTable,
        attr_power: AttributeTable,
        attr_region: AttributeTable,
        attr_time_period: AttributeTable,
        attr_volume: AttributeTable,
        attr_required_parameters: AttributeTable = None,
    ):
        self.fp_config = fp_config
        self.attr_required_parameters = attr_required_parameters

        # set tables
        self.attr_area = attr_area
        self.attr_energy = attr_energy
        self.attr_gas = attr_gas
        self.attr_length = attr_length
        self.attr_mass = attr_mass
        self.attr_monetary = attr_monetary
        self.attr_power = attr_power
        self.attr_region = attr_region
        self.attr_volume = attr_volume

        # set required parametrs by type
        self.params_bool = [
            "save_inputs"
        ]
        self.params_string = [
            "area_units",
            "energy_units",
            "energy_units_nemomod",
            "emissions_mass",
            "historical_solid_waste_method",
            "land_use_reallocation_max_out_directionality",
            "length_units",
            "monetary_units",
            "nemomod_solver",
            "output_method",
            "power_units",
            "region",
            "volume_units"
        ]
        self.params_float = [
            "days_per_year"
        ]
        self.params_float_fracs = [
            "discount_rate"
        ]
        self.params_int = [
            "global_warming_potential",
            "historical_back_proj_n_periods",
            "nemomod_solver_time_limit_seconds",
            "nemomod_time_periods",
            "num_lhc_samples",
            "random_seed",
            "time_period_u0"
        ]

        self.dict_config = self.get_config_information(
            attr_area,
            attr_energy,
            attr_gas,
            attr_length,
            attr_mass,
            attr_monetary,
            attr_power,
            attr_region,
            attr_time_period,
            attr_volume,
            attr_required_parameters,
            delim = "|",
        )



    def check_config_defaults(self,
        param,
        vals,
        dict_valid_values: dict = dict({}),
        set_as_literal_q: bool = False,
    ) -> List[Any]:
        """
        some restrictions on the config values
        """

        val_list = (
            [vals] 
            if ((not isinstance(vals, list)) or set_as_literal_q) 
            else vals
        )

        # loop over all values to check
        for val in val_list:
            if param in self.params_bool:
                val = bool(str(val) == "True")
            elif param in self.params_int:
                val = int(val)
            elif param in self.params_float:
                val = float(val)
            elif param in self.params_float_fracs:
                val = min(max(float(val), 0), 1)
            elif param in self.params_string:
                val = str(val)

            if param in dict_valid_values.keys():
                if val not in dict_valid_values[param]:
                    valid_vals = sf.format_print_list(dict_valid_values[param])
                    raise ValueError(f"Invalid specification of configuration parameter '{param}': {param} '{val}' not found. Valid values are {valid_vals}")

        return vals



    def get(self, 
        key: str, 
        raise_error_q: bool = False
    ) -> Any:
        """
        Retrieve a configuration value associated with key
        """
        out = self.dict_config.get(key)
        if (out is None) and raise_error_q:
            raise KeyError(f"Configuration parameter '{key}' not found.")

        return out



    def get_config_information(self,
        attr_area: AttributeTable = None,
        attr_energy: AttributeTable = None,
        attr_gas: AttributeTable = None,
        attr_length: AttributeTable = None,
        attr_mass: AttributeTable = None,
        attr_monetary: AttributeTable = None,
        attr_power: AttributeTable = None,
        attr_region: AttributeTable = None,
        attr_time_period: AttributeTable = None,
        attr_volume: AttributeTable = None,
        attr_parameters_required: AttributeTable = None,
        field_req_param: str = "configuration_file_parameter",
        field_default_val: str = "default_value",
        delim: str = ","
    ) -> dict: 
        """
        Retrieve a configuration file and population missing values with 
            defaults
        """

        # set some variables from defaults
        attr_area = attr_area if (attr_area is not None) else self.attr_area
        attr_energy = attr_energy if (attr_energy is not None) else self.attr_energy
        attr_gas = attr_gas if (attr_gas is not None) else self.attr_gas
        attr_length = attr_length if (attr_length is not None) else self.attr_length
        attr_mass = attr_mass if (attr_mass is not None) else self.attr_mass
        attr_monetary = attr_monetary if (attr_monetary is not None) else self.attr_monetary
        attr_power = attr_power if (attr_power is not None) else self.attr_power
        attr_region = attr_region if (attr_region is not None) else self.attr_region
        attr_time_period = attr_time_period if (attr_time_period is not None) else self.attr_time_period
        attr_volume = attr_volume if (attr_volume is not None) else self.attr_volume

        # check path and parse the config if it exists
        dict_conf = {}
        if self.fp_config is not None:
            if os.path.exists(self.fp_config):
                dict_conf = self.parse_config(self.fp_config, delim = delim)

        # update with defaults if a value is missing in the specified configuration
        if attr_parameters_required is not None:

            dict_key_to_required_param = (
                attr_parameters_required
                .field_maps
                .get(f"{attr_parameters_required.key}_to_{field_req_param}")
            )
            dict_key_to_default_value = (
                attr_parameters_required
                .field_maps
                .get(f"{attr_parameters_required.key}_to_{field_default_val}")
            )

            if (
                attr_parameters_required.key != field_req_param
            ) and (
                dict_key_to_required_param is not None
            ) and (
                dict_key_to_default_value is not None
            ):
                # add defaults
                for k in attr_parameters_required.key_values:
                    param_config = dict_key_to_required_param.get(k) if (attr_parameters_required.key != field_req_param) else k
                    if param_config not in dict_conf.keys():
                        val_default = self.infer_types(dict_key_to_default_value.get(k))
                        dict_conf.update({param_config: val_default})


        ##  MODIFY SOME PARAMETERS BEFORE CHECKING

        # set parameters to return as a list and ensure type return is list
        params_list = ["region", "nemomod_time_periods"]
        for p in params_list:
            if not isinstance(dict_conf[p], list):
                dict_conf.update({p: [dict_conf[p]]})

        # set some to lower case
        params_list = ["nemomod_solver", "output_method"]
        for p in params_list:
            dict_conf.update({p: str(dict_conf.get(p)).lower()})


        ##  CHECK VALID CONFIGURATION VALUES AND UPDATE IF APPROPRIATE

        valid_area = self.get_valid_values_from_attribute_column(
            attr_area, 
            "area_equivalent_", 
            str, 
            "unit_area_to_area",
        )

        valid_bool = [True, False]

        valid_energy = self.get_valid_values_from_attribute_column(
            attr_energy, 
            "energy_equivalent_", 
            str, 
            "unit_energy_to_energy",
        )

        valid_gwp = self.get_valid_values_from_attribute_column(
            attr_gas, 
            "global_warming_potential_", 
            int,
        )

        valid_historical_hwp_method = ["back_project", "historical"]

        valid_historical_solid_waste_method = ["back_project", "historical"]

        valid_lurmod = ["decrease_only", "increase_only", "decrease_and_increase"]

        valid_length = self.get_valid_values_from_attribute_column(
            attr_length, 
            "length_equivalent_", 
            str, 
            "unit_length_to_length",
        )

        valid_mass = self.get_valid_values_from_attribute_column(
            attr_mass, 
            "mass_equivalent_", 
            str, 
            "unit_mass_to_mass",
        )

        valid_monetary = self.get_valid_values_from_attribute_column(
            attr_monetary, 
            "monetary_equivalent_", 
            str, 
            "unit_monetary_to_monetary",
        )

        valid_output_method = ["csv", "sqlite"]
        valid_power = self.get_valid_values_from_attribute_column(
            attr_power, 
            "power_equivalent_", 
            str, 
            "unit_power_to_power",
        )
       
        valid_region = attr_region.key_values

        valid_solvers = ["cbc", "clp", "cplex", "gams_cplex", "glpk", "gurobi", "highs"]

        valid_time_period = attr_time_period.key_values

        valid_volume = self.get_valid_values_from_attribute_column(
            attr_volume, 
            "volume_equivalent_", 
            str,
        )
        

        # map parameters to valid values
        dict_checks = {
            "area_units": valid_area,
            "energy_units": valid_energy,
            "energy_units_nemomod": valid_energy,
            "emissions_mass": valid_mass,
            "global_warming_potential": valid_gwp,
            "historicall_harvested_wood_products_method": valid_historical_hwp_method,
            "historical_solid_waste_method": valid_historical_solid_waste_method,
            "land_use_reallocation_max_out_directionality": valid_lurmod,
            "length_units": valid_length,
            "monetary_units": valid_monetary,
            "nemomod_solver": valid_solvers,
            "nemomod_time_periods": valid_time_period,
            "output_method": valid_output_method,
            "power_units": valid_power,
            "region": valid_region,
            "save_inputs": valid_bool,
            "time_period_u0": valid_time_period,
            "volume_units": valid_volume
        }

        # allow some parameter switch values to valid values
        dict_params_switch = {"region": ["all"], "nemomod_time_periods": ["all"]}
        for p in dict_params_switch.keys():
            if dict_conf[p] == dict_params_switch[p]:
                dict_conf.update({p: dict_checks[p].copy()})

        dict_conf = dict(
            (k, self.check_config_defaults(k, v, dict_checks))
            for k, v in dict_conf.items()
        )

        ###   CHECK SOME PARAMETER RESITRICTIONS

        # positive integer restriction
        dict_conf.update({
            "historical_back_proj_n_periods": max(dict_conf.get("historical_back_proj_n_periods"), 1),
            "nemomod_solver_time_limit_seconds": max(dict_conf.get("nemomod_solver_time_limit_seconds"), 60), # set minimum solver limit to 60 seconds
            "num_lhc_samples": max(dict_conf.get("num_lhc_samples", 0), 0),
            "save_inputs": bool(str(dict_conf.get("save_inputs")).lower() == "true"),
            "random_seed": max(dict_conf.get("random_seed"), 1)
        })

        # set some attributes
        self.valid_area = valid_area
        self.valid_energy = valid_energy
        self.valid_gwp = valid_gwp
        self.valid_historical_solid_waste_method = valid_historical_solid_waste_method
        self.valid_land_use_reallocation_max_out_directionality = valid_lurmod
        self.valid_length = valid_length
        self.valid_mass = valid_mass
        self.valid_monetary = valid_monetary
        self.valid_power = valid_power
        self.valid_region = valid_region
        self.valid_save_inputs = valid_bool
        self.valid_solver = valid_solvers
        self.valid_time_period = valid_time_period
        self.valid_volume = valid_volume

        return dict_conf



    def get_valid_values_from_attribute_column(self,
        attribute_table: AttributeTable,
        column_match_str: str,
        return_type: type = None,
        field_map_to_val: str = None,
    ) -> List[str]:
        """
        Retrieve valid key values from an attribute column
        """
        cols = [
            x.replace(column_match_str, "") 
            for x in attribute_table.table.columns 
            if (x[0:min(len(column_match_str), len(x))] == column_match_str)
        ]
        if return_type is not None:
            cols = [return_type(x) for x in cols]

        # if a dictionary is specified, map the values to a name
        if field_map_to_val is not None:
            
            dict_map = attribute_table.field_maps.get(field_map_to_val)

            if not isinstance(dict_map, dict):
                msg = f"""
                Error in get_valid_values_from_attribute_column: the field map 
                '{field_map_to_val}' is not defined.
                """
                raise KeyError(msg)
            
            cols = [dict_map.get(x) for x in cols]

        return cols



    def infer_type(self,
        val: Union[int, float, str, None]
    ) -> Union[int, float, str, None]:
        """
        Guess the input type for a configuration file.
        """
        if val is not None:
            val = str(val)
            if val.replace(".", "").replace(",", "").isnumeric():
                num = float(val)
                val = int(num) if (num == int(num)) else float(num)

        return val



    def infer_types(self,
        val_in: Union[float, int, str, None],
        delim = ","
    ) -> Union[type, List[type], None]:
        """
        Guess the type of input value val_in
        """
        rv = None
        if val_in is not None:
            rv = (
                [self.infer_type(x) for x in val_in.split(delim)] 
                if (delim in val_in) 
                else self.infer_type(val_in)
            )

        return rv



    def parse_config(self,
        fp_config: str,
        delim: str = ","
    ) -> dict:
        """
            parse_config returns a dictionary of configuration values found in the
                configuration file (of form key: value) found at file path
                `fp_config`.

            Keyword Arguments
            -----------------
            delim: delimiter used to split input lists specified in the configuration file
        """

        #read in aws initialization
        if os.path.exists(fp_config):
        	with open(fp_config) as fl:
        		lines_config = fl.readlines()
        else:
            raise ValueError(f"Invalid configuation file {fp_config} specified: file not found.")

        dict_out = {}
        #remove unwanted blank characters
        for ln in lines_config:
            ln_new = sf.str_replace(ln.split("#")[0], {"\n": "", "\t": ""})
            if (":" in ln_new):
                ln_new = ln_new.split(":")
                key = str(ln_new[0])
                val = self.infer_types(str(ln_new[1]).strip(), delim = delim)
                dict_out.update({key: val})

        return dict_out



    def to_data_frame(self,
        list_delim: str = "|"
    ) -> pd.DataFrame:
        """
        List all configuration parameters as a single-rows dataframe. Converts
            lists to concatenated strings separated by the delimiter
            `list_delim`.

        Keyword Arguments
        -----------------
        - list_delim: delimiter to use to convert lists to concatenated strings
            as elements in the data frame.
        """
        dict_df = {}
        for key in self.dict_config.keys():
            val = self.dict_config.get(key)
            if isinstance(val, list):
                val = list_delim.join([str(x) for x in val])

            dict_df.update({key: [val]})

        return pd.DataFrame(dict_df)