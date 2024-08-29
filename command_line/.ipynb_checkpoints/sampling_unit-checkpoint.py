import numpy as np
import pandas as pd
import os, os.path
import support_functions as sf

#
#    SAVING AS BACKUP
#

##  class for sampling and generating experimental design
class sampling_unit:

    def __init__(self, df_var_def: pd.core.frame.DataFrame, dict_baseline_ids: dict, time_period_u0: int, field_strategy_id: str = "strategy_id", fan_function_specification: str = "linear"):

        # set some attributes

        self.field_strategy_id = field_strategy_id
        self.time_period_end_certainty = time_period_u0
        df_var = self.check_input_data_frame(df_var_def)
        self.fields_id = self.get_id_fields(df_var)
        self.field_min_scalar, self.field_max_scalar, self.time_period_scalar = self.get_scalar_time_period(df_var)
        self.fields_time_periods, self.time_periods = self.get_time_periods(df_var)
        self.variable_trajectory_group = self.get_trajgroup(df_var)
        self.uncertainty_fan_function_parameters = self.get_fan_function_parameters(fan_function_specification)
        self.uncertainty_ramp_vector = self.build_ramp_vector(self.uncertainty_fan_function_parameters)
        
        self.data_table, self.id_coordinates = self.check_scenario_variables(df_var, self.fields_id)
        self.dict_id_values, self.dict_baseline_ids = self.get_scenario_values(self.data_table, self.fields_id, dict_baseline_ids)
        self.num_scenarios = len(self.id_coordinates)
        self.variable_specifications = self.get_all_vs(self.data_table )
        self.dict_variable_info = self.get_variable_dictionary(self.data_table, self.fields_id, self.fields_time_periods, self.variable_specifications, self.field_max_scalar, self.field_min_scalar)
        self.ordered_trajectory_arrays = self.get_ordered_trajectory_arrays(self.data_table, self.fields_id, self.fields_time_periods, self.variable_specifications)
        self.scalar_diff_arrays = self.get_scalar_diff_arrays()
        
        # important components for different design ids + assessing uncertainty in lever acheivement
        self.fields_order_strat_diffs = [x for x in self.fields_id if (x != self.field_strategy_id)] + [self.field_var_spec, self.field_trajgroup_spec]
        self.xl_type, self.dict_strategy_info = self.infer_sampling_unit_type()

        

    # initialize some attributes
    field_trajgroup = "variable_trajectory_group"
    field_trajgroup_spec = "variable_trajectory_group_trajectory_type"
    field_uniform_scaling_q = "uniform_scaling_q"
    field_var_spec = "variable"
    # maps internal name (key) to classification in the input data frame (value)
    dict_required_tg_spec_fields = {"mixing_trajectory": "mix", "trajectory_boundary_0": "trajectory_boundary_0", "trajectory_boundary_1": "trajectory_boundary_1"}
    required_tg_specs = list(dict_required_tg_spec_fields.values())#[dict_required_tg_spec_fields[x] for x in ["mixing_trajectory", "boundary_trajectory_0", "boundary_trajectory_1"]]


    ##  functions to initialize attributes

    def check_input_data_frame(self, df_in: pd.DataFrame):

        # some standardized fields to require
        fields_req = [self.field_strategy_id, self.field_trajgroup, self.field_trajgroup_spec, self.field_uniform_scaling_q, self.field_var_spec]
        if len(set(fields_req) & set(df_in.columns)) < len(set(fields_req)):
            fields_missing = list(set(fields_req) - (set(fields_req) & set(df_in.columns)))
            fields_missing.sort()
            str_missing = ", ".join([f"'{x}'" for x in fields_missing])
            raise ValueError(f"Error: one or more columns are missing from the data frame. Columns {str_missing} not found")
        elif (self.field_strategy_id in df_in.columns) and ("_id" not in self.field_strategy_id):
            raise ValueError(f"Error: the strategy field '{self.field_strategy_id}' must contain the substring '_id'. Check to ensure this substring is specified.")

        return df_in.drop_duplicates()


    def check_scenario_variables(self, df_in: pd.DataFrame, fields_id: list):
        tups_id = set([tuple(x) for x in np.array(df_in[fields_id])])
        for tg_type in self.required_tg_specs:
            df_check = df_in[df_in[self.field_trajgroup_spec] == tg_type]
            for vs in list(df_check[self.field_var_spec].unique()):
                tups_id = tups_id & set([tuple(x) for x in np.array(df_check[df_check[self.field_var_spec] == vs][fields_id])])
        df_scen = pd.DataFrame(tups_id, columns = fields_id)
        df_in = pd.merge(df_in, df_scen, how = "inner", on = fields_id)

        return (df_in, tups_id)


    def get_all_vs(self, df_in: pd.DataFrame):
        if not self.field_var_spec in df_in.columns:
            raise ValueError(f"Field '{self.field_var_spec}' not found in data frame.")
        all_vs = list(df_in[self.field_var_spec].unique())
        all_vs.sort()
        return all_vs


    def get_id_fields(self, df_in: pd.DataFrame):
        fields_out = [x for x in df_in.columns if ("_id" in x)]
        fields_out.sort()
        if len(fields_out) == 0:
            raise ValueError(f"No id fields found in data frame.")

        return fields_out


    def get_ordered_trajectory_arrays(self, df_in: pd.DataFrame, fields_id: list, fields_time_periods: list, variable_specifications: list):
        # order trajectory arrays by id fields; used for quicker lhs application across id dimensions
        dict_out = {}
        for vs in variable_specifications:
            df_cur_vs = df_in[df_in[self.field_var_spec].isin([vs])].sort_values(by = fields_id)

            if self.variable_trajectory_group == None:
                dict_out.update({(vs, None): {"data": np.array(df_cur_vs[fields_time_periods]), "id_coordinates": df_cur_vs[fields_id]}})
            else:
                for tgs in self.required_tg_specs:
                    df_cur = df_cur_vs[df_cur_vs[self.field_trajgroup_spec] == tgs]
                    dict_out.update({(vs, tgs): {"data": np.array(df_cur[fields_time_periods]), "id_coordinates": df_cur[fields_id]}})
        return dict_out


    def get_scalar_time_period(self, df_in:pd.DataFrame):
        # determine min field/time period
        field_min = [x for x in df_in.columns if "min" in x]
        if len(field_min) == 0:
            raise ValueError("No field associated with a minimum scalar value found in data frame.")
        else:
            field_min = field_min[0]

        # determine max field/time period
        field_max = [x for x in df_in.columns if "max" in x]
        if len(field_max) == 0:
            raise ValueError("No field associated with a maximum scalar value found in data frame.")
        else:
            field_max = field_max[0]

        tp_min = int(field_min.split("_")[1])
        tp_max = int(field_max.split("_")[1])
        if (tp_min != tp_max) | (tp_min == None):
            raise ValueError(f"Fields '{tp_min}' and '{tp_max}' imply asymmetric final time periods.")
        else:
            tp_out = tp_min

        return (field_min, field_max, tp_out)


    def get_scenario_values(self, df_in: pd.DataFrame, fields_id: list, dict_baseline_ids: dict):
        # get scenario index values by scenario dimension
        dict_id_values = {}
        dict_id_baselines = dict_baseline_ids.copy()

        for fld in fields_id:
            dict_id_values.update({fld: list(df_in[fld].unique())})
            dict_id_values[fld].sort()

            # check if baseline for field is determined
            if fld in dict_id_baselines.keys():
                bv = int(dict_id_baselines[fld])

                if bv not in dict_id_values[fld]:
                    if fld == self.field_strategy_id:
                        print(bv)
                        raise ValueError(f"Error: baseline {self.field_strategy_id} scenario index '{bv}' not found in the variable trajectory input sheet. Please ensure the basline strategy is specified correctly.")
                    else:
                        msg_warning = f"The baseline id for dimension {fld} not found. The experimental design will not include futures along this dimension of analysis."
                        warnings.warn(msg_warning)
            else:
                # assume minimum >= 0
                bv = min([x for x in dict_id_values[fld] if x >= 0])
                dict_id_baselines.update({fld: bv})
                msg_warning = f"No baseline scenario index found for {fld}. It will be assigned to '{bv}', the lowest non-negative integer."
                warnings.warn(msg_warning)


        return dict_id_values, dict_id_baselines


    def get_time_periods(self, df_in: pd.DataFrame):
        fields_time_periods = [x for x in df_in.columns if x.isnumeric()]
        fields_time_periods = [x for x in fields_time_periods if int(x) == float(x)]
        if len(fields_time_periods) == 0:
            raise ValueError("No time periods found in data frame.")
        else:
            time_periods = [int(x) for x in fields_time_periods]

        time_periods.sort()
        fields_time_periods = [str(x) for x in time_periods]

        return (fields_time_periods, time_periods)

    # get the trajectory group for the sampling unit
    def get_trajgroup(self, df_in: pd.DataFrame):
        if not self.field_trajgroup in df_in.columns:
            raise ValueError(f"Field '{self.field_trajgroup}' not found in data frame.")
        # determine if this is associated with a trajectory group
        if len(df_in[~df_in[self.field_trajgroup].isna()]) > 0:
            return int(list(df_in[self.field_trajgroup].unique())[0])
        else:
            return None

    # the variable dictionary includes information on sampling ranges for time period scalars, wether the variables should be scaled uniformly, and the trajectories themselves
    def get_variable_dictionary(
        self, 
        df_in: pd.DataFrame, 
        fields_id: list, 
        fields_time_periods: list,
        variable_specifications: list,
        field_max: str,
        field_min: str
    ):
        dict_var_info = {}

        if self.variable_trajectory_group != None:
            tgs_loops = self.required_tg_specs
        else:
            tgs_loops = [None]

        for vs in variable_specifications:
            for tgs in tgs_loops:
                if tgs == None:
                    df_cur = sf.subset_df(df_in, {self.field_var_spec: vs})
                else:
                    df_cur = sf.subset_df(df_in, {self.field_var_spec: vs, self.field_trajgroup_spec: tgs})
                    
                dict_vs = {
                    "max_scalar": sf.build_dict(df_cur[fields_id + [field_max]]),
                    "min_scalar": sf.build_dict(df_cur[fields_id + [field_min]]),
                    "uniform_scaling_q": sf.build_dict(df_cur[fields_id + [self.field_uniform_scaling_q]]),
                    "trajectories": sf.build_dict(df_cur[fields_id + fields_time_periods], (len(self.fields_id), len(self.fields_time_periods)))}

                dict_var_info.update({(vs, tgs): dict_vs})

        return dict_var_info

    # determine if the sampling unit represents a strategy (L) or an uncertainty (X)
    def infer_sampling_unit_type(self, thresh: float = (10**(-12))):
        fields_id_no_strat = [x for x in self.fields_id if (x != self.field_strategy_id)]

        strat_base = self.dict_baseline_ids[self.field_strategy_id]
        strats_not_base = [x for x in self.dict_id_values[self.field_strategy_id] if (x != strat_base)]
        fields_ext = [self.field_var_spec, self.field_trajgroup_spec] + self.fields_id + self.fields_time_periods
        fields_ext = [x for x in fields_ext if (x != self.field_strategy_id)]
        fields_merge = [x for x in fields_ext if (x not in self.fields_time_periods)]
        # get the baseline strategy specification + set a renaming dictionary for merges
        df_base = self.data_table[self.data_table[self.field_strategy_id] == strat_base][fields_ext].sort_values(by = self.fields_order_strat_diffs).reset_index(drop = True)
        arr_base = np.array(df_base[self.fields_time_periods])
        fields_base = list(df_base.columns)

        dict_out = {
            "baseline_strategy_data_table": df_base, 
            "baseline_strategy_array": arr_base,
            "difference_arrays_by_strategy": {}
        }

        dict_diffs = {}
        strategy_q = False

        for strat in strats_not_base:

            df_base = pd.merge(df_base, self.data_table[self.data_table[self.field_strategy_id] == strat][fields_ext], how = "inner", on = fields_merge, suffixes = (None, "_y"))
            df_base.sort_values(by = self.fields_order_strat_diffs, inplace = True)
            arr_cur = np.array(df_base[[(x + "_y") for x in self.fields_time_periods]])
            arr_diff = arr_cur - arr_base
            dict_diffs.update({strat: arr_diff})
            df_base = df_base[fields_base]

            if max(np.abs(arr_diff.flatten())) > thresh:
                strategy_q = True

        if strategy_q:
            dict_out.update({"difference_arrays_by_strategy": dict_diffs})
            type_out = "L"
        else:
            type_out = "X"

        return type_out, dict_out




    ##  operational functions

    def get_scalar_diff_arrays(self):

        tp_end = self.fields_time_periods[-1]
        dict_out = {}

        for vs in self.variable_specifications:
            if self.variable_trajectory_group != None:
                tgs_loops = self.required_tg_specs
            else:
                tgs_loops = [None]
            for tgs in tgs_loops:
                # get the vector (dim is by scenario, sorted by self.fields_id )
                vec_tp_end = self.ordered_trajectory_arrays[(vs, tgs)]["data"][:,-1]
                tups_id_coords = [tuple(x) for x in np.array(self.ordered_trajectory_arrays[(vs, tgs)]["id_coordinates"])]
                # order the max/min scalars
                vec_scale_max = np.array([self.dict_variable_info[(vs, tgs)]["max_scalar"][x] for x in tups_id_coords])
                vec_scale_min = np.array([self.dict_variable_info[(vs, tgs)]["min_scalar"][x] for x in tups_id_coords])

                # difference, in final time period, between scaled value and baseline value-dimension is # of scenarios
                dict_tp_end_delta = {
                    "max_tp_end_delta": vec_tp_end*(vec_scale_max - 1),
                    "min_tp_end_delta": vec_tp_end*(vec_scale_min - 1)
                }

                dict_out.update({(vs, tgs): dict_tp_end_delta})

        return dict_out
    
    
    def mix_tensors(self, vec_b0, vec_b1, vec_mix, constraints_mix: tuple = (0, 1)):

        v_0 = np.array(vec_b0)
        v_1 = np.array(vec_b1)
        v_m = np.array(vec_mix)


        if constraints_mix != None:
            if constraints_mix[0] >= constraints_mix[1]:
                raise ValueError("Constraints to the mixing vector should be passed as (min, max)")
            v_alpha = v_m.clip(*constraints_mix)
        else:
            v_alpha = np.array(vec_mix)

        if len(v_alpha.shape) == 0:
            v_alpha = float(v_alpha)
            check_val = len(set([v_0.shape, v_1.shape]))
        else:
            check_val = len(set([v_0.shape, v_1.shape, v_alpha.shape]))

        if check_val > 1:
            raise ValueError("Incongruent shapes in mix_tensors")

        return v_0*(1 - v_alpha) + v_1*v_alpha

    
    def ordered_by_ota_from_fid_dict(self, dict_in: dict, key_tuple: tuple):
        return np.array([dict_in[tuple(x)] for x in np.array(self.ordered_trajectory_arrays[key_tuple]["id_coordinates"])])




    ## uncertainty fan functions
    
     # construct the "ramp" vector for uncertainties
    def build_ramp_vector(self, tuple_param):

        if tuple_param == None:
            tuple_param = self.get_f_fan_function_parameter_defaults(self.uncertainty_fan_function_type)

        if len(tuple_param) == 4:
            tp_0 = self.time_period_end_certainty
            n = len(self.time_periods) - tp_0 - 1

            return np.array([int(i > tp_0)*self.f_fan(i - tp_0 , n, *tuple_param) for i in range(len(self.time_periods))])
        else:
            raise ValueError(f"Error: tuple_param {tuple_param} in build_ramp_vector has invalid length. It should have 4 parameters.")

    # basic function that determines the shape; based on a generalization of the sigmoid (includes linear option)
    def f_fan(self, x, n, a, b, c, d):
        #
        # *defaults*
        #
        # for linear: 
        #    set a = 0, b = 2, c = 1, d = n/2 
        # for sigmoid:
        #    set a = 1, b = 0, c = math.e, d = n/2
        #
        #
        return (a*n + b*x)/(n*(1 + c**(d - x)))

    # parameter defaults for the fan, based on the number of periods n
    def get_f_fan_function_parameter_defaults(self, n: int, fan_type: str, return_type: str = "params"):
        dict_ret = {
            "linear": (0, 2, 1, n/2),
            "sigmoid": (1, 0, math.e, n/2)
        }

        if return_type == "params":
            return dict_ret[fan_type]
        elif return_type == "keys":
            return list(dict_ret.keys())
        else:
            str_avail_keys = ", ".join(list(dict_ret.keys()))
            raise ValueError(f"Error: invalid return_type '{return_type}'. Ensure it is one of the following: {str_avail_keys}.")

    # verify fan function parameters
    def get_fan_function_parameters(self, fan_type):
        
        if type(fan_type) == str:
            n = len(self.time_periods) - self.time_period_end_certainty
            keys = self.get_f_fan_function_parameter_defaults(n, fan_type, "keys")
            if fan_type in keys:
                return self.get_f_fan_function_parameter_defaults(n, fan_type, "params")
            else:
                str_avail_keys = ", ".join(keys)
                raise ValueError(f"Error: no defaults specified for uncertainty fan function of type {fan_type}. Use a default or specify parameters a, b, c, and d. Default functional parameters are available for each of the following: {str_avail_keys}")
        elif type(fan_type) == tuple:
            if len(fan_type) == 4:
                if set([type(x) for x in fan_type]).issubset({int, float}):
                    return fan_type
                else:
                    raise ValueError(f"Error: fan parameter specification {fan_type} contains invalid parameters. Ensure they are numeric (int or float)")
            else:
                raise ValueError(f"Error: fan parameter specification {fan_type} invalid. 4 Parameters are required.")
            
   

    def build_futures(self, n_samples: int, random_seed: int):
        print(f"sampling {self.id_values}")

        
    def generate_future(self, lhs_trial: float, lhs_trial_design: float = 1.0, constraints_mix_tg: tuple = (0, 1), baseline_future_q: bool = False):
        
        # index by variable_specification at keys
        dict_out = {}
        
        if not self.variable_trajectory_group == None:
            
            #list(set([x[0] for x in self.ordered_trajectory_arrays.keys()]))
            cat_mix = self.dict_required_tg_spec_fields["mixing_trajectory"]
            cat_b0 = self.dict_required_tg_spec_fields["trajectory_boundary_0"]
            cat_b1 = self.dict_required_tg_spec_fields["trajectory_boundary_1"]

            #if self.xl_type == "x":

            # use mix between 0/1 (0 = 100% trajectory_boundary_0, 1 = 100% trajectory_boundary_1)
            for vs in self.variable_specifications:
                
                dict_arrs = {
                    cat_b0: self.ordered_trajectory_arrays[(vs, cat_b0)]["data"],
                    cat_b1: self.ordered_trajectory_arrays[(vs, cat_b1)]["data"],
                    cat_mix: self.ordered_trajectory_arrays[(vs, cat_mix)]["data"]
                }


                if (baseline_future_q):
                    # for trajectory groups, the baseline is the specified mixing vector
                    arr_out = self.mix_tensors(dict_arrs[cat_b0], dict_arrs[cat_b1], dict_arrs[cat_mix], constraints_mix_tg)
                else:
                    arr_out = self.mix_tensors(dict_arrs[cat_b0], dict_arrs[cat_b1], lhs_trial, constraints_mix_tg)

                if self.xl_type == "L":
                    
                    if lhs_trial_design < 0:
                        raise ValueError(f"The value of lhs_trial_design = {lhs_trial_design} is invalid. lhs_trial_design must be >= 0.")
                    #
                    # if the XL is an L, then we use the modified future as a base (reduce to include only baseline strategy), then add the uncertainty around the strategy effect
                    #
                    
                    n_strat = len(self.dict_id_values[self.field_strategy_id])
                    # get id coordinates( any of cat_mix, cat_b0, or cat_b1 would work -- use cat_mix)
                    df_ids_ota = pd.concat([self.ordered_trajectory_arrays[(vs, cat_mix)]["id_coordinates"].copy().reset_index(drop = True), pd.DataFrame(arr_out, columns = self.fields_time_periods)], axis = 1)
                    w = np.where(df_ids_ota[self.field_strategy_id] == self.dict_baseline_ids[self.field_strategy_id])
                    df_ids_ota = df_ids_ota.iloc[w[0].repeat(n_strat)].reset_index(drop = True)
                    arr_out = np.array(df_ids_ota[self.fields_time_periods])
                    
                    l_modified_cats = []
                    inds0 = set(np.where(self.dict_strategy_info["baseline_strategy_data_table"][self.field_var_spec] == vs)[0])
                    
                    for cat_cur in [cat_b0, cat_b1, cat_mix]:
                        
                        # get the index for the current vs/cat_cur
                        inds = np.sort(np.array(list(inds0 & set(np.where(self.dict_strategy_info["baseline_strategy_data_table"][self.field_trajgroup_spec] == cat_cur)[0]))))
                        n_inds = len(inds)
                        df_ids0 = self.dict_strategy_info["baseline_strategy_data_table"][[x for x in self.fields_id if (x != self.field_strategy_id)]].loc[inds.repeat(n_strat)].reset_index(drop = True)
                        new_strats = list(np.zeros(len(df_ids0)).astype(int))
                        
                        # initialize as list - we only do this to guarantee the sort is correct
                        df_future_strat = np.zeros((n_inds*len(self.dict_id_values[self.field_strategy_id]), len(self.fields_time_periods)))
                        ind_repl = 0
                        
                        ##  start loop
                        for strat in self.dict_id_values[self.field_strategy_id]:

                            # strategy ids
                            new_strats[ind_repl*n_inds:((ind_repl + 1)*n_inds)] = [strat for x in inds]

                            # get the strategy difference that is adjusted by lhs_trial_delta; if baseline strategy, use 0s
                            df_repl = np.zeros((n_inds, len(self.fields_time_periods))) if (strat == self.dict_baseline_ids[self.field_strategy_id]) else self.dict_strategy_info["difference_arrays_by_strategy"][strat][inds, :]*lhs_trial_design
                            #df_repl = pd.concat([df_ids.reset_index(drop = True), pd.DataFrame(df_repl, columns = self.fields_time_periods)], axis = 1)
                            #df_repl = pd.DataFrame(df_repl, columns = self.fields_time_periods)
                            
                            np.put(df_future_strat, range(n_inds*len(self.fields_time_periods)*ind_repl, n_inds*len(self.fields_time_periods)*(ind_repl + 1)), df_repl)
                            
                            #if init_q:
                            #    df_future_strat = [df_repl for x in self.dict_id_values[self.field_strategy_id]]
                            #    init_q = False
                            #else:
                            #    df_future_strat[ind_repl] = df_repl

                            ind_repl += 1

                        #df_future_strat = pd.concat(df_future_strat, axis = 0).reset_index(drop = True)
                        df_ids0[self.field_strategy_id] = new_strats
                        df_future_strat = pd.concat([df_ids0, pd.DataFrame(df_future_strat, columns = self.fields_time_periods)], axis = 1).sort_values(by = self.fields_id).reset_index(drop = True)
                        l_modified_cats.append(dict_arrs[cat_cur] + np.array(df_future_strat[self.fields_time_periods]))
                        
                    arr_out = self.mix_tensors(*l_modified_cats, constraints_mix_tg)
                    
                    #
                    # one option for this approach is to compare the difference between the "L" design uncertainty and the baseline and add this to the uncertain future (final array)
                    #
                    
                dict_out.update({vs: arr_out})

                
        else:
             
            rv = self.uncertainty_ramp_vector
            
            for vs in self.variable_specifications:
                # order the uniform scaling by the ordered trajectory arrays
                vec_unif_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["uniform_scaling_q"], (vs, None))
                # gives 1s where we keep standard fanning (using the ramp vector) and 0s where we use uniform scaling 
                vec_base = 1 - vec_unif_scalar
                
                if max(vec_unif_scalar) > 0:
                    vec_max_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["max_scalar"], (vs, None))
                    vec_min_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["min_scalar"], (vs, None))
                    vec_unif_scalar = vec_unif_scalar*(vec_min_scalar + lhs_trial*(vec_max_scalar - vec_min_scalar))
                    
                vec_unif_scalar = np.array([vec_unif_scalar]).transpose()
                vec_base = np.array([vec_base]).transpose()
                
                delta_max = self.scalar_diff_arrays[(vs, None)]["max_tp_end_delta"]
                delta_min = self.scalar_diff_arrays[(vs, None)]["min_tp_end_delta"]
                delta_diff = delta_max - delta_min
                delta_val = delta_min + lhs_trial*delta_diff

                array_out = self.ordered_trajectory_arrays[(vs, None)]["data"] + (rv * np.array([delta_val]).transpose())
                array_out = array_out*vec_base + vec_unif_scalar*self.ordered_trajectory_arrays[(vs, None)]["data"]
                
                dict_out.update({vs: array_out})

        return dict_out