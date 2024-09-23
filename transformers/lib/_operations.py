

import pandas as pd


import sisepuede.core.model_attributes as ma
import sispuede.utilities._toolbox as sf
from sisepuede.transformers.lib._classes import Transformer


#####################################
###                               ###
###    COMBINE TRANSFORMATIONS    ###
###                               ###
#####################################

def combine_transformers(
    transformers: Union[List[Transformer], Transformer],
) -> Union[Transformer, None]:
    """
    Combine multiple transformations 
    """

    return None



class Transformers:
    """
    Build energy transformations using general transformations defined in
        auxiliary_definitions_transformations. Wraps more general forms from 
        auxiliary_definitions_transformations into functions and classes
        with shared ramps, paramters, and build functionality.

    NOTE: To update transformations, users need to follow three steps:

        1. Use or create a function in auxiliarty_definitions_transformations, 
            which modifies an input DataFrame, using the ModelAttributes object
            and any approprite SISEPUEDE models. 

        2. If the transformation is not the composition of existing 
            transformation functions, create a transformation definition 
            function using the following functional template:

             def transformation_sabv_###(
                df_input: pd.DataFrame,
                strat: Union[int, None] = None,
                **kwargs
             ) -> pd.DataFrame:
                #DOCSTRING
                ...
                return df_out

            This function is where parameterizations are defined (can be passed 
                through dict_config too if desired, but not done here)

            If using the composition of functions, can leverage the 
            trl.Transformer composition functionality, which lets the user
            enter lists of functions (see ?trl.Transformer for more 
            information)

        3. Finally, define the Transformer object using the 
            `trl.Transformer` class, which connects the function to the 
            Strategy name in attribute_strategy_id, assigns an id, and 
            simplifies the organization and running of strategies. 


    Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and
		coordination
    - dict_config: configuration dictionary used to pass parameters to 
        transformations. See ?TransformerEnergy._initialize_parameters() for
        more information on requirements.
    - dir_jl: location of Julia directory containing Julia environment and 
        support modules
    - fp_nemomod_reference_files: directory housing reference files called by
		NemoMod when running electricity model. Required to access data in 
        EnergyProduction. Needs the following CSVs:

        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
    - df_input: optional data frame to initialize with. If used, transformations
        can be called without arguments.
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
		used in Julia NemoMod Electricity model
        * If None, defaults to a temporary path sql database
    - logger: optional logger object
    - model_afolu: optional AFOLU object to pass for property and method access
    - model_enerprod: optional EnergyProduction object to pass for property and
        method access
        * NOTE: If passing, `dir_jl` and `fp_nemomod_reference_files` are 
            ignored (can pass None to those arguments if passing 
            model_enerprod)
    """
    
    def __init__(self,
        model_attributes: ma.ModelAttributes,
        dict_config: Dict,
        dir_jl: Union[str, None],
        fp_nemomod_reference_files: Union[str, None],
        df_input: Union[pd.DataFrame, None] = None,
        field_region: Union[str, None] = None,
		fp_nemomod_temp_sqlite_db: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
    ):

        self.logger = logger

        self._initialize_attributes(field_region, model_attributes)
        self._initialize_config(dict_config = dict_config)
    
        self._initialize_parameters(dict_config = dict_config)
        self._initialize_ramp()
        self._initialize_baseline_inputs(df_input)
        self._initialize_transformations()

        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def get_ramp_characteristics(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Get parameters for the implementation of transformations. Returns a 
            tuple with the following elements:

            (
                n_tp_ramp,
                vir_renewable_cap_delta_frac,
                vir_renewable_cap_max_frac,
                year_0_ramp, 
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires those keys in dict_config to set. If not found, will set
            the following defaults:
                * year_0_ramp: 9th year (10th time period)
                * n_tp_ramp: n_tp - t0_ramp - 1 (ramps to 1 at final time 
                    period)
                * vir_renewable_cap_delta_frac: 0.1
                * vir_renewable_cap_max_frac: 0.01 

        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_entc_renewable: list of categories to tag as renewable 
                for the Renewable Targets transformation.
        """

        dict_config = self.config if not isinstance(dict_config, dict) else dict_config
        n_tp = len(self.time_periods.all_time_periods)

        # get first year of non-baseline
        default_year = self.time_periods.all_years[min(9, n_tp - 1)]
        year_0_ramp = dict_config.get(self.key_config_year_0_ramp)
        year_0_ramp = (
            self.time_periods.all_years[default_year] 
            if not sf.isnumber(year_0_ramp, integer = True)
            else year_0_ramp
        )

        # shift by 2--1 to account for baseline having no uncertainty, 1 for py reindexing
        default_n_tp_ramp = n_tp - self.time_periods.year_to_tp(year_0_ramp) - 1
        n_tp_ramp = dict_config.get(self.key_config_n_tp_ramp)
        n_tp_ramp = (
            default_n_tp_ramp
            if not sf.isnumber(n_tp_ramp, integer = True)
            else n_tp_ramp
        )

        # get VIR (get_vir_max_capacity) delta_frac
        default_vir_renewable_cap_delta_frac = 0.01
        vir_renewable_cap_delta_frac = dict_config.get(self.key_config_vir_renewable_cap_delta_frac)
        vir_renewable_cap_delta_frac = (
            default_vir_renewable_cap_delta_frac
            if not sf.isnumber(vir_renewable_cap_delta_frac)
            else vir_renewable_cap_delta_frac
        )
        vir_renewable_cap_delta_frac = float(sf.vec_bounds(vir_renewable_cap_delta_frac, (0.0, 1.0)))

        # get VIR (get_vir_max_capacity) max_frac
        default_vir_renewable_cap_max_frac = 0.05
        vir_renewable_cap_max_frac = dict_config.get(self.key_config_vir_renewable_cap_max_frac)
        vir_renewable_cap_max_frac = (
            default_vir_renewable_cap_max_frac
            if not sf.isnumber(vir_renewable_cap_max_frac)
            else vir_renewable_cap_max_frac
        )
        vir_renewable_cap_max_frac = float(sf.vec_bounds(vir_renewable_cap_max_frac, (0.0, 1.0)))

        tup_out = (
            n_tp_ramp,
            vir_renewable_cap_delta_frac,
            vir_renewable_cap_max_frac,
            year_0_ramp, 
        )

        return tup_out



    def _initialize_attributes(self,
        field_region: Union[str, None],
        model_attributes: ma.ModelAttributes,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_transformer
            * self.key_region
            * self.model_attributes
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in transformations_energy")

        # get strategy attribute, baseline strategy, and some fields
        attribute_transformer = model_attributes.get_other_attribute_table(
            model_attributes.dim_transformer_code
        )

        baseline_strategy = int(
            attribute_strategy.table[
                attribute_strategy.table["baseline_strategy_id"] == 1
            ][attribute_strategy.key].iloc[0]
        )
        
        field_region = (
            model_attributes.dim_region 
            if (field_region is None) 
            else field_region
        )

        # add technology attribute
        attribute_technology = model_attributes.get_attribute_table(
            model_attributes.subsec_name_entc
        )

        # set some useful classes
        time_periods = sc.TimePeriods(model_attributes)
        regions = sc.Regions(model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_strategy = attribute_strategy
        self.attribute_technology = attribute_technology
        self.baseline_strategy = baseline_strategy
        self.key_region = field_region
        self.key_strategy = attribute_strategy.key
        self.model_attributes = model_attributes
        self.time_periods = time_periods
        self.regions = regions

        return None



    def _initialize_baseline_inputs(self,
        df_inputs: Union[pd.DataFrame, None],
    ) -> None:
        """
        Initialize the baseline inputs dataframe based on the initialization 
            value of df_inputs. It not initialied, sets as None. Sets the 
            following properties:

            * self.baseline_inputs
        """

        baseline_inputs = (
            self.transformation_baseline(df_inputs, strat = self.baseline_strategy) 
            if isinstance(df_inputs, pd.DataFrame) 
            else None
        )

        self.baseline_inputs = baseline_inputs

        return None



    def _initialize_config(self,
        dict_config: Union[Dict[str, Any], None],
    ) -> None:
        """
        Define the configuration dictionary and paramter keys. Sets the 
            following properties:

            * self.config (configuration dictionary)
            * self.key_* (keys)
            
        Function Arguments
        ------------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Can include the following keys:

            * "n_tp_ramp": number of time periods to use to ramp up. If None or
                not specified, builds to full implementation by the final time
                period
            * "year_0_ramp": last year with no diversion from baseline strategy
                (baseline for implementation ramp)
        """

        dict_config = {} if not isinstance(dict_config, dict) else dict_config

        # set parameters
        self.config = dict_config
        self.key_config_n_tp_ramp = "n_tp_ramp"
        self.key_config_year_0_ramp = "year_0_ramp" 

        return None



    
    def _initialize_parameters(self,
        dict_config: Union[Dict[str, Any], None] = None,
    ) -> None:
        """
        Define key parameters for transformation. For keys needed to initialize
            and define these parameters, see ?self._initialize_config
      
        """

        dict_config = (
            self.config 
            if not isinstance(dict_config, dict) 
            else dict_config
        )

        # get parameters from configuration dictionary
        (
            n_tp_ramp,
            year_0_ramp
        ) = self.get_ramp_characteristics()


        ##  SET PROPERTIES

        self.n_tp_ramp = n_tp_ramp
        self.year_0_ramp = year_0_ramp

        return None
    


    def _initialize_ramp(self,
    ) -> None: 
        """
        Initialize the ramp vector for implementing transformations. Sets the 
            following properties:

            * self.vec_implementation_ramp
        """
        
        vec_implementation_ramp = self.build_implementation_ramp_vector()
        

        ##  SET PROPERTIES

        self.vec_implementation_ramp = vec_implementation_ramp

        return None



    ################################################
    ###                                          ###
    ###    OTHER NON-TRANSFORMATION FUNCTIONS    ###
    ###                                          ###
    ################################################
                        
    def build_implementation_ramp_vector(self,
        year_0: Union[int, None] = None,
        n_tp_ramp: Union[int, None] = None,
    ) -> np.ndarray:
        """
        Build the implementation ramp vector

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
		- year_0: last year without change from baseline
        - n_tp_ramp: number of years to go from 0 to 1
        """
        year_0 = self.year_0_ramp if (year_0 is None) else year_0
        n_tp_ramp = self.n_tp_ramp if (n_tp_ramp is None) else n_tp_ramp

        tp_0 = self.time_periods.year_to_tp(year_0)
        n_tp = len(self.time_periods.all_time_periods)
        
        # use ramp vector function
        vec_out = sf.ramp_vector(
            n_tp, 
            alpha_logistic = 0.0, # default to linear 
            r_0 = tp_0,
            # r_1 = tp_0 + n_tp_ramp,
            # window_logistic = (-8, 8),
        )

        return vec_out



    def build_msp_cap_vector(self,
        vec_ramp: np.ndarray,
    ) -> np.ndarray:
        """
        Build the cap vector for MSP adjustments. 
            Derived from self.vec_implementation_ramp

        Function Arguments
		------------------
        - vec_ramp: implementation ramp vector to use. Will set cap at first 
            non-zero period

        Keyword Arguments
		-----------------
        """
        vec_out = np.array([
            (self.model_enerprod.drop_flag_tech_capacities if (x == 0) else 0) 
            for x in vec_ramp
        ])

        return vec_out
    


    def build_strategies_long(self,
        df_input: Union[pd.DataFrame, None] = None,
        include_base_df: bool = True,
        strategies: Union[List[str], List[int], None] = None,
    ) -> pd.DataFrame:
        """
        Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
        - df_input: baseline (untransformed) data frame to use to build 
            strategies. Must contain self.key_region and 
            self.model_attributes.dim_time_period in columns. If None, defaults
            to self.baseline_inputs
        - include_base_df: include df_input in the output DataFrame? If False,
            only includes strategies associated with transformation 
        - strategies: strategies to build for. Can be a mixture of strategy_ids
            and names. If None, runs all available. 
        """

        # INITIALIZE STRATEGIES TO LOOP OVER

        strategies = (
            self.all_transformations_non_baseline
            if strategies is None
            else [x for x in self.all_transformations_non_baseline if x in strategies]
        )
        strategies = [self.get_strategy(x) for x in strategies]
        strategies = sorted([x.id for x in strategies if x is not None])
        n = len(strategies)


        # LOOP TO BUILD
        
        t0 = time.time()
        self._log(
            f"TransformersEnergy.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        # initialize baseline
        df_out = (
            self.transformation_baseline(df_input)
            if df_input is not None
            else (
                self.baseline_inputs
                if self.baseline_inputs is not None
                else None
            )
        )

        if df_out is None:
            return None

        # initialize to overwrite dataframes
        iter_shift = int(include_base_df)
        df_out = [df_out for x in range(len(strategies) + iter_shift)]

        for i, strat in enumerate(strategies):
            t0_cur = time.time()
            transformation = self.get_strategy(strat)

            if transformation is not None:
                try:
                    df_out[i + iter_shift] = transformation(df_out[i + iter_shift])
                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built transformation {self.key_strategy} = {transformation.id} ('{transformation.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )

                except Exception as e: 
                    df_out[i + 1] = None
                    self._log(
                        f"\tError trying to build transformation {self.key_strategy} = {transformation.id}: {e}",
                        type_log = "error"
                    )
            else:
                df_out[i + iter_shift] = None
                self._log(
                    f"\tTransformer {self.key_strategy} not found: check that a support_classes.Transformer object has been defined associated with the code.",
                    type_log = "warning"
                )

        # concatenate, log time elapsed and completion
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out
    


    def check_implementation_ramp(self,
        vec_implementation_ramp: np.ndarray,
        df_input: Union[pd.DataFrame, None] = None,
    ) -> Union[np.ndarray, None]:
        """
        Check that vector `vec_implementation_ramp` ramp is the same length as 
            `df_input` and that it meets specifications for an implementation
            vector. If `df_input` is not specified, use `self.baseline_inputs`. 
        
        If anything fails, return `self.vec_implementation_ramp`.
        """

        df_input = (
            self.baseline_inputs 
            if not isinstance(df_input, pd.DataFrame)
            else df_input
        )

        out = tbg.check_implementation_ramp(
            vec_implementation_ramp,
            df_input,
            self,
        )

        return out

        

    def get(self,
        transformer: Union[int, str, None],
        field_transformer_code: str = "transformer_code",
        field_transformer_name: str = "transformer",
    ) -> None:
        """
        Get `transformer` based on transformer code, id, or name
        
        If strat is None or an invalid valid of strat is entered, returns None; 
            otherwise, returns the trl.Transformer object. 

            
        Function Arguments
        ------------------
        - transformer: transformer_id, transformer name, or transformer code to 
            use to retrieve sc.Trasnformation object
            
        Keyword Arguments
        ------------------
        - field_transformer_code: field in transformer attribute table 
            containing the transformer code
        - field_transformer_name: field in transformer attribute table 
            containing the transformer name
        """

        if not (sf.isnumber(transformer, integer = True) | isinstance(strat, str)):
            return None

        dict_code_to_strat = self.attribute_transformer.field_maps.get(
            f"{field_strategy_code}_to_{self.attribute_strategy.key}"
        )
        dict_name_to_strat = self.attribute_transformer.field_maps.get(
            f"{field_strategy_name}_to_{self.attribute_strategy.key}"
        )

        # check strategy by trying both dictionaries
        if isinstance(strat, str):
            strat = (
                dict_name_to_strat.get(strat)
                if strat in dict_name_to_strat.keys()
                else dict_code_to_strat.get(strat)
            )

        out = (
            None
            if strat not in self.attribute_strategy.key_values
            else self.dict_transformations.get(strat)
        )
        
        return out




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

        return None

