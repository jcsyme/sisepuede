import numpy as np
import pandas as pd
import sisepuede.core.support_classes as sc
import sisepuede.transformers as trf
import sisepuede.utilities._toolbox as sf

from sisepuede.transformers.strategies import is_strategies
from typing import *




######################
#    SOME GLOBALS    #
######################

# fields
_FIELD_LEVER_IMPLEMENATION_TABLE_MAXIMUM_MAGNITUDE = "maximum_magnitude"
_FIELD_LEVER_IMPLEMENATION_TABLE_SECTOR = "sector"
_FIELD_LEVER_IMPLEMENATION_TABLE_START_PERIOD = "start_period"
_FIELD_LEVER_IMPLEMENATION_TABLE_SUBSECTOR = "subsector"
_FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_CODE = "transformer_code"
_FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_DESCRIPTION = "transformer_description"
_FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_NAME = "transformer_name"
_FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_UNIT = "transformer_unit"

# module uuid
_MODULE_UUID = "853B026C-3CBA-488C-8EA3-0890BF4431FD"

# placeholds
_PLACEHOLDER_FIELDS_LEVER_IMPLEMENTATION_TABLE_STRATEGIES = "__PLACEHOLDER_STRATEGIES__"

# prefixes
_PREFIX_FIELD_STRATEGY = "strategy_"





#####################################
#    START FUNCTIONS AND CLASSES    #
#####################################

class LeversImplementationTable:
    """Build a strategy-based "lever implementation" table for visualization in 
        long format. Used to show magnitudes.

    Initialization Arguments
    ------------------------
    strategies : Strategies
        Strategies object used to coordatinate development of strategies
        and policy portfolios
    """

    
    def __init__(self,
        strategies: 'Strategies',
    ) -> None:

        self._initialize_strategy_attributes(strategies, )
        self._initialize_table_properties()
        self._initialize_summarizer()

        return None

    

    def _initialize_strategy_attributes(self,
        strategies, 
    ) -> None:
        """Initialize and check strategies and associated attributes. Sets 
            the following properties:

            * self.field_year
            * self.model_attributes
            * self.strategies
            * self.time_periods
            * self.transformations
            * self.transformers
        """

        if not is_strategies(strategies):
            tp = type(strategies)
            raise TypeError(f"Invalid type '{tp}' for strategies: must be a Strategies object")


        time_periods = sc.TimePeriods(strategies.model_attributes, )
        key_vir = (
            strategies
            .transformations
            .transformers
            .key_config_vec_implementation_ramp
        )

        
        ##  SET PROPERTIES

        self.field_year = time_periods.field_year
        self.key_vir = key_vir
        self.model_attributes = strategies.model_attributes
        self.strategies = strategies
        self.time_periods = time_periods
        self.transformations = strategies.transformations
        self.transformers = strategies.transformations.transformers

        return None


    
    def _initialize_summarizer(self,
    ) -> None:
        """Initialize the transformation summarizer. Sets the following 
            properties:

            * self.transformation_summarizer
        """

        transformation_summarizer = TransformationSummarizer(
            self.transformers,
        )


        ##  SET PROPERTIES

        self.transformation_summarizer = transformation_summarizer

        return None
    


    def _initialize_table_properties(self,
    ) -> None:
        """Initialize some properties of the table, including column ordering, 
            etc. Sets the following properties:

            * self.dict_sector_name_replacement
                Dictionary mapping any sector names to display replacements
            * self.fields_ordered
                Ordered output fields (including strategies placeholder) for 
                table
        """

        fields_ordered = [
            _FIELD_LEVER_IMPLEMENATION_TABLE_SECTOR,
            _FIELD_LEVER_IMPLEMENATION_TABLE_SUBSECTOR,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_CODE,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_NAME,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_DESCRIPTION,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_UNIT,
            _FIELD_LEVER_IMPLEMENATION_TABLE_START_PERIOD,
            _FIELD_LEVER_IMPLEMENATION_TABLE_MAXIMUM_MAGNITUDE,
            _PLACEHOLDER_FIELDS_LEVER_IMPLEMENTATION_TABLE_STRATEGIES,
            self.field_year
        ]

        dict_sector_name_replacement = {
            "pflo": "Cross"
        }

        
        ##  SET PROPERTIES

        self.dict_sector_name_replacement = dict_sector_name_replacement
        self.fields_ordered = fields_ordered

        return None
        


    ################################
    #    TABLE UPDATE FUNCTIONS    #
    ################################

    def _update_dict_from_element(self,
        dict_table_base: Dict[str, List],           
        key: str,
        element: Any,
        vec_time: np.ndarray,
    ) -> None:
        """Update a table dictionary using an arbitrary element (non-vector)

        Function Arguments
        ------------------
        dict_table_base : Dict[str, List]
            Base table dictionary
        key : str
            Key in the dictionary (field in table) to update
        element : Any
            Any element (e.g., int, float, str, etc.) to repeat length vec_time
        vec_time : np.ndarray
            Vector of time periods needed to expand on
            
        Keyword Arguments
        -----------------
        """
        vec = dict_table_base.get(key)
        if not isinstance(vec, list):
            return None
        
        # otherwise, append to the dictionary
        (
            dict_table_base[key]
            .extend(
                list(
                    np.repeat(element, len(vec_time))
                )
            )
        )
        

        return None


    
    def _update_dict_from_transformation_transformer_property(self,
        dict_table_base: Dict[str, List],           
        key: str,
        transformation: 'Transformation',
        property_name: str, 
        vec_time: np.ndarray,
        return_on_missing: Any = None,
    ) -> None:
        """Update a table dictionary using a property from the transformer 
            associated with the transformer.

        Function Arguments
        ------------------
        dict_table_base : Dict[str, List]
            Base table dictionary
        key : str
            Key in the dictionary (field in table) to update
        transformation : Transformation
            Transformation object storing relevant information
        property_name : str
            Property name in the Transformation object to use to assign
        vec_time : np.ndarray
            Vector of time periods needed to expand on
            
        Keyword Arguments
        -----------------
        return_on_missing : Any
            Optional value to return if no value is present
        """

        base_transformer = self.transformers.get_transformer(
            transformation.transformer_code,
        )

        # try to get the property
        prop = getattr(base_transformer, property_name, None)
        if prop is None:
            return return_on_missing
        
        # if available, pass as an element
        self._update_dict_from_element(
            dict_table_base,
            key,
            prop,
            vec_time,
        )
        
        return None



    def _update_dict_from_vector(self,
        dict_table_base: Dict[str, List],           
        key: str,
        vector: np.ndarray,
    ) -> None:
        """Update a table dictionary using a property from the transformation.

        Function Arguments
        ------------------
        dict_table_base : Dict[str, List]
            Base table dictionary
        key : str
            Key in the dictionary (field in table) to update
        vector : np.ndarray
            Vector to pass. Must have length vec_time
            
        Keyword Arguments
        -----------------
        """

        col = dict_table_base.get(key, )
        if not isinstance(col, list):
            return None
        
    
        # otherwise, append to the dictionary
        (
            dict_table_base[key]
            .extend(list(vector), )
        )

        return None


        
    def _update_code(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the transformation code in the table associated with the transformation
        """

        self._update_dict_from_transformation_transformer_property(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_CODE,
            transformation,
            "code",
            vec_time,
        )
        
        return None



    def _update_desc(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the description in the table associated with the 
            transformation
        """
        self._update_dict_from_transformation_transformer_property(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_DESCRIPTION,
            transformation,
            "description",
            vec_time,
            return_on_missing = "none",
        )

        return None



    def _update_maximum_magnitude(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the maximum magnitude associated with the transformation
        """

        # TEMPORARY--THIS IS A PLACEHOLDER
            
        # update sector
        self._update_dict_from_element(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_MAXIMUM_MAGNITUDE,
            0.0,
            vec_time,
        )

        return None



    def _update_name(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the description in the table associated with the transformation
        """
        self._update_dict_from_transformation_transformer_property(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_NAME,
            transformation,
            "name",
            vec_time,
        )

        return None



    def _update_sector_subsector(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the description in the table associated with the transformation
        """

        subsector, sector = self.get_sector_codes(transformation, )
            
        # update sector
        self._update_dict_from_element(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_SECTOR,
            sector,
            vec_time,
        )

        # update subsector
        self._update_dict_from_element(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_SUBSECTOR,
            subsector,
            vec_time,
        )
        
        return None



    def _update_units_description(self,
        dict_table_base: Dict[str, List],           
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the description in the table associated with the transformation
        """
        self._update_dict_from_transformation_transformer_property(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_UNIT,
            transformation,
            "description_units",
            vec_time,
        )
        
        """
        desc_units = self.get_units_description(transformation, )
            
        # update sector
        self._update_dict_from_element(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_UNIT,
            desc_units,
            vec_time,
        )
        """
        return None


        
    def _update_vir_and_start_period(self,
        dict_table_base: Dict[str, List],   
        transformation: 'Transformation',
        vec_time: np.ndarray,
        as_str: bool = False,
        decimals: int = 1,
        include_magnitude: bool = True,
    ) -> None:
        """Update the implementation ramp vector in the table associated 
            with the transformation
        """

        # summarize?
        magnitude = 1
        if include_magnitude:
            magnitude = (
                self
                .transformation_summarizer
                .summarize_transformation_magnitude(transformation, )
            )

        # get the ramp
        vir_try = transformation.dict_parameters.get(self.key_vir, )
        vec_implementation_ramp = (
            vir_try.copy()
            if isinstance(vir_try, np.ndarray)
            else self.transformers.vec_implementation_ramp.copy()
        )
            
        vec_implementation_ramp *= magnitude

        # verify consistency
        if vec_implementation_ramp.shape[0] != vec_time.shape[0]:
            raise RuntimeError(f"Invalid shape of vec_implementation_ramp: must be of length {vec_time.shape[0]}")

        # get the start period
        period_0 = int(np.where(vec_implementation_ramp == 0)[0].max())

        # adjust to string if needed
        if as_str:
            vec_implementation_ramp = [
                self.show_perc(x, decimals = decimals, ) for x in vec_implementation_ramp
            ]


        ##  UPDATE DICTS
        
        self._update_dict_from_vector(
            dict_table_base,
            _PLACEHOLDER_FIELDS_LEVER_IMPLEMENTATION_TABLE_STRATEGIES,
            vec_implementation_ramp,
        )

        self._update_dict_from_element(
            dict_table_base,
            _FIELD_LEVER_IMPLEMENATION_TABLE_START_PERIOD,
            period_0,
            vec_time,
        )
        
        return None



    def _update_year(self,
        dict_table_base: Dict[str, List],   
        transformation: 'Transformation',
        vec_time: np.ndarray,
    ) -> None:
        """Update the implementation ramp vector in the table associated 
            with the transformation
        """
        self._update_dict_from_vector(
            dict_table_base,
            self.field_year,
            vec_time,
        )
        
        return None




    ########################
    #    CORE FUNCTIONS    #
    ########################
    
    def build_strategy_field(self,
        strategy: 'Strategy',
        delim: str = ":",
    ) -> str:
        """Build a field for a strategy
        """
        code = strategy.code.replace(delim, "_")
        out = f"{_PREFIX_FIELD_STRATEGY}{code}"

        return out
    

    
    def build_table_for_strategies(self,
        strategy_specifications: Union[int, str, None],
    ) -> pd.DataFrame:
        """Build one component of the level table for Tableau.
    
        Function Arguments
        ------------------
        strategy_specification : Union[int, str, None]
            strategy_id, strategy name, or strategy code to use to retrieve 
            Strategy object
    
        Keyword Arguments
        -----------------
        
        """

        df_out = None
        fields_to_placeholder = []

        for spec in strategy_specifications:
            df_cur = self.build_table_for_strategy(spec, )
            
            # this will get the strategy field
            fields_strat = [
                x for x in df_cur.columns if x not in self.fields_ordered
            ]
            fields_to_placeholder.extend(fields_strat, )

            # set extraction fields for merge
            fields_ext = list(df_cur.columns)
            """
            [
                _FIELD_LEVER_IMPLEMENATION_TABLE_TRANSFORMER_CODE,
                self.field_year
            ]
            fields_ext.extend(fields_strat, )
            """

            # finally, update df out
            if df_out is None:
                df_out = df_cur
                continue

            df_out = (
                pd.merge(
                    df_out,
                    df_cur.get(fields_ext, ),
                    how = "outer",
                )
                .fillna(0.0)
            )


        # get ordered fields
        fields_ordered = []
        for x in self.fields_ordered:
            if x != _PLACEHOLDER_FIELDS_LEVER_IMPLEMENTATION_TABLE_STRATEGIES:
                fields_ordered.append(x)
                continue
            
            fields_ordered.extend(
                sorted(fields_to_placeholder)
            )


        df_out = df_out[fields_ordered]
        
        return df_out 
    


    def build_table_for_strategy(self,
        strategy_specification: Union[int, str, None],
    ) -> pd.DataFrame:
        """Build one component of the level table for Tableau.
    
        Function Arguments
        ------------------
        strategy_specification : Union[int, str, None]
            strategy_id, strategy name, or strategy code to use to retrieve 
            Strategy object
    
        Keyword Arguments
        -----------------
        
        """
    
        ##  SOME INITIALIZATION
        
        # retrieve the strategy and set a shortcut to transformers
        strat = self.strategies.get_strategy(
            strategy_specification,
            stop_on_error = True, 
        )

        
        # get transformations associated with it
        transformation_objs = strat.get_transformation_list(
            strat.transformation_specification,
            self.transformations,
        )
        
        
        ##  ITERATE OVER TRANSFORMATIONS TO BUILD 
        
        # initialize output table
        dict_table_base = self.instantiate_base_individual_strategy_table_dictionary()
        vec_time = np.array(self.time_periods.all_years)
        
        for transformation in transformation_objs:

            # get the transformer for relevant information
            transformer = self.transformers.get_transformer(
                transformation.transformer_code,
            )

            args_from_transformation = (
                dict_table_base, 
                transformation, 
                vec_time,
            )

            # update 
            self._update_code(*args_from_transformation, )
            self._update_desc(*args_from_transformation, )
            self._update_maximum_magnitude(*args_from_transformation, )
            self._update_name(*args_from_transformation, )
            self._update_sector_subsector(*args_from_transformation, )
            self._update_units_description(*args_from_transformation, )
            self._update_vir_and_start_period(*args_from_transformation, )
            self._update_year(*args_from_transformation, )


        ##  BUILD THE OUTPUT DATAFRAME

        # set the strategy's field
        field_strategy = self.build_strategy_field(strat, )

        df_out = (
            pd.DataFrame(
                dict((k, v) for k, v in dict_table_base.items() if len(v) > 0)
            )
            .rename(
                columns = {
                    _PLACEHOLDER_FIELDS_LEVER_IMPLEMENTATION_TABLE_STRATEGIES: field_strategy,
                }
            )
        )
            
        return df_out
        


    def get_sector_codes(self,
        transformation: 'Transformation',
        delim: str = ":",
        uppercase_subsector: bool = True,
    ) -> Tuple[str, str]:
        """Get the subsector and sector associated with a transformer. Returns a 
            tuple of the form

            (subsector_abbreviation, sector, )

        Set uppercase_subsector = False to return a lower case subsector_abbreviation
        """
        # get the associated transformer
        transformer = self.transformers.get_transformer(
            transformation.transformer_code,
        )

        # pull out the subsector abbreviation from the code and try to get the sector
        subsec_abv = transformer.code.split(delim)[1].lower()
        sector = self.model_attributes.get_subsector_attribute(subsec_abv, "sector")
        if sector is None:
            sector = self.dict_sector_name_replacement.get(subsec_abv, )

        # do some output formatting
        subsec_abv = subsec_abv.upper() if uppercase_subsector else subsec_abv
        out = (subsec_abv, sector, )
    
        return out



    def get_units_description(self,
        transformation: 'Transformation',
    ) -> str:
        """Get the units description from the transformer associated with the 
            transformation.
        """
        # get the associated transformer
        transformer = self.transformers.get_transformer(
            transformation.transformer_code,
        )

        # pull out the subsector abbreviation from the code and try to get the sector
        units_descrip = getattr(transformer, "description_units", None)
        
    
        return units_descrip


    
    def instantiate_base_individual_strategy_table_dictionary(self,
    ) -> Dict[str, List]:
        """Initialize a base table dictionary for a strtategy. Does
            Not include year.
        """
        dict_out = dict((x, []) for x in self.fields_ordered)

        return dict_out



    def show_perc(self,
        x: float, 
        decimals: int = 1,
    ) -> str:
        """Convert to display percentage
        """
        disp = str(np.round(100*x, decimals = decimals, ))
        if disp.endswith("."): 
            disp = disp + "0"*decimals
    
        disp = f"{disp}%"
    
        return disp
    




class TransformationSummarizer:

    def __init__(self,
        transformers: 'Transformers',
    ) -> None:
        
        self._initialize_transformers(transformers)
        
        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> float:

        out = self.summarize_transformation_magnitude(*args, )
        
        return out


    
    def _initialize_transformers(self,
        transformers: 'Transformers',               
    ) -> None:
        """Initialize the transformers object
        """
        if not trf.is_transformers(transformers):
            tp = type(transformers)
            raise TypeError(f"Invalid type '{tp}': must be a Transformers object.")


        self.transformers = transformers

        return None
    



    #################################
    #    START SUMMARY FUNCTIONS    #
    #################################
    
    def summarize_from_magnitude(self,
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],     
        key_magnitude: str = "magnitude",
    ) -> float:
        """Summarize transformations based on a single magnitude parameter
        """
        magnitude = dict_params.get(
            key_magnitude,
            kwargs.get(key_magnitude),
        )
    
        return magnitude


    
    def summarize_special_case_agrc_conservation_ag(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the conservation agriculture 
            transformer
        """
        dict_magnitude = dict_params.get("dict_categories_to_magnitude")
        if dict_magnitude is None:
            dict_magnitude = base_transformer(return_dict_magnitude = True, )
            
        magnitude = np.mean(list(dict_magnitude.values()))
    
        return magnitude

    

    def summarize_special_case_lndu_bounds(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the bound classes transformation
        """
        dict_dcm = dict_params.get("dict_directional_categories_to_magnitude")
        if dict_dcm is None:
            return 0
        
        dict_dcm = base_transformer(
            dict_directional_categories_to_magnitude = dict_dcm, 
            return_dict_dcm_only = True, 
        )

        magnitude = sum([v for k, v in dict_dcm.items() if k[1] == "min"])
    
        return magnitude
    


    def summarize_special_case_lsmm_pathways(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the manure management pathways
            transformer
        """
        dict_pathways = dict_params.get("dict_lsmm_pathways")
        if dict_pathways is None:
            dict_pathways = base_transformer(return_pathways = True, )
            
        magnitude = sum(dict_pathways.values())
    
        return magnitude



    def summarize_special_case_lvst_enteric_fermentation(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the enteric fermentation
            transformer
        """
        dict_reductions = dict_params.get("dict_lvst_reductions")
        if dict_reductions is None:
            dict_reductions = base_transformer(return_reductions_dict = True, )

        magnitude = np.mean(list(dict_reductions.values()))
    
        return magnitude



    def summarize_special_case_pflo_ccs(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the industrial CCS
            transformer
        """

        # get efficacy
        key = "dict_magnitude_eff"
        efficacy = dict_params.get(
            key,
            kwargs.get(key, )
        )
            
        # get prevalence
        dict_prev = dict_params.get("dict_magnitude_prev")
        if dict_prev is None:
            dict_prev = base_transformer(return_prevalence_dict = True, )

        # adjust efficacy as dictionary
        dict_efficacy = (
            dict((k, efficacy) for k in dict_prev.keys())
            if sf.isnumber(efficacy)
            else efficacy
        )

        # use a weighted mean real reduction, which is efficacy*prevalence
        n = 0
        magnitude = 0
        for k, v in dict_prev.items():
            eff = dict_efficacy.get(k)
            if eff is None: continue

            magnitude += eff*v
            n += 1

        magnitude /= n 
    
        return magnitude


    
    def summarize_special_case_trns_mode_shift_regional(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the regional mode shift (TRNS)
            transformer
        """
        key = "dict_categories_out"
        dict_cats = dict_params.get(
            key,
            kwargs.get(key),
        )
        
        magnitude = sum(dict_cats.values())

        return magnitude



    def summarize_special_case_wali_pathways(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based wastewater treatment pathways
            transformer
        """
        dict_pathways = dict_params.get("dict_magnitude")
        if dict_pathways is None:
            dict_pathways = base_transformer(return_pathways = True, )
            
        magnitude = sum(dict_pathways.values())

        return magnitude



    def summarize_special_case_waso_anaerobic_compost(self,
        base_transformer: 'Transformer',
        dict_params: Dict[str, Any],
        kwargs: Dict[str, Any],                                     
    ) -> float:
        """Summarize transformations based on the WASO anaerobic and compost 
            transformer
        """

        magnitude_anaerobic = self.summarize_from_magnitude(
            dict_params,
            kwargs,
            key_magnitude = "magnitude_biogas",
        )

        magnitude_compost = self.summarize_from_magnitude(
            dict_params,
            kwargs,
            key_magnitude = "magnitude_compost",
        )

        magnitude = magnitude_anaerobic + magnitude_compost

        return magnitude
        
        

    
    ##  CORE FUNCTION
    
    def summarize_transformation_magnitude(self,
        transformation: 'Transformation',
    ) -> float:
        """Summarize a transformation into a single magnitude
        """
    
        ##  INITIALIZATION
    
        prefix_transformer_code = trf.transformers._MODULE_CODE_SIGNATURE
        
        # get the transformer and look for magnitude etc.
        base_transformer = self.transformers.get_transformer(
            transformation.transformer_code
        )
        
        args, kwargs = sf.get_args(
            base_transformer.function,
            include_defaults = True,
        )
        
        # check if magnitude is defined
        dict_params = transformation.dict_parameters
        

        ##  TRY BASE MAGNITUDE 
        
        magnitude = self.summarize_from_magnitude(
            dict_params,
            kwargs,
            key_magnitude = "magnitude", 
        )

        if magnitude is not None:
            return magnitude
    

        
        ###################################
        #    A NUMBER OF SPECIAL CASES    #
        ###################################

        args_summary = (base_transformer, dict_params, kwargs, )
        
        # AGRC - Increase conservation agriculture
        if base_transformer.code == f"{prefix_transformer_code}:AGRC:INC_CONSERVATION_AGRICULTURE":
            out = self.summarize_special_case_agrc_conservation_ag(*args_summary, )
            return out
    
    
        # ENTC - Least cost solution
        if base_transformer.code == f"{prefix_transformer_code}:ENTC:LEAST_COST_SOLUTION":
            return 1
            
        
        # INEN - Shift fuel heat
        if base_transformer.code == f"{prefix_transformer_code}:INEN:SHIFT_FUEL_HEAT":
            magnitude = self.summarize_from_magnitude(
                dict_params,
                kwargs,
                key_magnitude = "frac_switchable", 
            )
            return magnitude
    
        
        # LNDU - Bound classes
        if base_transformer.code == f"{prefix_transformer_code}:LNDU:BOUND_CLASSES":
            magnitude = self.summarize_special_case_lndu_bounds(*args_summary, )
            return magnitude 
            
            
        # LSMM - Manure management pathways
        if base_transformer.code in [
            f"{prefix_transformer_code}:LSMM:INC_MANAGEMENT_CATTLE_PIGS",
            f"{prefix_transformer_code}:LSMM:INC_MANAGEMENT_OTHER",
            f"{prefix_transformer_code}:LSMM:INC_MANAGEMENT_POULTRY"
        ]:
            magnitude = self.summarize_special_case_lsmm_pathways(*args_summary, )
            return magnitude
    
    
        # LVST - Enteric fermentation groups
        if base_transformer.code == f"{prefix_transformer_code}:LVST:DEC_ENTERIC_FERMENTATION":
            magnitude = self.summarize_special_case_lvst_enteric_fermentation(*args_summary, )
            return magnitude
    
        
        # PFLO - Healthier diets
        if base_transformer.code == f"{prefix_transformer_code}:PFLO:INC_HEALTHIER_DIETS":
            magnitude = self.summarize_from_magnitude(
                dict_params,
                kwargs,
                key_magnitude = "magnitude_red_meat", 
            )
            return magnitude
    
        
        # PFLO - Industrial CCS
        if base_transformer.code == f"{prefix_transformer_code}:PFLO:INC_IND_CCS":
            magnitude = self.summarize_special_case_pflo_ccs(*args_summary, )
            return magnitude
    
    
        # TRNS - Regional mode shift
        if base_transformer.code == f"{prefix_transformer_code}:TRNS:SHIFT_MODE_REGIONAL":
            magnitude = self.summarize_special_case_trns_mode_shift_regional(*args_summary, )
            return magnitude
            
    
        # WALI - Wastewater treatment pathways
        if base_transformer.code in [
            f"{prefix_transformer_code}:WALI:INC_TREATMENT_INDUSTRIAL",
            f"{prefix_transformer_code}:WALI:INC_TREATMENT_RURAL",
            f"{prefix_transformer_code}:WALI:INC_TREATMENT_URBAN"
        ]:
            magnitude = self.summarize_special_case_wali_pathways(*args_summary, )
            return magnitude


        # WASO - Anaerobic digester and composting for organic waste
        if base_transformer.code == f"{prefix_transformer_code}:WASO:INC_ANAEROBIC_AND_COMPOST":
            magnitude = self.summarize_special_case_waso_anaerobic_compost(*args_summary, )
            return magnitude


        # RAISE AN ERROR HERE IF A TRANSFORMATION IS NOT DEALT WITH
        #
        raise RuntimeError(f"Unable to summarize transformation '{transformation.code}': check cases.")
        
        return 0