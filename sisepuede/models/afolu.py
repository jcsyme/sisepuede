
import itertools
import logging
import numpy as np
import pandas as pd
import time
from matplotlib.pyplot import subplots
from typing import *


from sisepuede.core.attribute_table import AttributeTable
from sisepuede.core.model_attributes import *
from sisepuede.core.model_variable import is_model_variable
from sisepuede.models.energy_consumption import EnergyConsumption
from sisepuede.models.ippu import IPPU
from sisepuede.models.socioeconomic import Socioeconomic
from sisepuede.utilities._plotting import is_valid_figtuple
import sisepuede.utilities._npp_curves as npp
import sisepuede.utilities._optimization as suo
import sisepuede.utilities._toolbox as sf



_MODULE_UUID = "53E0A234-5674-47C8-B950-5A419EEAAF00"  
_NPP_INTEGRATION_WINDOWS = [20, 480, 1000]



##########################
###                    ###
###     AFOLU MODEL    ###
###                    ###
##########################



class CarbonLedger:
    """Track carbon flows using this class. Support class for AFOLU model.

    NOTES:

        * Assumes that protected areas do not experience removals
        * Scales

    Initialization Arguments
    ------------------------
    df_afolu_trajectories : int
        Number of time periods to initialize for
    model_attributes : ModelAttributes
        ModelAttributes for some initialization
    cat_frst_secondary : str
        Category in FRST to use for secondary forest
    cats_lndu_frst : List[str]
        List of forest land use categories
    cats_lndu_track : Union[List[str], None]
        Land use categories to track
    dict_lndu_to_frst: Dict[str, str]
        Dictionary mapping land use categories to associated forest categories
    modvar_frst_frac_c_converted_available: Union['ModelVariable', str]
        ModelVariable that stores the fraction of C converted that is available
        for use
    modvar_frst_frac_max_degradation : Union[ModelVariable, str]
        ModelVariable giving maximum degredation fraction (by land use category)
    vec_c_removals_demanded: Union[np.ndarray, None],
        Vector (n_time_periods) of exogenous demands for removals (from fuelwood
        and harvested wood products). If None, sets to 0.
    vec_init_by_cat_area : np.ndarray
        Vector (length n_cats) of initial areas, ordered by cats_lndu_track
    vec_init_by_cat_c_stock_per_area : np.ndarray
        Vector (length n_cats) of initial C stock per unit area, ordered by 
        cats_lndu_track
    vec_init_by_cat_seq_per_area : np.ndarray
        Vector (length n_cats) of initial C sequestration per unit area, 
        ordered by cats_lndu_track
    vec_sequestration_per_tp_new : np.ndarray
        Vector storing sequestration rates (mass C/area/time_period) for new
        growth. Can be derived from NPP curve.

    Keyword Arguments
    -----------------
    n_tps_no_withdrawals_new_growth : int
        Number of time periods without removals while new growth occurs. 
    **kwargs: 
        Passed to initialize arrays. Can be used to set initial values for
        arrays. Use

            {"initval_ARRAYVARNAME": x}

        to set the initial value of ARRAYVARNAME in x

    """


    def __init__(self,
        df_afolu_trajectories: pd.DataFrame,
        model_attributes: ModelAttributes,
        cat_frst_secondary: str,
        cats_lndu_frst: List[str],
        cats_lndu_track: Union[List[str], None],
        dict_lndu_to_frst: Dict[str, str],
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],
        vec_c_removals_demanded: Union[np.ndarray, None],
        vec_init_by_cat_area: np.ndarray,
        vec_init_by_cat_c_stock_per_area: np.ndarray,
        vec_init_by_cat_seq_per_area: np.ndarray,
        vec_sequestration_per_tp_new: np.ndarray,
        n_tps_no_withdrawals_new_growth: int = 20,
        **kwargs,
    ) -> None:
        

        self._initialize_attributes(
            model_attributes,
            cat_frst_secondary,
            cats_lndu_frst,
            cats_lndu_track,
            dict_lndu_to_frst,
        )

        self._initialize_arrays(
            df_afolu_trajectories,
            modvar_frst_frac_c_converted_available,
            modvar_frst_frac_max_degradation,
            vec_c_removals_demanded,
            vec_init_by_cat_area,
            vec_init_by_cat_c_stock_per_area,
            vec_init_by_cat_seq_per_area,
        )

        self._initialize_new_forest_properties(
            vec_sequestration_per_tp_new,
            n_tps_no_withdrawals_new_growth,
        )

        self._initialize_nf_availability_mask()

        return None
    


    def _initialize_attributes(self,
        model_attributes: ModelAttributes,
        cat_frst_secondary: str,
        cats_lndu_frst: List[str],
        cats_lndu_track: Union[List[str], None],    
        dict_lndu_to_frst: Dict[str, str],
    ) -> None:
        """Initialize key attributes used to manage land use classes
        """

        attr_lndu = model_attributes.get_attribute_table(
            model_attributes.subsec_name_lndu,
        )
        # some dictionaries
        dict_frst_to_lndu = sf.reverse_dict(dict_lndu_to_frst,)


        # set categories
        cats_lndu_frst = [x for x in attr_lndu.key_values if (x in cats_lndu_frst)]
        
        cats_lndu_track = (
            [x for x in attr_lndu.key_values if (x in cats_lndu_track) and (x in cats_lndu_frst)]
            if sf.islistlike(cats_lndu_track)
            else cats_lndu_frst
        )
        cats_frst_track = [dict_lndu_to_frst.get(x) for x in self.cats_lndu_track]

        # check for secondary forest
        if cat_frst_secondary not in cats_frst_track:
            raise RuntimeError(f"Error: secondary forest category '{cat_frst_secondary}' must be associated with a land use class to track.")
        
        ind_frst_secondary = cats_frst_track.index(cat_frst_secondary)
        cat_lndu_fsts = dict_frst_to_lndu.get(cat_frst_secondary)


        # pycategory for landuse
        pycat_lndu = model_attributes.get_subsector_attribute(
            model_attributes.subsec_name_lndu,
            "pycategory_primary_element"
        )

        

        ##  SET PROPERTIES

        self.attr_lndu = attr_lndu
        self.cat_frst_secondary = cat_frst_secondary
        self.cat_lndu_fsts = cat_lndu_fsts
        self.cats_frst_track = cats_frst_track
        self.cats_lndu_frst = cats_lndu_frst
        self.cats_lndu_track = cats_lndu_track
        self.dict_frst_to_lndu = dict_frst_to_lndu
        self.dict_lndu_to_frst = dict_lndu_to_frst
        self.ind_frst_secondary = ind_frst_secondary
        self.model_attributes = model_attributes
        self.pycat_lndu = pycat_lndu

        return None
    


    def _initialize_arrays(self,
        df_afolu_trajectories: pd.DataFrame,
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],
        vec_c_removals_demanded: Union[np.ndarray, None],
        vec_init_by_cat_area: np.ndarray,
        vec_init_by_cat_c_stock_per_area: np.ndarray,
        vec_init_by_cat_seq_per_area: np.ndarray,
        **kwargs,
    ) -> None:
        """Initialize the arrays needed for tracking Carbon stock. Stores the
            following arrays:

            * self.arr_area_conversion_away    
                Area of tracked land use being converted away (deforestation)
            * self.arr_area_protected
                Area of tracked land that is protected; removals are not allowed
            * self.arr_area_original_remaining
                Area of tracked land use remaining from initial time period
            * self.arr_areas_new_forest
                Areas of new forests that are planted. Each column gives the 
            * self.arr_c_available_total_at_beginning_of_period
                Array of total carbon available at the beginning of the time
                period by tracked land use. Allocation of 
                vec_c_available_total_at_beginning_of_period to tracked classes.
            * self.arr_c_seq_per_time_period
                Sequestration per time period
            * self.arr_c_seq_per_time_period_new_forests
                Sequestration per time period in new forests
            * self.arr_c_stock_avg_per_area
                Avergage C stock per area
            * self.arr_c_stock_in_new_forest
                Array of C stock stored in new forest growth. 
            * self.arr_c_stock_removed
                C Stock actually removed, by land use class
            * self.arr_mask_new_forest_available
                A binary mask signaling wether or not new forest stock is 
                available for removals
            * self.arr_min_frac_c_required
                Minimum fraction of C required (modeling parameter--makes 
                fraction greater than this available for removal)
            * self.arr_ratio_c_avail_to_c_avail_without_degradation
                Ratio of total C available in forests to C available without
                degradataion; used to reduce carbon stock factors on land and
            * self.vec_c_available_from_new_forests
                Vector of C available for harvest from new forests (restricted 
                by self.n_tps_no_withdrawals_new_growth)
            * self.vec_c_available_total_at_beginning_of_period
                Vector of total carbon available at the beginning of the time
                period
            * self.vec_c_removals_demanded
                Demand for C removals (defined at initialization)
            * self.vec_c_removals_satisfiable
                Demand for C removals that can be met
            * self.vec_frac_conversion_c_available_for_use
                fraction of C from land use converted to other types that is
                available for use to satisfy removals
        
        """

        n_cats = len(self.cats_lndu_track)
        n_tp = df_afolu_trajectories.shape[0]
        shape_by_cat = (n_tp, n_cats)
        shape_by_cat_with_new = (n_tp, n_cats + 1)
        
        # initialize arrays that are dynamically updated
        arr_area_conversion_away = np.zeros(shape_by_cat)
        arr_area_protected = np.zeros(shape_by_cat)
        arr_area_original_remaining = np.zeros(shape_by_cat)
        arr_areas_new_forest = np.zeros((n_tp, n_tp))
        arr_c_available_total_at_beginning_of_period = np.zeros(shape_by_cat)
        arr_c_seq_per_time_period = np.zeros(shape_by_cat)
        arr_c_seq_per_time_period_new_forests = np.zeros((n_tp, n_tp))
        arr_c_stock_avg_per_area = np.zeros(shape_by_cat)
        arr_c_stock_in_new_forest = np.zeros((n_tp, n_tp))
        arr_c_stock_removed = np.zeros(shape_by_cat_with_new)
        arr_c_stock_without_degradation = np.zeros(shape_by_cat_with_new)
        arr_mask_new_forest_available = np.zeros((n_tp, n_tp)).astype(int)
        arr_ratio_c_avail_to_c_avail_without_degradation = np.ones(shape_by_cat)
        vec_c_available_from_conversion = np.zeros(n_cats)
        vec_c_available_from_new_forests = np.zeros(n_cats)
        vec_c_available_total_at_beginning_of_period = np.zeros(n_cats)
        vec_c_removals_satisfiable = np.zeros(n_cats)
        

        ##  INITIALIZE ARRAYS THAT ARE PULLED FROM INPUT DATA (EXOGENOUS)

        # maximum degradation fraction for forests
        arr_min_frac_c_required = self.model_attributes.extract_model_variable(
            df_afolu_trajectories,
            modvar_frst_frac_max_degradation,
            return_type = "data_frame",
            var_bounds = (0, 1),
        )

        fields_keep = modvar_frst_frac_max_degradation.build_fields(
            category_restrictions = self.cats_frst_track,
        )
        arr_min_frac_c_required = 1 - arr_min_frac_c_required[fields_keep].to_numpy()
        

        # Fraction of C available from conversion
        vec_frac_conversion_c_available_for_use = self.model_attributes.extract_model_variable(
            df_afolu_trajectories,
            modvar_frst_frac_c_converted_available,
            override_vector_for_single_mv_q = False,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        
        # exogenous C removals
        vec_c_removals_demanded = (
            np.zeros(n_cats)
            if not isinstance(vec_c_removals_demanded, np.ndarray)
            else vec_c_removals_demanded
        )


        ##  SET SOME STARTING CONDITIONS

        arr_area_original_remaining[0, :] = vec_init_by_cat_area
        arr_c_stock_avg_per_area[0, :] = vec_init_by_cat_c_stock_per_area
        arr_c_seq_per_time_period[0, :] = vec_init_by_cat_seq_per_area
        

        ##  SET PROPERTIES

        self.arr_area_conversion_away = arr_area_conversion_away
        self.arr_area_protected = arr_area_protected
        self.arr_area_original_remaining = arr_area_original_remaining
        self.arr_areas_new_forest = arr_areas_new_forest
        self.arr_c_available_total_at_beginning_of_period = arr_c_available_total_at_beginning_of_period
        self.arr_c_seq_per_time_period = arr_c_seq_per_time_period
        self.arr_c_seq_per_time_period_new_forests = arr_c_seq_per_time_period_new_forests
        self.arr_c_stock_avg_per_area = arr_c_stock_avg_per_area
        self.arr_c_stock_in_new_forest = arr_c_stock_in_new_forest
        self.arr_c_stock_removed = arr_c_stock_removed
        self.arr_c_stock_without_degradation = arr_c_stock_without_degradation
        self.arr_mask_new_forest_available = arr_mask_new_forest_available
        self.arr_min_frac_c_required = arr_min_frac_c_required
        self.arr_ratio_c_avail_to_c_avail_without_degradation = arr_ratio_c_avail_to_c_avail_without_degradation
        self.n_tp = n_tp
        self.vec_c_available_from_conversion = vec_c_available_from_conversion
        self.vec_c_available_from_new_forests = vec_c_available_from_new_forests
        self.vec_c_available_total_at_beginning_of_period = vec_c_available_total_at_beginning_of_period
        self.vec_c_removals_demanded = vec_c_removals_demanded
        self.vec_c_removals_satisfiable = vec_c_removals_satisfiable
        self.vec_frac_conversion_c_available_for_use = vec_frac_conversion_c_available_for_use
        

        return None



    def _initialize_nf_availability_mask(self,
    ) -> None:
        """Initialize the availability mask, which denotes a one if new forest
            has existed long enough to allow for removals (reasonably)
        """

        arr = self.arr_mask_new_forest_available
        n0 = self.n_tps_no_withdrawals_new_growth + 1

        for i in range(n0, arr.shape[0]):
            j_max = i - self.n_tps_no_withdrawals_new_growth
            arr[i, 0:j_max] = 1

        self.arr_mask_new_forest_available = arr

        return None
    


    def _initialize_new_forest_properties(self,
        vec_sequestration_per_tp_new: np.ndarray,
        n_tps_no_withdrawals_new_growth: int = 20,
    ) -> None:
        """Initialize key attributes used to manage land use classes
        """ 

        ##  CHECK vec_sequestration_per_tp_new

        if not sf.islistlike(vec_sequestration_per_tp_new):
            raise TypeError(f"vec_sequestration_per_tp_new must be a list or numpy array")
        
        # convert to array
        vec_sequestration_per_tp_new = np.array(vec_sequestration_per_tp_new)
        if vec_sequestration_per_tp_new.shape[0] != self.n_tp:
            raise RuntimeError(f"Invalid shape {vec_sequestration_per_tp_new.shape[0]} for vec_sequestration_per_tp_new: must be of length {self.n_tp}")

        vec_sequestration_per_tp_new_cumulative = np.cumsum(vec_sequestration_per_tp_new)


        ##  CHECK NUMBER OF TIME PERIODS WITH NO WITHDRAWALS FOR NEWLY PLANTED FORESTS

        n_tps_no_withdrawals_new_growth = (
            20 
            if not sf.isnumer(n_tps_no_withdrawals_new_growth, integer = True)
            else max(n_tps_no_withdrawals_new_growth, 1)
        )


        ##  SET PROPERTIES

        self.n_tps_no_withdrawals_new_growth = n_tps_no_withdrawals_new_growth
        self.vec_sequestration_per_tp_new = vec_sequestration_per_tp_new
        self.vec_sequestration_per_tp_new_cumulative = vec_sequestration_per_tp_new_cumulative

        return None
    


    
    




    def _update(self,
        i: int,
        area_new_forest: float,
        vec_converted_away: np.ndarray,
        vec_protected: np.ndarray,
    ) -> None:
        """Update the ledger with land use lose

        Function Arguments
        ------------------
        i : int
            Time period to update (row index)
        area_new_forest : float
            Area of new forest entering (through planting/aforestation or 
            natural conversion)
        vec_converted_away : np.ndarray 
            Vector of total land use area converted away from tracked land use 
            types
        vec_protected : np.ndarray
            Vector of protected arrays
        """

        # update exogenous parameters
        self.arr_area_conversion_away[i, :] = vec_converted_away
        self.arr_area_protected[i, :] = vec_protected

        if i < self.n_tp - 1:
            self.arr_areas_new_forest[(i + 1):, i] = area_new_forest


        #
        # make any area adjustments here for new forest
        # if deforestation continues, it can encroach on new forest
        #

        area_to_shift_from_new = 0

        # 1. update area remaining from original and 
        if i < self.n_tp - 1:
            self.arr_area_original_remaining[i + 1, :] = self.arr_area_original_remaining[i, :] - vec_converted_away
            area_to_shift_from_new = max(
                -1*self.arr_area_original_remaining[i + 1, self.ind_frst_secondary],
                0
            )

            # set to 0 to ensure that the shift from new doesn't accumulate
            self.arr_area_original_remaining[i + 1, :] = max(self.arr_area_original_remaining[i + 1, :], 0)


        # 2. eliminate any new forest if implied by deforestation of secondary forest
        #      - iterate over columns in "new" to remove forest if necessary
        if area_to_shift_from_new > 0:
            j = 0
            area_shifted = 0
            while (area_shifted < area_to_shift_from_new):
                area_avail_cur = self.arr_areas_new_forest[i + 1, j]
                area_shift_cur = min(
                    area_to_shift_from_new - area_shifted,
                    area_avail_cur
                )

                self.arr_areas_new_forest[(i + 1):, j] = area_avail_cur - area_shift_cur
                area_shifted += area_shift_cur
                
                # move to next column and implement a safety check here
                j += 1
                if j > self.arr_areas_new_forest.shape[1] - 1:
                    break
                

        # 3. Calculate the counterfactual ""


        # calculate actual sequestration here--want to do it in a loop so that it can adapt to changes in area
        for j in range(i):
            self.arr_c_seq_per_time_period_new_forests[i, j] = (
                self.arr_areas_new_forest[i, j]
                *self.vec_sequestration_per_tp_new[i - 1]
            )

            if i >= self.n_tp - 1: continue

            self.arr_c_stock_in_new_forest[(i + 1), j] = self.arr_c_seq_per_time_period_new_forests[0:(i + 1), j].sum()

            # self.arr_c_seq_per_time_period_new_forests[(i + 1):, i] = area_new_forest*self.vec_sequestration_per_tp_new[0:(self.n_tp - i - 1)]
            # self.arr_c_stock_in_new_forest[(i + 1):, i] = area_new_forest*self.vec_sequestration_per_tp_new_cumulative
        
        
        return None
        


    

class AFOLU:
    """Use AFOLU to calculate emissions from Agriculture, Forestry, and Land Use 
        in SISEPUEDE. Includes emissions from the following subsectors:

        * Agriculture (AGRC)
        * Forestry (FRST)
        * Land Use (LNDU)
        * Livestock (LVST)
        * Livestock Manure Management (LSMM)
        * Soil Management (SOIL)

    For additional information, see the SISEPUEDE readthedocs at:

        https://sisepuede.readthedocs.io/en/latest/afolu.html


    Intialization Arguments
    -----------------------
    model_attributes : ModelAttributes
        ModelAttributes object used in SISEPUEDE to manage variables and 
        categories

    Optional Arguments
    ------------------
    logger : Union[logging.Logger, None]
        optional logger object to use for event logging
    min_diag_qadj : float
        Optional specification of minimum diagonal constraint to adhere to when
        adjusting transition matrices. If None, defaults to config (IN 
        PROCESSS). Note that this threshold will not modify unadjusted diagonal
        transitions that begin below this constraint.
    npp_curve : Union[str, npp.NPPCurve, None] 
        Optional specification of an NPP curve to use for dynamic forest 
        sequestration. In dynamic forest sequestration, forest sequestration 
        factors are used to fit NPP (net primary productivity) curves, which 
        vary over time. In general, young secondary forests sequester much more 
        than older secondary forests, with much of the annual growth 
        concentrated in the first 5-50 years. 
            * None or invalid entry: dynamic NPP is not used
            * "gamma": use the gamma function
            * "sem": use the SEM function 

    npp_include_primary_forest : bool
        Include primary forest sequestration factor in integration target for
        Net Primary Productivity (NPP) curve? If False, uses secondary forest 
        (non-young) sequestration factor as long term integration target. Only
        applies if `npp_curve` is specified as a valid curve.

    npp_integration_windows : arraylike
        Used for NPP curve fitting--to fit, integration is performed (or 
        estimated) in the windows specified here, and the average value by time
        period is compared to average annual sequestration factors that are 
        read from data. Only applies if `npp_curve` is specified as a valid 
        curve.
    """

    def __init__(self,
        attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None,
        min_diag_qadj: float = 0.98,
        npp_curve: Union[str, npp.NPPCurve, None] = None,
        npp_include_primary_forest: bool = False,
        npp_integration_windows: Union[list, tuple, np.ndarray] = _NPP_INTEGRATION_WINDOWS,
        **kwargs,
    ) -> None:

        self.logger = logger
        self.model_attributes = attributes

        self._initialize_subsector_names()
        self._initialize_input_output_components()
        self._initialize_other_properties()


        ##  SET MODEL FIELDS

        self._initialize_subsector_vars_agrc()
        self._initialize_subsector_vars_frst()
        self._initialize_subsector_vars_lndu()
        self._initialize_subsector_vars_lsmm()
        self._initialize_subsector_vars_lvst()
        self._initialize_subsector_vars_soil()

        self._initialize_models(
            min_diag_qadj = min_diag_qadj,
        )
        self._initialize_npp_properties(
            npp_curve = npp_curve,
            npp_include_primary_forest = npp_include_primary_forest,
            npp_integration_windows = npp_integration_windows,
        )

        self._initialize_integrated_variables()
        

        self._initialize_uuid()

        return None
    


    def __call__(self,
        *args,
        **kwargs
    ) -> pd.DataFrame:

        return self.project(*args, **kwargs)





    ##############################################
    #    SUPPORT AND INITIALIZATION FUNCTIONS    #
    ##############################################

    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See
            ?sf._optional_log for more information.

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



    def _assign_cats_lndu(self
    ) -> None:
        """
        Assign shortcut properties for land use categories, including crosswalks
            mapping forests to their land use category
        """

        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        pycat_frst = self.model_attributes.get_subsector_attribute(
            self.subsec_name_frst, 
            "pycategory_primary_element"
        )
        

        ##  SET LAND USE CATEGORY SHORTCUTS: CATEGORIES AND INDICES

        # non-forest
        dict_attr_field_prepend_to_attr_shortcut = {
            "crops": "crop",
            "flooded_lands": "flod",
            "grasslands": "grss",
            "other": "othr",
            "pastures": "pstr",
            "settlements": "stlm",
            "shrublands": "shrb",
            "wetlands": "wetl"
        }

        for k, v in dict_attr_field_prepend_to_attr_shortcut.items():
            cat = self.model_attributes.filter_keys_by_attribute(
                self.subsec_name_lndu, 
                {f"{k}_category": 1}
            )[0]

            ind = attr_lndu.get_key_value_index(cat)
            
            # set the category and index
            setattr(self, f"cat_lndu_{v}", cat)
            setattr(self, f"ind_lndu_{v}", ind)


        # forest
        dict_attr_field_prepend_to_attr_shortcut_frst = {
            self.cat_frst_mang: "fstm",
            self.cat_frst_prim: "fstp",
            self.cat_frst_scnd: "fsts",
        }

        for k, v in dict_attr_field_prepend_to_attr_shortcut_frst.items():
            cat = self.model_attributes.filter_keys_by_attribute(
                self.subsec_name_lndu,
                {pycat_frst: f"``{k}``"}
            )[0]

            ind = attr_lndu.get_key_value_index(cat)

            # set the category and index
            setattr(self, f"cat_lndu_{v}", cat)
            setattr(self, f"ind_lndu_{v}", ind)


        # list of categories to use to "max out" transition probabilities when scaling land use prevelance
        self.cats_lndu_max_out_transition_probs = self.model_attributes.filter_keys_by_attribute(
            self.model_attributes.subsec_name_lndu,
            {
                "reallocation_transition_probability_exhaustion_category": 1
            }
        )

        # land use classes that are assumed stable under reallocation
        self.cats_lndu_stable_under_reallocation = self.model_attributes.filter_keys_by_attribute(
            self.model_attributes.subsec_name_lndu,
            {
                "reallocation_stable_prevalence_category": 1
            }
        )

        return None



    def _assign_lndu_frst_dicts(self,
    ) -> None:
        """
        Generate dictionaries mapping land use categories to forest categories.
            Sets the following properties:

            * self.dict_cats_lndu_to_cats_frst
            * self.dict_cats_frst_to_cats_lndu
        """
        pycat_frst = self.model_attributes.get_subsector_attribute(
            self.model_attributes.subsec_name_frst,
            "pycategory_primary_element"
        )

        dict_lndu_to_frst = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_lndu,
            pycat_frst,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True,
        )
        dict_frst_to_lndu = sf.reverse_dict(dict_lndu_to_frst)
        

        ##  SET PROPERTIES
        
        self.dict_cats_frst_to_cats_lndu = dict_frst_to_lndu
        self.dict_cats_lndu_to_cats_frst = dict_lndu_to_frst

        return None



    def check_df_fields(self, 
        df_afolu_trajectories: pd.DataFrame, 
        check_fields: Union[List[str], None] = None
    ) -> None:
        check_fields = self.required_variables if (check_fields is None) else check_fields
        # check for required variables
        if not set(check_fields).issubset(df_afolu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_afolu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"AFOLU projection cannot proceed: The fields {set_missing} are missing.")

        return None



    def _initialize_input_output_components(self,
    ) -> None:
        """
        Set a range of input components, including required dimensions, 
            subsectors, input and output fields, and integration variables.
            Sets the following properties:

            * self.output_variables
            * self.required_dimensions
            * self.required_subsectors
            * self.required_base_subsectors
            * self.required_variables
            
        """
        # initialzie dynamic variables
        
        # required dimensions of analysis
        required_doa = [self.model_attributes.dim_time_period]

        # required subsectors
        subsectors = self.model_attributes.get_sector_subsectors("AFOLU")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]
        
        # input/output
        required_vars, output_vars = self.model_attributes.get_input_output_fields(subsectors)
        required_vars += required_doa

        # set output properties
        self.required_dimensions = required_doa
        self.required_subsectors = subsectors
        self.required_base_subsectors = subsectors_base
        self.required_variables = required_vars
        self.output_variables = output_vars

        return None



    def _initialize_integrated_variables(self
    ) -> None:
        """
        Initialize variables required for integration. Sets the following 
            properties:

            * self.dict_integration_variables_by_subsector
            * self.integration_variables
        """

        # initialize some variables not initialized elsewhere
        modvar_entc_efficiency_factor_technology = "Technology Efficiency of Fuel Use"
        modvar_entc_nemomod_min_share_production = "NemoMod MinShareProduction"
        modvar_entc_nemomod_residual_capacity = "NemoMod ResidualCapacity"

        dict_vars_required_for_integration = {
            # enfu variables that are required
            self.subsec_name_enfu: [
                self.model_enercons.modvar_enfu_energy_density_gravimetric
            ],

            # entc variables required for estimating biomass demand from ENTC
            self.subsec_name_entc: [
                modvar_entc_efficiency_factor_technology,
                modvar_entc_nemomod_min_share_production,
                modvar_entc_nemomod_residual_capacity
            ],

            # ippu variables required for estimating HWP
            self.subsec_name_ippu: [
                self.model_ippu.modvar_ippu_average_lifespan_housing,
                self.model_ippu.modvar_ippu_change_net_imports,
                self.model_ippu.modvar_ippu_demand_for_harvested_wood,
                self.model_ippu.modvar_ippu_elast_ind_prod_to_gdp,
                self.model_ippu.modvar_ippu_max_recycled_material_ratio,
                self.model_ippu.model_socioeconomic.modvar_grnl_num_hh,
                self.model_ippu.modvar_ippu_prod_qty_init,
                self.model_ippu.modvar_ippu_qty_recycled_used_in_production,
                self.model_ippu.modvar_ippu_qty_total_production,
                self.model_ippu.modvar_ippu_ratio_of_production_to_harvested_wood,
                self.model_ippu.modvar_waso_waste_total_recycled
            ],

            # SCOE variables required for projecting changes to wood energy demand
            self.subsec_name_scoe: [
                self.model_enercons.modvar_scoe_consumpinit_energy_per_hh_elec,
                self.model_enercons.modvar_scoe_consumpinit_energy_per_hh_heat,
                self.model_enercons.modvar_scoe_consumpinit_energy_per_mmmgdp_elec,
                self.model_enercons.modvar_scoe_consumpinit_energy_per_mmmgdp_heat,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_coal,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_diesel,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_electricity,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_gasoline,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_hgl,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_hydrogen,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_kerosene,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_natural_gas,
                self.model_enercons.modvar_scoe_efficiency_fact_heat_en_solid_biomass,
                self.model_enercons.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc,
                self.model_enercons.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc,
                self.model_enercons.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc,
                self.model_enercons.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc,
                self.model_enercons.modvar_scoe_frac_heat_en_coal,
                self.model_enercons.modvar_scoe_frac_heat_en_diesel,
                self.model_enercons.modvar_scoe_frac_heat_en_electricity,
                self.model_enercons.modvar_scoe_frac_heat_en_gasoline,
                self.model_enercons.modvar_scoe_frac_heat_en_hgl,
                self.model_enercons.modvar_scoe_frac_heat_en_hydrogen,
                self.model_enercons.modvar_scoe_frac_heat_en_kerosene,
                self.model_enercons.modvar_scoe_frac_heat_en_natural_gas,
                self.model_enercons.modvar_scoe_frac_heat_en_solid_biomass
            ]
        }

        # set complete output list of integration variables
        list_vars_required_for_integration = []
        for k in dict_vars_required_for_integration.keys():
            list_vars_required_for_integration += dict_vars_required_for_integration[k]

        
        ##  SET PROPERTIES

        self.modvar_entc_efficiency_factor_technology = modvar_entc_efficiency_factor_technology
        self.modvar_entc_nemomod_min_share_production = modvar_entc_nemomod_min_share_production
        self.modvar_entc_nemomod_residual_capacity = modvar_entc_nemomod_residual_capacity
        self.dict_integration_variables_by_subsector = dict_vars_required_for_integration
        self.integration_variables = list_vars_required_for_integration

        return None



    def _initialize_models(self,
        min_diag_qadj: float = 0.98,
        model_attributes: Union[ModelAttributes, None] = None,
    ) -> None:
        """
        Initialize SISEPUEDE model classes for fetching variables and accessing 
            methods. Initializes:

            * SISEPUEDE Models

                * self.model_enercons
                * self.model_ippu
                * self.model_socioeconomic

            * Associated categories of interest

                * self.cat_enfu_biomass
                * self.cat_ippu_paper
                * self.cat_ippu_wood

            * The Land Use transition corrector optimization model

                * self.q_adjuster


        Keyword Arguments
        -----------------
        - npp_curve: specification of dynamic NPP curve. Set to None
            to use default secondary forest factor. 
        - model_attributes: ModelAttributes object used to instantiate
            models. If None, defaults to self.model_attributes.
        """

        model_attributes = (
            self.model_attributes 
            if (model_attributes is None) 
            else model_attributes
        )

        # add other model classes--required for integration variables
        model_enercons = EnergyConsumption(model_attributes)
        model_ippu = IPPU(model_attributes)
        model_socioeconomic = Socioeconomic(model_attributes)

        # key categories
        cat_enfu_biomass = model_attributes.filter_keys_by_attribute(
            self.subsec_name_enfu, 
            {"biomass_demand_category": 1, }
        )[0]

        cat_ippu_paper = model_attributes.filter_keys_by_attribute(
            self.subsec_name_ippu, 
            {"virgin_paper_category": 1}
        )[0]

        cat_ippu_wood = model_attributes.filter_keys_by_attribute(
            self.subsec_name_ippu, 
            {"virgin_wood_category": 1}
        )[0]
        

        ##  SET CLASSES USED TO SOLVE SOME INTERNAL PROBLEMS

        # used to adjust transition matrices using Quadratic Programming
        q_adjuster = suo.QAdjuster(
            flag_ignore = float(self.flag_ignore_constraint),
            min_solveable_diagonal = min_diag_qadj,
        )


        ##  SET PROPERTIES

        self.cat_enfu_biomass = cat_enfu_biomass
        self.cat_ippu_paper = cat_ippu_paper
        self.cat_ippu_wood = cat_ippu_wood
        self.model_enercons = model_enercons
        self.model_ippu = model_ippu
        self.model_socioeconomic = model_socioeconomic
        self.q_adjuster = q_adjuster

        return None



    def _initialize_npp_properties(self,
        npp_curve: Union[str, npp.NPPCurve, None] = None,
        npp_include_primary_forest: bool = False,
        npp_integration_windows: Union[list, tuple, np.ndarray] = _NPP_INTEGRATION_WINDOWS,
    ) -> None:
        """
        Initialize properties related to the Net Primary Productivity estimator.
            Sets the following properties:
            
            * NPP Curve fitting for dynamic forest sequestration

                * self.curves_npp
                * self.npp_include_primary_forest
                    Include primary forest in the NPP integration? If True,
                    will integrate so that the tail averages the primary forest
                    sequestration value. In general, however, 
                * self.npp_integration_windows
                    Time period windows for integration matching. Each element 
                    is the time span (sequentially) of young secondary, 
                    secondary, and primary forests.
        """

        # used to fit NPP curves numerically
        curves_npp = npp.NPPCurves([])
        npp_curve = (
            npp_curve 
            if npp_curve in curves_npp.curves
            else None
        )

        # check if an error needs to be thrown
        error_q = not sf.islistlike(npp_integration_windows) 
        error_q &= not isinstance(npp_integration_windows, tuple)
        error_q &= npp_curve is not None
        
        if error_q:
            tp = str(type(npp_integration_windows))
            raise RuntimeError(f"Unable to initialize npp_curve {npp_curve}: invalid integration window of type {tp} specified. Must be array-like.")


        ##  SET PROPERTIES

        self.curves_npp = curves_npp
        self.npp_curve = npp_curve
        self.npp_include_primary_forest = npp_include_primary_forest
        self.npp_integration_windows = npp_integration_windows

        return None



    def _initialize_other_properties(self,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.factor_c_to_co2
            * self.factor_n2on_to_n2o
            * self.flag_ignore_constraint
            * self.is_sisepuede_model_afolu
            * self.n_time_periods
            * self.time_dependence_stock_change
            * self.time_periods
        """
        
        # from IPCC docs
        factor_c_to_co2 = float(11/3)
        factor_n2on_to_n2o = float(11/7)

        # see IPCC 2006/2019R GNGHGI V4 CH2 FOR D = 20
        time_dependence_stock_change = 20

        # time variables
        time_periods, n_time_periods = self.model_attributes.get_time_periods()


        ##  SET PROPERTIES

        self.factor_c_to_co2 = factor_c_to_co2
        self.factor_n2on_to_n2o = factor_n2on_to_n2o
        self.flag_ignore_constraint = -999
        self.is_sisepuede_model_afolu = True
        self.time_periods = time_periods
        self.n_time_periods = n_time_periods
        self.time_dependence_stock_change = time_dependence_stock_change

        return None



    def _initialize_subsector_names(self,
    ) -> None:
        """
        Initialize all subsector names (self.subsec_name_****)
        """

        attr_subsec = self.model_attributes.get_subsector_attribute_table()
        for subsec in attr_subsec.key_values:
            subsec_name = attr_subsec.get_attribute(subsec, "subsector")
            attr_name = f"subsec_name_{subsec}"
            setattr(self, attr_name, subsec_name, )
        """
        # some subector reference variables
        self.subsec_name_agrc = "Agriculture"
        self.subsec_name_econ = "Economy"
        self.subsec_name_enfu = "Energy Fuels"
        self.subsec_name_frst = "Forest"
        self.subsec_name_gnrl = "General"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_lndu = "Land Use"
        self.subsec_name_lsmm = "Livestock Manure Management"
        self.subsec_name_lvst = "Livestock"
        self.subsec_name_scoe = "Stationary Combustion and Other Energy"
        self.subsec_name_soil = "Soil Management"
        """
        return None


    def _initialize_subsector_vars_agrc(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            AGRC (Agriculture). Sets the following properties:

            * self.cat_agrc_****
            * self.ind_agrc_****
            * self.modvar_agrc_****
            * self.modvar_dict_agrc_****
            * self.modvar_list_agrc_****
        """
        # agricultural model variables
        self.modvar_agrc_adjusted_equivalent_exports = "Adjusted Agriculture Equivalent Exports"
        self.modvar_agrc_adjusted_equivalent_imports = "Adjusted Agriculture Equivalent Imports"
        self.modvar_agrc_area_prop_calc = "Cropland Area Proportion"
        self.modvar_agrc_area_prop_init = "Initial Cropland Area Proportion"
        self.modvar_agrc_area_crop = "Crop Area"
        self.modvar_agrc_changes_to_net_imports_lost = "Changes to Agriculture Net Imports Lost"
        self.modvar_agrc_combustion_factor = "AGRC Combustion Factor"
        self.modvar_agrc_demand_crops = "Crop Demand"
        self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Anaerobic Decomposition Emission Factor"
        self.modvar_agrc_ef_ch4_burning = ":math:\\text{CH}_4 Crop Biomass Burning Emission Factor"
        self.modvar_agrc_ef_co2_biomass = ":math:\\text{CO}_2 Crop Biomass Emission Factor"
        self.modvar_agrc_ef_n2o_burning = ":math:\\text{N}_2\\text{O} Crop Biomass Burning Emission Factor"
        self.modvar_agrc_ef_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Crop Fertilizer and Lime Emission Factor"
        self.modvar_agrc_elas_crop_demand_income = "Crop Demand Income Elasticity"
        self.modvar_agrc_emissions_ch4_biomass_burning = ":math:\\text{CH}_4 Emissions from Biomass Burning"
        self.modvar_agrc_emissions_ch4_rice = ":math:\\text{CH}_4 Emissions from Rice"
        self.modvar_agrc_emissions_co2_biomass = ":math:\\text{CO}_2 Emissions from Biomass Carbon Stock Changes"
        self.modvar_agrc_emissions_n2o_biomass_burning = ":math:\\text{N}_2\\text{O} Emissions from Biomass Burning"
        self.modvar_agrc_emissions_n2o_crop_residues = ":math:\\text{N}_2\\text{O} Emissions from Crop Residues"
        self.modvar_agrc_equivalent_exports = "Agriculture Equivalent Exports"
        self.modvar_agrc_frac_animal_feed = "Crop Fraction Animal Feed"
        self.modvar_agrc_frac_demand_imported = "Fraction of Agriculture Demand Imported"
        self.modvar_agrc_frac_dry = "Agriculture Fraction Dry"
        self.modvar_agrc_frac_dry_matter_in_crop = "Dry Matter Fraction of Harvested Crop"
        self.modvar_agrc_frac_no_till = "No Till Crop Fraction"
        self.modvar_agrc_frac_production_lost = "Fraction of Food Produced Lost Before Consumption"
        self.modvar_agrc_frac_production_loss_to_msw = "Fraction of Food Loss Sent to Municipal Solid Waste"
        self.modvar_agrc_frac_residues_removed = "Fraction of Residues Removed"
        self.modvar_agrc_frac_residues_burned = "Fraction of Residues Burned"
        self.modvar_agrc_frac_temperate = "Agriculture Fraction Temperate"
        self.modvar_agrc_frac_tropical = "Agriculture Fraction Tropical"
        self.modvar_agrc_frac_wet = "Agriculture Fraction Wet"
        self.modvar_agrc_n_content_of_above_ground_residues = "N Content of Above Ground Residues"
        self.modvar_agrc_n_content_of_below_ground_residues = "N Content of Below Ground Residues"
        self.modvar_agrc_ratio_above_ground_residue_to_harvested_yield = "Ratio of Above Ground Residue to Harvested Yield"
        self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass = "Ratio of Below Ground Biomass to Above Ground Biomass"
        self.modvar_agrc_regression_m_above_ground_residue = "Above Ground Residue Dry Matter Slope"
        self.modvar_agrc_regression_b_above_ground_residue = "Above Ground Residue Dry Matter Intercept"
        self.modvar_agrc_total_food_lost_in_ag = "Total Food Produced Lost Before Consumption"
        self.modvar_agrc_total_food_lost_in_ag_to_msw = "Total Food Loss Sent to Municipal Solid Waste"
        self.modvar_agrc_yf = "Crop Yield Factor"
        self.modvar_agrc_yield = "Crop Yield"
        
        # additional lists
        self.modvar_list_agrc_frac_drywet = [
            self.modvar_agrc_frac_dry,
            self.modvar_agrc_frac_wet
        ]
        self.modvar_list_agrc_frac_temptrop = [
            self.modvar_agrc_frac_temperate,
            self.modvar_agrc_frac_tropical
        ]
        self.modvar_list_agrc_frac_residues_removed_burned = [
            self.modvar_agrc_frac_residues_burned,
            self.modvar_agrc_frac_residues_removed
        ]

        # some key categories
        self.cat_agrc_rice = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_agrc, 
            {"rice_category": 1}
        )[0]

        return None



    def _initialize_subsector_vars_frst(self,
    ) -> None:
        """Initialize model variables, categories, and indices associated with
            FRST (Forest). Sets the following properties:

            * self.cat_frst_****
            * self.ind_frst_****
            * self.modvar_frst_****
            * self.modvar_dict_frst_****
            * self.modvar_list_frst_****
        """

        # forest model variables
        self.modvar_frst_average_fraction_burned_annually = "Average Fraction of Forest Burned Annually"
        self.modvar_frst_biomass_consumed_fire_temperate = "Fire Biomass Consumption for Temperate Forests"
        self.modvar_frst_biomass_consumed_fire_tropical = "Fire Biomass Consumption for Tropical Forests"
        self.modvar_frst_c_stock = "Above Ground C Stock in Forests"
        self.modvar_frst_ef_co2_fires = ":math:\\text{CO}_2 Forest Fire Emission Factor"
        self.modvar_frst_ef_ch4 = ":math:\\text{CH}_4 Forest Methane Emissions"
        self.modvar_frst_emissions_co2_fires = ":math:\\text{CO}_2 Emissions from Forest Fires"
        self.modvar_frst_emissions_co2_hwp = ":math:\\text{CO}_2 Emissions from Harvested Wood Products"
        self.modvar_frst_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Forests"
        self.modvar_frst_emissions_co2_sequestration = ":math:\\text{CO}_2 Emissions from Forest Biomass Sequestration"
        self.modvar_frst_frac_c_converted_available = "Fraction of Forest Converted C Available for Use"
        self.modvar_frst_frac_c_per_dm = "Carbon Fraction Dry Matter"
        self.modvar_frst_frac_c_per_hwp = "Carbon Fraction Harvested Wood Products"
        self.modvar_frst_frac_max_degradation = "Maximum Degradation Fraction"
        self.modvar_frst_frac_temperate_nutrient_poor = "Forest Fraction Temperate Nutrient Poor"
        self.modvar_frst_frac_temperate_nutrient_rich = "Forest Fraction Temperate Nutrient Rich"
        self.modvar_frst_frac_tropical = "Forest Fraction Tropical"
        self.modvar_frst_hwp_half_life_paper = "HWP Half Life Paper"
        self.modvar_frst_hwp_half_life_wood = "HWP Half Life Wood"
        self.modvar_frst_sq_co2 = "Forest Sequestration Emission Factor"
        self.modvar_frst_sq_co2_young_secondary = "Young Secondary Forest Sequestration Emission Factor"
        
        #additional lists
        self.modvar_list_frst_frac_temptrop = [
            self.modvar_frst_frac_temperate_nutrient_poor,
            self.modvar_frst_frac_temperate_nutrient_rich,
            self.modvar_frst_frac_tropical
        ]

        # assign some key categories
        self.cat_frst_mang = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_frst, 
            {"mangroves_forest_category": 1}
        )[0]
        self.cat_frst_prim = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_frst, 
            {"primary_forest_category": 1}
        )[0]
        self.cat_frst_scnd = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_frst, 
            {"secondary_forest_category": 1}
        )[0]

        # assign indicies
        attr_frst = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_frst,
        )

        self.ind_frst_mang = attr_frst.get_key_value_index(self.cat_frst_mang, )
        self.ind_frst_prim = attr_frst.get_key_value_index(self.cat_frst_prim, )
        self.ind_frst_scnd = attr_frst.get_key_value_index(self.cat_frst_scnd, )

        return None



    def _initialize_subsector_vars_lndu(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LNDU (Land Use). Sets the following properties:

            * self.cat_lndu_****
            * self.ind_lndu_****
            * self.modvar_lndu_****
            * self.modvar_dict_lndu_****
            * self.modvar_list_lndu_****
        """
        # land use model variables
        self.modvar_lndu_area_by_cat = "Land Use Area"
        self.modvar_lndu_area_improved = "Area of Improved Land"
        self.modvar_lndu_area_converted = "Area of Land Use Converted"
        self.modvar_lndu_area_converted_from_type = "Area of Land Use Converted Away from Type"
        self.modvar_lndu_area_converted_to_type = "Area of Land Use Converted to Type"
        self.modvar_lndu_constraint_area_max = "Maximum Area"
        self.modvar_lndu_constraint_area_min = "Minimum Area"
        self.modvar_lndu_c_stock_bgb_ratio = "Below Ground Biomass Carbon Stock Ratio"
        self.modvar_lndu_c_stock_factor_abg = "Initial Above Ground Biomass Carbon Stock Factor"
        # self.modvar_lndu_ef_co2_conv = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
        self.modvar_lndu_emissions_conv = ":math:\\text{CO}_2 Emissions from Land Use Conversion"
        self.modvar_lndu_emissions_conv_away = ":math:\\text{CO}_2 Emissions from Conversion Away from Land Use Type"
        self.modvar_lndu_emissions_ch4_from_wetlands = ":math:\\text{CH}_4 Emissions from Wetlands"
        self.modvar_lndu_emissions_co2_sequestration = ":math:\\text{CO}_2 Emissions from Land Use Biomass Sequestration" #NEW
        self.modvar_lndu_emissions_co2_drained_organic_soils = ":math:\\text{CO}_2 Emissions from Drained Organic Soils"
        self.modvar_lndu_factor_soil_carbon = "Soil Carbon Land Use Factor"
        self.modvar_lndu_factor_soil_management_infinum = "Unimproved Soil Carbon Land Management Factor"
        self.modvar_lndu_factor_soil_management_supremum = "Maximum Soil Carbon Land Management Factor"
        self.modvar_lndu_factor_soil_inputs_supremum_no_manure = "Maximum Soil Carbon Land Input Factor Without Manure"
        self.modvar_lndu_frac_dry = "Land Use Fraction Dry"
        self.modvar_lndu_frac_fertilized = "Land Use Fraction Fertilized"
        self.modvar_lndu_frac_increasing_net_exports_met = "Fraction of Increasing Net Exports Met"
        self.modvar_lndu_frac_increasing_net_imports_met = "Fraction of Increasing Net Imports Met"
        self.modvar_lndu_frac_mineral_soils = "Fraction of Soils Mineral"
        self.modvar_lndu_frac_pastures_improved = "Fraction of Pastures Improved"
        self.modvar_lndu_frac_temperate = "Land Use Fraction Temperate"
        self.modvar_lndu_frac_tropical = "Land Use Fraction Tropical"
        self.modvar_lndu_frac_wet = "Land Use Fraction Wet"
        self.modvar_lndu_initial_frac = "Initial Land Use Area Proportion"
        self.modvar_lndu_initial_frac_area_dos = "Initial Fraction of Area with Drained Organic Soils"
        self.modvar_lndu_ef_ch4_boc = "Land Use BOC :math:\\text{CH}_4 Emission Factor"
        self.modvar_lndu_prob_transition = "Unadjusted Land Use Transition Probability"
        self.modvar_lndu_rate_utilization = "Utilization Rate" #NEW
        self.modvar_lndu_reallocation_factor = "Land Use Yield Reallocation Factor"
        self.modvar_lndu_sf_co2 = "Land Use Biomass Sequestration Factor" #NEW
        self.modvar_lndu_vdes = "Vegetarian Diet Exchange Scalar"
        self.modvar_lndu_yf_pasture_sup = "Maximum Pasture Dry Matter Yield Factor"


        # additional lists
        self.modvar_list_lndu_frac_drywet = [
            self.modvar_lndu_frac_dry,
            self.modvar_lndu_frac_wet
        ]
        self.modvar_list_lndu_frac_temptrop = [
            self.modvar_lndu_frac_temperate,
            self.modvar_lndu_frac_tropical
        ]
        
        # some key categories
        self._assign_cats_lndu()
        self._assign_lndu_frst_dicts()

        return None



    def _initialize_subsector_vars_lsmm(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LSMM (Livestock Manure Management). Sets the following 
            properties:

            * self.cat_lsmm_****
            * self.ind_lsmm_****
            * self.modvar_lsmm_****
            * self.modvar_dict_lsmm_****
        """
        
        # manure management variables
        self.modvar_lsmm_dung_incinerated = "Dung Incinerated"
        self.modvar_lsmm_ef_direct_n2o = ":math:\\text{N}_2\\text{O} Manure Management Emission Factor"
        self.modvar_lsmm_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Manure Management"
        self.modvar_lsmm_emissions_direct_n2o = ":math:\\text{N}_2\\text{O} Direct Emissions from Manure Management"
        self.modvar_lsmm_emissions_indirect_n2o = ":math:\\text{N}_2\\text{O} Indirect Emissions from Manure Management"
        self.modvar_lsmm_frac_loss_leaching = "Fraction of Nitrogen Lost to Leaching"
        self.modvar_lsmm_frac_loss_volatilisation = "Fraction of Nitrogen Lost to Volatilisation"
        self.modvar_lsmm_frac_n_available_used = "Fraction of Nitrogen Used in Fertilizer"
        self.modvar_lsmm_mcf_by_pathway = "Manure Management Methane Correction Factor"
        self.modvar_lsmm_n_from_bedding = "Nitrogen from Bedding per Animal"
        self.modvar_lsmm_n_from_codigestates = "Nitrogen from Co-Digestates Factor"
        self.modvar_lsmm_n_to_pastures = "Total Nitrogen to Pastures"
        self.modvar_lsmm_n_to_fertilizer = "Nitrogen Available for Fertilizer"
        self.modvar_lsmm_n_to_fertilizer_agg_dung = "Total Nitrogen Available for Fertilizer from Dung"
        self.modvar_lsmm_n_to_fertilizer_agg_urine = "Total Nitrogen Available for Fertilizer from Urine"
        self.modvar_lsmm_n_to_other_use = "Total Nitrogen Available for Construction/Feed/Other"
        self.modvar_lsmm_ratio_n2_to_n2o = "Ratio of :math:\\text{N}_2 to :math:\\text{N}_2\\text{O}"
        self.modvar_lsmm_recovered_biogas = "LSMM Biogas Recovered from Anaerobic Digesters"
        self.modvar_lsmm_rf_biogas = "Biogas Recovery Factor at LSMM Anaerobic Facilities"

        # some categories
        self.cat_lsmm_incineration = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_lsmm, 
            {"incineration_category": 1}
        )[0]
        self.cat_lsmm_pasture = self.model_attributes.filter_keys_by_attribute(
            self.subsec_name_lsmm, 
            {"pasture_category": 1}
        )[0]

        return None



    def _initialize_subsector_vars_lvst(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            LVST (Livestock). Sets the following properties:

            * self.cat_lvst_****
            * self.ind_lvst_****
            * self.modvar_lvst_****
            * self.modvar_dict_lvst_****
        """
        # livestock model variables
        self.modvar_lvst_adjusted_equivalent_exports = "Adjusted Livestock Equivalent Exports"
        self.modvar_lvst_adjusted_equivalent_imports = "Adjusted Livestock Equivalent Imports"
        self.modvar_lvst_animal_weight = "Animal Weight"
        self.modvar_lvst_carrying_capacity_scalar = "Carrying Capacity Scalar"
        self.modvar_lvst_changes_to_net_imports_lost = "Changes to Livestock Net Imports Lost"
        self.modvar_lvst_demand_livestock = "Livestock Demand"
        self.modvar_lvst_dry_matter_consumption = "Daily Dry Matter Consumption"
        self.modvar_lvst_ef_ch4_ef = ":math:\\text{CH}_4 Enteric Fermentation Emission Factor"
        self.modvar_lvst_elas_lvst_demand = "Elasticity of Livestock Demand to GDP per Capita"
        self.modvar_lvst_emissions_ch4_ef = ":math:\\text{CH}_4 Emissions from Livestock Enteric Fermentation"
        self.modvar_lvst_equivalent_exports = "Livestock Equivalent Exports"
        self.modvar_lvst_frac_demand_imported = "Fraction of Livestock Demand Imported"
        self.modvar_lvst_frac_exc_n_in_dung = "Fraction Nitrogen Excretion in Dung"
        self.modvar_lvst_frac_mm_anaerobic_digester = "Livestock Manure Management Fraction Anaerobic Digester"
        self.modvar_lvst_frac_mm_anaerobic_lagoon = "Livestock Manure Management Fraction Anaerobic Lagoon"
        self.modvar_lvst_frac_mm_composting = "Livestock Manure Management Fraction Composting"
        self.modvar_lvst_frac_mm_daily_spread = "Livestock Manure Management Fraction Daily Spread"
        self.modvar_lvst_frac_mm_deep_bedding = "Livestock Manure Management Fraction Deep Bedding"
        self.modvar_lvst_frac_mm_dry_lot = "Livestock Manure Management Fraction Dry Lot"
        self.modvar_lvst_frac_mm_incineration = "Livestock Manure Management Fraction Incineration"
        self.modvar_lvst_frac_mm_liquid_slurry = "Livestock Manure Management Fraction Liquid Slurry"
        self.modvar_lvst_frac_mm_poultry_manure = "Livestock Manure Management Fraction Poultry Manure"
        self.modvar_lvst_frac_mm_ppr = "Livestock Manure Management Fraction Paddock Pasture Range"
        self.modvar_lvst_frac_mm_solid_storage = "Livestock Manure Management Fraction Solid Storage"
        self.modvar_lvst_genfactor_nitrogen = "Daily Nitrogen Generation Factor"
        self.modvar_lvst_genfactor_volatile_solids = "Daily Volatile Solid Generation Factor"
        self.modvar_lvst_b0_manure_ch4 = "Maximum Manure :math:\\text{CH}_4 Generation Capacity"
        self.modvar_lvst_pop = "Livestock Head Count"
        self.modvar_lvst_pop_init = "Initial Livestock Head Count"
        self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"

        # dictionaries and list variables
        tup = self.get_lvst_dict_lsmm_categories_to_lvst_fraction_variables()
        modvar_list_lvst_mm_fractions = [v.get("mm_fraction") for v in tup[0].values()]
        modvar_list_lvst_mm_fractions = [x for x in modvar_list_lvst_mm_fractions if (x is not None)]

        self.dict_lsmm_categories_to_lvst_fraction_variables = tup[0]
        self.dict_lsmm_categories_to_unassigned_variables = tup[1]
        self.modvar_list_lvst_mm_fractions = modvar_list_lvst_mm_fractions
        """
        self.modvar_list_lvst_mm_fractions = [
            self.modvar_lvst_frac_mm_anaerobic_digester,
            self.modvar_lvst_frac_mm_anaerobic_lagoon,
            self.modvar_lvst_frac_mm_composting,
            self.modvar_lvst_frac_mm_daily_spread,
            self.modvar_lvst_frac_mm_deep_bedding,
            self.modvar_lvst_frac_mm_dry_lot,
            self.modvar_lvst_frac_mm_incineration,
            self.modvar_lvst_frac_mm_liquid_slurry,
            self.modvar_lvst_frac_mm_poultry_manure,
            self.modvar_lvst_frac_mm_ppr,
            self.modvar_lvst_frac_mm_solid_storage
        ]
        """;
        return None



    def _initialize_subsector_vars_soil(self,
    ) -> None:
        """
        Initialize model variables, categories, and indices associated with
            SOIL (Soil Management). Sets the following properties:

            * self.cat_soil_****
            * self.ind_soil_****
            * self.modvar_soil_****
            * self.modvar_dict_soil_****
        """
        # soil management variables
        self.modvar_soil_demscalar_fertilizer = "Fertilizer N Demand Scalar"
        self.modvar_soil_demscalar_liming = "Liming Demand Scalar"
        self.modvar_soil_ef1_n_managed_soils_rice = ":math:\\text{EF}_{1FR} - \\text{N}_2\\text{O} Rice Fields"
        self.modvar_soil_ef1_n_managed_soils_org_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Organic Amendments and Fertilizer"
        self.modvar_soil_ef1_n_managed_soils_syn_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Synthetic Fertilizer"
        self.modvar_soil_ef2_n_organic_soils = ":math:\\text{EF}_2 - \\text{N}_2\\text{O} Emissions from Drained and Managed Organic Soils"
        self.modvar_soil_ef3_n_prp = "EF3 N Pasture Range and Paddock"
        self.modvar_soil_ef4_n_volatilisation = "EF4 N Volatilisation and Re-Deposition Emission Factor"
        self.modvar_soil_ef5_n_leaching = "EF5 N Leaching and Runoff Emission Factor"
        self.modvar_soil_ef_c_liming_dolomite = "C Liming Emission Factor Dolomite"
        self.modvar_soil_ef_c_liming_limestone = "C Liming Emission Factor Limestone"
        self.modvar_soil_ef_c_organic_soils_croplands = "C Annual Croplands Drained Organic Soils Emission Factor"
        self.modvar_soil_ef_c_organic_soils_managed_grasslands = "C Annual Managed Grasslands Drained Organic Soils Emission Factor"
        self.modvar_soil_ef_c_urea = "C Urea Emission Factor"
        self.modvar_soil_emissions_co2_lime = ":math:\\text{CO}_2 Emissions from Lime"
        self.modvar_soil_emissions_co2_soil_carbon_mineral = ":math:\\text{CO}_2 Emissions from Soil Carbon in Mineral Soils"
        self.modvar_soil_emissions_co2_urea = ":math:\\text{CO}_2 Emissions from Urea"
        self.modvar_soil_emissions_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Emissions from Fertilizer Use"
        self.modvar_soil_emissions_n2o_mineral_soils = ":math:\\text{N}_2\\text{O} Emissions from Mineral Soils"
        self.modvar_soil_emissions_n2o_organic_soils = ":math:\\text{N}_2\\text{O} Emissions from Organic Soils"
        self.modvar_soil_emissions_n2o_ppr = ":math:\\text{N}_2\\text{O} Emissions from Paddock Pasture and Range"
        self.modvar_soil_frac_n_lost_leaching = "Leaching Fraction of N Lost"
        self.modvar_soil_frac_n_lost_volatilisation_on = "Volatilisation Fraction from Organic Amendments and Fertilizers"
        self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea = "Volatilisation Fraction from Non-Urea Synthetic Fertilizers"
        self.modvar_soil_frac_n_lost_volatilisation_sn_urea = "Volatilisation Fraction from Urea Synthetic Fertilizers"
        self.modvar_soil_frac_synethic_fertilizer_urea = "Fraction Synthetic Fertilizer Use Urea"
        self.modvar_soil_fertuse_final_organic = "Organic Fertilizer N Use"
        self.modvar_soil_fertuse_final_synthetic = "Synthetic Fertilizer N Use"
        self.modvar_soil_fertuse_final_total = "Total Fertilizer N Use"
        self.modvar_soil_fertuse_init_synthetic = "Initial Synthetic Fertilizer Use"
        self.modvar_soil_limeuse_total = "Total Lime Use"
        self.modvar_soil_organic_c_stocks = "Soil Organic C Stocks"
        self.modvar_soil_qtyinit_liming_dolomite = "Initial Liming Dolomite Applied to Soils"
        self.modvar_soil_qtyinit_liming_limestone = "Initial Liming Limestone Applied to Soils"
        self.modvar_soil_ratio_c_to_n_soil_organic_matter = "C to N Ratio of Soil Organic Matter"
        self.modvar_soil_ureause_total = "Total Urea Use"


        (
            self.dict_lndu_categories_to_soil_variables, 
            self.dict_lndu_categories_to_unassigned_soil_variables,
        ) = self.get_soil_dict_lndu_categories_to_soil_c_dos_variables()


        return None



    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID
        """

        self._uuid = _MODULE_UUID

        return None




    ###########################
    #    SUPPORT FUNCTIONS    #
    ###########################

    def adjust_transition_matrix(self,
        mat: np.ndarray,
        dict_tuples_scale: dict,
        ignore_diag_on_col_scale: bool = False,
        mat_bounds: tuple = (0, 1),
        response_columns = None
    ) -> np.ndarray:
        """Rescale elements of a row-stochastic transition matrix Q (n x n) to 
            account for scalars applied to columns or entries defined in 
            dict_tuples_scale. The columns that are adjusted in response to 
            these exogenous scalars are said to be to subject to automated 
            rescaling.

        NOTE: THIS FUNCTION IS DEPRECATED. 


        dict_tuples_scale is the mechanism for passing scalars to apply to a 
            transition matrix It accepts two types of tuples for keys

            - to scale an entire column, enter a single tuple (j, )
            - to scale a point, use (j, k)
            
        For example,

            dict_tuples_scale = {(i, ): scalar_1, (j, k): scalar_2}

        will scale:
            * all transition probabilities in column i using scalar_1
            * the transition probabilty at (j, k) using scalar_2
            * all other probabilities in a given row uniformly to ensure 
                summation to 1

        Function Arguments
        ------------------
        - mat: row-stochastic transition matrix to apply scalars to
        - dict_tuples_scale: dictionary of tuples defining columnar or 
            point-based probabilities to scale

        Keyword Arguments
        -----------------
        - ignore_diag_on_col_scale: if True, diagonals on the transition matrix 
            are not scaled in response to other changing probabilties.
        - mat_bounds: bounds for elements in the matrix (weak inequalities).
        - response_columns: the columns in the matrix that are subject to 
            automated rescaling in response to to exogenous scalars. If None, 
            then, for each row, columns that are not affected by exogenous 
            scalars are subject to automated rescaling.



        Example and Notes
        -----------------
        * The final transition matrix may not reflect the scalars that are 
            passed. For example, considr the matrix

            array([
                [0.5, 0, 0.5],
                [0.2, 0.7, 0.1],
                [0.0, 0.1, 0.9]
            ])

        if dict_tuples_scale = {(0, ): 2, (0, 2): 1.4}, then, before rescaling 
            probabilities that are not specified in dict_tuples_scale, the 
            matrix becomes

            array([
                [1.0, 0, 0.7],
                [0.4, 0.7, 0.1],
                [0.0, 0.1, 0.9]
            ])

        Since all of the non-zero elements in the first row are subject to 
            scaling, the normalization to sum to 1 reduces the effect of the 
            specified scalar. The final matrix becomes (approximately)

            array([
                [0.5883, 0, 0.4117],
                [0.4, 0.525, 0.075],
                [0.0, 0.1, 0.9]
            ])

        * NOTE: Applying large scalars will lead to dominance, and eventually 
            singular values.

        """

        # assume that the matrix is square - get the scalar, then get the mask to use adjust transition probabilities not specified as a scalar
        mat_scale = np.ones(mat.shape)
        mat_pos_scale = np.zeros(mat.shape)
        mat_mask = np.ones(mat.shape)
        
        # assign columns that will be adjusted in response to changes - default to all that aren't scaled
        if (response_columns == None):
            mat_mask_response_nodes = np.ones(mat.shape)
        else:
            mat_mask_response_nodes = np.zeros(mat.shape)
            for col in [x for x in response_columns if x < mat.shape[0]]:
                mat_mask_response_nodes[:, col] = 1

        m = mat_scale.shape[0]

        ##  PERFORM SCALING

        # adjust columns first
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 1]:
            # overwrite the column
            mat_scale[:, ind[0]] = np.ones(m)*dict_tuples_scale[ind]
            mat_pos_scale[:, ind[0]] = np.ones(m)
            mat_mask[:, ind[0]] = np.zeros(m)

        # it may be of interest to ignore the diagonals when scaling columns
        if ignore_diag_on_col_scale:
            mat_diag = np.diag(tuple(np.ones(m)))
            # reset ones on the diagonal
            mat_scale = (np.ones(mat.shape) - mat_diag)*mat_scale + mat_diag
            mat_pos_scale = sf.vec_bounds(mat_pos_scale - mat_diag, (0, 1))
            mat_mask =  sf.vec_bounds(mat_mask + mat_diag, (0, 1))
        
        # next, adjust points - operate on the transpose of the matrix
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 2]:
            i, j = ind
            mat_scale[i, j] = dict_tuples_scale.get(ind)
            mat_pos_scale[i, j] = 1
            mat_mask[i, j] = 0


        """
        Get the total that needs to be removed from masked elements (those that 
            are not scaled)

        NOTE: bound scalars at the low end by 0 (if 
            mask_shift_total_i > sums_row_mask_i, then the scalar is negative.
            This occurs if the row total of the adjusted values exceeds 1.Set 
            `mask_scalar` using a minimum value of 0 and implement row 
            normalization—if there's no way to rebalance response columns, 
            everything gets renormalized. We correct for this below by 
            implementing row normalization to mat_out.
        """
        # get new mat and restrict values to 0, 1
        mat_new_scaled = sf.vec_bounds(mat*mat_scale, mat_bounds)
        sums_row = sum(mat_new_scaled.transpose())
        sums_row_mask = sum((mat_mask_response_nodes*mat_mask*mat).transpose())
        
        # get shift and positive scalar to apply to valid masked elements
        mask_shift_total = sums_row - 1
        mask_scalar = np.nan_to_num(
            sf.vec_bounds(
                (sums_row_mask - mask_shift_total)/sums_row_mask, 
                (0, np.inf)
            ), 
            nan = 1.0, 
            posinf = 1.0,
        )

        # get the masked nodes, multiply by the response scalar (at applicable columns, denoted by mat_mask_response_nodes), then add to
        mat_out = ((mat_mask_response_nodes*mat_mask*mat).transpose() * mask_scalar).transpose()
        mat_out += sf.vec_bounds(mat_mask*(1 - mat_mask_response_nodes), (0, 1))*mat
        mat_out += mat_pos_scale*mat_new_scaled
        mat_out = (mat_out.transpose()/sum(mat_out.transpose())).transpose()

        mat_out = sf.vec_bounds(mat_out, mat_bounds)

        return mat_out



    def back_project_hwp_c_k(self,
        n_tps_lookback: int,
        vec_frst_c_paper: np.ndarray,
        vec_frst_c_wood: np.ndarray,
        vec_frst_k_hwp_paper: np.ndarray,
        vec_frst_k_hwp_wood: np.ndarray,
        n_tps_mean: int = 5,
    ) -> Union[Tuple[np.ndarray], None]:
        """Back project carbon stored in wood and paper products (c) and the 
            exponential decay parameter k.

        Returns a tuple with the following elements:

            (
                vec_frst_c_paper,
                vec_frst_c_wood,
                vec_frst_k_hwp_paper,
                vec_frst_k_hwp_wood,
            )
            

        Function Arguments
        ------------------
        n_tps_lookback : int
            Number of time periods to look back
        vec_frst_c_paper : np.ndarray
            Vector of c stored in paper products
        vec_frst_c_wood : np.ndarray
            Vector of c stored in wood products
        vec_frst_k_hwp_paper : np.ndarray
            Vector of decay variable k for paper
        vec_frst_k_hwp_wood : np.ndarray
            Vector of decay variable k for wood

        Keyword Arguments
        -----------------
        - n_tps_mean : int
            Number of years to use to generate the mean rate of growth in HWP. 
            Will not, in practice, exceed the number of observed years
        """
        # return None if the lookback is invalid
        if n_tps_lookback <= 0:
            return None

        # initialize time periods
        n_tps_mean = max(n_tps_mean, 1)
        n_tps_mean = min(n_tps_mean, len(vec_frst_c_paper) - 1) # -1 because we apply the mean to rates


        ##  CARBON CONTENT BACK PROJECTIONS

        vec_exp = (np.arange(n_tps_lookback) - n_tps_lookback)

        # paper products
        r_paper = np.mean(vec_frst_c_paper[1:(1 + n_tps_mean)]/vec_frst_c_paper[0:n_tps_mean])
        vec_frst_c_paper = np.concatenate([
            vec_frst_c_paper[0]*(r_paper**vec_exp), 
            vec_frst_c_paper
        ])
         
        # wood products
        r_wood = np.mean(vec_frst_c_wood[1:(1 + n_tps_mean)]/vec_frst_c_wood[0:n_tps_mean])
        vec_frst_c_wood = np.concatenate([
            vec_frst_c_wood[0]*(r_wood**vec_exp), 
            vec_frst_c_wood
        ])


        ##  DECAY FACTOR (ASSUME CONSTANT IN BACK PROJECTION)

        vec_frst_k_hwp_paper = np.concatenate([
            np.repeat(vec_frst_k_hwp_paper[0], n_tps_lookback),
            vec_frst_k_hwp_paper
        ])
        vec_frst_k_hwp_wood = np.concatenate([
            np.repeat(vec_frst_k_hwp_wood[0], n_tps_lookback),
            vec_frst_k_hwp_wood
        ])


        tup_out = (
            vec_frst_c_paper,
            vec_frst_c_wood,
            vec_frst_k_hwp_paper,
            vec_frst_k_hwp_wood,
        )

        return tup_out
    


    def build_c_conversion_matrices(self,
        df_afolu_trajectories: pd.DataFrame,
        attr_lndu: Union['AttributeTable', None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build conversion matrices for carbon stock loss from land use 
            conversion between types. Does not account for sequestration from 
            conversion *to* a type; only loss from conversion away. 
            Sequestration from conversion to a type is estimated over time using
            NPP.

        Returns a tuple of the form:

            (
                arrs_c_agb,     # conversion arrays for above-ground biomass
                arrs_c_bgb,     # conversion arrays for below-ground biomass
                arr_bmass_ag,   # array of initial above-ground biomass
            )

        Estimates for conversion between land use types i and j are 

            max(c_i - c_j, 0)

            where c_i is the average carbon stock in type i.

        Function Arguments
        ------------------
        df_afolu_trajectories : DataFrame
            Input DataFrame containing columnar variables with initial carbon
            stocks. Note that these carbon stocks are dynamic, and only the 
            first time step is not adjusted over time.
        """
        
        ##  INITIALIZATION

        if not is_attribute_table(attr_lndu):
            attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        
        # get variables
        modvar_bmass_ag = self.model_attributes.get_variable(
            self.modvar_lndu_c_stock_factor_abg,
        )
        modvar_bmass_ratio_bg = self.model_attributes.get_variable(
            self.modvar_lndu_c_stock_bgb_ratio,
        )
        
        # get each variable
        arr_bmass_ag = (
            self
            .model_attributes
            .extract_model_variable(
                df_afolu_trajectories,
                modvar_bmass_ag,
                all_cats_missing_val = 0.0,
                expand_to_all_cats = True,
                return_type = "array_base",
            )
        )

        # get the below-ground to above-ground biomass ratio
        arr_bmass_bgr = (
            self
            .model_attributes
            .extract_model_variable(
                df_afolu_trajectories,
                modvar_bmass_ratio_bg,
                all_cats_missing_val = 0.0,
                expand_to_all_cats = True,
                return_type = "array_base",
            )
        )

        arr_bmass_bg = arr_bmass_bgr*arr_bmass_ag


        ##  INITIALIZE ARRAYS OUT
        
        n_tp = df_afolu_trajectories.shape[0]
        shp = (n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values)
        arrs_out_ag = np.zeros(shp)
        arrs_out_bg = np.zeros(shp)

        # iterate over each year (now a row)
        for k in range(n_tp):
            arrs_out_ag[k] = sf.bounded_outer_diff(arr_bmass_ag[k, :], vec_2 = arr_bmass_ag[k, :], )
            arrs_out_bg[k] = sf.bounded_outer_diff(arr_bmass_bg[k, :], vec_2 = arr_bmass_bg[k, :], )

        # setup output tuple and return
        out = (
            arrs_out_ag, 
            arrs_out_bg, 
            arr_bmass_ag,    
        )

        return out



    def calculate_ipcc_soc_deltas(self,
        vec_soc: np.ndarray,
        approach: int = 1
    ) -> np.ndarray:
        """Calculate the annual change in soil carbon using Approach 1 (even 
            though we have a transition matrix). 

        Function Arguments
        ------------------
        vec_soc : np.ndarray
            Vector of soil carbon

        Keyword Arguments
        -----------------
        approach : int
            * 1: use IPCC approach 1
            * 2: use change in soil carbon year over year implied by vec_soc

        """
        if approach not in [1, 2]:
            self._log(
                f"Warning in 'calculate_ipcc_soc_deltas': approach '{approach}' not found--please enter 1 or 2. Default will be set to 1.",
                type_log = "warning"
            )
            approach = 1

        if approach == 1:
            vec_soc_delta = np.concatenate([np.ones(self.time_dependence_stock_change)*vec_soc[0], vec_soc])
            vec_soc_delta = (vec_soc_delta[self.time_dependence_stock_change:] - vec_soc_delta[0:(len(vec_soc_delta) - self.time_dependence_stock_change)])/self.time_dependence_stock_change
        elif approach == 2:
            vec_soc_delta = vec_soc[1:] - vec_soc[0:-1]
            vec_soc_delta = np.insert(vec_soc_delta, 0, vec_soc_delta[0])

        return vec_soc_delta



    def calculate_soc_stock_change_with_time_dependence(self,
        arrs_lndu_land_conv: np.ndarray,
        arrs_lndu_soc_conversion_factors: np.ndarray,
        time_dependence_stock_change: int,
        shape_param: Union[float, int, None] = None,
    ) -> np.ndarray:
        """Calculate the SOC stock change with time dependence (includes some 
            qualitative non-linearities)

        Function Arguments
        ------------------
        arrs_lndu_land_conv : np.ndarray
            Arrays with land use conversion totals
        arrs_lndu_soc_conversion_factors : np.ndarray
            Arrays with SOC conversion factors between types
        time_dependence_stock_change : int
            Time-dependent stock change factor to use

        Keyword Arguments
        -----------------
        shape_param :  Union[float, int, None]
            Parameter that expands the sigmoid. If None, defaults to 
            time_dependence_stock_change/10.

        Notes
        -----
        See Volume 4, Chapter 2 of the IPCC 2006 Guidance for National 
            Greenhouse Gas Inventories, page 2.38 for the following description 
            of changes to soil carbon:

            "Changes in C stocks normally occur in a non-linear fashion, and it
            is possible to further develop the time dependence of stock change
            factors to reflect this pattern. For changes in land use or
            management that cause a decrease in soil C content, the rate of
            change is highest during the first few years, and progressively
            declines with time. In contrast, when soil C is increasing due to
            land-use or management change, the rate of accumulation tends to
            follow a sigmoidal curve, with rates of change being slow at the
            beginning, then increasing and finally decreasing with time. If
            historical changes in land-use or management practices are
            explicitly tracked by re- surveying the same locations (i.e.,
            Approach 2 or 3 activity data, see Chapter 3), it may be possible
            to implement a Tier 2 method that incorporates the non-linearity
            of changes in soil C stock."

        We use this guidance to assign different shapes to SOC for releases and
            sequestration. 

        """

        # get length
        n = min(len(arrs_lndu_land_conv), len(arrs_lndu_soc_conversion_factors))
        D = int(np.round(time_dependence_stock_change))
        shape_param = D/10 if (shape_param is None) else shape_param

        # get functions that characterize SOC stock change from conversion 
        #    - sequestration (sigmoid) 
        #    - emissions (complementary sigmoid with double phase)
        def emission_curve(x: float) -> float:
            d = D/2
            x_pr = x/(2*shape_param) + d*(shape_param - 1)/shape_param
            sig = 1/(1 + np.e**(d - x_pr))

            return 2*(1 - sig) - 1


        def sequestration_curve(x: float) -> float:
            d = D/2
            #f_fan(x/shape_param + d*(shape_param - 1)/shape_param, D, 1, 0, np.e, d)
            x_pr = x/shape_param + d*(shape_param - 1)/shape_param
            sig = 1/(1 + np.e**(d - x_pr))

            return sig

        # get emission/sequestration proportions
        vec_proportional_emission = np.array([emission_curve(x) for x in range(D)])
        vec_proportional_emission /= np.sum(vec_proportional_emission)
        vec_proportional_sequestration = np.array([sequestration_curve(x) for x in range(D)])
        vec_proportional_sequestration /= np.sum(vec_proportional_sequestration)
        
        #vec_proportional_emission = np.ones(D)/D#np.array(range(D, 0, -1))/(D*(D + 1)/2)
        #vec_proportional_sequestration = np.ones(D)/D#np.array(range(1, D + 1))/(D*(D + 1)/2)
        # get arrays in terms of sequestration and emission
        arrs = arrs_lndu_land_conv*arrs_lndu_soc_conversion_factors
        arrs_sequestration = sf.vec_bounds(arrs, (0, np.inf))
        arrs_emission = sf.vec_bounds(arrs, (-np.inf, 0))

        v_out = np.zeros(n)
        
        for i in range(n):
            v_cur_emission = vec_proportional_emission*(arrs_emission[i].sum())
            v_cur_sequestration = vec_proportional_sequestration*(arrs_sequestration[i].sum())
            v_cur = v_cur_emission + v_cur_sequestration
            v_out[i:min(n, i + D)] += v_cur[0:min(D, n - i)]

        return v_out
    


    def check_cropland_fractions(self,
        df_in: pd.DataFrame,
        frac_type: str = "initial",
        thresh_for_correction: float = 0.01,
    ) -> np.ndarray:
        """
        Check cropland fractions and extract from data frame. Returns np.ndarray
            ordered by AGRC attribute keys
        """
        if frac_type not in ["initial", "calculated"]:
            raise ValueError(f"Error in frac_type '{frac_type}': valid values are 'initial' and 'calculated'.")

        varname = (
            self.modvar_agrc_area_prop_init 
            if frac_type == "initial"
            else self.modvar_agrc_area_prop_calc
        )

        arr = self.model_attributes.extract_model_variable(#
            df_in, 
            varname, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        totals = sum(arr.transpose())
        m = max(np.abs(totals - 1))
        if m > thresh_for_correction:
            msg = f"""
                Invalid crop areas found in check_cropland_fractions. The
                maximum fraction total was {m}; the maximum allowed deviation 
                from 1 is {thresh_for_correction}.
            """
            raise ValueError(msg)

        arr = (arr.transpose()/totals).transpose()

        return arr



    def check_markov_shapes(self, 
        arrs: np.ndarray, 
        function_var_name:str,
    ) -> None:
        """
        Check the shape of transition/emission factor matrices sent to 
            project_land_use
        """
        # get land use info
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)

        if len(arrs.shape) < 3:
            msg = f"""
            f"Invalid shape for array {function_var_name}; the array must be a 
            list of square matrices."
            """
            raise ValueError(msg)

        elif arrs.shape[1:3] != (attr_lndu.n_key_values, attr_lndu.n_key_values):
            msg = f"""
            f"Invalid shape of matrices in {function_var_name}. They must have 
            shape ({attr_lndu.n_key_values}, {attr_lndu.n_key_values})."
            """
            raise ValueError(msg)

        return None



    def convert_fuelwood_to_c_equivalent(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_energy_demand_fuelwood: Union[float, np.ndarray],
        modvar_frac_c: Union['ModelVariable', None] = None,
        units_energy: Union['ModelVariable', str, None] = None,
        units_mass: Union['ModelVariable', str, None] = None,
    ) -> Union[float, np.ndarray]:
        """Convert energy demand for wood into aboveground carbon removal 
            equivalent. 

        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            DataFrame containing trajectories used to get key variables
        vec_energy_demand_fuelwood : Union[float, np.ndarray]
            Vector or float denoting energy demand for biomass (fuelwood). 

        Keyword Arguments
        -----------------
        modvar_frac_c : Union['ModelVariable', None]
            Optional model variable to use to specify fraction of C that makes
            up mass. If None, defaults to `self.modvar_frst_frac_c_per_dm`
        units_energy : Union['ModelVariable', str, None]
            Optional specification of energy units for input vector. If None, 
            defaults to configuration values. 
        units_mass : Union['ModelVariable', str, None]
            Optional specification of mass units for C equivalent. If None, 
            defaults to configuration values. 
        """
        
        ##  INITIALIZATION 

        attr_enfu = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_enfu,
        )
        cat_biomass = self.model_enercons.cat_enfu_biomass
        ind_biomass = attr_enfu.get_key_value_index(cat_biomass, )

        # get model variables--start with gravimetric energy demand
        modvar_ged = self.model_attributes.get_variable(
            self.model_enercons.modvar_enfu_energy_density_gravimetric,
        )

        # get C per dry matter to use
        modvar_c_per_dm = self.model_attributes.get_variable(modvar_frac_c, )
        if modvar_c_per_dm is None:
            modvar_c_per_dm = self.model_attributes.get_variable(
                self.modvar_frst_frac_c_per_dm, 
            )

        
        ##  EXTRACT VARIABLES

        # extract gravimetric energy demand
        arr_enfu_ged = self.model_attributes.extract_model_variable(
            df_afolu_trajectories,
            modvar_ged,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # c fraction
        vec_frst_c_frac = self.model_attributes.extract_model_variable(
            df_afolu_trajectories,
            modvar_c_per_dm,
            override_vector_for_single_mv_q = False,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        
        ##  CONVERT
        
        um_energy = self.model_attributes.get_unit("energy")
        um_mass = self.model_attributes.get_unit("mass")
        units_energy = self.get_units_from_specification(units_energy, "energy", )
        units_mass = self.get_units_from_specification(units_mass, "mass", )

        # convert to energy units and mass units
        arr_enfu_ged *= um_energy.convert(
            modvar_ged.attribute("unit_energy"),
            units_energy,
        )

        #   
        arr_enfu_ged /= um_mass.convert(
            modvar_ged.attribute("unit_mass"),
            units_mass,
        )

        # get biomass as 
        vec_ged_biomass = arr_enfu_ged[:, ind_biomass]
        vec_mass_biomass = vec_energy_demand_fuelwood/vec_ged_biomass
        vec_mass_c = vec_mass_biomass*vec_frst_c_frac

        return vec_mass_c



    def estimate_biomass_demand_entc(self,
        df_energy_trajectories: pd.DataFrame,
        vec_fuel_demand_electricity: np.ndarray,
        kludge_inflation_factor: float = 0.05,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the demand for biomass coming from ENTC. Returns vector of
            estimated biomass demand in configuration units.

        Function Arguments
        ------------------
        df_energy_trajectories : pd.DataFrame
            DataFrame containing trajectories used to estimate demand for 
            biomass in ENTC
        vec_fuel_demand_electricity : np.ndarray
            Vector (with same length as df_energy_trajectories.shape[0])
            containing total electricity demand from consumption sectors in
            configuration units
        attr_enfu : AttributeTable
            AttributeTable for Energy Fuels
        ind_biomass: int
            Index in Energy Fuels categories storing fuel_biomass
        ind_electricity : int
            Index in Energy Fuels categories storing fuel_electricity

        Keyword Arguments
        -----------------
        kludge_inflation_factor : float
            Inflate the estimate by 1 + kludge_inflation_factor; accounts for
            endogenous electricity demand.

            NOTE: This could be too low in regions with high amounts of fuel
            production; however, those regions rarely use large amounts of 
            biomass for fuel (especially in electricity production), and this is 
            expected to be a rare problem.
        """

        ##  INITIALIZATION OF MODEL ATTRIBUTES INFO

        # attribute table
        attr_entc = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_entc,
        )

        # get the powerplant category associated with biomass
        cat_entc_biomass = self.model_attributes.filter_keys_by_attribute(
            self.model_attributes.subsec_name_entc,
            {
                "electricity_generation_cat_fuel": unclean_category(self.model_enercons.cat_enfu_biomass)
            }
        )[0]

        ind_entc_biomass = attr_entc.get_key_value_index(cat_entc_biomass, )


        # get model variables
        modvar_entc_efficiency = self.model_attributes.get_variable(
            self.modvar_entc_efficiency_factor_technology,
        )
        modvar_entc_msp = self.model_attributes.get_variable(
            self.modvar_entc_nemomod_min_share_production,
        )
        modvar_entc_rc = self.model_attributes.get_variable(
            self.modvar_entc_nemomod_residual_capacity,
        )

        # get some units managers
        um_power = self.model_attributes.get_unit("power")
        

        ##########################
        #    START ESTIMATION    #
        ##########################

        """ESTIMATION APPROACH:

            1.  Get residual capacity and estimate how much would be produced at 
                max (e_cap)
            2.  Using electricity required from consumption subsectors, 
                calculate residual MSP (e_res = 1 - total_msp_accounted)
            3.  Estimate e_msp as E_total*MSP_biomass (total electricity)
            4.  Set the estimate elementwise as:
                    min(max(e_cap, e_msp), e_msp + e_res)
        """

        ##  GET EFFICIENCY OF BIOMASS PLANT

        arr_efficiency_factors = self.model_attributes.extract_model_variable(
            df_energy_trajectories,
            modvar_entc_efficiency,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, np.inf), 
        )

        vec_efficiciency_biomass = arr_efficiency_factors[:, ind_entc_biomass]
        

        ##  GET RESIDUAL CAPACITY OF BIOMASS
        #   - use this if MSP is 0 and the residual MSP (1 - total accounted)
        arr_residual_capacity = self.model_attributes.extract_model_variable(
            df_energy_trajectories,
            modvar_entc_rc,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # convert to GW to ensure can be converted to GWY
        units_power_intermediate = "gw"
        units_energy = self.model_attributes.get_energy_power_swap(
            units_power_intermediate, 
        )

        # get the scalars we'll use to convert
        scalar_energy = self.model_attributes.get_energy_equivalent(
            units_energy,
        )
        scalar_power = um_power.convert(
            modvar_entc_rc.attribute("unit_power"),
            units_power_intermediate,
        ) 

        # convert to the intermediate power units,
        #  which are then translated to annual equivalents, 
        #  and finally config units
        arr_residual_capacity *= scalar_power
        arr_residual_capacity *= scalar_energy

        # get the estimated capacity number
        vec_biomass_cap = arr_residual_capacity[:, ind_entc_biomass]

        
        ##  ESTIMATE MSP

        arr_msp = self.model_attributes.extract_model_variable(
            df_energy_trajectories,
            modvar_entc_msp,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        vec_sums = arr_msp.sum(axis = 1, )
        if (vec_sums.max() > 1) or (vec_sums.min() < 0):
            raise RuntimeError(f"Invalid {modvar_entc_msp.name} values found: sum must not exceed 1 or be below 0.")
        
        # get biomass residual potential and specified share of production
        vec_biomass_residual_potential = (1 - vec_sums)*vec_fuel_demand_electricity
        vec_biomass_msp = arr_msp[:, ind_entc_biomass]*vec_fuel_demand_electricity
        
        # build the estimate
        vec_estimate = np.max(
            np.array(
                [
                    vec_biomass_cap,
                    vec_biomass_msp
                ]
            ),
            axis = 0,
        )

        vec_estimate = np.min(
            np.array(
                [
                    vec_estimate,
                    vec_biomass_msp + vec_biomass_residual_potential
                ]
            ),
            axis = 0,
        )
        
        # inflate by inverse efficiency factor to get input fuel demand
        vec_estimate = np.nan_to_num(
            vec_estimate/vec_efficiciency_biomass,
            nan = 0.0,
            posinf = 0.0,
        )

        # inflate with kludge?
        if sf.isnumber(kludge_inflation_factor):
            vec_estimate *= kludge_inflation_factor

        return vec_estimate
    


    def estimate_c_demand_fuelwood_noag(self,
        df_afolu_trajectories: pd.DataFrame,
        df_ippu_production: pd.DataFrame,
        convert_to_c: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the specified demand fir mass of c removals in term of a 
            target model variable. Later gets adjusted if stock is not 
            available.
        
        Combines biomass energy demands from SCOE, INEN, and ENTC (as estimated) 
            and combines those

        NOTE:
            - emissions from AG/LVST industrial use are estimated using GDP. 

        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            DataFrame containing trajectories used to estimate removals
        df_ippu_production : np.ndarray
            DataFrame containing IPPU production estimates (from internal model
            run; will adjust later)
        attr_enfu : AttributeTable
            AttributeTable for Energy Fuels
        ind_biomass: int
            Index in Energy Fuels categories storing fuel_biomass
        ind_electricity : int
            Index in Energy Fuels categories storing fuel_electricity

        Keyword Arguments
        -----------------
        convert_to_c : bool
            Set to True to convert output to C. If True, can pass units via 
            kwargs to convert_fuelwood_to_c_equivalent()
        kwargs :
            Passed to convert_fuelwood_to_c_equivalent()
        """
        ##  INITIALIZATION

        # get some model attribute related information
        attr_enfu = self.model_attributes.get_attribute_table(self.subsec_name_enfu)
        modvar_ag = self.model_attributes.get_variable(self.modvar_agrc_yield, )
        modvar_lvst = self.model_attributes.get_variable(self.modvar_lvst_total_animal_mass, )

        # get some indices
        ind_enfu_biomass = attr_enfu.get_key_value_index(self.cat_enfu_biomass)
        ind_enfu_electricity = attr_enfu.get_key_value_index(self.model_enercons.cat_enfu_electricity)


        ##  BUILD INPUT FOR EnergyConsumption

        # generate yield/lvst pop + production
        df_for_enercons = sf._concat_df(
            [
                df_afolu_trajectories.copy(),
                df_ippu_production,
            ], 
            axis = 1,
        )

        # set ag/lvst to 0 for now; will add later in project_integrated_land_use()
        df_for_enercons = modvar_ag.spawn_default_dataframe(
            df_base = df_for_enercons,
            fill_value = 0.0,
        )

        df_for_enercons = modvar_lvst.spawn_default_dataframe(
            df_base = df_for_enercons,
            fill_value = 0.0,
        )     

        # try to run the model 
        try:
            df_energy_cons_out = self.model_enercons(df_for_enercons, )

        except Exception as e:
            raise RuntimeError(f"Running of EnergyConsumption model for biomass estimates failed with error: {e}")

        
        ##  GET ENERGY DEMANDS--WILL REQUIRE ESTIMATING ELECTRICITY DEMANDS FROM BIOMASS

        modvars = [
            self.model_enercons.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.model_enercons.modvar_enfu_energy_demand_by_fuel_inen,
            self.model_enercons.modvar_enfu_energy_demand_by_fuel_scoe,
            self.model_enercons.modvar_enfu_energy_demand_by_fuel_trns,
        ] 

        # initialize fuel demands
        vec_fuel_demand_biomass = 0
        vec_fuel_demand_electricity = 0

        for modvar in modvars:
            # get energy demand for fuels
            arr_fuel_demand = self.model_attributes.extract_model_variable(#
                df_energy_cons_out,
                modvar,
                all_cats_missing_val = 0.0,
                extraction_logic = "all",
                expand_to_all_cats = True,
                return_type = "array_base",
            )

            # convert to config units for now
            arr_fuel_demand *= self.model_attributes.get_scalar(
                modvar, 
                return_type = "energy",
            )

            vec_fuel_demand_biomass += arr_fuel_demand[:, ind_enfu_biomass]
            vec_fuel_demand_electricity += arr_fuel_demand[:, ind_enfu_electricity]

    
        # estimate any specified ENTC demand
        vec_fuel_demand_biomass_entc = self.estimate_biomass_demand_entc(
            df_for_enercons,
            vec_fuel_demand_electricity,
            **kwargs,
        )

        # get biomass demand and return if not converting to C
        vec_fuel_demand_biomass_out = vec_fuel_demand_biomass + vec_fuel_demand_biomass_entc
        if not convert_to_c:
            return vec_fuel_demand_biomass_out
        
        
        # convert to C and return
        vec_c_demand = self.convert_fuelwood_to_c_equivalent(
            df_afolu_trajectories,
            vec_fuel_demand_biomass_out,
            modvar_frac_c = self.modvar_frst_frac_c_per_dm,
            **kwargs,
        )

        return vec_c_demand
    


    def estimate_c_demand_from_hwp_and_removals(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        #dict_check_integrated_variables: dict,
        units_mass_out: Union['ModelVariable', str, None] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the demand for C from Harvested Wood Production (HWP) and 
            fuel wood removal. Estimates total C requirement in terms of 
            configuration Emissions Mass units (or other unit specified using
            `units_mass_out` kwarg).

        Returns a Tuple of the form
        
            (
                vec_frst_c_fuel,
                vec_frst_c_paper,
                vec_frst_c_wood,
            )

        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            DataFrame containing trajectories used to estimate wood removals
        vec_rates_gdp : np.ndarray
            Vector giving rates of GDP growth in same order as 
            df_afolu_trajectories [length = n_rows(df_afolu_trajectories) - 1]
        dict_check_integrated_variables : Dict[str, List[str]]


        Keyword Arguments
        -----------------
        units_mass_out : Union['ModelVariable', str, None]
            Units of mass for vec_frst_c_paper and vec_frst_c_wood. If None, it
            is assumed that the units are configuration units.
        **kwargs :
            Passed to estimate_c_demand_fuelwood_noag()
        """
        ## 

        # initialize some small checks and shorthands 
        # check_ippu = dict_check_integrated_variables.get(self.subsec_name_ippu)
        # check_scoe = dict_check_integrated_variables.get(self.subsec_name_scoe)
        modvar_demand_hwp = self.model_attributes.get_variable(
            self.model_ippu.modvar_ippu_demand_for_harvested_wood,
        )


        ##  ESTIMATE C DEMAND FROM HWP

        # IPPU components--initialize hwp vectors
        arr_frst_harvested_wood_industrial = 0.0
        vec_frst_harvested_wood_industrial_paper = 0.0
        vec_frst_harvested_wood_industrial_wood = 0.0
    

        # get projections of industrial wood and paper product demand
        attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)
        ind_paper = attr_ippu.get_key_value_index(self.cat_ippu_paper)
        ind_wood = attr_ippu.get_key_value_index(self.cat_ippu_wood)

        # production data
        _, dfs_ippu_harvested_wood = self.model_ippu.get_production_with_recycling_adjustment(
            df_afolu_trajectories, 
            vec_rates_gdp
        )

        # get industrial production demand for HWP
        arr_frst_harvested_wood_industrial = self.model_attributes.extract_model_variable(#
            dfs_ippu_harvested_wood[1], 
            modvar_demand_hwp, 
            expand_to_all_cats = True,
            return_type = "array_base", 
        )

        vec_frst_harvested_wood_industrial_paper = arr_frst_harvested_wood_industrial[:, ind_paper]
        vec_frst_harvested_wood_industrial_wood = arr_frst_harvested_wood_industrial[:, ind_wood]

        print(f"vec_frst_harvested_wood_industrial_paper = {vec_frst_harvested_wood_industrial_paper}")

        # get C fraction of HWP
        vec_frst_ef_c = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_frst_frac_c_per_hwp,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )
        vec_frst_c_paper = vec_frst_harvested_wood_industrial_paper*vec_frst_ef_c
        vec_frst_c_wood = vec_frst_harvested_wood_industrial_wood*vec_frst_ef_c


        ##  DO UNIT CONVERSION

        # get target unit, and convert from IPPU production mass to unit mass
        um_mass = self.model_attributes.get_unit("mass")
        units_mass_out = self.get_units_from_specification(units_mass_out, "mass", )
        scalar = um_mass.convert(
            modvar_demand_hwp.attribute("unit_mass"),
            units_mass_out,
        )

        # update vecs out
        vec_frst_c_paper *= scalar
        vec_frst_c_wood *= scalar


        ##  GET REMOVALS OF FUELWOOD

        vec_frst_c_fuel = self.estimate_c_demand_fuelwood_noag(
            df_afolu_trajectories,
            dfs_ippu_harvested_wood[2],
            units_mass = units_mass_out,
            **kwargs,
        )


        out = (
            vec_frst_c_fuel,
            vec_frst_c_paper,
            vec_frst_c_wood,
        )

        return out


    
    def get_emissions_co2_from_hwp(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_frst_c_paper: np.ndarray,
        vec_frst_c_wood: np.ndarray,
        units_mass: Union['ModelVariable', str, None] = None,
    ) -> List[pd.DataFrame]:
        """Estimate the CO2 emissions from Harvested Wood Products.

        Function Arguments
        ------------------
        df_afolu_trajectories : 
            DataFrame containing trajectories used to estimate wood removals
        vec_frst_c_paper : np.ndarray
            Vector of mass of forest C needed for paper
        vec_frst_c_wood : np.ndarray
            Vector of mass of forest C needed for wood products

        Keyword Arguments
        -----------------
        units_mass : Union['ModelVariable', str, None]
            Units of mass for vec_frst_c_paper and vec_frst_c_wood. If None, it
            is assumed that the units are configuration units.
        """

        ##  INITIALIZATION

        # back projection
        historical_method = self.model_attributes.configuration.get("historical_harvested_wood_products_method")

        # units conversion
        um_mass = self.model_attributes.get_unit("mass")
        units_mass_input = self.get_units_from_specification(units_mass, "mass", )
        units_mass_config = self.get_units_from_specification(None, "mass", )

        scalar_mass = um_mass.convert(units_mass_input, units_mass_config, )

        
        ##  PREPARE FOR FOD

        # get half-life factors for FOD model - start with paper
        vec_frst_k_hwp_paper = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_frst_hwp_half_life_paper,
            return_type = "array_base",
        )
        vec_frst_k_hwp_paper = np.log(2)/vec_frst_k_hwp_paper

        # add wood
        vec_frst_k_hwp_wood = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_frst_hwp_half_life_wood,
            return_type = "array_base",
        )
        vec_frst_k_hwp_wood = np.log(2)/vec_frst_k_hwp_wood

        # totals
        vec_frst_ef_c = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_frst_frac_c_per_hwp,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )


        # set a lookback based on some number of years (max half-life to estimate some amount of carbon stock)
        if historical_method == "back_project":

            n_years_lookback = int(self.model_attributes.configuration.get("historical_back_proj_n_periods"))

            if n_years_lookback > 0:
                
                (
                    vec_frst_c_paper,
                    vec_frst_c_wood,
                    vec_frst_k_hwp_paper,
                    vec_frst_k_hwp_wood,
                ) = self.back_project_hwp_c_k(
                    n_years_lookback,
                    vec_frst_c_paper,
                    vec_frst_c_wood,
                    vec_frst_k_hwp_paper,
                    vec_frst_k_hwp_wood,
                )


        else:
            # set up n_years_lookback to be based on historical
            n_years_lookback = 0
            msg = f"""Error in estimate_c_demand_from_hwp(): 
            historical_harvested_wood_products_method 'historical' not supported 
            at the moment.
            """
            raise ValueError(msg)


        ## RUN ASSUMPTIONS USING STEADY-STATE ASSUMPTIONS (see Equation 12.4)

        # initialize using Equation 12.4 for paper and wood
        vec_frst_c_from_hwp_paper = np.zeros(len(vec_frst_k_hwp_paper))
        vec_frst_c_from_hwp_paper[0] = np.mean(
            vec_frst_c_paper[0:min(5, len(vec_frst_c_paper))]
        )/vec_frst_k_hwp_paper[0]

        vec_frst_c_from_hwp_wood = np.zeros(len(vec_frst_k_hwp_wood))
        vec_frst_c_from_hwp_wood[0] = np.mean(
            vec_frst_c_wood[0:min(5, len(vec_frst_c_wood))]
        )/vec_frst_k_hwp_wood[0]


        # First Order Decay (FOD) implementation
        for i in range(len(vec_frst_c_from_hwp_paper) - 1):
            # paper
            current_stock_paper = vec_frst_c_from_hwp_paper[0] if (i == 0) else vec_frst_c_from_hwp_paper[i]
            exp_k_paper = np.exp(-vec_frst_k_hwp_paper[i])
            vec_frst_c_from_hwp_paper[i + 1] = current_stock_paper*exp_k_paper + ((1 - exp_k_paper)/vec_frst_k_hwp_paper[i])*vec_frst_c_paper[i]

            # wood
            current_stock_wood = vec_frst_c_from_hwp_wood[0] if (i == 0) else vec_frst_c_from_hwp_wood[i]
            exp_k_wood = np.exp(-vec_frst_k_hwp_wood[i])
            vec_frst_c_from_hwp_wood[i + 1] = current_stock_wood*exp_k_wood + ((1 - exp_k_wood)/vec_frst_k_hwp_wood[i])*vec_frst_c_wood[i]

        # reduce from look back
        if n_years_lookback > 0:
            vec_frst_c_from_hwp_paper = vec_frst_c_from_hwp_paper[(n_years_lookback - 1):]
            vec_frst_c_from_hwp_wood = vec_frst_c_from_hwp_wood[(n_years_lookback - 1):]
            
        vec_frst_c_from_hwp_paper_delta = vec_frst_c_from_hwp_paper[1:] - vec_frst_c_from_hwp_paper[0:-1]
        vec_frst_c_from_hwp_wood_delta = vec_frst_c_from_hwp_wood[1:] - vec_frst_c_from_hwp_wood[0:-1]


        # get emissions from co2
        vec_frst_emissions_co2_hwp = vec_frst_c_from_hwp_paper_delta + vec_frst_c_from_hwp_wood_delta
        vec_frst_emissions_co2_hwp *= -1*self.factor_c_to_co2
        vec_frst_emissions_co2_hwp *= scalar_mass

        #*= self.model_attributes.get_scalar(
        #    self.model_ippu.modvar_ippu_demand_for_harvested_wood, 
        #    "mass",
        #)


        list_dfs_out = [
            self.model_attributes.array_to_df(
                vec_frst_emissions_co2_hwp, 
                self.modvar_frst_emissions_co2_hwp,
            )
        ]

        return list_dfs_out
    


    def format_lndu_conversion_emissions_and_scale_secondary_forest(self,
        arrs_lndu_emissions_conv_matrices: np.ndarray,
        scalar_secondary_forest_conversions: Union[np.ndarray, float] = 1.0,
    ) -> pd.DataFrame:
        """Scale conversion emissions *out* of secondary forest based on biomass
            accumulation from NPP (calculate assuming that original secondary
            is coverted first) 
        """
        arrs_out = arrs_lndu_emissions_conv_matrices.copy()
        n, _ = arrs_out[0].shape
        w = np.where(np.arange(n) != self.ind_lndu_fsts)[0]

        for i, arr in enumerate(arrs_out):
            arr[self.ind_lndu_fsts, w] *= (
                scalar_secondary_forest_conversions[i]
                if isinstance(scalar_secondary_forest_conversions, np.ndarray)
                else scalar_secondary_forest_conversions
            )
            arrs_out[i] = arr

        # scale the secondary forest conversion emissions and convert land use conversion emission totals to config
        df_lndu_emissions_conv_matrices = self.format_transition_matrix_as_input_dataframe(
            arrs_out,
            exclude_time_period = True,
            modvar = self.modvar_lndu_emissions_conv,
        )

        return df_lndu_emissions_conv_matrices
        


    def format_transition_matrix_as_input_dataframe(self,
        mat: np.ndarray,
        exclude_time_period: bool = False,
        field_key: str = "key",
        key_vals: Union[List[str], None] = None,
        modvar: Union[str, 'ModelVariable', None] = None,
        vec_time_periods: Union[List, np.ndarray, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """Convert an input transition matrix mat to a wide dataframe using 
            AFOLU.modvar_lndu_prob_transition as variable template

        NOTE: Returns None on error.


        Function Arguments
        ------------------
        mat : np.ndarray
            Row-stochastic transition matrix to format as wide data frame OR
            array of row-stochastic transition matrices to format. If specifying
            as array matrices, format_transition_matrix_as_input_dataframe()
            will interpret as being associated with sequential time periods. 
            * NOTE: Time periods can be specified using vec_time_periods. 
                Time periods specified in this way must have the same length
                as the array of matrices.

        Keyword Arguments
        -----------------
        exclude_time_period : bool
            Drop time period from data frame?
        field_key : str
            Temporary field to use for merging
        key_vals : Union[List[str], None]
            Ordered land use categories representing states in mat. If None, 
            defaults to attr_lndu.key_values. Should be of length n for an nxn 
            transition matrix.
        modvar : Union[str, 'ModelVariable', None]
            Optional specification of I/J model variable to use. If None,
            defaults to transition probability 
            (self.modvar_lndu_prob_transition)
        vec_time_periods : Union[List, np.ndarray, None]
            Optional vector of time periods to specify for matrices. Does not 
            affect implementation with single matrix. If None and mat is an 
            array of arrays, takes value of: 
                * ModelAttributes.time_period key_values (if mat is of same 
                    length)
                * range(len(mat)) otherwise
        """
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        field_time_period = self.model_attributes.dim_time_period
        attr_time_period = self.model_attributes.get_dimensional_attribute_table(field_time_period)
        
        key_vals = (
            attr_lndu.key_values 
            if (key_vals is None)
            else [x for x in key_vals if x in attr_lndu.key_values]
        )
        
        modvar = (
            modvar
            if self.model_attributes.get_variable(modvar) is not None
            else self.modvar_lndu_prob_transition
        )


        ##  INITIALIZATION AND CHECKS

        # shared variables in melt/pivot
        field_field = "field"
        field_value = "value"
        field_variable = "variable"

        # check if is a single matrix; if so, treat as list and reduce later
        is_single_matrix = (len(mat.shape) == 2)
        mat = np.array([mat]) if is_single_matrix else mat
        mat_shape = mat.shape
        vec_time_periods = np.zeros(1) if is_single_matrix else vec_time_periods

        # check specification of time periods
        if vec_time_periods is None:
            # default to model attributes time period key values if same length
            vec_time_periods = (
                np.arange(mat_shape[0]).astype(int)
                if mat_shape[0] != len(attr_time_period.key_values)
                else attr_time_period.key_values
            )

        elif not sf.islistlike(vec_time_periods):
            tp = str(type(vec_time_periods))
            self._log(
                f"Error in format_transition_matrix_as_input_dataframe: invalid type specifiation '{tp}' of vec_time_periods.",
                type_log = "error"
            )
            return None

        elif len(vec_time_periods) != len(mat):
            self._log(
                f"Error in format_transition_matrix_as_input_dataframe: invalid specifiation of vec_time_periods. The length of vec_time_periods ({len(vec_time_periods)}) and mat ({mat.shape[0]}) must be the same.",
                type_log = "error"
            )
            return None


        ##  BUILD DATA FRAME

        # setup new fields - key values and time periods
        field_key_vals = [x[1] for x in itertools.product(np.ones(mat_shape[0]), key_vals)]
        field_time_periods = (
            np.concatenate(
                np.outer(
                    vec_time_periods, 
                    np.ones(mat.shape[1])
                )
            )
            .astype(int)
        )

        # generate data frame before melting
        df_mat = pd.DataFrame(
            np.concatenate(mat), 
            columns = key_vals
        )
        df_mat[field_key] = field_key_vals
        df_mat[field_time_period] = field_time_periods


        ##  MELT, APPLY VARIABLES, THEN PIVOT

        # note: sorting is important here to ensure that variable names generated below are in correct order
        df_out = (
            pd.melt(
                df_mat, 
                id_vars = [field_time_period, field_key], 
                value_vars = key_vals,
                var_name = field_variable,
                value_name = field_value,
            )
            .sort_values(by = [field_time_period, field_key, field_variable])
            .reset_index(drop = True)
        )

        # get vars and expand 
        field_variables = self.model_attributes.build_variable_fields(
            modvar,
            restrict_to_category_values = key_vals,
        )
        field_variables = [x[1] for x in itertools.product(np.ones(mat_shape[0]), field_variables)]
        df_out[field_field] = field_variables

        # pivot and drop time period if returning as single matrix
        df_out = sf.pivot_df_clean(
            df_out[[field_time_period, field_value, field_field]],
            [field_field],
            [field_value]
        )
        (
            df_out.drop([field_time_period], axis = 1, inplace = True)
            if is_single_matrix | exclude_time_period
            else None
        )

        return df_out
    


    def get_lndu_area_drained_organic_soils(self,
        df_afolu_trajectories: pd.DataFrame,
        arr_land_use: np.ndarray,
        arrs_lndu_land_conv: np.ndarray,
    ) -> pd.DataFrame:
        """Get the area of drained organic soils in terms fo arr_land_use
            areas.
        
        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            DataFrame containing input trajectories
        arr_land_use : np.ndarray
            Array of land use areas by land use classification in terms of
            region area variable
        arrs_lndu_land_conv : np.ndarray
            Array of land use conversion areas by land use classification 
            in terms of region area variable
            
        Keyword Arguments
        -----------------
        
        """
        ##  INITIALIZE SOME MODEL ATTRIBUTES ELEMENTS

        # land use attribute table and land use pyategory element
        attr_lndu = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_lndu, 
        )
        pycat_lndu = self.model_attributes.get_subsector_attribute(
            self.model_attributes.subsec_name_lndu,
            "pycategory_primary_element",
        )

        # model variable for fraction drained organic soil
        modvar_dos = self.model_attributes.get_variable(
            self.modvar_lndu_initial_frac_area_dos,
        )

        
        # get fraction variables here
        arr_lndu_frac_drained_organic = self.model_attributes.extract_model_variable(
            df_afolu_trajectories,
            modvar_dos,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        arr_lndu_area_drained_organic = arr_lndu_frac_drained_organic*arr_land_use
        

        ##  ITERATE OVER TIME PERIODS TO ESTIMATE AREA WITH DOS
        #
        # - assume transitions out of classes are uniform across 
        #   land use
        # - use transition matrices to estimate DOS
        #
        
        # get categories and indices 
        cats_lndu_dos = modvar_dos.dict_category_keys.get(pycat_lndu)
        inds_lndu_dos = [attr_lndu.get_key_value_index(x) for x in cats_lndu_dos]
        inds_lndu_not_dos = [attr_lndu.get_key_value_index(x) for x in attr_lndu.key_values if x not in cats_lndu_dos]

        # initialize the fraction of each class that is DOS, then iterate over time periods
        vec_frac_dos = arr_lndu_frac_drained_organic[0].copy()

        for t in range(arr_lndu_area_drained_organic.shape[0] - 1):

            # use vec_fraction * arr_conv to calculate new area of DOS; then, filter to only include valid categories
            vec_frac_dos[self.ind_lndu_wetl] = 1.0
            vec_area_dos = vec_frac_dos.dot(arrs_lndu_land_conv[t])
            vec_area_dos[inds_lndu_not_dos] = 0

            #  update
            arr_lndu_area_drained_organic[t + 1] = vec_area_dos
            vec_frac_dos = np.nan_to_num(
                vec_area_dos/arr_land_use[t + 1],
                nan = 0.0,
                posinf = 0.0,
            )

        
        return arr_lndu_area_drained_organic



    def get_lndu_areas_to_from_remaining(self,
        arr_converted: np.ndarray, 
        ind: int,
    ) -> Tuple[float, float]:
        """For a conversion matrix `arr_converted` and land use index, calculate 
            how much secondary area was added to (column sums) and lost from 
            (row sums).

            Returns a tuple with 

            (
                area_to,
                area_from,
                area_remaining
            )
        """
        n, _ = arr_converted.shape
        w = np.array([i for i in range(n) if i != ind])

        area_to = arr_converted[w, ind].sum()
        area_from = arr_converted[ind, w].sum()
        area_remaining = arr_converted[w, w]
        out = (area_to, area_from, area_remaining, )

        return out



    def get_frst_area_from_df(self,
        df_land_use: pd.DataFrame,
        attr_frst: AttributeTable,
    ) -> np.ndarray:
        """Retrieve the area of forest types as a numpy array
        """
        # get ordered fields from land use
        fields_lndu_forest_ordered = [self.dict_cats_frst_to_cats_lndu.get(x) for x in attr_frst.key_values]
        fields_ext = self.model_attributes.build_variable_fields(
            self.modvar_lndu_area_by_cat,
            restrict_to_category_values = fields_lndu_forest_ordered,
        )
        arr_area_frst = np.array(df_land_use[fields_ext])
        
        return arr_area_frst
    

    
    def get_frst_methane_factors(self,
        df_afolu_trajectories: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get forest methane emission factors in terms of output emission mass 
            and self.model_socioeconomic.modvar_gnrl_area area units.
        """
        arr_frst_ef_methane = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_ef_ch4, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )
        arr_frst_ef_methane *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_ef_ch4,
            "area"
        )

        return arr_frst_ef_methane
    

    
    def get_frst_sequestration_and_land_use_conversion_emission(self,
        df_afolu_trajectories: pd.DataFrame,
        arrs_lndu_conversion: np.ndarray,
        arrs_lndu_conversion_emissions: np.ndarray,
        arr_lndu_areas: np.ndarray,
        arr_frst_areas: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Get forest sequestration and land use conversion emissions.
        """
        if self.npp_curve is None:

            out = self.get_frst_sequestration_static(
                df_afolu_trajectories,
                arr_frst_areas,
            )

            scalar = 1.0

        else:
            out, scalar = self.get_frst_sequestration_from_npp(
                df_afolu_trajectories,
                arrs_lndu_conversion,
                arr_lndu_areas,
                arr_frst_areas,
                **kwargs
            )

        # scale conversion estimates based on secondary forest biomass growth (scaled) and add to output list
        df_lndu_emissions_conversion = self.format_lndu_conversion_emissions_and_scale_secondary_forest(
            arrs_lndu_conversion_emissions, 
            scalar_secondary_forest_conversions = scalar,
        )

        out.append(df_lndu_emissions_conversion)

        return out
    
    

    def get_frst_sequestration_from_npp(self,
        df_afolu_trajectories: pd.DataFrame,
        arrs_lndu_conversion: np.ndarray,
        arr_lndu_areas: np.ndarray, #arr_land_use
        arr_frst_areas: np.ndarray,
        attr_lndu: Union[AttributeTable, None] = None,
        field_sfv_secondary: str = "secondary",
        key_cmf: str = "cmf",
        key_params: str = "params",
        npp_curve: Union[str, None] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Calculate forest sequestration using NPP.

        NOTE: all secondary sequestration at time t = 0 is assumed to use the
            average specified sequestration factor.

        
        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            data frame containing input trajectories
        arrs_lndu_conversion : np.ndarray
            arrays of converted areas in terms of self.modvar_lndu_area_by_cat
        arr_lndu_areas : np.ndarray
            Array (by lndu category) of land use prevalence long by time period
            

        Keyword Arguments
        ------------------
        attr_lndu : Union[AttributeTable, None]
            optional land use attribute table
        field_sfv_secondary : str
            field to use for standard secondary factor
        key_cmf : str
            key in curve group subdicts storing cmf
        key_params : str
            key in curve group subdicts storing parameter results
        npp_curve : Union[str, None]
            curve to use? If None, use self.npp_curve
        kwargs :
            passed to the following methods:
            * get_npp_frst_sequestration_factor_vectors()
            * get_npp_biomass_sequestration_curves()
        """
        
        ##  SOME INIT

        attr_lndu = (
            self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_lndu)
            if attr_lndu is None
            else attr_lndu
        )

        # stick to using the name
        npp_curve = self.npp_curve if (npp_curve is None) else npp_curve

        # 
        # get groups of sequestration factors and NPP curves
        (
            df_sf_groups,
            arr_frst_sf,
            _,
        ) = self.get_npp_frst_sequestration_factor_vectors(
            df_afolu_trajectories,
            field_ord_2 = field_sfv_secondary, 
            return_factors = True,
            **kwargs,
        )

        dict_curves = self.get_npp_biomass_sequestration_curves(
            df_sf_groups,
            key_cmf = key_cmf,
            key_params = key_params,
            **kwargs,
        )
        
        # get CH4 factors
        arr_frst_ef_methane = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_ef_ch4, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )
        arr_frst_ef_methane *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_ef_ch4,
            "area"
        )


        # get an array to collapse
        field_cmf = "cumulative_mass"
        field_tp = self.model_attributes.dim_time_period
        tps = df_afolu_trajectories[field_tp].to_numpy()
        n_tp = len(tps)

        # arrs to collapse: if the area of secondary forests declines, then assume that pre-existing forest was cut into first
        #    - arr_area_to_collapse:                areas converted in by year
        #    - arr_cumulative_mass_to_collapse:     cumulative mass by year for new forests; used to scale conversion factors 
        #    - arr_sequestration_to_collapse:       sequestration totals by year
        arr_area_to_collapse = np.zeros((n_tp, n_tp))
        arr_cumulative_mass_to_collapse = np.zeros((n_tp, n_tp))
        arr_sequestration_to_collapse = np.zeros((n_tp, n_tp))

        # initialize the tp, curve tuple and the curve
        tup_call = None 
        curve = self.curves_npp.get_curve(npp_curve, )


        ##  GET SEQUESTRATION BY YEAR
        
        sf_frst_secondary_init = arr_frst_sf[0, self.ind_frst_scnd]

        # ACCOUNTING VECTORS
        #    - vec_area_secondary_remaining_from_original:  stores the area of secondary forest that was initialized that remains after convesions occur 
        #    - vec_area_converted_from_total:               area in each time period of forest converted away from secondary forests
        #    - vec_area_converted_from_original:            area in each time period of forest converted away from secondary forests from original stock
        #    - vec_area_new_total:                          total area converted to new w/o removing losses 
        vec_area_secondary_remaining_from_original = np.ones(n_tp)*arr_lndu_areas[0, self.ind_lndu_fsts]
        vec_area_converted_from_total = np.zeros(n_tp)
        vec_area_new_total = np.zeros(n_tp)
        
        # 
        area_converted_from_total = 0
        area_converted_to_total = 0

        for (i, arr) in enumerate(arrs_lndu_conversion):

            tp = tps[i]
            tup = (tp, npp_curve)
            area_to, area_from, area_remaining = self.get_lndu_areas_to_from_remaining(arr, self.ind_lndu_fsts, )

            # this should always initialize the tuple since the first time period will have variables
            if tup in dict_curves.keys():

                tup_call = tup
                dict_cur = dict_curves.get(tup_call, )
                
                # ensure cmf and parameters are defined
                arr_cmf = dict_cur.get(key_cmf, )
                params = dict_cur.get(key_params, )
                if (params is None) | (arr_cmf is None):
                    continue 

                arr_cmf = self._get_frst_sequestration_from_npp_build_cmf_df(
                    df_afolu_trajectories,
                    arr_cmf,
                    field_cmf,
                    field_tp,
                    return_type = "array",
                )

            
            # safety check
            if tup_call is None:
                continue


            # get area to use original sequestration factor for; then remove land converted away for NEXT iteration
            vec_area_secondary_remaining_from_original[i] -= area_converted_from_total
            vec_area_converted_from_total[i] = area_from
            vec_area_new_total[i] = area_converted_to_total

            # update aggreate areas
            area_converted_from_total += area_from
            area_converted_to_total += area_to

            # get the domain and add to the sequestration total
            vec_t = tps[i:] - tps[i]
            arr_area_to_collapse[i:, i] = area_to
            """
            NOTE: This approach of iteratively pushing the cmf down (1 by 1) 
            assumes that time periods are implemented correctly as 
            t, t + 1, t + 2, etc. with no gaps

            NOTE: the value of arr_sequestration_to_collapse at time i is 0, so
                area_to doesn't apply until i + 1. This means that sequestration
                factors (and biomass conversion scalars) aren't affected until
                t = i + 1
            """
        
            arr_sequestration_to_collapse[i:, i] = curve(vec_t, *params.x)*area_to
            if i < n_tp - 1:
                arr_cumulative_mass_to_collapse[(i + 1):, i] = arr_cmf[0:(n_tp - i - 1), 1]*area_to


        # set area removed from new as any excess over "original" secondary
        vec_area_secondary_remaining_from_original = sf.vec_bounds(
            vec_area_secondary_remaining_from_original,
            (0, np.inf)
        )
        vec_area_converted_from_original = sf.vec_bounds(
            vec_area_converted_from_total,
            [(0, x) for x in vec_area_secondary_remaining_from_original]
        )
        vec_area_converted_from_new = vec_area_converted_from_total - vec_area_converted_from_original

        # new forest is total inflow, bounded above by area of forest
        vec_area_new_total_capped = sf.vec_bounds(
            vec_area_new_total,
            [(0, x) for x in arr_lndu_areas[:, self.ind_lndu_fsts]]
        )
        

        ##  CALCULATE OUTPUTS
        #   Assumes that conversions out of secondary hit older secondary forests first.
        #    - get secondary forest sequestration
        #    - get average biomass conversion scalar

        # get base sequestration
        arr_sequestration = arr_frst_areas*arr_frst_sf

        # total secondary is scaled by how much is actually near 
        vec_sequestration_secondary = arr_sequestration_to_collapse.sum(axis = 1)
        vec_sequestration_secondary *= np.nan_to_num(
            vec_area_new_total_capped/vec_area_new_total, 
            nan = 0.0, 
            posinf = 0.0, 
        )
        vec_sequestration_secondary += vec_area_secondary_remaining_from_original*sf_frst_secondary_init
        arr_sequestration[:, self.ind_frst_scnd] = vec_sequestration_secondary

        # get the scalar for land use conversion to apply
        vec_frac_area_converted_from_new = vec_area_converted_from_new/vec_area_converted_from_total
        vec_conversion_scalar = arr_cumulative_mass_to_collapse.sum(axis = 1)/vec_area_new_total # weighted average of 
        vec_conversion_scalar *= vec_frac_area_converted_from_new
        vec_conversion_scalar += 1 - vec_frac_area_converted_from_new
        vec_conversion_scalar = np.nan_to_num(vec_conversion_scalar, nan = 1.0, posinf = 1.0, )

        # get methane emissions
        arr_frst_ef_methane = self.get_frst_methane_factors(df_afolu_trajectories, )
        arr_frst_emission_methane = arr_frst_areas*arr_frst_ef_methane

        # this seris:
        #    1. normalizes the cumulative mass curve per unit new
        #    2. multiplies by total new area
        #    3. adds "1" * area of secondary remaining from original
        #    4. divides by total area of f
        vec_frst_prevalence_scalar = arr_cumulative_mass_to_collapse.sum(axis = 1)/vec_area_new_total
        vec_frst_prevalence_scalar *= vec_area_new_total_capped
        vec_frst_prevalence_scalar += vec_area_secondary_remaining_from_original
        vec_frst_prevalence_scalar /= arr_lndu_areas[:, self.ind_lndu_fsts]
        vec_frst_prevalence_scalar = np.nan_to_num(vec_frst_prevalence_scalar, nan = 1.0, posinf = 1.0, )
        
        vec_frst_ef_ch4_secondary = arr_frst_ef_methane[:, self.ind_frst_scnd]*vec_frst_prevalence_scalar
        arr_frst_emission_methane[:, self.ind_frst_scnd] = arr_frst_areas[:, self.ind_frst_scnd]*vec_frst_ef_ch4_secondary


        # convert to dataframes and return
        out = [
            self.model_attributes.array_to_df(
                -1*arr_sequestration, 
                self.modvar_frst_emissions_co2_sequestration
            ),
            self.model_attributes.array_to_df(
                arr_frst_emission_methane, 
                self.modvar_frst_emissions_ch4
            )
        ]

        out = (out, vec_conversion_scalar,)

        return out



    def _get_frst_sequestration_from_npp_build_cmf_df(self,
        df_afolu_trajectories: pd.DataFrame,
        cmf: np.ndarray, 
        field_cmf: str,
        field_tp: str,
        return_type: str = "dataframe",
    ) -> pd.DataFrame:
        """Support function for get_frst_sequestration_from_npp()
        
        Arguments
        ---------
        cmf : np.ndarray
            2d numpy array (column 1 -> time periods, column 2 -> values)
        return_type : str
            "dataframe" or "array"
        """
        # build the cmf data frame
        df_cmf = (
            pd.merge(
                df_afolu_trajectories[[field_tp]],
                pd.DataFrame(
                    cmf,
                    columns = [field_tp, field_cmf],
                ),
                how = "left",
            )
        )

        # fill in values
        sf.df_fillna_propagate_value(df_cmf, 0.0, field_cmf, forward = False,)
        sf.df_fillna_propagate_value(df_cmf, 1.0, field_cmf, forward = True,)
        df_cmf.interpolate(
            inplace = True,
            method = "linear",
        )

        if return_type == "array":
            df_cmf = df_cmf.to_numpy()

        return df_cmf
    


    def get_frst_sequestration_factors(self,
        df_afolu_trajectories: pd.DataFrame,
        modvar_area: Union[str, 'ModelVariable', None] = None,
        modvar_sequestration: Union[str, 'ModelVariable', None] = None,
        override_vector_for_single_mv_q: bool = True,
        **kwargs,
    ) -> Tuple:
        """Retrieve the sequestration factors for forest in terms of 
            modvar_sequestration units and modvar_area
        """
        # get area variable
        modvar_area = self.model_attributes.get_variable(modvar_area)
        if modvar_area is None:
            modvar_area = self.model_socioeconomic.modvar_gnrl_area 

        # get sequetration factor variable
        modvar_sequestration = self.model_attributes.get_variable(modvar_sequestration)
        if modvar_sequestration is None:
            modvar_sequestration = self.modvar_frst_sq_co2


        # get sequestration factors
        arr_frst_ef_sequestration = self.model_attributes.extract_model_variable(
            df_afolu_trajectories, 
            modvar_sequestration, 
            override_vector_for_single_mv_q = override_vector_for_single_mv_q, 
            return_type = "array_units_corrected",
            **kwargs,
        )

        arr_frst_ef_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_area,
            modvar_sequestration,
            "area"
        )

        return arr_frst_ef_sequestration
    


    def get_frst_sequestration_static(self,
        df_afolu_trajectories: pd.DataFrame,
        arr_area_frst: np.ndarray,
    ) -> List[pd.DataFrame]:
        """Retrieve static sequestration and ch4 emissions; used if NPP is not 
            specified. Returns a list with elements related to forests:

            [
                df_sequestration,
                df_emission_method,
            ]
        """
        # get different variables
        arr_frst_ef_sequestration = self.get_frst_sequestration_factors(df_afolu_trajectories, )
        arr_frst_ef_methane = self.get_frst_methane_factors(df_afolu_trajectories, )
    

        # build output variables
        out = [
            self.model_attributes.array_to_df(
                -1*arr_area_frst*arr_frst_ef_sequestration, 
                self.modvar_frst_emissions_co2_sequestration
            ),
            self.model_attributes.array_to_df(
                arr_area_frst*arr_frst_ef_methane, 
                self.modvar_frst_emissions_ch4
            )
        ]

        return out



    def get_lndu_class_bounds(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_lndu_area_init: np.ndarray,
        modvar_area_target: Union[str, 'ModelVariable', None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check constraints specified for land use classes. Ensures that 
            initial conditions do not conflict with constraints by adjusting any 
            specified constraints. Returns a tuple\
            
            (
                arr_constraints_inf,
                arr_constratins_sup
            )

            in terms of modvar_area_target

        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            DataFrame containing input trajectories
        vec_lndu_area_init : np.ndarray
            Initial land use prevalance vector in terms of 
            AFOLU.Socioeconomic.modvar_gnrl_area

        Keyword Arguments
        -----------------
        modvar_area_target : Union[str, ModelVariable, None]
            ModelVariable to use for output units. If not properly specified as 
            a varable name or model variable, defaults to
            AFOLU.model_socioeconomic.modvar_gnrl_area
        """
        modvar_area_target = self.model_attributes.get_variable(modvar_area_target)
        modvar_area_target = (
            self.model_socioeconomic.modvar_gnrl_area
            if modvar_area_target is None
            else modvar_area_target
        )

        # get the supremum
        arr_constraint_sup = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_constraint_area_max,
            return_type = "array_base",
        )

        arr_constraint_sup *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lndu_constraint_area_max, 
            modvar_area_target, 
            "area"
        )

        # get the infimum
        arr_constraint_inf = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_constraint_area_min,
            return_type = "array_base",
        )

        arr_constraint_inf *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lndu_constraint_area_min, 
            modvar_area_target, 
            "area"
        )

        
        # check the initial state of the maximum (must be the greater of the initial value and the max)
        arr_constraint_sup_first = [
            (
                max(arr_constraint_sup[0, i], vec_lndu_area_init[i]) 
                if arr_constraint_sup[0, i] != self.flag_ignore_constraint
                else self.flag_ignore_constraint
            )
            for i in range(len(vec_lndu_area_init))
        ]
        arr_constraint_sup[0] = np.array(arr_constraint_sup_first)

        # check the initial state
        arr_constraint_inf_first = [
            (
                min(arr_constraint_inf[0, i], vec_lndu_area_init[i]) 
                if arr_constraint_inf[0, i] != self.flag_ignore_constraint
                else self.flag_ignore_constraint
            )
            for i in range(len(vec_lndu_area_init))
        ]
        arr_constraint_inf[0] = np.array(arr_constraint_inf_first)

        # prep output and return
        tuple_out = (arr_constraint_inf, arr_constraint_sup)

        return tuple_out



    def get_lndu_scalar_max_out_states(self,
        scalar_in: Union[int, float],
        attribute_land_use: AttributeTable = None,
        cats_max_out: Union[List[str], None] = None,
        lmo_approach: Union[str, None] = None,
    ) -> np.ndarray:
        """Retrieve the vector of scalars to apply to land use based on an input 
            preliminary scalar and configuration parameters for "maxing out" 
            transition probabilities.

        Function Arguments
        ------------------
        scalar_in : Union[int, float]
            Input scalar

        Keyword Arguments
        -----------------
        attribute_land_use : AttributeTable
            AttributeTable for $CAT-LANDUSE$. If None, use self.model_attributes 
            default.
        cats_max_out : Union[List[str], None]
            Categories that are available to be maxed out. If None, defaults to 
            AFOLU.cats_lndu_max_out_transition_probs
        lmo_approach : Union[str, None]
            Approach to take for land use max out approach. If None, defaults to 
            configuration. Values are:

            * "decrease_only": apply "max_out" approach only if scalar < 1
            * "increase_only": apply "max_out" approach only if scalar > 1
            * "decrease_and_increase": apply "max_out" to categories in either 
                case
        """
        attribute_land_use = (
            self.model_attributes.get_attribute_table(self.subsec_name_lndu) 
            if (attribute_land_use is None) 
            else attribute_land_use
        )
        cats_max_out = (
            self.cats_lndu_max_out_transition_probs
            if not isinstance(cats_max_out, list)
            else [x for x in attribute_land_use.key_values if x in cats_max_out]
        )

        # check specification of land use max out approach
        valid_lmo = [
            "decrease_only",
            "decrease_and_increase",
            "increase_only"
        ]
        lmo_approach = (
            self.model_attributes.configuration.get("land_use_reallocation_max_out_directionality")
            if lmo_approach not in valid_lmo
            else lmo_approach
        )
        lmo_approach = (
            "decrease_only" 
            if (lmo_approach is None) 
            else lmo_approach
        )

        # proceed only if any of the following 3 conditions are true
        proceed_q = False
        proceed_q |= ((lmo_approach == "decrease_only") & (scalar_in < 1))
        proceed_q |= ((lmo_approach == "increase_only") & (scalar_in > 1))
        proceed_q |= (lmo_approach == "decrease_and_increase")

        out = (
            np.array([int(x in cats_max_out) for x in attribute_land_use.key_values]).astype(int)
            if proceed_q
            else np.zeros(attribute_land_use.n_key_values).astype(int)
        )

        return out
    


    def get_lndu_soil_soc_factors(self,
        df_afolu_trajectories: pd.DataFrame,
        arr_lndu_prevalence: np.ndarray,
        arr_agrc_crop_fractions: np.ndarray,
        dict_fracs_residues_removed_burned: Dict,
        default_fmg_grassland_not_pasture: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate soild soc factors and areas of land-use types under improved 
            management (in terms of configuration units).
        
        NOTE: Treats F_I and F_MG as independent; right now, F_I is scaled by 
            the fraction of residues that are not removed or burned, and it is 
            assumed that inputs can be used without no till.

        Returns a tuple of np.ndarrays

            arr_lndu_factor_soil_management, arr_lndu_area_improved

        Function Arguments
        ------------------
        - df_afolu_trajectories: data frame containing input AFOLU trajectories
        - arr_lndu_prevalence: array giving land use prevalence in terms of 
        - arr_agrc_crop_fractions: array giving crop areas as fraction of 
            cropland
        - dict_fracs_residues_removed_burned: dictionary with keys 
            self.modvar_agrc_frac_residues_burned and 
            self.modvar_agrc_frac_residues_removed

        Keyword Arguments
        -----------------
        - default_fmg_grassland_not_pasture: F_MG for grasslands not pasture. 
            Default is 1 (can be read in later)
        """
        # lower bound for F_MG
        arr_lndu_factor_soil_management_inf = self.model_attributes.extract_model_variable(# 
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_management_infinum,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # upper bound for F_MG
        arr_lndu_factor_soil_management_sup = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_management_supremum,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # upper bound for F_I, or input component w/o manure
        arr_lndu_factor_soil_inputs_no_manure_sup = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_inputs_supremum_no_manure,
            all_cats_missing_val = 1.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # fraction of croplands using no-till
        arr_agrc_frac_improved_by_crop = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_frac_no_till,
            all_cats_missing_val = 0.0,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # fraction of pastures that are improved
        vec_lndu_frac_pastures_improved = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_frac_pastures_improved,
            return_type = "array_base",
            expand_to_all_cats = False,
        )


        ##  1. START WITH CROPLAND FACTORS

        arr_agrc_crop_area = sf.do_array_mult(
            arr_agrc_crop_fractions,
            arr_lndu_prevalence[:, self.ind_lndu_crop]
        )

        # start by calculating average management factor, based on the fraction using no-till (sup) and unimproved (inf) 
        arr_agrc_crop_area_improved = arr_agrc_crop_area*arr_agrc_frac_improved_by_crop
        vec_lndu_frac_cropland_improved = (arr_agrc_crop_area_improved).sum(axis = 1)
        vec_lndu_frac_cropland_improved /= arr_lndu_prevalence[:, self.ind_lndu_crop] # should be a denominator of 1, but just to be safe

        # get the cropland management factor
        vec_lndu_fmg_cropland = vec_lndu_frac_cropland_improved*arr_lndu_factor_soil_management_sup[:, self.ind_lndu_crop]
        vec_lndu_fmg_cropland += (1 - vec_lndu_frac_cropland_improved)*arr_lndu_factor_soil_management_inf[:, self.ind_lndu_crop]

        # get fraction removed/burned
        vec_agrc_frac_residue_burned = dict_fracs_residues_removed_burned.get(self.modvar_agrc_frac_residues_burned).flatten()
        vec_agrc_frac_residue_removed = dict_fracs_residues_removed_burned.get(self.modvar_agrc_frac_residues_removed).flatten()
        vec_agrc_frac_residue_inputs = 1 - vec_agrc_frac_residue_burned - vec_agrc_frac_residue_removed
        
        # calculate the input componen
        vec_agrc_inputs_component = sf.vec_bounds(arr_lndu_factor_soil_inputs_no_manure_sup[:, self.ind_lndu_crop] - 1, (0, 1))
        vec_agrc_inputs_component *= vec_agrc_frac_residue_inputs
        vec_agrc_inputs_component += 1.0

        # assume that inputs can be provided independently of 
        vec_lndu_fmg_cropland *= vec_agrc_inputs_component


        ##  2. GET PASTURE (AND GRASSLAND) FACTOR

        # get pasture
        vec_lndu_fmg_pasture = (1 - vec_lndu_frac_pastures_improved)*arr_lndu_factor_soil_management_inf[:, self.ind_lndu_pstr]
        vec_lndu_fmg_pasture += vec_lndu_frac_pastures_improved*arr_lndu_factor_soil_management_sup[:, self.ind_lndu_pstr]
        vec_lndu_area_pasture_improved = arr_lndu_prevalence[:, self.ind_lndu_pstr]*vec_lndu_frac_pastures_improved

        # update output array
        arr_lndu_factor_soil_management = np.ones(arr_lndu_factor_soil_management_sup.shape)
        arr_lndu_factor_soil_management[:, self.ind_lndu_crop] = vec_lndu_fmg_cropland
        arr_lndu_factor_soil_management[:, self.ind_lndu_grss] = default_fmg_grassland_not_pasture
        arr_lndu_factor_soil_management[:, self.ind_lndu_pstr] = vec_lndu_fmg_pasture

        # build output areas
        arr_lndu_area_improved = np.ones(arr_lndu_factor_soil_management_sup.shape)
        arr_lndu_area_improved[:, self.ind_lndu_crop] = arr_agrc_crop_area_improved.sum(axis = 1)
        arr_lndu_area_improved[:, self.ind_lndu_pstr] = vec_lndu_area_pasture_improved

        out = (arr_lndu_factor_soil_management, arr_lndu_area_improved)

        return out



    def get_lvst_area_required(self,
        vec_lvst_annual_dry_matter_consumption: np.ndarray, # arr_lvst_annual_dry_matter_consumption_per_capita[i]
        vec_lvst_production_needed: np.ndarray, # arr_lvst_domestic_production_unadj[i]
        factor_lndu_dry_matter_production: float, #factor_lndu_init_avg_consumption_pstr * vec_lvst_carry_capacity_scale[i]
    ) -> np.ndarray:
        """
        Get the number of lvst that can be support

        NOTE: DOES NOT PERFORM UNITS CORRECTION, ASSUMES UNITS ARE PROPERLY 
            FORMATTED
        
        Function Arguments
        ------------------ 
        - area_pstr_proj: projected area of pasture
        - vec_lvst_annual_dry_matter_consumption: vector (by animal) of 
            consumption of dry matter per animal
        - vec_lvst_annual_production_unadj: vector (by animal) of expected 
            production demand by animal type
        - factor_lndu_dry_matter_production: dry matter production factor for 
            pastures
        """

        ##  AREA OF LAND NEEDED 
        #
        area_total_land_needed = (vec_lvst_annual_dry_matter_consumption*vec_lvst_production_needed).sum()
        area_total_land_needed /= factor_lndu_dry_matter_production

        return area_total_land_needed
    


    def get_lvst_dict_lsmm_categories_to_lvst_fraction_variables(self,
    ) -> Dict:
        """
        Return a dictionary with Livestock Manure Management categories as keys 
            based on the Livestock attribute table:

            {
                cat_lsmm: {
                    "mm_fraction": VARNAME_MANURE_MANAGEMENT, 
                    ...
                }
            }

            for each key, the dict includes variables associated with the 
                livestock manure management cat_lsmm: 

            - "mm_fraction"
        """
        subsec_lvst = self.model_attributes.subsec_name_lvst
        subsec_lsmm = self.model_attributes.subsec_name_lsmm
        pycat_lsmm = self.model_attributes.get_subsector_attribute(
            subsec_lsmm, 
            "pycategory_primary_element"
        )

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            subsec_lvst,
            pycat_lsmm,
            {
                "Livestock Manure Management Fraction": "mm_fraction"
            },
        )

        return dict_out



    def get_lvst_pasture_max_yield_and_carrying_capacity(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_lndu_area_initial: np.ndarray,
    ) -> np.ndarray:
        """
        Get initial maximum feasible yield and carrying capacity vector. Ensures 
            that the specified maximum yield doesn't conflict with assumptions 
            from exogenous specification of livestock population, dry matter 
            consumption, and consumption per head of livestock.

            out = (
                arr_lvst_annual_dry_matter_consumption, # annual dry matter consumption per head of lvst
                vec_max_yf, # maximum feasible yield factor for pastures in mass/area
                vec_carrying_capacity_scalar, # maximum feasible scalar applied to pasture yields
                factor_lndu_init_avg_consumption_pstr, # initial average estimate of consumption mass per area
            )

        Returns consumption in terms of 

            MASS: self.modvar_lndu_yf_pasture_sup
            AREA: self.model_socioeconomic.modvar_gnrl_area

        
        Function Arguments
        ------------------
        - df_afolu_trajectories: data frame containing AFOLU trajectory inputs
            to use in modeling
        - vec_lndu_area_initial: n x 1, where n is number of land use 
            categories. Vector giving initial areas of each land use category. 
            Area is in terms of input area of region.

        NOTE
        ----
        - The maximum feasible carrying capacity is defined using the pasture 
            "Maximum Pasture Dry Matter Yield Factor" in combination with the 
            pasture area. If the initial consumption by livestock implied by 
            total heads and dry matter consumption per head exceeds this, the 
            threshold is changed to reflect the initial state.
        """


        ##  GET SPECIFIED MAXIMUM YIELD
        
        # get initial pasture area and maximum feasible production
        area_lndu_pasture_init = vec_lndu_area_initial[self.ind_lndu_pstr]

        # get upper bound of annual dry matter production and convert area
        vec_lndu_yf_pasture_sup = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_yf_pasture_sup,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        vec_lndu_yf_pasture_sup /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lndu_yf_pasture_sup, 
            self.model_socioeconomic.modvar_gnrl_area, 
            "area"
        )

        # total yield supremum at time 0
        total_yield_lndu_pasture_sup = area_lndu_pasture_init*vec_lndu_yf_pasture_sup[0]


        ##  CALCULATE TOTAL DRY MATTER REQUIRED FOR INITIAL POPULATION
        
        dpy = self.model_attributes.configuration.get("days_per_year")

        vec_lvst_production_init = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_pop_init,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )[0]

        arr_lvst_annual_dry_matter_consumption_per_capita = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_dry_matter_consumption,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        scalar_lndu_ddm_to_yf = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_dry_matter_consumption,
            self.modvar_lndu_yf_pasture_sup,
            "mass"
        )

        arr_lvst_annual_dry_matter_consumption_per_capita *= dpy*scalar_lndu_ddm_to_yf

        # set annual total required by livestock category in terms of yield supremum
        lvst_total_dry_matter_required = vec_lvst_production_init*arr_lvst_annual_dry_matter_consumption_per_capita[0]
        lvst_total_dry_matter_required = lvst_total_dry_matter_required.sum()


        ##  GET ADJUSTED MAXIMUM FEASIBLE YIELD AND CARRYING CAPACITY SCALAR
        
        factor_lndu_init_avg_consumption_pstr = lvst_total_dry_matter_required/area_lndu_pasture_init

        # total feasible maximum yield
        vec_lndu_yf_pasture_sup_adj = sf.vec_bounds(
            vec_lndu_yf_pasture_sup, 
            (factor_lndu_init_avg_consumption_pstr, np.inf), 
        )

        # carrying capacity is capped at ratio from output to implicit bound
        vec_lvst_carry_capacity_scale = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_carrying_capacity_scalar,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )
        vec_lvst_carry_capacity_scale = np.nan_to_num(
            vec_lvst_carry_capacity_scale/vec_lvst_carry_capacity_scale[0],
            nan = 1.0,
            posinf = 1.0,
        )
        
        vec_lvst_carry_capacity_scale = sf.vec_bounds(
            vec_lvst_carry_capacity_scale,
            [(0.0, x) for x in vec_lndu_yf_pasture_sup_adj/factor_lndu_init_avg_consumption_pstr]
        )

        # output tuple
        out = (
            arr_lvst_annual_dry_matter_consumption_per_capita,
            vec_lndu_yf_pasture_sup_adj,
            vec_lvst_carry_capacity_scale,
            factor_lndu_init_avg_consumption_pstr,
        )
        
        return out



    def get_lvst_production_supported(self,
        area_pstr_proj: float, 
        vec_lvst_annual_dry_matter_consumption : np.ndarray, # arr_lvst_annual_dry_matter_consumption_per_capita[i]
        vec_lvst_annual_production_unadj: np.ndarray, # arr_lvst_domestic_production_unadj[i]
        factor_lndu_init_avg_consumption_pstr: float,
        scalar_lvst_cc: float # vec_lvst_carry_capacity_scale[i]s
    ) -> np.ndarray:
        """
        Get the number of lvst that can be support

        NOTE: DOES NOT PERFORM UNITS CORRECTION, ASSUMES UNITS ARE PROPERLY 
            FORMATTED
        
        Function Arguments
        ------------------
        - area_pstr_proj: projected area of pasture
        - vec_lvst_annual_dry_matter_consumption: vector (by animal) of 
            consumption of dry matter per animal
        - vec_lvst_annual_production_unadj: vector (by animal) of expected 
            production demand by animal type
        - factor_lndu_init_avg_consumption_pstr: initial estimated total average 
            production factor for pastures across all animal types
        - scalar_lvst_cc: scalar to apply to 
        """
        # average yield per ha of pasture
        # - NOTE: comes from output of self.get_lvst_pasture_max_yield_and_carrying_capacity, so it's already controlled
        vec_lndu_total_yield_pstr_unadj = factor_lndu_init_avg_consumption_pstr*scalar_lvst_cc
        vec_lndu_total_yield_pstr_unadj *= area_pstr_proj


        ##  NUMBER OF LIVESTOCK THAT CAN BE SUPPORTED
        #
        vec_lndu_total_yield_needed = (vec_lvst_annual_dry_matter_consumption*vec_lvst_annual_production_unadj).sum()
        vec_lvst_prod_supportable = vec_lndu_total_yield_pstr_unadj/vec_lndu_total_yield_needed
        vec_lvst_prod_supportable *= vec_lvst_annual_production_unadj

        #
        w = np.where(vec_lvst_annual_dry_matter_consumption == 0)[0]
        np.put(vec_lvst_prod_supportable, w, vec_lvst_annual_production_unadj[w])


        return vec_lvst_prod_supportable



    def get_markov_matrices(self,
        df_ordered_trajectories: pd.DataFrame,
        correct_emission_units: bool = True,
        n_tp: Union[int, None] = None,
        return_c_stock_conversion_factors: bool = True,
        target_units_area_modvar: Union[str, 'ModelVariable', None] = None,
        thresh_correct: float = 0.0001,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get the transition and emission factors matrices from the data frame 
            df_ordered_trajectories. Assumes that the input data frame is 
            ordered by time_period

        Function Arguments
        ------------------
        df_ordered_trajectories : DataFrame
            Input DataFrame containing columnar variables for conversion 
            transition probabilities and emission factors.

        Keyword Arguments
        -----------------
        correct_emission_units : bool
            Correct emissions to configuration units?
        return_c_stock_conversion_factors : bool
            Return C stock converion factors as well?
            * False:    Returns ONLY the array of transition probabilities (not
                        as a tuple, but as an individual array)
            * True:     Returns a tuple of the form:

                        (
                            arrs_pr,        # array of transition probabilities
                            arrs_c_agb,     # array of C stock change from 
                                            # i -> j for above-ground biomass
                            arrs_c_bgb,     # array of C stock change from 
                                            # i -> j for below-ground biomass
                            arr_c_lndu_agb, # array of c above ground biomass
                        )

        target_units_area_modvar : Union[str, 'ModelVariable', None]
            modvar that will drive emissions, used to 
            identify target area units to which conversion emission factors are 
            applied. 
            * If None, no adjustment is made
        n_tp : Union[int, None]
            The number of time periods. Default value is None, which implies all 
            time periods
        thresh_correct : float 
            Used to decide whether or not to correct the transition matrix 
            (assumed to be row stochastic) to sum to 1; if the absolute value of 
            the sum is outside this range, an error will be thrown

        Notes
        -----
        fields_pij and fields_efc will be properly ordered by categories for 
            this transformation
        """
        
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        n_cats = attr_lndu.n_key_values
        n_tp = n_tp if sf.isnumber(n_tp, integer = True) else self.n_time_periods

        # get fields that are needed
        dict_mv_to_fields = self.model_attributes.dict_model_variables_to_variable_fields
        fields_agb = dict_mv_to_fields.get(self.modvar_lndu_c_stock_factor_abg, )
        fields_bgb = dict_mv_to_fields.get(self.modvar_lndu_c_stock_bgb_ratio, )
        fields_pij = dict_mv_to_fields.get(self.modvar_lndu_prob_transition, )
        
        # return None?
        return_none = fields_pij is None
        return_none |= (((fields_agb is None) | (fields_bgb is None)) & return_c_stock_conversion_factors)
        if return_none:
            return None

        # verify fields are present
        fields_check = (
            fields_pij + fields_agb + fields_bgb
            if return_c_stock_conversion_factors
            else fields_pij
        )
        sf.check_fields(df_ordered_trajectories, fields_check)


        ##  GET TRANSITION PROBABILITIES

        # if not return C factors, return on the transition probabilities
        arr_pr = np.array(df_ordered_trajectories[fields_pij])
        arr_pr = arr_pr.reshape((n_tp, n_cats, n_cats))
        if not return_c_stock_conversion_factors:
            return arr_pr


        ##  GET C CONVERSION FACTORS

        arrs_c_agb, arrs_c_bgb, arr_c_lndu_agb = self.build_c_conversion_matrices(
            df_ordered_trajectories,
            attr_lndu = attr_lndu,
        )

        # convert units to output units
        if correct_emission_units:
            scalar = self.model_attributes.get_scalar(
                self.modvar_lndu_c_stock_factor_abg,
                "mass"
            )

            arrs_c_agb *= scalar
            arrs_c_bgb *= scalar
            arr_c_lndu_agb *= scalar

        # ensure area is accounted for
        if self.model_attributes.get_variable(target_units_area_modvar) is not None:
            scalar = self.model_attributes.get_variable_unit_conversion_factor(
                target_units_area_modvar,
                self.modvar_lndu_c_stock_factor_abg,
                "area"
            )

            arrs_c_agb *= scalar
            arrs_c_bgb *= scalar
            arr_c_lndu_agb *= scalar
    
        # build output tuple
        out = (
            arr_pr,
            arrs_c_agb,
            arrs_c_bgb,
            arr_c_lndu_agb,
        )

        return out
    


    def get_matrix_column_scalar(self,
        mat_column: np.ndarray,
        target_scalar: np.ndarray,
        vec_x: np.ndarray,
        eps: float = 0.000001,
        mask_max_out_states: np.ndarray = None,
        max_iter: int = 100,
    ) -> float:

        """
        Return a vector of scalars that need to be applied to acheieve a 
            specified scaled change in state, i.e. finds the true matrix column 
            scalar required to achieve x(1)_i -> x(0)_i*target_scalar. If any 
            transition probabilities are 1 w/scalar > 1 (or 0 with scalar < 0), 
            the scalar becomes inadequate; iterative function will search for 
            adequate column scalar.

        Function Arguments
        ------------------
        - mat_column: column in transition matrix (assuming row-stochastic) 
            representing probabilities of entry into a state
        - target_scalar: the target change in output area to achieve
        - vec_x: current state of areas. Target area is sum(vec_x)*target_scalar

        Keyword Arguments
        -----------------
        - eps: area convergence tolerance.
        - mask_max_out_states: np.array of same length as mat_column that 
            contains a mask for states that should be "maxed out" (sent to 0 
            or 1) first. The default is for all states to be scalable.
            * For example, to ensure that masses of area are first shifted 
                to/from state 1 out of states 0, 1, 2, 3, 4, 5, the vector 
                should be

                mask_max_out_states = np.array([0, 1, 0, 0, 0])

            * Default is None. If None, all states are scalable states.

        - max_iter: maximum number of iterations. Default is 100.


        Notes and Expanded Description
        ------------------------------

        For x(0), x(1) \in \mathbb{R}^(1 x n) and a row-stochastic transition 
            matrix Q in \mathbb{R}^{n x n}, the value

            x(1) = x(0)Q

        transforms the state vector x(0) -> x(1). In some cases, it is desirable 
            to ensure that x(1)_i = \alpha*x(0)_i, i.e., that a transition 
            matrix enforces an observed growth from period 0 -> 1. This can 
            generally be acheived in AFOLU using the 
            AFOLU.adjust_transition_matrix() method.

        However, some transition matrices may have 0s or 1s in column entries, 
            meaning that specified input scalar may be unable to acheieve a 
            desired scalar change in input area, so other columnar entries may 
            have to be scaled more to acheive the desired outcome of 
            x(1)_i = \alpha*x(0)_i.

        """
        # check input
        if not sf.islistlike(mask_max_out_states):
            # LOG HERE - INVALID STATE
            mask_max_out_states = None
        elif len(mask_max_out_states) != len(mask_max_out_states):
            mask_max_out_states = None
        else:
            mask_max_out_states = sf.vec_bounds(np.round(mask_max_out_states).astype(int), (0, 1))

        if mask_max_out_states is None:
            mask_max_out_states = np.zeros(len(mat_column))


        ##  START BY GETTING AREA THAT CAN BE MAXED OUT FROM INPUT CLASSES

        # true area, target area, and area incoming from
        area = np.dot(vec_x, mat_column)
        area_target = target_scalar*area
        area_incoming_from_max_out_states_at_full = np.sum(mask_max_out_states*vec_x)
        area_incoming_from_max_out_states_at_current = np.sum(mask_max_out_states*vec_x*mat_column)
        
        delta_area  = area_target - area    
        scalar_adj = (1 - mask_max_out_states)
    
        if np.sum(mask_max_out_states) > 0:
            # if the max-out states can't absorb the required change, then max them out and allow the rest to continue
            max_scale = (
                area_incoming_from_max_out_states_at_full/area_incoming_from_max_out_states_at_current
                if area_incoming_from_max_out_states_at_current != 0
                else 1.0
            )
            
            scale_max_out_states = np.nan_to_num(
                delta_area/area_incoming_from_max_out_states_at_current, 
                nan = 0.0, 
                posinf = 0.0,
            )
            scale_max_out_states = min(max(1 + scale_max_out_states, 0), max_scale)
            scalar_adj += mask_max_out_states*scale_max_out_states


        q_j = sf.vec_bounds(mat_column*scalar_adj, (0, 1))
        area_target = target_scalar*np.dot(vec_x, mat_column)

        # index of probabilities that are scalable
        w_scalable = np.where((q_j < 1) & (q_j > 0))[0]
        w_unscalable = np.where((q_j == 1) | (q_j == 0))[0]
        i = 0
        n = len(vec_x)
        n_iter = max_iter#*2 if (sum(mask_max_out_states) < len(mat_column)) else max_iter

        # prevent iterating if every entry is a 1/0, double number of iterations to allow to turn off mask halfway through (after max_iter) if still non-convergent
        while (np.abs(area/area_target - 1) > eps) and (len(w_unscalable) < n) and (i < n_iter):
            # scalar is capped at one, so we are effectively applying this to only scalable indices
            area = np.dot(vec_x, q_j)
            area_unscalable = np.dot(vec_x[w_unscalable], q_j[w_unscalable])
            scalar_cur = np.nan_to_num(
                (area_target - area_unscalable)/(area - area_unscalable), 
                nan = 1.0, 
                posinf = 1.0,
            )
            if (scalar_cur < 0):
                break
            scalar_adj *= scalar_cur

            # recalculate vector + implied area
            q_j = sf.vec_bounds(mat_column*scalar_adj, (0, 1))

            w_scalable = np.where((q_j < 1) & (q_j > 0))[0]
            w_unscalable = np.where((q_j == 1) | (q_j == 0))[0]
            i += 1

        return scalar_adj



    def get_mineral_soc_change_matrices(self,
        arr_agrc_crop_area: np.ndarray,
        arr_lndu_area: np.ndarray,
        arr_lndu_factor_soil_carbon: np.ndarray,
        arr_lndu_factor_soil_management: np.ndarray,
        arr_lndu_frac_mineral_soils: np.ndarray,
        arr_soil_ef1_organic: np.ndarray,
        arr_soil_soc_stock: np.ndarray,
        vec_soil_area_crop_pasture: np.ndarray,
        dict_soil_fracs_to_use_agrc: Dict[str, np.ndarray],
        dict_soil_fracs_to_use_frst: Dict[str, np.ndarray],
        dict_soil_fracs_to_use_lndu: Dict[str, np.ndarray]
    ) -> tuple:
        """Retrieve matrices of delta SOC (- means loss of SOC, + indicates 
            sequestration) for conversion from class i to class j in each time 
            period, as well as a land-area weighted average EF1 for each time 
            period. Note that all areas should be in the same units.

        Returns: tuple of form (arrs_delta_soc, vec_avg_ef1)

        Function Arguments
        ------------------
        arr_agrc_crop_area : np.ndarray 
            `array giving crop areas
        arr_lndu_area : np.ndarray
            Array of land use areas (ordered by attr_lndu.key_values)
        arr_lndu_factor_soil_carbon : np.ndarray
            Aarray of F_LU SOC factors by land use type
        arr_lndu_factor_soil_management : np.ndarray
            Array of F_MG SOC adjustment factors by land use type
        arr_lndu_frac_mineral_soils : np.ndarray
            Array giving the fraction of soils that are mineral by each land use 
            category
        arr_soil_ef1_organic : np.ndarray
            N2O Organic EF1 by soil type
        arr_soil_soc_stock : np.ndarray
            Array giving soil carbon content of soil types (30cm) for each time 
            period (n_tp x attr_soil.n_key_values)
        vec_soil_area_crop_pasture : np.ndarray
            Area of crop and pasture by time period
        dict_soil_fracs_to_use_agrc : Dict[str, np.ndarray]
            Dictionary mapping soil fraction model variable string to array 
            (wide by crop categories)
        dict_soil_fracs_to_use_frst : Dict[str, np.ndarray]
            Dictionary mapping soil fraction model variable string to fraction 
            array (wide by forest categories)
        dict_soil_fracs_to_use_lndu : Dict[str, np.ndarray]
            Dictionary mapping soil fraction model variable string to fraction 
            array (wide by land use categories)
        """

        # get attribute tables
        attr_agrc = self.model_attributes.get_attribute_table(self.subsec_name_agrc, )
        attr_frst = self.model_attributes.get_attribute_table(self.subsec_name_frst, )
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu, )
        attr_soil = self.model_attributes.get_attribute_table(self.subsec_name_soil, )

        # initialize SOC transition arrays
        n_tp = len(arr_agrc_crop_area)
        arrs_delta_soc_source = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_delta_soc_target = arrs_delta_soc_source.copy()
        
        # initialize dictionary for building matrix lists out and vector of average ef1 soild
        dict_lndu_avg_soc_vecs = {}
        vec_soil_ef1_soc_est = 0.0


        ##  GET AVERAGE SOC CONTENT IN CROPLANDS

        vec_agrc_area = np.sum(arr_agrc_crop_area, axis = 1)
        vec_agrc_avg_soc = 0.0
        
        for modvar in dict_soil_fracs_to_use_agrc.keys():

            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            #
            vec_agrc_avg_soc_cur = np.sum(arr_agrc_crop_area*dict_soil_fracs_to_use_agrc[modvar], axis = 1)
            vec_soil_ef1_soc_est += vec_agrc_avg_soc_cur*arr_soil_ef1_organic[:, ind_soil]/vec_soil_area_crop_pasture

            vec_agrc_avg_soc_cur = np.nan_to_num(vec_agrc_avg_soc_cur/vec_agrc_area, nan = 0.0, posinf = 0.0, )
            vec_agrc_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, self.ind_lndu_crop]*arr_lndu_factor_soil_management[:, self.ind_lndu_crop]
            vec_agrc_avg_soc += vec_agrc_avg_soc_cur*arr_lndu_frac_mineral_soils[:, self.ind_lndu_crop]
            
        dict_lndu_avg_soc_vecs.update({self.ind_lndu_crop: vec_agrc_avg_soc})


        ##  GET AVERAGE SOC CONTENT IN FORESTS

        arr_frst_avg_soc = 0.0
        dict_lndu_forest_to_frst_forest = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_lndu,
            attr_frst.key,
            clean_attribute_schema_q = True,
            return_type = dict,
            skip_none_q = True
        )
        inds_frst = np.array([(attr_lndu.get_key_value_index(x), attr_frst.get_key_value_index(dict_lndu_forest_to_frst_forest[x])) for x in dict_lndu_forest_to_frst_forest.keys()])
        inds_lndu = inds_frst[:, 0]
        inds_frst = inds_frst[:, 1]


        for modvar in dict_soil_fracs_to_use_frst.keys():
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            #
            """
            arr_frst_avg_soc_cur = arr_lndu_area[:, inds_lndu]*dict_soil_fracs_to_use_frst[modvar][:, inds_frst]
            arr_frst_avg_soc_cur = (
                np.nan_to_num(
                    arr_frst_avg_soc_cur/arr_lndu_area[:, inds_lndu], 
                    nan = 0.0, 
                    posinf = 0.0,
                )
                .transpose()
            )
            """;
            arr_frst_avg_soc_cur = (
                dict_soil_fracs_to_use_frst[modvar][:, inds_frst]
                .copy()
                .transpose()
            )

            arr_frst_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, inds_lndu].transpose()
            arr_frst_avg_soc += arr_frst_avg_soc_cur.transpose()*arr_lndu_frac_mineral_soils[:, inds_lndu]

        for i in enumerate(inds_lndu):
            i, ind = i
            dict_lndu_avg_soc_vecs.update({ind: arr_frst_avg_soc[:, i]})


        ##  GET AVERAGE SOC CONTENT IN ADDITIONAL LAND USE CATEGORIES HEREHEREHERE

        arr_lndu_avg_soc = 0.0
        cats_lndu = list(set(sum([self.model_attributes.get_variable_categories(x) for x in dict_soil_fracs_to_use_lndu.keys()], [])))
        cats_lndu = [x for x in attr_lndu.key_values if x in cats_lndu]
        inds_lndu = sorted([attr_lndu.get_key_value_index(x) for x in cats_lndu])
        w_pstr = np.where(np.array(inds_lndu) == self.ind_lndu_pstr)[0]

        for modvar, arr_frac in dict_soil_fracs_to_use_lndu.items():
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, attr_soil.key))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            # 
            arr_lndu_avg_soc_cur = arr_lndu_area[:, inds_lndu]*arr_frac[:, inds_lndu]
            vec_soil_ef1_soc_est += (
                arr_lndu_avg_soc_cur[:, w_pstr[0]]*arr_soil_ef1_organic[:, ind_soil]/vec_soil_area_crop_pasture
                if len(w_pstr) > 0
                else 0.0
            )

            # get average SOC for the curent soil type
            """
            arr_lndu_avg_soc_cur = (
                np.nan_to_num(
                    arr_lndu_avg_soc_cur/arr_lndu_area[:, inds_lndu], 
                    nan = 0.0, 
                    posinf = 0.0,
                )
                .transpose()
            )
            """
            arr_lndu_avg_soc_cur = (
                arr_frac[:, inds_lndu]
                .copy()
                .transpose()
            )
            arr_lndu_avg_soc_cur *= arr_soil_soc_stock[:, ind_soil]*arr_lndu_factor_soil_carbon[:, inds_lndu].transpose()
            arr_lndu_avg_soc_cur *= arr_lndu_factor_soil_management[:, inds_lndu].transpose()

            # add to weighted average
            arr_lndu_avg_soc += arr_lndu_avg_soc_cur.transpose()*arr_lndu_frac_mineral_soils[:, inds_lndu]

        for i, ind in enumerate(inds_lndu):
            dict_lndu_avg_soc_vecs.update({ind: arr_lndu_avg_soc[:, i]})

        
        ##  APPLY AN AVERAGE TO MISSING LAND USE TYPES

        # calculate an average to apply to other land use classes
        vec_lngu_avg_soc = (arr_lndu_avg_soc * arr_lndu_area[:, inds_lndu]).sum(axis = 1, )
        vec_lngu_avg_soc /= arr_lndu_area[:, inds_lndu].sum(axis = 1, )
        vec_lngu_avg_soc = np.nan_to_num(vec_lngu_avg_soc, nan = 0.0, posinf = 0.0, )

        # add a missing
        inds_missing = [x for x in range(attr_lndu.n_key_values) if x not in dict_lndu_avg_soc_vecs.keys()]
        for ind in inds_missing:
            dict_lndu_avg_soc_vecs.update({ind: vec_lngu_avg_soc, })
        

        ##  UPDATE SOURCE/TARGET ARRAYS USING AVERAGE SOC

        for k, v in dict_lndu_avg_soc_vecs.items():
            # ensure we're not overwriting dict
            vec = v.copy()
            w = np.where(vec != 0)[0]
            vec = np.interp(range(len(vec)), w, vec[w]) if (len(w) > 0) else vec

            for i in range(n_tp):
                arrs_delta_soc_source[i, k, :] = vec[i]
                arrs_delta_soc_target[i, :, k] = vec[i]

        arrs_delta = arrs_delta_soc_target - arrs_delta_soc_source

        return arrs_delta, vec_soil_ef1_soc_est
    


    def get_npp_biomass_sequestration_curves(self,
        df_ordered_sequestration: pd.DataFrame,
        curve_npp: Union[str, npp.NPPCurve, None] = None,
        field_ord_1: str = "young",
        field_ord_2: str = "secondary",
        field_ord_3: str = "primary",
        force_convergence: bool = False,
        key_cmf: str = "cmf",
        key_params: str = "params",
        include_primary_forest_in_npp: Union[bool, None] = None,
        maxiter: int = 500,
        method: str = "SLSQP",
        params_init: Union[np.ndarray, None] = None,
        vec_widths: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> Dict[int, 'scipy.optimize._optimize.OptimizeResult']:
        """Get sets of sequestration curves as specified in input trajectories.
            Reduces the number of times that optimization is called to only 
            deal with unique sets of 

            (young, secondary, primary, )

            sequestration factors. 

        Notes
        -----
        Can be called from any sector--in agrc (woody perennial), exclude
            primary forest. 


        Arguments
        ---------
        df_ordered_sequestration : pd.DataFrame
            data frame containing ordered sequestration by time period group   
            (output of, e.g., get_npp_frst_sequestration_factor_vectors())

        Keyword Arguments
        -----------------
        curve_npp : Union[str, npp.NPPCurve, None]
            Curve to use for NPP. Valid string options are:
                - "gamma"
                - "sem"
        field_ord_i : str
            ordered field name to use as sequestration factors
        force_convergence : bool
            force parameter a in SEM to converge to final value? Only applies
            if curve_npp is "sem"
        key_cmf : str
            Key in subdictionaries storing cmf
        key_params : str
            Key in subdictionaries storing parameters
        include_primary_forest_in_npp : bool
            Include primary forest in the NPP curve fitting? If None, defaults 
            to self.npp_include_primary_forest
        maxiter : int
            maximum number of iterations of solver
        method : str
            valid solver for scipy.optimize.minimize
        params_init : Union[np.ndarray, None]
            Initial parameters to perturb for sequestration curve shape
        vec_widths : Union[np.ndarray, None]
            Vector of widths to use for integral estimate in NPP curve
        """

        ##  INITIALIZATION

        curve_npp = self.npp_curve if curve_npp is None else curve_npp
        curve_npp = self.curves_npp.get_curve(curve_npp, )
        if curve_npp is None:
            return None

        # initialize info for NPPCurves input
        vec_widths = (
            self.npp_integration_windows.copy()
            if vec_widths is None
            else vec_widths
        )

        # include primary forest in curve fitting? 
        # reduce the targets/widths if not including it
        include_primary_forest_in_npp = (
            self.npp_include_primary_forest
            if not isinstance(include_primary_forest_in_npp, bool)
            else include_primary_forest_in_npp
        )

        fields_targets = [field_ord_1, field_ord_2, field_ord_3]
        if not include_primary_forest_in_npp:
            fields_targets = fields_targets[0:2]
            vec_widths = vec_widths[0:2]

        dict_out = {}


        ##  ITERATE OVER EACH GROUP TO BUILD PARAMETERS
        
        for i, row in df_ordered_sequestration.iterrows():

            # get the factors and the parameter curve
            vec_factors = row[fields_targets].to_numpy()
            tol = self.get_npp_convergence_tolerance(row, **kwargs)

            # update internal curves and get parameters
            inputs = list(zip(vec_factors, vec_widths, ))
            self.curves_npp._initialize_sequestration_targets(
                inputs,
                dt = self.curves_npp.dt,
                
            )
            
            self._log(
                f"Fitting NPP curve for time period {i}...",
                type_log = "info",
            )
            t0 = time.time()


            ##  GET PARAMETERS AND CMF

            params = self.curves_npp.fit(
                curve_npp,
                force_convergence = force_convergence,
                method = method,
                options = {
                    "maxiter": maxiter,
                },
                tol = tol, 
                vec_params_0 = params_init,
            )

            arr_cmf = self.curves_npp.get_assumed_steady_state_sem_cumulative_mass(
                *params.x,
                return_type = "array_collapsed_to_tp",
                vec_widths = vec_widths,
            )

            # report fit time
            t_elapse = sf.get_time_elapsed(t0)
            self._log(
                f"NPP curve for time period {i} complete in {t_elapse} seconds.",
                type_log = "info",
            )
    
            tp = int(row[self.model_attributes.dim_time_period])
            dict_out.update(
                {
                    (tp, curve_npp.name, ): {
                        key_cmf: arr_cmf,
                        key_params: params, 
                    }
                }
            )
            
            
        # remove existing inputs
        self.curves_npp._initialize_sequestration_targets(
            [],
            dt = self.curves_npp.dt,
            
        )

        return dict_out



    def get_npp_convergence_tolerance(self,
        series_factors: pd.Series,
        tol_base: float = 0.0000001,
        **kwargs,
    ) -> Tuple:
        """For optimization, rescale values so that minimum sequestration factor 
            has log_10 > 0. Returns the convergence tolerance.
        """
        logs = np.array(
            [
                np.log10(v) for k, v in series_factors.to_dict().items() 
                if k != self.model_attributes.dim_time_period
            ]
        )
        
        # 
        min_log = logs.min()
        exponent = np.sign(min_log)*np.ceil(np.abs(min_log))
        
        tol = tol_base*(10**exponent) if np.sign(min_log) < 0 else tol_base

        return tol



    def get_npp_factor_rescale(self,
        series_factors: pd.Series,
    ) -> Tuple:
        """For optimization, rescale values so that minimum sequestration factor 
            has log_10 > 0. Returns the mofified scalar.
        """
        logs = np.array(
            [
                np.log10(v) for k, v in series_factors.to_dict().items() 
                if k != self.model_attributes.dim_time_period
            ]
        )

        # 
        min_log = logs.min()
        exponent = np.ceil(np.abs(min_log)) if min_log < 0 else 0
        scalar = 10**exponent

        return scalar



    def get_npp_frst_sequestration_factor_vectors(self,
        df_afolu_trajectories: pd.DataFrame,
        field_ord_1: str = "young",
        field_ord_2: str = "secondary",
        field_ord_3: str = "primary",
        return_factors: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        """Retrieve sequestration factor vectors for use in building estimated 
            curve. 

        Return
        ------    
        Set return_factors = True to return a tuple of the form
            (
                df_duplicates,
                arr_frst_ef_sequestration,
                arr_frst_ef_sequestration_young,
            )

            with sequestration factors in terms of (self.modvar_frst_sq_co2)
            units of mass and self.model_socioeconomic.modvar_gnrl_area area
        """

        ##  GET SEQUESTRATION FACTORS
        #   - LU prevalence and incidence
        #   - put everything in terms of modvar_frst_sq_co2
            
        arr_frst_ef_sequestration = self.get_frst_sequestration_factors(
            df_afolu_trajectories, 
            expand_to_all_cats = True,
            
        )
        arr_frst_ef_sequestration_young = self.get_frst_sequestration_factors(
            df_afolu_trajectories, 
            override_vector_for_single_mv_q = False, 
            modvar_sequestration = self.modvar_frst_sq_co2_young_secondary, 
        )
        
        #arr_frst_ef_sequestration_young *= self.model_attributes.get_variable_unit_conversion_factor(
        #    self.modvar_frst_sq_co2_young_secondary,
        #    self.modvar_frst_sq_co2,
        #    "mass"
        #)
        
        # build data frame that is ordered--use this for sorting and dropping duplicates
        field_tp = self.model_attributes.dim_time_period
        
        df_duplicates = (
            pd.DataFrame(
                {
                    field_tp: df_afolu_trajectories[field_tp].to_numpy().copy(),
                    field_ord_1: arr_frst_ef_sequestration_young,
                    field_ord_2: arr_frst_ef_sequestration[:, self.ind_frst_scnd],
                    field_ord_3: arr_frst_ef_sequestration[:, self.ind_frst_prim],
                }
            )
            .drop_duplicates(subset = [field_ord_1, field_ord_2, field_ord_3])
            .sort_values(by = [field_tp])
        )

        out = (
            df_duplicates
            if not return_factors
            else (df_duplicates, arr_frst_ef_sequestration, arr_frst_ef_sequestration_young, )
        )

        return out
    


    def get_soil_arrs_ef_c_drained_organic_soils(self,
        df_afolu_trajectories: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Get arrays for C drained organic soils emission factors. Converts to

            (a) output mass units
            (b) area of self.model_socioeconomic.modvar_gnrl_area,

        Returns a dictionary mapping each land use class associated with DOS to
            the associated C factor variable in Soil Management, e.g., 

            {
                "croplands": arr_soil_ef_c_dos_croplands,    # for croplands
                "pastures": arr_soil_ef_c_dos_pastures,      # for managed grasslands
            }
        """

        dict_out = {}

        for k, v in self.dict_lndu_categories_to_soil_variables.items():
            
            modvar_cur = v.get("c_factor_drained_organic_soils")

            arr_soil_ef_c_dos = self.model_attributes.extract_model_variable(#
                df_afolu_trajectories, 
                modvar_cur, 
                expand_to_all_cats = True,
                override_vector_for_single_mv_q = True, 
                return_type = "array_base",
            )

            arr_soil_ef_c_dos *= self.model_attributes.get_scalar(modvar_cur, "mass", )
            arr_soil_ef_c_dos /= self.model_attributes.get_variable_unit_conversion_factor(
                modvar_cur,
                self.model_socioeconomic.modvar_gnrl_area,
                "area",
            )

            dict_out.update({k: arr_soil_ef_c_dos, })

        return dict_out
    



    def get_soil_dict_lndu_categories_to_soil_c_dos_variables(self,
    ) -> Dict:
        """Return a dictionary with land use categories as keys based on the 
            Soil Management attribute table

            {
                cat_lndu: {
                    "c_factor_drained_organic_soils": VARNAME_C_DOS, 
                    ...
                }
            }

            for each key, the dict includes variables associated with the land
            use category cat_lndu:

            - "c_factor_drained_organic_soils"
        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            self.model_attributes.subsec_name_soil,
            "cat_landuse",
            {
                "C Annual": "c_factor_drained_organic_soils"
            },
        )

        return dict_out
    


    def get_transition_matrix_from_long_df(self,
        df_transition: pd.DataFrame,
        field_i: str,
        field_j: str,
        field_pij: str,
    ) -> np.ndarray:
        """
        Convert a long data frame to a transition matrix
        
        Function Arguments
        ------------------
        - df_transition: data frame storing transition probabilities in long form
        - field_i: field storing source class
        - field_j: field sotring target class
        - field_pij: field storing probabilities
        """
        
        # get attribute table
        attr_lndu = self.model_attributes.get_attribute_table(
            self.model_attributes.subsec_name_lndu
        )
        
        return_df = not isinstance(df_transition, pd.DataFrame)
        return_df |= (
            not set([field_i, field_j, field_pij]).issubset(set(df_transition.columns))
            if not return_df
            else return_df
        )
        
        if return_df:
            return df_transition
        
        
        # filter out invalid rows
        df_transition_out = (
            df_transition[
                df_transition[field_i].isin(attr_lndu.key_values)
                & df_transition[field_j].isin(attr_lndu.key_values)
            ]
            .copy()
            .reset_index(drop = True)
        )
        
        # initialize output matrix and fill in
        n = attr_lndu.n_key_values
        mat_out = np.zeros(n**2)
        
        # indices
        vec_i = np.array(
            df_transition_out[field_i]
            .apply(attr_lndu.get_key_value_index)
        )
        
        vec_j = np.array(
            df_transition_out[field_j]
            .apply(attr_lndu.get_key_value_index)
        )
        
        vec_inds = vec_i*n + vec_j
        vec_probs = df_transition_out[field_pij].to_numpy()
        
        # overwrite values in the matrix and reshape
        np.put(mat_out, vec_inds, vec_probs)
        mat_out = mat_out.reshape((n, n))
        
        return mat_out
    


    def get_units_from_specification(self,
        units_specifier: Union['ModelVariable', str, None],
        units_type: str,
    ) -> Union[str, None]:
        """Based on a units specifier, get the units. Returns None if invalid
            units_type is set.

        Function Arguments
        ------------------
        units_specifier : Union['ModelVariable', str, None]
            Optional specification of units; if a variable, attempts to match
            units from the variable. If a string, attempts to set that as the 
            units, but must be of valid units_type. If None, defaults to 
            configuration values. 
        units_type : str
            "energy", "mass"
        """

        # only types supported in this shortcut
        dict_type_to_config = {
            "energy": "energy_units",
            "mass": "emissions_mass",
        }

        key_config = dict_type_to_config.get(units_type, )
        if key_config is None:
            return None

        # start with energy
        um = self.model_attributes.get_unit(units_type)
        if is_model_variable(units_specifier):
            units_specifier = units_specifier.attribute(f"unit_{units_type}")

        if isinstance(units_specifier, str):
            units_specifier = um.get_unit_key(units_specifier)

        if units_specifier is None:
            units_specifier = self.model_attributes.configuration.get(key_config, )
        
        return units_specifier



    def plot_npp_get_window(self,
        x_range: Union[tuple, None] = None,
    ) -> tuple:
        """Get the plot x_range for showing the fit NPP curve

        Arguments
        ---------
        x_range : Union[tuple, None]
            Optional range. If None, defaults to sum over 
            self.npp_integration_windows. If excluding primary, only takes
            first two elements of the window.
        """
        # check if valid
        if isinstance(x_range, tuple):
            ret = sf.isnumber(x_range[0])
            ret &= sf.isnumber(x_range[1])
            ret &= (x_range[0] >= 0) if ret else ret
            ret &= (x_range[0] < x_range[1]) if ret else ret
            
            if ret:
                return x_range

        # otherwise, use default
        window = self.npp_integration_windows.copy()
        if not self.npp_include_primary_forest:
            window = window[0:-1]
        window = (0, sum(window))

        return window



    def plot_npp_fit_curve(self,
        key: tuple,
        dict_fit_parameters: dict,
        figtuple: Union[tuple, None] = None,
        x_range: Union[tuple, None] = None,
        **kwargs,
    ) -> Union['plt.Plot', None]:
        """Plot a fitted NPP curve

        Arguments
        ---------
        key : tuple
            (tp_group, npp_curve) tuple to plot in dict_fit_parameters
        dict_fit_parameters : dict
            Dictionary of fit curve parameters to pull from
        figtuple : tuple
            Optional fig, ax tuple to pass from matplotlib.pyplot
        x_range : optional specification of x_range to plot
        """
        # try the result
        result = dict_fit_parameters.get(key)
        if result is None:
            return None 

        # some init of curve
        tp, curve_name = key
        curve = self.curves_npp.get_curve(curve_name)
        if curve is None:
            return None

        # get the per time-period estimated average 
        estimated_averages = self.curves_npp.estimate_integral(curve_name, *result.x, )
        x_range = self.plot_npp_get_window(x_range, )
        
        t = np.arange(x_range[0], x_range[1], self.curves_npp.dt)
        y = curve(t, *result.x)

        fig, ax = (
            subplots(figsize = (10, 8))
            if not is_valid_figtuple(figtuple, )
            else figtuple
        )
            
        
        out = ax.plot(
            t, 
            y,
            **kwargs,
        )
        
        return out



    def project_agrc_lvst_integrated_demands(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_modvar_lndu_initial_area: np.ndarray,
        vec_pop: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
    ) -> tuple:
        """
        Calculate agricultural and livestock demands, imports, exports, and 
            production. Agriculture and Livestock demands are related through 
            feed demands, requiring integration.

        Function Arguments
        ------------------
        - df_afolu_trajectories: data frame containing input variables
        - vec_modvar_lndu_initial_area: vector (long by category) of initial
            areas by land use class
        - vec_pop: vector of population
        - vec_rates_gdp_per_capita: vector of gdp/capita growth rates

        Notes
        -----

        Let:
            D = Domestic production to satisfy domestic demand
            E = Exports
            F = Fraction of domestic demand met by imports
            I = Imports
            M = Domestic demand
            P = Domestic production

        Then:
            P = D + E, MF = I, M - I = D => M - MF = D =>

            P = M - MF + E : Domestic Production
            M = (P - E)/(1 - F) : Domestic Demand

        """

        ###########################
        #    LIVESTOCK DEMANDS    #
        ###########################

        # get variables requried to estimate demand - start with exports
        arr_lvst_exports_unadj = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_equivalent_exports,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        arr_lvst_frac_imported = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_frac_demand_imported,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_lvst_elas_demand = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_elas_lvst_demand,
            expand_to_all_cats = True, # should already be expanded
            return_type = "array_base",
        )

        vec_gnrl_frac_eating_red_meat = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        vec_lvst_production_init = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_pop_init,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )[0]


        ##  1. CALCULATE DOMESTIC DEMAND
        
        # adjust exports to have cap at production; modify series downward
        vec_lvst_exports_modifier = sf.vec_bounds(vec_lvst_production_init - arr_lvst_exports_unadj[0], (-np.inf, 0))
        arr_lvst_exports_unadj = sf.vec_bounds(arr_lvst_exports_unadj + vec_lvst_exports_modifier, (0, np.inf))
        
        # get the demand
        vec_lvst_domestic_demand_init = sf.vec_bounds(vec_lvst_production_init - arr_lvst_exports_unadj[0], (0, np.inf))
        vec_lvst_domestic_demand_init /= (1 - arr_lvst_frac_imported[0])
        vec_lvst_domestic_demand_init = sf.vec_bounds(
            np.nan_to_num(
                vec_lvst_domestic_demand_init, 
                nan = 0.0, 
                posinf = 0.0,
            ), 
            (0, np.inf),
        )
        
        # project aggregate domestic demand forward
        vec_gnrl_frac_eating_red_meat_scalar = sf.vec_bounds(vec_gnrl_frac_eating_red_meat/vec_gnrl_frac_eating_red_meat[0], (0, 1))
        arr_lvst_domestic_demand_unadj = self.project_per_capita_demand(
            vec_lvst_domestic_demand_init,
            vec_pop,
            vec_rates_gdp_per_capita,
            arr_lvst_elas_demand,
            vec_gnrl_frac_eating_red_meat_scalar,
            int
        )

        # get imports, domestic production for domestic demand (unadj), and domestic production
        arr_lvst_imports_unadj = arr_lvst_domestic_demand_unadj*arr_lvst_frac_imported
        arr_lvst_domestic_production_for_domestic_demand_unadj = arr_lvst_domestic_demand_unadj - arr_lvst_imports_unadj
        arr_lvst_domestic_production_unadj = arr_lvst_domestic_production_for_domestic_demand_unadj + arr_lvst_exports_unadj


        ##  2. get weights for allocating grazing area and feed requirement to animals - based on first year only
        
        vec_lvst_base_graze_weights = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lvst_dry_matter_consumption,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )[0]

        vec_lvst_feed_allocation_weights = vec_lvst_production_init*vec_lvst_base_graze_weights
        vec_lvst_feed_allocation_weights /= np.dot(vec_lvst_production_init, vec_lvst_base_graze_weights)
        
        # get carrying capacity scalar, adjusted for maximum dry matter production and scaled to ensure first element is 1
        (
            arr_lvst_annual_dry_matter_consumption_per_capita,
            vec_lndu_yf_pasture_sup_adj, # in terms of pasture yield factor/LNDU initial area
            vec_lvst_carry_capacity_scale,
            factor_lndu_init_avg_consumption_pstr,
        ) = self.get_lvst_pasture_max_yield_and_carrying_capacity(
            df_afolu_trajectories,
            vec_modvar_lndu_initial_area,
        )


        #############################
        #    AGRICULTURE DEMANDS    #
        #############################

        # variables required for demand
        arr_agrc_elas_crop_demand = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_elas_crop_demand_income,
            return_type = "array_base",
        )
        arr_agrc_exports_unadj = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_equivalent_exports,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )
        arr_agrc_frac_feed = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_frac_animal_feed,
            return_type = "array_base",
        )
        arr_agrc_frac_imported = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_frac_demand_imported,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        # do some unit conversion
        arr_agrc_exports_unadj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_equivalent_exports,
            self.modvar_agrc_yf,
            "mass"
        )


        ##   1. get initial cropland areas and yields
        
        vec_agrc_frac_cropland_area = self.check_cropland_fractions(df_afolu_trajectories, "initial")[0]
        vec_agrc_cropland_area = vec_modvar_lndu_initial_area[self.ind_lndu_crop]*vec_agrc_frac_cropland_area

        # get yield factors and calculate yield
        arr_agrc_yf = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_yf,
            return_type = "array_base",
        )
        arr_agrc_yf *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        vec_agrc_yield_init = arr_agrc_yf[0]*vec_agrc_cropland_area


        ##  2. get dietary demand scalar for crop demand (increases based on reduction in red meat demand) - depends on how many people eat red meat (vec_gnrl_frac_eating_red_meat)
        
        vec_agrc_diet_exchange_scalar = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_vdes,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )
        vec_agrc_demscale = vec_gnrl_frac_eating_red_meat + vec_agrc_diet_exchange_scalar - vec_gnrl_frac_eating_red_meat*vec_agrc_diet_exchange_scalar
        vec_agrc_demscale = np.nan_to_num(vec_agrc_demscale/vec_agrc_demscale[0], nan = 1.0, posinf = 1.0, )
        
        # get categories that need to be scaled
        vec_agrc_scale_demands_for_veg = np.array(
            self.model_attributes.get_ordered_category_attribute(
                self.subsec_name_agrc, 
                "apply_vegetarian_exchange_scalar"
            )
        )

        arr_agrc_demscale = np.outer(vec_agrc_demscale, vec_agrc_scale_demands_for_veg)
        arr_agrc_demscale = arr_agrc_demscale + np.outer(np.ones(len(vec_agrc_demscale)), 1 - vec_agrc_scale_demands_for_veg)


        ##  3. Calculate crop demands split into yield for livestock feed (responsive to changes in domestic livestock population) and yield for consumption and export (nonlvstfeed)
        
        vec_agrc_domestic_demand_init = sf.vec_bounds(vec_agrc_yield_init - arr_agrc_exports_unadj[0], (0, np.inf))
        vec_agrc_domestic_demand_init /= (1 - arr_agrc_frac_imported[0])
        vec_agrc_domestic_demand_init = sf.vec_bounds(
            np.nan_to_num(
                vec_agrc_domestic_demand_init, 
                nan = 0.0, 
                posinf = 0.0,
            ), 
            (0, np.inf)
        )
        
        # split out livestock demands and human consumption
        vec_agrc_domestic_demand_init_lvstfeed = vec_agrc_domestic_demand_init*arr_agrc_frac_feed[0]
        vec_agrc_domestic_demand_init_nonlvstfeed = vec_agrc_domestic_demand_init - vec_agrc_domestic_demand_init_lvstfeed
        vec_agrc_production_init_lvstfeed = vec_agrc_domestic_demand_init_lvstfeed*(1 - arr_agrc_frac_imported[0])
        
        # project domestic demand
        arr_agrc_domestic_demand_nonfeed_unadj = self.project_per_capita_demand(
            vec_agrc_domestic_demand_init_nonlvstfeed,
            vec_pop,
            vec_rates_gdp_per_capita,
            arr_agrc_elas_crop_demand,
            arr_agrc_demscale,
            float
        )
        arr_agrc_imports_unadj = arr_agrc_domestic_demand_nonfeed_unadj*arr_agrc_frac_imported
        arr_agrc_production_nonfeed_unadj = arr_agrc_domestic_demand_nonfeed_unadj - arr_agrc_imports_unadj + arr_agrc_exports_unadj
 

        ##  4. Apply production wasted scalar to projected supply requirements, then re-calculate demand (reducing domestic production loss will drop future demands)
        
        vec_agrc_frac_production_wasted = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_agrc_frac_production_lost,
            return_type = "array_base",
            var_bounds = (0, 1),
        )
        vec_agrc_frac_production_wasted_scalar = np.nan_to_num(
            (1 - vec_agrc_frac_production_wasted[0])/(1 - vec_agrc_frac_production_wasted), 
            nan = 0.0, 
            posinf = 0.0,
        )
       
        arr_agrc_production_nonfeed_unadj = (arr_agrc_production_nonfeed_unadj.transpose()*vec_agrc_frac_production_wasted_scalar).transpose()
        arr_agrc_domestic_demand_nonfeed_unadj = arr_agrc_production_nonfeed_unadj + arr_agrc_imports_unadj
        
        # array gives the total yield of crop type i allocated to livestock type j at time 0
        arr_lndu_yield_i_reqd_lvst_j_init = np.outer(vec_agrc_production_init_lvstfeed, vec_lvst_feed_allocation_weights)

        out = (
            # agrc vars
            arr_agrc_domestic_demand_nonfeed_unadj,
            arr_agrc_exports_unadj,
            arr_agrc_imports_unadj,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            vec_agrc_frac_cropland_area,
            vec_agrc_frac_production_wasted,
            # lvst vars
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_annual_dry_matter_consumption_per_capita,
            arr_lvst_domestic_demand_unadj,
            arr_lvst_exports_unadj,
            arr_lvst_imports_unadj,
            arr_lvst_domestic_production_unadj,
            vec_lvst_carry_capacity_scale,
            vec_lvst_feed_allocation_weights,
            vec_lndu_yf_pasture_sup_adj,
            factor_lndu_init_avg_consumption_pstr,
        )

        return out



    def project_per_capita_demand(self,
        dem_0: np.ndarray, 
        pop: np.ndarray,
        gdp_per_capita_rates: np.ndarray, 
        elast: np.ndarray,
        dem_pc_scalar_exog: Union[float, None] = None, 
        return_type: type = float # return type of array
    ) -> np.ndarray:
        """
        Project per capita demand for agriculture and/or livestock

        Function Arguments
        ------------------
        dem_0 : np.ndarray
            Initial demand (e.g., total yield/livestock produced per acre)
        pop : np.ndarray
            Population (vec_pop)
        gdp_per_capita_rates driver of demand growth : np.ndarray
            GDP/capita (vec_rates_gdp_per_capita)
        elast :  np.ndarray
            Elasticity of demand per capita to growth in gdp/capita (e.g., 
            arr_lvst_elas_demand)
        
        Keyword Arguments
        -----------------
        dem_pc_scalar_exog : Union[float, None]
            Exogenous demand per capita scalar representing other changes in the 
            exogenous per-capita demand (can be used to represent population 
            changes)
        return_type : type
            Return type of array
        """
        
        # get the demand scalar to apply to per-capita demands
        dem_scale_proj_pc = (gdp_per_capita_rates.transpose()*elast[0:-1].transpose()).transpose()
        dem_scale_proj_pc = np.cumprod(1 + dem_scale_proj_pc, axis = 0)
        dem_scale_proj_pc = np.concatenate([np.ones((1,len(dem_scale_proj_pc[0]))), dem_scale_proj_pc])
        
        # estimate demand for livestock (used in CBA) - start with livestock population per capita
        dem_pc_scalar_exog = np.ones(pop.shape) if (dem_pc_scalar_exog is None) else dem_pc_scalar_exog

        if dem_pc_scalar_exog.shape == pop.shape:
            arr_dem_base = np.outer(pop*dem_pc_scalar_exog, dem_0/pop[0])

        elif dem_pc_scalar_exog.shape == dem_scale_proj_pc.shape:
            arr_pc = (dem_0/pop[0])*dem_pc_scalar_exog
            arr_dem_base = (pop*arr_pc.transpose()).transpose()
            
        else:
            raise ValueError(f"Invalid shape of dem_pc_scalar_exog: valid shapes are '{pop.shape}' and '{dem_scale_proj_pc.shape}'.")

        # get the total demand
        arr_dem_base = np.array(dem_scale_proj_pc*arr_dem_base).astype(return_type)

        return arr_dem_base



    def project_integrated_land_use(self,
        df_afolu_trajectories: pd.DataFrame,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_c_agb: np.ndarray,
        arrs_c_bgb: np.ndarray,
        arr_c_init_agb: np.ndarray,
        arr_agrc_production_nonfeed_unadj: np.ndarray,
        arr_agrc_yield_factors: np.ndarray,
        arr_lndu_constraints_inf: np.ndarray,
        arr_lndu_constraints_sup: np.ndarray,
        arr_lndu_frac_increasing_net_exports_met: np.ndarray,
        arr_lndu_frac_increasing_net_imports_met: np.ndarray,
        arr_lndu_yield_by_lvst: np.ndarray,
        factor_lndu_init_avg_consumption_pstr: float,
        arr_lvst_annual_dry_matter_consumption_per_capita: np.ndarray,
        arr_lvst_dem: np.ndarray,
        vec_agrc_frac_cropland_area: np.ndarray,
        vec_lndu_yrf: np.ndarray,
        vec_lvst_scale_cc: np.ndarray,
        vec_gnrl_area: np.ndarray,
        n_tp: Union[int, None] = None,
    ) -> Tuple:
        """Integrated land use model, which performs required land use 
            transition adjustments.

        NOTE: constraints are used to bound areas. However, they are not
            binding in the same way.

            * Minimum area constraints *do not* allow any land use transitions
                out of a state if, without accounting for inflows, the area
                of that state would fall below the minimum. This better reflects 
                the behavior of protected areas, where a fixed minimum area 
                does not change (if we assume the land cover mix within the
                protected area does not change)
            * Maximum area constraints are calculated after accounting for both
                inflows and outflows. 


        Function Arguments
        ------------------
        df_afolu_trajectories : pd.DataFrame
            Needed for accessing variables to estimate HWP/Biomass demands
        vec_initial_area : np.ndarray
            Initial state vector of area
        arrs_transitions : np.ndarray
            Array of transition matrices, ordered by time period
        arrs_c_agb : np.ndarray
            Array of above-ground biomass c stock conversion factors, in terms 
            of output units of mass per area (units of vec_initial_area)
        arrs_c_bgb : np.ndarray
            Array of below-ground biomass c stock conversion factors, in terms 
            of output units of mass per area (units of vec_initial_area)
        arr_c_init_agb : np.ndarray
            Initial 
        arr_agrc_production_nonfeed_unadj : np.ndarray
            Array of agricultural non-feed demand yield (human consumption)
        arr_agrc_yield_factors : np.ndarray
            Array of agricultural yield factors
        arr_lndu_constraints_inf : np.ndarray
            Minimum bounds on areas of each land use class (columns) for each 
            time period (rows). To set no constraint, use 
            AFOLU.flag_ignore_constraint (-999)
        arr_lndu_constraints_sup : np.ndarray
            Maximum bounds on areas of each land use class (columns) for each 
            time period (rows). To set no constraint, use 
            AFOLU.flag_ignore_constraint (-999)
        arr_lndu_frac_increasing_net_exports_met : np.ndarray
            Fraction--by land use type--of increases to net exports that are 
            met. Adjusts production demands downward if less than 1.
        arr_lndu_frac_increasing_net_exports_met : np.ndarray
            Fraction--by land use type--of increases to net imports that are 
            met. Adjusts production demands upward if less than 1.
        arr_lndu_yield_by_lvst : np.ndarray
            Array of lvst yield by land use category (used to project future 
            livestock supply). Array gives the total yield of crop type i 
            allocated to livestock type j at time 0 
            (attr_agriculture.n_key_values x attr_livestock.n_key_values)
        factor_lndu_init_avg_consumption_pstr : float
            Per unit area of pasture initial consumption by livestock 
            (estimated). Estimated as proxy for production of dry matter.
                * UNITS : pasture yield factor/gnrl area
        arr_lvst_annual_dry_matter_consumption_per_capita : np.ndarray
            Annual dry matter consumption per head of livestock
                * UNITS : pasture yield factor/head
        arr_lvst_dem : np.ndarray
            Array of livestock production requiremenets (unadjusted)
        vec_agrc_frac_cropland_area : np.ndarray
            Vector of fractions of agricultural area fractions by classes
        vec_lndu_yrf : np.ndarray
            Vector of land use reallocation factor 
        vec_lvst_scale_cc : np.ndarray
            Vector of livestock carrying capacity scalar to apply

        Keyword Arguments
        ------------------
        n_tp : int
            Number of time periods to run. If None, runs AFOLU.n_time_periods

            
        Returns
        -------
        Tuple with 14 elements:
        - arr_agrc_change_to_net_imports_lost,
        - arr_agrc_frac_cropland,
        - arr_agrc_net_import_increase,
        - arr_agrc_yield,
        - arr_emissions_conv,
        - arr_emissions_conv_matrices,
        - arr_land_use,
        - arr_lvst_change_to_net_imports_lost,
        - arr_lvst_net_import_increase,
        - arr_lvst_pop_ardj,
        - arrs_land_conv,
        - arrs_transitions_adj,
        - arrs_yields_per_livestock

        """
        
        t0 = time.time()

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_c_agb, "arrs_efs")
        self.check_markov_shapes(arrs_c_bgb, "arrs_efs")

        # get attributes
        attr_agrc = self.model_attributes.get_attribute_table(self.subsec_name_agrc)
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        attr_lvst = self.model_attributes.get_attribute_table(self.subsec_name_lvst)

        # set some commonly called attributes and indices in arrays
        inds_frst = [
            attr_lndu.get_key_value_index(x) 
            for x in self.dict_cats_lndu_to_cats_frst.keys()
        ]
        m = attr_lndu.n_key_values


        ##  INITIALIZE OUTPUT ARRAYS AND VARIABLES

        # intilize land use, land converted, emissions, and adjusted transitions
        arr_agrc_frac_cropland = np.array([vec_agrc_frac_cropland_area for k in range(n_tp)])
        arr_agrc_net_import_increase = np.zeros((n_tp, attr_agrc.n_key_values))
        arr_agrc_change_to_net_imports_lost = np.zeros((n_tp, attr_agrc.n_key_values))
        
        # initialize carbon stock stored in above-ground biomass in forests
        arr_c_stock_frsts = vec_initial_area[inds_frst]*arr_c_init_agb[0, inds_frst]
        arr_c_stock_frsts = np.repeat(
            [arr_c_stock_frsts],
            n_tp,
            axis = 0,
        )

        # 

        # get yield
        arr_agrc_yield = np.array([
            (vec_initial_area[self.ind_lndu_crop]*vec_agrc_frac_cropland_area*arr_agrc_yield_factors[0]) 
            for k in range(n_tp)
        ])
        
        arr_emissions_conv = np.zeros((n_tp, attr_lndu.n_key_values))
        arr_emissions_conv_matrices = np.zeros(arrs_transitions.shape)
        arr_land_use = np.array([vec_initial_area for k in range(n_tp)])
        
        # livestock demands
        arr_lvst_dem_adj = arr_lvst_dem.copy().astype(int)
        arr_lvst_pop_adj = arr_lvst_dem.copy().astype(int)

        arr_lvst_net_import_increase = np.zeros((n_tp, attr_lvst.n_key_values))
        arr_lvst_change_to_net_imports_lost = np.zeros((n_tp, attr_lvst.n_key_values))
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_transitions_adj = np.zeros(arrs_transitions.shape)

        arrs_yields_per_livestock = np.array([arr_lndu_yield_by_lvst for k in range(n_tp)])

        
        # initialize biomass removals demand
        ##  HARVESTED WOOD PRODUCTS


        # add to output
        df_out = self.estimate_c_demand_from_hwp(
            df_afolu_trajectories,
            vec_hh,
            vec_gdp,
            vec_rates_gdp,
            vec_rates_gdp_per_capita,
            dict_dims,
            n_projection_time_periods,
            projection_time_periods,
            self.dict_integration_variables_by_subsector,
        )

        ##  HEREHEREHERE

        ##  INITIALIZE VARIABLES

        vec_lvst_dem_gr_iterator = np.ones(len(arr_lvst_dem[0]))



        """
        Rough note on the transition adjustment process:

        a. Start with land use prevalance at time ind_first_nz 
            i. estimated as prevalence at 
                x_{ind_first_nz - 1}Q_{ind_first_nz - 1}
        b. Using land use carrying capacity and pasture fraction & crop demands,
             get requirements for livestock and use to adjust pasture and
             cropland requirement (see scalar_lndu_pstr and scalar_lndu_crop,
             respectively)
            i. Use `AFOLU.get_lndu_scalar_max_out_states` to get true 
                positional scalars. This accounts for states that might 
                "max out" (as determined by 
                AFOLU.mask_lndu_max_out_states), or reach 1 or 0 
                probability during the scaling process.
        c. Then, with the scalars obtained, adjust the matrix using 
            AFOLU.adjust_transition_matrix
        """
        

        ##  ITERATE OVER EACH TIME PERIOD
        #
        # 1. Calculate initial states
        # 2. Adjust based on land use reallocation factor and land class protections

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp - 1:
            
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            if i_ef != i:
                self._log(
                    f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "warning"
                )

            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            if i_tr != i:
                self._log(
                    f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "warning"
                )


            ##  GET PROJECTED (UNADJUSTED) LAND USE PREVALENCE

            x_proj_unadj = np.dot(x, arrs_transitions[i_tr])

            # crop values
            area_crop_cur = x[self.ind_lndu_crop]
            area_crop_proj = x_proj_unadj[self.ind_lndu_crop]
            vec_agrc_cropland_area_proj = area_crop_proj*arr_agrc_frac_cropland[i]

            # pasture values
            area_pstr_cur = x[self.ind_lndu_pstr]
            area_pstr_proj = x_proj_unadj[self.ind_lndu_pstr]


            # check areas where lvst has 0 pop
            inds_lvst_where_pop_noncc = np.where(arr_lvst_annual_dry_matter_consumption_per_capita[i + 1] == 0)[0]
            vec_lvst_prod_supported_pre_realloc = self.get_lvst_production_supported(
                area_pstr_proj,
                arr_lvst_annual_dry_matter_consumption_per_capita[i + 1],
                arr_lvst_dem[i + 1],
                factor_lndu_init_avg_consumption_pstr,
                vec_lvst_scale_cc[i + 1],
            )

            vec_lvst_unmet_demand = np.nan_to_num(
                arr_lvst_dem[i + 1] - vec_lvst_prod_supported_pre_realloc,
                nan = 0.0,
                posinf = 0.0,
            )

            # calculate net surplus met
            vec_lvst_unmet_demand_to_impexp = (
                sf.vec_bounds(vec_lvst_unmet_demand, (0, np.inf))
                *arr_lndu_frac_increasing_net_imports_met[i + 1, self.ind_lndu_pstr]
            )
            vec_lvst_unmet_demand_to_impexp += (
                sf.vec_bounds(vec_lvst_unmet_demand, (-np.inf, 0))
                *arr_lndu_frac_increasing_net_exports_met[i + 1, self.ind_lndu_pstr]
            )
            vec_lvst_unmet_demand_lost = vec_lvst_unmet_demand - vec_lvst_unmet_demand_to_impexp

            vec_lvst_pop_adj = vec_lvst_prod_supported_pre_realloc + vec_lvst_unmet_demand_lost
            if len(inds_lvst_where_pop_noncc) > 0:
                np.put(vec_lvst_unmet_demand_to_impexp, inds_lvst_where_pop_noncc, 0)
                np.put(vec_lvst_unmet_demand_lost, inds_lvst_where_pop_noncc, 0)
                np.put(vec_lvst_pop_adj, inds_lvst_where_pop_noncc, arr_lvst_dem[i + 1, inds_lvst_where_pop_noncc])


            vec_lvst_unmet_demand = vec_lvst_unmet_demand_to_impexp.copy()
            vec_lvst_reallocation_target = vec_lvst_unmet_demand*vec_lndu_yrf[i + 1] # demand for livestock met by reallocating land

            # get the target pasture area
            area_lndu_pstr_increase_0 = self.get_lvst_area_required(
                arr_lvst_annual_dry_matter_consumption_per_capita[i + 1],
                vec_lvst_reallocation_target, 
                factor_lndu_init_avg_consumption_pstr*vec_lvst_scale_cc[i + 1],
            )
            area_target_pstr = area_pstr_proj + area_lndu_pstr_increase_0



            ##  AGRICULTURE - calculate demand increase in crops, which is a function of gdp/capita (exogenous) and livestock demand (used for feed)
                        
            vec_agrc_feed_dem_yield = sum((arr_lndu_yield_by_lvst*vec_lvst_dem_gr_iterator).transpose())
            vec_agrc_total_dem_yield = (arr_agrc_production_nonfeed_unadj[i + 1] + vec_agrc_feed_dem_yield)

            # calculate net surplus for yields
            vec_agrc_proj_yields = vec_agrc_cropland_area_proj*arr_agrc_yield_factors[i + 1]
            vec_agrc_unmet_demand_yields = vec_agrc_total_dem_yield - vec_agrc_proj_yields
            vec_agrc_unmet_demand_yields_to_impexp = (
                sf.vec_bounds(
                    vec_agrc_unmet_demand_yields, 
                    (0, np.inf)
                )
                *arr_lndu_frac_increasing_net_imports_met[i + 1, self.ind_lndu_crop]
            )
            vec_agrc_unmet_demand_yields_to_impexp += (
                sf.vec_bounds(
                    vec_agrc_unmet_demand_yields, (-np.inf, 0)
                )
                *arr_lndu_frac_increasing_net_exports_met[i + 1, self.ind_lndu_crop]
            )
            vec_agrc_unmet_demand_yields_lost = vec_agrc_unmet_demand_yields - vec_agrc_unmet_demand_yields_to_impexp

            # adjust yields for import/export scalar
            vec_agrc_proj_yields_adj = vec_agrc_proj_yields + vec_agrc_unmet_demand_yields_lost
            vec_agrc_yield_factors_adj = np.nan_to_num(
                vec_agrc_proj_yields_adj/vec_agrc_cropland_area_proj, 
                nan = 0.0, 
                posinf = 0.0,
            ) # replaces arr_agrc_yield_factors[i + 1] below
            vec_agrc_total_dem_yield = vec_agrc_proj_yields_adj + vec_agrc_unmet_demand_yields_to_impexp

            # now, generate modified crop areas and net surplus of crop areas
            vec_agrc_dem_cropareas = np.nan_to_num(
                vec_agrc_total_dem_yield/vec_agrc_yield_factors_adj, 
                nan = 0.0, 
                posinf = 0.0,
            )
            vec_agrc_unmet_demand = vec_agrc_dem_cropareas - vec_agrc_cropland_area_proj
            vec_agrc_reallocation_target = vec_agrc_unmet_demand*vec_lndu_yrf[i + 1]

            # get surplus yield (increase to net imports)
            vec_agrc_net_imports_increase = (vec_agrc_unmet_demand - vec_agrc_reallocation_target)*vec_agrc_yield_factors_adj
            vec_agrc_cropareas_adj = vec_agrc_cropland_area_proj + vec_agrc_reallocation_target
            area_target_crop = vec_agrc_cropareas_adj.sum()


            ##  DO TRANSITION ADJUSTMENT (CALL OPTIMIZATION)

            # the reallocation may not have succeeded due to constraints on area
            dict_area_targets_exog = {
                self.ind_lndu_crop: area_target_crop,
                self.ind_lndu_pstr: area_target_pstr,
            }
            
            arr_transition_adj = self.qadj_adjust_transitions(
                arrs_transitions[i_tr],
                x,
                dict_area_targets_exog,
                arr_lndu_constraints_inf[i],
                arr_lndu_constraints_sup[i],
                area = vec_gnrl_area[i],
                x_proj_unadj = x_proj_unadj,
                solver = "quadprog",
            )

            x_next  = np.matmul(x, arr_transition_adj)


            ###   AFTER RUNNING OPT, WE HAVE TO ADJUSTE AREAS AND FINAL BALANCES, YIELDS, ETC.

            ##  LVST

            # livestock - calculate how much area extra is provided ()
            vec_lvst_net_imports_increase_adjustment = np.nan_to_num(
                (area_target_pstr - x_next[self.ind_lndu_pstr])/area_lndu_pstr_increase_0,
                nan = 0.0,
                posinf = 0.0,
                neginf = 0.0,
            )
            vec_lvst_net_imports_increase_adjustment *= vec_lvst_reallocation_target

            # adjust the lvst reallocation value and calculate net imports
            vec_lvst_reallocation = vec_lvst_reallocation_target - vec_lvst_net_imports_increase_adjustment
            vec_lvst_net_import_increase = vec_lvst_unmet_demand - vec_lvst_reallocation # demand for livestock met by increasing net imports (neg => net exports)
            vec_lvst_pop_adj += vec_lvst_reallocation
            vec_lvst_dem_gr_iterator = np.nan_to_num(vec_lvst_pop_adj/arr_lvst_dem[0], nan = 1.0, posinf = 1.0, )

            # update output arrays
            arr_lvst_pop_adj[i + 1] = np.round(vec_lvst_pop_adj).astype(int)


            ##  AGRC 

            # calculate how much area in surplus exists, then adjust yields down and add extra to net imports
            vec_agrc_net_imports_increase_adjustment = np.nan_to_num(
                (area_target_crop - x_next[self.ind_lndu_crop])/vec_agrc_reallocation_target.sum(),
                nan = 0.0,
                posinf = 0.0,
                neginf = 0.0,
            )
            vec_agrc_net_imports_increase_adjustment *= vec_agrc_yield_factors_adj

            # add the increase to the imports 
            vec_agrc_net_imports_increase += vec_agrc_net_imports_increase_adjustment
            vec_agrc_cropareas_adj = vec_agrc_cropareas_adj/vec_agrc_cropareas_adj.sum()
            vec_agrc_cropareas_adj *= x_next[self.ind_lndu_crop]
            vec_agrc_yield_adj = vec_agrc_total_dem_yield - vec_agrc_net_imports_increase


            ##  CALCULATE FINAL LAND CONVERSION AND EMISSIONS 

            arr_land_conv = (arr_transition_adj.transpose()*x.transpose()).transpose()
            arr_emissions_conv_matrix = (arr_transition_adj*arrs_efs[i_ef]).transpose()*x.transpose() # sums across columns
            arr_emissions_conv_matrix = arr_emissions_conv_matrix.transpose()
            vec_emissions_conv = arr_emissions_conv_matrix.sum(axis = 1)

            if i + 1 < n_tp:
                # update agriculture arrays
                rng_agrc = list(range((i + 1)*attr_agrc.n_key_values, (i + 2)*attr_agrc.n_key_values))
                np.put(arr_agrc_change_to_net_imports_lost, rng_agrc, vec_agrc_unmet_demand_yields_lost)
                np.put(arr_agrc_frac_cropland, rng_agrc, vec_agrc_cropareas_adj/sum(vec_agrc_cropareas_adj))
                np.put(arr_agrc_net_import_increase, rng_agrc, np.round(vec_agrc_net_imports_increase), 2)
                np.put(arr_agrc_yield, rng_agrc, vec_agrc_yield_adj)

                # update livestock arrays
                arr_lvst_change_to_net_imports_lost[i + 1] = vec_lvst_unmet_demand_lost
                arr_lvst_net_import_increase[i + 1] = np.round(vec_lvst_net_import_increase).astype(int)

            # non-ag arrays
            rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)

            arr_emissions_conv_matrices[i] = arr_emissions_conv_matrix
            arrs_land_conv[i] = arr_land_conv
            arrs_transitions_adj[i] = arr_transition_adj

            # update land use vector and iterate
            x = np.matmul(x, arr_transition_adj)
            i += 1


        # add on final time step by repeating the transition matrix
        trans_adj = arrs_transitions_adj[i - 1]

        # calculate final land conversion and emissions
        arr_land_conv = (trans_adj.transpose()*x.transpose()).transpose()
        arr_emissions_conv_matrix = (trans_adj*arrs_efs[len(arrs_efs) - 1]).transpose()*x.transpose() # sums across columns
        arr_emissions_conv_matrix = arr_emissions_conv_matrix.transpose()
        vec_emissions_conv = arr_emissions_conv_matrix.sum(axis = 1)
        
        # add to tables
        rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
        np.put(arr_land_use, rng_put, x)
        np.put(arr_emissions_conv, rng_put, vec_emissions_conv)

        # update matrix lists
        arr_emissions_conv_matrices[i] = arr_emissions_conv_matrix
        arrs_land_conv[i] = arr_land_conv
        arrs_transitions_adj[i] = trans_adj

        out = (
            arr_agrc_change_to_net_imports_lost,
            arr_agrc_frac_cropland,
            arr_agrc_net_import_increase,
            arr_agrc_yield,
            arr_emissions_conv,
            arr_emissions_conv_matrices,
            arr_land_use,
            arr_lvst_change_to_net_imports_lost,
            arr_lvst_net_import_increase,
            arr_lvst_pop_adj,
            arrs_land_conv,
            arrs_transitions_adj,
            arrs_yields_per_livestock,
        )
    
        return out



    def project_land_use(self,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_efs: np.ndarray,
        n_tp: Union[int, None] = None,
    ) -> Tuple:

        """
        Basic version of project_land_use_integrated() that only projects
            land use dynamics (performing Markov forward linear algebra 
            arithmetic)

        Function Arguments
        ------------------
        """

        t0 = time.time()

        np.seterr(divide = "ignore", invalid = "ignore")

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get land use info
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)

        # intilize the land use and conversion emissions array
        shp_init = (n_tp, attr_lndu.n_key_values)
        arr_land_use = np.zeros(shp_init)
        arr_emissions_conv = np.zeros(shp_init)
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp:
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            (
                self._log(
                    f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "info"
                )
                if i_ef != i
                else None
            )

            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            (
                self._log(
                    f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.",
                    type_log = "info"
                )
                if i_tr != i
                else None
            )

            # calculate land use, conversions, and emissions
            vec_emissions_conv = sum((arrs_transitions[i_tr] * arrs_efs[i_ef]).transpose()*x.transpose())
            arr_land_conv = (arrs_transitions[i_tr].transpose()*x.transpose()).transpose()
            
            # update matrices
            rng_put = np.arange(i*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)
            np.put(arrs_land_conv, np.arange(i*attr_lndu.n_key_values**2, (i + 1)*attr_lndu.n_key_values**2), arr_land_conv)
            
            # update land use vector
            x = np.matmul(x, arrs_transitions[i_tr])

            i += 1

        t1 = time.time()
        t_elapse = round(t1 - t0, 2)
        self._log(f"Land use projection complete in {t_elapse} seconds.", type_log = "info")

        return arr_emissions_conv, arr_land_use, arrs_land_conv



    def qadj_adjust_transitions(self,
        Q: np.ndarray,
        x_0: np.ndarray,
        dict_area_targets_exog: dict, 
        vec_infimum_in: np.ndarray,
        vec_supremum_in: np.ndarray,
        area: Union[np.ndarray, None] = None,
        x_proj_unadj: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> Dict:
        """Format adjustment problem inputs

        
        Function Arguments
        ------------------
        Q : np.ndarray
            Unadjusted transition matrix
        x_0 : np.ndarray
            Initial prevalence
        dict_area_targets_exog : dict
            Dictionary mapping indices to fixed values from reallocation
        vec_infima_in : np.ndarray
            Vector specifying class infima; use flag_ignore to set no infimum 
            for a class
        vec_suprema_in : np.ndarray
            Vector specifying class suprema; use flag_ignore to set no supremum 
            for a class
                
        Keyword Arguments
        -----------------
        area : Union[np.ndarray, None]
            Optional specification of area for normalization
        x_proj_unadj : Union[np.ndarray, None]
            Unadjusted projected land use derived from exogenously specified 
            (unadjusted) transition matrix

        """

        # retrieve inputs that are normalized
        (
            Q,
            x0,
            xT,
            vec_infimum,
            vec_supremum,
            costs_x,
        ) = self.qadj_get_inputs(
            Q,
            x_0,
            dict_area_targets_exog,
            vec_infimum_in,
            vec_supremum_in,
            area = area,
            x_proj_unadj = x_proj_unadj,
        )

        # run the optimization
        out = self.q_adjuster.solve(
            Q,
            x0,
            xT,
            vec_infimum,
            vec_supremum,
            self.flag_ignore_constraint,
            costs_x = costs_x, # definitely don't want to forget the prevalence costs   np.zeros(x0.shape),#
            #thresh_to_zero = 0.0001
            **kwargs,
        )
            
        return out



    def qadj_get_inputs(self,
        Q: np.ndarray,
        x_0: np.ndarray,
        dict_area_targets_exog: dict, 
        vec_infimum_in: np.ndarray,
        vec_supremum_in: np.ndarray, 
        area: Union[np.ndarray, None] = None, 
        x_proj_unadj: Union[np.ndarray, None] = None,
    ) -> Tuple:
        """Format inputs to the QAdjuster for land use. Renormalizes vectors and 
            builds costs for prevalence in objective function.

        Function Arguments
        ------------------
        Q : np.ndarray
            Unadjusted transition matrix
        x_0 : np.ndarray
            Initial prevalence
        dict_area_targets_exog : dict
            Dictionary mapping indices to fixed values from reallocation
        vec_infima_in : np.ndarray
            Vector specifying class infima; use flag_ignore to set no infimum 
            for a class
        vec_suprema_in : np.ndarray
            Vector specifying class suprema; use flag_ignore to set no supremum 
            for a class
                
        Keyword Arguments
        -----------------
        area : Union[np.ndarray, None]
            Optional specification of area for normalization
        x_proj_unadj : Union[np.ndarray, None]
            Unadjusted projected land use derived from exogenously specified 
            (unadjusted) transition matrix

        """
        # initialize the land area
        area = x_0.sum() if not sf.isnumber(area) else area
        x_proj_unadj = (
            x_proj_unadj
            if isinstance(x_proj_unadj, np.ndarray)
            else np.dot(x_0, Q)
        )

        # get x target
        x_target = self.qadj_get_x_target(
            dict_area_targets_exog,
            x_proj_unadj,
        )

        # get the infimum, supremum, and initial/target prevalence vectors
        vec_infimum = self.qadj_normalize_vector_for_adjustment(vec_infimum_in, area, )
        vec_supremum = self.qadj_normalize_vector_for_adjustment(vec_supremum_in, area, )
        x0 = self.qadj_normalize_vector_for_adjustment(x_0, area, )
        xT = self.qadj_normalize_vector_for_adjustment(x_target, area, )

        # get costs for prevalence targets--ignore those that don't need to hit a target
        costs_x = dict(
            (i, 0) for i, v in enumerate(xT) if (v == self.flag_ignore_constraint)
        )

        out = (
            Q,
            x0,
            xT,
            vec_infimum,
            vec_supremum,
            costs_x,
        )

        return out




    def qadj_get_x_target(self,
        dict_area_targets_exog: dict,
        x_proj_unadj: np.ndarray,
        attr_lndu: Union[AttributeTable, None] = None,
    ) -> np.ndarray:
        """
        Get vector of target land use prevalence areas.

        Function Arguments
        ------------------
        - dict_area_targets_exog: dictionary mapping indices to fixed values
        - x_proj_unadj: unadjusted projected land use derived from 
            exogenously specified (unadjusted) transition matrix

        Keyword Arguments
        -----------------
        - attr_lndu: Land use attribute table
        """
        
        # get the attribute table and initialize values
        attr_lndu = (
            self.model_attributes.get_attribute_table(
                self.model_attributes.subsec_name_lndu,
            )
            if attr_lndu is None
            else attr_lndu
        )

        # initialize the target using flags and any stable categories
        x_target = self.flag_ignore_constraint*np.ones(attr_lndu.n_key_values, )
        for cat in self.cats_lndu_stable_under_reallocation:
            ind = attr_lndu.get_key_value_index(cat)
            x_target[ind] = x_proj_unadj[ind]
        
        if not isinstance(dict_area_targets_exog, dict):
            return x_target

        # try overwriting entries with dictionary specification
        for k, v in dict_area_targets_exog.items():
            try:
                x_target[k] = v
            except: 
                continue
                
        return x_target



    def qadj_normalize_vector_for_adjustment(self,
        vec: np.ndarray,
        area: float,
    ) -> Union[np.ndarray, None]:
        """
        Normalize an input prevalence vector for use in the optimization. Divides
            non-flag entries by the total area. Returns None if any input is 
            invalid. 
        """
        # check inputs
        return_none = not isinstance(vec, np.ndarray)
        return_none |= not sf.isnumber(area)
        if return_none:
            return None

        # 
        vec_out = vec.copy()
        vec_out[vec_out != self.flag_ignore_constraint] /= area

        return vec_out
    




    ##  LIVESTOCK

    def reassign_pops_from_proj_to_carry(self,
        arr_lu_derived: np.ndarray,
        arr_dem_based: np.ndarray,
    ) -> np.ndarray:
        """
        Before assigning net imports, there are many non-grazing animals to 
            consider (note that these animals are generally not 
            emission-intensive animals).
        
        Due to 0 graze area, their estimated population is infinite, or stored 
            as a negative.

        We assign their population as the demand-estimated population.

        Function Arguments
        ------------------
        - arr_lu_derived: array of animal populations based on land use
        - arr_dem_based: array of animal populations based on demand
        """
        if arr_lu_derived.shape != arr_dem_based.shape:
            raise ValueError(f"Error in reassign_pops_from_proj_to_carry: array dimensions do not match: arr_lu_derived = {arr_lu_derived.shape}, arr_dem_based = {arr_dem_based.shape}.")

        cols = np.where(arr_lu_derived[0] < 0)[0]
        n_row, n_col = arr_lu_derived.shape

        for w in cols:
            rng = np.arange(w*n_row, (w + 1)*n_row)
            np.put(arr_lu_derived.transpose(), rng, arr_dem_based[:, w])

        return arr_lu_derived



    ####################################
    ###                              ###
    ###    PRIMARY MODEL FUNCTION    ###
    ###                              ###
    ####################################

    def project(self,
        df_afolu_trajectories: pd.DataFrame,
        passthrough_tmp: str = None
    ) -> pd.DataFrame:

        """
        The project() method takes a data frame of input variables (ordered by 
            time series) and returns a data frame of output variables (model
            projections for agriculture and livestock, forestry, and land use) 
            the same order.

        Function Arguments
        ------------------
        - df_afolu_trajectories: pd.DataFrame with all required input fields as 
            columns. The model will not run if any required variables are 
            missing, but errors will detail which fields are missing.

        Notes
        -----
        - The .project() method is designed to be parallelized or called from 
            command line via __main__ in run_sector_models.py.
        - df_afolu_trajectories should have all input fields required (see 
            AFOLU.required_variables for a list of variables to be defined)
        - the df_afolu_trajectories.project method will run on valid time 
            periods from 1 .. k, where k <= n (n is the number of time periods). 
            By default, it drops invalid time periods. If there are missing 
            time_periods between the first and maximum, data are interpolated.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_afolu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_afolu_trajectories)

        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_afolu_trajectories)
        (
            dict_dims, 
            df_afolu_trajectories, 
            n_projection_time_periods, 
            projection_time_periods
        ) = self.model_attributes.check_projection_input_df(
            df_afolu_trajectories, 
            True, 
            True, 
            True
        )

        # check integrated variables for HWP
        dict_check_integrated_variables = self.model_attributes.check_integrated_df_vars(
            df_afolu_trajectories, 
            self.dict_integration_variables_by_subsector, 
            "all"
        )


        ##  CATEGORY INITIALIZATION

        # pycat_ABRV is used to access the category elements (in context of 
        # variable schema) in the attribute tables
        pycat_lndu = self.model_attributes.get_subsector_attribute(
            self.subsec_name_lndu, 
            "pycategory_primary_element"
        )
        pycat_lsmm = self.model_attributes.get_subsector_attribute(
            self.subsec_name_lsmm, 
            "pycategory_primary_element"
        )
        pycat_soil = self.model_attributes.get_subsector_attribute(
            self.subsec_name_soil, 
            "pycategory_primary_element"
        )
        
        # attribute tables
        attr_agrc = self.model_attributes.get_attribute_table(self.subsec_name_agrc)
        attr_frst = self.model_attributes.get_attribute_table(self.subsec_name_frst)
        attr_lndu = self.model_attributes.get_attribute_table(self.subsec_name_lndu)
        attr_lsmm = self.model_attributes.get_attribute_table(self.subsec_name_lsmm)
        attr_lvst = self.model_attributes.get_attribute_table(self.subsec_name_lvst)
        attr_soil = self.model_attributes.get_attribute_table(self.subsec_name_soil)


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp,  
            return_type = "array_base",
        )

        vec_gdp_per_capita = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_econ_gdp_per_capita, 
            return_type = "array_base",
        )

        vec_hh = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_grnl_num_hh, 
            return_type = "array_base",
        )

        vec_pop = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.model_socioeconomic.modvar_gnrl_pop_total, 
            return_type = "array_base",
        )

        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_afolu_trajectories[self.required_dimensions].copy()]


        ########################################
        #    LAND USE - UNADJUSTED VARIABLES   #
        ########################################

        # area of the country + the applicable scalar used to convert outputs
        vec_area = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.model_socioeconomic.modvar_gnrl_area,
            return_type = "array_base",
        )

        scalar_lndu_input_area_to_output_area = self.model_attributes.get_scalar(
            self.model_socioeconomic.modvar_gnrl_area, 
            "area"
        )


        ##  INITIAL LAND DISTRIBUTION

        vec_modvar_lndu_initial_frac = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lndu_initial_frac, 
            return_type = "array_base",
        )[0]

        area_init = vec_area[0]
        vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_frac*area_init


        ##  TRANSITION MATRICES AND CARBON FACTORS

        tup = self.get_markov_matrices(
            df_afolu_trajectories, 
            correct_emission_units = True,
            n_tp = n_projection_time_periods,
            return_c_stock_conversion_factors = True,
            target_units_area_modvar = self.model_socioeconomic.modvar_gnrl_area,
        )

        (
            arrs_lndu_q_unadj,
            arrs_lndu_c_agb, 
            arrs_lndu_c_bgb,
            arr_lndu_c_init_agb,
        ) = tup

        
        ##  REALLOCATION FACTOR

        vec_lndu_reallocation_factor = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_reallocation_factor,
            return_type = "array_base",
        )


        ##  AREA CONSTRAINTS BY CLASS

        (
            arr_lndu_constraints_inf,
            arr_lndu_constraints_sup,
        ) = self.get_lndu_class_bounds(
            df_afolu_trajectories,
            vec_modvar_lndu_initial_area,
        )


        ##  FRACTIONS OF INCREASING NET IMPORTS/EXPORTS MET

        arr_lndu_frac_increasing_net_exports_met = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_frac_increasing_net_exports_met,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_lndu_frac_increasing_net_imports_met = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_frac_increasing_net_imports_met,
            expand_to_all_cats = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )


        ##  GET INTEGRATED AGRICULTURE AND LIVESTOCK DEMANDS

        (
            # agrc vars
            arr_agrc_domestic_demand_nonfeed_unadj,
            arr_agrc_exports_unadj,
            arr_agrc_imports_unadj,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            vec_agrc_frac_cropland_area,
            vec_agrc_frac_production_wasted,
            # lvst vars
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_annual_dry_matter_consumption_per_capita,
            arr_lvst_domestic_demand_unadj,
            arr_lvst_exports_unadj,
            arr_lvst_imports_unadj,
            arr_lvst_domestic_production_unadj,
            vec_lvst_carry_capacity_scale,
            vec_lvst_feed_allocation_weights,
            vec_lndu_yf_pasture_sup_adj,
            factor_lndu_init_avg_consumption_pstr,
        ) = self.project_agrc_lvst_integrated_demands(
            df_afolu_trajectories,
            vec_modvar_lndu_initial_area,
            vec_pop,
            vec_rates_gdp_per_capita,
        )
        


        ################################################
        #    CALCULATE LAND USE + AGRC/LVST DRIVERS    #
        ################################################

        # get land use projections (np arrays) - note, arrs_land_conv returns a list of matrices for troubleshooting
        (
            arr_agrc_change_to_net_imports_lost,
            arr_agrc_frac_cropland,
            arr_agrc_net_import_increase,
            arr_agrc_yield,
            arr_lndu_emissions_conv,
            arrs_lndu_emissions_conv_matrices,
            arr_land_use,                             # note: this is in terms of modvar_gnrl_area (Area of Region)
            arr_lvst_change_to_net_imports_lost,
            arr_lvst_net_import_increase,
            arr_lvst_pop,
            arrs_lndu_land_conv,
            arrs_lndu_q_adj,
            self.yields_per_livestock
        ) = self.project_integrated_land_use(
            vec_modvar_lndu_initial_area,
            arrs_lndu_q_unadj,
            arrs_lndu_c_agb,
            arrs_lndu_c_bgb,
            arr_lndu_c_init_agb,
            arr_agrc_production_nonfeed_unadj,
            arr_agrc_yf,
            arr_lndu_constraints_inf,
            arr_lndu_constraints_sup,
            arr_lndu_frac_increasing_net_exports_met,
            arr_lndu_frac_increasing_net_imports_met,
            arr_lndu_yield_i_reqd_lvst_j_init,
            factor_lndu_init_avg_consumption_pstr,
            arr_lvst_annual_dry_matter_consumption_per_capita,
            arr_lvst_domestic_production_unadj,
            vec_agrc_frac_cropland_area,
            vec_lndu_reallocation_factor,
            vec_lvst_carry_capacity_scale,
            vec_area,
            n_tp = n_projection_time_periods,
        )
        self.arrs_lndu_land_conv = arrs_lndu_land_conv
        self.arr_land_use = arr_land_use
        
        



        # update imports/exports for agriculture
        arr_agrc_exports_adj = sf.vec_bounds(
            arr_agrc_exports_unadj - sf.vec_bounds(arr_agrc_net_import_increase, (-np.inf, 0)),
            (np.zeros(arr_agrc_yield.shape), arr_agrc_yield)
        )
        arr_agrc_imports_adj = sf.vec_bounds(
            arr_agrc_imports_unadj + sf.vec_bounds(arr_agrc_net_import_increase, (0, np.inf)),
            (0, np.inf)
        )

        # update demand from population
        arr_lvst_exports_adj = sf.vec_bounds(
            arr_lvst_exports_unadj - sf.vec_bounds(arr_lvst_net_import_increase, (-np.inf, 0)),
            (np.zeros(arr_lvst_pop.shape), arr_lvst_pop)
        )
        arr_lvst_imports_adj = sf.vec_bounds(
            arr_lvst_imports_unadj + sf.vec_bounds(arr_lvst_net_import_increase, (0, np.inf)),
            (0, np.inf)
        )
        arr_lvst_demand = arr_lvst_pop + arr_lvst_imports_adj - arr_lvst_exports_adj

        # assign some dfs that are used below in other subsectors
        df_agrc_frac_cropland = self.model_attributes.array_to_df(
            arr_agrc_frac_cropland, 
            self.modvar_agrc_area_prop_calc
        )

        df_land_use = self.model_attributes.array_to_df(
            arr_land_use, 
            self.modvar_lndu_area_by_cat
        )


        # calculate land use conversions
        arrs_lndu_conv_to = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 0) for x in arrs_lndu_land_conv])
        arrs_lndu_conv_from = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 1) for x in arrs_lndu_land_conv])

        # get total production wasted FLAG!!HEREHERE - check if arr_agrc_production_nonfeed_unadj is correct

        vec_agrc_food_produced_wasted_before_consumption = np.sum(
            arr_agrc_production_nonfeed_unadj.transpose()*vec_agrc_frac_production_wasted, 
            axis = 0,
        )
        vec_agrc_food_produced_wasted_before_consumption *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_total_food_lost_in_ag,
            "mass"
        )
        
        # get total production that is wasted or lost that ends up in landfills
        vec_agrc_frac_production_loss_to_landfills = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_frac_production_loss_to_msw, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        vec_agrc_food_wasted_to_landfills = vec_agrc_food_produced_wasted_before_consumption*vec_agrc_frac_production_loss_to_landfills
        vec_agrc_food_wasted_to_landfills *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_total_food_lost_in_ag,
            self.modvar_agrc_total_food_lost_in_ag_to_msw,
            "mass",
        )


        ##  UNIT CONVERSIONS

        # convert yield out units
        arr_agrc_yield_out = arr_agrc_yield*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_yield,
            "mass"
        )
        # convert exports/imports
        arr_agrc_exports_adj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_adjusted_equivalent_exports,
            "mass"
        )
        arr_agrc_imports_adj *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_adjusted_equivalent_imports,
            "mass"
        )

        # convert change to net imports to yield units
        arr_agrc_change_to_net_imports_lost *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_changes_to_net_imports_lost,
            "mass"
        )
        """
        # JSYME REMOVED 2023-07-13: removing modvar_agrc_net_imports variable
        # self.modvar_agrc_net_imports = "Change to Net Imports of Crops"
        # convert change to net imports loss
        arr_agrc_net_import_increase *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.modvar_agrc_net_imports,
            "mass"
        )
        """;
        # get total domestic crop demand
        arr_agrc_demand_out = arr_agrc_yield_out*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yield,
            self.modvar_agrc_demand_crops,
            "mass"
        )
        arr_agrc_demand_out += arr_agrc_imports_adj*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_adjusted_equivalent_imports,
            self.modvar_agrc_demand_crops,
            "mass"
        )
        arr_agrc_demand_out -= arr_agrc_exports_adj*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_adjusted_equivalent_exports,
            self.modvar_agrc_demand_crops,
            "mass"
        )


        # convert land use conversion areas totals to config area
        df_lndu_area_conv_matrices = self.format_transition_matrix_as_input_dataframe(
            arrs_lndu_land_conv*scalar_lndu_input_area_to_output_area,
            exclude_time_period = True,
            modvar = self.modvar_lndu_area_converted,
        )
        
        # add to output data frame
        df_out += [
            df_agrc_frac_cropland,
            self.model_attributes.array_to_df(
                arr_agrc_change_to_net_imports_lost, 
                self.modvar_agrc_changes_to_net_imports_lost
            ),
            self.model_attributes.array_to_df(
                arr_agrc_demand_out, 
                self.modvar_agrc_demand_crops
            ),
            self.model_attributes.array_to_df(
                arr_agrc_exports_adj, 
                self.modvar_agrc_adjusted_equivalent_exports
            ),
            self.model_attributes.array_to_df(
                arr_agrc_imports_adj, 
                self.modvar_agrc_adjusted_equivalent_imports
            ),
            self.model_attributes.array_to_df(
                arr_agrc_yield_out, 
                self.modvar_agrc_yield
            ),
            self.model_attributes.array_to_df(
                vec_agrc_food_produced_wasted_before_consumption, 
                self.modvar_agrc_total_food_lost_in_ag
            ),
            self.model_attributes.array_to_df(
                vec_agrc_food_wasted_to_landfills, 
                self.modvar_agrc_total_food_lost_in_ag_to_msw
            ),
            self.model_attributes.array_to_df(
                arr_land_use*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_by_cat
            ),
            self.model_attributes.array_to_df(
                arrs_lndu_conv_from*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_converted_from_type
            ),
            self.model_attributes.array_to_df(
                arrs_lndu_conv_to*scalar_lndu_input_area_to_output_area, 
                self.modvar_lndu_area_converted_to_type
            ),
            # matrix of conversion areas
            df_lndu_area_conv_matrices,
            
            self.model_attributes.array_to_df(
                arr_lndu_emissions_conv, 
                self.modvar_lndu_emissions_conv_away, 
                include_scalars = True
            ),

            self.model_attributes.array_to_df(
                arr_lvst_change_to_net_imports_lost, 
                self.modvar_lvst_changes_to_net_imports_lost
            ),
            self.model_attributes.array_to_df(
                arr_lvst_demand, 
                self.modvar_lvst_demand_livestock
            ),
            self.model_attributes.array_to_df(
                arr_lvst_exports_adj, 
                self.modvar_lvst_adjusted_equivalent_exports
            ),
            self.model_attributes.array_to_df(
                arr_lvst_imports_adj, 
                self.modvar_lvst_adjusted_equivalent_imports
            )
        ]



        ####################################
        #    GENERIC LAND USE EMISSIONS    #
        ####################################

        # biomass sequestration by land use type
        arr_lndu_ef_sequestration = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lndu_sf_co2, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )
        arr_lndu_ef_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_lndu_sf_co2,
            "area"
        )
        
        # get land use sequestration in biomass
        arr_lndu_sequestration_co2e = -1*arr_land_use*arr_lndu_ef_sequestration
        arr_lndu_sequestration_co2e *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lndu_sf_co2,
            self.modvar_lndu_emissions_co2_sequestration,
            "mass"
        )

        # get CH4 emissions from wetlands
        arr_lndu_ef_ch4_boc = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lndu_ef_ch4_boc, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected"
        )
        arr_lndu_ef_ch4_boc *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_lndu_ef_ch4_boc,
            "area"
        )

        fields_lndu_for_ch4_boc = self.model_attributes.build_target_variable_fields_from_source_variable_categories(
            self.modvar_lndu_ef_ch4_boc, # source
            self.modvar_lndu_area_by_cat, # target
        )
        arr_lndu_area_ch4_boc = np.array(df_land_use[fields_lndu_for_ch4_boc])

        df_out += [
            # wetland CH4
            self.model_attributes.array_to_df(
                arr_lndu_area_ch4_boc*arr_lndu_ef_ch4_boc, 
                self.modvar_lndu_emissions_ch4_from_wetlands
            ),

            # biomass sequestration
            self.model_attributes.array_to_df(
                arr_lndu_sequestration_co2e, 
                self.modvar_lndu_emissions_co2_sequestration,
                reduce_from_all_cats_to_specified_cats = True,
            )
        ]
        """
        # build output variables
        df_out += [
            self.model_attributes.array_to_df(
                -1*arr_area_frst*arr_frst_ef_sequestration, 
                self.modvar_frst_emissions_co2_sequestration
            ),
            self.model_attributes.array_to_df(
                arr_area_frst*arr_frst_ef_methane, 
                self.modvar_frst_emissions_ch4
            )
        ]
        """



        ##########################################
        #    BUILD SOME SHARED FACTORS (EF_i)    #
        ##########################################

        # agriculture fractions in dry/wet climate
        dict_arrs_agrc_frac_drywet = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_drywet,
            1,
            force_sum_equality = True,
            msg_append = "Agriculture dry/wet fractions by category do not sum to 1. See definition of dict_arrs_agrc_frac_drywet."
        )
        # agriculture fractions in temperate/tropical climate
        dict_arrs_agrc_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Agriculture temperate/tropical fractions by category do not sum to 1. See definition of dict_arrs_agrc_frac_temptrop."
        )
        # forest fractions in temperate/tropical climate
        dict_arrs_frst_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_frst_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Forest temperate NP/temperate NR/tropical fractions by category do not sum to 1. See definition of dict_arrs_frst_frac_temptrop."
        )
        # land use fractions in dry/wet climate
        dict_arrs_lndu_frac_drywet = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lndu_frac_drywet,
            1,
            force_sum_equality = True,
            msg_append = "Land use dry/wet fractions by category do not sum to 1. See definition of dict_arrs_lndu_frac_drywet.",
        )

        # land use fractions in temperate/tropical climate
        dict_arrs_lndu_frac_temptrop = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lndu_frac_temptrop,
            1,
            force_sum_equality = True,
            msg_append = "Land use temperate/tropical fractions by category do not sum to 1. See definition of dict_arrs_lndu_frac_temptrop.",
        )

        ##  BUILD SOME FACTORS

        # get original EF4
        arr_soil_ef4_n_volatilisation = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_soil_ef4_n_volatilisation,
            expand_to_all_cats = True,
            return_type = "array_base",
        )

        # get EF4 for land use categories based on dry/wet (only applies to grassland)
        arr_lndu_ef4_n_volatilisation = 0.0
        for modvar_lndu_frac_drywet in dict_arrs_lndu_frac_drywet.keys():
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar_lndu_frac_drywet, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            arr_lndu_ef4_n_volatilisation += (dict_arrs_lndu_frac_drywet[modvar_lndu_frac_drywet].transpose()*arr_soil_ef4_n_volatilisation[:, ind_soil]).transpose()




        ##################
        #    FORESTRY    #
        ##################

        arr_area_frst = self.get_frst_area_from_df(df_land_use, attr_frst, )

        # TEMPTEMPTEMP
        list_frst_ests = self.get_frst_sequestration_and_land_use_conversion_emission(
            df_afolu_trajectories,
            arrs_lndu_land_conv,
            arrs_lndu_emissions_conv_matrices,
            arr_land_use,
            arr_area_frst,
        )

        df_out += list_frst_ests

        
        ##  FOREST FIRES

        # initialize some variables that are called below
        arr_frst_frac_burned = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_average_fraction_burned_annually,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        arr_frst_ef_co2_fires = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_ef_co2_fires,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        # temperate biomass burned
        arr_frst_biomass_consumed_temperate = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_biomass_consumed_fire_temperate, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        arr_frst_biomass_consumed_temperate *= self.model_attributes.get_scalar(
            self.modvar_frst_biomass_consumed_fire_temperate, 
            "mass"
        )
        arr_frst_biomass_consumed_temperate /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_frst_biomass_consumed_fire_temperate,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )

        # tropical biomass burned
        arr_frst_biomass_consumed_tropical = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_frst_biomass_consumed_fire_tropical, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_frst_biomass_consumed_tropical *= self.model_attributes.get_scalar(
            self.modvar_frst_biomass_consumed_fire_tropical, 
            "mass",
        )
        arr_frst_biomass_consumed_tropical /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_frst_biomass_consumed_fire_tropical,
            self.model_socioeconomic.modvar_gnrl_area,
            "area",
        )

        # setup biomass arrays as a dictionary
        dict_frst_modvar_to_array_forest_fires = {
            self.modvar_frst_frac_temperate_nutrient_poor: arr_frst_biomass_consumed_temperate,
            self.modvar_frst_frac_temperate_nutrient_rich: arr_frst_biomass_consumed_temperate,
            self.modvar_frst_frac_tropical: arr_frst_biomass_consumed_tropical
        }
        # loop over tropical/temperate NP/temperate NR
        arr_frst_emissions_co2_fires = 0.0
        for modvar in self.modvar_list_frst_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            self.model_attributes.extract_model_variable(#
                df_afolu_trajectories, 
                self.modvar_frst_ef_ch4, 
                override_vector_for_single_mv_q = True, 
                return_type = "array_units_corrected",
            )
            # get forest area
            arr_frst_area_temptrop_burned_cur = arr_area_frst*dict_arrs_frst_frac_temptrop[modvar]*arr_frst_frac_burned
            arr_frst_total_dry_mass_burned_cur = arr_frst_area_temptrop_burned_cur*dict_frst_modvar_to_array_forest_fires[modvar]
            arr_frst_emissions_co2_fires += arr_frst_total_dry_mass_burned_cur*arr_frst_ef_co2_fires

        # add to output
        df_out += [
            self.model_attributes.array_to_df(
                np.sum(arr_frst_emissions_co2_fires, axis = 1), 
                self.modvar_frst_emissions_co2_fires
            )
        ]




        #####################
        #    AGRICULTURE    #
        #####################

        # get area of cropland
        field_crop_array = self.model_attributes.build_variable_fields(
            self.modvar_lndu_area_by_cat, 
            restrict_to_category_values = self.cat_lndu_crop,
        )
        vec_cropland_area = np.array(df_land_use[field_crop_array])

        # fraction of cropland represented by each crop
        arr_agrc_frac_cropland_area = self.check_cropland_fractions(df_agrc_frac_cropland, "calculated")
        arr_agrc_crop_area = (arr_agrc_frac_cropland_area.transpose()*vec_cropland_area.transpose()).transpose()
       
        # unit-corrected emission factors - ch4
        arr_agrc_ef_ch4 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_ef_ch4,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )
        arr_agrc_ef_ch4 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_ch4,
            "area"
        )

        # biomass
        arr_agrc_ef_co2_biomass = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_ef_co2_biomass, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )
        arr_agrc_ef_co2_biomass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_co2_biomass,
            "area"
        )
        # biomass burning n2o is dealt with below in "soil management", where crop residues are calculated

        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(
                arr_agrc_crop_area*scalar_lndu_input_area_to_output_area, 
                self.modvar_agrc_area_crop
            ),
            self.model_attributes.array_to_df(
                arr_agrc_ef_ch4*arr_agrc_crop_area, 
                self.modvar_agrc_emissions_ch4_rice, 
                reduce_from_all_cats_to_specified_cats = True
            ),
            self.model_attributes.array_to_df(
                arr_agrc_ef_co2_biomass*arr_agrc_crop_area, 
                self.modvar_agrc_emissions_co2_biomass, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]



        ###################
        #    LIVESTOCK    #
        ###################

        # get area of grassland/pastures
        field_lvst_graze_array = self.model_attributes.build_variable_fields(
            self.modvar_lndu_area_by_cat, 
            restrict_to_category_values = self.cat_lndu_grss,
        )
        vec_lvst_graze_area = np.array(df_land_use[field_lvst_graze_array])

        # estimate the total number of livestock that are raised - arr_lvst_pop is a direct output of project_integrated_land_use
        arr_lvst_total_weight = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_animal_weight,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        arr_lvst_total_animal_mass = arr_lvst_pop*arr_lvst_total_weight
        arr_lvst_aggregate_animal_mass = np.sum(arr_lvst_total_animal_mass, axis = 1)
        arr_lvst_aggregate_animal_mass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lvst_total_animal_mass,
            "mass"
        )
        # get the enteric fermentation emission factor
        arr_lvst_emissions_ch4_ef = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_ef_ch4_ef, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected",
        )

        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(
                arr_lvst_emissions_ch4_ef*arr_lvst_pop, 
                self.modvar_lvst_emissions_ch4_ef
            ),

            self.model_attributes.array_to_df(
                arr_lvst_pop, 
                self.modvar_lvst_pop,
            ),

            self.model_attributes.array_to_df(
                arr_lvst_aggregate_animal_mass, 
                self.modvar_lvst_total_animal_mass
            )
        ]


        ##  MANURE MANAGEMENT DATA

        # nitrogen and volative solids generated (passed to manure management--unitless, so they take the mass of modvar_lvst_animal_weight)
        arr_lvst_nitrogen = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_genfactor_nitrogen,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )
        arr_lvst_nitrogen *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")

        arr_lvst_volatile_solids = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_genfactor_volatile_solids,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )
        arr_lvst_volatile_solids *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")

        arr_lvst_b0 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_b0_manure_ch4,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_units_corrected_gas",
        )
        # get ratio of n to volatile solids
        arr_lvst_ratio_vs_to_n = arr_lvst_volatile_solids/arr_lvst_nitrogen


        #####################################
        #    LIVESTOCK MANURE MANAGEMENT    #
        #####################################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_lsmm_frac_manure = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lvst_mm_fractions,
            1,
            force_sum_equality = True,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_inen_frac_energy.",
        )

        # get variables that can be indexed below 
        arr_lsmm_ef_direct_n2o = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_ef_direct_n2o,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        arr_lsmm_frac_lost_leaching = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_loss_leaching,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_lsmm_frac_lost_volatilisation = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_loss_volatilisation,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_lsmm_frac_used_for_fertilizer = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_frac_n_available_used,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_lsmm_mcf_by_pathway = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_mcf_by_pathway,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        arr_lsmm_n_from_bedding = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_n_from_bedding,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        arr_lsmm_n_from_codigestates = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_n_from_codigestates,
            expand_to_all_cats = True,  
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            var_bounds = (0, np.inf),
        )

        arr_lsmm_rf_biogas = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_rf_biogas,
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        vec_lsmm_frac_n_in_dung = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lvst_frac_exc_n_in_dung, 
            return_type = "array_base",
            var_bounds = (0, 1),
        )

        vec_lsmm_ratio_n2_to_n2o = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lsmm_ratio_n2_to_n2o, 
            return_type = "array_base",
        )

        # soil EF4/EF5 from Table 11.3 - use average fractions from grasslands
        vec_soil_ef_ef4 = attr_lndu.get_key_value_index(self.cat_lndu_grss)
        vec_soil_ef_ef4 = arr_lndu_ef4_n_volatilisation[:, vec_soil_ef_ef4]
        vec_soil_ef_ef5 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef5_n_leaching, 
            return_type = "array_base"
        )

        # convert bedding/co-digestates to animal weight masses
        arr_lsmm_n_from_bedding *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_from_bedding,
            self.modvar_lvst_animal_weight,
            "mass"
        )

        # initialize output arrays
        arr_lsmm_biogas_recovered = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_ch4 = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_n2o_direct = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_n2o_indirect = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_nitrogen_available = np.zeros(arr_lsmm_ef_direct_n2o.shape)

        # initialize some aggregations
        vec_lsmm_nitrogen_to_other = 0.0
        vec_lsmm_nitrogen_to_fertilizer_dung = 0.0
        vec_lsmm_nitrogen_to_fertilizer_urine = 0.0
        vec_lsmm_nitrogen_to_pasture = 0.0

        # categories that allow for manure retrieval and use in fertilizer
        cats_lsmm_manure_retrieval = self.model_attributes.get_variable_categories(self.modvar_lsmm_frac_n_available_used)

        # loop over manure pathways to
        for var_lvst_mm_frac in self.modvar_list_lvst_mm_fractions:
            # get the current variable
            arr_lsmm_fracs_by_lvst = dict_arrs_lsmm_frac_manure[var_lvst_mm_frac]
            arr_lsmm_total_nitrogen_cur = arr_lvst_nitrogen*arr_lsmm_fracs_by_lvst

            # retrieve the livestock management category
            cat_lsmm = clean_schema(
                self.model_attributes.get_variable_attribute(
                    var_lvst_mm_frac, 
                    pycat_lsmm
                )
            )

            index_cat_lsmm = attr_lsmm.get_key_value_index(cat_lsmm)


            ##  METHANE EMISSIONS

            # get MCF, b0, and total volatile solids - USE EQ. 10.23
            vec_lsmm_mcf_cur = arr_lsmm_mcf_by_pathway[:, index_cat_lsmm]
            arr_lsmm_emissions_ch4_cur = arr_lvst_b0*arr_lvst_volatile_solids*arr_lsmm_fracs_by_lvst
            arr_lsmm_emissions_ch4_cur = (arr_lsmm_emissions_ch4_cur.transpose()*vec_lsmm_mcf_cur).transpose()
            
            # get biogas recovery
            arr_lsmm_biogas_recovered_cur = (arr_lsmm_emissions_ch4_cur.transpose()*arr_lsmm_rf_biogas[:, index_cat_lsmm]).transpose()
            arr_lsmm_emissions_ch4_cur -= arr_lsmm_biogas_recovered_cur
            arr_lsmm_biogas_recovered[:, index_cat_lsmm] = np.sum(arr_lsmm_biogas_recovered_cur, axis = 1)
            
            # adjust
            arr_lsmm_emissions_ch4_cur *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")
            arr_lsmm_emission_ch4[:, index_cat_lsmm] = np.sum(arr_lsmm_emissions_ch4_cur, axis = 1)


            ##  NITROGEN EMISSIONS AND FERTILIZER AVAILABILITY

            # get total nitrogen deposited
            vec_lsmm_nitrogen_treated_cur = np.sum(arr_lsmm_total_nitrogen_cur, axis = 1)
            vec_lsmm_n_from_bedding = arr_lsmm_n_from_bedding[:, index_cat_lsmm]
            vec_lsmm_n_from_codigestates = arr_lsmm_n_from_codigestates[:, index_cat_lsmm]

            # get nitrogen from bedding per animal
            vec_lsmm_n_from_bedding *= np.sum(arr_lvst_pop*arr_lsmm_fracs_by_lvst, axis = 1)

            # get totals lost to different pathways
            vec_lsmm_frac_lost_direct = sf.vec_bounds((1 + vec_lsmm_ratio_n2_to_n2o)*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm], (0, 1))
            vec_lsmm_frac_lost_leaching = arr_lsmm_frac_lost_leaching[:, index_cat_lsmm]
            vec_lsmm_frac_lost_volatilisation = arr_lsmm_frac_lost_volatilisation[:, index_cat_lsmm]

            # apply the limiter, which prevents their total from exceeding 1
            vec_lsmm_frac_lost_direct, vec_lsmm_frac_lost_leaching, vec_lsmm_frac_lost_volatilisation = sf.vector_limiter(
                [
                    vec_lsmm_frac_lost_direct,
                    vec_lsmm_frac_lost_leaching,
                    vec_lsmm_frac_lost_volatilisation
                ],
                (0, 1)
            )
            vec_lsmm_frac_loss_ms = vec_lsmm_frac_lost_leaching + vec_lsmm_frac_lost_volatilisation + vec_lsmm_frac_lost_direct
            vec_lsmm_n_lost = vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates)*self.factor_n2on_to_n2o

            # 10.25 FOR DIRECT EMISSIONS
            arr_lsmm_emission_n2o_direct[:, index_cat_lsmm] = vec_lsmm_n_lost*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm]
            # 10.28 FOR LOSSES DUE TO VOLATILISATION
            arr_lsmm_emission_n2o_indirect[:, index_cat_lsmm] = vec_lsmm_n_lost*vec_soil_ef_ef4*vec_lsmm_frac_lost_volatilisation
            # 10.29 FOR LOSSES DUE TO LEACHING
            arr_lsmm_emission_n2o_indirect[:, index_cat_lsmm] += vec_lsmm_n_lost*vec_soil_ef_ef5*vec_lsmm_frac_lost_leaching
            # BASED ON EQ. 10.34A in IPCC GNGHGI 2019R FOR NITROGEN AVAILABILITY - note: co-digestates are entered as an inflation factor
            vec_lsmm_nitrogen_available = (vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates))*(1 - vec_lsmm_frac_loss_ms) + vec_lsmm_n_from_bedding

            # check categories
            if cat_lsmm in cats_lsmm_manure_retrieval:
                if cat_lsmm == self.cat_lsmm_incineration:

                    ##  MANURE (VOLATILE SOLIDS) FOR INCINERATION:

                    vec_lsmm_volatile_solids_incinerated = np.sum(arr_lvst_volatile_solids*arr_lsmm_fracs_by_lvst, axis = 1)
                    vec_lsmm_volatile_solids_incinerated *= self.model_attributes.get_variable_unit_conversion_factor(
                        self.modvar_lvst_animal_weight,
                        self.modvar_lsmm_dung_incinerated,
                        "mass"
                    )

                    ##  N2O WORK

                    # if incinerating, send urine nitrogen to pasture
                    vec_lsmm_nitrogen_to_pasture += vec_lsmm_nitrogen_available*(1 - vec_lsmm_frac_n_in_dung)
                    # get incinerated N in dung - not used yet
                    vec_lsmm_nitrogen_to_incinerator = vec_lsmm_nitrogen_available*vec_lsmm_frac_n_in_dung
                    vec_lsmm_nitrogen_to_incinerator *= self.model_attributes.get_variable_unit_conversion_factor(
                        self.modvar_lvst_animal_weight,
                        self.modvar_lsmm_dung_incinerated,
                        "mass"
                    )

                    # add to output
                    df_out += [
                        self.model_attributes.array_to_df(vec_lsmm_volatile_solids_incinerated, self.modvar_lsmm_dung_incinerated)
                    ]

                else:
                    # account for fraction used for fertilizer
                    vec_lsmm_nitrogen_cur = vec_lsmm_nitrogen_available*arr_lsmm_frac_used_for_fertilizer[:, index_cat_lsmm]
                    vec_lsmm_nitrogen_to_other += vec_lsmm_nitrogen_available - vec_lsmm_nitrogen_cur
                    # add to total by animal and splits by dung/urea (used in Soil Management subsector)
                    arr_lsmm_nitrogen_available[:, index_cat_lsmm] += vec_lsmm_nitrogen_cur
                    vec_lsmm_nitrogen_to_fertilizer_dung += vec_lsmm_nitrogen_cur*vec_lsmm_frac_n_in_dung
                    vec_lsmm_nitrogen_to_fertilizer_urine += vec_lsmm_nitrogen_cur*(1 - vec_lsmm_frac_n_in_dung)

            elif cat_lsmm == self.cat_lsmm_pasture:
                vec_lsmm_nitrogen_to_pasture += vec_lsmm_nitrogen_available


        ##  UNITS CONVERSTION

        # biogas recovery
        arr_lsmm_biogas_recovered *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_recovered_biogas,
            "mass"
        )
        # total nitrogen available for fertilizer by pathway
        arr_lsmm_nitrogen_available *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer,
            "mass"
        )
        # total nitrogen available for other uses by pathway
        vec_lsmm_nitrogen_to_other *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_other_use,
            "mass"
        )
        # total nitrogen sent to pasture
        vec_lsmm_nitrogen_to_pasture *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_pastures,
            "mass"
        )
        # nitrogen available from dung/urea
        vec_lsmm_nitrogen_to_fertilizer_dung *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_lsmm_nitrogen_to_fertilizer_urine *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_urine,
            "mass"
        )
        # n2o emissions - direct and indirect
        scalar_lsmm_n2o_di = self.model_attributes.get_scalar(self.modvar_lsmm_emissions_direct_n2o, "gas")
        scalar_lsmm_n2o_di *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")
        arr_lsmm_emission_n2o_direct *= scalar_lsmm_n2o_di
        arr_lsmm_emission_n2o_indirect *= scalar_lsmm_n2o_di

        df_out += [
            self.model_attributes.array_to_df(
                arr_lsmm_emission_ch4, 
                self.modvar_lsmm_emissions_ch4
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_emission_n2o_direct, 
                self.modvar_lsmm_emissions_direct_n2o,
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_emission_n2o_indirect, 
                self.modvar_lsmm_emissions_indirect_n2o,
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_pasture, 
                self.modvar_lsmm_n_to_pastures
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_nitrogen_available, 
                self.modvar_lsmm_n_to_fertilizer
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_other, 
                self.modvar_lsmm_n_to_other_use
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_fertilizer_dung, 
                self.modvar_lsmm_n_to_fertilizer_agg_dung
            ),
            self.model_attributes.array_to_df(
                vec_lsmm_nitrogen_to_fertilizer_urine, 
                self.modvar_lsmm_n_to_fertilizer_agg_urine
            ),
            self.model_attributes.array_to_df(
                arr_lsmm_biogas_recovered, 
                self.modvar_lsmm_recovered_biogas, 
                reduce_from_all_cats_to_specified_cats = True
            )
        ]




        #############################
        ###                       ###
        ###    SOIL MANAGEMENT    ###
        ###                       ###
        #############################

        # get inital demand for fertilizer N - start with area of land receiving fertilizer (grasslands and croplands)
        # put units in terms of modvar_lsmm_n_to_fertilizer_agg_dung
        #
        vec_soil_init_n_fertilizer_synthetic = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_fertuse_init_synthetic, 
            return_type = "array_base",
        )
        vec_soil_init_n_fertilizer_synthetic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_init_synthetic,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )

        vec_soil_n_fertilizer_use_organic = vec_lsmm_nitrogen_to_fertilizer_urine
        vec_soil_n_fertilizer_use_organic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_urine,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_soil_n_fertilizer_use_organic += vec_lsmm_nitrogen_to_fertilizer_dung
        vec_soil_init_n_fertilizer_total = vec_soil_init_n_fertilizer_synthetic + vec_soil_n_fertilizer_use_organic
        
        # get land that's fertilized and use to project fertilizer demand - add in fraction of grassland that is pasture
        arr_lndu_frac_fertilized = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_lndu_frac_fertilized, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1)
        )

        vec_soil_demscalar_fertilizer = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_demscalar_fertilizer, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )
        

        # estimate fertilizer demand
        vec_soil_area_fertilized = np.sum(arr_lndu_frac_fertilized*arr_land_use, axis = 1)
        vec_soil_n_fertilizer_use_total = np.concatenate([
            np.ones(1), 
            np.cumprod(vec_soil_area_fertilized[1:]/vec_soil_area_fertilized[0:-1])
        ])
        vec_soil_n_fertilizer_use_total *= vec_soil_demscalar_fertilizer
        vec_soil_n_fertilizer_use_total *= vec_soil_init_n_fertilizer_total[0]
        
        # estimate synthetic fertilizer demand - send extra manure back to pasture/paddock/range treatment flow
        vec_soil_n_fertilizer_use_synthetic = vec_soil_n_fertilizer_use_total - vec_soil_n_fertilizer_use_organic
        vec_soil_n_fertilizer_use_organic_to_pasture = sf.vec_bounds(vec_soil_n_fertilizer_use_synthetic, (0, np.inf))
        vec_soil_n_fertilizer_use_organic_to_pasture -= vec_soil_n_fertilizer_use_synthetic
        vec_soil_n_fertilizer_use_synthetic = sf.vec_bounds(vec_soil_n_fertilizer_use_synthetic, (0, np.inf))
        
        # split synthetic fertilizer use up
        vec_soil_frac_synthetic_fertilizer_urea = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_frac_synethic_fertilizer_urea,
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        vec_soil_n_fertilizer_use_synthetic_urea = vec_soil_n_fertilizer_use_synthetic*vec_soil_frac_synthetic_fertilizer_urea
        vec_soil_n_fertilizer_use_synthetic_nonurea = vec_soil_n_fertilizer_use_synthetic - vec_soil_n_fertilizer_use_synthetic_urea


        arr_soil_ef1_organic = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef1_n_managed_soils_org_fert,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        arr_soil_ef1_synthetic = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef1_n_managed_soils_syn_fert, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        vec_soil_ef1_rice = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_soil_ef1_n_managed_soils_rice, 
            return_type = "array_base",
        )



        ##############################################################
        #    N2O DIRECT - INPUT EMISSIONS (PT. 1 OF EQUATION 11.1)   #
        ##############################################################

        ##  SOME SHARED VARIABLES

        # get crop components of synthetic and organic fertilizers for ef1 (will overwrite rice)
        ind_rice = attr_agrc.get_key_value_index(self.cat_agrc_rice)

        # some variables
        arr_lndu_frac_mineral_soils = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_frac_mineral_soils, 
            expand_to_all_cats = True, 
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        # HEREHEREHERE
        arr_lndu_frac_organic_soils = sf.vec_bounds(1 - arr_lndu_frac_mineral_soils, (0.0, 1.0))
        vec_soil_area_crop_pasture = (
            arr_land_use[:, [self.ind_lndu_crop, self.ind_lndu_pstr]]
            .sum(axis = 1)
        )
        

        ##  F_ON AND F_SN - SYNTHETIC FERTILIZERS AND ORGANIC AMENDMENTS

        # initialize some components
        dict_soil_fertilizer_application_by_climate_organic = {}
        dict_soil_fertilizer_application_by_climate_synthetic = {}
        vec_soil_n2odirectn_fon = 0.0
        vec_soil_n2odirectn_fsn = 0.0
        vec_soil_n2odirectn_fon_rice = 0.0
        vec_soil_n2odirectn_fsn_rice = 0.0

        # crop component
        for modvar in self.modvar_list_agrc_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            
            # get current factors
            arr_agrc_cur_wetdry_fertilized_crop = dict_arrs_agrc_frac_drywet[modvar]*arr_agrc_crop_area
            arr_agrc_cur_wetdry_fertilized_crop = arr_agrc_cur_wetdry_fertilized_crop.transpose()*arr_lndu_frac_fertilized[:, self.ind_lndu_crop]
            
            # get fraction of fertilized land represented by current area of cropland
            arr_soil_frac_cur_drywet_crop = (arr_agrc_cur_wetdry_fertilized_crop/vec_soil_area_fertilized)
            arr_soil_frac_cur_drywet_crop_organic = arr_soil_frac_cur_drywet_crop*vec_soil_n_fertilizer_use_organic
            arr_soil_frac_cur_drywet_crop_synthetic = arr_soil_frac_cur_drywet_crop*vec_soil_n_fertilizer_use_synthetic
            
            # update the dictionary for use later
            dict_soil_fertilizer_application_by_climate_organic.update({cat_soil: np.sum(arr_soil_frac_cur_drywet_crop_organic, axis = 0)})
            dict_soil_fertilizer_application_by_climate_synthetic.update({cat_soil: np.sum(arr_soil_frac_cur_drywet_crop_synthetic, axis = 0)})
            
            # get rice components
            vec_soil_n2odirectn_fon_rice += arr_soil_frac_cur_drywet_crop_organic[ind_rice, :]*vec_soil_ef1_rice
            vec_soil_n2odirectn_fsn_rice += arr_soil_frac_cur_drywet_crop_synthetic[ind_rice, :]*vec_soil_ef1_rice
            
            # remove rice and carry on
            arr_soil_frac_cur_drywet_crop_organic[ind_rice, :] = 0.0
            arr_soil_frac_cur_drywet_crop_synthetic[ind_rice, :] = 0.0
            vec_soil_n2odirectn_fon += np.sum(arr_soil_frac_cur_drywet_crop_organic, axis = 0)*arr_soil_ef1_organic[:, ind_soil]
            vec_soil_n2odirectn_fsn += np.sum(arr_soil_frac_cur_drywet_crop_synthetic, axis = 0)*arr_soil_ef1_synthetic[:, ind_soil]


        # pasture component
        for modvar in self.modvar_list_lndu_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            
            # get current factors
            vec_soil_cur_wetdry_fertilized_pstr = dict_arrs_lndu_frac_drywet[modvar]*arr_land_use
            vec_soil_cur_wetdry_fertilized_pstr = (vec_soil_cur_wetdry_fertilized_pstr*arr_lndu_frac_fertilized)[:, self.ind_lndu_pstr]
            
            # get fraction of fertilized land represented by current area of cropland
            vec_soil_frac_cur_drywet_pstr = (vec_soil_cur_wetdry_fertilized_pstr/vec_soil_area_fertilized)
            vec_soil_n2odirectn_fon += (vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_organic)*arr_soil_ef1_organic[:, ind_soil]
            vec_soil_n2odirectn_fsn += (vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_synthetic)*arr_soil_ef1_synthetic[:, ind_soil]
            
            # update the dictionary for use later
            v_cur = dict_soil_fertilizer_application_by_climate_organic[cat_soil].copy()
            dict_soil_fertilizer_application_by_climate_organic.update({
                cat_soil: v_cur + vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_organic
            })
            v_cur = dict_soil_fertilizer_application_by_climate_synthetic[cat_soil].copy()
            dict_soil_fertilizer_application_by_climate_synthetic.update({
                cat_soil: v_cur + vec_soil_frac_cur_drywet_pstr*vec_soil_n_fertilizer_use_synthetic
            })


        ##  F_CR - CROP RESIDUES

        arr_soil_yield = arr_agrc_yield*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yield,
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )
        arr_soil_crop_area = arr_agrc_crop_area*self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_regression_m_above_ground_residue,
            "area"
        )

        # get the regression information
        arr_agrc_regression_m = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_regression_m_above_ground_residue, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_agrc_regression_b = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_regression_b_above_ground_residue,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_agrc_regression_b *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_b_above_ground_residue,
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass",
        )

        arr_agrc_regression_b /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_b_above_ground_residue,
            self.modvar_agrc_regression_m_above_ground_residue,
            "area",
        )

        # get crop dry matter
        arr_agrc_crop_frac_dry_matter = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_frac_dry_matter_in_crop, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_agrc_crop_drymatter_per_unit = np.nan_to_num(arr_soil_yield/arr_soil_crop_area, nan = 0.0, posinf = 0.0, )
        arr_agrc_crop_drymatter_per_unit = arr_agrc_regression_m*arr_agrc_crop_drymatter_per_unit + arr_agrc_regression_b
        arr_agrc_crop_drymatter_above_ground = arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area
        
        # get fraction removed/burned
        dict_agrc_frac_residues_removed_burned = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_agrc_frac_residues_removed_burned,
            1,
            force_sum_equality = False,
            msg_append = "Agriculture crop residue fractions by exceed 1. See definition of dict_agrc_frac_residues_removed_burned."
        )
        vec_agrc_frac_residue_burned = dict_agrc_frac_residues_removed_burned.get(self.modvar_agrc_frac_residues_burned).flatten()
        vec_agrc_frac_residue_removed = dict_agrc_frac_residues_removed_burned.get(self.modvar_agrc_frac_residues_removed).flatten()
        arr_agrc_combustion_factor = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_combustion_factor,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
            
        # get n available in above ground/below ground residues
        arr_agrc_n_content_ag_residues = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_n_content_of_above_ground_residues,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_agrc_n_content_bg_residues = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_n_content_of_below_ground_residues,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )
        
        # get total n HERE IS TOTAL N BURNED FROM CROP RESIDUE (in terms of modvar_agrc_regression_m_above_ground_residue)
        vec_agrc_total_n_residue_burned = np.sum(arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues, axis = 1)*vec_agrc_frac_residue_burned
        arr_agrc_total_n_residue_removed = (arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues).transpose()*vec_agrc_frac_residue_removed
        arr_agrc_total_n_above_ground_residues_burncomponent = (arr_agrc_crop_drymatter_above_ground*arr_agrc_combustion_factor*arr_agrc_n_content_ag_residues).transpose()*vec_agrc_frac_residue_burned
        arr_agrc_total_n_above_ground_residues = (arr_agrc_crop_drymatter_above_ground*arr_agrc_n_content_ag_residues).transpose() - arr_agrc_total_n_residue_removed - arr_agrc_total_n_above_ground_residues_burncomponent
        
        # get dry/wet and rice residuces
        vec_agrc_total_n_above_ground_residues_rice = arr_agrc_total_n_above_ground_residues[ind_rice, :].copy()
        arr_agrc_total_n_above_ground_residues[ind_rice, :] = 0
        vec_agrc_total_n_above_ground_residues_dry = np.sum(arr_agrc_total_n_above_ground_residues.transpose()*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_dry], axis = 1)
        vec_agrc_total_n_above_ground_residues_wet = np.sum(arr_agrc_total_n_above_ground_residues.transpose()*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_wet], axis = 1)
        
        # move to below ground and get total biomass (used for biomass burning)
        arr_agrc_ratio_bg_biomass_to_ag_biomass = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_agrc_bg_biomass = (arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area + arr_soil_yield)*arr_agrc_ratio_bg_biomass_to_ag_biomass
        vec_agrc_crop_residue_biomass = np.sum(
            arr_agrc_crop_drymatter_per_unit*arr_soil_crop_area + arr_agrc_bg_biomass, 
            axis = 1,
        )
       
        # get n from below ground residues
        arr_agrc_total_n_below_ground_residues = arr_agrc_bg_biomass*arr_agrc_n_content_bg_residues
        vec_agrc_total_n_below_ground_residues_rice = arr_agrc_total_n_below_ground_residues[:, ind_rice].copy()
        arr_agrc_total_n_below_ground_residues[:, ind_rice] = 0
        vec_agrc_total_n_below_ground_residues_dry = np.sum(
            arr_agrc_total_n_below_ground_residues*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_dry], 
            axis = 1,
        )
        vec_agrc_total_n_below_ground_residues_wet = np.sum(
            arr_agrc_total_n_below_ground_residues*dict_arrs_agrc_frac_drywet[self.modvar_agrc_frac_wet], 
            axis = 1,
        )
        
        # get total crop residue and conver to units of F_ON and F_SN
        scalar_soil_residue_to_fertilizer_equivalent = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_regression_m_above_ground_residue,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_agrc_total_n_residue_dry = (vec_agrc_total_n_above_ground_residues_dry + vec_agrc_total_n_below_ground_residues_dry)*scalar_soil_residue_to_fertilizer_equivalent
        vec_agrc_total_n_residue_wet = (vec_agrc_total_n_above_ground_residues_wet + vec_agrc_total_n_below_ground_residues_wet)*scalar_soil_residue_to_fertilizer_equivalent
        vec_agrc_total_n_residue_rice = (vec_agrc_total_n_above_ground_residues_rice + vec_agrc_total_n_below_ground_residues_rice)*scalar_soil_residue_to_fertilizer_equivalent
        
        # finally, get ef1 component
        dict_agrc_modvar_to_n_residue = {
            self.modvar_agrc_frac_dry: vec_agrc_total_n_residue_dry,
            self.modvar_agrc_frac_wet: vec_agrc_total_n_residue_wet,
        }
        vec_soil_n2odirectn_fcr = 0.0
        vec_soil_n2odirectn_fcr_rice = vec_agrc_total_n_residue_rice*vec_soil_ef1_rice

        # loop over dry/wet
        for modvar in self.modvar_list_agrc_frac_drywet:
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_n2odirectn_fcr += dict_agrc_modvar_to_n_residue[modvar]*arr_soil_ef1_organic[:, ind_soil]


        ##  ADD IN BIOMASS BURNING (FROM EQ. 2.27 & V4 SECTION 5.3.4.1)

        # dimensionless, buin terms of modvar_agrc_regression_m_above_ground_residue
        vec_agrc_ef_ch4_biomass_burning = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_ef_ch4_burning, 
            return_type = "array_units_corrected",
        )

        vec_agrc_ef_n2o_biomass_burning = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_ef_n2o_burning, 
            return_type = "array_units_corrected",
        )

        vec_agrc_crop_residue_burned = vec_agrc_crop_residue_biomass*vec_agrc_frac_residue_burned

        # get average combustion factor, or fraction of crops burned by fire
        vec_agrc_avg_combustion_factor = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_agrc_combustion_factor, 
            expand_to_all_cats = False,
            override_vector_for_single_mv_q = True,
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        cats_agrc_avg_combustion_factor = self.model_attributes.get_variable_categories(self.modvar_agrc_combustion_factor)
        inds_agrc_avg_combustion_factor = [attr_agrc.get_key_value_index(x) for x in cats_agrc_avg_combustion_factor]
        vec_agrc_avg_combustion_factor = np.sum(vec_agrc_avg_combustion_factor*arr_agrc_crop_area[:, inds_agrc_avg_combustion_factor], axis = 1)
        vec_agrc_avg_combustion_factor /= np.sum(arr_agrc_crop_area[:, inds_agrc_avg_combustion_factor], axis = 1)

        # multiply by combustion factor to get final mass of crops burned
        vec_agrc_crop_residue_burned *= vec_agrc_avg_combustion_factor

        # get estimate of emissions of ch4
        vec_agrc_emissions_ch4_biomass_burning = vec_agrc_crop_residue_burned*vec_agrc_ef_ch4_biomass_burning
        vec_agrc_emissions_ch4_biomass_burning *= self.model_attributes.get_scalar(
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )

        # get estimate of emissions of n2o
        vec_agrc_emissions_n2o_biomass_burning = vec_agrc_crop_residue_burned*vec_agrc_ef_n2o_biomass_burning
        vec_agrc_emissions_n2o_biomass_burning *= self.model_attributes.get_scalar(
            self.modvar_agrc_regression_m_above_ground_residue,
            "mass"
        )
        
        # add to output
        df_out += [
            # CH4 FROM BIOMASS BURNING
            self.model_attributes.array_to_df(
                vec_agrc_emissions_ch4_biomass_burning, 
                self.modvar_agrc_emissions_ch4_biomass_burning
            ),
            # N2O EMISSIONS FROM BIOMASS BURNING
            self.model_attributes.array_to_df(
                vec_agrc_emissions_n2o_biomass_burning, 
                self.modvar_agrc_emissions_n2o_biomass_burning
            )
        ]


        ##  F_SOM AND F_SO (DRAINED ORGANIC SOILS)

        # get soil management factors
        arr_lndu_factor_soil_management, arr_lndu_area_improved = self.get_lndu_soil_soc_factors(
            df_afolu_trajectories,
            arr_land_use,
            arr_agrc_frac_cropland,
            dict_agrc_frac_residues_removed_burned,
        )

        df_out += [
            self.model_attributes.array_to_df(
                arr_lndu_area_improved*scalar_lndu_input_area_to_output_area,
                self.modvar_lndu_area_improved,
                reduce_from_all_cats_to_specified_cats = True,
            )
        ]

        # get carbon stocks and ratio of c to n - convert to self.modvar_lsmm_n_to_fertilizer_agg_dung units for N2O/EF1
        arr_lndu_factor_soil_carbon = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_lndu_factor_soil_carbon,
            expand_to_all_cats = True,
            return_type = "array_base",
        )
        
        arr_soil_organic_c_stocks = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_soil_organic_c_stocks,
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
        )

        arr_soil_organic_c_stocks *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_organic_c_stocks,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass",
        )

        arr_soil_organic_c_stocks /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_organic_c_stocks,
            self.model_socioeconomic.modvar_gnrl_area,
            "area",
        )

        # get some other factors
        vec_soil_ratio_c_to_n_soil_organic_matter = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories,
            self.modvar_soil_ratio_c_to_n_soil_organic_matter,
            return_type = "array_base",
        )


        # get arrays of SOC conversion per area by land use
        arrs_lndu_soc_conversion_factors, vec_soil_ef1_soc_est = self.get_mineral_soc_change_matrices(
            arr_agrc_crop_area,
            arr_land_use,
            arr_lndu_factor_soil_carbon,
            arr_lndu_factor_soil_management,
            arr_lndu_frac_mineral_soils,
            arr_soil_ef1_organic,
            arr_soil_organic_c_stocks,
            vec_soil_area_crop_pasture,
            dict_arrs_agrc_frac_drywet,
            dict_arrs_frst_frac_temptrop,
            dict_arrs_lndu_frac_drywet
        )

        vec_soil_delta_soc_mineral = self.calculate_soc_stock_change_with_time_dependence(
            arrs_lndu_land_conv,
            arrs_lndu_soc_conversion_factors,
            20, # get from config HEREHERE
        )


        """
        Alternate approach to calculating SOC stock changes
        # calculate the change in soil carbon year over year for all and for mineral
        vec_soil_delta_soc = self.calculate_ipcc_soc_deltas(vec_soil_soc_total, 2)
        vec_soil_delta_soc_mineral = self.calculate_ipcc_soc_deltas(vec_soil_soc_total_mineral, 2)
        vec_soil_ratio_c_to_n_soil_organic_matter = vec_soil_ratio_c_to_n_soil_organic_matter
        """;

        # calculate FSOM from fraction mineral
        vec_soil_n2odirectn_fsom = -(sf.vec_bounds(vec_soil_delta_soc_mineral, (-np.inf, 0))/vec_soil_ratio_c_to_n_soil_organic_matter)*vec_soil_ef1_soc_est
        vec_soil_emission_co2_soil_carbon_mineral = -self.factor_c_to_co2*vec_soil_delta_soc_mineral
        vec_soil_emission_co2_soil_carbon_mineral *= self.model_attributes.get_scalar(self.modvar_lsmm_n_to_fertilizer_agg_dung, "mass")
        vec_soil_emission_co2_soil_carbon_mineral *= self.model_attributes.get_gwp("co2")


        ##  FINAL EF1 COMPONENTS

        # different tablulations (totals will run across EF1, EF2, EF3, EF4, and EF5)
        vec_soil_n2on_direct_input = (
            vec_soil_n2odirectn_fon 
            + vec_soil_n2odirectn_fon_rice 
            + vec_soil_n2odirectn_fsn 
            + vec_soil_n2odirectn_fsn_rice 
            + vec_soil_n2odirectn_fcr 
            + vec_soil_n2odirectn_fcr_rice 
            + vec_soil_n2odirectn_fsom
        )
        vec_soil_emission_n2o_crop_residue = vec_soil_n2odirectn_fcr + vec_soil_n2odirectn_fcr_rice
        vec_soil_emission_n2o_fertilizer = (
            vec_soil_n2odirectn_fon 
            + vec_soil_n2odirectn_fon_rice 
            + vec_soil_n2odirectn_fsn 
            + vec_soil_n2odirectn_fsn_rice
        )
        vec_soil_emission_n2o_mineral_soils = vec_soil_n2odirectn_fsom




        

        ########################################################################
        #    DRAINED ORGANIC SOILS                                             #
        #    - CO2 EMISSIONS IN CROPLANDS AND MANAGED GRASSLANDS               #
        #    - N2O DIRECT - ORGANIC SOIL EMISSIONS (PT. 2 OF EQUATION 11.1)    #
        ########################################################################

        # N2O emission factor variable (EF2)
        arr_soil_ef2 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef2_n_organic_soils, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        arr_soil_ef2 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_ef2_n_organic_soils,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        arr_soil_ef2 /= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_ef2_n_organic_soils,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )

        # soil C factors for drained organic soils
        # dict_soil_ef_c_dos, arr_lndu_frac_organic_soils
        # HEREHEREHERE arr_soil_ef_c_organic_cultivated_soils
        # get the emission factors for C in drained organic soils as part of soil carbon
        # NOTE: The factors are in terms of output unit emission mass (config) per self.model_socioeconomic.modvar_gnrl_area
        dict_soil_ef_c_dos = self.get_soil_arrs_ef_c_drained_organic_soils(df_afolu_trajectories, )

        # get areas of drained organic soils by time period
        arr_lndu_area_dos = self.get_lndu_area_drained_organic_soils(
            df_afolu_trajectories,
            arr_land_use, 
            arrs_lndu_land_conv,
        )


        ##  ITERATE OVER TROPICAL/TEMPERATE SOIL TYPES TO GET EMISSIONS ASSOCIATED WITH DOS
        
        # initialize the array of DOS emissions for CO2
        arr_lndu_emission_co2_drained_organic_soils = np.zeros(
            (
                n_projection_time_periods, 
                attr_lndu.n_key_values,
            )
        )

        # initialize direct N2O emissions from drained organic soils
        vec_soil_n2on_direct_organic = 0.0


        ##  CROPLAND (SINCE CROPS HAVE DATA BY WET/TEMPERATE/ETC)

        arr_agrc_ef_c_dos = dict_soil_ef_c_dos.get(self.cat_lndu_crop, )

        self.dict_soil_ef_c_dos = dict_soil_ef_c_dos
        self.arr_lndu_area_dos = arr_lndu_area_dos
        self.dict_arrs_agrc_frac_temptrop = dict_arrs_agrc_frac_temptrop

        for modvar in self.modvar_list_agrc_frac_temptrop:
            # get appropriate soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            
            #  keep the fraction of organic soils available
            #  arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]

            # area of DOS by crop
            arr_agrc_area_dos_by_crop = sf.do_array_mult(
                arr_agrc_frac_cropland_area,
                arr_lndu_area_dos[:, self.ind_lndu_crop]
            )
            vec_soil_dos_temptrop_cur = (arr_agrc_area_dos_by_crop*dict_arrs_agrc_frac_temptrop[modvar]).sum(axis = 1, )
            self.vec_soil_dos_temptrop_cur = vec_soil_dos_temptrop_cur
            # N component
            vec_soil_n2on_direct_organic += vec_soil_dos_temptrop_cur*arr_soil_ef2[:, ind_soil]

            # DOS Carbon - skip if not defined for crops
            if arr_agrc_ef_c_dos is not None: 
                vec_component = vec_soil_dos_temptrop_cur*arr_agrc_ef_c_dos[:, ind_soil]
                arr_lndu_emission_co2_drained_organic_soils[:, self.ind_lndu_crop] += vec_component

        
        ##  OTHER LAND USE DOS

        for k, v in dict_soil_ef_c_dos.items():
            
            # croplands are dealt with above
            if k == self.cat_lndu_crop: continue
            ind_lndu_cur = attr_lndu.get_key_value_index(k)

            for modvar in self.modvar_list_lndu_frac_temptrop:
                # get appropriate soil category
                cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
                ind_soil = attr_soil.get_key_value_index(cat_soil)

                #  keep the fraction of organic soils available
                #  arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]

                # get DOS total area in the current climate specification
                vec_lndu_area_dos_cur_climate = (arr_lndu_area_dos*dict_arrs_lndu_frac_temptrop[modvar]).sum(axis = 1, )

                # DOS Carbon
                arr_lndu_emission_co2_drained_organic_soils[:, ind_lndu_cur] += vec_lndu_area_dos_cur_climate*v[:, ind_soil]

                # update N2O component for pastures
                if ind_lndu_cur == self.ind_lndu_pstr:
                    vec_soil_n2on_direct_organic += vec_lndu_area_dos_cur_climate*arr_soil_ef2[:, ind_soil]

        
        self.arr_lndu_emission_co2_drained_organic_soils = arr_lndu_emission_co2_drained_organic_soils
        """
        # loop over dry/wet to estimate carbon stocks in crops
        for modvar in self.modvar_list_agrc_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_crop_temptrop_cur = np.sum(arr_agrc_crop_area*dict_arrs_agrc_frac_temptrop[modvar], axis = 1)
            vec_soil_crop_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]*arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += vec_soil_crop_temptrop_cur
    
        # loop over dry/wet to estimate carbon stocks in pastures (managed grasslands)
        for modvar in self.modvar_list_lndu_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_pstr_temptrop_cur = (arr_land_use*dict_arrs_lndu_frac_temptrop[modvar])[:, self.ind_lndu_pstr]
            vec_soil_pstr_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_pstr]*arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += vec_soil_pstr_temptrop_cur
        """

        # loop over tropical/temperate NP/temperate NR
        for modvar in self.modvar_list_frst_frac_temptrop:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            # get land use category for soil carbon factor
            cats_lndu = [
                clean_schema(x) for x in self.model_attributes.get_ordered_category_attribute(
                    self.subsec_name_frst, 
                    pycat_lndu
                )
            ]
            inds_lndu = [attr_lndu.get_key_value_index(x) for x in cats_lndu]
            arr_soil_frst_temptrop_cur = np.sum(arr_area_frst*dict_arrs_frst_frac_temptrop[modvar]*arr_lndu_frac_organic_soils[:, inds_lndu], axis = 1)
            arr_soil_frst_temptrop_cur *= arr_soil_ef2[:, ind_soil]
            vec_soil_n2on_direct_organic += arr_soil_frst_temptrop_cur

        
        ##  PREP OUTPUTS

        # get soil carbon from organic drained soils
        arr_lndu_emission_co2_drained_organic_soils *= self.factor_c_to_co2*self.model_attributes.get_gwp("co2")

        # initialize output emission vector
        vec_soil_emission_n2o_organic_soils = vec_soil_n2on_direct_organic



        ####################################################################
        #    N2O DIRECT - PASTURE/RANGE/PADDOCK (PT. 3 OF EQUATION 11.1)   #
        ####################################################################

        #
        arr_soil_ef3 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef3_n_prp, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base",
        )

        vec_lsmm_nitrogen_to_pasture *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_pastures,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )

        # loop over dry/wet for EF3, pasture, range, and paddock
        vec_soil_n2on_direct_prp = 0.0
        dict_soil_ppr_n_by_climate = {}
        for modvar in self.modvar_list_lndu_frac_drywet:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)
            vec_soil_frac_pstr_drywet_cur = (arr_land_use*dict_arrs_lndu_frac_drywet[modvar])[:, self.ind_lndu_pstr]/arr_land_use[:, self.ind_lndu_pstr]
            
            # add component to EF1 estimate for F_SOM
            vec_soil_prp_cur = (vec_lsmm_nitrogen_to_pasture + vec_soil_n_fertilizer_use_organic_to_pasture)*vec_soil_frac_pstr_drywet_cur
            vec_soil_n2on_direct_prp += vec_soil_prp_cur*arr_soil_ef3[:, ind_soil]
            dict_soil_ppr_n_by_climate.update({cat_soil: vec_soil_prp_cur})

        # initialize output emissions
        vec_soil_emission_n2o_ppr = vec_soil_n2on_direct_prp



        ###########################################################
        #    N2O INDIRECT - VOLATISED EMISSIONS (EQUATION 11.9)   #
        ###########################################################

        # get volatilisation vars
        vec_soil_frac_gasf_non_urea = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        vec_soil_frac_gasf_urea = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_sn_urea,
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        vec_soil_frac_gasm = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_volatilisation_on, 
            override_vector_for_single_mv_q = False, 
            return_type = "array_base", 
            var_bounds = (0, 1),
        )

        arr_soil_ef4 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef4_n_volatilisation, 
            expand_to_all_cats = True,
            override_vector_for_single_mv_q = True, 
            return_type = "array_base", 
        )

        # loop over dry/wet
        vec_soil_n2on_indirect_volatilisation = 0.0
        vec_soil_n2on_indirect_volatilisation_gasf = 0.0
        vec_soil_n2on_indirect_volatilisation_gasm_on = 0.0
        vec_soil_n2on_indirect_volatilisation_gasm_ppr = 0.0

        for modvar in self.modvar_list_lndu_frac_drywet:
            # soil category
            cat_soil = clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
            ind_soil = attr_soil.get_key_value_index(cat_soil)

            # GASF component--synthetic by urea/non-urea
            vec_soil_fert_sn_cur_non_urea = dict_soil_fertilizer_application_by_climate_synthetic[cat_soil].copy()
            vec_soil_fert_sn_cur_urea = vec_soil_fert_sn_cur_non_urea*vec_soil_frac_synthetic_fertilizer_urea
            vec_soil_fert_sn_cur_non_urea -= vec_soil_fert_sn_cur_urea
            vec_soil_component_gasf_cur = vec_soil_fert_sn_cur_non_urea*vec_soil_frac_gasf_non_urea + vec_soil_fert_sn_cur_urea*vec_soil_frac_gasf_urea
            vec_soil_component_gasf_cur *= arr_soil_ef4[:, ind_soil]

            # GASM component--organic
            vec_soil_component_gasm_on_cur = dict_soil_fertilizer_application_by_climate_organic[cat_soil]*vec_soil_frac_gasm*arr_soil_ef4[:, ind_soil]
            vec_soil_component_gasm_ppr_cur = dict_soil_ppr_n_by_climate[cat_soil]*vec_soil_frac_gasm*arr_soil_ef4[:, ind_soil]

            # aggregates
            vec_soil_n2on_indirect_volatilisation_gasf += vec_soil_component_gasf_cur
            vec_soil_n2on_indirect_volatilisation_gasm_on += vec_soil_component_gasm_on_cur
            vec_soil_n2on_indirect_volatilisation_gasm_ppr += vec_soil_component_gasm_ppr_cur
            vec_soil_n2on_indirect_volatilisation += vec_soil_component_gasf_cur + vec_soil_component_gasm_on_cur + vec_soil_component_gasm_ppr_cur

        # update emissions
        vec_soil_emission_n2o_fertilizer += vec_soil_n2on_indirect_volatilisation_gasf + vec_soil_n2on_indirect_volatilisation_gasm_on
        vec_soil_emission_n2o_ppr += vec_soil_n2on_indirect_volatilisation_gasm_ppr



        ###########################################################
        #    N2O INDIRECT - LEACHING EMISSIONS (EQUATION 11.10)   #
        ###########################################################

        # get some components
        vec_soil_ef5 = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef5_n_leaching, 
            return_type = "array_base",
        )

        vec_soil_frac_leaching = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_frac_n_lost_leaching, 
            return_type = "array_base", 
            var_bounds = (0, 1)
        )
        
        # add up sources of N
        vec_soil_n2on_indirect_leaching_fert = vec_soil_n_fertilizer_use_organic + vec_soil_n_fertilizer_use_synthetic
        vec_soil_n2on_indirect_leaching_fert *= vec_soil_frac_leaching*vec_soil_ef5
        vec_soil_n2on_indirect_leaching_ppr = vec_lsmm_nitrogen_to_pasture + vec_soil_n_fertilizer_use_organic_to_pasture
        vec_soil_n2on_indirect_leaching_ppr *= vec_soil_frac_leaching*vec_soil_ef5

        vec_soil_n2on_indirect_leaching_cr = vec_agrc_total_n_residue_dry + vec_agrc_total_n_residue_rice + vec_agrc_total_n_residue_wet
        vec_soil_n2on_indirect_leaching_cr *= vec_soil_frac_leaching*vec_soil_ef5
        vec_soil_n2on_indirect_leaching_mineral_soils = vec_soil_delta_soc_mineral/vec_soil_ratio_c_to_n_soil_organic_matter
        vec_soil_n2on_indirect_leaching_mineral_soils *= vec_soil_frac_leaching*vec_soil_ef5
        
        # build aggregate emissions
        vec_soil_n2on_indirect_leaching = (vec_soil_n2on_indirect_leaching_fert + vec_soil_n2on_indirect_leaching_ppr + vec_soil_n2on_indirect_leaching_cr + vec_soil_n2on_indirect_leaching_mineral_soils)
        vec_soil_emission_n2o_crop_residue += vec_soil_n2on_indirect_leaching_cr
        vec_soil_emission_n2o_fertilizer += vec_soil_n2on_indirect_leaching_fert
        vec_soil_emission_n2o_mineral_soils += vec_soil_n2on_indirect_leaching_mineral_soils
        vec_soil_emission_n2o_ppr += vec_soil_n2on_indirect_leaching_ppr

        # get fertilizer use totals
        vec_soil_n_fertilizer_use_organic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_fertuse_final_organic,
            "mass"
        )
        vec_soil_n_fertilizer_use_synthetic *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_fertuse_final_synthetic,
            "mass"
        )
        # total
        vec_soil_n_fertilizer_use_total = vec_soil_n_fertilizer_use_organic*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_final_organic,
            self.modvar_soil_fertuse_final_total,
            "mass"
        )
        vec_soil_n_fertilizer_use_total += vec_soil_n_fertilizer_use_synthetic*self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_fertuse_final_synthetic,
            self.modvar_soil_fertuse_final_total,
            "mass"
        )

        #####################################################
        #    SUMMARIZE N2O EMISSIONS AS DIRECT + INDIRECT   #
        #####################################################

        scalar_n2on_to_emission_out = self.factor_n2on_to_n2o*self.model_attributes.get_scalar(self.modvar_lsmm_n_to_fertilizer_agg_dung, "mass")
        scalar_n2on_to_emission_out *= self.model_attributes.get_gwp("n2o")
        
        # build emissions outputs
        df_out += [
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_crop_residue*scalar_n2on_to_emission_out, 
                self.modvar_agrc_emissions_n2o_crop_residues
            ),
            self.model_attributes.array_to_df(
                arr_lndu_emission_co2_drained_organic_soils, 
                self.modvar_lndu_emissions_co2_drained_organic_soils,
                reduce_from_all_cats_to_specified_cats = True,
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_soil_carbon_mineral, 
                self.modvar_soil_emissions_co2_soil_carbon_mineral
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_fertilizer*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_fertilizer
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_mineral_soils*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_mineral_soils
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_organic_soils*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_organic_soils
            ),
            self.model_attributes.array_to_df(
                vec_soil_emission_n2o_ppr*scalar_n2on_to_emission_out, 
                self.modvar_soil_emissions_n2o_ppr
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_organic, 
                self.modvar_soil_fertuse_final_organic
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_synthetic, 
                self.modvar_soil_fertuse_final_synthetic
            ),
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_total, 
                self.modvar_soil_fertuse_final_total
            )
        ]



        #####################################################
        #    CO2 EMISSIONS FROM LIMING + UREA APPLICATION   #
        #####################################################

        ##  LIMING

        # use land that's fertilized to project lime demand
        vec_soil_demscalar_liming = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_demscalar_liming, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        vec_soil_lime_init_dolomite = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_qtyinit_liming_dolomite,
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        vec_soil_lime_init_limestone = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_qtyinit_liming_limestone,
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        # get emission factors
        vec_soil_ef_liming_dolomite = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_liming_dolomite,
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        vec_soil_ef_liming_limestone = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_liming_limestone,
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        # write in terms of dolomite
        vec_soil_lime_init_limestone *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_qtyinit_liming_limestone,
            self.modvar_soil_qtyinit_liming_dolomite,
            "mass"
        )

        # estimate liming demand using the area of land that's fertilized
        vec_soil_lime_use_growth_rate = np.concatenate([np.ones(1), np.cumprod(vec_soil_area_fertilized[1:]/vec_soil_area_fertilized[0:-1])])
        vec_soil_lime_use_growth_rate *= vec_soil_demscalar_liming
        vec_soil_lime_use_dolomite = vec_soil_lime_init_dolomite[0]*vec_soil_lime_use_growth_rate
        vec_soil_lime_use_limestone = vec_soil_lime_init_limestone[0]*vec_soil_lime_use_growth_rate
       
        # get output emissions
        vec_soil_emission_co2_lime_use = vec_soil_lime_use_dolomite*vec_soil_ef_liming_dolomite + vec_soil_lime_use_limestone*vec_soil_ef_liming_limestone
        vec_soil_emission_co2_lime_use *= self.model_attributes.get_scalar(
            self.modvar_soil_qtyinit_liming_dolomite,
            "mass"
        )
        vec_soil_emission_co2_lime_use *= self.factor_c_to_co2
        vec_soil_emission_co2_lime_use *= self.model_attributes.get_gwp("co2")
        
        # total lime applied
        vec_soil_lime_use_total = vec_soil_lime_use_limestone + vec_soil_lime_use_dolomite
        vec_soil_lime_use_total *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_soil_qtyinit_liming_dolomite,
            self.modvar_soil_limeuse_total,
            "mass"
        )
         

        ##  UREA

        vec_soil_ef_urea = self.model_attributes.extract_model_variable(#
            df_afolu_trajectories, 
            self.modvar_soil_ef_c_urea, 
            return_type = "array_base", 
            var_bounds = (0, np.inf),
        )

        vec_soil_emission_co2_urea_use = vec_soil_ef_urea*vec_soil_n_fertilizer_use_synthetic_urea
        vec_soil_emission_co2_urea_use *= self.model_attributes.get_scalar(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass",
        )

        vec_soil_emission_co2_urea_use *= self.factor_c_to_co2
        vec_soil_emission_co2_urea_use *= self.model_attributes.get_gwp("co2")
        
        # get total urea applied (based on synthetic fertilizer, which was in terms of modvar_lsmm_n_to_fertilizer_agg_dung)
        vec_soil_n_fertilizer_use_synthetic_urea *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            self.modvar_soil_ureause_total,
            "mass",
        )
        

        # add to output
        df_out += [
            # CO2 EMISSIONS FROM LIMING
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_lime_use, 
                self.modvar_soil_emissions_co2_lime
            ),
            # CO2 EMISSIONS FROM UREA
            self.model_attributes.array_to_df(
                vec_soil_emission_co2_urea_use, 
                self.modvar_soil_emissions_co2_urea
            ),
            # TOTAL LIME USE
            self.model_attributes.array_to_df(
                vec_soil_lime_use_total, 
                self.modvar_soil_limeuse_total
            ),
            # TOTAL UREA USE
            self.model_attributes.array_to_df(
                vec_soil_n_fertilizer_use_synthetic_urea, 
                self.modvar_soil_ureause_total
            )
        ]



        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)

        if passthrough_tmp is None:
            return df_out
        else:
            return df_out, passthrough_tmp




###################################
###                             ###
###    SOME SIMPLE FUNCTIONS    ###
###                             ###
###################################


def is_sisepuede_model_afolu(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDE AFOLU model
    """

    out = hasattr(obj, "is_sisepuede_model_afolu")
    uuid = getattr(obj, "_uuid", None)
    
    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out