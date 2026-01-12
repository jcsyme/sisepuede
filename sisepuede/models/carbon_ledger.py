"""

"""

import numpy as np
import pandas as pd
import sisepuede.utilities._toolbox as sf
from typing import *


class ShapeError(Exception):
    pass





class BiomassCarbonLedger:
    """The CarbonLedger is a simplified dynamic model of forest carbon 
        sequestration, removal, and decay. It is designed to be iteratively 
        updated within the AFOLU model of the SISEPUEDE framework, but it can be 
        used outside of this model as well.

    The BiomassCarbonLedger uses 

        

    KEY ASSUMPTIONS
    ---------------
    * Conversions from young forest are assumed to be taken from the oldest
        young forest first.
    * Protected areas do not experience removals
    * Primary forests are in emissions equilibrium; i.e., that annual 
        sequestration is offset by emissions from decomposition of biomass. 
        This is used to estimate a fraction of total above-ground biomass that
        decomposes every year, which is then applied to all forests (SRC).
    * Removals occur homogenously across forests; since there is currently no 
        geospatial aspect to SISEPUEDE, this assumption is needed to ensure that
        carbon stock responds. 
    * Sequestration is dynamic, responding to changes in carbon stock to 
        represent degredation
    * Removals are also restricted in a buffer zone (representing reduced
        availability, which slows withdrawals) and cannot bring forest stock
        levels below a certain level, which would force a shift in land use
        class. This uses a reservoir analogy.

        

    NOTES
    -----
    * The carbon ledger is UNITLESS--all mass and area units are assumed to be 
        the same for all metrics. Users must control the input units to ensure
        they are uniform. 
    
        

    ARGUMENTS
    ---------

    # Initialization Arguments
    
    n_tp : int
        Number of time periods to use
    model_attributes : ModelAttributes
        ModelAttributes for some initialization
    cat_frst_secondary : str
        Secondary forest category
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

        
    # Keyword Arguments

    n_tps_no_withdrawals_new_growth : int
        Number of time periods without removals while new growth occurs. 
    **kwargs: 
        Passed to initialize arrays. Can be used to set initial values for
        arrays. Use

            {"initval_ARRAYVARNAME": x}

        to set the initial value of ARRAYVARNAME in x

        

    VARIABLE INFORMATION AND DESCRIPTIONS
    -------------------------------------

    To facilitate clarity, variables here can be traced to both documentation 
        and the Excel conceptual model contained in the SISEPUEDE reference
        directory.
        
        In variable dimensions below:
            * N = number of categories
            * T = number of time periods
            * UL = Unitless

            
        ##  INPUT VARIABLES

        * `arr_area` (T x N)
            Area of tracked categories.

        * `arr_area_conversion_away_total` (T x N)
            Area of land converted away; entry t gives the land converted 
            away in time period t.

        * `arr_area_conversion_into` (T x N)
            Area of conversion into each land use type. Long by time period,
            wide by category.

        * `arr_area_protected_total` (T x N)
            Area of each type protected in total (orig + young).

        * `arr_biomass_c_average_ag_stock_in_conversion_targets` (T x N)
            Average stock per ha in conversion target land use classes out of 
            each forest type. Used to restrict removals from converted land use
            classes.

        * `vec_biomass_c_ag_init_stst_storage` (N x 1)
            Array giving initial biomass C stock, i.e., mass of biomass c per 
            area for each category.

        * `vec_biomass_c_bg_to_ag_ratio` (N x 1)
            Array giving the ratio of below ground biomass to above ground
            biomass. This is used to produced crude estimates of below-ground
            biomass stock.

        * `vec_frac_biomass_adjustment_threshold` (T x 1) 
            Vector storing the biomass sequestration adjustment threshold by 
            time period. If the average carbon stock per area relative to the 
            steady state (initial) stock per area drops below this threshold,
            sequestration will start to decline linearly until the dead storage
            threshold is reach, where sequestration stops.
            (SOURCE FOR 33%).

        * `vec_frac_biomass_buffer` (T x 1) 
            Vector storing the biomass removal buffer zone by time period. If 
            the average carbon stock per area relative to the steady state 
            (initial) stock per area drops below this threshold, fewer removals 
            can be satisfied due to scarcity. Removals satisfiable will scale
            linearly downward until it reaches 0 at the dead storage threshold.

        * `vec_frac_biomass_from_conversion_available_for_use` (T x 1)
            Fraction of above-ground biomass that is available for use in 
            fuelwood, roundwood, and harvested wood products after land use 
            conversion. Any wood made available here reduces demand for removals
            from live forest.

        * `vec_frac_biomass_dead_storage` (T x 1)
            Vector giving the fraction of the steady-state (initial) assumed 
            carbon stock that is considered dead storage, i.e., minimum fraction 
            of original stock that must remain on the land to allow it to remain 
            that land use type. At or below this level, sequestration does not 
            occur (due to total degredation). This is required to prevent
            conversions that are separate from the land use model.

        * `vec_sf_nominal_initial` (N x 1)
            Nominal initial sequestration factors for original forests.

        * `vec_total_removals_demanded` (T x 1)
            Vector of total removals demanded, including HWP and fuelwood
            removals.

        * `vec_young_sf_curve` (T x 1)
            Vector of sequestration factors, over time, for newly planted
            forests; entries at time t are the sequestration factors t time 
            periods after planting. If entered as a single number, uses one 
            value for every time period.
        


        ##  INTERNAL CALCULATION VARIABLES

        * `arr_area_protected_original` (T x N)
            Area of each type (original) protected. Excludes young categories.
            
        * `arr_area_protected_original` (T x N)
            Area of each type (original) protected. Excludes young categories.

        * `arr_area_remaining_from_orig` (T x N)
            Area of tracked land use categories remaining from original,
            steady state assumption.

        * `arr_area_remaining_from_orig_after_conversion_away` (T x N)
            Area at end of time period for the land use class. I.e.,
            arr_area_remaining_from_orig_after_conversion_away[t] = arr_area_remaining_from_orig[t + 1]
        
        * `arr_biomass_c_ag_min_reqd_per_area` (T x N)
            Array that stores the minimum required biomass per area as the
            outer product of `vec_biomass_c_ag_init_stst_storage` and 
            `vec_frac_biomass_dead_storage`.
        
        * `arr_biomass_c_removals_from_converted_land_allocation` (T x N)
            Allocation of C removals from converted land 

        * `arr_orig_allocation_removals` (T x N)
            Fraction of total pool used to allocated withdrawls by forest type.
            Columns sum to 1.

        * `arr_orig_biomass_c_ag_available_from_conversion` (T x N)
            Above-ground biomass C from conversion made available for removals.
            
        * `arr_orig_biomass_c_ag_if_untouched` (T x N)
            Similar to `arr_orig_biomass_c_ag_if_untouched`, tracks biomass
            C stock that would be associated with areas of forest if no removals
            were made. This is the primary comparison made to determine 
            degredation and adjust sequestration factors accordingle. 

        * `arr_orig_biomass_c_ag_preserved_in_conversion` (T x N)
            Above ground biomass C that must be conserved--not allowed for 
            removals--in converted lands to maintain carbon stock for land use 
            conversion target classes.
        
        * `arr_orig_biomass_c_allocation_adjusted` (T x N)
            Adjusted stock available based on fraction of removals satisfiable.
    
        * `arr_orig_biomass_c_allocation_excluding_conversion` (T x N)
            Allocation of biomass extractions between forest types; used in
            estimates of above-ground C emissions or loss from conversion.

        * `arr_orig_biomass_c_average_per_area_no_ds` (T x N)
            Average biomass stock per area in original forest j at time i 
            excluding dead storage.

        * `arr_orig_frac_removables_satisfiable` (T x N)
            Fraction of removables satisfiable based on average carbon storage 
            level per unit area.

        * `arr_orig_frac_stock_available` (T x N)
            Average per area forest stock available excluding dead storage.

        * `arr_orig_sf_adjustment_factor` (T x N)
            Factors used to scale `arr_orig_biomass_c_ag_if_untouched` based on 
            the status of average carbon stock per ha in original forests (those
            present at the start of the simulation) in the previous time period. 
            If the average stock falls below the adjustment threshold (see
            `vec_frac_biomass_adjustment_threshold`), then sequestration factors
            are scaled linearly based on the distance between the adjustment 
            threshold and dead storage (see `vec_frac_biomass_dead_storage`).

        * `arr_young_area_by_tp_planted` (T x T)
            Area of young forest by time period planted.

        * `arr_young_area_by_tp_planted_cumvals` (T x T)
            Support array for dynamic updating of arr_young_area_by_tp_planted; 
            stores cumulative areas to identify when conversions away from young 
            forest (if they occur) take from older forests.

        * `arr_young_area_by_tp_planted_drops` (T x T)
            Support array for dynamic updating of arr_young_area_by_tp_planted; 
            stores area of conversion away from each area of young forest by 
            time period.

        * `arr_biomass_c_ag_converted_away` (T x N)
            Total stock of above-ground biomass C that is converted to a 
            different land use category BEFORE removals and minimum needed to
            stay for target land use class.

        * `arr_young_biomass_c_ag_converted_by_tp_planted` (T x T)
            Above-ground C stock that is converted by time period.

        * `arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted` (T x T)
            Young biomass that is converted that must remain to align with
            target land use class to prevent removal and replanting. Defined for
            each new forest class by time period planted. 

        * `arr_young_biomass_c_ag_stock` (T x T)
            Array storing the total estimated above-ground biomass C stock at 
            *the end* of time period i for forests planted at time j. EXCLUDES 
            CONVERSIONS FROM TIME PERIOD i.

        * `arr_young_biomass_c_ag_stock_if_untouched` (T x T)
            Total C stock in young forests if untouched for an area of forest
            planted in time period t (column).

        * `arr_young_biomass_c_available_for_removals_mask` (T x T)
            Biomass available for removal from each area in time i (row) of new 
            forest planted in time j (column).

        * `arr_young_biomass_c_bg_converted_by_tp_planted` (T x T)
            Below-ground C stock that is converted by time period.

        * `arr_young_biomass_c_bg_stock` (T x T)
            Array storing the total estimated below-ground biomass C stock at 
            *the end* of time period i for forests planted at time j. EXCLUDES 
            CONVERSIONS FROM TIME PERIOD i.

        * `arr_young_biomass_c_loss_from_decomposition` (T x T)
            Array storing loss of C assumed to decompose at time i (from dead 
            biomass or litter) and emit C at time j. This is estimated using the
            fraction of biomass that is estimated to die as a condition for 
            equilibrium based on primary forest factors.

        * `arr_young_biomass_c_stock_removal_allocation` (T x T)
            Array that stores how much biomass *is* removed in time i (row) of 
            new forest planted in time j (column). Allocates total removal
            under the assumption that older new forests are taken from first.

        * `arr_young_biomass_c_stock_removal_allocation_aux` (T x T)
            Auxiliary array to support calculations in 
            `arr_young_biomass_c_stock_removal_allocation`.

        * `arr_young_sf_adjusted_by_tp_planted` (T x T)
            Sequestration factors at time i for forests planted at time j. These
            scale the base adjustment factors stored in 
            `arr_young_sf_base_by_tp_planted` by factors stored in 
            `arr_young_sf_adjustment_factor`. The factors are responsive to
            changes in average stock per area from the previous time period.

        * `arr_young_sf_adjustment_factor` (T x T)
            Factors used to scale `arr_young_sf_base_by_tp_planted` based on the 
            status of average carbon stock per ha in the previous time period. 
            If the average stock falls below the adjustment threshold (see
            `vec_frac_biomass_adjustment_threshold`), then sequestration factors
            are scaled linearly based on the distance between the adjustment 
            threshold and dead storage (see `vec_frac_biomass_dead_storage`).

        * `arr_young_sf_base_by_tp_planted` (T x T)
            Array storing base sequestration factors by time period, which are
            generally based on NPP curves.

        * `vec_area_conversion_away_young_forest` (T x 1)
            Vector of total conversion away from young forest.

        * `vec_area_conversion_away_young_forest_no_protection` (T x 1)
            Vector of total conversion away from young forest EXCLUDING any 
            protected land. Used in intermediate calculations for carbon stock.

        * `vec_biomass_c_ag_init_healthy_available` (N x 1)
            Vector storing the initial amount of health stock available by 
            category.

        * `vec_frac_biomass_ag_decomposition` (N x 1)
            Estimated fraction of biomass that decomposes every year. This
            fraction is estimated using the equilibrium assumption for primary 
            forests, where total sequestration == total emission.
        
        * `vec_young_biomass_c_ag_converted` (T x 1)
            Total above-ground C converted from T x T; sum over columns.

        * `vec_young_biomass_c_ag_preserved_in_conversion` (T x 1)
            Total young biomass from conversion that must remain to align with
            target land use class to prevent removal and replanting.

        * `vec_young_biomass_c_available_for_removals_total` (T x 1)

            Total biomass available for removal from each area in time i; sum
            over columns of arr_young_biomass_c_available_for_removals_mask`.

        * `vec_young_biomass_c_bg_converted` (T x 1)
            Total below-ground C converted from T x T; sum over columns.
        
        * `vec_biomass_c_removals_from_forest_demanded` (T x 1)
            Total demand for removals from all forested land after accounting 
            for automatic removals from available converted biomass.

        * `vec_biomass_c_removed_from_original_demanded` (T x 1)
            Demand for removals from original forested land after accounting for
            automatic removals from available converted biomass.

        * `vec_biomass_c_removed_from_original_unmet` (T x N)
            Demand for removals unmet by original forests.

        * `vec_biomass_c_removed_from_young` (T x 1)
            Vector storing the mass of biomass removed from young forests at
            time t
        
        * `vec_orig_biomass_c_accessible_pool` (T x 1)
            Vector storing the total mass of biomass accessible. Used to
            constrain removals.
            


        ##  OUTPUT VARIABLES

        * `arr_biomass_c_ag_lost_conversion` (T x N)
            Total above-ground C lost ue to conversion

        * `arr_biomass_c_bg_lost_conversion` (T x N)
            Total below-ground C lost due to conversion
        
        * `arr_biomass_c_removed_from_forests_excluding_conversion` (T x 1)
            Total biomass removed from each forest type excluding conversions at
            time i.
            
        * `arr_orig_biomass_c_ag_average_per_area` (T x N)
            Average biomass Cin original forest per unit area are the start of
            the time period (before conversion)
        
        * `arr_biomass_c_ag_lost_decomposition` (T x N)
            Above-ground biomass C lost to decomposition in each forest type at 
            time i

        * `arr_orig_biomass_c_ag_starting` (T x N)
            Above-ground biomass C in original forest at the start of the time
            period.
        
        * `arr_biomass_c_bg_lost_removals` (T x N)
            Below-ground biomass C lost due to removals in each forest type at 
            time i
        
        * `arr_orig_biomass_c_removed_from_forests` (T x N)
            Total biomass removed from each forest type j at time i

        * `arr_orig_sf_adjusted` (T x N)
            Adjusted seuqestration factor, which the base sequestration factor
            (for each forest type) multiplied by the adjustment factor in 
            `arr_orig_sf_adjustment_factor`

        * `arr_total_biomass_c_ag_starting` (T x N)
            Total above-ground biomass in each forest type at the start of time 
            period t.
        
        * `arr_total_biomass_c_bg_starting` (T x N)
            Total below-ground biomass in each forest type at the start of time 
            period t.

        * `vec_biomass_c_removals_from_converted` (T x 1)
            Vector of actual removals from converted biomass available

        * `vec_total_removals_met` (T x 1)
            Total removals actually met

        * `vec_young_biomass_c_ag_starting` (T x 1)
            Vector storing the total above-ground biomass in young forests 
            at time t. Column sums of `arr_young_biomass_c_ag_stock`

    """


    def __init__(self,
        model_attributes: ModelAttributes,
        cat_frst_secondary: str,
        cats_lndu_track: Union[List[str], None],
        dict_lndu_to_frst: Dict[str, str],
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],
        n_tp: int,
        vec_biomass_c_ag_init_stst_storage: np.ndarray,
        vec_biomass_c_bg_to_ag_ratio: np.ndarray,
        vec_frac_biomass_adjustment_threshold: Union[float, np.ndarray],
        vec_frac_biomass_buffer: Union[float, np.ndarray],
        vec_frac_biomass_dead_storage: Union[float, int, np.ndarray],
        vec_frac_biomass_from_conversion_available_for_use: Union[float, int, np.ndarray],
        vec_sf_nominal_initial: np.ndarray,
        vec_total_removals_demanded: np.ndarray,
        n_tps_no_withdrawals_new_growth: int = 20,
        **kwargs,
    ) -> None:
        

        self._initialize_attributes(
            model_attributes,
            modvar_frst_frac_c_converted_available,
            modvar_frst_frac_max_degradation,
            cat_frst_secondary,
            cats_lndu_track,
            dict_lndu_to_frst,
            n_tp,
        )

        self._initialize_arrays(
            n_tp,
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_bg_to_ag_ratio,
            vec_frac_biomass_adjustment_threshold,
            vec_frac_biomass_buffer,
            vec_frac_biomass_dead_storage,
            vec_frac_biomass_from_conversion_available_for_use,
            vec_sf_nominal_initial,
            vec_total_removals_demanded,
        )

        self._initialize_new_forest_properties(
            vec_sequestration_per_tp_new,
            n_tps_no_withdrawals_new_growth,
        )

        self._initialize_nf_availability_mask()

        return None
    
    


    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def build_arr_young_sf_base_by_tp_planted(self,
        vec_young_sf_curve: np.ndarray,
    ) -> np.ndarray:
        """Build the arr_young_biomass_c_ag_converted_by_tp_planted array
        """

        arr_young_sf_base_by_tp_planted = np.zeros((self.n_tp, self.n_tp, ))

        for i in range(self.n_tp):
            n_end = self.n_tp - i
            arr_young_sf_base_by_tp_planted[i:] = vec_young_sf_curve[0:n_end]

        return arr_young_sf_base_by_tp_planted
    


    def _initialize_attributes(self,
        model_attributes: ModelAttributes,
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],
        cat_frst_secondary: str,
        cats_lndu_track: Union[List[str], None],    
        dict_lndu_to_frst: Dict[str, str],
        n_tp: int,
    ) -> None:
        """Initialize key attributes used to manage land use classes
        """

        attr_lndu = model_attributes.get_attribute_table(
            model_attributes.subsec_name_lndu,
        )
        # some dictionaries
        dict_frst_to_lndu = sf.reverse_dict(dict_lndu_to_frst,)


        # set categories
        cats_lndu_frst = [x for x in attr_lndu.key_values if (x in dict_lndu_to_frst.keys())]
        
        cats_lndu_track = (
            [x for x in attr_lndu.key_values if (x in cats_lndu_track) and (x in cats_lndu_frst)]
            if sf.islistlike(cats_lndu_track)
            else cats_lndu_frst
        )
        cats_frst_track = [dict_lndu_to_frst.get(x) for x in cats_lndu_track]

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


        # get some model variables
        modvar_frst_frac_c_converted_available = model_attributes.get_variable(
            modvar_frst_frac_c_converted_available,
        )
        modvar_frst_frac_max_degradation = model_attributes.get_variable(
            modvar_frst_frac_max_degradation,
        )

        if not sf.isnumber(n_tp, integer = True, ):
            raise TypeError(f"n_tp must be an integer")



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
        self.modvar_frst_frac_c_converted_available = modvar_frst_frac_c_converted_available
        self.modvar_frst_frac_max_degradation = modvar_frst_frac_max_degradation
        self.n_tp = n_tp
        self.pycat_lndu = pycat_lndu

        return None
    

    



    def _check_initialization_arrays(self,
        n_cats: int,
        n_tp: int,
        vec_biomass_c_ag_init_stst_storage: np.ndarray,
        vec_biomass_c_bg_to_ag_ratio: np.ndarray,
        vec_frac_biomass_adjustment_threshold: Union[float, np.ndarray],
        vec_frac_biomass_buffer: Union[float, np.ndarray],
        vec_frac_biomass_dead_storage: Union[float, int, np.ndarray],
        vec_frac_biomass_from_conversion_available_for_use: Union[float, int, np.ndarray],
        vec_sf_nominal_initial: np.ndarray,
        vec_total_removals_demanded: np.ndarray,
        vec_young_sf_curve: Union[float, np.ndarray],
    ) -> Tuple:
        """Check arrays used to initialize the ledger
        """

        ##  VERIFY AND CONVERT

        # initial steady-state storage
        vec_biomass_c_ag_init_stst_storage = self._verify_convert_array_input_to_array(
            vec_biomass_c_ag_init_stst_storage,
            n_cats,
            "vec_biomass_c_ag_init_stst_storage",
        )

        # initial below ground biomass to above ground biomass ratio
        vec_biomass_c_bg_to_ag_ratio = self._verify_convert_array_input_to_array(
            vec_biomass_c_bg_to_ag_ratio,
            n_cats,
            "vec_biomass_c_bg_to_ag_ratio",
        )

        # removals adjustment threshold
        vec_frac_biomass_adjustment_threshold = self._verify_convert_array_input_to_array(
            vec_frac_biomass_adjustment_threshold,
            n_tp,
            "vec_frac_biomass_adjustment_threshold",
        )

        # biomass buffer
        vec_frac_biomass_buffer = self._verify_convert_array_input_to_array(
            vec_frac_biomass_buffer,
            n_tp,
            "vec_frac_biomass_buffer",
        )

        # initial dead storage fraction
        vec_frac_biomass_dead_storage = self._verify_convert_array_input_to_array(
            vec_frac_biomass_dead_storage,
            n_tp,
            "vec_frac_biomass_dead_storage",
        )

        # how much converted biomass (ag) is available for use
        vec_frac_biomass_from_conversion_available_for_use  = self._verify_convert_array_input_to_array(
            vec_frac_biomass_from_conversion_available_for_use,
            n_tp,
            "vec_frac_biomass_from_conversion_available_for_use",
        )

        # initial annualized sequestration factors
        vec_sf_nominal_initial = self._verify_convert_array_input_to_array(
            vec_sf_nominal_initial,
            n_cats,
            "vec_sf_nominal_initial",
        )

        # vector of total C removals
        vec_total_removals_demanded = self._verify_convert_array_input_to_array(
            vec_total_removals_demanded,
            n_tp,
            "vec_total_removals_demanded",
        )
        
        # check the sequestration factor curve for young forests
        vec_young_sf_curve = self._verify_convert_array_input_to_array(
            vec_young_sf_curve,
            n_tp,
            "vec_young_sf_curve",
        )

        

        out = (
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_bg_to_ag_ratio,
            vec_frac_biomass_adjustment_threshold,
            vec_frac_biomass_buffer,
            vec_frac_biomass_dead_storage,
            vec_frac_biomass_from_conversion_available_for_use,
            vec_sf_nominal_initial,
            vec_total_removals_demanded,
            vec_young_sf_curve, 
        )

        return out
    


    def _initialize_arrays(self,
        vec_biomass_c_ag_init_stst_storage: np.ndarray,
        vec_biomass_c_bg_to_ag_ratio: np.ndarray,
        vec_frac_biomass_adjustment_threshold: Union[float, np.ndarray],
        vec_frac_biomass_buffer: Union[float, np.ndarray],
        vec_frac_biomass_dead_storage: Union[float, int, np.ndarray],
        vec_frac_biomass_from_conversion_available_for_use: Union[float, int, np.ndarray],
        vec_sf_nominal_initial: np.ndarray,
        vec_total_removals_demanded: np.ndarray,
        vec_young_sf_curve: Union[float, np.ndarray],
        **kwargs,
    ) -> None:
        """Initialize the arrays needed for tracking Carbon stock. Stores the
            following arrays:

        note this is temporary and will be moved up to the class docstring


        """

        n_cats = len(self.cats_lndu_track)
        n_tp = self.n_tp

        shape_by_cat = (n_tp, n_cats)
    
        ##  INITIALIZE ARRAYS

        # shape T x N
        arr_area = np.zeros(shape_by_cat, )
        arr_area_protected_original = np.zeros(shape_by_cat, )
        arr_area_protected_total = np.zeros(shape_by_cat, )
        arr_area_protected_young = np.zeros(shape_by_cat, )
        arr_area_remaining_from_orig = np.zeros(shape_by_cat, )
        arr_area_conversion_away_total = np.zeros(shape_by_cat, )
        arr_area_conversion_into = np.zeros(shape_by_cat, )
        arr_area_remaining_from_orig_after_conversion_away = np.zeros(shape_by_cat, )

        arr_biomass_c_ag_available_from_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_average_ag_stock_in_conversion_targets = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_converted_away = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_lost_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_lost_decomposition = np.zeros(shape_by_cat, )
        arr_biomass_c_bg_lost_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_bg_lost_removals = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_min_reqd_per_area = np.zeros(shape_by_cat, )
        arr_biomass_c_removals_from_converted_land_allocation = np.zeros(shape_by_cat, )
        arr_biomass_c_removed_from_forests_excluding_conversion = np.zeros(shape_by_cat, )

        arr_orig_allocation_removals = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_available_from_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_average_per_area = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_if_untouched = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_preserved_in_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_starting = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_allocation_adjusted = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_allocation_excluding_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_average_per_area_no_ds = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_removed_from_forests = np.zeros(shape_by_cat, )
        arr_orig_frac_stock_available = np.zeros(shape_by_cat, )
        arr_orig_frac_removables_satisfiable = np.zeros(shape_by_cat, )
        arr_orig_sf_adjusted = np.zeros(shape_by_cat, )
        arr_orig_sf_adjustment_factor = np.zeros(shape_by_cat, )

        arr_total_biomass_c_ag_starting = np.zeros(shape_by_cat, )
        arr_total_biomass_c_bg_starting = np.zeros(shape_by_cat, )



        ##  INITIALIZE VECTORS

        # by category
        vec_biomass_c_ag_init_healthy_available = np.zeros(n_cats, )
        vec_frac_biomass_ag_decomposition = np.zeros(n_cats, )

        # by time period
        vec_area_conversion_away_young_forest = np.zeros(n_tp, )
        vec_area_conversion_away_young_forest_no_protection = np.zeros(n_tp, )
        vec_biomass_c_removals_from_converted = np.zeros(n_tp, )
        vec_biomass_c_removals_from_forest_demanded = np.zeros(n_tp, )
        vec_biomass_c_removed_from_original_demanded = np.zeros(n_tp, )
        vec_biomass_c_removed_from_original_unmet = np.zeros(n_tp, )
        vec_biomass_c_removed_from_young = np.zeros(n_tp, )
        vec_orig_biomass_c_accessible_pool = np.zeros(n_tp, )
        vec_total_removals_met = np.zeros(n_tp, )
        vec_young_biomass_c_ag_starting = np.zeros(n_tp, )



        ##  GET AND VERIFY INITIALIZATION ARRAYS

        #   note: this call will check input types and convert to numpys
        #         entered as a number. Also verifies shape.

        (
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_bg_to_ag_ratio,
            vec_frac_biomass_adjustment_threshold,
            vec_frac_biomass_buffer,
            vec_frac_biomass_dead_storage,
            vec_frac_biomass_from_conversion_available_for_use,
            vec_sf_nominal_initial,
            vec_total_removals_demanded,
            vec_young_sf_curve,
        ) = self._check_initialization_arrays(
            n_cats,
            n_tp,
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_bg_to_ag_ratio,
            vec_frac_biomass_adjustment_threshold,
            vec_frac_biomass_buffer,
            vec_frac_biomass_dead_storage,
            vec_frac_biomass_from_conversion_available_for_use,
            vec_sf_nominal_initial,
            vec_total_removals_demanded,
            vec_young_sf_curve,
        )


        # set the young forest internal calculation arrays
        self._initialize_young_forest_internal_arrays(vec_young_sf_curve, )


        ##  SET PROPERTIES

        self.arr_area = arr_area
        self.arr_area_protected_original = arr_area_protected_original
        self.arr_area_protected_total = arr_area_protected_total
        self.arr_area_protected_young = arr_area_protected_young
        self.arr_area_remaining_from_orig = arr_area_remaining_from_orig
        self.arr_area_remaining_from_orig_after_conversion_away = arr_area_remaining_from_orig_after_conversion_away
        self.arr_area_conversion_away_total = arr_area_conversion_away_total
        self.arr_area_conversion_into = arr_area_conversion_into
        self.arr_biomass_c_ag_available_from_conversion = arr_biomass_c_ag_available_from_conversion
        self.arr_biomass_c_average_ag_stock_in_conversion_targets = arr_biomass_c_average_ag_stock_in_conversion_targets
        self.arr_biomass_c_ag_converted_away = arr_biomass_c_ag_converted_away
        self.arr_biomass_c_ag_min_reqd_per_area = arr_biomass_c_ag_min_reqd_per_area
        self.arr_biomass_c_ag_lost_conversion = arr_biomass_c_ag_lost_conversion
        self.arr_biomass_c_ag_lost_decomposition = arr_biomass_c_ag_lost_decomposition
        self.arr_biomass_c_bg_lost_conversion = arr_biomass_c_bg_lost_conversion
        self.arr_biomass_c_bg_lost_removals = arr_biomass_c_bg_lost_removals
        self.arr_biomass_c_removals_from_converted_land_allocation = arr_biomass_c_removals_from_converted_land_allocation
        self.arr_biomass_c_removed_from_forests_excluding_conversion = arr_biomass_c_removed_from_forests_excluding_conversion
        self.arr_orig_allocation_removals = arr_orig_allocation_removals
        self.arr_orig_biomass_c_ag_available_from_conversion = arr_orig_biomass_c_ag_available_from_conversion
        self.arr_orig_biomass_c_ag_average_per_area = arr_orig_biomass_c_ag_average_per_area
        self.arr_orig_biomass_c_ag_if_untouched = arr_orig_biomass_c_ag_if_untouched
        self.arr_orig_biomass_c_ag_preserved_in_conversion = arr_orig_biomass_c_ag_preserved_in_conversion
        self.arr_orig_biomass_c_ag_starting = arr_orig_biomass_c_ag_starting
        self.arr_orig_biomass_c_allocation_adjusted = arr_orig_biomass_c_allocation_adjusted
        self.arr_orig_biomass_c_allocation_excluding_conversion = arr_orig_biomass_c_allocation_excluding_conversion
        self.arr_orig_biomass_c_average_per_area_no_ds = arr_orig_biomass_c_average_per_area_no_ds
        self.arr_orig_biomass_c_removed_from_forests = arr_orig_biomass_c_removed_from_forests
        self.arr_orig_frac_stock_available = arr_orig_frac_stock_available
        self.arr_orig_frac_removables_satisfiable = arr_orig_frac_removables_satisfiable
        self.arr_orig_sf_adjusted = arr_orig_sf_adjusted
        self.arr_orig_sf_adjustment_factor = arr_orig_sf_adjustment_factor
        self.arr_total_biomass_c_ag_starting = arr_total_biomass_c_ag_starting
        self.arr_total_biomass_c_bg_starting = arr_total_biomass_c_bg_starting
        self.vec_area_conversion_away_young_forest = vec_area_conversion_away_young_forest
        self.vec_area_conversion_away_young_forest_no_protection = vec_area_conversion_away_young_forest_no_protection
        self.vec_biomass_c_ag_init_healthy_available = vec_biomass_c_ag_init_healthy_available
        self.vec_biomass_c_ag_init_stst_storage = vec_biomass_c_ag_init_stst_storage
        self.vec_biomass_c_bg_to_ag_ratio = vec_biomass_c_bg_to_ag_ratio
        self.vec_biomass_c_removals_from_converted = vec_biomass_c_removals_from_converted
        self.vec_biomass_c_removals_from_forest_demanded = vec_biomass_c_removals_from_forest_demanded
        self.vec_biomass_c_removed_from_original_demanded = vec_biomass_c_removed_from_original_demanded
        self.vec_biomass_c_removed_from_original_unmet = vec_biomass_c_removed_from_original_unmet
        self.vec_biomass_c_removed_from_young = vec_biomass_c_removed_from_young
        self.vec_frac_biomass_adjustment_threshold = vec_frac_biomass_adjustment_threshold
        self.vec_frac_biomass_ag_decomposition = vec_frac_biomass_ag_decomposition
        self.vec_frac_biomass_buffer = vec_frac_biomass_buffer
        self.vec_frac_biomass_dead_storage = vec_frac_biomass_dead_storage
        self.vec_frac_biomass_from_conversion_available_for_use = vec_frac_biomass_from_conversion_available_for_use
        self.vec_orig_biomass_c_accessible_pool = vec_orig_biomass_c_accessible_pool
        self.vec_sf_nominal_initial = vec_sf_nominal_initial
        self.vec_total_removals_demanded = vec_total_removals_demanded
        self.vec_total_removals_met = vec_total_removals_met
        self.vec_young_biomass_c_ag_starting = vec_young_biomass_c_ag_starting
        self.vec_young_sf_curve = vec_young_sf_curve

        return None



    def _initialize_young_forest_internal_arrays(self,
        vec_young_sf_curve: np.ndarray,
        **kwargs,
    ) -> None:
        """Init

        """
        shape_by_tp = (self.n_tp, self.n_tp, )

        # shape T x T
        arr_young_area_by_tp_planted = np.zeros(shape_by_tp, )
        arr_young_area_by_tp_planted_cumvals = np.zeros(shape_by_tp, )
        arr_young_area_by_tp_planted_drops = np.zeros(shape_by_tp, )
        arr_young_biomass_c_ag_converted_by_tp_planted = np.zeros(shape_by_tp, )
        arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted = np.zeros(shape_by_tp, )
        arr_young_biomass_c_ag_stock = np.zeros(shape_by_tp, )
        arr_young_biomass_c_ag_stock_if_untouched = np.zeros(shape_by_tp, )
        arr_young_biomass_c_available_for_removals_mask = np.zeros(shape_by_tp, )
        arr_young_biomass_c_bg_converted_by_tp_planted = np.zeros(shape_by_tp, )
        arr_young_biomass_c_bg_stock = np.zeros(shape_by_tp, )
        arr_young_biomass_c_loss_from_decomposition = np.zeros(shape_by_tp, )
        arr_young_biomass_c_stock_removal_allocation = np.zeros(shape_by_tp, )
        arr_young_biomass_c_stock_removal_allocation_aux = np.zeros(shape_by_tp, )
        arr_young_sf_adjusted_by_tp_planted = np.zeros(shape_by_tp, )
        arr_young_sf_adjustment_factor = np.zeros(shape_by_tp, )

        # vectors long by time period
        vec_young_biomass_c_ag_converted = np.zeros(self.n_tp, )
        vec_young_biomass_c_ag_preserved_in_conversion = np.zeros(self.n_tp, )
        vec_young_biomass_c_available_for_removals_total = np.zeros(self.n_tp, )
        vec_young_biomass_c_bg_converted = np.zeros(self.n_tp, )

        # build base sequestration factors by time period
        arr_young_sf_base_by_tp_planted = self.build_arr_young_sf_base_by_tp_planted(
            vec_young_sf_curve, 
        )


        ##  SET PROPERTIES

        self.arr_young_area_by_tp_planted = arr_young_area_by_tp_planted
        self.arr_young_area_by_tp_planted_cumvals = arr_young_area_by_tp_planted_cumvals
        self.arr_young_area_by_tp_planted_drops = arr_young_area_by_tp_planted_drops
        self.arr_young_sf_base_by_tp_planted = arr_young_sf_base_by_tp_planted
        self.arr_young_biomass_c_ag_converted_by_tp_planted = arr_young_biomass_c_ag_converted_by_tp_planted
        self.arr_young_biomass_c_ag_stock = arr_young_biomass_c_ag_stock
        self.arr_young_biomass_c_ag_stock_if_untouched = arr_young_biomass_c_ag_stock_if_untouched
        self.arr_young_biomass_c_available_for_removals_mask = arr_young_biomass_c_available_for_removals_mask
        self.arr_young_biomass_c_bg_converted_by_tp_planted = arr_young_biomass_c_bg_converted_by_tp_planted
        self.arr_young_biomass_c_bg_stock = arr_young_biomass_c_bg_stock
        self.arr_young_biomass_c_loss_from_decomposition = arr_young_biomass_c_loss_from_decomposition
        self.arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted = arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted
        self.arr_young_biomass_c_stock_removal_allocation = arr_young_biomass_c_stock_removal_allocation
        self.arr_young_biomass_c_stock_removal_allocation_aux = arr_young_biomass_c_stock_removal_allocation_aux
        self.arr_young_sf_adjusted_by_tp_planted = arr_young_sf_adjusted_by_tp_planted
        self.arr_young_sf_adjustment_factor = arr_young_sf_adjustment_factor
        self.vec_young_biomass_c_ag_converted = vec_young_biomass_c_ag_converted
        self.vec_young_biomass_c_ag_preserved_in_conversion = vec_young_biomass_c_ag_preserved_in_conversion
        self.vec_young_biomass_c_available_for_removals_total = vec_young_biomass_c_available_for_removals_total
        self.vec_young_biomass_c_bg_converted = vec_young_biomass_c_bg_converted

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
            if not sf.isnumber(n_tps_no_withdrawals_new_growth, integer = True)
            else max(n_tps_no_withdrawals_new_growth, 1)
        )


        ##  SET PROPERTIES

        self.n_tps_no_withdrawals_new_growth = n_tps_no_withdrawals_new_growth
        self.vec_sequestration_per_tp_new = vec_sequestration_per_tp_new
        self.vec_sequestration_per_tp_new_cumulative = vec_sequestration_per_tp_new_cumulative

        return None
    


    def _verify_convert_array_input_to_array(self,
        vec: Union[float, np.ndarray],
        dims: Union[int, Tuple],
        varname: str,
    ) -> np.ndarray:
        """Verify input types for an array and convert to an NumPy array. Allows
            for initialization using a single number, which is then assumed to
            be repeated for the length of the specified dimensions. Ensures
            that arrays have the correct shape.  
        """
        
        # first, check type
        ok = sf.isnumber(vec) | isinstance(vec, np.ndarray)
        if not ok:
            tp = type(vec)
            raise TypeError(f"Invalid type '{tp}' found for {varname}. Must be a number of an array")

        # verify dimensions
        dims = (dims, ) if sf.isnumer(dims, integer = True, ) else dims
        if not isinstance(dims, tuple):
            tp = type(dims)
            raise TypeError(f"Invalid type '{tp}' found for dims. Must be an integer or a tuple")
        
        # convert vector
        vec = (
            vec*np.ones(dims, )
            if not isinstance(vec, np.ndarray)
            else vec
        )

        # finally, verify shape
        if vec.shape != dims:
            raise ShapeError(f"Invalid shape {vec.shape} specified for {varname}. Must be {dims}")
        

        return vec
    




    ###################################
    ###                             ###
    ###    PRIMARY FUNCTIONALITY    ###
    ###                             ###
    ###################################


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
        
        
        return None#
    
    