"""

"""

import numpy as np
import pandas as pd
import sisepuede.utilities._toolbox as sf
from typing import *


class ShapeError(Exception):
    pass


"""
FOR USE IN AFOLU WRAPPER

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




        
        model_attributes: 'ModelAttributes',
        cat_frst_secondary: str,
        cats_lndu_track: Union[List[str], None],
        dict_lndu_to_frst: Dict[str, str],
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],



        model_attributes: 'ModelAttributes',
        modvar_frst_frac_c_converted_available: Union['ModelVariable', str],
        modvar_frst_frac_max_degradation: Union['ModelVariable', str],
        cat_frst_secondary: str,
        cats_lndu_track: Union[List[str], None],    
        dict_lndu_to_frst: Dict[str, str],


        def _initialize_attributes(self,
        
        n_tp: int,
    ) -> None:
        Initialize key attributes used to manage land use classes
        

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
""";





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


    n_tp: int,
    vec_area_init : np.ndarray
        Vector (length 2) of initial areas for primary and secondary forest
    vec_biomass_c_ag_init_stst_storage : np.ndarray
        Vector (length 2) of initial steady state storage of above-ground 
        biomass in primary (index 0) and secondary (index 1) forest in terms of 
        mass per unit area.
    vec_biomass_c_bg_to_ag_ratio : np.ndarray
        Vector (length 2) giving the ratio of below-ground biomass to 
        above-ground biomass for primary and secondary forest
    vec_frac_biomass_adjustment_threshold : Union[float, np.ndarray]
        Vector (length T) or float giving adjustment threshold, as a proportion,

        vec_frac_biomass_buffer: Union[float, np.ndarray],
        vec_frac_biomass_dead_storage: Union[float, int, np.ndarray],
        vec_frac_biomass_from_conversion_available_for_use: Union[float, int, np.ndarray],
        vec_sf_nominal_initial: np.ndarray,
    vec_total_removals_demanded: np.ndarray
        Vector (n_time_periods) of exogenous demands for removals (from fuelwood
        and harvested wood products). If None, sets to 0.
    vec_young_sf_curve: np.ndarray
        Vector storing sequestration rates (mass C/area/time_period) for new
        growth. Can be derived from NPP curve.

    vec_init_by_cat_area : np.ndarray
        V
    vec_init_by_cat_c_stock_per_area : np.ndarray
        Vector (length n_cats) of initial C stock per unit area, ordered by 
        cats_lndu_track
    vec_init_by_cat_seq_per_area : np.ndarray
        Vector (length n_cats) of initial C sequestration per unit area, 
        ordered by cats_lndu_track
    vec_sequestration_per_tp_new : np.ndarray
        

        
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
        
        * `arr_biomass_c_removals_from_converted_land_allocation` (T x N)
            Allocation of C removals from converted land 

        * `arr_orig_allocation_removals` (T x N)
            Fraction of total pool used to allocated withdrawls by forest type.
            Columns sum to 1.

        * `arr_orig_biomass_c_ag_available_from_conversion` (T x N)
            Above-ground biomass C from conversion made available for removals.
        
        * `arr_orig_biomass_c_ag_average_per_area_no_ds` (T x N)
            Average biomass stock per area in original forest j at time i 
            excluding dead storage.

        * `arr_orig_biomass_c_ag_converted_away` (T x N)
            Total stock of above-ground biomass C that is converted to a 
            different land use category BEFORE removals and minimum needed to
            stay for target land use class.

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

        * `vec_area_protected_young` (T x 1)
            Vector storing the area of protected young forest under the
            assumption that it is protected AFTER original forest.

        * `vec_biomass_c_ag_init_healthy_available` (N x 1)
            Vector storing the initial amount of health stock available by 
            category.

        * `vec_biomass_c_ag_min_reqd_per_area` (T x N)
            Array that stores the minimum required biomass per area as the
            outer product of `vec_biomass_c_ag_init_stst_storage` and 
            `vec_frac_biomass_dead_storage`.
        
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
        n_tp: int,
        vec_area_init: np.ndarray,
        vec_biomass_c_ag_init_stst_storage: np.ndarray,
        vec_biomass_c_bg_to_ag_ratio: np.ndarray,
        vec_frac_biomass_adjustment_threshold: Union[float, np.ndarray],
        vec_frac_biomass_buffer: Union[float, np.ndarray],
        vec_frac_biomass_dead_storage: Union[float, int, np.ndarray],
        vec_frac_biomass_from_conversion_available_for_use: Union[float, int, np.ndarray],
        vec_sf_nominal_initial: np.ndarray,
        vec_total_removals_demanded: np.ndarray,
        vec_young_sf_curve: np.ndarray,
        n_tps_no_withdrawals_new_growth: int = 20,
        **kwargs,
    ) -> None:
        

        self._initialize_attributes(
            n_tp,
            n_tps_no_withdrawals_new_growth = n_tps_no_withdrawals_new_growth,
        )

        self._initialize_arrays(
            vec_area_init,
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


        return None
    
    


    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _build_arr_young_sf_base_by_tp_planted(self,
        vec_young_sf_curve: np.ndarray,
    ) -> np.ndarray:
        """Build the arr_young_biomass_c_ag_converted_by_tp_planted array
        """

        arr_young_sf_base_by_tp_planted = np.zeros((self.n_tp, self.n_tp, ))

        for i in range(self.n_tp):
            n_end = self.n_tp - i
            arr_young_sf_base_by_tp_planted[i:self.n_tp, i] = vec_young_sf_curve[0:n_end]

        return arr_young_sf_base_by_tp_planted
    


    def _check_initialization_arrays(self,
        vec_area_init: np.ndarray,
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

        # initial area
        vec_area_init = self._verify_convert_array_input_to_array(
            vec_area_init,
            self.n_cats,
            "vec_area_init",
        )

        # initial steady-state storage
        vec_biomass_c_ag_init_stst_storage = self._verify_convert_array_input_to_array(
            vec_biomass_c_ag_init_stst_storage,
            self.n_cats,
            "vec_biomass_c_ag_init_stst_storage",
        )

        # initial below ground biomass to above ground biomass ratio
        vec_biomass_c_bg_to_ag_ratio = self._verify_convert_array_input_to_array(
            vec_biomass_c_bg_to_ag_ratio,
            self.n_cats,
            "vec_biomass_c_bg_to_ag_ratio",
        )

        # removals adjustment threshold
        vec_frac_biomass_adjustment_threshold = self._verify_convert_array_input_to_array(
            vec_frac_biomass_adjustment_threshold,
            self.n_tp,
            "vec_frac_biomass_adjustment_threshold",
        )

        # biomass buffer
        vec_frac_biomass_buffer = self._verify_convert_array_input_to_array(
            vec_frac_biomass_buffer,
            self.n_tp,
            "vec_frac_biomass_buffer",
        )

        # initial dead storage fraction
        vec_frac_biomass_dead_storage = self._verify_convert_array_input_to_array(
            vec_frac_biomass_dead_storage,
            self.n_tp,
            "vec_frac_biomass_dead_storage",
        )

        # how much converted biomass (ag) is available for use
        vec_frac_biomass_from_conversion_available_for_use  = self._verify_convert_array_input_to_array(
            vec_frac_biomass_from_conversion_available_for_use,
            self.n_tp,
            "vec_frac_biomass_from_conversion_available_for_use",
        )

        # initial annualized sequestration factors
        vec_sf_nominal_initial = self._verify_convert_array_input_to_array(
            vec_sf_nominal_initial,
            self.n_cats,
            "vec_sf_nominal_initial",
        )

        # vector of total C removals
        vec_total_removals_demanded = self._verify_convert_array_input_to_array(
            vec_total_removals_demanded,
            self.n_tp,
            "vec_total_removals_demanded",
        )
        
        # check the sequestration factor curve for young forests
        vec_young_sf_curve = self._verify_convert_array_input_to_array(
            vec_young_sf_curve,
            self.n_tp,
            "vec_young_sf_curve",
        )


        ##  INITIALIZE SOME OTHER DEPENDENTS

        vec_biomass_c_ag_min_reqd_per_area = vec_biomass_c_ag_init_stst_storage*vec_frac_biomass_dead_storage[0]
        vec_biomass_c_ag_init_healthy_available = vec_biomass_c_ag_init_stst_storage - vec_biomass_c_ag_min_reqd_per_area


        ##  RETURN

        out = (
            vec_area_init,
            vec_biomass_c_ag_init_healthy_available,
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_ag_min_reqd_per_area,
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
    


    def _get_sf_adjustment_factor(self,
        i: int,
        arr_stock: np.ndarray,
        arr_stock_untouched: np.ndarray,
    ) -> np.ndarray:
        """Support for young and original forests. Set vec strock (for tp i - 1)
            and vec_stock untouched (for i - 1), and calculate the adjustment
            factor accordingly.
        """

        if i == 0:
            out = np.ones(self.n_cats, )
            return out
        
        ##  START WITH THE ADJUSTMENT FACTOR arr_orig_sf_adjustment_factor

        # shortcuts
        frac_dead_storage = self.vec_frac_biomass_dead_storage[i - 1]
        thresh_adjustment = self.vec_frac_biomass_adjustment_threshold[i - 1]
        vec_stock = arr_stock[i - 1]
        vec_stock_untouched = arr_stock_untouched[i - 1]
        
        # the numerator and denominator compare stock available outside of dead storage
        vec_num = np.nan_to_num(
            vec_stock/vec_stock_untouched - frac_dead_storage,
            nan = 0.0,
            posinf = 0.0,
        )
        vec_denom = thresh_adjustment - frac_dead_storage

        # adjustment only occurs between adjustment thresh and dead storage
        vec_adj = np.clip(
            np.nan_to_num(
                vec_num/vec_denom,
                nan = 0.0,
                posinf = 0.0,
            ),
            0,
            1,
        )

        for j in range(vec_adj.shape[0]):
            if vec_stock[j] == 0:
                vec_adj[j] = 1

        return vec_adj
    


    def _get_vec_frac_biomass_ag_decomposition(self,
        vec_biomass_c_ag_init_stst_storage: np.ndarray,
        vec_sf_nominal_initial: np.ndarray,
    ) -> np.ndarray:
        """Initialize the fraction of above-ground biomass that is assumed to
            decompose each year. This is estimated under the assumption that
            primary forests are in equilibrium--i.e., they are a net 0 for 
            sequestration given that as much carbon decomposes as is sequestered
            by the forest.
        """
        
        # get indices
        ind_fp = self.ind_frst_primary
        ind_fs = self.ind_frst_secondary

        # ratio of loss to
        frac = vec_sf_nominal_initial[ind_fp]/vec_biomass_c_ag_init_stst_storage[ind_fp]
        vec_frac_biomass_ag_decomposition = self._verify_convert_array_input_to_array(
            frac,
            self.n_tp,
            "vec_frac_biomass_ag_decomposition",
        )

        return vec_frac_biomass_ag_decomposition
    


    def _initialize_attributes(self,
        n_tp: int,
        n_tps_no_withdrawals_new_growth: int = 20,
    ) -> None:
        """Initialize key attributes used to manage land use classes
        """

        if not sf.isnumber(n_tp, integer = True, ):
            raise TypeError(f"Invalid type for n_tp. Must be an integer.")


        ##  CHECK NUMBER OF TIME PERIODS WITH NO WITHDRAWALS FOR NEWLY PLANTED FORESTS

        n_tps_no_withdrawals_new_growth = (
            20 
            if not sf.isnumber(n_tps_no_withdrawals_new_growth, integer = True)
            else max(n_tps_no_withdrawals_new_growth, 1)
        )


        ##  SET PROPERTIES

        self.ind_frst_primary = 0
        self.ind_frst_secondary = 1
        self.n_cats = 2
        self.n_tp = n_tp
        self.n_tps_no_withdrawals_new_growth = n_tps_no_withdrawals_new_growth
    
        return None
    


    def _initialize_arrays(self,
        vec_area_init: np.ndarray,
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

        n_cats = self.n_cats
        n_tp = self.n_tp

        shape_by_cat = (n_tp, n_cats)
    
        ##  INITIALIZE ARRAYS

        # shape T x N
        arr_area = np.zeros(shape_by_cat, )
        arr_area_protected_original = np.zeros(shape_by_cat, )
        arr_area_protected_total = np.zeros(shape_by_cat, )
        arr_area_remaining_from_orig = np.zeros(shape_by_cat, )
        arr_area_conversion_away_total = np.zeros(shape_by_cat, )
        arr_area_conversion_into = np.zeros(shape_by_cat, )
        arr_area_remaining_from_orig_after_conversion_away = np.zeros(shape_by_cat, )

        arr_biomass_c_ag_available_from_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_average_ag_stock_in_conversion_targets = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_lost_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_ag_lost_decomposition = np.zeros(shape_by_cat, )
        arr_biomass_c_bg_lost_conversion = np.zeros(shape_by_cat, )
        arr_biomass_c_bg_lost_removals = np.zeros(shape_by_cat, )
        arr_biomass_c_removals_from_converted_land_allocation = np.zeros(shape_by_cat, )
        arr_biomass_c_removed_from_forests_excluding_conversion = np.zeros(shape_by_cat, )

        arr_orig_allocation_removals = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_available_from_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_average_per_area = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_average_per_area_no_ds = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_converted_away = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_if_untouched = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_preserved_in_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_ag_starting = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_allocation_adjusted = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_allocation_excluding_conversion = np.zeros(shape_by_cat, )
        arr_orig_biomass_c_removed_from_forests = np.zeros(shape_by_cat, )
        arr_orig_frac_stock_available = np.zeros(shape_by_cat, )
        arr_orig_frac_removables_satisfiable = np.zeros(shape_by_cat, )
        arr_orig_sf_adjusted = np.zeros(shape_by_cat, )
        arr_orig_sf_adjustment_factor = np.zeros(shape_by_cat, )

        arr_total_biomass_c_ag_starting = np.zeros(shape_by_cat, )
        arr_total_biomass_c_bg_starting = np.zeros(shape_by_cat, )



        ##  INITIALIZE VECTORS

        # by time period
        vec_area_conversion_away_young_forest = np.zeros(n_tp, )
        vec_area_conversion_away_young_forest_no_protection = np.zeros(n_tp, )
        vec_area_protected_young = np.zeros(n_tp, )
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
            vec_area_init,
            vec_biomass_c_ag_init_healthy_available,
            vec_biomass_c_ag_init_stst_storage,
            vec_biomass_c_ag_min_reqd_per_area,
            vec_biomass_c_bg_to_ag_ratio,
            vec_frac_biomass_adjustment_threshold,
            vec_frac_biomass_buffer,
            vec_frac_biomass_dead_storage,
            vec_frac_biomass_from_conversion_available_for_use,
            vec_sf_nominal_initial,
            vec_total_removals_demanded,
            vec_young_sf_curve,
        ) = self._check_initialization_arrays(
            vec_area_init,
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

        vec_young_sf_curve_cumulative = np.cumsum(vec_young_sf_curve)


        # set the young forest internal calculation arrays
        self._initialize_young_forest_internal_arrays(vec_young_sf_curve, )


        ##  GET ANY DERIVATIVE/CALCULATED VALUES HERE

        vec_frac_biomass_ag_decomposition = self._get_vec_frac_biomass_ag_decomposition(
            vec_biomass_c_ag_init_stst_storage,
            vec_sf_nominal_initial,
        )


        ##  DO ANY UPDATES TO INITIAL ARRAYS

        arr_area[0] = vec_area_init
        arr_area_remaining_from_orig[0] = vec_area_init


        ##  SET PROPERTIES

        self.arr_area = arr_area
        self.arr_area_protected_original = arr_area_protected_original
        self.arr_area_protected_total = arr_area_protected_total
        self.arr_area_remaining_from_orig = arr_area_remaining_from_orig
        self.arr_area_remaining_from_orig_after_conversion_away = arr_area_remaining_from_orig_after_conversion_away
        self.arr_area_conversion_away_total = arr_area_conversion_away_total
        self.arr_area_conversion_into = arr_area_conversion_into
        self.arr_biomass_c_ag_available_from_conversion = arr_biomass_c_ag_available_from_conversion
        self.arr_biomass_c_average_ag_stock_in_conversion_targets = arr_biomass_c_average_ag_stock_in_conversion_targets
        self.arr_biomass_c_ag_lost_conversion = arr_biomass_c_ag_lost_conversion
        self.arr_biomass_c_ag_lost_decomposition = arr_biomass_c_ag_lost_decomposition
        self.arr_biomass_c_bg_lost_conversion = arr_biomass_c_bg_lost_conversion
        self.arr_biomass_c_bg_lost_removals = arr_biomass_c_bg_lost_removals
        self.arr_biomass_c_removals_from_converted_land_allocation = arr_biomass_c_removals_from_converted_land_allocation
        self.arr_biomass_c_removed_from_forests_excluding_conversion = arr_biomass_c_removed_from_forests_excluding_conversion
        self.arr_orig_allocation_removals = arr_orig_allocation_removals
        self.arr_orig_biomass_c_ag_available_from_conversion = arr_orig_biomass_c_ag_available_from_conversion
        self.arr_orig_biomass_c_ag_average_per_area = arr_orig_biomass_c_ag_average_per_area
        self.arr_orig_biomass_c_ag_average_per_area_no_ds = arr_orig_biomass_c_ag_average_per_area_no_ds
        self.arr_orig_biomass_c_ag_converted_away = arr_orig_biomass_c_ag_converted_away
        self.arr_orig_biomass_c_ag_if_untouched = arr_orig_biomass_c_ag_if_untouched
        self.arr_orig_biomass_c_ag_preserved_in_conversion = arr_orig_biomass_c_ag_preserved_in_conversion
        self.arr_orig_biomass_c_ag_starting = arr_orig_biomass_c_ag_starting
        self.arr_orig_biomass_c_allocation_adjusted = arr_orig_biomass_c_allocation_adjusted
        self.arr_orig_biomass_c_allocation_excluding_conversion = arr_orig_biomass_c_allocation_excluding_conversion
        self.arr_orig_biomass_c_removed_from_forests = arr_orig_biomass_c_removed_from_forests
        self.arr_orig_frac_stock_available = arr_orig_frac_stock_available
        self.arr_orig_frac_removables_satisfiable = arr_orig_frac_removables_satisfiable
        self.arr_orig_sf_adjusted = arr_orig_sf_adjusted
        self.arr_orig_sf_adjustment_factor = arr_orig_sf_adjustment_factor
        self.arr_total_biomass_c_ag_starting = arr_total_biomass_c_ag_starting
        self.arr_total_biomass_c_bg_starting = arr_total_biomass_c_bg_starting
        self.vec_area_conversion_away_young_forest = vec_area_conversion_away_young_forest
        self.vec_area_conversion_away_young_forest_no_protection = vec_area_conversion_away_young_forest_no_protection
        self.vec_area_init = vec_area_init
        self.vec_area_protected_young = vec_area_protected_young
        self.vec_biomass_c_ag_init_healthy_available = vec_biomass_c_ag_init_healthy_available
        self.vec_biomass_c_ag_init_stst_storage = vec_biomass_c_ag_init_stst_storage
        self.vec_biomass_c_ag_min_reqd_per_area = vec_biomass_c_ag_min_reqd_per_area
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
        self.vec_young_sf_curve_cumulative = vec_young_sf_curve_cumulative

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
        arr_young_sf_base_by_tp_planted = self._build_arr_young_sf_base_by_tp_planted(
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
        ok = sf.isnumber(vec) | isinstance(vec, (np.ndarray, list))
        if not ok:
            tp = type(vec)
            raise TypeError(f"Invalid type '{tp}' found for {varname}. Must be a number of an array")

        # verify dimensions
        dims = (dims, ) if sf.isnumber(dims, integer = True, ) else dims
        if not isinstance(dims, tuple):
            tp = type(dims)
            raise TypeError(f"Invalid type '{tp}' found for dims. Must be an integer or a tuple")
        
        # convert vector
        vec = (
            vec*np.ones(dims, )
            if sf.isnumber(vec)
            else np.array(vec)
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
        vec_area_converted_away: Union[list, np.ndarray],
        vec_area_protected: np.ndarray,
        vec_biomass_c_average_ag_stock_in_conversion_targets: np.ndarray,
        unsafe: bool = False,
    ) -> None:
        """Update the ledger with land use lose

        Function Arguments
        ------------------
        i : int
            Time period to update (row index)
        area_new_forest : float
            Area of new (planted or regenerated) forest entering the young 
            secondary pipeline
        vec_area_converted_away : float
            Ordered vector of total land use area converted away from tracked 
            land use types
        vec_area_protected : np.ndarray 
            Ordered vector of of protected land use area
        vec_biomass_c_average_ag_stock_in_conversion_targets : np.ndarray
            Ordered vector of per unit area average carbon stock in target 
            land use classes--i.e., classes to which each forest type is
            transitioning into in time period i. This is used to restrict
            available removals from conversion and preserve logical consistency.
        """
        # skip if invalid
        if (i < 0) | (i >= self.n_tp):
            return None
        
        # update other inputs, including average stock in target classes
        self._update_other_inputs(
            i,
            vec_biomass_c_average_ag_stock_in_conversion_targets,
            unsafe = unsafe,
        )
        
        # update area inputs
        self._update_areas(
            i,
            area_new_forest,
            vec_area_converted_away,
            vec_area_protected,
            unsafe = unsafe,
        )
        
        # update young biomass
        self._update_forest_biomass_young(i, )

        # update original
        self._update_forest_biomass_original(i, )


        return None
    


    def _update_areas(self,
        i: int,
        area_new_forest: float,
        vec_area_converted_away: Union[list, np.ndarray],
        vec_area_protected: np.ndarray,
        unsafe: bool = False,
    ) -> None:
        """Update area arrays in support of _update(). Updates the following
            arrays:

            * arr_area
            * arr_area_conversion_away_total
            * arr_area_conversion_into
            * arr_area_protected_original
            * arr_area_protected_total
            * arr_area_protected_young
            * arr_area_remaining_from_orig
            * arr_area_remaining_from_orig_after_conversion_away
            * vec_area_conversion_away_young_forest
            * vec_area_conversion_away_young_forest_no_protection

        """

        ##  INITIALIZATION

        # check types
        if sf.islistlike(vec_area_converted_away):
            vec_area_converted_away = np.array(vec_area_converted_away)

        if sf.islistlike(vec_area_protected):
            vec_area_protected = np.array(vec_area_protected)

        # shortcuts
        ind_fs = self.ind_frst_secondary

        # check shapes?
        if not unsafe:
            if not sf.isnumber(area_new_forest):
                raise TypeError(f"area_new_forest must be a number type.")

            self._verify_convert_array_input_to_array(
                vec_area_converted_away,
                self.n_cats,
                "vec_area_converted_away"
            )

            self._verify_convert_array_input_to_array(
                vec_area_protected,
                self.n_cats,
                "vec_area_protected"
            )




        ##  UPDATE AREA ARRAYS (IN ORDER)
        
        # basic area arrays
        self.arr_area_conversion_away_total[i] = vec_area_converted_away
        self.arr_area_conversion_into[i, ind_fs] = area_new_forest

        # updates for key areas that are initialized on object creation but need to be updated
        if i > 0:

            # update area remaining from original forest
            self.arr_area_remaining_from_orig[i] = np.clip(
                self.arr_area_remaining_from_orig[i - 1] - vec_area_converted_away,
                0,
                np.inf,
            )

            # update area total
            self.arr_area[i] = np.clip(
                self.arr_area[i - 1] - vec_area_converted_away,
                0,
                np.inf,
            )
            self.arr_area[i] += self.arr_area_conversion_into[i - 1]

        # area of original after converting away
        self.arr_area_remaining_from_orig_after_conversion_away[i] = np.clip(
            self.arr_area[i] - vec_area_converted_away,
            0,
            np.inf,
        )

        # protected area
        self.arr_area_protected_total[i] = vec_area_protected
        
        
        ##  UPDATE PROTECTED AREAS

        area_protected_young = max(
            vec_area_protected[1] - self.arr_area_remaining_from_orig_after_conversion_away[i, ind_fs],
            0, 
        )
        vec_protected_original = np.min(
            [
                self.arr_area[i],
                vec_area_protected
            ],
            axis = 0,
        )

        # assign
        self.arr_area_protected_original[i] = vec_protected_original
        self.vec_area_protected_young[i] = area_protected_young


        ##  UPDATE OTHER AREAS DEPENDENT ON PROTECTED

        # update the hypothetical area converted away from young forest (EXCLUDING protection)
        area_conversion_away_young_forest_no_protection = -1*min(
            self.arr_area_remaining_from_orig[i, ind_fs] - vec_area_converted_away[1],
            0
        )
        self.vec_area_conversion_away_young_forest_no_protection[i] = area_conversion_away_young_forest_no_protection

        # assign the area of forest that is converted away
        area_conversion_away_young_forest = max(
            self.vec_area_conversion_away_young_forest_no_protection[i] - self.vec_area_protected_young[i],
            0
        )
        self.vec_area_conversion_away_young_forest[i] = area_conversion_away_young_forest


        return None



    def _update_forest_biomass_original(self,
        i: int,
    ) -> None:
        """Update the young biomass matrices.
        """

        # sequestration factors that are adjusted by previous time period carbon stock in yf
        self._update_of_dynamic_sequestration(i, )
        
        # update starting C stock
        self._update_of_c_stock(i, )

        # conversion and removal allocations
        self._update_of_conversion_and_removals_allocations(i, )

        # availability of c in pool
        self._update_of_c_availability(i, )

        # removals
        self._update_of_removals(i, )

        # finally, add some additional emissions of interest
        self._update_of_additional_losses(i, )

        return None
    


    def _update_forest_biomass_young(self,
        i: int,
    ) -> None:
        """Update the young biomass matrices.
        """

        # start with area, including conversions
        self._update_yf_area(i, )

        # move to untouched C stock counterfactual, which is only dependent on area
        self._update_yf_c_stock_untouched(i, )

        # conversions of biomass, including amount that remains, for AG/BG
        self._update_yf_biomass_conversions(i, )

        # allocation of removals from young forests
        self._update_yf_biomass_removals_allocations(i, )

        # sequestration factors that are adjusted by previous time period carbon stock in yf
        self._update_yf_dynamic_sequestration(i, )

        # loss from decomposition
        self._update_yf_biomass_loss_from_decomposition(i, )

        # finally, update stock in young forests
        self._update_yf_c_stock(i, )

        return None

        
    
    def _update_of_additional_losses(self,
        i: int,
    ) -> None:
        """Update additional system losses that need to be tracked for emission
            inventories. Updates:

            * arr_biomass_c_ag_lost_conversion
            * arr_biomass_c_ag_lost_decomposition
            * arr_biomass_c_bg_lost_conversion
            * arr_biomass_c_bg_lost_removals
            * vec_total_removals_met
            
        """


        # shortcuts
        frac_decomp = self.vec_frac_biomass_ag_decomposition[i]
        removals_from_orig = self.vec_biomass_c_removals_from_converted[i]
        vec_c_ag_conv = self.arr_orig_biomass_c_ag_converted_away[i]
        vec_c_ag_conv_pres = self.arr_orig_biomass_c_ag_preserved_in_conversion[i]
        vec_c_ag_removed = self.arr_orig_biomass_c_removed_from_forests[i]
        vec_c_ag_starting = self.arr_orig_biomass_c_ag_starting[i]
        vec_frac_c_rmv_alloc = self.arr_biomass_c_removals_from_converted_land_allocation[i]


        ##  UPDATES

        # 1. above-ground biomass lost to conversion: arr_biomass_c_ag_lost_conversion
        vec_c_ag_lost_conv = (
            vec_c_ag_conv - vec_c_ag_conv_pres - removals_from_orig*vec_frac_c_rmv_alloc
        )
        self.arr_biomass_c_ag_lost_conversion[i] = vec_c_ag_lost_conv


        # 2. above-ground biomass lost to decomposition: arr_biomass_c_ag_lost_decomposition
        vec_c_ag_lost_decomp = frac_decomp*(vec_c_ag_starting - vec_c_ag_conv - vec_c_ag_removed)
        self.arr_biomass_c_ag_lost_decomposition = vec_c_ag_lost_decomp


        # 2.     
        # 4. below-ground biomass lost to removals: arr_biomass_c_bg_lost_removals
        

        return None
    


    def _update_of_c_availability(self,
        i: int,
    ) -> None:
        """Update availability of C in forests for extraction. Updates:

            * arr_orig_allocation_removals
            * arr_orig_biomass_c_allocation_adjusted
            * arr_orig_frac_removables_satisfiable
            * arr_orig_frac_stock_available
            * vec_orig_biomass_c_accessible_pool
            
        """

        # shortcuts
        frac_buffer = self.vec_frac_biomass_buffer[i]
        frac_dead_storage = self.vec_frac_biomass_dead_storage[i]
        vec_removals_alloc = self.arr_orig_biomass_c_allocation_excluding_conversion[i]


        ##  UPDATE

        # 1. set arr_orig_frac_stock_available
        vec_frac_available = (
            self.arr_orig_biomass_c_ag_average_per_area_no_ds[i]
            /self.vec_biomass_c_ag_init_healthy_available
        )
        self.arr_orig_frac_stock_available[i] = vec_frac_available


        # 2. get fraction of removals that are satisfiable
        vec_frac_satisfiable = np.clip(
            (vec_frac_available - frac_dead_storage)/frac_buffer,
            0,
            1,
        )
        self.arr_orig_frac_removables_satisfiable[i] = vec_frac_satisfiable

        
        # 3. get adjusted removals satisfiable and allocated + total accessible pool + allocation frac:
        #    - arr_orig_allocation_removals
        #    - arr_orig_biomass_c_allocation_excluding_conversion
        #    - vec_orig_biomass_c_accessible_pool
        vec_alloc_adjusted = vec_removals_alloc*vec_frac_satisfiable
        alloc_total = vec_alloc_adjusted.sum()

        self.arr_orig_biomass_c_allocation_adjusted[i] = vec_alloc_adjusted
        self.vec_orig_biomass_c_accessible_pool[i] = alloc_total
        self.arr_orig_allocation_removals[i] = vec_alloc_adjusted/alloc_total


        return None
    

    
    def _update_of_c_stock(self,
        i: int, 
    ) -> None:
        """Update C stock in original forests. Updates:

            
            * arr_orig_biomass_c_ag_average_per_area
            * arr_orig_biomass_c_ag_average_per_area_no_ds
            * arr_orig_biomass_c_ag_starting

        """

        vec_area_remaining_orig = self.arr_area_remaining_from_orig[i]

        # if initializing, calculation is simple
        if i == 0:
            self.arr_orig_biomass_c_ag_starting[i] = (
                vec_area_remaining_orig
                * self.vec_biomass_c_ag_init_stst_storage
            )

            self.arr_orig_biomass_c_ag_average_per_area[i] = self.vec_biomass_c_ag_init_stst_storage

            self.arr_orig_biomass_c_ag_average_per_area_no_ds[i] = np.clip(
                self.vec_biomass_c_ag_init_stst_storage - self.vec_biomass_c_ag_min_reqd_per_area,
                0,
                np.inf,
            )

            return None


        ##  GET ABOVE-GROUND STOCK arr_orig_biomass_c_ag_starting
        
        # shortcuts
        vec_conv_prev = self.arr_orig_biomass_c_ag_converted_away[i - 1]
        vec_decomp_prev = self.arr_biomass_c_ag_lost_decomposition[i - 1]
        vec_removals_prev = self.arr_orig_biomass_c_removed_from_forests[i - 1]
        vec_seq = self.arr_orig_sf_adjusted[i]
        vec_stock_prev = self.arr_orig_biomass_c_ag_starting[i - 1]

        vec_stock = (
            vec_stock_prev
            - vec_conv_prev
            - vec_decomp_prev
            - vec_removals_prev
            + vec_seq*vec_area_remaining_orig
        )

        # update array
        self.arr_orig_biomass_c_ag_starting[i] = vec_stock


        ##  UPDATE AVERAGE arr_orig_biomass_c_ag_average_per_area and arr_orig_biomass_c_ag_average_per_area_no_ds

        vec_avg = np.nan_to_num(
            vec_stock/vec_area_remaining_orig,
            nan = 0.0,
            posinf = 0.0,
        )

        self.arr_orig_biomass_c_ag_average_per_area[i] = vec_avg
        self.arr_orig_biomass_c_ag_average_per_area_no_ds[i] = np.clip(
            vec_avg - self.vec_biomass_c_ag_min_reqd_per_area,
            0,
            np.inf,
        )

        return None
    


    def _update_of_conversion_and_removals_allocations(self,
        i: int, 
    ) -> None:
        """Update C stock allocations for conversion and removals. Updates:

            * arr_orig_biomass_c_ag_converted_away
            * arr_biomass_c_removals_from_converted_land_allocation
            * arr_orig_biomass_c_ag_available_from_conversion
            * arr_orig_biomass_c_ag_preserved_in_conversion
            * arr_orig_biomass_c_allocation_excluding_conversion
            * arr_total_biomass_c_ag_starting
            * arr_total_biomass_c_bg_starting
            * vec_biomass_c_removals_from_converted
        """

        ##  INITIALIZATION

        ind_fs = self.ind_frst_secondary

        # some shortcuts
        c_demanded = self.vec_total_removals_demanded[i]
        frac_c_converted_avail = self.vec_frac_biomass_from_conversion_available_for_use[i]
        vec_area_conv = self.arr_area_conversion_away_total[i]
        vec_area_protected = self.arr_area_protected_original[i]
        vec_area_remaining = self.arr_area_remaining_from_orig_after_conversion_away[i]
        vec_c_ag_total = self.arr_orig_biomass_c_ag_starting[i]
        vec_c_avg_per_area = self.arr_orig_biomass_c_ag_average_per_area[i]
        vec_c_avg_per_area_no_ds = self.arr_orig_biomass_c_ag_average_per_area_no_ds[i]
        vec_c_avg_preserved = self.arr_biomass_c_average_ag_stock_in_conversion_targets[i]

        
        ##  UPDATES

        # 1. update total above-ground biomass by type: arr_total_biomass_c_ag_starting
        vec_c_ag_total_update = vec_c_ag_total.copy()
        vec_c_ag_total_update[ind_fs] += self.vec_young_biomass_c_ag_starting[i]
        self.arr_total_biomass_c_ag_starting[i] = vec_c_ag_total_update


        # 2. update total below-ground biomass by type: arr_total_biomass_c_bg_starting
        self.arr_total_biomass_c_bg_starting[i] = (
            self.arr_orig_biomass_c_ag_starting[i]
            * self.vec_biomass_c_bg_to_ag_ratio
        )


        # 3. array of biomass converted away (total including removals): arr_orig_biomass_c_ag_converted_away
        vec_c_converted = vec_c_avg_per_area*vec_area_conv
        self.arr_orig_biomass_c_ag_converted_away[i] = vec_c_converted


        # 4. original biomass that must be preserved due to average target: arr_orig_biomass_c_ag_preserved_in_conversion
        vec_c_preserved = np.clip(
            vec_c_avg_preserved*vec_area_conv,
            0,
            vec_c_converted,
        )
        self.arr_orig_biomass_c_ag_preserved_in_conversion[i] = vec_c_preserved


        # 5. biomass in original forest conversion actually available for use to satistfy removals: arr_orig_biomass_c_ag_available_from_conversion
        vec_c_converted_available = (vec_c_converted - vec_c_preserved)*frac_c_converted_avail
        self.arr_orig_biomass_c_ag_available_from_conversion[i] = vec_c_converted_available
        

        # 6. total removals from converted biomass
        self.vec_biomass_c_removals_from_converted[i] = min(
            vec_c_converted_available.sum(),
            c_demanded,
        )


        # 7. allocation fractions for removed from original conversions: arr_biomass_c_removals_from_converted_land_allocation
        self.arr_biomass_c_removals_from_converted_land_allocation[i] = vec_c_converted_available/vec_c_converted_available.sum()

        
        # 8. C that is allocated to forest types but excluding conversions: arr_orig_biomass_c_allocation_excluding_conversion
        vec_allocate_biomass_for_removal = np.clip(
            vec_area_remaining - vec_area_protected, 
            0,
            np.inf,
        )
        vec_allocate_biomass_for_removal *= vec_c_avg_per_area_no_ds
        self.arr_orig_biomass_c_allocation_excluding_conversion[i] = vec_allocate_biomass_for_removal

        return None
    


    def _update_of_dynamic_sequestration(self,
        i: int,
    ) -> None:
        """Update dynamic sequestration in original forests and associated
            arrays. Updates:

            * arr_orig_biomass_c_ag_if_untouched
            * arr_orig_sf_adjusted
            * arr_orig_sf_adjustment_factor
        """
        
        ##  INITIALIZATION

        # get untouched biomass arr_orig_biomass_c_ag_if_untouched
        self.arr_orig_biomass_c_ag_if_untouched[i] = (
            self.arr_area_remaining_from_orig[i]
            *self.vec_biomass_c_ag_init_stst_storage
        )


        # update adjustment factor arr_orig_sf_adjustment_factors
        vec_adj = self._get_sf_adjustment_factor(
            i,
            self.arr_orig_biomass_c_ag_starting,
            self.arr_orig_biomass_c_ag_if_untouched,
        )

        self.arr_orig_sf_adjustment_factor[i] = vec_adj


        # update adjusted sequestration arr_orig_sf_adjusted
        self.arr_orig_sf_adjusted[i] = self.vec_sf_nominal_initial*vec_adj

        return None
    


    def _update_of_removals(self,
        i: int, 
    ) -> None:
        """Update removal calculations from original forest. Updates:

            * arr_biomass_c_removed_from_forests_excluding_conversion
            * arr_orig_biomass_c_removed_from_forests
            * vec_biomass_c_removals_from_forest_demanded
            * vec_biomass_c_removed_from_original_demanded
            * vec_biomass_c_removed_from_original_unmet
            * vec_biomass_c_removed_from_young
            
        """


        ##  INITIALIZATION

        ind_fs = self.ind_frst_secondary

        # some shortcuts
        c_available_orig = self.vec_orig_biomass_c_accessible_pool[i]
        c_available_young = self.vec_young_biomass_c_available_for_removals_total[i]
        c_demanded = self.vec_total_removals_demanded[i]
        c_rmv_from_conv = self.vec_biomass_c_removals_from_converted[i]
        vec_orig_frac_removals_alloc = self.arr_orig_allocation_removals[i]


        ##  UPDATES

        # 1. update demand for removals from forests: vec_biomass_c_removals_from_forest_demanded
        c_demanded_from_forest = max(c_demanded - c_rmv_from_conv, 0)
        self.vec_biomass_c_removals_from_forest_demanded[i] = c_demanded_from_forest


        # 2. get actual removals from original forests
        c_removed_from_orig = min(c_demanded_from_forest, c_available_orig)
        self.vec_biomass_c_removed_from_original_demanded[i] = c_removed_from_orig
        

        # 3. unmet demand: vec_biomass_c_removed_from_original_unmet
        c_demand_unmet = c_demanded_from_forest - c_removed_from_orig
        self.vec_biomass_c_removed_from_original_unmet[i] = c_demand_unmet


        # 4. biomass removed from young forests: vec_biomass_c_removed_from_young
        c_removed_from_young = min(c_demand_unmet, c_available_young, )
        self.vec_biomass_c_removed_from_young[i] = c_removed_from_young


        # 5. get total biomass taken from each original forest type: arr_orig_biomass_c_removed_from_forests
        vec_orig_removals = vec_orig_frac_removals_alloc*c_removed_from_orig
        self.arr_orig_biomass_c_removed_from_forests[i] = vec_orig_frac_removals_alloc


        # 6. get total biomass taken from each forest type (incl. young): arr_biomass_c_removed_from_forests_excluding_conversion
        vec_total_removals = vec_orig_removals.copy()
        vec_total_removals[ind_fs] += c_removed_from_young
        self.arr_biomass_c_removed_from_forests_excluding_conversion[i] = vec_total_removals

        return None



    def _update_other_inputs(self,
        i: int,
        vec_biomass_c_average_ag_stock_in_conversion_targets: np.ndarray,
        unsafe: bool = False,
    ) -> None:
        """Update other necessary arrays in support of _update(). Updates the 
            following arrays:

            * 

        """
        
        # verify shape?
        if not unsafe:
            self._verify_convert_array_input_to_array(
                vec_biomass_c_average_ag_stock_in_conversion_targets,
                self.n_cats,
                "vec_biomass_c_average_ag_stock_in_conversion_targets",
            )


        self.arr_biomass_c_average_ag_stock_in_conversion_targets[i] = vec_biomass_c_average_ag_stock_in_conversion_targets


        return None
    


    def _update_yf_area(self,
        i: int,
    ) -> None:
        """Update the area of young biomass arrays. Updates:

            * arr_young_area_by_tp_planted
            * arr_young_area_by_tp_planted_cumvals
            * arr_young_area_by_tp_planted_drops

        """

        ind_fs = self.ind_frst_secondary


        ##  INITIALIZE NEW AREA PLANTED

        self.arr_young_area_by_tp_planted[i, i] = self.arr_area_conversion_into[i, ind_fs]

        # no more action needs to be taken if on the first iteration
        if i == 0:
            return None


        ##  UPDATE CUMULATIVE VALUES: arr_young_area_by_tp_planted_cumvals
        #
        #   needed for conversions out
        
        arr_cv = self.arr_young_area_by_tp_planted_cumvals
        arr_cv[i] = np.cumsum(self.arr_young_area_by_tp_planted[i - 1])
        self.arr_young_area_by_tp_planted_cumvals = arr_cv



        ##  UPDATE CONVERSIONS OUT: arr_young_area_by_tp_planted_drops

        arr = self.arr_young_area_by_tp_planted_drops
        area_conv_away = self.vec_area_conversion_away_young_forest[i]
        
        for j in range(i):

            # if the cumulative area to this point is less than the total 
            # converted away, that means ALL of the new forest planted in this
            # time period will be converted
            if arr_cv[i, j] < area_conv_away:
                arr[i, j] = self.arr_young_area_by_tp_planted[i - 1, j]
                continue
            
            # otherwise, only some portion of this area will be converted away
            # (or NONE, if the previous step has already been met)
            base = arr_cv[i, j - 1] if j > 1 else 0
            arr[i, j] = (
                area_conv_away - base
                if base < area_conv_away
                else 0 
            )

        self.arr_young_area_by_tp_planted_drops = arr



        ##  UPDATE AREA: arr_young_area_by_tp_planted
        
        # set as previous area less conversions out
        arr1 = self.arr_young_area_by_tp_planted
        arr2 = self.arr_young_area_by_tp_planted_drops

        arr1[i, 0:i] = arr1[i - 1, 0:i] - arr2[i, 0:i]

        self.arr_young_area_by_tp_planted = arr1

        return None    
        


    def _update_yf_biomass_conversions(self,
        i: int,
    ) -> None:
        """Update the area of young biomass arrays. Updates:

            * arr_young_biomass_c_ag_converted_by_tp_planted
            * arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted 
            * arr_young_biomass_c_bg_converted_by_tp_planted
            * vec_young_biomass_c_ag_converted
            * vec_young_biomass_c_ag_preserved_in_conversion
            * vec_young_biomass_c_bg_converted
        """

        ind_fs = self.ind_frst_secondary


        ##  UPDATE arr_young_biomass_c_ag_converted_by_tp_planted

        if i > 0:
            new_row = np.nan_to_num(
                self.arr_young_biomass_c_ag_stock[i - 1]
                *(
                    self.arr_young_area_by_tp_planted_drops[i]
                    /self.arr_young_area_by_tp_planted[i - 1]
                ),
                nan = 0.0,
                posinf = 0.0,
            )

            self.arr_young_biomass_c_ag_converted_by_tp_planted[i] = new_row
            self.vec_young_biomass_c_ag_converted[i] = new_row.sum()

        
        ##  UPDATE arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted

        vec_min = self.arr_young_biomass_c_ag_converted_by_tp_planted[i]
        vec_conv = self.arr_young_area_by_tp_planted_drops[i]
        avg_c_target_luc = self.arr_biomass_c_average_ag_stock_in_conversion_targets[i, ind_fs]

        vec_preserved = np.clip(
            vec_conv*avg_c_target_luc,
            0,
            vec_min
        )

        self.arr_young_biomass_c_ag_preserved_in_conversion_by_tp_planted[i] = vec_preserved
        self.vec_young_biomass_c_ag_preserved_in_conversion[i] = vec_preserved.sum()


        ##  UPDATE arr_young_biomass_c_bg_converted_by_tp_planted

        self.arr_young_biomass_c_bg_converted_by_tp_planted[i] = (
            self.arr_young_biomass_c_ag_converted_by_tp_planted[i]
            * self.vec_biomass_c_bg_to_ag_ratio[ind_fs]
        )

        self.vec_young_biomass_c_bg_converted[i] = self.arr_young_biomass_c_bg_converted_by_tp_planted[i].sum()


        return None
    


    def _update_yf_biomass_loss_from_decomposition(self,
        i: int,
    ) -> None:
        """Update the counterfactual "untouched" c stock array. Updates:

            * arr_young_biomass_c_loss_from_decomposition
        """

        # nothing to do at first time step
        if i == 0:
            return None

        # shortcuts
        rate_decomp = self.vec_frac_biomass_ag_decomposition[i - 1]
        vec_conv = self.arr_young_biomass_c_ag_converted_by_tp_planted[i]
        vec_removals_cur = self.arr_young_biomass_c_stock_removal_allocation[i]
        vec_stock_prev = self.arr_young_biomass_c_ag_stock[i - 1]
        
        # calculation and assignment
        vec_loss = (vec_stock_prev - vec_removals_cur - vec_conv)*rate_decomp

        self.arr_young_biomass_c_loss_from_decomposition[i] = vec_loss

        return None



    def _update_yf_biomass_removals_allocations(self,
        i: int,
    ) -> None:
        """Update the counterfactual "untouched" c stock array. Updates:

            * arr_young_biomass_c_available_for_removals_mask
            * arr_young_biomass_c_stock_removal_allocation
            * arr_young_biomass_c_stock_removal_allocation_aux
            * vec_young_biomass_c_available_for_removals_total
        """
        
        # no action is taken if withdrawals aren't available
        if i <= self.n_tps_no_withdrawals_new_growth:
            return None
        

        ##  UPDATE arr_young_biomass_c_available_for_removals_mask
        #      and vec_young_biomass_c_available_for_removals_total

        arr_mask = self.arr_young_biomass_c_available_for_removals_mask

        # calculation for each time period for which biomass are available
        inds_col = list(range(self.n_tps_no_withdrawals_new_growth + 1, i))
        
        # shortcuts
        biomass_ag_min_per_area = self.vec_biomass_c_ag_min_reqd_per_area[i]
        vec_areas_conv = self.arr_young_area_by_tp_planted[i - 1, inds_col]
        vec_biomass_converted = self.arr_young_biomass_c_ag_converted_by_tp_planted[i, inds_col]
        vec_biomass_stock_prev = self.arr_young_biomass_c_ag_stock[i - 1, inds_col]

        # update mask
        mask_new = vec_biomass_stock_prev - vec_areas_conv*biomass_ag_min_per_area
        mask_new -= vec_biomass_converted
        arr_mask[i, inds_col] = np.clip(mask_new, 0)

        self.arr_young_biomass_c_available_for_removals_mask = arr_mask
        self.vec_young_biomass_c_available_for_removals_total[i] = arr_mask[i].sum()


        ##  UPDATE arr_young_biomass_c_stock_removal_allocation

        # first, use arr_young_biomass_c_stock_removal_allocation_aux for cumulative biomass
        vec_mask_cur = self.arr_young_biomass_c_available_for_removals_mask[i]
        self.arr_young_biomass_c_stock_removal_allocation_aux[i] = np.cumsum(vec_mask_cur, )
        
        # shortcuts
        arr_alloc = self.arr_young_biomass_c_stock_removal_allocation
        c_removals_demanded_from_young = self.vec_biomass_c_removed_from_young[i]
        vec_aux = self.arr_young_biomass_c_stock_removal_allocation_aux[i]

        # iterate to add 
        for j in range(i):

            # if the cumulative area to this point is less than the total 
            # demanded, that means ALL of the available removals will have to be
            # sent for satisfaction
            if vec_aux[j] < c_removals_demanded_from_young:
                arr_alloc[i, j] = vec_mask_cur[j]
                continue
            
            # otherwise, only some portion of availble removals will actually
            # be removed--or NONE, if the previous step has already been met
            base = vec_aux[j - 1] if j > 1 else 0
            arr_alloc[i, j] = (
                c_removals_demanded_from_young - base
                if base < c_removals_demanded_from_young
                else 0 
            )

        # reassign 
        self.arr_young_biomass_c_stock_removal_allocation = arr_alloc

        return None
    


    def _update_yf_c_stock(self,
        i: int, 
    ) -> None:
        """Update C stock for above- and below-ground biomass. Updates:

            * arr_young_biomass_c_ag_stock
            * arr_young_biomass_c_bg_stock
            * vec_young_biomass_c_ag_starting
        """
        # there's no stock in first time period
        if i == 0:
            return None
        
        # secondary forest index
        ind_fs = self.ind_frst_secondary


        ##  UPDATE ABOVE-GROUND: arr_young_biomass_c_ag_stock

        vec_area = self.arr_young_area_by_tp_planted[i]
        vec_conv = self.arr_young_biomass_c_ag_converted_by_tp_planted[i]
        vec_decomp = self.arr_young_biomass_c_loss_from_decomposition[i]
        vec_removals = self.arr_young_biomass_c_stock_removal_allocation[i]
        vec_sf_adj = self.arr_young_sf_adjusted_by_tp_planted[i]
        vec_stock_prev = self.arr_young_biomass_c_ag_stock[i - 1]
        
        # current stock is previous stock removals and decomposition + and plus new sequestration
        # note that new sequestration 
        vec_stock_ag = (
            vec_stock_prev 
            - vec_conv
            - vec_decomp
            - vec_removals
            + vec_area*vec_sf_adj
        )

        self.arr_young_biomass_c_ag_stock[i] = vec_stock_ag
        self.vec_young_biomass_c_ag_starting[i] = vec_stock_ag.sum()


        ##  UPDATE BELOW-GROUND: arr_young_biomass_c_bg_stock

        vec_stock_bg = vec_stock_ag*self.vec_biomass_c_bg_to_ag_ratio[ind_fs]
        
        self.arr_young_biomass_c_bg_stock[i] = vec_stock_bg

        return None



    def _update_yf_c_stock_untouched(self,
        i: int,
    ) -> None:
        """Update the counterfactual "untouched" c stock array. Updates:

            * arr_young_biomass_c_ag_stock_if_untouched
        """

        arr_area = self.arr_young_area_by_tp_planted
        vec_sf = self.vec_young_sf_curve

        # case for init
        if i == 0:
            self.arr_young_biomass_c_ag_stock_if_untouched[i, i] = vec_sf[i]*arr_area[i, i]
            return None
        
        
        # add as previous stock*(1 - decomp rate ) plus sequestration
        frac_decomp = self.vec_frac_biomass_ag_decomposition[i]

        self.arr_young_biomass_c_ag_stock_if_untouched[i] = (
            self.arr_young_biomass_c_ag_stock_if_untouched[i - 1] * (1 - frac_decomp)
            + arr_area[i] * self.arr_young_sf_base_by_tp_planted[i]
        )
        
        return None
    


    def _update_yf_dynamic_sequestration(self,
        i: int,
    ) -> None:
        """Update dynamic sequestration in young biomass arrays. Updates:

            * arr_young_sf_adjusted_by_tp_planted
            * arr_young_sf_adjustment_factor
        """
        
        ##  INITIALIZATION

        # no action taken at time 0;
        if i == 0:
            self.arr_young_sf_adjustment_factor[i] = 1
            self.arr_young_sf_adjusted_by_tp_planted[i] = self.arr_young_biomass_c_ag_stock_if_untouched[i]
            return None
        

        ##  START WITH THE ADJUSTMENT FACTOR arr_young_sf_adjustment_factor

        # shortcuts
        vec_adj = self._get_sf_adjustment_factor(
            i,
            self.arr_young_biomass_c_ag_stock,
            self.arr_young_biomass_c_ag_stock_if_untouched,
        )

        self.arr_young_sf_adjustment_factor[i] = vec_adj


        ##  NEXT, UPDATE ACTUAL ADJUSTED SEQUESTRATION FACTOR arr_young_sf_adjusted_by_tp_planted

        vec_sf = self.arr_young_sf_base_by_tp_planted[i].copy()
        vec_sf[0:i] *= vec_adj[0:i]

        self.arr_young_sf_adjusted_by_tp_planted[i] = vec_sf


        return None

    
