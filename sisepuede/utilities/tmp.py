from sisepuede.core.attribute_table import AttributeTable



# parameters a, b, c, d
_PARAMS_SEQUESTRATION_CURVE_EST = np.array([0.1323, 1.0642, 6.3342, 3.455], )








def get_frst_area_new_secondary(self,
    arr_converted: np.ndarray, 
    ind: Union[int, None] = None,
) -> float:
    """
    For a conversion matrix `arr_converted`, calculate how much secondary
        forest is newly formed.
    """
    ind = self.ind_lndu_fsts if not sf.isnumber(ind, integer = True) else ind

    n, _ = arr_converted.shape
    w = np.array([i for i in range(n) if i != ind])
    out = arr_converted[w, ind].sum()

    return out



def get_frst_sequestration_dynamic(self,
    df_afolu_trajectories: pd.DataFrame,
    arrs_lndu_conversion: np.ndarray,
    arrs_lndu_prevalence: np.ndarray,
    attr_lndu: Union[AttributeTable, None] = None,
    dynamic_forests: bool = True,
    max_age_young_secondary_forest: int = 20,
    **kwargs,
) -> Tuple:
    """
    Calculate forest sequestration. If using dynamic forest sequestration,
        estimates a sequestration curve and applies it over time. 
        Otherwise,

    NOTE: all secondary sequestration at time t = 0 

    
    Function Arguments
    ------------------
    - df_afolu_trajectories: data frame containing input trajectories

    Keyword Arguments
    ------------------
    - attr_lndu: optional land use attribute table
    - dynamic_forests: use dynamic forest sequestration?
    - max_age_young_secondary_forest: maximum age (in time periods) of young 
        secondary forests. Only applies if using dynamic forest 
        sequestration
    - kwargs: passed to the following methods:
         * get_npp_frst_sequestration_factor_vectors()
    """

    ##  SOME INIT

    attr_lndu = (
        self.model_attributes.get_attribute_table(self.model_attributes.subsec_name_lndu)
        if attr_lndu is None
        else attr_lndu
    )


    # get groups of sequestration factors
    df_sf_groups = self.get_npp_frst_sequestration_factor_vectors(
        df_afolu_trajectories,
        **kwargs,
    )



def get_frst_sequestration_factors(self,
    df_afolu_trajectories: pd.DataFrame,
    modvar_area: Union[str, mv.ModelVariable, None] = None,
    modvar_sequestration: Union[str, mv.ModelVariable, None] = None,
    override_vector_for_single_mv_q: bool = True,
    **kwargs,
) -> Tuple:
    """
    Retrieve the sequestration factors for forest in terms of 
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

    


        


