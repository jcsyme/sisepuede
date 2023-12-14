def build_transition_df_for_year(self,
        transition_matrix: rgd.RegionalGriddedData,
        year: int,
        dataset_prepend: str = "array_luc",
        df_areas_base: Union[pd.DataFrame, None] = None,
        dict_fields_add: Union[dict, None] = None,
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """
        Construct a dataframe from the land use change datasets from year `year` to
            `year + 1`
            
        Returns None on errors or misspecification of input data (e.g., invalid year
            or type)
        
        
        Function Arguments
        ------------------
        - transition_matrix: RegionalTransitionMatrix containing GriddedData divided
            into land use arrays. The arrays must be accessile using
            transition_matrix.get_regional_array(f"{dataset_prepend}_{year}")
        - year: base year (output transition)
        
        
        Keyword Arguments
        -----------------
        - dataset_prepend: dataset prependage in RegionalTransitionMatrix used to 
            access land use classification data
        - df_areas_base: areas of land use type; used for splitting mangroves from 
            wetlands (if specified). Should be a single row or series
        - dict_fields_add: optional fields to add to output dataframe. Dictionary maps new
            column to value
        **kwargs: passed to self.add_data_frame_fields_from_dict
        """
        
        ##  INITIALIZE SOME INTERNAL VARIABLES

        model_afolu = self.model_afolu
        
        # check input types
        if not isinstance(transition_matrix, rgd.RegionalGriddedData):
            return None
        
        arr_0 = transition_matrix.get_regional_array(f"{dataset_prepend}_{year}")
        arr_1 = transition_matrix.get_regional_array(f"{dataset_prepend}_{year + 1}")
        vec_areas = transition_matrix.get_regional_array("cell_areas")
        if (arr_0 is None) | (arr_1 is None) | (vec_areas is None):
            return None
        
        
        ##  GET DATAFRAME AND MODIFY
        
        df = transition_matrix.get_transition_data_frame(
            arr_0,
            arr_1,
            vec_areas,
            include_pij = False,
        )
        
        
        # call transition matrix
        df_transition = self.convert_transition_classes_to_sisepuede(
            df,
            df_areas_base = df_areas_base,
        )
        
        mat = model_afolu.get_transition_matrix_from_long_df(
            df_transition, 
            self.field_category_i,
            self.field_category_j,
            self.field_probability_transition,
        ) 

        df_out = model_afolu.format_transition_matrix_as_input_dataframe(
            mat,
        )
        
        if isinstance(dict_fields_add, dict):

            df_out = sf.add_data_frame_fields_from_dict(
                df_out,
                dict_fields_add,
                **kwargs
            )
            
        
        return df_out