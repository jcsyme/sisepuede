   # set some properties
        cat_mangrove = self.model_afolu.cat_lndu_fstm
        dict_copernicus_to_sisepuede = self.dict_copernicus_to_sisepuede
        model_afolu = self.model_afolu
        model_attrbutes = self.model_attributes

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        field_category_i = self.field_category_i
        field_category_i = self.field_category_j
        field_probability_transition = self.field_probability_transition

        # land use classes
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new

    def convert_herbaceous_vegetation_to_related_class(
        df_transition: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Herbaceous wetlands (HW) includes a number of categories that are 
            included elsewhere in SISEPUEDE v1.0; in general, it likely 
            corresponds best with grassland. Notably, transitions into wetlands 
            tend to be the highest off-diagonal probabilities, which is likely 
            unrealistic. 
        
        To account for this, we assume that HWs are best accounted for by the 
            correspondence class, and we eliminate most dynamics. 
            
            
        Function Arguments
        ------------------
        - df_transition: data frame containing aggregated transitions with 
            Copernicus classes.
        
        Keyword Arguments
        -----------------
        """

        # set some properties
        cat_mangrove = self.model_afolu.cat_lndu_fstm
        dict_copernicus_to_sisepuede = self.dict_copernicus_to_sisepuede
        model_afolu = self.model_afolu
        model_attrbutes = self.model_attributes

        # fields
        field_area = self.field_area
        field_area_total_0 = self.field_area_total_0
        field_array_0 = self.field_array_0
        field_array_1 = self.field_array_1
        
        # land use classes
        luc_copernicus_herbaceous_wetland = self.luc_copernicus_herbaceous_wetland
        luc_copernicus_herbaceous_wetland_new = self.luc_copernicus_herbaceous_wetland_new

        
        # first, get dictionary mapping array_0 to area_luc_0
        dict_state_0_to_area_total = sf.build_dict(
            df_transition[[field_array_0, field_area_total_0]]
        )
        
        # initialize new columns
        vec_new_0 = list(df_transition[field_array_0])
        vec_new_1 = list(df_transition[field_array_1])
        vec_new_array_total = list(df_transition[field_area_total_0])
        
        # initialize list of output classes that wetlands have been converted to
        output_edges_converted_to_wetlands = []
        
        for i, row in df_transition.iterrows():
            
            state_0 = int(row[field_array_0])
            state_1 = int(row[field_array_1])
            states = [state_0, state_1]
            
            add_area = True
            area = float(row[field_area])
            
            if luc_copernicus_herbaceous_wetland not in states:
                continue
                
                
            # get new state 
            state = (
                luc_copernicus_herbaceous_wetland_new
                if state_0 == state_1
                else [x for x in states if x != luc_copernicus_herbaceous_wetland][0]
            )
                
            # if new state was previously the inbound class, we have to add the area to total for outbound
            dict_state_0_to_area_total[state] += (
                area 
                if ((state == state_1) | (state_0 == state_1)) 
                else 0.0
            )   
                
            vec_new_0[i] = state
            vec_new_1[i] = state
        

        # update data frames
        df_transition[field_array_0] = vec_new_0
        df_transition[field_array_1] = vec_new_1
        df_transition[field_area_total_0] = (
            df_transition[field_array_0]
            .replace(dict_state_0_to_area_total)
        )
        
        # re-aggregate
        df_transition = sf.simple_df_agg(
            df_transition,
            [field_array_0, field_array_1],
            {
                field_area_total_0: "first",
                field_area: "sum",
            }
        )
        
        return df_transition