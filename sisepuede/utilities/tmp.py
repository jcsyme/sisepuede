def build_variable_dataframe_by_sector(self,
        sectors_build: Union[List[str], str, None],
        df_trajgroup: Union[pd.DataFrame, None] = None,
        field_subsector: str = "subsector",
        field_variable_field: str = "variable_field",
        field_variable_trajectory_group: str = "variable_trajectory_group",
        include_model_variable: bool = False,
        include_simplex_group_as_trajgroup: bool = False,
        include_time_periods: bool = True,
        vartype: str = "input",
    ) -> pd.DataFrame:
        """
        Build a data frame of all variables long by subsector and variable.
            Optional includion of time_periods.

        Function Arguments
        ------------------
        - sectors_build: sectors to include subsectors for

        Keyword Arguments
        -----------------
        - df_trajgroup: optional dataframe mapping each field variable to 
            trajectory groups. 
            * Must contain field_subsector, field_variable_field, and 
                field_variable_trajectory_group as fields
            * Overrides include_simplex_group_as_trajgroup if specified and 
                conflicts occur
        - field_subsector: subsector field for output data frame
        - field_variable_field: variable field for output data frame
        - field_variable_trajectory_group: field giving the output variable
            trajectory group (only included if
            include_simplex_group_as_trajgroup == True)
        - include_simplex_group_as_trajgroup: include variable trajectory group 
            defined by Simplex Group in attribute tables?
        - include_time_periods: include time periods? If True, makes data frame
            long by time period
        - vartype: "input" or "output"
        """
        df_out = []
        sectors_build = self.get_sector_list_from_projection_input(sectors_build)

        # loop over sectors/subsectors to construct subsector and all variables
        for sector in sectors_build:
            subsectors = self.get_sector_subsectors(sector)

            for subsector in subsectors:
                modvars_cur = self.get_subsector_variables(
                    subsector,
                    var_type = vartype
                )

                vars_cur = sum([self.dict_model_variables_to_variable_fields.get(x) for x in modvars_cur], [])
                df_out += [(subsector, x) for x in vars_cur]

        # convert to data frame and return
        fields_sort = [field_subsector, field_variable_field]
        df_out = pd.DataFrame(
            df_out,
            columns = fields_sort
        )

        # include simplex group as a trajectory group?
        if include_simplex_group_as_trajgroup:
            col_new = list(df_out[field_variable_field].apply(self.get_simplex_group))
            df_out[field_variable_trajectory_group] = col_new
            df_out[field_variable_trajectory_group] = df_out[field_variable_trajectory_group].astype("float")
        
        # use an exogenous specification of variable trajectory groups?
        if isinstance(df_trajgroup, pd.DataFrame):
            
            fields_sort_with_tg = fields_sort + [field_variable_trajectory_group]

            if (
                set([field_variable_field, field_variable_trajectory_group])
                .issubset(set(df_trajgroup.columns))
            ):
                df_trajgroup.dropna(
                    subset = [field_variable_field, field_variable_trajectory_group],
                    how = "any",
                    inplace = True
                )

                # if the trajgroup is already defined, split into 
                # - variables that are assigned by not in df_trajgroup
                # - variables that are assigned and in df_trajgroup
                if (field_variable_trajectory_group in df_out.columns):
                    
                    vars_to_assign = sorted(list(df_trajgroup[field_variable_field].unique()))
                    tgs_to_assign = sorted(list(df_trajgroup[field_variable_trajectory_group].unique()))
                    # split into values to keep (but re-index) and those to overwrite
                    df_out_keep = df_out[
                        ~df_out[field_variable_field]
                        .isin(vars_to_assign)
                    ]
                    df_out_overwrite = (
                        df_out[
                            df_out[field_variable_field]
                            .isin(vars_to_assign)
                        ]
                        .drop(
                            [field_variable_trajectory_group],
                            axis = 1
                        )
                    )

                    # get values to reindex and apply
                    dict_to_reindex = sorted(list(set(df_out_keep[field_variable_trajectory_group])))
                    dict_to_reindex = dict(
                        (x, i + max(tgs_to_assign) + 1)
                        for i, x in enumerate(dict_to_reindex)
                        if not np.isnan(x)
                    )
                    
                    (
                        df_out_keep[field_variable_trajectory_group]
                        .replace(
                            dict_to_reindex,
                            inplace = True,
                        )
                    )

                    # merge in required fields
                    df_out_overwrite = pd.merge(
                        df_out_overwrite,
                        df_trajgroup[
                            [x for x in fields_sort_with_tg if x in df_trajgroup.columns]
                        ]
                        .dropna(),
                        how = "left",
                    )

                    df_out = pd.concat(
                        [df_out_keep, df_out_overwrite],
                        axis = 0,
                    )
                
                else:
                    # merge in required fields
                    df_out = pd.merge(
                        df_out,
                        (
                            df_trajgroup[
                                [x for x in fields_sort_with_tg if x in df_trajgroup.columns]
                            ]
                            .dropna()
                        ),
                        how = "left",
                    )


        if include_time_periods:
            attr_time_period = self.get_dimensional_attribute_table(
                self.dim_time_period
            )

            df_out = sf.explode_merge(
                df_out,
                attr_time_period.table[[attr_time_period.key]]
            )

            fields_sort += [attr_time_period.key]

        df_out = (
            df_out
            .sort_values(by = fields_sort)
            .reset_index(drop = True)
        )

        return df_out