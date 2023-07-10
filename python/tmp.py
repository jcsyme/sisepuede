def get_average_forest_factors_by_iso(
    df_climate_by_iso: pd.DataFrame,
    df_carbon_factors_forest: pd.DataFrame,
    attr_kcc: AttributeTable,
    regions: sc.Regions,
    field_continent: str = field_continent,
    field_count: str = "count",
    field_ecological_zone: str = "ecological_zone1",
    field_forest_cat: str = "ipcc_forest",
    field_type_forest: str = field_type_forest,
) -> pd.DataFrame:
    """
    Generate average storage and sequestration rate factors by iso code
    
    Function Arguments
    ------------------
    - df_climate_by_iso: data frame containing KCC climate counts by ISO
        code
    - df_carbon_factors_forest: data frame storing IPCC GHG default carbon 
        biomass factors
    - attr_kcc: attribute table characterizing Kopen Climate 
        Classification 
        
    Keyword Arguments
    -----------------
    - field_continent: field storing continent
    - field_count: field giving # of cells by country assocaited with 
        each KCC category
    - field_ecological_zone: field in df_carbon_factors_forest containing IPCC
        forests
    - field_forest_cat: field in df_climate_by_iso containing the IPCC 
        forest category used to estimate factors
    - field_type_forest: field in df_carbon_factors_forest storing forest type
    """
    
    dict_un_region_to_continent = {"Americas": "North and South America"}
    continents_global = [
        "Asia\nEurope\nNorth and South America",
        "Asia Europe North\nAmerica",
        "Asia\nEurope\nNorth\nAmerica"
    ]
    
    fields_keep = [
        regions.field_iso,
        attr_kcc.key,
        field_count,
        field_forest_cat
    ]
    dfg_climate = (
        df_climate_by_iso[df_climate_by_iso[regions.field_iso].isin(["BRA"])][fields_keep]
        .groupby([regions.field_iso])
    )
    df_out = []

    # clean carbon factors df
    df_cf = (
        df_carbon_factors_forest
        .rename(
            columns = {
                field_ecological_zone: field_forest_cat
            }
        )
    )
    fields_ind_cf = [field_forest_cat, field_continent, field_type_forest]
    fields_dat_cf = [x for x in df_cf.columns if x not in fields_ind_cf]

    # split into continent-specific and global
    df_cf_by_continent = (
        df_cf[
            ~df_cf[field_continent].isin(continents_global)
        ]
        .reset_index(drop = True)
    )
    df_cf_global = (
        df_cf[
            df_cf[field_continent].isin(continents_global)
        ]
        .reset_index(drop = True)
    )
    
    # get forest vals by split
    all_forests_by_continent = set(df_cf_by_continent[field_forest_cat])
    all_forests_global = set(df_cf_global[field_forest_cat])
    
  
    for iso, df in dfg_climate:
        
        # get un region
        region_un = regions.get_un_region(iso)
        region_ipcc_forests = dict_un_region_to_continent.get(region_un, region_un)
        
        all_forests_cur = set(df[field_forest_cat])
        any_by_continent = len(all_forests_cur & all_forests_by_continent) > 0
        any_global = len(all_forests_cur & all_forests_global) > 0
        
        # total number of cells
        total_count = df[field_count].sum()
        
        # initialize splits
        df_by_continent = None
        df_global = None
        
        # deal with splits
        if any_by_continent:
            df[field_continent] = region_ipcc_forests

            df_by_continent = (
                pd.merge(
                    df,
                    df_cf_by_continent,
                )
                .drop(
                    [
                        field_continent, 
                        #field_kcc, 
                        #field_forest_cat,
                        regions.field_iso
                    ], 
                    axis = 1
                )
            )
            
            df.drop([field_continent], axis = 1, inplace = True)
            
            
        if any_global:
            df_global = (
                pd.merge(
                    df,
                    df_cf_global.drop([field_continent], axis = 1),
                )
                .drop(
                    [
                        #field_kcc, 
                        #field_forest_cat,
                        regions.field_iso
                    ], 
                    axis = 1
                )
            )

        
        # check that at least one was successfully merges
        if (df_global is None) & (df_by_continent is None):
            continue
        
        global df_cur
        global df_means
        global df_append
        
        df_means = pd.concat([df_by_continent, df_global], axis = 0)
        
        df_append = None
        for field in fields_dat_cf:
            
            df_cur = (
                df_means[
                    [
                        field_count,
                        field_type_forest,
                        field
                    ]
                ]
                .fillna(0)
            )
            
            df_cur[field] = np.array(df_cur[field])*np.array(df_cur[field_count])/total_count
            df_cur = sf.simple_df_agg(
                df_cur.drop([field_count], axis = 1),
                [field_type_forest],
                {
                    field: "sum"
                }
            )
            
            df_append = (
                df_cur
                if df_append is None
                else pd.merge(df_append, df_cur)
            )
            
            df_append[regions.field_iso] = iso
            
        df_out.append(df_append)
        
    df_out = pd.concat(df_out, axis = 0)
        
    return df_out
        


df_carbon_factors_forest_mod = get_average_forest_factors_by_iso(
    df_climate,
    df_carbon_factors_forest.drop([field_domain], axis = 1),
    attr_kcc,
    regions
);


