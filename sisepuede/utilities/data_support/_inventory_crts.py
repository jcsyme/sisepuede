"""Use this file to construct an inventory for historical based on Excel files.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import re
import sisepuede.utilities._toolbox as sf
import sisepuede.visualization.plots as svp
from typing import *




######################
#    SOME GLOBALS    #
######################

# categories
_CAT_ED_AGG = "Deforestation and Other Land Use Conversion"
_CAT_ED_DEFORESTATION = "Deforestation"
_CAT_ED_OTHER = "Other Land Use Conversion"

# fields
_FIELD_CW_CATEGORY_AGGREGATION = "aggregation_category"
_FIELD_CW_CATEGORY_SECONDARY = "secondary_category"
_FIELD_CW_EST_FROM_SSP = "est_from_sisepuede"
_FIELD_CW_GAS = "gas"
_FIELD_CW_INCLUDED_IN_INVENTORY = "included_in_inv"
_FIELD_CW_SISEPUEDE_FIELDS = "sisepuede_fields"
_FIELD_CW_SYNTHETIC_CATEGORY = "synthetic_categories"
_FIELD_CW_USE_SYNTHETIC = "use_synthetic"
_FIELD_ED_CATEGORY_AGGREGATION = "aggregation_category"
_FIELD_ED_GAS = "gas"
_FIELD_ED_VALUE = "value_kt"
_FIELD_ED_YEAR = "year"
_FIELD_INVTAB_CATEGORIES = "Categories"
_FIELD_INVTAB_GAS_CH4 = "CH4"
_FIELD_INVTAB_GAS_CO2 = "CO2"
_FIELD_INVTAB_GAS_HFCS = "HFCs"
_FIELD_INVTAB_GAS_N2O = "N2O"
_FIELD_INVTAB_GAS_NF3 = "NF3"
_FIELD_INVTAB_GAS_PFCS = "PFCs"
_FIELD_INVTAB_GAS_SF6 = "SF6"

# gas groups
_GASSES_NOT_CO2E = ["ch4", "n2o"]

# field prefixes
_PREFIX_FIELD_IPCC_CATEGORIES_LEVEL = "ipcc_categories_level_"
_PREFIX_FIELD_EMISSIONS_OUT = "emission_co2e_"

# regular expression
_REGEX_PATTERN_INVENTORY_TABLES = re.compile("GHG Emissions_(.*\d)_Complete File.xlsx")

# unit info
_UNITS_MASS_INV = "kt"



#########################
#    BUILD FUNCTIONS    #
#########################

def add_inv_table_emissions_to_cw(
    dict_dfs: Dict[int, Dict[str, pd.DataFrame]],
    df_cw: pd.DataFrame,
    fields_cat_ordered: List[str],
    dict_table_ind_to_sheet_name: Dict[Any, str],
    dict_cw_gas_to_itab_gas_field: Dict[str, str],
    model_attributes: 'ModelAttributes',
    emissions_flag_from_ssp: int = -99999,
    verbose: bool = True,
) -> pd.DataFrame:
    """

    Function Arguments
    ------------------
    dict_dfs : Dict[int, Dict[str, pd.DataFrame]]
        Dicitonary of inventory table dictionaries (Excel files) by year
    df_cw : pd.DataFrame
        Crosswalk DataFrame mapping inventory categories to SISEPUEDE fields
    fields_cat_ordered : List[str]
        Ordered (by priority) IPCC categories
        
    Keyword Arguments
    -----------------
    emissions_flag_from_ssp : int
        Flag to use in inventory to signal that emissions should be taken from
        SSP output
    verbose : bool
        Print warnings and status?
    """

    # new columns to add
    dict_new_cols = {}
    
    for k, v in dict_dfs.items():

        col = []
        
        for i, row in df_cw.iterrows():
        
            #if i > 159: break
            #if i < 159: continue
            
            if (row[_FIELD_CW_INCLUDED_IN_INVENTORY] == 0):
                appendage = (
                    emissions_flag_from_ssp
                    if row[_FIELD_CW_USE_SYNTHETIC]
                    else None
                )
        
                col.append(appendage)
                continue
                
        
            emissions_co2e = None
            
            try:
        
                emissions_co2e = retrieve_emissions_element_from_table(
                    row,
                    v,
                    fields_cat_ordered,
                    dict_table_ind_to_sheet_name,
                    dict_cw_gas_to_itab_gas_field,
                    model_attributes,
                    #return_dict = True, 
                )
            except Exception as e:
                print(f"Skipping row {i}: {e}")
                
            col.append(emissions_co2e, )

        # add to dictionary
        dict_new_cols.update({k: col, })

        if verbose:
            print(f"Completed year {k}")

    
    ##  BUILD OUTPUT DATAFRAME

    fields_ord = sorted(list(dict_new_cols.keys()))
    
    # concatenate cw to 
    df_out = pd.concat(
        [
            df_cw,
            pd.DataFrame(dict_new_cols)
            .get(fields_ord, )
        ],
        axis = 1,
    )

    return df_out



def aggregate_inventory_and_fill_from_ssp(
    df_cw_with_trajectories: pd.DataFrame,
    df_ssp: pd.DataFrame,
    df_emissions_deforestation: pd.DataFrame,
    model_attributes: 'ModelAttributes',
    time_periods: 'TimePeriods',
    cats_lndu_conversion: List[str] = [_CAT_ED_DEFORESTATION, _CAT_ED_OTHER],
    cat_forest_aggregate: str = "Forest Land Remaining Forest Land",
    cat_forest_sequestration: str = "Forest Land - Sequestration",
    cat_forest_removals_net: str = "Forest Land - Removals",
    cats_drop: Union[List[str], None] = None,
    delim: str = "|",
    # merge_forest_land_classes: bool = False, 
) -> pd.DataFrame:
    """
    Function Arguments
    ------------------


    Keyword Arguments
    -----------------
    cats_lndu_conversion : List[str]
        Categories in aggregation_categories giving conversion emissions
    cat_forest_aggregate : str
        NEW category (NOT in aggregation_categories) that combines sequestration 
        estimates and annual removals
    cat_forest_sequestration : str
        Category in aggregation_categories giving annual est sequestration
    cat_forest_removals_net : str
        Category in aggregation_categories storing net removals from forests
    cats_drop : str
        Optional specification of aggregation_categories to remove from the final
        historical
    delim : str
        Delimeter for SISEPUEDE fields
    merge_forest_land_classes : bool
        Set to True to combine cat_forest_sequestration and cat_forest_removals_net;
        useful since the removals category should account for this. 
    """

    ##  INITIALIZATION

    # get years
    fields_years = [x for x in df_cw_with_trajectories.columns if isinstance(x, int)]
    field_year = time_periods.field_year
    df_years = pd.DataFrame({field_year: fields_years, })
    
    # filter rows that are dealt with elsewhere 
    inds_ed_catagg = df_cw_with_trajectories[_FIELD_CW_CATEGORY_SECONDARY].isin(
        df_emissions_deforestation[_FIELD_ED_CATEGORY_AGGREGATION]
        .to_numpy()
    )
    df_filt = df_cw_with_trajectories[
        ~df_cw_with_trajectories[_FIELD_CW_SISEPUEDE_FIELDS].isna()
        & ~inds_ed_catagg
    ]

    # get fields associated with the emissions forom deforestation
    dfg = df_cw_with_trajectories[inds_ed_catagg].groupby([_FIELD_CW_CATEGORY_SECONDARY])
    dict_field_groups = {}
    for cat, df in dfg:
        fields = delim.join(
            df[_FIELD_CW_SISEPUEDE_FIELDS].to_numpy(),
        )

        dict_field_groups.update({cat[0]: fields, })
        
    
    # get some indices
    inds_accounted = df_filt[_FIELD_CW_INCLUDED_IN_INVENTORY].isin([1])
    inds_use_synthetic = df_filt[_FIELD_CW_USE_SYNTHETIC].isin([1])
    
    # split into accounted and unaccounted
    df_accounted = df_filt[inds_accounted]
    df_unaccounted = df_filt[~inds_accounted]


    
    ##  FOR ACCOUNTED, AGGREGATE

    fields_group = [
        _FIELD_CW_CATEGORY_AGGREGATION,
        #_FIELD_CW_CATEGORY_SECONDARY,
        _FIELD_CW_GAS,
    ]
    dfg = df_accounted.groupby(fields_group, )


    df_accounted_new = []
    for grp, df in dfg:
        fields = delim.join(
            df[_FIELD_CW_SISEPUEDE_FIELDS].to_numpy(),
        )
        tot = list(df[fields_years].sum(axis = 0, ))
        
        df_accounted_new.append(list(grp) + [fields] + tot)


    df_accounted_new = pd.DataFrame(
        df_accounted_new,
        columns = fields_group + [_FIELD_CW_SISEPUEDE_FIELDS] + fields_years,
    )


    ##  FORMAT THE EXOGENOUS DEFORESTATION DATA

    dfg = df_emissions_deforestation.groupby([_FIELD_ED_CATEGORY_AGGREGATION])
    df_ed = []

    for cat, df in dfg:
        df = (
            pd.merge(
                df_years,
                df,
                how = "left",
            )
            .interpolate()
            .bfill()
            .ffill()
        )

        df_ed.append(df)

    df_ed = sf._concat_df(df_ed, )

    # pivot and add sisepuede fields
    df_ed = sf.pivot_df_clean(
        df_ed,
        fields_column = [time_periods.field_year],
        fields_value = [_FIELD_ED_VALUE]
    )

    df_ed[_FIELD_CW_SISEPUEDE_FIELDS] = (
        df_ed[_FIELD_ED_CATEGORY_AGGREGATION]
        .apply(
            dict_field_groups.get
        )
    )


    ##  GET OTHER VALUES FROM SSP

    df_ssp_out = time_periods.tps_to_years(df_ssp,)
    dfg = df_unaccounted.groupby(fields_group)
    df_unaccounted_wide = []

    scalar = 1/model_attributes.get_mass_equivalent(_UNITS_MASS_INV, )
    
    for grp, df in dfg:
        fields_list = df[_FIELD_CW_SISEPUEDE_FIELDS].to_numpy()
        fields = delim.join(fields_list)
        fields_ext = sorted(fields.split(delim))
        

        # get the fields from SSP and sum them
        df_ext = df_ssp_out[[field_year] + fields_ext]
        df_ext[_FIELD_ED_VALUE] = df_ssp_out[fields_ext].sum(axis = 1, )
        df_ext = (
            pd.merge(
                df_years,
                df_ext,
                how = "left"
            )
            .interpolate()
            .ffill()
            .bfill()
            .drop(columns = fields_ext, )
            .set_index([time_periods.field_year])
            .transpose()
            .reset_index(drop = True, )
        )

        # add the ids and clean up
        df_ext.columns.name = None
        df_ext[_FIELD_CW_CATEGORY_AGGREGATION] = grp[0]
        df_ext[_FIELD_CW_GAS] = grp[1]
        df_ext[_FIELD_CW_SISEPUEDE_FIELDS] = fields

        # convert to target units
        df_ext[fields_years] *= scalar
        
        df_unaccounted_wide.append(df_ext)

    # 
    df_unaccounted_wide = sf._concat_df(df_unaccounted_wide, )


    ##  GROUP TOGETHER, THEN SPLIT THE REMOVALS UP

    fields_ext = [
        _FIELD_CW_CATEGORY_AGGREGATION,
        _FIELD_CW_GAS,
        _FIELD_CW_SISEPUEDE_FIELDS,
        _FIELD_CW_EST_FROM_SSP
    ]
    
    fields_ext += fields_years

    # add some info
    df_accounted_new[_FIELD_CW_EST_FROM_SSP] = 0
    df_ed[_FIELD_CW_EST_FROM_SSP] = 0
    df_unaccounted_wide[_FIELD_CW_EST_FROM_SSP] = 1

    # concatenate
    df_out = sf._concat_df(
        [
            df_accounted_new[fields_ext],
            df_unaccounted_wide[fields_ext],
            df_ed[fields_ext],
        ]
    )


    ##  ADJUST FOREST NUMBERS
    
    cats_adjust = cats_lndu_conversion + [cat_forest_sequestration, cat_forest_removals_net]

    dict_cats_adj_to_index = {}
    dict_cats_adj_to_vecs = {}
    for cat in cats_adjust:
        w = np.where(df_out[_FIELD_CW_CATEGORY_AGGREGATION].to_numpy() == cat)[0]
        if len(w) != 1:
            raise RuntimeError(f"Multiple instances of forest adjustment category {cat} found.")

        dict_cats_adj_to_index.update({cat: w, })
        dict_cats_adj_to_vecs.update({cat: df_out[fields_years].iloc[w].to_numpy()})
    
    # adjust removals so that total adds up
    vec_new_removals = dict_cats_adj_to_vecs.get(cat_forest_removals_net).copy()
    #vec_new_removals -= dict_cats_adj_to_vecs.get(cat_forest_sequestration)
    vec_new_removals -= np.array(
        [
            dict_cats_adj_to_vecs.get(x) for x in cats_lndu_conversion
        ]
    ).sum(axis = 0, )

    ind = dict_cats_adj_to_index.get(cat_forest_removals_net)
    df_out.loc[ind, fields_years] = vec_new_removals[0]

    # filter?
    if sf.islistlike(cats_drop, ):
        df_out = (
            df_out[
                ~df_out[_FIELD_CW_CATEGORY_AGGREGATION].isin(cats_drop)
            ]
            .reset_index(drop = True, )
        )

    ##  RETURN IF NOT COMBINING FOREST LAND CLASSES; OTHERWISE, KEEP GOING
    
    # if not merge_forest_land_classes:
    #     return df_out
    
    return df_out



def allocate_charcoal_to_inen_and_scoe(
    df_inv_trajectories: pd.DataFrame,
    field_emission: str,
    time_periods: 'TimePeriods',
    cat_buildings: str = "Buildings",
    cat_fuel_production: str = "Fuel Production",
    cat_industry: str = "Industrial Combustion",
) -> pd.DataFrame:
    """Allocate emissions from charcoal production to biomass CH4 (inen and scoe)
    """

    print("NOTE: reallocation charcoal emissions to SCOE and INEN...")
    
    ##  INITIALIZATION
    
    dict_dfs = {}

    # data to allocate
    df_allocate = df_inv_trajectories[
        df_inv_trajectories[_FIELD_CW_CATEGORY_AGGREGATION]
        .isin([cat_fuel_production])
    ]

    # some indices to split off
    inds_keep = (
        ~df_inv_trajectories[_FIELD_CW_CATEGORY_AGGREGATION]
        .isin([cat_buildings, cat_industry, cat_fuel_production])
    )

    
    ##  ITERATE OVER EACH GAS TO ALLOCATE

    cats_allocate = [cat_buildings, cat_industry]
    df_append = [df_inv_trajectories[inds_keep]]
    
    for gas in df_allocate[_FIELD_CW_GAS].unique():

        df_cur = df_allocate[
            df_allocate[_FIELD_CW_GAS].isin([gas])
        ].copy()

        # get allocation fractions
        for cat in cats_allocate:
            
            df_subinv = (
                df_inv_trajectories[
                    df_inv_trajectories[_FIELD_CW_CATEGORY_AGGREGATION].isin([cat])
                    & df_inv_trajectories[_FIELD_CW_GAS].isin([gas])
                ]
                .rename(columns = {field_emission: cat})
                .drop(columns = [_FIELD_CW_CATEGORY_AGGREGATION])
            )

            df_cur = (
                df_subinv
                if df_cur is None
                else pd.merge(
                    df_cur,
                    df_subinv,
                    how = "left",
                )
            )
            
            #dict_dfs.update({cat: })

        
        arr_allocate = sf.check_row_sums(
            df_cur[cats_allocate],
            thresh_correction = None,
        )

        arr_allocate = sf.do_array_mult(
            arr_allocate, 
            df_cur[field_emission].to_numpy(),
        )

        # update allocation
        df_cur[cats_allocate] += arr_allocate
        df_cur[field_emission] = 0.0

        
        # unwrap
        df_cur = (
            df_cur
            .drop(columns = _FIELD_CW_CATEGORY_AGGREGATION)
            .rename(columns = {field_emission: cat_fuel_production})
            .melt(
                id_vars = [
                    _FIELD_CW_GAS,
                    time_periods.field_year,
                ],
                value_name = field_emission,
                value_vars = cats_allocate + [cat_fuel_production],
                var_name = _FIELD_CW_CATEGORY_AGGREGATION,
            )
        )

        #
        df_append.append(df_cur)

    global dfa
    dfa = df_append
    
    # concatenate and sort
    df_out = (
        pd.concat(df_append, )
        .sort_values(
            by = [
                _FIELD_CW_CATEGORY_AGGREGATION,
                _FIELD_CW_GAS,
                time_periods.field_year
            ]
        )
        .reset_index(drop = True, )
    )
        
    return df_out



def field_category_level(
    level: int,
) -> str:
    """Build the IPCC categories level string
    """
    out = f"{_PREFIX_FIELD_IPCC_CATEGORIES_LEVEL}{level}"
    return out



def get_all_sheet_name_table_dict(
    dict_dfs: Dict[str, Dict[str, pd.DataFrame]],
) -> List[str]:
    """Get all sheet names associated with each of the inventory elements.
        Returns a dictionary that maps Table number to name.
    """
    
    all_keys = set({})
    for k, v in dict_dfs.items():
        keys_cur = set(v.keys())
        all_keys |= keys_cur

    # build output dictionary
    dict_out = {}
    
    for k in all_keys:
        ind = k.split()[1].strip()

        if ind.isnumeric():
            ind = int(ind)

        dict_out.update({ind: k, })

    return dict_out



def get_category_fields_ordered(
    df_cw: pd.DataFrame,
) -> List[str]:
    fields = [x for x in df_cw.columns if x.startswith(_PREFIX_FIELD_IPCC_CATEGORIES_LEVEL)]

    # use 'key' kwarg: see https://www.geeksforgeeks.org/python/python-sort-given-list-of-strings-by-part-the-numeric-part-of-string/
    fields = sorted(
        fields, 
        key = lambda x: int(re.search(r'\d+', x).group())
    )
    return fields



def get_emissions_deforestation(
    path: pathlib.Path,
) -> pd.DataFrame:
    """Retrieve the emissions from deforestation file and disaggregate 
        deforestation and other land use conversion proportionally.
    """

    # read the dataframe and set the type
    df_emissions_deforestation = pd.read_csv(path, )
    df_emissions_deforestation[_FIELD_ED_VALUE] = df_emissions_deforestation[_FIELD_ED_VALUE].astype(float)

    
    ##  AGGREGATE AND PIVOT

    fields_group = [
        _FIELD_ED_YEAR,
        _FIELD_ED_CATEGORY_AGGREGATION,
        _FIELD_ED_GAS
    ]
    
    fields_get = fields_group + [_FIELD_ED_VALUE]


    # do the aggregation
    df_emissions_deforestation_agg = (
        df_emissions_deforestation
        .get(fields_get)
        .groupby(fields_group)
        .sum()
        .reset_index()
    )
    
    # pivot and fill down deforestation 
    df_emissions_deforestation_agg = sf.pivot_df_clean(
        df_emissions_deforestation_agg,
        fields_column = [_FIELD_ED_CATEGORY_AGGREGATION],
        fields_value = [_FIELD_ED_VALUE],
    )
    
    
    # mark which years are actually broken out
    field_tb = "true_break"
    global dfag
    dfag = df_emissions_deforestation_agg.copy()

    # get inds where not NA
    vec_break = ~df_emissions_deforestation_agg[_CAT_ED_DEFORESTATION].isna()
    if _CAT_ED_OTHER in df_emissions_deforestation_agg.columns:
        vec_break &= ~df_emissions_deforestation_agg[_CAT_ED_OTHER].isna()
    
    vec_break = vec_break.astype(int)
    df_emissions_deforestation_agg[field_tb] = vec_break
    
    # fill fields and use as fracs
    vec_total = 0
    _CATS_ED = [_CAT_ED_DEFORESTATION, _CAT_ED_OTHER]
    _CATS_ED = [x for x in _CATS_ED if x in df_emissions_deforestation_agg.columns]

    for cat in _CATS_ED:
        df_emissions_deforestation_agg[cat] = (
            df_emissions_deforestation_agg[cat]
            .interpolate()
            .bfill()
        )
    
        vec_total += df_emissions_deforestation_agg[cat].to_numpy()
    
    
    
    ##  ALLOCATE USING PROPORTIONS FROM KNOWN DATA
    
    if _CAT_ED_AGG in df_emissions_deforestation_agg.columns:
        for cat in _CATS_ED:
            df_emissions_deforestation_agg[cat] = (
                df_emissions_deforestation_agg[cat]
                .to_numpy()
                *df_emissions_deforestation_agg[_CAT_ED_AGG].to_numpy()
                /vec_total
            )
    
    # clean and drop
    fields_drop = [field_tb, _CAT_ED_AGG]
    fields_drop = [x for x in fields_drop if x in df_emissions_deforestation_agg.columns]

    df_emissions_deforestation_agg = (
        df_emissions_deforestation_agg[
            df_emissions_deforestation_agg[field_tb].isin([0])
        ]
        .drop(columns = fields_drop)
        .melt(
            id_vars = [_FIELD_ED_YEAR, _FIELD_ED_GAS],
            value_name = _FIELD_ED_VALUE,
            value_vars = _CATS_ED,
            var_name = _FIELD_ED_CATEGORY_AGGREGATION,
        )
    )


    # recombine with known data
    df_out = (
        df_emissions_deforestation[
            ~df_emissions_deforestation[_FIELD_ED_YEAR].isin(
                df_emissions_deforestation_agg[_FIELD_ED_YEAR].to_numpy()
            )
        ]
        .get(df_emissions_deforestation_agg.columns, )
        .groupby(fields_group)
        .sum()
        .reset_index()
    )

    
    df_out = (
        pd.concat(
            [
                df_out,
                df_emissions_deforestation_agg
            ],
            axis = 0,
        )
        .sort_values(
            by = [
                _FIELD_ED_YEAR,
                _FIELD_ED_CATEGORY_AGGREGATION
            ]
        )
        .reset_index(drop = True, )
    )

    return df_out



def get_files(
    path: pathlib.Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Get inventory tables as a dictionary mapping years to the inventory 
        dictionaries (sheets to DataFrame)
    """

    dict_out = {}

    for path_file in path.iterdir():
        year = get_year_from_file_path(path_file, )
        if year is None: continue

        dict_dfs = pd.read_excel(
            path_file,
            header = 2,
            sheet_name = None,
        )
        
        dict_out.update({year: dict_dfs, })

    return dict_out



def get_inventory_gas_field_to_gas_dict(
) -> Dict[str, str]:
    """Map the inventory table gas fields to gasses in the crosswalk. 
        Returns two dictionaries:

            (
                dict_gas_field_to_gas,
                dict_gas_to_gas_field
            )
    """

    fields = [
        _FIELD_INVTAB_GAS_CH4,
        _FIELD_INVTAB_GAS_CO2,
        _FIELD_INVTAB_GAS_HFCS,
        _FIELD_INVTAB_GAS_N2O,
        _FIELD_INVTAB_GAS_NF3,
        _FIELD_INVTAB_GAS_PFCS,
        _FIELD_INVTAB_GAS_SF6
    ]

    dict_out = dict((x, x.lower()) for x in fields)
    dict_rev = sf.reverse_dict(dict_out, )
    
    out = (
        dict_out,
        dict_rev,
    )

    return out



def get_inventory_sheet(
    cat_0: str,
    dict_table_ind_to_sheet_name: Dict[Any, str],
) -> str:
    """Based on IPCC category, get the sheet name in inventory table
    """
    sheet_index = int(cat_0.split("-")[0].strip())
    sheet_name = dict_table_ind_to_sheet_name.get(sheet_index, )
    
    return sheet_name



def get_matchstrings(
    row: pd.Series,
    fields_category_ordered: str,
    delim: str = "|",
) -> str:
    """Retrieve strings to match inventory row on for a crosswalk row
    """

    # get 
    matchstrs = None
    for i, field in enumerate(fields_category_ordered):
        break_q = not isinstance(row[field], str)
        break_q |= (row[field] == "") if not break_q else break_q
        if break_q: break
    
    # get the matchstrings and cut up
    field = fields_category_ordered[i - 1]
    matchstrs = str(row[field]).split(delim)
    matchstrs = [x.strip() for x in matchstrs]

    return matchstrs



def get_year_from_file_path(
    path: pathlib.Path,
) -> Union[int, None]:
    """Checks a file name to see if it matches the regular expression for 
        inventory tables; if so, gets year. Otherwise, returns None.
    """
    file_name = path.parts[-1]
    match = _REGEX_PATTERN_INVENTORY_TABLES.match(file_name, )
    if match is None:
        return None
    
    # otherwise, get the year
    year = int(match.groups()[0])

    return year



def plot_and_export_inventory(
    df_trajectories: pd.DataFrame,
    time_periods: 'TimePeriods',
    path_out_figure: Union[None, pathlib.Path] = None,
) -> 'plt.Plot':
    """Plot the output historical trajectories
    """
    
    df_plot = df_trajectories.copy()
    
    # merge labels`
    field_label = "emissions_category"
    fields_reduce = [_FIELD_CW_CATEGORY_AGGREGATION, _FIELD_CW_GAS]
    col_label = df_plot[fields_reduce].apply(" ".join, axis = 1, ).to_numpy()
    
    df_plot[field_label] = col_label
    df_plot = df_plot.drop(columns = fields_reduce, )
    field_emissions = [x for x in df_plot.columns if x not in [time_periods.field_year, field_label]][0]

    df_plot = sf.pivot_df_clean(
        df_plot,
        [field_label],
        [field_emissions]
    )
    
    
    # plot and write
    fig, ax = plt.subplots(figsize = (13, 7))
    plot_out = svp.spu.plot_stack(
        df_plot,
        [x for x in df_plot.columns if (x not in [time_periods.field_year])],
        field_x = time_periods.field_year,
        figtuple = (fig, ax, ), 
    )

    if isinstance(path_out_figure, pathlib.Path):
        fig.savefig(path_out_figure, bbox_inches = "tight", dpi = 300, )

    out = (fig, ax, )
        
    return out
    


def retrieve_emissions_element_from_table(
    row: pd.Series,
    dict_inventory: Dict[str, pd.DataFrame],
    fields_category_ordered: List[str],
    dict_table_ind_to_sheet_name: Dict[Any, str],
    dict_cw_gas_to_itab_gas_field: Dict[str, str],
    model_attributes: 'ModelAttributes',
    correct_co2e: bool = True, 
    delim: str = "|",
    missing_val: Any = None,
    return_dict: bool = False,
) -> Tuple[str, str]:
    """Retrieve an element from an inventory table based on a specification in the 
        crosswalk row.


    Function Arguments
    ------------------
    row : pd.Series
        Pandas series from crosswalk (row)
    dict_table_ind_to_sheet_name : Dict[Any, str]
        Dictionary mapping the table index (1, 2, 3, 4 or "A")
        from the crosswalk category to the associated sheet in
        dict_inventory
    dict_cw_gas_to_itab_gas_field : Dict[str, str]
        Dictionary mapping each gas in the crosswalk to the
        associated field in the inventory table

    Keyword Arguments
    -----------------
    return_dict : bool
        Return the dictionary mapping inventory elements to raw
        (unadjusted to CO2e where applicable) values?
    """
    
    # get location of data
    cat_0 = row[field_category_level(0)]
    sheet_name = get_inventory_sheet(
        cat_0,
        dict_table_ind_to_sheet_name,
    )

    gas = row[_FIELD_CW_GAS]

    # get matchstrings used to pull rows from the inventory sheet
    matchstrs = get_matchstrings(
        row,
        fields_category_ordered,
        delim = delim, 
    )

    # get the values from the inventory
    dict_values = retrieve_values_from_matchstrs(
        dict_inventory.get(sheet_name, ),
        matchstrs,
        gas,
        dict_cw_gas_to_itab_gas_field,
        missing_val = missing_val,
    )
    
    if return_dict:
        return dict_values

    # get total
    total_emission = 0.0
    total_emission += sum([x for x in dict_values.values() if sf.isnumber(x)])

    #
    if gas in _GASSES_NOT_CO2E:
        total_emission *= model_attributes.get_gwp(gas)

    return total_emission
    


def retrieve_values_from_matchstrs(
    df_inv: pd.DataFrame,
    matchstrs: List[str],
    gas: str,
    dict_cw_gas_to_itab_gas_field: Dict[str, str],
    missing_val: Any = 0.0,
) -> Dict[str, Union[float, None]]:
    """Get the value from an inventory table
    """
    

    # initialize dictionary out
    dict_out = {}
    col = df_inv[_FIELD_INVTAB_CATEGORIES].values

    
    # iterate over each categoryu
    for matchstr in matchstrs:

        # find row and col--check for issues with CO2 in AFOLU
        ind = np.where([((matchstr in x) if isinstance(x, str) else False) for x in col])[0]
        field = dict_cw_gas_to_itab_gas_field.get(gas)
        if (gas == "co2") & (field not in df_inv.columns):
            field = "Net CO2 emissions / removals"
        
        if len(ind) == 0: 
            dict_out.update({matchstr: missing_val})
            continue

        # case where there are multiple rows; if only one of the rows is assocaited with values, use that row
        if len(ind) > 1:
            nums = []
            for i in ind:
                val = try_convert_value(
                    df_inv[field].iloc[i], 
                    None,
                )
                nums.append(sf.isnumber(val), )

            # if there's not a unique row associated with a number, raise an error
            if sum(nums) != 1:
                raise RuntimeError(f"Multiple entries found for matchstr = {matchstr}: {ind}")

            # otherwise, assign the 
            w_nums = np.where(nums)[0]
            ind = [ind[w_nums]]
        

        # value adjustment
        val = try_convert_value(
            df_inv[field].iloc[ind[0]], 
            missing_val,
        )
        

        dict_out.update({matchstr: val})

    return dict_out



def split_aggregate_inv_into_cw_and_trajectories(
    df_inv_aggregate: pd.DataFrame,
    model_attributes: 'ModelAttributes',
    time_periods: 'TimePeriods',
    add_stars_to_est_groups: bool = True,
    dict_overwrite_category_to_value: Union[Dict[str, float], None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split the aggregate inventory table into a crosswalk and separate out
        trajectories.
    """

    # some initialization
    fields_years = [x for x in df_inv_aggregate if isinstance(x, int)]
    fields_id = [
        _FIELD_CW_CATEGORY_AGGREGATION,
        _FIELD_CW_GAS
    ]
    fields_drop_from_traj = [
        _FIELD_CW_SISEPUEDE_FIELDS,
        _FIELD_CW_EST_FROM_SSP
    ]
    fields_cw = fields_id + fields_drop_from_traj

    # star the estimated fields?
    df_out = df_inv_aggregate.copy()
    if add_stars_to_est_groups:
        col_new = df_out[_FIELD_CW_CATEGORY_AGGREGATION].to_numpy()
        for i, el in enumerate(col_new):
            est = bool(df_out[_FIELD_CW_EST_FROM_SSP].iloc[i])
            if not est: continue
            
            col_new[i] = f"{el}*"
        
        df_out[_FIELD_CW_CATEGORY_AGGREGATION] = col_new

    
    # now, get crosswalk and check shape 
    df_cw = df_inv_aggregate[fields_cw].copy()
    df_cw_check = df_inv_aggregate[fields_id]
    if df_cw_check.shape != df_cw_check.drop_duplicates().shape:
        raise RuntimeError(f"Multiple entries for fields {fields_id} found in df_inv_aggregate")


    # split out trajectories and melt, then convert to same units as SSP
    scale_units = model_attributes.get_mass_equivalent(_UNITS_MASS_INV, )
    field_out = model_attributes.configuration.get("emissions_mass").lower()
    field_out = f"{_PREFIX_FIELD_EMISSIONS_OUT}{field_out}"

    # reshape
    df_trajectories = (
        df_inv_aggregate
        .drop(columns = fields_drop_from_traj, )
        .melt(
            id_vars = fields_id,
            value_name = field_out,
            value_vars = fields_years,
            var_name = time_periods.field_year,
            
        )
        .sort_values(
            by = [
                _FIELD_CW_CATEGORY_AGGREGATION,
                _FIELD_CW_GAS,
                time_periods.field_year
            ]
        )
        .reset_index(drop = True, )
    )

    # finally, scale emissions to SSP units
    df_trajectories[field_out] *= scale_units

    # overwrite directly from dictionary?
    if isinstance(dict_overwrite_category_to_value, dict):
        for k, v in dict_overwrite_category_to_value.items():

            # get the rows to overwrite
            ind = df_trajectories[_FIELD_CW_CATEGORY_AGGREGATION].isin([k])
            w = np.where(ind)[0]
            
            if len(w) == 0: continue
            
            df_trajectories.loc[w, field_out] = v

    
    # return the crosswalk and the trajectories
    out = (df_cw, df_trajectories, field_out, )
    
    return out



def try_convert_value(
    val: Any,
    missing_val: Any,
) -> float:
    """Try converting an inventory entry to a float
    """

    # value adjustment
    if isinstance(val, str):
        val = val.replace(",", "")

    try:
        val = float(val)

    except:
        val = missing_val

    return val





def main(
    df_ssp: pd.DataFrame,
    path_in_cw_inv_excels: pathlib.Path,
    path_in_emissions_deforestation: pathlib.Path,
    path_in_inv_excels: pathlib.Path,
    path_out_cw_new: pathlib.Path,
    path_out_trajectories: pathlib.Path,
    model_attributes: 'ModelAttributes',
    time_periods: 'TimePeriods',
    dict_overwrite_category_to_value: Union[Dict[str, float], None] = None,
    reallocate_charcoal_production: bool = True,
    path_out_figure: Union[pathlib.Path, None] = None,
    **kwargs,
) -> Dict[str, Union[Tuple[pd.DataFrame], 'plt.Plot']]:
    """Build the inventory table, split into a crosswalk and 
        trajectories, write to a CSV, and return a dictionary
        of the form

        {
            "data": (
                df_cw,
                df_trajectories,
            ),

            "plots": (
                fig,
                ax,
            )
        }

        
    Function Arguments
    ------------------
    df_ssp : pd.DataFrame
        DataFrame of baseline SISEPUEDE output used to fill gaps
    path_in_cw_inv_excels : pathlib.Path
        Input path where to CSV file where crosswalk for Excel 
        inventories and SISEPUEDE emission fields is stored
    path_in_emissions_deforestation : pathlib.Path
        Input path where the exogenous deforestation emissions (taken
        from NC3 and BUR2) are located
    path_in_inv_excels : pathlib.Path
        Input path to directory  where Excel inventories are located
    path_out_cw_new : pathlib.Path
        Output path for new crosswalk between aggregation categories
        and SISEPUEDE fields is stored
    path_out_trajectories: pathlib.Path
        Output path for new trajectories file is located
        and SISEPUEDE fields is stored
    model_attributes : ModelAttributes
        ModelAttributes used for GHG conversion etc.
    time_periods : TimePeriods
        TimePeriods object used to convert SSP output years based  

    Keyword Arguments
    -----------------
    dict_overwrite_category_to_value : Union[Dict[str, float], None]
        Optional dictionary used to overwrite values associated with category (k)
    path_out_figure : pathlib.Path
        Optional path to specify for writing output of plot of historical
        inventory trajectories.
    reallocate_charcoal_production: bool
        Send charchoal production to SCOE and INEN biomass?
    **kwargs : 
        passed to sf._write_csv
    """
    
    ##  INITIALIZATION

    # get files
    dict_dfs = get_files(path_in_inv_excels)
    df_cw = pd.read_csv(path_in_cw_inv_excels)
    df_emissions_deforestation = get_emissions_deforestation(path_in_emissions_deforestation, )
    
    # get some dictionaries
    dict_itab_gas_field_to_cw_gas, dict_cw_gas_to_itab_gas_field = get_inventory_gas_field_to_gas_dict()
    dict_table_ind_to_sheet_name = get_all_sheet_name_table_dict(dict_dfs, )
    fields_cat_ordered = get_category_fields_ordered(df_cw, )

    
    ##  BUILD TRAJECTORIES FROM INVENTORY TABLES

    df_cw_with_trajectories = add_inv_table_emissions_to_cw(
        dict_dfs,
        df_cw,
        fields_cat_ordered,
        dict_table_ind_to_sheet_name,
        dict_cw_gas_to_itab_gas_field,
        model_attributes,
    )


    ##  COMBINE WITH SSP TO FILL

    df_inv_aggregate = aggregate_inventory_and_fill_from_ssp(
        df_cw_with_trajectories,
        df_ssp,
        df_emissions_deforestation,
        model_attributes,
        time_periods,
        cats_drop = ["Forest Land - HWP"],
    )
    

    ##  SPLIT, WRITE, AND RETURN


    out = split_aggregate_inv_into_cw_and_trajectories(
        df_inv_aggregate,
        model_attributes,
        time_periods,
        dict_overwrite_category_to_value = dict_overwrite_category_to_value,
    )

    df_cw_out, df_trajectories, field_emission = out

    if reallocate_charcoal_production:
        # do some reallocation and update
        df_trajectories = allocate_charcoal_to_inen_and_scoe(
            df_trajectories,
            field_emission,
            time_periods,
        ) 
    
        out = (df_cw_out, df_trajectories, field_emission)

    

    # write output
    sf._write_csv(df_cw_out, path_out_cw_new, **kwargs, )
    sf._write_csv(df_trajectories, path_out_trajectories, **kwargs, )

    # plot and export if wanted
    plots = plot_and_export_inventory(
        df_trajectories,
        time_periods,
        path_out_figure = path_out_figure,
    )

    out = {
        "data": out,
        "plot": plots,
    }
    
    return out

    

