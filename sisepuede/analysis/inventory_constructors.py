"""Construct tables from inventory Common Reporting Tables (CRTs)
"""


"""Use this file to construct an inventory for historical based on Excel files.
"""
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import pathlib
import re
import sisepuede.analysis.inventory_tree as it
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.utilities._toolbox as sf
import sisepuede.visualization.plots as svp
import warnings
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
# _REGEX_PATTERN_INVENTORY_TABLES = re.compile("GHG Emissions_(.*\d)_Complete File.xlsx")

# unit info
_UNITS_MASS_INV = "kt"





#####################################
#    INITIALIZE GLOBAL VARIABLES    #
#####################################

# delimiters
_DELIM_IPCC = "."

# field storing ipcc codes
_FIELD_EMISSION_TOTAL_EST = "emission_co2e_total_kt"
_FIELD_IPCC_CODE = "ipcc_code"

# ipcc crt sectors
_IPCC_CRT_SECTOR_AGRICULTURE = "AGRICULTURE"
_IPCC_CRT_SECTOR_ENERGY = "ENERGY"
_IPCC_CRT_SECTOR_IPPU = "IPPU"
_IPCC_CRT_SECTOR_LULUCF = "LULUCF"
_IPCC_CRT_SECTOR_WASTE = "WASTE"



#################
#    CLASSES    #
#################

##  ERROR CLASSES

class MissingParentError(Exception):
    pass

class MissingTableError(Exception):
    pass



class CommonReportingTable:
    """A common reporting table from the UN. Instantiated with an Excel file. 
    """

    def __init__(self,
        template: Union[Dict, None] = None,
    ) -> None:

        self._initialize_model_attributes()
        self._initialize_structure(
            template = template, 
        )
        self._initialize_ipcc_sectors()
        
        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_model_attributes(self,
    ) -> None:
        """Initialize all ModelAttributes related objects
        """

        file_struct = sfs.SISEPUEDEFileStructure()
    

        ##  SET PROPERTIES

        self.file_struct = file_struct
        self.model_attributes = file_struct.model_attributes

        return None



    def _initialize_structure(self,
        template: Union[Dict, None] = None,
    ) -> None:
        """Initialize table structure and related properties. Reads CRT template
            from UNFCCC (in examples)
        """

        # set name/path
        fn_template = "crt_2.80.xlsx"
        fp_template = os.path.join(self.file_struct.dir_ref_examples, fn_template, )


        # read tempalte if not provided directly
        if not isinstance(template, dict):
            template = pd.read_excel(fp_template, sheet_name = None, )

        sheets = sorted(list(template.keys()))


        ##  SET SOME KEY SHEETS
        
        sheet_en = "Table1"
        sheet_ip = "Table2(I)"
        sheet_ag = "Table3"
        sheet_lu = "Table4"
        sheet_wt = "Table5"

        # set here so that check_sheets works--ordered
        sheets_sector_summary = [
            sheet_en,
            sheet_ip,
            sheet_ag,
            sheet_lu,
            sheet_wt
        ]

        #  check sheets
        self.sheets = sheets
        self.sheets_sector_summary = sheets_sector_summary
        self.check_sheets(
            template,
            only_require_for_sector_summaries = True, 
        )


        ##  SET PROPERTIES

        self.fn_template = fn_template
        self.sheet_en = sheet_en
        self.sheet_ip = sheet_ip
        self.sheet_ag = sheet_ag
        self.sheet_lu = sheet_lu
        self.sheet_wt = sheet_wt
        self.template = template
        
        return None



    def _initialize_ipcc_sectors(self,
    ) -> None:
        """Initialize some sector constants and properties.
        """

        ##  SET SOME SECTORAL SHORTCUTS

        # integer to sector name maps
        dict_ipcc_ind_to_ipcc_sector = {
            1: _IPCC_CRT_SECTOR_ENERGY,
            2: _IPCC_CRT_SECTOR_IPPU,
            3: _IPCC_CRT_SECTOR_AGRICULTURE,
            4: _IPCC_CRT_SECTOR_LULUCF,
            5: _IPCC_CRT_SECTOR_WASTE,
        }

        dict_ipcc_sector_to_ipcc_ind = sf.reverse_dict(
            dict_ipcc_ind_to_ipcc_sector, 
        )

        # name to table
        dict_ipcc_sector_to_summary_table = {
            _IPCC_CRT_SECTOR_AGRICULTURE: self.sheet_ag,
            _IPCC_CRT_SECTOR_ENERGY: self.sheet_en,
            _IPCC_CRT_SECTOR_IPPU: self.sheet_ip,
            _IPCC_CRT_SECTOR_LULUCF: self.sheet_lu,
            _IPCC_CRT_SECTOR_WASTE: self.sheet_wt,
        }


        ##  SET PROPERTIES

        self.crt_sector_agriculture = _IPCC_CRT_SECTOR_AGRICULTURE
        self.crt_sector_energy = _IPCC_CRT_SECTOR_ENERGY
        self.crt_sector_ippu = _IPCC_CRT_SECTOR_IPPU
        self.crt_sector_lulucf = _IPCC_CRT_SECTOR_LULUCF
        self.crt_sector_waste = _IPCC_CRT_SECTOR_WASTE
        self.dict_ipcc_ind_to_ipcc_sector = dict_ipcc_ind_to_ipcc_sector
        self.dict_ipcc_sector_to_ipcc_ind = dict_ipcc_sector_to_ipcc_ind
        self.dict_ipcc_sector_to_summary_table = dict_ipcc_sector_to_summary_table

        return None
        
        


    def check_sheets(self,
        candidate: Dict[str, pd.DataFrame],
        only_require_for_sector_summaries: bool = False,
        stop_on_error: bool = True,
    ) -> int:
        """Check sheets in candidate
        """

        if not isinstance(candidate, dict):
            raise TypeError(f"Invalide candidate type {type}")

        s_needed = (
            self.sheets_sector_summary
            if only_require_for_sector_summaries
            else self.sheets
        )
        s_needed = set(s_needed)
        s_avail = set(candidate.keys())

        if not s_needed.issubset(s_avail):
            msg = f"Candidate CRT tables failed check--one or more sheets missing."
            if stop_on_error:
                raise RuntimeError(msg)

            warnings.warn(msg)
            return 1

        return 0
            


    #########################
    #    PRIMARY METHODS    #
    #########################

    def build_tree_from_sector_table(self,
        dict_tables: Dict[str, pd.DataFrame],
        sector_spec: Union[int, str],
        field_code: str = _FIELD_IPCC_CODE,
        flag_memo_on: Union[str, None] = "memo",
        stop_on_error: bool = False,
    ) -> it.InventoryNode:
        """Using a sectoral CRT, build an inventory tree

        Function Arguments
        ------------------
        dict_tables : Dict[str, pd.DataFrame]
            Dictionary of tables from the self.read() method
        sector_spec : Union[int, str]
            Sector specification for reading the tables

        Keyword Arguments
        -----------------
        field_code : str
            Field in the table storing the IPCC code
        flag_memo_on : Union[str, None]
            Flag in the code field, when iterating through a table, marking that
            subsequent items are memo items instead of inventory items.
        stop_on_error : bool
            Stop if an error is encountered?
        """

        ##  INITIALIZATION


        # try to retrieve the table
        table = self.get_table(dict_tables, sector_spec, )
        if table is None:
            if stop_on_error:
                MissingTableError(f"Unable to build tree: table '{sector_spec}' not found.")
            
            return None


        ##  BUILD TREE

        # set memo item
        memo = False
        tree = None

        for i, row in table.iterrows():
            
            # update the memo and get the code
            memo |= ("memo" in str(row[field_code]).lower())
            code = self.get_ippc_code_from_row(
                row, 
                field = field_code, 
            )

            # if code isn't found, check if memo item needs to be added
            if code is None: 
                continue

            # get components of the IPCC code
            components = code.split(_DELIM_IPCC)
            n_comp = len(components)
            if n_comp == 1:
                dict_emissions = get_emissions(row, field_code, )
                tree = it.InventoryNode(
                    code, 
                    dict_emissions = dict_emissions,
                    memo = memo,    
                )
                
                continue


            ##  CHECK IF THE NODE ALREADY EXISTS
            #    if so, add emissions and move one
            
            node_cur = it.get_node_dfs(tree, code, )
            if node_cur is not None:
                dict_emissions = get_emissions(row, field_code, )
                node_cur.emissions = dict_emissions
                node_cur.memo = memo

                continue
                
            
            ##  OTHERWISE, TRY TO GET THE PARENT NODE FROM TREE
            
            node_parent = None
            parent_code = "  " # initialize as non-zero length string
            i = 0

            # try getting parent
            while (node_parent is None) & (len(parent_code) > 0):
                i += 1
                parent_code = _DELIM_IPCC.join(components[0:(n_comp - i)])
                node_parent = it.get_node_dfs(tree, parent_code, )

            # if no parent is found, something's gone terible wrong
            if node_parent is None:
                raise MissingParentError(f"No parent to '{code}' found in the current tree. Check code.")


            # Otherwise, build downward
            for k in range(n_comp - i + 1, n_comp + 1):
                code_new = _DELIM_IPCC.join(components[0:k])

                # if not n_comp, then we're adding parents that we haven't seem yet, 
                #    so emissions can be set as a blank {}
                dict_emissions = (
                    get_emissions(row, field_code, )
                    if k == n_comp
                    else {}
                )
                
                # make a new node
                node_new = it.InventoryNode(
                    code_new, 
                    dict_emissions = dict_emissions, 
                    memo = memo,
                )

                # add new node to tree, set parent, and update node_parent as the child
                node_parent.children.append(node_new)
                node_new.parent = node_parent
                node_parent = node_new

        return tree



    def build_tree_from_crt(self,
        path: pathlib.Path,
        field_code: str = _FIELD_IPCC_CODE,
        stop_on_error: bool = False,
        **kwargs,
    ) -> it.InventoryNode:
        """Using a CRT set of tables, build an inventory tree across all sector.

        Function Arguments
        ------------------
        path : pathlib.Path
            Path to CRT Excel to read
        
        Keyword Arguments
        -----------------
        **kwargs : 
            passed to self.read() method
        """

        dict_tables = self.read(
            path, 
            only_require_for_sector_summaries = True,
            stop_on_error = stop_on_error, 
            **kwargs, 
        )


        ##  BUILD TREE

        tree = it.InventoryNode("0 - CRT")

        # iterate in numerical order
        sector_ids_ord = sorted(
            list(
                self.dict_ipcc_ind_to_ipcc_sector.keys()
            )
        )

        for sector in sector_ids_ord:
            subtree = self.build_tree_from_sector_table(
                dict_tables,
                sector,
                field_code = field_code,
                stop_on_error = stop_on_error,
            )

            if subtree is None:
                raise RuntimeError(f"sector {sector} failed")
                
            global tree2
            tree2 = subtree
            # update tree
            subtree.parent = tree
            tree.append(subtree)


        return tree



    def get_ippc_code_from_row(self,
        row: pd.Series,
        delim: str = _DELIM_IPCC,
        field: str = _FIELD_IPCC_CODE,
    ) -> Union[str, None]:
        """Split the row
        """
        string = str(row[field])
        if not valid_code(string):
            return None

        # get the code and remove the trailing period--remove any known issues
        out = sf.str_replace(
            string
            .split(" ")[0]
            .strip(),
            {
                "Navigation": "",
            }
        )
        
        if out[-1] == delim:
            out = out[0:-1]


        return out



    def get_sector(self,
        sector_spec: Union[int, str],
    ) -> Union[str, None]:
        """Try to get the IPCC sector from the index
        """
        # try the string specification
        get_try = self.dict_ipcc_sector_to_ipcc_ind.get(sector_spec, )
        if get_try is not None:
            return sector_spec

        # otherwise, try the integer
        out = self.dict_ipcc_ind_to_ipcc_sector.get(sector_spec, )

        return out



    def get_table(self,
        dict_tables: Dict[str, pd.DataFrame],
        sector_spec: Union[int, str],
        field_code: str = _FIELD_IPCC_CODE,
    ) -> Union[pd.DataFrame, None]:
        """Retrieve the IPCC table based on one of the following codes:

            * ENERGY         (1)
            * IPPU           (2)
            * AGRICULTURE    (3)
            * LULUCF         (4)
            * WASTE          (5)
            
        Returns None if the sector can't be found. 
        """
        # check input type
        sector_name = self.get_sector(sector_spec, )
        if not isinstance(sector_name, str):
            return None
            
        # get the table name
        table_name = self.dict_ipcc_sector_to_summary_table.get(sector_name, )
        if table_name is None:
            raise RuntimeError(f"Error: table name '{table_name}' not found in the CommonReportingTable.")

        # format
        col = 1
        df_table = dict_tables.get(table_name).copy()
        field_tmp = df_table.columns[col]
        
        # some bespoke fixes included in the CRT
        df_table[field_tmp] = df_table[field_tmp].replace(
            {
                "Total Energy": f"1{_DELIM_IPCC} Total Energy",     # Energy is missing the 1.
            }
        )

        # next, get columns from row 6
        field_names = list(df_table.iloc[6])
        field_names[1] = field_code

        # reduce
        df_table = (
            df_table
            .iloc[8:]
            .reset_index(drop = True,)
        )
        df_table.columns = field_names

        df_table = self.rename_to_clean_emissions(
            df_table
            .drop(
                columns = field_names[0]
            )
        )

        return df_table


            
    def read(self,
        path: pathlib.Path,
        only_require_for_sector_summaries: bool = False,
        stop_on_error: bool = False,
        verify: bool = True,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """Retrieve a table Excel and check it against the CRT template.

        
        Function Arguments
        ------------------
        path : pathlib.Path
            Path to CRT Excel to read

        Keyword Arguments
        -----------------
        only_require_for_sector_summaries : bool
            If verifying, only requires that sectoral summary tables are present
        stop_on_error : bool
            Stop if an error occurs? If False, returns None on an error
        verify : bool
            Verify the structure of the Excel file
        """

        # try to read it
        try:
            dict_tabs = pd.read_excel(path, sheet_name = None, )
            
        except Exception as e:
            msg = f"Unable to read tempalte from '{path}': {e}."
            if stop_on_error:
                raise RuntimeError(msg)

            warnings.warn(msg)
            return None

        # check
        out = 0
        if verify:
            out = self.check_sheets(
                dict_tabs, 
                only_require_for_sector_summaries = only_require_for_sector_summaries,
                stop_on_error = stop_on_error,
            )

        # return output
        out = (
            dict_tabs
            if out == 0 
            else None
        )

        return out



    def rename_to_clean_emissions(self,
        df: pd.DataFrame,
        field_code: str = _FIELD_IPCC_CODE,
        field_total: str = _FIELD_EMISSION_TOTAL_EST,
    ) -> pd.DataFrame:
        """Clean field names of emissions so that they only store gasses
        """

        fields = list(df.columns)
        fields_new = []

        for i, field in enumerate(fields):
            if field == field_code: 
                fields_new.append(field)
                continue

            # CO2
            if "co2" in field.lower():
                fields_new.append("co2")
                continue

            # modify total
            if "total ghg" in field.lower():
                fields_new.append(field_total)
                continue

            # others
            field_new = field.split("(")[0].strip().lower().replace(" ", "_")
            fields_new.append(field_new)
        
        df.columns = fields_new

        return df





###############################
###                         ###
###    PRIMARY FUNCTIONS    ###
###                         ###
###############################

def get_emissions(
    row: str,
    field_code: str,
) -> Dict[str, float]:
    """Get emissions from a row
    """

    dict_emissions = {}
    for k, v in row.to_dict().items():
        if k == field_code: continue

        if isinstance(v, str):
            dict_emissions.update({k: 0.0, })
            continue

        dict_emissions.update({k: np.nan_to_num(v, nan = 0), })

    return dict_emissions


    








def valid_code(
    string: str,
) -> bool:
    """Check if a row is valid
    """
    good = isinstance(string, str)
    good &= (len(string) >= 2) if good else good
    good &= string[0].isnumeric() if good else good
    good &= (string[1] == ".") if good else good

    return good