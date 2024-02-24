###
###    SET OF TOOLS FOR READING/WRITING FROM/TO SISEPUEDE DATA REPOSITORY
###

import itertools
import logging
import model_attributes as ma
import model_afolu as mafl
import model_ippu as mi
import model_circular_economy as mc
import model_electricity as ml
import model_energy as me
import model_socioeconomic as se
import numpy as np
import os, os.path
import pandas as pd
import re
import support_classes as sc
import support_functions as sf
import time
from typing import *
import warnings




class SISEPUEDEBatchDataRepository:
    """
    Interact with the sisepuede_data git hub repository (read and write) using
        SISEPUEDE model variables.

    Initialization Arguments
    ------------------------
    - dir_repository: path to repository containing all data
    - model_attributes: model_attributes.ModelAttributes object used to 
        coordinate and access variables
    """

    def __init__(self,
        dir_repository: str,
        model_attributes: ma.ModelAttributes,
    ): 

        self._initialize_attributes(model_attributes)
        self._initialize_fields()
        self._initialize_repository(dir_repository)




    #################################
    #   INITIALIZATION FUNCTIONS    #
    #################################

    def _initialize_attributes(self,
        model_attributes: ma.ModelAttributes,
    ) -> None:
        """
        Initialize model attributes and associated support classes. Initializes
            the following properties:

            * self.model_attributes
            * self.regions (sc.Regions)
            * self.time_periods (sc.TimePeriods)
        """

        self.model_attributes = model_attributes
        self.regions = sc.Regions(model_attributes)
        self.time_periods = sc.TimePeriods(model_attributes)

        return None



    def _initialize_fields(self,
    ) -> None:
        """
        Initialize fields that are used in repository as well as groups of 
            fields that can be ignored etc. Sets the following properties:

            * self.field_repo_iso
            * self.field_repo_location_code
            * self.field_repo_nation
            * self.field_repo_year

        """

        self.field_repo_country = "country"
        self.field_repo_iso = "iso_code3"
        self.field_repo_location_code = "location_code"
        self.field_repo_nation = "Nation"
        self.field_repo_year = "Year"

        return None



    def _initialize_repository(self,
        dir_repository: str,
    ) -> None:
        """
        Initialize the data repository; check structure, etc. Sets the following
            properties:

            * self.dict_sector_to_subdir
            * self.dir_repository
            * self.key_historical
            * self.key_projected
            * self.subdir_input_to_sisepuede
        """

        # dictionary mapping SISEPUEDE sector name to sisepuede_data repository directory
        dict_sector_to_subdir = {
            "AFOLU": "AFOLU",
            "Circular Economy": "CircularEconomy",
            "Energy": "Energy",
            "IPPU": "IPPU", 
            "Socioeconomic": "SocioEconomic"
        }

        self.dict_sector_to_subdir = dict_sector_to_subdir
        self.dir_repository = sf.check_path(dir_repository, create_q = False)
        self.key_historical = "historical"
        self.key_projected = "projected"
        self.subdir_input_to_sisepuede = "input_to_sisepuede"

        return None



    

    ########################
    #    CORE FUNCTIONS    #
    ########################
    
    def check_periods_keep(self,
        periods_keep: Union[List[str], str, None] = None,
    ) -> Union[List[str], None]:
        """
        Check specification of periods_keep. Returns None if invalid.

        Function Arguments
        ------------------
        - periods_keep: projected or historical? If None, keep both. Can specify
            as a list or str containing 
            SISEPUEDEBatchDataRepository.key_historical or
            SISEPUEDEBatchDataRepository.key_projected
        """
        # check keys
        periods_all = [self.key_historical, self.key_projected]
        periods_keep = (
            [periods_keep] 
            if isinstance(periods_keep, str) 
            else periods_keep
        )
        periods_keep = (
            [self.key_historical, self.key_projected]
            if not sf.islistlike(periods_keep)
            else [x for x in periods_keep if x in periods_all]
        )
        if len(periods_keep) == 0:
            return None
        
        return periods_keep



    def file_to_sd_dirs(self,
        fp_csv: Union[pd.DataFrame, str],
        fields_ind: List[str],
        years_historical: List[int],
        dict_rename: Union[Dict[str, str], None] = None,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """
        Read a CSV file and return a dictionary of files to write
        
        Function Arguments
        ------------------
        - fp_csv: file path of CSV to read OR a DataFrame representing an input
            table
        - fields_ind: fields to use as index (present in all output CSVs). 
            NOTE: if dict_rename is not None and any index fielids are 
            renamed by dict_rename, then fields_ind should specify renaming
            targets.
        - years_historical: years to consider as historical
        
        Keyword Arguments
        -----------------
        - dict_rename: optional dictionary used to rename fields that are read
            in.
            NOTE: Renaming occurs *before* extracting fields_ind, so fields_ind
                should reference target renamed fields if applicable
        """
        
        # some quick checks
        return_none = (not (isinstance(fp_csv, pd.DataFrame) | isinstance(fp_csv, str)))
        return_none |= (
            not os.path.exists(fp_csv)
            if isinstance(return_none, str)
            else return_none
        )
        return_none |= (
            len(years_historical) == 0
            if sf.islistlike(years_historical)
            else False
        )

        if return_none:
            return None
        
        # check time periods/regions
        regions = self.regions
        time_periods = self.time_periods

        # read and check time indexing--add years if only time period is included
        df_csv = (
            pd.read_csv(fp_csv)
            if isinstance(fp_csv, str)
            else fp_csv
        )

        if (time_periods.field_year not in df_csv.columns) and (time_periods.field_time_period not in df_csv.columns):
            return None

        df_csv = (
            time_periods.tps_to_years(df_csv)
            if time_periods.field_year not in df_csv.columns
            else df_csv
        )
        
        # rename csv 
        dict_rnm = {}
        if isinstance(dict_rename, dict):
            for k, v in dict_rename.items():
                dict_rnm.update({k: v}) if (k in df_csv.columns) else None

        df_csv.rename(columns = dict_rnm, inplace = True)
        field_year = dict_rnm.get(time_periods.field_year, time_periods.field_year)
        
        # get fields and return None if invalid
        fields_ind = [x for x in fields_ind if x in df_csv.columns]
        fields_dat = [x for x in df_csv.columns if x in self.model_attributes.all_variable_fields]
        if min(len(fields_dat), len(fields_ind)) == 0:
            return None
        
        
        # initialize and write output
        dict_out = {}
        for fld in fields_dat:
            
            df_ext = df_csv[fields_ind + [fld]]
            
            df_ext_hist = df_ext[
                df_ext[field_year].isin(years_historical)
            ].reset_index(drop = True)
            
            df_ext_proj = df_ext[
                ~df_ext[field_year].isin(years_historical)
            ].reset_index(drop = True)
            
            dict_out.update(
                {
                    fld: {
                        self.key_historical: df_ext_hist,
                        self.key_projected: df_ext_proj
                    }
                }
            )
            
        return dict_out



    def field_to_path(self,
        fld: str,
        key_type: str,
    ) -> Union[str, None]:
        """
        Convert SISEPUEDE field `fld` to output path in sisepuede_data 
            repository

        Function Arguments
        ------------------
        - fld: valid SISEPUEDE field. If invalid, returns None
        - key_type: "historical" or "projected". If invalid, returns None
        """
        modvar = self.model_attributes.dict_variables_to_model_variables.get(fld)
        key = (
            self.key_historical 
            if (key_type in ["historical", self.key_historical])
            else (
                self.key_projected
                if (key_type in ["projected", self.key_projected])
                else None
            )
        )

        if (modvar is None) | (key is None):
            return None

        # otherwise, get sector info and outputs
        sector = self.model_attributes.get_variable_subsector(modvar)
        sector = self.model_attributes.get_subsector_attribute(sector, "sector")
        subdir_sector = self.dict_sector_to_subdir.get(sector, sector)

        # create outputs
        fp_out_base = os.path.join(self.dir_repository, subdir_sector, fld, self.subdir_input_to_sisepuede)
        fp_out = os.path.join(fp_out_base, key, f"{fld}.csv")

        return fp_out



    def write_from_df(self,
        df_write: pd.DataFrame,
        years_historical: List[int],
        field_iso_out: Union[str, None] = None,
        field_region_out: Union[str, None] = None,
        field_year_out: Union[str, None] = None,
        fps_ignore: Union[List[str], None] = None,
        fps_include: Union[List[str], None] = None,
        key_historical: Union[str, None] = None,
        key_projected: Union[str, None] = None,
        periods_write: Union[List[str], str, None] = None,
        write_q: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Using directory dir_batch (in SISEPUEDE repository), generate inputs
            for sisepuede_data repo
        
        NOTE: if both (dirs_ignore | fps_ignore) & fps_include are specified,
            then write_from_rbd will not write. Ensure to only specify datasets
            in terms of either exclusion (if writing all or most) or inclusion
            (if only writing in terms of a few).

        Function Arguments
        ------------------
        - df_write: data frame to use to write fields
        - years_historical: list of integer years to consider historical
        
        Keyword Arguments
        -----------------
        - dirs_ignore: list of subdirectories to ignore
        - ext_read: extension of input files to read
        - fields_ignore: list of fields to ignore in each input file when 
            checking for fields that will be written to self.dir_repository
        - fps_ignore: optional file paths to ignore
        - fps_include: optional file paths to include
        - key_historical: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_historical
        - key_projected: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_projected
        - periods_write: projected or historical? If None, keep both. Can specify
            as a list or str containing 
            SISEPUEDEBatchDataRepository.key_historical or
            SISEPUEDEBatchDataRepository.key_projected
        - write_q: write output data to files
        """

        # some field initialization
        field_iso_out = (
            self.field_repo_iso
            if field_iso_out is None
            else field_iso_out
        )
        field_region_out = (
            self.field_repo_nation
            if field_region_out is None
            else field_region_out
        )
        field_year_out = (
            self.field_repo_year
            if field_year_out is None
            else field_year_out
        )
        fields_ind = [field_year_out, field_iso_out]
        dict_rename = {
            self.model_attributes.dim_region: self.field_repo_nation,
            self.field_repo_country: self.field_repo_nation,
            self.field_repo_nation.lower(): self.field_repo_nation,
            self.regions.field_iso: self.field_repo_iso,
            self.time_periods.field_year: self.field_repo_year
        }

        # subdirectory keys
        periods_write = self.check_periods_keep(periods_write)
        if periods_write is None:
            return None

        key_historical = (
            self.key_historical
            if not isinstance(key_historical, str)
            else key_historical
        )

        key_projected = (
            self.key_projected
            if not isinstance(key_projected, str)
            else key_projected
        )
    

        # intialize split of data frame
        dict_rnm = dict((x, str(x).lower()) for x in df_write.columns)
        dict_out = self.file_to_sd_dirs(
            df_write.rename(columns = dict_rnm),
            fields_ind,
            years_historical,
            dict_rename = dict_rename,
        )
        
        if dict_out is None:
            return None


        # get paths for files
        dict_paths = {}

        for fld in dict_out.keys():
            fp_out_hist = self.field_to_path(fld, self.key_historical)
            fp_out_proj = self.field_to_path(fld, self.key_projected)
            
            dict_paths.update(
                {
                    fld: {
                        self.key_historical: fp_out_hist,
                        self.key_projected: fp_out_proj
                    }
                }
            )


        if write_q:
            
            for fld in dict_out.keys():

                dict_dfs_cur = dict_out.get(fld)
                dict_paths_cur = dict_paths.get(fld)
                
                for key in periods_write:
                    
                    # get df
                    df_write = dict_dfs_cur.get(key)
                    if df_write is None:
                        continue

                    # check directory
                    fp = dict_paths_cur.get(key)
                    dir_base = os.path.dirname(fp)
                    os.makedirs(dir_base, exist_ok = True) if not os.path.exists(dir_base) else None
                    
                    df_write.to_csv(
                        fp, 
                        index = None,
                        encoding = "UTF-8"
                    )
                    
                    print(f"DataFrame successfully written to '{fp}'")
                
        return dict_out, dict_paths



    def write_from_rbd(self,
        dir_batch: str,
        dict_years_historical: Union[Dict[str, List[int]], List[int]],
        dirs_ignore: Union[List[str], None] = None,
        ext_read: str = "csv",
        field_iso_out: Union[str, None] = None,
        field_region_out: Union[str, None] = None,
        field_year_out: Union[str, None] = None,
        fps_ignore: Union[List[str], None] = None,
        fps_include: Union[List[str], None] = None,
        key_historical: Union[str, None] = None,
        key_projected: Union[str, None] = None,
        periods_write: Union[List[str], str, None] = None,
        write_q: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Using directory dir_batch (in SISEPUEDE repository), generate inputs
            for sisepuede_data repo
        
        NOTE: if both (dirs_ignore | fps_ignore) & fps_include are specified,
            then write_from_rbd will not write. Ensure to only specify datasets
            in terms of either exclusion (if writing all or most) or inclusion
            (if only writing in terms of a few).

        Function Arguments
        ------------------
        - dir_batch: directory storing batch data using lac_decarbonization 
            structure
        - dict_years_historical: dictionary mapping a file to years historical 
            OR a list of integer years to consider histroical
        
        Keyword Arguments
        -----------------
        - dirs_ignore: list of subdirectories to ignore
        - ext_read: extension of input files to read
        - fields_ignore: list of fields to ignore in each input file when 
            checking for fields that will be written to self.dir_repository
        - fps_ignore: optional file paths to ignore
        - fps_include: optional file paths to include
        - key_historical: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_historical
        - key_projected: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_projected
        - periods_write: projected or historical? If None, keep both. Can 
            specify as a list or str containing 
            SISEPUEDEBatchDataRepository.key_historical or
            SISEPUEDEBatchDataRepository.key_projected
        - write_q: write output data to files
        """

        # check inclusion/exclusion specification
        return_none = (dirs_ignore is not None) | (fps_ignore is not None)
        return_none &= (fps_include is not None)

        if return_none:
            msg = f"""
                Error in write_from_rbd: cannot specify both as exclusion (at 
                least one of dirs_ignore or fps_ignore are None) and inclusion
                (fps_include is not None). Re-specify as exclusion only or 
                inclusion only. 
            """
            
            raise RuntimeError(msg)
            return None

        # some field initialization
        field_iso_out = (
            self.field_repo_iso
            if field_iso_out is None
            else field_iso_out
        )
        field_region_out = (
            self.field_repo_nation
            if field_region_out is None
            else field_region_out
        )
        field_year_out = (
            self.field_repo_year
            if field_year_out is None
            else field_year_out
        )
        fields_ind = [field_year_out, field_iso_out]
        dict_rename = {
            self.model_attributes.dim_region: self.field_repo_nation,
            self.field_repo_country: self.field_repo_nation,
            self.field_repo_nation.lower(): self.field_repo_nation,
            self.regions.field_iso: self.field_repo_iso,
            self.time_periods.field_year: self.field_repo_year
        }

        periods_write = self.check_periods_keep(periods_write)
        if periods_write is None:
            return None

        # subdirectory keys
        key_historical = (
            self.key_historical
            if not isinstance(key_historical, str)
            else key_historical
        )

        key_projected = (
            self.key_projected
            if not isinstance(key_projected, str)
            else key_projected
        )
        
        # directory checks--make output if not exstis + loop through subdirectories to check for available data
        subdirs = (
            [x for x in os.listdir(dir_batch) if os.path.join(dir_batch, x) not in dirs_ignore]
            if sf.islistlike(dirs_ignore)
            else os.listdir(dir_batch)
        )

        
        dict_out = {}
        dict_paths = {}
        
        for subdir in subdirs:
            fp_subdir = os.path.join(dir_batch, subdir)

            if os.path.isdir(fp_subdir):
                
                fns_read = [x for x in os.listdir(fp_subdir) if x.endswith(f".{ext_read}")]
                
                for fn in fns_read:
                    years_historical = (
                        dict_years_historical.get(fn)
                        if isinstance(dict_years_historical, dict)
                        else dict_years_historical
                    )
                    
                    fp_read = os.path.join(fp_subdir, fn)
                    if sf.islistlike(fps_ignore):
                        fp_read = None if (fp_read in fps_ignore) else fp_read
                    elif sf.islistlike(fps_include):
                        fp_read = None if (fp_read not in fps_include) else fp_read
                    
                    
                    dict_read = self.file_to_sd_dirs(
                        fp_read,
                        fields_ind,
                        years_historical,
                        dict_rename = dict_rename,
                    )
                    
                    # get variable information
                    if dict_read is None:
                        continue
                
                    for fld in dict_read.keys():
                        fp_out_hist = self.field_to_path(fld, self.key_historical)
                        fp_out_proj = self.field_to_path(fld, self.key_projected)
                        
                        dict_paths.update(
                            {
                                fld: {
                                    self.key_historical: fp_out_hist,
                                    self.key_projected: fp_out_proj
                                }
                            }
                        )
                        
                    dict_out.update(dict_read) 


        # write outputs?
        if write_q:
            
            for fld in dict_out.keys():

                dict_dfs_cur = dict_out.get(fld)
                dict_paths_cur = dict_paths.get(fld)
                
                for key in periods_write:
                    
                    # get df
                    df_write = dict_dfs_cur.get(key)
                    if df_write is None:
                        continue

                    # check directory
                    fp = dict_paths_cur.get(key)
                    dir_base = os.path.dirname(fp)
                    os.makedirs(dir_base, exist_ok = True) if not os.path.exists(dir_base) else None
                    
                    df_write.to_csv(
                        fp, 
                        index = None,
                        encoding = "UTF-8"
                    )
                    
                    print(f"DataFrame successfully written to '{fp}'")
                
        
        return dict_out, dict_paths


        
    def read(self,
        dict_modvars: Union[Dict[str, Union[List[str], None]], List[str], str, None],
        add_time_periods: bool = False,
        periods_keep: Union[List[str], str, None] = None,
    ) -> pd.DataFrame: 
        """
        Read inputs from the repository for use.
        
        Function Arguements
        -------------------
        - dict_modvars: dictionary with model variables as keys and a list of 
            categories to apply to (or None to read all applicable)
            
        Keyword Arguements
        ------------------
        - add_time_periods: add time periods to input?
        - periods_keep: projected or historical? If None, keep both. Can specify
            as a list or str containing 
            SISEPUEDEBatchDataRepository.key_historical or
            SISEPUEDEBatchDataRepository.key_projected
        """
        
        # SOME INITIALIZATION

        # some needed dictionaries
        dict_sector_to_subdir = self.dict_sector_to_subdir
        dict_subsec_abv_to_sector = (
            self.model_attributes
            .get_subsector_attribute_table()
            .field_maps
            .get("abbreviation_subsector_to_sector")
        )
        dict_subsec_to_subsec_abv = (
            self.model_attributes
            .get_subsector_attribute_table()
            .field_maps
            .get("subsector_to_abbreviation_subsector")
        )
        
        # check dictionary specification
        dict_modvars = (
            self.model_attributes.all_variables
            if dict_modvars is None
            else (
                {dict_modvars: None} 
                if isinstance(dict_modvars, str) 
                else dict_modvars
            )
        )

        dict_modvars = (
            dict((x, None) for x in dict_modvars)
            if sf.islistlike(dict_modvars)
            else dict_modvars
        ) 

        # check if subsector names are included and convert to modevl variables
        if isinstance(dict_modvars, dict):
            all_keys = dict_modvars.keys()
            sectors = [x for x in dict_modvars.keys() if x in self.model_attributes.all_sectors]
            
            for sector in sectors:
                vals = dict_modvars.get(sector)
                dict_modvars.update(
                    dict(
                        (k, vals) for k in self.model_attributes.get_sector_variables(sector)
                    )
                )
                dict_modvars.pop(sector)

        periods_keep = self.check_periods_keep(periods_keep)
        if periods_keep is None:
            return None
        

        # some fields
        field_iso = self.field_repo_iso.lower()
        field_year = self.field_repo_year.lower()
        fields_index = [field_iso, field_year] 
        fields_to_iso = [self.field_repo_location_code]
        
        # initialize output
        df_out = None
        df_index = None # used to govern merges
        dict_modvar_to_fields = {}
        
        modvars = list(dict_modvars.keys())

        for k, modvar in enumerate(modvars):
            
            cats_defined = self.model_attributes.get_variable_categories(modvar)
            cats = dict_modvars.get(modvar)
            cats = cats_defined if (cats is None) else cats
            cats = (
                [x for x in cats_defined if x in cats]
                if (cats_defined is not None)
                else None
            )
            
            subsec = self.model_attributes.get_variable_subsector(modvar)
            sector = dict_subsec_abv_to_sector.get(
                dict_subsec_to_subsec_abv.get(subsec)
            )
            sector_repo = dict_sector_to_subdir.get(sector)

            # check if need to skip iteration
            continue_q = (sector_repo is None)
            continue_q |= (len(cats) == 0) if (cats is not None) else False
            if continue_q:
                continue
            
            # build field names to retrieve
            var_names = self.model_attributes.build_varlist(
                subsec,
                modvar, 
                restrict_to_category_values = cats,
            )
            
            for var_name in var_names:
                
                """
                restriction = None if (cat is None) else [cat]
                var_name = self.model_attributes.build_varlist(
                    subsec,
                    modvar, 
                    restrict_to_category_values = restriction
                )[0]
                """;
                
                df_var = []
                i = 0

                for key in periods_keep:

                    fp_read = self.field_to_path(var_name, key)
                    if not os.path.exists(fp_read):
                        continue

                    try:
                        # read
                        df_var_cur = pd.read_csv(fp_read)
                        df_var_cur.dropna(inplace = True)
                        
                        # rename where necessary
                        dict_rnm_to_iso = dict(
                            (x, field_iso) 
                            for x in fields_to_iso
                            if x in df_var_cur.columns
                        )
                        df_var_cur.rename(
                            columns = dict_rnm_to_iso,
                            inplace = True
                        )
                        
                        # clean the fields
                        dict_rnm = dict((x, x.lower()) for x in df_var_cur.columns)
                        df_var_cur.rename(
                            columns = dict_rnm,
                            inplace = True
                        )

                        # drop any unwanted columns
                        df_var_cur = df_var_cur[fields_index + [var_name]]
                        df_var_cur.set_index(fields_index, inplace = True)

                        if i > 0:
                            inds_prev = df_var[0].index
                            df_var_cur = df_var_cur[
                                [(x not in inds_prev) for x in df_var_cur.index]
                            ]
                            df_var[0].reset_index(inplace = True)
                            df_var_cur.reset_index(inplace = True)
                        
                        df_var.append(df_var_cur)

                        i += 1

                    except Exception as e:
                        warnings.warn(f"Error trying to read {fp_read}: {e}")
                        
                (
                    df_var[0].reset_index(inplace = True) 
                    if len(df_var) == 1 
                    else None
                )

                # concatenate and sort
                df_var = pd.concat(df_var, axis = 0) if (len(df_var) > 0) else None
                global dfv
                dfv = df_var

                if ((fields_index is not None) and (df_var is not None)):                    
                    # get dictionaries
                    fields_add = sorted([x for x in df_var.columns if x not in fields_index])
                    fields_exist = dict_modvar_to_fields.get(modvar)
                    
                    (
                        dict_modvar_to_fields.update({modvar: fields_add}) 
                        if fields_exist is None
                        else dict_modvar_to_fields[modvar].extend(fields_add)
                    )
                    
                if df_var is not None:

                    df_var.sort_values(by = fields_index, inplace = True)
                    df_var.reset_index(drop = True, inplace = True)

                    if (df_out is None):

                        df_out = [df_var]
                        df_index = df_var[fields_index].copy()

                    else:
                        #df_out.append(df_var)
                        
                        fold_q = (
                            True
                            if df_var[fields_index].shape != df_index.shape
                            else not all(df_var[fields_index] == df_index)
                        )

                        # setup indexing data frame
                        if fold_q:
                            df_out = pd.concat(df_out, axis = 1)

                            df_index = (
                                pd.merge(
                                    df_index, 
                                    df_var[fields_index],
                                    how = "outer"
                                )
                                .sort_values(by = fields_index)
                                .reset_index(drop = True)
                            )


                            df_out = (
                                pd.merge(
                                    df_index, 
                                    df_out, 
                                    how = "left", 
                                    on = fields_index
                                )
                                .sort_index()
                                .reset_index(drop = True)
                            )
                            df_out = [df_out]
                            df_var = (
                                pd.merge(
                                    df_index, 
                                    df_var, 
                                    how = "left", 
                                    on = fields_index
                                )
                                .sort_index()
                                .reset_index(drop = True)
                            )

                        df_out.append(df_var[[var_name]])


        if (df_out is not None) and (fields_index is not None):

            """
            df_out = (
                df_out[0].join(
                    df_out[1:],
                    how = "outer"
                )
                if len(df_out) > 1
                else df_out[0]
            )
            """;
            df_out = pd.concat(df_out, axis = 1)
            df_out.sort_values(by = fields_index, inplace = True) 
            df_out.reset_index(drop = True, inplace = True) 

        df_out = (
            self.time_periods.years_to_tps(df_out)
            if add_time_periods
            else df_out
        )

        return df_out