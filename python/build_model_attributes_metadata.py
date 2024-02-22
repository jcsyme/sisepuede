"""
Use this file to build functions and methods that can generate metadata and/or
    tables/other information (including figures) based on the Model Attributes
    module.
"""
import itertools
import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import support_classes as sc
import support_functions as sf
from typing import *


class InformationTableProperties:
    """
    Class to preserve information table properties across uses
    """

    def __init__(self,
    ):
        self._initialize_properties()



    def _initialize_properties(self,
    ) -> None:
        """
        Set some properties, including shared fields. Sets the following
            properties:

            * self.field_categories
            * self.field_category_primary_name
            * self.field_field_emission
            * self.field_field_subsector_total
            * self.field_gas
            * self.field_gas_name
            * self.field_info
            * self.field_model_variable
            * self.field_sector
            * self.field_subsector
        """

        self.field_categories = "category_value"
        self.field_category_primary_name = "category_name"
        self.field_field_emission = "field"
        self.field_field_subsector_total = "subsector_total_field"
        self.field_gas = "gas"
        self.field_gas_name = "gas_name"
        self.field_info = "model_variable_information"
        self.field_model_variable = "model_variable"
        self.field_sector = "sector"
        self.field_subsector= "subsector"

        return None






def build_emissions_information_table(
    model_attributes: ma.ModelAttributes,
) -> pd.DataFrame:
    """
    Build a data frame with rows giving gasses, gas names, model variables, 
        subsector, sector, and subsector field totals.

    Function Arguments
    ------------------
    - model_attributes: model_attributes.ModelAttributes object used to generate
        and manage variables
    """
    attr_gas = model_attributes.dict_attributes.get("emission_gas")
    dict_gas_to_name = attr_gas.field_maps.get(f"{attr_gas.key}_to_name")
    dict_gas_to_emision_modvars = model_attributes.dict_gas_to_total_emission_modvars
    
    # get shared field names
    table_properties = InformationTableProperties()
    field_out_categories = table_properties.field_categories
    field_out_category_primary_name = table_properties.field_category_primary_name
    field_out_field_emission = table_properties.field_field_emission
    field_out_field_subsector_total = table_properties.field_field_subsector_total
    field_out_field_subsector_total = table_properties.field_field_subsector_total
    field_out_gas = table_properties.field_gas
    field_out_gas_name = table_properties.field_gas_name
    field_out_info = table_properties.field_info
    field_out_model_variable = table_properties.field_model_variable
    field_out_sector = table_properties.field_sector
    field_out_subsector = table_properties.field_subsector


    df_out = []
    
    for gas in dict_gas_to_emision_modvars.keys():
        
        gas_name = dict_gas_to_name.get(gas)
        modvars = dict_gas_to_emision_modvars.get(gas)
        
        # loop over available modvars
        for modvar in modvars:
            
            # get some attributes 
            subsec = model_attributes.get_variable_subsector(modvar)
            field_subsector_total = model_attributes.get_subsector_emission_total_field(subsec)
            sector = model_attributes.get_subsector_attribute(subsec, "sector")
            pycat_primary = model_attributes.get_subsector_attribute(subsec, "pycategory_primary_element")

            # fields and categories
            fields = model_attributes.build_varlist(None, modvar)
            cats = model_attributes.get_variable_categories(modvar)
            cats = [""] if (cats is None) else cats
            pycats_primary = [
                ("" if (x == "") else pycat_primary)
                for x in cats
            ]
            # attempt a description 
            info = model_attributes.get_variable_attribute(modvar, "information")
            info = "" if not isinstance(info, str) else info

            # build current component
            df_cur = pd.DataFrame({
                field_out_field_emission: fields,
                field_out_categories: cats,
                field_out_category_primary_name: pycats_primary,
            })
            df_cur[field_out_field_subsector_total] = field_subsector_total
            df_cur[field_out_model_variable] = modvar
            df_cur[field_out_gas] = gas
            df_cur[field_out_gas_name] = gas_name
            df_cur[field_out_info] = info
            df_cur[field_out_sector] = sector
            df_cur[field_out_subsector] = subsec
            
            df_out.append(df_cur)
    
    # set ordererd output fields
    fields_ord = [
        field_out_sector,
        field_out_subsector,
        field_out_field_emission,
        field_out_model_variable,
        field_out_categories,
        field_out_category_primary_name,
        field_out_gas,
        field_out_gas_name,
        field_out_field_subsector_total,
        field_out_info,
    ]
    
    df_out = (
        pd.concat(df_out, axis = 0)
        .sort_values(by = fields_ord)
        .reset_index(drop = True)
    )
    df_out = df_out[fields_ord]
    
    return df_out



def build_variable_information_table(
    model_attributes: ma.ModelAttributes,
    modvars: Union[List[str], None],
) -> Union[pd.DataFrame, None]:
    """
    Build a data frame with rows giving gasses, gas names, model variables, 
        subsector, sector, and subsector field totals.

    Function Arguments
    ------------------
    - model_attributes: model_attributes.ModelAttributes object used to generate
        and manage variables
    - modvars: model variables to build information for. If None, returns all
        model variables.
    """
    attr_gas = model_attributes.dict_attributes.get("emission_gas")
    dict_gas_to_name = attr_gas.field_maps.get(f"{attr_gas.key}_to_name")
    dict_gas_to_emision_modvars = model_attributes.dict_gas_to_total_emission_modvars
    
    # get shared field names
    table_properties = InformationTableProperties()
    field_out_categories = table_properties.field_categories
    field_out_category_primary_name = table_properties.field_category_primary_name
    field_out_field_emission = table_properties.field_field_emission
    field_out_field_subsector_total = table_properties.field_field_subsector_total
    field_out_field_subsector_total = table_properties.field_field_subsector_total
    field_out_gas = table_properties.field_gas
    field_out_gas_name = table_properties.field_gas_name
    field_out_info = table_properties.field_info
    field_out_model_variable = table_properties.field_model_variable
    field_out_sector = table_properties.field_sector
    field_out_subsector = table_properties.field_subsector
    
    modvars = (
        model_attributes.all_model_variables
        if not sf.islistlike(modvars)
        else list(modvars)
    )


    df_out = []
    
    # loop over available modvars
    for modvar in modvars:

        gas = model_attributes.get_variable_characteristic(modvar, model_attributes.varchar_str_emission_gas)
        gas_name = dict_gas_to_name.get(gas)
        gas = "" if (gas is None) else gas
        gas_name = "" if (gas_name is None) else gas_name
        
        # get some attributes 
        subsec = model_attributes.get_variable_subsector(modvar)
        sector = model_attributes.get_subsector_attribute(subsec, "sector")
        pycat_primary = model_attributes.get_subsector_attribute(subsec, "pycategory_primary_element")

        # fields and categories
        fields = model_attributes.build_varlist(None, modvar)
        cats = model_attributes.get_variable_categories(modvar)
        cats = [""] if (cats is None) else cats
        cats = (
            [str(x) for x in itertools.product(cats, cats)]
            if (len(cats)**2 == len(fields)) 
            else cats
        )

        pycats_primary = [
            ("" if (x == "") else pycat_primary)
            for x in cats
        ]
        # attempt a description 
        info = model_attributes.get_variable_attribute(modvar, "information")
        info = "" if not isinstance(info, str) else info

        lf = len(fields)
        lc = len(cats)
        lp = len(pycats_primary)
        if len(set({lf, lc, lp})) != 1:
            print(modvar)
            print(f"cats:\t{lc}")
            print(f"fields:\t{lf}")
            print(f"pycats_primary:\t{lp}")
            print("\n")

        # build current component
        df_cur = pd.DataFrame({
            field_out_field_emission: fields,
            field_out_categories: cats,
            field_out_category_primary_name: pycats_primary,
        })
        df_cur[field_out_model_variable] = modvar
        df_cur[field_out_gas] = gas
        df_cur[field_out_gas_name] = gas_name
        df_cur[field_out_info] = info
        df_cur[field_out_sector] = sector
        df_cur[field_out_subsector] = subsec
        
        df_out.append(df_cur)
    
    # set ordererd output fields
    fields_ord = [
        field_out_sector,
        field_out_subsector,
        field_out_field_emission,
        field_out_model_variable,
        field_out_categories,
        field_out_category_primary_name,
        field_out_gas,
        field_out_gas_name,
        field_out_info,
    ]
    
    df_out = (
        pd.concat(df_out, axis = 0)
        .sort_values(by = fields_ord)
        .reset_index(drop = True)
    )
    df_out = df_out[fields_ord]
    
    return df_out