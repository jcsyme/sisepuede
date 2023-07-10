import os, os.path
import numpy as np
import pandas as pd
import support_functions as sf


##  the AttributeTable class checks existence, keys, key values, and generates field maps
class AttributeTable:

    def __init__(self, fp_table: str, key: str, fields_to_dict: list, clean_table_fields: bool = True):

        # verify table exists and check keys
        table = pd.read_csv(sf.check_path(fp_table, False), skipinitialspace = True)
        fields_to_dict = [x for x in fields_to_dict if x != key]
        
        
        # clean the fields in the attribute table?
        dict_fields_clean_to_fields_orig = {}
        if clean_table_fields:
            fields_orig = list(table.columns)
            dict_fields_clean_to_fields_orig = dict(zip(sf.clean_field_names(fields_orig), fields_orig))
            table = sf.clean_field_names(table)
            fields_to_dict = sf.clean_field_names(fields_to_dict)
            key = sf.clean_field_names([key])[0]
        
        
        # add a key if not specified
        if not key in table.columns:
            print(f"Key {key} not found in table '{fp_table}''. Adding integer key.")
            table[key] = range(len(table))
        # check all fields
        sf.check_fields(table, [key] + fields_to_dict)
        # check key
        if len(set(table[key])) < len(table):
            raise ValueError(f"Invalid key {key} found in '{fp_table}': the key is not unique. Check the table and specify a unique key.")
            
        
        # if no fields for the dictionary are specified, default to all
        if len(fields_to_dict) == 0:
            fields_to_dict = [x for x in table.columns if (x != key)]

        # clear RST formatting in the table if applicable
        if table[key].dtype in [object, str]:
            table[key] = np.array([sf.str_replace(str(x), {"`": "", "\$": ""}) for x in list(table[key])]).astype(str)
        # set all keys
        key_values = list(table[key])
        key_values.sort()
        
        # next, create dict maps
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"

            field_maps.update({field_fwd: sf.build_dict(table[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(table[fld])
            if (len(vals_unique) == len(table)):
                field_maps.update({field_rev: sf.build_dict(table[[fld, key]])})


        self.dict_fields_clean_to_fields_orig = dict_fields_clean_to_fields_orig
        self.field_maps = field_maps
        self.fp_table = fp_table
        self.key = key
        self.key_values = key_values
        self.table = table



class ModelAttributes:
    
    def __init__(self, dir_attributes):
        self.attribute_file_extension = ".csv"
        self.substr_categories = "attribute_"
        self.substr_varreqs = "table_varreqs_by_"
        self.substr_varreqs_allcats = f"{self.substr_varreqs}category_"
        self.substr_varreqs_partialcats = f"{self.substr_varreqs}partial_category_"
        
        self.all_attributes, self.dict_attributes, self.dict_varreqs = self.get_attribute_tables(dir_attributes)
        
        # run checks and raise errors if there are problems
        self.check_land_use_tables()
        
    
    
    # retrieve and format attribute tables for use   
    def get_attribute_tables(self, dir_att):

        # get available types
        all_types = [x for x in os.listdir(dir_att) if (self.attribute_file_extension in x) and ((self.substr_categories in x) or (self.substr_varreqs_allcats in x) or (self.substr_varreqs_partialcats in x))]   
        ##  batch load attributes/variable requirements and turn them into AttributeTable objects
        dict_attributes = {} 
        dict_varreqs = {} 
        for att in all_types:
            fp = os.path.join(dir_att, att)
            if self.substr_categories in att:
                nm = sf.clean_field_names([x for x in pd.read_csv(fp, nrows = 0).columns if "$" in x])[0]
                att_table = AttributeTable(fp, nm, [])
                dict_attributes.update({nm: att_table})
            elif (self.substr_varreqs_allcats in att) or (self.substr_varreqs_partialcats in att): 
                nm = att.replace(self.substr_varreqs, "").replace(self.attribute_file_extension, "")
                att_table = AttributeTable(fp, "variable", [])
                dict_varreqs.update({nm: att_table})
            else:
                raise ValueError(f"Invalid attribute '{att}': ensure '{self.substr_categories}', '{self.substr_varreqs_allcats}', or '{self.substr_varreqs_partialcats}' is contained in the attribute file.")

        ##  add some subsector/python specific information into the subsector table
        field_category = "primary_category"
        field_category_py = field_category + "_py"
        # add a new field
        df_tmp = dict_attributes["abbreviation_subsector"].table
        df_tmp[field_category_py] = sf.clean_field_names(df_tmp[field_category])
        df_tmp = df_tmp[df_tmp[field_category_py] != "none"].reset_index(drop = True)
        # set a key and prepare new fields
        key = field_category_py
        fields_to_dict = [x for x in df_tmp.columns if x != key]
        # next, create dict maps to add to the table
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"
            field_maps.update({field_fwd: sf.build_dict(df_tmp[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(df_tmp[fld])
            if (len(vals_unique) == len(df_tmp)):
                field_maps.update({field_rev: sf.build_dict(df_tmp[[fld, key]])})

        dict_attributes["abbreviation_subsector"].field_maps.update(field_maps)

        return (all_types, dict_attributes, dict_varreqs)
    
    
    ####################################################
    #    SECTOR-SPECIFIC AND CROSS SECTORIAL CHECKS    #
    ####################################################
    
    # LAND USE checks
    def check_land_use_tables(self):

        # specify some generic variables
        catstr_forest = self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"]["Forest"]
        catstr_landuse = self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"]["Land Use"]
        attribute_forest = self.dict_attributes[catstr_forest]
        attribute_landuse = self.dict_attributes[catstr_landuse]
        cats_forest = attribute_forest.key_values
        cats_landuse = attribute_landuse.key_values
        matchstr_forest = "forest_"
        
        ##  check that all forest categories are in land use and that all categories specified as forest are in the land use table
        set_cats_forest_in_land_use = set([matchstr_forest + x for x in cats_forest])
        set_land_use_forest_cats = set([x.replace(matchstr_forest, "") for x in cats_landuse if (matchstr_forest in x)])
        
        if not set_cats_forest_in_land_use.issubset(set(cats_landuse)):
            missing_vals = set_cats_forest_in_land_use - set(cats_landuse)
            missing_str = sf.format_print_list(missing_vals)
            raise KeyError(f"Missing key values in land use attribute file '{attribute_landuse.fp_table}': did not find land use categories {missing_str}.")
        elif not set_land_use_forest_cats.issubset(cats_forest):
            extra_vals = set_land_use_forest_cats - set(cats_forest)
            extra_vals = sf.format_print_list(extra_vals)
            raise KeyError(f"Undefined forest categories specified in land use attribute file '{attribute_landuse.fp_table}': did not find forest categories {extra_vals}.")
            
            
    
    ###############################
    #    VARIABLE REQUIREMENTS    #
    ###############################
        
    ##  function for bulding a basic variable list from the (no complexitiies)
    def build_varlist_basic(self, dict_vr_vvs, category, attribute_table):
        vars_out = []
        self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"]["Forest"]
        # loop over required variables
        for var in dict_vr_vvs.keys():
            var_schema = ma.clean_schema(dict_vr_vvs[var])
            # fix the emission
            for catval in attribute_table.key_values:
                vars_out.append(var_schema.replace(category, catval))
        return vars_out
    
    
            
        
# function for cleaning a variable schema
def clean_schema(var_schema):
    
    var_schema = var_schema.split("(")
    var_schema[0] = var_schema[0].replace("`", "").replace(" ", "")
    
    if len(var_schema) > 1:
        repls =  var_schema[1].replace("`", "").split(",")
        dict_repls = {}
        for dr in repls:
            dr0 = dr.replace(" ", "").replace(")", "").split("=")
            var_schema[0] = var_schema[0].replace(dr0[0], dr0[1])
        
    return var_schema[0]

