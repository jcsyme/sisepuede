"""
SISEPUEDE
    Copyright (C) 2020-2024 James Syme

    LICENSE HERE

"""


from typing import *
import attribute_table as at
import logging
import numpy as np
import os, os.path
import pandas as pd
import re
import support_functions as sf




class ModelVariable:
    """
    Build a ModelVariable object. The ModelVariable stores information about
        the variables, including categories, dimensions to the variable, units,
        and more. 

    The initialization element (`variable_init`) must have the following keys
        specified: 

        * name: the name of the variable, used to access the variable in 
            dictionaries etc.
        * grouping: an optional grouping used to (in SISEPUEDE, this is a 
            subsector)
        * categories: 



    Initialization Arguments
    ------------------------
    - variable_init: initializer for the variable. Can be:
        * dict: a dictionary mapping keys to a value
        * pd.DataFrame: a Pandas DataFrame whose first row is used to initialize 
            variable elements by smapping fields (key) to values
        * pd.Series: a Pandas Series used to map indicies (key) to values
    - category_definition: initializer for variable categories (i.e., mutable
        elements in the variable schema). Can be:
        * attribute_table.AttributeTable: Can be provided if there is only a
            single known mutable element
        * Dict[str, attribute_table.AttributeTable]: Can be provided if there
            are multiple mutable elements or a number of tables in a centralized
            location. 
            * NOTE: Dictionary keys are assumed to be the corresponding 
                AttributeTable.key
        * List[attribute_table.AttributeTable]: optional list of AttributeTable
            objects

    Optional Initialization Arguments
    ---------------------------------
    - attribute_as_property: try to initialize attributes as properties of the
        model variable itself? If true, sets an attribute passed in the variable
        schema as an accessible property of the ModelVariable.
        * E.g., if "unit_mass" is an attribute from the schema, then it is 
            accessible from ModelVariable.unit_mass if 
            attribute_as_property == True.
        * NOTE: this should be used with caution. If an attribute conflicts with
            an existing property, it will not be set. 

    - delim_categories: delimeter used to split categories specified in 
        variable_init
    - flag_all_cats: flag in category specification used to point to the use of 
        all category values to define the variable
    - flag_no_dims: flag in category specification used to point to the use of 
        no category values to define the variable
    - key_categories: key in `variable_init` storing category specifications
        NOTE: set to "all" to specify all categories for a given mutable element
    - key_default_value: key in `variable_init` storing the default value of the
        variable if it is not found. 
    - key_name: key in `variable_init` storing the name of the variable
    - key_schema: key in `variable_init` storing the schema specification
    - keys_additional: additional keys to include as properties of the 
        ModelVariable. If None, will store all fields (indcies) specified in 
        variable_init as properties
    - logger: optional context-dependent logger to pass
    - stop_without_default: if no default value is found, stop?  If not, 
        defaults to np.nan
    """
    
    def __init__(self,
        variable_init: Union[dict, pd.DataFrame, pd.Series],
        element_definition: Union[at.AttributeTable, Dict[str, at.AttributeTable], List[at.AttributeTable]],
        attribute_as_property: bool = True,
        delim_categories: str = "|",
        flag_all_cats: str = "all",
        flag_no_dims: str = "none",
        key_categories: str = "categories",
        key_default_value: str = "default_value",
        key_name: str = "variable",
        key_schema: str = "variable_schema",
        keys_additional: Union[List[str], Dict[str, str], None] = None,
        logger: Union[logging.Logger, None] = None,
        stop_without_default: bool = False,
    ):

        self.logger = logger
        self._initialize_keys(
            key_categories = key_categories,
            key_default_value = key_default_value,
            key_name = key_name,
            key_schema = key_schema,
        )

        self._initialize_properties(
            variable_init, 
            attribute_as_property = attribute_as_property,
            delim_categories = delim_categories,
            flag_all_cats = flag_all_cats,
            flag_no_dims = flag_no_dims,
            stop_without_default = stop_without_default,
        )
        self._initialize_categories(
            element_definition, 
        )

        self._initialize_fields()

        return None

    

    def __call__(self,
        obj: Union[pd.DataFrame, None] = None,
        **kwargs,
    ) -> Union[List, np.ndarray, pd.DataFrame, str, None]:
        """
        Return the full variable list; if a DataFrame, extract the variable from 
            the DataFrame
        """

        if obj is None:
            return self.name

        out = self.get(obj, **kwargs,)

        return out



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_categories(self,
        category_definition: Union[at.AttributeTable, Dict[str, at.AttributeTable], List[at.AttributeTable]],
    ) -> None:
        """
        Initialize categories based on input attributes. Sets the following 
            properties:

            * self.dict_category_keys:
                dictionary mapping a cleaned category to the set of key values
                that are valid for the variable
            * self.dict_category_key_indices:
                for a given key, values map the elements associated with 
                dict_category_keys to their indices in dict_category_keys_space
            * self.dict_category_keys_space:
                dictionary mapping a cleaned category to the set of all 
                potential key values that the category can take on.

        * NOTE: for multidimensional specifications of mutable elements with the 
            same root (e.g., X-DIM1 and X-DIM2), 
            self.dict_category_keys_space only stores the space associated with
            the root element (e.g., those associated with X), and each dimension
            is referred back to its root element. 

            HOWEVER, self.dict_category_keys stores keys associated with the 
            individual dimensional implementations to allow for differentiation.  
            Similarly, self.dict_category_key_indices does as well, mapping the
            indicies of the restricted space in each dimension to the full 
            space of the root element.


        Function Arguments
        ------------------
        - category_definition: initializer for variable categories (i.e., 
            mutable elements in the variable schema). Can be:
            * attribute_table.AttributeTable: Can be provided if there is only a
                single known mutable element
            * Dict[str, attribute_table.AttributeTable]: Can be provided if 
                there are multiple mutable elements or a number of tables in a 
                centralized location. 
                * NOTE: Dictionary keys are assumed to be the corresponding 
                    AttributeTable.key
            * List[attribute_table.AttributeTable]: optional list of 
                AttributeTable objects
        """

        # initialize some outputs (modified below if valid)
        self.dict_category_keys = None
        self.dict_category_key_indices = None
        self.dict_category_key_space = None

        # iterate and compare to those included in the schema
        category_definition = self.get_category_definition(category_definition, )
        dict_category_key_space = {}
        
        # iterate over mutable elements
        """
        for k, v in category_definition.items():
            
            key = self.schema.clean_element(k)
            root_key, dim_index = self.schema.get_root_element(key)

            # some conditions to skip
            continue_q = (root_key not in self.schema.mutable_elements_clean_ordered)
            continue_q |= not isinstance(v.key_values, list)
            if continue_q:
                continue

            # update the dictionaries
            dict_category_key_space.update({key: v.key_values})
        """

        for elem in self.schema.mutable_elements_clean_ordered:

            if elem in dict_category_key_space.keys():
                continue
            
            # get the root element as a space
            root_elem, dim_index = self.schema.get_root_element(elem)
            attr = category_definition.get(root_elem)

            continue_q = attr is None
            continue_q |= (not isinstance(attr.key_values, list)) if not continue_q else False
            if continue_q:
                continue
            
            dict_category_key_space.update({root_elem: attr.key_values})
            

        # get categories 
        dict_category_keys = self.get_categories_by_element(
            dict_category_space = dict_category_key_space,
        )
            
            
        ##  SET PROPERTIES

        self.dict_category_keys = dict_category_keys
        self.dict_category_key_space = dict_category_key_space

        return None



    def _initialize_fields(self,
        **kwargs,
    ) -> None:
        """
        Set fields for quick access. Sets the following properties:

            * self.categories_are_restricted:
                boolean to store whether or not category restrictions are in 
                place for the variable
            * self.fields:
                fields associated with the variable
            * self.fields_indices:
                indices of self.fields in self.fields_space
            * self.fields_space:
                universe of potential fields (if built under all categories)
            * self.fields_space_complement:
                complement in fields_space of fields
        """

        fields = self.build_fields()

        # trick it into building variables across entire range
        fields_space = self.build_fields(
            category_restrictions = {}, 
            category_restrictions_as_full_spec = True,
        ) 

        # some derivatives
        fields_space_complement = [x for x in fields_space if x not in fields]
        fields_index = [fields_space.index(x) for x in fields]
        categories_are_restricted = (len(fields_space_complement) > 0)


        ##  SET PROPERTIES
        
        self.categories_are_restricted = categories_are_restricted
        self.fields = fields
        self.fields_index = fields_index
        self.fields_space = fields_space
        self.fields_space_complement = fields_space_complement

        return None



    def _initialize_keys(self,
        **kwargs,
    ) -> None:
        """
        Set required keys. Sets the following properties:

            * self.key_categories
            * self.key_default_value
            * self.key_name
            * self.key_schema
            * self.keys_required

        """

        # check keyword arguments for specification
        key_categories = kwargs.get("key_categories", "categories")
        key_default_value = kwargs.get("key_default_value", "default_value")
        key_name = kwargs.get("key_name", "variable")
        key_schema = kwargs.get("key_schema", "variable_schema")

        # set required keys
        keys_required = [
            key_categories,
            key_name,
            key_schema
        ]

        ##  SET PROPERTIES

        self.key_categories = key_categories
        self.key_default_value = key_default_value
        self.key_name = key_name
        self.key_schema = key_schema
        self.keys_required = keys_required

        return None



    def _initialize_properties(self,
        variable_init: Union[dict, pd.DataFrame, pd.Series],
        attribute_as_property: bool = True,
        delim_categories: str = "|",
        flag_all_cats: str = "all",
        flag_no_dims: str = "none",
        stop_without_default: bool = False,
    ) -> None:
        """
        Set properties derived from attribute table. Sets properties associated 
            with self.keys_required in addition to others specified in the 
            attribute table.
        
        Sets the following properties:

            * self.default_value:
                Default value of the variable if not found in a dataframe
            * self.dict_varinfo:
                Dictionary of variable information 
            * self.container_elements:
                Container to use in self.schema to parse mutable (e.g., category
                specifications) and immutable (e.g., attributes) elements.
                Default is "$"
            * self.container_expressions:
                Container to use in self.schema to parse expressions
            * self.flag_all_cats:
                Flag to use in variable specification to note that the variable
                is associated with all values in the category dimension
            * self.flag_no_dims:
                Flag to use in variable specification to note that the variable
                is not associated with any categories (0 dimension)
            * self.is_model_variable:
                Attribute used to identify as a model variable object
            * self.name:
                Name of the variable (passed in var_init via self.key_name)
            * self.name_clean:
                Name of the variable that is cleaned to remove LaTeX, RST, and
                markdown special characters
            * self.name_fs_safe:
                Name of the variable that excludes spaces and sets all 
                characters to lower case
            * self.regex_expression_bounds:
                Regular expression used to parse expressions (e.g., mutable 
                element dictionaries) from initialization strings
            * self.schema:
                VariableSchema object 
            * self.space_char:
                Character used to denote a space in a mutable element


        Function Arguments
        ------------------
        - variable_init: initializer for the variable. Can be:
            * dict: a dictionary mapping keys to a value
            * pd.DataFrame: a Pandas DataFrame whose first row is used to 
                initialize variable elements by mapping fields (key) to values
            * pd.Series: a Pandas Series used to map indicies (key) to values
        
        Keyword Arguments
        -----------------
        - attribute_as_property: try to initialize attributes as properties of 
            the model variable itself? If true, sets an attribute passed in the 
            variable schema as an accessible property of the ModelVariable.
            * E.g., if "unit_mass" is an attribute from the schema, then it is 
                accessible from ModelVariable.unit_mass if 
                attribute_as_property == True.
            * NOTE: this should be used with caution. If an attribute conflicts 
                with an existing property, it will not be set. 
        - delim_categories: delimeter used to split categories specified in 
            variable_init
        - flag_all_cats: flag in category specification used to point to the use
            of all category values to define the variable
        - stop_without_default: if no default value is found, stop?  If not, 
            defaults to np.nan
        """
        
        # get the variable information dictionary
        dict_varinfo = self.get_variable_init_dictionary(variable_init)
        name = dict_varinfo.get(self.key_name)
        name_clean = self.get_clean_name(name)
        name_fs_safe = self.get_clean_name(name, fs_safe = True)

        # set the schema
        try:
            schema = VariableSchema(dict_varinfo.get(self.key_schema))
        except Exception as e:
            msg = f"Error initializing variable schema: {e}"
            self._log(msg, type_log = "error")
            raise RuntimeError(msg)

        # try to set some properties
        if attribute_as_property:
            try:

                dict_attributes = dict(
                    (schema.dict_attribute_keys_to_attribute_keys_clean.get(k), v)
                    for k, v in schema.dict_attributes.items()
                )
                errors, num_errors, warns = sf.set_properties_from_dict(self, dict_attributes, )

                # pass errors
                errors = (":" + ("\n\t".join(errors))) if isinstance(errors, list) else "."
                msg = f"Properties for variable {name} successfully set with {num_error} errors{errors}."
                self._log(msg, type_log = "error")

                # pass warnings
                warns = ("\n\t".join(warns)) if isinstance(warns, list) else ""
                self._log(warns, type_log = "warning") if (warns != "") else None


            except Exception as e:
                msg = f"Error trying to initialize properties of variable {name} from dict_varinfo: {e}"
                self._log(msg, type_log = "warning") 

        # build the expression-bounding regular expression
        regex_expression_bounds = re.compile(f"(?<={schema.container_expressions})(.*?)(?={schema.container_expressions})")
        
        # get the default value
        default_value = self.get_default_value(
            dict_varinfo,
            stop_without_default = stop_without_default,
        )

        
        ##  SET PROPERTIES

        self.container_elements = schema.container_elements
        self.container_expressions = schema.container_expressions
        self.default_value = default_value
        self.delim_categories = delim_categories
        self.dict_varinfo = dict_varinfo
        self.flag_all_cats = flag_all_cats
        self.flag_no_dims = flag_no_dims
        self.is_model_variable = True
        self.name = name
        self.name_clean = name_clean
        self.name_fs_safe = name_fs_safe
        self.regex_expression_bounds = regex_expression_bounds
        self.schema = schema
        self.space_char = schema.space_char

        return None



    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger.
            See ?sf._optional_log for more information.

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)

        return None

    
    ############################
    #    NON-INIT FUNCTIONS    #
    ############################

    def build_init_me_to_cat_dictionary(self,
        dict_category_space: Union[Dict[str, List[str]], None],
        return_on_none: Union[Any, None] = None,
    ) -> Dict:
        """
        Initialize the dictionary mapping the variable schema's mutable elements 
            to categories in the variable field. Differs from 
            self.dict_category_key_space in that the keys are schema mutable 
            elements, not root elements.

        Support function for self.get_categories_by_element()

        
        Function Arguments
        ------------------
        - dict_category_space: optional dictionary specifying the space of 
            possible. If None, no checks are performed

        Keyword Arguments
        -----------------
        - return_on_none: element to return if dict_category_space is None
        """

        dict_out = (
            dict(
                (x, dict_category_space.get(self.schema.get_root_element(x)[0]))
                for x in self.schema.mutable_elements_clean_ordered
            )
            if isinstance(dict_category_space, dict)
            else return_on_none
        )

        return dict_out



    def get_categories_by_element(self,
        category_subspace: Union[Dict[str, str], str, None] = None,
        dict_category_space: Union[Dict[str, List[str]], None] = None,
    ) -> Union[Dict[str, Union[List[str], None]], None]:
        """
        Convert categories specified in an attribute table into a dictionary
            where keys are cleaned mutable elements of the variable schema and
            values are lists of category restrictions for that element. 
        
        NOTE: if category_subspace is specified as a single string, it will be
            passed to EACH mutable element stored in 
            self.schema.mutable_elements_clean_ordered


        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - category_subspace: either a single string of delimited categories or
            a dictionary mapping a mutable element to a delimited string. 
            * If None, uses those specified in self.dict_varinfo (in 
            self.key_categories)
        - dict_category_space: optional dictionary specifying the space of 
            possible. If None, no checks are performed
        """
        # check specification
        proceed = isinstance(category_subspace, dict)
        proceed |= isinstance(category_subspace, str)
        proceed |= category_subspace is None
        if not proceed:
            return None
        
        # get the subspace(s) of categories
        category_subspace = (
            self.dict_varinfo.get(self.key_categories)
            if category_subspace is None
            else category_subspace
        )


        ##  CHECK SPECIFICATION OF category_subspace AND CONVERT TO DICTIONARY

        if isinstance(category_subspace, str):

            # in this case, the variable is defined for all categories associated with each dimension
            if category_subspace.lower() == self.flag_all_cats:
                dict_out = self.build_init_me_to_cat_dictionary(dict_category_space)
                return dict_out

            # in this case, the variable is intentionally associated with no categories
            if category_subspace.lower() == self.flag_no_dims:
                return {}
            
            # try to convert to a dictionary
            category_subspace = self.parse_categories_string(category_subspace)
            if isinstance(category_subspace, str):
                # otherwise, assume the string is intended to restrict the space for all mutable elements
                category_subspace = dict(
                    (elem, category_subspace)
                    for elem in self.schema.mutable_elements_clean_ordered
                )
           

        ##  BUILD OUTPUT DICTIONARY MAPPING CLEANED MUTABLE ELEMENTS TO LISTS OF CATEGORIES

        # initialize the space as a list if defined that way; otherwise, each iteration will check for list elements
        space = (
            dict_category_space 
            if sf.islistlike(dict_category_space) 
            else None
        )
        flags_skip =  [
            self.flag_all_cats, 
            f"{self.container_expressions}{self.flag_all_cats}{self.container_expressions}"
        ]

        # initialize the dictionary out
        dict_out = self.build_init_me_to_cat_dictionary(
            dict_category_space,
            return_on_none = {},
        )


        ##  ITERATE

        # set ordering for iteration; put root elements first (if they are specified)
        elements_root = []
        elements_children = []
        for elem in category_subspace.keys():
            elems_children = self.schema.dict_root_element_to_children.get(elem)
            (
                elements_root.append(elem)
                if elems_children is not None
                else elements_children.append(elem)
            )
        elements_iter = elements_root + elements_children

        # iterate over each of the mutable element categories to get specified category subsets
        for elem in elements_iter:
            
            v = category_subspace.get(elem)
            children = self.schema.dict_root_element_to_children.get(elem, [elem])

            """
            since elements_iter applies roots first, if categories for roots are
                specified in the presence of multiple dimensions, it will 
                initialize those restrictions for the children. If children are
                also specified, then they will overwrite the root's 
                specification.

                E.g., X -> R will set X-DIM1, X-DIM2, and X-DIM3 to R; then, 
                additionally specifying X-DIM2 as s will overwrite X-DIM2 (only)
                as S.
            """;
            for k in children:
                # check the space to ensure it's defined
                if isinstance(dict_category_space, dict):
                    key_space = self.schema.get_root_element(k)
                    space = dict_category_space.get(key_space[0], "__skip")
                    if space == "__skip":
                        continue
                
                # if defined as all, set to the space
                categories = (
                    space
                    if (v in flags_skip)
                    else self.get_categories_from_specification(v)
                )
                if not isinstance(categories, list):
                    continue
                    
            
                categories = (
                    [x for x in space if x in categories]
                    if sf.islistlike(space)
                    else categories
                )
                
                dict_out.update({k: categories})
            

        return dict_out



    def get_category_definition(self,
        category_definition: Union[at.AttributeTable, Dict[str, at.AttributeTable], List[at.AttributeTable]],
    ) -> Union[Dict[str, at.AttributeTable], None]:
   
        """
        Read in category_definition and convert to dictionary for use in 
            _initialize_categories()


        Function Arguments
        ------------------
        - category_definition: initializer for variable categories (i.e., 
            mutable elements in the variable schema). Can be:
            * attribute_table.AttributeTable: Can be provided if there is only a
                single known mutable element
            * Dict[str, attribute_table.AttributeTable]: Can be provided if 
                there are multiple mutable elements or a number of tables in a 
                centralized location. 
                * NOTE: Dictionary keys are assumed to be the corresponding 
                    AttributeTable.key
            * List[attribute_table.AttributeTable]: optional list of 
                AttributeTable objects
        """
        # initialize
        category_definition_out = {}

        # check the specification of the element definition and turn into a dictionary for iteration
        if isinstance(category_definition, at.AttributeTable):
            category_definition_out = {category_definition.key: category_definition}

        elif sf.islistlike(category_definition):
            category_definition_out = dict((x.key, x) for x in category_definition if hasattr(x, "key"))

        elif isinstance(category_definition, dict):
            category_definition_out = dict(
                (v.key, v) for v in category_definition.values() 
                if hasattr(v, "key") & hasattr(v, "key_values")
            )


        return category_definition_out
    


    def get_categories_from_specification(self,
        category_str: Union[List[str], str],
    ) -> Union[List[str], None]:
        """
        Split categories defined in a string into a list. Performs checks on
            specification to ensure that elements are cleaned. 
        """
        
        # some cases--return a list
        if sf.islistlike(category_str):
            return list(category_str)

        if not isinstance(category_str, str):
            return None


        # MECHANISM FOR DICTIONARY EXTRACTION!
        
        categories = self.regex_expression_bounds.findall(category_str)
        if len(categories) == 0:
            return None
        
        if self.delim_categories in categories[0]:
            categories = categories[0].split(self.delim_categories)
        else:    
            categories = category_str.split(self.delim_categories)
            categories = [
                x.replace(self.container_expressions, "")
                for x in categories 
                if x.startswith(self.container_expressions) & x.endswith(self.container_expressions)
            ]

        return categories
    


    def get_clean_name(self,
        name: str,
        fs_safe: bool = False,
    ) -> str:
        """
        Return the cleaned version of the name, which excludes LaTeX, Markdown,
            and RST special characters that are part of the full name. Set 
            fs_safe = True to generate file system safe version (no spaces, all 
            lower case)
        """
        chars_repl = [":math:\\text", "{", "}_", "}"]

        name_clean = sf.str_replace(name, dict((x, "") for x in chars_repl))
        if fs_safe:
            name_clean = (
                name_clean
                .lower()
                .strip()
                .replace(" ", "_")
            )

        return name_clean



    def get_default_value(self,
        dict_varinfo: dict,
        default_if_missing: Any = np.nan,
        stop_without_default: bool = False,
    ) -> None:
        """
        Get the default value based on the initialization dictionary. 


        Function Arguments
        ------------------
        - dict_varinfo: dictionary containing key self.key_default_value
        
        Keyword Arguments
        -----------------
        - default_if_missing: default value if not found
        - stop_without_default: if no default value is found, stop?  If not, 
            defaults to np.nan
        """

        if not isinstance(dict_varinfo, dict):
            return None

        out = dict_varinfo.get(self.key_default_value)

        if out is None:
            if stop_without_default:
                msg = f"""
                Error instantiating variable: key {self.key_default_value}' 
                missing from var_init; no default value can be set.
                """
                raise KeyError(msg)
            
            out = default_if_missing

        return out
    


    def get_property(self,
        prop: str,
    ) -> Any:
        """
        Try to retrieve the variable prop--associated with dict_varinfo--
            for the variable.
        """
        out = self.dict_varinfo.get(prop)
        return out



    def get_variable_init_dictionary(self,
        variable_init: Union[dict, pd.DataFrame, pd.Series],
    ) -> dict:
        """
        Verify the variable initialization dictionary, pd.Series, or 
            pd.DataFrame and convert to a dictionary.

        Function Arguments
        ------------------
        - variable_init: initializer for the variable. Can be:
            * dict: a dictionary mapping keys to a value
            * pd.DataFrame: a Pandas DataFrame whose first row is used to 
                initialize variable elements by smapping fields (key) to values
            * pd.Series: a Pandas Series used to map indicies (key) to values
        """

        # check type
        invalid_type_q = not isinstance(variable_init, dict)
        invalid_type_q &= not isinstance(variable_init, pd.DataFrame)
        invalid_type_q &= not isinstance(variable_init, pd.Series)

        if invalid_type_q:
            tp = str(type(variable_init))
            msg = f"Error initializing ModelVariable: invalid type {tp} for variable_init"
            raise RuntimeError(msg)

        
        # convert type
        if isinstance(variable_init, pd.DataFrame):
            variable_init = variable_init.iloc[0].to_dict()
        
        if isinstance(variable_init, pd.Series):
            variable_init = variable_init.to_dict()
        
        # next, check keys
        s_req = set(self.keys_required)
        s_avail = set(variable_init.keys())
        if not s_req.issubset(s_avail):
            fields_missing = sf.format_print_list(s_req - s_avail)
            raise KeyError(f"Error initializing ModelVariable: keys {fields_missing} not found in variable_init")


        return variable_init
    


    def parse_categories_string(self,
        categories_string: str,
        assignment_dict: str = "=",
        delim_dict: str = ",",
        endpoints_dict: Tuple[str] = ("(", ")"),
    ) -> Union[Dict[str, str], str, None]:
        """
        Read input categories_string and parse into a string or dictionary 
            mapping mutable elements to the applicable category string. 

        If categories_string is specified as a dictionary, then 
            ModelVariable.parse_categories_string() returns a dictionary. If it
            is entered only as a string, then the categories string is returned.

        Function Arguments
        ------------------
        - categories_string: categories specification string from attribute 
            table

        Keyword Arguments
        -----------------
        - assignment_dict: assignment string in the dictionary that maps
            keys to values 
        - delim_dict: delimiter in dictionaries splitting key value pairs 
        - endpoints_dict
        """

        # check input
        return_none = not isinstance(categories_string, str)
        return_none |= not isinstance(endpoints_dict, tuple)
        return_none = (len(endpoints_dict) < 1) if not return_none else return_none
        if return_none:
            return None

        # drop endpoints and split
        if any([x in categories_string for x in endpoints_dict]):
            categories_string = categories_string.split(endpoints_dict[0])
            if len(categories_string) == 1:
                return None
            
            categories_string = categories_string[1].split(endpoints_dict[1])
            if len(categories_string) == 1:
                return None

            categories_string = categories_string[0]
        
        categories_string = categories_string.split(delim_dict)

        # 
        as_dict = False
        out = []

        # first, check if it should be read as a dictionary
        for cat in categories_string:
            
            get_exp = self.regex_expression_bounds.findall(cat)
            if len(get_exp) == 0:
                continue
            
            is_key_pair = assignment_dict in get_exp[0]
            as_dict |= is_key_pair

            if not as_dict:
                out.append(cat)
                continue

            if not is_key_pair:
                continue

            cat_new = get_exp[0].split(assignment_dict)
            cat_new = cat_new[0:2] if (len(cat_new) > 2) else cat_new
            """
            If the second element of cat_new is emtpy, don't add it to the 
                dictionary; in practice, this means that ALL elements
                associated with that mutable element will be included when
                building the field list (checks for null string).
            """;
            if len(cat_new[1]) == 0:
                continue
            
            key = self.schema.clean_element(cat_new[0].strip())
            value = cat_new[1].strip()
            # need to add the container so that self.get_categories_from_specification() will find the categories
            value = f"{self.container_expressions}{value}{self.container_expressions}"

            out.append((key, value))
        
        # check out 
        if len(out) == 0:
            return None

        out = (
            dict(x for x in out if isinstance(x, tuple)) 
            if as_dict 
            else out[0]
        )

        return out
    


    def replace_mutable_element(self,
        schema: str,
        elem: str,
        value: Any,
    ) -> str:
        """
        In schema `schema`, replace the mutable elemenet `elem` with value 
            `value`
        """
        
        # get the element
        elem = self.schema.dict_mutable_elements_clean_to_original.get(elem, elem)
        elem = f"{self.container_elements}{elem}{self.container_elements}"
        
        out = schema.replace(elem, str(value))
        
        return out
    



    ############################
    #    CORE FUNCTIONALITY    #
    ############################
    
    def attribute(self,
        attr: str,
        return_on_none: Union[Any, None] = None,
    ) -> Any:
        """
        Retrieve an attribute attr from the model variable. If not found,
            returns value return_on_none.
        """
        out = self.schema.get_attribute(
            attr, 
            return_on_none = return_on_none,
        )

        return out



    def build_fields(self,
        category_restrictions: Union[dict, None] = None,
        category_restrictions_as_full_spec: bool = False,
    ) -> Union[List[str], None]:
        """
        Build fields associated with the variable.
        
        Function Argumnents
        -------------------
        
        Keyword Argumnents
        ------------------
        - category_restrictions: optional dictionary to overwrite 
            `self.dict_category_keys` with; i.e., keys in category_restrictions 
            will overwrite those in `self.dict_category_keys` IF 
            `category_restrictions_as_full_spec == False`. 
            
            * NOTE: if `category_restrictions_as_full_spec == True`, then 
                category_restrictions is treated as the initialization 
                dictionary
                
        - category_restrictions_as_full_spec: Set to True to treat 
            `category_restrictions` as the full specification dictionary
        """
        
        # INITIALIZATION
        
        # basic
        dims = self.schema.mutable_elements_clean_ordered.copy()
        fields = [self.schema.schema]
        schema = self.schema.schema
        
        # keys to keep from the dictionary (if applicable)
        keys_keep = (
            list(category_restrictions.keys())
            if isinstance(category_restrictions, dict)
            else None
        )
        
        # run through filtering and split elements
        category_restrictions = self.get_categories_by_element(
            category_restrictions,
            dict_category_space = self.dict_category_key_space,
        )
        if not isinstance(category_restrictions, dict):
            category_restrictions = {}
        
        # if full specification, then get_categories_by_element() will  
        # fill out keys as though the dictionaries are being initialized
        if (not category_restrictions_as_full_spec) and isinstance(keys_keep, list):
            category_restrictions = dict(
                (k, v) for k, v in category_restrictions.items()
                if k in keys_keep
            )

        ##  ITERATE 
        
        for dim in dims:
            
            # try the input dictionary, return internal restrictions if not defined
            restrictions = category_restrictions.get(
                dim,
                self.dict_category_keys.get(dim)
            )

            fields_new = []
            for field_elem in fields:
                for cat in restrictions:
                    # field_elem.replace(dim, cat)
                    field = self.replace_mutable_element(field_elem, dim, cat)
                    fields_new.append(field)
                    
            fields = fields_new
            
        return fields

    

    def get(self,
        obj: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        expand_to_all_categories: bool = False,
        extraction_logic: str = "all",
        fill_value: Any = None,
        **kwargs,
    ) -> Union[List[Any], np.ndarray, pd.DataFrame, None]:
        """
        Retrieve the variable from an input object ordered by self.fields.

        Function Arguments
        ------------------
        - obj: the input object to retrieve the variable from

        Keyword Arguments
        -----------------
        - expand_to_all_categories: extract and expand output to all categories? 
        - extraction_logic: set logic used on extraction
            * "all": throws an error if any field in self.fields is missing
            * "any": extracts any field in self.fields available in `obj`
                and fills any missing values with fill_value (or default value)
        - fill_value: if `expand_to_all_categories == True` OR 
            `extract_any == True`, missing categories will be filled with this
            value (cannot be None). 
            * If None, reverts to self.default_value
        - **kwargs: additional method specific keyword arguments. 
            * If obj is a pd.DataFrame:
                - return_type: one of the following values:
                    * "data_frame": return the subset data frame that includes 
                        the variable
                    * "array": return a numpy array
        """

        out = None

        if isinstance(obj, pd.DataFrame):

            args, kwpass = sf.get_args(self.get_from_dataframe)
            kwpass = dict((k, v) for k, v in kwargs.items() if k in kwpass)

            out = self.get_from_dataframe(
                obj, 
                expand_to_all_categories = expand_to_all_categories,
                extraction_logic = extraction_logic,
                fill_value = fill_value,
                **kwpass
            )

        return out
    


    def get_from_dataframe(self,
        df: pd.DataFrame,
        expand_to_all_categories: bool = False,
        extraction_logic: str = "all",
        fill_value: Any = None,
        return_type: str = "data_frame",
    ) -> Union[List[Any], np.ndarray, pd.DataFrame, None]:
        """
        Retrieve the variable from a pandas DataFrame

        Function Arguments
        ------------------
        - df: the input DataFrame to retrieve the variable from

        Keyword Arguments
        -----------------
        - expand_to_all_categories: extract and expand output to all categories? 
        - extraction_logic: set logic used on extraction
            * "all": throws an error if any field in self.fields is missing
            * "any": extracts any field in self.fields available in `obj`
                and fills any missing values with fill_value (or default value)
        - fill_value: if `expand_to_all_categories == True` OR 
            `extract_any == True`, missing categories will be filled with this
            value. 
            * If None, reverts to self.default_value
        - return_type: one of the following values:
            * "data_frame": return the subset data frame that includes the 
                variable
            * "array": return a numpy array
        """
        
        ##  INITIALIZATION

        # verify input types 
        if not isinstance(df, pd.DataFrame):
            return None

        # check logic specification
        extraction_logic = (
            "all"
            if extraction_logic not in ["all", "any"]
            else extraction_logic
        )

        # check fill value
        fill_value = (
            self.default_value
            if fill_value is None
            else fill_value
        )

        # check return type specification
        return_type = (
            "data_frame"
            if return_type not in ["data_frame", "array"]
            else return_type
        )

        # check fields that can be extracted
        fields_ext = (
            self.fields
            if extraction_logic == "all"
            else [x for x in self.fields if x in df.columns]
        )
        if len(fields_ext) == 0:
            return None
        
        
        ##  MAIN OPERATIONS

        try:
            df_out = df[fields_ext]

        except Exception as e:

            fields_missing = sf.print_setdiff(
                fields_ext, 
                [x for x in fields_ext if x not in df.columns]
            )

            msg = f"Error trying to retrieve variable {self.name}: fields {fields_missing} not found."
            self._log(msg, type_log = "error")

            return None


        # expand to all categories?
        if expand_to_all_categories & self.categories_are_restricted:

            n_complement = len(self.fields_space_complement)
            df_concat = pd.DataFrame(
                np.full((df_out.shape[0], n_complement), fill_value),
                columns = self.fields_space_complement
            )

            df_out = pd.concat(
                [
                    df_out,
                    df_concat
                ],
                axis = 1
            )

            df_out = df_out[self.fields_space]
            print(df_out.shape)

        df_out.reset_index(drop = True, inplace = True)
        df_out = df_out.to_numpy() if (return_type == "array") else df_out

        return df_out







class VariableSchema:
    """
    Create a variable schema class that stores information on the base schema

    Initialization Arguments
    ------------------------
    - schema_raw: initialization schema
    
    Optional Arguments
    ------------------
    - container_elements: string used to delimit elements--such as a category, 
        unit, or gas--within a schema
    - container_expressions: substring used to parse out schema and associated
            elements
    - flag_dim: protected string that allows users to pass the same mutable
        element as multiple dimensions (e.g., field_$CAT-X-DIM1$_$CAT-X-DIM2$
        passes the element X twice) and giving the outer product of the space 
    - space_char: character used to replace spaces
    """

    def __init__(self,
        schema_raw: str,
        container_elements: str = "$",
        container_expressions: str = "``",
        flag_dim: str = "DIM",
        space_char: str = "-",
    ) -> None:

        self._initialize_properties(
            container_elements = container_elements,
            container_expressions = container_expressions,
            flag_dim = flag_dim,
            space_char = space_char,
        )

        self._initialize_schema(
            schema_raw,
        )




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_properties(self,
        container_elements: str = "$",
        container_expressions: str = "``",
        flag_dim: str = "DIM",
        space_char: str = "-",
    ) -> None:
        """
        Initialize some key properties of the schema. Sets the following 
            properties:

            * self.container_elements:
                container used to identify elements
            * self.container_expressions:
                container used to identy expressions in the schema
            * self.regex_elements (regular expression used to identify elements
                in the schema)
            * self.space_char
    

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - container_elements: string used to delimit elements--such as a 
            category, unit, or gas--within a schema
        - container_expressions: substring used to parse out schema and 
            associated elements
        - flag_dim: protected string that allows users to pass the same mutable
            element as multiple dimensions (e.g., 
            field_$CAT-X-DIM1$_$CAT-X-DIM2$ passes the element X twice) and 
            giving the outer product of the space 
        - space_char: character used to replace spaces
        """

        # check some specifications
        return_error = not isinstance(container_elements, str)
        return_error |= not isinstance(flag_dim, str)
        return_error |= not isinstance(container_expressions, str)
        return_error |= not isinstance(space_char, str)
        if return_error:
            msg = f"""
            Error initializing variable schema: invalid type detected for 
            container_elements, container_expressions, and/or space_char. Check 
            that all elements are a string.
            """
            raise RuntimeError(msg)


        # build the element-wise regular expression
        delim_regex = (
            f"\{container_elements}" 
            if container_elements in ["$", "[", "]"]
            else container_elements
        )
        regex_elements = re.compile(f"(?<={delim_regex})(.*?)(?={delim_regex})")

        # build the regular expression for identifying mutable elements with multiple dimensions
        regex_component = f"{space_char}{flag_dim}"
        regex_component_clean = clean_element(
            regex_component,
            container_elements = container_elements,
            container_expressions = container_expressions,
            space_char = space_char,
        )
        regex_dims = re.compile(f".*{regex_component}(\d+)$")
        regex_dims_clean = re.compile(f".*{regex_component_clean}(\d+)$")


        ##  SET PROPERTIES

        self.container_elements = container_elements
        self.container_expressions = container_expressions
        self.flag_dim = flag_dim
        self.regex_dims = regex_dims
        self.regex_dims_clean = regex_dims_clean
        self.regex_elements = regex_elements
        self.space_char = space_char

        return None
    


    def _initialize_schema(self,
        schema_raw: str,
    ) -> None:
        """
        Initialize some key properties of the schema. Sets the following 
            properties:

            * self.attributes: 
                list of attributes (keys) that can be accessed
            * self.dict_attribute_keys_clean_to_attribute_keys: 
                dictionary mapping attribute keys from the schema to cleaned 
                versions of the keys
            * self.dict_attribute_keys_to_attribute_keys_clean: 
                dictionary mapping cleaned versions of attribute keys to the 
                original version of the keys
            * self.dict_attributes: 
                dictionary mapping original attribute (not cleaned) to 
                specification
            * dict_mutable_elements_clean_to_original:
                dictionary mapping cleaned mutable elements to originals
            * dict_mutable_elements_original_to_clean: 
                dictionary mapping original mutable elements to cleaned elements
            * self.mutable_elements_clean_ordered:
                ordered list of elements--cleaned--that can be replaced in the 
                schema
            * self.mutable_elements_ordered: 
                ordered list of elements that can be replaced in the schema 
            * self.schema: schema to use that includes mutable elements
            * self.schema_raw: initial schema that includes elements set as
                attributes using the specification


        Function Arguments
        ------------------
        - schema_raw: raw input schema string to intialize

        Keyword Arguments
        -----------------
        """

        # verify input type
        sf._check_type(
            schema_raw,
            str, 
            prependage = "Error in schema_raw initializing variable schema: "
        )
        # replace any paired delimiters with null
        schema_raw = schema_raw.replace(self.container_elements*2, "")


        # check the elements in the input string
        n_delims = schema_raw.count(self.container_elements)
        if n_delims%2 != 0:
            msg = f"Error initializing variable schema from {schema_raw}: invalid number of element delimiters {n_delims}--it must be even."
            raise RuntimeError(msf)


        # decompose the schema
        dict_replacements, schema = decompose_schema(
            schema_raw,
            container_expressions = self.container_expressions,
            container_elements = self.container_elements,
            return_type = "all",
            space_char = self.space_char,
        )

        # map keys to clean keys/vis versa
        dict_attribute_keys_to_attribute_keys_clean = dict(
            (x, self.clean_element(x)) for x in dict_replacements.keys()
        )
        dict_attribute_keys_clean_to_attribute_keys = dict(
            (v, k) for k, v in dict_attribute_keys_to_attribute_keys_clean.items()
        )

        attributes_all = sorted(list(dict_attribute_keys_to_attribute_keys_clean.keys()))
        
        # next, get mutable elements
        mutable_elements_ordered = self.get_mutable_elements(schema)
        mutable_elements_clean_ordered = self.get_mutable_elements(schema, clean = True, )
        dict_mutable_elements_original_to_clean = dict(
            zip(
                mutable_elements_ordered,
                mutable_elements_clean_ordered
            )
        )
        dict_mutable_elements_clean_to_original = sf.reverse_dict(dict_mutable_elements_original_to_clean)
        
        # finally, get dictionary mapping root element to all its childred
        dict_root_element_to_children = self.get_child_elements(
            dict_melems_to_melems_clean = dict_mutable_elements_original_to_clean,
            mutable_elements_ordered = mutable_elements_ordered,
        )


        ##  SET PROPERTIES

        self.attributes = attributes_all
        self.dict_attribute_keys_clean_to_attribute_keys = dict_attribute_keys_clean_to_attribute_keys
        self.dict_attribute_keys_to_attribute_keys_clean = dict_attribute_keys_to_attribute_keys_clean
        self.dict_attributes = dict_replacements
        self.dict_mutable_elements_clean_to_original = dict_mutable_elements_clean_to_original
        self.dict_mutable_elements_original_to_clean = dict_mutable_elements_original_to_clean
        self.dict_root_element_to_children = dict_root_element_to_children
        self.mutable_elements_ordered = mutable_elements_ordered
        self.mutable_elements_clean_ordered = mutable_elements_clean_ordered
        self.schema = schema
        self.schema_raw = schema_raw

        return None
    


    def get_mutable_elements(self,
        schema: str,
        clean: bool = False,
    ) -> Union[List[str], None]:
        """
        Retrieve mutable elements (e.g., categories) from schema. Returns an
            ordered list of elements (ordered by appearance) or None if no 
            mutable elements are found. 

        Function Arguments
        ------------------
        - schema: schema (with attributes filled) to extract elements from

        Keyword Arguments
        -----------------
        - clean: clean the output elements?
        """

        list_elements = self.regex_elements.findall(schema)
        list_elements = [
            x for i, x in enumerate(list_elements) if i%2 == 0
        ]

        if clean:
            list_elements = [
                clean_element(
                    x,
                    container_elements = self.container_elements,
                    container_expressions = self.container_expressions,
                    space_char = self.space_char,
                ) 
                for x in list_elements
            ]

        return list_elements
    


    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def clean_element(self,
        element: str, 
    ) -> Union[str, None]:
        """
        Clean a variable schema element to a manageable, shared name that excludes
            special characters.

        Function Arguments
        ------------------
        - element: element to clean

        Keyword Arguments
        -----------------
        """

        out = clean_element(
            element,
            container_elements = self.container_elements,
            container_expressions = self.container_expressions,
            space_char = self.space_char,
        )

        return out



    def get_attribute(self,
        key: str,
        return_on_none: Union[Any, None] = None,
    ) -> Union[str, None]:
        """
        Retrieve an attribute of a variable schema associated with key. If none
            is found, returns None. 
            
        NOTE: Looks for the key in two stages. It starts by checking if the key
            is cleaned; if not found, it defaults to the original element and
            tries to retrieve the attribute.

        Function Arguments
        ------------------
        - key: attribute to retrieve

        Keyword Arguments
        -----------------
        - return_on_none: value to return if no key is found
        """
        # if in the clean dictionary, assume that it is the cleaned version of the key
        key = self.dict_attribute_keys_clean_to_attribute_keys.get(key, key) 
        out = self.dict_attributes.get(key, return_on_none)

        return out
    


    def get_child_elements(self,
        dict_melems_to_melems_clean: Union[Dict[str, str], None] = None,
        mutable_elements_ordered: Union[Dict[str, str], None] = None,
    ) -> Dict:
        """
        Build a dictionary to map root elements to all child elements; maps
            clean and unclean elements to the same dictionary. 

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - dict_melems_to_melems_clean: dictionary mapping mutable elements to 
            their cleaned version
        - mutable_elements_ordered: list of ordered (by replacement hierarchy) 
            mutable elements 
        """
        
        ##  INITIALIZE

        dict_melems_to_melems_clean = (
            self.dict_mutable_elements_original_to_clean
            if not isinstance(dict_melems_to_melems_clean, dict)
            else dict_melems_to_melems_clean
        )

        mutable_elements_ordered = (
            self.mutable_elements_ordered
            if not sf.islistlike(mutable_elements_ordered)
            else mutable_elements_ordered
        )


        dict_out = {}
        
        # loop over elements to add both origina & clean
        for elem in mutable_elements_ordered:
            elem_root = self.get_root_element(elem)[0]  
            (
                dict_out.update({elem_root: [elem]})
                if elem_root not in dict_out.keys()
                else dict_out[elem_root].append(elem)
            )
            
            # add clean
            elem_clean = dict_melems_to_melems_clean.get(elem)
            if elem_clean is None:
                continue

            elem_clean_root = self.get_root_element(elem_clean)[0]
            (
                dict_out.update({elem_clean_root: [elem_clean]})
                if elem_clean_root not in dict_out.keys()
                else dict_out[elem_clean_root].append(elem_clean)
            )


        return dict_out
    


    def get_mutable_element(self,
        field: str,
        element: str,
        return_on_none: Union[Any, None] = None,
        return_regex: bool = False,
    ) -> Union[str, None]:
        """
        Retrieve the value of a mutable element 
            
        NOTE: Looks for the element in two stages. It starts by checking if the 
            element is cleaned; if not found, it defaults to the original
            specification.

        Function Arguments
        ------------------
        - field: field to try to retrieve the element from
        - element: element to retrieve

        Keyword Arguments
        -----------------
        - return_on_none: value to return if no key is found
        - return_regex: return the regular expression used to match?
        """

        # try getting the element from the clean dictionary; if not present, assume original form
        element = self.dict_mutable_elements_clean_to_original.get(element, element)
        element = (
            f"{self.container_elements}{element}{self.container_elements}"
            if self.container_elements not in element
            else element
        )

        # build a regular expression to match on
        regex = self.schema.replace(element, "(.*)")
        regex = re.compile(f"{regex}$")
        if return_regex:
            return regex

        # 
        out = regex.match(field)
        out = return_on_none if (out is None) else out.groups()[0]

        return out
    


    def get_root_element(self,
        elem: str,
    ) -> Union[Tuple, None]:
        """
        For a mutable element `elem`, get the root element--i.e., stripped of 
            dimensional specification.
            
            E.g., maps 
                "XXXX-DIM1" -> "XXXX"  AND  "XXXX" -> "XXXX"
            (with DIM being self.flag_dim)
            
        NOTE: Returns a tuple of element (element, index) where `index` is the 
            dimensional index of the root element. If the input element is
            equivalent to the root, then the index is -1. Otherwise, the index
            is as specified.

            If `elem` is not a string, then index is None.
        
        
        Function Arguments
        ------------------
        - elem: element to retrieve root from
        """
        
        if not isinstance(elem, str):
            return (elem, None)

        #elem_orig = self.dict_mutable_elements_clean_to_original.get(elem, elem)

        is_clean = False
        if self.regex_dims.match(elem) is None:
            # check to see if elem is clean, will retrieve the original category
            is_clean = self.regex_dims_clean.match(elem) is not None
            if not is_clean:
                return (elem, None)

        space_char = self.clean_element(self.space_char) if is_clean else self.space_char 
        flag_dim = self.clean_element(self.flag_dim) if is_clean else self.flag_dim

        cat = elem.split(flag_dim)
        ind = int(cat[1]) # should successfully parse since the regular expressions searche for intes
        cat = cat[0].strip(space_char)
        
        cat = self.clean_element(cat) if is_clean else cat # clean root again if needed (may not be in dictionary)
        out = (cat, ind)

        return out 



    def replace(self,
        dict_repl: str,
        keys_are_clean: bool = False,
    ) -> str:
        """
        Replace the schema with mutable elements as specified in the dictionary
            dict_repl. Returns a string.

        Function Arguments
        ------------------
        - dict_repl: dictionary mapping mutable elements (unclean if clean_keys
            is False OR clean if clean_keys is True) to new strings

        Keyword Arguments
        -----------------
        - keys_are_clean: set to True if the keys in dict_repl are cleaned 
            mutable elements
        """

        dict_replace = dict_repl

        # if keys in dict_replace are cleaned, try to replace with known originals 
        if keys_are_clean:
            dict_replace = {}

            for k, v in dict_repl.items():
                elem_orig = self.dict_mutable_elements_clean_to_original.get(k)
                if elem_orig is None:
                    continue
                
                elem_orig = f"{self.container_elements}{elem_orig}{self.container_elements}"
                dict_replace.update({elem_orig: v})

        # replace the substrings and return
        out = sf.str_replace(self.schema, dict_replace, )

        return out


########################
#    SOME FUNCTIONS    #
########################


def clean_element(
    element: str, 
    container_elements: str = "$",
    container_expressions: str = "``",
    space_char: str = "-",
) -> Union[str, None]:
    """
    Clean a variable schema element to a manageable, shared name that excludes
        special characters.

    Function Arguments
    ------------------
    - element: element to clean

    Keyword Arguments
    -----------------
    - container_elements: string used to delimit elements--such as a category, 
        unit, or gas--within a schema
    - container_expressions: substring used to parse out schema and associated
        elements
    - space_char: character used within an element in place of a space
    """

    if not isinstance(element, str):
        return None

    element = (
        element
        .lower()
        .replace(container_elements, "")
        .replace(container_expressions, "")
        .replace(space_char, "_")
    )

    return element



def decompose_schema(
    var_schema: str, 
    container_elements: str = "$",
    container_expressions: str = "``",
    return_type: str = "schema",
    space_char: str = "-",
) -> str:
    """
    Decompose a variable schema input `var_schema` into elements used in the
        ModelVariable class.

    Function Arguments
    ------------------
    - var_schema: raw schema definition from variable definition

    Keyword Arguments
    -----------------
    - container_elements: string used to delimit elements--such as a category, 
        unit, or gas--within a schema
    - container_expressions: substring used to parse out schema and associated
        elements
    - return_type: return type to pass. Can take the following values:
        * "all": return an ordered tuple with the following elements:
            - dict_replacements
            - schema (cleaned)
            - category tuple
        * "replacements": return the replacement dictinary defined in the raw 
            schema
        * "schema": return the schema only
    - space_char: character used within an element in place of a space
    """
    valid_returns = ["all", "replacements", "schema"]
    return_type = "schema" if (return_type not in valid_returns) else return_type

    # split dictionary and variable schema apart
    var_schema = var_schema.split("(")
    var_schema[0] = (
        var_schema[0]
        .replace(container_expressions, "")
        .replace(" ", "")
    )

    # get dictionary of elements (without cleaning the keys)
    dict_repls = {}
    if len(var_schema) > 1:
        repls = var_schema[1].replace("`", "").split(",")
        for dr in repls:
            dr0 = (
                dr
                .replace(" ", "")
                .replace(")", "")
                .split("=")
            )

            var_schema[0] = var_schema[0].replace(dr0[0], dr0[1])
            key = dr0[0].replace(container_elements, "") # remove the delimiter
            dict_repls.update({key: dr0[1]})
    

    # return different elements based on desired return
    if return_type == "all":
        out = (dict_repls, var_schema[0])
    elif return_type == "replacements":
        out = dict_repls
    elif return_type == "schema":
        out = var_schema[0]

    return out



def is_model_variable(
    modvar: Any,
) -> bool:
    """
    Determine if the object is a ModelVariable
    """
    out = hasattr(modvar, "is_model_variable")
    return out



def unclean_category(
    cat: str
) -> str:
    """
    Convert a category to "unclean" by adding tick marks
    """
    return f"``{cat}``"


    
    