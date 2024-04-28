from attribute_table import AttributeTable
from model_attributes import *
import logging
import numpy as np
import pandas as pd
import support_functions as sf



class Socioeconomic:
    """
    Use Socioeconomic to calculate key drivers of emissions that are shared 
        across SISEPUEDE emissions models and sectors/subsectors. Includes 
        model variables for the following model subsectors (non-emission):

        * Economic (ECON)
        * General (GNRL)

    For additional information, see the SISEPUEDE readthedocs at:

        https://sisepuede.readthedocs.io/en/latest/energy_non_electric.html

    

    Intialization Arguments
    -----------------------
    - model_attributes: ModelAttributes object used in SISEPUEDE

    Optional Arguments
    ------------------
    - logger: optional logger object to use for event logging

    """
    def __init__(self, 
        model_attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None,
    ):

        self.logger = logger
        self.model_attributes = model_attributes
        self._initialize_input_output_components()

        # initialize subsector variables
        self._initialize_subsector_vars_econ()
        self._initialize_subsector_vars_gnrl()

        # initialize other properties
        self._initialize_other_properties()
        

    
    def __call__(self,
        *args,
        **kwargs
    ) -> pd.DataFrame:

        return self.project(*args, **kwargs)





    ##############################################
    #    INITIALIZATION AND SUPPORT FUNCTIONS    #
    ##############################################

    def check_df_fields(self, 
        df_se_trajectories: pd.DataFrame,
    ) -> None:
        """
        Check df fields to verify proper fields are present. If fill_value is
            not None, will instantiate missing values with fill_value.
        """
        check_fields = self.required_variables

        # check for required variables
        if not set(check_fields).issubset(df_se_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_se_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"Socioconomic projection cannot proceed: The fields {set_missing} are missing.")
        
        return None



    def _initialize_input_output_components(self,
    ) -> None:
        """
        Set a range of input components, including required dimensions, 
            subsectors, input and output fields, and integration variables.
            Sets the following properties:

            * self.output_model_variables
            * self.output_variables
            * self.required_dimensions
            * self.required_subsectors
            * self.required_base_subsectors
            * self.required_model_variables
            * self.required_variables
        """

        ##  START WITH REQUIRED DIMENSIONS (TEMPORARY - derive from attributes later)

        required_doa = [self.model_attributes.dim_time_period]
        self.required_dimensions = required_doa


        ##  ADD REQUIRED SUBSECTORS (TEMPORARY - derive from attributes)
        
        subsectors = sorted(list(
            sf.subset_df(
                self.model_attributes.get_subsector_attribute_table().table, 
                {
                    "sector": ["Socioeconomic"]
                }
            )["subsector"]
        ))
        subsectors_base = subsectors.copy()

        self.required_subsectors = subsectors
        self.required_base_subsectors = subsectors_base


        ##  SET INPUT OUTPUT VARIABLES

        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(subsectors)

        # get input/output model variables`
        required_model_vars = sorted(list(set(
            [
                self.model_attributes.dict_variable_fields_to_model_variables.get(x) 
                for x in required_vars
            ]
        )))

        output_model_vars = sorted(list(set(
            [
                self.model_attributes.dict_variable_fields_to_model_variables.get(x) 
                for x in output_vars
            ]
        )))

        self.output_model_variables = output_model_vars
        self.output_variables = output_vars
        self.required_model_variables = required_model_vars
        self.required_variables = required_vars + required_doa

        return None
    


    def _initialize_other_properties(self,
    ) -> None:
        """
        Initialize other properties that don't fit elsewhere. Sets the 
            following properties:

            * self.n_time_periods
            * self.time_periods
        """
        
        # time periods
        time_periods, n_time_periods = self.model_attributes.get_time_periods()


        ##  SET PROPERTIES

        self.n_time_periods = n_time_periods
        self.time_periods = time_periods

        return None



    def _initialize_subsector_vars_econ(self,
    ) -> None:
        """
        Initialize Economic (ECON) subsector vars for use in Socioeconomic. 
            Initializes the following properties:

            * self.cat_econ_*
            * self.dict_modvars_econ_*
            * self.ind_econ_*
            * self.modvar_econ_*
        """

        self.modvar_econ_gdp = "GDP"
        self.modvar_econ_gdp_per_capita = "GDP per Capita"

        return None


    
    def _initialize_subsector_vars_gnrl(self,
    ) -> None:
        """
        Initialize General (GNRL) subsector vars for use in Socioeconomic. 
            Initializes the following properties:

            * self.cat_gnrl_*
            * self.dict_modvars_gnrl_*
            * self.ind_gnrl_*
            * self.modvar_gnrl_*
        """

        self.modvar_gnrl_area = "Area of Region"
        self.modvar_gnrl_climate_change_hydropower_availability = "Climate Change Factor for Average Hydropower Availability"
        self.modvar_gnrl_elasticity_occrate_to_gdppc = "Elasticity National Occupation Rate to GDP Per Capita"
        self.modvar_gnrl_emission_limit_ch4 = ":math:\\text{CH}_4 Annual Emission Limit"
        self.modvar_gnrl_emission_limit_co2 = ":math:\\text{CO}_2 Annual Emission Limit"
        self.modvar_gnrl_emission_limit_n2o = ":math:\\text{N}_2\\text{O} Annual Emission Limit"
        self.modvar_gnrl_frac_eating_red_meat = "Fraction Eating Red Meat"
        self.modvar_gnrl_init_occ_rate = "Initial National Occupancy Rate"
        self.modvar_grnl_num_hh = "Number of Households"
        self.modvar_gnrl_occ_rate = "National Occupancy Rate"
        self.modvar_gnrl_subpop = "Population"
        self.modvar_gnrl_pop_total = "Total Population"

        return None


    
    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger.
            See ?sf._optional_log for more information

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)





    ###################################
    #    PRIMARY PROJECTION METHOD    #
    ###################################

    def project(self, 
        df_se_trajectories: pd.DataFrame,
        ignore_time_periods: bool = False,
        project_for_internal: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Function Arguments
        ------------------
        - df_se_trajectories: pd.DataFrame with input variable trajectories for 
            the Socioeconomic model.

        Keyword Arguments
        -----------------
        - ignore_time_periods: If True, will project independent of time period
            restrictions. Should generally be left as False
        - project_for_internal: 
            
            * If True, returns a tuple with the following ordered elements:

            [0] the first element of the return tuple is a modified version of 
                df_se_trajectories data frame that includes socioeconomic 
                projections. This should be passed to other models.

            [1] the second element of the return tuple is a data frame with 
                n_time_periods - 1 rows that represents growth rates in the 
                socioeconomic sector. Row i represents the growth rate from time 
                i to time i + 1.

            * If False, returns only the variables calculated in SE 
        """
        # check fields ands get some properties
        self.check_df_fields(df_se_trajectories)

        (
            dict_dims, 
            df_se_trajectories, 
            n_projection_time_periods, 
            projection_time_periods
        ) = self.model_attributes.check_projection_input_df(
            df_se_trajectories,
            override_time_periods = ignore_time_periods,
        )


        # initialize output
        df_out = (
            [df_se_trajectories.reset_index(drop = True)]
            if project_for_internal
            else [
                (
                    df_se_trajectories[self.required_dimensions]
                    .copy()
                    .reset_index(drop = True)
                )
            ]
        )

        # get some basic emission drivers
        vec_gdp = self.model_attributes.extract_model_variable(#
            df_se_trajectories, 
            self.modvar_econ_gdp, 
            return_type = "array_base",
        )

        vec_pop = np.sum(
            self.model_attributes.extract_model_variable(#
                df_se_trajectories, 
                self.modvar_gnrl_subpop, 
                return_type = "array_base"
            ), 
            axis = 1,
        )

        vec_gdp_per_capita = np.nan_to_num(vec_gdp/vec_pop, 0.0, posinf = 0.0)
        vec_gdp_per_capita *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_econ_gdp,
            self.modvar_econ_gdp_per_capita,
            "monetary"
        )

        # growth rates
        vec_rates_gdp = vec_gdp[1:]/vec_gdp[0:-1] - 1
        vec_rates_gdp_per_capita = vec_gdp_per_capita[1:]/vec_gdp_per_capita[0:-1] - 1

        # calculate the housing occupancy rate
        vec_gnrl_elast_occrate_to_gdppc = self.model_attributes.extract_model_variable(#
            df_se_trajectories, 
            self.modvar_gnrl_elasticity_occrate_to_gdppc, 
            return_type = "array_base",
        )

        vec_gnrl_init_occrate = self.model_attributes.extract_model_variable(#
            df_se_trajectories, 
            self.modvar_gnrl_init_occ_rate, 
            return_type = "array_base",
        )

        vec_gnrl_growth_occrate = sf.project_growth_scalar_from_elasticity(
            vec_rates_gdp_per_capita, 
            vec_gnrl_elast_occrate_to_gdppc, 
            False, 
            "standard",
        )

        vec_gnrl_occrate = vec_gnrl_init_occrate[0]*vec_gnrl_growth_occrate
        vec_gnrl_num_hh = np.round(vec_pop/vec_gnrl_occrate).astype(int)

        # add to output
        df_out += [
            self.model_attributes.array_to_df(
                vec_pop,
                self.modvar_gnrl_pop_total,
                False
            ),
            self.model_attributes.array_to_df(
                vec_gdp_per_capita, 
                self.modvar_econ_gdp_per_capita, 
                False
            ),
            self.model_attributes.array_to_df(
                vec_gnrl_occrate, 
                self.modvar_gnrl_occ_rate, 
                False
            ),
            self.model_attributes.array_to_df(
                vec_gnrl_num_hh, 
                self.modvar_grnl_num_hh, 
                False
            )
        ]

        df_se_trajectories = (
            pd.concat(
                df_out, 
                axis = 1
            )
            .reset_index(drop = True)
        )


        ##  setup output
        out = df_se_trajectories

        if project_for_internal:
            # get internal variables that are shared between downstream sectors
            time_periods_df = np.array(df_se_trajectories[self.model_attributes.dim_time_period])[0:-1]
            df_se_internal_shared_variables = df_se_trajectories[[self.model_attributes.dim_time_period]].copy().reset_index(drop = True)

            # build data frame of rates--will not have values in the final time period
            df_se_internal_shared_variables = pd.merge(
                df_se_internal_shared_variables,
                pd.DataFrame(
                    {
                        self.model_attributes.dim_time_period: time_periods_df, 
                        "vec_rates_gdp": vec_rates_gdp, 
                        "vec_rates_gdp_per_capita": vec_rates_gdp_per_capita
                    }
                ),
                how = "left"
            )

            out = (df_se_trajectories, df_se_internal_shared_variables)
        self.cols = list(df_se_trajectories.columns)
        return out
