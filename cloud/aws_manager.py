from sisepuede.core.analysis_id import *
from sisepuede.core.attribute_table import AttributeTable
import base64
import boto3
import logging
import sisepuede.core.model_attributes as ma
import numpy as np
import sisepuede.data_management.ordered_direct_product_table as odpt
import os, os.path
import pandas as pd
import random
import re
import setup_analysis as sa
import shutil
import sisepuede.manager.sisepuede as ssp
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.utilities.sql_utilities as squ
import sisepuede.core.support_classes as sc
import sisepuede.utilities.support_functions as sf
import sys
import time
from typing import *
import warnings




class ShellTemplate:
    """
    Class to build and modify shell scripts, including AWS User Data
    
    Initialization Arguments
    ------------------------
    - fp_template: file path to UserData template to modify and build from
    
    Keyword Arguments
    -----------------
    - char_esc: characters used to denote entrance and exit from a variable 
            specification in the template
    - raw_template_input: set to true to pass a string of the template as 
        fp_template
    """
    def __init__(self,
        fp_template: str,
        char_esc: str = "$$",
        raw_template_input: bool = False,
    ): 
        self._initialize_template(
            fp_template, 
            raw_template_input = raw_template_input,
        )
        self._initialize_replacement_matchstrings(
            char_esc = char_esc, 
        )
        
    
        
    def _initialize_replacement_matchstrings(self,
        char_esc: str = "$$",
    ) -> None:
        """
        Initialize matchstrings in a template. Sets the following properties:
            
            * self.char_esc
            * self.char_esc_regex
            * self.regex_matchvars
        
        Function Arguments
        ------------------
        - fp: path to yaml to read

        Keyword Arguments
        -----------------
        - char_esc: characters used to denote entrance and exit from a variable 
            specification in the template
        """
        # set regular expression to identify variables
        char_esc_regex = char_esc.replace("$", "\$")
        regex_matchvars = re.compile(f"{char_esc_regex}(.*){char_esc_regex}")
        

        ##  SET PROPERTIES
        
        self.char_esc = char_esc
        self.char_esc_regex = char_esc_regex
        self.regex_matchvars = regex_matchvars
        
        return None
    
    
    
    def _initialize_template(self,
        fp_template: str,
        raw_template_input: bool = False,
    ) -> None:
        """
        Initialize the template string from file path `fp_template`. Sets the 
            following properties:
        
            * self.template
            
        Function Arguments
        ------------------
        - fp: path to user data template

        Keyword Arguments
        -----------------
        - raw_template_input: set to true to pass a string of the template as 
            fp_template
        """
        
        template = (
            sf.read_text(fp_template, as_lines = False)
            if not raw_template_input
            else fp_template
        )

        if not isinstance(template, str):
            raise NotImplementedError(f"Error initializing template in UserDataTempalte: no template found at '{fp_template}'")
        
        self.template = template
        
        return None
    
    
    
    #####################
    #   CORE METHODS    #
    #####################
    
    def fill_template(self,
        dict_fill: Dict[str, str],
    ) -> str:
        """
        Fill the template by replacing match strings with new  
        
        Function Arguments
        ------------------
        - dict_fill: dictionary mapping matchstring variables to new strings

        Keyword Arguments
        -----------------
        """
        
        template_out = self.template
        
        for k, v in dict_fill.items():
            
            if not isinstance(v, str):
                continue
            
            k_new = self.get_template_matchstring(k)
            template_out = template_out.replace(k_new, v)

        return template_out
            
        
        
    def get_template_matchstring(self,
        matchstr: str,
        char_esc: Union[str, None] = None,
    ) -> str:
        """
        Get the substring in the template to replace on
        
        Function Arguments
        ------------------
        - matchstr: name of matchstr target variable (excludes escape characters)

        Keyword Arguments
        -----------------
        - char_esc: escape character in template delimiting variables
        """
        
        char_esc = (
            self.char_esc
            if not isinstance(char_esc, str)
            else char_esc
        )
        out = f"{char_esc}{matchstr}{char_esc}"
        
        return out



class AWSManager:
    """
    Set up some shared values used to manage runs on AWS

    Initialization Arguments
    ------------------------
    - sisepuede: SISEPUEDE object used to access ID information, 
        model_attributes, primary_key_database, and table names
    - primary_key_database: OrderedDirectProductTable used to identify primary
        keys 
    - config: path to YAML config file or sc.YAMLConfiguration object
    - fp_template_docker_shell: path to template shell file used to launch 
        application on docker
    - fp_template_user_data: path to template user data used in instance 
        launches

    Keyword Arguments
    -----------------
    - as_string_docker_shell: set to True to pass fp_template_docker_shell as a
        template string rather than a file path
    - as_string_user_data: set to True to pass fp_template_user_data as a 
        template string rather than a file path
    - logger: optional logging.Logger object to use for logs
    - partitions_ordered: ordering for partitions in S3 file paths for Athena.
        If None, no partitioning is performed.
    """

    def __init__(self,
        sisepuede: ssp.SISEPUEDE,
        config: Union[str, sc.YAMLConfiguration],
        fp_template_docker_shell: str,
        fp_template_user_data: str,
        as_string_docker_shell: bool = False,
        as_string_user_data: bool = False,
        logger: Union[logging.Logger, None] = None,
        partitions_ordered: Union[List[str], None] = None,
    ) -> None:
        
        self.logger = logger

        self._initialize_template_matchstrs()
        self._initialize_attributes(
            sisepuede,
        )
        self._initialize_config(config)
        self._initialize_templates(
            fp_template_docker_shell,
            fp_template_user_data,
            as_string_docker_shell = as_string_docker_shell,
            as_string_user_data = as_string_user_data,
        )

        # some dependent initialization
        self._initialize_shell_script_environment_vars()
        self._initialize_paths()
        self._initialize_docker_properties()
        self._initialize_table_attributes()

        # initialize aws 
        self._initialize_aws_properties(
            partitions_ordered = partitions_ordered,
        )
        self._initialize_aws()
        self._initialize_s3_paths()
        




    ########################
    #    INITIALIZATION    #
    ########################

    def build_s3_query_output_file_names(self,
        table_name_composite_default: str = "composite",
    ) -> Dict:
        """
        Return a dictionary of file names for query result files. Returns a 
            dictionary mapping types to file names

        Keyword Arguments
        -----------------
        - table_name_composite_default: default table name
        """

        valid_types = [
            "composite",
            "create_input",
            "create_output",
            "input",
            "output",
            "repair_input",
            "repair_output",
        ]

        # get appendage to input/output tables
        appendage_table_name = self.config.get("aws_athena.query_table_appendage")
        appendage_table_name = (
            ""
            if not isinstance(appendage_table_name, str)
            else appendage_table_name
        )
        
        # build dictionary of file base names
        keys_modify = ["input", "output"]
        dict_fbn = dict(
            (x, x) for x in valid_types
        )

        # update with tables from SISEPUEDE
        for key in keys_modify:
            table_name = self.dict_input_output_to_table_names.get(key)
            if table_name is None:
                continue

            table_name = f"{table_name}{appendage_table_name}"
            dict_fbn.update({key: table_name.lower()})


        # add in composite query
        table_name_composite = self.config.get("aws_athena.filename_composite_query")
        table_name_composite = (
            table_name_composite_default 
            if table_name_composite is None
            else table_name_composite
        )

        dict_fbn.update({"composite": table_name_composite})

        return dict_fbn



    def build_table_name_to_type_dicts(self,
    ) -> Tuple[Dict, Dict]:
        """
        Build a dictionary mapping table names to input/output based on 
            configuration. Returns a tuple of dictionaries of the following 
            form:

            (dict_table_names_to_io, dict_io_to_table_names)
        """
        dict_table_names_to_io = {
            self.sisepuede_database.table_name_input: "input",
            self.sisepuede_database.table_name_output: "output",
        }

        dict_io_to_table_names = sf.reverse_dict(dict_table_names_to_io)

        return dict_table_names_to_io, dict_io_to_table_names



    def get_sisepuede_relative_paths(self,
        path_type: str,
    ) -> Union[str, None]:
        """
        Get relative paths for use in instance templates. 

        Function Arguments
        ------------------
        - path_type: any of
            * "julia"
            * "python"
            * "out"
            * "ref"
            * "tmp"
        """

        if path_type == "julia":
            dir_target = self.file_struct.dir_jl
        elif path_type == "python":
            dir_target = self.file_struct.dir_py
        elif path_type == "out":
            dir_target = self.file_struct.dir_out
        elif path_type == "ref":
            dir_target = self.file_struct.dir_ref
        elif path_type == "tmp":
            dir_target = self.file_struct.dir_tmp
        else: 
            return None

        dir_return = os.path.relpath(
            dir_target, 
            start = self.file_struct.dir_proj
        )

        return dir_return



    def get_template(self,
        fp: str,
        raw_template_input: bool = False,
    ) -> ShellTemplate:
        """
        Get a shell template from file path fp. Set raw_template_input = True to 
            pass a string for the template. Uses escape character char_esc.
        """

        fp = "" if not isinstance(fp, str) else fp
        if (not os.path.exists(fp)) & (not raw_template_input):
            raise NotImplementedError("Unable to instantiate user data ShellTemplate: file '{fp_template_user_data}'")

        template = ShellTemplate(
            fp, 
            raw_template_input = raw_template_input,
        )

        return template
    


    def get_user_data_subtemplate_partitions(self,
        template: str,
        delim_end: str = "END",
        delim_newline: str = "\n",
        delim_start: str = "BEGIN",
        dict_repl_keys_with_text: Union[Dict[str, str], None] = None,
        key_main: str = "MAIN",
        restrict_to_keys: Union[List[str], None] = None,
    ) -> Dict:
        """
        For a template, look for subtemplates delimined by

        <{delim_start}:KEY>
        <{delim_end}:KEY>

        Always returns main template under `key_main`

        Function Arguments
        ------------------
        - template: input template string to split

        Keyword Arguments
        -----------------
        - delim_end: delimiter <{delim_end}:key> used to close a split
        - delim_newline: delimter used to split template into new lines
        - delim_start: delimiter <{delim_start}:key> used to open a split
        - key_main: key in output dictionary for unpartitioned lines
        - restrict_to_keys: set allowable keys to spliton  (will drop other 
            keys)
        """

        dict_out = {
            key_main: []
        }

        lines_template = template.split(delim_newline)
        lines_skip = set({})

        # get valid delineations
        regex_start = re.compile(f"<{delim_start}:(.*)>")
        regex_end = re.compile(f"<{delim_end}:(.*)>")
        valid_splits_start = set(re.findall(regex_start, template))
        valid_splits_end = set(re.findall(regex_start, template))


        ##  VALIDATE DELIMITTING SETUP

        valid_splits = []

        for split in valid_splits_start:
            # index starting line and count
            l0 = [0, 0]
            l1 = [0, 0]
            str_start = f"<{delim_start}:{split}>"
            str_end = f"<{delim_end}:{split}>"

            # iterate to check lines
            for i, line in enumerate(lines_template):

                # check starting index and count
                l0[0] = i if (str_start in line) else l0[0]
                l0[1] += 1 if (str_start in line) else 0

                l1[0] = i if (str_end in line) else l1[0]
                l1[1] += 1 if (str_end in line) else 0

            validated = (l1[0] > l0[0]) & (l0[1] == 1) & (l1[1] == 1)
            validated &= delim_newline not in split
            valid_splits.append(split) if validated else None

        valid_splits.sort()
        
        # reassign to actual delimiters
        valid_splits_start = [f"<{delim_start}:{x}" for x in valid_splits]
        valid_splits_end = [f"<{delim_end}:{x}" for x in valid_splits]
        invalid_splits = (
            [x for x in valid_splits if x not in restrict_to_keys]
            if sf.islistlike(restrict_to_keys)
            else []
        )


        ##  ITERATE OVER LINES TO SPLIT OUT

        in_delim = False
        key = None
        keys_drop = []

        # get begin statements (cannot nest)
        for i, line in enumerate(lines_template):
            
            if not in_delim:
                if any([(x in line) for x in valid_splits_start]):
                    key_try = regex_start.match(line).groups()[0]
                    if key_try not in invalid_splits:
                        key = key_try
                        dict_out.update({key: []})
                        in_delim = True
                        
                        # optional replacement string?
                        str_repl = (
                            dict_repl_keys_with_text.get(key)
                            if dict_repl_keys_with_text is not None
                            else None
                        )

                        (
                            dict_out[key_main].append(str_repl)
                            if str_repl is not None
                            else None
                        )

                    else:
                        keys_drop.append(key_try)

                else:
                    (
                        dict_out[key_main].append(line)
                        if not any([(f"<{delim_end}:{x}>" in line) for x in keys_drop])
                        else None
                    )

            else:
                if in_delim:
                    str_close = f"<{delim_end}:{key}>"
                    if str_close in line:
                        in_delim = False
                    else:
                        (
                            dict_out[key].append(line)
                            if not any([(f"<{delim_end}:{x}>" in line) for x in keys_drop])
                            else None
                        )
        
        dict_out = dict(
            (
                k, 
                ShellTemplate(
                    delim_newline.join(v),
                    raw_template_input = True
                )
            ) 
            for k, v in dict_out.items()
        )
        
        return dict_out



    def _initialize_attributes(self,
        sisepuede: ssp.SISEPUEDE,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.dict_dims_to_docker_flags (map SISEPUEDE dimensions to docker 
                flags)
            * self.field_launch_index
            * self.field_instance_id
            * self.field_random_seed
            * self.field_ip_address
            * self.file_struct
            * self.flag_key_database_type
            * self.flag_key_id
            * self.flag_key_max_solve_attempts
            * self.flag_key_random_seed
            * self.flag_key_save_inputs
            * self.flag_key_try_exogenous_xl_type
            * self.key_design
            * self.key_future
            * self.key_region
            * self.key_primary
            * self.key_strategy
            * self.model_attributes
            * self.primary_key_database
            * self.regex_config_experiment_components
            * self.regions
            * self.sisepuede_database
        """
        # pull in some SISEPUEDE attributes
        model_attributes = sisepuede.model_attributes
        primary_key_database = sisepuede.experimental_manager.primary_key_database
        database = sisepuede.database

        # analysis dimensional keys
        key_design = sisepuede.key_design
        key_future = sisepuede.key_future
        key_region = sisepuede.key_region
        key_primary = sisepuede.key_primary
        key_strategy = sisepuede.key_strategy
        key_time_period = sisepuede.key_time_period

        # additional command line flags
        flag_key_database_type = "database_type"
        flag_key_id = "id"
        flag_key_max_solve_attempts = "max_solve_attempts"
        flag_key_random_seed = "random_seed"
        flag_key_save_inputs = "save_inputs"
        flag_key_try_exogenous_xl_type = "try_exogenous_xl_types"

        # map keys to flag values
        dict_dims_to_docker_flags = {
            key_design: "keys-design",
            key_future: "keys-future",
            key_region: "regions",
            key_primary: "keys-primary",
            key_strategy: "keys-strategy",
            flag_key_database_type: "database-type",
            flag_key_id: "id",
            flag_key_max_solve_attempts: "max-solve-attempts",
            flag_key_random_seed: "random-seed",
            flag_key_save_inputs: "save-inputs",
            flag_key_try_exogenous_xl_type: "try-exogenous-xl-types",
        }

        regions = sc.Regions(model_attributes)

        # set some fields for output fields
        field_instance_id = "instance_id"
        field_ip_address = "ip_address"
        field_launch_index = "launch_index"
        field_n_launch_tries = "n_launch_tries"
        field_random_seed = "random_seed"
        
        


        ##  SOME EXPERIMENTAL PROPERTIES

        # regular expression to use for experimental components
        regex_config_experiment_components = re.compile("part_(\d*$)")


        ##  SET PROPERTIES

        self.dict_dims_to_docker_flags = dict_dims_to_docker_flags
        self.field_instance_id = field_instance_id
        self.field_ip_address = field_ip_address
        self.field_launch_index = field_launch_index
        self.field_n_launch_tries = field_n_launch_tries
        self.field_random_seed = field_random_seed
        self.file_struct = sisepuede.file_struct
        self.flag_key_database_type = flag_key_database_type
        self.flag_key_id = flag_key_id
        self.flag_key_max_solve_attempts = flag_key_max_solve_attempts
        self.flag_key_random_seed = flag_key_random_seed
        self.flag_key_save_inputs = flag_key_save_inputs
        self.flag_key_try_exogenous_xl_type = flag_key_try_exogenous_xl_type
        self.key_design = key_design
        self.key_future = key_future
        self.key_region = key_region
        self.key_primary = key_primary
        self.key_strategy = key_strategy
        self.key_time_period = key_time_period
        self.model_attributes = model_attributes
        self.primary_key_database = primary_key_database
        self.regex_config_experiment_components = regex_config_experiment_components
        self.regions = regions
        self.sisepuede_database = database

        return None
    


    def _initialize_aws(self,
        default_profile_name: str = "default",
    ) -> None:
        """
        Initialize AWS session components. Sets the following properties:
            
            * self.b3_session
            * self.client_athena
            * self.client_ec2
            * self.client_s3
            * self.command_aws
            * self.resource_s3
            * self.s3_buckets
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - default_profile_name: default passed to boto3.Session as profile_name
            if none found in configuration file
        """
        
        ##  GET AWS KEYS
        
        aws_access_key_id = self.config.get("aws_config.aws_access_key_id")
        aws_secret_access_key = self.config.get("aws_config.aws_secret_access_key")
        aws_session_token = self.config.get("aws_config.aws_session_token")
        aws_region = self.config.get("aws_config.region")
        profile_name = self.config.get("aws_config.profile_name")
        
        # set all to None to rely on config defaults
        any_none_imples_all_none = [aws_access_key_id, aws_secret_access_key, aws_session_token]
        if any([(x is None) for x in any_none_imples_all_none]):
            aws_access_key_id = None
            aws_secret_access_key = None
            aws_session_token = None

        # check profile
        profile_name = (
            default_profile_name
            if not isinstance(profile_name, str)
            else profile_name
        )
            
            
        ##  INITIALIZE boto3 SESSION

        b3_session = boto3.Session(
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
            aws_session_token = aws_session_token,
            profile_name = profile_name,
            #region_name = aws_region,
        )
        
        # get clients and resources
        client_athena = b3_session.client("athena")
        client_ec2 = b3_session.client("ec2")
        client_s3 = b3_session.client("s3")
        resource_s3 = b3_session.resource("s3")
        
        # get some often-used derivatives
        buckets = self.get_buckets(client_s3)


        ##  OTHER PROPERTIES

        command_aws = self.config.get("aws_ec2.path_aws")
        command_aws = (
            "aws"
            if not isinstance(command_aws, str)
            else command_aws
        )

        
        ##  SET PROPERTIES

        self.b3_session = b3_session
        self.client_athena = client_athena
        self.client_ec2 = client_ec2
        self.client_s3 = client_s3
        self.command_aws = command_aws
        self.resource_s3 = resource_s3
        self.s3_buckets = buckets

        return None



    def _initialize_aws_properties(self,
        partitions_ordered: Union[List[str], None] = None,
    ) -> None:
        """
        Use the configuration file to set some key shared properties for AWS. 
            Sets the following properties:

            * self.athena_database
            * self.athena_partitions_ordered
            * self.athena_reponse_key_exec_id
            * self.aws_dict_tags
            * self.aws_url_instance_id_metadata
            * self.ec2_image
            * self.ec2_instance_base_name
            * self.ec2_instance_type
            * self.ec2_max_n_instances
            * self.s3_bucket
            * self.s3_key_database
            * self.s3_key_logs
            * self.s3_key_metadata

        NOTE: *Excludes* paths on instances. Those are set in
            _initialize_paths()

        Keyword Arguments
        -----------------
        - partitions_ordered: ordering for partitions in S3 file paths for 
            Athena. If None, no partitioning is performed.
        """

        # general AWS properties
        self.aws_dict_tags = self.config.get("aws_general.tag")
        self.aws_region_name = self.config.get("aws_config.region_name")
        self.aws_url_instance_id_metadata = "http://169.254.169.254/latest/meta-data/instance-id/"

        # athena properties
        self.athena_database = self.config.get("aws_athena.database")
        self.athena_partitions_ordered = partitions_ordered
        self.athena_reponse_key_exec_id = "QueryExecutionId"

        # EC2 properties
        self.ec2_image = self.config.get("aws_ec2.database")
        self.ec2_instance_base_name = self.config.get("aws_ec2.instance_base_name")
        self.ec2_instance_type = self.config.get("aws_ec2.instance_type")
        self.ec2_max_n_instances = int(self.config.get("aws_ec2.max_n_instances"))

        # S3 properties
        self.s3_bucket = self.config.get("aws_s3.bucket")
        self.s3_key_database = self.config.get("aws_s3.key_database")
        self.s3_key_logs = self.config.get("aws_s3.key_logs")
        self.s3_key_metadata = self.config.get("aws_s3.key_metadata")

        return None



    def _initialize_config(self,
        config: Union[str, sc.YAMLConfiguration],
    ) -> None:
        """
        Initialize the configuration object. Initializes the following 
            properties:

            * self.config

        Function Arguments
        ------------------
        - config: file path to yaml configuration object OR YAMLConfiguration
            object
        """

        self.config = None

        if isinstance(config, str):
            try:
                config = sc.YAMLConfiguration(config)
            except Exception as e:
                raise RuntimeError(f"Error in AWSManager trying to initialize YAMLConfiguration: {e}")

        if isinstance(config, sc.YAMLConfiguration):
            self.config = config

        return None
    


    def _initialize_docker_properties(self,
    ) -> None:
        """
        Use the configuration file to set some key shared properties for AWS. 
            Sets the following properties:

            * self.docker_image_name
            * self.use_ecr

        NOTE: *Excludes* paths on instances. Those are set in
            _initialize_paths()
        """

        # check if using AWS ECR
        use_ecr = self.config.get("docker.use_ecr")
        use_ecr = bool(use_ecr) if (use_ecr is not None) else False

        name_docker_image_public = self.config.get("docker.image_name")
        name_docker_image_ecr = self.config.get("docker.image_name_ecr")

        # get the image name
        name_docker_image = (
            name_docker_image_ecr
            if use_ecr & (name_docker_image_ecr is not None)
            else name_docker_image_public
        )


        self.docker_image_name = name_docker_image
        self.use_ecr = use_ecr

        return None



    def _initialize_paths(self,
    ) -> None:
        """
        Initialize some paths. Initializes the following properties:

            * self.dir_docker_sisepuede_python
            * self.dir_docker_sisepuede_out
            * self.dir_docker_sisepuede_repo
            * self.dir_instance_home
            * self.dir_instance_out
            * self.dir_instance_out_db
            * self.dir_instance_out_run_package
            * self.fp_instance_instance_info
            * self.fp_instance_shell_script
            * self.fp_instance_sisepuede_log

        NOTE: must be initialized *after* 
            _initialize_shell_script_environment_vars()

        Function Arguments
        ------------------
        """

        ##  INSTANCE PATHS

        # directories
        dir_instance_home = self.config.get("aws_ec2.dir_home")
        dir_instance_out = self.config.get("aws_ec2.dir_output")
        dir_instance_out_db = self.get_path_instance_output_db(dir_instance_out)
        dir_instance_out_run_package = self.get_path_instance_output_run_package(dir_instance_out)


        ##  DOCKER PATHS

        # directories
        dir_docker_sisepuede_repo = self.config.get("docker.dir_repository")
        dir_docker_sisepuede_out = os.path.join(
            dir_docker_sisepuede_repo,
            self.get_sisepuede_relative_paths("out")
        )
        dir_docker_sisepuede_python = os.path.join(
            dir_docker_sisepuede_repo,
            self.get_sisepuede_relative_paths("python")
        )


        ##  FILE PATHS
        
        fp_instance_instance_info = self.get_path_instance_metadata(dir_instance_out)
        fp_instance_shell_script = self.config.get("aws_ec2.instance_shell_path")
        fp_instance_sisepuede_log = self.get_path_instance_log(dir_instance_out_run_package)


        ##  SET PROPERTIES

        self.dir_docker_sisepuede_python = dir_docker_sisepuede_python
        self.dir_docker_sisepuede_out = dir_docker_sisepuede_out
        self.dir_docker_sisepuede_repo = dir_docker_sisepuede_repo
        self.dir_instance_home = dir_instance_home
        self.dir_instance_out = dir_instance_out
        self.dir_instance_out_db = dir_instance_out_db
        self.dir_instance_out_run_package = dir_instance_out_run_package
        self.fp_instance_instance_info = fp_instance_instance_info
        self.fp_instance_shell_script = fp_instance_shell_script
        self.fp_instance_sisepuede_log = fp_instance_sisepuede_log

        return None
    


    def _initialize_s3_paths(self,
    ) -> None:
        """
        Initialize some s3 paths. Initializes the following properties:

            * self.s3p_athena_database
            * self.s3p_run_metadata

        NOTE: must be initialized *after* 
            _initialize_aws()

        Function Arguments
        ------------------
        """
        ##  S3 PATHS

        s3p_athena_database = self.get_s3_path_athena_output("database")
        s3p_athena_queries = self.get_s3_path_athena_output("queries")
        s3p_run_log = self.get_s3_path_athena_output("logs")
        s3p_run_metadata = self.get_s3_path_athena_output("metadata")

        dict_s3p_query_fbns_by_type = self.build_s3_query_output_file_names()

        ##  SET PROPERTIES

        self.dict_s3p_query_fbns_by_type = dict_s3p_query_fbns_by_type
        self.s3p_athena_database = s3p_athena_database
        self.s3p_athena_queries = s3p_athena_queries
        self.s3p_run_log = s3p_run_log
        self.s3p_run_metadata = s3p_run_metadata

        return None
    


    def _initialize_shell_script_environment_vars(self,
    ) -> None:
        """
        Initialize some environment variables used to shell scripts. Sets the
            following properties:

            * self.shell_env_instance_id

        Function Arguments
        ------------------
        """

        self.heredoc_eof = "XEOF"
        self.shell_env_instance_id = "$INSTANCEID"

        return None
    


    def _initialize_table_attributes(self,
    ) -> None:
        """
        Initialize key table information. Sets the following properties:

            * self.dict_input_output_to_table_names
            * self.dict_keys_table_name_to_index_ordered
            * self.dict_table_names_to_input_output

        NOTE: These hardcode some information from 
        """

        # dictionary mapping 
        (
            dict_table_names_to_input_output, 
            dict_input_output_to_table_names
        ) = self.build_table_name_to_type_dicts()


        # get table indicies for input
        dict_keys_table_name_to_index_ordered = {}

        for key, table in dict_input_output_to_table_names.items():
            keys_table_index_ordered_cur = (
                self.sisepuede_database
                .db
                .dict_iterative_database_tables
                .get(table)
                .fields_index
                .copy()
            )
            keys_table_index_ordered_cur.extend([self.key_time_period])

            keys_table_index_ordered_cur = [
                x for x in self.model_attributes.sort_ordered_dimensions_of_analysis
                if x in keys_table_index_ordered_cur
            ]

            dict_keys_table_name_to_index_ordered.update({table: keys_table_index_ordered_cur})

    
        ##  SET PROPERTIES

        self.dict_input_output_to_table_names = dict_input_output_to_table_names
        self.dict_keys_table_name_to_index_ordered = dict_keys_table_name_to_index_ordered
        self.dict_table_names_to_input_output = dict_table_names_to_input_output

        return None



    def _initialize_templates(self,
        fp_template_docker_shell: str,
        fp_template_user_data: str,
        as_string_docker_shell: bool = False,
        as_string_user_data: bool = False,
    ) -> None:
        """
        Initialize the configuration object. Initializes the following 
            properties:

            * self.template_docker_shell
            * self.template_user_data

        Function Arguments
        ------------------
        - config: file path to yaml configuration object OR YAMLConfiguration
            object

        Keyword Arguments
        -----------------
        - as_string: set to true to treat fp_template_user_data as a raw string
        """

        # build templates
        template_docker_shell = self.get_template(
            fp_template_docker_shell,
            raw_template_input = as_string_docker_shell,
        )
        template_user_data_raw = self.get_template(
            fp_template_user_data,
            raw_template_input = as_string_user_data,
        )

        # set some keys and split off
        dict_template_key_main = "MAIN"
        dict_template_key_screen = "SCREEN"

        dict_templates = self.get_user_data_subtemplate_partitions(
            template_user_data_raw.template,
            dict_repl_keys_with_text = {
                dict_template_key_screen: template_user_data_raw.get_template_matchstring(
                    self.matchstr_screen_shell_dummy
                )
            },
            key_main = dict_template_key_main,
            restrict_to_keys = [dict_template_key_screen],
        )


        ##  SET PROPERTIES

        self.dict_templates = dict_templates
        self.dict_template_key_main = dict_template_key_main
        self.dict_template_key_screen = dict_template_key_screen
        self.template_docker_shell = template_docker_shell
        self.template_user_data_raw = template_user_data_raw

        return None
    


    def _initialize_template_matchstrs(self,
    ) -> None:
        """
        Initialize matchstrings in User Data and Docker Shell templates. Sets
            the following properties:

            * self.matchstr_

        NOTE: needs some work to be gener
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """

        ##  DOCKER SHELL MATCHSTRINGS

        self.matchstr_shell_sispeude_cl_flags = "FLAGS_SISEPUEDE_CL"

        ##  USER DATA MATCHSTRINGS
        
        self.matchstr_copy_to_s3 = "LINES_COPY_TO_S3"
        self.matchstr_dimensional_reads = "LINES_DIMENSIONAL_READS"
        self.matchstr_docker_generate_shell = "LINES_CREATE_DOCKER_SHELL"
        self.matchstr_docker_pull = "LINE_DOCKER_PULL"
        self.matchstr_docker_run = "LINE_DOCKER_RUN"
        self.matchstr_generate_instance_info = "LINES_GENERATE_INSTANCE_INFO"
        self.matchstr_homedir = "LINE_SET_HOME_DIRECTORY"
        self.matchstr_mkdir_instance_out = "LINES_CREATE_INSTANCE_OUT"
        self.matchstr_screen_shell_dummy = "LINES_SCREEN_WRAP"
        self.matchstr_terminate_instance = "LINES_TERMINATE_INSTANCE"

        return None
    


    def _log(self,
        msg: str,
        type_log: str = "log",
        **kwargs
    ) -> None:
        """
        Clean implementation of sf._optional_log in-line using default logger. See
            ?sf._optional_log for more information.

        Function Arguments
        ------------------
        - msg: message to log

        Keyword Arguments
        -----------------
        - type_log: type of log to use
        - **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
        """
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)





    #######################
    #    KEY FUNCTIONS    #
    #######################

    def build_docker_shell_flags(self,
        dict_subset: Dict[str, List[int]],
        delim_args: str = " ",
        delim_elems: str = ",",
        delim_newline: str = "\n",
        output_type: str = "echo",
    ) -> str:
        """
        Replace the template line that specified dimensional subsets 
            to run.
        
        Function Arguments
        ------------------
        - dict_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template.
        - manager: AWS manager used for method access

        Keyword Arguments
        -----------------
        - delim_args: argument separator
        - delim_elems: element delimiter for arguments passed via flags
        - delim_newline: string delimiter used for new lines
        - output_type: 
            * "echo": write the echo statemest for UserData
            * "flags": write python flags for Docker Shell
        """
        
        flags = []

        # add scenario flags, random seed, and id
        for k, v in dict_subset.items():
            
            v_out = (
                delim_elems.join([str(x) for x in v])
                if sf.islistlike(v)
                else v
            )
            flag = self.dict_dims_to_docker_flags.get(k, k)
                
            flag = f"--{flag} {v_out}"
            
            flags.append(flag)
        
        # add in database type
        db_type = self.config.get("sisepuede_runtime.database_type")
        db_type = "csv" if (db_type is None) else db_type
        flag = self.dict_dims_to_docker_flags.get(self.flag_key_database_type)
        (
            flags.append(f"--{flag} {db_type}") 
            if isinstance(flag, str)
            else None
        )

        # add in id type
        id_str = self.file_struct.id
        flag = self.dict_dims_to_docker_flags.get(self.flag_key_id)
        (
            flags.append(f"--{flag} \"{id_str}\"") 
            if isinstance(flag, str)
            else None
        )

        # add in max solve attempts
        max_attempts = self.config.get("sisepuede_runtime.max_solve_attempts")
        flag = self.dict_dims_to_docker_flags.get(self.flag_key_max_solve_attempts)
        (
            flags.append(f"--{flag} {max(max_attempts, 1)}") 
            if isinstance(max_attempts, int)
            else None
        )

        # add in save inputs
        save_inputs = self.config.get("sisepuede_runtime.save_inputs")
        save_inputs = True if not isinstance(save_inputs, bool) else save_inputs
        flag = self.dict_dims_to_docker_flags.get(self.flag_key_save_inputs)
        (
            flags.append(f"--{flag}") 
            if isinstance(flag, str) & save_inputs
            else None
        )

        # add in try exogenous xl types
        try_exog_xl = self.config.get("sisepuede_runtime.try_exogenous_xl_types")
        try_exog_xl = False if not isinstance(try_exog_xl, bool) else try_exog_xl
        flag = self.dict_dims_to_docker_flags.get(self.flag_key_try_exogenous_xl_type)
        (
            flags.append(f"--{flag}") 
            if isinstance(flag, str) & try_exog_xl
            else None
        )

        flags = delim_args.join(flags)

        return flags
    


    def build_docker_shell(self,
        dict_subset: Dict[str, List[int]],
        **kwargs,
    ) -> str:
        """
        Build the user data to uplod to instances based on the dictionary of 
            dimensional key values to run. 

        Function Arguments
        ------------------
        - dict_dimensional_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template.

        Keyword Arguments
        -----------------
        - **kwargs: passed to build_docker_shell_flags()
        """

        dict_fill = {}
        # get flags
        flags = self.build_docker_shell_flags(dict_subset, **kwargs)
        dict_fill.update({
            self.matchstr_shell_sispeude_cl_flags: flags
        })

        
        # fill the template
        template_out = self.template_docker_shell.fill_template(
            dict_fill
        )

        return template_out



    def build_keyvals_echo_string(self,
        keyname: Union[str, None], 
        values: List[Any],
        delim: str = "\n",
    ) -> Union[str, None]:
        """
        Build a string to 

        Function Arguments
        ------------------
        - keyname: columne header to include in csv. If None, skips
        - values: list of values to print out

        Keyword Arguments
        -----------------
        - delim: delimiter for printed values
        """

        if not isinstance(delim, str):
            return None
        
        # build output string as list
        str_out = (
            list(values).copy()
            if sf.islistlike(values)
            else []
        )

        (
            str_out.insert(0, keyname)
            if isinstance(keyname, str)
            else None
        )

        (
            str_out.append("")
            if len(str_out) > 0
            else None
        )

        if (len(str_out) == 0):
            return None

        str_out = delim.join([str(x) for x in str_out])

        return str_out



    def build_user_data(self,
        dict_dimensional_subset: Dict[str, List[int]],
        instance_launch_index: Union[int, None] = None,
        restrict_partition_to_keys: Union[List[str], None] = None,
    ) -> str:
        """
        Build the user data to uplod to instances based on the dictionary of 
            dimensional key values to run. 

        Function Arguments
        ------------------
        - dict_dimensional_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template
        - instance_launch_index: optional instance launch index to pass
        - restrict_partition_to_keys: in template, restricting partitions to 
            only certain keys?
        """
        
        # dictionaries that will be used to fill templates
        dict_fill_user_data = {}


        ##  SHELL (PASSED TO USER DATA)

        str_docker_shell = self.build_user_data_str_docker_shell(dict_dimensional_subset)
        (
            dict_fill_user_data.update({
                self.matchstr_docker_generate_shell: str_docker_shell
            })
            if str_docker_shell is not None
            else None
        )

        ##  USER DATA

        # cd homedir command
        str_cd_home = self.build_user_data_str_cd_home_dir()
        (
            dict_fill_user_data.update({
                self.matchstr_homedir: str_cd_home
            })
            if str_cd_home is not None
            else None
        )

        # dimensional reads (if passing dimensional subsets through echo > file)
        str_dimensional_reads = self.build_user_data_str_dimensional_reads(
            dict_dimensional_subset,
        )
        (
            dict_fill_user_data.update({
                self.matchstr_dimensional_reads: str_dimensional_reads
            })
            if str_dimensional_reads is not None
            else None
        )

        # docker pull
        str_docker_pull = self.build_user_data_str_docker_pull()
        (
            dict_fill_user_data.update({
                self.matchstr_docker_pull: str_docker_pull
            })
            if str_docker_pull is not None
            else None
        )

        # docker run command
        str_docker_run = self.build_user_data_str_docker_run()
        (
            dict_fill_user_data.update({
                self.matchstr_docker_run: str_docker_run
            })
            if str_docker_run is not None
            else None
        )

        # instance info echo dump
        str_instance_info_echo = self.build_user_data_str_instance_metadata(
            dict_dimensional_subset,
            analysis_id = self.file_struct.id,
            instance_launch_index = instance_launch_index,
        )
        (
            dict_fill_user_data.update({
                self.matchstr_generate_instance_info: str_instance_info_echo
            })
            if str_instance_info_echo is not None
            else None
        )

        # make output directory to mount with docker command
        str_output_mount = self.build_user_data_str_mkdir_instance_mount()
        dict_fill_user_data.update({
            self.matchstr_mkdir_instance_out: str_output_mount
        })

        # S3 copy command
        str_cp_to_s3 = self.build_user_data_str_copy_to_s3(
            dict_dimensional_subset, 
            instance_launch_index,
        )
        (
            dict_fill_user_data.update({
                self.matchstr_copy_to_s3: str_cp_to_s3
            })
            if isinstance(str_cp_to_s3, str)
            else None
        )

        # terminate instance command
        str_terminate_instance = self.build_user_data_str_terminate_instance()
        dict_fill_user_data.update({
            self.matchstr_terminate_instance: str_terminate_instance
        })

        

        ##  SPLIT TO ALLOW USE OF "screen" REMOTELY?

        if self.dict_template_key_screen in self.dict_templates.keys():
            
            screen_script = (
                self.dict_templates
                .get(self.dict_template_key_screen)
                .fill_template(dict_fill_user_data)
            )
            
            # build the screen wrap component and add
            str_screen_wrap = self.build_user_data_str_screen_wrapper(
                screen_script,
                self.dict_template_key_screen
            )
            (
                dict_fill_user_data.update({
                    self.matchstr_screen_shell_dummy: str_screen_wrap
                })
                if str_screen_wrap is not None
                else None
            )

            user_data = self.dict_templates.get(self.dict_template_key_main)
            user_data = user_data.fill_template(
                dict_fill_user_data
            )



        else:
            # fill user data
            user_data = self.template_user_data_raw.fill_template(
                dict_fill_user_data
            )
        

        # run checks


        return user_data
    


    def build_user_data_str_cd_home_dir(self,
    ) -> str:
        """
        template line to cd to a home directory
        """

        out = f"cd {self.dir_instance_home}"

        return out
    


    def build_user_data_str_copy_to_s3(self,
        dict_partition: Union[Dict[str, Any], None],
        launch_index: int, 
        delim_newline: str = "\n",
    ) -> Union[str, None]:
        """
        Copy outputs to S3

        Function Arguments
        ------------------
        - dict_partition: dictionary representing the partition generated by
            the instance
        - launch_index: the launch index for the user data

        Keyword Arguments
        -----------------
        - delim_newline: new line delimitter
        """
        
        bucket = self.config.get("aws_s3.bucket")
        bucket = (
            bucket
            if self.validate_bucket(bucket)
            else None
        )
        return_none = (bucket is None)
        
        # get keys and check
        key_database = self.s3_key_database #config.get("aws_s3.key_database")
        key_logs = self.s3_key_logs #config.get("aws_s3.key_logs")
        key_metadata = self.s3_key_metadata #
        return_none |= any([(x is None) for x in [key_database, key_logs, key_metadata]])
        
        if return_none is None:
            return None
        
        
        ##  PREPARE PATHS AND NAMES
        
        # get database path and get tables to try to copy out
        tables_copy = self.get_tables_to_copy_to_s3()
        lines_out = []
        
        # copy output tables
        for table in tables_copy:
            
            fp_source = os.path.join(self.dir_instance_out_db, f"{table}.csv")
            fp_target = self.get_s3_path_athena_table_file(
                table,
                dict_partition,
                index = launch_index,
            )
            
            comm = f"{self.command_aws} s3 cp '{fp_source}' '{fp_target}'"
            lines_out.append(comm)
        
        # copy instance metadata
        fp_target = self.get_s3_path_instance_metadata()
        comm = f"{self.command_aws} s3 cp '{self.fp_instance_instance_info}' '{fp_target}'"
        lines_out.append(comm)

        # copy log
        fp_target = self.get_s3_path_instance_log()
        comm = f"{self.command_aws} s3 cp \"{self.fp_instance_sisepuede_log}\" \"{fp_target}\""
        lines_out.append(comm)
        
        lines_out = delim_newline.join([str(x) for x in lines_out])
        
        return lines_out



    def build_user_data_str_dimensional_reads(self,
        dict_subset: Dict[str, List[int]],
        delim_newline: str = "\n",
        output_type: str = "echo",
    ) -> str:
        """
        Replace the template line that specified dimensional subsets 
            to run.
        
        Function Arguments
        ------------------
        - dict_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template.
        - manager: AWS manager used for method access

        Keyword Arguments
        -----------------
        - delim_newline: string delimiter used for new lines
        - output_type: 
            * "echo": write the echo statemest for UserData
            * "flags": write python flags for Docker Shell
        """
        
        template_component = []
        output_type = "echo" if (output_type not in ["echo", "flags"]) else output_type
        delim = (
            delim_newline
            if output_type == "echo"
            else " "
        )
        
        for k, v in dict_subset.items():
            if not sf.islistlike(v):
                continue
                
            line = self.build_user_data_str_echo_from_keyvals(
                k,
                v,
                k,
                dir_instance = self.dir_instance_home
            )
            
            template_component.append(line)
        

        template_component = delim.join(template_component)

        return template_component



    def build_user_data_str_docker_pull(self,
        delim_imgname: str = "/",
        delim_newline: str = "\n",
    ) -> str:
        """
        Build the docker pull command
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - delim_imgname: delimiter in AWS ECR image name separating docker 
            account info from image
        - delim_newline: new line delimiter
        """

        # get some key properties
        aws_region = self.aws_region_name
        name_docker_image = self.docker_image_name
        use_ecr = self.use_ecr
        
        # initialize output command - if using ECR, must retrieve credentials on the instance 
        # see https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html
        command_out = []

        if use_ecr:
            std_in = name_docker_image.split(delim_imgname)[0]
            comm_ecr = f"{self.command_aws} ecr get-login-password --region {aws_region}"
            comm_ecr = f"{comm_ecr} | docker login --username AWS --password-stdin {std_in}"

            command_out.append(comm_ecr)


        # build the basic docker command and join the list out
        command_pull = f"docker pull {name_docker_image}"
        command_out.append(command_pull)

        command_out = delim_newline.join(command_out)

        return command_out



    def build_user_data_str_docker_run(self,
    ) -> str:
        """
        Build the docker run command
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """

        # get the image name
        name_docker_image = self.docker_image_name
        fp_instance_sh = self.fp_instance_shell_script
        

        # build the command as a list
        command_out = ["docker run"]
        command_out.append(
            f"-v \"{self.dir_instance_out}:{self.dir_docker_sisepuede_out}\""
        )
        command_out.append(
            f"-w \"{self.dir_docker_sisepuede_python}\""
        )
        command_out.append(
            name_docker_image
        )
        command_out.append(
            f"-c \"$(cat {fp_instance_sh})\""
        )

        command_out = " ".join(command_out)

        return command_out
    


    def build_user_data_str_docker_shell(self,
        dict_dimensional_subset: Dict[str, List[int]],
        **kwargs,
    ) -> str:
        """
        Build the user data to uplod to instances based on the dictionary of 
            dimensional key values to run. 

        Function Arguments
        ------------------
        - dict_dimensional_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template.

        Keyword Arguments
        -----------------
        - **kwargs: passed to build_docker_shell_flags()
        """
        # setup output path
        fp_out = self.fp_instance_shell_script
        str_docker_shell = self.build_docker_shell(dict_dimensional_subset)

        if False:
            # keep here unless it needs to come back
            str_docker_shell = sf.str_replace(
                str_docker_shell,
                {
                    "\"": "\\\"",
                    "!": "\\!"
                }
            )
        # user heredoc: https://linuxopsys.com/topics/create-multiline-strings-in-bash
        str_docker_shell = f"cat > {fp_out} << {self.heredoc_eof}\n{str_docker_shell}\n{self.heredoc_eof}\n"
        
        return str_docker_shell
        


    def build_user_data_str_echo_from_keyvals(self,
        keyname: Union[str, None], 
        values: List[Any],
        fbn_values: str,
        delim: str = "\\n",
        dir_instance: Union[str, None] = None,
        ext: str = "csv",
        return_type: str = "command",
    ) -> Union[str, None]:
        """
        Based on a key and values, build a user data line to keys to pass to a 
            CSV.

        Function Arguments
        ------------------
        - keyname: columne header to include in csv. If None, skips
        - values: list of values to print out
        - fbn_values: base name (excluding directory and extension) of output 
            file

        Keyword Arguments
        -----------------
        - delim: delimiter for printed values
        - dir_instance: optional directory to prepend in front of output file
        - ext: file extension
        - return_type: 
            * "command": return the echo command
            * "file_path": only return the file path out
        """
        
        # build output string as list
        str_echo = self.build_keyvals_echo_string(
            keyname, 
            values,
            delim = delim,
        )

        fp_out = f"{fbn_values}.{ext}"
        fp_out = (
            os.path.join(dir_instance, fp_out)
            if isinstance(dir_instance, str)
            else fp_out
        )
        str_out = f"echo \"{str_echo}\" > \"{fp_out}\""

        return str_out
    


    def build_user_data_str_instance_metadata(self,
        dict_subset: Dict[str, List[int]],
        analysis_id: Union[str, None] = None,
        char_tab: str = "  ",
        delim_elems: str = ",",
        delim_newline: str = "\n",
        instance_launch_index: Union[int, None] = None,
        key_instance_analysis_id: str = "analysis_id",
        key_instance_id: str = "instance_id",
        key_instance_launch_index: str = "launch_index",
    ) -> str:
        """
        Generate shell lines to create instance_id metadata file as YAML. 
            Returns a tuple with two elements:

            (
                lines_user_data,
                fn_instance_data
            )
        
        Function Arguments
        ------------------
        - dict_subset: dictionary mapping dimensional keys to values
            to pass as subset to run via template.

        Keyword Arguments
        -----------------
        - analysis_id: optional analysis ID to pass 
        - char_tab: tab delimiter for use in YAML. By default, uses two spaces.
        - delim_elems: delimiter for stored elements in a list
        - delim_newline: new line delimiter to use
        - instance_launch_index: optional instance launch index to pass
        - key_instance_analysis_id: optional analysis ID to pass as a key in 
            YAML
        - key_instance_id: YAML key for instance ID
        - key_instance_launch_hash: hash to connect 
        """
        env_var = self.shell_env_instance_id
        env_var_assign = env_var.replace("$", "")
        fp_out = self.fp_instance_instance_info

        # initialize instance ID shell variable from instance meta data
        command_get_instance_id = f"{env_var_assign}=$(curl {self.aws_url_instance_id_metadata})"
        
        # add key information
        command_out = (
            ["launch_info:", f"{char_tab}{key_instance_analysis_id}: {analysis_id}"]
            if analysis_id is not None
            else []
        )
        command_out += ["instance:"]
        command_out.append(
            f"{char_tab}{key_instance_id}: {env_var}"
        )
        (
            command_out.append(
                f"{char_tab}{key_instance_launch_index}: {instance_launch_index}"
            )
            if isinstance(instance_launch_index, int)
            else None
        )


        # build scenario information
        for k, v in dict_subset.items():
            if not sf.islistlike(v):
                continue
            
            v_out = ",".join([str(x) for x in v])
            line = f"{char_tab}{k}: {v_out}"
            command_out.append(line)
        
        command_out.append("")
        command_out = delim_newline.join(command_out)
        command_out = f"{command_get_instance_id}{delim_newline}cat > {fp_out} << {self.heredoc_eof}\n{command_out}\n{self.heredoc_eof}\n"

        return command_out
    
    

    def build_user_data_str_mkdir_instance_mount(self,
        permissions: str = "777",
    ) -> str:
        """
        Build the docker pull command
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - permissions: permissions for the output directory
        """

        # get the image name
        dir_out = self.dir_instance_out
        command_out = f"mkdir {dir_out}\nchmod {permissions} {dir_out}"

        return command_out
    


    def build_user_data_str_screen_wrapper(self,
        template_to_wrap: str,
        key: str,
        default_screen_name: str = "checkr",
        delim_newline: str = "\n",
    ) -> str:
        """
        Wrap a template in a screen command by making a dummy shell file
        
        Function Arguments
        ------------------
        - template_to_wrap: template component to dump into dummy shell file 
            shell_script
        - key: name to provide to file

        Keyword Arguments
        -----------------
        - default_screen_name: screen name used when logging in via ssh
        - delim_newline: new line delimeter
        """

        if template_to_wrap is None:
            return None

        # set shell script path
        fp_shell_script =  (
            os.path.join(self.dir_instance_home, f"{key}.sh")
        )

        # get the screen name
        screen_name = self.config.get("aws_ec2.screen_name")
        screen_name = (
            default_screen_name 
            if not isinstance(screen_name, str) 
            else screen_name
        )

        # build the command
        command_out = []
        command_out.append(
            f"cat > {fp_shell_script} << {self.heredoc_eof}\n{template_to_wrap}\n{self.heredoc_eof}\n"
        )

        command_out.append(
            f"su -c \"screen -dmS {screen_name} bash -c 'source {fp_shell_script}'\" ec2-user"
        )

        command_out = delim_newline.join(command_out)

        return command_out
    


    def build_user_data_str_terminate_instance(self,
    ) -> str:
        """
        Build the line to terminate the instance
        """
        aws_region = self.aws_region_name

        comm = f"{self.command_aws} ec2 terminate-instances --instance-ids \"{self.shell_env_instance_id}\""
        comm = f"{comm} --region {aws_region}"
        
        return comm
    


    
    ###########################
    #    SOME AWS COMMANDS    #
    ###########################

    def build_ec2_iam_instance_profile_regex(self,
        delim_name: str = "/",
    ) -> Tuple[re.Pattern, str]:
        """
        Build a regular expression for verifying iam instance profiles. Returns 
            a tuple of the following form
            
            (re_iam_instance_profile, delim_name)
            
        Keyword Arguments
        -----------------
        - delim_name: delimiter in IAM instance profile used to separate ARN from Name
        """
        re_iam_instance_profile = re.compile(f"arn:aws:iam::(.*):instance-profile{delim_name}(.*)")
        
        out = (re_iam_instance_profile, delim_name)
        
        return out
    


    def format_for_ec2_iam_instance_profile(self,
        instance_iam_role: str,
        key_arn: str = "Arn",
        key_name: str = "Name",
        return_arn: bool = True,
        return_name: bool = False,
    ) -> Union[Dict[str, str], None]:
        """
        Transform IAM Instance Profile string to a dictionary for upload with 
            client_ec2.run_instances().
        
        Function Arguments
        ------------------
        - instance_iam_role: role to format 

        Keyword Arguments
        -----------------
        - key_arn: key in output dictionary containing ARN
        - key_name: key in output dictionary containing profile name
        - return_arn: return ARN in the output dictionary?
        - return_name: return name in the output dictionary?
        """
        
        regex_check, delim_name = self.build_ec2_iam_instance_profile_regex()
        
        # see info here: https://stackoverflow.com/questions/41518334/how-do-i-use-boto3-to-launch-an-ec2-instance-with-an-iam-role
        if not isinstance(instance_iam_role, str):
            return None
        
        if regex_check.match(instance_iam_role) is None:
            return None
        
        # split into ARN and name, extract name (2nd element)
        list_components = instance_iam_role.split(delim_name)
        if len(list_components) != 2:
            return None
        
        dict_out = {}
        (
            dict_out.update({key_arn: instance_iam_role})
            if return_arn
            else None
        )
        (
            dict_out.update({key_name: list_components[1]})
            if return_name
            else None
        )

        return dict_out



    def format_for_ec2_tags(self,
        dict_tags: Dict[str, Any],
    ) -> Union[List[Dict[str, Any]], None]:
        """
        Transform configuration tags as a list of dictionaries for upload with 
            client_ec2.run_instances()
        """
        
        if not isinstance(dict_tags, dict):
            return None
        
        tags_out = []
        
        for k, v in dict_tags.items():
            tags_out.append(
                {
                    "Key": k,
                    "Value": v
                }
            )
            
        return tags_out
    


    def format_table_name_for_athena(self,
        table_name: str,
    ) -> str:
        """
        Format a table name for Athena using the analysis run id
        """
        
        appendage = self.get_time_hash_for_table()
        
        out = f"`{table_name}_{appendage}`"
        out = out.lower()
        
        return out



    def get_buckets(self,
        client_s3: Union[boto3.client, None] = None,
    ) -> Union[List[str], None]:
        """
        Retrieve buckets associated with an S3 client
        """

        client_s3 = (
            self.client_s3
            if client_s3 is None
            else client_s3
        )

        try:
            buckets = client_s3.list_buckets()
        except Exception as e:
            self._log(
                f"Error retrieving buckets using AWSManager.get_buckets(): {e}",
                type_log = "error",
            )

            return None
        
        bucket_names = [x.get("Name") for x in buckets.get("Buckets")]

        return bucket_names
    


    def get_path_instance_log(self,
        dir_out: str,
    ) -> str:
        """
        Retrieve the instance log file path

        Function Arguments
        ------------------
        - dir_out: output directory to store the file
        """
        fp_out = os.path.join(
            dir_out,
            os.path.basename(self.file_struct.fp_log_default)
        )
        
        return fp_out
    


    def get_path_instance_metadata(self,
        dir_out: str,
    ) -> str:
        """
        Build the file path for instance metadata.

        Function Arguments
        ------------------
        - dir_out: output directory to store the file
        """
        env_var = self.shell_env_instance_id
        fp_out = os.path.join(dir_out, f"instance_info_{env_var}.yaml")

        return fp_out
    


    def get_path_instance_output_db(self,
        dir_instance_out: Union[str, None] = None,
    ) -> str:
        """
        Get the directory on the instance containing output from the Docker
            image. This is the CSV database path, which is contained within the
            broader run package.

        NOTE: contained within get_path_instance_output_run_package()
        """
        
        dir_instance_out = (
            self.dir_instance_out
            if dir_instance_out is None
            else dir_instance_out
        )

        dir_instance_db = self.file_struct.fp_base_output_raw
        dir_instance_db = dir_instance_db.replace(
            self.file_struct.dir_out, 
            dir_instance_out
        )
        
        return dir_instance_db
    


    def get_path_instance_output_run_package(self,
        dir_instance_out: Union[str, None] = None,
    ) -> str:
        """
        Get the directory on the instance containing the run package for 
            SISEPUEDE, including the output database, pickle, and log.
        """
        
        dir_instance_out = (
            self.dir_instance_out
            if dir_instance_out is None
            else dir_instance_out
        )

        dir_instance_db = self.file_struct.dir_base_output_raw
        dir_instance_db = dir_instance_db.replace(
            self.file_struct.dir_out, 
            dir_instance_out
        )
        
        return dir_instance_db
    


    def get_time_hash_for_table(self,
    ) -> str:
        """
        Get the component of the analysis id that is prepended to runs
        """
        id_prependage = self.file_struct.regex_template_analysis_id.pattern.replace("(.+$)" , "")
        id_str = self.file_struct.id
        
        dict_repl = dict(
            (x, "") for x in ["-", ";", ":", ".", "T", id_prependage]
        )
        
        id_str = sf.str_replace(id_str, dict_repl)
        
        return id_str
    


    def get_regions(self,
        regions: Union[List[str], str, None] = None,
        delim: str = ",",
    ) -> Union[List[str], None]:
        """
        Return valid regions from a string or `delim` delimitted list. If 
            None, calls from configuration.
        """
        
        regions_config = self.config.get("experiment.regions")
        
        if (regions is None) & (regions_config is None):
            return None
        
        regions = (
            regions_config
            if regions is None
            else regions
        )
        
        regions = (
            regions.split(delim)
            if isinstance(regions, str)
            else regions
        )
        
        
        regions = [self.regions.return_region_or_iso(x) for x in regions]
        regions = [x for x in self.regions.all_regions if x in regions]
        
        return regions
    



    def get_s3_bucket_and_key_from_address(self,
        address: str,
        sep: str = "/",
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Return a tuple giving the bucket and key from a full s3 address
        """
        # check inputs
        return_none = not isinstance(address, str)
        return_none |= not isinstance(sep, str)
        if return_none:
            return None
        
        # use bucket as splitter
        
        bucket = f"{self.s3_bucket}{sep}"
        tup = address.split(bucket)
        key = (
            tup[1]
            if len(tup) > 1
            else None
        )
        
        out = (self.s3_bucket, key)
        
        return out


    


    def get_s3_path_athena_output(self,
        return_type: str,
    ) -> Union[str, None]:
        """
        Build the output path for the database on S3
        
            NOTE: All tables are *within* the database path
        
        Function Arguments
        ------------------
        - return_type: "database", "logs", "metadata", or "queries
        """
        
        # get bucket
        bucket = self.config.get("aws_s3.bucket")
        bucket = (
            bucket
            if self.validate_bucket(bucket)
            else None
        )
        return_none = (bucket is None)
        
        # get keys
        key_database = self.config.get("aws_s3.key_database")
        key_logs = self.config.get("aws_s3.key_logs")
        key_metadata = self.config.get("aws_s3.key_metadata")
        key_queries = self.config.get("aws_s3.key_queries")
        
        return_none |= ((key_database is None) & (return_type == "database"))
        return_none |= ((key_logs is None) & (return_type == "logs"))
        return_none |= ((key_metadata is None) & (return_type == "metadata"))
        return_none |= ((key_queries is None) & (return_type == "queries"))
        
        if return_none:
            return None
        
        # map key type to specified configuration key
        dict_key_S3 = {
            "database": key_database,
            "logs": key_logs,
            "metadata": key_metadata,
            "queries": key_queries,
        }
        key = dict_key_S3.get(return_type)
        
        address_out = os.path.join(bucket, key, self.file_struct.id_fs_safe)
        address_out = f"s3://{address_out}"
        # required for partitioning and MSCK REPAIR TABLE
        address_out = address_out.lower()

        return address_out
    


    def get_s3_path_athena_table(self,
        table_name: str,
    ) -> str:
        """
        Get the path to the Athena table with table name `table_name`.

        NOTE: Sets to lower case to allow partitioning using MSCK REPAIR TABLE
        (see https://docs.aws.amazon.com/athena/latest/ug/partitions.html)
        """

        out = os.path.join(self.s3p_athena_database, table_name.lower())

        return out
    


    def get_s3_path_athena_table_file(self,
        table_name: str,
        dict_partitions: Union[Dict[str, Any], None],
        ext: str = "csv",
        file_name: str = "data",
        index: Union[Any, None] = None,
    ) -> Union[str, None]:
        """
        Build the output path for the database on S3
        
            NOTE: All tables are *within* the database path
        
        Paths are stored as

        s3://BUCKET/key_database/table_name/(partitions).../table_name_$(ind)/data.csv

        Function Arguments
        ------------------
        - table_name: table name
        - dict_partitions: dictionary of dimensions to partition on. If None, no
            partitioning is performed

        Keyword Arguments
        -----------------
        - ext: file extension to use
        - file_name: file name on S3 of CSV
        - index: optional index to add to the table (such as launch index)
        """

        if not isinstance(table_name, str):
            return None

        # get basename
        base_name = (
            table_name
            if not isinstance(index, int)
            else f"{table_name}_{index}"
        )
        base_name = base_name.lower()
        file_name = f"{file_name}.{ext}"

        # initialize as base name
        address_out = [
            #self.s3p_athena_database, table_name
            self.get_s3_path_athena_table(table_name)
        ]

        if isinstance(dict_partitions, dict) & (self.athena_partitions_ordered is not None):
            for key in self.athena_partitions_ordered:
                val = dict_partitions.get(key)

                # skip if value not found or the value is a list-like object with not 1-element
                continue_q = (val is None)
                continue_q |= (sf.islistlike(val) & len(val) != 1)
                if continue_q:
                    continue
                
                # if listlike, take only element
                val = val[0] if sf.islistlike(val) else val
                address_out.append(f"{key}={val}")

        # add output table and join
        address_out.extend([base_name, file_name])
        address_out = os.path.join(*address_out)

        return address_out
    


    def get_s3_path_instance_log(self,
    ) -> Union[str, None]:
        """
        Build the output path for the instance log file to store
        
            NOTE: All tables are *within* the database path
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - ext: file extension to use
        - index: optional index to add to the table (such as launch index)
        """

        fp_out = os.path.join(
            self.s3p_run_log,
            f"log_{self.shell_env_instance_id}.log"
        )

        return fp_out
    


    def get_s3_path_instance_metadata(self,
    ) -> Union[str, None]:
        """
        Build the output path for the instance metadata file to store
        
            NOTE: All tables are *within* the database path
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - ext: file extension to use
        - index: optional index to add to the table (such as launch index)
        """

        fp_out = os.path.join(
            self.s3p_run_metadata,
            os.path.basename(self.fp_instance_instance_info)
        )

        return fp_out
    


    def get_s3_path_query_output(self,
        output_type: str,
    ) -> Union[str, None]:
        """
        Return the path of queries that are output. 

        Function Arguments
        ------------------
        - output_type: specification of file type within query S3 key to build.
            Valid options are:

            * "input": returns path to query-modified SISEPUEDE input table 
            * "output": returns path to query-modified SISEPUEDE input table
            * "create_input": path to the create input table query output
            * "create_output": path to the create output table query output
            * "repair_input": path to the repair input table query
            * "repair_output": path to the repair output table query

        Returns None if invalid specification is found.
        """

        # now, build output path
        address_out = self.dict_s3p_query_fbns_by_type.get(output_type)
        if address_out is None:
            return address_out

        address_out = os.path.join(self.s3p_athena_queries, f"{address_out}.csv")#HEREHERE

        return address_out



    def get_tables_to_copy_to_s3(self,
        tables: Union[str, List[str], None] = None,
        delim: str = ",",
    ) -> Union[List[str], None]:
        """
        Using a lit of tables (string delimited by delim or list or names), 
            return the list of tables to extract from the instance.
        """
        tables = (
            self.config.get("aws_s3.tables_copy")
            if tables is None
            else tables
        )
        
        tables = (
            tables.split(delim)
            if isinstance(tables, str)
            else tables
        )
        
        tables = (
            tables
            if sf.islistlike(tables)
            else None
        )
        
        return tables



    def launch_instance(self,
        user_data: Union[str, None],
        ami: Union[str, None] = None,
        iam_instance_profile: Union[str, None] = None,
        instance_type: Union[str, None] = None,
        key_name: Union[str, None] = None,
        max_count: int = 1,
        min_count: int = 1,
        name: Union[str, None] = None,
        security_group: Union[str, None] = None,
        subnet: Union[str, None] = None,
        tags: Union[Dict[str, str], None] = None,
    ) -> Union[Dict, None]:
        """
        Launch an instance 
        
        Function Arguments
        ------------------
        - user_data: string of user data to use on launch. If None, launches 
            with no user data.

        Keyword Arguments
        -----------------
        - ami: optional image id. If None, calls from configuration
        - iam_instance_profile: optional iam instance profile to use on launch.
            If None, calls from configuration
        - instance_type: optional instance type. If None, calls from 
            configuration
        - key_name: optional key pair name to pass. If None, calls from 
            configuration
        - max_count: maximum number of instances to launch
        - min_count: minimum number of instances to launch
        - name: optional name for the instance
        - security_group: optional security group id. If None, calls from 
            configuration
        - subnet: optional subnet id. If None, calls from configuration
        - tags: optional tags to specify (as dictionary). None, calls from 
            configuration
        """
        
        ##  SOME CHECKS
        
        return_none = not isinstance(user_data, str)
        return_none |= not isinstance(max_count, int)
        return_none |= not isinstance(min_count, int)
        
        if return_none:
            return None
        
        
        ##  EC2 LAUNCH CONFIG
        
        ami = (
            self.config.get("aws_ec2.image")
            if not isinstance(ami, str)
            else ami
        )
        
        iam_instance_profile = (
            self.config.get("aws_ec2.iam_instance_profile")
            if iam_instance_profile is None
            else iam_instance_profile
        )
        iam_instance_profile = self.format_for_ec2_iam_instance_profile(
            iam_instance_profile
        )

        instance_type = (
            self.config.get("aws_ec2.instance_type")
            if not isinstance(instance_type, str)
            else instance_type
        )

        key_name = (
            self.config.get("aws_ec2.key_name")
            if not isinstance(key_name, str)
            else key_name
        )
        
        security_group = (
            self.config.get("aws_general.security_group")
            if not isinstance(security_group, str)
            else security_group
        )
        
        subnet = (
            self.config.get("aws_general.subnet")
            if not isinstance(subnet, str)
            else subnet
        )
        
        tags = (
            self.config.get("aws_general.tag")
            if not isinstance(tags, Dict)
            else tags
        )
        (
            tags.update({"Name": name})
            if isinstance(name, str)
            else None
        )
        
        list_tag_dicts = self.format_for_ec2_tags(tags)

        
        # initialize output and
        out = None

        try:
            out = self.client_ec2.run_instances(
                IamInstanceProfile = iam_instance_profile,
                ImageId = ami,
                InstanceType = instance_type,
                KeyName = key_name,
                MinCount = min_count,
                MaxCount = max_count,
                SecurityGroupIds = [security_group],
                SubnetId = subnet,
                TagSpecifications = [{
                    "ResourceType": "instance",
                    "Tags": list_tag_dicts
                }],
                UserData = user_data.encode("ascii")
            )
        
        except Exception as e:
            
            self._log(
                f"Error occured trying to launch instance using AWSManager.launch_instance(): {e}",
                type_log = "error",
            )
            
            return out
        
        return out

    


    def validate_bucket(self,
        bucket_name: str
    ) -> bool:
        """
        Checks if bucket_name is valid and present in S3
        """
        
        out = (bucket_name in self.s3_buckets)
        return out





    ##################################
    ###                            ###
    ###    EXPERIMENT FUNCTIONS    ###
    ###                            ###
    ##################################

    def build_athena_sql_query_component_primary_keys_all(self,
        delim: str = ",",
        n_regions: Union[int, None] = None,
        table_base: str = "output",
    ) -> Union[str, Dict[str, str]]:
        """
        Using a list of SISEPUEDE fields, setup the retrieval queries.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - delim: delimiter used in fields_retrieve if specified as string
        - n_regions: number of regions to require for identifying primary
            keys. If None, defaults to len(self.get_regions()) from config
        - table_base: base table for primary key values; must be present in 
            AWSManager.dict_input_output_to_table_names().keys()
        """

        # get the database name and table names
        database_name = f"\"{self.athena_database}\""
        table_name = self.dict_input_output_to_table_names.get(table_base)
        table_name_db = self.format_table_name_for_athena(table_name)
        table_name_db = table_name_db.replace("`", "\"")

        n_regions = (
            len(self.get_regions())
            if not isinstance(n_regions, int)
            else n_regions
        )

        

        query = f"""
        SELECT \"{self.key_primary}\"
        FROM (
            SELECT DISTINCT \"{self.key_primary}\", \"{self.key_region}\" from {database_name}.{table_name_db}
        )
        GROUP BY \"{self.key_primary}\"
        HAVING COUNT(\"{self.key_region}\") = {n_regions}
        """

        return query



    def build_athena_query_create_table(self,
        table_name: str,
        schema: str, 
        address_s3: Union[str, None] = None,
    ) -> str:
        """
        Build a create table query for target athena database. 
        
        Function Arguments
        ------------------
        - table_name: name of table to create
        - schema: table schema used in create table argument
        
        Keyword Arguments
        -----------------
        - address_s3: address on S3 where the table is stored. Set to None
            to keep with AWSManager structure
        """
        
        ##  INITIALIZE BASE QUERY AND S3 ADDRESS

        # get the output address
        address_s3 = (
            self.get_s3_path_athena_table(table_name)
            if not isinstance(address_s3, str)
            else address_s3
        )

        database_name = f"`{self.athena_database}`"
        
        # initialize query, then add some components below
        table_name_out = self.format_table_name_for_athena(table_name)
        query_create_table = f"CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name_out}"
        query_create_table = f"{query_create_table} (\n{schema}\n)"
        

        ##  PREPARE CONDITIONS

        # appendages have to be ordered
        appends_ordered = [
            "ROW FORMAT",
            #"WITH SERDEPROPERTIES",
            #"STORED AS",
            "LOCATION",
            # TBLPROPERTIES follows everything else; see https://docs.aws.amazon.com/athena/latest/ug/vpc-flow-logs.html for example
            "TBLPROPERTIES"
        ]
        
        (
            appends_ordered.insert(0, "PARTITIONED BY")
            if self.athena_partitions_ordered is not None
            else []
        )
        
        # get the partition string
        str_partition = self.build_athena_sql_table_schema_for_inputs_outputs(
            self.athena_partitions_ordered,
            None,
            partition_field_appendage = None,
        )
        str_partition = (
            f"(\n{str_partition[0]}\n)"
            if str_partition is not None
            else str_partition
        )

        # add the string for row formatting
        str_row_format = f"""DELIMITED
            FIELDS TERMINATED BY ','
            ESCAPED BY '\\\\'
            LINES TERMINATED BY '\\n'
        """


        ##  MAP COMPONENTS TO VALUES, THEN APPEND TO QUERY

        # map each query component to its value
        dict_query_components = {
            "PARTITIONED BY": str_partition, 
            "LOCATION": f"'{address_s3}'",
            "ROW FORMAT": str_row_format,
            # see https://docs.aws.amazon.com/athena/latest/ug/lazy-simple-serde.html for info on lazy
            "ROW FORMAT SERDE": "'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'",
            "STORED AS": "TEXTFILE",
            "TBLPROPERTIES": "(\"skip.header.line.count\"=\"1\")",
            "WITH SERDEPROPERTIES": f"""(
                'separatorChar' = ',',
                'quoteChar' = '\\"',
                'escapeChar' = '\\\\'
            )
            """,
        }
        
        for k in appends_ordered:
            v = dict_query_components.get(k)
            query_create_table = f"{query_create_table}\n{k} {v}"

        query_create_table = f"{query_create_table};"
        
        return query_create_table
    


    def build_athena_query_repair_table(self,
        table_name: str,
    ) -> str:
        """
        Build the MSCK REPAIR TABLE query that indexes partitions
        
        Function Arguments
        ------------------
        - table_name: base table name used to generate ttable
        """
        
        database_name = f"`{self.athena_database}`"

        # get table name and write query
        query = self.format_table_name_for_athena(table_name)
        query = f"MSCK REPAIR TABLE {database_name}.{query};"
        
        return query



    def build_athena_sql_table_schema_for_inputs_outputs(self,
        index_fields_ordered: List[str],
        table_type: str,
        dict_fields_to_dtype_sql: Union[Dict[str, Union[str, type]], None] = None,
        fields_data_integer: Union[List[str], None] = None,
        float_as_double: bool = False,
        model_attributes: Union[ma.ModelAttributes, None] = None,
        partition_field_appendage: Union[str, None] = "dummy",
        sep: str = ",\n",
        sql_type_object_default: str = "STRING",
    ) -> Tuple[str, List[str]]:
        """
        Generate an SQL schema from a data frame. For use in automating table
            setup in remote SQL or SQL-like databases.
        
        Returns a two-ple of the following form:
        
            (schema_out, fields_out)
    
    
        Function Arguments
        ------------------
        - index_fields_ordered: ordered index fields, which are prepended to all
            input/output fields
        - table_type: "input" or "output"
        
        Keyword Arguments
        -----------------
        - dict_fields_to_dtype_sql: dictionary mapping fields to specific SQL 
            data types; keys are fields and dtypes are SQL data types.
            * If None, infers from DataFrame dtype
        - fields_data_integer: optional data fields to store as integer instead 
            of float
        - model_attributes: Model Attributes object used to identify input and
            output variables
        - partition_field_appendage: appendage to add to output fields included
            in the table that are used to partition the Athena db. If None, no
            appendage is added.
        - sep: string separator to use in schema (", " or "\n, ", should always 
            have a comma)
        - sql_type_object_default: default data type for fields specified as 
            object (any valid SQL data type)
        """

        ##  INITIALIZATION AND CHECKS

        # initialize model attributes
        model_attributes = (
            self.model_attributes
            if model_attributes is None
            else model_attributes
        )

        # initialize fields, schema out, and fields that were successfully pushed to the schema
        return_none = any(
            [
                (x not in model_attributes.sort_ordered_dimensions_of_analysis) 
                for x in index_fields_ordered
            ]
        )
        return_none |= not sf.islistlike(index_fields_ordered)
        if return_none:
            return None
        

        ##  SECONDARY INIT
    
        dict_dtypes_indices = model_attributes.dict_dtypes_doas
        fields_data_integer = (
            [] 
            if not sf.islistlike(fields_data_integer) 
            else fields_data_integer
        )

        schema_out = []
        fields_out = []
        
        
        fields = sorted(list(index_fields_ordered))
        if table_type in ["input", "output"]:
            fields += (
                sorted(model_attributes.all_variable_fields_output)
                if table_type.lower() == "output"
                else sorted(model_attributes.all_variable_fields_input)
            )

        # some data type dictionaries that are used
        dict_dtypes_to_sql_types = {
            "string": "string",
            "o": "string", 
            "float64": "double",
            "int64": "int",
        }
    
        for i, field in enumerate(fields):
            
            # 1. try getting sql data type from indexing dictionary; if not, 
            # 2. check if specified as integer; if not
            # 3. specify as float
            dtype_sql = dict_dtypes_indices.get(field)
            
            if dtype_sql is None:
                
                dtype_sql = (
                    "int64"  
                    if field in fields_data_integer
                    else "float64"
                )
            
            # skip if failed
            dtype_sql = dict_dtypes_to_sql_types.get(dtype_sql, "string")
            if dtype_sql is None:
                continue

            # finally, if the field is used to partition, append a dummy 
            field_name = (
                f"{field}_{partition_field_appendage}"
                if (field in self.athena_partitions_ordered) & isinstance(partition_field_appendage, str)
                else field
            )
                
            # otherwise, build schema
            schema_out.append(f"{field_name} {dtype_sql}")
            fields_out.append(field)
        
        schema_out = str(sep).join(schema_out)
        
        return schema_out, fields_out
    


    def build_athena_sql_query_for_retrieval(self,
        delim: str = ",",
        fields_retrieve: Union[List[str], str, None] = None,
        fields_retrieval_key: str = "field",
        key_model_inputs: str = "model_inputs",
        key_model_outputs: str = "model_outputs",
        key_model_primary_all: str = "primary_keys_all",
        n_regions: Union[int, None] = None,
        order_by: bool = True,
        return_type: str = "composite",
        table_name_intermediate: str = "merge_1",
    ) -> Union[str, Dict[str, str]]:
        """
        Using a list of SISEPUEDE fields, setup the retrieval queries.
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - delim: delimiter used in fields_retrieve if specified as string
        - fields_retrieve: optional specification of fields to retrieve from
            query. 
            Can be specified as delimited string (using delim) or file path
        - fields_retrieval_key: key used to read fields from `fields_retrieval`
            if specified as a file
        - key_model_inputs: model inputs table key in output dictionary + 
            temporary table name in SQL query
        - key_model_outputs: model outputs table  key in output dictionary + 
            temporary table name in SQL query
        - key_model_primary_all: intersctinal primary keys key in output 
            dictionary + temporary table name in SQL query
        - n_regions: number of regions to filter on when identifying primary
            key values to keep
        - order_by: order table by indices? Can increase query execution time
        - return_type: 
            * "composite": return a composite query
            * "input": return input query only
            * "output": return output query only
            * "primary_keys_all": return the query for only primary keys
            * "query_dictionary": return queries in a dictionary with keys
                - key_model_inputs
                - key_model_outputs
                - key_model_primary_all
        - table_name_intermediate: intermediate table name for staged merge in 
            query
        """
        
        ##  INITIALIZATION AND CHECKS
        
        fields_retrieve = (
            self.config.get("aws_athena.fields_retrieve")
            if not isinstance(fields_retrieve, str)
            else fields_retrieve
        )
        return_none = (fields_retrieve is None)

        if return_none:
            return None
        
        # set order by 
        order_by_internal = order_by & (return_type != "composite")
       
    
        ##
        
        # retrieve fields, which can be specified in a file
        fields_retrieve = sf.get_dimensional_values(
            fields_retrieve,
            fields_retrieval_key,
            delim = delim,
            return_type = str,
        )
        fields_retrieve = [
            x for x in fields_retrieve
            if x in (self.model_attributes.all_variable_fields_input)
            or x in (self.model_attributes.all_variable_fields_output)
        ]

        
        # build input table query and return if desired
        query_input = self.build_athena_sql_query_for_retrieval_by_io(
            "input", 
            fields_retrieve,
            order_by = order_by_internal,
        )
        if return_type == "input":
            return query_input
        
        
        # build output table query and return if desired
        query_output = self.build_athena_sql_query_for_retrieval_by_io(
            "output", 
            fields_retrieve,
            order_by = order_by_internal,
        )
        if return_type == "output":
            return query_output
        
        
        # build primary filter and return if desired
        query_primary_filter = self.build_athena_sql_query_component_primary_keys_all(
            n_regions = n_regions,
        )
        if return_type == "primary_keys_all":
            return query_primary_filter
        
        
        # if dictionary, return
        dict_out = {
            key_model_inputs: query_input,
            key_model_outputs: query_output,
            key_model_primary_all: query_primary_filter,
        }
        if return_type == "query_dictionary":
            return dict_out
        
        
        
        ##  OTHERWISE, BUILD COMPOSITE QUERY

        # set up shared index fields - output
        table_name_input = self.dict_input_output_to_table_names.get("input")
        table_name_output = self.dict_input_output_to_table_names.get("output")
        fields_ind = self.dict_keys_table_name_to_index_ordered.get(table_name_input).copy()
        fields_ind = [
            x for x in fields_ind if x in
            self.dict_keys_table_name_to_index_ordered.get(table_name_output)
        ]
        
        # get fields to select as index
        fields_full_select = [f"{key_model_outputs}.\"{x}\"" for x in fields_ind]
        fields_full_select = delim.join(fields_full_select)
        
        fields_full_select_retrieve = squ.join_list_for_query(
            fields_retrieve, 
            delim  = delim, 
            quote = "\""
        )
        fields_full_select = f"{fields_full_select}{delim}{fields_full_select_retrieve}"
              
        # setup merge columns
        fields_merge = [
            f"{table_name_intermediate}.\"{x}\" = {key_model_outputs}.\"{x}\""
            for x in fields_ind
        ]
        fields_merge_str = " AND ".join(fields_merge)
        
        # do some quick cleaning
        query_input = query_input.replace(";", "")
        query_output = query_output.replace(";", "")
        query_primary_filter = query_primary_filter.replace(";", "")
        

        # the query filters for primary ids that succeeded
        # then merges inputs and outputs with those
        query = f"""WITH {key_model_primary_all} AS ({query_primary_filter}), 
        {key_model_inputs} AS ({query_input}),
        {key_model_outputs} AS ({query_output}),
        {table_name_intermediate} AS (
         SELECT {key_model_inputs}.* FROM {key_model_primary_all}
         INNER JOIN {key_model_inputs}
         ON {key_model_inputs}.{self.key_primary} = {key_model_primary_all}.{self.key_primary}
        )
        SELECT
        {fields_full_select}
        FROM {table_name_intermediate}
        INNER JOIN {key_model_outputs}
        ON {fields_merge_str}"""
        
        # order by indexing fields?
        if order_by:
            query_order_by = squ.join_list_for_query(
                fields_ind, 
                delim  = ",", 
                quote = "\""
            )
            query_order_by = f"ORDER BY ({query_order_by})"
            query = f"{query}\n{query_order_by}"

        query = f"{query};"

        return query



    def build_athena_sql_query_for_retrieval_by_io(self,
        table_type: str,
        fields_retrieve: List[str],
        order_by: bool = True,
    ) -> str:
        """
        Build a query to select from a table by table type

        Function Arguments
        ------------------
        - table_type: input or output
        - fields_retrieve: fields to retrieve

        Keyword Arguments
        -----------------
        - order_by: order output by indices? Can increase query execution time
        """

        ##  INITIALIZATION

        table_type = (
            "output"
            if table_type not in self.dict_input_output_to_table_names.keys()
            else table_type
        )
        domain = (
            self.model_attributes.all_variable_fields_input
            if table_type == "input"
            else self.model_attributes.all_variable_fields_output
        )

        database_name = f"\"{self.athena_database}\""


        ##  GET NAME AND FIELDS

        table_name = self.dict_input_output_to_table_names.get(table_type)
        fields_ind = self.dict_keys_table_name_to_index_ordered.get(table_name).copy()
        fields = fields_ind + sorted([x for x in domain if x in fields_retrieve])
        
        table_name_db = self.format_table_name_for_athena(table_name)
        table_name_db = table_name_db.replace("`", "\"") # clean tick marks 
        query = squ.join_list_for_query(
            fields, 
            delim  = ",", 
            quote = "\""
        )

        
        ##  BUILD QUERY

        query = f"SELECT {query} FROM {database_name}.{table_name_db}"

        # order by indexing fields?
        if order_by:
            query_order_by = squ.join_list_for_query(
                fields_ind, 
                delim  = ",", 
                quote = "\""
            )
            query_order_by = f"ORDER BY ({query_order_by})"
            query = f"{query} {query_order_by}"

        query = f"{query};"

        return query 



    def build_experiment(self,
        dict_experimental_components: Union[Dict[str, Union[Dict[str, List[int]], List[int]]], None] = None,
        regex_config_parts: Union[re.Pattern, None] = None,
        return_type: str = "list",
    ) -> Union[Dict[str, int], None]:
        """
        Build the experimental design to split. Returns a list of unique
            key_primary (generally primary_id) values to run.
        
        Keyword Arguments
        -----------------
        - dict_experimental_components: dictionary of components with keys
            matching regex_config_parts
        - regex_config_parts: regular expression for determining experimental
            components. If None, defaults to 
            self.regex_config_experiment_components
        - return_type: "list" or "indexing_data_frame"
        """

        ##  CHECKS
        
        # can't build without db
        return_none = (self.primary_key_database is None)
        
        # get the dictionary, return None if invalidly specified
        dict_experimental_components = self.get_experimental_dictionary(
            dict_experimental_components = dict_experimental_components,
            regex_config_parts = regex_config_parts,
        )
        return_none |= (dict_experimental_components is None)
    
        if return_none:
            return None
        
        
        ##  BUILD PRIMARY KEYS

        # initialize some elements
        return_type = (
            return_type
            if return_type in ["list", "indexing_data_frame"]
            else "list"
        )
        set_primaries = set({})
        
        for k, v in dict_experimental_components.items():

            keys_primary_cur = (
                self.primary_key_database.get_indexing_dataframe_from_primary_key(
                    v.get(self.key_primary)
                )
                if self.key_primary in v.keys()
                else self.primary_key_database.get_indexing_dataframe(v)
            )

            if keys_primary_cur is None:
                continue
                
            keys_primary_cur = keys_primary_cur[self.key_primary]
            set_primaries |= set(keys_primary_cur)
        
        # built final output
        set_primaries = sorted(list(set_primaries))
        set_primaries = (
            self.primary_key_database.get_indexing_dataframe_from_primary_key(set_primaries)
            if return_type == "indexing_data_frame"
            else set_primaries
        )
        
        return set_primaries
    


    def check_queries_success(self,
        execution_ids: Union[List[str], str],
        max_execution: int = 10,
        return_single_val_on_str: bool = True,
        sleep_time: int = 5,
        state_queued: str = "QUEUED",
        state_running: str = "RUNNING",
        state_succeeded: str = "SUCCEEDED",
    ) -> Dict[str, bool]:
        """
        Check whether or not a query has succeeded. Will use thread.sleep to 
            occupy, at most, sleep_time*max_execution seconds. Adapted from 
            
            https://www.learnaws.org/2022/01/16/aws-athena-boto3-guide/#how-to-create-a-new-database-in-athena
        
        Function Arguments
        ------------------
        - execution_ids: list of execution ids to check or a single execution id
        
        Keyword Arguments
        -----------------
        - max_execution: maximum number of times to check
        - return_single_val_on_str: if execution_ids is a string (singleton), 
            return only the value?
        - sleep_time: time to sleep (in seconds) while waiting for status 
            updates
        - state_queued: AWS state returned if query is queued
        - state_running: AWS state returned if query is running
        - state_succeeded: AWS state returned if query succeeeded
        """
        
        exec_ids_entered_as_str = isinstance(execution_ids, str) 
        
        # check execution ids
        execution_ids = (
            [execution_ids] 
            if exec_ids_entered_as_str
            else execution_ids
        )
        execution_ids = (
            execution_ids 
            if sf.islistlike(execution_ids) 
            else None
        )
        return_none = execution_ids is None
        
        # check max execution
        return_none = not isinstance(return_none, int)
        if return_none:
            return None
        
        
        # initialize states
        dict_states = {}
        for x in execution_ids:
            state_cur = self.get_query_state(x)
            dict_states.update({x: state_cur})

        all_states = set(dict_states.values())
        states_continue = set({state_running, state_queued})
        ind_iter = 0
        
        while (ind_iter < max_execution) & (len(all_states & states_continue) > 0):
            
            ind_iter += 1
            
            for exec_id, state in dict_states.items():
                
                if state == state_succeeded:
                    continue
                
                state_cur = self.get_query_state(exec_id)
                dict_states.update({exec_id: state_cur})
            
            all_states = set(dict_states.values())

            time.sleep(sleep_time)
            
            
        # return bools - if entered as a string originally, the dict will have length one
        out = dict(
            (x, (dict_states.get(x) == state_succeeded))
            for x in execution_ids
        )
        out = (
            list(out.values())[0]
            if exec_ids_entered_as_str
            else out
        )

        return out
    


    def execute_create_table_athena_query(self,
        fields_data_integer: Union[List[str], None] = None,
        s3p_out: Union[str, None] = None,
        **kwargs
    ) -> Dict:
        """
        Execute an athena create table query based on the model id. Returns
            the response from AWS as a dictionary
        
        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - fields_data_integer: optional data fields to store as integer instead 
            of float
        - s3p_out: path on s3 containing output database information
        - **kwargs: passed to self.check_queries_success
        """
        
        ##  SOME INITIALIZATION
        
        # output path for queries
        s3p_out = (
            self.s3p_athena_queries
            if not isinstance(s3p_out, str)
            else s3p_out
        )
        
        # initialize some dictionaries
        dict_table_to_schema_type = self.dict_table_names_to_input_output
        dict_query_info = {}
        
        
        ##  BUILD QUERIES AND TABLES
        
        # build queries
        for k, v in dict_table_to_schema_type.items():
            
            dict_query_info.update({v: {}})
            response_create = None
            response_repair = None
            
            # pull index keys from database
            keys_index_ordered = self.dict_keys_table_name_to_index_ordered.get(k)
            tup = self.build_athena_sql_table_schema_for_inputs_outputs(
                keys_index_ordered,
                v,
                fields_data_integer = fields_data_integer,
            )
            

            ##  TRY TO CREATE THE TABLE

            query_create = self.build_athena_query_create_table(k, tup[0])
            response_create = self.exec_query(
                query_create,
                self.get_s3_path_query_output(f"create_{v}"),
                log_msg_supplement = " CREATE TABLE",
            )
            if response_create is None:
                continue

            # check the creation query
            qe_id = response_create.get(self.athena_reponse_key_exec_id)
            success = self.check_queries_success(qe_id, **kwargs)

            # log the outcome and update output and update the output dictionary
            if not success:
                self._log(
                    f"CREATE TABLE {k} with QueryExecutionId {qe_id} has not successfully completed in time allotted for checking. Skipping repair query.",
                    type_log = "info",
                )
                continue
            
            # update the output
            dict_query_info[v].update({"create": (query_create, qe_id)})
            self._log(
                f"Successfully executed CREATE TABLE {k} with QueryExecutionId {qe_id}",
                type_log = "info",
            )
                
                
            ##  NEXT, TRY TO PARTITION THE TABLE THROUGH REPAIR

            query_repair = self.build_athena_query_repair_table(k)
            response_repair = self.exec_query(
                query_repair,
                self.get_s3_path_query_output(f"repair_{v}"),
                log_msg_supplement = " MSCK REPAIR TABLE",
            )   
            if response_repair is None:
                continue

            # check the creation query
            qe_id = response_repair.get(self.athena_reponse_key_exec_id)
            success = self.check_queries_success(qe_id, **kwargs)

            # log the outcome and update output and update the output dictionary
            if not success:
                self._log(
                    f"MSCK REPAIR TABLE {k} with QueryExecutionId {qe_id} has not successfully completed in time allotted for checking.",
                    type_log = "info",
                )
                continue
            
            # update the output
            dict_query_info[v].update({"repair": (query_repair, qe_id)})
            self._log(
                f"Successfully executed MSCK REPAIR TABLE {k} with QueryExecutionId {qe_id}",
                type_log = "info",
            )


        return dict_query_info
    


    def exec_query(self,
        query: str,
        s3p_query: str,
        log_msg_supplement: Union[str, None] = None
    ) -> Dict:
        """
        Execute query with output sent to location s3p_query. Optionally
            pass logging info supplement log_msg_supplement
        """
        log_msg_supplement = (
            "" 
            if not isinstance(log_msg_supplement, str) 
            else log_msg_supplement
        )
        response = None
        
        
        # try the CreateTable query
        try:
            response = self.client_athena.start_query_execution(
                QueryString = query,
                ResultConfiguration = {"OutputLocation": s3p_query}
            )

        except Exception as e:
            self._log(
                f"Error trying to execute query{log_msg_supplement}: {e}",
                type_log = "error",
            )
        
        return response
            
    


    def get_design_splits_by_launch_index(self,
        ind_base: Union[int, None] = None,
        max_n_instances: Union[int, None] = None,
        primary_keys: Union[Dict[str, int], List[int], None] = None,
        regions: Union[List[str], str, None] = None,
    ) -> Union[Dict, None]:
        """
        Divide the experimental design by region/primary keys. Generates a 
            dictionary with launch indices by key and each value is a tuple of
            the form

            (
                regions,
                primary_keys
            )

            to run with that launch index.

        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - ind_base: base index to use for launching instances
        - max_n_instances: maximum number of instances to spawn
        - primary_keys: dictionary of key dimensions to dimensional values 
            (collapsed using AND logic) OR list of primary key indices
        - regions: optional regions to specify. Must be defined in 
            configuration.
        """

        ##  INITIALIZE

        # get regions and implement checks
        regions = self.get_regions(regions)
        return_none = (regions is None)
        return_none |= (
            len(regions) == 0
            if not return_none
            else return_none
        )

        # number of instances
        max_n_instances = (
            self.ec2_max_n_instances
            if not isinstance(max_n_instances, int)
            else max_n_instances
        )
        return_none |= (max_n_instances is None)

        # primary keys to run
        primary_keys = self.build_experiment(primary_keys)
        return_none |= (primary_keys is None)

        # if any conditions are met, return none
        if return_none:
            return None


        ##  SECONDARY INIT (POST FILTERING WITH return_non)
        
        dict_launch_index = self.get_region_groups(
            regions, 
            max_n_instances, 
            primary_keys,
            ind_launch_base = ind_base,
        )
        
        
        return dict_launch_index
    


    def get_design_splits_by_launch_index_from_run_information(self,
        df_run_information: pd.DataFrame,
        delim_key_primary: str = ",",
        str_unassigned: str = "UNASSIGNED",
    ) -> Union[Dict, None]:
        """
        Generate a dictionary with launch indices by key using the 
            df_run_information generated by AWSManager.launch_experiment(). For 
            each launch index, the value is a tuple of the form

            (
                regions,
                primary_keys
            )

            to run with that launch index.

        Function Arguments
        ------------------
        - df_run_information: output run information data frame from 
            AWSManager.launch_experiment() 

        Keyword Arguments
        -----------------
        - delim_key_primary: primary key delimiter used to store primary keys in 
            field AWSManager.key_primary in df_run_information
        - str_unassigned: unassigned string used to identify runs that could not 
            launch
        """
        
        field_instance_id = self.field_instance_id
        field_launch_index = self.field_launch_index
        key_primary = self.key_primary
        key_region = self.key_region
        
        fields_req = set(
            [
                field_launch_index,
                field_instance_id,
                key_primary,
                key_region
            ]
        )
        
        if not fields_req.issubset(set(df_run_information.columns)):
            return None
        
        
        ##  FILTER AND GENERATE DICTIONARY
        
        df_run_information_filt = df_run_information[
            df_run_information[field_instance_id].isin([str_unassigned])
        ]
        
        if len(df_run_information_filt) == 0:
            return None
        
        
        dict_out = {}
        
        for i, row in df_run_information_filt.iterrows():
            
            index = int(row[field_launch_index])
            primaries = [int(x) for x in str(row[key_primary]).split(delim_key_primary)]
            regions = [str(row[key_region])]
            
            dict_out.update({index: (regions, primaries)})
        
        
        return dict_out



    def get_experimental_dictionary(self,
        delim: str = ",",
        dict_experimental_components: Union[Dict[str, Union[Dict[str, List[int]], List[int]]], None] = None,
        regex_config_parts: Union[re.Pattern, None] = None,
    ) -> Union[Dict[str, int], None]:
        """
        Check the specification of dict_experimental_components

        Function Arguments
        ------------------
            
        Keyword Arguments
        -----------------
        - delim: string delimiter for configuration
        - dict_experimental_components: dictionary of components with keys
            matching regex_config_parts
        - regex_config_parts: regular expression for determining experimental
            components. If None, defaults to 
            self.regex_config_experiment_components
        """
        
        # look for configuration components
        regex_config_parts = (
            self.regex_config_experiment_components
            if not isinstance(regex_config_parts, re.Pattern)
            else regex_config_parts
        )
        key_default = regex_config_parts.pattern.replace("(\\d*$)", "0")
        
        # default key if no parts are specified
        dict_experimental_components = (
            {self.key_primary: dict_experimental_components}
            if sf.islistlike(dict_experimental_components)
            else dict_experimental_components
        )
        dict_experimental_components = (
            self.config.get("experiment")
            if not isinstance(dict_experimental_components, dict)
            else dict_experimental_components
        )
        if dict_experimental_components is None:
            return None


        # check for keys breaking it into parts; if none are present, then look for any dimensional keys
        keys_keep = [
            self.key_design,
            self.key_future,
            self.key_primary,
            self.key_strategy
        ]
        
        keys_parts = [
            x for x in list(dict_experimental_components.keys())
            if regex_config_parts.match(x) is not None
        ]

        if len(keys_parts) > 0:
            dict_experimental_components = dict(
                (k, v)
                for k, v in dict_experimental_components.items()
                if isinstance(v, dict) & (k in keys_parts)
            )
        else:
            dict_experimental_components = {
                key_default: dict_experimental_components
            }

        # next, verify presernce of any dimensional keys
        dict_out = {}

        for k, v in dict_experimental_components.items():

            if not any([x in keys_keep for x in v.keys()]):
                continue
            
            # drop any other dimensions if primary is specified (avoids conflict)
            v = (
                {self.key_primary: v.get(self.key_primary)}
                if self.key_primary in v.keys()
                else v
            )
            
            # split to ensure list output
            v = dict(
                (j, sf.get_dimensional_values(z, j))
                for j, z in v.items()
                if j in keys_keep
            )
            
            dict_out.update({k: v})

        dict_experimental_components = (
            dict_out
            if len(dict_out) > 0
            else None
        )

        return dict_experimental_components

    

    def get_number_of_instances(self,
        n_regions: int,
        n_scenarios: int,
        max_n_instances: int,
    ) -> int:
        """
        Calculate the number of instances to spawn based on the number of regions,
            number of scenarios to run (per each region), and maximum number of
            instances
        """

        n_instances = min(
            max_n_instances,
            n_regions*n_scenarios
        )

        return n_instances
    


    def get_query_state(self,
        exec_id: str,
        client: Union[boto3.client, None] = None,
    ) -> Union[str, None]:
        """
        Get the state of a query with execution id exec_id
        """

        client = (
            self.client_athena
            if client is None
            else client
        )
        
        response = client.get_query_execution(
            QueryExecutionId = exec_id
        )

        query_exec = (
            response.get("QueryExecution")
            if isinstance(response, dict)
            else None
        )
        query_exec = (
            query_exec.get("Status")
            if isinstance(query_exec, dict)
            else None
        )
        state_cur = (
            query_exec.get("State")
            if isinstance(query_exec, dict)
            else None
        )
        
        return state_cur
    


    def get_random_seeds(self,
        regions: List[str],
        key_config_all: str = "ALL",
        random_seed_max: int = 10**9,
    ) -> Dict[str, int]:
        """
        Map regions to a random seed. Seeds can be specified as a uniform
            seed (defaults to configuration, or can specify as ALL) or as
            a region name or ISO. If a conflict occurs, pull from ISO. 
            
            Specify in configuration under 
            
            experiment.random_seed.ISO
            experiment.random_seed.REGION_NAME
            
            or 
            
            experiment.random_seed.ALL for all
            
        Function Arguments
        ------------------
        - regions: regions to assign random seeds for

        Keyword Arguments
        -----------------
        - key_config_all: key in configuration to use for assigning a seed
            for all countries
        - random_seed_max: max seed for generating a random seed
        """
        
        uniform_seed_q = self.config.get("experiment.uniform_random_seed")
        uniform_seed_q = (
            uniform_seed_q
            if isinstance(uniform_seed_q, bool)
            else False
        )
        
        # set up the seed
        seed_all = self.config.get(f"experiment.random_seed.{key_config_all}")
        seed_all = (
            self.model_attributes.configuration.get("random_seed")
            if not isinstance(seed_all, int)
            else seed_all
        )
        seed_all = (
            random.randint(0, random_seed_max)
            if not isinstance(seed_all, int)
            else seed_all
        )
        
        
        ##  ITERATE TO BUILD DICTIONARY
        
        dict_out = {}
        
        for region in regions:
            
            region_name = self.regions.return_region_or_iso(region, return_type = "region")
            region_iso = self.regions.return_region_or_iso(region, return_type = "iso")
            
            if uniform_seed_q:
                dict_out.update({region: seed_all})
                continue
                
            # check for a seed, def
            config_key_name = f"experiment.random_seed.{region_name}"
            config_key_iso = f"experiment.random_seed.{region_iso}"
            seed_name = self.config.get(config_key_name)
            seed_iso = self.config.get(config_key_iso)
            
            seed = (
                seed_iso
                if isinstance(seed_iso, int)
                else (
                    seed_name
                    if isinstance(seed_iso, int)
                    else random.randint(0, random_seed_max)
                )
            )

            dict_out.update({region: seed})
            
        
        return dict_out



    def get_region_groups(self,
        regions: List[str],
        max_n_instances: int,
        primary_keys: List[int],
        ind_launch_base: int = 0,
    ) -> Dict[int, Tuple]:
        """
        Get groups of regions for instances. Returns a dictionary that maps a
            launch group to a tuple of the following form

            {
                i: (
                    [regions_i...],
                    [primary_keys_i...]
                )
            }

            for launch index i, regions to launch regions_i, and primary_keys
            primary_keys_i.

            
        Function Arguments
        ------------------
        - regions: list of regions to split
        - max_n_instances: maximum number of instances to spawn
        - primary_keys: list of primary keys that will be run

        Keyword Arguments
        -----------------
        - ind_launch_base: starting index for the launch
        """
        # some initialization
        n_primary_keys = len(primary_keys)
        ind_launch_base = 0 if not sf.isnumber(ind_launch_base) else ind_launch_base
        ind_launch_base = max(ind_launch_base, 0)

        # set number of instances
        n_regions = len(regions)
        n_instances = self.get_number_of_instances(
            n_regions,
            len(primary_keys),
            max_n_instances,
        )
        base_n_regions_per_instance, n_instances_w_extra_region = sf.div_with_modulo(n_regions, n_instances)
        base_n_regions_per_instance = max(base_n_regions_per_instance, 1)


        ##  GENERATE REGION GROUPS

        # randomize order of regions
        dict_region_groups = {}
        regions_shuffled = regions.copy()
        random.shuffle(regions_shuffled)

        ind_extra_regions = 0
        ind_range = 0

        for i in range(min(n_regions, n_instances)):

            ind_0 = ind_range
            ind_1 = ind_range + base_n_regions_per_instance
            ind_1 += int(ind_extra_regions < n_instances_w_extra_region)

            ind_extra_regions += 1

            # select slice
            grp = regions_shuffled[ind_0:ind_1]
            dict_region_groups.update({i: grp})

            # update base
            ind_range = ind_1

        n_region_groups = len(dict_region_groups)


        ##  ASSIGN INSTANCES

        base_n_instances_per_rg, n_rg_w_extra_instances = sf.div_with_modulo(n_instances, n_region_groups)

        # get keys to assign extra instances to (sorted from largest to leas)
        assignment_queue = sorted(
            [
                (len(v), k) for k, v in dict_region_groups.items()
            ], 
            reverse = True
        )
        assignment_queue = [x[1] for x in assignment_queue]

        # iterate to assign
        dict_launch_index = {}
        ind_extra_instance = 0
        ind_launch_index = ind_launch_base#0

        for i, k in enumerate(assignment_queue):
            # get the 
            v = dict_region_groups.get(k)

            n_instances_cur_group = base_n_instances_per_rg
            n_instances_cur_group += int(ind_extra_instance < n_rg_w_extra_instances)

            # now, split up primary keys
            base_n_keys_per_instance, n_instances_with_extra_keys = sf.div_with_modulo(n_primary_keys, n_instances_cur_group)

            # some iterators - note, ind_launch_index should be outside the assignment queue
            ind_pk_0 = 0
            ind_key_extra = 0
            list_splits = {}

            for j in range(n_instances_cur_group):

                ind_pk_1 = ind_pk_0 + base_n_keys_per_instance
                ind_pk_1 += int(ind_key_extra < n_instances_with_extra_keys)
                ind_key_extra += 1

                dict_launch_index.update(
                    {
                        ind_launch_index: (
                            v,
                            primary_keys[ind_pk_0:ind_pk_1]
                        )
                    }
                )

                ind_pk_0 = ind_pk_1

                # inner iteration; we want to map each set of primary keys/region to a launch index
                ind_launch_index += 1

            ind_extra_instance += 1

        """
        # get number of instances  
        out = (
            dict_launch_index,
            dict_region_groups, 
            n_instances, 
            n_region_groups,
        )
        """;

        return dict_launch_index
    


    def launch_experiment(self,
        delim: str = ",",
        dict_launcher: Union[Dict[int, Tuple[List[str], List[int]]], None] = None,
        launch: bool = True,
        setup_database: bool = False,
        try_number: int = 1,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Launch a design specified in dict_launcher
        
        Function Arguments
        ------------------
        
        Keyword Arugments
        -----------------
        - delim: delimiter to use to print regions in output DataFrame
        - dict_launcher: dictionary building launcher information (see 
            self.get_design_splits_by_launch_index())
        - launch: set to False to return random seeds and launch index 
            dictionary
        - setup_database: set up the Athena database? 
            NOTE: The repair query has to be launched completion even if True
        - try_number: pass the try number for this scenario
        - **kwargs: passed to AWSManager.execute_create_table_athena_query()
        """
        
        dict_launcher = (
            self.get_design_splits_by_launch_index()
            if not isinstance(dict_launcher, dict)
            else dict_launcher
        )
    
        # get the regions that will be launched, and get the seed dictionary
        regions_launch = [
            [x[0] for x in dict_launcher.values()]
        ]
        regions_launch = sorted(list(set(sum(sum(regions_launch, []), []))))
        dict_region_to_seed = self.get_random_seeds(regions_launch)    
        
        
        # initialize information for information dataframe + dictionary of UserData
        df_run_information = []
        fields_out = [
            self.field_launch_index,
            self.field_instance_id,
            self.key_region,
            self.key_primary,
            self.field_random_seed,
            self.field_ip_address,
            self.field_n_launch_tries
        ]
        
        dict_ud = {}
        
        for ind_launch, spec_tuple in dict_launcher.items():
            
            continue_q = not isinstance(ind_launch, int)
            continue_q |= not sf.islistlike(spec_tuple[0])
            continue_q |= not sf.islistlike(spec_tuple[1])
            
            if continue_q:
                continue
            
            # 
            
            random_seed = dict_region_to_seed.get(spec_tuple[0][0])
            
            dict_specification = {
                self.key_region: spec_tuple[0],
                self.key_primary: spec_tuple[1],
                self.flag_key_random_seed: random_seed,
            }
            
            # get user data and add to dictionary
            user_data = self.build_user_data(
                dict_specification, 
                instance_launch_index = ind_launch
            )
            dict_ud.update({ind_launch: user_data})
            
            
            # try to run instance
            try:
                dict_instance = self.launch_instance(user_data) if launch else None

            except Exception as e:
                self._log(
                    f"Error trying to launch index {ind_launch}: {e}",
                    type_log = "error"
                )
            
            
            # initialize
            instance_id = "UNASSIGNED"
            ip_address = "00.00.00.00"

            # overwrite if information is available from a successful launch
            if isinstance(dict_instance, dict):

                info = dict_instance.get("Instances")[0]
                instance_id = info.get("InstanceId")
                ip_address = info.get("PrivateIpAddress")

                self._log(
                    f"Launch index {ind_launch} successfully launched instance {instance_id}",
                    type_log = "info"
                )
  
                

            # add output roww
            row = [
                ind_launch,
                instance_id,
                delim.join(spec_tuple[0]),
                delim.join([str(x) for x in spec_tuple[1]]),
                random_seed,
                ip_address,
                try_number,
            ]
            
            df_run_information.append(row)

                
        df_run_information = pd.DataFrame(df_run_information, columns = fields_out)
        

        # then, build the queries to create the Athena database if specified
        dict_query_responses = (
            self.execute_create_table_athena_query(**kwargs)
            if (setup_database & launch)
            else None
        )

        return dict_query_responses, dict_ud, df_run_information



    #############################
    #    PACKAGING FUNCTIONS    #
    #############################

    def generate_presign_command(self,
        dict_response: Dict,
        table_type: str,
        expiration: int = 10800,
    ) -> Union[str, None]:
        """
        Generate a presigned AWS S3 share command for a file based on a query
            execution response dictionary.
        
        Function Arguments
        ------------------
        - dict_response: response dictionary generated by 
            self.client_athena.get_query_execution OR execution id string
        - table_type: any valid input to self.get_s3_path_query_output (see 
            ?get_s3_path_query_output())

        Keyword Arguments
        -----------------
        - expiration: expiration time of link in seconds
        """
        # some checks
        return_none = not isinstance(dict_response, dict)
        return_none &= not isinstance(dict_response, str)
        
        address_s3_query = self.get_s3_path_query_output(table_type)
        return_none |= not isinstance(address_s3_query, str)
        return_none |= not isinstance(expiration, int)
        
        if return_none:
            return None
        
        # try getting execution id
        exec_id = (
            dict_response.get(self.athena_reponse_key_exec_id)
            if isinstance(dict_response, dict)
            else dict_response
        )
        
        if exec_id is None:
            return None
        
        
        address_s3_query = self.get_s3_path_query_output(table_type)
        address_s3_query = os.path.join(address_s3_query, f"{exec_id}.csv")
        comm = f"aws s3 presign \"{address_s3_query}\" --expires-in {expiration}"
        
        return comm



    def _write_attribute_generic(self,
        dims: Union[str, List[str], None],
        dir_local_write: Union[str, None] = None,
        strip_ids_from_table_names: bool = True,
        **kwargs,
    ) -> None:
        """
        Write generic model attribute files (including design and strategy) to
            files.
        
        Function Arguments
        ------------------
        - dims: valid dimensions include
            * design
            * strategy
            * time_period

        Keyword Arguments
        -----------------
        - dir_local_write: output directory to write files to. If None, defaults
            to self.file_struct.fp_base_output_raw
        - strip_ids_from_table_names: drop any string from the table name that 
            matches "_ID"
        - **kwargs: passed to AWSManager.build_experiment()
        """
        
        
        ##  CHECKS

        dims = [dims] if isinstance(dims, str) else dims
        dims = (
            [x for x in dims if f"dim_{x}" in self.model_attributes.dict_attributes.keys()] 
            if sf.islistlike(dims) 
            else None
        )
        
        if dims is None:
            return None


        # check output directory
        dir_local_write = self.file_struct.fp_base_output_raw
        (
            os.makedirs(dir_local_write, exist_ok = True)
            if not os.path.exists(dir_local_write)
            else None
        )
        
        for dim in dims:
            
            attr = self.model_attributes.get_dimensional_attribute_table(dim)

            if attr is None:
                self._log(
                    f"From AWSManager._write_attribute_generic(): table key '{key}' not found in ModelAttributes. It will not be written.",
                    type_log = "warning",
                )
                continue
            
            # get the base dimension
            table_dim = dim.upper()
            table_dim = (
                table_dim.replace("_ID", "")
                if strip_ids_from_table_names
                else table_dim
            )

            attr.table.to_csv(
                os.path.join(
                    dir_local_write,
                    f"ATTRIBUTE_{table_dim}.csv"
                ),
                index = None,
                encoding = "UTF-8",
            )
        
        return None



    def _write_attribute_primary(self,
        dict_experimental_components: Union[Dict, None] = None,
        dir_local_write: Union[str, None] = None,
        strip_ids_from_table_names: bool = True,
        **kwargs,
    ) -> None:
        """
        Retrieve output, input, and other files from S3 and add to 
            SISEPUEDE.id output location
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        - dict_experimental_components: optional experimental definition 
            dictionary to pass to AWSManager.build_experiment() (see docstring
            at ?AWSManager.build_experiment for more information)
        - dir_local_write: output directory to write files to. If None, defaults
            to self.file_struct.fp_base_output_raw
        - strip_ids_from_table_names: drop any string from the table name that 
            matches "_ID"
        - **kwargs: passed to AWSManager.build_experiment()
        """
        
        
        ##  CHECKS
        
        return_none = False
        if return_none:
            return None

        # check output directory
        dir_local_write = self.file_struct.fp_base_output_raw
        (
            os.makedirs(dir_local_write, exist_ok = True)
            if not os.path.exists(dir_local_write)
            else None
        )

        # get the base dimension
        table_dim = self.sisepuede_database.table_name_attribute_primary.upper()
        table_dim = (
            table_dim.replace("_ID", "")
            if strip_ids_from_table_names
            else table_dim
        )
        
        (
            self.build_experiment(
                dict_experimental_components = dict_experimental_components,
                return_type = "indexing_data_frame",
            )
            .to_csv(
                os.path.join(
                    dir_local_write,
                    f"{table_dim}.csv"
                ),
                index = None,
                encoding = "UTF-8",
            )
        )
        
        return None
        


