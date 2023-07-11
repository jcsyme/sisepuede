from analysis_id import *
from attribute_table import AttributeTable
import base64
import boto3
import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import re
import setup_analysis as sa
import shutil
import sisepuede_file_structure as sfs
import support_classes as sc
import support_functions as sf
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
    - file_struct: SISEPUEDEFileStructure used to access ID information and
        model_attributes
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
    """

    def __init__(self,
        file_struct: sfs.SISEPUEDEFileStructure,
        config: Union[str, sc.YAMLConfiguration],
        fp_template_docker_shell: str,
        fp_template_user_data: str,
        as_string_docker_shell: bool = False,
        as_string_user_data: bool = False,
    ) -> None:
        
        self._initialize_template_matchstrs()
        self._initialize_attributes(file_struct)
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

        # initialize aws 
        self._initialize_aws_properties()
        self._initialize_aws()




    ########################
    #    INITIALIZATION    #
    ########################

    def get_instance_metadata_file_path(self,
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
        file_struct: sfs.SISEPUEDEFileStructure,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.dict_dims_to_docker_flags (map SISEPUEDE dimensions to docker 
                flags)
            * self.file_struct = file_struct
            * self.key_design
            * self.key_future
            * self.key_region
            * self.key_primary
            * self.key_strategy
            * self.model_attributes
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """
        model_attributes = file_struct.model_attributes

        # analysis dimensional keys
        key_design = model_attributes.dim_design_id
        key_future = model_attributes.dim_future_id
        key_region = model_attributes.dim_region
        key_primary = model_attributes.dim_primary_id
        key_strategy = model_attributes.dim_strategy_id

        # map keys to flag values
        dict_dims_to_docker_flags = {
            key_design: "keys-design",
            key_future: "keys-future",
            key_region: "regions",
            key_primary: "keys-primary",
            key_strategy: "keys-strategy",
            "database_type": "database-type",
            "max_solve_attempts": "max-solve-attempts",
            "random_seed": "random-seed",
            "save_inputs": "save-inputs",
        }

        ##  SET PROPERTIES

        self.dict_dims_to_docker_flags = dict_dims_to_docker_flags
        self.file_struct = file_struct
        self.key_design = key_design
        self.key_future = key_future
        self.key_region = key_region
        self.key_primary = key_primary
        self.key_strategy = key_strategy
        self.model_attributes = model_attributes

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
            * self.resource_s3
        
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
        
        #get clients and resources
        client_athena = b3_session.client("athena")
        client_ec2 = b3_session.client("ec2")
        client_s3 = b3_session.client("s3")
        resource_s3 = b3_session.resource("s3")
        
        
        ##  SET PROPERTIES

        self.b3_session = b3_session
        self.client_athena = client_athena
        self.client_ec2 = client_ec2
        self.client_s3 = client_s3
        self.resource_s3 = resource_s3

        return None



    def _initialize_aws_properties(self,
    ) -> None:
        """
        Use the configuration file to set some key shared properties for AWS. 
            Sets the following properties:

            * self.athena_database
            * self.aws_dict_tags
            * self.aws_url_instance_id_metadata
            * self.ec2_image
            * self.ec2_instance_base_name
            * self.ec2_instance_type
            * self.ec2_n_instances
            * self.s3_bucket
            * self.s3_key_database
            * self.s3_key_metadata

        NOTE: *Excludes* paths on instances. Those are set in
            _initialize_paths()
        """

        # general AWS properties
        self.aws_dict_tags = self.config.get("aws_general.tag")
        self.aws_url_instance_id_metadata = "http://169.254.169.254/latest/meta-data/instance-id/"

        # athena properties
        self.athena_database = self.config.get("aws_athena.database")

        # EC2 properties
        self.ec2_image = self.config.get("aws_ec2.database")
        self.ec2_instance_base_name = self.config.get("aws_ec2.instance_base_name")
        self.ec2_instance_type = self.config.get("aws_ec2.instance_type")
        self.ec2_n_instances = int(self.config.get("aws_ec2.n_instances"))

        # S3 properties
        self.s3_bucket = self.config.get("aws_s3.bucket")
        self.s3_key_database = self.config.get("aws_s3.key_database")
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

            * self.

        NOTE: *Excludes* paths on instances. Those are set in
            _initialize_paths()
        """

        self.docker_image_name = self.config.get("docker.image_name")

        return None



    def _initialize_paths(self,
    ) -> None:
        """
        Initialize some paths. Initializes the following properties:

            * self.dir_docker_sisepuede_out
            * self.dir_docker_sisepuede_python
            * self.dir_docker_sisepuede_repo
            * self.dir_instance_home
            * self.dir_instance_out
            * self.fp_instance_instance_info
            * self.fp_instance_shell_script

        NOTE: must be initialized *after* 
            _initialize_shell_script_environment_vars()

        Function Arguments
        ------------------
        """

        ##  INSTANCE PATHS

        # directories
        dir_instance_home = self.config.get("aws_ec2.dir_home")
        dir_instance_out = self.config.get("aws_ec2.dir_output")

        # file paths
        fp_instance_instance_info = self.get_instance_metadata_file_path(dir_instance_out)
        fp_instance_shell_script = self.config.get("aws_ec2.instance_shell_path")
        

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

        # file paths



        ##  SET PROPERTIES

        self.dir_docker_sisepuede_python = dir_docker_sisepuede_python
        self.dir_docker_sisepuede_out = dir_docker_sisepuede_out
        self.dir_docker_sisepuede_repo = dir_docker_sisepuede_repo
        self.dir_instance_home = dir_instance_home
        self.dir_instance_out = dir_instance_out
        self.fp_instance_instance_info = fp_instance_instance_info
        self.fp_instance_shell_script = fp_instance_shell_script

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
        
        # add in max solve attempts
        max_attempts = self.config.get("sisepuede_runtime.max_solve_attempts")
        flag = self.dict_dims_to_docker_flags.get("max_solve_attempts")
        (
            flags.append(f"--{flag} {max(max_attempts, 1)}") 
            if isinstance(max_attempts, int)
            else None
        )

        # add in database type
        db_type = self.config.get("sisepuede_runtime.database_type")
        db_type = "csv" if (db_type is None) else db_type
        flag = self.dict_dims_to_docker_flags.get("database_type")
        (
            flags.append(f"--{flag} {db_type}") 
            if isinstance(flag, str)
            else None
        )

        # add in save inputs
        save_inputs = self.config.get("sisepuede_runtime.save_inputs")
        save_inputs = True if (save_inputs is None) else save_inputs
        flag = self.dict_dims_to_docker_flags.get("save_inputs")
        (
            flags.append(f"--{flag} {save_inputs}") 
            if isinstance(flag, str)
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
        dict_subset: Dict[str, List[int]],
        delim_newline: str = "\n",
        output_type: str = "echo",
    ) -> str:
        """
        Copy outputs to S3
        
        Function Arguments
        ------------------
        


        Keyword Arguments
        -----------------
        
        """





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
    ) -> str:
        """
        Build the docker pull command
        
        Function Arguments
        ------------------

        Keyword Arguments
        -----------------
        """

        # get the image name
        name_docker_image = self.docker_image_name

        # build the command as a list
        command_out = ["docker pull"]
        command_out.append(name_docker_image)
       
        # join commands
        command_out = " ".join(command_out)

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

        # get the image nameHEREHERE
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
        

        


