
import pandas as pd
import os, os.path
from typing import *


import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf




class SISEPUEDEExamples:
    """
    Load and access example data used to demonstrate SISEPUEDE.


    Optional Arguments
    ------------------
    
    """


    def __init__(self,
    ) -> None:

        self._initialize_file_structure()
        self._initialize_examples()

        return None
    


    def __call__(self,
        *args,
    ) -> None:

        out = self.get_example(*args,)
        
        return out
    


    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_file_structure(self,
    ) -> None:

        """
        Intialize the SISEPUEDEFileStructure object and model_attributes object.
            Initializes the following properties:

            * self.analysis_id
            * self.file_struct
            * self.fp_base_output_raw
            * self.fp_log 	(via self._initialize_logger())
            * self.id		(via self._initialize_logger())
            * self.id_fs_safe
            * self.logger
            * self.model_attributes

        Optional Arguments
        ------------------
        """

        file_struct = None

        try:
            file_struct = sfs.SISEPUEDEFileStructure(
                initialize_directories = False,
            )

        except Exception as e:
            msg = f"Error trying to initialize SISEPUEDEFileStructure: {e}"
            raise RuntimeError(msg)


        ##  SET PROPERTIES

        self.file_struct = file_struct
        self.model_attributes = file_struct.model_attributes

        return None
    


    def _initialize_examples(self,
    ) -> None:

        """
        Intialize examples as properties. Sets the following properties:

            * self.EXAMPLE_NAME_HERE (files contained in ref/examples)
            * self.all_examples


        Optional Arguments
        ------------------
        """

        dir_examples = self.file_struct.dir_ref_examples
        files_example = os.listdir(dir_examples)

        ##  INIT

        all_examples = []


        ##  CSVs

        files_example_csvs = [x for x in files_example if x.endswith(".csv")]

        for fl in files_example_csvs:

            fp_read = os.path.join(dir_examples, fl)

            try:
                df_cur = pd.read_csv(fp_read)
                attr_name = fl.replace(".csv", "")

                setattr(self, attr_name, df_cur)

            except Exception as e:
                continue

            all_examples.append(attr_name)

        
        ##  OTHER FILE TYPES (PLACEHOLDER)


        ##  SET PROPERTIES

        self.all_examples = all_examples

        return None




    ########################
    #    CORE FUNCTIONS    #
    ########################

    def get_example(self,
        example: str,
    ) -> Any:
        """
        Retrieve an example dataset from the SISEPUEDE Example system.
        """

        out = getattr(self, example, None)

        return out



