
import os, os.path
import sys
from typing import *

dir_data_construction = os.path.dirname(os.path.realpath(__file__))
dir_python = os.path.dirname(dir_data_construction)
(
    sys.path.append(dir_python)
    if dir_python not in sys.path
    else None
)
import model_attributes as ma
import sisepuede_file_structure as sfs
import support_classes as sc
import support_functions as sf



class SISEPUEDEDataConstruction:
    """
    SISEPUEDEDataConstruction facilitates the generation and management of batch 
        data for SISEPUEDE. This structure sets some shared paths and methods
        for 
    """

    def __init__(self,
    ):
        self._initialize_paths()

        return None


    
    def _initialize_paths(self,
    ) -> None:
        """
        Instantiate file paths and set the following properties:

            * self.dir_data_construction
            * self.sisepuede_file_struct
        """

        dir_data_construction = os.path.dirname(os.path.realpath(__file__))
        dir_python = os.path.dirname(dir_data_construction)

        sisepuede_file_struct = sfs.SISEPUEDEFileStructure(
            initialize_directories = False,
        )

        # some attributes to set



        ##  SET PROPERTIES

        self.dir_data_construction = dir_data_construction
        self.sisepuede_file_struct = sisepuede_file_struct

        

        return None
