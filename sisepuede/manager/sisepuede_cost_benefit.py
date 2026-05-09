import logging
import numpy as np
import os, os.path
import pandas as pd
import tempfile
from typing import *


from sisepuede.core.model_attributes import ModelAttributes
from sisepuede.models.afolu import AFOLU
from sisepuede.models.circular_economy import CircularEconomy
import sisepuede.models.energy_production as mep
import sisepuede.models.energy_consumption as mec
from sisepuede.models.ippu import IPPU
from sisepuede.models.socioeconomic import Socioeconomic
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf



##########################
#    GLOBAL VARIABLES    #
##########################

# INITIALIZE UUID
_MODULE_UUID = "B2E533AC-F477-4315-AB64-9B2C230B74EB"



####################
#    MAIN CLASS    #
####################

class SISEPUEDECostBenefit:
    """Instantiate cost benefit calculator for SISEPUEDE. Operates on outputs 
        from runs to calculate and estimate cost benefits. Can be 

    
    Initialization Arguments
    ------------------------
    model_attributes : ModelAttributes
        ModelAttributes object used to manage variables and coordination

    Optional Arguments
    ------------------
    df_baseline : Union[pd.DataFrame, None]
        Optional DataFrame to use as baseline for comparing costs and benefits
        to.
    logger : Union[logging.Logger, None]
        optional logging.Logger object used to log model events
    """
    def __init__(self,
        model_attributes: ModelAttributes,
        df_baseline: Union[pd.DataFrame, None] = None,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        # initialize input objects
        self._initialize_attributes(
            model_attributes,
            logger = logger,
        )

        # set the UUID
        self._initialize_uuid()

        return None
    


    def __call__(self,
        *args,
        **kwargs,             
    ) -> pd.DataFrame:
        
        out = self.project(*args, **kwargs)

        return out




    ##############################################
    #	SUPPORT AND INITIALIZATION FUNCTIONS	#
    ##############################################

    def _initialize_attributes(self,
        model_attributes: ModelAttributes,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        """Initialize key attributes for the model. Initializes the following 
            properties:

            * self.logger
            * self.model_attributes
            * self.time_periods
        """

        time_periods = sc.TimePeriods(model_attributes)

        self.logger = logger
        self.model_attributes = model_attributes
        self.time_periods = time_periods

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """
        Initialize the UUID. Sets the following properties:

            * self.is_sisepuede_examples
            * self._uuid
        """

        self.is_sisepuede_cost_benefit = True
        self._uuid = _MODULE_UUID

        return None






###################################
###                             ###
###    SOME SIMPLE FUNCTIONS    ###
###                             ###
###################################

def is_sisepuede_examples(
    obj: Any,
) -> bool:
    """
    check if obj is a SISEPUEDECostBenefit object
    """

    out = hasattr(obj, "is_sisepuede_cost_benefit")
    uuid = getattr(obj, "_uuid", None)

    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out