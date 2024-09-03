

import pandas as pd


import sisepuede.core.model_attributes as ma
import sispuede.utilities._toolbox as sf
from sisepuede.transformers.lib._classes import Transformer


#####################################
###                               ###
###    COMBINE TRANSFORMATIONS    ###
###                               ###
#####################################

def combine_transformers(
    transformers: Union[List[Transformer], Transformer],
) -> Union[Transformer, None]:
