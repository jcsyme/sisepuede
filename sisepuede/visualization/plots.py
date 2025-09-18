
import matplotlib.pyplot as plt
import pandas as pd
from typing import *

import sisepuede.core.model_attributes as ma
import sisepuede.utilities._plotting as spu
import sisepuede.utilities._toolbox as sf



"""
fields = matt.get_all_subsector_emission_total_fields()#[x for x in df_out.columns if (x.startswith("emission_co2e_subsector_total"))]
dict_format = dict(
    (k, {"color": v}) for (k, v) in
    matt.get_subsector_color_map().items()
)

dict_format

"""


def plot_emissions_stack(
    df: pd.DataFrame,
    model_attributes: 'ModelAttributes',
    dict_format: Union[dict, None] = None,
    **kwargs,
) -> 'plt.Plot':
    """Plot subsector emission total fields using 

    Function Arguments
    ------------------
    df : DataFrame
        DataFrame containing SISEPUEDE output emissions to plot
    model_attributes : ModelAttributes
        ModelAttributes object used to identify fields and default colors

    Keyword Arguments
    -----------------
    dict_format : Union[Dict, None]
        Optional dictionary to pass to spu.plot_stack (see spu.plot_stack for 
        more information on formatting). Can be used to overwrite default colors
        for sectors
    **kwargs
        Passed to spu.plot_stack
    """

    fields = model_attributes.get_all_subsector_emission_total_fields()#[x for x in df_out.columns if (x.startswith("emission_co2e_subsector_total"))]
    dict_formatting = dict(
        (k, {"color": v}) for (k, v) in
        model_attributes.get_subsector_color_map().items()
    )

    if isinstance(dict_format, dict):
        dict_formatting.update(dict_format, )

    out = spu.plot_stack(
        df,
        fields,
        dict_formatting = dict_formatting,
        **kwargs,
    )

    return out



def plot_variable_stack(
    df: pd.DataFrame,
    modvar: 'ModelVariable',
    model_attributes: 'ModelAttributes',
    **kwargs,
) -> 'plt.Plot':
    """Plot a variable using stack.

    """

    # get fields to extract
    fields = model_attributes.build_variable_fields(
        modvar,
        **kwargs
    )

    out = spu.plot_stack(
        df,
        fields,
        **kwargs,
    )

    return out



def plot_variable(
    df: pd.DataFrame,
    modvar: 'ModelVariable',
    model_attributes: 'ModelAttributes',
    kind: str,
    **kwargs,
) -> 'plt.Plot':
    """
    Plot a SISEPUEDE variable.
    """

    return None