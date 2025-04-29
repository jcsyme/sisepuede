
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
    model_attributes: 'ModelAttributes',
) -> 'plt.Plot':
    """
    """

    fields = matt.get_all_subsector_emission_total_fields()#[x for x in df_out.columns if (x.startswith("emission_co2e_subsector_total"))]
    dict_format = dict(
        (k, {"color": v}) for (k, v) in
        matt.get_subsector_color_map().items()
    )

    dict_format

    return None




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