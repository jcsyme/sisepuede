
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Figure
from matplotlib.axes._subplots import Axes
from typing import *

import sisepuede.utilities._toolbox as sf




##########################################
###                                    ###
###    BEGIN PLOT FUNCTIONS LIBRARY    ###
###                                    ###
##########################################

def is_valid_figtuple(
    figtuple: Any,
) -> bool:
    """
    Check if `figtuple` is a valid figure tuple of the form (fig, ax) for use
        in specifying plots. 
    """

    is_valid = isinstance(figtuple, tuple)
    is_valid &= (len(figtuple) >= 2) if is_valid else False
    is_valid &= (
        (isinstance(figtuple[0], Figure) & isinstance(figtuple[1], Axes)) 
        if is_valid 
        else False
    )

    return is_valid



def plot_stack(
    df: pd.DataFrame,
    fields: pd.DataFrame, 
    dict_formatting: Union[Dict[str, Dict[str, Any]], None] = None,
    field_x: Union[str, None] = None,
    figsize: Tuple = (18, 12),
    figtuple: Union[Tuple, None] = None,
    label_x: Union[str, None] = None,
    label_y: Union[str, None] = None,
    title: Union[str, None] = None,
    **kwargs,
) -> Union['matplotlib.Plot', None]:
    """
    plt.plot.area() cannot handle negative trajctories. Use plt.stackplot to 
        facilitate stacked area charts for trajectories that may shift between 
        positive and negative.
    
    Function Arguments
    ------------------
    - df: data frame to pull data from
    - fields: fields to plot

    Keyword Arguments
    ----------------- 
    - dict_formatting: optional dictionary used to pass field-specific 
        formatting; e.g., use 
        
        dict_formatting = {
            field_i: {
                "kwarg_i": val_i,
                ...
            }
        }
        
        to pass formatting keywords for fields

    - field_x: optional field `x` in data frame to use for x axis
    - figsize: figure size to use. Only used if `figtuple` is not a valid 
        (Figure, Axis) pair
    - figtuple: optional tuple of form `(fig, ax)` (result of plt.subplots) to 
        pass. Allows users to predefine information about the fig, ax outside of
        this function, then plot within those confines
    - label_x: optional label to pass for x axis. Only used if `figtuple` is
        not a valid (Figure, Axis) pair
    - label_y: optional label to pass for y axis. Only used if `figtuple` is
        not a valid (Figure, Axis) pair
    - title: optional title to pass. Only used if `figtuple` is not a valid 
        (Figure, Axis) pair
    - **kwargs: passed to ax.stackplot, ax.set_xlabel, ax.set_ylabel, and 
        ax.set_title
    """
    
    # check field x
    add_x = False
    field_x = field_x if field_x in df.columns else None
    if field_x is None:
        field_x = "x"
        add_x = True
    
    # check all fields
    fields = [x for x in df.columns if x in fields and (x != field_x)]
    if len(fields) == 0:
        return None
    
    if add_x:

        # if adding a dummy, copy first, then add field x
        df_plot = df[fields].copy()
        df_plot[field_x] = range(len(df_plot))
        fields.append(field_x)

    else:

        # if field_x is in the df, add it to fields, then copy
        fields.append(field_x)
        df_plot = df[fields].copy()


    # check the color dictionary
    if not isinstance(dict_formatting, dict):
        dict_formatting = {}
    
    
    ##  BUILD PLOTTED DATA
    
    # initialize ordered plot fields
    fields_plot = [x for x in fields if x != field_x]
    
    # split into positive and negative
    df_neg = pd.concat(
        [
            df_plot[[field_x]],
            df_plot[fields_plot].clip(upper = 0, )
        ], 
        axis = 1,
    )
    df_pos = pd.concat(
        [
            df_plot[[field_x]],
            df_plot[fields_plot].clip(lower = 0, )
        ], 
        axis = 1,
    )
    
    # if colors are specified, specify an ordered vector that matches the fields
    if isinstance(dict_formatting, dict):
        try:
            # use try to avoid checking if "x" is specified
            color = [dict_formatting.get(x).get("color") for x in fields_plot]
            
        except:
            color = None

    

    ##  SPECIFY AND FORMAT AXES
    
    # check if the figtuple specification is ok
    accept_figtuple = is_valid_figtuple(figtuple, )
    
    if accept_figtuple:
        fig, ax = figtuple

    else:
        fig, ax = plt.subplots(1, 1, figsize = figsize, )

        # check label x
        if isinstance(label_x, str):
            sf.call_with_varkwargs(
                ax.set_xlabel,
                label_x,
                dict_kwargs = dict_kwargs,
            )
        
        # check label y
        if isinstance(label_y, str):
            sf.call_with_varkwargs(
                ax.set_xlabel,
                label_y,
                dict_kwargs = dict_kwargs,
            )
        
        # check title
        if isinstance(title, str):
            sf.call_with_varkwargs(
                ax.set_title,
                title,
                dict_kwargs = dict_kwargs,
            )

            
    
    ##  PLOT THE POSITIVE AND NEGATIVE COMPONENTS
    
    dict_kwargs = dict((k, v) for (k, v) in kwargs.items())
    dict_kwargs.update(
        {
            "colors": color,
            "data": df_pos,
            "labels": fields_plot
        }
    )
    
    sf.call_with_varkwargs(
        ax.stackplot,
        field_x, 
        *fields_plot,
        dict_kwargs = dict_kwargs,
    )
    
    # update before plotting next one
    dict_kwargs.update({"data": df_neg})
    sf.call_with_varkwargs(
        ax.stackplot,
        field_x, 
        *fields_plot,
        dict_kwargs = dict_kwargs,
    )
    
    return fig, ax
    
           

        






