===============
Transformations
===============

SISEPUEDE includes a predefined library of **transformations** that can be customized and combined to create detailed and comprehensize strategies--based on expected outcomes--for reducing emissions of greenhouse gasses at scale. 

Transformations are outcomes that can be acheieved through the implementation of one or more policies. For example, mode shifting transportation can be achieved through a number of potential policies, including urban planning, taxes on fuels, reductions of public transportation fares, construction of infrastructure, and more. 

Transformations are defined in `Transformation` classes in python. Collections of transformations are called **strategies** (link).


Transformers vs. Transformations
================================

SISEPUEDE uses simlar terms to refer to two different components of the framework for modifying  

**Transformers** are pre-defined callable classes in Python that modify a base set of trajectories to reflect a desired outcome. These classes, which include default values, can be called with different functional specifications that are defined in configuration files to allow for flexibility in applying the transformation. 

Transformer Code: 

.. note:: Example: Transformer ``TX:AGRC_INC_CONSERVATION_AGRICULTURE``



Defining Transformations
========================

Transformations are parameterizations of Transformers. 

.. the module should be available from path set in conf.py https://sphinx-tutorial.readthedocs.io/step-2/
.. autoclass:: sisepuede.transformers.transformations.Transformations


Parameterizing Transformations
==============================


Common Transformation Nomenclature
==================================


`magnitude_type`
----------------

- `baseline_additive`: add the magnitude to the baseline
- `baseline_scalar`: multiply baseline value by magnitude
- `baseline_scalar_diff_reduction`: reduce the difference between
    the value in the baseline time period and the upper bound (NOTE:
    requires specification of bounds to work) by magnitude
- `final_value`: magnitude is the final value for the variable to
    take (achieved in accordance with vec_ramp)
- `final_value_ceiling`: magnitude is the lesser of (a) the existing 
    final value for the variable to take (achieved in accordance 
    with vec_ramp) or (b) the existing specified final value,
    whichever is smaller
- `final_value_floor`: magnitude is the greater of (a) the existing 
    final value for the variable to take (achieved in accordance 
    with vec_ramp) or (b) the existing specified final value,
    whichever is greater
- `transfer_value`: transfer value from categories to other
    categories. Must specify "categories_source" &
    "categories_target" in dict_modvar_specs. See description below
    in OPTIONAL for information on specifying this.
- `transfer_scalar_value`: transfer value from categories to other
    categories based on a scalar. Must specify "categories_source" &
    "categories_target" in dict_modvar_specs. See description below
    in OPTIONAL for information on specifying this.
- `transfer_value_to_acheieve_magnitude`: transfer value from
    categories to other categories to acheive a target magnitude.
    Must specify "categories_source" & "categories_target" in
    dict_modvar_specs. See description below in OPTIONAL for
    information on specifying this.
- `vector_specification`: simply enter a vector to use for region


`vec_implementation_ramp`
-------------------------

The implementation ramp vector is a vector that defines the fractional implementation of a policy over time periods.

.. equation:: 

- `n_tp_ramp`: Number of time periods it takes for the intervention to reach full effect. If not specified
- `tp_0_ramp`: Final time period with 0 change from baseline
- `a`: sigmoid magnitude parameter; set to 0 for linear, 1 for full sigmoid
- `b`: linear coefficient; set to 2 for linear (div by 2) or 0 for sigmoid
- `c`: denominator exponee--in linear, set to 1 (adds term 1 + 1 to denominator); for sigmoid, set to np.e
- `d` (optional): centroid for sigmoid/linear function. If using a sigmoid, this is the position of 0.5 in years :math:`\geq r_0`

Linear vector
set a = 0, b = 2, c = 1, d = r_0 + (n - r_0 - r_1)/2

Sigmoid:
set a = 1, b = 0, c = math.e, d = r_0 + (n - r_0 - r_1)/2

