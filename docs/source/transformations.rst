===============
Transformations
===============

SISEPUEDE includes a predefined library of **transformations** that can be customized and combined to create detailed and comprehensize strategies--based on expected outcomes--for reducing emissions of greenhouse gasses at scale. 

Transformations are outcomes that can be acheieved through the implementation of one or more policies. For example, mode shifting transportation can be achieved through a number of potential policies, including urban planning, taxes on fuels, reductions of public transportation fares, construction of infrastructure, and more. 

Transformations are defined in `Transformation` classes in python. Collections of transformations are called **strategies** (link).


Predefined Transformations
==========================



Parameterizing Transformations
==============================


Transformation Nomenclature
===========================



`magnitude_type`

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
* `vector_specification`: simply enter a vector to use for region
