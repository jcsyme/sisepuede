============
Transformers
============

SISEPUEDE includes a predefined library of **transformations** that can be customized and combined to create detailed and comprehensize strategies--based on expected outcomes--for reducing emissions of greenhouse gasses at scale. 

Transformations are outcomes that can be acheieved through the implementation of one or more policies. For example, mode shifting transportation can be achieved through a number of potential policies, including urban planning, taxes on fuels, reductions of public transportation fares, construction of infrastructure, and more. 

Transformations are defined in `Transformation` classes in python. Collections of transformations are called **strategies** (link).


Transformers vs. Transformations
================================

SISEPUEDE uses simlar terms to refer to two different components of the framework for modifying  

**Transformers** are pre-defined callable classes in Python that modify a base set of trajectories to reflect a desired outcome. These classes, which include default values, can be called with different functional specifications that are defined in configuration files to allow for flexibility in applying the transformation. 


Transformer Conventions
-----------------------

Transformers follow some conventions. Understanding these conventions can help increase readability and interpretation of conventions. 

Transformer Codes follow a simple structure

``TFR:SUBSEC_CODE:ACTION_DESCRIPTION``

Actions are generally defined using the following abbreviations:

- `DEC`: Decrease
- `INC`: Increase
- `SHIFT`: Move from one category to another

For example,


Transformers
============

.. csv-table:: Transformers included in SISEPUEDE
   :file: ../../sisepuede/attributes/attribute_transformer_code.csv
   :header-rows: 1


.. the module should be available from path set in conf.py https://sphinx-tutorial.readthedocs.io/step-2/
.. autoclass :: sisepuede.transformers.transformers.Transformers
    :members: _trfunc_agrc_improve_rice_management

Transformer Docstrings 
----------------------

Note that parameters shown in the docstrings below are keyword arguments. In general, the user should **not** set a value for the ``strat`` keyword argument.

.. generate list of all transformers
.. autofunction:: transformers.Transformers.agrc_decrease_exports
.. autofunction:: transformers.Transformers.agrc_expand_conservation_agriculture
.. autofunction:: transformers.Transformers.agrc_improve_rice_management
.. autofunction:: transformers.Transformers.agrc_increase_crop_productivity
.. autofunction:: transformers.Transformers.agrc_reduce_supply_chain_losses
.. autofunction:: transformers.Transformers.baseline
.. autofunction:: transformers.Transformers.ccsq_increase_air_capture
.. autofunction:: transformers.Transformers.entc_clean_hydrogen
.. autofunction:: transformers.Transformers.entc_least_cost
.. autofunction:: transformers.Transformers.entc_reduce_transmission_losses
.. autofunction:: transformers.Transformers.entc_renewable_electricity
.. autofunction:: transformers.Transformers.fgtv_maximize_flaring
.. autofunction:: transformers.Transformers.fgtv_minimize_leaks
.. autofunction:: transformers.Transformers.inen_fuel_switch_heat
.. autofunction:: transformers.Transformers.inen_maximize_energy_efficiency
.. autofunction:: transformers.Transformers.inen_maximize_production_efficiency
.. autofunction:: transformers.Transformers.ippu_demand_managment
.. autofunction:: transformers.Transformers.ippu_reduce_cement_clinker
.. autofunction:: transformers.Transformers.ippu_reduce_hfcs
.. autofunction:: transformers.Transformers.ippu_reduce_n2o
.. autofunction:: transformers.Transformers.ippu_reduce_other_fcs
.. autofunction:: transformers.Transformers.ippu_reduce_pfcs
.. autofunction:: transformers.Transformers.lndu_expand_silvopasture
.. autofunction:: transformers.Transformers.lndu_expand_sustainable_grazing
.. autofunction:: transformers.Transformers.lndu_increase_reforestation
.. autofunction:: transformers.Transformers.lndu_partial_reallocation
.. autofunction:: transformers.Transformers.lndu_stop_deforestation
.. autofunction:: transformers.Transformers.lsmm_improve_manure_management_cattle_pigs
.. autofunction:: transformers.Transformers.lsmm_improve_manure_management_other
.. autofunction:: transformers.Transformers.lsmm_improve_manure_management_poultry
.. autofunction:: transformers.Transformers.lsmm_increase_biogas_capture
.. autofunction:: transformers.Transformers.lvst_decrease_exports
.. autofunction:: transformers.Transformers.lvst_increase_productivity
.. autofunction:: transformers.Transformers.lvst_reduce_enteric_fermentation
.. autofunction:: transformers.Transformers.pflo_industrial_ccs
.. autofunction:: transformers.Transformers.plfo_healthier_diets
.. autofunction:: transformers.Transformers.scoe_fuel_switch_electrify
.. autofunction:: transformers.Transformers.scoe_increase_applicance_efficiency
.. autofunction:: transformers.Transformers.scoe_reduce_heat_energy_demand
.. autofunction:: transformers.Transformers.soil_reduce_excess_fertilizer
.. autofunction:: transformers.Transformers.soil_reduce_excess_liming
.. autofunction:: transformers.Transformers.trde_reduce_demand
.. autofunction:: transformers.Transformers.trns_electrify_light_duty_road
.. autofunction:: transformers.Transformers.trns_electrify_rail
.. autofunction:: transformers.Transformers.trns_fuel_switch_maritime
.. autofunction:: transformers.Transformers.trns_fuel_switch_medium_duty_road
.. autofunction:: transformers.Transformers.trns_increase_efficiency_electric
.. autofunction:: transformers.Transformers.trns_increase_efficiency_non_electric
.. autofunction:: transformers.Transformers.trns_increase_occupancy_light_duty
.. autofunction:: transformers.Transformers.trns_mode_shift_freight
.. autofunction:: transformers.Transformers.trns_mode_shift_public_private
.. autofunction:: transformers.Transformers.trns_mode_shift_regional
.. autofunction:: transformers.Transformers.trww_increase_biogas_capture
.. autofunction:: transformers.Transformers.trww_increase_septic_compliance
.. autofunction:: transformers.Transformers.wali_improve_sanitation_industrial
.. autofunction:: transformers.Transformers.wali_improve_sanitation_rural
.. autofunction:: transformers.Transformers.wali_improve_sanitation_urban
.. autofunction:: transformers.Transformers.waso_descrease_consumer_food_waste
.. autofunction:: transformers.Transformers.waso_energy_from_biogas
.. autofunction:: transformers.Transformers.waso_energy_from_incineration
.. autofunction:: transformers.Transformers.waso_increase_anaerobic_treatment_and_composting
.. autofunction:: transformers.Transformers.waso_increase_biogas_capture
.. autofunction:: transformers.Transformers.waso_increase_landfilling
.. autofunction:: transformers.Transformers.waso_increase_recycling




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

