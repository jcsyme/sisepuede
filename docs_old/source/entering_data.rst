=============
Entering Data
=============


Interacting with SISEPUEDE
==========================


Category Attribute Tables
=========================



Variable Attribute Tables
=========================

Attribute tables are fundamental components of the data framework underlying SISEPUEDE. Attribute tables define sectors and categories associated with those sectors (those starting with ``cat_``); model variables and associated attributes, including variable schema (which include information on units--those attribute tables starting with 


Design
Region
Strategy
Time Period

See `Dimensions of Analysis <../dimensions_of_analysis.html>`_ for more information on the dimensions of analysis and what they control.

The SISEPUEDE framework relies on a collection of tools and classes--including the ``InputTemplate``, ``FutureTrajectories``, ``SamplingUnit``, and ``LHSDesign`` classes--to represent input data, including uncertainties and levers, and modify them to facilitate robust exploratory modeling (see `Entering Data <../entering_data.html>`_ for more information on these classes)


Variable Name
-------------

Variable Schema
---------------

The ``$VARNAME$`` notation to denote metavariables as components of variable schema. These variable schema are used to store data in the ``model_input_variables.csv`` file.

For example, model input variables used to denote agricultural activity emission factors by crop type and gas in ``model_input_variables.csv`` may have the following structure:
``ef_agactivity_$CAT-AGRICULTURE$_$UNIT-MASS$_$EMISSION-GAS$_$UNIT-AREA$``, where

- ``$CAT-AGRICULTURE$`` is the categorical crop type (e.g., cereals, oil crops, pulses, etc.);
- ``$EMISSION-GAS$`` is the greenhouse gas that is being emitted (e.g. ``co2``, ``ch4``, ``n2o``, ``sf6``, etc.)
- ``$UNIT-MASS$`` is the unit of mass for gas emission (e.g., ``kg`` for kilograms; some sector variables may use ``gg`` for gigagrams);
- ``$UNIT-AREA$`` is the area unit (e.g., ``ha`` is hectares).

These components are referred to as *metavariables*--they characterize and describe the notation for naming model input variables. Each variable is associated with some naming *schema*, which presents a standardized format for variable entry depending on the relevant metavariables.

.. note::
   Example: the :math:`\text{CO}_2` emission factor for maize crop production, which captures crop burning, decomposition, and other factors, would be entered as ``ef_agactivity_maize_kg_co2_ha`` since, in this case, ``$CAT-AGRICULTURE$ = maize``, ``$EMISSION-GAS$ = co2``, ``$UNIT-AREA$ = ha``, and ``$UNIT-MASS$ = kg``. Similarly, the :math:`\text{N}_2\text{O}` factor, which includes crop liming and fertilization, would be captured as ``ef_agactivity_maize_kg_n2o_ha``.

.. Variable ranges can be set for individual dimensions and for the space of any dimensions ``$CAT-INDUSTRY$ = paper|cement|plastic``, ``$CAT-INDUSTRY-DIM1$ = product_use_lubricants|product_use_paraffin_wax|cement|plastic``, ``$CAT-TRANSPORTATION$ = aviation|rail_freight|dymm` `since elements_iter applies roots first, if categories for roots are
                specified in the presence of multiple dimensions, it will 
                initialize those restrictions for the children. If children are
                also specified, then they will overwrite the root's 
                specification.

                E.g., X -> R will set X-DIM1, X-DIM2, and X-DIM3 to R; then, 
                additionally specifying X-DIM2 as s will overwrite X-DIM2 (only)
                as S.


Simplex Group
-------------

Simplex groups are unique to each attribute table and are entered in the *Simplex Group* field as integers. This marker allows users to specify variable trajectory groups on a standard simplex with a flag. Which fields must sum to 1 are defined by how the groups are entered: 

* If one variable is associated with a unique simplex group (i.e., 1:1), then the sum of all fields associated with that variable must equal 1.
* If multiple variable are associated with a unique simplex group (i.e., one simplex group maps to four variables), the the sum across variables must be 1 *for each category across the variables*.

Simplex groups cannot be specified outside of a single attribute table.



Input Templates
===============

- Each region has a template for each sector
- five templates
- 3 different types: calibrated, uncalibrated, demo



Defining Futures
----------------


Defining Strategies
-------------------



Variable Trajectory Groups
--------------------------



*Variable trajectory groups* allow users to specify collctions of variables that will vary using the same Latin Hypercube Sample. There are two approaches to defining

Defining a Variable Trajectory Group using the Bound-Mix Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(:math:`b_0`, :math:`b_1`, :math:`m`)

.. note: If using a variable trajectory group with a Bound-Mix approach, note that exogenous uncertainties are not applied to bounds. This ensure that any standard simplex summation (e.g., fractions that sum to 1) requirements are preserved. 


Defining a Variable Trajectory Group using Trajectories Alone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users do not have to use trjaectories 


Note on Variable Trajectory Groups and the Standard Simplex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. note: *Similarly*, if specifying trajectories on a standard simplex using only the variable trajectory group, the user should take care to avoid specifying scalars for uncertainty bounds; these can pull trajectories off of the standard simplex. 



Sampling Units
==============

SamplingUnits represent variable or collection of variables that are perturbed by a single Latin Hybercube (LHC) sample. There are a number of ways to define sampling units in the input templates.

... describe approaches here






Other Input Files
=================


NemoMod Reference Files
-----------------------
