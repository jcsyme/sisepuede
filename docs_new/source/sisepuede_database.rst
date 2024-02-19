======================
The SISEPUEDE Database
======================

Each instantiation of SISEPUEDE relies on a unique AnalysisID, which is used to define an output database. The use of a unique ID ensures that a SISEPUEDE session can be restored easily.

- Describe database concepts, structure, etc.

- Recommended merges

- Discuss the AnalysisID
  - code on instantiating using an existing analysis id

.. note::
  AnalysisIDs from file names require the replacement of semicolons with colons (add example, clarify)


Default Tables
==============

ANALYSIS_METADATA
-----------------
The `ANALYSIS_METADATA` table contains the configuration parameters used to setup SISEPUEDE for the session. These parameters include information on time periods, output units (for emissions, length, area, etc.), assumptions about historical data in Circular Economy and Energy, and more. For more information on configuration parameters, see `Analytical Parameters <./analytical_parameters.html>`_.


ATTRIBUTE_DESIGN
----------------
The `ATTRIBUTE_DESIGN` table contains the attribute table describing the `design_id <./dimensions_of_analysis.html#designs-and-lever-effects>`_ dimension of analysis, which characterizes the magnitudes of lever effects and uncertainty fans.

.. add table


ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES
---------------------------------------------
The `ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES` table contains raw LHC samples for each exogenous uncertainty sample group (fields). Each row contains the LHC trials used to generate `exogenous uncertainties (future_id) <./dimensions_of_analysis.html#futures>`_ in SISEPUEDE. These LHC trials are separate from those used to develop uncertainties in lever effects (see below).


ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS
-----------------------------------
The `ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES` table contains raw LHC samples associated with levers (wide by strategy sample group). Each row contains the LHC trials used to generate `lever effect uncertainties (future_id) <./dimensions_of_analysis.html#futures>`_ in SISEPUEDE.


ATTRIBUTE_PRIMARY
-----------------
The `ATTRIBUTE_PRIMARY` table defines the `primary key (primary_id) <dimensions_of_analysis.html#primary-key>`_ for the runs contained in the database.

.. note:: To save memory and disk space, SISEPUEDE stores all primary key information in an ``OrderedDirectProductTable`` class, which does not explicitly store all rows of the table. Instead, SISEPUEDE **only stores rows associated with `primary_id` values that are contained within the output database**. The following ``OrderedDirectProductTable`` methods can be used to find more information on the primary key indexing.

    * ``OrderedDirectProductTable.get_dims_from_key()``: Get dimensional values associated with a key (inverse of ``get_key_value``)
    * ``OrderedDirectProductTable.get_key_value()``: Get a key value associated with dimensional values (inverse of ``get_dims_from_key``)
    * ``OrderedDirectProductTable.get_indexing_dataframe()``: Get a data frame associated with select dimensional values or with key values.


ATTRIBUTE_STRATEGY
------------------
The `ATTRIBUTE_STRATEGY` table contains all information used to define strategies in SISEPUEDE, including the name of the strategy, baseline specification (should only exist for one strategy--by convention, it is recommended that this strategy is set to 0), and the `strategy_id <./dimensions_of_analysis.html#strategies>`_.

.. add table


MODEL_BASE_INPUT_DATABASE
-------------------------
The `MODEL_BASE_INPUT_DATABASE` explicitly stores all information used to define input data for baseline futures for all regions; it is long by `region`, `strategy`, and `time_period` and wide by input variable.


MODEL_INPUT
-----------
The `MODEL_INPUT` table stores explicit model inputs for every scenario that is available in the database. It is long by `primary_id`, `region`, and `time_period`. The `MODEL_INPUT` table is **optional**; by default, it is not stored to save space. **NOTE: ADD MECHANISM FOR ALLOWING INPUT TABLE TO BE WRITTEN**

.. note:: Individual futures (and the table itself) can be reproduced quickly using SISEPUEDE's internal functions in combination with LHS tables, which are saved by default. To generate an input table associated with a primary key, use ``SISEPUEDE.generate_scenario_database_from_primary_key()``.


MODEL_OUTPUT
------------
The `MODEL_OUTPUT` table stores explicit model outputs for every scenario (indexed by `primary_id`) that is run under the given AnalysisID. It is long by `primary_id`, `region`, and `time_period`.


Optional and Derivative Tables
==============================

SISEPUEDE contains a mechanism for storing derivative tables using specified functions. DESCRIBE LATER
