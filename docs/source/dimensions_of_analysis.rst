======================
Dimensions of Analysis
======================

Dimensions of Analysis - this section will detail the dimensions of the analysis, including future_id, strategy_id, time_series_id,

SISEPUEDE is composed of three key experimental dimensions that are used to evaluate emissions under different policy strategies and uncertainties:

# Futures
# Strategies
# Design and Lever Effects

Futures
=======

**Futures** are the


Strategies
==========
**Strategies**, indexed by ``strategy_id``, combine transformations to generate whole-sector and economy-wide transformations. 

.. csv-table:: A number of strategies. Note the numbering scheme; AFOLU-specific transformations occupy 1001-1999; Circular Economy, 2001-2999; Energy, 3001-3999; IPPU, 4001-4999; and cross-sector, 5001-5999.
   :file: ./csvs/attribute_dim_strategy_id.csv
   :header-rows: 1


Designs and Lever Effects
=========================
The ``design_id`` controls experiments and how lever effects are explored using uncertainty.

.. csv-table:: Current design specifications  
   :file: ./csvs/attribute_dim_design_id.csv
   :header-rows: 1


Primary Key
===========
