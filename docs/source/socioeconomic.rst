=============
Socioeconomic
=============

The socioeconomic sector represents the primary drivers of emissions. These include economic and demographic factors that influence all emissions sectors of the economy. These factors are treated as exogenous uncertainties.

General (GNRL)
==============

Categories
----------

Categories associated with the General subsector are identified by the ``$CAT-GENERAL$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: General categories (``$CAT-GENERAL$`` attribute table)
   :file: ./csvs/attribute_cat_general.csv
   :header-rows: 1


Variables
---------

Variables associated with the General subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the General subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_se_gnrl.csv
   :header-rows: 1


----


Economy (ECON)
==============

The Economy subsector is used to represent exogenous economic drivers of emissions and is separate from the economic impact analysis. Note that Gross Domestic Product (GDP), which is treated as an exogenous uncertainty, in one of the key drivers of emissions, combining with elasticities to both GDP and GDP/capita to affect projections of future demands of products and transportation.


Categories
----------
Categories associated with the Economy subsector are identified by the ``$CAT-ECONOMY$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: Economy categories (``$CAT-ECONOMY$`` attribute table)
   :file: ./csvs/attribute_cat_economy.csv
   :header-rows: 1


Variables
---------

Variables associated with the Economy subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Economy subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_se_econ.csv
   :header-rows: 1
