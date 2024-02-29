=============
Socioeconomic
=============

The socioeconomic sector represents the primary drivers of emissions. These include economic and demographic factors that influence all emissions sectors of the economy. These factors are treated as exogenous uncertainties.

General
=======

Variables by Category
---------------------

.. csv-table:: For each general category category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_se_gnrl.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

The general socioeconomic subsector includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column. If the variable applies to no specific categorical value, the entry will read **none**.

.. csv-table:: Trajectories of the following variables are needed for **some** (or no) general categories.
   :file: ./csvs/table_varreqs_by_partial_category_se_gnrl.csv
   :header-rows: 1

Categories
----------

General should be divided into the following categories, given by ``$CAT-GENERAL$``.

.. csv-table:: General categories (``$CAT-GENERAL$`` attribute table)
   :file: ./csvs/attribute_cat_general.csv
   :header-rows: 1

----


Economy
=======

The *Economy* subsector is used to represent exogenous economic drivers of emissions and is separate from the economic impact analysis.
 .. note:: Gross Domestic Product (GDP) is entered as an exogenous uncertainty. Value added trajectories are included under ``$CAT-ECONOMY$`` and represent subsets of economic activity. The categories specified under ``$CAT-ECONOMY$`` do not have to be mutually exclusive, but they are considered exogenous and used to drive industrial emissions.

Variables by Category
---------------------

.. csv-table:: For each general exogenous economic category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_se_econ.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

   The economy socioeconomic subsector includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column. If the variable applies to no specific categorical value, the entry will read **none**.

   .. csv-table:: Trajectories of the following variables are needed for **some** (or no) economy categories.
      :file: ./csvs/table_varreqs_by_partial_category_se_econ.csv
      :header-rows: 1



Categories
----------
Economic value added trajectories should be divided into the following categories, given by ``$CAT-ECONOMY$``.

.. csv-table:: Economy categories (``$CAT-ECONOMY$`` attribute table)
   :file: ./csvs/attribute_cat_economy.csv
   :header-rows: 1
