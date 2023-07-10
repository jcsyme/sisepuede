===========================================
Agriculture, Forestry, and Land Use (AFOLU)
===========================================


AFOLU includes six subsectors (* indicates non-emission subsector): 

* Agriculture (AGRC)
* Forestry (FRST)
* Land Use (LNDU)
* Livestock (LVST)
* Livestock Manure Management (LSMM)
* Soil (SOIL)

These six sectors are based on Volume 4 of the IPCC guidance for national 
greenhouse gas inventories and include extensive treatment of key emission 
phenomena, including crop residues and burning, forest sequestration, land use 
conversion and use, enteric fermentation, manure management, fertilizer 
application, soil carbon sequestration in mineral and organic solis, and more. 
Land use models transitions directly as a discrete Markov chain, allowing for a 
detailed accounting of emissions due to land use conversion. Furthermore, it 
includes a novel mechanism to model land use changes that occur in response to 
changing demands for livestock and crops. Demands for crops and livestock 
production are generally based on historical production, imports, and exports 
and are responsive to changes in trade and GDP, GDP/capita, and population.  


See the `AFOLU Mathematical Documentation <./mathdoc_afolu.htm>`_ for more information on the model structure, including mathematical formulae and assumptions.


Agriculture
===========

The **Agriculture** subsector is used to quantify emissions associated with growing crops, including emissions from the release of soil carbon, fertilizer applications and crop liming, crop burning, methane emissions from paddy rice fields, **AND MORE;CONTINUE**. Agriculture is divided into the following categories (crops), given by the metavariable ``$CAT-AGRICULTURE$``. Each crop should be associated an FAO classifications. `See the FAO <https://www.fao.org/waicent/faoinfo/economic/faodef/annexe.htm>`_ for the source of these classifications and a complete mapping of crop types to categories. On the git, the table ``ingestion/FAOSTAT/ref/attribute_fao_crop.csv`` contains the information mapping each crop to this crop type. Note, this table can be used to merge and aggregate data from FAO into these categories. If a crop type is not present in a country, set the associated area as a fraction of crop area to 0.

.. note:: Carbon stocks are scaled by 44/12 to estimate :math:`\text{CO}_2` emissions. See Section 2.2.3 of the `IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc.ch/report/2019-refinement-to-the-2006-ipcc-guidelines-for-national-greenhouse-gas-inventories/>`_.

Variables by Category
---------------------

Agriculture requires the following variables.

.. csv-table:: For each agricultural category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_agrc.csv
   :header-rows: 1
.. :widths: 20, 30, 30, 10, 10

.. note::  | To reduce the number of potential variables, types are associated with some key physical characteristics that are used to estimate :math:`\text{N}_2\text{O}` emissions, including :math:`\text{N}_{AG(T)}`, :math:`\text{N}_{BG(T)}`, :math:`\text{R}_{AG(T)}`, :math:`\text{RS}_{T}`, and :math:`DRY`, which are derived from Table 11.1 in Volume 4, Chapter 11 of the `IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf>`_.
 |
 | These variables are used in Equations 11.6 and 11.7 (Volume 4) of the IPCC NGHGI to estimate :math:`\text{N}_2\text{O}`


Variables by Partial Category
-----------------------------

 Agriculture includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

 .. csv-table:: Trajectories of the following variables are needed for **some** categories `$CAT-AGRICULTURE`. If they are independent of categories, the category will show up as **none**.
    :file: ./csvs/table_varreqs_by_partial_category_af_agrc.csv
    :header-rows: 1


Categories
----------

Agriculture is divided into the following categories.

.. csv-table:: Agricultural categories (``$CAT-AGRICULTURE$`` attribute table)
   :file: ./csvs/attribute_cat_agriculture.csv
   :header-rows: 1
..   :widths: 15,15,30,15,10,15



**Costs to be added**

----

Forestry
========

Variables by Category
---------------------

.. csv-table:: For each forest category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_frst.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

Forestry includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** forest categories. If they are independent of categories, the category will show up as **none**.
   :file: ./csvs/table_varreqs_by_partial_category_af_frst.csv
   :header-rows: 1

Categories
----------

Forestry is divided into the following categories. These categories reflect an aggregation of forestry types into emission-relevant categories. Note that areas of forested land are determined in the **Land Use** subsector. The land use at time *t* is determined by an ergodic Markov Chain (probabilities are set in the variable input table and subject to uncertainty using the mixing approach)

.. csv-table:: Forest categories (``$CAT-FOREST$`` attribute table)
   :file: ./csvs/attribute_cat_forest.csv
   :header-rows: 1
..   :widths: 15,15,30,15,10,15


----

Land Use
========

Land use projections are driven by a Markov Chain, represented by a transition matrix :math:`Q(t)` (the matrix is specified for each time period in the input template). The model requires initial states (entered as a fraction of total land area) for all land use categories ``$CAT-LANDUSE$``. See the `AFOLU Mathematical Documentation <./mathdoc_afolu.htm>`_ for more information on the integrated land use model.

.. note::
   The entries :math:`Q_{ij}(t)` give the transition probability of land use category :math:`i` to land use category :math:`j`. :math:`Q` is row stochastic, so that :math:`\sum_{j}Q_{ij}(t) = 1` for each land use category :math:`i` and time period :math:`t`. To preserve row stochasticity, it is highly recommended that strategies and uncertainty be represented using the trajectory mixing approach, where bounding trajectories on transitions probabilities are specified and uncertainty exploration gives a mix between them.

Land Use Mechanisms
-------------------

**BRIEF DESCRIPTION**

Land Use Reallocation Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The land use transition model includes what is referred to as the *Land Use Reallocation Factor* (LURF). The LURF helps reconcile differences between an exogenous projection of a land use transition matrix and endogenous changes that would be required to adapt to changing demands for production of livestock and crops. The LURF--which is referred to throughout the SISEPUEDE documentation as :math:`\eta`--can be set to any real number in the interval [0, 1], i.e., :math:`\eta \in [0, 1]`.

When running a model with an exogenous specification of land use transition probabilities, the demand (:math:`D`) for production of crops and livestock may exceed (or not meet) the supply (:math:`S`) that is implied by the area of land and the production per area (grazing livestock per area for pastures, yield per area in crops, and livestock feed yield per area of relevant crop classes). If demand is not equal to supply, then there is an imbalance :math:`I = D - S` (also referred to as *surplus demand*). This imbalance can be compensated in any combination of two ways:

#. Changing net imports of the crop or animal (:math:`I > 0 \implies` the change to net imports is positive); and/or

#. Reallocating land use categories away from the exogenous transition matrix to increase or decrease available supply.

The value of :math:`\eta` represents the fraction of unmet demand, in pasture and cropland categories, that is allocated to the second option, i.e., the amount of demand that is used to calculate changes to pasture and cropland areas. If :math:`\eta = 0`, then no land is reallocated to account for the demand/supply imbalance, and surplus demand is added to net imports (surplus demand can be negative). If :math:`\eta = 1`, then **all** imbalance is reconciled by reallocating cropland and pastures so that supply is equal to demand, and :math:`D = S \implies I = 0`. For values of :math:`\eta \in (0, 1)`, some surplus demand is met through changes to net imports, while some is met through land use reallocation.

.. note:: In the ``$CAT-LANDUSE$`` attribute file, categories can be specified as a *Reallocation Transition Probability Exhaustion Category* The configuration file includes the *land_use_reallocation_max_out_directionality* parameter. This parameter can take on three values:

   #. decrease_only (Default): If, during land use reallocation, the demand for cropland and/or pasture **decreases**, then transition probabilities out of land use categories specified in as Reallocation Transition Probability Exhaustion Categories (into cropland or pastures) will be minimized before scaling other inbound transition probabilities (they are bound by 0). If increasing, all inbound transition probabilities to cropland and pastures are scaled uniformly.

   #. increase_only: If, during land use reallocation, the demand for cropland and/or pasture **increases**, then transition probabilities out of land use categories specified as Reallocation Transition Probability Exhaustion Categories (into cropland or pastures) will be maximized before scaling other inbound transition probabilities (they are bound by 1). If decreasing, all inbound transition probabilities to cropland and pastures are scaled uniformly.

   #. decrease_and_increase: If the demand for cropland and/or pasture **decreases** or **increases**, then transition probabilities out of land use categories specified as Reallocation Transition Probability Exhaustion Categories (into cropland or pastures) will be minimized or maximized (respectively) before scaling other inbound transition probabilities (they are bound by 0 and 1, respectively).


Fraction of Increasing Net Exports/Imports Met
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text here


Variables by Category
---------------------

.. csv-table:: For each land use category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_lndu.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

Land use includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. note::
   Note that the sum of all initial fractions of area across land use categories *u* should be should equal 1to , i.e. :math:`\sum_u \varphi_u = 1`, where :math:`\varphi_{\text{$CAT-LANDUSE$}} \to` ``frac_lu_$CAT-LANDUSE$`` at period *t*.

.. csv-table:: Trajectories of the following variables are needed for **some** land use categories.
   :file: ./csvs/table_varreqs_by_partial_category_af_lndu.csv
   :header-rows: 1
.. :widths: 15, 15, 20, 10, 10, 10, 10, 10

Categories
----------

Land use should be divided into the following categories, given by ``$CAT-LANDUSE$``.

.. csv-table:: Land Use categories (``$CAT-LANDUSE$`` attribute table)
   :file: ./csvs/attribute_cat_land_use.csv
   :header-rows: 1

----


Livestock
=========

For each category, the following variables are needed. Information on enteric fermentation can be found from `the EPA <https://www3.epa.gov/ttnchie1/ap42/ch14/final/c14s04.pdf>`_ and **ADDITIONAL LINKS HERE**.

Variables by Category
---------------------

.. csv-table:: For each livestock category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_lvst.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

Livestock includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** livestock categories.
   :file: ./csvs/table_varreqs_by_partial_category_af_lvst.csv
   :header-rows: 1


Categories
----------

Livestock should be divided into the following categories, given by ``$CAT-LIVESTOCK$``.

.. note::
   Animal weights are only used to estimate the increase in protein consumption in liquid waste (which contribute to :math:`\text{N}_2\text{O}` emissions). All estimates are adapted from `Holechek 1988 <https://journals.uair.arizona.edu/index.php/rangelands/article/download/10362/9633>`_ (using 2.2 lbs/kg) unless otherwise noted.

.. csv-table:: Livestock categories (``$CAT-LIVESTOCK$`` attribute table)
   :file: ./csvs/attribute_cat_livestock.csv
   :header-rows: 1



Livestock Manure Management
===========================

EXPLANATION HERE

Variables by Category
---------------------

.. csv-table:: For each livestock category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_lsmm.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

Livestock manure management includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

   .. csv-table:: Trajectories of the following variables are needed for **some** livestock manure management categories.
      :file: ./csvs/table_varreqs_by_partial_category_af_lsmm.csv
      :header-rows: 1


Categories
----------

Manure management is divided into the following categories, given by ``$CAT-MANURE-MANAGEMENT$``.

.. csv-table:: Livestock manure management categories (``$CAT-MANURE-MANAGEMENT$`` attribute table)
   :file: ./csvs/attribute_cat_manure_management.csv
   :header-rows: 1



Soil Management
===============

EXPLANATION HERE

Variables by Category
---------------------

.. csv-table:: For each soil management category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_soil.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

   Soil management includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

   .. csv-table:: Trajectories of the following variables are needed for **some** soil management categories.
      :file: ./csvs/table_varreqs_by_partial_category_af_soil.csv
      :header-rows: 1


Categories
----------

Soil management is divided into the following categories, given by ``$CAT-SOIL-MANAGEMENT$``.

.. csv-table:: Soil management categories (``$CAT-SOIL-MANAGEMENT$`` attribute table)
   :file: ./csvs/attribute_cat_soil_management.csv
   :header-rows: 1
