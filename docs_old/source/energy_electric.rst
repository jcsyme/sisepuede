========================================
Energy - Electricity and Fuel Production
========================================

Electricity and fuel production are modeled using `Julia NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. Two key subsectors are exclusive to electricity and fuel production (* indicates non-emission subsector): 

* Energy Technology (ENTC--NOTE this is marked as **Electricity and Fuel Production** on the `SISEPUEDE DAG <./sisepuede_concept.html>`_)
* Energy Storage (ENST)*

The SISEPUEDE model is highly aggregated at the country level, that can be built and improved upon by countries with deeper systemic knowledge at a later date. In general, SISEPUEDE acts as a wrapper for Julia NEMO, formatting input data and integrating uncertainty into an integrated modeling framework for NemoMod. **Energy Technology** is the subsector in which emissions accrue and contains key attributes surrounding energy generation technologies, including capacity factors, costs, lifetimes, input and output fuels, and more. **Energy Storage** includes key properties of energy storage. All storage categories are also technology categories (this is necessary to model storage).

NemoMod requires several dimensions of data; these data include:

* EMISSIONS
* FUEL
* MODE
* STORAGE
* TECHNOLOGY
* TIMESLICE
* TSGROUP1
* TSGROUP2
* YEARS

These dimensions are associated with attribute tables in SISEPUEDE scripts and the ModelAttributes class, and some--such as FUEL (ENFU), STORAGE (ENST), and TECHNOLOGY (ENTC)--are subsectors in SISEPUEDE. 

.. note::
   Most of the variables that are required by category are explained in further detail in the `NemoMod Parameter Documentation <https://sei-international.github.io/NemoMod.jl/stable/parameters/>`_. For example, if it is unclear what the *Capacity Factor* is (see Categories - TECHNOLOGY below), the NemoMod parameter documentation can provide additional information.


NemoMod Tables and Default Values
=================================

NemoMod includes a range of tables used to integrate components of an energy system. The following table includes SISEPUEDE default parameter values for each table as well as information about how each table is populated in SISEPUEDE.

.. csv-table:: NemoMod Table attributes
   :file: ./csvs/attribute_nemomod_table.csv
   :header-rows: 1



Categories - Region
-------------------

NemoMod allows users to specify regions, and policies can be modeled that represent cross-regional power transfers, storage, etc. In the SISEPUEDE NemoMod implementation, each country is treated as a region.

----


Energy Storage (STORAGE)
========================

**DESCRIPTION OF STORAGE AND INTERACTIONS WITH TECH**


Variables by Categories
-----------------------

The following variables are required for each category ``$CAT-STORAGE$``.

.. csv-table:: For different Energy Storage categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_category_en_enst.csv
   :header-rows: 1


Categories
----------

.. csv-table:: The following STORAGE dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_storage.csv
   :header-rows: 1

---



Energy Technology (TECHNOLOGY)
==============================

The SISEPUEDE model (v1.0) uses NemoMod *only* to model the electricity sector. Therefore, technologies are limited to power generation (power plants) and storage.

.. csv-table:: The following TECHNOLOGY dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_technology.csv
   :header-rows: 1


Variables by Category
---------------------

The following variables are required for each category ``$CAT-TECHNOLOGY$``. Note that these technologies represent consumers of fuel (including electricity); in SISEPUEDE, this is restricted to generation technology and storage.

.. csv-table:: For each Energy Technology category, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_category_en_entc.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

The following variables are required for some categories ``$CAT-TECHNOLOGY$``.

.. csv-table:: For different technology categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_entc.csv
   :header-rows: 1

----
