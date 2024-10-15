===================================================
Energy Production - Electricity and Fuel Production
===================================================

Electricity and Fuel Production is modeled using `Julia NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. Two key subsectors are exclusive to electricity and fuel production (* indicates non-emission subsector): 

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
   :file: ../../sisepuede/attributes/attribute_nemomod_table.csv
   :header-rows: 1


----


Energy Storage (ENST)
=====================

**DESCRIPTION OF STORAGE AND INTERACTIONS WITH TECH**


Categories
----------

Categories associated with Energy Storage are identified by the ``$CAT-STORAGE$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: Energy Storage categories (``$CAT-STORAGE$`` attribute table) included in the SISEPUEDE NemoMod model.
   :file: ../../sisepuede/attributes/attribute_cat_storage.csv
   :header-rows: 1


Variables
---------

Variables associated with the Energy Storage subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Energy Storage subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ../../sisepuede/attributes/variable_definitions_en_enst.csv
   :header-rows: 1


---



Energy Technology (ENTC)
========================

The SISEPUEDE model (v1.0) uses NemoMod to model the production of energy, including:

* Electricity generation technology
* Fuel production and374585
 refinement, including petroleum and natural gas
* Mining and raw materials extraction, including coal mining and natural gas and oil exploration and extraction (*excluding* fugitive emissions, which are estimated in the `Fugitive Emissions (FGTV) <../energy_consumption.html#fugitive-emissions-fgtv>`_ subsector)

Categories
----------

Categories associated with Energy Technology are identified by the ``$CAT-TECHNOLOGY$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: Technology categories (``$CAT-TECHNOLOGY$`` attribute table) included in the SISEPUEDE NemoMod model.
   :file: ../../sisepuede/attributes/attribute_cat_technology.csv
   :header-rows: 1


Variables
---------

Variables associated with the Energy Technology subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Energy Technology subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ../../sisepuede/attributes/variable_definitions_en_entc.csv
   :header-rows: 1


Modeling Notes
--------------

Default input activity ratios are based on 