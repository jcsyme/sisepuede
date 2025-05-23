============
General Data
============

**SISEPUEDE** includes key elements to support conversion between different variables and support different reporting quantities (e.g., emissions in MT or GT). The following section enumerates conversion factors and units for some of the key elements of SISEPUEDE, including gasses and defined units of energy, length, mass, and volume (used in emission accounting).


Sectors and Subsectors
======================

SISEPUEDE models emissions in four key sectors: AFOLU, Circular Economy, Energy, and IPPU. Additional, emissions are driven by activity in the Socioeconomic sector.

.. csv-table:: Emissions sectors in SISEPUEDE
   :file: ../../sisepuede/attributes/attribute_sector.csv
   :header-rows: 1

Each of the four key emissions sectors and the socioeconomic sector are divided into several subsectors, which are detailed below.

.. csv-table:: Subsectors modeled in SISEPUEDE
   :file: ../../sisepuede/attributes/attribute_subsector.csv
   :header-rows: 1

----


Gasses
======

Emissions are calculated in a unit of mass (default MT) for each relevant gas included. For :math:`\text{CO}_2\text{e}` conversions, the default Global Warming Potential (GWP) time horizon is 100 years. However, the GWP time horizon can be changed in the `Analytical Parameters <../analytical_parameters.html>`_ configuration file. Most GWP conversion factors below are taken from `IPCC AR6 WG1 Chapter 7 - Table 7.SM.7 <https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter_07_Supplementary_Material.pdf>`_ (referred to as IPCC AR6 below), though GWPs for a few gasses were sourced elsewhere.

See `Chapter 7, Section 6.1 of the IPCC Sixth Assessment Report (AR6) <https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07.pdf>`_ for more detail on global warming potential and how it is calculated.

.. csv-table:: Gasses potentially included in SISEPUEDE and their CO2 equivalent
   :file: ../../sisepuede/attributes/attribute_gas.csv
   :header-rows: 1


----


Regions
=======

SISEPUEDE includes aggregated data from over 170 countries worldwide, though more refined data are available for Latin America and select additional countries. These regions are associated with different NDCs, power grids, governmental structures and political regimes. 

Currently, regions are treated as independent in the model, and exports are exogenously defined. Future updates to SISEPUEDE will include broader regional integration to improve understanding of optimal power sharing and trade. 

.. note:: Regions are associated with population-centroids of latitude and longitude; these geospatial coordinates are primarily used to estimate country-level average solar irradiance and availability for solar power generation. Updates to SISEPUEDE will include a more refined treatment of sectoral-level demand variability due to solar cylces.

.. csv-table:: The following REGION dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ../../sisepuede/attributes/attribute_region.csv
   :header-rows: 1


----


Units
=====

SISEPUEDE includes versatile units reporting and conversion to allow for users to set output units in terms relevant to their locale and/or question. 

Units - Area
------------
The standard reporting output for area (e.g., energy demand) can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of area for reporting is ha (hectares).

.. csv-table:: Area units defined in SISEPUEDE and relationships between them.
   :file: ../../sisepuede/attributes/attribute_unit_area.csv
   :header-rows: 1


Units - Energy
--------------
The standard reporting output for energy (e.g., energy demand) can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of energy for reporting is PJ (Petajoule).

.. note:: The energy attribute table includes relationships mapping certain units of Energy to associated time-durational power units (e.g., kWh to kW or GWy to GW).


.. csv-table:: Energy units defined in SISEPUEDE and relationships between them and power.
   :file: ../../sisepuede/attributes/attribute_unit_energy.csv
   :header-rows: 1


Units - Length
--------------
The standard reporting output for any output lengths can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of length for reporting length-relevant information (e.g., transportation demand) is km (kilometers).

.. csv-table:: Length units defined in SISEPUEDE and relationships between them.
   :file: ../../sisepuede/attributes/attribute_unit_length.csv
   :header-rows: 1


Units - Mass
------------
The emissions accounting mass can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of mass for reporting emissions is MT (megatons).

.. csv-table:: Mass units defined in SISEPUEDE and relationships between them.
   :file: ../../sisepuede/attributes/attribute_unit_mass.csv
   :header-rows: 1


Units - Monetary
----------------
The default output units for CAPEX and OPEX from the Fuel Production model are set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default units for monetary units is million USD (mm_usd).

.. note::SISEPUEDE currently requires an external R script, SISEPUEDE-CBA, to estimate costs and benefits associated with policy transformations. SISEPUEDE is currently undergoing updates to integrate these scripts into the SISEPUEDE fraemwork.

.. csv-table:: Monetary units defined in SISEPUEDE and relationships between them.
   :file: ../../sisepuede/attributes/attribute_unit_monetary.csv
   :header-rows: 1


Units - Power
-------------
The standard reporting output for power (e.g., produced power) can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of power for reporting is GW (Gigawatt).

.. note:: The power attribute table includes relationships mapping certain units of Power to associated energy units over an assumed period of time (e.g., kW to kWh or GW to GWy).


.. csv-table:: Power units defined in SISEPUEDE and relationships between them and energy.
   :file: ../../sisepuede/attributes/attribute_unit_power.csv
   :header-rows: 1


Units - Volume
--------------
The standard output volume for output volume units can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of volume for reporting volumes (such as wastewater) is :math:`m^3` (cubic meters).

.. csv-table:: Volume units defined in SISEPUEDE and relationships between them.
   :file: ../../sisepuede/attributes/attribute_unit_volume.csv
   :header-rows: 1