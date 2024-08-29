===========================================
Industrial Processes and Product Use (IPPU)
===========================================

**Industrial Processes and Product Use (IPPU)** includes only one subsector (* indicates non-emission subsector): 

* Industrial Processes and Product Use (IPPU)

Emission accounting for IPPU is based primarily on Volume 3 of the IPCC guidance for national greenhouse gas inventories. The IPPU sector includes estimates of emissions of a range of gasses released during industrial production, including a by-gas accounting of a number of fluorinated compounds (including HFCs, PFCs, and other FCs) derived from other bottom-up estimates in the literature. Industrial production is driven primarily by domestic demands and trade and is responsive to changes in GDP, GDP/capita, and recycling (for applicable industries). Production is generally written in terms of mass produced.

.. note:: The IPPU model contains the python method for industrial production, which is accessed in both AFOLU and CircularEconomy. 


Industrial Processes and Product Use
====================================

Industrial processes and product use is used exclusively to account for emissions from the industrial sector **excluding** industrial energy. Examples of emissions in IPPU include HFCs and PFCs emitted from refrigeration, HFCs/PFCs/FCs emitted in electronics manufacturing, :math:`\text{CO}_2` from cement clinker  etc.

All industrial categorization is derived from the *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories*, Volume 3, *Industrial Processes* (see Chapter 1, page 1.7 for a chart describing these categorizations). The entire guidance is available from the `Task Force on National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2019rf/index.html>`_

.. note::Note that emissions from and energy use, as well as energy demands, from industrial activity (excluding fuel and energy production) are accounted for in `Industrial Energy <./energy_consumption.html#industrial-energy-inen>`_. Emissions from fuel production, including electricity, coal mining, and natural gas exploration, extraction, and processing and refinement are accounted for in `Electricity and Fuel Production <./energy_production.html>`_, which is constructed using NEMO. The categories enumerated below, however, are used to estimate those emissions and electrical demands.


Categories
----------

Categories associated with Industrial Processes and Product Use are identified by the ``$CAT-INDUSTRY$`` variable schema element and shown in the category attribute table shown below.


.. csv-table:: Industrial categories (``$CAT-INDUSTRY$`` attribute table)
   :file: ./csvs/attribute_cat_industry.csv
   :header-rows: 1

.. note:: The Electronics Industry is broken into four key subcategories given the high degree of variation in emission factors and emission-relevant gasses governing each of these subcategories and the relative magnitude of the GWP of fluorinated compounds used in the electronics manufacturing process. Other categories--like metals--are not broken into subcategories (e.g., such as steel, aluminum, magnesium, zinc, etc... ) given the relatively similar magnitude of emission factors across gasses and the low prevalence of gasses associated with very-high GWPs. See `V3, C6 IPCC GNGHGI (2019R)<https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/3_Volume3/19R_V3_Ch06_Electronics.pdf>`_ Table 6.6 for default emission factors associated with electronics manufacturing.


Variables
---------

Variables associated with the Industrial Processes and Product Use subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Industrial Processes and Product Use subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_ip_ippu.csv
   :header-rows: 1


----


Known Issues and Future Improvements
====================================
* IPPU models most production in terms of tonnage. SISEPUEDE 2.0 will include higher refinement of industrial emissions modeling using more common units (e.g., :math:`m^2` of substrate in electronics manufacturing).