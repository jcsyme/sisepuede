===================
Non-Electric Energy
===================

Energy includes a range of variables and categories. Given the integrated nature of the energy sector, there are several "cross-subsector" categories required to construct the NemoMod energy model. These categories include Fuels and Technologies, which . The dimensions required for the NemoMod framework are available from the `NemoMod Categories Documentation <https://sei-international.github.io/NemoMod.jl/stable/dimensions/>`_.


Modeling Concepts and Important Notes
=====================================

Fuels and Heat Energy
---------------------

In general, energy is produced by stationary or mobile combustion of different fuels. The combustion of fuels releases :math:`\text{CO}_2`, :math:`\text{CH}_4`, and :math:`\text{N}_2\text{O}` (and other gasses, which may not be captured). These fuels are utilized by different technologies, which may use fuels at different efficiencies. Energy can also be stored (specifically electricity). The term **fuel** is explicitly used in all energy subsectors, while **technology** and **storage** are used in the NemoMod electricity model.

.. note:: Fuels are used in both non-electric energy and electric energy. However, since they are a required component of the NemoMod electricity model, variables and attributes associated with fuel are available in the `Energy - Electricity <./energy_production.html>`_ section.

The combination of fuel and efficiency is an important concept for entering input data. Energy use in SCOE and CCSQ both use a fraction of energy *demand at point of use* to project future changes in fuel mixtures. However, many empirical data that are used rely on energy *consumption*, and both SCOE and CCSQ take initial consumption as inputs to SISEPUEDE.

Let

* :math:`D_t` be the total energy demand at time :math:`t`
* :math:`C_t` be the total energy consumption at time :math:`t` in question (**input to model**)
* :math:`\alpha^{(C)}_t \in \mathbb{R}^n` be the vector of fuel mix fractions of *consumption* at time :math:`t` for :math:`n` fuels
* :math:`\alpha^{(D)}_t \in \mathbb{R}^n` be the vector of fuel mix fractions of *demand* at time :math:`t` for :math:`n` fuels (**input to model**)
* :math:`e_t \in \mathbb{R}^n` be the vector of fuel-technology average efficiencies at time :math:`t` for :math:`n` fuels, the demand is

:math:`D_t = C_t\left(\alpha^{(C)}_t \cdot e_t\right)`.

The fraction at point-of-use demand :math:`\alpha^{(D)}_{ti}` for fuel :math:`i` is then calculated as

:math:`\alpha^{(D)}_{ti} = \frac{\alpha^{(C)}_{ti}e_{ti}}{\alpha^{(C)}_t \cdot e_t}`,

i.e., the point-of-use demand is the efficiency-weighted fraction of consumption. For more information on the energy models' mathematical specification, see the `Mathematical Documentation of Energy Models <./mathdoc_energy.html>`_.

----


Energy Fuels (ENFU)
===================

Fuel is cross-cutting, affecting all energy sectors (including `Electricity <../energy_production.htm>`_). **EXPAND DESCRIPTION**

Categories
----------

Categories associated with Energy Technology are identified by the ``$CAT-FUEL$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: Fuel categories (``$CAT-FUEL$`` attribute table)
   :file: ./csvs/attribute_cat_fuel.csv
   :header-rows: 1


Variables
---------

Variables associated with the Energy Fuels subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Energy Fuels subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_enfu.csv
   :header-rows: 1


----


Fugitive Emissions (FGTV)
=========================

Fugitive emissions includes emission from coal, natural gas, and oil production, transmission, and distribution.


Categories
----------

Fugitive emissions relies on the `Energy Fuels <./energy_consumption.html#energy-fuels-enfu>`_ category (``$CAT-FUEL$``) as the primary input category, but are limited to categories associated with coal, oil, and natural gas production--i.e., ``fuel_coal``, ``fuel_natural_gas``, and ``fuel_oil``. 


Variables
---------

Variables associated with the Fugitive Emissions subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Fugitive Emissions subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_fgtv.csv
   :header-rows: 1


----


Industrial Energy (INEN)
========================

Industrial energy includes emission from **DESCRIPTION**

Categories
----------

Industrial Energy uses indstrial categories (variable scehma element ``$CAT-INDUSTRY$``), defined in `Industial Processes and Product Use (IPPU) <../ippu.html>`_, as the primary input category space.


Variables
---------

Variables associated with the Industrial Energy subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Industrial Energy subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_inen.csv
   :header-rows: 1


----


Stationary Combustion and Other Energy (SCOE)
=============================================

SCOE (**S**ationary **C**ombustion and **O**ther **E**nergy) captures stationary emissions, primarily from combustion occuring in buildings (split out by differing drivers) and allows for the specification of other fuel combustion emissions not captured elsewhere.

.. note:: | Energy efficiency factor represents the technological efficiency for the system of heat energy delivery. Some system/fuels may conserve energy more efficiently than others.
          |
          | For example, a value of 0.8 would indicate that 20% (1 - 0.8) of the input energy to the system (e.g., for heating, cooking, water heaters, etc.) is lost (e.g., 1.25 TJ of input energy satisfies 1 TJ of end-use demand), while a value of 1 would indicate perfect efficiency (1 TJ in :math:`\implies` 1 TJ out)
          |
          | At time :math:`t = 0`, the efficiencies are used to calculate an end-user demand for energy, which elasticities are applied to to estimate a point-of-use demand. In subsequent time steps, as the mix of energy use changes, input energy demands are calculated using the efficiency factors of different mixes of fuels.


Categories
----------

Categories associated with Stationary Combustion and Other Energy are identified by the ``$CAT-SCOE$`` variable schema element and shown in the category attribute table shown below.

.. csv-table:: Other categories (``$CAT-SCOE$`` attribute table)
   :file: ./csvs/attribute_cat_scoe.csv
   :header-rows: 1


Variables
---------

Variables associated with the Stationary Combustion and Other Energy subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Stationary Combustion and Other Energy subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_scoe.csv
   :header-rows: 1


----


Transportation (TRNS)
=====================

Transportation consists of different categories (or modes) of transportation that are used to satisfy different types of demand. In general

* Socioeconomic: 
* Transportation Demand: 
* Energy Production: 


Known Issues
------------

**Discuss how variables that are set in Transportation have to be added to the EnergyConsumption class as well**


Categories
----------

Categories associated with Transportation are identified by the ``$CAT-TRANSPORTATION$`` variable schema element and shown in the category attribute table shown below. 

.. note:: These categories are associated with different `Transportation Demand <./energy_consumption.html#transportation-demand-trde>`_ categories, which govern mode-shifting.

.. csv-table:: Other categories (``$CAT-TRANSPORTATION$`` attribute table)
   :file: ./csvs/attribute_cat_transportation.csv
   :header-rows: 1


Variables
---------

Variables associated with the Transportation subsector are shown below. 

.. note::
   :math:`\text{CH}_4` and :math:`\text{N}_4\text{O}` emissions from mobile combustion of fuels are highly dependent on the technologies (e.g., types of cars) that use the fuels. Therefore, emission factors for mobile combustion of fuels are contained in the Transportation subsector instead of the Energy Fuels subsector. See Section Volume 2, Chapter 3, Section 3.2.1.2 of the `2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_3_Ch3_Mobile_Combustion.pdf>`_ for more information.


.. csv-table:: Trajectories of the following variables are needed for the Transportation subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_trns.csv
   :header-rows: 1


----


Transportation Demand (TRDE)
============================

Transportation demand is broken into its own subsector given some of the complexities that drive transportation demand (unlike other subsectors, like SCOE, that do not contain categorical mode-shifting within demands). The SISEPUEDE transportation demand subsector allows for more complex interactions--e.g., interactions with industrial production, growth in tourism, waste collection, and imports and exports--to be integrated, though these are not dealt with explicitly at this time.


Categories
----------

Categories associated with Transportation are identified by the ``$CAT-TRANSPORTATION-DEMAND$`` variable schema element and shown in the category attribute table shown below. These categories are associated with different allowable mode shifts between vehicle types.

.. csv-table:: Transportation Demand categories (``$CAT-TRANSPORTATION-DEMAND$`` attribute table)
   :file: ./csvs/attribute_cat_transportation_demand.csv
   :header-rows: 1


Variables
---------

Variables associated with the Transportation Demand subsector are shown below. 

.. csv-table:: Trajectories of the following variables are needed for the Transportation Demand subsector. The categories that variables apply to are described in the ``category`` column.
   :file: ./csvs/variable_definitions_en_trde.csv
   :header-rows: 1


