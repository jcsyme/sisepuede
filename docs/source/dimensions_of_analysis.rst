======================
Dimensions of Analysis
======================

SISEPUEDE facilitates large-scale computational experiments by organizing data around **Dimensions of Analysis**. These dimensions of analysis--including ``design_id``, ``future_id``, ``strategy_id``, and ``time_series_id`` (under development)-- are based on the XLRM logical framework (see `Lempert, Popper, and Bankes (2003) <https://www.rand.org/pubs/monograph_reports/MR1626.html>`_). The framework organizes robust, scenario-based modeling excercises into the following components:

* **X** Exogenous uncertertainties
* **L** Levers (controls and strategies)
* **R** Relationships (including models)
* **M** Metrics

In SISEPUEDE, *Strategies* (**Ls**) are used to denote different collections of levers that can be applied to a baseline, and strategies are indexed by ``strategy_id``. A strategy can represent any combination of levers, including no levers (baseline, which is required to specify); a single lever; or a collection of levers. *Futures* represent different plausible states of the future--including both exogenous uncertainties (**Xs**) and the ability of levers to acheieve expected effects (**LEs**)-- and are indexed by dimension ``future_id``. Futures are explored over using Latin Hybercube (LHC) samples, which ensure that the uncertainty space is maximally explored given some number of futures (defined at runtime or in configuration files). 

SISPEUEDE furthermore allows users to combine Xs and different characterizations of LEs through different *Experimental Designs*, which are identified using the ``design_id``. All inputs for variables, strategies and futures (sampling ranges) are managed using templates and attribute tables, each of which are discussed below.

.. note 
   The SISEPUEDE framework relies on a collection of tools and classes--including the ``InputTemplate``, ``FutureTrajectories``, ``SamplingUnit``, and ``LHSDesign`` classes--to represent input data, including uncertainties and levers, and modify them to facilitate robust exploratory modeling (see `Entering Data <../entering_data.html>`_ for more information on these classes).


Futures
=======

**Futures** are the


Strategies
==========
**Strategies**, indexed by ``strategy_id``, combine transformations to generate whole-sector and economy-wide transformations. 

.. csv-table:: A number of strategies. Note the numbering scheme; AFOLU-specific transformations occupy 1001-1999; Circular Economy, 2001-2999; Energy, 3001-3999; IPPU, 4001-4999; and cross-sector, 5001-5999.
   :file: ./csvs/attribute_dim_strategy_id.csv
   :header-rows: 1

Using the input data system, the ``SamplingUnits`` is instantiated for each variable or collection of variables (specified as a variable trajectory group) and infers whether or not a variable is an X or an L, then determines the implicit lever effect for each strategy since effects might vary by strategy.

.. note
   The baseline strategy is always entered as ``strategy_id = 0`` in the strategy attribute table.

Designs and Lever Effects
=========================

The **Design** dimension of analyais, which is indexed by ``design_id``, is used to manage computational experiments and control two uncertainties: *exogenous uncertainties*, characterized by **X** in the XLRM matrix, and *lever effect uncertainties*, which represent undertainties in the ability to acheieve lever or strategy specifications. As described above in the Strategy section, the implicit lever effect, or lever delta, is inferred by the ``SamplingUnit`` class (``sampling_units.py``). The ``design_id`` allows the user to sample arund this effect and consider scenarios where strategic goals or expectations are not met or are exceeded. The specification of uncertainty designs are controled in the ``design_id`` attribute table, located at `./csvs/attribute_dim_design_id.csv`.

.. csv-table:: Current specifications of designs in ``attribute_dim_design_id.csv``
   :file: ./csvs/attribute_dim_design_id.csv
   :header-rows: 1

A brief description of input fields is included below. Note that fields that begin with ``linear_transform_ld_`` give parameter values for the linear transformation of LHC samples, which are then applied as scalars to LEs. Mathematically, suppose an LHC sample :math:`x` is such that :math:`x ~ U(0, 1)`. Then the transformation applied to generate scalars for lever effects (LEs) is :math:`d(x) = \max\{\min\{mx + b, a_1\}, a_0\}`.

.. csv-table:: Description of fields in ``attribute_dim_design_id.csv``
   :file: ./csvs/attribute_field_design_id.csv
   :header-rows: 1



 

Primary Key
===========
