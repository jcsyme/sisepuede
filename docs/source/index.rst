=======================
SISEPUEDE Documentation
=======================

SISEPUEDE (**SI**\ mulating **SE**\ ctoral **P**\ athways and **U**\ ncertainty **E**\ xploration for **DE**\ carbonization) is an integrated Python/Julia modeling framework that facilitates exploratory analyses of decarbonization transformations within emissions sectors at the region level. It includes several key components:

- Integrated yet separable sectoral models of emissions based on IPCC guidelines for greenhouse gas inventories
- Uncertainty specification and trajectory sampling mechanism
- A data pipeline management system
- Scalable architecture
- Customizable variable setup through sector-level categorization


Check out the `General Data <../general_data.htm>`_ section to get started.


About the Model
===============

SISEPUEDE is a compartmentalized, sector-based model of emissions based primarily on two key publications from the IPCC:

#. `2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/index.html>`_ and

#.  `2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/index.html>`_

These two documents are often abbreviated as **V##, C## IPCC GNGHGI** in attribute tables. In this notation, **V##** gives the volume number, while **C##** gives the chapter number. For example, V5, C6 refers to Volume 5, Chapter 6 (*Wastewaster Treatment and Discharge*).

**EXPAND TO DESCRIBE ABSTRACT STRUCTURE**


SISEPUEDE and Documentation Terminology 
---------------------------------------


Subsectors
^^^^^^^^^^
- What are subsectors?

Categories
^^^^^^^^^^
- what are categories?

Variables and Fields
^^^^^^^^^^^^^^^^^^^^
The SISEPUEDE integrated modeling framework makes use of a generalizable variable schematic to define input variables for models. There are two components to this naming system:

#. **Model Variables** These are conceptual variables--for example, `Crop Yield Factor`--that are used to group

#. **Variable Fields** These are direct inputs to the SISEPUEDE models, entered as fields in a data frame. For example, the input variables associated with `Crop Yield Factor` include...

- variables are abstract groupings of variables for a defined category
   - some variables represent no categories
   - some represent all
   - some represent only a few
- the model fundamentally reads in data frames with fields; those fields are defined by the variable construct
- reading the variable definition tables
   - Variable Name
   - Variable Schema
   - Categories
   - Simplex Group (probability simplex)
.. note::
   SIMPLEX NOTE EXAMPLE Note that the sum of all initial fractions of area across land use categories *u* should be should equal 1 to , i.e. :math:`\sum_u \varphi_u = 1`, where :math:`\varphi_{\text{$CAT-LANDUSE$}} \to` ``frac_lu_$CAT-LANDUSE$`` at period *t*.

   - Default value
   - Other attributes



Metavariables and Constructing Input Parameters
-----------------------------------------------


-


 



Contents
--------
.. toc example struct from https://github.com/readthedocs/sphinx_rtd_theme/blob/c9b1bde560d8ee31400e4e4f92f2e8d7a42265ce/docs/index.rst
.. https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html

.. toctree::
   :caption: Getting Started
   :hidden:

   installation
   quick_start
   sisepuede_concept
   analytical_parameters

.. toctree::
   :caption: Variables, Categories, and Data
   :hidden:

   general_data
   afolu
   circular_economy
   energy_electric
   energy_non_electric
   ippu
   socioeconomic

.. toctree::
   :caption: Managing Experiments
   :hidden:

   dimensions_of_analysis
   entering_data
   running_models
   sisepuede_database

.. toctree::
   :caption: Mathematical Specifications
   :hidden:

   mathdoc_afolu
   mathdoc_circular_economy
   mathdoc_economic_impact
   mathdoc_energy
   mathdoc_ippu

.. toctree::
   :caption: Community
   :hidden:

   contribute
