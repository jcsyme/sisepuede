=========================
SISEPUEDE - Model Concept
=========================

**SISEPUEDE**, or **SI**\ mulation of **SE**\ ctoral **P**\ athways and **U**\ ncertainty **E**\ xploration for **DE**\ carbonization, is an integrated emissions modeling framework used to evaluate emissions pathways and transformations through the integration of extensive cross-sectoral dynamics. Fundamentally, SISEPUEDE is a bottom-up emissions accounting model that is based primarily on the IPCC guidance on greenhouse gas inventories (2006/2019R) at Tier 1 or higher. SISEPUEDE accounts for emissions by gas in each of the 4 key IPCC emission sectors–Agriculture, Forestry, and Land Use (**AFOLU**); Waste Management (**Circular Economy**); **Energy**; and Industrial Processes and Product Use (**IPPU**). SISEPUEDE also includes a fifth **Socioeconomic** subsector, which is used to coordinate shared drivers among the emission models. These five sectors are divided into 16 emission and 5 non-emission model sectors. SISEPUEDE integrates these sectors using a directed-acyclic graph (DAG, see figure below) that passes key outputs from one subsector to another. For example, livestock manure management creates opportunities for replacement of synthetic fertilizers with manure in croplands and pastures; increasing recycling modifies industrial production to reduce demands for virgin materials; and fuel switching in energy subsectors leads to changes in fuel production, including for electricity and hydrogen. 


SISEPUEDE Directed Acyclic Graph (DAG)
======================================

SISEPUEDE model integration is best reflect in a directed acylic graph, also known as a DAG.

.. image:: img/sisepuede_dag.jpg


Known Issues and Future Improvements
====================================
* Demand for harvested wood products in AFOLU is based off industrial demands that are not adjusted for recycling. SISEPUEDE 2.0 will fix this.



