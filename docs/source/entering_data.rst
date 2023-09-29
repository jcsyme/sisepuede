=============
Entering Data
=============


Interacting with SISEPUEDE
==========================


Attribute Tables
================

Attribute tables are fundamental components of the data framework underlying SISEPUEDE. Attribute tables define sectors and categories associated with those sectors (those starting with ``cat_``); model variables and associated attributes, including variable schema (which include information on units--those attribute tables starting with 


Design
Region
Strategy
Time Period

See `Dimensions of Analysis <../dimensions_of_analysis.html>`_ for more information on the dimensions of analysis and what they control.

The SISEPUEDE framework relies on a collection of tools and classes--including the ``InputTemplate``, ``FutureTrajectories``, ``SamplingUnit``, and ``LHSDesign`` classes--to represent input data, including uncertainties and levers, and modify them to facilitate robust exploratory modeling (see `Entering Data <../entering_data.html>`_ for more information on these classes)


Input Templates
===============

- Each region has a template for each sector
- five templates
- 3 different types: calibrated, uncalibrated, demo


Sampling Units
--------------

SamplingUnits represent variable or collection of variables that are perturbed by a single Latin Hybercube (LHC) sample. There are a number of ways to define sampling units in the input templates.

... describe approaches here


Defining Futures
----------------

Trajectory Groups
-----------------


Defining Strategies
-------------------






Other Input Files
=================

NemoMod Reference Files
-----------------------
