==========
Strategies
==========

**Strategies** are collections of Transformations.


Defining Strategies
===================

Strategies are defined using a directory structure and require the specification of three types of files:

(1) General configuration and baseline:
    Each directory MUST include a **config.yaml** file that configures some general properties as well as the baseline.

(2) Transformations:
    Transformations are defined in YAML configuration files in a directory. Each transformation is associated with a Transformer, and different parameters can be passed to the transformer, including things like the magnitude, applicable categories, fractional mixes, and the timing of the transformation. 

(3) Strategy definitions:
    A strategy definiiton table is used to combine transformations,


Generate a Default Set of Transformations
-----------------------------------------
The fastest way to get started is to use SISEPUEDE to generate a default set of transformations based on Transformer default.

