==========
Strategies
==========

**Strategies** are collections of Transformations.


Defining Strategies
===================

Strategies are defined using a directory structure and two steps:

(1) Defining the universe of Transformations
    Transformations are defined in YAML configuration files in a directory. Each transformation is associated with a Transformer, and different parameters can be passed to the transformer, including things like the magnitude, applicable categories, fractional mixes, and the timing of the transformation. 

(2) Defining strategies
    A strategy definiiton table is used to combine transformations,


Each directory MUST include a **config.yaml** file that configures some general properties as well as the baseline.
