==================================================
Mathematical Documentation - Economic Impact Model
==================================================

Data
====

* IO Tables

The `Global Input-Output Tables <https://www.cepal.org/en/events/global-input-output-tables-tools-analysis-integration-latin-america-world>`_ is an effort to complete, among other,the multiregional input-output (MRIO) tables in Latin American and Asia Pacific. The institutions that
contributed to this project were: the Economic Commission for Latin America and the Caribbean (ECLAC), the Economic and Social Commission for Asia and the Pacific (ESCAP) and the Asian Development Bank (ADB).

Specifically we use the I-O table for 2011 that includes 25 sectors and 78 economies plus the Rest of the World. This table contains
disaggregated data for 18 economies of Latin America and is consistent with the new I-O table by the Asian Development Bank, which considers 38 economic sectors.

The data of 18 Latin America economies is `here <https://public.tableau.com/app/profile/edmundo.molina/viz/I-O_Viz/I-OTablesLAC>`_ . Availability of IO models by country can be found in the `$CAT-REGION$` attribute table in `General Data <./general_data.html#regions-countries>`_.



* Tables 1, 2 and 3 of `Garrett-Peltier, H. (2017) <https://www.sciencedirect.com/science/article/abs/pii/S026499931630709X>`_

.. table:: Weights in `Garrett-Peltier 2011 <https://www.amazon.com/Creating-Clean-Energy-Economy-Investments-Sustainable/dp/3844306455>`_ .

    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |    | I-O Industry (from 71-industry table)                         | Weatherization   | Mass Transit & Freight Rail   | Smart Grid   | Wind   | Solar   | Biomass   |   Oil & Gas |   Coal |
    +====+===============================================================+==================+===============================+==============+========+=========+===========+=============+========+
    |  1 | Farm products (unprocessed)                                   | –                | –                             | –            | –      | –       | 0.250     |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  2 | Forestry, fishing and related                                 | –                | –                             | –            | –      | –       | 0.250     |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  3 | Oil and Gas extraction                                        |                  |                               |              |        |         |           |        0.3  |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  4 | Coal Mining                                                   |                  |                               |              |        |         |           |             |   0.44 |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  5 | Support activities for extraction and mining                  |                  |                               |              |        |         |           |        0.04 |   0.08 |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  6 | Natural gas distribution                                      |                  |                               |              |        |         |           |        0.1  |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  7 | Construction                                                  | 1.000            | 0.450                         | 0.250        | 0.260  | 0.300   | 0.250     |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  8 | Petroleum and Coal Products                                   |                  |                               |              |        |         |           |        0.53 |   0.48 |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    |  9 | Chemical products                                             | –                | –                             | –            | –      | –       | 0.125     |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 10 | Plastics and rubber products                                  | –                | –                             | –            | 0.120  | –       | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 11 | Fabricated metal products                                     | –                | –                             | –            | 0.120  | 0.175   | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 12 | Machinery                                                     | –                | –                             | 0.250        | 0.370  | –       | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 13 | Computer and electronic products                              | –                | –                             | 0.250        | 0.030  | 0.175   | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 14 | Electrical equipment, appliances, and components              | –                | –                             | 0.250        | 0.030  | 0.175   | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 15 | Rail transportation                                           | –                | 0.100                         | –            | –      | –       | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 16 | Transit and ground passenger transportation                   | –                | 0.450                         | –            | –      | –       | –         |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 17 | Pipeline transportation                                       |                  |                               |              |        |         |           |        0.03 |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 18 | Miscellaneous professional, scientific and technical services | –                | –                             | –            | 0.070  | 0.175   | 0.125     |             |        |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+
    | 19 | sum of weights                                                | 1.000            | 1.000                         | 1.000        | 1.000  | 1.000   | 1.000     |        1    |   1    |
    +----+---------------------------------------------------------------+------------------+-------------------------------+--------------+--------+---------+-----------+-------------+--------+

.. table:: Weights in `Pollin et al. 2015 <https://peri.umass.edu/publication/item/689-global-green-growth-clean-energy-industrial-investments-and-expanding-job-opportunities>`_ .

    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |    |                                                  |   Wind |   Solar |   Bioenergy |   Geothermal |   Hydro (small) |   Weatherization |   Industrial EE |   Smart Grid |   Oil and Gas |   Coal |
    +====+==================================================+========+=========+=============+==============+=================+==================+=================+==============+===============+========+
    |  1 | Farms                                            |        |         |       0.25  |              |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  2 | Forestry, fishing, and related activities        |        |         |       0.25  |              |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  3 | Oil and gas extraction                           |        |         |             |              |                 |                  |                 |              |          0.5  |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  4 | Mining, except oil and gas                       |        |         |             |              |                 |                  |                 |              |               |    0.5 |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  5 | Support activities for mining                    |        |         |             |         0.15 |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  6 | Construction                                     |   0.26 |   0.3   |       0.25  |         0.45 |            0.18 |                1 |             0.2 |         0.25 |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  7 | Fabricated metal products                        |   0.12 |   0.175 |             |              |            0.18 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  8 | Machinery                                        |   0.37 |   0.175 |             |         0.1  |            0.07 |                  |             0.5 |         0.25 |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    |  9 | Computer and electronic products                 |   0.03 |   0.175 |             |              |                 |                  |                 |         0.25 |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 10 | Electrical equipment, appliances, and components |   0.03 |         |             |              |            0.14 |                  |                 |         0.25 |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 11 | Petroleum and coal products                      |        |         |             |              |                 |                  |                 |              |          0.25 |    0.5 |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 12 | Chemical products                                |        |         |       0.125 |              |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 13 | Plastics and rubber products                     |   0.12 |         |             |              |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 14 | Pipeline transportation                          |        |         |             |              |                 |                  |                 |              |          0.25 |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 15 | Miscellaneous professional, scientific, and      |   0.07 |   0.175 |       0.125 |         0.3  |            0.43 |                  |             0.3 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 16 | technical services                               |        |         |             |              |                 |                  |                 |              |               |        |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+
    | 17 | Sum of weights                                   |   1    |   1     |       1     |         1    |            1    |                1 |             1   |         1    |          1    |    1   |
    +----+--------------------------------------------------+--------+---------+-------------+--------------+-----------------+------------------+-----------------+--------------+---------------+--------+

.. table:: Composition of RE industries using alternative cost structures.

    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |    |                                    | Wind                       | Wind                 | Wind (onshore)      | Solar PV (central)   | Solar                | Solar         | Geothermal   |
    +====+====================================+============================+======================+=====================+======================+======================+===============+==============+
    |  0 |                                    | Tegen et at. (2013) [#f1]_ | IRENA (2012b) [#f2]_ | B & V (2012) [#f3]_ | B & V (2012)         | IRENA (2012a) [#f4]_ | BNEF-SEA 2013 | B & V (2012) |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  1 | Support activities for mining      |                            |                      |                     |                      |                      |               | 0.39         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  2 | Construction                       | 0.200                      | 0.276                | 0.255               | 0.095                | 0.125                | 0.290         | 0.25         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  3 | Nonmetallic mineral products       | 0.030                      | 0.160                |                     | 0.120                | 0.050                |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  4 | Fabricated metal products          | 0.160                      | 0.160                | 0.340               | 0.410                | 0.210                | 0.200         | 0.14         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  5 | Machinery                          | 0.370                      |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  6 | Computer and electronic products   |                            |                      |                     |                      | 0.385                |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  7 | Electrical equipment, appliances,  | 0.150                      | 0.314                | 0.340               | 0.330                | 0.122                | 0.250         | 0.08         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  8 | and components                     |                            |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    |  9 | Truck transportation               | 0.030                      |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 10 | Insurance carriers and related     | 0.030                      |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 11 | activities                         |                            |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 12 | Miscellaneous professional,        | 0.020                      | 0.090                | 0.040               | 0.020                | 0.109                | 0.210         | 0.07         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 13 | scientific, and technical services |                            |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 14 | Management of companies and        | 0.010                      |                      | 0.025               | 0.025                |                      | 0.050         | 0.07         |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 15 | enterprises                        |                            |                      |                     |                      |                      |               |              |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
    | 16 | Sum of weights                     | 1.000                      | 1.000                | 1.000               | 1.000                | 1.000                | 1.000         | 1.0          |
    +----+------------------------------------+----------------------------+----------------------+---------------------+----------------------+----------------------+---------------+--------------+
.. rubric:: Footnotes

.. [#f1] `Tegen et at. (2013) <https://www.osti.gov/biblio/1072784>`_ .
.. [#f2] `IRENA (2012b) <https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2013/Renewable_Power_Generation_Costs_in_2012_summary.pdf?la=en&hash=548B1D4A7BEAF616A19B26D8DF07011A8B8F49E7;>`_ .
.. [#f3] `B & V (2012) <https://refman.energytransitionmodel.com/publications/1921>`_ .
.. [#f4] `IRENA (2012a) <https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2013/Renewable_Power_Generation_Costs_in_2012_summary.pdf?la=en&hash=548B1D4A7BEAF616A19B26D8DF07011A8B8F49E7;>`_ .



* Investment tables (Investment shock per sector and per country, per scenario)

I-O Impact Assessment Model
===========================

New Industry: The Final-Demand Approach (Miller and Blair, 2009, cap 13)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the sake of simplicity, consider an example economy of two sectors

.. math::

   A=\begin{bmatrix}
      a_{11} & a_{12}\\
      a_{21} & a_{22}
   \end{bmatrix}

We assume that a new industry is introduced to the economy (sector 3, as a result of decarbonization). The model is built on the premise that it is possible to estimate the inputs of sectors 1 and 2 and the value of production of the new sector 3; this is, :math:`a_{13}` y :math:`a_{23}`.

In order to quantity the impact of the entry of sector 3 in the economy, we have to use some of measure of the *magnitude* of the new economic activities associated with sector 3.

Thus for the I-O model, we specify:

* The level of production of sector 3, :math:`x_3`, o
* The final demand, :math:`f_3`

For this example, then the level of production of sector can be denoted as :math:`\bar{x}_3`.

The new demand that results for sectors 1 and 2 from the new production of sector 3 is then :math:`a_{13}\bar{x}_3`  and :math:`a_{23}\bar{x}_3` , respectively.

This means, that these new demands can be model as a **exogenous** shock to the two original sectors;

.. math::
  \Delta \mathbf{f}= \begin{bmatrix}
  a_{13}\bar{x}_3 \\
  a_{23}\bar{x}_3
  \end{bmatrix}

Thus the impacts, in terms of the production of these two sectors, are given by :math:`\Delta\mathbf{x} = \mathbf{L}\Delta \mathbf{f}`:

.. math::
  \Delta\mathbf{x}=\begin{bmatrix}
  l_{11} & l_{12}\\
  l_{21} & l_{22}
  \end{bmatrix}
  \begin{bmatrix}
  a_{13}\bar{x}_3 \\
  a_{23}\bar{x}_3
  \end{bmatrix}
  = \begin{bmatrix}
  l_{11}a_{13}\bar{x}_3 + l_{12}a_{23}\bar{x}_3 \\
  l_{21}a_{13}\bar{x}_3 + l_{22}a_{23}\bar{x}_3 \\
  \end{bmatrix}


Since there is also a baseline demand, independent of the new demand associated with sector 3, :math:`\bar{f}_1` and :math:`\bar{f}_2`, for these two sectors
the gross total production is given by:

.. math::
  \begin{bmatrix}
  x_1 \\
  x_2
  \end{bmatrix} =\begin{bmatrix}
  l_{11} & l_{12}\\
  l_{21} & l_{22}
  \end{bmatrix}
  \begin{bmatrix}
  \bar{f}_1 + a_{13}\bar{x}_3 \\
  \bar{f}_2 + a_{23}\bar{x}_3
  \end{bmatrix}
  = \begin{bmatrix}
  l_{11}(\bar{f}_1 + a_{13}\bar{x}_3) + l_{12}(\bar{f}_2 + a_{23}\bar{x}_3) \\
  l_{21}(\bar{f}_1+a_{13}\bar{x}_3) + l_{22}(\bar{f}_2 + a_{23}\bar{x}_3) \\
  \end{bmatrix}


when :math:`\bar{f}_1 = 0` y :math:`\bar{f}_2=0`, we can isolate the impact of incorporating the new sector

This logic is operationalized in the following example
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Be

.. math::
  \mathbf{A}=\begin{bmatrix}
      0.15 & 0.25\\
      0.20& 0.05
  \end{bmatrix},

Then :math:`(\mathbf{I} - \mathbf{A})^{-1}` is equal to:

.. math::
  \mathbf{A}=\begin{bmatrix}
      1.25412541 & 0.330033\\
      0.2640264  & 1.12211221
  \end{bmatrix}

::

  import numpy as np

  A= np.array([[0.15,0.25],[0.20,0.05]])
  L = np.linalg.inv(np.identity(2)-A)
  L
  >> array([[1.25412541, 0.330033  ],[0.2640264 , 1.12211221]])

We assume that input demand for sector 3 is given by:

* :math:`a_{13}=0.30`
* :math:`a_{23} = 0.18`

and in this example we estimate that sector 3 will produce at a level of 100,000 units per year.

Such that :math:`\bar{x}_3 = 100000`

.. math::
  \Delta \mathbf{f}= \begin{bmatrix}
  0.30 \times 100000  \\
  0.18 \times 100000
  \end{bmatrix}
  =
   \begin{bmatrix}
  30000  \\
  18000
  \end{bmatrix}


Then the impact of incorporating the new sector is equal to:

.. math::
   \Delta \mathbf{x} = \begin{bmatrix}
  43564  \\
  28118
  \end{bmatrix}

::

  x_bar_3 = 100000
  delta_f = np.array([x_bar_3 * 0.30 ,x_bar_3 * 0.18])
  L@delta_f
  >> array([43564.35643564, 28118.81188119])

Sector 1 would have to meet the new demand level of 30,000,
thus its production level would need to increase to 43,560. In a similar way, the new
demand level for sector 2, derived from the introduction of sector 3, would be 18,000, but in the end sector 2
will need to produce 28,116 units more. These impact analysis simulated the effect that the introduction of
a new industry can have on an economy.

Impact on employment
"""""""""""""""""""""""

Let be :math:`\mathscr{E}` total employment and :math:`E = [e_1,e_2,\dots,e_n]` a row vector
containing the job coefficients of each sector, then total employment will be estimated as:

.. math::
  \begin{equation}
  \mathscr{E} = EX
  \end{equation}

Considering a two sectors economy, the **impact** (direct plus indirect change) on employment will be given by the exogenous change in final demand for sector 2 is

.. math::
  \Delta\mathscr{E}_{d} =
  \begin{bmatrix}
      e_1 & e_2\\
  \end{bmatrix}
  \begin{bmatrix}
      l_{11} & l_{12}\\
      l_{12} & l_{22}
  \end{bmatrix}
  \begin{bmatrix}
      \Delta f_{1} \\
      \Delta f_{2}
  \end{bmatrix}
  =
  E (\mathbf{I} - \mathbf{A})^{-1} \Delta \mathbf{f}
  = E \Delta X

The total direct employment change resulting from the demand change is :math:`\Delta\mathscr{E}_{d'}`

.. math::
  \Delta\mathscr{E}_{d'} =
  \begin{bmatrix}
      e_1 & e_2\\
  \end{bmatrix}
  \begin{bmatrix}
      \Delta f_{1} \\
      \Delta f_{2}
  \end{bmatrix}
  =
  E\Delta \mathbf{f}



Example 2
"""""""""""""

Following the previous example, the estimated impact on employment, derived from the introduction of sector 3 is estimated as follows.

The employment coefficients are given by:

.. math::
  E = \begin{bmatrix}
      0.25 & 0.15\\
  \end{bmatrix}

Then the change in employment to meet the final demand:

.. math::
  \Delta\mathscr{E}_{d} =
  E \Delta X
  =
  E (\mathbf{I} - \mathbf{A})^{-1} \Delta \mathbf{f}

.. math::
  \Delta\mathscr{E}_{d} =
  \begin{bmatrix}
  0.25 & 0.15
  \end{bmatrix}
  \begin{bmatrix}
  43564  \\
  28118
  \end{bmatrix}
  =
  15108

::

  E = np.array([0.25,0.15])
  E.dot(L@delta_f)
  >> 15108.910891089108

To estimate the percent change in employment, we need to estimate  :math:`X`. Asumming that :math:`\bar{f}_1 = 120000` and :math:`\bar{f}_2=90000`.
Recordando la expresión para calcular :math:`X`;

.. math::
  \begin{bmatrix}
  x_1 \\
  x_2
  \end{bmatrix} =\begin{bmatrix}
  l_{11} & l_{12}\\
  l_{21} & l_{22}
  \end{bmatrix}
  \begin{bmatrix}
  \bar{f}_1  \\
  \bar{f}_2
  \end{bmatrix}

::

  x_bar_3 = 100000
  f1_usual = 120000
  f2_usual = 90000
  X = np.array([f1_usual ,f2_usual])
  (E.dot(L@delta_f)/sum(L@X))*100
  >> 4.829113924050633
