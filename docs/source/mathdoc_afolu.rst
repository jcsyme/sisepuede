==================================
Mathematical Documentation - AFOLU
==================================

Expand here

Integrated Land Use Model and Assumptions
=========================================

SISEPUEDE does not calculate a general equilibrium for land use demand. Instead, land use is specified as a policy that is adjusted based on demand for livestock (which modifies the area of grassland available) and crops. A rule-based approach is to reallocate land at each time step in response to changing demands. This ruleset is described below.

.. note::
   Non-grazing livestock--including pigs and chickens--are assumed to be produced endogenously independently of land use.

#. **Estimate the livestock carrying capacity of land at each time step for grazing livestock**. First, a carrying capacity :math:`\chi_v(t)` of livestock/hectare is calculated for grazing livestock. The capacity at time *t* is estimated in a few steps.

   * The initial carrying capacity implied by the historical period :math:`\chi_v(0)` is estimated as :math:`\chi_v(0) = L_v(0)G_v(0)^{-1}`, where:

      * :math:`L_v(0)` is the initial population of livestock cateogry *v* at time 0, given by the **Initial Livestock Head Count** variable.

      * :math:`G_v(0)` is the area of grassland (or grazing land/pastures) allocated to each livestock of type *v* and is fixed over time. This is estimated using the **Daily Dry Matter Consumption** of each animal at time :math:`t = 0`, :math:`F_v(0)`; the population of livestock at :math:`t = 0`, :math:`L_v(0)` (and the vector :math:`L(0)`); and the initial area of grassland, :math:`x_g(0)`, where *g* is the index for grassland in the land use initial vector and :math:`x_g(0) = Ap_g(0)` , where :math:`p_j(0)` is the **Initial Land Use Area Proportion** of land use type *j* and *A* is specified as the **Area of Region**.

      * The estimated carrying capacity of the land :math:`G_v(0) = G(0)\frac{L_v(0)F_v(0)}{L(0)\cdot F(0)}`.

   * Note that the calculation of :math:`\chi_v(0)` simplifies to :math:`\chi_v(0) = \frac{L(0)\cdot F(0)}{G(0)F_v(0)}`.

   * The livestock carrying capacity at time *t* is then :math:`\chi_v(t) = c(t)\chi_v(0)`, where :math:`c(t)` is the **Carrying Capacity Scalar**, which is to represent changes to pasture management that enable increases in the number of livestock that can graze per acre--including activities like shared grazing.

#. **Estimate preliminary land use areas** Land use is estimated using a discrete-time ergodic Markov Chain.

   * Let :math:`x(t) \in \mathbb{R}^m` be the vector of land use by type at time *t* (where there are *m* categories of land use).

   * Then let :math:`\tilde{Q}(t) \in \mathbb{R}^{m \times m}` be the exogenous (specified as a policy), unadjusted row-stochastic land use transition matrix from time :math:`t \to t + 1`, so that that :math:`\tilde{Q}_{ij}(t)` gives the transition probability of land use category :math:`i` to land use category :math:`j`.

   * Without adjustments, the area of each land use type at time *t* is :math:`\tilde{x}(t + 1) = x(t)^{T}\tilde{Q}_{ij}(t)`.

#. **Estimate livestock demand and unadjusted production** Livestock demand at time *t* :math:`D_v^{(lvst)}(t)` is represented as a function of population and GDP/capita, with the assumption that demand and supply are in equilibrium at :math:`t = 0` (i.e., :math:`D_v^{(lvst)}(0) = L_v(0)`).

   * Let :math:`M(t)` be the GDP/capita at time *t* (GDP and population are both exogenously defined, where the population is the sum of the rural and urban population), and let :math:`\Delta M(t) = \frac{M(t + 1)}{M(t)} - 1` be the growth rate of GDP/capita.

   * Let :math:`\lambda_v` be the demand elasticity for livestock category *v* to changes in GDP/capita (we use income elasticities as a proxy).

   * Let :math:`P(t)` be the total population at time *t*, and let the per-capita demand for livestock at :math:`t = 0` be :math:`\hat{D}_v^{(lvst)}(0) = P(0)/D_v^{(lvst)}(0)`.

   * The demand for livestock at time :math:`t > 0` is :math:`D_v^{(lvst)}(t) = P(t)\hat{D}_v^{(lvst)}(t)`, where the per-capita demand :math:`\hat{D}_v^{(lvst)}(t + 1)` is calculated recursively as :math:`\hat{D}_v^{(lvst)}(t + 1) = \hat{D}_v^{(lvst)}(t)\left[1 + \lambda \Delta M(t)\right]`.

   * Let *g* be the index (:math:`1 \leq g \leq m`) of grassland in the land use vector *x*. Then the unadjusted area of grassland available for each livestock of type *v* to graze on is :math:`\tilde{G}_v(t) = \frac{G_v(0)}{x_g(0)}\tilde{x}_g(t)`.

   * The policy-specified unadjusted production capacity of each livestock type *v* is :math:`\tilde{P}_v^{(lvst)}(t) = \tilde{G}_v(t)\chi(t)`

#. **Adjust grassland area to reflect demand changes and livestock import factor** Under uncertain futures, land use area and demand for livestock may conflict. This step resolves the conflict by reallocating land use, using an uncertain parameter, to either (a) completely meet demand (at one extreme, where :math:`\eta = 1`) or (b) produce only what the land allows and adjust net imports (where :math:`\eta = 0`).

   * The net surplus demand for livestock of type *v* at time *t* is :math:`S_v^{(lvst)}(t) = D_v^{(lvst)}(t) - \tilde{P}_v^{(lvst)}(t)`

   * In the SISEPUEDE land use model, some of this surplus demand can be met endogenously (by adjusting the land use transition), while some can be met from net imports. The quantity used to adjust the land-use transition is found as :math:`\eta S_v^{(lvst)}`, where :math:`0 \leq \eta \leq 1` is the **Land Use Yield Reallocation Factor**.
      * If :math:`S_v^{(lvst)}(t) < 0`, the area of grassland will be reapportioned back to cropland, where a second adjustment occurs.

#. **Estimate cropland demand and unadjusted production**


#. **Adjust cropland area to reflect demand changes and crop import factor**

#. **Calculate adjusted transitions and emissions from conversion**

.. note::
   SISEPEUDE accounts for increases in crop demand for livestock feed, but changes to diet are not reflected in crop mix.


Adjusting the Land Use Transition Matrix
========================================

**Description of adjustment process here**
