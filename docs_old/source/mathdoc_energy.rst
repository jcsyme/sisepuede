===================================
Mathematical Documentation - Energy
===================================

Energy math here



Demand vs. Consumption
======================


Let

- :math:`L` be the set of fuels (and fuel-technologies, which are assumed to be 1:1 to fuels) with index :math:`i` (:math:`i \in L`). Let :math:`n_L = |L|` be the number of fuels.
- :math:`T` be the set of time periods :math:`T = \{0, 1, \ldots, n_T - 1\}`, where :math:`n_T` is the number of time periods. :math:`T` is indexed to start at 0 for conceptual consistency (time :math:`t = 0` is the `baseline` or `initial state`).


Model Inputs for Estimating Energy Demands
------------------------------------------

Energy demands in SISEPUEDE are estimated as a function of several components that are entered by the user.

#. :math:`\alpha^{(D)}_{it}`: the fraction of **demand** that is met by a certain fuel at time :math:`t` (**model input**). The fractions :math:`\alpha^{(D)}_{it}` are also written as a vector :math:`\alpha^{(D)}_t = \left(\alpha^{(D)}_{1t}, \ldots, \alpha^{(D)}_{nt}\right)`. Of course, the fractions should sum to unity, i.e.,

.. math::
   :label: frac_demand_unity

   \sum_{i \in L}\alpha^{(D)}_{it} = 1\,\forall\, t \in T

#. :math:`I_0`: the initial consumption intensity of energy, measured in units of energy; :math:`I_t` is the consumption intensity at time :math:`t`. Current SISEPUEDE units are TJ/tonne for production driven-categories and TJ/billion GDP (GDP only applies to the `other_product_manufacturing` category right now)

#. :math:`e_{it}`: the average fuel efficiency of fuel-technology :math:`i` at time :math:`t`, dimensionless. This factor represents the proportion of input energy that is output by the fuel-technology (e.g., coal-burning ovens might provide 40\ , or 0.4, of the input energy as output energy). The factors :math:`e_{it}` are also written as a vector :math:`e_t = \left(e_{1t}, \ldots, e_{nt}\right)`.

#. :math:`R_t`: the driver of demand, which can be production or GDP (INEN, SCOE), passenger- or megatonne- kilometer traveled (TRNS), population (SCOE), and/or sequestration (CCSQ). For most industries, :math:`R_t` is production at time :math:`t` (in tons). However, in \texttt{other\_product\_manufacturing}, :math:`R_t` is GDP in \` billion. Production values are calculated in IPPU/Circular Economy, while GDP is exogenously defined.

#. :math:`\sigma_t`: the demand scalar at time :math:`t`, which is applied to an industry's energy demand (not consumption). If :math:`\sigma_t = 1`, demands in SISEPUEDE are estimated according to drivers (production or GDP), energy intensity, and fuel mix, and the average fuel-technology efficiency factor. The demand projection is scaled by :math:`\sigma`.  Note that SISEPUEDE always corrects the sequence :math:`\{\sigma_t\}_{t \in T} \to \left\{\frac{\sigma_t}{\sigma_0}\right\}_{t \in T}` to ensure that :math:`\sigma_0 = 1` (not that this leaves the sequence unchanged if :math:`\sigma_0 = 1`). See equation :math:numref:`demand_projection` for a description of the demand calculation.

.. note::
   The value of :math:`\alpha^{(D)}_{it}` shown in equation :math:numref:`frac_demand_unity` contrasts with the fraction of consumption that is met by a certain fuel; the relationship between these two is defined below in equation :math:numref:`frac_demand_frac_consumption_relationship`. **Equation :math:numref:`frac_demand_frac_consumption_relationship` should be used to translate empirical observations of initial consumption into initial demands based on assumptions about efficiency.** Demands are entered in place of consumption to preserve logical consistency of transformations; in general, fuel mix transformations are geared towards demands rather than consumption.



Entering Demand Fractions
^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
  As noted above, instead of specifying energy consumption fuel fractions, users specify exogenous *demand* fuel fractions in SISEPUEDE. This is done to preserve logical consistency of transformations; in general, policy targets for fuel mixtures are geared towards satisfying energy demands rather than consumption. It also means, intuitively, that consumption is a logical function of end-use demand rather than demand being a logical function of consumption. The relationship between demand fuel-fractions (entered by the user) and the fraction of fuel consumed (:math:`\alpha^{(C)}_{it}`, data which are often empirically available and should be used for time :math:`t = 0`) is found as

  .. math::
     :label: frac_demand_frac_consumption_relationship

     \alpha^{(D)}_{it} = \frac{\alpha^{(C)}_{it}e_{it}}{\alpha^{(C)}_t \cdot e_t},

  i.e., the point-of-use demand is the efficiency-weighted fraction of consumption. **Demand fuel mix fraction inputs to SISEPUEDE are** :math:`\alpha^{(D)}_{it}` **, and equation** :math:numref:`frac_demand_frac_consumption_relationship` **should be used to calculate them for the initial time period.**


How SISEPUEDE Projects Demands
------------------------------

The components described above in sections components are used to project demands and future consumption in SISEPUEDE. Demands for energy are calculated *at the point of use*. This differs from consumption. The differences between these terms are discussed below.

- **Demand: Demand for energy at point-of-use.** Generalized as :math:`D_t`. Energy demands are energy requirements for completing a task or process.
- **Consumption: Final total energy consumption (by fuel) required to satisfy demands.** Generalized as :math:`C_t`. Energy consumption is higher than demand when energy production systems are inefficient. The more efficient a fuel-technology, the lower the difference between consumption and demand satisfied by that fuel. Fuel consumption is the eventual driver of emissions, while fuel switching to more efficient fuel-techs (or improving the efficiency of existing fuel-techs) can reduce system losses and cut emissions.


Estimating Initial Demand
^^^^^^^^^^^^^^^^^^^^^^^^^

Initial total demand is estimated using the initial consumption and the fraction of consumption as

.. math::
   :label: demand_projection_initial

   D_0 = \sigma_0I_0R_0\left(\alpha^{(C)}_0 \cdot e_0\right) = I_0R_0\left(\alpha^{(C)}_0 \cdot e_0\right)

since :math:`\sigma_0 = 1`.


Projecting Demand and Consumption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the initial point of use demand for energy is estimated, future demand is estimated by scaling the base demand proportionally to growth in the driver and applying the demand scalar ; i.e.,

.. math::
   :label: demand_projection

   D_t = \sigma_t\frac{R_t}{R_o}D_0.

Using :math:`D_t`, it is possible to then estimate how much of each demand is satisfied by fuel :math:`i`, i.e.,

.. math::
   :label: demand_projection_by_fuel

   D_{it} = \alpha^{(D)}_{it}D_t.

Finally, the total consumption of each fuel :math:`i`--the driver of greenhouse gas emissions--is estimated as a function of :math:`D_{it}` as

.. math::
   :label: consumption_projection_by_fuel

   C_{it} = D_{it}e_{it}^{-1}.


Transmission Losses
-------------------
* To model transmission losses in the absence of a network, electricity 
      demands are modeled by inflating two key model elements by the 
      factor \*= 1/(1 - loss):
         1. the InputActivityRatio for electricity-consuming
            technologies (edogenizes transmission loss for fuel 
            production) and
         2. demands from other energy sectors, which are passed in
               SpecifiedAnnualDemand.
      The retrieve_nemomod_tables_fuel_production_demand_and_trade() method accounts for this increase by scaling production and 
      demands by \*= (1 - loss) and assigning the total loss.

* Demands that are returned EXCLUDE transmission losses, keeping with 
      the demands for fuels from other sectors. This is logically 
      consistent with the approach taken with fuel "consumption" by 
      energy subsectors (how much is actually used) versus fuel demands
      by other subsectors (point of use). Here, fuel is demanded to meet
      consumption requirements; what is produced in excess of demand--
      accounting for systemic inefficencies--does not count toward demand
      but consumption (or production in this case). 

* The mass balance equations are as follows:

      (demands + exports)/(1 - transmission_loss_fractions)
      =
      (production + imports)

      Fuel Imports includes electricity eventually lost to transmission 
      because it represents how much electricity originates from other 
      regions (before loss reaches point of use demand). 
      
      Conversely, Fuel Exports does *not* include transmission loss 
      because it represents how much arrives in the other region, which 
      occurs *after* within-region transmission loss between production 
      and exporting.