<title>Title</title>


[toc]

#How to Use this Documentation

This documentation provides a detailed look at data requirements and interactions between models in the Latin American—Integrated Decarbonization Pathways Model (LAC-IDPM).

### Metavariables and Constructing Input Parameters
This document makes use of the \$VARNAME\$ notation to denote metavariables for parameter and model input variable name.

For example, model input variables used to denote agricutlural activity emission factors by crop type and gas in `model_input_variables.csv` may have the following structure:
`$CAT-CROP$_ef_$UNIT-MASS$_$EMISSION-GAS$_$UNIT-AREA$`, where

- `$CAT-CROP$` is the categorical crop type (e.g., pineapple, banana, coffee, etc.);
- `$EMISSION-GAS$` is the greenhouse gas that is being emitted (e.g. CO$_2$, N$_2$O, CH$_4$)
- `$UNIT-MASS$` is the unit of mass for gas emission (e.g., `kg` for kilograms; some sector variables may use `gg` for gigagrams);
- `$UNIT-AREA$` is the area unit (e.g., `ha` is hectares).

As a tangible example, the CO$_2$ emission factor for banana production, which captures crop burning, decomposition, and other factors, would be entered as `banana_ef_kg_CO2_ha` since, in this case, `$CAT-CROP$ = banana`, `$EMISSION-GAS$ = CO2`, `$UNIT-AREA$ = ha`, `$UNIT-MASS$ = kg`. Similarly, the N$_2$O factor, which includes crop liming and fertilization, would be captured as `banana_ef_kg_N2O_ha`.

These are _metavariables_, which characterize and describe the notation for naming model input variables. Each variable is associated with some _schema_. In the following sections, data requirements and the associated naming schema for `model_input_variables.csv` are 


#Preliminary Variable Estimates for Calibration

### continue description here....
<br><br>

#Entering Variable Trajectories for Model Runs

The input sheet `model_input_variables.csv` stores input variables trajectories for all variables and parameters that are included in the 
### continue description here....
<br><br>

#Data Requirements and Variable Schema


## Cross Sector Data

### Emission Attributes and Information

Per IPCC AR6 guidance and the emission metrics used to quantify emission targets in the Paris Agreement, we use the 100-year global warming potential for GHGs. See IPCC AR6 WG1 Physical Basis Chapter 7 and Table 7.SM.7 for the source of these values.

| Gas | Name | CO$_2$ equivalent factor | `$EMISSION-GAS$` |
| --------- | --------- | --------- | ----------- |
| CH$_4$ | Methane | 27.9 | `CH$_4$` |
| CO$_2$ | Carbon Dioxide | 1 | `CO$_2$` |
| HFC-23 | HFC-23 | 14600 | `hfc23` |
| HFC-32 | HFC-32 | 771 | `hfc32` |
| HFC-125 | HFC-32 | 3740 | `hfc125` |
| HFC-134a | HFC-134a | 1530 | `hfc134a` |
| HFC-143a | HFC-143a | 5810 | `hfc143a` |
| HFC-152 | HFC-152 | 21.5 | `hfc152` |
| HFC-152a | HFC-152a | 164 | `hfc152a` |
| N$_2$O | Nitrus Oxide | 273 | `N$_2$O` |
| SF6 | Sulfur Hexflouride | 25200 | `sf6` |

Additional gasses can be added as necessary

In the git, the file `ref/attribute_ghg.csv` stores this relevant information for use by the integrated modeling system.
<br>


### General Data by Country

#### General Variables

The following variables are required for each country.

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area of Region | Units: Hectares (ha). Total area of the country, including land and water. | `area_country_ha` |
| Urban Population | Units: # of people | `population_urban` |
| Rural Population | Units: # of people | `population_rural` |

#### continue description here and more attributes....
<br>

#### Economic Variables

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Gross Domestic Product | Units: Billion USD (2020\$) | `gdp_mmm_usd` |
| Commercial Value Added — ### | Units: Billion USD (2020\$) | `va_industry_mmm_usd` |
| Industrial Value Added — ### | Units: Billion USD (2020\$) | `va_e###` |

<br><br>



#### Sector Attributes and Information

There are four primary emission sectors included in LAC-IDPM: Agriculture, Forestry, and Land Use (AFOLU), Circular Economy, Energy, and Industrial Processes and Product Use (IPPU). Metavariable details for each of the sectors is located in the table below.  

|  Sector | Sector Name | `$ABBREVIATION-SECTOR$` | Description |
| --------- | --------- | --------- | --------- |--------- |
| AFOLU | Agriculture, Forestry, and Land Use | `af` | Agricultural activity .... | Python |
| Circular Economy | Circular Economy | `ce` | Activity (demand and emission) associated with emissions from domestic |
| Energy | Energy | `en` | Activity (demand and emission) associated with energy generation and consumption, including:  <ul><li>stationary emissions in commercial, municipal, and residential buildings;<li>electricity generation; <li>industrial energy use; and <li>transportation.</ul> |
| IPPU | Industrial Processes and Product Use | `ip` | Activity (demand and emission) associated with industrial processes and product use. |



#### Subsector Attributes and Information

The table below contains metavariable information and descriptions for all subsectors associated with each sector included in the analysis.

|  Sector | Subsector | `$ABBREVIATION-SUBSECTOR$` | Description | Model |
| --------- | --------- | --------- | --------- |--------- |
| AFOLU | Agriculture | `agrc` | Agricultural activity .... | Python |
| AFOLU | Forest | `frst` | Forest activity .... | Python |
| AFOLU | Land Use | `lndu` | Land use activity .... | Python |
| AFOLU | Livestock | `lvst` | Livestock activity .... | Python |
| Circular Economy | Liquid Waste | `wali` | Activity (demand and emission) associated with emissions from domestic and industrial liquid waste disposal and treatment | Python |
| Circular Economy | Solid Waste | `waso` | Activity (demand and emission) associated with emissions from domestic and industrial solid waste disposal and treatment  | Python |
| Energy | Buildings  | `bldg` | Stationary emissions in commercial, residential, and municipal buildings (e.g., water heaters, gas stoves, fireplaces, etc.).... | Python |
| Energy | Electricity Generation | `elec` | Activity (demand and emission) associated with the generation of electricity .... | Julia |
| Energy | Industrial Energy | `inen` | Activity (demand and emission) associated with the energy generation and use in industry | Julia |
| Energy | Transportation | `trns` | Activity (demand and emission) associated with the generation of electricity .... | Julia |
| IPPU | IPPU | `ippu` | Activity (demand and emission) associated with industrial processes and product use | Python |






## AFOLU

### <u>Agriculture</u>

#### Variables by Category

For each agricultural category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema | Variable Type | Notes |
| --------- | --------- | ----------- | ----------- | ----------- |
| Cropland area proportion | Proportion of total **crop** area (%/100) | `frac_area_cropland_$CAT-AGRICULTURE$` | 
| CH$_4$ Emission Factor | Annual average CH$_4$ (methane) emitted per ha of crop grown. <b>RICE is the only crop this is needed for. This will be 0 for most crops (or negligible).</b> | `ef_agactivity_$CAT-AGRICULTURE$_$UNIT-MASS$_CH4_$UNIT-AREA$` (`$EMISSION-GAS$ = CH$_4$`, `$UNIT-AREA$ = ha`)|
| CO$_2$ Emission Factor | Annual average CO$_2$ (carbon dioxide) emitted per ha of crop; for the purposes of accounting and calibration, this includes the following categories: crop burning, ##CONTINUE LISTING HERE## | `ef_agactivity_$CAT-AGRICULTURE$_$UNIT-MASS$_CO2_$UNIT-AREA$` (`$EMISSION-GAS$ = CO2`, `$UNIT-AREA$ = ha`)|
| N$_2$O Emission Factor | Annual average N$_2$O (nitrous oxide) emitted per ha of crop grown  (≥ 0) | `ef_agactivity_$CAT-AGRICULTURE$_$UNIT-MASS$_N2O_$UNIT-AREA$` (`$EMISSION-GAS$ = N$_2$O`, `$UNIT-AREA$ = ha`)|

#### Categories
Agriculture is divided into the following categories (crops), given by `$CAT-AGRICULTURE$`. Each crop is associated with one of these classifciations (see https://www.fao.org/waicent/faoinfo/economic/faodef/annexe.htm for the source of these classifications—on the git, the table `ingestion/FAOSTAT/ref/attribute_fao_crop.csv` contains the information mapping each crop to this crop type). If a crop type is not present in a country, set the associated area as a fraction of crop area to 0.

| Category Name | `$CAT-AGRICULTURE$` | Description | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- |  ----------- |  ----------- |  ----------- |
| Beverage crops | `bevs_and_spices` | tea, cinnamon, vanilla, cocoa, matte, ginger, peppermint, ... |
| Coffee (café) | `coffee` | Coffee crops |
| Fibers | `fibers` | Cotton, tobacco, flax, hemp, ... |
| Fruits (frutas) | `fruits` | Fruits including bananas, mangoes, plantains, oranges, lemons, apples,...  |
| Maize (maize) | `maize` | Corn (maize) crops |
| Nuts | `nuts` | brazil nuts, cashew nuts, chestnuts, almonds, walnuts, pistachios, hazelnuts, kolanuts,... | 
| Pulses and derived crops | `pulses` | Pulses include dry beans, peas, lentiles, ... | 
| Oil bearing crops | `oil_bearing` | Perennial crops used primarily for oil, inlcuding soy, palm, coconuts, olives,...| 
| Other Cereal and Grains | `cereals` | Crops including wheat, sorghum, barley, ... |
| Rice | `rice` | Paddy rice crops (methane emitting) |
| Roots and Tubers | `tubers` | Roots and tubers, including cassava, potatoes, yams, ...|
| Vegetables | `vegetables` | Vegetables including cabbages, artichokes, asparagus, lettuce, spinach, tomatoes, cauliflower, chilies, eggplant, onions, garlic ...|


Emission factors for these crops can be estimated by starting with default values from the IPCC guidance on 

### continue additional crop types, will look through FAO


#### Cost Requirements by Category

For each agricultural category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema | Variable Type |
| --------- | --------- | ----------- | ----------- |

To be completed.

<br>



### <u>Forestry</u>


#### Variables by Category

For each forest category, the following variables are needed. 

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area proportion | Proportion of total **forested** area (%/100) | `frac_area_forest_$CAT-FOREST$` |
| Sequestration Emission Factor | Annual average CO$_2$ emitted per ha from sequestration (< 0 – this is a negative number) | `ef_sequestration_$CAT-FOREST$_$UNIT-MASS$_CO2_$UNIT-AREA$` (`$EMISSION-GAS$ = CO2`)|
| Forest Fire Emission Factor | Annual average CO$_2$ emitted per ha due to forest fires (≥ 0) | `$CAT-FOREST$_ef_ff_$UNIT-MASS$_CO2_$UNIT-AREA$` (`$EMISSION-GAS$ = CO2`)|
<br>


#### Categories
Forest should divided into the following categories, given by `$CAT-FOREST$`.

| Category Name | `$CAT-FOREST$` | Definition |
| --------- | --------- | ----------- |
| Mangroves | `mangroves` | |
| Primary Wet Forest | `primary_wet` | |
| Primary Dry Forest | `primary_dry` | |
| Secondary Wet Forest | `secondary_wet` | |
| Secondary Dry Forest | `secondary_dry` | |




### <u>Land Use</u>


#### Variables by Category
For each category, the following variables are needed. 

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area proportion | Proportion of total **country** area (%/100) | `frac_lu_$CAT-LANDUSE$` |
| Transition probability | Transition probability in steady state Markov Chain Model | `pij_$CAT-LANDUSE-I$_$CAT-LANDUSE-J$` |


#### Variables by Category (Partial Categories)
For each category, the following variables are needed. 

| Variable | Information | Variable Schema | Categories |
| --------- | --------- | ----------- | ----------- |
| CO2 Emission Factor | Annual average CO$_2$ emitted per ha due to soil carbon release from grazing (**CHECK THIS**) | `$CAT-LANDUSE$_ef_soil_carbon_$UNIT-MASS$_CO2_$UNIT-AREA$` | `pasture` |
| N2O Emission Factor | Annual average N$_2$O emitted per ha due to soil carbon release from liming and fertilization (**CHECK THIS**) | `$CAT-LANDUSE$_ef_exist_$UNIT-MASS$_N2O_$UNIT-AREA$` | `pasture` |
| CH4 Emission Factor | Annual average CH$_4$ emitted per ha due decay of organic matter (**CHECK THIS**). | `$CAT-LANDUSE$_ef_boc_$UNIT-MASS$_CH4_$UNIT-AREA$` | `wetlands` |


#### Categories
Land use should be divided into the following categories, given by `$CAT-LANDUSE$`. Note that the sum of land use area across all land use, forestry, agriculture, and livestock land use categories should equal the total amount of land available in the country. 

| Category Name | `$CAT-LANDUSE$` | Definition | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- | --------- | --------- | ----------- |
| Cropland | `cropland` | Area of land devoted to growing crops for consumption and industrial purposes. Does not include pastures. |
| Forests | `forest` | Area of land covered in forest (including mangroves) |
| Grasslands | `grassland` | Area of non-pasture grassland |
| Other | `other` | Other land use categories |
| Pastures | `pasture` | Area of land used to support grazing for livestock |
| Settlements | `settlement` | Area of land devoted to urban/suburban development. Assume no emissions from land use (primary emissions captured elsewhere). |
| Water | `water` | Area of the country considered to be water (non-land area). |
| Wetlands | `wetlands` | Wetlands, which emit CH4|

<br>

### <u>Livestock</u>

#### Variables by Category
For each category, the following variables are needed. Information on enteric fermentation can be found <a href = "https://www3.epa.gov/ttnchie1/ap42/ch14/final/c14s04.pdf">the EPA</a> and ADDITIONAL LINKS HERE.

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Head count | Population of animal | `livestock_pop_$CAT-LIVESTOCK$` |
| CH4 Enteric Fermentation Emission Factor  | This represents emissions from enteric fermentation, primarily occuring in multigastric ruminant livestock (including buffalo, camels, cattle, goats, and sheep) | `ef_lvst_entferm_$CAT-LIVESTOCK$_$UNIT-MASS$_N2O_$UNIT-POPULATION$` | 
| CH4 Manure Management Emission Factor  | Methane emissions from manure management | `ef_lvst_mm_$CAT-LANDUSE$_$UNIT-MASS$_N2O_$UNIT-AREA$` | 
| N2O Enteric Fermentation Emission Factor  | This represents emissions from enteric fermentation, primarily occuring in multigastric ruminant livestock (including buffalo, camels, cattle, goats, and sheep) | `ef_lvst_$CAT-LANDUSE$_$UNIT-MASS$_N2O_$UNIT-AREA$` | 

#### Categories
Emissions in the livestock subsector are assumed to be driven by the number of animals raised for dairy and slaughter. Livestock is associated with divided into the following categories, given by `$CAT-LIVESTOCK$`. 

| Category Name | `$CAT-LIVESTOCK$` | Definition | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- | --------- | --------- | ----------- |
| Buffalo | `buffalo` | Buffalo livestock | | | |
| Cattle - Meat | `cattle_meat` | Cattle livestock raised exclusively for meat| | | |
| Cattle - Dual Purpose| `cattle_dual_purpose` | Cattle livestock raised for dairy, then meat | | | |
| Cattle - Dairy | `cattle_dual_purpose` | Cattle livestock raised for dairy only | | | |
| Chickens | `chickens` | Chicken livestock| | | |
| Goats | `goats` | Goats livestock| | | |
| Horses | `horses` | Horses livestock| | | |
| Mules | `mules` | Mules livestock| | | |
| Sheep | `sheep` | Sheep livestock| | | |
| Pigs | `pigs` | Pigs livestock| | | |



`https://www.epa.gov/sites/default/files/2020-10/documents/ag_module_users_guide.pdf`
<br><br>

## Energy

Energy includes a range of variables and categories. Given the integrated nature of the energy sector, there are several "cross subsector" categories required to construct the NemoMOD energy model. These categories include Fuel, Technology

The dimensions required for the NemoMOD framework are available from the <a href="https://sei-international.github.io/NemoMod.jl/stable/dimensions/">NemoMOD Categories Documentation</a>.

### <u>Stationary Emissions</u>

#### Variables by Category

For each stationary emissions category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema | Variable Type | Notes |
| --------- | --------- | ----------- | ----------- | ----------- |
| Demand for fuel type per activity | Fraction of energy used to power processes with each fuel `FUEL`. Note: for each process, across fuels, these variables must sum to 1. | `dem_se_$CAT-STATIONARY-EMISSIONS$_$FUEL$_per_$ACTIVITY(CAT-STATIONARY-EMISSIONS)$` |||

#### Categories

| Category Name | `$CAT-STATIONARY-EMISSIONS$` | `$ACTIVITY(CAT-STATIONARY-EMISSIONS)$` | Description | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- |  ----------- |  ----------- |  ----------- |
| Commercial | `commercial` | `va_commercial` | Commercial stationary emissions, including heating, cooking, etc... | | | |
| Municipal and Public | `public` | `population` | Municipal and public stationary emissions, including from heating, cooking, etc... | | | |
| Residential | `residential` | `households` | Residential stationary emissions, including heating, cooking, etc... | | | |

<br>

### <u>Electricity Generation</u>
<br>

### <u>Industrial Energy</u>

#### Variables by Category

For each agricultural category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema | Variable Type | Notes |
| --------- | --------- | ----------- | ----------- | ----------- |
| Fraction of power from fuel `FUEL` | Fraction of energy used to power industrial processes associated with `CAT-INDUSTRY` that come from fuel `FUEL`. Note: for each process, across fuels, these variables must sum to 1. | `` |||

#### Categories

| Category Name | `$CAT-WASTE-LIQUID$` | Description | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- |  ----------- |  ----------- |  ----------- |
| Beverage crops | `bevs_and_spices` | tea, cinnamon, vanilla, cocoa, matte, ginger, peppermint, ... |
<br>

### <u>Transportation</u>
<br><br>


## Industrial Processes and Product Use

### <u>Industrial Processes and Product Use</u>

<br><br>

## Circular Economy

### <u>Liquid Waste</u>

#### Variables by Category

For each agricultural category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema | Variable Type | Notes |
| --------- | --------- | ----------- | ----------- | ----------- |
| Cropland area proportion |||||

#### Categories

| Category Name | `$CAT-WASTE-LIQUID$` | Description | Data Source | Hyperlink | Notes |
| --------- | --------- | ----------- |  ----------- |  ----------- |  ----------- |
| Beverage crops | `bevs_and_spices` | tea, cinnamon, vanilla, cocoa, matte, ginger, peppermint, ... |

<br>

### <u>Solid Waste</u>
<br>

<br><br>




#Data Glossary

### Aggregate, easy to read variable table here for quick reference... See below...

## Summary of metavariables

etc.

* `EMISSION-GAS`
* `FUEL`
* `CAT-AGRICULTURE`
* `CAT-FOREST`
* `CAT-INDUSTRY`
* `CAT-LANDUSE`
* `CAT-LIVESTOCK`
* `CAT-WASTE-LIQUID`
* `CAT-WASTE-SOLID`
* `TECHNOLOGY`
* `UNIT-AREA`
* `UNIT-MASS `

## Summary of variables required

| Sector | Subsector | Variable | Information | Variable Schema | Varies by |
| --------- | --------- | --------- | --------- | ----------- | ----------- |
| All | - | Area of Region | Units: Hectares (ha) | `area_country_ha` | - |
| All | - | Urban Population | Units: # of people | `population_urban` | - |
| All | - | Rural Population | Units: # of people | `population_rural` | - |
| AFOLU | Forestry | Area proportion by crop type | Proportion of total country area (%/100) | `frac_lu_$CAT-FOREST$` | `$CAT-FOREST$` |
| AFOLU | Forestry | Sequestration Emission Factor | Annual average CO$_2$ emitted per ha from sequestration (< 0 – this is a negative number) | `$CAT-FOREST$_ef_seq_$UNIT-MASS$_CO2_$UNIT-AREA$` (`$EMISSION-GAS$ = CO2`)| `$CAT-FOREST$` |
| AFOLU | Forestry | Forest Fire Emission Factor | Annual average CO$_2$ emitted per ha due to forest fires (≥ 0) | `$CAT-CROP$_ef_ff_$UNIT-MASS$_CO2_$UNIT-AREA$` (`$EMISSION-GAS$ = CO2`)| `$CAT-FOREST$` |