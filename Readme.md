# Data Models Documentation

This repository contains ETL scripts for building dimension tables from raw survey data. Each model extracts, cleans, and transforms relevant columns into standardized formats suitable for analytics and reporting.

---

## 📁 Modules

### dim_entity

Captures core identification metadata for each farmer entity.

- Keys: `entity_id`
- Attributes:
  - `entity_name`: Name of the farmer or household head :: From 
  - `farm_id`: Unique plot or farm reference
  - `gender`: Participant’s gender
  - `year_of_birth`: For age calculation and cohort segmentation
  - `relation_to_entity`: Respondent's role (e.g., self, spouse)
  - `lead_farmer`: Lead farmer designation flag
  - `local_group`: Group or cooperative affiliation
- Notes:
  - Uses semantic column matching for flexible ingestion
  - Deduplicates based on `entity_id` for clean joins


### dim_farmer

Captures personal and contact information for farmer participants.

- Keys: `entity_id`
- Attributes:
  - `first_name`: Farmer’s given name
  - `last_name`: Surname or family name
  - `phone`: Contact number (mobile or landline)
  - `email`: Email address, if available
  - `education_level`: Highest level of formal education completed
- Logic:
  - Semantic column matching across fuzzy survey phrasing
  - Filters out rows missing all fields except `entity_id`
  - Deduplicated for one farmer record per entity


### dim_location

Captures the geospatial and administrative positioning of each farmer or household entity.

- Keys: `entity_id`
- Attributes:
  - `country`: Country name or ISO code
  - `region`: Broad administrative region
  - `sub_region`: Smaller unit within the region
  - `unit`: Operational cluster, cooperative unit, or local admin tag
  - `latitude`, `longitude`: GPS coordinates if available
  - `altitude`: Elevation data (optional)
  - `address_1`, `address_2`: Freeform address fields
- Notes:
  - Fuzzy matched from common geospatial phrasing
  - Non-matching columns flagged in logs
  - Deduplicated on `entity_id`


### dim_plot

Defines foundational plot information linked to an `entity_id`. Helps anchor agronomic activity across seasons or planting cycles.

- Keys: `entity_id`, `plot_number`
- Attributes:
  - `year_started`: Indicates the inception year of the plot
- Cleaning:
  - Filters out records missing plot context
  - Deduplicates rows on full set

### dim_emission_sources

Standardizes metadata about emission sources and their corresponding greenhouse gases. Ensures traceability across variable column naming and pre-cleaned fields.

- Keys: `source_id` (auto-generated)
- Attributes:
  - `raw_column`: Original column name in source data
  - `source_type`: Normalized label for the emission origin
  - `gas_type`: Greenhouse gas detected (e.g., CO2, CH4, N2O, CO2e)
- Cleaning:
  - Regex-driven normalization of column artifacts
  - Updates `Context.used_columns` for pipeline tracking
  - Deduplicates based on cleaned metadata


### dim_entity_cft

Captures foundational entity details collected via CFT (Carbon Footprint Tool) interviews.

- Keys: `entity_id`
- Attributes:
  - `entity_name`: Label for the interviewed household, farm, or unit
  - `interviewee`: Respondent name or role
- Matching:
  - Flexible column resolution using keyword matcher
  - Supports synonyms for improved robustness (e.g., 'Respondent')
- Cleaning:
  - Deduplicated across full entity-level record


### fact_agro_input

Captures reported agrochemical and management inputs for each entity across interview sessions or records.

- Keys: `entity_id`
- Attributes:
  - `fertilizer_used`: Most common fertilizer reported
  - `n_pct`, `p_pct`, `k_pct`: Nutrient composition (% of Nitrogen, Phosphorus, Potassium)
  - `cash_incentive`: Financial support or incentives tied to input usage
  - `unacceptable_4c`: Flags inputs violating 4C standard (e.g. sustainability criteria)
  - `herbicide_usage`: Whether herbicides are applied
  - `weed_management`: Method or degree of weed control practiced
- Cleaning:
  - Dynamically matches column keywords
  - Drops records with no meaningful data
  - Deduplicates responses to avoid inflation

  ## fact_biodiversity_assessment

Captures biodiversity and agroforestry attributes at plot or entity level. Enables longitudinal and sustainability assessments via structured field responses.

### Keys
- `entity_id`

### Attributes
- `coffee_age_above_20_percent`
- `coffee_age_0_2_percent`
- `coffee_age_3_20`
- `bee_hives`
- `biodiversity_habitat`
- `non_coffee_trees_total`
- `non_coffee_trees_per_hectare`
- `non_coffee_tree_species_variety`
- `tree_species`
- `agro_land_percent`
- `interplanted_crops`
- `num_commercial_species`
- `num_robusta_coffee_trees`
- `num_shade_species_excl_commercial`
- `shade_coverage_percent`
- `native_shade_percent`
- `num_shade_species_total`
- `agro_input_risks`
- `weed_management_practices`
- `herbicide_application_frequency`
- `young_tree_stock_percent`

### Notes
- Keyword matching with duplicate safeguard
- Partial synonym support
- Drops fully null rows
- Filters out orphaned `entity_id` entries with no associated data

## fact_co_product

Captures co-product metadata linked to interview results and entities. Enables traceable alignment of secondary products, byproducts, and associated valuation relative to main crops.

### Keys
- `co_product_id` (surrogate key)
- `entity_id`

### Attributes
- `result_id`
- `result_id_parent`
- `co_product_uid`
- `entity_system_id`
- `type`
- `value_relative_to_crop`
- `country`
- `business_name`
- `date_created`
- `date_synched`
- `fa_name`
- `data_versioning`
- `last_modified_by`
- `last_updated`

### Notes
- Semantic column matching with fallback aliases
- Updates `Context.used_columns` for pipeline tracking
- Maps `entity_id` from `entity_system_id` using lookup dictionary
- Removes records with no meaningful data beyond surrogate key
- Deduplicated by all columns


## fact_derived_emission_metrics

Calculates normalized greenhouse gas emission metrics based on crop area and farm gate yield. Supports agronomic and sustainability analysis across spatial and production scales.

### Inputs
- `fact_df`: Emission records containing `emission_value`, `result_id`, and `emissions_id`
- `crop_df`: Crop metadata with `result_id`, `crop_area`, `farm_gate_ready_amount`

### Derived Metrics
- `ghg_per_ha`: Emissions per hectare of land
- `ghg_per_tonne_gbe`: Emissions per tonne of farm gate ready green bean equivalent

### Output
- `emissions_id`
- `ghg_per_ha`
- `ghg_per_tonne_gbe`

### Notes
- Joins on `result_id`
- Replaces 0 with `NaN` to prevent divide-by-zero
- Updates `Context.used_columns` for audit trail
- Filters out records missing both metrics

## fact_economics

Aggregates economics and crop-specific metadata at plot and entity level. Supports profitability modeling, yield assessment, and valuation traceability across cycles.

### Keys
- `entity_id`
- `plot_id` (constructed from `entity_id` + `plot_number`)

### Attributes
- `plot_number`
- `cash_crop_1`
- `cash_crop_2`
- `other_crop_1`
- `other_crop_2`
- `livestock`
- `coffee_sales`
- `crop_name`
- `crop_area`
- `uom_crop_area`
- `harvested_amount`
- `uom_harvested_amount`
- `farmgate_ready_amount`
- `uom_farmgate_ready_amount`
- `residue_amount`
- `uom_residue_amount`
- `residue_management`
- `co_products_used_or_sold`
- `business_name`
- `country`
- `harvest_year`
- `year_of_reporting`

### Notes
- Matches column names using semantic aliases
- Constructs `plot_id` for uniqueness and traceability
- Normalizes string columns to lowercased, stripped values
- Filters out records with no crop or economics data
- Deduplicates across full column set


## fact_emissions

Transforms wide-format emissions columns into structured, analyzable records. Enables downstream greenhouse gas metrics, source tagging, and footprint modeling.

### Keys
- `result_id`

### Attributes
- `source_raw`: Original column name capturing emission source
- `emission_value`: Reported emission value
- `gas_type`: Greenhouse gas type (CO2, CH4, N2O, CO2e)
- `source_type`: Normalized source label

### Notes
- Auto-detects emission columns via gas keyword filtering
- Uses regex to extract gas types from column names
- Cleans raw column names to derive semantic `source_type`
- Converts emission values to numeric with graceful coercion
- Updates `Context.used_columns` for pipeline auditing


## fact_energy_usage

Structures reported energy usage tied to crop or process metadata. Enables source-type tracking, usage quantification, and entity-level auditing.

### Keys
- `energy_usage_id` (surrogate key)
- `entity_id`

### Attributes
- `result_id`
- `result_id_parent`
- `energy_usage_uid`
- `entity_system_id`
- `source_type`
- `usage_quantity`
- `uom_usage_quantity`
- `category`
- `label`
- `fa_name`
- `date_created`
- `date_synched`
- `last_updated`
- `last_modified_by`
- `data_versioning`

### Notes
- Uses semantic alias matching for flexible schema resolution
- Maps `entity_id` from `entity_system_id` using lookup dictionary
- Generates surrogate primary key for join compatibility
- Deduplicates for clean data payloads



## fact_feedback_demographics

Reshapes structured demographic survey responses into a feedback-friendly format. Supports thematic tagging, visualization, and aggregation by entity.

### Keys
- `feedback_id` (surrogate key)
- `entity_id`

### Attributes
- `question_label`: Original column name or semantic label
- `response_text`: Respondent’s answer or numeric value
- `theme_label`: Static tag = 'demographics'

### Notes
- Matches semantic aliases for robustness across survey versions
- Converts wide-format demographics into long-form feedback entries
- Drops empty responses to preserve signal
- Deduplicates to avoid inflation


## fact_intercropped_crops

Explodes multi-valued intercropped crop entries into normalized records for plot-level analysis.

### Keys
- `plot_id`

### Attributes
- `crop_name`: Individual crop name parsed from semi-structured input

### Notes
- Accepts comma, slash, semicolon, period, and ampersand delimiters
- Strips whitespace and filters blanks
- Ensures clean one-to-one mapping of crops per plot
- Drops rows missing either `plot_id` or `intercropped_crops`

## fact_labor_effort

Normalizes household labor effort across crop activities with inferred wage intelligence.

### Keys
- `entity_id`

### Attributes
- `activity_type`: Target enterprise (e.g., coffee, cash_crop_1)
- `labor_source_type`: Origin of labor (family, temporary, permanent)
- `labor_group`: Demographic sub-group (e.g. male, female, other family members)
- `labor_days_per_year`: Days of effort, inferred from wage when applicable
- `average_daily_wage`: Reported or inferred daily wage
- `total_wage_value`: Raw wage value if days couldn't be estimated
- `wage_currency`: Standardized currency code

### Notes
- Dynamically identifies input columns via fuzzy matching
- Converts monetary values into time estimates if threshold triggers
- Ignores “total” aggregates to avoid data duplication
- Lowercases keys for consistency and joins


## fact_nescafe_plan

Consolidates key engagement metrics for the Nescafé Plan participants.

### Keys
- `entity_id`: Unique household or farm identifier

### Attributes
- `year_joined_nescafe_plan`: Year of initial program enrollment
- `training_sessions_male`: Count of trainings attended by male members
- `training_sessions_female`: Count of trainings attended by female members
- `training_sessions_youth`: Count of trainings attended by youth
- `technical_visits`: Total technical support visits received
- `plantlets_received`: Number of coffee plantlets distributed
- `plantlets_survived`: Number of plantlets that survived
- `condition_plantlets`: Qualitative assessment of plantlet health
- `satisfaction_plantlets`: Reported satisfaction with plantlet quality
- `renovation`: Indicator or value reflecting plot renovation efforts
- `expansion`: Indicator or value reflecting plot expansion efforts

### Notes
- Dynamically resolves column names using fuzzy matching
- Skips entirely blank rows and removes duplicates
- Logs missing columns for downstream auditability

## fact_plot_production

Captures plot-level crop production and tree composition with entity linkage and precision typing.

### Keys
- `plot_id`: Composite key from `entity_id` + `plot_number`

### Attributes
- `entity_id`: Household or farm identifier
- `plot_number`: Local plot reference
- `production_description`: Reported production type or notes
- `production_kg`: Harvested yield in kilograms
- `total_coffee_tree`: Total coffee trees recorded
- `coffee_tree_active`: Trees actively producing
- `coffee_tree_rejuvenation`: Trees under recovery stage
- `coffee_tree_intercropping`: Coffee trees in mixed cropping systems
- `coffee_tree_monocropping`: Trees under pure coffee cultivation
- `total_tree_other_crops`: Other tree crops observed
- `total_tree_other_crops_update`: Updated count of non-coffee crops
- `total_tree_conservation`: Trees managed for conservation or natural habitat
- `total_tree_other_uses`: Trees serving alternate land use purposes
- `total_tree`: All tree types aggregated

### Notes
- Auto-detects column names via fuzzy matching
- Parses and standardizes plot numbers as strings
- Constructs unique `plot_id` for clean joins and indexing
- Filters out rows lacking meaningful data beyond plot identifiers



## fact_plot_structure

Captures structured spatial attributes of farm plots, linking entity identifiers to granular crop area allocations.

### Keys
- `plot_id`: Composite key from `entity_id` and `plot_number`

### Attributes
- `entity_id`: Household or farm identifier
- `plot_number`: Field-level reference number
- `total_area`: Reported total size of the plot
- `total_coffee_area`: Area dedicated to coffee cultivation
- `coffee_area_active`: Area under active coffee production
- `coffee_area_rejuvenation`: Coffee area in recovery phase
- `coffee_area_intercropping`: Mixed cropping systems involving coffee
- `coffee_area_monocropping`: Sole coffee cultivation area
- `land_area_other_crops`: Area planted with non-coffee crops
- `land_area_natural_habitat_or_conservation`: Area allocated to conservation or natural habitat
- `farm_area_other_uses`: Land designated for other agricultural or non-agricultural purposes
- `intercropped_crops`: List or description of crops intercropped with coffee

### Notes
- Dynamically resolves input column names with fuzzy matching
- Drops empty rows and removes duplicates
- Normalizes plot numbers to string format for consistency
- Filters incomplete entities while preserving plot structure fidelity


## fact_recordkeeping

Standardizes household-level recordkeeping and financial inclusion indicators.

### Keys
- `entity_id`: Unique household or farm identifier

### Attributes
- `financial_records`: Presence or quality of financial management records
- `cash_incentive`: Participation in cash-based support programs
- `insurance`: Access to agricultural or financial insurance
- `vsla`: Participation in Village Savings and Loan Associations (VSLA)

### Notes
- Resolves column names using fuzzy matching
- Logs missing column references for audit support
- Filters out fully blank rows and duplicate entries


## fact_revenue_economics

Extracts core economic indicators tied to crop yield, production, and input costs.

### Keys
- `entity_id`: Household or farm identifier

### Attributes
- `yield_gc_per_ha`: Gross crop yield per hectare
- `production_kg`: Total harvested production in kilograms
- `price`: Reported market price per unit
- `total_fertilizer_applied_kg_per_ha`: Quantity of all fertilizers applied per hectare
- `organic_fertilizer_applied_kg_per_ha`: Quantity of organic fertilizers used per hectare

### Notes
- Column names resolved via fuzzy matching
- Applies snake_case formatting for pipeline compatibility
- Removes duplicate entries to ensure clean joins


## fact_soil_assessment

Normalizes soil-related agronomic practices and analysis metrics into structured survey-ready format.

### Keys
- `entity_id`: Unique farm or household identifier

### Attributes
- `cover_crops`: Use or extent of cover cropping and residue application
- `erosion_practices`: Adoption of erosion control measures (e.g., terracing, windbreaks)
- `erosion_risk_estimate`: Reported risk severity across erosion types
- `soil_analysis`: Indicator of soil testing practices
- `interval_soil_analysis`: Frequency or interval of soil analysis activity
- `fertilizer_plan`: Existence of a formal fertilizer planning approach
- `fertilizer_ratio`: Ratio of organic to total fertilizer usage
- `fertilizer_type`: Most used fertilizer type
- `fertilizer_total_kg_ha`: Total fertilizer applied per hectare
- `yield_gc_per_ha`: Yield per hectare (gross crop)
- `soil_organic_matter`: Reported level of organic matter
- `soil_ph`: Measured soil pH level

### Notes
- Proactive column inspection detects duplicates and unmapped expectations
- Fuzzy matching accommodates variation in field naming conventions
- Dropped blanks and duplicates prior to return
- Filters out entities with no meaningful agronomic detail

## fact_water_management

Standardizes data related to on-farm water use, irrigation practices, and water conservation strategies for traceability and sustainability reporting.

### Keys
- `entity_id`: Unique household or farm identifier

### Attributes
- `irrigate_coffee`
- `irrigation_source`
- `irrigation_rounds`
- `irrigation_quantity`
- `wet_processing`
- `do_you_wet_process`
- `monitor_water_usage_wet_processing`
- `monitor_water_usage_irrigation`
- `water_usage_wet_processing`
- `soil_moisture_monitoring`
- `wastewater_treatment`
- `do_you_have_water_bodies`
- `riparian_buffer`
- `min_distance_field_to_water_body`
- `water_usage_irrigation_avg`

### Notes
- Applies semantic keyword matching for column resolution
- Logs missing fields for schema validation transparency
- Filters rows with empty entity IDs or non-informative records
- Deduplicates data and resets index for model compatibility

## fact_feedback_demographics

Processes household-level feedback survey data focused on educational background, family structure, and succession planning.

### Keys
- `entity_id`: Unique household or respondent identifier

### Attributes
- `survey_year`: Annual reporting period
- `collection_date`: Date when survey was conducted
- `num_adults`: Count of adult individuals in the household
- `num_boys`: Count of male children
- `num_girls`: Count of female children
- `successor`: Stated farm successor or inheritance plan
- `education_id`: Normalized education level based on `dim_education_df`

### Notes
- Combines semantic matching with fuzzy logic for column resolution
- Integrates education dimension via left join for relational normalization
- Filters duplicates by household and survey year
- Minimizes risk of missing demographic values by logging unmatched mappings


# Utils 
## match function

Performs token-level, case-insensitive partial matching between column names and keyword phrases to identify semantically aligned fields in unstandardized data.

### Purpose
- Enables schema-agnostic extraction by matching keyword tokens to candidate column names.

### Logic
- Tokenizes both keywords and column names by:
  - Lowercasing text
  - Removing punctuation
  - Splitting into individual words
- Returns first column whose tokens contain all keyword tokens.

### Parameters
- `df`: Input DataFrame containing columns to search
- `keywords`: List of keyword phrases to attempt matching

### Returns
- `str`: Name of the matched column or `None` if no match found

### Notes
- Ignores column case and punctuation for robust matching
- Designed for flexibility in survey schema intake workflows
- Prioritizes subset token containment for intelligent alignment

## matcher

Implements partial, case-insensitive token matching to locate columns whose names align with loosely structured keyword phrases—ideal for handling noisy survey headers or ad hoc data entry formats.

### Purpose
- Enables schema-independent extraction of columns when headings vary across datasets.
- Provides fallback matching logic for ingestion pipelines when standard semantic resolution fails.

### Matching Logic
- Extracts tokens from both keywords and column names using:
  - Lowercasing
  - Punctuation removal
  - Word boundary tokenization
- Returns the first column whose token set fully contains the tokens from any keyword in `keywords`.

### Parameters
- `df`: Source `pandas.DataFrame` to evaluate.
- `keywords`: List of descriptive strings that may refer to a column.

### Returns
- `str`: Column name from `df.columns` that matches any keyword token set.
- Returns `None` if no column passes containment test.

### Notes
- Works well on real-world survey and agricultural datasets where column headers often lack consistency.
- Can be composed with fuzzy or semantic matchers for hybrid resolution pipelines.
- Avoids strict string equivalence for better resilience during ETL.

# pipeline.py

Central orchestrator function for executing a full ETL workflow across multiple dimensions, facts, and feedback survey models—standardizing, cleaning, linking, and preparing data for agricultural analytics, climate modeling, and sustainability reporting.

### Purpose
- Applies unified preprocessing to raw datasets
- Executes modular builders across `dim`, `fact`, `feedback`, and `cft` categories
- Resolves entity references and column provenance
- Produces a dictionary of normalized DataFrames for downstream modeling or dashboarding

---

### Parameters

| Name                     | Type           | Description                                                                 |
|--------------------------|----------------|-----------------------------------------------------------------------------|
| `df_raw`                 | `pd.DataFrame` | Primary dataset with all survey and farm-level responses                   |
| `df_cft`                 | `pd.DataFrame` | Dataset related to CFT (Carbon Footprint Tool) and emissions               |
| `df_co_product_cleaned` | `pd.DataFrame` | Pre-cleaned co-product dataset for emissions linkage                       |
| `df_energy`              | `pd.DataFrame` | Farm-level energy usage entries                                            |
| `df_fertilizer_input`    | `pd.DataFrame` | Fertilizer input dataset (currently unused in this pipeline)               |

---

### Workflow Overview

#### Data Cleaning
- `df_raw` → cleaned via `process_dataframe`
- `df_cft` → cleaned via `secondary_cleaner.clean_dataframe`

#### Dimension Builders
- `dim_*`: Constructs structural records—entity, location, plot, farmer, education, etc.
- Education dimension (`dim_education`) is reused for survey enrichment

#### Fact Builders
- Executes domain-specific normalization for soil, water, inputs, labor, economics, plot structure
- `fact_intercropped_crops`: derived from `fact_plot_structure`
- Feedback demographics merged with `dim_education` for enrichment

#### Feedback Modules
- Resolves questions, survey structure, and validator views
- Aggregates used columns to avoid duplication in `fact_survey_feedback`

#### CFT-Specific Models
- Normalizes `dim_entity_cft` and `dim_geolocation_cft` from CFT data
- Resolves entity ID lookup mapping for shared facts
- Builds emissions, co-product, and energy usage models

---

### Returns

- `dict`: Keyed dictionary of all dimension, fact, feedback, and CFT DataFrames

---

### Notes
- `df_fertilizer_input` is passed but not yet used—placeholder for future expansion
- Modular design supports easy extension (e.g., adding `fact_soil_health_index`)
- `used_columns` tracking improves feedback deduplication and semantic integrity

## run.py [ETL Pipeline Entrypoint]

Entry-level function that coordinates the entire ETL execution—from data ingestion and cleaning to transformation and export—powered by `pipeline`. This script is designed to be run as a standalone module and is suitable for automation within scheduled tasks or containerized workflows.

---

### Responsibilities
- Resolves absolute paths for input and output directories
- Loads raw CSV and Excel input files
- Cleans both main and CFT datasets using specialized cleaning utilities
- Executes the modular pipeline via `pipeline`
- Saves each output DataFrame as a CSV artifact in a structured folder
- Prints completion status and output locations for verification

---

### Key Paths

| Name         | Description                          |
|--------------|--------------------------------------|
| `raw_dir`    | Directory containing unprocessed data |
| `output_dir` | Destination for all processed outputs |
| `cleaned_file_path` | Saved copy of cleaned raw data for inspection |

---

### Input Files
- `raw_data.csv`: Primary farm survey and structural data
- `cft.xlsx`: Multi-sheet input for Carbon Footprint Tool (CFT), energy usage, and emissions
  - Sheet 0: CFT master
  - Sheet 1: Co-product & fertilizer input
  - Sheet 4: Energy usage

---

### Output
- Individual CSV files per normalized `dim`, `fact`, and `feedback` dataset
- All saved to `output_dir` with schema-ready formatting

---

### Notes
- Encoding set to `ISO-8859-1` for compatibility with legacy survey exports
- Uses `low_memory=False` to ensure data type inference doesn't fail on large CSVs


---

### Execution
```bash
python main.py

