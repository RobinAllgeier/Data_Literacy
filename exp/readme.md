# Experiment Pipeline Overview

This folder contains exploratory and experimental notebooks following a
structured, reproducible data analysis workflow.

The pipeline is intentionally split into clearly separated stages to avoid
data leakage, unclear assumptions, and undocumented data modifications.

## Workflow

Raw data  
↓  
**01_dataset_overview (describe)**  
Understand the structure of the raw data:

- schema, columns, data types
- time coverage and basic metadata  
  No assumptions or modifications are made at this stage.

↓  
**02_sanity_checks (validate)**  
Validate data correctness:

- detect impossible values
- check consistency rules
- identify duplicates and system errors  
  Issues are detected but not fixed here.

↓  
**03_data_cleaning (fix)**  
Apply documented and reproducible corrections:

- remove or fix objectively incorrect data
- standardize formats and encodings
- handle duplicates  
  Only factual errors are corrected.

↓  
**04_eda (analyze patterns)**  
Exploratory Data Analysis on cleaned data to understand and explain
behavioral and structural patterns in the data:

- target definition, class balance, and baseline rates
- feature distributions and missingness behavior
- feature–target relationships and non-linear effects
- temporal dynamics and segment-based differences

This stage is purely observational. No data is modified.

↓  
**05_modeling (quantify & validate)**  
Simple and interpretable models are used to quantify and validate
patterns identified during EDA:

- feature engineering based on EDA findings
- estimation of effect sizes and directions
- validation of pattern stability
- uncertainty and performance assessment

Modeling serves as a supporting analytical tool rather than the primary objective.

## Design Principles

- Each stage has a single responsibility
- Data modifications occur only in the cleaning stage
- EDA is observational, not corrective
- All steps are reproducible and auditable
