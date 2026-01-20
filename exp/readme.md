# Experiment Pipeline Overview

This folder contains exploratory notebooks that follow a **clear and structured data analysis workflow**.  
The goal is to understand the data step by step and discover meaningful patterns based on a single original dataset.

Throughout the analysis, additional tables may be derived from the cleaned data, but all results ultimately build on the same initial dataset.

## Workflow

Raw data  
↓

### **01_dataset_overview (understand the data)**

Initial exploration of the raw data:

- available columns and data types
- time coverage and basic structure
- rough size and completeness

No assumptions are made and no data is modified at this stage.

↓

### **02_sanity_checks (identify issues)**

Check whether the data is plausible and internally consistent:

- impossible or suspicious values
- inconsistencies between columns
- duplicates or system-related artifacts

Issues are **detected but not corrected** here.

↓

### **03_data_cleaning (apply fixes)**

Apply clearly justified and reproducible corrections:

- remove or fix objectively incorrect records
- standardize formats and encodings
- handle duplicates

Only **factual errors** are corrected.  
All changes are documented and reproducible.

↓

### **04_eda (explore patterns)**

Exploratory Data Analysis on cleaned data to understand behavior and structure:

- distributions and baseline rates
- relationships between variables
- differences across user groups, media types, and time
- analysis based on derived tables created from the cleaned dataset

This stage is **purely observational**.  
No data is modified. The focus is on _finding and explaining interesting patterns in the data_.
