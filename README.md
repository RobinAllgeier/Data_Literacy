# Data Literacy Project: Learning and Habits: The Borrowing Behavior of Library Users

📄 **Final Report:**  
[![PDF](https://img.shields.io/badge/PDF-Download-blue)](report.pdf)

## Introduction

This repository contains the code and analysis for a data literacy project examining user behavior in a library borrowing system.
Using a single, structured dataset, we perform descriptive analyses to study learning effects, temporal usage patterns, and regularity in user behavior over repeated borrowing sessions. The project follows a transparent and reproducible workflow, including dataset inspection, sanity checks, data cleaning, and exploratory analysis.
The focus is on understanding behavioral patterns in the data rather than building predictive models.

## Project Structure

```
DATA_LITERACY/
├── report.pdf          # Final project report (compiled PDF)
│
├── data/
│   ├── raw/            # Original, unmodified input data as provided
│   └── processed/      # Cleaned and derived datasets used for analysis
│
├── doc/
│   ├── report/         # LaTeX sources and figures for the report
│   ├── presentations/ # Slides used for project presentations
│   └── protocols/     # Meeting notes, project protocols, decisions
│
├── exp/
│   ├── *.ipynb         # Exploratory analysis notebooks
│   │                  # (dataset overview, sanity checks, data cleaning, EDA)
│   └── utils/          # Helper functions used by the notebooks
│
├── src/
│   ├── plotting/       # Reusable plotting code for final figures
│   ├── config/         # Central configuration (column names, constants)
│   ├── features/       # Feature construction and aggregation logic
│   ├── io/             # Data loading and saving utilities
│   ├── preprocess/     # Data cleaning and preprocessing steps
│   └── validate/       # Sanity checks and data validation logic
```
