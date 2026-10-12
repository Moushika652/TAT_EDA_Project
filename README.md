# EDA and Deep Learning on Edge-Server Task Turnaround Times (TAT)

Course project - U21ADP05 Exploratory Data Analysis and Visualization (Assignment II), KPR Institute of Engineering and Technology.

## Overview
Exploratory analysis, distribution fitting, visualization and MLP-based prediction of image-recognition task
turnaround times measured on four edge servers (2 x MacBook Pro, VM, Raspberry Pi; 4,000 records).
The raw files have 2 columns, so 22 fields were derived (lags, rolling statistics, gaps, time parts) -> 24 columns.

## Contents
- `report/EDA_Case_Study_Report.docx` - full project report
- `src/tat_eda_dl.py` - complete pipeline (load, EDA, preprocessing, 21 figures, MLP training, evaluation)
- `data/` - the four source CSVs and the dataset description PDF
- `figures/` - all generated figures
- `outputs/` - results.json, engineered dataset, tuning tables, `interactive_tat.html` (interactive explorer)

## Run
    pip install -r requirements.txt
    python src/tat_eda_dl.py

## Key findings
- Median TAT: 0.107 s (MacBookPro1) to 1.018 s (Raspberry Pi); all distributions right-skewed and heavy-tailed.
- Slow outliers are bursty (dispersion index 4.1); P(outlier | previous outlier) = 24.6% vs 2.7%.
- MLP regression of log TAT: MAE 0.046 s, R2 0.958 (log) - on par with Ridge; device identity explains most of it.
- Slow-task classifier is near chance (AUC 0.50): individual spikes are not predictable from history.

## Dataset
Execution Times for Image Recognition Task Offloading in Mobile Edge Computing (supplied with the assignment; included in `data/`).
