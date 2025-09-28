# coastal-ocean-utils

A collection of **Python, Fortran, and Bash scripts** for workflows in **coastal ocean modeling**.  
These utilities mainly support **SCHISM** and **WW3 (WaveWatch III)** model setup and testing, as well as **observational data analysis** and **integration with large-scale/global models**.  

## Features
- **Model Setup & Testing**
  - Grid preparation and preprocessing for SCHISM
  - Input generation and configuration support for WW3
  - Automated test runs and validation scripts
- **Observational Data**
  - Tools to download and organize in-situ and satellite datasets
  - Scripts for time series extraction, interpolation, and QC
- **Global Model Integration**
  - Processing outputs from large-scale models (e.g., CMEMS, CFSv2, HYCOM)
  - Building open boundary conditions and nudging fields for SCHISM and WW3
- **Validation & Analysis**
  - Compare SCHISM and WW3 outputs with observations and reanalyses
  - Metrics: RMSE, bias, correlation, skill scores
- **Automation & HPC**
  - Bash scripts for job submission on HPC clusters
  - Workflow automation for large-scale experiments
