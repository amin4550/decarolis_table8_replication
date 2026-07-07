# Replication of Decarolis (2015), Table 8

Student authors: Hasan Amin, Abhishek Arya

This repository replicates Table 8 from:

Decarolis, Francesco. 2015. "Medicare Part D: Are Insurers Gaming the Low Income Subsidy Design?" *American Economic Review* 105(4): 1547-1580.

Paper link: https://doi.org/10.1257/aer.20130903

## Target result

The target result is Table 8, "Growth of the Basic Premium: OLS and 2SLS Regressions." The table estimates the relationship between LIPSA weight concentration (`wLIS4`) and growth in the enrollment-weighted basic premium at the Medicare Part D region-year level.

## Repository structure

```
decarolis_table8_replication/
|-- README.md
|-- AI_DISCLOSURE.md
|-- requirements.txt
|-- code/
|   |-- replicate_table8.py
|-- data/
|   |-- README.md
|   |-- master_data_file.dta
|-- output/
|   |-- table4_panel_c_market_summary.csv
|   |-- table8_key_coefficients.csv
|   |-- table8_replication.csv
|-- report/
|   |-- replication_report.pdf
```

Note: `data/master_data_file.dta` is not included in this GitHub repository or submission ZIP file. It must be downloaded separately and placed in the `data/` folder before running the code.

## Data

The script expects the author replication dataset to be named exactly:

```
master_data_file.dta
```

The file must be placed here:

```
data/master_data_file.dta
```

The dataset can be obtained from the paper's replication package on openICPSR:

```
https://www.openicpsr.org/openicpsr/project/112933/version/V1/view
```

Download the replication package from openICPSR, extract the ZIP file, locate `master_data_file.dta`, and place it in the `data/` folder so that the final path is:

```
data/master_data_file.dta
```

The original author replication materials include the master dataset, variable list, main Stata do-file, and supporting Stata do-files. This Python replication only requires `master_data_file.dta` to reproduce Table 8.

The dataset is not included in this GitHub repository or submission ZIP file because it is obtained through the official openICPSR replication package. The `data/README.md` file is included so the grader knows where to place the dataset before running the code.

## How to run

For first-time users, from the repository root, run the following commands in order:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code/replicate_table8.py
```

For someone running it again:

```
source .venv/bin/activate
python code/replicate_table8.py
```

The script performs the full replication workflow: it loads the data, recreates the region-year dataset, estimates the OLS and 2SLS specifications, and writes the output tables.

No notebook is required.

## Python packages

See `requirements.txt`. The code was tested with Python 3.11 and uses:

- pandas
- numpy
- statsmodels
- scipy

## Expected outputs

The script saves all table outputs to:

```
output/
```

The expected output files are:

### `table8_replication.csv`

Full replicated Table 8 output in CSV format. This includes the OLS and 2SLS specifications, coefficients, clustered standard errors, region time-trend indicators, R-squared values, and observation counts.

### `table8_key_coefficients.csv`

A simplified output file containing the key coefficients on `wLIS4`, the main explanatory variable of interest.

### `table4_panel_c_market_summary.csv`

A market-level summary-statistics check for the constructed region-year dataset. This is not the main replicated result, but it helps verify that the market-level data construction is working correctly.

## Main replicated coefficients

The main output is `output/table8_replication.csv`. The key replicated coefficients on `wLIS4` are:

| Panel | Spec 1 | Spec 2 | Spec 3 | Spec 4 | Spec 5 | Spec 6 |
|---|---:|---:|---:|---:|---:|---:|
| OLS | 0.274 | 0.301 | 0.276 | 0.302 | 0.304 | 0.352 |
| 2SLS | 0.544 | 0.616 | 0.689 | 0.632 | 0.750 | 0.702 |

These match Decarolis Table 8 up to rounding for the coefficient of interest and its clustered standard errors.

## Known limitations

This is a Python replication of the original Stata workflow. The main coefficients and clustered standard errors match the paper. Some constant terms in specifications with region-specific time trends may differ because Python and Stata can choose different omitted variables when fixed effects and region-specific trends create collinearity. This does not affect the coefficient of interest, `wLIS4`.

This replication uses the author-provided analysis dataset from the official openICPSR replication package rather than reconstructing the full dataset from the original CMS, IPUMS, formulary, and Q1Medicare source files. Therefore, the replication verifies the Table 8 analysis from the provided dataset, but it does not independently reproduce the entire raw-data construction process.