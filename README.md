# DataScience_StatX

A group project for **CSE558: Data Science — Monsoon 2026**.

## Overview

This project explores statistical inference through data analysis and hypothesis testing. It follows a data science workflow that includes data cleaning, categorical encoding, exploratory data analysis (EDA), and statistical hypothesis tests.

The repository contains Python scripts and a Jupyter notebook used to prepare data, generate visualizations, and conduct statistical analyses.

## Repository Structure

```text
DataScience_StatX/
├── preprocessingData/
│   ├── 02_cleaning.py
│   └── 03_encoding.py
│
├── EDA_Plots/
│   ├── eda_plots.py
│   └── eda_plots/
│       ├── F1_default_by_score.png
│       ├── F2_default_by_ltv.png
│       ├── F3_default_by_borrower.png
│       ├── F4_default_by_age.png
│       └── F5_loan_volume_by_days_to_month_end.png
│
├── Hypothesis Tests/
│   ├── Test_1.py
│   ├── Test_2.py
│   ├── Test_3.py
│   └── Test_4.py
│   └── Test_5.py
│
└── README.md
```

### Directory Description

- **`preprocessingData/`** — Contains scripts for cleaning the data and encoding categorical variables.
- **`EDA_Plots/`** — Contains the script for generating exploratory data analysis visualizations and the resulting plots.
- **`Hypothesis Tests/`** — Contains Python scripts and a Jupyter notebook for statistical hypothesis testing.
- **`README.md`** — Provides an overview of the project and its repository structure.

## Getting Started

### Prerequisites

The project uses Python and the following libraries across its analysis scripts:

- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn


### Setup

Clone the repository:

```bash
git clone https://github.com/Krish-agg/DataScience_StatX.git
cd DataScience_StatX
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

### Running the Scripts

Run the relevant Python scripts from their respective directories. Some scripts expect input CSV files to be available at specific paths, so ensure the required files are present before execution.

For example, the cleaning script can be run from the preprocessing directory:

```bash
cd preprocessingData
python 02_cleaning.py
```

The encoding script follows the cleaning step and uses the cleaned CSV produced by it.

The EDA and hypothesis-testing scripts have their own input-file requirements. Refer to the individual scripts for their expected paths and execution details.

## Project Status

**Current stage:** Midterm submission.

The repository is under active development, and its contents may be updated as the project progresses.

