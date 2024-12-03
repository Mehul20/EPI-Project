# Bridging the Gap: Enhancing COVID-19 Epidemic Forecasting by Integrating Factors like Vaccination Rates, Mobility, Stringency, Socio-Economic Indicators, and Health Metrics into Time Series Models

## Contributors

- Mehul Rastogi (mehulrastogi at gatech dot edu)
- Akshat Karwa (akshatkarwa at gatech dot edu)

## Setup

```
pip install prophet scikit-learn statsmodels pandas matplotlib sktime scalecast
```

## Package Structure

- All the code files are in the path: `src/`. 
- The raw and cleaned data is in `data/`. 
- All the plots are in `plots/`.

## Data Cleaning

```
cd src
python3 source.py
```

All the data cleaning and pre-processing, changes can be seen in `/data`

## SARIMA and Mobility Graphs

```
cd src
python3 mobility_model.py
```

This will run the SARIMA model, return the metrics and update all the plots in `plots/SARIMA`. This will also generate the mobility graphs in `plots/mobility`.

FAQ - If you're interested - Add a new case `case5` array on line 105 to try out the computation with different sets of regressor/exogenous variables. Also, add that array in the for loop on line 107.

## Vaccination Indicator

```
cd src
python3 vaccination_indicator.py
```

This will generate all the plots in `plots/vaccination`

## Meta Prophet Model

```
cd src
python3 meta_prophet_model.py
```

Simulate the Meta Prophet model. Also, generate graphs and metrics for the same. Plots to be found in `plots/Prophet`.

FAQ - If you're interested - Add a new case `case4` array on line 51 to try out the computation with different sets of regressor/exogenous variables. Also, add that array in the for loop on line 53.

## TBATS Model

```
cd src
python3 tbats_model.py
```

Simulate the TBATS model. Also, generate plots that will be saved in `plots/tbats`.

## SI Model

```
cd src
python3 si_model.py
```

Simulate the SI model after data processing. Also, generate plots that will be saved in `plots/SI_model`.

## RNN and LSTM models

```
cd src
python3 run_RNN_LSTM.py
```

This file runs the RNN and LSTM models and generates plots that are saved in `plots/RNN`. Layer Functions are in `LSTM_RNN.py`. Also generates metrics that are in `src/results.xlsx`.

## Documentation & Reports

The following reports can be found in the `doc/` directory:

- Project Proposal
- Project Milestone
- Final Project Report