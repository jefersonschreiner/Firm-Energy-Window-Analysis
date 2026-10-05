# Firm Energy Window Analysis

A Python pipeline that tests a long historical streamflow record for **non-stationarity** and generates **synthetic daily inflow ensembles** under three different calibration-window strategies. The goal is to study how the choice of historical window affects the inflow scenarios used to estimate a hydropower plant's **firm energy**.

The pipeline works in two stages:

1. **Statistical analysis** of the observed series (Mann-Kendall, Spearman's Rho and recursive Pettitt tests) to detect trends and change points.
2. **Stochastic simulation** (SAMS: Analysis, Modeling and Simulation of Stochastic series) that calibrates ARMA and SARIMA models on different windows of the record and generates ensembles of 500 synthetic series of 50 years each.

The generated files follow the input format of the companion project [Reservoir Operation Dispatch Simulator](https://github.com/), so they can be fed directly into the dispatch simulation.

## Methodologies

All three methodologies use the same stochastic engine and differ only in **which part of the historical record is used for calibration**. The default break year (1968) is the change point detected by the Pettitt test.

| Method | Name | Calibration window | Cycles |
|---|---|---|---|
| **MFB** | Fixed Base Model | One fixed window, from the first year to the break year (1931-1968) | 1 |
| **MOF** | Fixed Origin Model | Fixed start year with a growing end year (1931-1969, 1931-1970, ... up to the last year of the record) | One per year |
| **MJD** | Sliding Window Model | Fixed-length window (same length as the base period) that slides forward one year at a time (1932-1969, 1933-1970, ...) | One per year |

For each window, the model is re-fitted and a new ensemble is generated and saved.

## How the stochastic model works

For every calibration window, SAMS does the following:

1. **Transformation.** Daily flows are converted to logarithms, and the leap day (Feb 29) is discarded so every year has 365 days.
2. **Deseasonalization.** The mean and standard deviation of log-flow are computed for each day of the year, and the series is standardized: `z = (log Q - mean_day) / std_day`.
3. **Model selection.** A grid search fits many candidate models in parallel and keeps the best of each family by **AIC**:
   - **ARMA(p, q)** with p, q in 0..3
   - **SARIMA(p, q)(P, Q)<sub>7</sub>** with p, q in 0..2 and P, Q in 0..1
   
   The family with the lowest AIC is the winner. About 50 candidate fits are evaluated per window.
4. **Generation.** Synthetic standardized series are produced from the winning model's parameters using Gaussian white noise, initialized with the end of the observed series, and rescaled to match the observed standard deviation.
5. **Back-transformation.** The daily seasonality is re-applied and the series is converted back to flow with `exp()`.

Each series is generated with its own seed (`semente_base + i`), so results are **fully reproducible**.

## Statistical analysis

`modulo/estatistica.py` runs on the annual mean flow of the full record and reports:

- Basic statistics (mean, standard deviation, variance)
- **Mann-Kendall** trend test
- **Spearman's Rho** correlation between time and flow
- **Pettitt** change-point test, applied recursively so that every significant break is found (the series is split at each break and the test is repeated on both segments)
- A final conclusion on whether the series is stationary (stationary only if all three tests are non-significant, at the 5% level)

## Project structure

```
.
├── main.py                      # Pipeline orchestrator
├── dados/
│   ├── Banco.csv                # Raw observed flows (input)
│   └── Banco_processado.csv     # Processed series (generated automatically)
├── modulo/
│   ├── banco.py                 # Data loading, cleaning and window extraction
│   ├── estatistica.py           # Mann-Kendall, Spearman and Pettitt tests
│   ├── sams.py                  # Model selection (ARMA/SARIMA) and ensemble generation
│   ├── hymodel.py               # Stochastic generators (AR, ARMA, SARIMA)
│   └── metodologia/
│       ├── mfb.py               # Fixed Base Model
│       ├── mof.py               # Fixed Origin Model
│       └── mjd.py               # Sliding Window Model
└── resultado/                   # Outputs (generated)
    ├── estatisticas_resultado.csv
    ├── mfb/
    ├── mof/
    └── mjd/
```

## Requirements

- Python 3.10+
- NumPy
- pandas
- SciPy
- statsmodels
- pymannkendall

```bash
pip install numpy pandas scipy statsmodels pymannkendall
```

## Input data

Place the raw flow record at `dados/Banco.csv` (`;` separator, `,` decimal), in the wide monthly format commonly exported by Brazilian hydrological databases:

| Data | Vazao01 | Vazao02 | ... | Vazao31 |
|---|---|---|---|---|
| 01/01/1931 | 812,4 | 805,7 | ... | 790,1 |
| 01/02/1931 | 640,2 | 655,8 | ... | |

- The first column holds the first day of each month (`dd/mm/yyyy`).
- Columns `Vazao01` to `Vazao31` hold the daily flow (m³/s) for each day of that month.
- Non-existent days, missing values and Feb 29 are dropped automatically.

On the first run, the raw file is cleaned and converted to a daily series and saved as `dados/Banco_processado.csv`, which is reused on later runs.

## Usage

```bash
python main.py
```

The pipeline runs in four steps: statistical analysis, MFB, MOF and MJD. All four use a single shared pool of worker processes, which parallelizes the ARMA/SARIMA grid search.

### Configuration

Parameters are defined at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `RODAR_ESTATISTICA`, `RODAR_MFB`, `RODAR_MOF`, `RODAR_MJD` | `True` | Enable or disable each step |
| `ANO_INICIO_BASE` | 1931 | First year of the record / stationary base period |
| `ANO_QUEBRA` | 1968 | End of the stationary period (Pettitt break year) |
| `N_ANOS_SIMULACAO` | 50 | Length of each synthetic series, in years |
| `N_SERIES` | 500 | Number of synthetic series per ensemble |
| `SEMENTE_BASE` | 13 | Base random seed |
| `N_WORKERS` | 4 | Number of CPU processes |

> The MOF and MJD methods fit one model per year of the record, so the full pipeline can take **several hours**. Each cycle prints an estimate of the remaining time.

## Output

| File | Content |
|---|---|
| `resultado/estatisticas_resultado.csv` | Results of the statistical tests in long format |
| `resultado/mfb/MFB_1931-1968.csv` | Ensemble from the fixed base window |
| `resultado/mof/MOF_<start>-<end>.csv` | One ensemble per MOF cycle |
| `resultado/mjd/MJD_<start>-<end>.csv` | One ensemble per MJD window |
| `resultado/mof/MOF_log.csv`, `resultado/mjd/MJD_log.csv` | One row per cycle: window, winning model, AIC, BIC, run time |

Each ensemble CSV (`;` separator, `,` decimal) has one row per simulated day and one column per synthetic series:

| ano_simulado | dia_ano | serie_0001 | serie_0002 | ... | serie_0500 |
|---|---|---|---|---|---|
| 1 | 1 | 812,4 | 790,1 | ... | 835,9 |
| 1 | 2 | 805,7 | 801,3 | ... | 828,2 |

## License

See the [LICENSE](LICENSE) file for details.
