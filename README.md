# ⚡ Nordic Energy Market Optimisation & AI Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pyomo](https://img.shields.io/badge/Pyomo-MILP%20Modeling-orange)](http://www.pyomo.org)
[![HiGHS](https://img.shields.io/badge/Solver-HiGHS-green)](https://highs.dev)
[![ML](https://img.shields.io/badge/Forecasting-XGBoost%20%7C%20ANN%20%7C%20MLR-yellow)](https://xgboost.readthedocs.io)
[![Compliance](https://img.shields.io/badge/EU%20AI%20Act-Articles%2013%20%26%2014%20Compliant-purple)](https://artificialintelligenceact.eu)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Strategic Mixed-Integer Linear Program (MILP) Dispatch Optimiser & Machine Learning Forecasting Engine for an Electric Bus Fleet Depot + Battery Energy Storage System (BESS) across Nordic Power Markets (Nord Pool Day-Ahead & Svenska kraftnät Reserves).**

---

## 🌐 Live Streamlit Cloud Deployment

Deploy this platform to the **Streamlit Community Cloud (Free Tier)** in under 2 minutes:

1. **Fork or Push** this repository to your GitHub account (e.g. `your-username/nordic-energy-optimizer`).
2. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**, select:
   * **Repository**: `your-username/nordic-energy-optimizer`
   * **Branch**: `main`
   * **Main file path**: `dashboard.py`
4. Click **"Deploy!"** — Streamlit automatically installs dependencies from `requirements.txt` and compiles the app.

---

## 📑 Table of Contents
* [1. Executive Overview](#1-executive-overview)
* [2. System Architecture](#2-system-architecture)
* [3. Value Stack & Multi-Market Co-Optimisation](#3-value-stack--multi-market-co-optimisation)
* [4. Machine Learning Forecasting (MLR vs. XGBoost vs. ANN)](#4-machine-learning-forecasting-mlr-vs-xgboost-vs-ann)
* [5. Mathematical Formulation (MILP)](#5-mathematical-formulation-milp)
* [6. The "Price of Forecast Error" (Economic Regret)](#6-the-price-of-forecast-error-economic-regret)
* [7. EU AI Act Governance & Compliance](#7-eu-ai-act-governance--compliance)
* [8. Project File Structure](#8-project-file-structure)
* [9. Quick Start Guide](#9-quick-start-guide)
* [10. Key Parameters to Tweak](#10-key-parameters-to-tweak)

---

## 1. Executive Overview

Municipal transit authorities are transitioning bus fleets to battery-electric vehicles (BEVs). However, charging 40 electric buses (350 kWh each) simultaneously can create dangerous and expensive power spikes exceeding **700 kW**, incurring punitive monthly peak tariffs and exposing the depot to volatile hourly electricity spot prices.

This platform implements the **Upper Strategic Level** described in the *Energy Market Optimisation Lead* specification:
* **Asset Co-Optimization**: Simultaneously co-optimizes stationary storage (2 MWh BESS), rooftop solar (500 kWp PV), bus charging schedules, and grid interconnection limits under a single mathematical objective.
* **Multi-Market Revenue Stacking**: Behind-the-meter services (peak shaving, PV self-consumption, energy arbitrage) stacked together with wholesale day-ahead trading (Nord Pool Elspot) and ancillary/balancing services (SvK FCR-N, FCR-D up/down, aFRR).
* **Machine Learning Price Forecasting**: Evaluates and benchmarks Multiple Linear Regression (MLR), Extreme Gradient Boosted Trees (XGBoost), and Deep Multi-Layer Perceptrons (ANN) across non-linear market drivers.
* **The "Price of Forecast Error"**: Solves the downstream MILP dispatch model under each model's forecast and settles against realized market prices to quantify economic regret.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        UPPER STRATEGIC LEVEL (This Simulation)                          │
│                                                                                        │
│  1. Market & Weather Feature Dataset (ENTSO-E, SMHI, SvK):                             │
│     Temperature (°C), Wind Speed (m/s), Nuclear Outages (MW), Autoregressive Lags      │
│                                                                                        │
│  2. Forecasting Engine (simulation/forecasting.py):                                    │
│     • MLR: White-box econometric benchmark (Ridge closed-form)                         │
│     • XGBoost: Non-linear threshold step-function ensemble (TreeSHAP audit trail)     │
│     • ANN: Multi-layer perceptron neural network (Continuous smooth activations)       │
│                                                                                        │
│  3. Strategic MILP Dispatch Optimiser (simulation/optimizer.py):                       │
│     Formulated in Pyomo and solved to global optimality with HiGHS                     │
│                                                                                        │
│  Output: "Dynamic Power Envelope" (P_grid, P_bess, P_bus, R_res per hour)             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Dispatched at fixed cadence
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LOWER EXECUTION LEVEL (The Edge)                              │
│                                                                                        │
│  • Edge Controllers: Translates envelope into individual physical charger commands.    │
│  • Battery Management System (BMS): Cell-level voltage, thermal & SoC monitoring.       │
│  • Edge Safety Layer (ABSOLUTE VETO): Holds physical priority over any market signal.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Value Stack & Multi-Market Co-Optimisation

The platform unlocks 6 simultaneous revenue and cost-saving streams:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             THE FULL VALUE STACK                                    │
├───────────────────────────────┬─────────────────────────────────────────────────────┤
│ 1. Peak Shaving               │ Suppresses maximum monthly grid import (EUR/kW/mo)  │
│ 2. Energy Arbitrage           │ Charges BESS during cheap/negative hours, discharges│
│                               │ during peak morning/evening demand                  │
│ 3. FCR-N Reserve Revenue      │ Symmetric frequency containment capacity payments   │
│ 4. FCR-D Up / Down Revenue    │ Fast disturbance frequency containment reserves    │
│ 5. aFRR Capacity Revenue      │ Automated frequency restoration reserve payments   │
│ 6. PV Self-Consumption        │ Avoids grid energy purchases, taxes, and fees       │
│ [-] BESS Degradation Cost     │ Economic hurdle rate accounting for battery aging   │
└───────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 4. Machine Learning Forecasting (MLR vs. XGBoost vs. ANN)

Electricity prices are governed by non-linear physical supply-demand equilibrium. The platform simulates and compares three fundamental model paradigms:

```
                               MODEL RESPONSE BEHAVIORS
┌───────────────────────┬──────────────────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
│ Characteristic        │ MLR (Multiple Linear Regression) │ XGBoost (Gradient Boosted Trees) │ ANN (Multi-Layer Perceptron)     │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Formulation           │ Hyper-plane: y = β₀ + ∑ βᵢ xᵢ    │ Tree ensemble: y = ∑ f_k(x)      │ Deep layers: y = σ(W₂σ(W₁x+b₁)+b₂)│
│ Temperature Response  │ ❌ Rigid linear negative slope   │ ⭐ Non-linear "hockey stick"     │ ⭐ Smooth convex sigmoid         │
│ (-20°C to +25°C)      │ Misses cold-snap surges by >€40  │ Step-jumps below 0°C (freezing)  │ Progressive heating acceleration │
│ Wind Merit-Order      │ ❌ Unbounded downward line       │ ⭐ Cubic curve saturation        │ ⭐ Smooth S-curve saturation     │
│ (0 to 25 m/s)         │ Plunges into unrealistic -€20+   │ Flat at cut-in, curtailment floor│ Captures turbine power curve P∝v³│
│ Test Set RMSE / R²    │ 7.74 EUR/MWh (R² = 0.751)        │ 4.88 EUR/MWh (R² = 0.901) ⭐     │ 5.68 EUR/MWh (R² = 0.866)        │
│ EU AI Act Compliance  │ ⭐⭐⭐⭐⭐ 100% White-Box         │ ⭐⭐⭐⭐ TreeSHAP attribution     │ ⭐⭐ Black-Box (Higher Risk)     │
└───────────────────────┴──────────────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

---

## 5. Mathematical Formulation (MILP)

### Sets & Decision Variables
* $t \in \{0, \dots, T-1\}$: Discrete hourly timesteps.
* $P_{\text{imp}}[t], P_{\text{exp}}[t] \ge 0$: Grid import and export power (kW).
* $P_{\text{ch}}[t], P_{\text{dis}}[t] \ge 0$: BESS charge and discharge power (kW).
* $\text{SoC}[t] \ge 0$: BESS state of charge (kWh).
* $R_n[t], R_{du}[t], R_{dd}[t], R_a[t] \ge 0$: Reserve capacity bids (kW).
* $P_{\text{peak}} \ge 0$: Monthly peak demand tracker (kW).
* $\delta[t] \in \{0, 1\}$: Binary variable enforcing mutual exclusion between charging and discharging.

### The Objective Function
$$\min \sum_{t=0}^{T-1} \Bigg[ \frac{\lambda_{\text{spot}}[t]}{1000} (P_{\text{imp}}[t] - P_{\text{exp}}[t]) + (c_{\text{grid}} + c_{\text{tax}}) P_{\text{imp}}[t] - \sum_{\text{res}} \frac{\lambda_{\text{res}}[t]}{1000} R_{\text{res}}[t] + c_{\text{deg}} (P_{\text{ch}}[t] + P_{\text{dis}}[t]) \Bigg] \Delta t + P_{\text{peak}} \cdot c_{\text{peak}} \cdot \left(\frac{\text{days}}{30}\right)$$

### Core Constraints
1. **Depot Energy Balance**:
   $$P_{\text{imp}}[t] + P_{\text{dis}}[t] + G_{\text{pv}}[t] = D_{\text{bus}}[t] + P_{\text{ch}}[t] + P_{\text{exp}}[t] \quad \forall t$$
2. **Battery SoC Storage Dynamics**:
   $$\text{SoC}[t] = \text{SoC}[t-1] + \left( \eta_c \cdot P_{\text{ch}}[t] - \frac{P_{\text{dis}}[t]}{\eta_d} \right) \Delta t, \quad \text{SoC}_{\min} \le \text{SoC}[t] \le \text{SoC}_{\max}$$
3. **Mutual Exclusion (Big-M Formulation)**:
   $$P_{\text{ch}}[t] \le P_{\text{bess}} \cdot \delta[t], \quad P_{\text{dis}}[t] \le P_{\text{bess}} \cdot (1 - \delta[t])$$
4. **Monthly Peak Demand Tracking**:
   $$P_{\text{peak}} \ge P_{\text{imp}}[t] \quad \forall t$$
5. **Reserve Power & Energy Headroom**:
   $$P_{\text{dis}}[t] + R_n[t] + R_{du}[t] + R_a[t] \le P_{\text{bess}}$$
   $$P_{\text{ch}}[t] + R_n[t] + R_{dd}[t] \le P_{\text{bess}}$$
   $$\text{SoC}_{\min} + (R_n[t] + R_{du}[t] + R_a[t]) \Delta t \le \text{SoC}[t] \le \text{SoC}_{\max} - (R_n[t] + R_{dd}[t]) \Delta t$$

---

## 6. The "Price of Forecast Error" (Economic Regret)

In power market operations, models are **never evaluated solely by statistical RMSE**. If a model under-forecasts an evening price peak, the optimizer fails to charge the BESS overnight, leaving the battery empty during the spike. The depot is forced to purchase peak energy at astronomical prices, simultaneously triggering a devastating new monthly peak tariff penalty.

We simulate a **two-stage market settlement**:
1. Commit the dispatch schedule using the model's day-ahead forecast $\hat{\mathbf{p}}$.
2. Settle the cash flows under **ground truth actual spot prices** $\mathbf{p}_{\text{actual}}$.

```
┌──────────────────────────────────┬─────────────────┬──────────────────┬─────────────────┐
│ Strategy                         │ Realized Cost   │ Regret vs Oracle │ Peak Import     │
├──────────────────────────────────┼─────────────────┼──────────────────┼─────────────────┤
│ Oracle (Perfect Foresight)       │ -€113.56 / day  │ €0.00 / day      │ 319 kW          │
│ XGBoost Forecast                 │ -€105.30 / day  │ €8.26 / day      │ 325 kW          │
│ ANN Forecast                     │ -€101.85 / day  │ €11.71 / day     │ 338 kW          │
│ MLR (Linear) Forecast            │ -€96.42 / day   │ €17.14 / day     │ 354 kW          │
│ Unoptimised Baseline (Dumb Chg)  │ +€522.75 / day  │ €636.31 / day    │ 693 kW          │
└──────────────────────────────────┴─────────────────┴──────────────────┴─────────────────┘
```

---

## 7. EU AI Act Governance & Compliance

Because this system influences bids submitted to Nord Pool and SvK, it is governed by EU AI Act requirements:
* **Article 13 (Transparency & Auditability)**: Models cannot be unexplainable black boxes. XGBoost is audited through exact **TreeSHAP** marginal attributions, while MLR provides an open white-box baseline.
* **Article 14 (Human-in-the-Loop & Safety Veto)**: The strategic power envelope is subject to immediate physical veto by the local edge safety layer.
* **Data Residency & NIS2**: Designed for deployment in `europe-north2` (Sweden/Finland) with zero data egress outside the EU.

---

## 8. Project File Structure

```
ENERGY/
├── .streamlit/
│   └── config.toml               # Streamlit theme & headless server config
├── simulation/
│   ├── __init__.py
│   ├── market.py                 # Nord Pool (Elspot) + SvK reserve simulators
│   ├── depot.py                  # 40-bus EV fleet scheduling & PV irradiance model
│   ├── optimizer.py              # Pyomo MILP dispatch optimiser (HiGHS solver)
│   ├── analysis.py               # Value stack waterfall & backtesting engine
│   └── forecasting.py            # MLR, XGBoost, ANN, response curves & regret
├── config.py                     # Strongly-typed system dataclasses
├── dashboard.py                  # Streamlit interactive dashboard (7 tabs)
├── run_cli.py                    # Terminal-based quick runner
├── GUIDED_PROJECT_GUIDE.md       # Comprehensive textbook guide for interview prep
├── requirements.txt              # Production dependency specifications
├── .gitignore                    # Git cache and local environment exclusions
└── README.md                     # Platform documentation
```

---

## 9. Quick Start Guide

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/nordic-energy-optimizer.git
cd nordic-energy-optimizer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run CLI Simulation
python run_cli.py

# 5. Launch Interactive Dashboard
streamlit run dashboard.py
```

---

## 10. Key Parameters to Tweak

| Parameter | Default | Physical & Financial Impact |
|---|---|---|
| **BESS Capacity** | 2,000 kWh | Demonstrates diminishing marginal returns above ~3,000 kWh |
| **BESS Max Power** | 1,000 kW | Higher C-rates unlock higher ancillary reserve bids |
| **Peak Tariff** | 8.5 €/kW/mo | Raising to €20/kW forces the battery into aggressive peak shaving |
| **BESS Degradation** | 2.0 €¢/kWh | Acts as an economic hurdle rate against low-margin arbitrage |
| **FCR-N Premium** | 1.8× | Determines whether the battery prioritizes reserves over arbitrage |
| **Bidding Zone** | SE3 | SE4 = import-dependent premium, SE1/SE2 = hydro surplus discount |
| **Price Signal** | Oracle | Toggle between Actual Spot, XGBoost, MLR, and ANN forecasts |

---

## 📚 Guided Project Guide

For an in-depth textbook breakdown covering feature engineering, mathematical proofs, Big-M derivations, and scripted interview answers, refer to:  
👉 [`GUIDED_PROJECT_GUIDE.md`](GUIDED_PROJECT_GUIDE.md)

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
