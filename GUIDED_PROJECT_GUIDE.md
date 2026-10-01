# ⚡ Energy Market Optimisation & AI Forecasting: The Definitive Guided Project Guide

> **Target Audience**: Beginners to Applied AI & Operations Research, and Quantitative Energy Analysts preparing for the **Energy Market Optimisation Lead** interview.  
> **Philosophy**: Zero buzzwords without definitions. Every equation, slider, parameter, and ML model is explained from first principles with real-world energy economics.

---

## 📑 Table of Contents
1. [The Big Picture: What Problem Are We Solving?](#1-the-big-picture-what-problem-are-we-solving)
2. [Master Glossary: Energy, Optimization & AI Terms](#2-master-glossary-energy-optimization--ai-terms)
3. [The Data: Features, Ground Truth & Physical Market Dynamics](#3-the-data-features-ground-truth--physical-market-dynamics)
4. [How Machine Learning Forecasting Works (MLR vs. XGBoost vs. ANN)](#4-how-machine-learning-forecasting-works-mlr-vs-xgboost-vs-ann)
5. [The Optimization Engine: Pyomo, HiGHS & Linear Programming](#5-the-optimization-engine-pyomo-highs--linear-programming)
6. [Mathematical Formulation: How the Equations Change When You Tweak Parameters](#6-mathematical-formulation-how-the-equations-change-when-you-tweak-parameters)
7. [The "Price of Forecast Error": Linking AI to Financial Value](#7-the-price-of-forecast-error-linking-ai-to-financial-value)
8. [EU AI Act & Regulatory Governance (Articles 13 & 14)](#8-eu-ai-act--regulatory-governance-articles-13--14)
9. [Hands-On Experimentation & Interview Prep Walkthrough](#9-hands-on-experimentation--interview-prep-walkthrough)

---

## 1. The Big Picture: What Problem Are We Solving?

Imagine a municipal bus depot in Stockholm, Sweden:
* You operate **40 electric buses**, each with a heavy 350 kWh battery pack.
* When buses return from their morning and evening routes, they must recharge so they are ready for the next shift. If a bus isn't charged in the morning, city commuters are stranded. **Bus charging schedules are sacred.**
* However, plugging 40 buses directly into the grid simultaneously creates an enormous electrical spike (over **700 kW**). 
* In Sweden and across the Nordics, electricity costs are twofold:
  1. **Spot Price (EUR/MWh)**: Fluctuates every single hour based on wind, nuclear, hydro, and demand. Charging at 18:00 (dinner peak) can cost 5× more than charging at 03:00 (night baseload).
  2. **Peak Demand Tariff (EUR/kW/month)**: A punitive penalty based on the highest power spike your depot draws from the grid during the entire month.
* To solve this, the depot installs:
  * A **Rooftop Solar PV Array** (500 kWp): Generates free clean energy when the sun shines.
  * A **BESS (Battery Energy Storage System)** (2,000 kWh / 1,000 kW): A massive stationary battery container parked at the depot.

### The Two-Level Architecture
In real industrial energy systems, this is organized into two distinct layers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        UPPER STRATEGIC LEVEL (This Simulation)                          │
│                                                                                        │
│  1. Machine Learning Forecasters (MLR, XGBoost, ANN):                                  │
│     Predict tomorrow's hourly spot prices, reserve values, bus demand, and solar PV.   │
│                                                                                        │
│  2. Mathematical Optimiser (Pyomo + HiGHS MILP):                                       │
│     Builds the optimal 24-hour master plan:                                            │
│     - When should the BESS charge from the grid (cheap hours)?                         │
│     - When should the BESS discharge to power buses or export to grid (peak hours)?    │
│     - How much battery capacity to commit to SvK balancing reserves (FCR-N, aFRR)?     │
│                                                                                        │
│  Output: "Dynamic Power Envelope" (P_grid, P_bess, P_bus per hour)                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Dispatched every hour / 15 minutes
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LOWER EXECUTION LEVEL (The Edge)                              │
│                                                                                        │
│  • Edge Controllers: Distribute power across individual physical chargers (CCS-2).     │
│  • Battery Management System (BMS): Monitors cell voltages and thermal limits.         │
│  • Edge Safety Layer (ABSOLUTE VETO): If the optimiser recommends 1,200 kW but the     │
│    transformer is overheating, the edge safety layer vetoes the command immediately.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Master Glossary: Energy, Optimization & AI Terms

### Energy Market Concepts
* **Nord Pool**: The European power exchange operating across the Nordic and Baltic countries.
* **Day-Ahead Spot Market (Elspot)**: The auction that closes every day at 12:00 CET for power delivery during each hour of the following day. Prices clear at the intersection of supply and demand curves.
* **Bidding Zones (SE1, SE2, SE3, SE4)**: Sweden is partitioned into 4 price zones. SE1/SE2 (North) have massive hydropower surpluses and cheap power. SE3 (Stockholm) and SE4 (Malmö) are demand-heavy with transmission bottlenecks, leading to higher prices and volatility.
* **Ancillary / Balancing Services (SvK)**: Svenska kraftnät (the Swedish Transmission System Operator) pays battery owners to keep the grid frequency at exactly 50.0 Hz:
  * **FCR-N (Frequency Containment Reserve - Normal)**: Symmetric reserve operating within 49.9–50.1 Hz. The battery must be ready to either charge (if frequency is high) or discharge (if frequency is low). Highly lucrative capacity payments (EUR/MW/h).
  * **FCR-D (Disturbance Up/Down)**: Activated during sudden generator trips (under 49.9 Hz) or wind surges (above 50.1 Hz). Rapid 5-second response.
  * **aFRR / mFRR (Restoration Reserves)**: Secondary and manual reserves used to restore system frequency over 2 to 15-minute horizons.
* **Energy Arbitrage**: Buying power when it is cheap (e.g. at night or during wind surpluses at -€5 to €20/MWh), storing it in the BESS, and discharging it when prices spike (€100+/MWh).
* **Peak Shaving**: Discharging the stationary battery during hours when bus charging demand is high to suppress the depot's maximum grid import, directly reducing monthly grid capacity charges.

### BESS & Battery Physics
* **Capacity (kWh or MWh)**: The total amount of energy the battery container holds (e.g., 2,000 kWh = 2 MWh).
* **Max Power (kW or MW)**: The maximum speed at which energy can flow in or out (e.g., 1,000 kW). A 2,000 kWh battery with 1,000 kW power has a **C-rate of 0.5C** (takes 2 hours to fully charge).
* **State of Charge (SoC)**: The fuel gauge of the battery, expressed as a fraction or percentage (0.0 to 1.0, or 0% to 100%).
* **Round-Trip Efficiency ($\eta_{\text{RTE}}$)**: The fraction of energy recovered after charging and discharging. If $\eta_{\text{RTE}} = 90\%$, then one-way efficiency $\eta_c = \eta_d = \sqrt{0.90} \approx 0.9487$. For every 100 kWh put in, ~90 kWh comes back out; 10 kWh is lost as heat.
* **Degradation Cost ($c_{\text{deg}}$)**: Lithium-ion cells degrade with each charge/discharge cycle. If a 2,000 kWh battery costs €300,000 and lasts 4,000 full cycles, cycling 1 kWh incurs ~€0.02 of battery wear. The optimiser accounts for this so it does not cycle the battery for a trivial €0.01 price difference.

### Machine Learning & AI Terms
* **Feature ($X$)**: An input variable given to a model to make a prediction (e.g., temperature, wind speed, hour of day).
* **Target ($y$)**: The ground-truth value the model is trying to predict (e.g., actual Nord Pool spot price).
* **Inductive Bias**: The underlying structural assumptions built into a model architecture. (e.g., Linear Regression assumes all relationships are straight lines; Decision Trees assume relationships are step-wise threshold boxes).
* **Out-of-Sample / Test Set**: Data the model has never seen during training, used to prove real-world predictive validity.
* **Residual ($y - \hat{y}$)**: The error between the actual price ($y$) and the model's prediction ($\hat{y}$).
* **MAE (Mean Absolute Error)**: Average absolute magnitude of errors in EUR/MWh. $\frac{1}{N}\sum |y - \hat{y}|$.
* **RMSE (Root Mean Squared Error)**: Standard deviation of residuals. Heavily penalizes large errors because errors are squared before averaging. $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$.
* **$R^2$ (Coefficient of Determination)**: The percentage of price variation explained by the model (1.0 = perfect forecast, 0.0 = no better than predicting the historical average).

### Optimization & Operations Research Terms
* **MILP (Mixed-Integer Linear Programming)**: A mathematical optimization problem where the objective and constraints are linear, but some variables are restricted to integers or binary values (0 or 1).
* **Decision Variables**: The knobs the optimizer turns (e.g., how many kW to charge, how many kW to bid into FCR-N).
* **Parameters**: Given constants the optimizer cannot change (e.g., tomorrow's electricity prices, bus arrival times, battery size).
* **Constraints**: Hard physical rules that cannot be violated (e.g., BESS cannot exceed 100% SoC; grid import cannot exceed 3,000 kW).
* **Objective Function**: The single mathematical formula the solver minimizes (Total Cost) or maximizes (Total Profit).
* **Solver (HiGHS)**: The high-performance C++ software algorithm that navigates the multi-dimensional feasible region using the Simplex algorithm and Branch-and-Bound to guarantee the globally optimal mathematical solution.
* **Big-M Method**: A mathematical formulation technique using a large constant $M$ and a binary variable $\delta \in \{0, 1\}$ to enforce logical conditions (e.g., "either charge OR discharge, but never both simultaneously").

---

## 3. The Data: Features, Ground Truth & Physical Market Dynamics

Electricity spot prices are not random walks. They are governed by the **laws of physics and merit-order economics**. In [`simulation/forecasting.py`](file:///c:/Users/imac/Desktop/ENERGY/simulation/forecasting.py), the dataset generator models 9 key physical features:

```
┌────────────────────┬───────────┬───────────────────────────────────────────────────────────────┐
│ Feature Name       │ Unit      │ Physical & Market Mechanism                                   │
├────────────────────┼───────────┼───────────────────────────────────────────────────────────────┤
│ hour               │ 0–23      │ Daily diurnal curve: industry starts at 07:00, dinner at 18:00│
│ day_of_week        │ 0–6       │ Working days vs weekend: factory demand drops ~15–20%         │
│ is_weekend         │ 0 or 1    │ Binary indicator for industrial shutdown                      │
│ temperature        │ °C        │ Nordic degree-day electric heating surge below 0°C            │
│ wind_speed         │ m/s       │ Cubic power curve P ∝ v³ driving renewable merit-order crash  │
│ hydro_inflow       │ 0.0–1.0   │ Reservoir storage dampening peak prices in summer/autumn      │
│ nuclear_outage_mw  │ MW        │ Baseload supply deficit causing severe price spikes           │
│ lag_24h_price      │ EUR/MWh   │ Yesterday's price at the exact same hour (persistence)        │
│ lag_168h_price     │ EUR/MWh   │ Same hour last week (captures recurring weekly schedules)     │
└────────────────────┴───────────┴───────────────────────────────────────────────────────────────┘
```

### The Non-Linear Physics of Electricity Price Formation

#### 1. The Temperature "Hockey Stick"
* When temperature drops from 20°C down to 10°C, electricity demand is relatively flat.
* When temperature drops **below 0°C** into severe Nordic freezing (-5°C, -15°C, -20°C):
  1. Air-source heat pumps lose their Coefficient of Performance (COP).
  2. Auxiliary direct resistive electric heating engages.
  3. Space heating demand skyrockets non-linearly.
* **Mathematical expression**:
  $$\text{Surge} = (10 - T) \cdot 0.8 + \mathbf{1}_{\{T < 0\}} \cdot (0 - T)^{1.6} \cdot 1.2$$

#### 2. The Wind Merit-Order Cannibalization Effect
* A wind turbine produces zero power below cut-in speed (~3 m/s).
* Between 3 m/s and 12 m/s, power output increases **cubically**:
  $$P_{\text{wind}} \propto v^3$$
* Between 12 m/s and 25 m/s, the turbine reaches rated capacity and plateaus.
* In power markets, wind has zero fuel cost (zero marginal cost). When wind is strong, wind generators bid at 0 EUR/MWh (or negative prices). This pushes expensive fossil-gas and coal peaker plants out of the clearing stack (the **Merit-Order Effect**), crashing spot prices down to near zero or even negative numbers.

---

## 4. How Machine Learning Forecasting Works (MLR vs. XGBoost vs. ANN)

When an interview asks: *"Why can't you just use Multiple Linear Regression (MLR) or a basic Neural Net?"*, this section gives you the exact technical rationale.

```
                               VISUALIZING INDUCTIVE BIAS
       MLR (Straight Line)          XGBoost (Threshold Steps)            ANN (Smooth Curve)
     Price                         Price                             Price
       │          /                  │         ┌─────                  │            .---
       │        /                    │         │                       │          .-'
       │      /                      │   ┌─────┘                       │       .-'
       │    /                        │   │                             │    .-'
       │  /                          │───┘                             │.--'
       └────────────── Temp          └────────────── Temp              └────────────── Temp
     Misses cold-snap surge!       Captures freezing trigger!        Smooth non-linear curve!
```

---

### Model 1: Multiple Linear Regression (MLR / Ridge)

#### How it works:
MLR fits a flat hyper-plane through the data points using the equation:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$

In matrix form:
$$\hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\beta}$$

To prevent individual coefficients from exploding when features are correlated (like temperature and 24h lagged prices), we apply **Ridge Regularization ($L_2$ penalty)**:
$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2$$

This has a closed-form analytical solution:
$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

#### Why it responds the way it does:
* **Temperature**: Because $\beta_{\text{temp}} \approx -1.8$, MLR assumes that every 1°C temperature drop causes the exact same linear €1.8/MWh price increase, whether going from 20°C to 19°C, or from -14°C to -15°C.
* **The Failure**: During an Arctic Cold Snap (-18°C), MLR predicts **€83/MWh**, while the actual market spikes to **€125+/MWh**. It severely underpredicts tail spikes!
* **Wind Speed**: When wind reaches 22 m/s, MLR continues downward in a straight line, predicting an unrealistic **-€20.7/MWh** because it lacks a saturation ceiling.
* **Why keep it?**: It is 100% white-box, requires no GPU/iterations, and satisfies EU AI Act explainability instantly.

---

### Model 2: XGBoost (Extreme Gradient Boosted Decision Trees)

#### How it works:
XGBoost does not use straight lines. It builds an **ensemble of shallow decision trees** sequentially:
$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

Each individual tree $f_k$ consists of split questions:
```
                [ Temperature < -2.5°C ? ]
                       /        \
                    YES          NO
                    /              \
        [ Wind < 4.0 m/s ? ]      [ Hour ∈ {17,18,19} ? ]
            /          \             /             \
        +€45.20      +€12.10      +€18.40         -€4.10
```

#### How it trains (Gradient Boosting with 2nd-Order Taylor Expansion):
Unlike standard gradient descent, XGBoost calculates the first derivative ($g_i$, gradient) and second derivative ($h_i$, hessian) of the loss function for each data point:
$$g_i = \frac{\partial L(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}} = (\hat{y}^{(t-1)} - y_i)$$
$$h_i = \frac{\partial^2 L(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2} = 1 \quad (\text{for MSE loss})$$

The optimal weight $w_j^*$ for a tree leaf is:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

#### Why it wins on Tabular Energy Markets ($R^2 = 0.901$):
* **Piecewise Step-Function Response**: Electricity markets operate on discrete trigger events (e.g. "Did temperature fall below freezing?", "Did a 1,400 MW reactor trip?", "Is it peak hour 18:00?").
* **Regime Isolation**: XGBoost routes cold snap hours with low wind into specialized leaves with large price markups without distorting normal mild days.
* **Wind Saturation**: Once wind exceeds 14 m/s, the splits recognize that turbines are at rated power; predictions flatten into a sensible curtailment floor (-€6.5/MWh).

---

### Model 3: Artificial Neural Network (ANN / Multi-Layer Perceptron)

#### How it works:
The ANN projects input features through stacked layers of artificial neurons:
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{y} = \mathbf{W}_3 \mathbf{h}_2 + b_3$$

Where $\text{ReLU}(z) = \max(0, z)$ is the non-linear activation function.

```
   Inputs (x)               Hidden Layer 1 (64)        Hidden Layer 2 (32)        Output (ŷ)
   [Temperature] ───────┐      ┌─────────────┐            ┌─────────────┐
   [Wind Speed]  ───────┼─────>│  Neuron 1   ├───────────>│  Neuron 1   ├─────────> [Spot Price]
   [Hour]        ───────┼─────>│  Neuron 2   ├───────────>│  Neuron 2   │
   [Lags...]     ───────┘      └─────────────┘            └─────────────┘
```

#### Why it responds the way it does:
* **Continuous Smooth Curves**: Because matrix multiplications are smoothed by continuous activations, the ANN generates a graceful, convex exponential response curve for cold temperatures.
* **Complex Cross-Feature Synergy**: The ANN naturally blends temperature, wind, and hour simultaneously.
* **The Production Risk**: Neural networks struggle with out-of-distribution tabular extrapolations. If a geopolitical event or catastrophic cold wave (-30°C) occurs that was never in the training set, an ANN can output wild or unstable numbers, requiring strict safety bounds.

---

## 5. The Optimization Engine: Pyomo, HiGHS & Linear Programming

### AI Forecasting vs. Mathematical Optimization

Many people confuse **Machine Learning** with **Optimization**:
* **Machine Learning** is *descriptive & predictive*: *"Given the weather, what will tomorrow's spot price likely be?"*
* **Mathematical Optimization (MILP)** is *prescriptive*: *"Given that price forecast, physical bus schedules, grid export limits, and battery degradation, what exact kilowatt setpoints should the battery charge and discharge at every hour to minimize total operating costs?"*

```
   ┌───────────────────────┐          ┌───────────────────────┐          ┌───────────────────────┐
   │ Machine Learning (AI) │          │  MILP Optimiser       │          │ Physical Dispatch     │
   │ Predictions:          │ ───────> │  (Pyomo + HiGHS)      │ ───────> │ Setpoints:            │
   │ • Spot Prices (EUR)   │          │  Subject to:          │          │ • BESS: +850 kW (Chg) │
   │ • PV Solar Gen (kW)   │          │  • Battery bounds     │          │ • Grid: 350 kW (Imp)  │
   │ • Bus Demand (kW)     │          │  • Bus arrival times  │          │ • FCR-N: 150 kW (Bid) │
   └───────────────────────┘          └───────────────────────┘          └───────────────────────┘
```

### Pyomo vs. HiGHS: Who Does What?
1. **Pyomo**: An open-source Python library used to **formulate** optimization problems using high-level Python code (sets, parameters, variables, constraints, objective). It translates Python code into a standardized mathematical matrix representation (MPS or LP file).
2. **HiGHS (`highspy`)**: The underlying **solver engine** written in optimized C++. It takes the mathematical matrix and uses advanced linear algebra (Dual Simplex, Interior Point, and Branch-and-Cut) to prove global mathematical optimality in milliseconds.

---

## 6. Mathematical Formulation: How the Equations Change When You Tweak Parameters

In [`simulation/optimizer.py`](file:///c:/Users/imac/Desktop/ENERGY/simulation/optimizer.py), the entire depot operation is converted into a Mixed-Integer Linear Program.

### 1. Decision Variables (What the solver decides for each hour $t \in \{0, \dots, T-1\}$)
* $P_{\text{imp}}[t] \ge 0$: Power imported from the grid (kW)
* $P_{\text{exp}}[t] \ge 0$: Power exported to the grid (kW)
* $P_{\text{ch}}[t] \ge 0$: Power charging into the BESS (kW)
* $P_{\text{dis}}[t] \ge 0$: Power discharging from the BESS (kW)
* $\text{SoC}[t] \ge 0$: State of Charge of the BESS at the end of hour $t$ (kWh)
* $R_n[t] \ge 0$: Capacity bid into FCR-N reserve (kW)
* $R_{du}[t] \ge 0$: Capacity bid into FCR-D Up reserve (kW)
* $R_{dd}[t] \ge 0$: Capacity bid into FCR-D Down reserve (kW)
* $R_a[t] \ge 0$: Capacity bid into aFRR reserve (kW)
* $P_{\text{peak}} \ge 0$: Maximum peak power imported from the grid across the entire month (kW)
* $\delta[t] \in \{0, 1\}$: Binary variable (1 = BESS charging, 0 = BESS discharging)

---

### 2. The Objective Function

The optimizer minimizes **Total Net Operating Cost**:

$$\min \sum_{t=0}^{T-1} \Big[ \text{Energy Purchases} + \text{Grid Fees} + \text{Energy Taxes} - \text{Export Revenue} - \text{Reserve Revenues} + \text{BESS Degradation} \Big] + \text{Peak Demand Charge}$$

$$\begin{aligned}
\min \sum_{t=0}^{T-1} \Bigg[ & \frac{\lambda_{\text{spot}}[t]}{1000} \cdot P_{\text{imp}}[t] \cdot \Delta t \\
& + c_{\text{grid}} \cdot P_{\text{imp}}[t] \cdot \Delta t \\
& + c_{\text{tax}} \cdot P_{\text{imp}}[t] \cdot \Delta t \\
& - \frac{\lambda_{\text{spot}}[t]}{1000} \cdot P_{\text{exp}}[t] \cdot \Delta t \\
& - \frac{\lambda_{\text{fcr\_n}}[t]}{1000} \cdot R_n[t] \cdot \Delta t \\
& - \frac{\lambda_{\text{fcr\_d\_up}}[t]}{1000} \cdot R_{du}[t] \cdot \Delta t \\
& - \frac{\lambda_{\text{fcr\_d\_dn}}[t]}{1000} \cdot R_{dd}[t] \cdot \Delta t \\
& - \frac{\lambda_{\text{afrr}}[t]}{1000} \cdot R_a[t] \cdot \Delta t \\
& + c_{\text{deg}} \cdot (P_{\text{ch}}[t] + P_{\text{dis}}[t]) \cdot \Delta t \Bigg] \\
& + P_{\text{peak}} \cdot c_{\text{peak}} \cdot \left(\frac{\text{days}}{30}\right)
\end{aligned}$$

---

### 3. The Core Constraints (The Physical Laws)

#### Constraint 1: Depot Energy Balance (Kirchhoff's Current Law)
At every single hour, power flowing into the depot busbar must equal power flowing out:
$$P_{\text{imp}}[t] + P_{\text{dis}}[t] + G_{\text{pv}}[t] = D_{\text{bus}}[t] + P_{\text{ch}}[t] + P_{\text{exp}}[t] \quad \forall t$$

#### Constraint 2: Battery State of Charge (SoC) Dynamics
The energy in the battery this hour equals the energy from last hour plus incoming charge minus outgoing discharge, factoring in efficiency losses ($\eta_c, \eta_d$):
$$\text{SoC}[t] = \text{SoC}[t-1] + \left( \eta_c \cdot P_{\text{ch}}[t] - \frac{P_{\text{dis}}[t]}{\eta_d} \right) \cdot \Delta t$$
$$\text{SoC}_{\min} \le \text{SoC}[t] \le \text{SoC}_{\max}$$

#### Constraint 3: Binary Mutual Exclusion (The Big-M Formulation)
A battery cannot physically charge and discharge at the exact same instant. Without binary constraints, linear solvers will exploit round-trip losses to cheat the system. We use binary variable $\delta[t] \in \{0, 1\}$ and big constant $M = P_{\text{bess}}$:
$$P_{\text{ch}}[t] \le P_{\text{bess}} \cdot \delta[t]$$
$$P_{\text{dis}}[t] \le P_{\text{bess}} \cdot (1 - \delta[t])$$
* When $\delta[t] = 1$: $P_{\text{ch}} \le P_{\text{bess}}$ and $P_{\text{dis}} \le 0$ (charging only).
* When $\delta[t] = 0$: $P_{\text{ch}} \le 0$ and $P_{\text{dis}} \le P_{\text{bess}}$ (discharging only).

#### Constraint 4: Monthly Peak Tracking
The peak demand variable $P_{\text{peak}}$ must be at least as large as the highest grid import drawn in any individual hour:
$$P_{\text{peak}} \ge P_{\text{imp}}[t] \quad \forall t$$

#### Constraint 5: Reserve Headroom (The SvK Reliability Rules)
When committing to FCR-N (symmetric frequency control), the battery must reserve both power and energy capacity so it never defaults on a grid call:
* **Power Headroom**:
  $$P_{\text{dis}}[t] + R_n[t] + R_{du}[t] + R_a[t] \le P_{\text{bess}}$$
  $$P_{\text{ch}}[t] + R_n[t] + R_{dd}[t] \le P_{\text{bess}}$$
* **Energy Headroom (SoC Floor and Ceiling)**:
  $$\text{SoC}[t] \ge \text{SoC}_{\min} + (R_n[t] + R_{du}[t] + R_a[t]) \cdot \Delta t$$
  $$\text{SoC}[t] \le \text{SoC}_{\max} - (R_n[t] + R_{dd}[t]) \cdot \Delta t$$

---

### What Changes in the Equation When You Tweak Sidebar Settings?

| Sidebar Action | Code Change | Exact Mathematical Effect | Real-World Consequence |
|---|---|---|---|
| **Toggle "Enable Reserves" OFF** | `enable_reserves = False` | Sets $R_n[t] = R_{du}[t] = R_{dd}[t] = R_a[t] = 0$. Drops all reserve revenue terms from objective. Removes headroom bounds from SoC constraints. | The battery is 100% free to cycle from 10% to 90% SoC purely for energy arbitrage. Total revenue drops because FCR-N payments are lost. |
| **Increase Peak Tariff (€8.5 → €25/kW)** | `grid_tariff = 25.0` | Multiplies $P_{\text{peak}} \cdot c_{\text{peak}}$ penalty by 3× in objective function. | The solver drastically restricts $P_{\text{imp}}[t]$. The BESS discharges aggressively during peak bus charging hours to keep the peak flat. |
| **Increase BESS Power (1 MW → 2 MW)** | `max_power_kw = 2000.0` | Relaxes Big-M constraints and power headroom: $P_{\text{bess}} = 2000$. | The battery can charge twice as fast during brief negative-price hours and bid twice as much capacity into FCR-N. |
| **Increase Degradation (€0.02 → €0.05/kWh)** | `bess_deg = 0.05` | Increases cost coefficient $c_{\text{deg}}$ on $(P_{\text{ch}} + P_{\text{dis}})$. | The solver becomes conservative. It will only cycle the battery if the price spread between night and day exceeds ~€0.12/kWh. |

---

## 7. The "Price of Forecast Error": Linking AI to Financial Value

In traditional machine learning courses, models are judged on **RMSE (Root Mean Squared Error)** or **$R^2$**.  
**In real quantitative energy trading, this is a dangerous trap.**

### Why RMSE is Insufficient: The Asymmetric Cost of Being Wrong
Consider two forecast errors:
1. **Case A**: At 03:00 AM (cheap night baseload), your model predicts €20/MWh, but the actual price is €25/MWh (Error = +€5).
   - *Impact*: The battery charges anyway. Minor €5/MWh sub-optimality.
2. **Case B**: At 18:00 PM (evening peak), your model predicts €45/MWh, but a nuclear reactor trips and actual prices spike to €140/MWh (Error = -€95).
   - *Impact*: Because the model predicted a mild price, the optimizer did not bother to preserve battery energy. When the €140 spike hits, the battery is empty, and the depot is forced to purchase power from the grid at astronomical prices, simultaneously setting a devastating new monthly peak demand charge!

### The Two-Stage Market Settlement Simulation
In [`simulation/forecasting.py`](file:///c:/Users/imac/Desktop/ENERGY/simulation/forecasting.py), we simulate what happens when plans collide with reality:
* **Stage 1 (Day-Ahead Optimization)**: Solve the MILP using the model's predicted prices $\hat{\mathbf{p}}$ to obtain the committed schedule ($P_{\text{imp}}^*, P_{\text{ch}}^*, P_{\text{dis}}^*, P_{\text{peak}}^*$).
* **Stage 2 (Real-Time Settlement)**: Settle the cash flows using the **actual real-world prices** $\mathbf{p}_{\text{actual}}$:

$$\text{Realized Cost} = \sum_{t} \left[ \frac{\lambda_{\text{actual}}[t]}{1000} (P_{\text{imp}}^*[t] - P_{\text{exp}}^*[t]) + \text{Taxes} + \text{Fees} - \text{Reserves} + \text{Degradation} \right] + P_{\text{peak}}^* \cdot c_{\text{peak}}$$

$$\text{Forecast Regret} = \text{Realized Cost}_{\text{model}} - \text{Realized Cost}_{\text{Oracle}}$$

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
**Key Takeaway**: XGBoost's superior handling of non-linear threshold steps saves an additional **€8.88/day (~€3,240/year)** over MLR, and cuts peak grid exposure by 29 kW.

---

## 8. EU AI Act & Regulatory Governance (Articles 13 & 14)

Because our AI system directly calculates bidding schedules submitted to Nord Pool and Svenska kraftnät, it influences critical European energy infrastructure. Under the **EU AI Act**, algorithmic energy trading systems are classified under strict governance rules:

### Mandatory Requirements:
1. **Explainability & Transparency (Article 13)**:
   - "No black-box models in production."
   - Every price forecast and dispatch decision must be attributable to its input parameters.
   - For **XGBoost**: We achieve compliance through **TreeSHAP (SHapley Additive exPlanations)**, which breaks down the exact monetary attribution of every feature (e.g. "Spot price predicted at €92/MWh: +€24 due to -15°C temperature, -€11 due to 14 m/s wind, +€34 due to evening peak hour").
2. **Human Oversight & Auditing (Article 14)**:
   - Operators must be able to audit and override automated decisions.
   - The system maintains an append-only immutable audit trail of feature store snapshots, model versions, and solver logs.
3. **Data Residency & NIS2**:
   - Deployed strictly in `europe-north2` (Stockholm / Finland). Zero data egress outside the EU.
4. **The Absolute Safety Veto**:
   - The mathematical optimizer output is treated as a **recommendation envelope**. The edge safety layer on the physical charger holds hardware-level priority to veto commands that would breach breaker ratings or thermal thresholds.

---

## 9. Hands-On Experimentation & Interview Prep Walkthrough

To prepare for your interview, open the Streamlit dashboard (`streamlit run dashboard.py`) and perform these 5 interactive experiments:

### Experiment 1: The Cold Snap Hockey Stick
* Go to **Tab 2: 🤖 ML Forecasting**.
* Select the **❄️ Temperature Response** tab.
* Slide your eyes along the curve from +20°C to -20°C:
  * Look at MLR's straight dashed line: notice how it stubbornly underpredicts cold temperatures.
  * Look at XGBoost's step jumps below 0°C.
  * *Interview Question*: *"How does temperature non-linearity affect Nordic price forecasting?"*
  * *Answer*: *"Nordic heating demand is piecewise non-linear. Below 0°C, heat pumps lose COP and resistive heating kicks in. Linear models miss the tail spikes, which is why tree-based ensembles (XGBoost/LightGBM) with threshold splits are standard."*

### Experiment 2: Running Optimization on ML Forecasts vs. Oracle
* Go to **Tab 4: ⚡ Optimisation**.
* In the **Price Signal** dropdown, select **XGBoost Forecast**, then click **Run Optimisation**.
* Notice how the BESS charges overnight (hours 01:00–04:00) when prices are lowest, and discharges at 08:00 and 18:00.
* Switch the Price Signal to **Actual Spot (Oracle)** and rerun.
* Compare the total cost. The gap is the **Price of Forecast Error**.

### Experiment 3: Diminishing Returns on Battery Size
* In the sidebar, change **BESS Capacity** from 2,000 kWh to 4,000 kWh.
* Re-run optimization. Does savings double?
* **No!** You will observe diminishing marginal returns because the primary depot peak (evening bus return) is already shaved by the first 2,000 kWh. Adding more battery capacity only unlocks lower-margin secondary arbitrage cycles.

### Experiment 4: The Value of Ancillary Services (FCR-N)
* In Tab 4, uncheck **Enable Reserve Markets** and click **Run Optimisation**.
* Notice what happens to the State of Charge (SoC) line:
  * With reserves enabled, SoC stays hovering around **40%–60%** because FCR-N requires symmetric upward and downward headroom.
  * With reserves disabled, the battery aggressively charges to 90% and dumps to 10% purely for spot arbitrage.
  * *Interview Question*: *"Why does an FCR-N battery maintain ~50% SoC?"*
  * *Answer*: *"SvK rules mandate symmetric reserve availability. If frequency spikes high, the battery must absorb power; if frequency drops low, it must inject power. Operating near 50% SoC guarantees headroom in both directions."*

### Experiment 5: The Impact of Degradation Cost
* In the sidebar, set **Degradation Cost** to **0 €¢/kWh**. Run optimization.
  * The battery cycles back and forth constantly, chasing even tiny €1/MWh price variations.
* Now raise **Degradation Cost** to **5 €¢/kWh** (€0.05/kWh throughput). Run optimization.
  * The battery remains stationary for most of the day, only firing during the single largest morning/evening price spread.
  * *Interview Question*: *"How do you model battery cycle aging in MILP?"*
  * *Answer*: *"We add a linear degradation cost coefficient $c_{\text{deg}}$ to charge and discharge power variables in the objective function. This creates an automatic economic hurdle rate: the solver will only dispatch the battery if the market spread exceeds round-trip efficiency losses plus degradation wear."*
