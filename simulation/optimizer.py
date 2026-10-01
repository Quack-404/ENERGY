"""
Multi-Horizon MILP Dispatch Optimiser
══════════════════════════════════════

This is the CORE ENGINE of the platform — the "upper strategic level"
from the two-tier architecture described in the JD.

It formulates a Mixed-Integer Linear Program (MILP) that co-optimises:
  1. Bus fleet charging (hard constraints — transit schedules are sacred)
  2. BESS dispatch (charge/discharge scheduling)
  3. Grid import/export (spot price arbitrage)
  4. Peak demand management (peak shaving)
  5. Reserve capacity allocation (FCR-N, FCR-D up/down, aFRR)
  6. PV self-consumption maximisation

The output is a "dynamic power envelope" — hour-by-hour setpoints that
the lower execution layer (edge controllers) translate into per-charger
commands. The edge safety layer holds absolute physical veto.

═══════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION (for interview discussion)
═══════════════════════════════════════════════════════════════

SETS:
  T = {0, 1, ..., H-1}  — timesteps (hourly for day-ahead)

PARAMETERS:
  λ_spot[t]     — Spot price (EUR/MWh)
  λ_fcr_n[t]    — FCR-N capacity price (EUR/MW/h)
  λ_fcr_d_up[t] — FCR-D up capacity price
  λ_fcr_d_dn[t] — FCR-D down capacity price
  λ_afrr[t]     — aFRR capacity price
  D_bus[t]      — Bus charging demand (kW)
  G_pv[t]       — PV generation (kW)
  C_bess        — BESS energy capacity (kWh)
  P_bess        — BESS max power (kW)
  P_grid_max    — Grid connection limit (kW)
  η_c, η_d      — Charge/discharge efficiency
  SoC_min/max   — SoC bounds (kWh)
  c_peak        — Peak demand tariff (EUR/kW/month)
  c_grid        — Grid energy fee (EUR/kWh)
  c_tax         — Energy tax (EUR/kWh)
  c_deg         — BESS degradation cost (EUR/kWh cycled)

DECISION VARIABLES:
  P_imp[t] ≥ 0   — Grid import power (kW)
  P_exp[t] ≥ 0   — Grid export power (kW)
  P_ch[t]  ≥ 0   — BESS charging power (kW)
  P_dis[t] ≥ 0   — BESS discharging power (kW)
  SoC[t]   ≥ 0   — BESS state of charge (kWh)
  R_n[t]   ≥ 0   — FCR-N capacity bid (kW)
  R_du[t]  ≥ 0   — FCR-D up capacity bid (kW)
  R_dd[t]  ≥ 0   — FCR-D down capacity bid (kW)
  R_a[t]   ≥ 0   — aFRR up capacity bid (kW)
  P_peak   ≥ 0   — Monthly peak demand (kW)
  δ[t]     ∈ {0,1} — Binary: 1=charging, 0=discharging

OBJECTIVE (Minimize total cost):
  min Σ_t [ λ_spot[t]/1000 · P_imp[t] · Δt          (energy purchase)
          + c_grid · P_imp[t] · Δt                     (grid fees)
          + c_tax  · P_imp[t] · Δt                     (energy tax)
          - λ_spot[t]/1000 · P_exp[t] · Δt            (export revenue)
          - λ_fcr_n[t]/1000 · R_n[t] · Δt             (FCR-N revenue)
          - λ_fcr_d_up[t]/1000 · R_du[t] · Δt         (FCR-D up revenue)
          - λ_fcr_d_dn[t]/1000 · R_dd[t] · Δt         (FCR-D dn revenue)
          - λ_afrr[t]/1000 · R_a[t] · Δt              (aFRR revenue)
          + c_deg · (P_ch[t] + P_dis[t]) · Δt          (degradation)
        ]
      + P_peak · c_peak · (days/30)                     (peak demand charge)

CONSTRAINTS:

  (C1) Energy balance:
       P_imp[t] + P_dis[t] + G_pv[t] = D_bus[t] + P_ch[t] + P_exp[t]

  (C2) SoC dynamics:
       SoC[0] = SoC_init + (η_c·P_ch[0] - P_dis[0]/η_d) · Δt
       SoC[t] = SoC[t-1] + (η_c·P_ch[t] - P_dis[t]/η_d) · Δt

  (C3) SoC bounds:
       SoC_min ≤ SoC[t] ≤ SoC_max

  (C4) Charge/discharge mutual exclusion (Big-M):
       P_ch[t]  ≤ P_bess · δ[t]
       P_dis[t] ≤ P_bess · (1 - δ[t])

  (C5) Peak demand tracking:
       P_peak ≥ P_imp[t]  ∀t

  (C6) Reserve power headroom (upward):
       P_dis[t] + R_n[t] + R_du[t] + R_a[t] ≤ P_bess

  (C7) Reserve power headroom (downward):
       P_ch[t] + R_n[t] + R_dd[t] ≤ P_bess

  (C8) Reserve energy headroom (upward — enough SoC to discharge):
       SoC[t] ≥ SoC_min + (R_n[t] + R_du[t] + R_a[t]) · Δt

  (C9) Reserve energy headroom (downward — enough room to absorb):
       SoC[t] ≤ SoC_max - (R_n[t] + R_dd[t]) · Δt

  (C10) Grid connection limits:
        P_imp[t] ≤ P_grid_max_import
        P_exp[t] ≤ P_grid_max_export

═══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DepotConfig


def _get_solver(preferred: str = 'appsi_highs'):
    """
    Find an available MILP solver.

    Priority order: appsi_highs → highs → glpk → cbc
    HiGHS is preferred because it's open-source, fast, and pip-installable.
    """
    import pyomo.environ as pyo
    for solver_name in [preferred, 'appsi_highs', 'highs', 'glpk', 'cbc']:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver.available():
                return solver, solver_name
        except Exception:
            continue
    raise RuntimeError(
        "No MILP solver found. Install HiGHS: pip install highspy\n"
        "Or GLPK: conda install -c conda-forge glpk"
    )


class DepotOptimizer:
    """
    Multi-horizon MILP dispatch optimiser for BESS + Depot co-optimisation.

    Usage:
        optimizer = DepotOptimizer(config)
        optimizer.build_model(spot_prices, fcr_n_prices, ..., bus_demand, pv_gen)
        status = optimizer.solve()
        results_df = optimizer.extract_results()
    """

    def __init__(self, config: DepotConfig):
        self.config = config
        self.model = None
        self.solve_result = None
        self.solve_time = None

    def build_model(self,
                    spot_prices: np.ndarray,
                    fcr_n_prices: np.ndarray,
                    fcr_d_up_prices: np.ndarray,
                    fcr_d_down_prices: np.ndarray,
                    afrr_prices: np.ndarray,
                    bus_demand: np.ndarray,
                    pv_generation: np.ndarray,
                    enable_reserves: bool = True,
                    enable_export: bool = True,
                    enable_peak_shaving: bool = True):
        """
        Build the Pyomo ConcreteModel.

        Feature flags allow disabling value streams for marginal
        value decomposition (run with vs. without each stream).
        """
        import pyomo.environ as pyo

        c = self.config
        model = pyo.ConcreteModel("DepotBESS_Dispatch")

        T = len(spot_prices)
        dt = c.simulation.timestep_minutes / 60.0  # Hours per timestep
        days = c.simulation.horizon_hours / 24.0

        # ══════════════════════════════════════
        # SETS
        # ══════════════════════════════════════
        model.T = pyo.RangeSet(0, T - 1)

        # ══════════════════════════════════════
        # PARAMETERS
        # ══════════════════════════════════════
        model.spot_price = pyo.Param(model.T, initialize=dict(enumerate(spot_prices)))
        model.fcr_n_price = pyo.Param(model.T, initialize=dict(enumerate(fcr_n_prices)))
        model.fcr_d_up_price = pyo.Param(model.T, initialize=dict(enumerate(fcr_d_up_prices)))
        model.fcr_d_down_price = pyo.Param(model.T, initialize=dict(enumerate(fcr_d_down_prices)))
        model.afrr_price = pyo.Param(model.T, initialize=dict(enumerate(afrr_prices)))
        model.bus_demand = pyo.Param(model.T, initialize=dict(enumerate(bus_demand)))
        model.pv_gen = pyo.Param(model.T, initialize=dict(enumerate(pv_generation)))

        # ══════════════════════════════════════
        # DECISION VARIABLES
        # ══════════════════════════════════════

        # Grid power
        model.p_grid_import = pyo.Var(
            model.T, domain=pyo.NonNegativeReals,
            bounds=(0, c.grid.max_import_kw)
        )
        model.p_grid_export = pyo.Var(
            model.T, domain=pyo.NonNegativeReals,
            bounds=(0, c.grid.max_export_kw if enable_export else 0)
        )

        # Peak demand tracker
        model.p_peak = pyo.Var(domain=pyo.NonNegativeReals)

        # BESS power
        model.p_bess_charge = pyo.Var(
            model.T, domain=pyo.NonNegativeReals,
            bounds=(0, c.bess.max_power_kw)
        )
        model.p_bess_discharge = pyo.Var(
            model.T, domain=pyo.NonNegativeReals,
            bounds=(0, c.bess.max_power_kw)
        )

        # BESS state of charge
        soc_min_kwh = c.bess.min_soc * c.bess.capacity_kwh
        soc_max_kwh = c.bess.max_soc * c.bess.capacity_kwh
        model.soc = pyo.Var(
            model.T, domain=pyo.NonNegativeReals,
            bounds=(soc_min_kwh, soc_max_kwh)
        )

        # Binary: charge/discharge mutual exclusion
        model.is_charging = pyo.Var(model.T, domain=pyo.Binary)

        # Reserve capacity bids
        reserve_ub = c.bess.max_power_kw if enable_reserves else 0
        model.fcr_n = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, reserve_ub))
        model.fcr_d_up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, reserve_ub))
        model.fcr_d_down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, reserve_ub))
        model.afrr_up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, reserve_ub))

        # ══════════════════════════════════════
        # OBJECTIVE FUNCTION
        # ══════════════════════════════════════
        def objective_rule(m):
            total_cost = 0.0
            for t in m.T:
                # --- COSTS ---
                # Energy purchase (spot price × import)
                total_cost += m.spot_price[t] / 1000 * m.p_grid_import[t] * dt
                # Grid transmission + distribution fee
                total_cost += c.market.grid_energy_fee_eur_kwh * m.p_grid_import[t] * dt
                # Energy tax
                total_cost += c.market.energy_tax_eur_kwh * m.p_grid_import[t] * dt
                # BESS degradation (proportional to throughput)
                total_cost += c.bess.degradation_cost_eur_per_kwh * (
                    m.p_bess_charge[t] + m.p_bess_discharge[t]
                ) * dt

                # --- REVENUES (negative cost) ---
                # Export revenue
                if enable_export:
                    total_cost -= m.spot_price[t] / 1000 * m.p_grid_export[t] * dt
                # Reserve capacity revenues
                if enable_reserves:
                    total_cost -= m.fcr_n_price[t] / 1000 * m.fcr_n[t] * dt
                    total_cost -= m.fcr_d_up_price[t] / 1000 * m.fcr_d_up[t] * dt
                    total_cost -= m.fcr_d_down_price[t] / 1000 * m.fcr_d_down[t] * dt
                    total_cost -= m.afrr_price[t] / 1000 * m.afrr_up[t] * dt

            # Peak demand charge (prorated from monthly to horizon)
            if enable_peak_shaving:
                total_cost += m.p_peak * c.market.grid_tariff_eur_kw_month * days / 30.0

            return total_cost

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # ══════════════════════════════════════
        # CONSTRAINTS
        # ══════════════════════════════════════

        # (C1) Energy balance at each timestep
        #      Supply = Demand
        #      grid_import + bess_discharge + pv = bus_demand + bess_charge + grid_export
        def energy_balance_rule(m, t):
            supply = m.p_grid_import[t] + m.p_bess_discharge[t] + m.pv_gen[t]
            demand = m.bus_demand[t] + m.p_bess_charge[t] + m.p_grid_export[t]
            return supply == demand
        model.c_energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

        # (C2) SoC dynamics
        soc_init_kwh = c.bess.initial_soc * c.bess.capacity_kwh
        def soc_dynamics_rule(m, t):
            energy_in = c.bess.charge_efficiency * m.p_bess_charge[t] * dt
            energy_out = m.p_bess_discharge[t] / c.bess.discharge_efficiency * dt
            if t == 0:
                return m.soc[t] == soc_init_kwh + energy_in - energy_out
            return m.soc[t] == m.soc[t - 1] + energy_in - energy_out
        model.c_soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

        # (C4) Charge/discharge mutual exclusion via Big-M
        def charge_exclusive_rule(m, t):
            return m.p_bess_charge[t] <= c.bess.max_power_kw * m.is_charging[t]
        model.c_charge_excl = pyo.Constraint(model.T, rule=charge_exclusive_rule)

        def discharge_exclusive_rule(m, t):
            return m.p_bess_discharge[t] <= c.bess.max_power_kw * (1 - m.is_charging[t])
        model.c_discharge_excl = pyo.Constraint(model.T, rule=discharge_exclusive_rule)

        # (C5) Peak demand tracking
        if enable_peak_shaving:
            def peak_demand_rule(m, t):
                return m.p_peak >= m.p_grid_import[t]
            model.c_peak_demand = pyo.Constraint(model.T, rule=peak_demand_rule)

        # (C6) Reserve power headroom — upward
        #      When called to provide upward reserve (increase generation / decrease load),
        #      the BESS must have enough discharge headroom.
        if enable_reserves:
            def reserve_up_power_rule(m, t):
                return (m.p_bess_discharge[t] + m.fcr_n[t] + m.fcr_d_up[t] +
                        m.afrr_up[t]) <= c.bess.max_power_kw
            model.c_reserve_up_power = pyo.Constraint(model.T, rule=reserve_up_power_rule)

            # (C7) Reserve power headroom — downward
            def reserve_down_power_rule(m, t):
                return (m.p_bess_charge[t] + m.fcr_n[t] +
                        m.fcr_d_down[t]) <= c.bess.max_power_kw
            model.c_reserve_down_power = pyo.Constraint(model.T, rule=reserve_down_power_rule)

            # (C8) Reserve energy headroom — upward (must have enough SoC to sustain)
            #      Assume 1-hour sustained activation for energy headroom
            activation_hours = 1.0
            def reserve_up_energy_rule(m, t):
                return m.soc[t] >= soc_min_kwh + (
                    m.fcr_n[t] + m.fcr_d_up[t] + m.afrr_up[t]
                ) * activation_hours
            model.c_reserve_up_energy = pyo.Constraint(model.T, rule=reserve_up_energy_rule)

            # (C9) Reserve energy headroom — downward (must have room to absorb)
            def reserve_down_energy_rule(m, t):
                return m.soc[t] <= soc_max_kwh - (
                    m.fcr_n[t] + m.fcr_d_down[t]
                ) * activation_hours
            model.c_reserve_down_energy = pyo.Constraint(model.T, rule=reserve_down_energy_rule)

        self.model = model
        return model

    def solve(self, time_limit: Optional[int] = None) -> str:
        """
        Solve the MILP model.

        Returns:
            Solver status string ('optimal', 'feasible', 'infeasible', etc.)
        """
        import pyomo.environ as pyo

        if self.model is None:
            raise RuntimeError("Call build_model() before solve()")

        tl = time_limit or self.config.simulation.solver_time_limit
        solver, solver_name = _get_solver(self.config.simulation.solver)

        # Configure solver options
        if 'highs' in solver_name:
            solver.options['time_limit'] = float(tl)
            solver.options['mip_rel_gap'] = self.config.simulation.mip_gap
        elif solver_name == 'glpk':
            solver.options['tmlim'] = tl
            solver.options['mipgap'] = self.config.simulation.mip_gap

        t_start = time.time()
        result = solver.solve(self.model, tee=False)
        self.solve_time = time.time() - t_start
        self.solve_result = result

        status = str(result.solver.termination_condition)
        return status

    def extract_results(self) -> pd.DataFrame:
        """
        Extract optimisation results into a pandas DataFrame.

        Each row = one timestep. Columns include all decision variables,
        parameters, and derived quantities.
        """
        import pyomo.environ as pyo

        if self.model is None:
            raise RuntimeError("Model not built yet")

        m = self.model
        T = len(m.T)
        dt = self.config.simulation.timestep_minutes / 60.0

        data = {
            'hour': list(range(T)),
            'spot_price': [pyo.value(m.spot_price[t]) for t in range(T)],
            'fcr_n_price': [pyo.value(m.fcr_n_price[t]) for t in range(T)],
            'fcr_d_up_price': [pyo.value(m.fcr_d_up_price[t]) for t in range(T)],
            'fcr_d_down_price': [pyo.value(m.fcr_d_down_price[t]) for t in range(T)],
            'afrr_price': [pyo.value(m.afrr_price[t]) for t in range(T)],
            'bus_demand': [pyo.value(m.bus_demand[t]) for t in range(T)],
            'pv_gen': [pyo.value(m.pv_gen[t]) for t in range(T)],
            'p_grid_import': [pyo.value(m.p_grid_import[t]) for t in range(T)],
            'p_grid_export': [pyo.value(m.p_grid_export[t]) for t in range(T)],
            'p_bess_charge': [pyo.value(m.p_bess_charge[t]) for t in range(T)],
            'p_bess_discharge': [pyo.value(m.p_bess_discharge[t]) for t in range(T)],
            'soc': [pyo.value(m.soc[t]) for t in range(T)],
            'soc_pct': [pyo.value(m.soc[t]) / self.config.bess.capacity_kwh * 100 for t in range(T)],
            'fcr_n': [pyo.value(m.fcr_n[t]) for t in range(T)],
            'fcr_d_up': [pyo.value(m.fcr_d_up[t]) for t in range(T)],
            'fcr_d_down': [pyo.value(m.fcr_d_down[t]) for t in range(T)],
            'afrr_up': [pyo.value(m.afrr_up[t]) for t in range(T)],
            'is_charging': [pyo.value(m.is_charging[t]) for t in range(T)],
        }

        df = pd.DataFrame(data)

        # Derived quantities
        df['net_bess_power'] = df['p_bess_discharge'] - df['p_bess_charge']
        df['net_grid_power'] = df['p_grid_import'] - df['p_grid_export']
        df['total_reserve'] = df['fcr_n'] + df['fcr_d_up'] + df['fcr_d_down'] + df['afrr_up']
        df['p_peak'] = pyo.value(m.p_peak)
        df['objective_value'] = pyo.value(m.objective)

        return df

    def get_dual_prices(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract dual (shadow) prices from constraints.

        Interview talking point:
          "What are shadow prices and why do they matter?"
          → The dual price of a constraint tells you the marginal cost
            of tightening that constraint by one unit. For the energy
            balance constraint, the dual gives the marginal value of
            electricity at each timestep — the depot's internal
            "locational marginal price" (LMP analogue).

        Note: Duals are only available for LP relaxation (not MILP with
        binaries active). For MILP, fix binaries and re-solve as LP.
        """
        import pyomo.environ as pyo

        if not hasattr(self.model, 'dual'):
            return None

        try:
            T = len(self.model.T)
            duals = {
                'energy_balance': [
                    self.model.dual.get(self.model.c_energy_balance[t], 0)
                    for t in range(T)
                ],
            }
            return duals
        except Exception:
            return None

    def get_solve_statistics(self) -> Dict:
        """Return solve statistics for reporting."""
        import pyomo.environ as pyo

        obj_val = None
        peak_val = None
        if self.model and self.solve_result:
            try:
                obj_val = round(pyo.value(self.model.objective), 2)
                peak_val = round(pyo.value(self.model.p_peak), 1)
            except (ValueError, RuntimeError):
                pass

        stats = {
            'solve_time_seconds': round(self.solve_time, 2) if self.solve_time else None,
            'objective_value_eur': obj_val,
            'peak_demand_kw': peak_val,
            'status': str(self.solve_result.solver.termination_condition) if self.solve_result else None,
            'num_variables': self.model.nvariables() if self.model else None,
            'num_constraints': self.model.nconstraints() if self.model else None,
        }
        return stats

