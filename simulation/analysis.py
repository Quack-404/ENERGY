"""
Value Stack Analysis & Backtesting Engine
═════════════════════════════════════════

Provides tools for:
  1. Decomposing optimised results into individual value streams
  2. Computing baseline (unoptimised) costs for comparison
  3. Running marginal value analysis (toggle features on/off)
  4. Backtesting across multiple price scenarios and seasons
  5. Computing financial metrics (savings, ROI, payback)

Interview talking point:
  "How do you validate that your optimiser actually creates value?"
  → Compare against a baseline (dumb charging) AND backtest across
    historical/simulated price scenarios. Report P10/P50/P90 savings
    to communicate risk. Shadow-mode deployment before live trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import copy

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DepotConfig


class ValueStackAnalyzer:
    """
    Decomposes optimisation results into individual value streams.

    The "value stack" shows how much each revenue/cost stream contributes:
      - Energy arbitrage (buy low, sell high)
      - Peak shaving (reduce capacity charges)
      - Reserve revenues (FCR-N, FCR-D, aFRR)
      - PV self-consumption (avoided grid purchases)
      - BESS degradation (cycling cost)
    """

    def __init__(self, config: DepotConfig):
        self.config = config

    def decompose(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose total cost into individual value streams (EUR).

        Positive values = costs, negative values = revenues/savings.
        """
        c = self.config
        dt = c.simulation.timestep_minutes / 60.0
        days = c.simulation.horizon_hours / 24.0

        streams = {}

        # 1. Energy purchase cost
        streams['Energy Purchase'] = float(
            (results_df['spot_price'] / 1000 * results_df['p_grid_import'] * dt).sum()
        )

        # 2. Grid transmission & distribution fees
        streams['Grid Fees'] = float(
            (c.market.grid_energy_fee_eur_kwh * results_df['p_grid_import'] * dt).sum()
        )

        # 3. Energy tax
        streams['Energy Tax'] = float(
            (c.market.energy_tax_eur_kwh * results_df['p_grid_import'] * dt).sum()
        )

        # 4. Peak demand charge (monthly, prorated)
        streams['Peak Demand Charge'] = float(
            results_df['p_peak'].iloc[0] * c.market.grid_tariff_eur_kw_month * days / 30
        )

        # 5. Export revenue (negative = revenue)
        streams['Export Revenue'] = float(
            -(results_df['spot_price'] / 1000 * results_df['p_grid_export'] * dt).sum()
        )

        # 6. FCR-N revenue
        streams['FCR-N Revenue'] = float(
            -(results_df['fcr_n'] * results_df['fcr_n_price'] / 1000 * dt).sum()
        )

        # 7. FCR-D Up revenue
        streams['FCR-D Up Revenue'] = float(
            -(results_df['fcr_d_up'] * results_df['fcr_d_up_price'] / 1000 * dt).sum()
        )

        # 8. FCR-D Down revenue
        streams['FCR-D Down Revenue'] = float(
            -(results_df['fcr_d_down'] * results_df['fcr_d_down_price'] / 1000 * dt).sum()
        )

        # 9. aFRR revenue
        streams['aFRR Revenue'] = float(
            -(results_df['afrr_up'] * results_df['afrr_price'] / 1000 * dt).sum()
        )

        # 10. BESS degradation cost
        streams['BESS Degradation'] = float(
            (c.bess.degradation_cost_eur_per_kwh *
             (results_df['p_bess_charge'] + results_df['p_bess_discharge']) * dt).sum()
        )

        return streams

    def compute_baseline_cost(self, bus_demand: np.ndarray,
                               spot_prices: np.ndarray,
                               pv_generation: np.ndarray) -> Dict[str, float]:
        """
        Compute baseline cost: unoptimised "dumb charging".

        Baseline = charge buses directly from grid with no BESS,
        no reserve participation, no PV optimisation, no peak management.
        PV is used when available but surplus is curtailed (no export).
        """
        c = self.config
        dt = c.simulation.timestep_minutes / 60.0
        days = c.simulation.horizon_hours / 24.0

        # Net demand after PV self-consumption
        pv_used = np.minimum(pv_generation, bus_demand)
        grid_demand = bus_demand - pv_used

        # All from grid at spot price
        energy_cost = float((spot_prices / 1000 * grid_demand * dt).sum())
        grid_fees = float((c.market.grid_energy_fee_eur_kwh * grid_demand * dt).sum())
        energy_tax = float((c.market.energy_tax_eur_kwh * grid_demand * dt).sum())
        peak_charge = float(np.max(grid_demand) * c.market.grid_tariff_eur_kw_month * days / 30)

        total = energy_cost + grid_fees + energy_tax + peak_charge

        return {
            'total': total,
            'energy_cost': energy_cost,
            'grid_fees': grid_fees,
            'energy_tax': energy_tax,
            'peak_charge': peak_charge,
            'pv_self_consumed_kwh': float((pv_used * dt).sum()),
            'peak_demand_kw': float(np.max(grid_demand)),
        }

    def compute_savings_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive savings summary: optimised vs baseline.

        Returns metrics useful for business case and interview discussion.
        """
        value_stack = self.decompose(results_df)
        optimized_total = sum(value_stack.values())

        baseline = self.compute_baseline_cost(
            results_df['bus_demand'].values,
            results_df['spot_price'].values,
            results_df['pv_gen'].values,
        )

        total_savings = baseline['total'] - optimized_total
        days = self.config.simulation.horizon_hours / 24.0

        # Annualised savings estimate
        annual_savings = total_savings * (365 / max(days, 0.01))

        # Simple payback period for BESS investment
        bess_capex_estimate = self.config.bess.capacity_kwh * 300  # ~€300/kWh
        payback_years = bess_capex_estimate / annual_savings if annual_savings > 0 else float('inf')

        # Revenue breakdown (only the negative values in value stack)
        revenues = {k: -v for k, v in value_stack.items() if v < 0}
        total_revenue = sum(revenues.values())

        return {
            'baseline_cost_eur': round(baseline['total'], 2),
            'optimized_cost_eur': round(optimized_total, 2),
            'total_savings_eur': round(total_savings, 2),
            'savings_pct': round((total_savings / baseline['total']) * 100, 1) if baseline['total'] > 0 else 0,
            'annual_savings_estimate_eur': round(annual_savings, 0),
            'bess_capex_estimate_eur': round(bess_capex_estimate, 0),
            'simple_payback_years': round(payback_years, 1),
            'peak_reduction_kw': round(baseline['peak_demand_kw'] - results_df['p_peak'].iloc[0], 1),
            'peak_baseline_kw': round(baseline['peak_demand_kw'], 1),
            'peak_optimized_kw': round(results_df['p_peak'].iloc[0], 1),
            'total_reserve_revenue_eur': round(total_revenue, 2),
            'revenue_breakdown': revenues,
            'value_stack': value_stack,
            'baseline_detail': baseline,
        }

    def evaluate_dispatch_under_actual_prices(
        self,
        dispatch_plan: pd.DataFrame,
        actual_spot_prices: np.ndarray,
        market_data: Dict[str, np.ndarray],
        bus_demand: np.ndarray,
        pv_gen: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate what a pre-committed dispatch plan actually costs and earns
        when settled against the REALIZED actual spot market prices.
        """
        c = self.config
        dt = c.simulation.timestep_minutes / 60.0
        days = c.simulation.horizon_hours / 24.0

        p_import = dispatch_plan['p_grid_import'].values
        p_export = dispatch_plan['p_grid_export'].values
        p_ch = dispatch_plan['p_bess_charge'].values
        p_dis = dispatch_plan['p_bess_discharge'].values
        peak_kw = float(dispatch_plan['p_peak'].iloc[0])

        # Energy purchase and export under ACTUAL spot prices
        energy_cost = float((actual_spot_prices / 1000.0 * p_import * dt).sum())
        export_rev = float((actual_spot_prices / 1000.0 * p_export * dt).sum())
        grid_fees = float((c.market.grid_energy_fee_eur_kwh * p_import * dt).sum())
        energy_tax = float((c.market.energy_tax_eur_kwh * p_import * dt).sum())
        peak_tariff = float(peak_kw * c.market.grid_tariff_eur_kw_month * days / 30.0)
        deg_cost = float((c.bess.degradation_cost_eur_per_kwh * (p_ch + p_dis) * dt).sum())

        # Reserves
        fcr_n_rev = float((dispatch_plan['fcr_n'].values * market_data['fcr_n_price'] / 1000.0 * dt).sum())
        fcr_d_up_rev = float((dispatch_plan['fcr_d_up'].values * market_data['fcr_d_up_price'] / 1000.0 * dt).sum())
        fcr_d_down_rev = float((dispatch_plan['fcr_d_down'].values * market_data['fcr_d_down_price'] / 1000.0 * dt).sum())
        afrr_rev = float((dispatch_plan['afrr_up'].values * market_data['afrr_price'] / 1000.0 * dt).sum())
        reserve_rev = fcr_n_rev + fcr_d_up_rev + fcr_d_down_rev + afrr_rev

        net_cost = energy_cost + grid_fees + energy_tax + peak_tariff + deg_cost - export_rev - reserve_rev
        arbitrage = float(((actual_spot_prices / 1000.0 * p_dis * dt).sum()) - ((actual_spot_prices / 1000.0 * p_ch * dt).sum()))

        return {
            'net_cost_eur': round(net_cost, 2),
            'energy_cost_eur': round(energy_cost, 2),
            'export_revenue_eur': round(export_rev, 2),
            'peak_kw': round(peak_kw, 1),
            'peak_tariff_eur': round(peak_tariff, 2),
            'degradation_eur': round(deg_cost, 2),
            'reserve_revenue_eur': round(reserve_rev, 2),
            'arbitrage_eur': round(arbitrage, 2),
        }


class Backtester:
    """
    Backtesting engine: run the optimiser across multiple scenarios
    to assess robustness and expected performance distribution.

    Interview talking point:
      "What's your backtesting strategy?"
      → Generate scenario ensemble varying: price level (±30%),
        volatility, season (all 12 months), weekday/weekend,
        fleet size variation, and PV output. Report P10/P50/P90
        of daily savings. Production: use historical ENTSO-E data.
    """

    def __init__(self, base_config: DepotConfig):
        self.base_config = base_config

    def run_scenario_sweep(self, optimizer_class, market_generator,
                            fleet_model_class, pv_model_class,
                            num_scenarios: int = 12,
                            progress_callback=None) -> pd.DataFrame:
        """
        Run optimisation across multiple scenarios (one per month).

        Returns a DataFrame with one row per scenario containing
        key metrics.
        """
        results = []

        for i in range(num_scenarios):
            if progress_callback:
                progress_callback(i / num_scenarios)

            # Create scenario-specific config
            scenario_config = self._create_scenario(i, num_scenarios)

            try:
                # Generate data
                market_data = market_generator(scenario_config.market, scenario_config.simulation)

                fleet = fleet_model_class(
                    scenario_config.fleet,
                    scenario_config.chargers,
                    scenario_config.simulation
                )
                bus_demand = fleet.compute_charging_demand()

                pv = pv_model_class(scenario_config.pv, scenario_config.simulation)
                pv_gen = pv.generate_production()

                # Run optimizer
                opt = optimizer_class(scenario_config)
                opt.build_model(
                    spot_prices=market_data['spot_price'],
                    fcr_n_prices=market_data['fcr_n_price'],
                    fcr_d_up_prices=market_data['fcr_d_up_price'],
                    fcr_d_down_prices=market_data['fcr_d_down_price'],
                    afrr_prices=market_data['afrr_price'],
                    bus_demand=bus_demand,
                    pv_generation=pv_gen,
                )
                status = opt.solve(time_limit=60)
                result_df = opt.extract_results()

                # Analyse
                analyzer = ValueStackAnalyzer(scenario_config)
                savings = analyzer.compute_savings_summary(result_df)

                results.append({
                    'scenario': i,
                    'month': scenario_config.simulation.month,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][
                        scenario_config.simulation.month - 1],
                    'avg_spot_price': float(np.mean(market_data['spot_price'])),
                    'status': status,
                    'baseline_cost': savings['baseline_cost_eur'],
                    'optimized_cost': savings['optimized_cost_eur'],
                    'total_savings': savings['total_savings_eur'],
                    'savings_pct': savings['savings_pct'],
                    'peak_reduction_kw': savings['peak_reduction_kw'],
                    'reserve_revenue': savings['total_reserve_revenue_eur'],
                    'solve_time': opt.solve_time,
                })

            except Exception as e:
                results.append({
                    'scenario': i,
                    'month': (i % 12) + 1,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i % 12],
                    'status': 'error',
                    'error': str(e),
                    'total_savings': 0,
                    'savings_pct': 0,
                })

        if progress_callback:
            progress_callback(1.0)

        return pd.DataFrame(results)

    def _create_scenario(self, scenario_id: int, total: int) -> DepotConfig:
        """Create a scenario config varying month and price level."""
        config = copy.deepcopy(self.base_config)

        # Vary month (cycle through 12 months)
        config.simulation.month = (scenario_id % 12) + 1

        # Vary seed for different random realisations
        config.simulation.seed = self.base_config.simulation.seed + scenario_id * 100

        # Slight price level variation
        rng = np.random.RandomState(scenario_id + 999)
        price_factor = 1.0 + rng.uniform(-0.15, 0.15)
        config.market.base_spot_price_eur_mwh *= price_factor

        return config


class SensitivityAnalyzer:
    """
    Parameter sensitivity analysis.

    Sweeps a single parameter while holding others constant to show
    marginal value curves and diminishing returns.
    """

    @staticmethod
    def sweep_bess_capacity(base_config: DepotConfig,
                             optimizer_class,
                             market_data: Dict,
                             bus_demand: np.ndarray,
                             pv_gen: np.ndarray,
                             capacities: List[float],
                             progress_callback=None) -> pd.DataFrame:
        """Sweep BESS capacity and record total savings at each level."""
        results = []
        for i, cap in enumerate(capacities):
            if progress_callback:
                progress_callback(i / len(capacities))

            config = copy.deepcopy(base_config)
            config.bess.capacity_kwh = cap
            # Scale power proportionally (maintain C-rate)
            config.bess.max_power_kw = cap / 2  # C/2 rate

            try:
                opt = optimizer_class(config)
                opt.build_model(
                    spot_prices=market_data['spot_price'],
                    fcr_n_prices=market_data['fcr_n_price'],
                    fcr_d_up_prices=market_data['fcr_d_up_price'],
                    fcr_d_down_prices=market_data['fcr_d_down_price'],
                    afrr_prices=market_data['afrr_price'],
                    bus_demand=bus_demand,
                    pv_generation=pv_gen,
                )
                opt.solve(time_limit=30)
                result_df = opt.extract_results()

                analyzer = ValueStackAnalyzer(config)
                savings = analyzer.compute_savings_summary(result_df)

                results.append({
                    'bess_capacity_kwh': cap,
                    'bess_power_kw': cap / 2,
                    'total_savings_eur': savings['total_savings_eur'],
                    'savings_pct': savings['savings_pct'],
                    'reserve_revenue': savings['total_reserve_revenue_eur'],
                    'peak_reduction_kw': savings['peak_reduction_kw'],
                    'annual_savings': savings['annual_savings_estimate_eur'],
                    'payback_years': savings['simple_payback_years'],
                })
            except Exception as e:
                results.append({
                    'bess_capacity_kwh': cap,
                    'error': str(e),
                    'total_savings_eur': 0,
                })

        if progress_callback:
            progress_callback(1.0)
        return pd.DataFrame(results)

    @staticmethod
    def sweep_spot_price(base_config: DepotConfig,
                          optimizer_class,
                          market_generator,
                          bus_demand: np.ndarray,
                          pv_gen: np.ndarray,
                          price_levels: List[float],
                          progress_callback=None) -> pd.DataFrame:
        """Sweep base spot price level."""
        results = []
        for i, price in enumerate(price_levels):
            if progress_callback:
                progress_callback(i / len(price_levels))

            config = copy.deepcopy(base_config)
            config.market.base_spot_price_eur_mwh = price

            try:
                market_data = market_generator(config.market, config.simulation)

                opt = optimizer_class(config)
                opt.build_model(
                    spot_prices=market_data['spot_price'],
                    fcr_n_prices=market_data['fcr_n_price'],
                    fcr_d_up_prices=market_data['fcr_d_up_price'],
                    fcr_d_down_prices=market_data['fcr_d_down_price'],
                    afrr_prices=market_data['afrr_price'],
                    bus_demand=bus_demand,
                    pv_generation=pv_gen,
                )
                opt.solve(time_limit=30)
                result_df = opt.extract_results()

                analyzer = ValueStackAnalyzer(config)
                savings = analyzer.compute_savings_summary(result_df)

                results.append({
                    'spot_price_eur_mwh': price,
                    'total_savings_eur': savings['total_savings_eur'],
                    'savings_pct': savings['savings_pct'],
                    'reserve_revenue': savings['total_reserve_revenue_eur'],
                    'annual_savings': savings['annual_savings_estimate_eur'],
                })
            except Exception as e:
                results.append({
                    'spot_price_eur_mwh': price,
                    'error': str(e),
                    'total_savings_eur': 0,
                })

        if progress_callback:
            progress_callback(1.0)
        return pd.DataFrame(results)
