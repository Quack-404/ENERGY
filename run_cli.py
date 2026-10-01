"""
CLI runner for the Energy Market Optimisation Simulation.
Run without Streamlit for quick results in the terminal.

Usage: python run_cli.py
"""
import numpy as np
import sys
import os
import io

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DepotConfig
from simulation.market import generate_all_market_data
from simulation.depot import FleetModel, PVModel
from simulation.optimizer import DepotOptimizer
from simulation.analysis import ValueStackAnalyzer


def main():
    print("=" * 70)
    print("[*] Energy Market Optimisation Simulation -- CLI Mode")
    print("=" * 70)

    # Default configuration
    config = DepotConfig()
    print(f"\n[CONFIG] Configuration:")
    print(f"   BESS: {config.bess.capacity_kwh} kWh / {config.bess.max_power_kw} kW")
    print(f"   Fleet: {config.fleet.num_buses} buses x {config.fleet.bus_battery_kwh} kWh")
    print(f"   PV: {config.pv.capacity_kwp} kWp")
    print(f"   Grid: {config.grid.max_import_kw} kW import / {config.grid.max_export_kw} kW export")
    print(f"   Zone: {config.market.bidding_zone.value}")
    print(f"   Horizon: {config.simulation.horizon_hours}h")
    print(f"   Month: {config.simulation.month}")

    # Generate data
    print("\n[MARKET] Generating market data...")
    market_data = generate_all_market_data(config.market, config.simulation)
    print(f"   Spot price: {np.mean(market_data['spot_price']):.1f} EUR/MWh (avg), "
          f"{np.min(market_data['spot_price']):.1f} - {np.max(market_data['spot_price']):.1f} range")

    print("\n[FLEET] Generating fleet schedule...")
    fleet = FleetModel(config.fleet, config.chargers, config.simulation)
    schedule = fleet.generate_bus_schedule()
    bus_demand = fleet.compute_charging_demand(schedule)
    print(f"   Total charging demand: {np.sum(bus_demand):.0f} kWh")
    print(f"   Peak demand: {np.max(bus_demand):.0f} kW")

    print("\n[PV] Generating PV profile...")
    pv = PVModel(config.pv, config.simulation)
    pv_gen = pv.generate_production()
    print(f"   Total PV generation: {np.sum(pv_gen):.0f} kWh")

    # Run optimisation
    print("\n[OPTIM] Building MILP model...")
    optimizer = DepotOptimizer(config)
    optimizer.build_model(
        spot_prices=market_data['spot_price'],
        fcr_n_prices=market_data['fcr_n_price'],
        fcr_d_up_prices=market_data['fcr_d_up_price'],
        fcr_d_down_prices=market_data['fcr_d_down_price'],
        afrr_prices=market_data['afrr_price'],
        bus_demand=bus_demand,
        pv_generation=pv_gen,
    )

    stats = optimizer.get_solve_statistics()
    print(f"   Variables: {stats['num_variables']}")
    print(f"   Constraints: {stats['num_constraints']}")

    print("\n[SOLVE] Solving with HiGHS...")
    status = optimizer.solve()
    stats = optimizer.get_solve_statistics()
    print(f"   Status: {status}")
    print(f"   Solve time: {stats['solve_time_seconds']}s")
    print(f"   Objective: EUR {stats['objective_value_eur']:.2f}")

    if status in ['optimal', 'feasible']:
        results_df = optimizer.extract_results()

        # Value stack
        analyzer = ValueStackAnalyzer(config)
        savings = analyzer.compute_savings_summary(results_df)

        print("\n" + "=" * 70)
        print("[$$] VALUE STACK DECOMPOSITION")
        print("=" * 70)
        print(f"\n   Baseline cost (dumb charging):  EUR {savings['baseline_cost_eur']:>10.2f}")
        print(f"   Optimised cost:                 EUR {savings['optimized_cost_eur']:>10.2f}")
        print(f"   -----------------------------------------------")
        print(f"   Daily savings:                  EUR {savings['total_savings_eur']:>10.2f}  "
              f"({savings['savings_pct']:.1f}%)")
        print(f"   Annual savings (est):           EUR {savings['annual_savings_estimate_eur']:>10,.0f}")
        print(f"   BESS payback period:            {savings['simple_payback_years']:.1f} years")

        print(f"\n   Peak demand: {savings['peak_baseline_kw']:.0f} -> "
              f"{savings['peak_optimized_kw']:.0f} kW "
              f"(-{savings['peak_reduction_kw']:.0f} kW)")

        print("\n   Revenue/Cost breakdown:")
        for stream, value in savings['value_stack'].items():
            indicator = " [+]" if value < 0 else " [-]"
            print(f"    {indicator} {stream:<25s} EUR {value:>8.2f}")

        print("\n   Reserve revenue breakdown:")
        for stream, value in savings.get('revenue_breakdown', {}).items():
            print(f"     >>> {stream:<25s} EUR {value:>8.2f}")

        # ── ML Forecasting Benchmark & Model Responses ──
        print("\n" + "=" * 70)
        print("[ML] FORECASTING BENCHMARK: MLR vs XGBoost vs ANN")
        print("=" * 70)
        from simulation.forecasting import (
            MarketFeatureDataset, ModelSuite, ModelResponseAnalyzer,
            evaluate_all_models, evaluate_forecast_dispatch_cost
        )
        ds_gen = MarketFeatureDataset(base_price=config.market.base_spot_price_eur_mwh, seed=config.simulation.seed)
        df_hist = ds_gen.generate_historical_dataset(n_days=90)
        train_df = df_hist.iloc[:1800]
        test_df = df_hist.iloc[1800:1800 + config.simulation.horizon_hours].copy()
        suite = ModelSuite(random_state=config.simulation.seed)
        suite.fit(train_df)
        preds = suite.predict_all(test_df)
        metrics = evaluate_all_models(test_df, preds)

        print("\n   Out-of-sample Accuracy Metrics:")
        print(f"   {'Model':<10} {'MAE (EUR/MWh)':<14} {'RMSE (EUR/MWh)':<14} {'R2 Score':<10} {'MAPE (%)':<10}")
        print("   " + "-" * 56)
        for _, row in metrics.iterrows():
            print(f"   {row['Model']:<10} {row['MAE (€/MWh)']:<14.2f} {row['RMSE (€/MWh)']:<14.2f} {row['R² Score']:<10.3f} {row['MAPE (%)']:<10.1f}")

        # Physical response stress test
        analyzer = ModelResponseAnalyzer(suite)
        stress_df = analyzer.compute_stress_test_responses()
        print("\n   Physical Stress Test Responses (Regime Adaptation):")
        for _, s in stress_df.iterrows():
            print(f"    * {s['Scenario']:<24}: MLR EUR {s['MLR Predicted (€/MWh)']:>6.1f} | XGBoost EUR {s['XGBoost Predicted (€/MWh)']:>6.1f} | ANN EUR {s['ANN Predicted (€/MWh)']:>6.1f}")

        # Downstream Price of Forecast Error
        print("\n   The 'Price of Forecast Error' (Realized Cost Settled Under Actual Spot):")
        for m_name in ['XGBoost', 'MLR', 'ANN']:
            m_res = evaluate_forecast_dispatch_cost(
                actual_prices=test_df['spot_price'].values,
                forecast_prices=preds[m_name],
                market_data=market_data,
                bus_demand=bus_demand,
                pv_generation=pv_gen,
                config=config
            )
            if m_res['success']:
                regret = m_res['realized_cost_eur'] - savings['optimized_cost_eur']
                print(f"    * {m_name:<10} Realized: EUR {m_res['realized_cost_eur']:>7.2f}/day | Regret vs Oracle: EUR {max(0.0, regret):>6.2f}/day")

        print("\n" + "=" * 70)
        print("[OK] Simulation complete! Run 'streamlit run dashboard.py' for interactive mode.")
        print("=" * 70)
    else:
        print(f"\n[FAIL] Optimisation failed with status: {status}")


if __name__ == '__main__':
    main()
