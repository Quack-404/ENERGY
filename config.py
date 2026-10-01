"""
Configuration module for the Energy Market Optimisation Simulation.
═══════════════════════════════════════════════════════════════════

All system parameters are defined as dataclasses for easy tweaking.
This mirrors the real system design where each subsystem (BESS, fleet,
market, grid) has its own configuration contract.

Key design decisions documented inline for interview discussion.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BiddingZone(Enum):
    """Swedish electricity bidding zones (part of Nord Pool)."""
    SE1 = "SE1"  # Luleå  — Northern Sweden, large hydro surplus
    SE2 = "SE2"  # Sundsvall — Mid Sweden
    SE3 = "SE3"  # Stockholm — Largest consumption area, reference zone
    SE4 = "SE4"  # Malmö — Southern Sweden, often highest prices


@dataclass
class BESSConfig:
    """
    Battery Energy Storage System configuration.

    Interview talking point: Why separate charge/discharge efficiency?
    → Asymmetric losses in power electronics; charge path includes
      AC→DC conversion + BMS overhead, discharge path is DC→AC.
      Round-trip η = η_charge × η_discharge.
    """
    capacity_kwh: float = 2000.0          # Total energy capacity (kWh) — 2 MWh
    max_power_kw: float = 1000.0          # Max charge/discharge power (kW) — 1 MW C/2 rate
    min_soc: float = 0.10                 # Minimum SoC (fraction) — protects cycle life
    max_soc: float = 0.90                 # Maximum SoC (fraction) — protects calendar life
    initial_soc: float = 0.50             # Initial SoC at horizon start
    charge_efficiency: float = 0.95       # One-way charging efficiency
    discharge_efficiency: float = 0.95    # One-way discharging efficiency
    degradation_cost_eur_per_kwh: float = 0.015  # Cycling degradation cost €/kWh throughput
    calendar_life_years: float = 15.0     # Calendar lifetime
    cycle_life_equivalent: float = 5000   # Full equivalent cycle life


@dataclass
class FleetConfig:
    """
    Electric bus fleet configuration.

    Interview talking point: Why model individual bus schedules?
    → Aggregate demand profiles miss the time-coupling: each bus
      must reach its target SoC BEFORE departure. Aggregation
      hides critical constraints and overestimates flexibility.
    """
    num_buses: int = 40
    bus_battery_kwh: float = 350.0        # Battery capacity per bus (kWh)
    min_departure_soc: float = 0.90       # Must leave depot with ≥90% SoC
    avg_consumption_kwh_per_km: float = 1.2
    avg_route_km: float = 180.0
    # Schedule patterns (Nordic city bus operations)
    morning_departure_hour: float = 5.0   # First morning departure
    morning_return_hour: float = 14.0     # Morning shift return
    evening_departure_hour: float = 14.5  # Evening shift departure
    evening_return_hour: float = 23.0     # Evening shift return
    morning_fleet_fraction: float = 0.60  # 60% do morning routes
    return_soc_mean: float = 0.30         # Average SoC when returning
    return_soc_std: float = 0.05          # Std dev of return SoC


@dataclass
class ChargerConfig:
    """Charging infrastructure configuration."""
    num_dc_fast: int = 10                 # DC fast chargers (opportunity/buffer)
    dc_fast_power_kw: float = 150.0       # kW per DC fast charger
    num_ac_depot: int = 30                # AC depot chargers (overnight/long-dwell)
    ac_depot_power_kw: float = 50.0       # kW per AC depot charger


@dataclass
class PVConfig:
    """
    On-site photovoltaic system configuration.

    Interview talking point: Why include PV in the depot model?
    → Behind-the-meter PV reduces grid import (saves on energy fees,
      grid tariff, and tax). Its intermittency also creates scheduling
      complexity that the optimiser must handle.
    """
    capacity_kwp: float = 500.0           # Peak capacity (kWp)
    latitude: float = 59.33               # Stockholm latitude
    panel_efficiency: float = 0.20
    system_losses: float = 0.14           # Inverter, wiring, soiling, etc.
    tilt_degrees: float = 30.0


@dataclass
class GridConfig:
    """Grid connection configuration."""
    max_import_kw: float = 3000.0         # Maximum grid import (kW) — 3 MW
    max_export_kw: float = 1500.0         # Maximum grid export (kW) — 1.5 MW
    connection_voltage_kv: float = 10.0   # Medium voltage connection


@dataclass
class MarketConfig:
    """
    Nordic electricity market configuration.

    Interview talking points:
    - Nord Pool operates day-ahead (Elspot) and intraday (Elbas)
    - Svenska kraftnät (SvK) procures balancing reserves:
      FCR-N (symmetric, continuous), FCR-D (asymmetric, disturbance),
      aFRR (automatic restoration), mFRR (manual restoration)
    - Grid tariff has two components: capacity (peak kW) and energy (kWh)
    - Swedish energy tax applies to consumed electricity
    """
    bidding_zone: BiddingZone = BiddingZone.SE3
    # Spot market parameters
    base_spot_price_eur_mwh: float = 45.0     # Average day-ahead price
    spot_price_volatility: float = 0.35        # Volatility scaling factor
    # Grid tariff structure (Swedish DSO model)
    grid_tariff_eur_kw_month: float = 8.50     # Peak demand charge (€/kW/month)
    grid_energy_fee_eur_kwh: float = 0.025     # Transmission + distribution energy fee
    energy_tax_eur_kwh: float = 0.005          # Energy tax
    # Reserve market premium factors (relative to spot price level)
    # These control how reserve prices relate to spot — tweak to explore
    fcr_n_premium_factor: float = 1.8          # FCR-N typically 1.5–2.5× spot equivalent
    fcr_d_up_premium_factor: float = 1.2       # FCR-D upward
    fcr_d_down_premium_factor: float = 0.8     # FCR-D downward
    afrr_premium_factor: float = 1.5           # aFRR
    mfrr_premium_factor: float = 1.0           # mFRR
    # Minimum bid sizes (kW) — real SvK requirement
    min_reserve_bid_kw: float = 100.0          # Minimum bid for any reserve product


@dataclass
class SimulationConfig:
    """Simulation control parameters."""
    horizon_hours: int = 24               # Planning horizon (hours)
    timestep_minutes: int = 60            # Timestep resolution (minutes)
    num_scenarios: int = 10               # Number of backtest scenarios
    seed: int = 42                        # Random seed for reproducibility
    # Solver settings
    solver: str = "appsi_highs"           # Primary solver (HiGHS via Pyomo APPSI)
    solver_time_limit: int = 120          # Solver time limit (seconds)
    mip_gap: float = 0.01                # Acceptable MIP optimality gap (1%)
    # Temporal context (affects prices and PV generation)
    month: int = 6                        # Month (1–12), default June
    day_of_week: int = 2                  # Day of week (0=Mon … 6=Sun), default Wednesday


@dataclass
class DepotConfig:
    """
    Complete depot configuration — aggregates all subsystem configs.

    This is the single top-level config object passed throughout the system.
    In production, this would be loaded from a versioned YAML/JSON config
    with schema validation (Pydantic BaseModel), tracked in a config registry.
    """
    bess: BESSConfig = field(default_factory=BESSConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    chargers: ChargerConfig = field(default_factory=ChargerConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
