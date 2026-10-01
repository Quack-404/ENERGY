"""
Electric Bus Depot Model
════════════════════════

Models the physical depot with:
  • Bus fleet schedules (arrivals, departures, SoC requirements)
  • Aggregate charging demand profile
  • On-site PV generation
  • Charger infrastructure constraints

Interview talking point:
  "What makes depot optimisation harder than a standalone BESS?"
  → The fleet imposes hard time-coupled constraints: each bus MUST
    reach its target SoC before departure. These are not soft constraints
    — a bus that can't complete its route is a contract violation and
    a safety issue. The optimiser must respect transit schedules as
    inviolable while finding flexibility in the remaining degrees of freedom.
"""
import numpy as np
import pandas as pd
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import FleetConfig, ChargerConfig, PVConfig, SimulationConfig


class FleetModel:
    """
    Electric bus fleet model.

    Generates individual bus schedules and computes the aggregate
    minimum charging demand that the optimiser must satisfy at each timestep.
    """

    def __init__(self, fleet_config: FleetConfig, charger_config: ChargerConfig,
                 sim_config: SimulationConfig):
        self.fleet = fleet_config
        self.chargers = charger_config
        self.sim = sim_config
        self.rng = np.random.RandomState(sim_config.seed + 10)

    def generate_bus_schedule(self) -> pd.DataFrame:
        """
        Generate individual bus schedules.

        Each bus has:
          - route_type: 'morning' or 'evening'
          - departure_hour: when it leaves the depot
          - return_hour: when it returns
          - return_soc: SoC when it arrives back at depot
          - required_departure_soc: minimum SoC needed before departure
          - energy_to_charge: kWh that must be added before next departure
        """
        schedules = []
        num_morning = int(self.fleet.num_buses * self.fleet.morning_fleet_fraction)
        num_evening = self.fleet.num_buses - num_morning

        for i in range(num_morning):
            departure = self.fleet.morning_departure_hour + self.rng.uniform(-0.25, 0.25)
            arrival = self.fleet.morning_return_hour + self.rng.uniform(-0.5, 0.5)
            return_soc = np.clip(
                self.rng.normal(self.fleet.return_soc_mean, self.fleet.return_soc_std),
                0.10, 0.50
            )
            energy_needed = (self.fleet.min_departure_soc - return_soc) * self.fleet.bus_battery_kwh
            schedules.append({
                'bus_id': f'BUS_{i:03d}',
                'route_type': 'morning',
                'departure_hour': round(departure, 2),
                'return_hour': round(arrival, 2),
                'return_soc': round(return_soc, 3),
                'required_departure_soc': self.fleet.min_departure_soc,
                'battery_kwh': self.fleet.bus_battery_kwh,
                'energy_to_charge_kwh': round(max(0, energy_needed), 1),
            })

        for i in range(num_evening):
            departure = self.fleet.evening_departure_hour + self.rng.uniform(-0.25, 0.25)
            arrival = self.fleet.evening_return_hour + self.rng.uniform(-0.5, 0.5)
            return_soc = np.clip(
                self.rng.normal(self.fleet.return_soc_mean, self.fleet.return_soc_std),
                0.10, 0.50
            )
            energy_needed = (self.fleet.min_departure_soc - return_soc) * self.fleet.bus_battery_kwh
            schedules.append({
                'bus_id': f'BUS_{num_morning + i:03d}',
                'route_type': 'evening',
                'departure_hour': round(departure, 2),
                'return_hour': round(arrival, 2),
                'return_soc': round(return_soc, 3),
                'required_departure_soc': self.fleet.min_departure_soc,
                'battery_kwh': self.fleet.bus_battery_kwh,
                'energy_to_charge_kwh': round(max(0, energy_needed), 1),
            })

        return pd.DataFrame(schedules)

    def compute_charging_demand(self, schedule: pd.DataFrame = None) -> np.ndarray:
        """
        Compute aggregate minimum charging demand profile (kW) per timestep.

        This represents the non-negotiable load: buses MUST be charged.
        The optimiser can shift WHEN within the dwell window, but the
        total energy must be delivered before departure.

        For this simulation, we model a "baseline" (unoptimised) demand
        that charges each bus as soon as it arrives — the optimiser's job
        is to reshape this within constraints.
        """
        if schedule is None:
            schedule = self.generate_bus_schedule()

        T = self.sim.horizon_hours
        dt = self.sim.timestep_minutes / 60.0
        demand = np.zeros(T)

        # Max available charger power (aggregate)
        max_ac_power = self.chargers.num_ac_depot * self.chargers.ac_depot_power_kw
        max_dc_power = self.chargers.num_dc_fast * self.chargers.dc_fast_power_kw
        max_total_charger_power = max_ac_power + max_dc_power

        for _, bus in schedule.iterrows():
            energy_needed = bus['energy_to_charge_kwh']
            if energy_needed <= 0:
                continue

            if bus['route_type'] == 'morning':
                # Morning buses return ~14:00, must be ready by ~05:00 next day
                # Dwell window: return_hour → 24 (end of horizon for 24h sim)
                # In a rolling horizon, this would wrap to next day
                charge_start = int(np.floor(bus['return_hour']))
                charge_end = T  # Charge until end of horizon

                available_hours = max(1, charge_end - charge_start)
                # Spread charging evenly across dwell window (baseline strategy)
                charge_rate = min(
                    energy_needed / (available_hours * dt),
                    self.chargers.ac_depot_power_kw  # Limited by charger
                )

                remaining = energy_needed
                for t in range(charge_start, min(charge_end, T)):
                    if remaining <= 0:
                        break
                    actual_rate = min(charge_rate, remaining / dt)
                    demand[t] += actual_rate
                    remaining -= actual_rate * dt

            elif bus['route_type'] == 'evening':
                # Evening buses return ~23:00, must be ready by ~14:30 next day
                # Charge overnight: return_hour → 24, then 0 → departure_hour
                charge_start = int(np.floor(min(bus['return_hour'], T - 1)))
                departure_next = int(np.floor(self.fleet.evening_departure_hour))

                # Calculate total available hours (wrapping around midnight)
                hours_tonight = max(0, T - charge_start)
                hours_morning = departure_next  # 0 to departure
                available_hours = max(1, hours_tonight + hours_morning)

                charge_rate = min(
                    energy_needed / (available_hours * dt),
                    self.chargers.ac_depot_power_kw
                )

                remaining = energy_needed
                # Tonight
                for t in range(charge_start, T):
                    if remaining <= 0:
                        break
                    actual_rate = min(charge_rate, remaining / dt)
                    demand[t] += actual_rate
                    remaining -= actual_rate * dt

                # Early morning (wraps around in 24h horizon)
                for t in range(0, min(departure_next, T)):
                    if remaining <= 0:
                        break
                    actual_rate = min(charge_rate, remaining / dt)
                    demand[t] += actual_rate
                    remaining -= actual_rate * dt

        # Cap at total charger capacity
        demand = np.minimum(demand, max_total_charger_power)

        return demand

    def compute_flexibility_window(self, schedule: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the flexible charging envelope: min and max demand at each timestep.

        min_demand[t] = minimum charging that must happen at t (considering deadlines)
        max_demand[t] = maximum possible charging at t (all available chargers)

        The optimiser can schedule anywhere within this envelope.
        """
        if schedule is None:
            schedule = self.generate_bus_schedule()

        T = self.sim.horizon_hours

        # Minimum: essential charging to meet all departure deadlines
        # (computed as the baseline demand — must deliver this total energy)
        min_demand = np.zeros(T)

        # Maximum: all chargers running at full power when buses are present
        max_demand = np.zeros(T)
        max_charger_power = (self.chargers.num_ac_depot * self.chargers.ac_depot_power_kw +
                             self.chargers.num_dc_fast * self.chargers.dc_fast_power_kw)

        for _, bus in schedule.iterrows():
            return_hour = int(np.floor(bus['return_hour']))
            if bus['route_type'] == 'morning':
                # Bus present from return to end of horizon
                for t in range(min(return_hour, T), T):
                    max_demand[t] += self.chargers.dc_fast_power_kw  # One charger per bus
            else:
                # Evening: present from return to end, then start of day to departure
                for t in range(min(return_hour, T), T):
                    max_demand[t] += self.chargers.ac_depot_power_kw
                departure = int(np.floor(self.fleet.evening_departure_hour))
                for t in range(0, min(departure, T)):
                    max_demand[t] += self.chargers.ac_depot_power_kw

        max_demand = np.minimum(max_demand, max_charger_power)

        return min_demand, max_demand


class PVModel:
    """
    On-site photovoltaic generation model.

    Uses a clear-sky sinusoidal model with cloud factor variation.
    In production, replace with weather-API-driven forecasts or
    satellite-based nowcasting.
    """

    def __init__(self, pv_config: PVConfig, sim_config: SimulationConfig):
        self.pv = pv_config
        self.sim = sim_config
        self.rng = np.random.RandomState(sim_config.seed + 20)

    def generate_production(self) -> np.ndarray:
        """
        Generate PV production profile (kW) for the planning horizon.

        Uses solar geometry for the given latitude and month to compute
        sunrise/sunset, then applies a sinusoidal clear-sky model with
        stochastic cloud cover.
        """
        T = self.sim.horizon_hours
        hours = np.arange(T) % 24

        # Compute day length for this month and latitude
        day_length = self._day_length(self.sim.month, self.pv.latitude)
        sunrise = 12 - day_length / 2
        sunset = 12 + day_length / 2

        production = np.zeros(T)
        for i, h in enumerate(hours):
            if sunrise < h < sunset:
                # Clear-sky irradiance: sinusoidal curve
                hour_angle = np.pi * (h - sunrise) / (sunset - sunrise)
                clear_sky_kw = self.pv.capacity_kwp * np.sin(hour_angle)

                # Stochastic cloud factor (beta distribution: mostly clear)
                cloud_factor = self.rng.beta(7, 2)  # Mean ~0.78

                # Seasonal irradiance factor (less in winter, more in summer)
                seasonal = 0.5 + 0.5 * np.sin(np.pi * (self.sim.month - 3) / 6)
                seasonal = np.clip(seasonal, 0.2, 1.0)

                production[i] = (clear_sky_kw * cloud_factor * seasonal *
                                 (1 - self.pv.system_losses))

        return np.maximum(production, 0)

    def _day_length(self, month: int, latitude: float) -> float:
        """
        Approximate day length in hours for given month and latitude.

        Uses the solar declination angle formula. Critical for Nordic
        latitudes where day length varies dramatically:
          Stockholm June: ~18.5h daylight
          Stockholm December: ~6h daylight
        """
        # Day of year (approximate, mid-month)
        day_of_year = (month - 1) * 30.44 + 15
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 / 365 * (284 + day_of_year)))
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)

        cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)

        day_length_hours = 2 * np.degrees(np.arccos(cos_hour_angle)) / 15
        return day_length_hours
