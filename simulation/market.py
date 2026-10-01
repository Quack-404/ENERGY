"""
Nordic Electricity Market Simulator
════════════════════════════════════

Generates realistic price profiles for:
  • Nord Pool day-ahead spot market (Elspot)
  • SvK balancing reserve markets (FCR-N, FCR-D, aFRR, mFRR)

Price patterns reflect real Nordic market dynamics:
  - Morning and evening peaks driven by demand
  - Night valley (low demand, high wind/hydro)
  - Seasonal variation (winter = expensive, summer = cheap)
  - Occasional price spikes (low wind + cold snap)
  - Negative prices (surplus renewables)
  - Reserve prices correlated with spot but with independent dynamics

Interview talking point:
  "How would you generate realistic price scenarios for backtesting?"
  → Combine structural patterns (daily shape, seasonality) with
    stochastic noise and fat-tailed spikes. In production, replace
    with ML-based probabilistic forecasts (quantile regression,
    LightGBM, or normalising flows for full joint distribution).
"""
import numpy as np
import pandas as pd
from typing import Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MarketConfig, SimulationConfig


class NordPoolSimulator:
    """
    Simulates Nord Pool Elspot day-ahead prices for a Swedish bidding zone.

    The model uses a multiplicative decomposition:
        price(t) = base × daily_shape(h) × seasonal_factor(month)
                   × weekday_factor × (1 + noise) + spikes
    """

    def __init__(self, market_config: MarketConfig, sim_config: SimulationConfig):
        self.market = market_config
        self.sim = sim_config
        self.rng = np.random.RandomState(sim_config.seed)

    def generate_spot_prices(self) -> np.ndarray:
        """
        Generate hourly spot prices (EUR/MWh) for the planning horizon.

        Returns:
            np.ndarray of shape (T,) with hourly prices.
        """
        T = self.sim.horizon_hours
        hours = np.arange(T) % 24

        base = self.market.base_spot_price_eur_mwh

        # 1. Intraday shape — two-peak pattern typical for Nordic load
        daily_shape = self._daily_price_pattern(hours)

        # 2. Seasonal adjustment — Nordic: expensive winter, cheap summer
        seasonal = self._seasonal_factor(self.sim.month)

        # 3. Weekday/weekend — demand ~15% lower on weekends
        weekend = 0.85 if self.sim.day_of_week >= 5 else 1.0

        # 4. Zone premium — SE4 typically more expensive, SE1 cheapest
        zone_premium = self._zone_premium()

        # 5. Stochastic noise (log-normal for realistic distribution)
        noise = self.rng.lognormal(
            mean=0,
            sigma=self.market.spot_price_volatility * 0.15,
            size=T
        )

        # 6. Occasional price spikes (Poisson arrival, exponential magnitude)
        spikes = self._generate_price_spikes(T)

        prices = base * daily_shape * seasonal * weekend * zone_premium * noise + spikes

        # Allow negative prices (occurs in Nordics during wind surpluses)
        prices = np.maximum(prices, -50.0)

        return prices

    def _daily_price_pattern(self, hours: np.ndarray) -> np.ndarray:
        """
        Typical Nordic intraday price shape.

        Two peaks:
          - Morning: 07:00–09:00 (industry start + heating)
          - Evening: 17:00–20:00 (residential + lighting)
        Valley:
          - Night: 00:00–05:00 (low demand, baseload hydro/nuclear)
        """
        # Use a smooth sinusoidal decomposition rather than step function
        pattern = np.ones_like(hours, dtype=float)
        for i, h in enumerate(hours):
            # Base load curve (two Gaussians for peaks + constant floor)
            morning_peak = 0.35 * np.exp(-0.5 * ((h - 8) / 1.5) ** 2)
            evening_peak = 0.40 * np.exp(-0.5 * ((h - 18) / 2.0) ** 2)
            night_valley = -0.35 * np.exp(-0.5 * ((h - 3) / 2.5) ** 2)
            # Solar depression in summer afternoons
            solar_dip = -0.15 * np.exp(-0.5 * ((h - 13) / 1.5) ** 2) if self.sim.month in [5, 6, 7, 8] else 0

            pattern[i] = 1.0 + morning_peak + evening_peak + night_valley + solar_dip

        return pattern

    def _seasonal_factor(self, month: int) -> float:
        """
        Seasonal price adjustment.

        Nordic markets: winter prices ~40% above average (heating demand,
        low hydro inflow), summer prices ~30% below (high hydro, solar, low demand).
        """
        # Cosine: peak in January (month=1), trough in July (month=7)
        return 1.0 + 0.35 * np.cos(2 * np.pi * (month - 1) / 12)

    def _zone_premium(self) -> float:
        """Bidding zone price premium relative to system price."""
        premiums = {
            'SE1': 0.85,  # Hydro surplus → cheap
            'SE2': 0.90,
            'SE3': 1.00,  # Reference zone
            'SE4': 1.15,  # Import-dependent → expensive
        }
        return premiums.get(self.market.bidding_zone.value, 1.0)

    def _generate_price_spikes(self, T: int) -> np.ndarray:
        """
        Generate occasional price spikes.

        Spikes occur due to: low wind + cold snap, interconnector outage,
        nuclear maintenance, or demand forecast errors.
        """
        spikes = np.zeros(T)
        # Poisson process: ~1 spike per 24h on average
        num_spikes = self.rng.poisson(max(1, T / 24))
        if num_spikes > 0:
            spike_hours = self.rng.choice(T, size=min(num_spikes, T), replace=False)
            for h in spike_hours:
                # Exponential spike magnitude (mean 40 EUR/MWh)
                spike_magnitude = self.rng.exponential(40)
                spikes[h] = spike_magnitude
                # Spikes tend to cluster (autocorrelation)
                if h + 1 < T:
                    spikes[h + 1] += spike_magnitude * 0.5
        return spikes


class SvKReserveSimulator:
    """
    Simulates Svenska kraftnät (SvK) reserve market prices.

    Products modeled:
      FCR-N  — Frequency Containment Reserve - Normal
               Symmetric, continuous activation within 49.9–50.1 Hz
               Paid for capacity (EUR/MW/h)

      FCR-D  — Frequency Containment Reserve - Disturbance
               Asymmetric (up/down separate), activated at ±0.1 Hz
               Full activation within 5 seconds

      aFRR   — Automatic Frequency Restoration Reserve
               Automatic, full activation within 120 seconds
               Capacity + energy payment

      mFRR   — Manual Frequency Restoration Reserve
               Manual activation, 15-minute response
               Capacity + energy payment

    Interview talking point:
      "How do reserve prices relate to spot prices?"
      → Positively correlated (high spot = tight system = high reserve value)
        but with their own dynamics: FCR-N commands a premium because the
        BESS must maintain symmetric headroom (costly SoC management).
    """

    def __init__(self, market_config: MarketConfig, sim_config: SimulationConfig,
                 spot_prices: np.ndarray):
        self.market = market_config
        self.sim = sim_config
        self.spot = spot_prices
        self.rng = np.random.RandomState(sim_config.seed + 1)

    def generate_fcr_n_prices(self) -> np.ndarray:
        """
        FCR-N capacity prices (EUR/MW/h).

        FCR-N is the most valuable BESS product because:
        1. Symmetric requirement → BESS must stay near 50% SoC → opportunity cost
        2. Continuous activation → high utilisation
        3. Recently moved to marginal pricing → higher clearing prices
        """
        base = np.maximum(
            self.spot * self.market.fcr_n_premium_factor / 3,
            5.0  # Floor price
        )
        noise = self.rng.lognormal(0, 0.2, len(self.spot))
        prices = base * noise
        return np.maximum(prices, 1.0)

    def generate_fcr_d_up_prices(self) -> np.ndarray:
        """
        FCR-D upward (under-frequency) capacity prices.
        Activated when frequency drops below 49.9 Hz (generation deficit).
        """
        base = np.maximum(
            self.spot * self.market.fcr_d_up_premium_factor / 4,
            3.0
        )
        noise = self.rng.lognormal(0, 0.25, len(self.spot))
        return np.maximum(base * noise, 0.5)

    def generate_fcr_d_down_prices(self) -> np.ndarray:
        """
        FCR-D downward (over-frequency) capacity prices.
        Activated when frequency rises above 50.1 Hz (generation surplus).
        Typically cheaper than FCR-D up.
        """
        base = np.maximum(
            self.spot * self.market.fcr_d_down_premium_factor / 5,
            2.0
        )
        noise = self.rng.lognormal(0, 0.2, len(self.spot))
        return np.maximum(base * noise, 0.3)

    def generate_afrr_prices(self) -> np.ndarray:
        """
        aFRR capacity prices.
        Slower response (120s) but often lucrative capacity payments.
        """
        base = np.maximum(
            self.spot * self.market.afrr_premium_factor / 3.5,
            4.0
        )
        noise = self.rng.lognormal(0, 0.2, len(self.spot))
        return np.maximum(base * noise, 1.0)

    def generate_mfrr_prices(self) -> np.ndarray:
        """
        mFRR capacity prices.
        Manual activation (15 min), lowest capacity value for BESS.
        """
        base = np.maximum(
            self.spot * self.market.mfrr_premium_factor / 5,
            2.0
        )
        noise = self.rng.lognormal(0, 0.2, len(self.spot))
        return np.maximum(base * noise, 0.5)


def generate_all_market_data(market_config: MarketConfig,
                              sim_config: SimulationConfig) -> Dict[str, np.ndarray]:
    """
    Generate complete market data for a simulation run.

    Returns dict with keys:
        spot_price, fcr_n_price, fcr_d_up_price, fcr_d_down_price,
        afrr_price, mfrr_price
    All arrays of shape (T,) in EUR/MWh or EUR/MW/h.
    """
    nordpool = NordPoolSimulator(market_config, sim_config)
    spot_prices = nordpool.generate_spot_prices()

    svk = SvKReserveSimulator(market_config, sim_config, spot_prices)

    return {
        'spot_price': spot_prices,
        'fcr_n_price': svk.generate_fcr_n_prices(),
        'fcr_d_up_price': svk.generate_fcr_d_up_prices(),
        'fcr_d_down_price': svk.generate_fcr_d_down_prices(),
        'afrr_price': svk.generate_afrr_prices(),
        'mfrr_price': svk.generate_mfrr_prices(),
    }
