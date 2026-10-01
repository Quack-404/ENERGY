"""
⚡ Energy Market Optimisation Simulator — Interactive Dashboard
═══════════════════════════════════════════════════════════════

Streamlit-based interactive dashboard for exploring the MILP dispatch
optimiser and preparing for the Energy Market Optimisation Lead interview.

Run with:  streamlit run dashboard.py
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import sys
import os

# Ensure imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DepotConfig, BESSConfig, FleetConfig, ChargerConfig,
    PVConfig, GridConfig, MarketConfig, SimulationConfig, BiddingZone
)
from simulation.market import generate_all_market_data
from simulation.depot import FleetModel, PVModel
from simulation.optimizer import DepotOptimizer
from simulation.analysis import ValueStackAnalyzer, Backtester, SensitivityAnalyzer
from simulation.forecasting import (
    MarketFeatureDataset, ModelSuite, ModelResponseAnalyzer,
    evaluate_all_models, evaluate_forecast_dispatch_cost, get_model_governance_matrix
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Energy Market Optimisation Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(44, 83, 100, 0.5);
    }
    .main-header h1 {
        color: #e0f7fa;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }
    .main-header p {
        color: #80cbc4;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a2332 0%, #0d1b2a 100%);
        border: 1px solid #1b3a4b;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00bcd4;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00e5ff;
        margin: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #78909c;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .metric-value.positive { color: #69f0ae; }
    .metric-value.negative { color: #ff5252; }
    .metric-value.neutral  { color: #ffd740; }

    /* Interview prep cards */
    .interview-card {
        background: linear-gradient(145deg, #1a2332 0%, #0d1b2a 100%);
        border-left: 4px solid #00bcd4;
        border-radius: 0 10px 10px 0;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .interview-card h4 {
        color: #00e5ff;
        margin: 0 0 0.5rem 0;
    }
    .interview-card p {
        color: #b0bec5;
        margin: 0;
        line-height: 1.6;
    }

    /* Tab styling enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1b2a 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #00e5ff;
        font-size: 1.3rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #4dd0e1;
        font-size: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>⚡ Energy Market Optimisation Simulator</h1>
    <p>MILP Dispatch Optimiser for Electric Bus Depot + BESS — Nordic Markets (Nord Pool / SvK)</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ System Config")

    st.markdown("---")
    st.subheader("🔋 BESS")
    bess_capacity = st.slider("Capacity (kWh)", 500, 5000, 2000, 100,
                               help="Total BESS energy capacity")
    bess_power = st.slider("Max Power (kW)", 100, 2500, 1000, 50,
                            help="Max charge/discharge rate")
    bess_min_soc = st.slider("Min SoC (%)", 5, 30, 10, help="Lower SoC bound — protects cycle life")
    bess_max_soc = st.slider("Max SoC (%)", 70, 98, 90, help="Upper SoC bound — protects calendar life")
    bess_initial_soc = st.slider("Initial SoC (%)", 10, 90, 50)
    bess_rte = st.slider("Round-trip Efficiency (%)", 80, 98, 90,
                          help="η_charge × η_discharge")
    bess_deg = st.slider("Degradation Cost (€¢/kWh)", 0, 5, 2,
                          help="Cost per kWh of throughput")

    st.markdown("---")
    st.subheader("🚌 Fleet")
    num_buses = st.slider("Number of Buses", 10, 100, 40)
    bus_battery = st.slider("Bus Battery (kWh)", 200, 600, 350, 10)
    morning_fraction = st.slider("Morning Shift (%)", 30, 80, 60,
                                  help="Fraction of fleet doing morning routes")

    st.markdown("---")
    st.subheader("☀️ PV System")
    pv_capacity = st.slider("PV Capacity (kWp)", 0, 1500, 500, 50)

    st.markdown("---")
    st.subheader("🔌 Grid")
    grid_import_max = st.slider("Max Import (kW)", 1000, 6000, 3000, 100)
    grid_export_max = st.slider("Max Export (kW)", 0, 3000, 1500, 100)

    st.markdown("---")
    st.subheader("📈 Market")
    bidding_zone = st.selectbox("Bidding Zone", ['SE1', 'SE2', 'SE3', 'SE4'], index=2)
    spot_price_base = st.slider("Avg Spot Price (€/MWh)", 10, 200, 45, 5)
    spot_volatility = st.slider("Price Volatility", 0.10, 0.80, 0.35, 0.05)
    grid_tariff = st.slider("Peak Tariff (€/kW/mo)", 2.0, 25.0, 8.5, 0.5)
    fcr_n_premium = st.slider("FCR-N Premium Factor", 0.5, 4.0, 1.8, 0.1,
                               help="FCR-N price relative to spot")

    st.markdown("---")
    st.subheader("⏱️ Simulation")
    horizon = st.selectbox("Horizon (hours)", [24, 48, 72], index=0)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month = st.selectbox("Month", list(range(1, 13)), index=5,
                          format_func=lambda m: month_names[m - 1])
    day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

# ══════════════════════════════════════════════════════════════
# BUILD CONFIG FROM SIDEBAR
# ══════════════════════════════════════════════════════════════
eta_one_way = np.sqrt(bess_rte / 100.0)
config = DepotConfig(
    bess=BESSConfig(
        capacity_kwh=float(bess_capacity),
        max_power_kw=float(bess_power),
        min_soc=bess_min_soc / 100.0,
        max_soc=bess_max_soc / 100.0,
        initial_soc=bess_initial_soc / 100.0,
        charge_efficiency=eta_one_way,
        discharge_efficiency=eta_one_way,
        degradation_cost_eur_per_kwh=bess_deg / 100.0,
    ),
    fleet=FleetConfig(
        num_buses=num_buses,
        bus_battery_kwh=float(bus_battery),
        morning_fleet_fraction=morning_fraction / 100.0,
    ),
    chargers=ChargerConfig(),
    pv=PVConfig(capacity_kwp=float(pv_capacity)),
    grid=GridConfig(max_import_kw=float(grid_import_max), max_export_kw=float(grid_export_max)),
    market=MarketConfig(
        bidding_zone=BiddingZone(bidding_zone),
        base_spot_price_eur_mwh=float(spot_price_base),
        spot_price_volatility=spot_volatility,
        grid_tariff_eur_kw_month=grid_tariff,
        fcr_n_premium_factor=fcr_n_premium,
    ),
    simulation=SimulationConfig(
        horizon_hours=horizon,
        month=month,
        day_of_week=0 if day_type == "Weekday" else 5,
    ),
)

# ══════════════════════════════════════════════════════════════
# GENERATE SIMULATION DATA
# ══════════════════════════════════════════════════════════════

def _config_hash(c: DepotConfig) -> str:
    """Create a hash string for config caching."""
    parts = [
        c.bess.capacity_kwh, c.bess.max_power_kw, c.bess.min_soc, c.bess.max_soc,
        c.bess.initial_soc, c.bess.charge_efficiency, c.bess.degradation_cost_eur_per_kwh,
        c.fleet.num_buses, c.fleet.bus_battery_kwh, c.fleet.morning_fleet_fraction,
        c.pv.capacity_kwp, c.grid.max_import_kw, c.grid.max_export_kw,
        c.market.bidding_zone.value, c.market.base_spot_price_eur_mwh,
        c.market.spot_price_volatility, c.market.grid_tariff_eur_kw_month,
        c.market.fcr_n_premium_factor,
        c.simulation.horizon_hours, c.simulation.month, c.simulation.day_of_week,
        c.simulation.seed,
    ]
    return str(parts)


@st.cache_data
def generate_data(config_hash_str):
    """Generate market, fleet, and PV data (cached on config hash)."""
    # We use st.session_state to pass the actual config object
    # The hash string is just for cache invalidation
    cfg = st.session_state['_current_config']
    market_data = generate_all_market_data(cfg.market, cfg.simulation)
    fleet = FleetModel(cfg.fleet, cfg.chargers, cfg.simulation)
    schedule = fleet.generate_bus_schedule()
    bus_demand = fleet.compute_charging_demand(schedule)
    pv_model = PVModel(cfg.pv, cfg.simulation)
    pv_gen = pv_model.generate_production()
    return market_data, schedule, bus_demand, pv_gen


@st.cache_data
def get_forecasting_data(seed: int, base_price: float, horizon_hours: int):
    """Train MLR, XGBoost, and ANN models on Nordic market features."""
    ds_gen = MarketFeatureDataset(base_price=base_price, seed=seed)
    df = ds_gen.generate_historical_dataset(n_days=90)
    train_df = df.iloc[:1800]
    test_df = df.iloc[1800:1800 + horizon_hours].copy()
    suite = ModelSuite(random_state=seed)
    suite.fit(train_df)
    preds = suite.predict_all(test_df)
    metrics_df = evaluate_all_models(test_df, preds)
    analyzer = ModelResponseAnalyzer(suite)
    gov_matrix = get_model_governance_matrix()
    return train_df, test_df, preds, metrics_df, suite, analyzer, gov_matrix


# Store config in session state for the cached function to access
st.session_state['_current_config'] = config
cfg_hash = _config_hash(config)
market_data, schedule, bus_demand, pv_gen = generate_data(cfg_hash)
ml_train_df, ml_test_df, ml_preds, ml_metrics, ml_suite, ml_analyzer, ml_gov_matrix = get_forecasting_data(
    config.simulation.seed, config.market.base_spot_price_eur_mwh, config.simulation.horizon_hours
)
hours = np.arange(config.simulation.horizon_hours)

# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,22,40,0.8)',
    font=dict(family='Inter', color='#b0bec5'),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)

COLORS = {
    'spot': '#ffd740',
    'fcr_n': '#00e5ff',
    'fcr_d_up': '#69f0ae',
    'fcr_d_down': '#b388ff',
    'afrr': '#ff8a80',
    'grid_import': '#42a5f5',
    'grid_export': '#ef5350',
    'bess_charge': '#26c6da',
    'bess_discharge': '#ffa726',
    'pv': '#ffee58',
    'bus_demand': '#78909c',
    'soc': '#00e5ff',
    'peak': '#ff5252',
    'mlr': '#b388ff',
    'xgboost': '#00e5ff',
    'ann': '#ff9100',
}


# ══════════════════════════════════════════════════════════════
# TAB CONTENT
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Market Data",
    "🤖 ML Forecasting (MLR, XGBoost, ANN)",
    "🚌 Fleet & Depot",
    "⚡ Optimisation",
    "💰 Value Stack",
    "📈 Sensitivity",
    "🎯 Interview Prep",
])

# ─────────────────────────────────────────────────────────────
# TAB 1: MARKET DATA
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Nordic Electricity Market Prices")
    st.caption(f"Bidding Zone: {config.market.bidding_zone.value} | "
               f"Month: {month_names[month - 1]} | "
               f"{'Weekend' if day_type == 'Weekend' else 'Weekday'}")

    # Spot price chart
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(
        x=hours, y=market_data['spot_price'],
        mode='lines', name='Day-Ahead Spot',
        line=dict(color=COLORS['spot'], width=2.5),
        fill='tozeroy', fillcolor='rgba(255,215,64,0.1)',
    ))
    fig_spot.update_layout(
        **PLOTLY_LAYOUT,
        title='Nord Pool Day-Ahead Spot Price',
        xaxis_title='Hour',
        yaxis_title='EUR/MWh',
        height=350,
    )
    st.plotly_chart(fig_spot, use_container_width=True)

    # Reserve prices
    col1, col2 = st.columns(2)
    with col1:
        fig_fcr = go.Figure()
        fig_fcr.add_trace(go.Scatter(
            x=hours, y=market_data['fcr_n_price'],
            mode='lines', name='FCR-N',
            line=dict(color=COLORS['fcr_n'], width=2),
        ))
        fig_fcr.add_trace(go.Scatter(
            x=hours, y=market_data['fcr_d_up_price'],
            mode='lines', name='FCR-D Up',
            line=dict(color=COLORS['fcr_d_up'], width=2),
        ))
        fig_fcr.add_trace(go.Scatter(
            x=hours, y=market_data['fcr_d_down_price'],
            mode='lines', name='FCR-D Down',
            line=dict(color=COLORS['fcr_d_down'], width=2, dash='dash'),
        ))
        fig_fcr.update_layout(
            **PLOTLY_LAYOUT,
            title='Frequency Containment Reserves (FCR)',
            xaxis_title='Hour', yaxis_title='EUR/MW/h',
            height=300,
        )
        st.plotly_chart(fig_fcr, use_container_width=True)

    with col2:
        fig_afrr = go.Figure()
        fig_afrr.add_trace(go.Scatter(
            x=hours, y=market_data['afrr_price'],
            mode='lines', name='aFRR',
            line=dict(color=COLORS['afrr'], width=2),
        ))
        fig_afrr.add_trace(go.Scatter(
            x=hours, y=market_data['mfrr_price'],
            mode='lines', name='mFRR',
            line=dict(color='#ce93d8', width=2, dash='dash'),
        ))
        fig_afrr.update_layout(
            **PLOTLY_LAYOUT,
            title='Restoration Reserves (aFRR / mFRR)',
            xaxis_title='Hour', yaxis_title='EUR/MW/h',
            height=300,
        )
        st.plotly_chart(fig_afrr, use_container_width=True)

    # Price statistics
    st.subheader("Price Statistics")
    stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Price Spread'],
        'Spot (€/MWh)': [
            f"{np.mean(market_data['spot_price']):.1f}",
            f"{np.median(market_data['spot_price']):.1f}",
            f"{np.min(market_data['spot_price']):.1f}",
            f"{np.max(market_data['spot_price']):.1f}",
            f"{np.std(market_data['spot_price']):.1f}",
            f"{np.max(market_data['spot_price']) - np.min(market_data['spot_price']):.1f}",
        ],
        'FCR-N (€/MW/h)': [
            f"{np.mean(market_data['fcr_n_price']):.1f}",
            f"{np.median(market_data['fcr_n_price']):.1f}",
            f"{np.min(market_data['fcr_n_price']):.1f}",
            f"{np.max(market_data['fcr_n_price']):.1f}",
            f"{np.std(market_data['fcr_n_price']):.1f}",
            f"{np.max(market_data['fcr_n_price']) - np.min(market_data['fcr_n_price']):.1f}",
        ],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 2: ML FORECASTING (MLR vs XGBoost vs ANN)
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("🤖 Machine Learning Forecasting: MLR vs XGBoost vs ANN")
    st.caption("Comparing econometric, gradient boosting, and deep learning architectures for Nordic Day-Ahead price formation, physical non-linear sensitivities, and downstream dispatch optimization.")

    # High-level metric KPI cards
    best_row = ml_metrics.sort_values(by='RMSE (€/MWh)').iloc[0]
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value positive">{best_row['Model']}</div>
            <div class="metric-label">Best Performing Model (RMSE: €{best_row['RMSE (€/MWh)']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c2:
        mlr_r2 = ml_metrics[ml_metrics['Model'] == 'MLR']['R² Score'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mlr_r2:.3f}</div>
            <div class="metric-label">MLR R² (White-Box Baseline)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c3:
        xgb_r2 = ml_metrics[ml_metrics['Model'] == 'XGBoost']['R² Score'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value positive">{xgb_r2:.3f}</div>
            <div class="metric-label">XGBoost R² (Industry Standard)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c4:
        ann_r2 = ml_metrics[ml_metrics['Model'] == 'ANN']['R² Score'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value neutral">{ann_r2:.3f}</div>
            <div class="metric-label">ANN R² (Deep MLP)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Overview Cards
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.markdown("""
        <div class="interview-card">
            <h4>📐 MLR (Multiple Linear Regression)</h4>
            <p><b>Inductive Bias:</b> Hyper-planar equation: <code>y = β₀ + ∑ βᵢ xᵢ</code><br>
            <b>Key Strength:</b> 100% white-box explainability under EU AI Act. Instant closed-form training, transparent feature elasticities (€/MWh per m/s wind).<br>
            <b>Critical Flaw:</b> Incapable of capturing non-linear merit-order cannibalisation or freezing heating surges without manual polynomial feature engineering.</p>
        </div>
        """, unsafe_allow_html=True)
    with arch_col2:
        st.markdown("""
        <div class="interview-card">
            <h4>🌲 XGBoost (Gradient Boosted Trees)</h4>
            <p><b>Inductive Bias:</b> Ensemble of decision trees: <code>y = ∑ f_k(x)</code><br>
            <b>Key Strength:</b> Captures sharp threshold step functions (e.g. freezing temp degree-day jumps, negative price cliffs at high wind). Primary JD technology.<br>
            <b>Auditability:</b> Supported by exact TreeSHAP values, feature split gain ranking, and monotonicity constraints.</p>
        </div>
        """, unsafe_allow_html=True)
    with arch_col3:
        st.markdown("""
        <div class="interview-card">
            <h4>🧠 ANN (Artificial Neural Network / MLP)</h4>
            <p><b>Inductive Bias:</b> Deep layered activations: <code>y = σ(W₂ σ(W₁ x + b₁) + b₂)</code><br>
            <b>Key Strength:</b> Continuous smooth non-linear surfaces. Naturally captures multi-way feature interactions (e.g. cold temp + low wind + peak hour).<br>
            <b>Governance:</b> Classified as higher-risk black-box under EU AI Act; requires post-hoc Integrated Gradients and drift guardrails.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # SECTION 1: MODEL RESPONSE CURVES & SENSITIVITY
    st.subheader("1. 📈 Physical Response Curves & Marginal Sensitivities")
    st.caption("Inspect how each model responds when varying fundamental market drivers while holding other features constant.")

    resp_tab1, resp_tab2, resp_tab3, resp_tab4 = st.tabs([
        "❄️ Temperature Response",
        "💨 Wind Speed Response",
        "⏰ Intraday 24h Profile",
        "⚡ Extreme Stress Regimes"
    ])

    with resp_tab1:
        st.markdown("""
        **Physics of Temperature Response in Nordic Power Markets:**
        Electricity demand exhibits a pronounced **"hockey-stick"** non-linear curve.
        Between 10°C and 20°C, demand is flat. When temperature drops below freezing (0°C to -20°C),
        air-source heat pumps lose COP efficiency and auxiliary resistive heating activates, triggering exponential price surges.
        """)
        df_temp = ml_analyzer.compute_temperature_response()
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_temp['temperature'], y=df_temp['MLR'], mode='lines', name='MLR (Linear)', line=dict(color=COLORS['mlr'], width=2.5, dash='dash')))
        fig_temp.add_trace(go.Scatter(x=df_temp['temperature'], y=df_temp['XGBoost'], mode='lines', name='XGBoost (Threshold Steps)', line=dict(color=COLORS['xgboost'], width=3)))
        fig_temp.add_trace(go.Scatter(x=df_temp['temperature'], y=df_temp['ANN'], mode='lines', name='ANN (Smooth Non-linear)', line=dict(color=COLORS['ann'], width=2.5)))
        fig_temp.add_vline(x=0, line_width=1, line_dash="dot", line_color="#ff5252", annotation_text="0°C Freezing Threshold", annotation_position="top right")
        fig_temp.update_layout(**PLOTLY_LAYOUT, title='Model Price Response vs Ambient Temperature (°C)', xaxis_title='Ambient Temperature (°C)', yaxis_title='Predicted Spot Price (EUR/MWh)', height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.info("💡 **Interview Takeaway**: Notice MLR's rigid straight line misses the winter peak price spike by over €40/MWh. XGBoost and ANN accurately capture the non-linear cold snap demand explosion.")

    with resp_tab2:
        st.markdown("""
        **Physics of Wind Speed & The Renewable Merit-Order Effect:**
        Nordic wind generation follows a cubic power curve: P ∝ v³ between cut-in (3 m/s) and rated (12 m/s).
        As wind generation floods the grid, zero-marginal-cost generation pushes thermal generation to the right of the merit-order stack,
        crashing prices and causing negative prices during high wind + low demand periods.
        """)
        df_wind = ml_analyzer.compute_wind_response()
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(x=df_wind['wind_speed'], y=df_wind['MLR'], mode='lines', name='MLR (Linear)', line=dict(color=COLORS['mlr'], width=2.5, dash='dash')))
        fig_wind.add_trace(go.Scatter(x=df_wind['wind_speed'], y=df_wind['XGBoost'], mode='lines', name='XGBoost (Merit-Order Steps)', line=dict(color=COLORS['xgboost'], width=3)))
        fig_wind.add_trace(go.Scatter(x=df_wind['wind_speed'], y=df_wind['ANN'], mode='lines', name='ANN (S-Curve Saturation)', line=dict(color=COLORS['ann'], width=2.5)))
        fig_wind.add_vline(x=3, line_width=1, line_dash="dot", line_color="#80cbc4", annotation_text="Cut-in (3 m/s)", annotation_position="bottom right")
        fig_wind.add_vline(x=12, line_width=1, line_dash="dot", line_color="#ffd740", annotation_text="Rated Power (12 m/s)", annotation_position="top right")
        fig_wind.update_layout(**PLOTLY_LAYOUT, title='Model Price Response vs Wind Speed (m/s)', xaxis_title='Wind Speed (m/s)', yaxis_title='Predicted Spot Price (EUR/MWh)', height=400)
        st.plotly_chart(fig_wind, use_container_width=True)
        st.info("💡 **Interview Takeaway**: MLR plunges into unconstrained negative values because it lacks saturation gates. XGBoost and ANN correctly reflect turbine power plateaus and market curtailment floors.")

    with resp_tab3:
        st.markdown("**Intraday Load Curve Reconstruction:**")
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            wknd_toggle = st.radio("Day Type", ["Weekday", "Weekend"], horizontal=True, key='ml_wknd_toggle')
        df_intra = ml_analyzer.compute_intraday_response(is_weekend=(wknd_toggle == "Weekend"))
        fig_intra = go.Figure()
        fig_intra.add_trace(go.Scatter(x=df_intra['hour'], y=df_intra['MLR'], mode='lines', name='MLR', line=dict(color=COLORS['mlr'], width=2, dash='dash')))
        fig_intra.add_trace(go.Scatter(x=df_intra['hour'], y=df_intra['XGBoost'], mode='lines', name='XGBoost', line=dict(color=COLORS['xgboost'], width=2.5)))
        fig_intra.add_trace(go.Scatter(x=df_intra['hour'], y=df_intra['ANN'], mode='lines', name='ANN', line=dict(color=COLORS['ann'], width=2.5)))
        fig_intra.update_layout(**PLOTLY_LAYOUT, title=f'24-Hour Price Profile ({wknd_toggle})', xaxis_title='Hour of Day', yaxis_title='Predicted Price (EUR/MWh)', height=380)
        st.plotly_chart(fig_intra, use_container_width=True)

    with resp_tab4:
        st.markdown("**Response to Critical Nordic Market Stress Scenarios:**")
        df_stress = ml_analyzer.compute_stress_test_responses()
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

        fig_stress = go.Figure()
        scenarios = df_stress['Scenario'].tolist()
        fig_stress.add_trace(go.Bar(name='MLR', x=scenarios, y=df_stress['MLR Predicted (€/MWh)'], marker_color=COLORS['mlr']))
        fig_stress.add_trace(go.Bar(name='XGBoost', x=scenarios, y=df_stress['XGBoost Predicted (€/MWh)'], marker_color=COLORS['xgboost']))
        fig_stress.add_trace(go.Bar(name='ANN', x=scenarios, y=df_stress['ANN Predicted (€/MWh)'], marker_color=COLORS['ann']))
        fig_stress.update_layout(**PLOTLY_LAYOUT, barmode='group', title='Stress Test Predicted Prices by Scenario', yaxis_title='EUR/MWh', height=380)
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("---")

    # SECTION 2: ACCURACY BENCHMARK & RESIDUALS
    st.subheader("2. 🎯 Out-of-Sample Accuracy Benchmarking")
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        st.markdown("**Test Set Evaluation Metrics (24h Out-of-Sample):**")
        st.dataframe(ml_metrics, use_container_width=True, hide_index=True)
    with b_col2:
        st.markdown("""
        **Forecasting Takeaways for the Energy Optimisation Lead Role:**
        - **XGBoost** achieves the highest $R^2$ and lowest RMSE because electricity prices are governed by supply-curve steps and discrete threshold dispatch logic (e.g. wind generation thresholds, fuel switching).
        - **MLR** serves as an indispensable regulatory white-box benchmark and safety envelope fallback.
        - **ANN** excels at learning continuous multi-horizon sequences, but requires strong regularisation to avoid overfitting to transient price spikes.
        """)

    # 24h Forecast vs Actual Chart
    fig_f_vs_a = go.Figure()
    fig_f_vs_a.add_trace(go.Scatter(x=hours, y=ml_test_df['spot_price'], mode='lines+markers', name='Ground Truth Spot', line=dict(color='#ffd740', width=3.5)))
    fig_f_vs_a.add_trace(go.Scatter(x=hours, y=ml_preds['XGBoost'], mode='lines', name='XGBoost Forecast', line=dict(color=COLORS['xgboost'], width=2.5)))
    fig_f_vs_a.add_trace(go.Scatter(x=hours, y=ml_preds['ANN'], mode='lines', name='ANN Forecast', line=dict(color=COLORS['ann'], width=2)))
    fig_f_vs_a.add_trace(go.Scatter(x=hours, y=ml_preds['MLR'], mode='lines', name='MLR Forecast', line=dict(color=COLORS['mlr'], width=2, dash='dash')))
    fig_f_vs_a.update_layout(**PLOTLY_LAYOUT, title='Out-of-Sample 24h Spot Price: Forecast vs Realized', xaxis_title='Hour', yaxis_title='EUR/MWh', height=400)
    st.plotly_chart(fig_f_vs_a, use_container_width=True)

    # Residuals
    fig_res = go.Figure()
    for m_name, col in [('MLR', COLORS['mlr']), ('XGBoost', COLORS['xgboost']), ('ANN', COLORS['ann'])]:
        err = ml_test_df['spot_price'].values - ml_preds[m_name]
        fig_res.add_trace(go.Bar(x=hours, y=err, name=f'{m_name} Error', marker_color=col, opacity=0.75))
    fig_res.update_layout(**PLOTLY_LAYOUT, barmode='group', title='Forecast Residuals (Actual - Predicted, EUR/MWh)', xaxis_title='Hour', yaxis_title='Error (EUR/MWh)', height=320)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # SECTION 3: FEATURE IMPORTANCES & EXPLAINABILITY (EU AI ACT)
    st.subheader("3. 🔍 Feature Importance & Explainability (EU AI Act Compliance)")
    st.caption("EU AI Act Article 13 requires all AI systems in energy trading to provide auditable input attribution.")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("**MLR Standardized Coefficients (Marginal Elasticities):**")
        mlr_coef_df = pd.DataFrame({
            'Feature': list(ml_suite.feature_coefficients.keys()),
            'Coefficient (EUR/MWh / 1σ)': [round(v, 2) for v in ml_suite.feature_coefficients.values()]
        }).sort_values(by='Coefficient (EUR/MWh / 1σ)', key=abs, ascending=True)
        fig_coef = go.Figure(go.Bar(
            x=mlr_coef_df['Coefficient (EUR/MWh / 1σ)'],
            y=mlr_coef_df['Feature'],
            orientation='h',
            marker_color=[COLORS['grid_export'] if c < 0 else COLORS['fcr_n'] for c in mlr_coef_df['Coefficient (EUR/MWh / 1σ)']]
        ))
        fig_coef.update_layout(**PLOTLY_LAYOUT, title='MLR Marginal Price Elasticity (White-Box)', height=340)
        st.plotly_chart(fig_coef, use_container_width=True)

    with exp_col2:
        st.markdown("**XGBoost Feature Importance (Tree Split Gain):**")
        xgb_imp_df = pd.DataFrame({
            'Feature': list(ml_suite.feature_importances.keys()),
            'Importance': [round(v, 3) for v in ml_suite.feature_importances.values()]
        }).sort_values(by='Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=xgb_imp_df['Importance'],
            y=xgb_imp_df['Feature'],
            orientation='h',
            marker_color=COLORS['xgboost']
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT, title='XGBoost Tree Importance (TreeSHAP Equivalent)', height=340)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # SECTION 4: THE PRICE OF FORECAST ERROR (DOWNSTREAM MILP IMPACT)
    st.subheader("4. 💰 The 'Price of Forecast Error' — Downstream MILP Dispatch Impact")
    st.markdown("""
    > **JD Core Mandate**: *"A wrong forecast has a price, so calibration, backtesting, and validation are part of the job, not an afterthought."*

    In energy trading, a model is **never judged by RMSE alone**.
    If a model under-forecasts an evening peak, the BESS fails to reserve capacity or discharges too early, causing massive economic regret.
    Below, we solve the Pyomo MILP dispatch model under each model's forecast, and settle against the **realized actual spot prices**:
    """)

    btn_eval_dispatch = st.button("⚡ Run 4-Model Dispatch & Economic Regret Benchmark", type="primary", use_container_width=True)

    if btn_eval_dispatch or 'dispatch_comp_df' in st.session_state:
        if btn_eval_dispatch:
            with st.spinner("Solving Pyomo MILP across Oracle, XGBoost, MLR, and ANN forecasts..."):
                actual_p = ml_test_df['spot_price'].values
                eval_models = [
                    ('Oracle (Perfect Foresight)', actual_p),
                    ('XGBoost Forecast', ml_preds['XGBoost']),
                    ('MLR Forecast', ml_preds['MLR']),
                    ('ANN Forecast', ml_preds['ANN']),
                ]
                comp_rows = []
                oracle_cost = None
                for name, p_forecast in eval_models:
                    d_res = evaluate_forecast_dispatch_cost(
                        actual_prices=actual_p,
                        forecast_prices=p_forecast,
                        market_data=market_data,
                        bus_demand=bus_demand,
                        pv_generation=pv_gen,
                        config=config
                    )
                    if d_res['success']:
                        if oracle_cost is None:
                            oracle_cost = d_res['realized_cost_eur']
                        regret = d_res['realized_cost_eur'] - oracle_cost
                        comp_rows.append({
                            'Dispatch Model': name,
                            'Realized Cost (€/day)': round(d_res['realized_cost_eur'], 2),
                            'Regret vs Oracle (€/day)': round(max(0.0, regret), 2),
                            'Arbitrage Revenue (€)': round(d_res['arbitrage_eur'], 2),
                            'Peak Grid Import (kW)': round(d_res['peak_kw'], 1),
                            'Degradation Cost (€)': round(d_res['degradation_eur'], 2),
                            'Reserve Revenue (€)': round(d_res['reserve_revenue_eur'], 2),
                        })
                st.session_state['dispatch_comp_df'] = pd.DataFrame(comp_rows)

        comp_df = st.session_state['dispatch_comp_df']
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig_regret = go.Figure()
        fig_regret.add_trace(go.Bar(
            x=comp_df['Dispatch Model'],
            y=comp_df['Realized Cost (€/day)'],
            marker_color=[COLORS['spot'], COLORS['xgboost'], COLORS['mlr'], COLORS['ann']],
            text=[f"€{c:,.2f}" for c in comp_df['Realized Cost (€/day)']],
            textposition='outside'
        ))
        fig_regret.update_layout(**PLOTLY_LAYOUT, title='Realized Operating Cost Settled Under Ground Truth Prices (€/day)', yaxis_title='EUR/day', height=360)
        st.plotly_chart(fig_regret, use_container_width=True)

    st.markdown("---")

    # SECTION 5: EU AI ACT & GOVERNANCE MATRIX
    st.subheader("5. ⚖️ EU AI Act & Architecture Governance Comparison")
    st.dataframe(ml_gov_matrix, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 3: FLEET & DEPOT
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Electric Bus Depot")

    # Fleet schedule
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Bus Fleet Schedule")
        fig_schedule = go.Figure()

        morning_buses = schedule[schedule['route_type'] == 'morning']
        evening_buses = schedule[schedule['route_type'] == 'evening']

        # Show bus presence at depot (Gantt-like)
        for _, bus in morning_buses.iterrows():
            # At depot: 0 to departure, then return to 24
            fig_schedule.add_trace(go.Bar(
                x=[bus['departure_hour']], y=[bus['bus_id']],
                orientation='h', width=0.6,
                marker_color='rgba(38,198,218,0.4)',
                name='At Depot (AM)', showlegend=False,
            ))
            fig_schedule.add_trace(go.Bar(
                x=[24 - bus['return_hour']], y=[bus['bus_id']],
                base=bus['return_hour'],
                orientation='h', width=0.6,
                marker_color='rgba(38,198,218,0.4)',
                showlegend=False,
            ))
            fig_schedule.add_trace(go.Bar(
                x=[bus['return_hour'] - bus['departure_hour']], y=[bus['bus_id']],
                base=bus['departure_hour'],
                orientation='h', width=0.6,
                marker_color='rgba(255,167,38,0.6)',
                name='On Route', showlegend=False,
            ))

        for _, bus in evening_buses.iterrows():
            fig_schedule.add_trace(go.Bar(
                x=[bus['departure_hour']], y=[bus['bus_id']],
                orientation='h', width=0.6,
                marker_color='rgba(38,198,218,0.4)',
                showlegend=False,
            ))
            fig_schedule.add_trace(go.Bar(
                x=[24 - bus['return_hour']], y=[bus['bus_id']],
                base=bus['return_hour'],
                orientation='h', width=0.6,
                marker_color='rgba(38,198,218,0.4)',
                showlegend=False,
            ))
            fig_schedule.add_trace(go.Bar(
                x=[bus['return_hour'] - bus['departure_hour']], y=[bus['bus_id']],
                base=bus['departure_hour'],
                orientation='h', width=0.6,
                marker_color='rgba(239,83,80,0.6)',
                showlegend=False,
            ))

        fig_schedule.update_layout(
            **PLOTLY_LAYOUT,
            title='Fleet Schedule (Blue=Depot, Orange/Red=On Route)',
            xaxis_title='Hour of Day',
            barmode='overlay',
            height=max(300, len(schedule) * 15),
            showlegend=False,
        )
        fig_schedule.update_xaxes(range=[0, 24], dtick=2)
        st.plotly_chart(fig_schedule, use_container_width=True)

    with col2:
        st.subheader("Fleet Summary")
        total_energy = schedule['energy_to_charge_kwh'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{config.fleet.num_buses}</p>
            <p class="metric-label">Total Buses</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_energy:,.0f}</p>
            <p class="metric-label">Total Energy to Charge (kWh)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(morning_buses)}/{len(evening_buses)}</p>
            <p class="metric-label">Morning / Evening Shift</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Bus Schedule Data")
        st.dataframe(
            schedule[['bus_id', 'route_type', 'departure_hour', 'return_hour',
                       'return_soc', 'energy_to_charge_kwh']].round(2),
            use_container_width=True,
            height=300,
        )

    # Demand and PV profiles
    st.subheader("Charging Demand & PV Generation")
    fig_demand = make_subplots(specs=[[{"secondary_y": True}]])
    fig_demand.add_trace(go.Scatter(
        x=hours, y=bus_demand,
        mode='lines', name='Bus Charging Demand',
        fill='tozeroy', fillcolor='rgba(120,144,156,0.2)',
        line=dict(color=COLORS['bus_demand'], width=2),
    ), secondary_y=False)
    fig_demand.add_trace(go.Scatter(
        x=hours, y=pv_gen,
        mode='lines', name='PV Generation',
        fill='tozeroy', fillcolor='rgba(255,238,88,0.15)',
        line=dict(color=COLORS['pv'], width=2),
    ), secondary_y=False)
    fig_demand.add_trace(go.Scatter(
        x=hours, y=bus_demand - np.minimum(pv_gen, bus_demand),
        mode='lines', name='Net Grid Demand',
        line=dict(color=COLORS['grid_import'], width=2, dash='dash'),
    ), secondary_y=False)
    fig_demand.update_layout(
        **PLOTLY_LAYOUT,
        title='Depot Load Profile (Unoptimised Baseline)',
        height=350,
    )
    fig_demand.update_yaxes(title_text='Power (kW)', secondary_y=False)
    st.plotly_chart(fig_demand, use_container_width=True)

    # Key stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Demand", f"{np.max(bus_demand):,.0f} kW")
    c2.metric("Total Energy", f"{np.sum(bus_demand):,.0f} kWh")
    c3.metric("PV Generation", f"{np.sum(pv_gen):,.0f} kWh")
    c4.metric("PV Coverage", f"{np.sum(np.minimum(pv_gen, bus_demand)) / max(np.sum(bus_demand), 1) * 100:.1f}%")


# ─────────────────────────────────────────────────────────────
# TAB 4: OPTIMISATION
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("MILP Dispatch Optimisation")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 1])
    with col_ctrl1:
        enable_reserves = st.checkbox("Enable Reserve Markets", value=True,
                                       help="FCR-N, FCR-D, aFRR participation")
    with col_ctrl2:
        enable_export = st.checkbox("Enable Grid Export", value=True,
                                     help="Allow selling to grid")
    with col_ctrl3:
        enable_peak_shaving = st.checkbox("Enable Peak Shaving", value=True,
                                           help="Optimise peak demand charge")
    with col_ctrl4:
        price_signal_opt = st.selectbox(
            "Price Signal",
            ["Actual Spot (Oracle)", "XGBoost Forecast", "MLR Forecast", "ANN Forecast"],
            help="Choose which price curve drives the dispatch optimization"
        )

    # Determine input price array
    if price_signal_opt == "XGBoost Forecast":
        opt_spot_prices = ml_preds['XGBoost']
    elif price_signal_opt == "MLR Forecast":
        opt_spot_prices = ml_preds['MLR']
    elif price_signal_opt == "ANN Forecast":
        opt_spot_prices = ml_preds['ANN']
    else:
        opt_spot_prices = market_data['spot_price']

    run_button = st.button("🚀 Run Optimisation", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("Building MILP model and solving with HiGHS..."):
            try:
                optimizer = DepotOptimizer(config)
                optimizer.build_model(
                    spot_prices=opt_spot_prices,
                    fcr_n_prices=market_data['fcr_n_price'],
                    fcr_d_up_prices=market_data['fcr_d_up_price'],
                    fcr_d_down_prices=market_data['fcr_d_down_price'],
                    afrr_prices=market_data['afrr_price'],
                    bus_demand=bus_demand,
                    pv_generation=pv_gen,
                    enable_reserves=enable_reserves,
                    enable_export=enable_export,
                    enable_peak_shaving=enable_peak_shaving,
                )
                status = optimizer.solve()
                results_df = optimizer.extract_results()
                solve_stats = optimizer.get_solve_statistics()

                st.session_state['results_df'] = results_df
                st.session_state['solve_stats'] = solve_stats
                st.session_state['opt_status'] = status
                st.session_state['enable_reserves'] = enable_reserves
                st.session_state['enable_export'] = enable_export
                st.session_state['enable_peak_shaving'] = enable_peak_shaving

                if status in ['optimal', 'feasible']:
                    st.success(f"✅ Solver status: **{status}** | "
                              f"Time: {solve_stats['solve_time_seconds']}s | "
                              f"Variables: {solve_stats['num_variables']} | "
                              f"Constraints: {solve_stats['num_constraints']}")
                else:
                    st.error(f"❌ Solver status: {status}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Make sure you have Pyomo and HiGHS installed: "
                        "`pip install pyomo highspy`")

    # Display results if available
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        # KPI metrics
        st.subheader("Key Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Objective (€)", f"{results_df['objective_value'].iloc[0]:,.2f}")
        m2.metric("Peak Demand", f"{results_df['p_peak'].iloc[0]:,.0f} kW",
                   delta=f"{results_df['p_peak'].iloc[0] - np.max(bus_demand):,.0f} kW")
        m3.metric("Total Reserve", f"{results_df['total_reserve'].mean():,.0f} kW avg")
        m4.metric("Max Export", f"{results_df['p_grid_export'].max():,.0f} kW")
        m5.metric("Solve Time", f"{st.session_state['solve_stats']['solve_time_seconds']}s")

        # Dispatch chart
        st.subheader("Optimal Dispatch Schedule")
        fig_dispatch = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Power Flows (kW)', 'BESS State of Charge', 'Reserve Allocation (kW)'),
            row_heights=[0.45, 0.25, 0.30],
            vertical_spacing=0.08,
        )

        # Row 1: Power flows
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['bus_demand'],
            mode='lines', name='Bus Demand',
            line=dict(color=COLORS['bus_demand'], width=2, dash='dot'),
        ), row=1, col=1)
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['p_grid_import'],
            mode='lines', name='Grid Import',
            fill='tozeroy', fillcolor='rgba(66,165,245,0.15)',
            line=dict(color=COLORS['grid_import'], width=2),
        ), row=1, col=1)
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=-results_df['p_grid_export'],
            mode='lines', name='Grid Export',
            fill='tozeroy', fillcolor='rgba(239,83,80,0.15)',
            line=dict(color=COLORS['grid_export'], width=2),
        ), row=1, col=1)
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['p_bess_discharge'],
            mode='lines', name='BESS Discharge',
            line=dict(color=COLORS['bess_discharge'], width=2),
        ), row=1, col=1)
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=-results_df['p_bess_charge'],
            mode='lines', name='BESS Charge',
            line=dict(color=COLORS['bess_charge'], width=2),
        ), row=1, col=1)
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['pv_gen'],
            mode='lines', name='PV Generation',
            line=dict(color=COLORS['pv'], width=1.5, dash='dash'),
        ), row=1, col=1)

        # Row 2: SoC
        fig_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['soc_pct'],
            mode='lines', name='SoC (%)',
            fill='tozeroy', fillcolor='rgba(0,229,255,0.1)',
            line=dict(color=COLORS['soc'], width=2.5),
        ), row=2, col=1)
        # SoC bounds
        fig_dispatch.add_hline(y=config.bess.min_soc * 100, line_dash="dash",
                               line_color="red", opacity=0.5, row=2, col=1)
        fig_dispatch.add_hline(y=config.bess.max_soc * 100, line_dash="dash",
                               line_color="red", opacity=0.5, row=2, col=1)

        # Row 3: Reserves
        fig_dispatch.add_trace(go.Bar(
            x=hours, y=results_df['fcr_n'],
            name='FCR-N', marker_color=COLORS['fcr_n'], opacity=0.8,
        ), row=3, col=1)
        fig_dispatch.add_trace(go.Bar(
            x=hours, y=results_df['fcr_d_up'],
            name='FCR-D Up', marker_color=COLORS['fcr_d_up'], opacity=0.8,
        ), row=3, col=1)
        fig_dispatch.add_trace(go.Bar(
            x=hours, y=results_df['fcr_d_down'],
            name='FCR-D Down', marker_color=COLORS['fcr_d_down'], opacity=0.8,
        ), row=3, col=1)
        fig_dispatch.add_trace(go.Bar(
            x=hours, y=results_df['afrr_up'],
            name='aFRR Up', marker_color=COLORS['afrr'], opacity=0.8,
        ), row=3, col=1)

        fig_dispatch.update_layout(
            **PLOTLY_LAYOUT,
            height=900,
            barmode='stack',
        )
        fig_dispatch.update_xaxes(title_text='Hour', row=3, col=1)
        fig_dispatch.update_yaxes(title_text='kW', row=1, col=1)
        fig_dispatch.update_yaxes(title_text='SoC %', row=2, col=1)
        fig_dispatch.update_yaxes(title_text='kW', row=3, col=1)
        st.plotly_chart(fig_dispatch, use_container_width=True)

        # Price overlay with dispatch
        st.subheader("Dispatch vs. Price Signal")
        fig_price_dispatch = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price_dispatch.add_trace(go.Scatter(
            x=hours, y=results_df['spot_price'],
            mode='lines', name='Spot Price',
            line=dict(color=COLORS['spot'], width=2),
        ), secondary_y=True)
        fig_price_dispatch.add_trace(go.Bar(
            x=hours, y=results_df['net_bess_power'],
            name='Net BESS (+ discharge / − charge)',
            marker_color=[COLORS['bess_discharge'] if v >= 0 else COLORS['bess_charge']
                         for v in results_df['net_bess_power']],
            opacity=0.7,
        ), secondary_y=False)
        fig_price_dispatch.update_layout(
            **PLOTLY_LAYOUT,
            title='BESS Response to Price Signal (arbitrage behaviour)',
            height=350,
        )
        fig_price_dispatch.update_yaxes(title_text='BESS Power (kW)', secondary_y=False)
        fig_price_dispatch.update_yaxes(title_text='Spot Price (€/MWh)', secondary_y=True)
        st.plotly_chart(fig_price_dispatch, use_container_width=True)

        # Raw data
        with st.expander("📋 Raw Optimisation Results"):
            st.dataframe(results_df.round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 5: VALUE STACK
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Value Stack Decomposition")

    if 'results_df' not in st.session_state:
        st.info("⚡ Run the optimisation first (Optimisation tab) to see the value stack.")
    else:
        results_df = st.session_state['results_df']
        analyzer = ValueStackAnalyzer(config)
        savings = analyzer.compute_savings_summary(results_df)

        # Top-level metrics
        st.subheader("Savings Summary")
        m1, m2, m3, m4 = st.columns(4)

        savings_class = "positive" if savings['total_savings_eur'] > 0 else "negative"
        m1.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {savings_class}">€{savings['total_savings_eur']:,.2f}</p>
            <p class="metric-label">Daily Savings</p>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
            <p class="metric-value positive">€{savings['annual_savings_estimate_eur']:,.0f}</p>
            <p class="metric-label">Annual Savings (Est.)</p>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
            <p class="metric-value neutral">{savings['savings_pct']:.1f}%</p>
            <p class="metric-label">Cost Reduction</p>
        </div>
        """, unsafe_allow_html=True)

        payback_class = "positive" if savings['simple_payback_years'] < 10 else "negative"
        m4.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {payback_class}">{savings['simple_payback_years']:.1f} yr</p>
            <p class="metric-label">Simple Payback</p>
        </div>
        """, unsafe_allow_html=True)

        # Waterfall chart
        st.subheader("Cost Waterfall: Baseline → Optimised")
        vs = savings['value_stack']
        labels = list(vs.keys())
        values = list(vs.values())

        # Add baseline and total
        waterfall_labels = ['Baseline Cost'] + labels + ['Optimised Cost']
        waterfall_values = [savings['baseline_cost_eur']] + values + [savings['optimized_cost_eur']]
        waterfall_measures = ['absolute'] + ['relative'] * len(labels) + ['total']

        fig_waterfall = go.Figure(go.Waterfall(
            x=waterfall_labels,
            y=waterfall_values,
            measure=waterfall_measures,
            connector=dict(line=dict(color='rgba(0,229,255,0.3)')),
            increasing=dict(marker=dict(color='#ef5350')),  # Costs = red
            decreasing=dict(marker=dict(color='#69f0ae')),   # Revenues = green
            totals=dict(marker=dict(color='#42a5f5')),
            textposition='outside',
            text=[f"€{v:,.1f}" for v in waterfall_values],
        ))
        fig_waterfall.update_layout(
            **PLOTLY_LAYOUT,
            title='Value Stack Waterfall',
            height=450,
            yaxis_title='EUR',
        )
        fig_waterfall.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Revenue pie chart
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue Sources")
            rev = savings.get('revenue_breakdown', {})
            if rev:
                fig_pie = go.Figure(go.Pie(
                    labels=list(rev.keys()),
                    values=list(rev.values()),
                    hole=0.4,
                    marker=dict(colors=[COLORS.get('fcr_n', '#00e5ff'),
                                        COLORS.get('fcr_d_up', '#69f0ae'),
                                        COLORS.get('fcr_d_down', '#b388ff'),
                                        COLORS.get('afrr', '#ff8a80'),
                                        COLORS.get('grid_export', '#ef5350')]),
                ))
                fig_pie.update_layout(
                    **PLOTLY_LAYOUT,
                    title='Revenue Breakdown',
                    height=350,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No revenues generated in this scenario.")

        with col2:
            st.subheader("Baseline vs Optimised")
            baseline_detail = savings['baseline_detail']
            comparison = pd.DataFrame({
                'Component': ['Energy Cost', 'Grid Fees', 'Energy Tax', 'Peak Charge', 'Total'],
                'Baseline (€)': [
                    baseline_detail['energy_cost'],
                    baseline_detail['grid_fees'],
                    baseline_detail['energy_tax'],
                    baseline_detail['peak_charge'],
                    baseline_detail['total'],
                ],
                'Optimised (€)': [
                    vs.get('Energy Purchase', 0),
                    vs.get('Grid Fees', 0),
                    vs.get('Energy Tax', 0),
                    vs.get('Peak Demand Charge', 0),
                    savings['optimized_cost_eur'],
                ],
            })
            comparison['Saving (€)'] = comparison['Baseline (€)'] - comparison['Optimised (€)']
            st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

            # Peak shaving metrics
            st.subheader("Peak Shaving")
            st.metric("Peak Reduction",
                       f"{savings['peak_reduction_kw']:,.0f} kW",
                       delta=f"{savings['peak_reduction_kw'] / max(savings['peak_baseline_kw'], 1) * 100:.1f}% reduction")
            st.metric("Baseline Peak", f"{savings['peak_baseline_kw']:,.0f} kW")
            st.metric("Optimised Peak", f"{savings['peak_optimized_kw']:,.0f} kW")


# ─────────────────────────────────────────────────────────────
# TAB 6: SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Sensitivity Analysis")
    st.caption("Sweep key parameters to understand marginal value and diminishing returns")

    analysis_type = st.selectbox("Analysis Type", [
        "BESS Capacity Sweep",
        "Spot Price Level Sweep",
        "Monthly Backtest (12 months)",
    ])

    if analysis_type == "BESS Capacity Sweep":
        st.markdown("**Sweep BESS capacity** from 500 to 5000 kWh (power scales at C/2 rate)")
        num_points = st.slider("Number of sweep points", 4, 12, 6)
        capacities = np.linspace(500, 5000, num_points).tolist()

        if st.button("🔬 Run BESS Sweep", type="primary"):
            progress = st.progress(0)
            with st.spinner("Running sensitivity sweep..."):
                try:
                    sweep_df = SensitivityAnalyzer.sweep_bess_capacity(
                        base_config=config,
                        optimizer_class=DepotOptimizer,
                        market_data=market_data,
                        bus_demand=bus_demand,
                        pv_gen=pv_gen,
                        capacities=capacities,
                        progress_callback=lambda p: progress.progress(p),
                    )
                    st.session_state['bess_sweep'] = sweep_df
                    progress.progress(1.0)
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'bess_sweep' in st.session_state:
            sweep_df = st.session_state['bess_sweep']
            if 'error' not in sweep_df.columns or sweep_df['error'].isna().all():
                col1, col2 = st.columns(2)
                with col1:
                    fig_cap = go.Figure()
                    fig_cap.add_trace(go.Scatter(
                        x=sweep_df['bess_capacity_kwh'],
                        y=sweep_df['total_savings_eur'],
                        mode='lines+markers',
                        name='Daily Savings',
                        line=dict(color='#69f0ae', width=2.5),
                        marker=dict(size=8),
                    ))
                    fig_cap.update_layout(
                        **PLOTLY_LAYOUT,
                        title='Savings vs BESS Capacity',
                        xaxis_title='BESS Capacity (kWh)',
                        yaxis_title='Daily Savings (€)',
                        height=400,
                    )
                    st.plotly_chart(fig_cap, use_container_width=True)

                with col2:
                    fig_payback = go.Figure()
                    fig_payback.add_trace(go.Scatter(
                        x=sweep_df['bess_capacity_kwh'],
                        y=sweep_df['payback_years'],
                        mode='lines+markers',
                        name='Payback Period',
                        line=dict(color='#ffd740', width=2.5),
                        marker=dict(size=8),
                    ))
                    fig_payback.update_layout(
                        **PLOTLY_LAYOUT,
                        title='Payback Period vs BESS Capacity',
                        xaxis_title='BESS Capacity (kWh)',
                        yaxis_title='Simple Payback (Years)',
                        height=400,
                    )
                    st.plotly_chart(fig_payback, use_container_width=True)

                # Marginal value
                if len(sweep_df) > 1:
                    sweep_df['marginal_savings'] = sweep_df['total_savings_eur'].diff() / sweep_df['bess_capacity_kwh'].diff()
                    fig_marginal = go.Figure()
                    fig_marginal.add_trace(go.Bar(
                        x=sweep_df['bess_capacity_kwh'].iloc[1:],
                        y=sweep_df['marginal_savings'].iloc[1:],
                        marker_color='#00bcd4',
                        opacity=0.8,
                    ))
                    fig_marginal.update_layout(
                        **PLOTLY_LAYOUT,
                        title='Marginal Value of Additional BESS Capacity (€/kWh/day)',
                        xaxis_title='BESS Capacity (kWh)',
                        yaxis_title='€ saved per additional kWh',
                        height=300,
                    )
                    st.plotly_chart(fig_marginal, use_container_width=True)
                    st.info("📉 **Diminishing returns**: Notice how the marginal value decreases "
                            "as BESS capacity grows. This is key for BESS sizing decisions.")

                st.dataframe(sweep_df.round(2), use_container_width=True, hide_index=True)

    elif analysis_type == "Spot Price Level Sweep":
        st.markdown("**Sweep average spot price** from €10 to €150/MWh")
        num_points = st.slider("Number of sweep points", 4, 12, 8)
        prices = np.linspace(10, 150, num_points).tolist()

        if st.button("🔬 Run Price Sweep", type="primary"):
            progress = st.progress(0)
            with st.spinner("Running price sensitivity..."):
                try:
                    price_df = SensitivityAnalyzer.sweep_spot_price(
                        base_config=config,
                        optimizer_class=DepotOptimizer,
                        market_generator=generate_all_market_data,
                        bus_demand=bus_demand,
                        pv_gen=pv_gen,
                        price_levels=prices,
                        progress_callback=lambda p: progress.progress(p),
                    )
                    st.session_state['price_sweep'] = price_df
                    progress.progress(1.0)
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'price_sweep' in st.session_state:
            price_df = st.session_state['price_sweep']
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=price_df['spot_price_eur_mwh'],
                y=price_df['total_savings_eur'],
                mode='lines+markers',
                name='Daily Savings',
                line=dict(color='#69f0ae', width=2.5),
                marker=dict(size=8),
            ))
            fig_price.update_layout(
                **PLOTLY_LAYOUT,
                title='Savings vs Average Spot Price Level',
                xaxis_title='Average Spot Price (€/MWh)',
                yaxis_title='Daily Savings (€)',
                height=400,
            )
            st.plotly_chart(fig_price, use_container_width=True)
            st.dataframe(price_df.round(2), use_container_width=True, hide_index=True)

    elif analysis_type == "Monthly Backtest (12 months)":
        st.markdown("**Backtest across all 12 months** to see seasonal variation in value")

        if st.button("🔬 Run 12-Month Backtest", type="primary"):
            progress = st.progress(0)
            with st.spinner("Running 12 monthly scenarios..."):
                try:
                    backtester = Backtester(config)
                    bt_df = backtester.run_scenario_sweep(
                        optimizer_class=DepotOptimizer,
                        market_generator=generate_all_market_data,
                        fleet_model_class=FleetModel,
                        pv_model_class=PVModel,
                        num_scenarios=12,
                        progress_callback=lambda p: progress.progress(p),
                    )
                    st.session_state['backtest'] = bt_df
                    progress.progress(1.0)
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'backtest' in st.session_state:
            bt_df = st.session_state['backtest']
            valid = bt_df[bt_df['status'].isin(['optimal', 'feasible'])] if 'status' in bt_df.columns else bt_df

            if len(valid) > 0:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Bar(
                    x=valid['month_name'],
                    y=valid['total_savings'],
                    marker_color=[COLORS['fcr_n'] if s > 0 else COLORS['grid_export']
                                  for s in valid['total_savings']],
                    opacity=0.85,
                    text=[f"€{s:,.0f}" for s in valid['total_savings']],
                    textposition='outside',
                ))
                fig_bt.update_layout(
                    **PLOTLY_LAYOUT,
                    title='Daily Savings by Month',
                    xaxis_title='Month',
                    yaxis_title='Daily Savings (€)',
                    height=400,
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Daily Savings", f"€{valid['total_savings'].mean():,.1f}")
                col2.metric("Best Month", f"{valid.loc[valid['total_savings'].idxmax(), 'month_name']}")
                col3.metric("Annual Estimate", f"€{valid['total_savings'].mean() * 365:,.0f}")

                st.dataframe(valid.round(2), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 7: INTERVIEW PREP
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("🎯 Interview Preparation Guide")
    st.caption("Key concepts, common questions, and exploration exercises tied to this simulation")

    st.markdown("---")

    # Q1
    with st.expander("**Q: How would you formulate the multi-horizon dispatch problem?**", expanded=True):
        st.markdown("""
        **Key answer points:**

        1. **Objective**: Minimise total operating cost = energy purchases + grid fees + peak charges
           − reserve revenues − export revenues + degradation costs

        2. **Decision variables**: Grid import/export, BESS charge/discharge, SoC trajectory,
           reserve capacity bids (FCR-N, FCR-D, aFRR), peak demand tracker

        3. **Why MILP not LP?**: The charge/discharge mutual exclusion requires binary variables
           (Big-M formulation). Without this, the solver could simultaneously charge and discharge
           (free energy from efficiency losses — physically impossible).

        4. **Why not nonlinear?**: LP/MILP gives global optimality guarantees and fast solve times.
           Battery degradation is linearised. Temperature effects could add nonlinearity but are
           handled in the lower (edge) layer.

        5. **Multi-horizon**: Day-ahead (24h, hourly) → Intraday update (6h, 15-min) → Real-time
           (1h, 5-min). Each horizon refines the previous using a rolling window with terminal
           SoC constraints to avoid myopic behaviour.

        **👉 Try it**: Go to the Optimisation tab, run the model, and inspect the dispatch chart.
        Notice how the BESS charges during low-price hours and discharges during peaks.
        """)

    # Q2
    with st.expander("**Q: How do you co-optimise reserve markets with energy arbitrage?**"):
        st.markdown("""
        **Key answer points:**

        1. **Reserve capacity is an opportunity cost**: Allocating kW to FCR-N means that power
           can't be used for arbitrage. The optimiser weighs reserve revenue vs arbitrage value.

        2. **Headroom constraints are the key**: Both power headroom (C6, C7) and energy headroom
           (C8, C9) must be maintained. FCR-N is symmetric — must reserve equal up AND down.

        3. **SoC management for FCR-N**: Because FCR-N is symmetric, the BESS naturally stays
           near ~50% SoC when providing FCR-N. This limits arbitrage depth.

        4. **Product stacking**: FCR-N + FCR-D + aFRR can be stacked if total doesn't exceed
           BESS power. The optimiser finds the optimal mix hour by hour.

        **👉 Try it**: Toggle "Enable Reserve Markets" on/off in the Optimisation tab and compare
        the SoC trajectory and total cost. Notice FCR-N forces SoC to stay mid-range.
        """)

    # Q3
    with st.expander("**Q: What is the value stack and how do you decompose it?**"):
        st.markdown("""
        **Key answer points:**

        1. **Value stack** = the total economic value broken down by revenue/savings stream:
           - Peak shaving (reduced capacity charges)
           - Energy arbitrage (buy low, sell high)
           - FCR-N revenue (highest per-MW, but symmetric constraint)
           - FCR-D up/down revenue
           - aFRR revenue
           - PV self-consumption (avoided grid purchases)

        2. **Decomposition method**: Run the optimiser multiple times with feature flags:
           - Full optimisation (all streams)
           - No reserves → reserve marginal value = difference
           - No export → export marginal value
           - No BESS → BESS total value

        3. **Non-additive**: Individual stream values don't add up to total because they interact
           (e.g., reserve allocation affects arbitrage). Report marginal contributions.

        **👉 Try it**: Check the Value Stack tab for the waterfall chart. Then go to Sensitivity
        and sweep BESS capacity to see diminishing returns.
        """)

    # Q4
    with st.expander("**Q: How do you handle bid deliverability and grid safety?**"):
        st.markdown("""
        **Key answer points:**

        1. **"What was bid gets delivered"** — this is a contract integrity principle.
           Bids must reflect actual physical constraints.

        2. **Two-tier architecture**: The optimiser (upper level) produces a power envelope.
           The edge controller (lower level) translates this to per-charger commands.
           The edge layer has an **absolute physical veto** — it can reject any setpoint
           that would violate grid limits, transformer ratings, or charger specs.

        3. **Conservative bidding**: Reserve bids include safety margins. The energy headroom
           constraint (C8, C9) ensures SoC can sustain full activation for the required duration.

        4. **Monitoring**: Real-time telemetry compares actual delivery vs bid. Deviations
           trigger alerts and model recalibration.

        **👉 Try it**: Set BESS min SoC very low (5%) and max SoC very high (98%) — notice
        the optimiser still maintains headroom for reserves. The constraints enforce safety.
        """)

    # Q5
    with st.expander("**Q: How would you backtest the optimiser?**"):
        st.markdown("""
        **Key answer points:**

        1. **Historical replay**: Run the optimiser against historical ENTSO-E Transparency
           Platform data (day-ahead prices, FCR clearing prices) for SE3.

        2. **Scenario ensemble**: Generate 100+ scenarios varying price level, volatility,
           season, PV output, fleet size. Report P10/P50/P90 of daily savings.

        3. **Out-of-sample validation**: Train forecasting models on 2023 data, backtest
           on 2024. Compare realised vs forecasted savings.

        4. **Shadow mode**: Before live deployment, run the optimiser in parallel with
           the existing (dumb charging) system. Compare what WOULD have happened.

        5. **Metrics**: Track Sharpe ratio of daily savings, maximum drawdown (worst day),
           and delivery accuracy (bid vs actual).

        **👉 Try it**: Run the 12-month backtest in the Sensitivity tab. Notice seasonal
        patterns — winter typically higher savings due to higher prices.
        """)

    # Q6
    with st.expander("**Q: How do Nordic reserve markets (FCR-N, FCR-D, aFRR) work?**"):
        st.markdown("""
        **FCR-N (Frequency Containment Reserve - Normal)**:
        - **What**: Symmetric, continuous response to frequency deviations within 49.9–50.1 Hz
        - **How**: BESS adjusts power proportionally to frequency deviation (droop control)
        - **Payment**: Capacity payment (€/MW/h), recently moved to marginal pricing
        - **Key constraint**: Must provide BOTH up AND down → forces mid-SoC operation

        **FCR-D (Frequency Containment Reserve - Disturbance)**:
        - **What**: Asymmetric, activated only during large disturbances (>±0.1 Hz)
        - **FCR-D Up**: Activated when frequency drops below 49.9 Hz (generation deficit)
        - **FCR-D Down**: Activated when frequency rises above 50.1 Hz (generation surplus)
        - **Response**: Full activation within 5 seconds → BESS is ideal for this

        **aFRR (Automatic Frequency Restoration Reserve)**:
        - **What**: Automatic, activated by TSO control signal to restore frequency to 50 Hz
        - **Response**: Full activation within 120 seconds
        - **Payment**: Capacity + energy payment

        **mFRR (Manual Frequency Restoration Reserve)**:
        - **What**: Manual activation by TSO for larger imbalances
        - **Response**: Full activation within 15 minutes
        - **Payment**: Capacity + energy payment

        **Value ranking for BESS**: FCR-N > FCR-D > aFRR > mFRR (typically)
        """)

    # Q7
    with st.expander("**Q: What about EU AI Act compliance and explainability?**"):
        st.markdown("""
        **Key answer points:**

        1. **Traceability**: Every optimisation decision is fully traceable — the MILP
           formulation is deterministic given inputs. You can extract dual/shadow prices
           to explain WHY the optimiser made each decision.

        2. **No black boxes**: Unlike RL-based dispatch, MILP optimisation is interpretable.
           Each constraint has a physical meaning. Shadow prices show the marginal cost
           of each constraint.

        3. **Audit trail**: Log all inputs (prices, forecasts, constraints) and outputs
           (dispatch schedule, bids) with timestamps. Any bid can be reconstructed.

        4. **Model registry**: Version all model parameters, solver versions, and config.
           Changes require review and backtesting before production deployment.
        """)

    # Q8
    with st.expander("**Q: What technologies would you use in production?**"):
        st.markdown("""
        | Component | Technology | Why |
        |-----------|------------|-----|
        | Optimisation | **Pyomo + HiGHS** (or Gurobi) | LP/MILP, open-source, fast |
        | Forecasting | **LightGBM / XGBoost** | Probabilistic price forecasts |
        | HPO | **Optuna** | Bayesian hyperparameter tuning |
        | Drift detection | **Evidently AI** | Monitor model performance |
        | API | **FastAPI + gRPC** | Low-latency dispatch commands |
        | Data | **BigQuery + TimescaleDB** | Analytics + time-series |
        | Messaging | **GCP Pub/Sub** | Event-driven architecture |
        | Orchestration | **GKE (europe-north2)** | Data residency compliance |
        | MLOps | **Vertex AI / MLflow** | Model versioning & serving |
        | Config | **Pydantic** | Typed, validated configuration |
        """)

    # Q9
    with st.expander("**Q: How do MLR, XGBoost, and ANN respond differently to electricity price drivers?**"):
        st.markdown("""
        **Key answer points:**

        1. **Temperature & Electric Heating (Degree-Days)**:
           - *Nordic Reality*: Above 10°C, demand is inelastic. Below 0°C, heat pumps hit COP limits and resistive electric heaters kick in, causing exponential demand surges ("hockey-stick effect").
           - *MLR*: Rigid linear line — severely underpredicts cold snap peak prices by €30–50/MWh.
           - *XGBoost*: Step-wise threshold jumps that isolate the freezing regime into distinct tree leaves.
           - *ANN*: Smooth non-linear sigmoid/exponential curve reflecting continuous heating acceleration.

        2. **Wind Speed & Renewable Merit-Order Depression**:
           - *Nordic Reality*: Wind turbine power is cubic (P ∝ v³) between 3 and 12 m/s, then saturates. Zero-marginal-cost wind shifts the supply curve right, precipitating negative prices.
           - *MLR*: Plunges linearly into unbounded negative prices because it lacks saturation gates.
           - *XGBoost & ANN*: Correctly capture the flat cut-in floor, the steep price collapse during rated generation, and the curtailment price floor.

        3. **Inductive Bias & Overfitting**:
           - *MLR*: High bias, zero variance — cannot overfit, but underfits non-linear regimes.
           - *XGBoost*: Low bias, moderate variance — handles tabular threshold logic natively. Best empirical accuracy.
           - *ANN*: High representation capacity — requires substantial data, early stopping, and regularisation to avoid hallucinating price spikes.
        """)

    # Q10
    with st.expander("**Q: What is the 'Price of Forecast Error', and how do you comply with the EU AI Act?**"):
        st.markdown("""
        **Key answer points:**

        1. **The Price of Forecast Error (Economic Regret)**:
           - In dispatch optimization, models cannot be judged solely by RMSE or MAE.
           - *Asymmetric Penalty*: Under-forecasting an evening peak price means the BESS will fail to charge overnight or discharge prematurely, incurring hundreds of euros in lost arbitrage and peak tariff penalties.
           - *Two-Stage Evaluation*: We evaluate models by **Realized Dispatch Cost** and **Regret vs Oracle** (lost opportunity compared to perfect foresight).

        2. **EU AI Act Compliance (Articles 13 & 14)**:
           - Systems influencing real power grids and market bids cannot use black-box models without explainability.
           - *XGBoost*: Compliant via **TreeSHAP** (exact Shapley values attributing every euro of predicted price to temperature, wind, and lag features).
           - *MLR*: 100% white-box baseline and fallback safety envelope.
           - *Safety Override*: The physical edge safety layer retains absolute veto over any automated bid or power envelope.
        """)

    st.markdown("---")
    st.subheader("🧪 Exploration Exercises")
    st.markdown("""
    Use the sidebar to tweak parameters and observe the effects:

    1. **Double BESS capacity** (2000 → 4000 kWh) — does savings double? (Hint: diminishing returns)
    2. **Set PV to 0** — how much does the value stack change?
    3. **Switch to SE4** (most expensive zone) — how do reserve revenues change?
    4. **Set spot price to €100/MWh** — when does arbitrage dominate reserves?
    5. **Try winter (December)** vs **summer (June)** — seasonal dynamics
    6. **Disable reserves** — how much are they worth?
    7. **Increase fleet to 80 buses** — does the BESS become constrained?
    8. **Set peak tariff to €20/kW** — does peak shaving become dominant?
    """)

    st.markdown("---")
    st.subheader("📐 Architecture Diagram")
    st.code("""
    ┌──────────────────────────────────────────────────────────────────┐
    │                    UPPER STRATEGIC LEVEL                         │
    │                                                                  │
    │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
    │  │  Forecaster  │→│   MILP       │→│  Bid Formulation       │  │
    │  │  (Price,     │  │  Dispatch    │  │  (Nord Pool, SvK)      │  │
    │  │   Load, PV)  │  │  Optimiser   │  │                        │  │
    │  └─────────────┘  └──────┬───────┘  └────────────────────────┘  │
    │                          │                                       │
    │              Dynamic Power Envelope                              │
    └──────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                    LOWER EXECUTION LEVEL                         │
    │                                                                  │
    │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
    │  │  Per-Charger │  │  BESS        │  │  Edge Safety Layer     │  │
    │  │  Controller  │  │  Controller  │  │  (ABSOLUTE VETO)       │  │
    │  └─────────────┘  └──────────────┘  └────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────┘
    """, language=None)
