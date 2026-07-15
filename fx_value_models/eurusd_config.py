"""Default Bloomberg universe and model specifications for EURUSD."""

from __future__ import annotations

PAIR_TICKER = "EURUSD Curncy"
FIELD = "PX_LAST"

TICKERS = [
    PAIR_TICKER,
    "USGG2YR Index",
    "GDBR2 Index",
    "USGG10YR Index",
    "GDBR10 Index",
    "USOSFR2 Curncy",
    "EUSA2 Curncy",
    "EESWE2 Curncy",
    "CESIUSD Index",
    "CESIEUR Index",
    "VIX Index",
    "MOVE Index",
    "SPX Index",
    "SX5E Index",
    "SXXP Index",
    "CO1 Comdty",
    "TZT1 Comdty",
    "EURUSDV1M Curncy",
    "EURUSD25R1M Curncy",
    "EURUSDV3M Curncy",
    "EURUSD25R3M Curncy",
]

FEATURE_DEFS = [
    {"name": "de_us_2y_yield", "kind": "diff", "left": "GDBR2 Index", "right": "USGG2YR Index"},
    {"name": "de_us_10y_yield", "kind": "diff", "left": "GDBR10 Index", "right": "USGG10YR Index"},
    {"name": "eur_us_2y_swap", "kind": "diff", "left": "EUSA2 Curncy", "right": "USOSFR2 Curncy"},
    {"name": "estr_us_2y_ois", "kind": "diff", "left": "EESWE2 Curncy", "right": "USOSFR2 Curncy"},
    {"name": "cesi_eur_us", "kind": "diff", "left": "CESIEUR Index", "right": "CESIUSD Index"},
    {"name": "sx5e_spx_ratio", "kind": "log_ratio", "left": "SX5E Index", "right": "SPX Index"},
    {"name": "sxxp_spx_ratio", "kind": "log_ratio", "left": "SXXP Index", "right": "SPX Index"},
    {"name": "brent_log", "kind": "log", "ticker": "CO1 Comdty"},
    {"name": "ttf_gas_log", "kind": "log", "ticker": "TZT1 Comdty"},
    {"name": "vix_log", "kind": "log", "ticker": "VIX Index"},
    {"name": "move_log", "kind": "log", "ticker": "MOVE Index"},
    {"name": "eurusd_1m_vol", "kind": "level", "ticker": "EURUSDV1M Curncy"},
    {"name": "eurusd_1m_rr", "kind": "level", "ticker": "EURUSD25R1M Curncy"},
    {"name": "eurusd_3m_vol", "kind": "level", "ticker": "EURUSDV3M Curncy"},
    {"name": "eurusd_3m_rr", "kind": "level", "ticker": "EURUSD25R3M Curncy"},
]

MODEL_SPECS = [
    {"name": "gov_2y", "features": ["de_us_2y_yield"], "ridge_alpha": 0.0},
    {"name": "gov_curve", "features": ["de_us_2y_yield", "de_us_10y_yield"], "ridge_alpha": 0.25},
    {"name": "swap_2y", "features": ["eur_us_2y_swap"], "ridge_alpha": 0.0},
    {"name": "rates_surprises", "features": ["de_us_2y_yield", "cesi_eur_us"], "ridge_alpha": 0.25},
    {"name": "macro_core", "features": ["de_us_2y_yield", "cesi_eur_us", "sx5e_spx_ratio", "brent_log", "vix_log"], "ridge_alpha": 1.0},
    {"name": "macro_energy", "features": ["de_us_2y_yield", "cesi_eur_us", "sx5e_spx_ratio", "brent_log", "ttf_gas_log", "vix_log", "move_log"], "ridge_alpha": 2.0},
    {"name": "rates_options", "features": ["de_us_2y_yield", "cesi_eur_us", "eurusd_1m_vol", "eurusd_1m_rr", "eurusd_3m_rr"], "ridge_alpha": 1.0},
    {"name": "market_full", "features": ["de_us_2y_yield", "de_us_10y_yield", "cesi_eur_us", "sx5e_spx_ratio", "brent_log", "ttf_gas_log", "vix_log", "move_log", "eurusd_1m_vol", "eurusd_1m_rr"], "ridge_alpha": 3.0},
]

ROLLING_WINDOWS = [252, 504, 756]
ENTRY_ZS = [1.25, 1.5, 1.75, 2.0]
EXIT_ZS = [0.25, 0.5, 0.75]
STOP_Z = 3.0
MAX_HOLD_DAYS = 20
Z_WINDOW = 252
COST_BPS_PER_SIDE = 0.5
