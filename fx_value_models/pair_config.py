"""Pair-specific Bloomberg universes for the FX dashboard."""

from __future__ import annotations

from dataclasses import dataclass

FIELD = "PX_LAST"


@dataclass(frozen=True)
class PairConfig:
    name: str
    spot_ticker: str
    rate_2y_left: str
    rate_2y_right: str
    rate_10y_left: str
    rate_10y_right: str
    cesi_left: str
    cesi_right: str
    equity_left: str
    equity_right: str
    vol_1m: str
    rr_1m: str
    extra_features: tuple[str, ...] = ()

    @property
    def tickers(self) -> list[str]:
        base = [
            self.spot_ticker,
            self.rate_2y_left,
            self.rate_2y_right,
            self.rate_10y_left,
            self.rate_10y_right,
            self.cesi_left,
            self.cesi_right,
            self.equity_left,
            self.equity_right,
            self.vol_1m,
            self.rr_1m,
            "VIX Index",
            "MOVE Index",
        ]
        if "brent_log" in self.extra_features:
            base.append("CO1 Comdty")
        if "ttf_gas_log" in self.extra_features:
            base.append("TZT1 Comdty")
        if "gold_log" in self.extra_features:
            base.append("XAU Curncy")
        return sorted(set(base))

    @property
    def feature_defs(self) -> list[dict]:
        features = [
            {"name": "rate_2y_diff", "kind": "diff", "left": self.rate_2y_left, "right": self.rate_2y_right},
            {"name": "rate_10y_diff", "kind": "diff", "left": self.rate_10y_left, "right": self.rate_10y_right},
            {"name": "cesi_diff", "kind": "diff", "left": self.cesi_left, "right": self.cesi_right},
            {"name": "equity_ratio", "kind": "log_ratio", "left": self.equity_left, "right": self.equity_right},
            {"name": "vix_log", "kind": "log", "ticker": "VIX Index"},
            {"name": "move_log", "kind": "log", "ticker": "MOVE Index"},
            {"name": "fx_1m_vol", "kind": "level", "ticker": self.vol_1m},
            {"name": "fx_1m_rr", "kind": "level", "ticker": self.rr_1m},
        ]
        if "brent_log" in self.extra_features:
            features.append({"name": "brent_log", "kind": "log", "ticker": "CO1 Comdty"})
        if "ttf_gas_log" in self.extra_features:
            features.append({"name": "ttf_gas_log", "kind": "log", "ticker": "TZT1 Comdty"})
        if "gold_log" in self.extra_features:
            features.append({"name": "gold_log", "kind": "log", "ticker": "XAU Curncy"})
        return features

    @property
    def model_specs(self) -> list[dict]:
        macro_core = ["rate_2y_diff", "cesi_diff", "equity_ratio", "vix_log"]
        macro_extended = [
            "rate_2y_diff",
            "rate_10y_diff",
            "cesi_diff",
            "equity_ratio",
            "vix_log",
            "move_log",
            *self.extra_features,
        ]
        return [
            {"name": "rates_2y", "features": ["rate_2y_diff"], "ridge_alpha": 0.0},
            {"name": "rates_curve", "features": ["rate_2y_diff", "rate_10y_diff"], "ridge_alpha": 0.25},
            {"name": "rates_surprises", "features": ["rate_2y_diff", "cesi_diff"], "ridge_alpha": 0.25},
            {"name": "macro_core", "features": macro_core, "ridge_alpha": 1.0},
            {"name": "macro_extended", "features": macro_extended, "ridge_alpha": 2.0},
            {"name": "options_risk", "features": ["rate_2y_diff", "cesi_diff", "fx_1m_vol", "fx_1m_rr", "vix_log"], "ridge_alpha": 1.0},
        ]


PAIR_CONFIGS = [
    PairConfig("EURUSD", "EURUSD Curncy", "GDBR2 Index", "USGG2YR Index", "GDBR10 Index", "USGG10YR Index", "CESIEUR Index", "CESIUSD Index", "SX5E Index", "SPX Index", "EURUSDV1M Curncy", "EURUSD25R1M Curncy", ("brent_log", "ttf_gas_log")),
    PairConfig("GBPUSD", "GBPUSD Curncy", "GUKG2 Index", "USGG2YR Index", "GUKG10 Index", "USGG10YR Index", "CESIGBP Index", "CESIUSD Index", "UKX Index", "SPX Index", "GBPUSDV1M Curncy", "GBPUSD25R1M Curncy", ("brent_log",)),
    PairConfig("AUDUSD", "AUDUSD Curncy", "GACGB2 Index", "USGG2YR Index", "GACGB10 Index", "USGG10YR Index", "CESIAUD Index", "CESIUSD Index", "AS51 Index", "SPX Index", "AUDUSDV1M Curncy", "AUDUSD25R1M Curncy", ("brent_log", "gold_log")),
    PairConfig("NZDUSD", "NZDUSD Curncy", "GNZGB2 Index", "USGG2YR Index", "GNZGB10 Index", "USGG10YR Index", "CESINZD Index", "CESIUSD Index", "NZSE50FG Index", "SPX Index", "NZDUSDV1M Curncy", "NZDUSD25R1M Curncy", ("brent_log", "gold_log")),
    PairConfig("USDJPY", "USDJPY Curncy", "USGG2YR Index", "GJGB2 Index", "USGG10YR Index", "GJGB10 Index", "CESIUSD Index", "CESIJPY Index", "SPX Index", "NKY Index", "USDJPYV1M Curncy", "USDJPY25R1M Curncy", ("gold_log",)),
    PairConfig("USDCHF", "USDCHF Curncy", "USGG2YR Index", "GSWISS2 Index", "USGG10YR Index", "GSWISS10 Index", "CESIUSD Index", "CESICHF Index", "SPX Index", "SMI Index", "USDCHFV1M Curncy", "USDCHF25R1M Curncy", ("gold_log",)),
    PairConfig("USDNOK", "USDNOK Curncy", "USGG2YR Index", "GNOR2 Index", "USGG10YR Index", "GNOR10 Index", "CESIUSD Index", "CESINOK Index", "SPX Index", "OBX Index", "USDNOKV1M Curncy", "USDNOK25R1M Curncy", ("brent_log",)),
    PairConfig("USDSEK", "USDSEK Curncy", "USGG2YR Index", "GSGB2YR Index", "USGG10YR Index", "GSGB10YR Index", "CESIUSD Index", "CESISEK Index", "SPX Index", "OMX Index", "USDSEKV1M Curncy", "USDSEK25R1M Curncy", ("brent_log",)),
]

ROLLING_WINDOWS = [252, 504, 756]
ENTRY_ZS = [1.25, 1.5, 1.75, 2.0]
EXIT_ZS = [0.25, 0.5, 0.75]
STOP_Z = 3.0
MAX_HOLD_DAYS = 20
Z_WINDOW = 252
COST_BPS_PER_SIDE = 0.5


def all_tickers() -> list[str]:
    tickers: set[str] = set()
    for pair in PAIR_CONFIGS:
        tickers.update(pair.tickers)
    return sorted(tickers)
