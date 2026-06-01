from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .market_session import is_friday, is_regular_session, is_thursday, ny_date, session_label, to_ny
from .option_quotes import DefaultOptionQuoteProvider, OptionQuote, QuoteProvider
from .signal_types import SignalType, StrategyDecision, normalize_signal_type

MONTHS = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


@dataclass
class OptionsConfig:
    symbol_root: str = "QQQ"
    cache_dir: str = ".cache"
    logs_dir: str = "logs"
    regular_session_only: bool = True
    target_delta: float = 0.38
    min_delta: float = 0.35
    max_delta: float = 0.40
    iv_assumption: float = 0.25
    risk_free_rate: float = 0.05
    preferred_business_dte: int = 2
    enable_0dte: bool = True
    require_ml_for_0dte: bool = True
    require_strong_move_for_0dte: bool = True
    min_ml_0dte_long_05atr: float = 0.60
    min_ml_0dte_long_10atr: float = 0.42
    min_ml_0dte_short_05atr: float = 0.60
    min_ml_0dte_short_10atr: float = 0.42
    allow_open_put: bool = True
    allow_flip_same_bar: bool = False
    strike_increment: float = 1.0
    log_option_quotes: bool = True

    @classmethod
    def from_env(cls) -> "OptionsConfig":
        return cls(
            symbol_root=os.getenv("OPTIONS_SYMBOL_ROOT", "QQQ"),
            cache_dir=os.getenv("CACHE_DIR", ".cache"),
            logs_dir=os.getenv("LOGS_DIR", "logs"),
            regular_session_only=_env_bool("OPTIONS_REGULAR_SESSION_ONLY", True),
            target_delta=_env_float("OPTIONS_TARGET_DELTA", 0.38),
            min_delta=_env_float("OPTIONS_MIN_DELTA", 0.35),
            max_delta=_env_float("OPTIONS_MAX_DELTA", 0.40),
            iv_assumption=_env_float("OPTIONS_IV_ASSUMPTION", 0.25),
            risk_free_rate=_env_float("OPTIONS_RISK_FREE_RATE", 0.05),
            preferred_business_dte=_env_int("OPTIONS_PREFERRED_BUSINESS_DTE", 2),
            enable_0dte=_env_bool("OPTIONS_ENABLE_0DTE", True),
            require_ml_for_0dte=_env_bool("OPTIONS_REQUIRE_ML_FOR_0DTE", True),
            require_strong_move_for_0dte=_env_bool("OPTIONS_REQUIRE_STRONG_MOVE_FOR_0DTE", True),
            min_ml_0dte_long_05atr=_env_float("OPTIONS_MIN_ML_0DTE_LONG_05ATR", 0.60),
            min_ml_0dte_long_10atr=_env_float("OPTIONS_MIN_ML_0DTE_LONG_10ATR", 0.42),
            min_ml_0dte_short_05atr=_env_float("OPTIONS_MIN_ML_0DTE_SHORT_05ATR", 0.60),
            min_ml_0dte_short_10atr=_env_float("OPTIONS_MIN_ML_0DTE_SHORT_10ATR", 0.42),
            allow_open_put=_env_bool("OPTIONS_ALLOW_OPEN_PUT", True),
            allow_flip_same_bar=_env_bool("OPTIONS_ALLOW_FLIP_SAME_BAR", False),
            strike_increment=_env_float("OPTIONS_STRIKE_INCREMENT", 1.0),
            log_option_quotes=_env_bool("OPTIONS_LOG_OPTION_QUOTES", True),
        )


@dataclass
class OptionPosition:
    side: str  # CALL or PUT
    ticker: str
    opened_ticker: str
    strike: float
    expiration: str
    dte: int
    opened_spot: float
    opened_ts: str
    entry_quote: Optional[Dict[str, Any]] = None


@dataclass
class OptionsState:
    active_position: Optional[OptionPosition] = None
    last_event_ts: Optional[str] = None
    last_action: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active_position": asdict(self.active_position) if self.active_position else None,
            "last_event_ts": self.last_event_ts,
            "last_action": self.last_action,
        }


@dataclass
class OptionsDecision:
    action: str
    ticker: Optional[str] = None
    opened_ticker: Optional[str] = None
    side: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[str] = None
    dte: Optional[int] = None
    spot: Optional[float] = None
    session: str = "unknown"
    reason: str = ""
    quote: Optional[OptionQuote] = None
    state_after: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "ticker": self.ticker,
            "opened_ticker": self.opened_ticker,
            "side": self.side,
            "strike": self.strike,
            "expiration": self.expiration,
            "dte": self.dte,
            "spot": self.spot,
            "session": self.session,
            "reason": self.reason,
            "quote": self.quote.as_dict() if self.quote else None,
            "state_after": self.state_after,
            "meta": self.meta,
        }

    def telegram_block(self) -> str:
        if self.action == "NO_ACTION":
            return f"QQQ options block\nOptions action: NO ACTION\nReason: {self.reason}"
        lines = ["QQQ options block", f"Options action: {self.action}"]
        if self.ticker:
            lines.append(f"Ticker: {self.ticker}")
        if self.opened_ticker:
            lines.append(f"Opened ticker: {self.opened_ticker}")
        if self.spot is not None:
            lines.append(f"Spot={self.spot:.2f} | Strike={self.strike:g} | Exp={self.expiration} | DTE={self.dte}")
        lines.append(f"Session={self.session}")
        if self.quote:
            q = self.quote
            lines.append(
                "Quote: "
                f"bid={_fmt(q.bid)} ask={_fmt(q.ask)} mid={_fmt(q.mid)} last={_fmt(q.last)} "
                f"IV={_fmt(q.iv)} Δ={_fmt(q.delta)} θ={_fmt(q.theta)}"
            )
        lines.append(f"Note: {self.reason}")
        return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4g}"


def state_path(cfg: OptionsConfig) -> Path:
    return Path(cfg.cache_dir) / "options_pending_QQQ.json"


def load_state(cfg: Optional[OptionsConfig] = None) -> OptionsState:
    cfg = cfg or OptionsConfig.from_env()
    p = state_path(cfg)
    if not p.exists():
        return OptionsState()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pos_data = data.get("active_position")
        pos = OptionPosition(**pos_data) if pos_data else None
        return OptionsState(active_position=pos, last_event_ts=data.get("last_event_ts"), last_action=data.get("last_action"))
    except Exception:
        return OptionsState()


def save_state(state: OptionsState, cfg: Optional[OptionsConfig] = None) -> None:
    cfg = cfg or OptionsConfig.from_env()
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_delta(spot: float, strike: float, t_years: float, iv: float, r: float, option_side: str) -> float:
    t_years = max(t_years, 1 / 365.0)
    iv = max(iv, 0.01)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    if option_side == "CALL":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def _round_strike(strike: float, increment: float) -> float:
    if increment <= 0:
        return round(strike)
    return round(strike / increment) * increment


def estimate_strike_by_delta(
    spot: float,
    option_side: str,
    target_abs_delta: float,
    dte: int,
    iv: float,
    r: float,
    increment: float,
) -> Tuple[float, float]:
    t = max(dte, 1) / 252.0
    low = spot * 0.75
    high = spot * 1.25
    target = target_abs_delta
    best_k = spot
    best_delta = 0.0
    for _ in range(80):
        mid = (low + high) / 2.0
        d = _bs_delta(spot, mid, t, iv, r, option_side)
        abs_d = abs(d)
        best_k, best_delta = mid, d
        if option_side == "CALL":
            if abs_d > target:
                low = mid
            else:
                high = mid
        else:
            # Put abs delta increases when strike goes up.
            if abs_d > target:
                high = mid
            else:
                low = mid
    k = _round_strike(best_k, increment)
    d = _bs_delta(spot, k, t, iv, r, option_side)
    return k, d


def add_business_days(d: date, n: int) -> date:
    out = d
    added = 0
    while added < n:
        out += timedelta(days=1)
        if out.weekday() < 5:
            added += 1
    return out


def choose_expiration(ts: str, cfg: OptionsConfig) -> Tuple[date, int, str]:
    today = ny_date(ts)
    if is_friday(ts):
        return today, 0, "Friday rule: use 0DTE, do not roll to next week"
    if is_thursday(ts):
        exp = add_business_days(today, 1)
        return exp, 1, "Thursday rule: use Friday expiry, not Monday"
    exp = add_business_days(today, cfg.preferred_business_dte)
    return exp, cfg.preferred_business_dte, f"Target {cfg.preferred_business_dte} business DTE"


def format_exp(exp: date) -> str:
    return f"{exp.day:02d}{MONTHS[exp.month]}{exp.year}"


def make_option_ticker(root: str, exp: date, side: str, strike: float) -> str:
    strike_str = str(int(strike)) if abs(strike - int(strike)) < 1e-9 else f"{strike:g}"
    cp = "C" if side == "CALL" else "P"
    return f"+{root}.{format_exp(exp)}.{cp}{strike_str}"


def _ml_allows_0dte(decision: StrategyDecision, side: str, cfg: OptionsConfig) -> Tuple[bool, str]:
    ml = decision.ml
    if not cfg.require_ml_for_0dte:
        return True, "ML approval not required for 0DTE by config"
    if not ml.enabled or not ml.model_loaded:
        return False, "0DTE blocked: ML outcome model is not loaded"
    if side == "CALL":
        ok = ml.long_05atr >= cfg.min_ml_0dte_long_05atr and ml.long_10atr >= cfg.min_ml_0dte_long_10atr
        return ok, f"ML CALL approval: p+0.5ATR={ml.long_05atr:.2f}, p+1ATR={ml.long_10atr:.2f}"
    ok = ml.short_05atr >= cfg.min_ml_0dte_short_05atr and ml.short_10atr >= cfg.min_ml_0dte_short_10atr
    return ok, f"ML PUT approval: p-0.5ATR={ml.short_05atr:.2f}, p-1ATR={ml.short_10atr:.2f}"


def _normalize_decision(decision: Any) -> StrategyDecision:
    if isinstance(decision, StrategyDecision):
        return decision
    if isinstance(decision, dict):
        st = normalize_signal_type(decision.get("signal_type") or decision.get("action"))
        from .signal_types import MLApproval, SignalMode, TradeDirection, legacy_action_for_signal_type

        ml_raw = decision.get("ml") or {}
        ml = ml_raw if isinstance(ml_raw, MLApproval) else MLApproval(**{k: v for k, v in ml_raw.items() if k in MLApproval.__dataclass_fields__})
        return StrategyDecision(
            signal_type=st,
            action=legacy_action_for_signal_type(st),
            mode=SignalMode(decision.get("mode", "none")) if str(decision.get("mode", "none")) in SignalMode._value2member_map_ else SignalMode.NONE,
            direction=TradeDirection(decision.get("direction", "none")) if str(decision.get("direction", "none")) in TradeDirection._value2member_map_ else TradeDirection.NONE,
            reason=str(decision.get("reason", "")),
            bar_ts=decision.get("bar_ts"),
            close=decision.get("close"),
            atr=decision.get("atr"),
            regular_session=bool(decision.get("regular_session", False)),
            strong_move=bool(decision.get("strong_move", False)),
            option_open_allowed=bool(decision.get("option_open_allowed", False)),
            ml=ml,
            details=decision.get("details") or {},
        )
    st = normalize_signal_type(decision)
    from .signal_types import legacy_action_for_signal_type

    return StrategyDecision(signal_type=st, action=legacy_action_for_signal_type(st), reason="legacy signal")


def process_options_signal(
    decision: Any,
    *,
    cfg: Optional[OptionsConfig] = None,
    state: Optional[OptionsState] = None,
    quote_provider: Optional[QuoteProvider] = None,
    tradernet_client: Any = None,
    persist: bool = True,
) -> OptionsDecision:
    """Build and persist option action from explicit strategy decision.

    Rules implemented:
      - CLOSE_LONG closes active CALL only; never opens PUT.
      - OPEN_SHORT is required to buy PUT.
      - BUY/OPEN_LONG while active PUT closes PUT first; no same-bar flip by default.
      - 0DTE is allowed only on strong movement plus ML approval when configured.
      - Signals outside regular session do not open options.
      - Real quote fields are fetched/logged when quote provider/client is available.
    """
    cfg = cfg or OptionsConfig.from_env()
    state = state or load_state(cfg)
    qprov = quote_provider or DefaultOptionQuoteProvider(tradernet_client)
    d = _normalize_decision(decision)
    ts = d.bar_ts or datetime.now(timezone.utc).isoformat()
    spot = float(d.close) if d.close is not None else None
    session = session_label(ts)

    if cfg.regular_session_only and not is_regular_session(ts):
        od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="options are regular-session only")
        return _finish(od, d, state, cfg, persist=False)

    active = state.active_position

    # Close actions are explicit and conservative.
    if d.signal_type == SignalType.CLOSE_LONG:
        if active and active.side == "CALL":
            return _close_position("CLOSE CALL", active, d, state, cfg, qprov, persist)
        od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="CLOSE_LONG received, but no active CALL; PUT is not opened")
        return _finish(od, d, state, cfg, persist)

    if d.signal_type == SignalType.CLOSE_SHORT:
        if active and active.side == "PUT":
            return _close_position("CLOSE PUT", active, d, state, cfg, qprov, persist)
        od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="CLOSE_SHORT received, but no active PUT")
        return _finish(od, d, state, cfg, persist)

    if d.signal_type == SignalType.OPEN_LONG:
        if active and active.side == "PUT":
            od = _close_position("CLOSE PUT", active, d, state, cfg, qprov, persist)
            od.reason = "OPEN_LONG closes active PUT; no CALL opened on same bar"
            return od
        if active and active.side == "CALL":
            od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="OPEN_LONG ignored: CALL is already active")
            return _finish(od, d, state, cfg, persist)
        return _open_position("CALL", d, state, cfg, qprov, persist)

    if d.signal_type == SignalType.OPEN_SHORT:
        if not cfg.allow_open_put:
            od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="OPEN_SHORT blocked: PUT opening disabled")
            return _finish(od, d, state, cfg, persist)
        if active and active.side == "CALL":
            od = _close_position("CLOSE CALL", active, d, state, cfg, qprov, persist)
            od.reason = "OPEN_SHORT closes active CALL; no PUT opened on same bar"
            return od
        if active and active.side == "PUT":
            od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="OPEN_SHORT ignored: PUT is already active")
            return _finish(od, d, state, cfg, persist)
        return _open_position("PUT", d, state, cfg, qprov, persist)

    od = OptionsDecision(action="NO_ACTION", spot=spot, session=session, reason="HOLD/no option action")
    return _finish(od, d, state, cfg, persist)


def _open_position(side: str, d: StrategyDecision, state: OptionsState, cfg: OptionsConfig, qprov: QuoteProvider, persist: bool) -> OptionsDecision:
    if d.close is None:
        od = OptionsDecision(action="NO_ACTION", session=session_label(d.bar_ts or ""), reason="cannot open option: missing spot price")
        return _finish(od, d, state, cfg, persist)
    if not d.option_open_allowed:
        od = OptionsDecision(action="NO_ACTION", spot=d.close, session=session_label(d.bar_ts or ""), reason="strategy did not allow option opening")
        return _finish(od, d, state, cfg, persist)

    exp, dte, exp_reason = choose_expiration(d.bar_ts or datetime.now(timezone.utc).isoformat(), cfg)
    if dte == 0:
        if not cfg.enable_0dte:
            od = OptionsDecision(action="NO_ACTION", spot=d.close, session="regular", reason="0DTE disabled")
            return _finish(od, d, state, cfg, persist)
        if cfg.require_strong_move_for_0dte and not d.strong_move:
            od = OptionsDecision(action="NO_ACTION", spot=d.close, session="regular", reason="0DTE blocked: signal is not strong breakout/momentum")
            return _finish(od, d, state, cfg, persist)
        ml_ok, ml_reason = _ml_allows_0dte(d, side, cfg)
        if not ml_ok:
            od = OptionsDecision(action="NO_ACTION", spot=d.close, session="regular", reason=ml_reason)
            return _finish(od, d, state, cfg, persist)
        strike = _round_strike(float(d.close), cfg.strike_increment)
        est_delta = None
        strike_method = "ATM_0DTE"
        reason = f"{exp_reason}; 0DTE allowed: strong movement + {ml_reason}; strike=ATM"
    else:
        strike, est_delta = estimate_strike_by_delta(
            float(d.close), side, cfg.target_delta, dte, cfg.iv_assumption, cfg.risk_free_rate, cfg.strike_increment
        )
        strike_method = "DELTA"
        reason = (
            f"{exp_reason}; strike=DELTA target abs Δ={cfg.target_delta:.2f}, "
            f"est Δ={est_delta:.2f}, IV assumption={cfg.iv_assumption:.0%}"
        )

    ticker = make_option_ticker(cfg.symbol_root, exp, side, strike)
    quote = qprov.get_option_quote(ticker) if cfg.log_option_quotes else None
    pos = OptionPosition(
        side=side,
        ticker=ticker,
        opened_ticker=ticker,
        strike=float(strike),
        expiration=format_exp(exp),
        dte=dte,
        opened_spot=float(d.close),
        opened_ts=d.bar_ts or datetime.now(timezone.utc).isoformat(),
        entry_quote=quote.as_dict() if quote else None,
    )
    state.active_position = pos
    state.last_action = f"BUY {side}"
    state.last_event_ts = d.bar_ts
    od = OptionsDecision(
        action=f"BUY {side}",
        ticker=ticker,
        side=side,
        strike=float(strike),
        expiration=format_exp(exp),
        dte=dte,
        spot=float(d.close),
        session="regular",
        reason=reason,
        quote=quote,
        meta={"strike_method": strike_method, "est_delta": est_delta, "strategy": d.as_dict()},
    )
    return _finish(od, d, state, cfg, persist)


def _close_position(action: str, active: OptionPosition, d: StrategyDecision, state: OptionsState, cfg: OptionsConfig, qprov: QuoteProvider, persist: bool) -> OptionsDecision:
    quote = qprov.get_option_quote(active.ticker) if cfg.log_option_quotes else None
    spot = float(d.close) if d.close is not None else None
    state.active_position = None
    state.last_action = action
    state.last_event_ts = d.bar_ts
    od = OptionsDecision(
        action=action,
        ticker="-" + active.ticker.lstrip("+"),
        opened_ticker=active.opened_ticker,
        side=active.side,
        strike=active.strike,
        expiration=active.expiration,
        dte=active.dte,
        spot=spot,
        session="regular",
        reason=f"{d.signal_type.value} closes active {active.side}; no opposite option is opened",
        quote=quote,
        meta={"entry_position": asdict(active), "strategy": d.as_dict()},
    )
    return _finish(od, d, state, cfg, persist)


def _finish(od: OptionsDecision, d: StrategyDecision, state: OptionsState, cfg: OptionsConfig, persist: bool) -> OptionsDecision:
    od.state_after = state.as_dict()
    logs_dir = Path(cfg.logs_dir)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "bar_ts": d.bar_ts,
        "strategy_signal_type": d.signal_type.value,
        "strategy_action": d.action,
        "option_action": od.action,
        "option_ticker": od.ticker,
        "spot": od.spot,
        "reason": od.reason,
        "quote": od.quote.as_dict() if od.quote else None,
        "decision": od.as_dict(),
    }
    append_jsonl(logs_dir / "options_events.jsonl", row)
    if od.quote:
        append_jsonl(logs_dir / "options_quotes.jsonl", {"bar_ts": d.bar_ts, **od.quote.as_dict()})
    elif cfg.log_option_quotes and od.ticker and od.action != "NO_ACTION":
        append_jsonl(logs_dir / "options_quotes_missing.jsonl", {"bar_ts": d.bar_ts, "ticker": od.ticker, "reason": "quote_provider_returned_none"})
    if persist:
        save_state(state, cfg)
    return od


# Compatibility alias for older scheduler/bot code.
def build_options_signal(decision: Any, **kwargs: Any) -> OptionsDecision:
    return process_options_signal(decision, **kwargs)
