from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo


MONTHS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

_NORMAL = NormalDist()


@dataclass(frozen=True)
class OptionsSignalConfig:
    enabled: bool = True
    regular_session_only: bool = True

    # Expiration rules.
    target_dte: int = 2
    friday_0dte: bool = True
    thursday_to_friday: bool = True
    atm_on_0dte: bool = True

    # Strike rules.
    strike_increment: float = 1.0
    use_delta_strike: bool = True
    delta_min: float = 0.35
    delta_max: float = 0.40
    delta_target: float = 0.375
    iv_assumption: float = 0.25
    risk_free_rate: float = 0.05

    # Fallback when delta strike is disabled.
    strike_mode: str = "OTM"  # ATM or OTM
    otm_pct: float = 0.0025

    # Smart position handling.
    # BUY + flat => open CALL. SELL + active CALL => close CALL.
    # SELL + flat => open PUT if open_put_when_flat=True.
    # BUY + active PUT => close PUT if close_put_on_buy=True.
    open_call_on_buy: bool = True
    open_put_when_flat: bool = True
    close_call_on_sell: bool = True
    close_put_on_buy: bool = True

    # ML is shown but not used as a filter unless min_ml_prob > 0.
    min_ml_prob: float = 0.0
    show_ml_quality: bool = True

    # Persist option position across bot restarts.
    persist_position: bool = True


@dataclass(frozen=True)
class OptionsPosition:
    status: str               # OPEN
    option_side: str          # CALL / PUT
    cp: str                   # C / P
    ticker: str               # +QQQ.20MAY2026.C714
    underlying: str
    expiration: str           # YYYY-MM-DD
    strike: float
    opened_at: str            # ISO UTC
    opened_spot: float
    signal_action: str        # BUY / SELL that opened it


@dataclass(frozen=True)
class OptionTradeSignal:
    action_type: str          # OPEN / CLOSE
    instruction: str          # BUY CALL / BUY PUT / CLOSE CALL / CLOSE PUT
    underlying: str
    signal_action: str        # base signal BUY / SELL
    option_side: str          # CALL / PUT
    cp: str                   # C / P
    expiration: date
    dte: int
    strike: float
    ticker: str               # +QQQ... for open, -QQQ... for close
    position_ticker: str      # +QQQ... stored/opened ticker
    spot: float
    session: str
    strike_method: str        # ATM_0DTE / DELTA / ATM / OTM
    delta_target: float | None
    estimated_delta: float | None
    iv_assumption: float | None
    ml_quality: float | None
    ml_ok: bool
    note: str
    next_position: OptionsPosition | None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_options_signal_config_from_env() -> OptionsSignalConfig:
    delta_min = _clamp(_env_float("OPTIONS_DELTA_MIN", 0.35), 0.01, 0.99)
    delta_max = _clamp(_env_float("OPTIONS_DELTA_MAX", 0.40), 0.01, 0.99)
    if delta_max < delta_min:
        delta_min, delta_max = delta_max, delta_min
    delta_target = _env_float("OPTIONS_DELTA_TARGET", (delta_min + delta_max) / 2.0)
    delta_target = _clamp(delta_target, delta_min, delta_max)

    return OptionsSignalConfig(
        enabled=_env_bool("OPTIONS_SIGNALS_ENABLED", True),
        regular_session_only=_env_bool("OPTIONS_REGULAR_SESSION_ONLY", True),
        target_dte=max(0, min(10, _env_int("OPTIONS_TARGET_DTE", 2))),
        friday_0dte=_env_bool("OPTIONS_FRIDAY_0DTE", True),
        thursday_to_friday=_env_bool("OPTIONS_THURSDAY_TO_FRIDAY", True),
        atm_on_0dte=_env_bool("OPTIONS_ATM_ON_0DTE", True),
        strike_increment=max(0.01, _env_float("OPTIONS_STRIKE_INCREMENT", 1.0)),
        use_delta_strike=_env_bool("OPTIONS_USE_DELTA_STRIKE", True),
        delta_min=delta_min,
        delta_max=delta_max,
        delta_target=delta_target,
        iv_assumption=max(0.01, min(5.0, _env_float("OPTIONS_IV_ASSUMPTION", 0.25))),
        risk_free_rate=max(-0.05, min(0.25, _env_float("OPTIONS_RISK_FREE_RATE", 0.05))),
        strike_mode=os.getenv("OPTIONS_STRIKE_MODE", "OTM").strip().upper(),
        otm_pct=max(0.0, _env_float("OPTIONS_OTM_PCT", 0.0025)),
        open_call_on_buy=_env_bool("OPTIONS_OPEN_CALL_ON_BUY", True),
        open_put_when_flat=_env_bool("OPTIONS_OPEN_PUT_WHEN_FLAT", True),
        close_call_on_sell=_env_bool("OPTIONS_CLOSE_CALL_ON_SELL", True),
        close_put_on_buy=_env_bool("OPTIONS_CLOSE_PUT_ON_BUY", True),
        min_ml_prob=max(0.0, min(1.0, _env_float("OPTIONS_MIN_ML_PROB", 0.0))),
        show_ml_quality=_env_bool("OPTIONS_SHOW_ML_QUALITY", True),
        persist_position=_env_bool("OPTIONS_PERSIST_POSITION", True),
    )


def normalize_underlying(symbol: str) -> str:
    return str(symbol).split(".")[0].upper().strip()


def _to_local_dt(ts: datetime, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    if ts is None:
        ts = datetime.now(timezone.utc)
    if not isinstance(ts, datetime):
        # Handles pandas Timestamp and similar objects.
        try:
            ts = ts.to_pydatetime()
        except Exception:
            ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(tz)


def _to_utc_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def market_session(ts: datetime, tz_name: str = "America/New_York") -> str:
    local = _to_local_dt(ts, tz_name)

    if local.weekday() >= 5:
        return "closed"

    t = local.time()
    if time(9, 30) <= t < time(16, 0):
        return "regular"
    if time(4, 0) <= t < time(9, 30):
        return "premarket"
    if time(16, 0) <= t < time(20, 0):
        return "afterhours"
    return "closed"


def _add_business_days(d: date, n: int) -> date:
    out = d
    added = 0
    while added < n:
        out = out + timedelta(days=1)
        if out.weekday() < 5:
            added += 1
    return out


def choose_expiration(bar_ts: datetime, target_dte: int, tz_name: str, cfg: OptionsSignalConfig) -> tuple[date, int, str]:
    local = _to_local_dt(bar_ts, tz_name)
    today = local.date()
    weekday = local.weekday()  # Mon=0, Thu=3, Fri=4

    if weekday == 4 and cfg.friday_0dte:
        return today, 0, "Friday rule: use 0DTE, do not roll to next week"

    if weekday == 3 and cfg.thursday_to_friday:
        exp = today + timedelta(days=1)
        return exp, 1, "Thursday rule: use Friday expiry, not Monday"

    dte = max(0, int(target_dte))
    if dte == 0:
        return today, 0, "0DTE by config"

    exp = _add_business_days(today, dte)
    actual_dte = sum(1 for i in range(1, (exp - today).days + 1) if (today + timedelta(days=i)).weekday() < 5)
    return exp, actual_dte, f"Target {target_dte} business DTE"


def _round_nearest(x: float, inc: float) -> float:
    return round(x / inc) * inc


def _round_up(x: float, inc: float) -> float:
    return math.ceil(x / inc) * inc


def _round_down(x: float, inc: float) -> float:
    return math.floor(x / inc) * inc


def _normal_cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def _normal_inv_cdf(p: float) -> float:
    return _NORMAL.inv_cdf(_clamp(p, 1e-6, 1.0 - 1e-6))


def _bs_delta(spot: float, strike: float, cp: str, t_years: float, sigma: float, r: float) -> float:
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        if cp == "C":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))
    if cp == "C":
        return _normal_cdf(d1)
    return _normal_cdf(d1) - 1.0


def _choose_delta_strike(spot: float, cp: str, dte: int, cfg: OptionsSignalConfig) -> tuple[float, float]:
    inc = cfg.strike_increment
    target_delta = cfg.delta_target
    sigma = cfg.iv_assumption
    r = cfg.risk_free_rate

    # Calendar-day approximation. This is enough for a practical ticker suggestion,
    # but not a substitute for a real option-chain delta.
    t_years = max(float(dte), 1.0) / 365.0

    if cp == "C":
        d1 = _normal_inv_cdf(target_delta)
    else:
        # Put delta is N(d1)-1. Abs put delta target => N(-d1)=target.
        d1 = -_normal_inv_cdf(target_delta)

    raw_strike = spot * math.exp((r + 0.5 * sigma * sigma) * t_years - d1 * sigma * math.sqrt(t_years))
    strike = _round_nearest(raw_strike, inc)

    # Enforce OTM for non-0DTE delta-based selection.
    if cp == "C":
        if strike <= spot:
            strike = _round_up(spot, inc)
            if strike <= spot:
                strike += inc
    else:
        if strike >= spot:
            strike = _round_down(spot, inc)
            if strike >= spot:
                strike -= inc
        strike = max(inc, strike)

    est_delta = _bs_delta(spot, strike, cp, t_years, sigma, r)
    return float(strike), float(est_delta)


def _choose_fallback_strike(spot: float, cp: str, cfg: OptionsSignalConfig) -> float:
    mode = cfg.strike_mode.upper().strip()
    inc = cfg.strike_increment

    if cp == "C":
        target = spot * (1.0 + cfg.otm_pct)
        strike = _round_nearest(target, inc) if mode == "ATM" else _round_up(target, inc)
        if mode == "OTM" and strike <= spot:
            strike += inc
    else:
        target = spot * (1.0 - cfg.otm_pct)
        strike = _round_nearest(target, inc) if mode == "ATM" else _round_down(target, inc)
        if mode == "OTM" and strike >= spot:
            strike -= inc

    return max(inc, float(strike))


def choose_strike(spot: float, cp: str, dte: int, cfg: OptionsSignalConfig) -> tuple[float, str, float | None]:
    if dte == 0 and cfg.atm_on_0dte:
        strike = _round_nearest(spot, cfg.strike_increment)
        return max(cfg.strike_increment, float(strike)), "ATM_0DTE", None

    if cfg.use_delta_strike:
        strike, est_delta = _choose_delta_strike(spot, cp, dte, cfg)
        return strike, "DELTA", est_delta

    strike = _choose_fallback_strike(spot, cp, cfg)
    method = cfg.strike_mode.upper().strip() if cfg.strike_mode.upper().strip() in {"ATM", "OTM"} else "FALLBACK"
    return strike, method, None


def _format_expiration(d: date) -> str:
    return f"{d.day:02d}{MONTHS[d.month]}{d.year}"


def _format_strike(strike: float) -> str:
    if abs(strike - round(strike)) < 1e-9:
        return str(int(round(strike)))
    return f"{strike:.2f}".rstrip("0").rstrip(".")


def _open_ticker(underlying: str, exp: date, cp: str, strike: float) -> str:
    return f"+{underlying}.{_format_expiration(exp)}.{cp}{_format_strike(strike)}"


def _close_ticker(open_ticker: str) -> str:
    if open_ticker.startswith("+"):
        return "-" + open_ticker[1:]
    if open_ticker.startswith("-"):
        return open_ticker
    return "-" + open_ticker


def _parse_expiration(exp: str | date) -> date:
    if isinstance(exp, date):
        return exp
    return date.fromisoformat(str(exp))


def _position_from_dict(data: dict) -> OptionsPosition | None:
    try:
        if str(data.get("status", "")).upper() != "OPEN":
            return None
        return OptionsPosition(
            status="OPEN",
            option_side=str(data["option_side"]).upper(),
            cp=str(data["cp"]).upper(),
            ticker=str(data["ticker"]),
            underlying=str(data["underlying"]).upper(),
            expiration=str(data["expiration"]),
            strike=float(data["strike"]),
            opened_at=str(data["opened_at"]),
            opened_spot=float(data["opened_spot"]),
            signal_action=str(data.get("signal_action", "")).upper(),
        )
    except Exception:
        return None


def options_position_file(cache_dir: str | Path, symbol: str) -> Path:
    underlying = normalize_underlying(symbol)
    return Path(cache_dir).expanduser().resolve() / f"options_position_{underlying}.json"


def load_options_position(cache_dir: str | Path, symbol: str) -> OptionsPosition | None:
    path = options_position_file(cache_dir, symbol)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _position_from_dict(data)
    except Exception:
        return None


def save_options_position(cache_dir: str | Path, symbol: str, position: OptionsPosition | None) -> None:
    path = options_position_file(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    if position is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(json.dumps(asdict(position), ensure_ascii=False, indent=2), encoding="utf-8")


def _ml_quality_for_signal(signal_action: str, ml_decision) -> float | None:
    if ml_decision is None:
        return None

    action = str(signal_action).upper()
    if action == "BUY":
        value = getattr(ml_decision, "long_prob", None)
    elif action == "SELL":
        value = getattr(ml_decision, "short_prob", None)
    else:
        value = None

    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _ml_ok(signal_action: str, ml_decision, cfg: OptionsSignalConfig) -> tuple[float | None, bool]:
    quality = _ml_quality_for_signal(signal_action, ml_decision)
    if cfg.min_ml_prob <= 0.0 or quality is None:
        return quality, True
    return quality, quality >= cfg.min_ml_prob


def _make_open_position(
    *,
    underlying: str,
    option_side: str,
    cp: str,
    ticker: str,
    exp: date,
    strike: float,
    bar_ts: datetime,
    spot: float,
    signal_action: str,
) -> OptionsPosition:
    return OptionsPosition(
        status="OPEN",
        option_side=option_side,
        cp=cp,
        ticker=ticker,
        underlying=underlying,
        expiration=exp.isoformat(),
        strike=float(strike),
        opened_at=_to_utc_iso(bar_ts),
        opened_spot=float(spot),
        signal_action=signal_action,
    )


def build_options_signal(
    *,
    symbol: str,
    signal_action: str,
    spot: float,
    bar_ts: datetime,
    tz_name: str = "America/New_York",
    ml_decision=None,
    current_position: OptionsPosition | None = None,
    config: OptionsSignalConfig | None = None,
) -> OptionTradeSignal | None:
    cfg = config or load_options_signal_config_from_env()

    if not cfg.enabled:
        return None

    action = str(signal_action).upper().strip()
    if action not in {"BUY", "SELL"}:
        return None

    session = market_session(bar_ts, tz_name)
    if cfg.regular_session_only and session != "regular":
        return None

    try:
        spot_f = float(spot)
    except Exception:
        return None
    if spot_f <= 0:
        return None

    underlying = normalize_underlying(symbol)
    position = current_position
    if position is not None and position.underlying != underlying:
        position = None

    ml_quality, ml_ok = _ml_ok(action, ml_decision, cfg)

    # Smart close logic has priority over opening an opposite option.
    if action == "SELL" and position is not None and position.option_side == "CALL" and cfg.close_call_on_sell:
        exp = _parse_expiration(position.expiration)
        dte = max(0, (exp - _to_local_dt(bar_ts, tz_name).date()).days)
        return OptionTradeSignal(
            action_type="CLOSE",
            instruction="CLOSE CALL",
            underlying=underlying,
            signal_action=action,
            option_side="CALL",
            cp="C",
            expiration=exp,
            dte=dte,
            strike=position.strike,
            ticker=_close_ticker(position.ticker),
            position_ticker=position.ticker,
            spot=spot_f,
            session=session,
            strike_method="EXISTING_POSITION",
            delta_target=None,
            estimated_delta=None,
            iv_assumption=None,
            ml_quality=ml_quality,
            ml_ok=ml_ok,
            note="SELL signal closes the active CALL; no PUT is opened",
            next_position=None,
        )

    if action == "BUY" and position is not None and position.option_side == "PUT" and cfg.close_put_on_buy:
        exp = _parse_expiration(position.expiration)
        dte = max(0, (exp - _to_local_dt(bar_ts, tz_name).date()).days)
        return OptionTradeSignal(
            action_type="CLOSE",
            instruction="CLOSE PUT",
            underlying=underlying,
            signal_action=action,
            option_side="PUT",
            cp="P",
            expiration=exp,
            dte=dte,
            strike=position.strike,
            ticker=_close_ticker(position.ticker),
            position_ticker=position.ticker,
            spot=spot_f,
            session=session,
            strike_method="EXISTING_POSITION",
            delta_target=None,
            estimated_delta=None,
            iv_assumption=None,
            ml_quality=ml_quality,
            ml_ok=ml_ok,
            note="BUY signal closes the active PUT; no CALL is opened",
            next_position=None,
        )

    # Avoid duplicate option-open signals while the same side is already open.
    if action == "BUY" and position is not None and position.option_side == "CALL":
        return None
    if action == "SELL" and position is not None and position.option_side == "PUT":
        return None

    if action == "BUY":
        if not cfg.open_call_on_buy:
            return None
        option_side = "CALL"
        cp = "C"
        instruction = "BUY CALL"
    else:
        if not cfg.open_put_when_flat:
            return None
        option_side = "PUT"
        cp = "P"
        instruction = "BUY PUT"

    exp, dte, exp_note = choose_expiration(bar_ts, cfg.target_dte, tz_name, cfg)
    strike, strike_method, est_delta = choose_strike(spot_f, cp, dte, cfg)
    open_ticker = _open_ticker(underlying, exp, cp, strike)

    next_position = _make_open_position(
        underlying=underlying,
        option_side=option_side,
        cp=cp,
        ticker=open_ticker,
        exp=exp,
        strike=strike,
        bar_ts=bar_ts,
        spot=spot_f,
        signal_action=action,
    )

    note_parts = [
        exp_note,
        "regular-session only" if cfg.regular_session_only else "all-sessions",
        f"strike={strike_method}",
    ]
    if strike_method == "DELTA":
        note_parts.append(f"target delta {cfg.delta_min:.2f}-{cfg.delta_max:.2f}")
        note_parts.append(f"IV assumption {cfg.iv_assumption:.0%}")
    elif strike_method == "ATM_0DTE":
        note_parts.append("0DTE uses ATM strike")
    if cfg.min_ml_prob > 0.0 and ml_quality is not None:
        note_parts.append(f"ML {'OK' if ml_ok else 'weak'}")

    return OptionTradeSignal(
        action_type="OPEN",
        instruction=instruction,
        underlying=underlying,
        signal_action=action,
        option_side=option_side,
        cp=cp,
        expiration=exp,
        dte=dte,
        strike=strike,
        ticker=open_ticker,
        position_ticker=open_ticker,
        spot=spot_f,
        session=session,
        strike_method=strike_method,
        delta_target=cfg.delta_target if strike_method == "DELTA" else None,
        estimated_delta=est_delta,
        iv_assumption=cfg.iv_assumption if strike_method == "DELTA" else None,
        ml_quality=ml_quality,
        ml_ok=ml_ok,
        note=", ".join(note_parts),
        next_position=next_position,
    )


# Backward-compatible alias for the earlier patch.
def build_option_idea(**kwargs):
    return build_options_signal(**kwargs)


def option_signal_text_lines(signal: OptionTradeSignal) -> list[str]:
    quality = "n/a" if signal.ml_quality is None else f"{signal.ml_quality:.2f}"
    est_delta = "n/a" if signal.estimated_delta is None else f"{signal.estimated_delta:.2f}"
    delta_target = "n/a" if signal.delta_target is None else f"{signal.delta_target:.2f}"
    iv_txt = "n/a" if signal.iv_assumption is None else f"{signal.iv_assumption:.0%}"

    if signal.action_type == "CLOSE":
        return [
            f"Options action: {signal.instruction}",
            f"Ticker: {signal.ticker}",
            f"Opened ticker: {signal.position_ticker}",
            f"Spot={signal.spot:.2f} | Strike={_format_strike(signal.strike)} | Exp={_format_expiration(signal.expiration)} | DTE={signal.dte}",
            f"Session={signal.session} | ML quality={quality}",
            f"Note: {signal.note}",
        ]

    return [
        f"Options action: {signal.instruction}",
        f"Ticker: {signal.ticker}",
        f"Spot={signal.spot:.2f} | Strike={_format_strike(signal.strike)} | Exp={_format_expiration(signal.expiration)} | DTE={signal.dte}",
        f"Strike method={signal.strike_method} | target Δ={delta_target} | est Δ={est_delta} | IV={iv_txt}",
        f"Session={signal.session} | ML quality={quality}",
        f"Note: {signal.note}",
    ]


# Backward-compatible alias for the earlier patch.
def option_idea_text_lines(signal: OptionTradeSignal) -> list[str]:
    return option_signal_text_lines(signal)
