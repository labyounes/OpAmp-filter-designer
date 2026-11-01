"""Numerics, value parsing/formatting, E-series and time/frequency simulation.

No GUI code lives here, so everything can be tested headless.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

TWO_PI = 2.0 * math.pi

# --------------------------------------------------------------------------
# E-series
# --------------------------------------------------------------------------
_E96 = [
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
    1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
    1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
    2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
    3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
    5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
    7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
]
E_SERIES = {
    "E6": [1.0, 1.5, 2.2, 3.3, 4.7, 6.8],
    "E12": [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
    "E24": [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
            3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
    "E48": _E96[::2],
    "E96": _E96,
}
EXACT = "Exact (no rounding)"
SERIES_CHOICES = [EXACT, "E6", "E12", "E24", "E48", "E96"]


def snap(value: float, series: str | None, mode: str = "nearest") -> float:
    """Round `value` to a standard E-series value.

    mode: "nearest" (log distance), "down" (<= value) or "up" (>= value).
    """
    if series not in E_SERIES or not math.isfinite(value) or value <= 0:
        return value
    mant = E_SERIES[series]
    dec = math.floor(math.log10(value))
    cands = [float(f"{m * 10.0 ** d:.3g}") for d in (dec - 1, dec, dec + 1) for m in mant]
    if mode == "down":
        return max(c for c in cands if c <= value * (1 + 1e-9))
    if mode == "up":
        return min(c for c in cands if c >= value * (1 - 1e-9))
    return min(cands, key=lambda c: abs(math.log(c / value)))


# --------------------------------------------------------------------------
# Value parsing / formatting
# --------------------------------------------------------------------------
_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3,
           "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}
_UNITS = {"", "%", "Ω", "ohm", "ohms", "Ohm", "F", "H", "Hz", "hz", "HZ", "V", "A", "s", "dB", "db"}
_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$")
_RKM = re.compile(r"^(\d+)([pnuµμmkKMGR])(\d+)$")


def parse_value(text: str) -> float:
    """Parse '4.7k', '4k7', '100nF', '2.2u', '1e-9', '10 Meg', 'inf' ..."""
    s = str(text).strip().replace(",", ".")
    if not s:
        raise ValueError("empty")
    if s.lower() in ("inf", "∞", "infinity"):
        return math.inf
    m = _RKM.match(s)
    if m:
        mult = 1.0 if m.group(2) == "R" else _PREFIX[m.group(2)]
        return float(f"{m.group(1)}.{m.group(3)}") * mult
    m = _NUM.match(s)
    if not m:
        raise ValueError(f"cannot read '{text}'")
    val = float(m.group(1))
    suf = m.group(2).strip()
    if suf in _UNITS:
        return val
    if suf.lower().startswith("meg") and suf[3:] in _UNITS:
        return val * 1e6
    if suf[0] in _PREFIX and suf[1:] in _UNITS:
        return val * _PREFIX[suf[0]]
    raise ValueError(f"cannot read '{text}'")


_PFX = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
        (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")]


def fmt(value, unit: str = "", digits: int = 4) -> str:
    """Engineering notation: 4700 -> '4.7 kΩ'."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "∞" + (" " + unit if unit else "") if value > 0 else "-∞"
    if value == 0:
        return f"0 {unit}".strip()
    a = abs(value)
    scale, pre = _PFX[-1]
    for sc, p in _PFX:
        if a >= sc * 0.99995:
            scale, pre = sc, p
            break
    return f"{value / scale:.{digits}g} {pre}{unit}".strip()


def to_db(x) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-300))


# --------------------------------------------------------------------------
# Field / result structures shared by all circuits
# --------------------------------------------------------------------------
@dataclass
class Field:
    key: str
    label: str
    default: str
    unit: str = ""            # "Ω", "F", "H", "Hz", "V", "" ...
    kind: str = "num"         # "num" | "int" | "choice"
    choices: list | None = None
    zero_ok: bool = False     # allow 0 for component-like fields
    signed: bool = False      # allow negative values (e.g. gain)
    show_if: tuple | None = None   # (choice_key, prefix or tuple of prefixes): only used when that choice matches


@dataclass
class Model:
    """Transfer function H(s) = gain * prod(num_i/den_i) and a constant output term."""
    stages: list = field(default_factory=list)   # [(num, den), ...] highest power first
    gain: float = 1.0
    dc_out: float = 0.0                           # constant added to the output (volts)


@dataclass
class Outcome:
    rows: list = field(default_factory=list)      # ("h"|"kv"|"warn"|"note", label, value)
    model: Model | None = None
    netlist: str = ""
    f_char: float = 1e3                           # characteristic frequency (for plot range / default f_in)
    clip: bool = True                             # op-amp output limited by supply rails
    info: dict | None = None                      # extra data for the schematic (stages, ladder parts)
    std: dict | None = None                       # component values used (design mode) for 'use in analyze'


def H(title):
    return ("h", title, "")


def KV(label, value):
    return ("kv", label, value)


def WARN(text):
    return ("warn", text, "")


def NOTE(text):
    return ("note", text, "")


def field_visible(f: Field, fields: list, texts: dict) -> bool:
    if not f.show_if:
        return True
    key, prefixes = f.show_if
    cur = texts.get(key)
    if cur is None:
        cur = next((x.default for x in fields if x.key == key), "")
    return str(cur).startswith(prefixes)


def parse_fields(fields: list[Field], texts: dict) -> dict:
    """Convert raw text entries to numbers; raises ValueError with a readable message.
    Fields hidden by their show_if condition are skipped."""
    out = {}
    for f in fields:
        if not field_visible(f, fields, texts):
            continue
        raw = texts.get(f.key, f.default)
        if f.kind == "choice":
            out[f.key] = raw
            continue
        try:
            v = parse_value(raw)
            if f.kind == "int":
                v = int(round(v))
        except ValueError:
            raise ValueError(f"{f.label}: cannot read '{raw}'") from None
        if f.unit in ("Ω", "F", "H", "Hz") and not f.signed:
            if v < 0 or (v == 0 and not f.zero_ok):
                raise ValueError(f"{f.label} must be greater than 0")
        out[f.key] = v
    return out


# --------------------------------------------------------------------------
# Polynomial helpers
# --------------------------------------------------------------------------
def P(*coeffs):
    """Polynomial (highest power first) with leading zeros removed."""
    a = np.trim_zeros(np.array(coeffs, dtype=float), "f")
    return a if a.size else np.array([0.0])


def _strip_common_s(num, den):
    num, den = list(num), list(den)
    while len(num) > 1 and len(den) > 1 and num[-1] == 0 and den[-1] == 0:
        num.pop()
        den.pop()
    return P(*num), P(*den)


def with_gbw(num, den, noise_gain, gbw):
    """Closed loop with a single-pole op-amp A(s) = wt/s.

    H = H0 / (1 + NG(s)/A(s)),  NG(s) = n/d  ->  H = H0 * wt*d / (wt*d + s*n)
    """
    if not gbw or gbw <= 0:
        return P(*num), P(*den)
    wt = TWO_PI * gbw
    n, d = (P(*x) for x in noise_gain)
    num2 = np.polymul(num, wt * d)
    den2 = np.polymul(den, np.polyadd(wt * d, np.polymul([1.0, 0.0], n)))
    return _strip_common_s(num2, den2)


# --------------------------------------------------------------------------
# Frequency response
# --------------------------------------------------------------------------
def freq_response(model: Model, f):
    s = 2j * math.pi * np.asarray(f, dtype=float)
    h = np.full(s.shape, model.gain, dtype=complex)
    for num, den in model.stages:
        h = h * np.polyval(num, s) / np.polyval(den, s)
    return h


def characteristic_freqs(model: Model):
    """Absolute pole/zero frequencies in Hz (used to choose a sensible plot range)."""
    out = []
    for num, den in model.stages:
        for poly in (num, den):
            p = np.trim_zeros(np.atleast_1d(poly), "f")
            if len(p) > 1:
                for r in np.roots(p):
                    if abs(r) > 1e-3:
                        out.append(abs(r) / TWO_PI)
    return out


def default_freq_axis(model: Model, f_char: float, f_extra: float | None = None, n: int = 800):
    fs = characteristic_freqs(model) + [f_char]
    if f_extra and f_extra > 0:
        fs.append(f_extra)
    lo, hi = min(fs), max(fs)
    lo, hi = lo / 30.0, hi * 30.0
    if hi / lo < 1e3:
        c = math.sqrt(lo * hi)
        lo, hi = c / 40.0, c * 40.0
    return np.logspace(math.log10(lo), math.log10(hi), n)


# --------------------------------------------------------------------------
# Time-domain simulation (exact discretisation of every 1st/2nd order stage)
# --------------------------------------------------------------------------
def _make_proper(num, den, w_hf):
    num, den = P(*num), P(*den)
    extra = (len(num) - 1) - (len(den) - 1)
    for _ in range(max(0, extra)):
        den = np.polymul(den, [1.0 / w_hf, 1.0])
    return num, den


def _natural_w(den, fallback):
    d = np.trim_zeros(np.atleast_1d(den), "f")
    if len(d) >= 2 and d[-1] != 0:
        return abs(d[-1] / d[0]) ** (1.0 / (len(d) - 1))
    return fallback


def _run_stage(num, den, u, dt, w_ref):
    # Improper stages (ideal differentiator) get a pole at fs/10 so they can be simulated.
    num, den = _make_proper(num, den, TWO_PI / (10.0 * dt))
    w = _natural_w(den, w_ref)
    # time scaling s = w * s' keeps all coefficients near 1 (well conditioned)
    num_s = num * w ** np.arange(len(num) - 1, -1, -1)
    den_s = den * w ** np.arange(len(den) - 1, -1, -1)
    bd, ad, _ = signal.cont2discrete((num_s, den_s), dt * w, method="foh")
    return signal.lfilter(np.squeeze(bd), ad, u)


def simulate(model: Model, t, u, f_ref: float):
    dt = float(t[1] - t[0])
    y = np.asarray(u, dtype=float)
    w_ref = TWO_PI * max(f_ref, 1e-9)
    for num, den in model.stages:
        y = _run_stage(num, den, y, dt, w_ref)
    return model.gain * y + model.dc_out


def make_input(kind: str, t, amp: float, offset: float, f: float):
    w = TWO_PI * f * np.asarray(t)
    if kind == "Sine":
        x = amp * np.sin(w)
    elif kind == "Square":
        x = amp * signal.square(w)
    elif kind == "Triangle":
        x = amp * signal.sawtooth(w, 0.5)
    elif kind == "Step":
        x = amp * (np.asarray(t) >= 0.1 * t[-1]).astype(float)
    else:  # DC
        x = np.zeros_like(w)
    return x + offset


def minus3db_points(model: Model, f_lo: float, f_hi: float, n: int = 4000):
    """Frequencies where |H| crosses (peak / sqrt 2); returns (list_of_Hz, peak_gain_dB)."""
    f = np.logspace(math.log10(f_lo), math.log10(f_hi), n)
    mag = np.abs(freq_response(model, f))
    peak = mag.max()
    if peak <= 0:
        return [], -300.0
    lvl = peak / math.sqrt(2.0)
    out = []
    for i in range(n - 1):
        a, b = mag[i] - lvl, mag[i + 1] - lvl
        if a == 0 or a * b < 0:
            x = a / (a - b) if a != b else 0.0
            out.append(10 ** (math.log10(f[i]) + x * (math.log10(f[i + 1]) - math.log10(f[i]))))
    return out, to_db(peak)
