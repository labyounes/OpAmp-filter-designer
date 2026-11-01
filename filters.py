"""Filter synthesis: pole tables, active 2nd-order stage designs and passive LC ladders."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import signal

from core import E_SERIES, TWO_PI, P, snap

RESPONSES = ["Butterworth", "Bessel", "Chebyshev 0.1 dB", "Chebyshev 0.5 dB",
             "Chebyshev 1 dB", "Chebyshev 2 dB", "Chebyshev 3 dB"]


def cheb_ripple(response: str) -> float:
    return float(response.split()[1])


def proto_poles(response: str, n: int):
    """Poles of the normalised analogue low-pass prototype (cut-off 1 rad/s).

    Butterworth/Bessel: -3 dB at 1 rad/s.  Chebyshev: ripple edge at 1 rad/s.
    """
    if response == "Butterworth":
        _, p, _ = signal.buttap(n)
    elif response == "Bessel":
        _, p, _ = signal.besselap(n, norm="mag")
    elif response.startswith("Chebyshev"):
        _, p, _ = signal.cheb1ap(n, cheb_ripple(response))
    else:
        raise ValueError(response)
    return np.asarray(p, dtype=complex)


def pole_pairs(poles):
    """-> (pairs [(w0, Q)] sorted by Q, reals [w0])."""
    pairs, reals = [], []
    for r in poles:
        if abs(r.imag) <= 1e-9 * max(abs(r), 1e-12):
            reals.append(-r.real)
        elif r.imag > 0:
            pairs.append((abs(r), abs(r) / (-2.0 * r.real)))
    pairs.sort(key=lambda t: t[1])
    return pairs, reals


# --------------------------------------------------------------------------
# Active 2nd-order stage transfer functions (all ideal op-amps)
# --------------------------------------------------------------------------
def sk_lp_tf(R1, R2, C1, C2):
    return P(1.0), P(R1 * R2 * C1 * C2, C2 * (R1 + R2), 1.0)


def sk_hp_tf(R1, R2, C1, C2):
    a = R1 * R2 * C1 * C2
    return P(a, 0.0, 0.0), P(a, R1 * (C1 + C2), 1.0)


def mfb_lp_tf(R1, R2, R3, Cg, Cf):
    k = R3 / R1
    return P(-k), P(R2 * R3 * Cg * Cf, Cf * (R2 + R3 + R2 * R3 / R1), 1.0)


def mfb_bp_tf(R1, R2, R3, C1, C2):
    return (P(-1.0 / (R1 * C1), 0.0),
            P(1.0, (1.0 / C1 + 1.0 / C2) / R2, (R1 + R3) / (R1 * R2 * R3 * C1 * C2)))


def twin_t_tf(R, C, k=0.0):
    w0 = 1.0 / (R * C)
    return P(1.0, 0.0, w0 ** 2), P(1.0, 4.0 * (1.0 - k) * w0, w0 ** 2)


def f0_q(num, den):
    """f0 [Hz] and Q of a 2nd-order denominator."""
    d = P(*den)
    if len(d) != 3:
        return None, None
    w0 = math.sqrt(d[2] / d[0])
    return w0 / TWO_PI, w0 / (d[1] / d[0]) if d[1] else math.inf


# --------------------------------------------------------------------------
# Stage designs: return a component dict. `snapc` rounds capacitors to the E-series.
# --------------------------------------------------------------------------
def sk_lp_design(f0, Q, C2, snapc):
    """Unity-gain Sallen-Key low-pass; R1 != R2 so that f0 *and* Q are exact for any C1 >= 4Q^2 C2."""
    w0 = TWO_PI * f0
    C2 = snapc(C2)
    C1 = snapc(4.0 * Q * Q * C2, "up")
    S = 1.0 / (Q * w0 * C2)
    Pr = 1.0 / (w0 * w0 * C1 * C2)
    D = math.sqrt(max(S * S - 4.0 * Pr, 0.0))
    return {"R1": (S + D) / 2.0, "R2": (S - D) / 2.0, "C1": C1, "C2": C2}


def sk_hp_design(f0, Q, C, snapc):
    w0 = TWO_PI * f0
    C = snapc(C)
    return {"R1": 1.0 / (2.0 * Q * w0 * C), "R2": 2.0 * Q / (w0 * C), "C1": C, "C2": C}


def mfb_lp_design(f0, Q, K, Cf, snapc):
    """Inverting MFB low-pass with gain -K.  Cg (to ground) must be >= 4 Q^2 (1+K) Cf."""
    w0 = TWO_PI * f0
    Cf = snapc(Cf)
    Cg = snapc(4.0 * Q * Q * (1.0 + K) * Cf, "up")
    S = 1.0 / (Q * w0 * Cf)
    Pr = (1.0 + K) / (w0 * w0 * Cg * Cf)
    D = math.sqrt(max(S * S - 4.0 * Pr, 0.0))
    R3 = (S + D) / 2.0
    R2 = (S - D) / 2.0 / (1.0 + K)
    return {"R1": R3 / K, "R2": R2, "R3": R3, "Cg": Cg, "Cf": Cf}


def mfb_bp_design(f0, Q, H0, C, snapc):
    """Delyiannis-Friend band-pass, C1 = C2 = C, gain -H0 at f0 (needs H0 < 2 Q^2)."""
    w0 = TWO_PI * f0
    C = snapc(C)
    R2 = 2.0 * Q / (w0 * C)
    R1 = Q / (H0 * w0 * C)
    den = 2.0 * Q * Q / H0 - 1.0
    R3 = R1 / den if den > 0 else math.inf
    return {"R1": R1, "R2": R2, "R3": R3, "C1": C, "C2": C}


# --------------------------------------------------------------------------
# Passive LC ladder
# --------------------------------------------------------------------------
def ladder_g(response: str, n: int):
    """Prototype element values g1..gn and the normalised load g_{n+1} (source = 1)."""
    if response == "Butterworth":
        g = [2.0 * math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
        return g, 1.0
    if response.startswith("Chebyshev"):
        ap = cheb_ripple(response)
        beta = math.log(1.0 / math.tanh(ap / 17.37))
        gam = math.sinh(beta / (2.0 * n))
        a = [math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
        b = [gam ** 2 + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)]
        g = [2.0 * a[0] / gam]
        for k in range(2, n + 1):
            g.append(4.0 * a[k - 2] * a[k - 1] / (b[k - 2] * g[-1]))
        load = 1.0 if n % 2 else 1.0 / math.tanh(beta / 4.0) ** 2
        return g, load
    raise ValueError(f"ladder not supported for {response}")


def ladder_proto_denominator(g, series_first: bool, load_norm: float):
    """Denominator polynomial P(s) of H_lp(s) = 1/P(s) for the normalised ladder (Rs = 1)."""
    if series_first:
        RL = load_norm          # g0 is a source resistance of 1, load = g_{n+1}
    else:
        RL = 1.0 / load_norm    # g0 is a source conductance of 1, load conductance = g_{n+1}
    # ABCD chain of polynomial entries
    A, B, C, D = P(1.0), P(0.0), P(0.0), P(1.0)
    series = series_first
    for gk in g:
        if series:   # series L = g*s
            e = (P(1.0), P(gk, 0.0), P(0.0), P(1.0))
        else:        # shunt C = g*s
            e = (P(1.0), P(0.0), P(gk, 0.0), P(1.0))
        A, B, C, D = (np.polyadd(np.polymul(A, e[0]), np.polymul(B, e[2])),
                      np.polyadd(np.polymul(A, e[1]), np.polymul(B, e[3])),
                      np.polyadd(np.polymul(C, e[0]), np.polymul(D, e[2])),
                      np.polyadd(np.polymul(C, e[1]), np.polymul(D, e[3])))
        series = not series
    # H = V2/Vs = 1 / (A + B/RL + Rs*C + Rs*D/RL), Rs = 1
    den = np.polyadd(np.polyadd(A, B / RL), np.polyadd(C, D / RL))
    return P(*den), RL


def _sections(poles):
    """Group poles into real 2nd-order sections: list of den polys (len 3), plus singles."""
    poles = list(poles)
    cplx = [p for p in poles if p.imag > 1e-9 * max(abs(p), 1e-30)]
    reals = sorted([p.real for p in poles if abs(p.imag) <= 1e-9 * max(abs(p), 1e-30)])
    dens = [np.real(np.poly([p, np.conj(p)])) for p in cplx]
    singles = []
    while len(reals) >= 2:
        a, b = reals.pop(), reals.pop()
        dens.append(np.real(np.poly([a, b])))
    for r in reals:
        singles.append(np.real(np.poly([r])))
    return dens, singles


def ladder_model(kind: str, response: str, n: int, series_first: bool, fc: float, bw: float):
    """Model (stages, gain) of the denormalised ladder incl. source and load terminations."""
    from core import Model
    g, load = ladder_g(response, n)
    Pden, RL = ladder_proto_denominator(g, series_first, load)
    a_n = Pden[0]
    a_0 = Pden[-1]
    pp = np.roots(Pden)
    wc = TWO_PI * fc
    stages = []
    k = 1.0
    if kind == "Low-pass":
        poles = pp * wc
        dens, singles = _sections(poles)
        nums = [P(1.0)] * (len(dens) + len(singles))
        k = wc ** n / a_n
    elif kind == "High-pass":
        poles = wc / pp
        dens, singles = _sections(poles)
        nums = [P(1.0, 0.0, 0.0)] * len(dens) + [P(1.0, 0.0)] * len(singles)
        k = 1.0 / a_0
    else:  # Band-pass: fc = centre, bw = bandwidth in Hz
        B = TWO_PI * bw
        w0 = wc
        poles = []
        for p in pp:
            disc = np.sqrt(complex(p * p * B * B - 4.0 * w0 * w0))
            poles += [(p * B + disc) / 2.0, (p * B - disc) / 2.0]
        dens, singles = _sections(poles)
        nums = [P(1.0, 0.0)] * len(dens)
        k = B ** n / a_n
    allden = dens + singles
    # normalise every section to unit magnitude at its own natural frequency
    gain = k
    for num, den in zip(nums, allden):
        wn = abs(den[-1] / den[0]) ** (1.0 / (len(den) - 1))
        s = 1j * wn
        g = abs(np.polyval(num, s) / np.polyval(den, s))
        gain *= g
        stages.append((num / g, den))
    return Model(stages=stages, gain=gain), RL


def ladder_components(kind: str, g, series_first: bool, Z0: float, fc: float, bw: float, RL_norm: float):
    """Denormalised component list: [(designator, role, {name: value})]."""
    wc = TWO_PI * fc
    B = TWO_PI * bw
    out = []
    series = series_first
    ls = cs = 0
    for gk in g:
        if kind == "Low-pass":
            if series:
                ls += 1
                out.append((f"L{ls}", "series", {"L": gk * Z0 / wc}))
            else:
                cs += 1
                out.append((f"C{cs}", "shunt", {"C": gk / (Z0 * wc)}))
        elif kind == "High-pass":
            if series:
                cs += 1
                out.append((f"C{cs}", "series", {"C": 1.0 / (wc * gk * Z0)}))
            else:
                ls += 1
                out.append((f"L{ls}", "shunt", {"L": Z0 / (wc * gk)}))
        else:
            if series:
                ls += 1
                out.append((f"L{ls}/C{ls}", "series LC (in series)",
                            {"L": gk * Z0 / B, "C": B / (wc * wc * gk * Z0)}))
            else:
                ls += 1
                out.append((f"L{ls}/C{ls}", "shunt LC (in parallel)",
                            {"L": Z0 * B / (wc * wc * gk), "C": gk / (Z0 * B)}))
        series = not series
    return out, RL_norm * Z0


def _sc(poly, w):
    """Coefficients of p(w*s~) (highest power first)."""
    p = np.atleast_1d(np.array(poly, dtype=float))
    return p * w ** np.arange(len(p) - 1, -1, -1)


def ladder_model_from_elements(kind: str, elems, Rs: float, RL: float, ws: float):
    """Exact model of a ladder built from the given (possibly rounded) element values.

    elems: [(role, {"L":..,"C":..}), ...] ordered from the source.  role starts with
    "series" or "shunt".  Works in the scaled variable s~ = s/ws for good conditioning.
    """
    from core import Model
    A, B, C, D = P(1.0), P(0.0), P(0.0), P(1.0)
    prod_d = P(1.0)
    for role, v in elems:
        series = role.startswith("series")
        if kind == "Low-pass":
            zn, zd = ([v["L"], 0.0], [1.0]) if series else ([v["C"], 0.0], [1.0])
        elif kind == "High-pass":
            zn, zd = ([1.0], [v["C"], 0.0]) if series else ([1.0], [v["L"], 0.0])
        else:
            if series:
                zn, zd = [v["L"] * v["C"], 0.0, 1.0], [v["C"], 0.0]
            else:
                zn, zd = [v["L"] * v["C"], 0.0, 1.0], [v["L"], 0.0]
        zn, zd = P(*_sc(zn, ws)), P(*_sc(zd, ws))
        e = (zd, zn, P(0.0), zd) if series else (zd, P(0.0), zn, zd)
        A, B, C, D = (np.polyadd(np.polymul(A, e[0]), np.polymul(B, e[2])),
                      np.polyadd(np.polymul(A, e[1]), np.polymul(B, e[3])),
                      np.polyadd(np.polymul(C, e[0]), np.polymul(D, e[2])),
                      np.polyadd(np.polymul(C, e[1]), np.polymul(D, e[3])))
        prod_d = np.polymul(prod_d, zd)
    den = P(*np.polyadd(np.polyadd(A, B / RL), np.polyadd(Rs * C, Rs * D / RL)))
    num = P(*prod_d)
    n, m = len(den) - 1, len(num) - 1
    if m > 0 and not np.allclose(num[1:], 0.0):
        raise ValueError("unexpected transmission zeros in ladder")
    poles = np.roots(den) * ws
    dens, singles = _sections(poles)
    allden = dens + singles
    orders = [len(d) - 1 for d in allden]
    if m == 0:
        nums = [P(1.0)] * len(allden)
    elif m == sum(orders):
        nums = [P(*([1.0] + [0.0] * o)) for o in orders]
    elif 2 * m == sum(orders):
        nums = [P(*([1.0] + [0.0] * (o // 2))) for o in orders]
    else:
        raise ValueError("cannot distribute zeros")
    k = (num[0] / den[0]) * ws ** (n - m)
    gain = k
    stages = []
    for nu, de in zip(nums, allden):
        wn = abs(de[-1] / de[0]) ** (1.0 / (len(de) - 1))
        g = abs(np.polyval(nu, 1j * wn) / np.polyval(de, 1j * wn))
        gain *= g
        stages.append((nu / g, de))
    return Model(stages=stages, gain=gain)


# --------------------------------------------------------------------------
# Best standard-value resistor combination for a 2nd-order stage
# --------------------------------------------------------------------------
def nearest_values(v: float, series: str, n: int = 5):
    dec = math.floor(math.log10(v))
    cands = {float(f"{m * 10.0 ** d:.3g}") for d in (dec - 1, dec, dec + 1) for m in E_SERIES[series]}
    return sorted(cands, key=lambda c: abs(math.log(c / v)))[:n]


def refine_stage(tf_of, ideal_res: dict, fixed: dict, series: str, f0t: float, Qt: float, extra=None, n_near: int = 5):
    """Try the n nearest standard values for every resistor and keep the combination whose
    f0 and Q (and optionally gain via `extra(tf, f0)`) come closest to the target."""
    names = list(ideal_res)
    lists = [nearest_values(ideal_res[k], series, n_near) for k in names]
    best_err, best = None, None
    for combo in itertools.product(*lists):
        comps = dict(zip(names, combo))
        comps.update(fixed)
        tf = tf_of(comps)
        f0, Q = f0_q(*tf)
        if not f0 or not Q or not math.isfinite(Q) or Q <= 0:
            continue
        err = abs(math.log(f0 / f0t)) + abs(math.log(Q / Qt))
        if extra is not None:
            err += extra(tf, f0)
        if best_err is None or err < best_err:
            best_err, best = err, comps
    return best or {k: snap(ideal_res[k], series) for k in names}
