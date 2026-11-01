"""Self-test: checks every formula against an independent nodal (MNA) circuit solver.

Run:  python tests/test_circuits.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import circuits as cc  # noqa: E402
import core  # noqa: E402
import filters as flt  # noqa: E402
from core import Model, freq_response, parse_fields, to_db  # noqa: E402

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------- MNA solver
def mna_gain(elems, f, out, src="in"):
    s = 2j * math.pi * f
    nodes = set()
    for e in elems:
        k = e[0]
        if k in ("R", "C", "L", "VS"):
            nodes.update(e[1:3])
        elif k == "OA":
            nodes.update(e[1:4])
        elif k == "E":
            nodes.update(e[1:5])
    nodes.discard("0")
    nodes = sorted(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    extra = [e for e in elems if e[0] in ("OA", "VS", "E")]
    N = len(nodes) + len(extra)
    A = np.zeros((N, N), complex)
    b = np.zeros(N, complex)

    def add(r, c, v):
        if r != "0" and c != "0":
            A[idx[r], idx[c]] += v

    for e in elems:
        if e[0] in ("R", "C", "L"):
            y = {"R": 1 / e[3] if e[0] == "R" else 0, "C": s * e[3] if e[0] == "C" else 0,
                 "L": 1 / (s * e[3]) if e[0] == "L" else 0}[e[0]]
            a, c = e[1], e[2]
            add(a, a, y)
            add(c, c, y)
            add(a, c, -y)
            add(c, a, -y)
    for k, e in enumerate(extra):
        row = len(nodes) + k
        if e[0] == "OA":
            _, o, p, m = e
            A[idx[o], row] -= 1
            if p != "0":
                A[row, idx[p]] += 1
            if m != "0":
                A[row, idx[m]] -= 1
        elif e[0] == "VS":
            _, p, m, val = e
            if p != "0":
                A[idx[p], row] += 1
                A[row, idx[p]] += 1
            if m != "0":
                A[idx[m], row] -= 1
                A[row, idx[m]] -= 1
            b[row] = val
        elif e[0] == "E":
            _, op, om, cp, cm, g = e
            if op != "0":
                A[idx[op], row] += 1
                A[row, idx[op]] += 1
            if om != "0":
                A[idx[om], row] -= 1
                A[row, idx[om]] -= 1
            if cp != "0":
                A[row, idx[cp]] -= g
            if cm != "0":
                A[row, idx[cm]] += g
    x = np.linalg.solve(A, b)
    return x[idx[out]]


def compare(model, elems, freqs, out="out", tol=1e-6):
    worst = 0.0
    for f in freqs:
        ref = mna_gain(elems, f, out)
        got = freq_response(model, [f])[0]
        err = abs(ref - got) / max(abs(ref), 1e-12)
        worst = max(worst, err)
    assert worst < tol, f"max rel. error {worst:.2e}"
    return worst


FR = np.logspace(1, 6, 40)


def check(name, fn):
    try:
        w = fn()
        print(f"  ok   {name}" + (f"  (worst error {w:.1e})" if isinstance(w, float) else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name}: {exc}")
        return False


# ---------------------------------------------------------------- tests
def t_values():
    assert core.parse_value("4k7") == 4700
    assert abs(core.parse_value("100nF") - 1e-7) < 1e-20
    assert abs(core.parse_value("2.2 Meg") - 2.2e6) < 1e-6
    assert core.parse_value("1%") == 1
    assert core.snap(4.9e3, "E24") == 4.7e3 or core.snap(4.9e3, "E24") == 5.1e3
    assert core.snap(1.04e3, "E96") == 1.05e3
    assert len(core._E96) == 96
    assert core.snap(9.9e3, "E12") == 1e4
    assert core.snap(4.6e-8, "E24", "up") == 4.7e-8
    assert core.snap(4.8e-8, "E24", "down") == 4.7e-8


def t_sk_lp():
    p = dict(R1=11e3, R2=7.3e3, C1=33e-9, C2=4.7e-9)
    tf = flt.sk_lp_tf(**p)
    el = [("VS", "in", "0", 1), ("R", "in", "A", p["R1"]), ("R", "A", "B", p["R2"]), ("C", "A", "out", p["C1"]),
          ("C", "B", "0", p["C2"]), ("OA", "out", "B", "out")]
    return compare(Model([tf]), el, FR)


def t_sk_hp():
    p = dict(R1=11e3, R2=7.3e3, C1=33e-9, C2=4.7e-9)
    tf = flt.sk_hp_tf(**p)
    el = [("VS", "in", "0", 1), ("C", "in", "A", p["C1"]), ("C", "A", "B", p["C2"]), ("R", "A", "out", p["R1"]),
          ("R", "B", "0", p["R2"]), ("OA", "out", "B", "out")]
    return compare(Model([tf]), el, FR)


def t_mfb_lp():
    p = dict(R1=4.7e3, R2=12e3, R3=9.1e3, Cg=33e-9, Cf=2.2e-9)
    tf = flt.mfb_lp_tf(**p)
    el = [("VS", "in", "0", 1), ("R", "in", "A", p["R1"]), ("R", "A", "out", p["R3"]), ("C", "A", "0", p["Cg"]),
          ("R", "A", "B", p["R2"]), ("C", "B", "out", p["Cf"]), ("OA", "out", "0", "B")]
    return compare(Model([tf]), el, FR)


def t_mfb_bp():
    p = dict(R1=4.7e3, R2=39e3, R3=1.2e3, C1=10e-9, C2=10e-9)
    tf = flt.mfb_bp_tf(**p)
    el = [("VS", "in", "0", 1), ("R", "in", "A", p["R1"]), ("R", "A", "0", p["R3"]), ("C", "A", "B", p["C1"]),
          ("C", "A", "out", p["C2"]), ("R", "B", "out", p["R2"]), ("OA", "out", "0", "B")]
    return compare(Model([tf]), el, FR)


def t_twint():
    R, C, k = 15.9e3, 10e-9, 0.8
    tf = flt.twin_t_tf(R, C, k)
    el = [("VS", "in", "0", 1), ("R", "in", "N1", R), ("R", "N1", "out", R), ("C", "N1", "G", 2 * C),
          ("C", "in", "N2", C), ("C", "N2", "out", C), ("R", "N2", "G", R / 2), ("E", "G", "0", "out", "0", k)]
    return compare(Model([tf]), el, FR, tol=1e-5)


def ladder_elems(kind, comps, Z0, RL):
    el = [("VS", "in", "0", 1), ("R", "in", "n0", Z0)]
    cur, cnt = "n0", 0
    for name, role, v in comps:
        cnt += 1
        if role.startswith("series"):
            nxt = f"n{cnt}"
            if kind == "Low-pass":
                el.append(("L", cur, nxt, v["L"]))
            elif kind == "High-pass":
                el.append(("C", cur, nxt, v["C"]))
            else:
                mid = f"m{cnt}"
                el += [("L", cur, mid, v["L"]), ("C", mid, nxt, v["C"])]
            cur = nxt
        else:
            if kind == "Low-pass":
                el.append(("C", cur, "0", v["C"]))
            elif kind == "High-pass":
                el.append(("L", cur, "0", v["L"]))
            else:
                el += [("L", cur, "0", v["L"]), ("C", cur, "0", v["C"])]
    el.append(("R", cur, "0", RL))
    return el, cur


def ladder_case(kind, resp, n, series_first, fc, bw, Z0=50.0):
    g, load = flt.ladder_g(resp, n)
    _, RLn = flt.ladder_proto_denominator(g, series_first, load)
    comps, RL = flt.ladder_components(kind, g, series_first, Z0, fc, bw, RLn)
    el, out = ladder_elems(kind, comps, Z0, RL)
    f = np.logspace(math.log10(fc / 8), math.log10(fc * 8), 50)
    w1 = compare(flt.ladder_model(kind, resp, n, series_first, fc, bw)[0], el, f, out=out, tol=1e-5)
    w2 = compare(flt.ladder_model_from_elements(kind, [(r, v) for _, r, v in comps], Z0, RL, TWO_PI * fc), el, f,
                 out=out, tol=1e-5)
    # response shape: Butterworth is 3.01 dB below its passband value at fc
    m = flt.ladder_model(kind, resp, n, series_first, fc, bw)[0]
    pts, pk = core.minus3db_points(m, fc / 8, fc * 8)
    return max(w1, w2), pts, pk, RL


def t_ladders():
    worst = 0
    for kind in ("Low-pass", "High-pass", "Band-pass"):
        for resp in ("Butterworth", "Chebyshev 1 dB", "Chebyshev 0.5 dB"):
            for n in (3, 4, 5, 6):
                for sf in (False, True):
                    w, pts, pk, RL = ladder_case(kind, resp, n, sf, 1e6, 3e5)
                    worst = max(worst, w)
    return worst


def t_butterworth_ladder_3db():
    _, pts, pk, RL = ladder_case("Low-pass", "Butterworth", 5, False, 1e6, 0)
    assert abs(pts[0] - 1e6) / 1e6 < 1e-3, pts
    assert abs(pk + 6.02) < 0.01
    _, pts, pk, RL = ladder_case("High-pass", "Butterworth", 5, True, 1e6, 0)
    assert abs(pts[0] - 1e6) / 1e6 < 1e-3, pts
    _, pts, pk, RL = ladder_case("Band-pass", "Butterworth", 4, False, 1e6, 2e5)
    assert len(pts) == 2 and abs((pts[1] - pts[0]) - 2e5) / 2e5 < 2e-3, pts
    assert abs(math.sqrt(pts[0] * pts[1]) - 1e6) / 1e6 < 2e-3


def t_cheb_even_load():
    # even-order Chebyshev: peak transmission 0.5*sqrt(RL/Rs), ripple equals the specified value
    g, load = flt.ladder_g("Chebyshev 1 dB", 4)
    assert load > 1.5
    for sf in (False, True):
        m, RL = flt.ladder_model("Low-pass", "Chebyshev 1 dB", 4, sf, 1e6, 0)
        gs = abs(freq_response(m, np.logspace(3, 6, 4000)))
        # ripple in passband
        band = gs[np.logspace(3, 6, 4000) <= 1e6 * 1.0001]
        assert abs(to_db(band.max()) - to_db(band.min()) - 1.0) < 0.02, (to_db(band.max()), to_db(band.min()))


def active_check(resp, n, typ, fc=1e3):
    ctx = cc.Ctx()
    out = cc.ActiveFilterDesigner().design(dict(typ=typ, resp=resp, n=n, fc=fc, topo="Sallen-Key", C=10e-9, K=1.0), ctx)
    return out.model


def t_active():
    for typ in ("Low-pass", "High-pass"):
        for n in range(1, 9):
            m = active_check("Butterworth", n, typ)
            ref = abs(freq_response(m, [1e-3 * 1e3 if typ == "Low-pass" else 1e3 * 1e3])[0])
            g = abs(freq_response(m, [1e3])[0]) / ref
            assert abs(to_db(g) + 3.0103) < 0.02, (typ, n, to_db(g))
            m = active_check("Bessel", n, typ)
            ref = abs(freq_response(m, [1e-3 * 1e3 if typ == "Low-pass" else 1e6])[0])
            g = abs(freq_response(m, [1e3])[0]) / ref
            assert abs(to_db(g) + 3.0103) < 0.1, ("bessel", typ, n, to_db(g))
    # Chebyshev: ripple band edge at fc, ripple = rp (odd order: 0 .. -rp relative to DC)
    for n in (3, 4, 5):
        m = active_check("Chebyshev 1 dB", n, "Low-pass")
        f = np.logspace(1, math.log10(1e3), 2000)
        mag = to_db(1) + 20 * np.log10(np.abs(freq_response(m, f)))
        assert abs((mag.max() - mag.min()) - 1.0) < 0.02, (n, mag.max() - mag.min())
        g_fc = to_db(abs(freq_response(m, [1e3])[0]))
        assert abs(g_fc - mag.min()) < 0.02


def t_rounding_matches_mna():
    # snapped component values must still be analysed exactly: compare SK LP design incl. E24 rounding with MNA
    ctx = cc.Ctx(series="E24")
    c = cc.SallenKeyLP()
    p = parse_fields(c.des_fields, dict(f0="2.5k", Q="1.2", C2="4.7n"))
    ideal = c.ideal(p, ctx)
    std = {k: ctx.snap(v) for k, v in ideal.items()}
    out = c.forward(std, ctx)
    el = [("VS", "in", "0", 1), ("R", "in", "A", std["R1"]), ("R", "A", "B", std["R2"]), ("C", "A", "out", std["C1"]),
          ("C", "B", "0", std["C2"]), ("OA", "out", "B", "out")]
    return compare(out.model, el, FR)


def t_refine_and_cap_rounding():
    ctx = cc.Ctx(series="E24", cseries="E12")
    for cls, vals in ((cc.SallenKeyLP(), dict(f0="1k", Q="0.9", C2="10n")), (cc.SallenKeyHP(), dict(f0="3k", Q="2", C="22n")),
                      (cc.MFBLowPass(), dict(f0="1k", Q="0.9", K="2", Cf="4.7n")),
                      (cc.MFBBandPass(), dict(f0="2k", Q="5", H0="3", C="10n"))):
        p = parse_fields(cls.des_fields, vals)
        ideal = cls.ideal(p, ctx)
        std = cls.standardize(ideal, p, ctx)
        naive = {k: (ctx.snapc if k.startswith("C") else ctx.snap)(v) for k, v in ideal.items()}

        def err(d):
            f0, Q = flt.f0_q(*cls.forward(d, ctx).model.stages[0])
            return abs(math.log(f0 / p["f0"])) + abs(math.log(Q / p["Q"]))
        assert err(std) <= err(naive) + 1e-12, (cls.name, err(std), err(naive))
        for k, v in std.items():
            if k.startswith("C"):
                assert core.snap(v, "E12") == v, (cls.name, k, v)
    o = cc.LCLadder().design(dict(kind="Low-pass", resp="Butterworth", n=5, fc=1e6, bw=0.0, Z0=50.0,
                                  first="Shunt element first (π)"), ctx)
    assert o.model is not None


def t_solve_for():
    exact = cc.Ctx()
    rc = cc.RCFilter()
    for tx, chk in ((dict(solve=rc.SOLVE[0], fc="50k", C="10n"), "R"), (dict(solve=rc.SOLVE[1], fc="50k", R="1k"), "C"),
                    (dict(solve=rc.SOLVE[2], R="318.3", C="10n"), "fc")):
        p = parse_fields(rc.des_fields, tx)
        assert (chk == "fc") == ("fc" not in p), p
        o = rc.design(p, exact)
        fc = 1 / (TWO_PI * o.std["R"] * o.std["C"])
        if chk != "fc":
            assert abs(fc - 50e3) < 1e-6 * 50e3, (chk, fc)
    # known part is never rounded, solved part is
    ctx = cc.Ctx(series="E24", cseries="E12")
    p = parse_fields(rc.des_fields, dict(solve=rc.SOLVE[0], fc="50k", C="12.3n"))
    o = rc.design(p, ctx)
    assert abs(o.std["C"] - 12.3e-9) < 1e-18 and core.snap(o.std["R"], "E24") == o.std["R"]
    rl = cc.RLFilter()
    for i, tx in enumerate((dict(solve=rl.SOLVE[0], fc="1k", L="100m"), dict(solve=rl.SOLVE[1], fc="1k", R="1k"))):
        o = rl.design(parse_fields(rl.des_fields, tx), exact)
        assert abs(o.std["R"] / (TWO_PI * o.std["L"]) - 1e3) < 1e-6
    lc = cc.LCResonance()
    for tx in (dict(solve=lc.SOLVE[0], f0="1k", C="2.2u"), dict(solve=lc.SOLVE[1], f0="1k", L="10m")):
        o = lc.design(parse_fields(lc.des_fields, tx), exact)
        assert abs(1 / (TWO_PI * math.sqrt(o.std["L"] * o.std["C"])) - 1e3) < 1e-6


def t_diagrams():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import schem
    n = 0
    for tab, lst in cc.TABS.items():
        for circ in lst:
            for series in (None, "E24"):
                ctx = cc.Ctx(series=series, cseries="E12" if series else None)
                for mode in ("forward", "design"):
                    if (mode == "forward" and not circ.has_forward) or (mode == "design" and not circ.has_design):
                        continue
                    fl = circ.fwd_fields if mode == "forward" else circ.des_fields
                    p = parse_fields(fl, {})
                    o = circ.forward(p, ctx) if mode == "forward" else circ.design(p, ctx)
                    vals = p if mode == "forward" else (o.std or p)
                    for detail in (False, True):
                        fig = Figure(figsize=(5, 3))
                        ax = fig.add_axes([0, 0, 1, 1])
                        schem.draw(circ.key, ax, vals, o, fs=8, detail=detail)
                        n += 1
    # values really appear on the drawing
    fig = Figure(figsize=(5, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    schem.draw("inv", ax, dict(Rin=4700.0, Rf=47000.0), None)
    txt = [t.get_text() for t in ax.texts]
    assert "4.7 kΩ" in txt and "47 kΩ" in txt and "Rin" in txt and "Rf" in txt, txt
    assert n > 60


def t_design_hits_targets():
    ctx = cc.Ctx()
    for cls, key, vals in ((cc.SallenKeyLP(), None, dict(f0="1k", Q="0.7071", C2="10n")),
                           (cc.SallenKeyHP(), None, dict(f0="3k", Q="2", C="22n")),
                           (cc.MFBLowPass(), None, dict(f0="1k", Q="0.9", K="2", Cf="4.7n")),
                           (cc.MFBBandPass(), None, dict(f0="2k", Q="5", H0="3", C="10n"))):
        p = parse_fields(cls.des_fields, vals)
        o = cls.design(p, ctx)
        tf = o.model.stages[0]
        f0, Q = flt.f0_q(*tf)
        assert abs(f0 - p["f0"]) / p["f0"] < 1e-9, (cls.name, f0)
        assert abs(Q - p["Q"]) / p["Q"] < 1e-9, (cls.name, Q)
    # MFB gain
    p = parse_fields(cc.MFBLowPass().des_fields, dict(f0="1k", Q="0.9", K="2", Cf="4.7n"))
    o = cc.MFBLowPass().design(p, ctx)
    assert abs(abs(freq_response(o.model, [1.0])[0]) - 2.0) < 1e-6
    p = parse_fields(cc.MFBBandPass().des_fields, dict(f0="2k", Q="5", H0="3", C="10n"))
    o = cc.MFBBandPass().design(p, ctx)
    assert abs(abs(freq_response(o.model, [2e3])[0]) - 3.0) < 1e-6
    # notch
    p = parse_fields(cc.TwinTNotch().des_fields, dict(f0="1k", Q="2.5", C="10n"))
    o = cc.TwinTNotch().design(p, ctx)
    assert abs(freq_response(o.model, [1e3])[0]) < 1e-9


def t_amps():
    ctx = cc.Ctx(gbw=1e6)
    p = parse_fields(cc.InvertingAmp().des_fields, dict(gain="-10", Rin="10k"))
    o = cc.InvertingAmp().design(p, cc.Ctx())
    assert abs(freq_response(o.model, [1.0])[0] + 10) < 1e-9
    o = cc.InvertingAmp().forward(dict(Rin=1e4, Rf=1e5), ctx)
    f3 = 1e6 / 11
    assert abs(abs(freq_response(o.model, [f3])[0]) - 10 / math.sqrt(2)) < 1e-6
    o = cc.NonInvertingAmp().forward(dict(Rf=9e4, Rg=1e4), ctx)
    assert abs(abs(freq_response(o.model, [1e5])[0]) - 10 / math.sqrt(2)) < 1e-6
    # integrator unity-gain frequency, with finite GBW still close at f << GBW
    o = cc.Integrator().forward(dict(R=1e4, C=1e-8, Rp=0.0), ctx)
    fu = 1 / (TWO_PI * 1e-4 * 1.0)
    assert abs(abs(freq_response(o.model, [fu])[0]) - 1) < 5e-3
    o = cc.Integrator().forward(dict(R=1e4, C=1e-8, Rp=0.0), cc.Ctx())
    assert abs(abs(freq_response(o.model, [fu])[0]) - 1) < 1e-9
    o = cc.Differentiator().forward(dict(C=1e-8, R=1e4, Rs=1e3, Cf=0.0), cc.Ctx())
    assert abs(abs(freq_response(o.model, [fu / 10])[0]) - 0.1) < 2e-3
    # summing / difference / instrumentation
    o = cc.SummingAmp().forward(dict(R1=1e4, R2=2e4, Rf=1e4, v2=0.5), cc.Ctx())
    assert abs(o.model.dc_out + 0.25) < 1e-12 and abs(freq_response(o.model, [10.0])[0] + 1) < 1e-9
    o = cc.InstrumentationAmp().forward(dict(R1=25e3, Rg=1.04e3, R2=1e4, R3=1e4), cc.Ctx())
    assert abs(abs(freq_response(o.model, [10.0])[0]) - (1 + 50e3 / 1.04e3)) < 1e-9


def t_rlc():
    for out_i, out in enumerate(cc.SeriesRLC.OUTS):
        R_, L, C = 12.0, 8e-3, 3.3e-6
        o = cc.SeriesRLC().forward(dict(out=out, R=R_, L=L, C=C), cc.Ctx())
        el = [("VS", "in", "0", 1), ("L", "in", "a", L), ("C", "a", "b", C), ("R", "b", "0", R_)]
        if out_i == 0:
            node = "b"
        elif out_i == 1:
            el = [("VS", "in", "0", 1), ("R", "in", "a", R_), ("L", "a", "b", L), ("C", "b", "0", C)]
            node = "b"
        elif out_i == 2:
            el = [("VS", "in", "0", 1), ("R", "in", "a", R_), ("C", "a", "b", C), ("L", "b", "0", L)]
            node = "b"
        else:
            el = [("VS", "in", "0", 1), ("R", "in", "a", R_), ("L", "a", "b", L), ("C", "b", "0", C)]
            # notch output = across L+C == node a
            node = "a"
        compare(o.model, el, FR, out=node, tol=1e-6)


def t_simulation():
    # steady-state amplitude/phase of every model type equals |H(jw)|
    ctx = cc.Ctx()
    cases = []
    out = cc.ActiveFilterDesigner().design(dict(typ="Low-pass", resp="Chebyshev 1 dB", n=5, fc=1e3, topo="Sallen-Key",
                                                C=10e-9, K=1.0), ctx)
    cases.append((out.model, 1.3e3))
    out = cc.LCLadder().design(dict(kind="Band-pass", resp="Butterworth", n=3, fc=1e6, bw=3e5, Z0=50.0,
                                    first="Shunt element first (π)"), ctx)
    cases.append((out.model, 1.1e6))
    out = cc.LCLadder().design(dict(kind="High-pass", resp="Chebyshev 0.5 dB", n=5, fc=1e6, bw=0.0, Z0=50.0,
                                    first="Series element first (T)"), cc.Ctx(series="E24"))
    cases.append((out.model, 1.2e6))
    out = cc.Differentiator().forward(dict(C=1e-8, R=1e4, Rs=1e3, Cf=0.0), cc.Ctx(gbw=1e6))
    cases.append((out.model, 2e3))
    for m, f in cases:
        T = 40 / f
        t = np.linspace(0, T, 40 * 200 + 1)
        u = core.make_input("Sine", t, 1.0, 0.0, f)
        y = core.simulate(m, t, u, f)
        last = t >= T - 3 / f
        amp = (y[last].max() - y[last].min()) / 2
        ref = abs(freq_response(m, [f])[0])
        assert abs(amp - ref) / ref < 0.01, (amp, ref)


def t_all_defaults():
    n = 0
    for tab, lst in cc.TABS.items():
        for circ in lst:
            for series in (None, "E12", "E96"):
                ctx = cc.Ctx(series=series, gbw=3e6)
                if circ.has_forward:
                    o = circ.forward(parse_fields(circ.fwd_fields, {}), ctx)
                    n += 1
                    if o.model:
                        freq_response(o.model, [1e3])
                if circ.has_design:
                    o = circ.design(parse_fields(circ.des_fields, {}), ctx)
                    n += 1
                    if o.model:
                        freq_response(o.model, [1e3])
    assert n > 60


if __name__ == "__main__":
    tests = [t_values, t_diagrams, t_solve_for, t_refine_and_cap_rounding, t_sk_lp, t_sk_hp, t_mfb_lp, t_mfb_bp, t_twint, t_ladders, t_butterworth_ladder_3db,
             t_cheb_even_load, t_active, t_rounding_matches_mna, t_design_hits_targets, t_amps, t_rlc,
             t_simulation, t_all_defaults]
    res = [check(t.__name__, t) for t in tests]
    print(f"\n{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)
