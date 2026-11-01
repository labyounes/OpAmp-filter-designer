"""Circuit definitions: every calculator in the app is one Circuit subclass.

A circuit offers up to two modes:
  forward(p)  components -> characteristics (+ model for the simulation)
  design(p)   target values -> components, rounded to an E-series, then analysed again
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core import (EXACT, TWO_PI, H, KV, NOTE, WARN, Field, Model, Outcome, P, fmt,
                  freq_response, minus3db_points, snap, to_db, with_gbw)
from filters import (RESPONSES, ladder_components, ladder_g, ladder_model,
                     ladder_model_from_elements, ladder_proto_denominator, f0_q,
                     mfb_bp_design, mfb_bp_tf, mfb_lp_design, mfb_lp_tf, pole_pairs,
                     proto_poles, refine_stage, sk_hp_design, sk_hp_tf, sk_lp_design, sk_lp_tf,
                     twin_t_tf)


@dataclass
class Ctx:
    series: str | None = None    # E-series for resistors (None = exact)
    gbw: float = 0.0             # op-amp gain-bandwidth product in Hz (0 = ideal)
    cseries: str | None = None   # E-series for capacitors and coils (None = exact)

    @staticmethod
    def _snap(v, series, mode):
        if not series or v in (0, math.inf) or not math.isfinite(v):
            return v
        return snap(v, series, mode)

    def snap(self, v, mode="nearest"):
        """Round a resistor value."""
        return self._snap(v, self.series, mode)

    def snapc(self, v, mode="nearest"):
        """Round a capacitor or coil value."""
        return self._snap(v, self.cseries, mode)

    @property
    def rounding(self):
        return bool(self.series or self.cseries)

    @property
    def label(self):
        names = [f"R {self.series}" if self.series else "", f"C/L {self.cseries}" if self.cseries else ""]
        return ", ".join(n for n in names if n)


def R(key, label, default, **kw):
    return Field(key, label, default, "Ω", **kw)


def Cap(key, label, default, **kw):
    return Field(key, label, default, "F", **kw)


def Ind(key, label, default, **kw):
    return Field(key, label, default, "H", **kw)


def Hz(key, label, default, **kw):
    return Field(key, label, default, "Hz", **kw)


def Num(key, label, default, unit="", **kw):
    return Field(key, label, default, unit, **kw)


def Choice(key, label, choices, default=None):
    return Field(key, label, default or choices[0], "", kind="choice", choices=list(choices))


def when(field, key, prefixes):
    """Show `field` only while choice `key` starts with one of `prefixes`."""
    field.show_if = (key, prefixes)
    return field


def gain_text(g):
    return f"{g:.5g} V/V   ({to_db(g):.2f} dB)"


COMP_UNITS = ("Ω", "F", "H")


class Circuit:
    key = ""
    name = ""
    blurb = ""
    fwd_fields: list = []
    des_fields: list = []
    uses_gbw = False
    has_model = True
    default_clip = True

    # -- subclass API ------------------------------------------------------
    def forward(self, p, ctx) -> Outcome:
        raise NotImplementedError

    def ideal(self, p, ctx) -> dict:
        raise NotImplementedError

    # -- shared design flow ---------------------------------------------------
    @property
    def has_forward(self):
        return bool(self.fwd_fields)

    @property
    def has_design(self):
        return bool(self.des_fields)

    def known_keys(self, p):
        """Parts the user typed in; they are never rounded."""
        return ()

    def standardize(self, ideal, p, ctx):
        """Round every component to the chosen E-series (subclasses may search for a better combination)."""
        std = dict(ideal)
        known = self.known_keys(p)
        for f in self.fwd_fields:
            if f.key in ideal and f.key not in known and f.unit in COMP_UNITS:
                std[f.key] = (ctx.snapc if f.unit in ("F", "H") else ctx.snap)(ideal[f.key])
        return std

    def comp_rows(self, fields, ideal, std, ctx):
        rows = []
        for f in fields:
            if f.key not in ideal or f.unit not in COMP_UNITS:
                continue
            v = ideal[f.key]
            if v == 0 and f.zero_ok:
                rows.append(KV(f.label, "not fitted"))
            elif v == math.inf:
                rows.append(KV(f.label, "not fitted (open)"))
            else:
                sv = std[f.key]
                if abs(sv - v) > 1e-9 * abs(v):
                    rows.append(KV(f.label, f"{fmt(sv, f.unit)}  (ideal {fmt(v, f.unit)}, {100 * (sv - v) / v:+.1f} %)"))
                else:
                    rows.append(KV(f.label, fmt(sv, f.unit)))
        return rows

    def design(self, p, ctx) -> Outcome:
        ideal = self.ideal(p, ctx)
        std = self.standardize(ideal, p, ctx)
        rows = self.comp_rows(self.fwd_fields, ideal, std, ctx)
        out = self.forward(std, ctx)
        head = [H("Component values" + (f"  (rounded: {ctx.label})" if ctx.rounding else ""))] + rows
        out.rows = head + [H("Result with these values")] + out.rows
        out.std = {k: v for k, v in std.items() if any(f.key == k for f in self.fwd_fields)}
        return out


# ===========================================================================
# Op-amp circuits
# ===========================================================================
class InvertingAmp(Circuit):
    key = "inv"
    name = "Inverting amplifier"
    blurb = "Vout = -(Rf / Rin) · Vin.  Input impedance equals Rin."
    uses_gbw = True
    fwd_fields = [R("Rin", "Rin  (input resistor)", "10k"), R("Rf", "Rf  (feedback resistor)", "100k")]
    des_fields = [Num("gain", "Gain (V/V, sign ignored)", "-10", signed=True), R("Rin", "Rin  (your choice)", "10k")]

    def ideal(self, p, ctx):
        g = abs(p["gain"])
        if g == 0:
            raise ValueError("Gain must not be 0")
        return {"Rin": p["Rin"], "Rf": g * p["Rin"]}

    def forward(self, p, ctx):
        Rin, Rf = p["Rin"], p["Rf"]
        G = Rf / Rin
        num, den = with_gbw([-G], [1.0], ([1 + G], [1.0]), ctx.gbw)
        rows = [H("Characteristics"), KV("Gain", gain_text(-G) + "   phase 180°"),
                KV("Input impedance", fmt(Rin, "Ω")), KV("Noise gain", f"{1 + G:.5g}"),
                KV("Bias-balance resistor", fmt(Rf * Rin / (Rf + Rin), "Ω") + "  (Rf ∥ Rin, in series with the + input)")]
        if ctx.gbw:
            rows.append(KV("-3 dB bandwidth", fmt(ctx.gbw / (1 + G), "Hz")))
        else:
            rows.append(NOTE("Ideal op-amp. Choose a GBW in the options to see the bandwidth limit."))
        net = ("Vin ── Rin ── (−) input,  Rf from output to (−),  (+) input to GND\n"
               "Vout = op-amp output")
        return Outcome(rows, Model([(num, den)]), net, 1e3)


class NonInvertingAmp(Circuit):
    key = "noninv"
    name = "Non-inverting amplifier"
    blurb = "Vout = (1 + Rf / Rg) · Vin.  Very high input impedance.  Rg = inf gives a voltage follower."
    uses_gbw = True
    fwd_fields = [R("Rf", "Rf  (feedback resistor)", "90k", zero_ok=True), R("Rg", "Rg  (resistor to GND)", "10k")]
    des_fields = [Num("gain", "Gain (V/V, ≥ 1)", "10"), R("Rg", "Rg  (your choice)", "10k")]

    def ideal(self, p, ctx):
        g = p["gain"]
        if g < 1:
            raise ValueError("A non-inverting stage cannot have a gain below 1")
        return {"Rf": (g - 1) * p["Rg"], "Rg": p["Rg"]}

    def forward(self, p, ctx):
        Rf, Rg = p["Rf"], p["Rg"]
        G = 1 + Rf / Rg if math.isfinite(Rg) else 1.0
        num, den = with_gbw([G], [1.0], ([G], [1.0]), ctx.gbw)
        rows = [H("Characteristics"), KV("Gain", gain_text(G) + "   phase 0°"),
                KV("Input impedance", "very high (op-amp input)"),
                KV("Rf || Rg", fmt(Rf * Rg / (Rf + Rg), "Ω") if Rf and math.isfinite(Rg) else "—")]
        if ctx.gbw:
            rows.append(KV("-3 dB bandwidth", fmt(ctx.gbw / G, "Hz")))
        else:
            rows.append(NOTE("Ideal op-amp. Choose a GBW in the options to see the bandwidth limit."))
        net = ("Vin ── (+) input\n"
               "Rf from output to (−),  Rg from (−) to GND\n"
               "Vout = op-amp output")
        return Outcome(rows, Model([(num, den)]), net, 1e3)


class SummingAmp(Circuit):
    key = "sum"
    name = "Summing amplifier (2 inputs)"
    blurb = "Vout = -Rf · (V1/R1 + V2/R2).  V1 is the simulated input, V2 is a fixed DC voltage."
    uses_gbw = True
    fwd_fields = [R("R1", "R1  (input 1)", "10k"), R("R2", "R2  (input 2)", "20k"),
                  R("Rf", "Rf  (feedback)", "10k"), Num("v2", "V2 (DC on input 2)", "0.5", "V", signed=True)]
    des_fields = [Num("g1", "Gain for input 1 (V/V)", "1", signed=True), Num("g2", "Gain for input 2 (V/V)", "0.5", signed=True),
                  R("Rf", "Rf  (your choice)", "10k"), Num("v2", "V2 (DC on input 2)", "0.5", "V", signed=True)]

    def ideal(self, p, ctx):
        if p["g1"] == 0 or p["g2"] == 0:
            raise ValueError("Gains must not be 0")
        return {"R1": p["Rf"] / abs(p["g1"]), "R2": p["Rf"] / abs(p["g2"]), "Rf": p["Rf"], "v2": p["v2"]}

    def forward(self, p, ctx):
        R1, R2, Rf, v2 = p["R1"], p["R2"], p["Rf"], p["v2"]
        G1, G2 = -Rf / R1, -Rf / R2
        ng = 1 + Rf * (R1 + R2) / (R1 * R2)
        num, den = with_gbw([G1], [1.0], ([ng], [1.0]), ctx.gbw)
        rows = [H("Characteristics"), KV("Gain input 1", gain_text(G1)), KV("Gain input 2", gain_text(G2)),
                KV("Output from V2 alone", fmt(G2 * v2, "V")), KV("Input impedances", f"{fmt(R1, 'Ω')} / {fmt(R2, 'Ω')}"),
                KV("Noise gain", f"{ng:.5g}")]
        if ctx.gbw:
            rows.append(KV("-3 dB bandwidth", fmt(ctx.gbw / ng, "Hz")))
        net = ("V1 ── R1 ──┐\n"
               "V2 ── R2 ──┴── (−) input,  Rf from output to (−),  (+) to GND")
        return Outcome(rows, Model([(num, den)], dc_out=G2 * v2), net, 1e3)


class DifferenceAmp(Circuit):
    key = "diff"
    name = "Difference (subtractor) amplifier"
    blurb = "Vout = (R2 / R1) · (V+ − V−) with matched resistor pairs.  Simulation drives the differential input."
    uses_gbw = True
    fwd_fields = [R("R1", "R1  (both input resistors)", "10k"), R("R2", "R2  (feedback = resistor to GND)", "10k"),
                  Num("tol", "Resistor tolerance", "1", "%")]
    des_fields = [Num("gain", "Gain (V/V)", "1"), R("R1", "R1  (your choice)", "10k"), Num("tol", "Resistor tolerance", "1", "%")]

    def ideal(self, p, ctx):
        return {"R1": p["R1"], "R2": p["gain"] * p["R1"], "tol": p["tol"]}

    def forward(self, p, ctx):
        R1, R2, tol = p["R1"], p["R2"], p["tol"]
        G = R2 / R1
        num, den = with_gbw([G], [1.0], ([1 + G], [1.0]), ctx.gbw)
        cmrr = ((1 + G) / (4 * tol / 100.0)) if tol > 0 else math.inf
        rows = [H("Characteristics"), KV("Differential gain", gain_text(G)),
                KV("Input impedance (V−)", fmt(R1, "Ω")), KV("Input impedance (V+)", fmt(R1 + R2, "Ω")),
                KV("Worst-case CMRR from resistors", f"{to_db(cmrr):.1f} dB" if math.isfinite(cmrr) else "∞")]
        if tol > 0:
            rows.append(NOTE("CMRR ≈ (1+G) / (4·tolerance). Use 0.1 % resistors or trim one resistor for better rejection."))
        if ctx.gbw:
            rows.append(KV("-3 dB bandwidth", fmt(ctx.gbw / (1 + G), "Hz")))
        net = ("V− ── R1 ── (−) input,  R2 from output to (−)\n"
               "V+ ── R1 ── (+) input,  R2 from (+) to GND")
        return Outcome(rows, Model([(num, den)]), net, 1e3)


class Integrator(Circuit):
    key = "int"
    name = "Integrator"
    blurb = "Vout = -(1/RC) ∫Vin dt.  A resistor Rp across C keeps the DC operating point stable."
    uses_gbw = True
    fwd_fields = [R("R", "R  (input resistor)", "10k"), Cap("C", "C  (feedback capacitor)", "10n"),
                  R("Rp", "Rp  (parallel to C)", "100k", zero_ok=True)]
    des_fields = [Hz("fu", "Unity-gain frequency", "1.59k"), Cap("C", "C  (your choice)", "10n"),
                  Num("ratio", "Rp / R  (0 = no Rp)", "10", zero_ok=True)]

    def ideal(self, p, ctx):
        Rv = 1.0 / (TWO_PI * p["fu"] * p["C"])
        return {"R": Rv, "C": p["C"], "Rp": p["ratio"] * Rv}

    def forward(self, p, ctx):
        Rv, C, Rp = p["R"], p["C"], p["Rp"]
        fu = 1.0 / (TWO_PI * Rv * C)
        if Rp > 0:
            num0, den0 = [-Rp / Rv], [Rp * C, 1.0]
            ng = ([Rv * Rp * C, Rv + Rp], [Rv * Rp * C, Rv])
        else:
            num0, den0 = [-1.0], [Rv * C, 0.0]
            ng = ([Rv * C, 1.0], [Rv * C, 0.0])
        num, den = with_gbw(num0, den0, ng, ctx.gbw)
        rows = [H("Characteristics"), KV("Time constant R·C", fmt(Rv * C, "s")),
                KV("Unity-gain frequency", fmt(fu, "Hz")),
                KV("Output slope for 1 V DC in", fmt(-1.0 / (Rv * C), "V/s"))]
        if Rp > 0:
            rows += [KV("DC gain (limited by Rp)", gain_text(-Rp / Rv)),
                     KV("Integrates above", fmt(1.0 / (TWO_PI * Rp * C), "Hz"))]
        else:
            rows.append(WARN("No Rp: any DC offset makes the output drift into the supply rails."))
        net = ("Vin ── R ── (−) input\n"
               "C (and Rp in parallel) from output to (−),  (+) to GND")
        return Outcome(rows, Model([(num, den)]), net, fu)


class Differentiator(Circuit):
    key = "dif"
    name = "Differentiator"
    blurb = "Vout = -R·C · dVin/dt.  Rs (series with C) and Cf (parallel to R) limit high-frequency gain and noise."
    uses_gbw = True
    fwd_fields = [Cap("C", "C  (input capacitor)", "10n"), R("R", "R  (feedback resistor)", "10k"),
                  R("Rs", "Rs  (series with C)", "1k", zero_ok=True), Cap("Cf", "Cf  (parallel to R)", "0", zero_ok=True)]
    des_fields = [Hz("fu", "Unity-gain frequency", "1.59k"), Cap("C", "C  (your choice)", "10n"),
                  Hz("flim", "High-frequency limit (0 = none)", "16k", zero_ok=True)]

    def ideal(self, p, ctx):
        Rv = 1.0 / (TWO_PI * p["fu"] * p["C"])
        Rs = 1.0 / (TWO_PI * p["flim"] * p["C"]) if p["flim"] > 0 else 0.0
        return {"C": p["C"], "R": Rv, "Rs": Rs, "Cf": 0.0}

    def forward(self, p, ctx):
        C, Rv, Rs, Cf = p["C"], p["R"], p["Rs"], p["Cf"]
        a = P(Rs * C, 1.0)          # 1 + s Rs C
        b = P(Rv * Cf, 1.0)         # 1 + s R Cf
        den0 = np.polymul(a, b)
        num0 = P(-Rv * C, 0.0)
        ng_n = np.polyadd(den0, P(Rv * C, 0.0))
        num, den = with_gbw(num0, den0, (ng_n, den0), ctx.gbw)
        fu = 1.0 / (TWO_PI * Rv * C)
        rows = [H("Characteristics"), KV("Unity-gain frequency", fmt(fu, "Hz")),
                KV("Gain at 10× lower frequency", f"{to_db(0.1):.0f} dB  (+20 dB/decade slope)")]
        if Rs > 0:
            rows += [KV("High-frequency corner (Rs·C)", fmt(1.0 / (TWO_PI * Rs * C), "Hz")),
                     KV("Maximum gain", gain_text(-Rv / Rs))]
        else:
            rows.append(WARN("No Rs: gain keeps rising, noise and oscillation risk. Add Rs and/or Cf."))
        if Cf > 0:
            rows.append(KV("Roll-off corner (R·Cf)", fmt(1.0 / (TWO_PI * Rv * Cf), "Hz")))
        net = ("Vin ── C ── Rs ── (−) input\n"
               "R (and Cf in parallel) from output to (−),  (+) to GND")
        return Outcome(rows, Model([(num, den)]), net, fu)


class InstrumentationAmp(Circuit):
    key = "ina"
    name = "Instrumentation amplifier (3 op-amps)"
    blurb = "G = (1 + 2·R1/Rg) · (R3/R2).  Gain is set with the single resistor Rg.  Ideal op-amps only."
    fwd_fields = [R("R1", "R1  (two equal feedback resistors)", "25k"), R("Rg", "Rg  (gain resistor)", "1.04k"),
                  R("R2", "R2  (difference stage input)", "10k"), R("R3", "R3  (difference stage feedback)", "10k")]
    des_fields = [Num("gain", "Total gain (V/V)", "50"), R("R1", "R1  (your choice)", "25k"),
                  R("R2", "R2 = R3  (your choice)", "10k")]

    def ideal(self, p, ctx):
        g = p["gain"]
        if g <= 1:
            raise ValueError("Gain must be above 1")
        return {"R1": p["R1"], "Rg": 2 * p["R1"] / (g - 1), "R2": p["R2"], "R3": p["R2"]}

    def forward(self, p, ctx):
        R1, Rg, R2, R3 = p["R1"], p["Rg"], p["R2"], p["R3"]
        g1 = 1 + 2 * R1 / Rg
        G = g1 * R3 / R2
        rows = [H("Characteristics"), KV("Input stage gain", f"{g1:.5g}"), KV("Difference stage gain", f"{R3 / R2:.5g}"),
                KV("Total gain", gain_text(G)), NOTE("Common-mode gain of the first stage is 1, so CMRR is set mostly by R2/R3 matching.")]
        net = ("A1: (+) = V+, A2: (+) = V−.  R1 from each output to its (−) input, Rg between the two (−) inputs.\n"
               "A3: A1 out ── R2 ── (−);  A2 out ── R2 ── (+);  R3 from out to (−) and from (+) to GND")
        return Outcome(rows, Model([(P(G), P(1.0))]), net, 1e3)


# ===========================================================================
# Active filters (single stages)
# ===========================================================================
NET_SK_LP = ("Sallen-Key low-pass:  Vin ── R1 ── A ── R2 ── B ── (+) input\n"
             "   C1 from A to Vout (feedback capacitor),  C2 from B to GND\n"
             "   (−) input connected directly to Vout (unity gain)")
NET_SK_HP = ("Sallen-Key high-pass:  Vin ── C1 ── A ── C2 ── B ── (+) input\n"
             "   R1 from A to Vout,  R2 from B to GND\n"
             "   (−) input connected directly to Vout (unity gain)")
NET_MFB_LP = ("MFB low-pass:  Vin ── R1 ── A ── R2 ── B ── (−) input,  (+) to GND\n"
              "   R3 from A to Vout,  Cg from A to GND,  Cf from B to Vout\n"
              "   Gain = -R3/R1")
NET_MFB_BP = ("MFB band-pass:  Vin ── R1 ── A ── C1 ── B ── (−) input,  (+) to GND\n"
              "   R3 from A to GND,  C2 from A to Vout,  R2 from B to Vout\n"
              "   Gain = -R2/(2·R1) at the centre frequency")


def _q_hint(Q):
    for q, name in ((1 / math.sqrt(2), "Butterworth"), (1 / math.sqrt(3), "Bessel"), (0.5, "critically damped")):
        if abs(Q - q) / q < 0.03:
            return f"  ≈ {name}"
    return ""


def _stage_rows(tf, label="Pole frequency f0"):
    f0, Q = f0_q(*tf)
    return [KV(label, fmt(f0, "Hz")), KV("Quality factor Q", f"{Q:.4g}" + _q_hint(Q))], f0, Q


class SallenKeyLP(Circuit):
    key = "sklp"
    name = "Sallen-Key low-pass (2nd order)"
    blurb = "Unity-gain 2nd-order low-pass. Capacitor C1 (feedback) must be at least 4Q² · C2."
    fwd_fields = [R("R1", "R1", "10k"), R("R2", "R2", "10k"), Cap("C1", "C1  (feedback capacitor)", "22n"), Cap("C2", "C2  (to GND)", "10n")]
    des_fields = [Hz("f0", "Cut-off / pole frequency f0", "1k"), Num("Q", "Quality factor Q", "0.7071"), Cap("C2", "C2  (your choice)", "10n")]

    def ideal(self, p, ctx):
        if p["Q"] < 0.5:
            raise ValueError("Q must be at least 0.5 for this circuit")
        return sk_lp_design(p["f0"], p["Q"], p["C2"], ctx.snapc)

    def standardize(self, ideal, p, ctx):
        std = Circuit.standardize(self, ideal, p, ctx)
        if ctx.series:
            std.update(refine_stage(lambda c: sk_lp_tf(c["R1"], c["R2"], c["C1"], c["C2"]),
                                    {k: ideal[k] for k in ("R1", "R2")}, {k: std[k] for k in ("C1", "C2")},
                                    ctx.series, p["f0"], p["Q"]))
        return std

    def forward(self, p, ctx):
        tf = sk_lp_tf(p["R1"], p["R2"], p["C1"], p["C2"])
        r, f0, Q = _stage_rows(tf)
        rows = [H("Characteristics")] + r + [KV("DC gain", "1  (0 dB)"), KV("Roll-off", "-40 dB/decade")]
        return Outcome(rows, Model([tf]), NET_SK_LP, f0)


class SallenKeyHP(Circuit):
    key = "skhp"
    name = "Sallen-Key high-pass (2nd order)"
    blurb = "Unity-gain 2nd-order high-pass with equal capacitors."
    fwd_fields = [R("R1", "R1  (A to Vout)", "11.25k"), R("R2", "R2  (B to GND)", "22.5k"), Cap("C1", "C1", "10n"), Cap("C2", "C2", "10n")]
    des_fields = [Hz("f0", "Cut-off / pole frequency f0", "1k"), Num("Q", "Quality factor Q", "0.7071"), Cap("C", "C1 = C2  (your choice)", "10n")]

    def ideal(self, p, ctx):
        return sk_hp_design(p["f0"], p["Q"], p["C"], ctx.snapc)

    def standardize(self, ideal, p, ctx):
        std = Circuit.standardize(self, ideal, p, ctx)
        if ctx.series:
            std.update(refine_stage(lambda c: sk_hp_tf(c["R1"], c["R2"], c["C1"], c["C2"]),
                                    {k: ideal[k] for k in ("R1", "R2")}, {k: std[k] for k in ("C1", "C2")},
                                    ctx.series, p["f0"], p["Q"]))
        return std

    def forward(self, p, ctx):
        tf = sk_hp_tf(p["R1"], p["R2"], p["C1"], p["C2"])
        r, f0, Q = _stage_rows(tf)
        rows = [H("Characteristics")] + r + [KV("Passband gain", "1  (0 dB)"), KV("Roll-off", "+40 dB/decade")]
        return Outcome(rows, Model([tf]), NET_SK_HP, f0)


class MFBLowPass(Circuit):
    key = "mfblp"
    name = "Multiple-feedback low-pass (2nd order)"
    blurb = "Inverting 2nd-order low-pass with adjustable gain -R3/R1.  Needs Cg ≥ 4Q²(1+K)·Cf."
    fwd_fields = [R("R1", "R1  (input)", "10k"), R("R2", "R2", "10k"), R("R3", "R3  (A to Vout)", "10k"),
                  Cap("Cg", "Cg  (A to GND)", "10n"), Cap("Cf", "Cf  (B to Vout)", "1n")]
    des_fields = [Hz("f0", "Cut-off / pole frequency f0", "1k"), Num("Q", "Quality factor Q", "0.7071"),
                  Num("K", "Gain magnitude K (V/V)", "1"), Cap("Cf", "Cf  (your choice)", "10n")]

    def ideal(self, p, ctx):
        if p["K"] <= 0:
            raise ValueError("K must be positive")
        return mfb_lp_design(p["f0"], p["Q"], p["K"], p["Cf"], ctx.snapc)

    def standardize(self, ideal, p, ctx):
        std = Circuit.standardize(self, ideal, p, ctx)
        if ctx.series:
            std.update(refine_stage(lambda c: mfb_lp_tf(c["R1"], c["R2"], c["R3"], c["Cg"], c["Cf"]),
                                    {k: ideal[k] for k in ("R1", "R2", "R3")}, {k: std[k] for k in ("Cg", "Cf")},
                                    ctx.series, p["f0"], p["Q"],
                                    extra=lambda tf, f0: abs(math.log(abs(tf[0][-1] / tf[1][-1]) / p["K"]))))
        return std

    def forward(self, p, ctx):
        tf = mfb_lp_tf(p["R1"], p["R2"], p["R3"], p["Cg"], p["Cf"])
        r, f0, Q = _stage_rows(tf)
        rows = [H("Characteristics")] + r + [KV("DC gain", gain_text(-p["R3"] / p["R1"]) + "   (inverting)")]
        return Outcome(rows, Model([tf]), NET_MFB_LP, f0)


class MFBBandPass(Circuit):
    key = "mfbbp"
    name = "Multiple-feedback band-pass (2nd order)"
    blurb = "Inverting band-pass (Delyiannis-Friend).  Centre gain H0 must be below 2Q²."
    fwd_fields = [R("R1", "R1  (input)", "10k"), R("R2", "R2  (B to Vout)", "20k"), R("R3", "R3  (A to GND)", "10k"),
                  Cap("C1", "C1  (A to B)", "10n"), Cap("C2", "C2  (A to Vout)", "10n")]
    des_fields = [Hz("f0", "Centre frequency f0", "1k"), Num("Q", "Quality factor Q", "5"),
                  Num("H0", "Gain at f0 (V/V)", "2"), Cap("C", "C1 = C2  (your choice)", "10n")]

    def ideal(self, p, ctx):
        if p["H0"] >= 2 * p["Q"] ** 2:
            raise ValueError(f"Gain must be below 2·Q² = {2 * p['Q'] ** 2:.4g} for this circuit")
        return mfb_bp_design(p["f0"], p["Q"], p["H0"], p["C"], ctx.snapc)

    def standardize(self, ideal, p, ctx):
        std = Circuit.standardize(self, ideal, p, ctx)
        if ctx.series:
            def gain_err(tf, f0):
                g = abs(freq_response(Model([tf]), [f0])[0])
                return abs(math.log(max(g, 1e-12) / p["H0"]))
            std.update(refine_stage(lambda c: mfb_bp_tf(c["R1"], c["R2"], c["R3"], c["C1"], c["C2"]),
                                    {k: ideal[k] for k in ("R1", "R2", "R3")}, {k: std[k] for k in ("C1", "C2")},
                                    ctx.series, p["f0"], p["Q"], extra=gain_err))
        return std

    def forward(self, p, ctx):
        tf = mfb_bp_tf(p["R1"], p["R2"], p["R3"], p["C1"], p["C2"])
        f0, Q = f0_q(*tf)
        h0 = abs(freq_response(Model([tf]), [f0])[0])
        rows = [H("Characteristics"), KV("Centre frequency f0", fmt(f0, "Hz")), KV("Quality factor Q", f"{Q:.4g}"),
                KV("Bandwidth f0/Q", fmt(f0 / Q, "Hz")), KV("Gain at f0", gain_text(-h0) + "   (inverting)")]
        return Outcome(rows, Model([tf]), NET_MFB_BP, f0)


class TwinTNotch(Circuit):
    key = "twint"
    name = "Twin-T notch filter"
    blurb = "Rejects one frequency f0 = 1/(2πRC).  Passive Q = 0.25; bootstrapping the ground leg raises Q = 1/(4(1−k))."
    fwd_fields = [R("R", "R", "15.9k"), Cap("C", "C", "10n"), Num("k", "Feedback fraction k  (0 = passive)", "0.9")]
    des_fields = [Hz("f0", "Notch frequency f0", "1k"), Num("Q", "Quality factor Q (≥ 0.25)", "2.5"), Cap("C", "C  (your choice)", "10n")]

    def ideal(self, p, ctx):
        if p["Q"] < 0.25:
            raise ValueError("Twin-T Q must be at least 0.25")
        return {"R": 1.0 / (TWO_PI * p["f0"] * p["C"]), "C": p["C"], "k": 1.0 - 1.0 / (4.0 * p["Q"])}

    def forward(self, p, ctx):
        R_, C, k = p["R"], p["C"], p["k"]
        if not 0 <= k < 1:
            raise ValueError("k must be between 0 and 1")
        tf = twin_t_tf(R_, C, k)
        f0 = 1.0 / (TWO_PI * R_ * C)
        rows = [H("Characteristics"), KV("Notch frequency f0", fmt(f0, "Hz")), KV("Quality factor Q", f"{1.0 / (4.0 * (1 - k)):.4g}"),
                KV("Bandwidth f0/Q", fmt(f0 * 4.0 * (1 - k), "Hz")),
                KV("Other parts", f"2 × R = {fmt(R_, 'Ω')},  R/2 = {fmt(R_ / 2, 'Ω')},  2 × C = {fmt(C, 'F')},  2C = {fmt(2 * C, 'F')}"),
                NOTE("Output must be buffered; the load should be much larger than R.")]
        net = ("High-pass leg:  Vin ── C ── N1 ── C ── Vout,   R/2 from N1 to G\n"
               "Low-pass leg:   Vin ── R ── N2 ── R ── Vout,   2C from N2 to G\n"
               "G = GND for k = 0. For k > 0, G is driven by a buffer fed from a divider across Vout giving k·Vout.")
        return Outcome(rows, Model([tf]), net, f0)


# ===========================================================================
# Active filter designer (cascade of stages)
# ===========================================================================
class ActiveFilterDesigner(Circuit):
    key = "afd"
    name = "Active filter designer (any order)"
    blurb = "Cascade of 2nd-order stages (+ one RC buffer for odd orders).  Pick type, response, order, cut-off and a base capacitor."
    fwd_fields = []
    des_fields = [Choice("typ", "Type", ["Low-pass", "High-pass"]), Choice("resp", "Response", RESPONSES),
                  Num("n", "Order (1 - 10)", "4", kind="int"), Hz("fc", "Cut-off frequency fc", "1k"),
                  Choice("topo", "Stage topology", ["Sallen-Key", "Multiple feedback (low-pass only)"]),
                  Cap("C", "Reference capacitor", "10n"), Num("K", "Gain per MFB stage K (V/V)", "1")]

    def design(self, p, ctx):
        typ, resp, n, fc, topo, C, K = (p[k] for k in ("typ", "resp", "n", "fc", "topo", "C", "K"))
        if not 1 <= n <= 10:
            raise ValueError("Order must be between 1 and 10")
        if K <= 0:
            raise ValueError("K must be positive")
        lp = typ == "Low-pass"
        use_mfb = lp and topo.startswith("Multiple")
        pairs, reals = pole_pairs(proto_poles(resp, n))
        rows = [H(f"{resp} {typ.lower()}, order {n}, fc = {fmt(fc, 'Hz')}")]
        if not lp and topo.startswith("Multiple"):
            rows.append(NOTE("MFB high-pass is not offered; Sallen-Key high-pass stages are used instead."))
        stages, nets = [], []
        sinfo = []
        opamps = 0
        i = 0

        def comp_line(label, ideal_v, unit):
            sv = ctx.snap(ideal_v)
            if abs(sv - ideal_v) > 1e-9 * abs(ideal_v):
                return sv, KV(label, f"{fmt(sv, unit)}  (ideal {fmt(ideal_v, unit)}, {100 * (sv - ideal_v) / ideal_v:+.1f} %)")
            return sv, KV(label, fmt(sv, unit))

        for w0n in reals:                          # first-order section: RC + buffer
            i += 1
            f0 = fc * w0n if lp else fc / w0n
            Cs = ctx.snapc(C)
            Rv = 1.0 / (TWO_PI * f0 * Cs)
            rows.append(H(f"Stage {i}: 1st-order RC + buffer ({'low' if lp else 'high'}-pass, f0 = {fmt(f0, 'Hz')})"))
            rows.append(KV("C", fmt(Cs, "F")))
            Rs_, line = comp_line("R", Rv, "Ω")
            rows.append(line)
            tau = Rs_ * Cs
            stages.append((P(1.0), P(tau, 1.0)) if lp else (P(tau, 0.0), P(tau, 1.0)))
            sinfo.append({"title": f"RC {'low' if lp else 'high'}-pass + buffer", "parts": [("R", Rs_, "Ω"), ("C", Cs, "F")],
                          "kind": "rclp" if lp else "rchp", "vals": {"R": Rs_, "C": Cs}})
            opamps += 1
            nets.append("RC stage:  " + ("Vin ── R ── N ── buffer (+),  C from N to GND" if lp else "Vin ── C ── N ── buffer (+),  R from N to GND"))
        for w0n, Q in pairs:
            i += 1
            f0 = fc * w0n if lp else fc / w0n
            if lp and not use_mfb:
                comps = sk_lp_design(f0, Q, C, ctx.snapc)
                names = [("R1", "Ω"), ("R2", "Ω"), ("C1", "F"), ("C2", "F")]
                kind = "Sallen-Key low-pass"
            elif lp:
                comps = mfb_lp_design(f0, Q, K, C, ctx.snapc)
                names = [("R1", "Ω"), ("R2", "Ω"), ("R3", "Ω"), ("Cg", "F"), ("Cf", "F")]
                kind = "MFB low-pass"
            else:
                comps = sk_hp_design(f0, Q, C, ctx.snapc)
                names = [("R1", "Ω"), ("R2", "Ω"), ("C1", "F"), ("C2", "F")]
                kind = "Sallen-Key high-pass"
            res_names = [nm for nm, un in names if un == "Ω"]
            std = {nm: comps[nm] for nm, un in names if un != "Ω"}
            if ctx.series:
                if kind == "Sallen-Key low-pass":
                    tf_of, extra = (lambda c: sk_lp_tf(c["R1"], c["R2"], c["C1"], c["C2"])), None
                elif kind == "MFB low-pass":
                    tf_of = lambda c: mfb_lp_tf(c["R1"], c["R2"], c["R3"], c["Cg"], c["Cf"])
                    extra = lambda tf, f0: abs(math.log(abs(tf[0][-1] / tf[1][-1]) / K))
                else:
                    tf_of, extra = (lambda c: sk_hp_tf(c["R1"], c["R2"], c["C1"], c["C2"])), None
                std.update(refine_stage(tf_of, {k: comps[k] for k in res_names}, dict(std), ctx.series, f0, Q, extra=extra))
            else:
                std.update({k: comps[k] for k in res_names})
            lines = []
            for nm, un in names:
                v, iv = std[nm], comps[nm]
                if abs(v - iv) > 1e-9 * abs(iv):
                    lines.append(KV(nm, f"{fmt(v, un)}  (ideal {fmt(iv, un)}, {100 * (v - iv) / iv:+.1f} %)"))
                else:
                    lines.append(KV(nm, fmt(v, un)))
            if kind == "Sallen-Key low-pass":
                tf, net = sk_lp_tf(std["R1"], std["R2"], std["C1"], std["C2"]), NET_SK_LP
            elif kind == "MFB low-pass":
                tf, net = mfb_lp_tf(std["R1"], std["R2"], std["R3"], std["Cg"], std["Cf"]), NET_MFB_LP
            else:
                tf, net = sk_hp_tf(std["R1"], std["R2"], std["C1"], std["C2"]), NET_SK_HP
            a_f0, a_Q = f0_q(*tf)
            rows.append(H(f"Stage {i}: {kind}   (target f0 = {fmt(f0, 'Hz')}, Q = {Q:.4g})"))
            rows += lines
            rows.append(KV("→ actual f0 / Q", f"{fmt(a_f0, 'Hz')} / {a_Q:.4g}"))
            stages.append(tf)
            sinfo.append({"title": f"{kind}  f0 {fmt(a_f0, 'Hz')}  Q {a_Q:.3g}", "parts": [(nm, std[nm], un) for nm, un in names],
                          "kind": {"Sallen-Key low-pass": "sklp", "MFB low-pass": "mfblp", "Sallen-Key high-pass": "skhp"}[kind],
                          "vals": dict(std)})
            opamps += 1
            if net not in nets:
                nets.append(net)
        model = Model(stages=stages)
        # overall performance
        ref = abs(freq_response(model, [fc / 1e3 if lp else fc * 1e3])[0])
        g_fc = abs(freq_response(model, [fc])[0])
        rows.append(H("Overall response"))
        rows.append(KV("Passband gain", f"{to_db(ref):.2f} dB" + ("  (stage count with MFB is inverting)" if use_mfb and (len(pairs) % 2) else "")))
        rows.append(KV("Gain at fc", f"{to_db(g_fc / ref):.2f} dB  (relative to passband)"))
        pts, _ = minus3db_points(model, fc / 20.0, fc * 20.0)
        if pts:
            rows.append(KV("-3 dB frequency", ", ".join(fmt(x, "Hz") for x in pts)))
        for mult in ((2, 4, 10) if lp else (0.5, 0.25, 0.1)):
            gm = abs(freq_response(model, [fc * mult])[0])
            rows.append(KV(f"Attenuation at {mult:g} × fc", f"{to_db(gm / ref):.1f} dB"))
        rows.append(KV("Op-amps needed", str(opamps)))
        if use_mfb:
            rows.append(NOTE("MFB stages invert: the total phase includes 180° per stage."))
        if resp.startswith("Chebyshev") and n % 2 == 0:
            rows.append(NOTE("Even-order Chebyshev: stages are normalised to 0 dB at DC (ripple runs 0 … +ripple)."))
        o = Outcome(rows, model, "Signal path: Vin → stage 1 → stage 2 → … → Vout\n" + "\n".join(nets), fc)
        o.info = {"stages": sinfo}
        return o


# ===========================================================================
# Passive filters (resistors, capacitors, coils)
# ===========================================================================
class RCFilter(Circuit):
    key = "rc"
    name = "RC filter (1st order)"
    blurb = "Passive RC with ideal source and no load.  fc = 1/(2πRC).  Design: give any two of fc, R, C and get the third."
    default_clip = False
    SOLVE = ["R  (from fc and C)", "C  (from fc and R)", "fc  (from R and C)"]
    fwd_fields = [Choice("out", "Type", ["Low-pass", "High-pass"]), R("R", "R", "10k"), Cap("C", "C", "15.9n")]
    des_fields = [Choice("out", "Type", ["Low-pass", "High-pass"]), Choice("solve", "Solve for", SOLVE),
                  when(Hz("fc", "Cut-off frequency fc", "1k"), "solve", ("R", "C")),
                  when(R("R", "R  (known)", "10k"), "solve", ("C", "fc")),
                  when(Cap("C", "C  (known)", "10n"), "solve", ("R", "fc"))]

    def known_keys(self, p):
        return [k for k, s in (("R", ("C", "fc")), ("C", ("R", "fc"))) if p["solve"].startswith(s)]

    def ideal(self, p, ctx):
        s = p["solve"]
        if s.startswith("R"):
            Rv, Cv = 1.0 / (TWO_PI * p["fc"] * p["C"]), p["C"]
        elif s.startswith("C"):
            Rv, Cv = p["R"], 1.0 / (TWO_PI * p["fc"] * p["R"])
        else:
            Rv, Cv = p["R"], p["C"]
        return {"out": p["out"], "R": Rv, "C": Cv}

    def forward(self, p, ctx):
        tau = p["R"] * p["C"]
        lp = p["out"] == "Low-pass"
        tf = (P(1.0), P(tau, 1.0)) if lp else (P(tau, 0.0), P(tau, 1.0))
        fc = 1.0 / (TWO_PI * tau)
        rows = [H("Characteristics"), KV("Cut-off frequency (-3 dB)", fmt(fc, "Hz")), KV("Time constant", fmt(tau, "s")),
                KV("Roll-off", "-20 dB/decade" if lp else "+20 dB/decade"), KV("Phase at fc", "-45°" if lp else "+45°"),
                KV("Rise time 10-90 %", fmt(2.197 * tau, "s"))]
        net = ("Low-pass:  Vin ── R ── Vout,  C from Vout to GND" if lp else "High-pass:  Vin ── C ── Vout,  R from Vout to GND")
        return Outcome(rows, Model([tf]), net, fc, clip=False)


class RLFilter(Circuit):
    key = "rl"
    name = "RL filter with coil (1st order)"
    blurb = "Passive RL with ideal source and no load.  fc = R/(2πL).  Design: give any two of fc, R, L and get the third."
    default_clip = False
    SOLVE = ["R  (from fc and L)", "L  (from fc and R)", "fc  (from R and L)"]
    fwd_fields = [Choice("out", "Type", ["Low-pass", "High-pass"]), R("R", "R", "628"), Ind("L", "L  (coil)", "100m")]
    des_fields = [Choice("out", "Type", ["Low-pass", "High-pass"]), Choice("solve", "Solve for", SOLVE),
                  when(Hz("fc", "Cut-off frequency fc", "1k"), "solve", ("R", "L")),
                  when(R("R", "R  (known)", "628"), "solve", ("L", "fc")),
                  when(Ind("L", "L  (known)", "100m"), "solve", ("R", "fc"))]

    def known_keys(self, p):
        return [k for k, s in (("R", ("L", "fc")), ("L", ("R", "fc"))) if p["solve"].startswith(s)]

    def ideal(self, p, ctx):
        s = p["solve"]
        if s.startswith("R"):
            Rv, Lv = TWO_PI * p["fc"] * p["L"], p["L"]
        elif s.startswith("L"):
            Rv, Lv = p["R"], p["R"] / (TWO_PI * p["fc"])
        else:
            Rv, Lv = p["R"], p["L"]
        return {"out": p["out"], "R": Rv, "L": Lv}

    def forward(self, p, ctx):
        tau = p["L"] / p["R"]
        lp = p["out"] == "Low-pass"
        tf = (P(1.0), P(tau, 1.0)) if lp else (P(tau, 0.0), P(tau, 1.0))
        fc = 1.0 / (TWO_PI * tau)
        rows = [H("Characteristics"), KV("Cut-off frequency (-3 dB)", fmt(fc, "Hz")), KV("Time constant L/R", fmt(tau, "s")),
                KV("Roll-off", "-20 dB/decade" if lp else "+20 dB/decade"), KV("Phase at fc", "-45°" if lp else "+45°")]
        if lp:
            net = "Low-pass:   Vin ── L ── Vout,  R from Vout to GND   (Vout = voltage across R)"
        else:
            net = "High-pass:  Vin ── R ── Vout,  L from Vout to GND   (Vout = voltage across L)"
        return Outcome(rows, Model([tf]), net, fc, clip=False)


class SeriesRLC(Circuit):
    key = "rlc"
    name = "Series RLC resonant circuit"
    blurb = "Source drives R + L + C in series; choose where Vout is taken.  f0 = 1/(2π√(LC)),  Q = √(L/C)/R."
    default_clip = False
    OUTS = ["Across R (band-pass)", "Across C (low-pass)", "Across L (high-pass)", "Across L+C (notch)"]
    fwd_fields = [Choice("out", "Output taken", OUTS), R("R", "R", "10"), Ind("L", "L  (coil)", "10m"), Cap("C", "C", "2.53u")]
    des_fields = [Choice("out", "Output taken", OUTS), Hz("f0", "Resonance frequency f0", "1k"), Num("Q", "Quality factor Q", "5"),
                  Cap("C", "C  (your choice)", "2.2u")]

    def ideal(self, p, ctx):
        w0 = TWO_PI * p["f0"]
        L = 1.0 / (w0 * w0 * p["C"])
        return {"out": p["out"], "R": w0 * L / p["Q"], "L": L, "C": p["C"]}

    def forward(self, p, ctx):
        R_, L, C = p["R"], p["L"], p["C"]
        den = P(L * C, R_ * C, 1.0)
        num = {0: P(R_ * C, 0.0), 1: P(1.0), 2: P(L * C, 0.0, 0.0), 3: P(L * C, 0.0, 1.0)}[self.OUTS.index(p["out"])]
        f0 = 1.0 / (TWO_PI * math.sqrt(L * C))
        Q = math.sqrt(L / C) / R_
        rows = [H("Characteristics"), KV("Resonance frequency f0", fmt(f0, "Hz")), KV("Quality factor Q", f"{Q:.4g}"),
                KV("Bandwidth f0/Q", fmt(f0 / Q, "Hz")), KV("Characteristic impedance √(L/C)", fmt(math.sqrt(L / C), "Ω")),
                KV("Damping ratio ζ", f"{1.0 / (2.0 * Q):.4g}")]
        if p["out"].startswith("Across C") or p["out"].startswith("Across L ("):
            rows.append(KV("Peak gain at resonance", f"{to_db(Q):.1f} dB  (≈ Q)"))
        net = "Vin ── R ── L ── C ── GND  (series).  Vout: " + p["out"].lower()
        return Outcome(rows, Model([(num, den)]), net, f0, clip=False)


class LCLadder(Circuit):
    key = "lcl"
    name = "LC ladder filter (coils + capacitors)"
    blurb = "Doubly-terminated passive Butterworth / Chebyshev ladder: low-pass, high-pass or band-pass.  Response includes source and load."
    default_clip = False
    fwd_fields = []
    des_fields = [Choice("kind", "Type", ["Low-pass", "High-pass", "Band-pass"]),
                  Choice("resp", "Response", [r for r in RESPONSES if r != "Bessel"]),
                  Num("n", "Order (2 - 9)", "5", kind="int"),
                  Hz("fc", "Cut-off  (band-pass: centre frequency)", "1M"),
                  Hz("bw", "Bandwidth (band-pass only)", "200k"),
                  R("Z0", "Source / load impedance", "50"),
                  Choice("first", "First element from the source", ["Shunt element first (π)", "Series element first (T)"])]

    def design(self, p, ctx):
        kind, resp, n, fc, bw, Z0, first = (p[k] for k in ("kind", "resp", "n", "fc", "bw", "Z0", "first"))
        if not 2 <= n <= 9:
            raise ValueError("Order must be between 2 and 9")
        series_first = first.startswith("Series")
        if kind == "Band-pass" and bw >= 1.6 * fc:
            raise ValueError("Bandwidth too large compared with the centre frequency")
        g, load = ladder_g(resp, n)
        _, RLn = ladder_proto_denominator(g, series_first, load)
        comps, RL = ladder_components(kind, g, series_first, Z0, fc, bw, RLn)
        rows = [H(f"{resp} {kind.lower()} ladder, order {n}"),
                KV("Source resistance", fmt(Z0, "Ω")), KV("Load resistance", fmt(RL, "Ω"))]
        if abs(RL - Z0) / Z0 > 0.01:
            rows.append(WARN(f"Even-order Chebyshev needs a load of {fmt(RL, 'Ω')} (not equal to the source)."))
        rows.append(H("Components, in order from the source" + (f"  (rounded: {ctx.label})" if ctx.rounding else "")))
        elems = []
        for name, role, vals in comps:
            sv, parts = {}, []
            for k, v in vals.items():
                s = ctx.snapc(v)
                sv[k] = s
                u = "H" if k == "L" else "F"
                txt = fmt(s, u)
                if abs(s - v) > 1e-9 * abs(v):
                    txt += f"  (ideal {fmt(v, u)}, {100 * (s - v) / v:+.1f} %)"
                parts.append(txt if len(vals) == 1 else f"{k} = {txt}")
            rows.append(KV(f"{name}  {role.split(' (')[0]}", "   ".join(parts)))
            elems.append((role, sv))
        ws = TWO_PI * fc
        if ctx.rounding:
            model = ladder_model_from_elements(kind, elems, Z0, RL, ws)
            rows.append(NOTE("Response below uses the rounded component values."))
        else:
            model, _ = ladder_model(kind, resp, n, series_first, fc, bw)
        pts, pk = minus3db_points(model, fc / 20.0 if kind != "Band-pass" else max(fc - bw * 3, fc / 20.0), fc * 20.0 if kind != "Band-pass" else fc + bw * 3)
        rows.append(H("Resulting response"))
        rows.append(KV("Peak passband gain", f"{pk:.2f} dB" + ("  (6.02 dB loss: source/load divider)" if abs(pk + 6.02) < 0.4 else "")))
        if pts:
            rows.append(KV("-3 dB points (from peak)", ", ".join(fmt(x, "Hz") for x in pts)))
        att_f = {"Low-pass": (2, 4), "High-pass": (0.5, 0.25)}.get(kind)
        if att_f:
            for m in att_f:
                gm = abs(freq_response(model, [fc * m])[0])
                rows.append(KV(f"Gain at {m:g} × fc", f"{to_db(gm):.1f} dB"))
        net = ("Series branches are in the signal path, shunt branches go to GND.\n"
               "Source (Z0) → first element → … → last element → load RL.")
        o = Outcome(rows, model, net, fc, clip=False)
        o.info = {"ladder": {"kind": kind, "elems": [(nm, rl, vl) for (nm, rl, _), (_, vl) in zip(comps, elems)],
                             "Z0": Z0, "RL": RL}}
        return o


# ===========================================================================
# Tools (no simulation)
# ===========================================================================
class Reactance(Circuit):
    key = "xc"
    name = "Reactance of C and L"
    blurb = "Xc = 1/(2πfC),  XL = 2πfL at a given frequency."
    has_model = False
    fwd_fields = [Hz("f", "Frequency", "1k"), Cap("C", "Capacitor C", "100n", zero_ok=True), Ind("L", "Coil L", "10m", zero_ok=True)]

    def forward(self, p, ctx):
        f, C, L = p["f"], p["C"], p["L"]
        rows = [H("Reactance")]
        if C > 0:
            rows.append(KV("Xc = 1/(2πfC)", fmt(1.0 / (TWO_PI * f * C), "Ω") + "   (phase -90°)"))
        if L > 0:
            rows.append(KV("XL = 2πfL", fmt(TWO_PI * f * L, "Ω") + "   (phase +90°)"))
        if C > 0 and L > 0:
            x = TWO_PI * f * L - 1.0 / (TWO_PI * f * C)
            rows.append(KV("Series L + C", fmt(x, "Ω") + ("  (inductive)" if x > 0 else "  (capacitive)")))
            rows.append(KV("Series resonance", fmt(1.0 / (TWO_PI * math.sqrt(L * C)), "Hz")))
        return Outcome(rows)


class LCResonance(Circuit):
    key = "lc"
    name = "LC resonance"
    blurb = "f0 = 1/(2π√(LC)).  Design: give any two of f0, L, C and get the third."
    has_model = False
    SOLVE = ["L  (from f0 and C)", "C  (from f0 and L)", "f0  (from L and C)"]
    fwd_fields = [Ind("L", "L  (coil)", "10m"), Cap("C", "C", "2.53u")]
    des_fields = [Choice("solve", "Solve for", SOLVE),
                  when(Hz("f0", "Resonance frequency f0", "1k"), "solve", ("L", "C")),
                  when(Ind("L", "L  (known)", "10m"), "solve", ("C", "f0")),
                  when(Cap("C", "C  (known)", "2.2u"), "solve", ("L", "f0"))]

    def known_keys(self, p):
        return [k for k, s in (("L", ("C", "f0")), ("C", ("L", "f0"))) if p["solve"].startswith(s)]

    def ideal(self, p, ctx):
        s = p["solve"]
        if s.startswith("L"):
            w0 = TWO_PI * p["f0"]
            return {"L": 1.0 / (w0 * w0 * p["C"]), "C": p["C"]}
        if s.startswith("C"):
            w0 = TWO_PI * p["f0"]
            return {"L": p["L"], "C": 1.0 / (w0 * w0 * p["L"])}
        return {"L": p["L"], "C": p["C"]}

    def forward(self, p, ctx):
        L, C = p["L"], p["C"]
        f0 = 1.0 / (TWO_PI * math.sqrt(L * C))
        rows = [H("Resonance"), KV("f0", fmt(f0, "Hz")),
                KV("Characteristic impedance √(L/C)", fmt(math.sqrt(L / C), "Ω"))]
        return Outcome(rows, f_char=f0)


class VoltageDivider(Circuit):
    key = "vd"
    name = "Voltage divider"
    blurb = "Vout = Vin · R2 / (R1 + R2)."
    has_model = False
    fwd_fields = [R("R1", "R1  (top)", "10k"), R("R2", "R2  (bottom)", "10k"), Num("vin", "Vin", "5", "V")]
    des_fields = [Num("vin", "Vin", "5", "V"), Num("vout", "Wanted Vout", "3.3", "V"), R("R2", "R2  (your choice)", "10k")]

    def ideal(self, p, ctx):
        if not 0 < p["vout"] < p["vin"]:
            raise ValueError("Vout must be between 0 and Vin")
        return {"R1": p["R2"] * (p["vin"] - p["vout"]) / p["vout"], "R2": p["R2"], "vin": p["vin"]}

    def forward(self, p, ctx):
        R1, R2, vin = p["R1"], p["R2"], p["vin"]
        rows = [H("Result"), KV("Vout", fmt(vin * R2 / (R1 + R2), "V")), KV("Ratio", f"{R2 / (R1 + R2):.5g}"),
                KV("Output resistance", fmt(R1 * R2 / (R1 + R2), "Ω")), KV("Current", fmt(vin / (R1 + R2), "A"))]
        return Outcome(rows)


class DecibelConverter(Circuit):
    key = "db"
    name = "dB converter"
    blurb = "Converts a voltage ratio to dB and back."
    has_model = False
    fwd_fields = [Num("ratio", "Voltage ratio Vout/Vin", "10", signed=True)]
    des_fields = [Num("db", "Gain in dB", "20", "dB", signed=True)]

    def design(self, p, ctx):
        r = 10 ** (p["db"] / 20.0)
        rows = [H("Result"), KV("Voltage ratio", f"{r:.6g}"), KV("Power ratio", f"{r * r:.6g}")]
        return Outcome(rows)

    def forward(self, p, ctx):
        r = abs(p["ratio"])
        if r == 0:
            raise ValueError("Ratio must not be 0")
        return Outcome([H("Result"), KV("Voltage gain", f"{to_db(r):.4f} dB"), KV("Power ratio", f"{r * r:.6g}")])


class NearestValue(Circuit):
    key = "e"
    name = "Nearest standard value"
    blurb = "Shows the closest E-series value for any resistor, capacitor or coil value."
    has_model = False
    fwd_fields = [Num("v", "Value (e.g. 4.7k, 33n, 2.2u)", "5.1k")]

    def forward(self, p, ctx):
        v = p["v"]
        if v <= 0:
            raise ValueError("Value must be positive")
        rows = [H("Nearest values")]
        for s in ("E6", "E12", "E24", "E48", "E96"):
            n = snap(v, s)
            rows.append(KV(s, f"{fmt(n)}    ({100 * (n - v) / v:+.2f} %)"))
        return Outcome(rows)


# ===========================================================================
TABS = {
    "Op-Amp Circuits": [InvertingAmp(), NonInvertingAmp(), SummingAmp(), DifferenceAmp(), Integrator(),
                        Differentiator(), InstrumentationAmp()],
    "Active Filters": [ActiveFilterDesigner(), SallenKeyLP(), SallenKeyHP(), MFBLowPass(), MFBBandPass(), TwinTNotch()],
    "Passive (L, C, R)": [LCLadder(), SeriesRLC(), RCFilter(), RLFilter()],
    "Tools": [Reactance(), LCResonance(), VoltageDivider(), DecibelConverter(), NearestValue()],
}
