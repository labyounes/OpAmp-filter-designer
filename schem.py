"""Schematic drawing with matplotlib: parts carry their real values.

`Drawer` offers the symbols; `draw(key, d, values, outcome)` draws the diagram of one circuit.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from core import fmt

WIRE = "#1d2733"
PART = "#16365c"
VALC = "#b8470f"
FILL = "#eaf1f9"
MUTED = "#6b717c"


def fr(v):
    return fmt(v, "Ω")


def fc(v):
    return fmt(v, "F")


def fl(v):
    return fmt(v, "H")


class Drawer:
    def __init__(self, ax, fs=8.0):
        self.ax = ax
        self.fs = fs
        self.bb = [1e9, 1e9, -1e9, -1e9]
        self.texts = []
        self.margin = 0.5
        self.off = (0.0, 0.0)
        self.lbls = []
        self.upp = 0.5          # data units per point; refined in finish()
        self.off_of = {}

    # -- bookkeeping ---------------------------------------------------------
    def _b(self, x, y):
        b = self.bb
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)

    def text(self, x, y, s, color=PART, ha="center", va="center", weight="normal", size=None, style="normal"):
        x, y = x + self.off[0], y + self.off[1]
        t = self.ax.text(x, y, s, color=color, ha=ha, va=va, fontsize=size or self.fs, fontweight=weight,
                         fontstyle=style, zorder=6)
        self.texts.append(t)
        self._b(x, y)
        self.off_of[id(t)] = self.off
        return t

    # -- basic symbols ---------------------------------------------------------
    def wire(self, *pts):
        pts = [(x + self.off[0], y + self.off[1]) for x, y in pts]
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=WIRE, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        for x, y in pts:
            self._b(x, y)

    def dot(self, x, y):
        x, y = x + self.off[0], y + self.off[1]
        self.ax.plot([x], [y], "o", ms=5, color=WIRE, zorder=5)
        self._b(x, y)

    def term(self, x, y, name, side="left", sub=None):
        self.ax.add_patch(Circle((x + self.off[0], y + self.off[1]), 0.17, fc="white", ec=WIRE, lw=1.4, zorder=5))
        self._b(x + self.off[0], y + self.off[1])
        dx = -0.45 if side == "left" else 0.45
        ha = "right" if side == "left" else "left"
        self.text(x + dx, y + (0.22 if sub else 0), name, ha=ha, weight="bold")
        if sub:
            self.text(x + dx, y - 0.45, sub, ha=ha, color=VALC)

    def ground(self, x, y):
        self.wire((x, y), (x, y - 0.35))
        for i, w in enumerate((1.0, 0.64, 0.28)):
            yy = y - 0.35 - 0.2 * i
            self.wire((x - w / 2, yy), (x + w / 2, yy))

    # -- two-terminal parts ----------------------------------------------------
    def _frame(self, p, q):
        o = np.array(self.off, float)
        p, q = np.array(p, float) + o, np.array(q, float) + o
        d = q - p
        L = float(np.hypot(*d))
        u = d / L
        n = np.array([-u[1], u[0]])
        c = (p + q) / 2
        return p, q, u, n, c, L

    def _lab(self, c, u, n, name, value, lab):
        c = np.array(c, float) - np.array(self.off, float)
        horiz = abs(u[0]) >= abs(u[1])
        if lab is None:
            lab = "above" if horiz else "right"
        if horiz:
            ha = "center"
            va = "bottom" if lab == "above" else "top"
        else:
            ha = "left" if lab == "right" else "right"
            va = "bottom"
        t_name = self.text(c[0], c[1], name, ha=ha, va=va, weight="bold")
        t_val = self.text(c[0], c[1], value, ha=ha, va="bottom" if not horiz else va, color=VALC)
        if not horiz:
            t_val.set_va("top")
        self.lbls.append((t_name, t_val, c, horiz, lab))

    def _place_labels(self):
        lh = self.fs * 1.38 * self.upp          # one text line in data units
        gap = 0.8
        for t_name, t_val, c, horiz, lab in self.lbls:
            X, Y = self.off_of[id(t_name)] if hasattr(self, "off_of") and id(t_name) in self.off_of else (0.0, 0.0)
            if horiz:
                if lab == "above":
                    t_val.set_position((c[0] + X, c[1] + Y + gap))
                    t_name.set_position((c[0] + X, c[1] + Y + gap + lh))
                else:
                    t_name.set_position((c[0] + X, c[1] + Y - gap))
                    t_val.set_position((c[0] + X, c[1] + Y - gap - lh))
            else:
                sgn = 1 if lab == "right" else -1
                x = c[0] + X + sgn * 0.9
                t_name.set_position((x, c[1] + Y + 0.04))
                t_val.set_position((x, c[1] + Y - 0.04))

    def _raw_wire(self, *pts):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=WIRE, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        for x, y in pts:
            self._b(x, y)

    def _poly(self, pts):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=PART, lw=1.7, solid_joinstyle="round", zorder=3)

    def R(self, p, q, name, value, lab=None):
        p, q, u, n, c, L = self._frame(p, q)
        loc = [(-1.0, 0), (-0.83, .3), (-0.5, -.3), (-0.17, .3), (0.17, -.3), (0.5, .3), (0.83, -.3), (1.0, 0)]
        pts = [tuple(c + s * u + t * n) for s, t in loc]
        self._raw_wire(tuple(p), pts[0])
        self._poly(pts)
        self._raw_wire(pts[-1], tuple(q))
        self._lab(c, u, n, name, value, lab)

    def C(self, p, q, name, value, lab=None):
        p, q, u, n, c, L = self._frame(p, q)
        a, b = c - 0.17 * u, c + 0.17 * u
        self._raw_wire(tuple(p), tuple(a))
        self._raw_wire(tuple(b), tuple(q))
        for e in (a, b):
            self._poly([tuple(e + 0.62 * n), tuple(e - 0.62 * n)])
        self._lab(c, u, n, name, value, lab)

    def L(self, p, q, name, value, lab=None):
        p, q, u, n, c, L = self._frame(p, q)
        pts = []
        for k in range(4):
            s0 = -1.1 + 0.55 * k
            for th in np.linspace(math.pi, 0, 12):
                pts.append(tuple(c + (s0 + 0.275 + 0.275 * math.cos(th)) * u + 0.34 * math.sin(th) * n))
        self._raw_wire(tuple(p), pts[0])
        self._poly(pts)
        self._raw_wire(pts[-1], tuple(q))
        self._lab(c, u, n, name, value, lab)

    def part(self, kind, p, q, name, value, lab=None):
        {"R": self.R, "C": self.C, "L": self.L}[kind](p, q, name, value, lab)

    # -- op-amp ---------------------------------------------------------------
    def opamp(self, ox, oy, minus_top=True, name=None):
        h = 1.7
        X, Y = self.off
        self.ax.add_patch(Polygon([(ox + X, oy + Y + h), (ox + X, oy + Y - h), (ox + X + 3.2, oy + Y)], closed=True,
                                  fc=FILL, ec=PART, lw=1.8, zorder=3))
        for x, y in ((ox, oy + h), (ox, oy - h), (ox + 3.2, oy)):
            self._b(x + X, y + Y)
        top, bot = ("−", "+") if minus_top else ("+", "−")
        self.text(ox + 0.35, oy + 0.8, top, size=self.fs + 3, weight="bold")
        self.text(ox + 0.35, oy - 0.8, bot, size=self.fs + 3, weight="bold")
        if name:
            self.text(ox + 1.0, oy, name, color=MUTED, size=self.fs - 1)
        top_p, bot_p = (ox, oy + 0.8), (ox, oy - 0.8)
        return {"inm": top_p if minus_top else bot_p, "inp": bot_p if minus_top else top_p, "out": (ox + 3.2, oy)}

    def box(self, x, y, w, h, fill=FILL):
        x, y = x + self.off[0], y + self.off[1]
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=PART, lw=1.6, zorder=3))
        self._b(x, y)
        self._b(x + w, y + h)

    def arrow(self, p, q):
        p = (p[0] + self.off[0], p[1] + self.off[1])
        q = (q[0] + self.off[0], q[1] + self.off[1])
        self.ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.5), zorder=4)
        self._b(*p)
        self._b(*q)

    # -- final scaling ---------------------------------------------------------
    def _scale(self):
        ax = self.ax
        fig = ax.figure
        ax.apply_aspect()
        pos = ax.get_position()
        w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        per_in = max((x1 - x0) / max(w_in, 1e-6), (y1 - y0) / max(h_in, 1e-6))
        self.upp = per_in / 72.0

    def finish(self):
        ax = self.ax
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        if self.bb[0] > self.bb[2]:
            return
        m = self.margin

        def apply(b):
            ax.set_xlim(b[0] - m, b[2] + m)
            ax.set_ylim(b[1] - m, b[3] + m)

        b = list(self.bb)
        apply(b)
        fig = ax.figure
        for _ in range(6):
            try:
                self._scale()
                self._place_labels()
                fig.canvas.draw()
                r = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else fig._get_renderer()
                inv = ax.transData.inverted()
                nb = list(self.bb)
                for t in self.texts:
                    bb = t.get_window_extent(r)
                    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
                    nb = [min(nb[0], x0), min(nb[1], y0), max(nb[2], x1), max(nb[3], y1)]
                if all(abs(nb[i] - b[i]) < 0.02 for i in range(4)):
                    break
                b = nb
                apply(b)
            except Exception:  # noqa: BLE001
                break


# ===========================================================================
# helpers shared by several circuits
# ===========================================================================
def _vout(d, x, y, x_term):
    d.wire((x, y), (x_term, y))
    d.term(x_term, y, "Vout", side="right")


def _inverting_frame(d, rin_parts, feedback_parts, vin_label="Vin", sub=None):
    """Inverting op-amp: input chain of parts to (-), feedback branch(es) in parallel, (+) to GND."""
    oa = d.opamp(10, 0, minus_top=True)
    d.term(1, .8, vin_label, sub=sub)
    x = 1.0
    d.wire((1, .8), (2.2, .8))
    x = 2.2
    for kind, name, val, ln in rin_parts:
        if kind == "W":
            d.wire((x, .8), (x + 1.2, .8))
            x += 1.2
        else:
            d.part(kind, (x, .8), (x + ln, .8), name, val)
            x += ln
    d.wire((x, .8), (10, .8))
    nx = 9.0
    d.dot(nx, .8)
    return oa, nx


def _feedback_stack(d, nx, parts, y0=3.4, dy=2.6, x1=9.6, x2=13.2, ox=14.2):
    """Parallel feedback branches from node (nx, 0.8) to output node (ox, 0)."""
    top = y0 + dy * (len(parts) - 1)
    d.wire((nx, .8), (nx, top))
    d.wire((ox, 0), (ox, top))
    d.dot(ox, 0)
    for i, (kind, name, val) in enumerate(parts):
        y = y0 + dy * i
        if i > 0:
            d.dot(nx, y)
            d.dot(ox, y)
        if kind == "W":
            d.wire((nx, y), (ox, y))
        else:
            d.wire((nx, y), (x1, y))
            d.part(kind, (x1, y), (x2, y), name, val)
            d.wire((x2, y), (ox, y))


# ===========================================================================
# Op-amp circuits
# ===========================================================================
def draw_inv(d, v, out):
    oa, nx = _inverting_frame(d, [("R", "Rin", fr(v["Rin"]), 3.8)], None)
    _feedback_stack(d, nx, [("R", "Rf", fr(v["Rf"]))])
    d.wire((10, -.8), (9, -.8), (9, -1.5))
    d.ground(9, -1.5)
    d.wire((13.2, 0), (14.2, 0))
    _vout(d, 14.2, 0, 16.2)
    d.wire(oa["out"], (14.2, 0))


def draw_noninv(d, v, out):
    oa = d.opamp(10, 0, minus_top=False)
    d.term(1, .8, "Vin")
    d.wire((1, .8), (10, .8))
    Rf, Rg = v["Rf"], v["Rg"]
    d.wire((10, -.8), (9, -.8), (9, -3.4))
    d.dot(9, -3.4)
    if Rf > 0:
        d.wire((9, -3.4), (9.6, -3.4))
        d.R((9.6, -3.4), (13.2, -3.4), "Rf", fr(Rf), lab="below")
        d.wire((13.2, -3.4), (14.2, -3.4), (14.2, 0))
    else:
        d.wire((9, -3.4), (14.2, -3.4), (14.2, 0))
    d.dot(14.2, 0)
    d.wire(oa["out"], (14.2, 0))
    _vout(d, 14.2, 0, 16.2)
    if math.isfinite(Rg):
        d.R((9, -3.4), (9, -6.8), "Rg", fr(Rg))
        d.ground(9, -6.8)
    else:
        d.text(10.2, -4.6, "no Rg: voltage follower", color=MUTED, ha="left", style="italic")


def draw_sum(d, v, out):
    oa = d.opamp(10, 0, minus_top=True)
    bus = 8.0
    d.term(1, 2.6, "V1")
    d.wire((1, 2.6), (2.2, 2.6))
    d.R((2.2, 2.6), (6.0, 2.6), "R1", fr(v["R1"]))
    d.wire((6.0, 2.6), (bus, 2.6))
    d.term(1, 0.0, "V2", sub=fmt(v["v2"], "V") + " DC")
    d.wire((1, 0), (2.2, 0))
    d.R((2.2, 0), (6.0, 0), "R2", fr(v["R2"]))
    d.wire((6.0, 0), (bus, 0))
    d.wire((bus, 0), (bus, 5.2))
    d.dot(bus, 2.6)
    d.dot(bus, .8)
    d.wire((bus, .8), (10, .8))
    d.dot(bus, 5.2)
    d.wire((bus, 5.2), (9.6, 5.2))
    d.R((9.6, 5.2), (13.2, 5.2), "Rf", fr(v["Rf"]))
    d.wire((13.2, 5.2), (14.2, 5.2), (14.2, 0))
    d.dot(14.2, 0)
    d.wire(oa["out"], (14.2, 0))
    _vout(d, 14.2, 0, 16.2)
    d.wire((10, -.8), (9, -.8), (9, -1.5))
    d.ground(9, -1.5)


def draw_diff(d, v, out):
    oa = d.opamp(10, 0, minus_top=True)
    d.term(1, .8, "V−")
    d.wire((1, .8), (2.2, .8))
    d.R((2.2, .8), (6.0, .8), "R1", fr(v["R1"]))
    d.wire((6.0, .8), (8.2, .8), (10, .8))
    d.dot(8.2, .8)
    d.wire((8.2, .8), (8.2, 3.6), (9.6, 3.6))
    d.R((9.6, 3.6), (13.2, 3.6), "R2", fr(v["R2"]))
    d.wire((13.2, 3.6), (14.2, 3.6), (14.2, 0))
    d.dot(14.2, 0)
    d.wire(oa["out"], (14.2, 0))
    _vout(d, 14.2, 0, 16.2)
    d.term(1, -3.2, "V+")
    d.wire((1, -3.2), (2.2, -3.2))
    d.R((2.2, -3.2), (6.0, -3.2), "R1", fr(v["R1"]), lab="below")
    d.wire((6.0, -3.2), (8.2, -3.2))
    d.dot(8.2, -3.2)
    d.wire((8.2, -3.2), (8.2, -.8), (10, -.8))
    d.R((8.2, -3.2), (8.2, -6.6), "R2", fr(v["R2"]))
    d.ground(8.2, -6.6)


def draw_int(d, v, out):
    oa, nx = _inverting_frame(d, [("R", "R", fr(v["R"]), 3.8)], None)
    fb = [("C", "C", fc(v["C"]))]
    if v.get("Rp", 0) > 0:
        fb.append(("R", "Rp", fr(v["Rp"])))
    _feedback_stack(d, nx, fb)
    d.wire((10, -.8), (9, -.8), (9, -1.5))
    d.ground(9, -1.5)
    d.wire((13.2, 0), (14.2, 0))
    d.wire(oa["out"], (14.2, 0))
    _vout(d, 14.2, 0, 16.2)


def draw_dif(d, v, out):
    oa = d.opamp(11, 0, minus_top=True)
    d.term(1, .8, "Vin")
    d.wire((1, .8), (2.0, .8))
    d.C((2.0, .8), (5.2, .8), "C", fc(v["C"]))
    if v.get("Rs", 0) > 0:
        d.R((5.2, .8), (8.8, .8), "Rs", fr(v["Rs"]))
        x = 8.8
    else:
        d.wire((5.2, .8), (8.8, .8))
        x = 8.8
    d.wire((x, .8), (11, .8))
    nx = 10.0
    d.dot(nx, .8)
    fb = [("R", "R", fr(v["R"]))]
    if v.get("Cf", 0) > 0:
        fb.append(("C", "Cf", fc(v["Cf"])))
    _feedback_stack(d, nx, fb, x1=10.6, x2=14.2, ox=15.2)
    d.wire((11, -.8), (10, -.8), (10, -1.5))
    d.ground(10, -1.5)
    d.wire(oa["out"], (15.2, 0))
    _vout(d, 15.2, 0, 17.2)


def draw_ina(d, v, out):
    a1 = d.opamp(6, 7.0, minus_top=False, name="A1")
    a2 = d.opamp(6, -2.2, minus_top=True, name="A2")
    # inputs
    d.term(1, 7.8, "V+")
    d.wire((1, 7.8), (6, 7.8))
    d.term(1, -3.0, "V−")
    d.wire((1, -3.0), (6, -3.0))
    # gain resistor between the two (-) inputs
    d.wire((6, 6.2), (3.6, 6.2), (3.6, 5.0))
    d.R((3.6, 5.0), (3.6, 0.0), "Rg", fr(v["Rg"]), lab="left")
    d.wire((3.6, 0.0), (3.6, -1.4), (6, -1.4))
    # feedback resistors R1 of A1 and A2
    d.wire(a1["out"], (10.2, 7.0))
    d.dot(10.2, 7.0)
    d.wire((10.2, 7.0), (10.2, 4.6), (9.6, 4.6))
    d.R((9.6, 4.6), (6.4, 4.6), "R1", fr(v["R1"]), lab="below")
    d.wire((6.4, 4.6), (5.0, 4.6), (5.0, 6.2))
    d.dot(3.6, 6.2)
    d.wire((6, 6.2), (5.0, 6.2))
    d.wire(a2["out"], (10.2, -2.2))
    d.dot(10.2, -2.2)
    d.wire((10.2, -2.2), (10.2, 0.2), (9.6, 0.2))
    d.R((9.6, 0.2), (6.4, 0.2), "R1", fr(v["R1"]), lab="above")
    d.wire((6.4, 0.2), (5.0, 0.2), (5.0, -1.4))
    d.dot(5.0, -1.4)
    # difference stage
    a3 = d.opamp(19, 2.4, minus_top=True, name="A3")
    d.wire((10.2, 7.0), (11.0, 7.0))
    d.R((11.0, 7.0), (14.6, 7.0), "R2", fr(v["R2"]))
    d.wire((14.6, 7.0), (16.4, 7.0), (16.4, 3.2), (19, 3.2))
    d.dot(16.4, 4.8)
    d.wire((16.4, 4.8), (17.2, 4.8))
    d.R((17.2, 4.8), (20.8, 4.8), "R3", fr(v["R3"]))
    d.wire((20.8, 4.8), (23.0, 4.8), (23.0, 2.4))
    d.dot(23.0, 2.4)
    d.wire(a3["out"], (23.0, 2.4))
    d.wire((10.2, -2.2), (11.0, -2.2))
    d.R((11.0, -2.2), (14.6, -2.2), "R2", fr(v["R2"]), lab="below")
    d.wire((14.6, -2.2), (17.4, -2.2), (17.4, 1.6), (19, 1.6))
    # R3 to ground (below the lower input)
    d.dot(18.2, 1.6)
    d.R((18.2, 1.6), (18.2, -1.6), "R3", fr(v["R3"]), lab="right")
    d.ground(18.2, -1.6)
    _vout(d, 23.0, 2.4, 25.0)


# ===========================================================================
# Active filter stages
# ===========================================================================
def _sallen_key(d, series, feedback, shunt, ox=11.0):
    """series=(k1,n1,v1,k2,n2,v2); feedback/shunt=(kind,name,value)."""
    oa = d.opamp(ox, 0, minus_top=True)
    k1, n1, v1, k2, n2, v2 = series
    d.term(1, -.8, "Vin")
    d.wire((1, -.8), (2.0, -.8))
    d.part(k1, (2.0, -.8), (5.2, -.8), n1, v1)
    ax = 6.4
    d.wire((5.2, -.8), (ax, -.8))
    d.dot(ax, -.8)
    d.wire((ax, -.8), (7.0, -.8))
    d.part(k2, (7.0, -.8), (10.2 if ox <= 11 else ox - 0.8, -.8), n2, v2)
    bx = 10.6
    d.wire((10.2, -.8), (ox, -.8))
    d.dot(bx, -.8)
    d.part(shunt[0], (bx, -.8), (bx, -4.0), shunt[1], shunt[2])
    d.ground(bx, -4.0)
    # feedback from A to Vout
    top = 3.6
    ox2 = ox + 4.2
    d.wire((ax, -.8), (ax, top), (7.2, top))
    d.part(feedback[0], (7.2, top), (10.8, top), feedback[1], feedback[2])
    d.wire((10.8, top), (ox2, top), (ox2, 0))
    d.dot(ox2, 0)
    d.wire(oa["out"], (ox2, 0))
    # (-) input tied to the output
    d.wire(oa["inm"], (10.4, .8), (10.4, 1.9), (ox2, 1.9))
    d.dot(ox2, 1.9)
    _vout(d, ox2, 0, ox2 + 2.0)


def draw_sklp(d, v, out):
    _sallen_key(d, ("R", "R1", fr(v["R1"]), "R", "R2", fr(v["R2"])), ("C", "C1", fc(v["C1"])), ("C", "C2", fc(v["C2"])))


def draw_skhp(d, v, out):
    _sallen_key(d, ("C", "C1", fc(v["C1"]), "C", "C2", fc(v["C2"])), ("R", "R1", fr(v["R1"])), ("R", "R2", fr(v["R2"])))


def _mfb(d, rin, ab, top, shunt, bout, ox=12.0):
    """Multiple-feedback stage. Each part = (kind, name, value)."""
    oa = d.opamp(ox, 0, minus_top=True)
    d.term(1, .8, "Vin")
    d.wire((1, .8), (2.0, .8))
    d.part(rin[0], (2.0, .8), (5.4, .8), rin[1], rin[2])
    ax = 6.6
    d.wire((5.4, .8), (ax, .8))
    d.dot(ax, .8)
    d.wire((ax, .8), (7.2, .8))
    d.part(ab[0], (7.2, .8), (10.4, .8), ab[1], ab[2])
    bx = 11.0
    d.wire((10.4, .8), (ox, .8))
    d.dot(bx, .8)
    d.part(shunt[0], (ax, .8), (ax, -2.8), shunt[1], shunt[2])
    d.ground(ax, -2.8)
    ox2 = ox + 4.2
    # top feedback from A to out
    d.wire((ax, .8), (ax, 5.2), (7.4, 5.2))
    d.part(top[0], (7.4, 5.2), (11.0, 5.2), top[1], top[2])
    d.wire((11.0, 5.2), (ox2, 5.2), (ox2, 0))
    d.dot(ox2, 0)
    # B to out
    d.wire((bx, .8), (bx, 2.7), (12.0, 2.7))
    d.part(bout[0], (12.0, 2.7), (15.2, 2.7), bout[1], bout[2])
    d.wire((15.2, 2.7), (ox2, 2.7))
    d.dot(ox2, 2.7)
    d.wire(oa["out"], (ox2, 0))
    d.wire((ox, -.8), (ox - 1.0, -.8), (ox - 1.0, -1.5))
    d.ground(ox - 1.0, -1.5)
    _vout(d, ox2, 0, ox2 + 2.0)


def draw_mfblp(d, v, out):
    _mfb(d, ("R", "R1", fr(v["R1"])), ("R", "R2", fr(v["R2"])), ("R", "R3", fr(v["R3"])),
         ("C", "Cg", fc(v["Cg"])), ("C", "Cf", fc(v["Cf"])))


def draw_mfbbp(d, v, out):
    _mfb(d, ("R", "R1", fr(v["R1"])), ("C", "C1", fc(v["C1"])), ("C", "C2", fc(v["C2"])),
         ("R", "R3", fr(v["R3"])), ("R", "R2", fr(v["R2"])))


def draw_twint(d, v, out):
    R_, C, k = v["R"], v["C"], v["k"]
    d.term(1, 0, "Vin")
    d.wire((1, 0), (2.4, 0), (2.4, 3.8))
    d.wire((2.4, 0), (2.4, -3.8))
    d.dot(2.4, 0)
    # upper branch: R - R with 2C to G
    d.wire((2.4, 3.8), (3.2, 3.8))
    d.R((3.2, 3.8), (6.6, 3.8), "R", fr(R_))
    d.wire((6.6, 3.8), (8.0, 3.8))
    d.dot(8.0, 3.8)
    d.wire((8.0, 3.8), (8.8, 3.8))
    d.R((8.8, 3.8), (12.2, 3.8), "R", fr(R_))
    d.wire((12.2, 3.8), (13.6, 3.8), (13.6, -3.8))
    d.C((8.0, 3.8), (8.0, 1.0), "2C", fc(2 * C), lab="right")
    # lower branch: C - C with R/2 to G
    d.wire((2.4, -3.8), (3.2, -3.8))
    d.C((3.2, -3.8), (6.6, -3.8), "C", fc(C), lab="below")
    d.wire((6.6, -3.8), (8.0, -3.8))
    d.dot(8.0, -3.8)
    d.wire((8.0, -3.8), (8.8, -3.8))
    d.C((8.8, -3.8), (12.2, -3.8), "C", fc(C), lab="below")
    d.wire((12.2, -3.8), (13.6, -3.8))
    d.R((8.0, -3.8), (8.0, -1.0), "R/2", fr(R_ / 2), lab="right")
    d.wire((8.0, 1.0), (8.0, -1.0))
    d.dot(8.0, 0)
    d.dot(13.6, 0)
    _vout(d, 13.6, 0, 15.6)
    d.wire((8.0, 0), (9.2, 0))
    if k > 0:
        d.text(9.4, 0.55, f"G = {k:.2g} \u00b7 Vout", ha="left", color=VALC)
        d.text(9.4, -0.45, "(buffered)", ha="left", color=MUTED, style="italic")
    else:
        d.ground(9.2, 0)
        d.text(10.0, -0.9, "G = GND", ha="left", color=MUTED)


def _rc_buffer(d, lp, R_, C_):
    oa = d.opamp(10, 0, minus_top=False, name="buffer")
    d.term(1, .8, "Vin")
    d.wire((1, .8), (2.2, .8))
    if lp:
        d.R((2.2, .8), (6.0, .8), "R", fr(R_))
        d.wire((6.0, .8), (10, .8))
        d.dot(7.4, .8)
        d.C((7.4, .8), (7.4, -2.8), "C", fc(C_))
        d.ground(7.4, -2.8)
    else:
        d.C((2.2, .8), (6.0, .8), "C", fc(C_))
        d.wire((6.0, .8), (10, .8))
        d.dot(7.4, .8)
        d.R((7.4, .8), (7.4, -2.8), "R", fr(R_))
        d.ground(7.4, -2.8)
    d.wire((10, -.8), (9, -.8), (9, -2.4), (14.2, -2.4), (14.2, 0))
    d.dot(14.2, 0)
    d.wire(oa["out"], (14.2, 0))
    _vout(d, 14.2, 0, 16.2)


STAGE_DRAW = {"sklp": draw_sklp, "skhp": draw_skhp, "mfblp": draw_mfblp,
              "rclp": lambda d, v, o: _rc_buffer(d, True, v["R"], v["C"]),
              "rchp": lambda d, v, o: _rc_buffer(d, False, v["R"], v["C"])}


def draw_afd(d, v, out):
    stages = (out.info or {}).get("stages", [])
    if getattr(d, "detail", False):
        # every stage as a real circuit, stacked top to bottom; the output of one feeds the next
        for i, st in enumerate(stages):
            d.off = (0.0, -i * 15.0)
            d.text(0.2, 5.8, f"Stage {i + 1}", ha="left", weight="bold", size=d.fs + 1)
            d.text(2.6 + 0.2 * len(str(i + 1)), 5.8, st["title"], ha="left", color=MUTED)
            STAGE_DRAW[st["kind"]](d, st["vals"], out)
        d.off = (0.0, 0.0)
        return
    per_row = 4
    w, h, gap = 6.6, 4.4, 2.0
    d.term(0.5, 0, "Vin")
    x0 = 1.8
    prev = (0.5 + 0.17, 0.0)
    for i, st in enumerate(stages):
        row, col = divmod(i, per_row)
        x = x0 + col * (w + gap)
        y = -row * (h + 3.0)
        d.box(x, y - h / 2, w, h)
        d.text(x + w / 2, y + h / 2 - 0.55, f"Stage {i + 1}", weight="bold")
        d.text(x + w / 2, y + h / 2 - 1.2, st["title"], size=d.fs - 1, color=MUTED)
        parts = st["parts"]
        lines = []
        for j in range(0, len(parts), 2):
            lines.append("   ".join(f"{n} = {fmt(val, u)}" for n, val, u in parts[j:j + 2]))
        for j, ln in enumerate(lines):
            d.text(x + w / 2, y + 0.15 - j * 0.75, ln, size=d.fs - 1, color=VALC)
        if i == 0:
            d.arrow(prev, (x, y))
        elif col == 0:
            px = x0 + (per_row - 1) * (w + gap) + w
            py = -(row - 1) * (h + 3.0)
            d.wire((px, py), (px + 0.8, py), (px + 0.8, py - (h / 2 + 1.5)), (x0 - 0.8, py - (h / 2 + 1.5)),
                   (x0 - 0.8, y), (x, y))
        else:
            d.arrow((x - gap, y), (x, y))
    if stages:
        n = len(stages)
        row, col = divmod(n - 1, per_row)
        x = x0 + col * (w + gap) + w
        y = -row * (h + 3.0)
        d.arrow((x, y), (x + gap, y))
        d.term(x + gap + 0.2, y, "Vout", side="right")


# ===========================================================================
# Passive circuits
# ===========================================================================
def _two_part_divider(d, series, shunt, vin="Vin"):
    d.term(1, 0, vin)
    d.wire((1, 0), (2.4, 0))
    d.part(series[0], (2.4, 0), (6.6, 0), series[1], series[2])
    d.wire((6.6, 0), (8.6, 0))
    d.dot(8.6, 0)
    d.part(shunt[0], (8.6, 0), (8.6, -3.6), shunt[1], shunt[2])
    d.ground(8.6, -3.6)
    _vout(d, 8.6, 0, 11.4)


def draw_rc(d, v, out):
    if v.get("out", "Low-pass") == "Low-pass":
        _two_part_divider(d, ("R", "R", fr(v["R"])), ("C", "C", fc(v["C"])))
    else:
        _two_part_divider(d, ("C", "C", fc(v["C"])), ("R", "R", fr(v["R"])))


def draw_rl(d, v, out):
    if v.get("out", "Low-pass") == "Low-pass":
        _two_part_divider(d, ("L", "L", fl(v["L"])), ("R", "R", fr(v["R"])))
    else:
        _two_part_divider(d, ("R", "R", fr(v["R"])), ("L", "L", fl(v["L"])))


def draw_rlc(d, v, out):
    sel = v["out"]
    R_, L, C = ("R", "R", fr(v["R"])), ("L", "L", fl(v["L"])), ("C", "C", fc(v["C"]))
    d.term(1, 0, "Vin")
    d.wire((1, 0), (2.4, 0))
    if sel.startswith("Across R"):
        seq = [L, C, R_]
        tap_after = 2
    elif sel.startswith("Across C"):
        seq = [R_, L, C]
        tap_after = 2
    elif sel.startswith("Across L ("):
        seq = [R_, C, L]
        tap_after = 2
    else:
        seq = [R_, L, C]
        tap_after = 1
    x = 2.4
    for i, p in enumerate(seq[:2]):
        d.part(p[0], (x, 0), (x + 3.6, 0), p[1], p[2])
        x += 3.6
        d.wire((x, 0), (x + 1.0, 0))
        x += 1.0
    last = seq[2]
    d.dot(x, 0)
    d.part(last[0], (x, 0), (x, -3.8), last[1], last[2])
    d.ground(x, -3.8)
    if tap_after == 2:
        _vout(d, x, 0, x + 3.0)
    else:
        # notch: Vout across L+C, i.e. the node before L
        nx = 2.4 + 3.6 + 1.0
        d.dot(nx, 0)
        d.wire((nx, 0), (nx, 2.6), (nx + 3.0, 2.6))
        d.term(nx + 3.0, 2.6, "Vout", side="right")
        d.text(nx + 0.3, 3.4, "(across L + C)", ha="left", color=MUTED, style="italic")


def draw_lcl(d, v, out):
    info = (out.info or {}).get("ladder")
    if not info:
        return
    kind, elems, Z0, RL = info["kind"], info["elems"], info["Z0"], info["RL"]
    d.term(0.8, 0, "Vin")
    d.wire((0.8, 0), (1.8, 0))
    d.R((1.8, 0), (5.4, 0), "Rs", fr(Z0))
    x = 5.4
    d.wire((x, 0), (x + 1.0, 0))
    x += 1.0
    for name, role, vals in elems:
        if role.startswith("series"):
            if kind == "Low-pass":
                d.L((x, 0), (x + 3.6, 0), name, fl(vals["L"]))
                x += 3.6
            elif kind == "High-pass":
                d.C((x, 0), (x + 3.6, 0), name, fc(vals["C"]))
                x += 3.6
            else:
                ln, cn = name.split("/")
                d.L((x, 0), (x + 3.4, 0), ln, fl(vals["L"]))
                d.C((x + 3.4, 0), (x + 6.8, 0), cn, fc(vals["C"]))
                x += 6.8
            d.wire((x, 0), (x + 1.0, 0))
            x += 1.0
        else:
            d.dot(x, 0)
            if kind == "Low-pass":
                d.C((x, 0), (x, -3.8), name, fc(vals["C"]))
                d.ground(x, -3.8)
                x += 3.2
            elif kind == "High-pass":
                d.L((x, 0), (x, -3.8), name, fl(vals["L"]))
                d.ground(x, -3.8)
                x += 3.2
            else:
                ln, cn = name.split("/")
                d.L((x, 0), (x, -3.8), ln, fl(vals["L"]), lab="left")
                d.ground(x, -3.8)
                x2 = x + 3.6
                d.wire((x, 0), (x2, 0))
                d.dot(x2, 0)
                d.C((x2, 0), (x2, -3.8), cn, fc(vals["C"]))
                d.ground(x2, -3.8)
                x = x2 + 3.2
            d.wire((x - 3.2, 0), (x, 0))
    d.dot(x, 0)
    d.R((x, 0), (x, -3.8), "RL", fr(RL))
    d.ground(x, -3.8)
    _vout(d, x, 0, x + 2.6)


# ===========================================================================
# Tools
# ===========================================================================
def draw_vd(d, v, out):
    d.term(1, 3.4, "Vin", sub=fmt(v["vin"], "V"))
    d.wire((1, 3.4), (6, 3.4), (6, 2.6))
    d.R((6, 3.4), (6, 0), "R1", fr(v["R1"]))
    d.dot(6, 0)
    d.R((6, 0), (6, -3.4), "R2", fr(v["R2"]))
    d.ground(6, -3.4)
    d.wire((6, 0), (9.5, 0))
    d.term(9.5, 0, "Vout", side="right", sub=fmt(v["vin"] * v["R2"] / (v["R1"] + v["R2"]), "V"))


def draw_lc(d, v, out):
    d.wire((2, 3.4), (7, 3.4))
    d.wire((2, 3.4), (2, 2.4))
    d.dot(2, 3.4)
    d.L((2, 3.4), (2, 0), "L", fl(v["L"]), lab="left")
    d.ground(2, 0)
    d.wire((7, 3.4), (7, 2.4))
    d.C((7, 3.4), (7, 0), "C", fc(v["C"]))
    d.ground(7, 0)
    d.dot(7, 3.4)
    d.text(4.5, 4.4, "resonant tank", color=MUTED, style="italic")
    d.text(4.5, -2.4, f"f0 = {fmt(1.0 / (2 * math.pi * math.sqrt(v['L'] * v['C'])), 'Hz')}", color=VALC, weight="bold")


def draw_xc(d, v, out):
    d.term(1, 0, "f", sub=fmt(v["f"], "Hz"))
    d.wire((1, 0), (2.4, 0))
    y = 0.0
    x = 2.4
    if v["C"] > 0:
        d.C((x, 0), (x + 3.6, 0), "C", fc(v["C"]))
        x += 3.6
    if v["L"] > 0:
        d.wire((x, 0), (x + 0.8, 0))
        x += 0.8
        d.L((x, 0), (x + 3.6, 0), "L", fl(v["L"]))
        x += 3.6
    d.wire((x, 0), (x + 1.0, 0), (x + 1.0, -1.0))
    d.ground(x + 1.0, -1.0)


DRAWERS = {
    "inv": draw_inv, "noninv": draw_noninv, "sum": draw_sum, "diff": draw_diff, "int": draw_int, "dif": draw_dif,
    "ina": draw_ina, "sklp": draw_sklp, "skhp": draw_skhp, "mfblp": draw_mfblp, "mfbbp": draw_mfbbp,
    "twint": draw_twint, "afd": draw_afd, "rc": draw_rc, "rl": draw_rl, "rlc": draw_rlc, "lcl": draw_lcl,
    "vd": draw_vd, "lc": draw_lc, "xc": draw_xc,
}


def draw(key, ax, values, outcome, fs=8.0, detail=False):
    """Draw the diagram of circuit `key` on `ax`. Returns True when something was drawn."""
    ax.clear()
    ax.axis("off")
    fn = DRAWERS.get(key)
    d = Drawer(ax, fs)
    d.detail = detail
    if fn is None or values is None and key not in ("afd", "lcl"):
        d.text(0.5, 0.5, "No diagram for this helper", color=MUTED, style="italic")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return False
    fn(d, values or {}, outcome)
    d.finish()
    return True
