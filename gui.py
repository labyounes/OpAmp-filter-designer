"""Tkinter GUI: calculator tabs with a Bode plot and a Vin/Vout time-domain simulation."""
from __future__ import annotations

import math
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

import core  # noqa: E402
import schem  # noqa: E402
from circuits import TABS, Ctx  # noqa: E402
from core import SERIES_CHOICES, EXACT, field_visible, fmt, parse_fields, parse_value  # noqa: E402

APP_NAME = "OpAmp Lab"
C_IN = "#2f6fb5"
C_OUT = "#d9531e"
C_GRID = "#d5d9df"
C_RAIL = "#b23b3b"

GBW_PRESETS = {
    "Ideal op-amp (infinite GBW)": 0.0,
    "LM358 / LM324  (1 MHz)": 1e6,
    "TL072 / TL082  (3 MHz)": 3e6,
    "NE5532  (10 MHz)": 10e6,
    "OPA2134  (8 MHz)": 8e6,
    "MCP6002  (1 MHz)": 1e6,
    "OPA2340  (5.5 MHz)": 5.5e6,
}

WAVES = ["Sine", "Square", "Triangle", "Step", "DC"]


def short(v: float) -> str:
    """Compact text for entry fields: 4700 -> '4.7k'."""
    return fmt(v, "").replace(" ", "")


def time_unit(T: float):
    if T < 1e-6:
        return 1e9, "ns"
    if T < 1e-3:
        return 1e6, "µs"
    if T < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


# ===========================================================================
class SimPanel(ttk.Frame):
    """Bode plot (magnitude + phase) and Vin/Vout waveform with live read-outs."""

    def __init__(self, master, mono):
        super().__init__(master)
        self.model = None
        self.f_char = 1e3
        self._job = None
        self.mono = mono

        self.fig = Figure(figsize=(7.4, 6.0), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        self.ax_mag = self.fig.add_subplot(gs[0, 0])
        self.ax_ph = self.fig.add_subplot(gs[0, 1])
        self.ax_t = self.fig.add_subplot(gs[1, :])
        self.readout = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.readout, font=self.mono, justify="left").pack(side="bottom", fill="x", padx=8, pady=(2, 6))
        box = ttk.LabelFrame(self, text="Input signal (Vin) and supply")
        box.pack(side="bottom", fill="x", padx=4, pady=(4, 2))
        tb = ttk.Frame(self)
        tb.pack(side="bottom", fill="x")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        NavigationToolbar2Tk(self.canvas, tb).update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.v_wave = tk.StringVar(value="Sine")
        self.v_amp = tk.StringVar(value="1")
        self.v_off = tk.StringVar(value="0")
        self.v_freq = tk.StringVar(value="1k")
        self.v_cyc = tk.StringVar(value="5")
        self.v_clip = tk.BooleanVar(value=True)
        self.v_sat = tk.StringVar(value="12")
        cells = [("Waveform", ttk.Combobox(box, textvariable=self.v_wave, values=WAVES, state="readonly", width=9)),
                 ("Amplitude (V)", ttk.Entry(box, textvariable=self.v_amp, width=8)),
                 ("Offset (V)", ttk.Entry(box, textvariable=self.v_off, width=8)),
                 ("Frequency (Hz)", ttk.Entry(box, textvariable=self.v_freq, width=10)),
                 ("Periods shown", ttk.Entry(box, textvariable=self.v_cyc, width=6))]
        for i, (lab, w) in enumerate(cells):
            ttk.Label(box, text=lab).grid(row=0, column=i, sticky="w", padx=(6, 2))
            w.grid(row=1, column=i, sticky="w", padx=(6, 2), pady=(0, 4))
        self.chk = ttk.Checkbutton(box, text="Limit output to ±", variable=self.v_clip)
        self.chk.grid(row=0, column=5, sticky="w", padx=(14, 2))
        sat = ttk.Frame(box)
        sat.grid(row=1, column=5, sticky="w", padx=(14, 2), pady=(0, 4))
        ttk.Entry(sat, textvariable=self.v_sat, width=6).pack(side="left")
        ttk.Label(sat, text=" V  (supply rail)").pack(side="left")
        ttk.Button(box, text="f = f₀", width=7, command=self.use_fchar).grid(row=1, column=6, padx=(14, 4), pady=(0, 4))

        for v in (self.v_wave, self.v_amp, self.v_off, self.v_freq, self.v_cyc, self.v_clip, self.v_sat):
            v.trace_add("write", lambda *a: self.schedule())

    # ------------------------------------------------------------------
    def schedule(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(220, self.refresh)

    def use_fchar(self):
        self.v_freq.set(short(self.f_char))

    def set_model(self, out, reset_freq=False, clip_default=None):
        self.model = out.model
        self.f_char = out.f_char
        if clip_default is not None:
            self.v_clip.set(clip_default and out.clip)
        if reset_freq:
            self.use_fchar()
        self.schedule()

    # ------------------------------------------------------------------
    def _read(self):
        amp = parse_value(self.v_amp.get())
        off = parse_value(self.v_off.get())
        f = parse_value(self.v_freq.get())
        cyc = parse_value(self.v_cyc.get())
        sat = parse_value(self.v_sat.get())
        if f <= 0:
            raise ValueError("frequency must be above 0")
        if not 0.25 <= cyc <= 200:
            raise ValueError("periods shown must be between 0.25 and 200")
        if sat <= 0:
            raise ValueError("supply rail must be above 0")
        return amp, off, f, cyc, sat

    def refresh(self):
        self._job = None
        for ax in (self.ax_mag, self.ax_ph, self.ax_t):
            ax.clear()
        if self.model is None:
            self.canvas.draw_idle()
            return
        try:
            amp, off, f, cyc, sat = self._read()
        except ValueError as exc:
            self.readout.set(f"Input signal: {exc}")
            self.canvas.draw_idle()
            return
        m = self.model
        # ---- Bode ------------------------------------------------------
        fa = core.default_freq_axis(m, self.f_char, f)
        h = core.freq_response(m, fa)
        mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        ph = np.degrees(np.unwrap(np.angle(h)))
        hi = core.freq_response(m, [f])[0]
        self.ax_mag.semilogx(fa, mag, color=C_OUT, lw=1.8)
        top = float(mag.max())
        self.ax_mag.set_ylim(max(float(mag.min()), top - 90) - 3, top + 6)
        self.ax_mag.set_title("Gain", fontsize=10, loc="left")
        self.ax_mag.set_ylabel("dB")
        self.ax_ph.semilogx(fa, ph, color=C_OUT, lw=1.8)
        self.ax_ph.set_title("Phase", fontsize=10, loc="left")
        self.ax_ph.set_ylabel("degrees")
        span = float(ph.max() - ph.min())
        if span < 20:
            mid = float((ph.max() + ph.min()) / 2)
            self.ax_ph.set_ylim(mid - 100, mid + 100)
        self.ax_ph.yaxis.set_major_locator(MultipleLocator(45 if span <= 360 else 90))
        for ax, val in ((self.ax_mag, 20 * math.log10(max(abs(hi), 1e-12))), (self.ax_ph, math.degrees(np.angle(hi)))):
            ax.axvline(f, color=C_IN, ls=":", lw=1.2)
            ax.grid(True, which="both", color=C_GRID, lw=0.5)
            ax.set_xlabel("Frequency (Hz)")
        # phase marker should sit on the unwrapped curve
        idx = int(np.argmin(np.abs(np.log(fa / f))))
        self.ax_mag.plot([f], [20 * math.log10(max(abs(hi), 1e-12))], "o", color=C_IN, ms=5)
        self.ax_ph.plot([f], [ph[idx]], "o", color=C_IN, ms=5)
        # ---- time domain -----------------------------------------------
        T = cyc / f
        N = int(min(max(cyc * 160, 1600), 14000))
        t = np.linspace(0.0, T, N + 1)
        u = core.make_input(self.v_wave.get(), t, amp, off, f)
        y = core.simulate(m, t, u, f)
        clipped = False
        if self.v_clip.get():
            yc = np.clip(y, -sat, sat)
            clipped = bool(np.any(np.abs(y) > sat))
            y_plot = yc
        else:
            y_plot = y
        sc, un = time_unit(T)
        self.ax_t.plot(t * sc, u, color=C_IN, lw=1.4, label="Vin")
        self.ax_t.plot(t * sc, y_plot, color=C_OUT, lw=1.6, label="Vout")
        lo = float(min(u.min(), y_plot.min()))
        hi_ = float(max(u.max(), y_plot.max()))
        pad = 0.08 * (hi_ - lo if hi_ > lo else 1.0)
        self.ax_t.set_ylim(lo - pad, hi_ + pad)
        if self.v_clip.get() and (clipped or max(abs(lo), abs(hi_)) > 0.5 * sat):
            for s in (sat, -sat):
                self.ax_t.axhline(s, color=C_RAIL, ls="--", lw=0.9, alpha=0.7)
        self.ax_t.set_xlim(0, T * sc)
        self.ax_t.set_xlabel(f"Time ({un})")
        self.ax_t.set_ylabel("Volts")
        self.ax_t.set_title("Vin and Vout", fontsize=10, loc="left")
        self.ax_t.grid(True, color=C_GRID, lw=0.5)
        self.ax_t.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, frameon=False)
        self.canvas.draw_idle()
        # ---- read-outs (last period) -----------------------------------
        mask = t >= T - min(1.0 / f, T)
        ui, yi = u[mask], y_plot[mask]
        line1 = (f"f = {fmt(f, 'Hz'):<10} gain {20 * math.log10(max(abs(hi), 1e-12)):7.2f} dB ({abs(hi):.4g}×)   "
                 f"phase {math.degrees(np.angle(hi)):7.1f}°")
        line2 = (f"Vin  pk-pk {fmt(ui.max() - ui.min(), 'V'):<10} mean {fmt(ui.mean(), 'V'):<10}  "
                 f"Vout pk-pk {fmt(yi.max() - yi.min(), 'V'):<10} mean {fmt(yi.mean(), 'V')}")
        if clipped:
            line2 += f"\n⚠ Output clipped at ±{fmt(sat, 'V')} (supply rail)"
        self.readout.set(line1 + "\n" + line2)


# ===========================================================================
class CalcTab(ttk.Frame):
    def __init__(self, master, circuits, fonts, with_sim=True, show_gbw=True):
        super().__init__(master)
        self.circuits = circuits
        self.by_name = {c.name: c for c in circuits}
        self.fonts = fonts
        self.show_gbw = show_gbw
        self.memory = {}
        self.vars = {}
        self._job = None
        self._reset_freq = True
        self.last = None

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=8)
        pw.add(left, weight=0)
        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        # ---- circuit + mode --------------------------------------------
        ttk.Label(left, text="Circuit", font=fonts["bold"]).pack(anchor="w")
        self.v_circ = tk.StringVar(value=circuits[0].name)
        cb = ttk.Combobox(left, textvariable=self.v_circ, values=[c.name for c in circuits], state="readonly", width=46)
        cb.pack(fill="x", pady=(2, 4))
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_select())
        self.blurb = ttk.Label(left, wraplength=470, justify="left", foreground="#4a4f58")
        self.blurb.pack(fill="x", pady=(0, 6))

        mrow = ttk.Frame(left)
        mrow.pack(fill="x")
        self.v_mode = tk.StringVar(value="forward")
        self.r_fwd = ttk.Radiobutton(mrow, text="Analyze  (parts → result)", value="forward", variable=self.v_mode, command=self.on_select)
        self.r_des = ttk.Radiobutton(mrow, text="Design  (target → parts)", value="design", variable=self.v_mode, command=self.on_select)
        self.r_fwd.pack(side="left", padx=(0, 12))
        self.r_des.pack(side="left")

        self.fields = ttk.Frame(left)
        self.fields.pack(fill="x", pady=8)

        opt = ttk.LabelFrame(left, text="Options")
        opt.pack(fill="x")
        ttk.Label(opt, text="Round resistors to").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.v_series = tk.StringVar(value="E24")
        sc = ttk.Combobox(opt, textvariable=self.v_series, values=SERIES_CHOICES, state="readonly", width=20)
        sc.grid(row=0, column=1, sticky="w", pady=3)
        sc.bind("<<ComboboxSelected>>", lambda e: self.schedule())
        ttk.Label(opt, text="Round C and coils to").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.v_cseries = tk.StringVar(value="E12")
        sc2 = ttk.Combobox(opt, textvariable=self.v_cseries, values=SERIES_CHOICES, state="readonly", width=20)
        sc2.grid(row=1, column=1, sticky="w", pady=3)
        sc2.bind("<<ComboboxSelected>>", lambda e: self.schedule())
        self.lbl_gbw = ttk.Label(opt, text="Op-amp GBW")
        self.v_gbw = tk.StringVar(value=list(GBW_PRESETS)[0])
        self.cb_gbw = ttk.Combobox(opt, textvariable=self.v_gbw, values=list(GBW_PRESETS), width=28)
        self.lbl_gbw.grid(row=2, column=0, sticky="w", padx=6, pady=3)
        self.cb_gbw.grid(row=2, column=1, sticky="w", pady=3)
        self.cb_gbw.bind("<<ComboboxSelected>>", lambda e: self.schedule())
        self.v_gbw.trace_add("write", lambda *a: self.schedule())
        if not show_gbw:
            self.lbl_gbw.grid_remove()
            self.cb_gbw.grid_remove()

        # ---- results ------------------------------------------------------
        rf = ttk.Frame(left)
        rf.pack(fill="both", expand=True, pady=(8, 0))
        self.txt = tk.Text(rf, height=9, width=60, wrap="word", font=fonts["mono"], relief="solid", bd=1,
                           padx=6, pady=4, state="disabled", background="#fbfbfc")
        sb = ttk.Scrollbar(rf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)
        cw = fonts["mono"].measure("0")
        self.txt.tag_configure("h", font=fonts["monobold"], foreground="#16365c", spacing1=8, spacing3=2)
        self.txt.tag_configure("kv", lmargin1=0, lmargin2=int(cw * 28))
        self.txt.tag_configure("k", foreground="#444a55")
        self.txt.tag_configure("v", font=fonts["monobold"])
        self.txt.tag_configure("warn", foreground="#a4380c")
        self.txt.tag_configure("note", foreground="#6b717c", font=fonts["monoital"])
        self.txt.tag_configure("err", foreground="#b00020", font=fonts["monobold"])

        brow = ttk.Frame(left)
        brow.pack(fill="x", pady=(4, 2))
        ttk.Button(brow, text="Copy results", command=self.copy).pack(side="left")
        self.btn_use = ttk.Button(brow, text="Use these parts in Analyze mode", command=self.use_parts)
        self.btn_use.pack(side="left", padx=6)

        hdr = ttk.Frame(left)
        hdr.pack(fill="x", pady=(6, 0))
        ttk.Label(hdr, text="Circuit diagram", font=fonts["bold"]).pack(side="left")
        ttk.Button(hdr, text="Enlarge / save ⤢", command=self.enlarge).pack(side="right")
        self.sch_fig = Figure(figsize=(5.2, 2.75), dpi=100, facecolor="white")
        self.sch_ax = self.sch_fig.add_axes([0, 0, 1, 1])
        self.sch_canvas = FigureCanvasTkAgg(self.sch_fig, master=left)
        self.sch_canvas.get_tk_widget().configure(highlightthickness=1, highlightbackground="#c9ced6")
        self.sch_canvas.get_tk_widget().pack(fill="x", pady=(2, 0))
        self._sch_state = None
        self._big = None

        # ---- simulation ---------------------------------------------------
        self.sim = None
        if with_sim:
            self.sim = SimPanel(right, fonts["mono"])
            self.sim.pack(fill="both", expand=True)
        else:
            ttk.Label(right, text="These helpers need no simulation.\nChange a value on the left and the result updates.",
                      justify="center", foreground="#6b717c", font=fonts["bold"]).place(relx=0.5, rely=0.4, anchor="center")
        self.on_select()

    # ------------------------------------------------------------------
    @property
    def circ(self):
        return self.by_name[self.v_circ.get()]

    def on_select(self):
        c = self.circ
        self.blurb.configure(text=c.blurb)
        if self.v_mode.get() == "forward" and not c.has_forward:
            self.v_mode.set("design")
        elif self.v_mode.get() == "design" and not c.has_design:
            self.v_mode.set("forward")
        self.r_fwd.state(["!disabled"] if c.has_forward else ["disabled"])
        self.r_des.state(["!disabled"] if c.has_design else ["disabled"])
        if self.show_gbw:
            for w in (self.lbl_gbw, self.cb_gbw):
                (w.grid if c.uses_gbw else w.grid_remove)()
        self._build_fields()
        self._reset_freq = True
        self.schedule(0)

    def _build_fields(self):
        for w in self.fields.winfo_children():
            w.destroy()
        self.vars = {}
        self.calc_fields = {}
        c, mode = self.circ, self.v_mode.get()
        all_fl = c.fwd_fields if mode == "forward" else c.des_fields
        texts = {f.key: self.memory.get((c.key, mode, f.key), f.default) for f in all_fl}
        dynamic = any(f.show_if for f in all_fl)
        self.fields.columnconfigure(1, weight=1)
        self.calc_fields = {}
        for r, f in enumerate(all_fl):
            active = field_visible(f, all_fl, texts)
            lab = f.label if active else f.label.split("  (")[0] + "   = calculated"
            ttk.Label(self.fields, text=lab, foreground="#000000" if active else "#2f6fb5").grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8))
            sv = tk.StringVar(value=texts[f.key])
            if f.kind == "choice":
                self.vars[f.key] = sv
                w = ttk.Combobox(self.fields, textvariable=sv, values=f.choices, state="readonly", width=30)
                w.grid(row=r, column=1, sticky="ew", pady=2)
                if dynamic:
                    w.bind("<<ComboboxSelected>>", lambda e: self.after_idle(self._rebuild))
                else:
                    w.bind("<<ComboboxSelected>>", lambda e: self.schedule())
            else:
                w = ttk.Entry(self.fields, textvariable=sv, width=14, justify="right")
                w.grid(row=r, column=1, sticky="e", pady=2)
                ttk.Label(self.fields, text=f.unit, width=3, foreground="#6b717c").grid(row=r, column=2, sticky="w", padx=(3, 0))
                if active:
                    self.vars[f.key] = sv
                    w.bind("<Return>", lambda e: self.schedule(0))
                else:
                    sv.set("")
                    w.state(["readonly"])
                    self.calc_fields[f.key] = sv
            if f.key in self.vars:
                sv.trace_add("write", lambda *a, k=f.key: self._changed(k))

    def _rebuild(self):
        self._build_fields()
        self.schedule(0)

    def _changed(self, key):
        if getattr(self, "_quiet", False):
            return
        c, mode = self.circ, self.v_mode.get()
        if key in self.vars:
            self.memory[(c.key, mode, key)] = self.vars[key].get()
        self.schedule()

    def schedule(self, delay=250):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(delay, self.update)

    # ------------------------------------------------------------------
    def _ctx(self):
        s = self.v_series.get()
        gbw = 0.0
        if self.show_gbw and self.circ.uses_gbw:
            txt = self.v_gbw.get().strip()
            if txt in GBW_PRESETS:
                gbw = GBW_PRESETS[txt]
            else:
                try:
                    gbw = parse_value(txt.split("(")[0] if "(" in txt else txt)
                except ValueError:
                    raise ValueError(f"Op-amp GBW: cannot read '{txt}'  (try 2M or 500k)") from None
                if gbw < 0:
                    raise ValueError("GBW must not be negative")
        cs = self.v_cseries.get()
        return Ctx(series=None if s == EXACT else s, gbw=gbw, cseries=None if cs == EXACT else cs)

    def update(self):
        self._job = None
        c, mode = self.circ, self.v_mode.get()
        fl = c.fwd_fields if mode == "forward" else c.des_fields
        try:
            p = parse_fields(fl, {k: v.get() for k, v in self.vars.items()})
            ctx = self._ctx()
            out = c.forward(p, ctx) if mode == "forward" else c.design(p, ctx)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        except (ArithmeticError, np.linalg.LinAlgError) as exc:
            self._show_error(f"These values cannot be calculated ({type(exc).__name__}). Check for 0 or extreme values.")
            return
        self.last = out
        self._sch_state = (c, (p if mode == "forward" else (out.std or p)), out)
        self._show(out)
        self.redraw_schematic()
        self._fill_calculated(out, mode)
        self.btn_use.state(["!disabled"] if (mode == "design" and out.std and c.has_forward) else ["disabled"])
        if self.sim is not None and out.model is not None:
            self.sim.set_model(out, reset_freq=self._reset_freq, clip_default=c.default_clip)
            self._reset_freq = False

    def _fill_calculated(self, out, mode):
        self._quiet = True
        try:
            for key, sv in getattr(self, "calc_fields", {}).items():
                if out.std and key in out.std:
                    val = out.std[key]
                elif key in ("fc", "f0"):
                    val = out.f_char
                else:
                    continue
                sv.set(short(val))
        finally:
            self._quiet = False

    # ------------------------------------------------------------------
    def _set_text(self, widget, writer):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        writer(widget)
        widget.configure(state="disabled")

    def _show_error(self, msg):
        self._set_text(self.txt, lambda w: w.insert("end", "Check your input:\n", "err") or w.insert("end", msg, "warn"))

    def _show(self, out):
        def write(w):
            for kind, a, b in out.rows:
                if kind == "h":
                    w.insert("end", a + "\n", "h")
                elif kind == "kv":
                    w.insert("end", a.ljust(27) + " ", ("kv", "k"))
                    w.insert("end", str(b) + "\n", ("kv", "v"))
                elif kind == "warn":
                    w.insert("end", "⚠ " + a + "\n", "warn")
                else:
                    w.insert("end", a + "\n", "note")
        self._set_text(self.txt, write)

    def _render(self, fig, ax, fs, detail):
        if self._sch_state is None:
            return
        c, vals, out = self._sch_state
        try:
            schem.draw(c.key, ax, vals, out, fs=fs, detail=detail)
        except Exception as exc:  # noqa: BLE001  (a drawing problem must never break the calculator)
            ax.clear()
            ax.axis("off")
            ax.text(0.5, 0.5, f"Diagram not available\n({type(exc).__name__}: {exc})", ha="center", va="center", fontsize=8, color="#a4380c")

    def redraw_schematic(self):
        self._render(self.sch_fig, self.sch_ax, 7.5, False)
        self.sch_canvas.draw_idle()
        if self._big is not None:
            win, fig, ax, canvas = self._big
            if win.winfo_exists():
                self._render(fig, ax, 10, True)
                canvas.draw_idle()
            else:
                self._big = None

    def enlarge(self):
        if self._big is not None and self._big[0].winfo_exists():
            self._big[0].lift()
            return
        win = tk.Toplevel(self)
        win.title(f"{APP_NAME} – circuit diagram")
        win.geometry("1100x720")
        fig = Figure(figsize=(11, 7), dpi=100, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        canvas = FigureCanvasTkAgg(fig, master=win)
        tb = ttk.Frame(win)
        tb.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(canvas, tb).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._big = (win, fig, ax, canvas)
        self._render(fig, ax, 10, True)
        canvas.draw_idle()

    def copy(self):
        text = self.txt.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)

    def use_parts(self):
        out = self.last
        c = self.circ
        if not out or not out.std:
            return
        for f in c.fwd_fields:
            if f.key in out.std:
                v = out.std[f.key]
                self.memory[(c.key, "forward", f.key)] = v if isinstance(v, str) else short(v) if v not in (math.inf,) else "inf"
        self.v_mode.set("forward")
        self.on_select()


# ===========================================================================
def make_fonts(root):
    base = tkfont.nametofont("TkDefaultFont")
    families = set(tkfont.families())
    if sys.platform.startswith("win") and "Segoe UI" in families:
        base.configure(family="Segoe UI", size=10)
        tkfont.nametofont("TkTextFont").configure(family="Segoe UI", size=10)
    mono_family = next((f for f in ("Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Menlo", "Courier New") if f in families), "TkFixedFont")
    size = 10
    return {
        "bold": tkfont.Font(family=base.actual("family"), size=base.actual("size"), weight="bold"),
        "mono": tkfont.Font(family=mono_family, size=size),
        "monobold": tkfont.Font(family=mono_family, size=size, weight="bold"),
        "monoital": tkfont.Font(family=mono_family, size=size, slant="italic"),
    }


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001  (not on Windows / old Windows)
        pass
    root = tk.Tk()
    root.title(f"{APP_NAME} – op-amp and filter calculator with simulation")
    root.geometry("1380x880")
    root.minsize(1120, 720)
    style = ttk.Style(root)
    for name in ("vista", "clam"):
        if name in style.theme_names():
            style.theme_use(name)
            break
    fonts = make_fonts(root)
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    for title, circuits in TABS.items():
        tools = title == "Tools"
        nb.add(CalcTab(nb, circuits, fonts, with_sim=not tools, show_gbw=(title == "Op-Amp Circuits")), text=f"  {title}  ")
    root.mainloop()


if __name__ == "__main__":
    main()
