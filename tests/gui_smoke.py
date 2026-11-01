"""Headless GUI smoke test (run with xvfb-run). Walks through every tab and circuit in both modes."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tkinter as tk
from tkinter import ttk
import gui
from circuits import TABS

errors = []
tk.Tk.report_callback_exception = lambda self, *a: errors.append(a)

root = tk.Tk()
root.geometry("1380x880")
fonts = gui.make_fonts(root)
nb = ttk.Notebook(root); nb.pack(fill="both", expand=True)
tabs = []
for title, circuits in TABS.items():
    t = gui.CalcTab(nb, circuits, fonts, with_sim=(title != "Tools"), show_gbw=(title == "Op-Amp Circuits"))
    nb.add(t, text=title); tabs.append(t)

def pump(sec=0.7):
    end = time.time() + sec
    while time.time() < end:
        root.update(); time.sleep(0.02)

shots = sys.argv[1] if len(sys.argv) > 1 else None
n = 0
for t in tabs:
    nb.select(t); pump(0.4)
    for c in t.circuits:
        for mode in ("forward", "design"):
            if (mode == "forward" and not c.has_forward) or (mode == "design" and not c.has_design):
                continue
            t.v_circ.set(c.name); t.v_mode.set(mode); t.on_select(); pump(0.7)
            txt = t.txt.get("1.0", "end")
            assert "Check your input" not in txt, (c.name, mode, txt)
            n += 1
            if shots and t.sim is not None and mode == ("design" if c.name in ("Active filter designer (any order)", "LC ladder filter (coils + capacitors)") else "forward"):
                t.sim.fig.savefig(os.path.join(shots, f"{c.key}_{mode}.png"), dpi=80)
# "Solve for": the solved field is read-only and shows the calculated value, the others are editable
t = tabs[2]; t.v_circ.set("RC filter (1st order)"); t.v_mode.set("design"); t.on_select(); pump(0.8)
assert set(t.vars) == {"out", "solve", "fc", "C"} and set(t.calc_fields) == {"R"}, (set(t.vars), set(t.calc_fields))
assert t.calc_fields["R"].get() not in ("", None), "R should show the calculated value"
t.vars["solve"].set("C  (from fc and R)"); t._rebuild(); pump(0.8)
assert set(t.vars) == {"out", "solve", "fc", "R"} and set(t.calc_fields) == {"C"}
t.vars["R"].set("1k"); pump(0.8)
assert t.calc_fields["C"].get() not in ("",), "C should show the calculated value"
t.vars["solve"].set("fc  (from R and C)"); t._rebuild(); pump(0.8)
assert set(t.vars) == {"out", "solve", "R", "C"} and set(t.calc_fields) == {"fc"}
assert t.calc_fields["fc"].get() != ""
assert "Check your input" not in t.txt.get("1.0", "end")
# diagram + enlarge window exist and follow the inputs
t = tabs[0]; t.v_circ.set("Inverting amplifier"); t.v_mode.set("forward"); t.on_select(); pump(0.8)
n_art = len(t.sch_ax.texts)
assert n_art > 5, n_art
t.enlarge(); pump(0.6)
assert t._big is not None and len(t._big[2].texts) > 5
t.vars["Rf"].set("47k"); pump(0.8)
assert any("47 kΩ" in x.get_text() for x in t.sch_ax.texts), [x.get_text() for x in t.sch_ax.texts]
assert any("47 kΩ" in x.get_text() for x in t._big[2].texts)
t._big[0].destroy(); pump(0.4)
# the "use parts" button
t = tabs[0]; t.v_circ.set("Inverting amplifier"); t.v_mode.set("design"); t.on_select(); pump(0.6)
t.use_parts(); pump(0.6)
assert t.v_mode.get() == "forward"
# bad input shows message, not an exception
t.vars["Rin"].set("abc"); pump(0.6)
assert "Check your input" in t.txt.get("1.0", "end")
t.vars["Rin"].set("10k"); pump(0.6)
if shots:
    root.update(); pump(0.5)
print("combos checked:", n, "errors:", len(errors))
for e in errors[:3]:
    import traceback; traceback.print_exception(*e)
root.destroy()
sys.exit(1 if errors else 0)
