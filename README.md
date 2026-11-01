# OpAmp Lab

A Windows desktop tool (Python + Tkinter) for designing and simulating op-amp circuits and filters.
Enter a frequency or gain and get resistor / capacitor / coil values, or enter parts and get the response,
with a live circuit diagram, Bode plot and Vin/Vout time plot.

## Features
- **Op-amp circuits:** inverting, non-inverting, summing, difference, integrator, differentiator, instrumentation amp (optional real GBW, e.g. LM358, TL072).
- **Active filters:** order 1-10, Butterworth / Bessel / Chebyshev, low-/high-pass, Sallen-Key or MFB, plus band-pass and twin-T notch.
- **Passive filters:** LC ladders with coils, series RLC, RC, RL.
- **Tools:** reactance, LC resonance, voltage divider, dB converter, nearest E-series value.
- **Two modes per circuit:** *Analyze* (parts -> result) and *Design* (target -> parts, rounded to E6...E96).
- **Simulation:** Bode (gain/phase) and time plot with sine/square/triangle/step/DC input, offset and supply-rail clipping.

## Install
Requires Python 3.9+ ([python.org](https://www.python.org/downloads/), tick "Add python.exe to PATH").

```bash
git clone https://github.com/<your-user>/opamp-lab.git
cd opamp-lab
pip install -r requirements.txt
python main.py
```

On Windows you can also just double-click `run_windows.bat`.
For a single `.exe` (no Python needed on the target PC) run `build_exe.bat` -> `dist\OpAmpLab.exe`.

## Tests
```bash
python tests/test_circuits.py
```
Checks every formula against an independent nodal circuit solver.

## Limits
Op-amps are ideal apart from the single-pole GBW model; slew rate, offsets and noise are not modelled.
Inductors and capacitors are ideal (no ESR / Q).

## Layout
`core.py` maths and simulation, `circuits.py` calculators, `filters.py` filter synthesis, `schem.py` diagrams, `gui.py` window, `main.py` entry point.
